# 双第三视角到第一视角的视频生成两阶段训练设计

## 当前实现对应的主方案

这版方案尽量贴近现有 WAN world-model 结构，并贴近“先保留原有三视角方式，再做时序注入微调”的思路。

## 阶段一：masked target first frame

- 输入仍使用三视角联合视频 `video`
- 视角顺序固定为：
  - `view 0`: 第三视角 1
  - `view 1`: 第三视角 2
  - `view 2`: 第一视角
- 训练时内部构造：
  - `video_gt`: 原始真实三视角视频
  - `video_cond`: 仅把第一视角第 1 帧替换为占位帧，其余帧保持真实
- 条件分支读取 `video_cond`
- 监督与去噪目标读取 `video_gt`
- `num_history_frames` 固定为 `1`
- loss 设计：
  - 第一视角为主损失
  - 第三视角为弱辅助损失

## 阶段二：完整第三视角时序注入

- 在阶段一 checkpoint 上继续训练
- 新增 `CrossViewSourceVideoProjector3D`
- 输入两路第三视角完整视频，先经冻结 VAE 编码，再经过 3D projector 生成 `source_tokens`
- `source_tokens` 注入到 DiT 的 cross-attention context
- 旧的 image-conditioning 分支保留，但在训练时做 dropout 弱化
- 阶段二只监督第一视角损失，不再保留第三视角辅助损失

## 关键约束

- 当前实现要求 `num_history_frames=1`
- 当前实现默认 `video` 已经按三视角顺序组织好
- 推理时如果沿用该方案，也应保留三视角槽位，并把第一视角首帧替换为固定占位帧

## 对应任务名

- `cross_view_stage1`
- `cross_view_stage2`




train :/home/xuehao/xh/projects/DiffSynth-Studio/bash/train_droid_crossview_small200.sh
inference：
  stage1：/home/xuehao/xh/projects/DiffSynth-Studio/examples/wanvideo/model_inference/infer_cross_view_stage1.py
  stage2：/home/xuehao/xh/projects/DiffSynth-Studio/examples/wanvideo/model_inference/infer_cross_view_stage2.py


CUDA_VISIBLE_DEVICES=0 python /home/xuehao/xh/projects/DiffSynth-Studio/examples/wanvideo/model_inference/infer_cross_view_stage1.py \ 
  --ckpt_path /home/xuehao/xh/projects/DiffSynth-Studio/Ckpt/droid_crossview_10000_stage1/epoch-4/epoch-4.safetensors \
  --config_json /home/xuehao/xh/projects/DiffSynth-Studio/Ckpt/droid_crossview_10000_stage1/config.json \
  --dataset_base_path /data1/xuehao/datasets/droid_1.0.1_crossview_meta \
  --dataset_metadata_path /data1/xuehao/datasets/droid_1.0.1_crossview_meta/meta/episodes_cross_view_val_81.jsonl  \
  --train_metadata_path /data1/xuehao/datasets/droid_1.0.1_crossview_meta/meta/episodes_cross_view_train_81_10000.jsonl \



# V2版本
## 改动总览
  这次实现的核心，是把 stage2 从“把 source view 压成一串全局 token 再拼到 context”升级成“按时间组织的 source memory + 局部时窗注入 + 更强辅助约束”的训练方
  案。

  - 在 cross_view_projector.py:73 新增了 CrossViewSourceVideoProjector3DTemporal。它把 source views 的 VAE latent 编成 B x T_latent x N_source_tokens x D，
    而不是旧版 B x N_all_tokens x D。这样后续可以逐帧或局部时窗注入。
  - 在 wan_video_dit.py:228 给 DiTBlock 增加了 temporal_source_cross_attn。每个目标 latent 帧 t 只看 t-r ~ t+r 的 source memory，再做 cross-attn，而不是所有
    时刻一起灌进去。
  - 在 wan_video.py:539 扩展了 model_fn_wan_video，支持 source_memory_by_time、source_window_radius、return_hidden_by_time。同时修掉了 gradient
    checkpointing 路径里 stage2 会重复传 source_memory_by_time 的 bug。
  - 在 train.py:44 增加了两个新模块：CrossViewSourceTemporalGate 和 CrossViewTargetStateHead。前者控制 source memory 注入强度，后者从 DiT hidden 预测
    target-state，作为 stage2 辅助监督。
  - 在 runner.py:14 加了训练进度注入，让 cross_view_old_branch_dropout_schedule=linear_warmup_to_high 真正按训练进度提高旧图像分支 dropout。
  - 在 infer_cross_view_stage2.py 和 metric.py:58 同步了推理/评估逻辑：stage2 推理走同一套 temporal source builder，评估新增“target view 去首帧”的指标。

## 训练链路
  以 cross_view_stage2 为例，训练数据流现在是这样走的，主入口在 train.py:775。

  - 输入仍是 video,state,prompt_emb。默认脚本里 LOAD_MODULES=dit,text:emb,vae,image,action:noise，所以文本走预提取 prompt_emb，状态 state_pose_7d 复用
    action encoder 路径，不额外要求数据里有 action 字段，见 train.py:900。
  - video 的原始形状是 V x C x T x H x W。代码先构造 cond_video，把 target view 的第 1 帧替换成 placeholder，逻辑在 train.py:465。
  - cross-view 训练显式跳过了 WanVideoUnit_InputVideoEmbedder，只跑后面的 VAE image / CLIP / prompt / action(state) 单元，见 train.py:536。原因是 stage1/
    stage2 需要同时保留两套 latent：GT latent 和 masked cond latent，不能复用原本单一路径的 input_latents。
  - video_gt 和 cond_video 会各自经过 VAE 编码成 joint latent，形状是 1 x C_lat x T_lat x (V*H_lat) x W_lat，见 train.py:542。
  - stage2 额外取 source views，经 VAE 后送入 projector。若 cross_view_source_injection_mode=temporal_local，得到 source_memory_by_time；若是
    global_concat，得到旧版 source_tokens，逻辑在 train.py:587。
  - source_temporal_gate 会在 projector 输出后、进入 DiT 前起作用。scalar 是全局一个 sigmoid gate，state_aware 是由 downsample 后的状态序列逐时刻生成 gate，
    见 train.py:44。
  - 加噪方式仍是 flow matching。先对 input_latents_gt 采样时间步、加高斯噪声，再把第一个 history latent timestep 强行替换回 input_latents_cond，见
    train.py:797。因为当前 cross-view 训练强约束 num_history_frames=1，所以这里只固定第一段 latent history。
  - DiT 内部如果是 temporal_local，每个 block 在 cross-attn 时，对目标时间步 t 只拼接局部 source memory 窗口 [t-r, t+r]，见 wan_video_dit.py:228。这就是你之
    前问的“逐帧注入/局部注入”，现在已经落地。
  - 一个关键实现细节是 source memory 始终追加在已有 context 后面，而不是插到前面，见 wan_video.py:583。这是为了不破坏 CrossAttention.has_image_input 对前
    257 个 CLIP token 的位置假设。

## 损失与约束
  stage1 和 stage2 的目标现在明确分层了。

  - stage1 仍是“全视角 joint reconstruction”，target view 为主，source views 作为辅助重建项，权重 cross_view_source_loss_weight，见 train.py:726。
  - stage2 的主损失改成只对 target view 的 flow matching loss。也就是 source view 不再要求重建，而是转成条件来源，见 train.py:822。
  - 新增 temporal consistency loss。实现方式是在 latent x0 空间上计算：pred_x0 = noisy_latent - sigma * noise_pred，再比较 target view 相邻时间差分和 GT 差
    分，见 train.py:678。这比直接对噪声做时序一致性更合理。
  - 新增 state alignment loss。做法是取 DiT 的 hidden_by_time，按 joint latent 高度拆回各个 view，只保留 target view，再对 token 维做 pooling，用
    target_state_head 预测 downsample 后的状态序列，见 train.py:703。
  - old branch dropout 仍然只打掉旧图像条件分支 y 和 clip_feature，不会打掉文本或状态条件，见 train.py:759。现在支持 linear_warmup_to_high，训练越往后越强迫
    模型依赖 source memory 而不是首帧图像分支。
  - 新模块 source_video_projector、source_temporal_gate、target_state_head 都加入了 checkpoint 加载和 trainable models 自动扩展，见 train.py:289 和train.py:309。


  
## 训练方式
  训练入口还是 train_droid_crossview_small200.sh:76，默认是 accelerate launch + bf16 + gradient checkpointing。

  - 优化器是 AdamW，学习率调度是 ConstantLR，见 runner.py:47。
  - DataLoader 实际是单样本迭代，默认 batch size 等价于 1，有效 batch 主要靠 gradient_accumulation_steps 和多卡数决定，见 runner.py:49。
  - 默认脚本参数是 NUM_FRAMES=81、HEIGHT=180、WIDTH=320、GRAD_ACCUM_STEPS=4、MIXED_PRECISION=bf16，状态输入是 state_pose_7d，见
    train_droid_crossview_small200.sh:9。
  - stage2 必须从 stage1 checkpoint 接着训，脚本里强制要求 CKPT_PATH，见 train_droid_crossview_small200.sh:69。

  建议按下面两步训练。

  CUDA_VISIBLE_DEVICES=2,3 \
  TAG=droid_crossview_small200_stage1 \
  TASK=cross_view_stage1 \
  OUTPUT_PATH=/home/xuehao/xh/projects/DiffSynth-Studio/Ckpt/droid_crossview_small200_stage1 \
  bash /home/xuehao/xh/projects/DiffSynth-Studio/bash/train_droid_crossview_small200.sh

  CUDA_VISIBLE_DEVICES=2,3 \
  TAG=droid_crossview_small200_stage2 \
  TASK=cross_view_stage2 \
  CKPT_PATH=/home/xuehao/xh/projects/DiffSynth-Studio/Ckpt/droid_crossview_small200_stage1/epoch-79/epoch-79.safetensors \
  OUTPUT_PATH=/home/xuehao/xh/projects/DiffSynth-Studio/Ckpt/droid_crossview_small200_stage2 \
  CROSS_VIEW_SOURCE_INJECTION_MODE=temporal_local \
  CROSS_VIEW_SOURCE_WINDOW_RADIUS=1 \
  CROSS_VIEW_SOURCE_GATE_MODE=scalar \
  CROSS_VIEW_TEMP_LOSS_WEIGHT=0.1 \
  CROSS_VIEW_STATE_LOSS_WEIGHT=0.05 \
  CROSS_VIEW_OLD_BRANCH_DROPOUT=0.3 \
  CROSS_VIEW_OLD_BRANCH_DROPOUT_SCHEDULE=linear_warmup_to_high \
  bash /home/xuehao/xh/projects/DiffSynth-Studio/bash/train_droid_crossview_small200.sh




 # 两阶段方案重定版：Stage1 预训练 + Stage2 混合观测源视角

  ## 摘要

  - 训练仍然保持 两阶段，不新增 stage2.5。
  - Stage1 保持现有定义：三视角 joint reconstruction，学基本视频生成、视角共享表征和 target 补全。
  - Stage2 直接改成最终版混合方案：
    source full videos -> external temporal memory，同时 source branches -> sigma-matched clamped latents；只生成 target first-person view，只对 target 算损失。
  - 这个 Stage2 从最佳 Stage1 checkpoint 启动，不从旧 Stage2 启动。

  ## 训练设计

  - Stage1
      - 保持现有 cross_view_stage1 不变。
      - 输入仍是三视角视频，target 首帧做 placeholder。
      - 主损失仍是 target view 主导、source views 辅助的 joint flow-matching。
      - 输出目标仍是学到稳定的 joint video prior 和跨视角共享特征。
  - Stage2
      - 任务语义固定为：两路第三视角是观测量，一路第一视角是生成量。
      - source full videos 走两条路：
          - 走 projector/gate，形成 source_memory_by_time，供 block 的 temporal-local cross-attn 使用。
          - 走 VAE latent 后，按当前 timestep t 重建 sigma-matched source latents，直接覆盖 joint latent 中 source branches，供 self-attn 使用。
      - target view 仍是唯一需要预测的分支：
          - x_t_target = add_noise(target_x0, eps_target, t)
          - source branches 使用 x_t_source = add_noise(source_x0, eps_source, t)，不是 clean clamp
      - target history 继续固定为 num_history_frames=1，但 history latent 只能由“target placeholder 首帧 + 未来全零 padding”编码得到；不能再用完整 target GT video 编码 history
        条件。
      - y 和 clip_feature 旧图像分支在 Stage2 中训练和推理都直接禁用；不再保留 old-branch dropout 机制。
      - Stage2 只训练：dit, action_encoder, source_video_projector, source_temporal_gate, target_state_head

  ## 关键实现变更

  - 任务与配置
      - 仍使用 cross_view_stage2 任务名，不新增第三阶段任务名。
      - cross_view_stage2 固定语义为最终版混合方案。
      - 固定默认：
          - cross_view_source_injection_mode=temporal_local
          - cross_view_source_window_radius=1
          - cross_view_source_gate_mode=scalar
          - cross_view_temp_loss_weight=0.1
          - cross_view_state_loss_weight=0.05
          - cross_view_old_branch_dropout=0
  - 前向数据流
      - 新增单独的 source latent builder：只编码 source views，得到 source_x0_latents
      - 新增单独的 target history builder：只编码 target placeholder 第一帧，未来填零
      - 新增 joint latent assembler：
          - source view 区域替换为 sigma-matched x_t_source
          - target view 区域替换为 x_t_target
          - target history slice 再覆盖为 placeholder-history latent
      - 外部 source_memory_by_time 始终由 clean source_x0_latents 生成，不随 timestep 变化
  - 损失
      - 主损失：target-only flow-matching loss
      - 辅助损失：
          - target-only temporal consistency loss
  - 推理
          - source_x0_latents
          - 固定 eps_source
      - 每个 denoising step 前重建并覆盖 source branches：
          - x_t_source = (1-sigma_t) * source_x0 + sigma_t * eps_source
      - 每步同时覆盖 target history slice
      - scheduler 仍作用于整块 joint latent，但下一步开始前必须重新覆盖 source branches 和 target history
      - 正式输出只取 target view；全视角 decode 仅用于调试和可视化

  ## 训练方式

  - Stage1
      - 按现有脚本训练到收敛，选择最佳 target_view_exclude_first 或验证集目标质量的 checkpoint
  - Stage2
      - 从该 Stage1 checkpoint 初始化
      - 学习率设为 Stage1 的 0.5x
      - 总训练步数设为原先 Stage2 预算的 1.0x
      - 不从你已训练的旧 Stage2 checkpoint 继续；因为新的 Stage2 输入状态、约束和条件路径已变，继续训会把旧解耦假设带进来
  - 默认脚本策略
      - 保持两阶段脚本入口不变
      - TASK=cross_view_stage1 训练第一阶段
      - TASK=cross_view_stage2 训练第二阶段，但其内部实现已切换为混合方案

  ## 测试与验收

  - 静态/单元
      - source branch overwrite 后，source 区域必须逐元素等于 scheduler.add_noise(source_x0, eps_source, t)
      - target history slice 必须逐元素等于 placeholder-history latent
      - 改变 target future GT，不得改变 source memory 和 target history latent
  - 训练 smoke test
      - cross_view_stage2 在开启 gradient checkpointing 时可前反向一步
  - 推理一致性
      - 固定 seed 时，每个 denoising step 重建的 source branches 必须与当前 sigma_t 对应
  - 验收指标
      - 主指标：target_view_exclude_first
      - 次指标：all_views_exclude_first
      - 新 Stage2 必须优于旧 external-memory-only Stage2，且不能出现更重的 target 首帧伪影或中后段漂移

  ## 假设

  - 推理场景始终提供两路 source full videos
  - 目标仍是单一路第一视角视频生成，不需要同时输出 source views
  - Stage1 仍然有价值，因为它负责学通用 joint video prior；Stage2 负责把任务语义切换为“observed source -> generated target”