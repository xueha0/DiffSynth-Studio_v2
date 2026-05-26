# DiffSynth-Studio v2 Userbook

本手册面向两类读者：一类是希望直接跑通数据缓存、训练、推理评估的用户；另一类是需要理解代码结构、模型架构和关键实现逻辑的研发者或 AI/Codex 阅读者。文档重点覆盖本仓库当前的 WAN cross-view robot world model 流程，尤其是 DROID 三视角数据、两阶段训练、latent cache、LagerNVS geometry sidecar 与 stage2 几何增强逻辑。

## 1. 项目概览

DiffSynth-Studio v2 基于 WAN 视频扩散模型，扩展了机器人场景下的跨视角视频生成能力。当前 cross-view 任务的典型目标是：给定两个外部相机视角、机器人状态轨迹，以及目标相机的起始条件，生成目标 wrist/第一人称视角未来视频。

### 1.1 核心用途

本仓库主要支持以下能力：

- Cross-view stage1 训练：联合建模多视角视频，把多个视角沿 latent 高度维拼接，让模型学习跨视角一致的视频去噪。
- Cross-view stage2 训练：从 stage1 checkpoint 初始化，只预测目标视角，同时把源视角视频编码成 source memory 注入 DiT，让模型专注目标视角生成。
- LagerNVS geometry sidecar：使用 LagerNVS 的 camera-aware scene tokens 与目标相机 tokens 强化 3D 几何条件。
- 离线 latent cache：预先缓存 VAE latents、CLIP/image branch、prompt embedding 引用、state/action 等，显著降低训练时 CPU/IO/VAE 编码开销。
- 推理与评估：批量生成 GT|Pred 对比视频，并计算 PSNR、SSIM、LPIPS-like、FVD-style 指标。

### 1.2 典型输入输出

**训练输入**

- 多视角视频：默认 3 个 view，shape 为 `(V, C, T, H, W)`，例如 `(3, 3, 81, 180, 320)`，像素范围 `[-1, 1]`。
- 状态序列：`state_pose_7d`，shape 为 `(1, T, 7)`，归一化到 `[-1, 1]`。
- Prompt embedding：预先保存的 `.pt` 文件路径。
- 可选相机条件：source camera tokens、target camera tokens、scene tokens。

**模型输出**

- 训练时输出预测的 flow/noise target，shape 对齐 VAE latent，例如目标视角 stage2 为 `(B, 16, 21, 22, 40)` 左右。
- 推理时输出目标视角视频和对比视频，通常保存为 `.mp4`。
- checkpoint 保存为 `epoch-N/epoch-N.safetensors`，并配套保存 `config.json`。

### 1.3 最小 smoke check

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2

/env/conda/envs/studio/bin/python -m py_compile \
  tool/build_cross_view_latent_cache.py \
  tool/build_cross_view_geometry_sidecar_cache.py \
  examples/wanvideo/model_training/train.py \
  examples/wanvideo/model_inference/infer_cross_view_stage2.py
```

## 2. 仓库结构

```text
DiffSynth-Studio_v2/
├── diffsynth/
│   ├── pipelines/                 # WAN pipeline、pipeline units、model_fn
│   ├── models/                    # DiT、VAE、action encoder、geometry modules
│   ├── core/                      # UnifiedDataset、data operators、loader
│   └── diffusion/                 # 训练 runner、logger、parser、scheduler
├── examples/wanvideo/
│   ├── model_training/train.py    # 主训练入口，stage1/stage2 逻辑集中在这里
│   └── model_inference/           # stage1/stage2 推理与闭环推理
├── tool/
│   ├── build_cross_view_latent_cache.py
│   ├── build_cross_view_geometry_sidecar_cache.py
│   ├── build_prompt_embeddings.py
│   └── evaluate_generated_videos.py
├── bash/
│   ├── train_droid_success_high_quality_crossview_cache.sh
│   ├── train_droid_crossview_small200.sh
│   └── infer_all_epoch_stage2.sh
├── Ckpt/                          # 本地训练输出，不建议提交
└── userbook.md
```

### 2.1 模块调用关系

```text
bash/*.sh
  -> examples/wanvideo/model_training/train.py
    -> diffsynth.diffusion.parsers.prepare_wan_runtime
    -> diffsynth.core.UnifiedDataset
    -> WanTrainingModule
      -> WanVideoPipeline
        -> WanVideoUnit_NoiseInitializer
        -> WanVideoUnit_InputVideoEmbedder
        -> WanVideoUnit_ImageEmbedderVAE
        -> WanVideoUnit_ImageEmbedderCLIP
        -> WanVideoUnit_PromptEmbedder
        -> WanVideoUnit_ActionEmbedder
      -> model.forward_cross_view / forward_cross_view_cached
        -> pipe.model_fn
          -> diffsynth.pipelines.wan_video.model_fn_wan_video
            -> diffsynth.models.wan_video_dit.WanModel blocks
```

### 2.2 代码阅读示例

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2

grep -n "def forward_cross_view" examples/wanvideo/model_training/train.py
grep -n "def model_fn_wan_video" diffsynth/pipelines/wan_video.py
grep -n "geometry_aware_cross_attn" diffsynth/models/wan_video_dit.py
```

## 3. 环境准备

### 3.1 推荐 Python 环境

当前机器上建议使用：

```bash
export PYTHON_BIN=/env/conda/envs/studio/bin/python
$PYTHON_BIN -V
```

如果需要从头创建环境，可参考仓库提供的环境文件：

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2
conda env create -f diff-environment.yml
```

### 3.2 预训练模型目录

训练脚本通过 `MODEL_DIR` 指向 WAN/PAI 权重根目录，常用路径：

```bash
export MODEL_DIR=/data_ywj/data_xh/projects/datasets/PAI
```

`prepare_wan_runtime()` 会按 `load_modules` 自动寻找以下权重：

| 模块 | 默认文件候选 | 说明 |
| --- | --- | --- |
| `dit` | `diffusion_pytorch_model.safetensors` 或 `.pth` | WAN DiT 主干 |
| `text` | `models_t5_umt5-xxl-enc-bf16.pth` 或 `.safetensors` | T5 文本编码器，`text:emb` 时不加载 |
| `vae` | `Wan2.1_VAE.pth` 或 `.safetensors` | 视频 VAE |
| `image` | `models_clip_open-clip-xlm-roberta-large-vit-huge-14.*` | CLIP image encoder |
| `action` | 仓库内动态创建 | action/state 条件 encoder |

### 3.3 环境检查示例

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2

$PYTHON_BIN - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), torch.cuda.device_count())
PY
```

## 4. 数据准备

### 4.1 数据目录约定

本项目当前 DROID cross-view 数据通常分为原始/视频数据和 metadata 数据：

```bash
export DATASET_META_ROOT=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta
export TRAIN_MANIFEST=$DATASET_META_ROOT/meta/episodes_cross_view_train_81_small16567.jsonl
export VAL_MANIFEST=$DATASET_META_ROOT/meta/episodes_cross_view_val_81_small200.jsonl
export STATE_STAT_PATH=$DATASET_META_ROOT/meta/stat_state_pose_7d.json
export NEG_PROMPT_EMB=$DATASET_META_ROOT/prompt_emb/neg_prompt.pt
```

如果 manifest 内路径是相对路径，`UnifiedDataset` 会用 `dataset_base_path` 拼接；因此 `DATASET_META_ROOT` 必须能解析 manifest 中的 `video`、`state`、`prompt_emb` 引用。

### 4.2 manifest JSONL 格式

每一行是一个 episode clip。典型字段如下：

```json
{
  "video": [
    {"data": "videos/episode_000001_left_external.mp4", "start_frame": 0, "end_frame": 80, "pad_to_frames": 81, "pad_mode": "repeat_last"},
    {"data": "videos/episode_000001_right_external.mp4", "start_frame": 0, "end_frame": 80, "pad_to_frames": 81, "pad_mode": "repeat_last"},
    {"data": "videos/episode_000001_wrist.mp4", "start_frame": 0, "end_frame": 80, "pad_to_frames": 81, "pad_mode": "repeat_last"}
  ],
  "state": {"data": "states/episode_000001.parquet", "start_frame": 0, "end_frame": 80, "pad_to_frames": 81, "pad_mode": "repeat_last"},
  "prompt_emb": "prompt_emb/episode_000001.pt",
  "episode_index": 1,
  "start_frame": 0,
  "end_frame": 80
}
```

### 4.3 视频与 state shape

加载流程在 `diffsynth/core/data/unified_dataset.py` 和 `diffsynth/core/data/operators.py` 中：

- `LoadVideo` 读取 `[start_frame, end_frame]`，不足时可按 `repeat_last` 补齐。
- `ImageCropAndResize` 支持 `resize_mode=fit` 或 `crop`。
- `ToVideoTensor` 输出 `(V, C, T, H, W)`，值域为 `[-1, 1]`。
- `LoadDroidState` 读取 `state_pose_7d`，输出 `(1, T, 7)`。
- `ResolvePromptEmbPath` 只解析 prompt embedding 路径，真正加载在 prompt embedder 中完成。

`state_pose_7d` 支持三种 parquet 结构：

- DROID 列：`observation.state.cartesian_position` + `observation.state.gripper_position`
- LeRobot pose 列：`observation.cartesian_position` + `observation.gripper_position`
- 26D `observation.state`，取右臂 pose/gripper 子集 `[19:26]`

### 4.4 数据检查命令

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2

$PYTHON_BIN - <<'PY'
import json, os
root = "/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta"
manifest = root + "/meta/episodes_cross_view_train_81_small16567.jsonl"
row = json.loads(open(manifest).readline())
print(row.keys())
for item in row["video"]:
    print(os.path.exists(os.path.join(root, item["data"])), item["data"])
print(os.path.exists(os.path.join(root, row["state"]["data"])), row["state"]["data"])
print(os.path.exists(os.path.join(root, row["prompt_emb"])), row["prompt_emb"])
PY
```

## 5. Cache 构建

训练可以直接从原始视频读数据，但实际推荐先构建 cache。主 cache 保存 VAE latent、history latent、legacy image branch、state、prompt embedding 路径和元信息；geometry sidecar 保存 camera-aware scene tokens 与 target camera tokens。两者分开构建更灵活：主 cache 只依赖 WAN/VAE/CLIP，sidecar 依赖 LagerNVS 与相机参数。

### 5.1 主 cache 内容

`tool/build_cross_view_latent_cache.py` 每个样本保存一个 `.pth`：

| key | shape/类型 | 作用 |
| --- | --- | --- |
| `latent_views_gt` | `(V, 16, T_lat, H_lat, W_lat)` | 每个视角独立 VAE encode 的 GT latent |
| `target_history_latents` | `(1, 16, T_hist, H_lat, W_lat)` | stage2 目标视角已知历史 latent |
| `cond_history_latents` | `(1, 16, T_hist, V*H_lat, W_lat)` | stage1 联合视角历史条件 latent |
| `y` | `(1, 20, T_lat, V*H_lat, W_lat)` 或 target-only | 旧 image branch 的 mask + VAE 条件。**方案 A 启用 dual-end 时**，`y` 中 mask 通道在 head 与 tail 的 latent slot 都置 1，VAE 通道是整段 81 像素帧（含 head/tail 合成帧）一次性 encode 的结果 |
| `clip_feature` | `(1, 257, 1280)` | CLIP 首帧语义条件 |
| `state` | `(1, T, 7)` | 机器人状态条件 |
| `prompt_emb` | path 或 tensor | 文本条件 |
| `scene_tokens` | `(1, N, 768)` 可选 | 主 cache 内 zero-camera scene tokens |
| `source_first_frames` | `(V_src, 3, H, W)` | 给 sidecar/runtime scene token 提取使用 |
| `height,width,num_frames` | int | cache 配置检查 |

对于 `T=81`，VAE 时间 latent 长度通常为：

```python
T_lat = (81 - 1) // 4 + 1  # 21
```

`cache_config.json` 字段：除常规 `num_frames` / `cross_view_source_views` 等外，方案 A 加入 `cross_view_use_tail_anchor`、`num_tail_frames`、`cross_view_tail_anchor_dropout`、`tail_anchor_segment_stride`，训练时由 `validate_cross_view_cache_config` 严格匹配。

### 5.2 主 cache 一键命令

如果已经完成 cache，不要设置 `FORCE_REBUILD_CACHE=1`。`FORCE_REBUILD_CACHE=0` 时脚本会复用完整 cache；不完整时会在 `--skip-existing` 下跳过已有 `.pth` 继续补齐。

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2

PYTHON_BIN=/env/conda/envs/studio/bin/python \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_DIR=/data_ywj/data_xh/projects/datasets/PAI \
DATASET_META_ROOT=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta \
TRAIN_MANIFEST=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/episodes_cross_view_train_81_small16567.jsonl \
VAL_MANIFEST=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/episodes_cross_view_val_81_small200.jsonl \
STATE_STAT_PATH=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/stat_state_pose_7d.json \
CACHE_ROOT=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/cache_crossview_81f_180x320_main \
WRIST_FIRST_FRAME_INDEX=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/wrist_first_frame_index_all.json \
CACHE_NUM_SHARDS=8 \
CACHE_SHARD_MODE=strided \
CACHE_NUM_WORKERS=1 \
FORCE_REBUILD_CACHE=0 \
RUN_TRAIN_AFTER_CACHE=0 \
bash bash/train_droid_success_high_quality_crossview_cache.sh
```

### 5.3 主 cache 参数说明

| 参数/环境变量 | 默认值 | 建议 | 说明 |
| --- | --- | --- | --- |
| `CACHE_NUM_SHARDS` | `8` | GPU 数或略少 | 并行构建进程数 |
| `CACHE_SHARD_MODE` | `strided` | 续跑用 `strided` | `strided` 均衡跳过已有样本，`contiguous` 减少视频重复解码 |
| `CACHE_NUM_WORKERS` | `0` | `0` 或 `1` | 每个 shard 的 DataLoader worker，过高会导致 CPU/IO 压力 |
| `CACHE_VAE_TILED_ENCODE` | `0` | 通常 `0` | 180x320 下 tiled encode 多数更慢，只用于显存兜底 |
| `CACHE_SKIP_LEGACY_BRANCH` | `0` | 通常 `0` | 只有训练时 `CROSS_VIEW_DISABLE_LEGACY_IMAGE_BRANCH=1` 才可跳过 |
| `WRIST_FIRST_FRAME_INDEX` | 空 | 有 LagerNVS 合成首帧时设置 | 替换 wrist target frame0 条件 |
| `SCENE_TOKEN_CHECKPOINT` | 空 | 主 cache 一般不设置 | 若设置，主 cache 提取的是 zero-camera scene tokens |

### 5.4 Geometry sidecar 内容

`tool/build_cross_view_geometry_sidecar_cache.py` 会读取主 cache 的 `source_first_frames`，再从 DROID parquet 读取相机内外参，并复用 LagerNVS 的相机归一化逻辑生成严格一致的 camera tokens。

每个 sidecar `.pth` 包含：

| key | shape | 作用 |
| --- | --- | --- |
| `source_cam_tokens` | `(1, V_src, 11)` | 源视角首帧相机 token |
| `target_cam_tokens` | `(1, T, 11)` | 目标视角逐帧相机 token |
| `target_cam_tokens_latent` | `(1, T_lat, 11)` | 下采样到 latent 时间长度，给 `target_camera_encoder` |
| `scene_tokens_camera_aware` | `(1, N, 768)` | LagerNVS 使用真实 source camera tokens 提取的 scene tokens |

camera token 语义与 LagerNVS 保持一致：

```text
[pose/FOV 9D, camera_scale, world_points_scale] -> 11D
```

### 5.5 Geometry sidecar 构建命令

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2

CUDA_VISIBLE_DEVICES=0 \
/env/conda/envs/studio/bin/python tool/build_cross_view_geometry_sidecar_cache.py \
  --dataset_base_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta \
  --main_cache_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/cache_crossview_81f_180x320_main \
  --train_metadata_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/episodes_cross_view_train_81_small16567.jsonl \
  --val_metadata_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/episodes_cross_view_val_81_small200.jsonl \
  --output_root /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/geometry_sidecar_lagernvs_strict_iter060000 \
  --scene_token_checkpoint /data_ywj/data_xh/projects/DiffSynth-Studio_v2/lagernvs/ckpt/droid_base_stage2/checkpoint_0060000.pt \
  --lagernvs_root /data_ywj/data_xh/projects/LagerNVS \
  --cross_view_source_views 0,1 \
  --cross_view_target_view 2 \
  --num_frames 81 \
  --num_history_frames 1 \
  --device cuda \
  --dtype bf16 \
  --skip-existing
```

### 5.6 多卡 sidecar 示例

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2
mkdir -p logs/sidecar_8shards

for shard in $(seq 0 7); do
  gpu=$((shard % 8))
  CUDA_VISIBLE_DEVICES=$gpu \
  /env/conda/envs/studio/bin/python tool/build_cross_view_geometry_sidecar_cache.py \
    --dataset_base_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta \
    --main_cache_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/cache_crossview_81f_180x320_main \
    --train_metadata_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/episodes_cross_view_train_81_small16567.jsonl \
    --val_metadata_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/episodes_cross_view_val_81_small200.jsonl \
    --output_root /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/geometry_sidecar_lagernvs_strict_iter060000 \
    --scene_token_checkpoint /data_ywj/data_xh/projects/DiffSynth-Studio_v2/lagernvs/ckpt/droid_base_stage2/checkpoint_0060000.pt \
    --lagernvs_root /data_ywj/data_xh/projects/LagerNVS \
    --num_shards 8 \
    --shard_index $shard \
    --skip-existing \
    > logs/sidecar_8shards/shard_${shard}_gpu_${gpu}.log 2>&1 &
done

wait
```

### 5.7 Dual-end anchor 主 cache 重建（方案 A）

当 stage2 启用 dual-end anchor (`CROSS_VIEW_USE_TAIL_ANCHOR=1`) 时，cache 必须重新构建——`y` 通道需要包含 tail 锚帧（mask + VAE 编码），旧 head-only cache 无法直接复用。`validate_cross_view_cache_config` 会显式拒绝 mismatch（"Cached dataset was built with cross_view_use_tail_anchor=False ... but training requested dual-end anchoring"）。

cache 重建命令（先构 cache，不训练）：

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2

PYTHON_BIN=/env/conda/envs/studio/bin/python \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TASK=cross_view_stage1 \
TAG=droid_planA_cache_build \
MODEL_DIR=/data_ywj/data_xh/projects/datasets/PAI \
DATASET_META_ROOT=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta \
TRAIN_MANIFEST=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/episodes_cross_view_train_81_small16567.jsonl \
VAL_MANIFEST=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/episodes_cross_view_val_81_small200.jsonl \
STATE_STAT_PATH=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/stat_state_pose_7d.json \
WRIST_FIRST_FRAME_INDEX=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/wrist_first_frame_index_all.json \
CACHE_ROOT=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/cache_crossview_81f_180x320_lagernvs_iter060001_planA \
SCENE_TOKEN_CHECKPOINT=/data_ywj/data_xh/projects/DiffSynth-Studio_v2/lagernvs/ckpt/droid_base_stage2/checkpoint_0060000.pt \
BUILD_CACHE=1 \
RUN_TRAIN_AFTER_CACHE=0 \
FORCE_REBUILD_CACHE=1 \
CACHE_NUM_SHARDS=8 \
CACHE_SHARD_MODE=strided \
CROSS_VIEW_USE_TAIL_ANCHOR=1 \
NUM_TAIL_FRAMES=1 \
CROSS_VIEW_TAIL_ANCHOR_DROPOUT=0.5 \
bash bash/train_droid_success_high_quality_crossview_cache.sh
```

要点：

- `CACHE_ROOT` 起新名（`..._planA` 后缀）避免覆盖现有 v0 cache，方便回退对照。
- `RUN_TRAIN_AFTER_CACHE=0` 表示只构 cache 不训练；构完用 §7.3 命令训。
- `CROSS_VIEW_TAIL_ANCHOR_DROPOUT=0.5` **必须在 cache 构建阶段传入**——cached 训练路径下 cache 中的 `y` 字段已经固化，训练时不会再做 dropout。设 0.5 让 cache 中 50% 样本 tail = next-synth、50% = zero placeholder，模型学软依赖。
- cache 构建过程会调用 `_load_first_frame_image` 取本段首帧，再调用 `_lookup_wrist_next_segment_first_frame_path(start_frame_offset=81)` 取下一段首帧；末段查不到时退回 zero placeholder（与训练分布一致）。
- 末尾 cache 文件夹会写入 `cache_config.json`，含 `cross_view_use_tail_anchor: true / num_tail_frames: 1 / cross_view_tail_anchor_dropout: 0.5`，训练时 `validate_cross_view_cache_config` 据此严格匹配。

构 cache 期间 GPU 一直跑 VAE encode，速度受 IO 与显存影响，预估 8 卡 16567+200 样本 ~2-3 小时。完成后 §7.3 stage2 训练命令把 `CACHE_ROOT` 指到这个新目录。

## 6. 模型架构

### 6.1 WAN 视频扩散主干

`WanVideoPipeline` 在 `diffsynth/pipelines/wan_video.py` 中定义，核心组件包括：

| 组件 | 类/函数 | 作用 |
| --- | --- | --- |
| DiT | `WanModel` | 对视频 latent token 做 flow/noise 预测 |
| VAE | `WanVideoVAE` | 像素视频与 latent 的双向转换 |
| Text | `WanTextEncoder` 或 `text:emb` | prompt 条件 |
| Image | `WanImageEncoder` | CLIP 首帧语义条件 |
| Action/State | `WanVideoActionEncoder` | 机器人状态/动作条件注入 |
| Scheduler | `FlowMatchScheduler("Wan")` | 加噪、target、training weight |

`model_fn_wan_video()` 的主流程：

```text
latents + timestep
  -> time embedding / AdaLN modulation
  -> optional action embedding
  -> optional y(mask+VAE image condition)
  -> patchify 3D latent volume
  -> optional target_camera_emb
  -> text / CLIP / source / scene cross-attention
  -> DiT blocks with RoPE
  -> unpatchify
  -> predicted flow/noise
```

### 6.2 Cross-view stage1

Stage1 把多个视角的 VAE latent 沿高度维拼接：

```python
latent_views_gt: (V, C, T_lat, H_lat, W_lat)
joint_latent:    (1, C, T_lat, V * H_lat, W_lat)
```

训练目标是联合重建所有视角未来 latent。目标 wrist 第一帧会被 placeholder 或 LagerNVS 合成首帧替换，作为 image branch 条件。Stage1 的作用是让 WAN DiT 适应多视角拼接空间和机器人状态条件。

Stage1 forward 关键逻辑位于：

```text
examples/wanvideo/model_training/train.py
  -> forward_cross_view()
  -> encode_joint_video_latents()
  -> cross_view_weighted_loss()
```

### 6.3 Cross-view stage2

Stage2 从 stage1 checkpoint 初始化，但训练目标变为只预测目标视角：

```python
input_latents_gt = select_target_latents(latent_views_gt)
source_x0_latents = select_source_latents(latent_views_gt)
# Plan A: head/tail anchor 通过 cond_video 像素帧 + WAN-Fun-InP y 通道注入
cond_video[wrist, :, 0]  = LagerNVS_synth(current_segment_first)
cond_video[wrist, :, -1] = LagerNVS_synth(next_segment_first)  # 启用 dual-end 时
```

Stage2 新增模块：

| 模块 | 默认大小 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- | --- |
| `source_video_projector` | `16 -> 512 -> 1536` Conv3D | `(B,Vsrc,16,Tlat,Hlat,Wlat)` | temporal: `(B,Tp,Nsrc,1536)` | 把源视角 latent 编成可 cross-attend 的 source memory |
| `source_temporal_gate` | scalar 或 state MLP | source memory + state | gated source memory | 控制源视角条件强度 |
| `target_state_head` | `1536 -> 1536 -> 7` | DiT hidden by time | `(B,T,7)` | 辅助预测状态，提升动作/状态一致性 |
| `target_camera_encoder` | `11 -> 1536 -> 1536` | `(B,Tlat,11)` | `(B,Tlat,1536)` | 当 `geometry_target_camera_mode=add_time_mlp` 时把目标相机注入 token |
| `scene_token_adapter` | `768 -> 1536` | `(B,N,768)` | `(B,Npool,1536)` | 将 LagerNVS scene tokens 对齐 DiT 维度 |
| `geometry_gates` | 每个 DiT block 一个 MLP | timestep embedding | gate_scene/gate_source | 动态调节 scene/source 条件 |

Stage2 训练时，source 分支不是预测目标，而是作为条件：

```text
source_latents
  -> source_video_projector
  -> source_temporal_gate
  -> source_memory_by_time
  -> DiT block cross-attention
```

目标分支锚帧注入（方案 A，对齐 WAN-Fun-InP）：

```text
cond_video[wrist, :, 0]  = LagerNVS 合成首帧
cond_video[wrist, :, -1] = 下一段 LagerNVS 合成首帧 (启用 dual-end 时)
  -> WanVideoUnit_ImageEmbedderVAE 整段 81 帧 VAE encode
  -> y 通道 (mask 4 + latent 16) → DiT 36 通道输入
loss 监督整段 noise prediction (head/tail 锚帧也参与 loss)
```

旧版本（v0 dual-anchor）走 latent slot overwrite + loss-mask 范式，已被替换为方案 A。

### 6.4 Geometry-aware attention

当设置：

```bash
SCENE_TOKEN_CHECKPOINT=...
GEOMETRY_SCENE_TOKEN_SOURCE=camera_aware_sidecar
GEOMETRY_SIDECAR_CACHE_PATH=...
```

训练会从 sidecar 读取 `scene_tokens_camera_aware`，经 `SceneTokenAdapter` 投到 DiT 维度，再在每个 DiT block 的 cross-attention 中与 text/source context 拼接。

`WanAttentionBlock.geometry_aware_cross_attn()` 逻辑：

```text
context = [text_context, clip_context, gated_scene_tokens, gated_source_memory]
query   = current target video tokens
key/value = context
```

如果启用 `GEOMETRY_TARGET_CAMERA_MODE=add_time_mlp`，`target_cam_tokens_latent` 会经过 `CrossViewTargetCameraEncoder`，在 patchify 后加到 target latent token grid 上：

```python
target_camera_emb: (B, T_patch, D)
x = x + rearrange(target_camera_emb, "b f d -> b d f 1 1")
```

### 6.5 架构调试示例

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2

grep -n "class CrossViewTargetCameraEncoder" examples/wanvideo/model_training/train.py
grep -n "class CrossViewSourceVideoProjector3DTemporal" diffsynth/models/cross_view_projector.py
grep -n "def geometry_aware_cross_attn" diffsynth/models/wan_video_dit.py
```

## 7. 训练流程

### 7.1 通用训练输出

训练由 `accelerate` 启动。`ModelLogger` 会保存：

```text
OUTPUT_PATH/
├── config.json
├── epoch-0/
│   ├── config.json
│   └── epoch-0.safetensors
├── epoch-1/
│   ├── config.json
│   └── epoch-1.safetensors
└── ...
```

`ckpt_path` 是“模型权重初始化”，`resume_from` 是 accelerate 完整训练状态恢复。二者不能同时使用。

即使使用 cached dataset，仍建议在最终训练命令或配置中保留 `state_stat_path`。原因是推理脚本会从 `config.json` 恢复归一化统计路径，再读取 raw manifest 生成评估样本；如果 `config.json` 中该字段为 `null`，stage2 推理可能无法正常加载 state。

### 7.2 Stage1 训练命令

如果 cache 已构建好，直接训练 stage1（多视角联合去噪，head-only InP，不启用 dual-end）：

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2

PYTHON_BIN=/env/conda/envs/studio/bin/python \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TASK=cross_view_stage1 \
TAG=droid_stage1_planA \
MODEL_DIR=/data_ywj/data_xh/projects/datasets/PAI \
DATASET_META_ROOT=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta \
TRAIN_MANIFEST=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/episodes_cross_view_train_81_small16567.jsonl \
VAL_MANIFEST=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/episodes_cross_view_val_81_small200.jsonl \
STATE_STAT_PATH=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/stat_state_pose_7d.json \
WRIST_FIRST_FRAME_INDEX=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/wrist_first_frame_index_all.json \
CACHE_ROOT=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/cache_crossview_81f_180x320_lagernvs_iter060001 \
BUILD_CACHE=0 \
RUN_TRAIN_AFTER_CACHE=1 \
NUM_EPOCHS=7 \
GRAD_ACCUM_STEPS=4 \
LEARNING_RATE=1e-4 \
CROSS_VIEW_USE_TAIL_ANCHOR=0 \
bash bash/train_droid_success_high_quality_crossview_cache.sh
```

stage1 不启用 dual-end，因为 stage1 是多视角联合去噪，wrist 视角与外部两视角都有完整 GT，没有"未来帧"概念。

### 7.3 Stage2 训练命令（方案 A，dual-end y 通道）

使用 stage1 checkpoint 初始化。**方案 A 必须用 §5.7 重建好的 dual-end cache**——cache 中的 `y` 通道已包含 head/tail 双端 mask 与 VAE 编码，训练时 `attach_cached_legacy_image_branch` 直接读取，无须任何 latent overwrite。`validate_cross_view_cache_config` 会校验训练侧 `cross_view_use_tail_anchor` / `num_tail_frames` 与 cache_config 是否一致，mismatch 时直接报错。

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2

PYTHON_BIN=/env/conda/envs/studio/bin/python \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TASK=cross_view_stage2 \
TAG=droid_stage2_planA \
CKPT_PATH=/data_ywj/data_xh/projects/DiffSynth-Studio_v2/Ckpt/droid_stage1_planA/epoch-6/epoch-6.safetensors \
MODEL_DIR=/data_ywj/data_xh/projects/datasets/PAI \
DATASET_META_ROOT=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta \
TRAIN_MANIFEST=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/episodes_cross_view_train_81_small16567.jsonl \
VAL_MANIFEST=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/episodes_cross_view_val_81_small200.jsonl \
STATE_STAT_PATH=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/stat_state_pose_7d.json \
WRIST_FIRST_FRAME_INDEX=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/wrist_first_frame_index_all.json \
CACHE_ROOT=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/cache_crossview_81f_180x320_lagernvs_iter060001_planA \
GEOMETRY_SIDECAR_CACHE_PATH=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/geometry_sidecar_lagernvs_strict_iter060000 \
SCENE_TOKEN_CHECKPOINT=/data_ywj/data_xh/projects/DiffSynth-Studio_v2/lagernvs/ckpt/droid_base_stage2/checkpoint_0060000.pt \
GEOMETRY_SCENE_TOKEN_SOURCE=camera_aware_sidecar \
GEOMETRY_USE_CAMERA_TOKENS=1 \
GEOMETRY_TARGET_CAMERA_MODE=add_time_mlp \
BUILD_CACHE=0 \
RUN_TRAIN_AFTER_CACHE=1 \
NUM_EPOCHS=10 \
GRAD_ACCUM_STEPS=8 \
LEARNING_RATE=8e-5 \
CROSS_VIEW_USE_TAIL_ANCHOR=1 \
NUM_TAIL_FRAMES=1 \
CROSS_VIEW_TAIL_ANCHOR_DROPOUT=0.0 \
bash bash/train_droid_success_high_quality_crossview_cache.sh
```

> **方案 A 关键点**
>
> - `cond_video[wrist, :, 0]` = LagerNVS 合成首帧；`cond_video[wrist, :, -1]` = 下一段合成首帧（DROID stride=81，由 `_load_wrist_next_segment_first_frame` 查 `wrist_first_frame_index[f"{ep}_{sf+81}"]`）；末段或 dropout 触发时退回 zero placeholder。
> - DiT 通过 36 通道输入接收 `[noisy_latent(16), y_channel(20)]`：mask 通道在 head 与 tail 的 latent slot 都置 1，VAE encode 一次完整 81 像素帧得到 slot 自洽的语义。
> - 没有任何 latent slot overwrite。stage2 loss 监督**整段** noise prediction（不再切 `[history_t : -tail_t]`），head/tail 锚帧位置也参与 loss——这与 WAN-Fun-InP 原版训练目标完全一致。
> - **dropout 只在 cache 构建时生效**（cached 训练路径直接读取 cache 中的 `y` 字段，不会再过 `WanVideoUnit_ImageEmbedderVAE`）。所以训练命令里 `CROSS_VIEW_TAIL_ANCHOR_DROPOUT=0.0`，真正的 dropout 概率应该在 §5.7 cache 重建命令里设为 0.5。
> - `validate_cross_view_cache_config` 会严格匹配训练侧 `cross_view_use_tail_anchor` / `num_tail_frames` 与 cache_config 的对应字段，mismatch 直接报错。
> - 旧 dual-anchor ckpt（v0 latent-overwrite 范式）不可直接接续——它的权重适应了 latent slot 0 / slot 20 都是 clean anchor 的分布，方案 A 下这两个 slot 改为 noisy GT。建议从 stage1 重新训练 stage2，或从未启用 dual-end 的 stage2_sidecar epoch-N 接续训。
> - 第一个 epoch loss 可能短暂上升（500-1000 step）然后下降，属于分布迁移正常现象。如果一直不降，应回退到 stage1 重训。

> **常见坑：训练时漏传 `WRIST_FIRST_FRAME_INDEX` / `STATE_STAT_PATH`**
>
> 即使 cache 里 `target_history_latents` 已经是 `VAE(LagerNVS_synth)`、`stat` 已写入磁盘，训练时不传这两个变量会导致 `config.json` 把对应字段保存为 `null`。后续推理脚本（§8）会因此分别表现为「wrist 首帧整张灰」和「`KeyError: missing normalization stats`」。
>
> 方案 A 还多一层依赖：dual-end 训练在 `build_cross_view_condition_video` 调用 `_load_wrist_next_segment_first_frame(meta, ...)` 查下一段合成帧。如果 `WRIST_FIRST_FRAME_INDEX` 没传，**所有非末段样本的 tail anchor 都退回 zero placeholder**，训练分布与单端无异。务必传齐。
>
> 解决：训练命令始终把这两个变量加上，让 `config.json` 写正确路径。当前推理脚本即使 config 中为 null 也会 fallback 到 `${DATASET_META_ROOT}/meta/{wrist_first_frame_index_all,stat_state_pose_7d}.json`，但前提是文件确实存在于那个位置。

### 7.4 训练参数速查

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `TASK` | `cross_view_stage1` | `cross_view_stage1` 或 `cross_view_stage2` |
| `NUM_FRAMES` | `81` | 视频帧数，要求满足 WAN 时间下采样 |
| `HEIGHT/WIDTH` | `180/320` | 输入分辨率 |
| `NUM_HISTORY_FRAMES` | `1` | cross-view 当前要求为 1 |
| `LOAD_MODULES` | `dit,text:emb,vae,image,action:noise` | 使用 prompt embedding、VAE、CLIP、action/state encoder |
| `CROSS_VIEW_SOURCE_VIEWS` | `0,1` | 条件源视角 |
| `CROSS_VIEW_TARGET_VIEW` | `2` | 目标 wrist 视角 |
| `CROSS_VIEW_PLACEHOLDER_MODE` | `zeros` | 目标首帧占位方式 |
| `CROSS_VIEW_USE_TAIL_ANCHOR` | `0` | **方案 A** dual-end 总开关；stage2 推荐 `1`，stage1 保持 `0` |
| `NUM_TAIL_FRAMES` | `1` | 尾锚像素帧数（latent slot 数 = `((N-1)//4)+1`） |
| `CROSS_VIEW_TAIL_ANCHOR_DROPOUT` | `0.0` | tail 像素帧被 zero placeholder 替换的概率。**只在 cache 构建（`BUILD_CACHE=1`）时生效**——cached 训练路径直接读取 cache 中的 `y` 字段，不会再过 `WanVideoUnit_ImageEmbedderVAE`。训练阶段命令里固定 `0.0`，真正想 0.5 的话在 §5.7 cache 重建时传。 |
| `CROSS_VIEW_SOURCE_LOSS_WEIGHT` | `0.8` in bash | stage1 source auxiliary loss 权重 |
| `CROSS_VIEW_OLD_BRANCH_DROPOUT` | `0.5` in bash | stage2 legacy image branch dropout |
| `CROSS_VIEW_SOURCE_INJECTION_MODE` | `temporal_local` | 按时间局部注入 source memory |
| `CROSS_VIEW_SOURCE_BRANCH_MODE` | `sigma_matched_clamp` | source latent 分支按 timestep 加噪匹配 |
| `CROSS_VIEW_SOURCE_GATE_MODE` | `scalar` | source memory gate |
| `CROSS_VIEW_TEMP_LOSS_WEIGHT` | `0.1` | stage2 temporal consistency loss |
| `CROSS_VIEW_STATE_LOSS_WEIGHT` | `0.05` | stage2 state prediction auxiliary loss |
| `SCENE_TOKEN_POOL_SIZE` | `512` | scene token pooling 数 |
| `GEOMETRY_GATE_MODE` | `learned` | scene/source gate 类型 |
| `GEOMETRY_TARGET_CAMERA_MODE` | `none` | `add_time_mlp` 才启用 target camera encoder |
| `GEOMETRY_SCENE_TOKEN_SOURCE` | `cached_zero_cam` | 推荐 sidecar 时设为 `camera_aware_sidecar` |
| `ALIGNMENT_LOSS_WEIGHT` | `0.1` | hidden 与 scene token 对齐 loss |

### 7.5 断点与初始化

从 stage1 权重初始化 stage2：

```bash
CKPT_PATH=/path/to/stage1/epoch-6.safetensors
```

从 accelerate 状态恢复完整训练：

```bash
/env/conda/envs/studio/bin/python -m accelerate.commands.launch \
  examples/wanvideo/model_training/train.py \
  --resume_from /path/to/output/epoch-3 \
  ...
```

不要同时传 `--ckpt_path` 和 `--resume_from`。

## 8. 推理与评估

### 8.1 Stage2 推理（多卡，推荐）

`infer_cross_view_stage2.py` 在 stage2 中以 **target-only**（仅腕部视角）模式运行，每次只 denoise 目标视角的 latent，输入条件包含 source memory（外部视角 VAE latent）、scene tokens、target camera tokens、首帧 anchor。`bash/infer_stage2_multi_gpu.sh` 用 8 个进程把 val 样本平均分到 8 张卡，最后跑一次聚合 pass 计算指标。每个 shard 只处理自己负责的样本（`idx % num_shards == shard_index`），生成的视频文件用全局 idx 命名，因此**多 shard 同时写同一目录不会冲突**；聚合 pass 对已存在的视频幂等跳过，只算指标。

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2

CKPT_PATH=/data_ywj/data_xh/projects/DiffSynth-Studio_v2/Ckpt/droid_success_high_quality_crossview_stage2_sidecar/epoch-0/epoch-0.safetensors \
CONFIG_JSON=/data_ywj/data_xh/projects/DiffSynth-Studio_v2/Ckpt/droid_success_high_quality_crossview_stage2_sidecar/epoch-0/config.json \
OUTPUT_DIR=/data_ywj/data_xh/projects/DiffSynth-Studio_v2/Ckpt/droid_success_high_quality_crossview_stage2_sidecar/epoch-0/stage2_eval_8gpu \
GEOMETRY_SIDECAR_CACHE_PATH=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/geometry_sidecar_lagernvs_strict_iter060000 \
DATASET_METADATA_PATH=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/episodes_cross_view_val_81_small200.jsonl \
STATE_STAT_PATH=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/stat_state_pose_7d.json \
WRIST_FIRST_FRAME_INDEX=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/wrist_first_frame_index_all.json \
SAMPLE_LIMIT=200 \
GPUS=0,1,2,3,4,5,6,7 \
CFG_SCALE=1.0 \
NUM_INFERENCE_STEPS=50 \
SKIP_TRAIN_PREVIEW=1 \
bash bash/infer_stage2_multi_gpu.sh
```

可覆盖的环境变量（默认值见脚本顶部）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `CKPT_PATH` | `Ckpt/.../epoch-0.safetensors` | Stage2 ckpt 路径 |
| `CONFIG_JSON` | `Ckpt/.../config.json` | 训练时保存的 grouped config，用来恢复模型结构 |
| `OUTPUT_DIR` | `${CKPT 同级}/stage2_eval_8gpu` | 多卡推理总输出目录 |
| `GEOMETRY_SIDECAR_CACHE_PATH` | sidecar 路径 | `geometry_scene_token_source=camera_aware_sidecar` 时必填 |
| `DATASET_METADATA_PATH` | val small200 manifest | 推理用 jsonl |
| `STATE_STAT_PATH` | `${DATASET_BASE_PATH}/meta/stat_state_pose_7d.json` | DROID state 归一化统计；config.json 中为 null 时必须显式提供 |
| `WRIST_FIRST_FRAME_INDEX` | `${DATASET_BASE_PATH}/meta/wrist_first_frame_index_all.json` | LagerNVS 合成首帧索引；缺失时 wrist 视角首帧退回 zero placeholder（视觉上为灰），与训练分布不匹配 |
| `SAMPLE_LIMIT` | `200` | 推理样本数上限（与 manifest 长度取小） |
| `GPUS` | `0,1,2,3,4,5,6,7` | 用逗号分隔的 GPU id 列表，决定 shard 数 |
| `CFG_SCALE` | `1.0` | 1.0 等价于训练时 cfg_merge=False |
| `NUM_INFERENCE_STEPS` | `50` | flow matching 去噪步数 |
| `SKIP_TRAIN_PREVIEW` | `1` | 跳过 train preview，纯 val 推理时建议保留 |

> **重要：训练-推理一致性**
>
> 训练命令如果没有显式传 `WRIST_FIRST_FRAME_INDEX` 与 `STATE_STAT_PATH`，保存到 `config.json` 的对应字段会是 `null`。即使 cache 里 `target_history_latents` 已经是 `VAE(LagerNVS_synth)`，推理脚本仍会因为 `model.wrist_first_frame_index = None` 而在 `cond_video[wrist, 0]` 上回退到 zero placeholder——表现为生成视频的 wrist 视角首帧整体偏灰。
>
> 解决方法有两种：
> - **推荐**：训练时直接把这两个变量加进 bash 命令，让 config 里写正确路径
> - **当前 ckpt 兜底**：launcher 已经默认从 `${DATASET_BASE_PATH}/meta/` 找两个文件，无需重训
>
> 见 §8.3 训练-推理一致性要点。

输出：

```text
${OUTPUT_DIR}/
├── shard_logs/                    # 每张卡 + aggregator 的日志
│   ├── shard_0_gpu_0.log ... shard_7_gpu_7.log
│   └── aggregator.log
├── comparisons/val/               # 共 SAMPLE_LIMIT 个 mp4 (3 视角 × 2 列 GT|Pred)
│   ├── val_000_ep12968.mp4
│   └── ...
├── config_eval.json               # 推理 EvalConfig 快照
├── metrics_shard0.json ...        # 每 shard 的元信息（不含指标，仅 shard_info）
└── metrics.json                   # 聚合 pass 计算的最终指标（PSNR/SSIM/LPIPS/FVD by view）
```

> **关于「3 视角输出」**：`VideoSaver.save_comparison` 始终按 `V 行 × 2 列 (GT|Pred)` 网格输出。stage2 模型只 denoise wrist 单视角，但保存视频时 source 视角直接复制 GT，target 视角嵌入预测。这是评估和肉眼对比的便利设计，不是模型行为异常。看 `metrics.json` 时只关注 `target_view` 字段的 PSNR/SSIM/LPIPS/FVD（其他视角因为 GT vs GT，PSNR 会是无穷大）。

加速预期：单卡 50 步 / sample 大约 30 秒，200 样本单卡 ~100 分钟；8 卡并行总耗时 ~15-20 分钟（含聚合阶段的指标计算）。约 6-7× 加速。

#### 8.1.1 Dual-end anchor（双端锚定）评估 — 方案 A

如果 ckpt 是用 `CROSS_VIEW_USE_TAIL_ANCHOR=1` 训出的（`config.json` 中 `cross_view_use_tail_anchor: 1`），方案 A 下推理脚本不再做 latent 覆盖，而是：

- `cond_video[wrist, :, 0]` = 当前段 LagerNVS 合成首帧（沿用旧逻辑）
- `cond_video[wrist, :, -1]` = **下一段** LagerNVS 合成首帧（DROID stride=81 处理）；末段退回 zero placeholder
- DiT 通过 36 通道输入吃 `[noisy_latent, y_channel]`，y 通道由 `WanVideoUnit_ImageEmbedderVAE` 整段 encode 81 像素帧得到，mask 通道在 head 与 tail 的 latent slot 都置 1
- denoise 循环不做任何 latent slot overwrite——anchor 信号完全由 y 通道软引导

**重要**：旧 v0 dual-anchor ckpt（用 latent overwrite 训出的）**不能**用方案 A 推理，行为分布不一致。需要用方案 A 重训 stage2（见 §7.3）。

可用的 manifest：

| manifest | 行数 | 用途 |
| --- | --- | --- |
| `episodes_cross_view_val_81_small200.jsonl` | 674 | 完整 val（含每个 ep 的最末段） |
| `episodes_cross_view_val_81_small200_without_last_chunk.jsonl` | 479 | 剔除末段，每个样本都有非零 tail anchor |

方案 A 推理评估命令：

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2

CKPT_DIR=/data_ywj/data_xh/projects/DiffSynth-Studio_v2/Ckpt/droid_stage2_planA/epoch-9
DATA_BASE=/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta

CKPT_PATH="$CKPT_DIR/epoch-9.safetensors" \
CONFIG_JSON="$CKPT_DIR/config.json" \
OUTPUT_DIR="$CKPT_DIR/stage2_eval_8gpu_planA" \
DATASET_BASE_PATH="$DATA_BASE" \
DATASET_METADATA_PATH="$DATA_BASE/meta/episodes_cross_view_val_81_small200.jsonl" \
GEOMETRY_SIDECAR_CACHE_PATH="$DATA_BASE/geometry_sidecar_lagernvs_strict_iter060000" \
WRIST_FIRST_FRAME_INDEX="$DATA_BASE/meta/wrist_first_frame_index_all.json" \
STATE_STAT_PATH="$DATA_BASE/meta/stat_state_pose_7d.json" \
SAMPLE_LIMIT=2000 CFG_SCALE=1.0 NUM_INFERENCE_STEPS=50 \
GPUS=0,1,2,3,4,5,6,7 SKIP_TRAIN_PREVIEW=1 \
PYTHON_BIN=/env/conda/envs/studio/bin/python \
bash bash/infer_stage2_multi_gpu.sh
```

方案 A 不需要 `TAIL_ANCHOR_SEGMENT_INDEX_MANIFEST` —— SegmentIndex 不再使用，下一段合成首帧通过 `wrist_first_frame_index[f"{ep}_{sf+81}"]` 直接查表。完整 manifest 也不需要剔除末段——末段的 tail pixel 退回 zero placeholder 是训练时见过的分布。

**Ablation 开关**：`DISABLE_TAIL_ANCHOR_AT_INFERENCE=1` 强制 `num_tail_frames=0`，让同一个 dual-end ckpt 跑 head-only InP 推理。可用于消融对比"双端 vs 单端"的真实增益：

```bash
DISABLE_TAIL_ANCHOR_AT_INFERENCE=1 \
OUTPUT_DIR="$CKPT_DIR/stage2_eval_8gpu_planA_NO_TAIL" \
... (其他参数同上) \
bash bash/infer_stage2_multi_gpu.sh
```

向后兼容：旧 ckpt（`config.json` 中无 `cross_view_use_tail_anchor` 或为 `0`）→ `num_tail_frames=0` → `WanVideoUnit_ImageEmbedderVAE` 退化为原 head-only 行为，bit-for-bit 等价。

### 8.2 Stage2 单卡推理（小样本快速验证）

如果只想快速 sanity check（例如 SAMPLE_LIMIT=4），仍可用单卡命令：

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2

CUDA_VISIBLE_DEVICES=0 \
/env/conda/envs/studio/bin/python examples/wanvideo/model_inference/infer_cross_view_stage2.py \
  --ckpt_path /data_ywj/data_xh/projects/DiffSynth-Studio_v2/Ckpt/droid_stage2_planA/epoch-9/epoch-9.safetensors \
  --config_json /data_ywj/data_xh/projects/DiffSynth-Studio_v2/Ckpt/droid_stage2_planA/epoch-9/config.json \
  --dataset_base_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta \
  --dataset_metadata_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/episodes_cross_view_val_81_small200.jsonl \
  --geometry_sidecar_cache_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/geometry_sidecar_lagernvs_strict_iter060000 \
  --state_stat_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/stat_state_pose_7d.json \
  --wrist_first_frame_index /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/wrist_first_frame_index_all.json \
  --output_dir /data_ywj/data_xh/projects/DiffSynth-Studio_v2/Ckpt/stage2_eval_single_gpu \
  --cfg_scale 1.0 \
  --num_inference_steps 50 \
  --sample_limit 4 \
  --skip_train_preview
```

新增 CLI 参数（多卡脚本会用，单卡通常不需要传）：

| 参数 | 含义 |
| --- | --- |
| `--num_shards N` | 总分片数；多卡时由 launcher 传递 |
| `--shard_index i` | 当前分片号 [0, N)；本进程只处理 `idx % N == i` 的样本 |
| `--skip_metrics` | 多卡阶段每个 shard 都加这个，避免每张卡重复在不完整集合上算指标 |
| `--skip_train_preview` | 跳过 train preview 阶段，纯 val 推理常用 |
| `--state_stat_path` | DROID state 归一化 JSON；config.json 中为 null 时必填 |
| `--wrist_first_frame_index` | LagerNVS 合成首帧索引；缺失时 wrist 首帧灰，推理与训练分布不匹配 |
| `--disable_tail_anchor_at_inference` | 方案 A ablation：强制 `num_tail_frames=0`，让同一 ckpt 跑 head-only InP 推理 |

### 8.3 训练-推理一致性要点

stage2 推理与训练严格对齐 target-only 模式（方案 A，y 通道双端 anchor）：

- DiT 输入 latent shape `(1, 16, T_lat, H_lat, W_lat)`（**单视角**，不再是 stage1 的 V·H_lat 联合 grid）
- DiT 36 通道输入：`[noisy_latent(16), y_channel(20)]`；y_channel = `[mask(4), VAE_encoded(16)]`，由 `WanVideoUnit_ImageEmbedderVAE` 整段 encode 81 像素帧得到
- `cond_video[wrist, :, 0]`、`cond_video[wrist, :, -1]` 通过 `meta=data` 透传 `wrist_first_frame_index` → 加载 LagerNVS 合成首帧（当前段 + 下一段）
- source memory、scene tokens、target camera tokens 全部走 sidecar 路径，与训练侧 `forward_cross_view_cached` 完全相同
- denoise 循环**不做任何 latent slot overwrite**——head/tail anchor 完全靠 y 通道软引导，与 WAN-Fun-InP 原版 i2v 范式一致

#### 三个最容易踩的训练-推理坑

每个坑都有一个共同特征：训练命令缺一个参数，模型还是能正常训练（cache 已包含正确数据），但保存到 `config.json` 的对应字段为 `null`，推理脚本读 config 时静默退回到 placeholder。**修复要么补训练命令并重训，要么推理时显式覆盖**。

| 训练命令缺 | config.json 表现 | 推理症状 | 推理端兜底 |
| --- | --- | --- | --- |
| `STATE_STAT_PATH` | `state_stat_path: null` | `LoadDroidState._get_min_max` 抛 `KeyError` | `--state_stat_path` 或 launcher `STATE_STAT_PATH` |
| `WRIST_FIRST_FRAME_INDEX` | `wrist_first_frame_index: null` | wrist 视角首帧 + 末帧整张灰；wrist 整段质量劣化 | `--wrist_first_frame_index` 或 launcher `WRIST_FIRST_FRAME_INDEX` |
| `GEOMETRY_SIDECAR_CACHE_PATH`（且使用 sidecar 模式） | `geometry_sidecar_cache_path: null` | 加载 sidecar 时找不到文件，抛 `FileNotFoundError` | `--geometry_sidecar_cache_path` 显式指定 |

当前推理脚本会按以下顺序解析 `state_stat_path` 与 `wrist_first_frame_index`：

1. 命令行 `--xxx`（最高优先级）
2. `config.json` 中读到的字段
3. fallback 到 `${dataset_base_path}/meta/{stat_state_pose_7d,wrist_first_frame_index_all}.json`
4. 找不到则 fail-fast（state stat）或 silently fall back to placeholder（wrist）

> 推荐做法：训练时显式传齐 `STATE_STAT_PATH` 和 `WRIST_FIRST_FRAME_INDEX`，让 `config.json` 中保存正确路径，未来推理零配置即可。

### 8.4 Stage1 推理

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2

CUDA_VISIBLE_DEVICES=0 \
/env/conda/envs/studio/bin/python examples/wanvideo/model_inference/infer_cross_view_stage1.py \
  --ckpt_path /path/to/stage1/epoch-6/epoch-6.safetensors \
  --config_json /path/to/stage1/epoch-6/config.json \
  --dataset_base_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta \
  --dataset_metadata_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/episodes_cross_view_val_81_small200.jsonl \
  --output_dir /data_ywj/data_xh/projects/DiffSynth-Studio_v2/Ckpt/stage1_eval_example \
  --cfg_scale 1.0 \
  --num_inference_steps 50 \
  --sample_limit 20
```

Stage1 仍是多视角联合去噪（patch 高度 = V·H_lat），目前 `infer_cross_view_stage1.py` 是单卡脚本，多卡需要类似 stage2 的 launcher 包装（暂未提供）。

### 8.5 评估指标

如果已经生成对比视频，可使用：

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2

/env/conda/envs/studio/bin/python tool/evaluate_generated_videos.py \
  --comparison-dir /data_ywj/data_xh/projects/DiffSynth-Studio_v2/Ckpt/droid_success_high_quality_crossview_stage2_sidecar/epoch-0/stage2_eval_8gpu/comparisons/val \
  --output-json /data_ywj/data_xh/projects/DiffSynth-Studio_v2/Ckpt/droid_success_high_quality_crossview_stage2_sidecar/epoch-0/stage2_eval_8gpu/fvd_metrics.json \
  --num-views 3 \
  --metrics fvd,lpips,ssim,psnr \
  --device cuda \
  --sample-limit 200
```

评估脚本默认认为对比视频是 3 行视角、2 列 `GT|Pred` 网格。若输出格式不同，需要调整 `--num-views` 或修改 `split_comparison_grid()`。

### 8.6 批量评估多个 epoch

`bash/infer_all_epoch_stage2.sh` 是模板脚本，但里面有硬编码旧路径，使用前需要改 `BASE_CKPT_DIR`、`DATASET_METADATA_PATH`、`TRAIN_METADATA_PATH` 和 Python 环境。把内层调用替换为 `bash/infer_stage2_multi_gpu.sh` 即可享受多卡加速。

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2
sed -n '1,80p' bash/infer_all_epoch_stage2.sh
```

## 9. 关键算法与实现逻辑

### 9.1 Flow matching 训练目标

训练时流程是：

```python
timestep = sample_training_timestep()
noise = torch.randn_like(input_latents_gt)
latents = scheduler.add_noise(input_latents_gt, noise, timestep)
training_target = scheduler.training_target(input_latents_gt, noise, timestep)
noise_pred = pipe.model_fn(..., latents=latents, timestep=timestep)
loss = mse(noise_pred_future, training_target_future)
```

核心点是 history latent 不参与 loss，模型学习从条件和 noisy future latent 中恢复未来目标。

### 9.2 多视角 latent 拼接

Stage1 使用：

```python
rearrange(latent_views, "v c t h w -> 1 c t (v h) w")
```

优点是不用修改 WAN DiT 的空间结构，只把多视角当作“更高的视频帧”。缺点是视角之间的几何关系没有显式表达，因此 stage2 引入 source memory 和 geometry sidecar 补强。

### 9.3 Stage2 source memory

`CrossViewSourceVideoProjector3DTemporal` 使用两层 Conv3D：

```text
Conv3d(16, 512, stride=(1,2,2))
GELU
Conv3d(512, 1536, stride=(1,2,2))
```

输出 temporal local 格式：

```python
(B, V, C, T, H, W) -> (B, T, V*H'*W', 1536)
```

DiT block 对每个 target 时间片只 attend 附近 source 时间片：

```python
left = max(0, frame_id - source_window_radius)
right = min(num_frames, frame_id + source_window_radius + 1)
```

这样既保留跨视角信息，又避免所有 source tokens 全局拼接导致显存过大。

### 9.4 LagerNVS scene tokens

`SceneTokenExtractor` 包装 LagerNVS `Reconstructor`：

```python
images:    (B, Vsrc, 3, H, W), range [-1,1]
cam_token: (B, Vsrc, 11)
tokens:    (B, Vsrc * P, 768)
```

`SceneTokenAdapter` 做：

```text
LayerNorm(768)
Linear(768 -> 1536)
GELU
Linear(1536 -> 1536)
LayerNorm(1536)
+ type embedding
```

如果 `scene_token_pool_size=512` 且 token 数超过 512，会先 `adaptive_avg_pool1d` 到 512。

### 9.5 Camera token 与 target camera encoder

sidecar 的 camera tokens 是严格按 LagerNVS 逻辑构造：

```text
DROID parquet intr/extr
  -> c2w
  -> normalize_extrinsics，以 source first-frame cameras 为条件相机
  -> pose/FOV encoding 9D
  -> append camera_scale, world_points_scale
  -> 11D camera token
```

`target_camera_encoder` 只有在：

```bash
GEOMETRY_TARGET_CAMERA_MODE=add_time_mlp
```

时创建并训练。它将 `(B,Tlat,11)` 映射到 `(B,Tlat,1536)`，再加到 patchify 后的时间 token 上。

### 9.6 辅助 loss

Stage2 总 loss 由几部分组成：

```text
total_loss =
  main_future_mse
  + temporal_loss_weight * temporal_consistency_loss
  + state_loss_weight * target_state_prediction_loss
  + alignment_loss_weight * geometry_alignment_loss
```

其中：

- `temporal_consistency_loss` 约束预测视频在 latent 时间上更平滑。
- `target_state_head` 从 DiT hidden 预测状态序列，作为辅助监督。
- `geometry_alignment_loss` 让目标 hidden 和 scene tokens 的 pooled 表征更接近。
- warmup 由 `cross_view_aux_loss_warmup_ratio` 和 `alignment_loss_warmup_ratio` 控制。

### 9.7 代码定位示例

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2

grep -n "def compute_geometry_alignment_loss" examples/wanvideo/model_training/train.py
grep -n "def build_cross_view_source_condition" examples/wanvideo/model_training/train.py
grep -n "def build_target_camera_condition" examples/wanvideo/model_training/train.py
```

## 10. 配置与默认参数

### 10.1 parser 默认值

这些默认值来自 `diffsynth/diffusion/parsers.py`：

| CLI | 默认值 | 说明 |
| --- | --- | --- |
| `--learning_rate` | `1e-4` | 学习率 |
| `--num_epochs` | `1` | 训练 epoch 数 |
| `--trainable_models` | `dit` | 训练模块，stage2 会自动追加 projector/gates/head |
| `--mixed_precision` | `bf16` | 混合精度 |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--gradient_accumulation_steps` | `1` | 梯度累积 |
| `--max_grad_norm` | `0.5` | 梯度裁剪 |
| `--dataset_num_workers` | `8` | 训练 DataLoader worker |
| `--num_frames` | `81` | 视频帧数 |
| `--num_history_frames` | `1` | 历史条件帧数 |
| `--resize_mode` | `fit` | resize 策略 |
| `--cross_view_source_views` | `0,1` | 源视角 |
| `--cross_view_target_view` | `2` | 目标视角 |
| `--geometry_scene_token_source` | `cached_zero_cam` | scene token 来源 |

### 10.2 bash 脚本覆盖默认值

`bash/train_droid_success_high_quality_crossview_cache.sh` 对若干参数有项目级默认：

```bash
NUM_FRAMES=81
HEIGHT=180
WIDTH=320
NUM_HISTORY_FRAMES=1
NUM_EPOCHS=1
LEARNING_RATE=1e-4
GRAD_ACCUM_STEPS=4
MIXED_PRECISION=bf16
LOAD_MODULES=dit,text:emb,vae,image,action:noise
CROSS_VIEW_SOURCE_LOSS_WEIGHT=0.8
CROSS_VIEW_OLD_BRANCH_DROPOUT=0.5
CROSS_VIEW_SOURCE_INJECTION_MODE=temporal_local
CROSS_VIEW_SOURCE_BRANCH_MODE=sigma_matched_clamp
CROSS_VIEW_SOURCE_GATE_MODE=scalar
```

### 10.3 查看真实命令

所有 bash 脚本都会打印最终 `Running command`。建议每次训练前保存日志：

```bash
mkdir -p logs
bash bash/train_droid_success_high_quality_crossview_cache.sh 2>&1 | tee logs/train_stage2_$(date +%Y%m%d_%H%M%S).log
```

## 11. FAQ 常见问题

### Q1: `/root/miniconda3/envs/studio/bin/python: No such file or directory`

当前可用环境通常是：

```bash
PYTHON_BIN=/env/conda/envs/studio/bin/python
```

训练和 cache 命令都建议显式传 `PYTHON_BIN`。

### Q2: `[ERROR] train manifest references missing assets`

说明 manifest 中 `video`、`state` 或 `prompt_emb` 路径无法在 `DATASET_META_ROOT` 下找到。检查：

```bash
head -n 1 $TRAIN_MANIFEST
ls $DATASET_META_ROOT
```

如果实际视频在另一个根目录，需要重新生成 manifest 或调整 `DATASET_META_ROOT` 到能解析相对路径的位置。

### Q3: 主 cache 会跳过已存在文件吗？

会。wrapper 调用 `tool/build_cross_view_latent_cache.py` 时传了 `--skip-existing`。但如果设置：

```bash
FORCE_REBUILD_CACHE=1
```

脚本会先删除整个 `CACHE_ROOT`，不要在续跑时使用。

### Q4: shard 按 contiguous 分片，前面样本已缓存，会空一张卡吗？

会。`contiguous` 下 shard0 可能全部命中已有 cache，很快结束，GPU 空闲。续跑推荐：

```bash
CACHE_SHARD_MODE=strided
```

从零构建且视频读取局部性更重要时可用 `contiguous`。

### Q5: 多 shard 后 GPU 利用率接近 0，CPU 很高怎么办？

cache 构建常见瓶颈是视频解码、parquet 读取和磁盘 IO，不一定是 GPU。建议：

```bash
CACHE_NUM_SHARDS=4
CACHE_NUM_WORKERS=0
CACHE_SHARD_MODE=strided
```

或单卡小 shard 测速。`CACHE_NUM_WORKERS` 过高会让 CPU 更忙，不一定更快。

### Q6: 是否建议开启 VAE tiled encode？

180x320 分辨率下通常不建议。tiled encode 更多是 OOM 兜底：

```bash
CACHE_VAE_TILED_ENCODE=1
```

如果没有显存问题，保持 `0` 通常更快。

### Q7: `target_camera_encoder` 为什么没有启用？

它只在以下参数为真时创建：

```bash
GEOMETRY_TARGET_CAMERA_MODE=add_time_mlp
```

同时需要 sidecar 或 runtime 数据里有 `target_cam_tokens` / `target_cam_tokens_latent`。推荐配套：

```bash
GEOMETRY_SCENE_TOKEN_SOURCE=camera_aware_sidecar
GEOMETRY_USE_CAMERA_TOKENS=1
GEOMETRY_SIDECAR_CACHE_PATH=/path/to/geometry_sidecar
```

### Q8: 主 cache 中的 scene tokens 是真实相机 token 吗？

不是严格真实相机。主 cache 如果传 `--scene_token_checkpoint`，当前逻辑给 scene extractor 的 `cam_token` 是全零，属于 `cached_zero_cam`。严格 camera-aware 训练应使用 geometry sidecar 的 `scene_tokens_camera_aware`。

### Q9: `Geometry sidecar not found`

当：

```bash
GEOMETRY_SCENE_TOKEN_SOURCE=camera_aware_sidecar
```

训练会强制要求每个主 cache 样本都有对应 sidecar：

```text
GEOMETRY_SIDECAR_CACHE_PATH/train/0000000.pth
GEOMETRY_SIDECAR_CACHE_PATH/val/0000000.pth
```

确认 `main_cache_path`、manifest、sidecar output 三者使用的是同一份样本顺序。

### Q10: `Cached dataset config mismatch for load_modules`

训练时 `LOAD_MODULES` 必须和 cache 构建时一致。推荐统一使用：

```bash
LOAD_MODULES=dit,text:emb,vae,image,action:noise
```

如果 cache 是跳过 legacy branch 构建的，训练也必须：

```bash
CROSS_VIEW_DISABLE_LEGACY_IMAGE_BRANCH=1
```

### Q10b: `Cached dataset was built with cross_view_use_tail_anchor=False ... but training requested dual-end anchoring`

方案 A 下 dual-end 训练必须配合 dual-end cache。如果用旧 head-only cache + `CROSS_VIEW_USE_TAIL_ANCHOR=1` 训练会被 `validate_cross_view_cache_config` 直接拒绝。

解决方法：

1. 用 §5.7 命令重建 cache，加 `CROSS_VIEW_USE_TAIL_ANCHOR=1` 与匹配的 `NUM_TAIL_FRAMES`；新 cache 起新名（例如 `..._planA` 后缀）；
2. 训练命令把 `CACHE_ROOT` 指到新目录。

类似地，如果 `num_tail_frames` 训练值与 cache_config 中的值不一致，也会直接报错。两个值必须严格相等。

如果你只是想"用 dual-end cache 跑一次 head-only 训练做对照"，可以保留 cache 不变，把 `CROSS_VIEW_USE_TAIL_ANCHOR=0` 训练 —— 此时会打印一条 WARN，cache 中的 tail 信号被忽略，但训练能跑（这是设计支持的 ablation 模式）。

### Q11: `CKPT_PATH is required for cross_view_stage2`

Stage2 必须从 stage1 或已有 stage2 checkpoint 初始化：

```bash
CKPT_PATH=/data_ywj/data_xh/projects/DiffSynth-Studio_v1/Ckpt/droid_success_lagernvs_180x320_stage1/epoch-6/epoch-6.safetensors
```

### Q12: 如何一键杀掉 cache 进程？

先确认命令：

```bash
ps -ef | grep build_cross_view_latent_cache.py | grep -v grep
```

再杀：

```bash
pkill -f build_cross_view_latent_cache.py
pkill -f build_cross_view_geometry_sidecar_cache.py
```

如需更谨慎，可只杀当前用户进程：

```bash
pkill -u "$USER" -f build_cross_view_latent_cache.py
```

### Q13: OOM 怎么办？

优先尝试：

```bash
GRAD_ACCUM_STEPS=8
CUDA_VISIBLE_DEVICES=0
```

如果是 cache 阶段 OOM：

```bash
CACHE_NUM_SHARDS=1
CACHE_VAE_TILED_ENCODE=1
```

如果是训练阶段 OOM，需要在训练命令中直接加 `--use_gradient_checkpointing`，当前 bash 模板里该选项注释掉了，可按需修改脚本或直接运行 Python 训练命令。

## 12. 直接 Python 训练模板

如果不想走 bash wrapper，可以直接调用训练入口。下面是 cached stage2 的最小模板：

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/env/conda/envs/studio/bin/python -m accelerate.commands.launch \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta \
  --cached_dataset_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/cache_crossview_81f_180x320_main \
  --data_file_keys video,state,prompt_emb \
  --state_type state_pose_7d \
  --state_stat_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/stat_state_pose_7d.json \
  --height 180 \
  --width 320 \
  --num_frames 81 \
  --num_history_frames 1 \
  --dataset_repeat 1 \
  --model_paths /data_ywj/data_xh/projects/datasets/PAI \
  --load_modules dit,text:emb,vae,image,action:noise \
  --learning_rate 1e-4 \
  --num_epochs 20 \
  --output_path /data_ywj/data_xh/projects/DiffSynth-Studio_v2/Ckpt/direct_stage2_example \
  --gradient_accumulation_steps 4 \
  --mixed_precision bf16 \
  --task cross_view_stage2 \
  --trainable_models dit \
  --ckpt_path /data_ywj/data_xh/projects/DiffSynth-Studio_v1/Ckpt/droid_success_lagernvs_180x320_stage1/epoch-6/epoch-6.safetensors \
  --cross_view_source_views 0,1 \
  --cross_view_target_view 2 \
  --cross_view_placeholder_mode zeros \
  --cross_view_source_injection_mode temporal_local \
  --cross_view_source_branch_mode sigma_matched_clamp \
  --cross_view_source_gate_mode scalar \
  --cross_view_old_branch_dropout 0.5 \
  --cross_view_temp_loss_weight 0.1 \
  --cross_view_state_loss_weight 0.05 \
  --scene_token_checkpoint /data_ywj/data_xh/projects/DiffSynth-Studio_v2/lagernvs/ckpt/droid_base_stage2/checkpoint_0060000.pt \
  --geometry_sidecar_cache_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/geometry_sidecar_lagernvs_strict_iter060000 \
  --geometry_scene_token_source camera_aware_sidecar \
  --geometry_use_camera_tokens 1 \
  --geometry_target_camera_mode add_time_mlp \
  --alignment_loss_weight 0.1
```

注意：即使 `--trainable_models dit`，stage2 初始化逻辑会自动追加 `source_video_projector`、`source_temporal_gate`、`target_state_head`、`target_camera_encoder`、`scene_token_adapter`、`geometry_gates` 等需要训练的模块。

## 13. 推荐工作流

### 13.1 从零到 stage2

```text
1. 准备 manifest、prompt_emb、state stats
2. 构建 wrist_first_frame_index，可选
3. 构建主 cache
4. 构建 geometry sidecar
5. 训练 stage1
6. 用 stage1 epoch-N.safetensors 初始化 stage2
7. stage2 推理生成对比视频
8. evaluate_generated_videos.py 计算指标
```

### 13.2 每步检查点

```bash
# 主 cache 文件数
find $CACHE_ROOT/train -maxdepth 1 -name '*.pth' | wc -l
find $CACHE_ROOT/val -maxdepth 1 -name '*.pth' | wc -l

# sidecar 文件数
find $GEOMETRY_SIDECAR_CACHE_PATH/train -maxdepth 1 -name '*.pth' | wc -l
find $GEOMETRY_SIDECAR_CACHE_PATH/val -maxdepth 1 -name '*.pth' | wc -l

# checkpoint
find /data_ywj/data_xh/projects/DiffSynth-Studio_v2/Ckpt -path '*epoch-*/*.safetensors' | sort
```

## 14. 给维护者和 AI/Codex 的实现提示

- 改训练行为优先看 `examples/wanvideo/model_training/train.py`，这里集中处理 stage1/stage2、cache、sidecar、loss。
- 改 DiT 条件注入优先看 `diffsynth/pipelines/wan_video.py:model_fn_wan_video` 和 `diffsynth/models/wan_video_dit.py`。
- 改数据格式优先看 `diffsynth/core/data/unified_dataset.py` 与 `diffsynth/core/data/operators.py`。
- 改 cache 字段时必须同步 `build_cross_view_latent_cache.py`、`validate_cached_dataset_config()`、`validate_cross_view_cached_batch()` 和推理脚本。
- 新增开关时优先加到 `diffsynth/diffusion/parsers.py`，再传入 `WanTrainingModule.__init__`。
- 不要静默改变旧 cache 格式；应增加版本字段或新路径，避免旧实验不可复现。

代码定位示例：

```bash
cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2

grep -n "validate_cached_dataset_config" examples/wanvideo/model_training/train.py
grep -n "cache_format_version" tool/build_cross_view_latent_cache.py
grep -n "geometry_cache_version" tool/build_cross_view_geometry_sidecar_cache.py
```
