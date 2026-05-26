# 双第三视角到单第一视角的视频生成两阶段训练设计（备选方案）

## 核心思想

这版方案不再保留“三视角联合视频生成”的旧语义，而是从第一阶段开始就显式拆分：

- 条件输入：两路第三视角视频
- 监督目标：单路第一视角视频

目标是做更干净的跨视角视频翻译，而不是在原有三视角联合 world model 上打补丁。

## 阶段一

- 训练样本字段拆成：
  - `source_video`: 两路第三视角
  - `target_video`: 单路第一视角
  - `action`
  - `prompt/prompt_emb`
- 条件分支只编码 `source_video`
- 用冻结 VAE 编码 `target_video`，在 latent 空间做 Flow Matching SFT
- loss 只监督 `target_video`，不做 source reconstruction
- 不依赖目标首帧，不保留目标占位槽

## 阶段二

- 新增 `SourceVideoProjector3D`
- 用冻结 VAE 先编码两路第三视角完整视频，再通过轻量 3D projector 生成 `source_tokens`
- `source_tokens` 作为 cross-attention 条件注入 DiT
- `action` 建议也改成 token 条件，而不是直接做 noise injection
- 继续只监督第一视角视频

## 优点

- 任务定义更干净，训练和推理分布一致
- 不需要为目标视角设计占位帧
- 更适合后续扩展到更多机位或变化机位

## 缺点

- 对现有代码改动更大
- 需要新增样本字段和专用推理接口
- 无法最大化复用当前“三视角拼接 latent”训练逻辑
