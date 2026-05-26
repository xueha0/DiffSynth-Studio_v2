# 数据集说明

本文档说明本仓库训练 `examples/wanvideo/model_training/train.py` 时，数据集需要整理成什么格式，以及如何用仓库内脚本把原始机器人数据加工成可训练版本。

## 1. 训练时实际读取哪些文件

训练入口最终会从 `UnifiedDataset` 读取以下几类文件：

- `meta/*.jsonl`：样本级元数据，定义视频路径、动作路径、帧范围、文本条件等。
- `videos/**/*.mp4`：视频文件，支持单视角或多视角。
- `data/**/*.parquet`：动作或状态序列。
- `meta/stat.json`：动作归一化统计量。
- `prompt_emb/*.pt`：预提取文本 embedding，仅在 `text:emb` 模式下需要。

最常见的训练配置是：

```bash
--load_modules "dit,text:emb,vae,image,action:noise"
```

在这个配置下，每条样本最少需要：

- `video`
- `start_frame`
- `end_frame`
- `action`
- `prompt_emb`

如果改成 `text` 或 `text:t5`，则需要 `prompt`；如果关闭 `action`，则可以不提供 `action` 和 `stat.json`。

## 2. 推荐目录结构

推荐按下面的目录布局组织数据：

```text
dataset_root/
  data/
    chunk-000/
      episode_000000.parquet
      episode_000001.parquet
  videos/
    chunk-000/
      observation.images.image/
        episode_000000.mp4
      observation.images.wrist_image/
        episode_000000.mp4
  prompt_emb/
    pos_0.pt
    pos_1.pt
    neg_prompt.pt
  meta/
    info.json
    tasks.jsonl
    episodes.jsonl
    episodes_train.jsonl
    episodes_val.jsonl
    stat.json
```

说明：

- `data/` 和 `videos/` 是训练必需目录。
- `meta/info.json` 不是训练强依赖，但仓库内的 `robot_data/compute_stat.py` 会使用它，建议保留。
- `prompt_emb/` 只在 `text:emb` 模式下必需。

## 3. 元数据 `jsonl` 格式

### 3.1 推荐样本格式

`episodes_train.jsonl` 或 `episodes_val.jsonl` 中，每行推荐如下：

```json
{
  "episode_index": 0,
  "length": 17,
  "start_frame": 18,
  "end_frame": 34,
  "video": [
    "videos/chunk-000/observation.images.image/episode_000000.mp4",
    "videos/chunk-000/observation.images.wrist_image/episode_000000.mp4"
  ],
  "action": "data/chunk-000/episode_000000.parquet",
  "prompt": "Pick up the cube and place on the plate",
  "prompt_emb": "prompt_emb/pos_0.pt"
}
```

### 3.2 字段说明

- `episode_index`：样本所属 episode 编号。训练不强依赖，但建议保留。
- `length`：当前样本片段长度。通常应满足 `length = end_frame - start_frame + 1`。
- `start_frame` / `end_frame`：必需。视频和 parquet 都按这个区间切片。
- `video`：必需。
  - 可为字符串：单个视频文件。
  - 可为列表：多个相机的视频文件，训练时会组成多视角输入。
- `action`：动作或状态 parquet 路径。开启 `action:*` 模块时必需。
- `prompt`：文本提示词。使用 `text`/`text:t5` 时必需。
- `prompt_emb`：预提取文本 embedding 路径。使用 `text:emb` 时必需。

### 3.3 支持的两种视频组织方式

本仓库支持两种常见组织方式：

#### 方式 A：多相机分文件，推荐

```json
"video": [
  "videos/chunk-000/observation.images.image/episode_000000.mp4",
  "videos/chunk-000/observation.images.wrist_image/episode_000000.mp4"
]
```

特点：

- `V=视角数`，模型能显式知道是多视角输入。
- 这是 `robot_data/piper` 当前采用的方式。

#### 方式 B：多视角先拼接成单个视频

```json
"video": "videos_new_81/chunk-000/observation.images.cam_all_views_rgb/episode_000049.mp4"
```

特点：

- 视频帧通常已经按高度方向拼接，例如 3 个相机拼成 `1440 x 640`。
- 对训练来说也能工作，因为模型最终看到的是拼接后的大画面。
- 这是 `Data/Val_new_81` 一类数据常见的方式。

如果你是新建训练集，优先推荐方式 A，可读性和可维护性更好。

## 4. 视频文件要求

- 推荐格式：`mp4`
- 帧像素格式：普通 RGB 视频即可。
- 示例尺寸：
  - 单视角：`480 x 640 x 3`
  - 三视角竖向拼接：`1440 x 640 x 3`
- 训练脚本会按 `--height`、`--width` 做 resize，不要求原始尺寸完全一致。
- 但建议保证：
  - 同一数据集的视频编码稳定；
  - `start_frame` 到 `end_frame` 区间真实存在；
  - 不同模态的帧数与 parquet 行数一致。

### 4.1 帧数约束

WAN 视频训练默认要求时间长度满足：

```text
num_frames % 4 == 1
```

常见值：

- `17`
- `81`

如果视频太长，通常先在元数据里切成训练片段，而不是一次喂完整 episode。

## 5. parquet 文件要求

### 5.1 必需列

当训练带动作条件时，parquet 至少应包含：

- `action`
- `observation.state`

仓库中真实样例还会带：

- `timestamp`
- `frame_index`
- `episode_index`
- `task_index`

这些附加列不会影响训练，可以保留。

### 5.2 维度要求

仓库内部动作加载器支持两类宽度：

- `26` 维完整状态/动作
- `14` 维已经切好的单种表示

但**强烈建议统一成 26 维**，因为：

- `robot_data/compute_stat.py` 默认会把 7 维数据扩展或重写成 26 维；
- `stat.json` 也是从 26 维再切出 `joint/pose` 的 14 维统计量；
- 这条路径和仓库现有脚本最兼容。

### 5.3 26 维字段顺序

26 维顺序应与 `meta/info.json` 中 `action.names` / `observation.state.names` 对齐，即：

1. 左臂 6 个关节
2. 左夹爪开合
3. 左臂末端位姿 `xyz + rpy`
4. 右臂 6 个关节
5. 右夹爪开合
6. 右臂末端位姿 `xyz + rpy`

如果你的原始数据只有 7 维单臂 `xyz rpy + gripper`，不能直接拿来训练当前默认流程，需要先映射或扩展。

## 6. `stat.json` 格式

带动作条件训练时，需要通过 `--action_stat_path` 指向一个 `stat.json`。

推荐结构如下：

```json
{
  "state_joint": {
    "shape": [14],
    "min": [...],
    "max": [...],
    "p01": [...],
    "p99": [...],
    "mean": [...],
    "std": [...]
  },
  "action_joint": { "...": "..." },
  "state_pose": { "...": "..." },
  "action_pose": { "...": "..." }
}
```

关键点：

- 四个键最好都提供：`state_joint`、`action_joint`、`state_pose`、`action_pose`
- 每组统计量长度都应为 `14`
- 当前动作归一化默认优先使用 `p01` / `p99`

## 7. `tasks.jsonl` 与 `prompt_emb`

### 7.1 推荐 `tasks.jsonl`

为兼容仓库内不同脚本，建议每条 task 同时保留 `task` 和 `prompt`：

```json
{"task_index": 0, "task": "Pick up the cube and place on the plate", "prompt": "Pick up the cube and place on the plate"}
```

这样可以兼容：

- `robot_data/compute_stat.py` 中按 `task` 做映射的逻辑
- `tool/build_prompt_embeddings.py` 中按 `prompt` 生成 embedding 的逻辑

### 7.2 `prompt_emb` 文件

- 文件格式：`torch.save()` 保存的 `.pt`
- 一般由 `tool/build_prompt_embeddings.py` 生成，不建议手写
- 正样本文本 embedding 常命名为：
  - `prompt_emb/pos_0.pt`
  - `prompt_emb/pos_1.pt`
- 负样本 embedding 常命名为：
  - `prompt_emb/neg_prompt.pt`

## 8. 从原始数据加工到可训练格式

### 8.1 如果你已有接近 `robot_data/piper` 的目录

可以直接使用仓库脚本：

```bash
python robot_data/compute_stat.py \
  --dataset-root robot_data/piper \
  --train-window-size 17 \
  --train-stride 16 \
  --train-min-length 5
```

这个脚本会做几件事：

1. 把 parquet 中的动作/状态重写到 26 维
2. 重写 `meta/episodes.jsonl`
3. 生成 `meta/episodes_train.jsonl`
4. 重写 `meta/info.json`
5. 生成 `meta/stat.json`

如果还要顺便生成 prompt embedding：

```bash
python robot_data/compute_stat.py \
  --dataset-root robot_data/piper \
  --build-prompt-emb \
  --prompt-emb-model-root /path/to/Wan2.1-Fun-V1.1-1.3B-InP
```

### 8.2 如果你只想单独切训练片段

```bash
python tool/build_piper_train_1_from_episodes.py \
  --input robot_data/piper/meta/episodes.jsonl \
  --output robot_data/piper/meta/episodes_train_1.jsonl \
  --clip 17 \
  --overlap 1 \
  --min-tail 5
```

### 8.3 如果你只想单独生成 prompt embedding

```bash
python tool/build_prompt_embeddings.py \
  --mode pos \
  --pos-jsonl robot_data/piper/meta/tasks.jsonl \
  --pos-output robot_data/piper/prompt_emb \
  --model-root /path/to/Wan2.1-Fun-V1.1-1.3B-InP
```

然后把 embedding 路径写回训练元数据：

```bash
python tool/add_prompt_emb_to_episode.py \
  --val-jsonl robot_data/piper/meta/episodes_train.jsonl \
  --task-jsonl robot_data/piper/meta/tasks.jsonl
```

## 9. 一条可直接参考的训练命令

```bash
python -m accelerate.commands.launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /path/to/dataset_root \
  --dataset_metadata_path /path/to/dataset_root/meta/episodes_train.jsonl \
  --action_stat_path /path/to/dataset_root/meta/stat.json \
  --action_type state_pose \
  --load_modules "dit,text:emb,vae,image,action:noise" \
  --height 240 \
  --width 320 \
  --num_frames 17 \
  --num_history_frames 1 \
  --model_paths /path/to/Wan2.1-Fun-V1.1-1.3B-InP \
  --output_path Ckpt/my_run
```

## 10. 提交前的自检清单

在开始训练前，建议逐项确认：

- `episodes_train.jsonl` 中每条样本都能找到对应的 `video` 和 `action`
- `length == end_frame - start_frame + 1`
- `start_frame/end_frame` 不越界
- `parquet` 的 `action` 和 `observation.state` 行数与视频帧数一致
- `stat.json` 中四类统计量都存在，长度为 `14`
- `prompt_emb` 路径能被 `dataset_base_path` 正确解析
- 训练的 `--load_modules` 与数据字段匹配

## 11. 常见错误

### 11.1 `Prompt not found` 或 `Missing prompt_emb`

原因通常是：

- `tasks.jsonl` 里只有 `task` 没有 `prompt`
- `episodes_train.jsonl` 里的 `prompt` 文本和 `tasks.jsonl` 对不上
- `prompt_emb/*.pt` 没生成或路径写错

建议：让 `tasks.jsonl` 同时保存 `task` 和 `prompt`，且两者内容一致。

### 11.2 `Unexpected action width`

说明 parquet 中动作维度既不是完整的 `26`，也不是当前 `action_type` 对应的 `14`。

建议：统一转换成 26 维，再重新生成 `stat.json`。

### 11.3 `Not enough rows in parquet` 或视频越界

说明 `start_frame/end_frame` 切片范围超过了真实 episode 长度。

建议：优先从原始 episode 长度重新生成 `episodes_train.jsonl`，不要手工改一部分字段。

---

如果后续要继续扩展这份文档，建议优先补充两类内容：

- 新机器人平台的数据映射规则，例如 7D/14D/26D 如何互转
- 各种 `load_modules` 组合对应的最小字段集合
