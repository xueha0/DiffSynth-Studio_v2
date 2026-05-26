# Repository Guidelines（中文版）

## 项目结构与模块组织
- `diffsynth/`：核心库代码。`pipelines/` 负责 WAN 视频流程，`models/` 放 DiT、VAE、文本/图像/动作编码器，`core/` 提供数据与显存管理，`diffusion/` 放训练、调度器和 loss。
- `examples/wanvideo/`：可直接运行的入口。训练入口是 `model_training/train.py`，推理和闭环 rollout 在 `model_inference/`。
- `tool/`：数据切片、prompt embedding、统计量生成等脚本。
- `VLA/interface/`：外部机器人策略的 ZeroMQ 通信封装。
- `Data/`、`robot_data/`、`Ckpt/`：本地数据、机器人样本和训练产物。不要提交生成文件或大体积资产。

## 构建、测试与开发命令
- `conda env create -f diff-environment.yml`：创建仓库默认的 Python 3.10 环境。
- `pip install -e /data1/blm/DiffSynth-Studio`：以可编辑模式安装本仓库。
- `python -m accelerate.commands.launch examples/wanvideo/model_training/train.py ...`：启动训练。
- `python examples/wanvideo/model_inference/infer_robot.py ...`：执行单 checkpoint 视频推理。
- `python examples/wanvideo/model_inference/infer_vla_wm_closed_loop.py ...`：执行 VLA + world model 闭环推理。
- `python -m compileall diffsynth examples tool VLA`：提交前做一次轻量语法检查。

## 代码风格与命名规范
- 遵循 PEP 8，使用 4 空格缩进，变量和函数采用 `snake_case`，类名采用 `PascalCase`。
- 模块文件名保持小写；CLI 参数风格与现有脚本一致，例如 `--dataset_base_path`、`--action_stat_path`。
- 新增流程单元时，优先沿用 `WanVideoUnit_*` 这类命名模式。
- 仓库内没有统一格式化工具；请保持 import 整洁，并避免提交机器相关的硬编码路径。
- 新增行为必须挂在显式版本开关或新路径上，不要静默改旧逻辑。

## 测试指南
- 当前正式测试较少，新增逻辑时至少补充可复现的 smoke test。
- 测试文件建议命名为 `test_*.py`，小型测试可放在相关模块旁边，规模更大的测试建议新建 `tests/` 目录。
- 涉及数据处理或推理改动时，请在 PR 中附上复现命令、输出张量形状或产物路径。

## 提交与 Pull Request 规范
- 提交信息使用简短祈使句，例如 `Optimize history-frame conditioning`、`Add action normalization check`。
- 每个 commit 聚焦一个独立改动，避免把数据、脚本和模型修改混在一起。
- PR 需要说明目的、影响的入口脚本或数据格式、验证命令，以及推理结果变化时的示例输出。

## 配置与资产管理
- 不要把密钥、令牌、内网主机地址写入提交内容。
- 合并前把硬编码本地路径改成环境变量或明确的 CLI 参数。
