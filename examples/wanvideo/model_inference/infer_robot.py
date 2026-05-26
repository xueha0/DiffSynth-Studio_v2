"""
机器人视频推理主入口
负责：命令行参数解析、配置管理、启动推理引擎
"""
import os
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from PIL import Image

from diffsynth.diffusion.parsers import (
    add_action_config,
    add_dataset_base_config,
    add_infer_config,
    add_model_config,
    add_training_config,
    add_video_size_config,
    build_grouped_config,
    prepare_wan_runtime,
)
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline


# 禁用 tokenizers 的并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class FrameConverter:
    """帧格式转换工具"""

    @staticmethod
    def to_uint8(frame) -> np.ndarray:
        """将任意格式的帧转换为 uint8 numpy 数组"""
        if isinstance(frame, Image.Image):
            return np.array(frame)

        if isinstance(frame, torch.Tensor):
            if frame.dim() == 3 and frame.shape[0] in [1, 3, 4]:
                frame = frame.permute(1, 2, 0)
            frame = frame.detach().to(dtype=torch.float32)
            array = frame.cpu().numpy()
            return FrameConverter._normalize_to_uint8(array)

        if isinstance(frame, np.ndarray):
            return FrameConverter._normalize_to_uint8(frame)

        return frame

    @staticmethod
    def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
        """归一化到 uint8 范围"""
        if array.dtype == np.uint8:
            return array
        max_value = float(array.max())
        min_value = float(array.min())
        if min_value >= -1.0 and max_value <= 1.0:
            if min_value < 0.0:
                array = (array + 1.0) * 127.5
            else:
                array = array * 255.0
            return np.clip(array, 0, 255).astype(np.uint8)

        return np.clip(array, 0, 255).astype(np.uint8)

    @staticmethod
    def ensure_rgb(frame: np.ndarray) -> np.ndarray:
        """确保帧是 RGB 三通道"""
        if len(frame.shape) == 2:
            return np.stack([frame] * 3, axis=-1)
        return frame


class VideoSaver:
    """视频保存工具"""

    def __init__(self, fps: int = 5, quality: int = 5) -> None:
        self.fps = fps
        self.quality = quality
        self.converter = FrameConverter()

    def save_comparison(
        self,
        original_video,
        predicted_video,
        output_dir: Path,
        video_name: str,
        length_mode: str = "truncate_min",
    ) -> Path:
        """保存多视角对比视频：每行一个视角，左 GT，右 Pred。"""
        if not isinstance(original_video, torch.Tensor) or original_video.ndim != 5:
            raise TypeError("`original_video` must be torch.Tensor with shape (V,C,T,H,W).")
        if not isinstance(predicted_video, torch.Tensor) or predicted_video.ndim != 5:
            raise TypeError("`predicted_video` must be torch.Tensor with shape (V,C,T,H,W).")

        original_video = original_video.detach().to(dtype=torch.float32).cpu()
        predicted_video = predicted_video.detach().to(dtype=torch.float32).cpu()
        num_views = min(int(original_video.shape[0]), int(predicted_video.shape[0]))
        gt_frames = int(original_video.shape[2])
        pred_frames = int(predicted_video.shape[2])
        if length_mode == "truncate_min":
            num_frames = min(gt_frames, pred_frames)
        elif length_mode == "pad_to_pred_black_gt":
            num_frames = max(gt_frames, pred_frames)
        else:
            raise ValueError(f"Unknown length_mode={length_mode!r}")

        comparison_frames: List[np.ndarray] = []
        for t in range(num_frames):
            rows: List[np.ndarray] = []
            for view_idx in range(num_views):
                if t < gt_frames:
                    gt_frame = self.converter.ensure_rgb(
                        self.converter.to_uint8(original_video[view_idx, :, t])
                    )
                else:
                    pred_h = int(predicted_video.shape[3])
                    pred_w = int(predicted_video.shape[4])
                    gt_frame = np.zeros((pred_h, pred_w, 3), dtype=np.uint8)

                if t < pred_frames:
                    pred_frame = self.converter.ensure_rgb(
                        self.converter.to_uint8(predicted_video[view_idx, :, t])
                    )
                else:
                    gt_h = int(original_video.shape[3])
                    gt_w = int(original_video.shape[4])
                    pred_frame = np.zeros((gt_h, gt_w, 3), dtype=np.uint8)
                rows.append(np.hstack([gt_frame, pred_frame]))
            comparison_frames.append(np.vstack(rows))

        comp_array = np.asarray(comparison_frames)

        output_path = output_dir / video_name
        save_video(comp_array, str(output_path), fps=self.fps, quality=self.quality)
        return output_path


class TimeFormatter:
    """时间格式化工具"""

    @staticmethod
    def format(seconds: float) -> str:
        """格式化时间显示（秒转为 时:分:秒）"""
        if seconds < 60:
            return f"{seconds:.2f}s"
        if seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.2f}s"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


@dataclass
class Config:
    """推理配置（从训练 config + 推理参数合并）"""
    values: Dict[str, Any]
    grouped_config: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name in self.values:
            return self.values[name]
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    @property
    def metadata_path(self) -> str:
        """元数据文件路径"""
        return str(self._resolve_metadata_path())

    @property
    def resolved_data_base(self) -> str:
        """数据集根目录（用于输出路径）"""
        return str(self._resolve_data_base())

    @property
    def videos_path(self) -> str:
        """视频数据路径"""
        base_dir = self._resolve_data_base()
        return str(base_dir / "videos")

    def _resolve_metadata_path(self) -> Path:
        metadata_value = self.values.get("dataset_metadata_path")
        if not metadata_value:
            raise ValueError("`dataset_metadata_path` is required for inference.")
        metadata_path = Path(metadata_value)
        if metadata_path.is_absolute():
            return metadata_path

        has_sep = os.sep in metadata_value or (
            os.altsep and os.altsep in metadata_value
        )
        if has_sep:
            return metadata_path

        return Path(self.dataset_base_path) / "meta" / metadata_value

    def _resolve_data_base(self) -> Path:
        metadata_path = self._resolve_metadata_path()
        if metadata_path.parent.name == "meta":
            return metadata_path.parent.parent
        return Path(self.dataset_base_path)


class CheckpointPipelineManager:
    """检查点发现、Pipeline 初始化与权重更新"""

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config, self.logger, self.pipeline = config, logger, None

    def discover_checkpoints(self) -> List[Path]:
        if not self.config.checkpoint_path:
            raise ValueError("`--ckpt_path` is required and must be a .safetensors file path.")
        ckpt_path = Path(self.config.checkpoint_path)
        self.logger.info("Mode: SINGLE CHECKPOINT")
        self.logger.info(f"  - {ckpt_path.name}")
        return [ckpt_path]
    
    def initialize_pipeline(self, checkpoints: List[Path]) -> WanVideoPipeline:
        if not checkpoints:
            raise ValueError("No checkpoints provided for pipeline initialization")
        module_set = {str(item).strip().lower() for item in self.config.modules if str(item).strip()}
        model_configs = [ModelConfig(path=p, offload_device="cpu") for p in self._parse_model_paths()]
        tokenizer_config = (
            ModelConfig(path=self.config.tokenizer_path)
            if "text" in module_set and self.config.tokenizer_path
            else None
        )
        first_ckpt = checkpoints[0]
        msg = f"Using pretrained WAN weights; will apply checkpoint: {first_ckpt}"
        self.logger.info(msg)
        self.pipeline = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            modules=list(self.config.modules),
        )
        self.logger.info("Pipeline initialized successfully!\n"); return self.pipeline
    
    def update_checkpoint(self, checkpoint: Path) -> None:
        if self.pipeline is None: raise RuntimeError("Pipeline is not initialized")
        from diffsynth.core.loader import load_state_dict
        self.logger.info(f"Updating checkpoint weights: {checkpoint.name}")
        state_dict = load_state_dict(str(checkpoint), torch_dtype=torch.bfloat16, device="cpu")
        dit_state = {k: v for k, v in state_dict.items() if not k.startswith("pipe.action_encoder.")}
        act_state = {k[len("pipe.action_encoder."):]: v for k, v in state_dict.items() if k.startswith("pipe.action_encoder.")}
        if self.pipeline.dit is not None and dit_state:
            self.pipeline.dit.load_state_dict(dit_state, strict=False); self.logger.info(f"  - Updated dit: {len(dit_state)} parameters")
        if self.pipeline.action_encoder is not None and act_state:
            self.pipeline.action_encoder.load_state_dict(act_state, strict=False); self.logger.info(f"  - Updated action_encoder: {len(act_state)} parameters")
        elif act_state:
            self.logger.warning("  - action_encoder weights found but pipeline.action_encoder is None")
        self.logger.info("")

    def _parse_model_paths(self) -> List[str]:
        if not self.config.model_paths:
            raise ValueError("Model paths are empty; check `--model_paths` and `--load_modules`.")
        return json.loads(self.config.model_paths)


def _load_config_defaults(ckpt_path: Optional[str]) -> Dict[str, Any]:
    if not ckpt_path:
        return {}
    path = Path(ckpt_path)
    config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    if not any(isinstance(value, dict) for value in data.values()):
        return data
    flat: Dict[str, Any] = {}
    for value in data.values():
        if isinstance(value, dict):
            flat.update(value)
    return flat


def parse_args() -> Config:
    """解析命令行参数并返回配置对象"""
    parser = argparse.ArgumentParser(description="Robot inference with WanVideo pipeline")

    # 数据参数
    parser = add_model_config(parser)
    parser = add_dataset_base_config(parser)
    parser = add_action_config(parser)
    parser = add_video_size_config(parser)
    parser = add_training_config(parser)
    parser = add_infer_config(parser)

    for action in parser._actions:
        if getattr(action, "dest", None) == "dataset_base_path":
            action.required = False
            break

    pre_args, _ = parser.parse_known_args()
    base_config = _load_config_defaults(pre_args.checkpoint_path)
    if base_config:
        known = {action.dest for action in parser._actions if getattr(action, "dest", None)}
        defaults = {k: v for k, v in base_config.items() if k in known}
        if defaults:
            parser.set_defaults(**defaults)
    args = parser.parse_args()

    if not args.dataset_base_path:
        raise ValueError("`--dataset_base_path` is required (or provide a config.json next to the checkpoint).")

    grouped_config = build_grouped_config(parser, args) or {}
    values = vars(args).copy()
    data_file_keys = values.get("data_file_keys")
    data_file_keys = (
        [item.strip() for item in str(data_file_keys).split(",") if item.strip()]
        if data_file_keys is not None
        else []
    )
    runtime = prepare_wan_runtime(
        args.model_paths,
        args.load_modules,
        data_file_keys,
    )
    values["data_file_keys"] = tuple(runtime["data_file_keys"])
    values["modules"] = tuple(runtime["modules"])
    values["model_paths"] = json.dumps(runtime["model_paths"])
    values["tokenizer_path"] = runtime["tokenizer_path"]

    return Config(values=values, grouped_config=grouped_config)


def setup_logging() -> logging.Logger:
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def main() -> None:
    """主入口"""
    from inference_engine import InferenceEngine

    config = parse_args()
    logger = setup_logging()

    engine = InferenceEngine(config, logger)
    engine.run()


if __name__ == "__main__":
    main()
