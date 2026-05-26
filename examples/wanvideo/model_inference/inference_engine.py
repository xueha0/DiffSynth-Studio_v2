"""
推理引擎：负责所有推理逻辑
包括：检查点管理、数据加载、视频生成、指标评估
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadCobotAction, ResolvePromptEmbPath
from diffsynth.pipelines.wan_video import WanVideoPipeline
from infer_robot import VideoSaver, CheckpointPipelineManager

DATASET_LOAD_NUM_FRAMES = 100001


class InferenceEngine:
    """
    推理引擎：负责整个推理流程

    核心流程：
    1. 加载数据集并选择样本
    2. 发现所有 checkpoint
    3. 初始化 pipeline（只加载一次 VAE/CLIP/T5）
    4. 遍历每个 checkpoint：
       - 更新 DiT/Action Encoder 权重
       - 生成所有样本的视频
       - 计算评估指标
    5. 输出总结报告
    """

    def __init__(self, config, logger) -> None:
        self.config = config
        self.logger = logger

        self.video_saver = VideoSaver(fps=config.fps, quality=config.quality)
        self.pipeline_manager = CheckpointPipelineManager(config, logger)

        self.dataset: Optional[UnifiedDataset] = None
        self.pipeline: Optional[WanVideoPipeline] = None
        self.sample_indices: List[int] = []
        self.checkpoints: List[Path] = []
        self.num_views: Optional[int] = None

    def run(self) -> None:
        """运行完整推理流程"""
        try:
            self._prepare()
            self.pipeline = self.pipeline_manager.initialize_pipeline(self.checkpoints)

            summaries: List[Dict] = []
            for ckpt_idx, checkpoint in enumerate(self.checkpoints, 1):
                summary = self._process_checkpoint(checkpoint, ckpt_idx)
                summaries.append(summary)

        except Exception as exc:  # pragma: no cover - runtime protection
            self.logger.error(f"Inference failed: {exc}", exc_info=True)
            raise

    # ---------- 准备阶段 ----------

    def _prepare(self) -> None:
        """准备数据集和 checkpoints"""
        self._load_dataset()
        self.sample_indices = list(range(len(self.dataset)))
        self.checkpoints = self.pipeline_manager.discover_checkpoints()

        self.logger.info(f"Total samples: {len(self.dataset)}")
        self.logger.info(f"Selected samples: {len(self.sample_indices)}")
        self.logger.info(f"Checkpoints to process: {len(self.checkpoints)}")
        self.logger.info("")

    def _load_dataset(self) -> None:
        """加载数据集"""
        self.logger.info("Loading dataset...")
        module_bases = {
            str(module).partition(":")[0].strip().lower()
            for module in (self.config.modules or [])
            if str(module).strip()
        }
        special_operator_map = {}
        action_stat_path = self.config.action_stat_path
        if "action" in self.config.data_file_keys:
            self.logger.info(f"Action stat path: {action_stat_path}")
        if "text" in module_bases and "prompt_emb" in self.config.data_file_keys:
            special_operator_map["prompt_emb"] = ResolvePromptEmbPath(
                base_path=self.config.dataset_base_path
            )
        if "text" in module_bases and "negative_prompt_emb" in self.config.data_file_keys:
            special_operator_map["negative_prompt_emb"] = ResolvePromptEmbPath(
                base_path=self.config.dataset_base_path
            )
        self.dataset = UnifiedDataset(
            base_path=self.config.dataset_base_path,
            metadata_path=self.config.dataset_metadata_path,
            repeat=1,
            data_file_keys=self.config.data_file_keys,
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=self.config.dataset_base_path,
                height=self.config.height,
                width=self.config.width,
                num_frames=DATASET_LOAD_NUM_FRAMES,
                resize_mode=self.config.resize_mode,
            ),
            special_operator_map=special_operator_map,
            stat_path=action_stat_path,
            action_type=self.config.action_type,
        )

        if "action" in self.config.data_file_keys:
            self.dataset.special_operator_map["action"] = LoadCobotAction(
                base_path=self.config.dataset_base_path,
                action_type=self.config.action_type,
                stat=self.dataset.stat,
                num_frames=DATASET_LOAD_NUM_FRAMES,
            )

    # ---------- Checkpoint 处理 ----------

    def _process_checkpoint(self, checkpoint: Path, ckpt_idx: int) -> Dict:
        """处理单个 checkpoint 的推理流程"""
        self.logger.info("\n" + "#" * 80)
        self.logger.info(f"Processing checkpoint {ckpt_idx}/{len(self.checkpoints)}: {checkpoint.name}")
        self.logger.info("#" * 80 + "\n")

        self.pipeline_manager.update_checkpoint(checkpoint)

        output_dirs = self._create_output_dirs(checkpoint)

        self._generate_all_videos(output_dirs, ckpt_idx)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("VIDEO GENERATION COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Total videos: {len(self.sample_indices)}")
        self.logger.info("=" * 60 + "\n")

        checkpoint_name = checkpoint.name
        metric_results = self._evaluate_metrics(output_dirs, checkpoint_name)

        return {
            "checkpoint": checkpoint_name,
            "output_dir": output_dirs["root"].resolve(),
            "num_videos": len(self.sample_indices),
            "metrics": metric_results,
        }

    def _create_output_dirs(self, checkpoint: Path) -> Dict[str, Path]:
        """创建输出目录结构（时间戳目录）"""
        from datetime import datetime

        # 生成时间戳：格式为 {月}M{日}D_{小时}H{分钟}Min（例如：12M15D_14H30Min）
        timestamp = datetime.now().strftime("%mM%dD_%HH%MMin")

        base_dir = checkpoint.parent / checkpoint.stem
        output_dir = base_dir / timestamp

        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_config(output_dir)

        dirs = {
            "root": output_dir,
        }

        self.logger.info(f"Output directory: {output_dir.resolve()}\n")
        return dirs

    def _save_config(self, output_dir: Path) -> None:
        config_path = output_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            payload = self.config.grouped_config or self.config.values
            json.dump(payload, f, indent=2, ensure_ascii=True)

    def _evaluate_metrics(self, output_dirs: Dict[str, Path], checkpoint_name: str) -> Optional[Dict]:
        """评估对比视频的 PSNR/SSIM/MSE 指标"""
        if not self.config.enable_metrics:
            self.logger.info("Metrics disabled; skipping evaluation.\n")
            return None

        comparison_dir = output_dirs["root"]
        self.logger.info(f"Evaluating metrics for checkpoint: {checkpoint_name}")

        from diffsynth.core.metric.metric import evaluate

        view_names = [
            "cam_high_rgb",
            "cam_left_wrist_rgb",
            "cam_right_wrist_rgb",
        ]

        def _format_view_metrics(view_metrics):
            formatted = {}
            for idx, metrics in enumerate(view_metrics):
                name = view_names[idx] if idx < len(view_names) else f"view_{idx + 1}"
                formatted[name] = {
                    "psnr": float(metrics["psnr"]),
                    "ssim": float(metrics["ssim"]),
                    "mse": float(metrics["mse"]),
                    "frames": int(metrics["frames"]),
                }
            return formatted

        def _format_metrics(results):
            return {
                "avg_psnr": float(results["avg_psnr"]),
                "avg_ssim": float(results["avg_ssim"]),
                "avg_mse": float(results["avg_mse"]),
                "view_metrics": _format_view_metrics(results["view_metrics"]),
            }

        def _extract_epoch(checkpoint):
            import re
            match = re.search(r"epoch[-_]?(\d+)", checkpoint)
            if match:
                return f"epoch-{match.group(1)}"
            return checkpoint

        type_dirs = sorted([p for p in comparison_dir.iterdir() if p.is_dir()])

        num_views = self.num_views
        overall_results = evaluate(str(comparison_dir), num_views=num_views)
        metrics = {
            "overall": _format_metrics(overall_results),
            "types": {},
        }
        for type_dir in type_dirs:
            type_results = evaluate(str(type_dir), num_views=num_views)
            metrics["types"][type_dir.name] = _format_metrics(type_results)

        metrics_path = output_dirs["root"] / "metrics.json"
        epoch_label = _extract_epoch(checkpoint_name)
        payload = {
            "checkpoint": checkpoint_name,
            "epoch": epoch_label,
            "overall": {
                "comparison_dir": str(comparison_dir.resolve()),
                **metrics["overall"],
            },
            "types": {},
        }
        for type_name, type_metrics in metrics["types"].items():
            type_dir = comparison_dir / type_name
            payload["types"][type_name] = {
                "comparison_dir": str(type_dir.resolve()),
                **type_metrics,
            }
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)

        self.logger.info(f"Saved metrics to: {metrics_path.resolve()}")
        self.logger.info(
            f"Metrics for {checkpoint_name}: PSNR={metrics['overall']['avg_psnr']:.4f} "
            f"SSIM={metrics['overall']['avg_ssim']:.4f} MSE={metrics['overall']['avg_mse']:.6f}\n"
        )
        return metrics

    # ---------- 视频生成 ----------

    def _generate_all_videos(self, output_dirs: Dict[str, Path], ckpt_idx: int) -> None:
        """批量生成所有样本的视频"""
        for i, sample_idx in enumerate(self.sample_indices, 1):
            self.logger.info("\n" + "#" * 60)
            if len(self.checkpoints) > 1:
                self.logger.info(
                    f"[Checkpoint {ckpt_idx}/{len(self.checkpoints)}] Video {i}/{len(self.sample_indices)} "
                    f"(Sample {sample_idx})"
                )
            else:
                self.logger.info(f"Video {i}/{len(self.sample_indices)} (Sample {sample_idx})")
            self.logger.info("#" * 60)

            sample = self.dataset[sample_idx]
            self._generate_single_video(sample, output_dirs, i)

    def _resolve_prompt_emb_path(self, path_value):
        if path_value in (None, ""):
            return None
        if isinstance(path_value, (str, Path)):
            path = Path(path_value)
            if path.is_absolute():
                return str(path)
            return str(Path(self.config.dataset_base_path) / path)
        return path_value

    @staticmethod
    def _align_num_frames(num_frames: int) -> int:
        if (num_frames - 1) % 4 == 0:
            return num_frames
        return num_frames + (4 - ((num_frames - 1) % 4))

    @staticmethod
    def _slice_and_pad_action(action, start: int, target_frames: int, infer_frames: int):
        if action is None:
            return None
        chunk_action = action[:, start : start + target_frames]
        current_frames = int(chunk_action.shape[1])
        if current_frames >= infer_frames:
            return chunk_action[:, :infer_frames]
        if current_frames <= 0:
            return chunk_action
        pad_frames = infer_frames - current_frames
        if isinstance(chunk_action, torch.Tensor):
            pad = chunk_action[:, -1:, :].repeat(1, pad_frames, 1)
            return torch.cat([chunk_action, pad], dim=1)
        pad = np.repeat(chunk_action[:, -1:, :], repeats=pad_frames, axis=1)
        return np.concatenate([chunk_action, pad], axis=1)


    def _generate_single_video(
        self, sample: Dict, output_dirs: Dict[str, Path], video_index: int
    ) -> None:
        """生成单个样本的视频"""
        original_video = sample["video"]
        if not isinstance(original_video, torch.Tensor) or original_video.ndim != 5:
            raise TypeError(
                f"`sample['video']` must be torch.Tensor with shape (V,C,T,H,W), got {type(original_video)}"
            )
        original_video = original_video.detach().cpu()
        action = sample.get("action")
        if getattr(self.pipeline, "action_injection_mode", "none") == "none":
            action = None
        episode_index = sample["episode_index"]
        prompt = sample.get("prompt")
        prompt_emb = self._resolve_prompt_emb_path(sample.get("prompt_emb"))
        negative_prompt_emb = sample.get("negative_prompt_emb", self.config.negative_prompt_emb)
        negative_prompt_emb = self._resolve_prompt_emb_path(negative_prompt_emb)
        sample_type = str(sample.get("type") or "unknown")

        num_views, _, total_frames, input_height, input_width = original_video.shape
        num_views = int(num_views)
        total_frames = int(total_frames)
        input_height = int(input_height)
        input_width = int(input_width)
        if self.num_views is None:
            self.num_views = num_views
        elif self.num_views != num_views:
            self.logger.warning(f"Mixed num_views detected: {self.num_views} -> {num_views}")

        self.logger.info(f"Episode: {episode_index}")
        self.logger.info(f"Loaded shape: {tuple(original_video.shape)}")
        self.logger.info(f"Type: {sample_type}")
        self.logger.info(f"Generating with diffusion (frames: {total_frames})")
        self.logger.info(f"Using input size: {input_width}x{input_height}")
        if action is not None:
            action = action[:, :total_frames]

        history_frames = int(self.config.num_history_frames)
        chunk_size = int(self.config.num_frames)

        if self.config.chunk_infer and total_frames > chunk_size:
            chunk_stride = chunk_size - history_frames
            chunk_starts = [0]
            while chunk_starts[-1] + chunk_size < total_frames:
                chunk_starts.append(chunk_starts[-1] + chunk_stride)

            predicted_chunks = []
            last_input_video = original_video[:, :, :history_frames]
            for chunk_idx, start in enumerate(chunk_starts):
                chunk_frames = min(chunk_size, total_frames - start)
                infer_frames = self._align_num_frames(chunk_frames)
                chunk_action = self._slice_and_pad_action(
                    action=action,
                    start=start,
                    target_frames=chunk_frames,
                    infer_frames=infer_frames,
                )
                chunk_input_history = last_input_video
                chunk_video = self.pipeline(
                    prompt=prompt,
                    negative_prompt=self.config.negative_prompt,
                    prompt_emb=prompt_emb,
                    negative_prompt_emb=negative_prompt_emb,
                    input_video=last_input_video,
                    action=chunk_action,
                    seed=self.config.seed,
                    tiled=False,
                    height=input_height,
                    width=input_width,
                    num_frames=infer_frames,
                    num_history_frames=history_frames,
                    cfg_scale=self.config.cfg_scale,
                    num_inference_steps=self.config.num_inference_steps,
                )
                if not isinstance(chunk_video, torch.Tensor) or chunk_video.ndim != 5:
                    raise TypeError(f"Pipeline output must be (V,C,T,H,W), got {type(chunk_video)}")

                chunk_video = chunk_video.detach().cpu()[:, :, :infer_frames]
                chunk_video = chunk_video[:, :, :chunk_frames]
                history_to_copy = min(
                    history_frames,
                    int(chunk_video.shape[2]),
                    int(chunk_input_history.shape[2]),
                )
                if history_to_copy > 0:
                    chunk_video[:, :, :history_to_copy] = chunk_input_history[:, :, :history_to_copy]
                if chunk_video.shape[2] > 0:
                    last_input_video = chunk_video[:, :, -history_frames:].contiguous()
                if chunk_idx > 0:
                    chunk_video = chunk_video[:, :, history_frames:]
                if chunk_video.shape[2] > 0:
                    predicted_chunks.append(chunk_video)

            if predicted_chunks:
                predicted_video = torch.cat(predicted_chunks, dim=2)
            else:
                predicted_video = original_video[:, :, :0]
        else:
            infer_frames = self._align_num_frames(total_frames)
            infer_action = self._slice_and_pad_action(
                action=action,
                start=0,
                target_frames=total_frames,
                infer_frames=infer_frames,
            )
            predicted_video = self.pipeline(
                prompt=prompt,
                negative_prompt=self.config.negative_prompt,
                prompt_emb=prompt_emb,
                negative_prompt_emb=negative_prompt_emb,
                input_video=original_video[:, :, :history_frames],
                action=infer_action,
                seed=self.config.seed,
                tiled=False,
                height=input_height,
                width=input_width,
                num_frames=infer_frames,
                num_history_frames=history_frames,
                cfg_scale=self.config.cfg_scale,
                num_inference_steps=self.config.num_inference_steps,
            )
            if not isinstance(predicted_video, torch.Tensor) or predicted_video.ndim != 5:
                raise TypeError(f"Pipeline output must be (V,C,T,H,W), got {type(predicted_video)}")
            predicted_video = predicted_video.detach().cpu()
            predicted_video = predicted_video[:, :, :total_frames]
            history_to_copy = min(
                history_frames,
                int(predicted_video.shape[2]),
                int(original_video.shape[2]),
            )
            if history_to_copy > 0:
                predicted_video[:, :, :history_to_copy] = original_video[:, :, :history_to_copy]

        expected_frames = total_frames
        pred_frames = int(predicted_video.shape[2]) if isinstance(predicted_video, torch.Tensor) else 0
        if pred_frames > expected_frames:
            predicted_video = predicted_video[:, :, :expected_frames]
        elif pred_frames < expected_frames:
            self.logger.warning(
                f"Generated {pred_frames} frames, expected {expected_frames}"
            )
        original_video = original_video[:, :, :expected_frames]

        type_dir = output_dirs["root"] / sample_type
        type_dir.mkdir(parents=True, exist_ok=True)
        video_name = f"video_{video_index:03d}_ep{episode_index}.mp4"
        saved_path = self.video_saver.save_comparison(
            original_video, predicted_video, type_dir, video_name
        )

        self.logger.info(f"Saved: {saved_path.resolve()}\n")
