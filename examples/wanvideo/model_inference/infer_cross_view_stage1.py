#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadDroidState, ResolvePromptEmbPath
from diffsynth.diffusion.parsers import prepare_wan_runtime
from infer_robot import VideoSaver

from examples.wanvideo.model_training.train import WanTrainingModule


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("infer_cross_view_stage1")


def flatten_grouped_config(grouped: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for value in grouped.values():
        if isinstance(value, dict):
            merged.update(value)
    return merged


@dataclass
class EvalConfig:
    checkpoint_path: str
    config_json: str
    dataset_base_path: str
    dataset_metadata_path: str
    train_metadata_path: Optional[str]
    output_dir: str
    cfg_scale: float
    num_inference_steps: int
    sigma_shift: float
    seed: int
    fps: int
    quality: int
    sample_limit: Optional[int]
    num_train_preview: int
    negative_prompt: str
    negative_prompt_emb: Optional[str]
    state_type: str
    state_stat_path: str
    model_paths: str
    load_modules: str
    num_frames: int
    num_history_frames: int
    height: int
    width: int
    resize_mode: str
    cross_view_source_views: str
    cross_view_target_view: int
    cross_view_placeholder_mode: str
    cross_view_source_loss_weight: float
    cross_view_old_branch_dropout: float
    cross_view_projector_hidden_dim: int
    task: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch inference/evaluation for cross_view_stage1 checkpoints."
    )
    parser.add_argument(
        "--ckpt_path",
        default="/data1/blm/DiffSynth-Studio/Ckpt/droid_crossview_small200_stage1/epoch-79/epoch-79.safetensors",
        help="Path to the stage1 checkpoint file.",
    )
    parser.add_argument(
        "--config_json",
        default="/data1/blm/DiffSynth-Studio/Ckpt/droid_crossview_small200_stage1/epoch-79/config.json",
        help="Path to the checkpoint config.json.",
    )
    parser.add_argument(
        "--dataset_base_path",
        default=None,
        help="Override dataset base path. Defaults to config.json value.",
    )
    parser.add_argument(
        "--dataset_metadata_path",
        default=None,
        help="Override val manifest path. Defaults to <dataset_base_path>/meta/episodes_cross_view_val_81_small50.jsonl if it exists.",
    )
    parser.add_argument(
        "--train_metadata_path",
        default=None,
        help="Optional train manifest path for preview generation. Defaults to <dataset_base_path>/meta/episodes_cross_view_train_81_small200.jsonl if it exists.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to <ckpt_dir>/stage1_eval.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="CFG scale for evaluation. Defaults to 1.0 to match training semantics.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--sigma_shift",
        type=float,
        default=5.0,
        help="Flow-matching sigma shift.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed. Defaults to config.json seed.",
    )
    parser.add_argument("--fps", type=int, default=None, help="Output comparison FPS.")
    parser.add_argument("--quality", type=int, default=5, help="Output video quality.")
    parser.add_argument("--sample_limit", type=int, default=None, help="Optional cap on number of val samples.")
    parser.add_argument("--num_train_preview", type=int, default=8, help="Number of train preview samples.")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "The video is not of a high quality, it has a low resolution. "
            "Watermark present in each frame. The background is solid. "
            "Strange body and strange trajectory. Distortion"
        ),
        help="Negative prompt used when cfg_scale != 1.",
    )
    parser.add_argument(
        "--negative_prompt_emb",
        type=str,
        default=None,
        help="Optional negative prompt embedding path.",
    )
    parser.add_argument(
        "--state_stat_path",
        type=str,
        default=None,
        help="Optional override for state normalization stats JSON.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EvalConfig:
    with open(args.config_json, "r", encoding="utf-8") as f:
        grouped = json.load(f)
    merged = flatten_grouped_config(grouped)
    dataset_base_path = args.dataset_base_path or merged["dataset_base_path"]
    default_val_manifest = os.path.join(
        dataset_base_path, "meta", "episodes_cross_view_val_81_small50.jsonl"
    )
    default_train_manifest = os.path.join(
        dataset_base_path, "meta", "episodes_cross_view_train_81_small200.jsonl"
    )
    dataset_metadata_path = args.dataset_metadata_path or (
        default_val_manifest
        if os.path.exists(default_val_manifest)
        else merged["dataset_metadata_path"]
    )
    train_metadata_path = args.train_metadata_path or (
        default_train_manifest if os.path.exists(default_train_manifest) else None
    )
    output_dir = args.output_dir or str(Path(args.ckpt_path).resolve().parent / "stage1_eval")
    negative_prompt_emb = args.negative_prompt_emb
    if negative_prompt_emb is None:
        candidate = os.path.join(dataset_base_path, "prompt_emb", "neg_prompt.pt")
        negative_prompt_emb = candidate if os.path.exists(candidate) else None
    state_stat_path = args.state_stat_path or merged.get("state_stat_path")
    if not state_stat_path:
        candidate = os.path.join(dataset_base_path, "meta", "stat_state_pose_7d.json")
        state_stat_path = candidate if os.path.exists(candidate) else None
    if not state_stat_path:
        raise FileNotFoundError(
            "Unable to resolve state_stat_path. Pass --state_stat_path explicitly "
            "or ensure <dataset_base_path>/meta/stat_state_pose_7d.json exists."
        )

    return EvalConfig(
        checkpoint_path=args.ckpt_path,
        config_json=args.config_json,
        dataset_base_path=dataset_base_path,
        dataset_metadata_path=dataset_metadata_path,
        train_metadata_path=train_metadata_path,
        output_dir=output_dir,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        seed=int(args.seed if args.seed is not None else merged["seed"]),
        fps=int(args.fps if args.fps is not None else 15),
        quality=int(args.quality),
        sample_limit=args.sample_limit,
        num_train_preview=int(args.num_train_preview),
        negative_prompt=args.negative_prompt,
        negative_prompt_emb=negative_prompt_emb,
        state_type=merged["state_type"],
        state_stat_path=state_stat_path,
        model_paths=merged["model_paths"],
        load_modules=merged["load_modules"],
        num_frames=int(merged["num_frames"]),
        num_history_frames=int(merged["num_history_frames"]),
        height=int(merged["height"]),
        width=int(merged["width"]),
        resize_mode=merged.get("resize_mode", "fit"),
        cross_view_source_views=merged["cross_view_source_views"],
        cross_view_target_view=int(merged["cross_view_target_view"]),
        cross_view_placeholder_mode=merged["cross_view_placeholder_mode"],
        cross_view_source_loss_weight=float(merged["cross_view_source_loss_weight"]),
        cross_view_old_branch_dropout=float(merged["cross_view_old_branch_dropout"]),
        cross_view_projector_hidden_dim=int(merged["cross_view_projector_hidden_dim"]),
        task=merged["task"],
    )


def build_dataset(metadata_path: str, config: EvalConfig) -> UnifiedDataset:
    dataset = UnifiedDataset(
        base_path=config.dataset_base_path,
        metadata_path=metadata_path,
        repeat=1,
        data_file_keys=("video", "state", "prompt_emb"),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=config.dataset_base_path,
            height=config.height,
            width=config.width,
            num_frames=config.num_frames,
            resize_mode=config.resize_mode,
        ),
        special_operator_map={
            "prompt_emb": ResolvePromptEmbPath(base_path=config.dataset_base_path),
        },
        stat_path=config.state_stat_path,
        action_type=None,
    )
    dataset.special_operator_map["state"] = LoadDroidState(
        base_path=config.dataset_base_path,
        state_type=config.state_type,
        stat=dataset.stat,
        num_frames=config.num_frames,
    )
    return dataset


def initialize_model(config: EvalConfig) -> WanTrainingModule:
    runtime = prepare_wan_runtime(
        config.model_paths,
        config.load_modules,
        ["video", "state", "prompt_emb"],
    )
    model = WanTrainingModule(
        model_paths=json.dumps(runtime["model_paths"]),
        tokenizer_path=runtime["tokenizer_path"],
        trainable_models="dit,action_encoder",
        modules=runtime["modules"],
        ckpt_path=config.checkpoint_path,
        task=config.task,
        device="cuda",
        num_history_frames=config.num_history_frames,
        cross_view_source_views=config.cross_view_source_views,
        cross_view_target_view=config.cross_view_target_view,
        cross_view_placeholder_mode=config.cross_view_placeholder_mode,
        cross_view_source_loss_weight=config.cross_view_source_loss_weight,
        cross_view_old_branch_dropout=config.cross_view_old_branch_dropout,
        cross_view_projector_hidden_dim=config.cross_view_projector_hidden_dim,
        state_type=config.state_type,
    )
    model.eval()
    model.requires_grad_(False)
    model.pipe.eval()
    return model


@torch.no_grad()
def generate_cross_view_stage1(
    model: WanTrainingModule,
    sample: Dict[str, Any],
    config: EvalConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = model.transfer_data_to_device(sample, model.pipe.device, model.pipe.torch_dtype)
    video_gt = data["video"]
    model.validate_cross_view_video(video_gt)
    cond_video = model.build_cross_view_condition_video(video_gt)
    inputs = model.build_cross_view_inputs(data, cond_video, seed=config.seed)
    inputs_shared, inputs_posi, inputs_nega = inputs
    inputs_shared["cfg_scale"] = config.cfg_scale
    inputs_shared["seed"] = config.seed
    inputs_nega["negative_prompt"] = config.negative_prompt
    if config.negative_prompt_emb is not None:
        inputs_nega["prompt_emb"] = config.negative_prompt_emb
    for unit in model.iter_cross_view_units():
        inputs = model.pipe.unit_runner(unit, model.pipe, *inputs)
    inputs_shared, inputs_posi, inputs_nega = inputs

    cond_latents = model.encode_joint_video_latents(cond_video)
    history_t = ((config.num_history_frames - 1) // 4) + 1

    model.pipe.scheduler.set_timesteps(
        config.num_inference_steps,
        denoising_strength=1.0,
        shift=config.sigma_shift,
    )
    latents = inputs_shared["noise"]
    latents[:, :, :history_t] = cond_latents[:, :, :history_t]
    inputs_shared["latents"] = latents

    models = {name: getattr(model.pipe, name) for name in model.pipe.in_iteration_models}
    for progress_id, timestep in enumerate(tqdm(model.pipe.scheduler.timesteps, leave=False)):
        timestep = timestep.unsqueeze(0).to(dtype=model.pipe.torch_dtype, device=model.pipe.device)
        noise_pred_posi = model.pipe.model_fn(
            **models,
            **inputs_shared,
            **inputs_posi,
            timestep=timestep,
        )
        if config.cfg_scale != 1.0:
            noise_pred_nega = model.pipe.model_fn(
                **models,
                **inputs_shared,
                **inputs_nega,
                timestep=timestep,
            )
            noise_pred = noise_pred_nega + config.cfg_scale * (
                noise_pred_posi - noise_pred_nega
            )
        else:
            noise_pred = noise_pred_posi

        inputs_shared["latents"] = model.pipe.scheduler.step(
            noise_pred,
            model.pipe.scheduler.timesteps[progress_id],
            inputs_shared["latents"],
        )
        inputs_shared["latents"][:, :, :history_t] = cond_latents[:, :, :history_t]

    latents = inputs_shared["latents"]
    num_views = int(video_gt.shape[0])
    latents_by_view = torch.reshape(
        latents,
        (
            latents.shape[0],
            latents.shape[1],
            latents.shape[2],
            num_views,
            latents.shape[-2] // num_views,
            latents.shape[-1],
        ),
    )
    latents_by_view = latents_by_view.permute(0, 3, 1, 2, 4, 5).reshape(
        -1,
        latents.shape[1],
        latents.shape[2],
        latents.shape[-2] // num_views,
        latents.shape[-1],
    )
    predicted_video = model.pipe.vae.decode(
        latents_by_view,
        device=model.pipe.device,
        tiled=False,
    )
    predicted_video = predicted_video.detach().cpu()
    predicted_video = predicted_video[:num_views, :, : video_gt.shape[2]]
    return sample["video"].detach().cpu(), predicted_video


def save_split_predictions(
    dataset: UnifiedDataset,
    config: EvalConfig,
    model: WanTrainingModule,
    output_root: Path,
    split_name: str,
    limit: Optional[int],
    logger: logging.Logger,
) -> Path:
    saver = VideoSaver(fps=config.fps, quality=config.quality)
    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    total = len(dataset) if limit is None else min(int(limit), len(dataset))
    logger.info("Generating %s samples for split=%s", total, split_name)
    for idx in range(total):
        sample = dataset[idx]
        original_video, predicted_video = generate_cross_view_stage1(model, sample, config)
        video_name = f"{split_name}_{idx:03d}_ep{sample['episode_index']}.mp4"
        saver.save_comparison(original_video, predicted_video, split_dir, video_name)
    return split_dir


def compute_metrics_for_split(
    split_dir: Path,
    num_views: int,
    frame_start: int = 0,
    target_view_index: Optional[int] = None,
) -> dict:
    try:
        from diffsynth.core.metric.metric import evaluate as builtin_evaluate

        return builtin_evaluate(
            str(split_dir),
            num_workers=16,
            num_views=num_views,
            frame_start=frame_start,
            target_view_index=target_view_index,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "decord":
            raise
        return fallback_evaluate(
            str(split_dir),
            num_views=num_views,
            frame_start=frame_start,
        )


def fallback_evaluate(comparison_dir: str, num_views: int, frame_start: int = 0) -> dict:
    import imageio
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    comparison_videos = []
    for root, _, files in os.walk(comparison_dir):
        for filename in files:
            if filename.lower().endswith(".mp4"):
                comparison_videos.append(os.path.join(root, filename))
    comparison_videos = sorted(comparison_videos)
    if not comparison_videos:
        raise RuntimeError(f"No comparison videos found in {comparison_dir}")

    view_sums = [{"psnr": 0.0, "ssim": 0.0, "mse": 0.0, "frames": 0} for _ in range(num_views)]
    total_psnr_sum = 0.0
    total_ssim_sum = 0.0
    total_mse_sum = 0.0
    total_frames = 0

    for video_path in tqdm(comparison_videos, desc="Computing fallback metrics"):
        reader = imageio.get_reader(video_path)
        frames = []
        for frame in reader:
            frames.append(frame.astype(np.float32) / 255.0)
        reader.close()
        frames = np.asarray(frames)
        row_splits = np.array_split(frames, num_views, axis=1)
        for view_idx, row in enumerate(row_splits):
            gt_video, pred_video = np.array_split(row, 2, axis=2)
            min_frames = min(len(gt_video), len(pred_video))
            if min_frames <= frame_start:
                continue
            for frame_idx in range(frame_start, min_frames):
                gt_frame = gt_video[frame_idx]
                pred_frame = pred_video[frame_idx]
                min_h = min(gt_frame.shape[0], pred_frame.shape[0])
                min_w = min(gt_frame.shape[1], pred_frame.shape[1])
                gt_frame = gt_frame[:min_h, :min_w]
                pred_frame = pred_frame[:min_h, :min_w]
                psnr = peak_signal_noise_ratio(pred_frame, gt_frame, data_range=1.0)
                ssim = structural_similarity(pred_frame, gt_frame, channel_axis=-1, data_range=1.0)
                mse = float(np.mean((pred_frame - gt_frame) ** 2))
                view_sums[view_idx]["psnr"] += psnr
                view_sums[view_idx]["ssim"] += ssim
                view_sums[view_idx]["mse"] += mse
                view_sums[view_idx]["frames"] += 1
                total_psnr_sum += psnr
                total_ssim_sum += ssim
                total_mse_sum += mse
                total_frames += 1

    view_metrics = []
    for view in view_sums:
        if view["frames"] > 0:
            view_metrics.append(
                {
                    "psnr": view["psnr"] / view["frames"],
                    "ssim": view["ssim"] / view["frames"],
                    "mse": view["mse"] / view["frames"],
                    "lpips": -1.0,
                    "frames": view["frames"],
                }
            )
        else:
            view_metrics.append(
                {"psnr": 0.0, "ssim": 0.0, "mse": 0.0, "lpips": -1.0, "frames": 0}
            )

    avg_psnr = total_psnr_sum / total_frames if total_frames else 0.0
    avg_ssim = total_ssim_sum / total_frames if total_frames else 0.0
    avg_mse = total_mse_sum / total_frames if total_frames else 0.0
    return {
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "avg_mse": avg_mse,
        "avg_lpips": -1.0,
        "view_metrics": view_metrics,
        "extended_metrics": None,
    }


def summarize_metrics(metrics_all: dict, metrics_exclude_first: dict, target_view: int) -> dict:
    return {
        "all_views": metrics_all,
        "all_views_exclude_first": metrics_exclude_first,
        "target_view_index": int(target_view),
        "target_view": metrics_all["view_metrics"][target_view],
        "target_view_exclude_first": metrics_exclude_first["view_metrics"][target_view],
        "target_view_extended": metrics_all.get("extended_metrics"),
        "target_view_exclude_first_extended": metrics_exclude_first.get("extended_metrics"),
        "evaluated_frame_start": {
            "all_views": 0,
            "all_views_exclude_first": 1,
            "target_view_exclude_first": 1,
        },
    }


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> None:
    args = parse_args()
    logger = setup_logger()
    config = build_config(args)
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "config_eval.json").open("w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2, ensure_ascii=False)

    logger.info("Loading model...")
    model = initialize_model(config)
    num_views = max(
        [config.cross_view_target_view]
        + [int(item) for item in str(config.cross_view_source_views).split(",") if item.strip()]
    ) + 1

    logger.info("Loading val dataset: %s", config.dataset_metadata_path)
    val_dataset = build_dataset(config.dataset_metadata_path, config)
    val_dir = save_split_predictions(
        val_dataset,
        config,
        model,
        output_root / "comparisons",
        "val",
        config.sample_limit,
        logger,
    )
    logger.info("Computing val metrics...")
    val_metrics = summarize_metrics(
        compute_metrics_for_split(
            val_dir,
            num_views=num_views,
            frame_start=0,
            target_view_index=config.cross_view_target_view,
        ),
        compute_metrics_for_split(
            val_dir,
            num_views=num_views,
            frame_start=1,
            target_view_index=config.cross_view_target_view,
        ),
        config.cross_view_target_view,
    )

    train_metrics = None
    train_dir = None
    if config.train_metadata_path and config.num_train_preview > 0:
        logger.info("Loading train preview dataset: %s", config.train_metadata_path)
        train_dataset = build_dataset(config.train_metadata_path, config)
        train_dir = save_split_predictions(
            train_dataset,
            config,
            model,
            output_root / "comparisons",
            "train_preview",
            config.num_train_preview,
            logger,
        )
        logger.info("Computing train preview metrics...")
        train_metrics = summarize_metrics(
            compute_metrics_for_split(
                train_dir,
                num_views=num_views,
                frame_start=0,
                target_view_index=config.cross_view_target_view,
            ),
            compute_metrics_for_split(
                train_dir,
                num_views=num_views,
                frame_start=1,
                target_view_index=config.cross_view_target_view,
            ),
            config.cross_view_target_view,
        )

    payload = {
        "checkpoint_path": config.checkpoint_path,
        "val_comparison_dir": str(val_dir.resolve()),
        "val_metrics": val_metrics,
        "train_preview_comparison_dir": str(train_dir.resolve()) if train_dir else None,
        "train_preview_metrics": train_metrics,
    }
    with (output_root / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)
    logger.info("Saved metrics to %s", (output_root / "metrics.json").resolve())


if __name__ == "__main__":
    main()
