#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from tqdm import tqdm

import infer_cross_view_stage1 as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch inference/evaluation for cross_view_stage1 with LagerNVS synthesized first-frame conditioning."
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Path to the stage1 checkpoint file.",
    )
    parser.add_argument(
        "--config_json",
        required=True,
        help="Path to the checkpoint config.json.",
    )
    parser.add_argument(
        "--wrist_first_frame_index",
        required=True,
        help="JSON mapping (episode_index,start_frame) to LagerNVS synthesized wrist first-frame PNG path.",
    )
    parser.add_argument(
        "--strict_wrist_first_frame",
        action="store_true",
        help="Raise an error if any evaluated sample is missing from --wrist_first_frame_index.",
    )
    parser.add_argument(
        "--dataset_base_path",
        default=None,
        help="Override dataset base path. Defaults to config.json value.",
    )
    parser.add_argument(
        "--dataset_metadata_path",
        default=None,
        help="Override val manifest path.",
    )
    parser.add_argument(
        "--train_metadata_path",
        default=None,
        help="Optional train manifest path for preview generation.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to <ckpt_dir>/stage1_lagernvs_eval.",
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
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use. Values >1 launch one worker process per GPU.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated GPU ids for multi-GPU inference, e.g. 0,1,2,3. Defaults to 0..num_gpus-1.",
    )
    parser.add_argument("--worker_shard_index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_num_shards", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker_split",
        choices=["val", "train_preview"],
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace):
    config = base.build_config(args)
    config.wrist_first_frame_index = args.wrist_first_frame_index
    config.strict_wrist_first_frame = bool(args.strict_wrist_first_frame)
    config.num_gpus = int(args.num_gpus)
    config.gpu_ids = args.gpu_ids
    config.worker_shard_index = args.worker_shard_index
    config.worker_num_shards = args.worker_num_shards
    config.worker_split = args.worker_split
    if args.output_dir is None:
        config.output_dir = str(Path(args.ckpt_path).resolve().parent / "stage1_lagernvs_eval")
    return config


def _scalar_int(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.flatten()[0].item()
    elif isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    return int(value)


def _sample_index_key(sample: Dict[str, Any]) -> Optional[str]:
    episode_index = _scalar_int(sample.get("episode_index"))
    start_frame = _scalar_int(sample.get("start_frame"))
    if episode_index is None or start_frame is None:
        return None
    return f"{episode_index}_{start_frame}"


def initialize_model(config, logger):
    model = base.initialize_model(config)
    index_path = Path(config.wrist_first_frame_index)
    if not index_path.is_file():
        raise FileNotFoundError(f"Wrist first-frame index not found: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        model.wrist_first_frame_index = json.load(f)
    logger.info(
        "Loaded LagerNVS wrist first-frame index: %s entries from %s",
        len(model.wrist_first_frame_index),
        index_path,
    )
    return model


@torch.no_grad()
def generate_cross_view_stage1_lagernvs(
    model,
    sample: Dict[str, Any],
    config,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = model.transfer_data_to_device(sample, model.pipe.device, model.pipe.torch_dtype)
    video_gt = data["video"]
    model.validate_cross_view_video(video_gt)

    if getattr(config, "strict_wrist_first_frame", False):
        key = _sample_index_key(data)
        path = model.wrist_first_frame_index.get(key) if key is not None else None
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(
                "Missing LagerNVS wrist first frame for "
                f"key={key!r}; index={config.wrist_first_frame_index}"
            )

    cond_video = model.build_cross_view_condition_video(video_gt, meta=data)
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


def _video_to_unit_range(video: torch.Tensor) -> torch.Tensor:
    return ((video.float() + 1.0) * 0.5).clamp(0.0, 1.0)


def _compute_video_metrics(
    original_video: torch.Tensor,
    predicted_video: torch.Tensor,
    target_view_index: int,
    frame_start: int,
) -> Dict[str, float]:
    gt = _video_to_unit_range(original_video[target_view_index])
    pred = _video_to_unit_range(predicted_video[target_view_index])
    frames = min(int(gt.shape[1]), int(pred.shape[1]))
    if frame_start >= frames:
        return {"psnr": 0.0, "ssim": 0.0, "mse": 0.0, "frames": 0}
    gt = gt[:, frame_start:frames]
    pred = pred[:, frame_start:frames]
    mse_tensor = (pred - gt).pow(2).mean(dim=(0, 2, 3))
    mse = float(mse_tensor.mean().item())
    psnr_values = []
    for value in mse_tensor:
        mse_value = float(value.item())
        if mse_value <= 0:
            psnr_values.append(float("inf"))
        else:
            psnr_values.append(10.0 * math.log10(1.0 / mse_value))
    finite_psnr = [value for value in psnr_values if math.isfinite(value)]
    psnr = float(sum(finite_psnr) / len(finite_psnr)) if finite_psnr else float("inf")
    try:
        from skimage.metrics import structural_similarity

        gt_np = gt.permute(1, 2, 3, 0).numpy()
        pred_np = pred.permute(1, 2, 3, 0).numpy()
        ssim_values = [
            structural_similarity(
                gt_np[index],
                pred_np[index],
                channel_axis=-1,
                data_range=1.0,
            )
            for index in range(gt_np.shape[0])
        ]
        ssim = float(sum(ssim_values) / len(ssim_values)) if ssim_values else 0.0
    except Exception:
        ssim = -1.0
    return {"psnr": psnr, "ssim": ssim, "mse": mse, "frames": frames - frame_start}


def _sample_metadata(sample: Dict[str, Any], index: int) -> Dict[str, Any]:
    metadata = {"sample_index": int(index)}
    for key in ("episode_index", "start_frame", "end_frame", "valid_frames"):
        value = sample.get(key)
        if value is None:
            continue
        scalar = _scalar_int(value)
        metadata[key] = scalar if scalar is not None else value
    metadata["wrist_first_frame_key"] = _sample_index_key(sample)
    return metadata


def _summarize_sample_metrics(records: list[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"count": len(records)}
    for scope in ("target_all_frames", "target_exclude_first"):
        scoped = [record[scope] for record in records if record.get(scope, {}).get("frames", 0) > 0]
        if not scoped:
            summary[scope] = {"psnr": 0.0, "ssim": 0.0, "mse": 0.0, "frames": 0}
            continue
        out = {"frames": int(sum(item["frames"] for item in scoped))}
        for key in ("psnr", "ssim", "mse"):
            values = [float(item[key]) for item in scoped if math.isfinite(float(item[key]))]
            out[key] = float(sum(values) / len(values)) if values else float("inf")
        summary[scope] = out
    return summary


def save_split_predictions_with_sample_metrics(
    dataset,
    config,
    model,
    output_root: Path,
    split_name: str,
    limit: Optional[int],
    logger,
    sample_indices: Optional[list[int]] = None,
    metrics_suffix: str = "",
    video_subdir: Optional[str] = None,
) -> tuple[Path, Path, Path]:
    saver = base.VideoSaver(fps=config.fps, quality=config.quality)
    split_dir = output_root / (video_subdir or split_name)
    split_dir.mkdir(parents=True, exist_ok=True)
    total = len(dataset) if limit is None else min(int(limit), len(dataset))
    if sample_indices is None:
        indices = list(range(total))
    else:
        indices = [int(index) for index in sample_indices if 0 <= int(index) < total]
    logger.info(
        "Generating %s/%s samples for split=%s%s",
        len(indices),
        total,
        split_name,
        f" shard={metrics_suffix}" if metrics_suffix else "",
    )
    jsonl_path = output_root / f"{split_name}_sample_metrics{metrics_suffix}.jsonl"
    summary_path = output_root / f"{split_name}_sample_metrics{metrics_suffix}_summary.json"
    records = []
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for idx in indices:
            sample = dataset[idx]
            original_video, predicted_video = generate_cross_view_stage1_lagernvs(model, sample, config)
            episode_index = _scalar_int(sample.get("episode_index"))
            episode_label = episode_index if episode_index is not None else "unknown"
            video_name = f"{split_name}_{idx:06d}_ep{episode_label}.mp4"
            saver.save_comparison(original_video, predicted_video, split_dir, video_name)
            record = _sample_metadata(sample, idx)
            record["video_name"] = video_name
            record["target_view_index"] = int(config.cross_view_target_view)
            record["target_all_frames"] = _compute_video_metrics(
                original_video,
                predicted_video,
                target_view_index=int(config.cross_view_target_view),
                frame_start=0,
            )
            record["target_exclude_first"] = _compute_video_metrics(
                original_video,
                predicted_video,
                target_view_index=int(config.cross_view_target_view),
                frame_start=1,
            )
            records.append(record)
            handle.write(json.dumps(base.to_jsonable(record), ensure_ascii=False) + "\n")
            handle.flush()
    summary = _summarize_sample_metrics(records)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(base.to_jsonable(summary), handle, indent=2, ensure_ascii=False)
    logger.info("Saved per-sample metrics to %s", jsonl_path.resolve())
    logger.info("Saved per-sample metric summary to %s", summary_path.resolve())
    return split_dir, jsonl_path, summary_path


def _parse_gpu_ids(gpu_ids: Optional[str], num_gpus: int) -> list[str]:
    if num_gpus <= 0:
        raise ValueError(f"--num_gpus must be positive, got {num_gpus}")
    if gpu_ids:
        parsed = [item.strip() for item in gpu_ids.split(",") if item.strip()]
    elif os.environ.get("CUDA_VISIBLE_DEVICES"):
        parsed = [
            item.strip()
            for item in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            if item.strip()
        ]
    else:
        parsed = [str(index) for index in range(num_gpus)]
    if len(parsed) < num_gpus:
        raise ValueError(
            f"--gpu_ids provides {len(parsed)} ids but --num_gpus={num_gpus}: {gpu_ids!r}"
        )
    return parsed[:num_gpus]


def _append_cli_arg(command: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(f"--{name}")
        return
    command.extend([f"--{name}", str(value)])


def _build_worker_command(
    args: argparse.Namespace,
    output_dir: str,
    split_name: str,
    shard_index: int,
    num_shards: int,
) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve())]
    skip = {
        "num_gpus",
        "gpu_ids",
        "output_dir",
        "worker_shard_index",
        "worker_num_shards",
        "worker_split",
    }
    for name, value in vars(args).items():
        if name in skip:
            continue
        _append_cli_arg(command, name, value)
    command.extend(
        [
            "--output_dir",
            output_dir,
            "--num_gpus",
            "1",
            "--worker_shard_index",
            str(shard_index),
            "--worker_num_shards",
            str(num_shards),
            "--worker_split",
            split_name,
        ]
    )
    return command


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    records = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _merge_shard_sample_metrics(
    comparisons_root: Path,
    split_name: str,
    num_shards: int,
) -> tuple[Path, Path, Dict[str, Any]]:
    records = []
    missing = []
    for shard_index in range(num_shards):
        shard_path = comparisons_root / f"{split_name}_sample_metrics_shard_{shard_index:03d}.jsonl"
        if not shard_path.is_file():
            missing.append(str(shard_path))
            continue
        records.extend(_read_jsonl(shard_path))
    if missing:
        raise FileNotFoundError(
            "Missing shard metric files:\n" + "\n".join(missing)
        )
    records.sort(key=lambda item: int(item.get("sample_index", -1)))

    jsonl_path = comparisons_root / f"{split_name}_sample_metrics.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(base.to_jsonable(record), ensure_ascii=False) + "\n")

    summary = _summarize_sample_metrics(records)
    summary_path = comparisons_root / f"{split_name}_sample_metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(base.to_jsonable(summary), handle, indent=2, ensure_ascii=False)
    return jsonl_path, summary_path, summary


def _run_worker_split(args: argparse.Namespace, config, logger) -> None:
    if config.worker_shard_index is None or config.worker_num_shards is None or config.worker_split is None:
        raise ValueError("Worker mode requires --worker_shard_index, --worker_num_shards, and --worker_split")
    if config.worker_num_shards <= 0:
        raise ValueError(f"Invalid --worker_num_shards={config.worker_num_shards}")
    if not (0 <= config.worker_shard_index < config.worker_num_shards):
        raise ValueError(
            f"Invalid shard index {config.worker_shard_index} for {config.worker_num_shards} shards"
        )

    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    worker_config_path = (
        output_root
        / f"config_eval_{config.worker_split}_shard_{config.worker_shard_index:03d}.json"
    )
    with worker_config_path.open("w", encoding="utf-8") as handle:
        json.dump(config.__dict__, handle, indent=2, ensure_ascii=False)

    logger.info(
        "Worker starting: split=%s shard=%s/%s CUDA_VISIBLE_DEVICES=%s",
        config.worker_split,
        config.worker_shard_index,
        config.worker_num_shards,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    model = initialize_model(config, logger)
    base.generate_cross_view_stage1 = generate_cross_view_stage1_lagernvs

    if config.worker_split == "val":
        metadata_path = config.dataset_metadata_path
        limit = config.sample_limit
    else:
        if not config.train_metadata_path:
            raise ValueError("--worker_split=train_preview requires --train_metadata_path")
        metadata_path = config.train_metadata_path
        limit = config.num_train_preview

    dataset = base.build_dataset(metadata_path, config)
    total = len(dataset) if limit is None else min(int(limit), len(dataset))
    sample_indices = list(range(config.worker_shard_index, total, config.worker_num_shards))
    _, jsonl_path, summary_path = save_split_predictions_with_sample_metrics(
        dataset,
        config,
        model,
        output_root / "comparisons",
        config.worker_split,
        limit,
        logger,
        sample_indices=sample_indices,
        metrics_suffix=f"_shard_{config.worker_shard_index:03d}",
        video_subdir=config.worker_split,
    )

    payload = {
        "split": config.worker_split,
        "shard_index": config.worker_shard_index,
        "num_shards": config.worker_num_shards,
        "sample_count": len(sample_indices),
        "sample_metrics_jsonl": str(jsonl_path.resolve()),
        "sample_metrics_summary": str(summary_path.resolve()),
    }
    payload_path = (
        output_root
        / "comparisons"
        / f"{config.worker_split}_worker_shard_{config.worker_shard_index:03d}.json"
    )
    with payload_path.open("w", encoding="utf-8") as handle:
        json.dump(base.to_jsonable(payload), handle, indent=2, ensure_ascii=False)
    logger.info("Worker finished: %s", payload_path.resolve())


def _launch_split_workers(
    args: argparse.Namespace,
    config,
    split_name: str,
    gpu_ids: list[str],
    logger,
) -> None:
    output_root = Path(config.output_dir)
    log_dir = output_root / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]
    processes = []
    log_handles = []
    for shard_index, gpu_id in enumerate(gpu_ids):
        command = _build_worker_command(
            args,
            str(output_root),
            split_name,
            shard_index,
            len(gpu_ids),
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["PYTHONUNBUFFERED"] = "1"
        log_path = log_dir / f"{split_name}_shard_{shard_index:03d}_gpu_{gpu_id}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((shard_index, gpu_id, log_path, process))
        log_handles.append(log_handle)
        logger.info(
            "Launched %s worker shard=%03d gpu=%s pid=%s log=%s",
            split_name,
            shard_index,
            gpu_id,
            process.pid,
            log_path.resolve(),
        )

    failures = []
    try:
        for shard_index, gpu_id, log_path, process in processes:
            return_code = process.wait()
            logger.info(
                "%s worker shard=%03d gpu=%s exited with code %s",
                split_name,
                shard_index,
                gpu_id,
                return_code,
            )
            if return_code != 0:
                failures.append((shard_index, gpu_id, return_code, log_path))
    finally:
        for handle in log_handles:
            handle.close()
    if failures:
        details = "\n".join(
            f"shard={shard_index} gpu={gpu_id} code={code} log={log_path}"
            for shard_index, gpu_id, code, log_path in failures
        )
        raise RuntimeError(f"{split_name} worker failure(s):\n{details}")


def run_multigpu(args: argparse.Namespace, logger) -> None:
    config = build_config(args)
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "config_eval.json").open("w", encoding="utf-8") as handle:
        json.dump(config.__dict__, handle, indent=2, ensure_ascii=False)

    gpu_ids = _parse_gpu_ids(config.gpu_ids, config.num_gpus)
    comparisons_root = output_root / "comparisons"
    comparisons_root.mkdir(parents=True, exist_ok=True)
    logger.info("Starting multi-GPU inference with GPUs: %s", ",".join(gpu_ids))

    num_views = max(
        [config.cross_view_target_view]
        + [int(item) for item in str(config.cross_view_source_views).split(",") if item.strip()]
    ) + 1

    _launch_split_workers(args, config, "val", gpu_ids, logger)
    val_sample_metrics, val_sample_metrics_summary, _ = _merge_shard_sample_metrics(
        comparisons_root,
        "val",
        len(gpu_ids),
    )
    val_dir = comparisons_root / "val"
    logger.info("Computing val metrics from merged multi-GPU outputs...")
    val_metrics = base.summarize_metrics(
        base.compute_metrics_for_split(
            val_dir,
            num_views=num_views,
            frame_start=0,
            target_view_index=config.cross_view_target_view,
        ),
        base.compute_metrics_for_split(
            val_dir,
            num_views=num_views,
            frame_start=1,
            target_view_index=config.cross_view_target_view,
        ),
        config.cross_view_target_view,
    )

    train_metrics = None
    train_dir = None
    train_sample_metrics = None
    train_sample_metrics_summary = None
    if config.train_metadata_path and config.num_train_preview > 0:
        _launch_split_workers(args, config, "train_preview", gpu_ids, logger)
        train_sample_metrics, train_sample_metrics_summary, _ = _merge_shard_sample_metrics(
            comparisons_root,
            "train_preview",
            len(gpu_ids),
        )
        train_dir = comparisons_root / "train_preview"
        logger.info("Computing train preview metrics from merged multi-GPU outputs...")
        train_metrics = base.summarize_metrics(
            base.compute_metrics_for_split(
                train_dir,
                num_views=num_views,
                frame_start=0,
                target_view_index=config.cross_view_target_view,
            ),
            base.compute_metrics_for_split(
                train_dir,
                num_views=num_views,
                frame_start=1,
                target_view_index=config.cross_view_target_view,
            ),
            config.cross_view_target_view,
        )

    payload = {
        "checkpoint_path": config.checkpoint_path,
        "wrist_first_frame_index": config.wrist_first_frame_index,
        "multi_gpu": True,
        "gpu_ids": gpu_ids,
        "num_gpus": len(gpu_ids),
        "val_comparison_dir": str(val_dir.resolve()),
        "val_metrics": val_metrics,
        "val_sample_metrics_jsonl": str(val_sample_metrics.resolve()),
        "val_sample_metrics_summary": str(val_sample_metrics_summary.resolve()),
        "train_preview_comparison_dir": str(train_dir.resolve()) if train_dir else None,
        "train_preview_metrics": train_metrics,
        "train_preview_sample_metrics_jsonl": str(train_sample_metrics.resolve()) if train_sample_metrics else None,
        "train_preview_sample_metrics_summary": str(train_sample_metrics_summary.resolve()) if train_sample_metrics_summary else None,
    }
    with (output_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(base.to_jsonable(payload), handle, indent=2, ensure_ascii=False)
    logger.info("Saved multi-GPU metrics to %s", (output_root / "metrics.json").resolve())


def main() -> None:
    args = parse_args()
    logger = base.setup_logger()
    if args.worker_shard_index is None and int(args.num_gpus) > 1:
        run_multigpu(args, logger)
        return

    config = build_config(args)
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.worker_shard_index is not None:
        _run_worker_split(args, config, logger)
        return

    with (output_root / "config_eval.json").open("w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2, ensure_ascii=False)

    logger.info("Loading model with LagerNVS first-frame conditioning...")
    model = initialize_model(config, logger)
    base.generate_cross_view_stage1 = generate_cross_view_stage1_lagernvs

    num_views = max(
        [config.cross_view_target_view]
        + [int(item) for item in str(config.cross_view_source_views).split(",") if item.strip()]
    ) + 1

    logger.info("Loading val dataset: %s", config.dataset_metadata_path)
    val_dataset = base.build_dataset(config.dataset_metadata_path, config)
    val_dir, val_sample_metrics, val_sample_metrics_summary = save_split_predictions_with_sample_metrics(
        val_dataset,
        config,
        model,
        output_root / "comparisons",
        "val",
        config.sample_limit,
        logger,
    )
    logger.info("Computing val metrics...")
    val_metrics = base.summarize_metrics(
        base.compute_metrics_for_split(
            val_dir,
            num_views=num_views,
            frame_start=0,
            target_view_index=config.cross_view_target_view,
        ),
        base.compute_metrics_for_split(
            val_dir,
            num_views=num_views,
            frame_start=1,
            target_view_index=config.cross_view_target_view,
        ),
        config.cross_view_target_view,
    )

    train_metrics = None
    train_dir = None
    train_sample_metrics = None
    train_sample_metrics_summary = None
    if config.train_metadata_path and config.num_train_preview > 0:
        logger.info("Loading train preview dataset: %s", config.train_metadata_path)
        train_dataset = base.build_dataset(config.train_metadata_path, config)
        train_dir, train_sample_metrics, train_sample_metrics_summary = save_split_predictions_with_sample_metrics(
            train_dataset,
            config,
            model,
            output_root / "comparisons",
            "train_preview",
            config.num_train_preview,
            logger,
        )
        logger.info("Computing train preview metrics...")
        train_metrics = base.summarize_metrics(
            base.compute_metrics_for_split(
                train_dir,
                num_views=num_views,
                frame_start=0,
                target_view_index=config.cross_view_target_view,
            ),
            base.compute_metrics_for_split(
                train_dir,
                num_views=num_views,
                frame_start=1,
                target_view_index=config.cross_view_target_view,
            ),
            config.cross_view_target_view,
        )

    payload = {
        "checkpoint_path": config.checkpoint_path,
        "wrist_first_frame_index": config.wrist_first_frame_index,
        "val_comparison_dir": str(val_dir.resolve()),
        "val_metrics": val_metrics,
        "val_sample_metrics_jsonl": str(val_sample_metrics.resolve()),
        "val_sample_metrics_summary": str(val_sample_metrics_summary.resolve()),
        "train_preview_comparison_dir": str(train_dir.resolve()) if train_dir else None,
        "train_preview_metrics": train_metrics,
        "train_preview_sample_metrics_jsonl": str(train_sample_metrics.resolve()) if train_sample_metrics else None,
        "train_preview_sample_metrics_summary": str(train_sample_metrics_summary.resolve()) if train_sample_metrics_summary else None,
    }
    with (output_root / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(base.to_jsonable(payload), f, indent=2, ensure_ascii=False)
    logger.info("Saved metrics to %s", (output_root / "metrics.json").resolve())


if __name__ == "__main__":
    main()
