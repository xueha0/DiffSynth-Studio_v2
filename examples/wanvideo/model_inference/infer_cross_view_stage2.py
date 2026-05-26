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
from diffsynth.core.data.operators import (
    LoadDroidCameraTokens,
    LoadDroidState,
    ResolvePromptEmbPath,
)
from diffsynth.diffusion.parsers import prepare_wan_runtime
from infer_robot import VideoSaver

from examples.wanvideo.model_training.train import WanTrainingModule


VIEW_CAMERA_PREFIX = {
    0: "left_external",
    1: "right_external",
    2: "wrist",
}


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("infer_cross_view_stage2")


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
    cross_view_source_injection_mode: str
    cross_view_source_branch_mode: str
    cross_view_source_window_radius: int
    cross_view_source_gate_mode: str
    cross_view_temp_loss_weight: float
    cross_view_state_loss_weight: float
    cross_view_global_source_tokens: int
    cross_view_old_branch_dropout_schedule: str
    cross_view_legacy_branch_schedule: Optional[str]
    cross_view_disable_legacy_image_branch: int
    # Dual-end anchor (tail anchor) — must match the training-time flags so the
    # model gets `cross_view_use_tail_anchor=1` at inference. If the checkpoint
    # was trained without dual-end anchoring these stay at 0 / 1 / 0.0 and the
    # tail-anchor code paths degrade to the original single-end behavior.
    cross_view_use_tail_anchor: int
    num_tail_frames: int
    cross_view_tail_anchor_dropout: float
    # Inference-time ablation switch: when True, force num_tail_frames=0 in
    # the y-channel encoder so the model sees a head-only InP signal even
    # when the ckpt was trained dual-end. Useful for diagnosing whether the
    # tail anchor signal helps at inference.
    disable_tail_anchor_at_inference: bool
    scene_token_checkpoint: Optional[str]
    scene_token_pool_size: int
    geometry_gate_mode: str
    geometry_sidecar_cache_path: Optional[str]
    geometry_use_camera_tokens: int
    geometry_target_camera_mode: str
    geometry_scene_token_source: str
    wrist_first_frame_index: Optional[str]
    task: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch inference/evaluation for cross_view_stage2 checkpoints."
    )
    parser.add_argument(
        "--ckpt_path",
        default="/data1/blm/DiffSynth-Studio/Ckpt/droid_crossview_small200_stage2/epoch-79/epoch-79.safetensors",
        help="Path to the stage2 checkpoint file.",
    )
    parser.add_argument(
        "--config_json",
        default="/data1/blm/DiffSynth-Studio/Ckpt/droid_crossview_small200_stage2/epoch-79/config.json",
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
        "--disable_tail_anchor_at_inference",
        action="store_true",
        help=(
            "Inference-time ablation: skip the tail-anchor latent overwrite "
            "(force tail_t=0 in the denoise loop) even if the ckpt was "
            "trained with cross_view_use_tail_anchor=1. Useful to diagnose "
            "whether a blurry synthesized tail anchor (e.g. VAE(LagerNVS)) "
            "is dragging down inference quality. Does not change the model "
            "instance's training-time flags; only short-circuits the "
            "tail_t / target_tail_latents derivation in the inference loop."
        ),
    )
    parser.add_argument(
        "--train_metadata_path",
        default=None,
        help="Optional train manifest path for preview generation. Defaults to <dataset_base_path>/meta/episodes_cross_view_train_81_small200.jsonl if it exists.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to <ckpt_dir>/stage2_eval.",
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
    parser.add_argument("--quality", type=int, default=9, help="Output video quality (imageio scale 0-10; higher = less compression artifacts contaminating PSNR/SSIM/LPIPS).")
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
        "--geometry_sidecar_cache_path",
        type=str,
        default=None,
        help="Optional geometry sidecar cache root. Defaults to training config when present.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of inference shards (typically = number of GPUs).",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Index of the current shard in [0, num_shards). Each shard owns sample indices i where i %% num_shards == shard_index.",
    )
    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        help="Skip metric computation. Useful when running per-GPU shards in parallel; "
             "compute metrics once after all shards finish via a separate aggregator pass.",
    )
    parser.add_argument(
        "--skip_train_preview",
        action="store_true",
        help="Skip the training-set preview step (saves time during multi-GPU val-only inference).",
    )
    parser.add_argument(
        "--state_stat_path",
        type=str,
        default=None,
        help=(
            "Override state normalization stats JSON. Required when the training "
            "config.json has state_stat_path=null. If unset, falls back to "
            "<dataset_base_path>/meta/stat_state_pose_7d.json when present."
        ),
    )
    parser.add_argument(
        "--wrist_first_frame_index",
        type=str,
        default=None,
        help=(
            "Override the wrist_first_frame_index JSON path. When the training "
            "config.json has this field as null (e.g. cache-only training), the "
            "wrist anchor frame silently falls back to a zero placeholder, which "
            "causes a hard train-test distribution mismatch (gray first frame at "
            "inference). If unset here, also falls back to "
            "<dataset_base_path>/meta/wrist_first_frame_index_all.json when present."
        ),
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
    output_dir = args.output_dir or str(Path(args.ckpt_path).resolve().parent / "stage2_eval")
    negative_prompt_emb = args.negative_prompt_emb
    if negative_prompt_emb is None:
        candidate = os.path.join(dataset_base_path, "prompt_emb", "neg_prompt.pt")
        negative_prompt_emb = candidate if os.path.exists(candidate) else None

    # state_stat_path resolution order:
    #   1. CLI override (--state_stat_path)
    #   2. training config.json (may legitimately be None for cached runs)
    #   3. fallback to <dataset_base_path>/meta/stat_state_pose_7d.json (repo convention)
    state_stat_path = (
        args.state_stat_path
        if args.state_stat_path
        else merged.get("state_stat_path")
    )
    if not state_stat_path:
        candidate = os.path.join(
            dataset_base_path, "meta", "stat_state_pose_7d.json"
        )
        if os.path.exists(candidate):
            state_stat_path = candidate
    if not state_stat_path:
        raise FileNotFoundError(
            "state_stat_path is required for stage2 inference but was not found. "
            "Pass --state_stat_path explicitly, or place stat_state_pose_7d.json under "
            f"{dataset_base_path}/meta/."
        )

    # wrist_first_frame_index resolution order (CLI > config.json > fallback).
    # When the training command did not pass --wrist_first_frame_index, the
    # saved config.json keeps it as None, but the cache may still have been
    # built with synthesized first frames (has_wrist_first_frame=true). Use the
    # repo-convention fallback to recover the train-time anchor distribution.
    wrist_first_frame_index = (
        args.wrist_first_frame_index
        if args.wrist_first_frame_index
        else merged.get("wrist_first_frame_index")
    )
    if not wrist_first_frame_index:
        candidate = os.path.join(
            dataset_base_path, "meta", "wrist_first_frame_index_all.json"
        )
        if os.path.exists(candidate):
            wrist_first_frame_index = candidate

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
        cross_view_old_branch_dropout=float(
            merged.get("cross_view_old_branch_dropout", 0.0)
        ),
        cross_view_projector_hidden_dim=int(merged["cross_view_projector_hidden_dim"]),
        cross_view_source_injection_mode=merged.get(
            "cross_view_source_injection_mode",
            "temporal_local",
        ),
        cross_view_source_branch_mode=merged.get(
            "cross_view_source_branch_mode",
            "sigma_matched_clamp",
        ),
        cross_view_source_window_radius=int(
            merged.get("cross_view_source_window_radius", 1)
        ),
        cross_view_source_gate_mode=merged.get(
            "cross_view_source_gate_mode",
            "scalar",
        ),
        cross_view_temp_loss_weight=float(
            merged.get("cross_view_temp_loss_weight", 0.1)
        ),
        cross_view_state_loss_weight=float(
            merged.get("cross_view_state_loss_weight", 0.05)
        ),
        cross_view_global_source_tokens=int(
            merged.get("cross_view_global_source_tokens", 0)
        ),
        cross_view_old_branch_dropout_schedule=merged.get(
            "cross_view_old_branch_dropout_schedule",
            "linear_warmup_to_high",
        ),
        cross_view_legacy_branch_schedule=merged.get(
            "cross_view_legacy_branch_schedule",
            None,
        ),
        cross_view_disable_legacy_image_branch=int(
            merged.get("cross_view_disable_legacy_image_branch", 0)
        ),
        cross_view_use_tail_anchor=int(
            merged.get("cross_view_use_tail_anchor", 0)
        ),
        num_tail_frames=int(merged.get("num_tail_frames", 1)),
        # Force-disable dropout at inference. Training-time stochastic dropout
        # (e.g. 0.1) is a regularizer; replaying it at eval would inject random
        # tail-anchor masking and ruin reproducibility.
        cross_view_tail_anchor_dropout=0.0,
        disable_tail_anchor_at_inference=bool(
            getattr(args, "disable_tail_anchor_at_inference", False)
        ),
        scene_token_checkpoint=merged.get("scene_token_checkpoint", None),
        scene_token_pool_size=int(merged.get("scene_token_pool_size", 512)),
        geometry_gate_mode=merged.get("geometry_gate_mode", "learned"),
        geometry_sidecar_cache_path=(
            args.geometry_sidecar_cache_path
            if args.geometry_sidecar_cache_path is not None
            else merged.get("geometry_sidecar_cache_path", None)
        ),
        geometry_use_camera_tokens=int(merged.get("geometry_use_camera_tokens", 0)),
        geometry_target_camera_mode=merged.get("geometry_target_camera_mode", "none"),
        geometry_scene_token_source=merged.get(
            "geometry_scene_token_source",
            "cached_zero_cam",
        ),
        wrist_first_frame_index=wrist_first_frame_index,
        task=merged["task"],
    )


def build_dataset(metadata_path: str, config: EvalConfig) -> UnifiedDataset:
    # Camera tokens at inference are sourced from (priority order):
    #   1. geometry sidecar (preferred; matches training distribution)
    #   2. attach_runtime_camera_tokens fallback (reads parquet directly)
    # Letting UnifiedDataset's special_operator_map also load them creates a
    # subtle ordering bug: LoadDroidState mutates data["state"] from
    # {"data": parquet, ...} into a (1,T,7) ndarray *in place*; the camera
    # operator that runs afterwards then receives the ndarray and crashes in
    # `_resolve_parquet_info` ("array truth ambiguous"). Since sidecar +
    # runtime fallback already cover both source and target cam tokens, we
    # never need the dataset-level camera operators.
    use_sidecar_or_runtime_for_cams = True
    load_cam_tokens_via_dataset = (
        int(config.geometry_use_camera_tokens) and not use_sidecar_or_runtime_for_cams
    )
    data_file_keys = (
        ("video", "state", "prompt_emb", "source_camera_tokens", "target_camera_tokens")
        if load_cam_tokens_via_dataset
        else ("video", "state", "prompt_emb")
    )
    dataset = UnifiedDataset(
        base_path=config.dataset_base_path,
        metadata_path=metadata_path,
        repeat=1,
        data_file_keys=data_file_keys,
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
    if load_cam_tokens_via_dataset:
        source_views = tuple(
            int(item)
            for item in str(config.cross_view_source_views).split(",")
            if item.strip()
        )
        dataset.special_operator_map["source_camera_tokens"] = LoadDroidCameraTokens(
            base_path=config.dataset_base_path,
            role="source",
            view_indices=source_views,
            num_frames=config.num_frames,
        )
        dataset.special_operator_map["target_camera_tokens"] = LoadDroidCameraTokens(
            base_path=config.dataset_base_path,
            role="target",
            view_indices=(int(config.cross_view_target_view),),
            num_frames=config.num_frames,
        )
    return dataset


def camera_columns_for_view(view_index: int) -> tuple[str, str]:
    prefix = VIEW_CAMERA_PREFIX.get(int(view_index))
    if prefix is None:
        raise ValueError(f"No camera prefix mapping for view index {view_index}.")
    return f"{prefix}_camera_intrinsics", f"{prefix}_camera_to_robot_extrinsics"


def resolve_state_parquet_path(dataset_base_path: str, state_ref) -> str:
    path = state_ref.get("data") if isinstance(state_ref, dict) else state_ref
    if not path:
        raise KeyError("Missing state parquet path in sample metadata.")
    if os.path.isabs(path):
        return path
    return os.path.join(dataset_base_path, path)


def sample_metadata_row(sample: Dict[str, Any]) -> Dict[str, Any]:
    row = sample.get("__metadata_row__")
    return row if isinstance(row, dict) else sample


def resolve_state_slice(row: dict, num_frames: int) -> tuple[int, int, int | None, str | None]:
    state_ref = row.get("state")
    if isinstance(state_ref, dict):
        start = int(state_ref.get("start_frame", row.get("start_frame", 0)))
        end = int(state_ref.get("end_frame", start + int(num_frames) - 1))
        pad_to_frames = state_ref.get("pad_to_frames")
        pad_mode = state_ref.get("pad_mode")
    else:
        start = int(row.get("start_frame", 0))
        end = int(row.get("end_frame", start + int(num_frames) - 1))
        pad_to_frames = row.get("pad_to_frames")
        pad_mode = row.get("pad_mode")
    return (
        start,
        end,
        None if pad_to_frames is None else int(pad_to_frames),
        None if pad_mode is None else str(pad_mode),
    )


def read_camera_sequence_from_sample(
    sample: Dict[str, Any],
    config: EvalConfig,
    view_index: int,
    num_frames: int,
) -> torch.Tensor:
    import pyarrow.parquet as pq

    intr_col, extr_col = camera_columns_for_view(view_index)
    row = sample_metadata_row(sample)
    parquet_path = resolve_state_parquet_path(config.dataset_base_path, row.get("state"))
    start, row_end, pad_to_frames, pad_mode = resolve_state_slice(row, num_frames)
    table = pq.read_table(parquet_path, columns=[intr_col, extr_col])
    intr = np.asarray(table[intr_col].to_pylist(), dtype=np.float32)
    extr = np.asarray(table[extr_col].to_pylist(), dtype=np.float32)
    read_length = min(int(num_frames), max(1, row_end - start + 1))
    end = min(start + read_length, intr.shape[0], extr.shape[0])
    intr = intr[start:end]
    extr = extr[start:end]
    if intr.shape[0] == 0:
        raise ValueError(f"Empty camera slice for {parquet_path}, start={start}.")
    if intr.shape[0] < int(num_frames):
        if pad_mode not in (None, "repeat_last"):
            raise ValueError(f"Unsupported camera pad_mode={pad_mode!r} for {parquet_path}.")
        pad = int(num_frames) - intr.shape[0]
        intr = np.concatenate([intr, np.repeat(intr[-1:], pad, axis=0)], axis=0)
        extr = np.concatenate([extr, np.repeat(extr[-1:], pad, axis=0)], axis=0)
    return torch.from_numpy(np.concatenate([intr[:num_frames], extr[:num_frames]], axis=-1))


def attach_runtime_camera_tokens(
    sample: Dict[str, Any],
    config: EvalConfig,
    model: WanTrainingModule,
) -> Dict[str, Any]:
    need_source_tokens = bool(int(config.geometry_use_camera_tokens))
    need_target_tokens = config.geometry_target_camera_mode == "add_time_mlp"
    if not (need_source_tokens or need_target_tokens):
        return sample
    if (
        (not need_source_tokens or "source_cam_tokens" in sample or "source_camera_tokens" in sample)
        and (
            not need_target_tokens
            or "target_cam_tokens_latent" in sample
            or "target_cam_tokens" in sample
            or "target_camera_tokens" in sample
        )
    ):
        return sample
    updated = dict(sample)
    try:
        source_views = [
            int(item)
            for item in str(config.cross_view_source_views).split(",")
            if item.strip()
        ]
        if need_source_tokens and "source_cam_tokens" not in updated and "source_camera_tokens" not in updated:
            source_tokens = [
                read_camera_sequence_from_sample(sample, config, view_index, 1)[0]
                for view_index in source_views
            ]
            updated["source_cam_tokens"] = torch.stack(source_tokens, dim=0).unsqueeze(0)
        if (
            need_target_tokens
            and "target_cam_tokens_latent" not in updated
            and "target_cam_tokens" not in updated
            and "target_camera_tokens" not in updated
        ):
            target_tokens = read_camera_sequence_from_sample(
                sample,
                config,
                int(config.cross_view_target_view),
                int(config.num_frames),
            ).unsqueeze(0)
            updated["target_cam_tokens"] = target_tokens
            latent_length = ((int(config.num_frames) - 1) // 4) + 1
            updated["target_cam_tokens_latent"] = model.downsample_camera_sequence(
                target_tokens.to(dtype=model.pipe.torch_dtype, device=model.pipe.device),
                latent_length,
            ).detach().cpu()
    except Exception as exc:
        logging.getLogger("infer_cross_view_stage2").warning(
            "Camera-token loading failed; geometry will fall back when possible: %s",
            exc,
        )
    return updated


def attach_geometry_sidecar_for_inference(
    sample: Dict[str, Any],
    split_name: str,
    sample_index: int,
    config: EvalConfig,
) -> Dict[str, Any]:
    if not config.geometry_sidecar_cache_path:
        return sample
    sidecar_path = (
        Path(config.geometry_sidecar_cache_path)
        / split_name
        / f"{int(sample_index):07d}.pth"
    )
    if not sidecar_path.is_file():
        if config.geometry_scene_token_source == "camera_aware_sidecar":
            raise FileNotFoundError(f"Geometry sidecar not found: {sidecar_path}")
        return sample
    sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    if not isinstance(sidecar, dict):
        raise TypeError(f"Geometry sidecar must contain a dict: {sidecar_path}")
    updated = dict(sample)
    updated.update(sidecar)
    return updated


def initialize_model(config: EvalConfig) -> WanTrainingModule:
    runtime = prepare_wan_runtime(
        config.model_paths,
        config.load_modules,
        ["video", "state", "prompt_emb"],
    )
    trainable_models = [
        "dit",
        "action_encoder",
        "source_video_projector",
        "source_temporal_gate",
        "target_state_head",
    ]
    if config.geometry_target_camera_mode == "add_time_mlp":
        trainable_models.append("target_camera_encoder")
    if config.scene_token_checkpoint is not None:
        trainable_models.extend(["scene_token_adapter", "geometry_gates"])
    model = WanTrainingModule(
        model_paths=json.dumps(runtime["model_paths"]),
        tokenizer_path=runtime["tokenizer_path"],
        trainable_models=",".join(dict.fromkeys(trainable_models)),
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
        cross_view_source_injection_mode=config.cross_view_source_injection_mode,
        cross_view_source_branch_mode=config.cross_view_source_branch_mode,
        cross_view_source_window_radius=config.cross_view_source_window_radius,
        cross_view_source_gate_mode=config.cross_view_source_gate_mode,
        cross_view_temp_loss_weight=config.cross_view_temp_loss_weight,
        cross_view_state_loss_weight=config.cross_view_state_loss_weight,
        cross_view_global_source_tokens=config.cross_view_global_source_tokens,
        cross_view_old_branch_dropout_schedule=config.cross_view_old_branch_dropout_schedule,
        cross_view_legacy_branch_schedule=config.cross_view_legacy_branch_schedule,
        cross_view_disable_legacy_image_branch=config.cross_view_disable_legacy_image_branch,
        cross_view_use_tail_anchor=config.cross_view_use_tail_anchor,
        num_tail_frames=config.num_tail_frames,
        cross_view_tail_anchor_dropout=config.cross_view_tail_anchor_dropout,
        state_type=config.state_type,
        scene_token_checkpoint=config.scene_token_checkpoint,
        scene_token_pool_size=config.scene_token_pool_size,
        geometry_gate_mode=config.geometry_gate_mode,
        geometry_sidecar_cache_path=config.geometry_sidecar_cache_path,
        geometry_use_camera_tokens=config.geometry_use_camera_tokens,
        geometry_target_camera_mode=config.geometry_target_camera_mode,
        geometry_scene_token_source=config.geometry_scene_token_source,
    )
    model.eval()
    model.requires_grad_(False)
    model.pipe.eval()
    if config.wrist_first_frame_index and os.path.exists(config.wrist_first_frame_index):
        with open(config.wrist_first_frame_index, "r", encoding="utf-8") as f:
            model.wrist_first_frame_index = json.load(f)
    return model


@torch.no_grad()
def generate_cross_view_stage2(
    model: WanTrainingModule,
    sample: Dict[str, Any],
    config: EvalConfig,
    dataset: Optional[UnifiedDataset] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample = attach_runtime_camera_tokens(sample, config, model)
    data = model.transfer_data_to_device(sample, model.pipe.device, model.pipe.torch_dtype)
    video_gt = data["video"]
    model.validate_cross_view_video(video_gt)

    # Plan A (WAN-Fun-InP aligned): cond_video[wrist, :, 0] = synth first
    # frame, cond_video[wrist, :, -1] = next-segment first frame (or zero
    # placeholder for the last segment of an episode). This is the same
    # mechanism `build_cross_view_condition_video` uses at training time.
    # The y-channel (built by WanVideoUnit_ImageEmbedderVAE in iter_cross_view_units)
    # encodes both anchors with the correct slot-position semantics, and
    # the mask channel marks the head & tail latent slots as known.
    # No latent-overwrite is performed inside the denoise loop.
    cond_video = model.build_cross_view_condition_video(video_gt, meta=data)
    inputs = model.build_cross_view_inputs(data, cond_video, seed=config.seed)
    inputs_shared, inputs_posi, inputs_nega = inputs
    # Inference-time tail-anchor ablation: force num_tail_frames=0 in the
    # pipeline-unit input so the y channel is built head-only, even when the
    # ckpt was trained dual-end. Useful for diagnosing whether end-anchor
    # signal helps at inference.
    if config.disable_tail_anchor_at_inference:
        inputs_shared["num_tail_frames"] = 0
    inputs_shared["cfg_scale"] = config.cfg_scale
    inputs_shared["seed"] = config.seed
    inputs_nega["negative_prompt"] = config.negative_prompt
    if config.negative_prompt_emb is not None:
        inputs_nega["prompt_emb"] = config.negative_prompt_emb
    for unit in model.iter_cross_view_units():
        inputs = model.pipe.unit_runner(unit, model.pipe, *inputs)
    inputs_shared, inputs_posi, inputs_nega = inputs

    inputs_shared = model.maybe_drop_old_branch(inputs_shared, allow_dropout=False)

    latent_views_gt = model.encode_video_latents_by_view(video_gt)
    target_x0_latents = model.select_target_latents(latent_views_gt)
    source_x0_latents = model.select_source_latents(latent_views_gt)
    if "y" in inputs_shared:
        inputs_shared["y"] = model.select_target_legacy_y(
            inputs_shared["y"], int(video_gt.shape[0])
        )
    condition_sequence = model.get_cross_view_condition_sequence(
        data,
        num_frames=int(video_gt.shape[2]),
    )
    inputs_shared.update(
        model.build_cross_view_source_condition(
            condition_sequence=condition_sequence,
            source_latents=source_x0_latents,
        )
    )
    inputs_shared.update(
        model.build_target_camera_condition(data, latent_length=target_x0_latents.shape[2])
    )
    if config.geometry_scene_token_source == "camera_aware_sidecar":
        inputs_shared.update(model._build_geometry_aware_inputs(data))
    else:
        inputs_shared.update(model._build_geometry_aware_inputs_from_video(video_gt, data=data))

    model.pipe.scheduler.set_timesteps(
        config.num_inference_steps,
        denoising_strength=1.0,
        shift=config.sigma_shift,
    )
    latents = model.pipe.generate_noise(
        tuple(target_x0_latents.shape),
        seed=config.seed,
        rand_device="cpu",
        device=model.pipe.device,
        torch_dtype=model.pipe.torch_dtype,
    )
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

    latents = inputs_shared["latents"]
    predicted_video = model.pipe.vae.decode(
        latents,
        device=model.pipe.device,
        tiled=False,
    )
    predicted_video = predicted_video.detach().cpu()
    predicted_video = predicted_video[:1, :, : video_gt.shape[2]]
    full_predicted_video = sample["video"].detach().cpu().clone()
    full_predicted_video[config.cross_view_target_view : config.cross_view_target_view + 1] = (
        predicted_video[:, :, : full_predicted_video.shape[2]]
    )
    return sample["video"].detach().cpu(), full_predicted_video


def save_split_predictions(
    dataset: UnifiedDataset,
    config: EvalConfig,
    model: WanTrainingModule,
    output_root: Path,
    split_name: str,
    limit: Optional[int],
    logger: logging.Logger,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Path:
    saver = VideoSaver(fps=config.fps, quality=config.quality)
    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    total = len(dataset) if limit is None else min(int(limit), len(dataset))
    num_shards = max(1, int(num_shards))
    shard_index = int(shard_index)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index={shard_index} must be in [0, {num_shards})"
        )
    my_indices = list(range(shard_index, total, num_shards))
    logger.info(
        "Generating split=%s shard=%d/%d -> %d samples (out of %d)",
        split_name, shard_index, num_shards, len(my_indices), total,
    )
    sidecar_split = "train" if split_name.startswith("train") else split_name
    for idx in my_indices:
        video_name = f"{split_name}_{idx:03d}_ep__placeholder__.mp4"  # ep filled below
        # idempotent skip: if a previous shard already wrote this idx, skip.
        existing = list(split_dir.glob(f"{split_name}_{idx:03d}_ep*.mp4"))
        if existing:
            logger.info("  skip idx=%d (already exists: %s)", idx, existing[0].name)
            continue
        sample = dataset[idx]
        if not getattr(dataset, "load_from_cache", False) and getattr(dataset, "data", None):
            sample = dict(sample)
            sample["__metadata_row__"] = dataset.data[idx % len(dataset.data)]
        sample = attach_geometry_sidecar_for_inference(sample, sidecar_split, idx, config)
        original_video, predicted_video = generate_cross_view_stage2(model, sample, config, dataset=dataset)
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


def summarize_metrics(metrics_all: dict, metrics_exclude_first: dict | None, target_view: int) -> dict:
    summary = {
        "all_views": metrics_all,
        "target_view_index": int(target_view),
        "metric_view_index": int(target_view),
        "target_view": metrics_all["view_metrics"][target_view],
        "target_view_extended": metrics_all.get("extended_metrics"),
        "evaluated_frame_start": {
            "all_views": 0,
        },
    }
    # frame_start=1 (exclude-first) evaluation is disabled by default to avoid
    # running the full PSNR/SSIM/LPIPS/FID/FVD pipeline twice. To re-enable, pass
    # a non-None metrics_exclude_first dict from main() — see infer_cross_view_stage2.py:main.
    if metrics_exclude_first is not None:
        summary["all_views_exclude_first"] = metrics_exclude_first
        summary["target_view_exclude_first"] = metrics_exclude_first["view_metrics"][target_view]
        summary["target_view_exclude_first_extended"] = metrics_exclude_first.get("extended_metrics")
        summary["evaluated_frame_start"].update(
            {
                "all_views_exclude_first": 1,
                "target_view_exclude_first": 1,
            }
        )
    return summary


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
        num_shards=int(getattr(args, "num_shards", 1)),
        shard_index=int(getattr(args, "shard_index", 0)),
    )
    if getattr(args, "skip_metrics", False):
        logger.info(
            "skip_metrics=True; this shard finished generation. "
            "Run a separate aggregator pass (num_shards=1, shard_index=0) "
            "without --skip_metrics to compute metrics across all shards."
        )
        val_metrics = None
    else:
        logger.info("Computing val metrics (frame_start=0 only)...")
        val_metrics = summarize_metrics(
            compute_metrics_for_split(
                val_dir,
                num_views=num_views,
                frame_start=0,
                target_view_index=config.cross_view_target_view,
            ),
            # NOTE: frame_start=1 (exclude-first) pass is disabled to avoid
            # running the full PSNR/SSIM/LPIPS/FID/FVD pipeline twice. To
            # re-enable, replace `None` below with a second
            # compute_metrics_for_split(..., frame_start=1, ...) call.
            # compute_metrics_for_split(
            #     val_dir,
            #     num_views=num_views,
            #     frame_start=1,
            #     target_view_index=config.cross_view_target_view,
            # ),
            None,
            config.cross_view_target_view,
        )

    train_metrics = None
    train_dir = None
    if (
        config.train_metadata_path
        and config.num_train_preview > 0
        and not getattr(args, "skip_train_preview", False)
    ):
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
            num_shards=int(getattr(args, "num_shards", 1)),
            shard_index=int(getattr(args, "shard_index", 0)),
        )
        if getattr(args, "skip_metrics", False):
            train_metrics = None
        else:
            logger.info("Computing train preview metrics (frame_start=0 only)...")
            train_metrics = summarize_metrics(
                compute_metrics_for_split(
                    train_dir,
                    num_views=num_views,
                    frame_start=0,
                    target_view_index=config.cross_view_target_view,
                ),
                # NOTE: frame_start=1 pass disabled (see val_metrics above).
                # compute_metrics_for_split(
                #     train_dir,
                #     num_views=num_views,
                #     frame_start=1,
                #     target_view_index=config.cross_view_target_view,
                # ),
                None,
                config.cross_view_target_view,
            )

    payload = {
        "checkpoint_path": config.checkpoint_path,
        "val_comparison_dir": str(val_dir.resolve()),
        "val_metrics": val_metrics,
        "train_preview_comparison_dir": str(train_dir.resolve()) if train_dir else None,
        "train_preview_metrics": train_metrics,
        "shard_info": {
            "num_shards": int(getattr(args, "num_shards", 1)),
            "shard_index": int(getattr(args, "shard_index", 0)),
            "skip_metrics": bool(getattr(args, "skip_metrics", False)),
        },
    }
    metrics_filename = (
        f"metrics_shard{int(getattr(args, 'shard_index', 0))}.json"
        if int(getattr(args, "num_shards", 1)) > 1 and getattr(args, "skip_metrics", False)
        else "metrics.json"
    )
    with (output_root / metrics_filename).open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)
    logger.info("Saved metrics to %s", (output_root / "metrics.json").resolve())


if __name__ == "__main__":
    main()
