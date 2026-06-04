#!/usr/bin/env python
"""Refresh cached cross-view samples with keyframe anchors in the y channel."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffsynth.diffusion.parsers import prepare_wan_runtime
from examples.wanvideo.model_training.train import WanTrainingModule, set_global_seed
from tool.build_cross_view_latent_cache import (
    _build_legacy_image_branch,
    _load_first_frame_image,
    _load_first_frame_image_from_path,
    _lookup_wrist_tail_frame_path,
    _parse_int_pair,
)
from tool.cross_view_keyframe_helpers import (
    KEYFRAME_ANCHOR_LOOKUP_MODE,
    load_keyframe_anchor_index,
    resolve_keyframe_anchors,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Copy an existing cross-view cache and refresh only `y` with keyframe anchors."
    )
    parser.add_argument("--dataset_base_path", type=str, required=True)
    parser.add_argument("--train_metadata_path", type=str, required=True)
    parser.add_argument("--val_metadata_path", type=str, default=None)
    parser.add_argument("--src_cache_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--model_paths", type=str, required=True)
    parser.add_argument("--load_modules", type=str, default="dit,text:emb,vae,image,action:noise")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_history_frames", type=int, default=1)
    parser.add_argument("--cross_view_source_views", type=str, default="0,1")
    parser.add_argument("--cross_view_target_view", type=int, default=2)
    parser.add_argument("--cross_view_placeholder_mode", type=str, default="zeros")
    parser.add_argument("--state_type", type=str, default="state_pose_7d")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wrist_first_frame_index", type=str, required=True)
    parser.add_argument("--cross_view_use_tail_anchor", type=int, default=1, choices=[0, 1])
    parser.add_argument("--num_tail_frames", type=int, default=1)
    parser.add_argument("--cross_view_tail_anchor_dropout", type=float, default=0.0)
    parser.add_argument("--tail_anchor_segment_stride", type=int, default=81)
    parser.add_argument("--num_keyframe_anchors", type=int, default=3)
    parser.add_argument("--keyframe_anchor_dropout", type=float, default=0.0)
    parser.add_argument("--keyframe_anchor_manifest_train", type=str, required=True)
    parser.add_argument("--keyframe_anchor_manifest_val", type=str, default=None)
    parser.add_argument("--keyframe_anchor_image_root_train", type=str, required=True)
    parser.add_argument("--keyframe_anchor_image_root_val", type=str, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--vae_tiled_encode", action="store_true")
    parser.add_argument("--vae_tile_size", type=str, default="34,34")
    parser.add_argument("--vae_tile_stride", type=str, default="18,16")
    return parser


def load_manifest(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_model(args, runtime):
    model = WanTrainingModule(
        model_paths=json.dumps(runtime["model_paths"]),
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=runtime["tokenizer_path"],
        trainable_models=None,
        modules=runtime["modules"],
        device=args.device,
        task="cross_view_stage1",
        num_history_frames=args.num_history_frames,
        cross_view_source_views=args.cross_view_source_views,
        cross_view_target_view=args.cross_view_target_view,
        cross_view_placeholder_mode=args.cross_view_placeholder_mode,
        state_type=args.state_type,
        cross_view_use_tail_anchor=int(args.cross_view_use_tail_anchor),
        num_tail_frames=int(args.num_tail_frames),
        cross_view_tail_anchor_dropout=0.0,
        cross_view_use_keyframe_anchor=1,
        num_keyframe_anchors=int(args.num_keyframe_anchors),
        keyframe_anchor_dropout=0.0,
    )
    model.eval()
    model.requires_grad_(False)
    model.pipe.eval()
    return model


def load_video_from_latents_shape(sample: dict, model) -> torch.Tensor:
    latent_views = sample["latent_views_gt"]
    num_views = int(latent_views.shape[0])
    return torch.zeros(
        num_views,
        3,
        int(sample["num_frames"]),
        int(sample["height"]),
        int(sample["width"]),
        dtype=model.pipe.torch_dtype,
        device=model.pipe.device,
    )


@torch.no_grad()
def refresh_split(
    model,
    split_name: str,
    manifest_rows: list[dict],
    src_root: Path,
    dst_root: Path,
    wrist_index: dict,
    keyframe_index: dict,
    args,
):
    src_dir = src_root / split_name
    dst_dir = dst_root / split_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source cache split not found: {src_dir}")
    vae_tile_size = _parse_int_pair(args.vae_tile_size, "vae_tile_size")
    vae_tile_stride = _parse_int_pair(args.vae_tile_stride, "vae_tile_stride")
    indices = list(range(len(manifest_rows)))
    if int(args.num_shards) > 1:
        indices = indices[int(args.shard_index)::int(args.num_shards)]

    for sample_id in tqdm(indices, desc=f"refresh-y:{split_name}"):
        src_path = src_dir / f"{sample_id:07d}.pth"
        dst_path = dst_dir / f"{sample_id:07d}.pth"
        if args.skip_existing and dst_path.is_file():
            continue
        if not src_path.is_file():
            raise FileNotFoundError(f"Source cache sample not found: {src_path}")
        sample = torch.load(src_path, map_location="cpu", weights_only=False)
        if not isinstance(sample, dict):
            raise TypeError(f"Cache sample must be a dict: {src_path}")
        meta = dict(manifest_rows[sample_id])
        meta["sample_id"] = int(sample_id)
        video_gt = load_video_from_latents_shape(sample, model)
        target_view = model.cross_view_target_view

        first_frame_image = _load_first_frame_image(
            meta,
            wrist_index,
            int(sample["height"]),
            int(sample["width"]),
            model.pipe.device,
            model.pipe.torch_dtype,
        )
        if first_frame_image is not None:
            video_gt[target_view, :, 0] = first_frame_image.squeeze(0)

        tail_frames = int(args.num_tail_frames) if bool(int(args.cross_view_use_tail_anchor)) else 0
        if tail_frames > 0:
            tail_frame = None
            if float(args.cross_view_tail_anchor_dropout) <= 0.0 or float(torch.rand(()).item()) >= float(args.cross_view_tail_anchor_dropout):
                tail_path = _lookup_wrist_tail_frame_path(
                    meta,
                    wrist_index,
                    segment_stride=int(args.tail_anchor_segment_stride),
                )
                tail_frame = _load_first_frame_image_from_path(
                    tail_path,
                    int(sample["height"]),
                    int(sample["width"]),
                    model.pipe.device,
                    model.pipe.torch_dtype,
                )
            if tail_frame is not None:
                video_gt[target_view, :, -1] = tail_frame.squeeze(0)

        keyframe_anchor_indices: list[int] = []
        anchors = resolve_keyframe_anchors(keyframe_index, meta, sample_id=int(sample_id))
        if len(anchors) != int(args.num_keyframe_anchors):
            raise KeyError(
                f"Expected {int(args.num_keyframe_anchors)} keyframes for {split_name} "
                f"sample {sample_id}, got {len(anchors)}."
            )
        for anchor in anchors:
            offset = int(anchor["offset"])
            keyframe_anchor_indices.append(offset)
            if (
                float(args.keyframe_anchor_dropout) > 0.0
                and float(torch.rand(()).item()) < float(args.keyframe_anchor_dropout)
            ):
                continue
            frame = _load_first_frame_image_from_path(
                anchor.get("path"),
                int(sample["height"]),
                int(sample["width"]),
                model.pipe.device,
                model.pipe.torch_dtype,
            )
            if frame is None:
                raise FileNotFoundError(f"Missing keyframe anchor image: {anchor}")
            video_gt[target_view, :, offset] = frame.squeeze(0)

        data = {
            "prompt_emb": sample.get("prompt_emb"),
            "state": sample.get("state"),
            "action": sample.get("action"),
            "sample_id": int(sample_id),
            **meta,
        }
        legacy_branch = _build_legacy_image_branch(
            model,
            data,
            video_gt,
            tiled=bool(args.vae_tiled_encode),
            tile_size=vae_tile_size,
            tile_stride=vae_tile_stride,
            num_tail_frames=tail_frames,
            anchor_frame_indices=keyframe_anchor_indices,
            include_clip=False,
        )
        refreshed = dict(sample)
        refreshed["y"] = legacy_branch["y"].detach().cpu()
        refreshed["sample_id"] = int(sample_id)
        refreshed["anchor_frame_indices"] = torch.tensor(
            sorted(set(keyframe_anchor_indices)), dtype=torch.long
        )
        tmp = dst_path.with_suffix(".pth.tmp")
        torch.save({k: v for k, v in refreshed.items() if v is not None}, tmp)
        os.replace(tmp, dst_path)


def main():
    parser = build_parser()
    args = parser.parse_args()
    set_global_seed(args.seed)
    if int(args.num_shards) < 1:
        raise ValueError("--num_shards must be >= 1")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard_index must be in [0, num_shards)")

    runtime = prepare_wan_runtime(
        args.model_paths,
        args.load_modules,
        ["video", "state", "prompt_emb"],
    )
    model = build_model(args, runtime)
    with open(args.wrist_first_frame_index, "r", encoding="utf-8") as handle:
        wrist_index = json.load(handle)

    src_root = Path(args.src_cache_root)
    dst_root = Path(args.output_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    src_config_path = src_root / "cache_config.json"
    if src_config_path.is_file():
        with src_config_path.open("r", encoding="utf-8") as handle:
            cache_config = json.load(handle)
    else:
        cache_config = {}
    cache_config.update(
        {
            "cross_view_use_keyframe_anchor": True,
            "num_keyframe_anchors": int(args.num_keyframe_anchors),
            "keyframe_anchor_lookup_mode": KEYFRAME_ANCHOR_LOOKUP_MODE,
            "keyframe_anchor_dropout": float(args.keyframe_anchor_dropout),
            "keyframe_anchor_manifest_train": args.keyframe_anchor_manifest_train,
            "keyframe_anchor_manifest_val": args.keyframe_anchor_manifest_val,
            "keyframe_anchor_image_root_train": args.keyframe_anchor_image_root_train,
            "keyframe_anchor_image_root_val": args.keyframe_anchor_image_root_val,
            "cross_view_use_tail_anchor": bool(int(args.cross_view_use_tail_anchor)),
            "num_tail_frames": int(args.num_tail_frames),
            "tail_anchor_lookup_mode": "end_frame_index",
            "cross_view_tail_anchor_dropout": float(args.cross_view_tail_anchor_dropout),
        }
    )
    tmp_config = dst_root / f".cache_config.{os.getpid()}.tmp"
    with tmp_config.open("w", encoding="utf-8") as handle:
        json.dump(cache_config, handle, indent=2)
    os.replace(tmp_config, dst_root / "cache_config.json")

    train_rows = load_manifest(args.train_metadata_path)
    train_index = load_keyframe_anchor_index(
        args.keyframe_anchor_manifest_train,
        args.keyframe_anchor_image_root_train,
        num_keyframes=int(args.num_keyframe_anchors),
        num_frames=int(args.num_frames),
    )
    refresh_split(model, "train", train_rows, src_root, dst_root, wrist_index, train_index, args)

    if args.val_metadata_path:
        if not args.keyframe_anchor_manifest_val or not args.keyframe_anchor_image_root_val:
            raise ValueError("Val keyframe manifest/root are required when --val_metadata_path is set.")
        val_rows = load_manifest(args.val_metadata_path)
        val_index = load_keyframe_anchor_index(
            args.keyframe_anchor_manifest_val,
            args.keyframe_anchor_image_root_val,
            num_keyframes=int(args.num_keyframe_anchors),
            num_frames=int(args.num_frames),
        )
        refresh_split(model, "val", val_rows, src_root, dst_root, wrist_index, val_index, args)
    elif (src_root / "val").is_dir() and not (dst_root / "val").is_dir():
        shutil.copytree(src_root / "val", dst_root / "val")


if __name__ == "__main__":
    main()
