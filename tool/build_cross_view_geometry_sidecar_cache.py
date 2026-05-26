#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from diffsynth.models.scene_token_extractor import SceneTokenExtractor


VIEW_CAMERA_PREFIX = {
    0: "left_external",
    1: "right_external",
    2: "wrist",
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build camera-aware geometry sidecar caches for cross-view WAN training."
    )
    parser.add_argument("--dataset_base_path", type=str, required=True)
    parser.add_argument("--main_cache_path", type=str, required=True)
    parser.add_argument("--train_metadata_path", type=str, required=True)
    parser.add_argument("--val_metadata_path", type=str, default=None)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--scene_token_checkpoint", type=str, required=True)
    parser.add_argument("--cross_view_source_views", type=str, default="0,1")
    parser.add_argument("--cross_view_target_view", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_history_frames", type=int, default=1)
    parser.add_argument(
        "--lagernvs_root",
        type=str,
        default=os.environ.get("LAGERNVS_ROOT", "/data_ywj/data_xh/projects/LagerNVS"),
        help="Path to the LagerNVS repo. Used to reuse its camera normalization utilities.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def parse_views(value: str) -> tuple[int, ...]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(items) == 0:
        raise ValueError("Expected at least one source view.")
    return tuple(int(item) for item in items)


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_parquet_path(dataset_base_path: str, state_ref) -> str:
    if isinstance(state_ref, dict):
        path = state_ref.get("data")
    else:
        path = state_ref
    if not path:
        raise KeyError("Missing state parquet path in metadata row.")
    if os.path.isabs(path):
        return path
    return os.path.join(dataset_base_path, path)


def resolve_video_path(dataset_base_path: str, video_ref) -> str:
    if isinstance(video_ref, dict):
        path = video_ref.get("data")
    else:
        path = video_ref
    if not path:
        raise KeyError("Missing video path in metadata row.")
    if os.path.isabs(path):
        return path
    return os.path.join(dataset_base_path, path)


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


def camera_columns_for_view(view_index: int) -> tuple[str, str]:
    prefix = VIEW_CAMERA_PREFIX.get(int(view_index))
    if prefix is None:
        raise ValueError(f"No camera prefix mapping for view index {view_index}.")
    return f"{prefix}_camera_intrinsics", f"{prefix}_camera_to_robot_extrinsics"


@lru_cache(maxsize=1)
def load_lagernvs_camera_modules(lagernvs_root: str):
    root = Path(lagernvs_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LagerNVS repo not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from data import camera_utils, normalization
    from vggt.utils import pose_enc
    from vggt.utils.rotation import quat_to_mat

    return camera_utils, normalization, pose_enc, quat_to_mat


@lru_cache(maxsize=16)
def load_camera_column_pair(parquet_path: str, intr_col: str, extr_col: str) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=[intr_col, extr_col])
    intr = np.asarray(table[intr_col].to_pylist(), dtype=np.float32)
    extr = np.asarray(table[extr_col].to_pylist(), dtype=np.float32)
    return intr, extr


@lru_cache(maxsize=512)
def read_video_hw(video_path: str) -> tuple[int, int]:
    import imageio

    reader = imageio.get_reader(video_path)
    try:
        meta = reader.get_meta_data()
        size = meta.get("size")
        if size is not None and len(size) == 2:
            width, height = int(size[0]), int(size[1])
            return height, width
        frame = reader.get_data(0)
        return int(frame.shape[0]), int(frame.shape[1])
    finally:
        reader.close()


def video_hw_for_view(dataset_base_path: str, row: dict, view_index: int) -> tuple[int, int]:
    videos = row.get("video")
    if not isinstance(videos, (list, tuple)) or int(view_index) >= len(videos):
        raise KeyError(f"Metadata row is missing video entry for view {view_index}.")
    return read_video_hw(resolve_video_path(dataset_base_path, videos[int(view_index)]))


def read_camera_raw_sequence(
    dataset_base_path: str,
    row: dict,
    view_index: int,
    num_frames: int,
) -> np.ndarray:
    intr_col, extr_col = camera_columns_for_view(view_index)
    parquet_path = resolve_parquet_path(dataset_base_path, row.get("state"))
    start, row_end, pad_to_frames, pad_mode = resolve_state_slice(row, num_frames)
    read_length = min(int(num_frames), max(1, row_end - start + 1))
    end = start + read_length
    intr, extr = load_camera_column_pair(parquet_path, intr_col, extr_col)
    if end > intr.shape[0] or end > extr.shape[0]:
        end = min(intr.shape[0], extr.shape[0])
    intr = intr[start:end]
    extr = extr[start:end]
    if intr.shape[0] == 0:
        raise ValueError(f"Empty camera slice for {parquet_path}, start={start}.")
    if intr.shape[0] < num_frames:
        if pad_mode not in (None, "repeat_last"):
            raise ValueError(f"Unsupported camera pad_mode={pad_mode!r} for {parquet_path}.")
        pad = num_frames - intr.shape[0]
        intr = np.concatenate([intr, np.repeat(intr[-1:], pad, axis=0)], axis=0)
        extr = np.concatenate([extr, np.repeat(extr[-1:], pad, axis=0)], axis=0)
    return intr[:num_frames], extr[:num_frames]


def pose_to_c2w(pose_xyz_qwxyz: np.ndarray, quat_to_mat) -> torch.Tensor:
    pose = torch.as_tensor(pose_xyz_qwxyz, dtype=torch.float32)
    quat_xyzw = torch.stack([pose[4], pose[5], pose[6], pose[3]])
    rot = quat_to_mat(quat_xyzw.unsqueeze(0)).squeeze(0)
    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, :3] = rot
    c2w[:3, 3] = pose[:3]
    return c2w


def intrinsics_to_lagernvs_fxfycxcy(
    intrinsics_raw: np.ndarray,
    orig_hw: tuple[int, int],
    tgt_hw: tuple[int, int],
    camera_utils,
) -> list[float]:
    # DROID stores [fx, cx, fy, cy]. LagerNVS expects [fx, fy, cx, cy]
    # after applying the same center-crop/resize intrinsics correction used
    # by its DROID dataset.
    fx_raw, cx_raw, fy_raw, cy_raw = np.asarray(intrinsics_raw, dtype=np.float32).tolist()
    crop_hw_in_orig = camera_utils.get_full_res_crop_dims_constant_ar(orig_hw, tgt_hw)
    fx, fy, cx, cy = camera_utils.adjust_intrinsics_for_crop_and_resize(
        (fx_raw, fy_raw, cx_raw, cy_raw),
        orig_hw,
        crop_hw_in_orig,
        tgt_hw,
    )
    return [float(fx), float(fy), float(cx), float(cy)]


def build_lagernvs_cam_cond_token(
    c2w_poses: torch.Tensor,
    intrinsics_fxfycxcy_px: torch.Tensor,
    num_cond_views: int,
    tgt_hw: tuple[int, int],
    camera_scale,
    pose_enc,
) -> torch.Tensor:
    # Mirrors LagerNVS data/normalization.py::build_cam_cond for the token
    # branch, without materializing Plucker ray maps that sidecar caching does
    # not need.
    cam_cond_token = pose_enc.extri_intri_to_pose_encoding(
        c2w_poses.unsqueeze(0),
        intrinsics_fxfycxcy_px.unsqueeze(0),
        image_size_hw=tgt_hw,
    ).squeeze(0)
    world_points_scale = torch.zeros(
        (),
        device=cam_cond_token.device,
        dtype=cam_cond_token.dtype,
    )
    camera_scale = torch.as_tensor(
        camera_scale,
        device=cam_cond_token.device,
        dtype=cam_cond_token.dtype,
    )
    scene_scale_tokens = torch.stack([camera_scale, world_points_scale]).unsqueeze(0)
    scene_scale_tokens = scene_scale_tokens.expand(cam_cond_token.shape[0], -1)
    return torch.cat([cam_cond_token, scene_scale_tokens], dim=-1)


def build_lagernvs_camera_tokens(
    dataset_base_path: str,
    row: dict,
    source_views: tuple[int, ...],
    target_view: int,
    num_frames: int,
    tgt_hw: tuple[int, int],
    lagernvs_root: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera_utils, normalization, pose_enc, quat_to_mat = load_lagernvs_camera_modules(
        lagernvs_root
    )
    intrinsics = []
    c2w_poses = []

    for view_index in source_views:
        intr_raw, extr_raw = read_camera_raw_sequence(
            dataset_base_path,
            row,
            int(view_index),
            num_frames=1,
        )
        orig_hw = video_hw_for_view(dataset_base_path, row, int(view_index))
        intrinsics.append(
            intrinsics_to_lagernvs_fxfycxcy(
                intr_raw[0],
                orig_hw,
                tgt_hw,
                camera_utils,
            )
        )
        c2w_poses.append(pose_to_c2w(extr_raw[0], quat_to_mat))

    target_intr_raw, target_extr_raw = read_camera_raw_sequence(
        dataset_base_path,
        row,
        int(target_view),
        num_frames=int(num_frames),
    )
    target_orig_hw = video_hw_for_view(dataset_base_path, row, int(target_view))
    for frame_id in range(int(num_frames)):
        intrinsics.append(
            intrinsics_to_lagernvs_fxfycxcy(
                target_intr_raw[frame_id],
                target_orig_hw,
                tgt_hw,
                camera_utils,
            )
        )
        c2w_poses.append(pose_to_c2w(target_extr_raw[frame_id], quat_to_mat))

    intrinsics_tensor = torch.tensor(np.asarray(intrinsics, dtype=np.float32))
    c2w_tensor = torch.stack(c2w_poses, dim=0)
    c2w_norm, camera_scale, _ = normalization.normalize_extrinsics(
        c2w_tensor,
        num_cond_views=len(source_views),
    )
    cam_tokens = build_lagernvs_cam_cond_token(
        c2w_norm,
        intrinsics_tensor,
        num_cond_views=len(source_views),
        tgt_hw=tgt_hw,
        camera_scale=camera_scale,
        pose_enc=pose_enc,
    )
    source_cam_tokens = cam_tokens[: len(source_views)].unsqueeze(0)
    target_cam_tokens = cam_tokens[len(source_views) :].unsqueeze(0)
    return source_cam_tokens, target_cam_tokens


def downsample_camera_sequence(camera_tokens: torch.Tensor, latent_length: int) -> torch.Tensor:
    if camera_tokens.shape[1] == latent_length:
        return camera_tokens
    target_length = int(latent_length) * 4
    tokens = torch.cat(
        [torch.repeat_interleave(camera_tokens[:, 0:1], repeats=4, dim=1), camera_tokens[:, 1:]],
        dim=1,
    )
    if tokens.shape[1] < target_length:
        tokens = torch.cat(
            [tokens, tokens[:, -1:].repeat(1, target_length - tokens.shape[1], 1)],
            dim=1,
        )
    else:
        tokens = tokens[:, :target_length]
    return tokens.reshape(tokens.shape[0], latent_length, 4, tokens.shape[-1]).mean(dim=2)


def dtype_from_arg(value: str):
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    return torch.float32


def build_split(
    split_name: str,
    rows: list[dict],
    args,
    extractor: SceneTokenExtractor,
    source_views: tuple[int, ...],
) -> None:
    main_split = Path(args.main_cache_path) / split_name
    out_split = Path(args.output_root) / split_name
    out_split.mkdir(parents=True, exist_ok=True)
    dtype = dtype_from_arg(args.dtype)
    device = torch.device(args.device)
    indices = list(range(len(rows)))
    if int(args.num_shards) > 1:
        indices = indices[int(args.shard_index) :: int(args.num_shards)]

    for global_idx in tqdm(indices, desc=f"geometry:{split_name}"):
        cache_file = main_split / f"{global_idx:07d}.pth"
        if not cache_file.is_file():
            continue
        out_file = out_split / cache_file.name
        if args.skip_existing and out_file.is_file():
            continue
        cached = torch.load(cache_file, map_location="cpu", weights_only=False)
        source_first_frames = cached["source_first_frames"].unsqueeze(0).to(
            device=device, dtype=dtype
        )
        row = rows[global_idx]
        if int(cached.get("episode_index", row.get("episode_index", -1))) != int(row.get("episode_index", -1)):
            raise ValueError(
                f"Cache/metadata episode mismatch at {cache_file}: "
                f"cache={cached.get('episode_index')}, metadata={row.get('episode_index')}"
            )
        tgt_hw = (int(cached["height"]), int(cached["width"]))
        source_cam_tokens, target_cam_tokens = build_lagernvs_camera_tokens(
            args.dataset_base_path,
            row,
            source_views,
            int(args.cross_view_target_view),
            int(args.num_frames),
            tgt_hw,
            args.lagernvs_root,
        )
        source_cam_tokens = source_cam_tokens.to(device=device, dtype=dtype)
        target_cam_tokens = target_cam_tokens.to(device=device, dtype=dtype)
        latent_length = int(cached["latent_views_gt"].shape[2])
        target_cam_tokens_latent = downsample_camera_sequence(
            target_cam_tokens, latent_length
        )
        scene_tokens = extractor(source_first_frames, source_cam_tokens).detach().cpu()
        sidecar = {
            "geometry_cache_version": 3,
            "source_cam_tokens": source_cam_tokens.detach().cpu(),
            "target_cam_tokens": target_cam_tokens.detach().cpu(),
            "target_cam_tokens_latent": target_cam_tokens_latent.detach().cpu(),
            "scene_tokens_camera_aware": scene_tokens,
        }
        temp_file = out_file.with_suffix(".pth.tmp")
        torch.save(sidecar, temp_file)
        os.replace(temp_file, out_file)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if int(args.num_shards) < 1:
        raise ValueError("--num_shards must be >= 1")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard_index must be in [0, num_shards)")
    source_views = parse_views(args.cross_view_source_views)
    device = torch.device(args.device)
    dtype = dtype_from_arg(args.dtype)
    extractor = SceneTokenExtractor(
        checkpoint_path=args.scene_token_checkpoint,
        freeze=True,
        input_value_range="minus1_1",
    ).to(device=device, dtype=dtype).eval()
    train_rows = load_jsonl(args.train_metadata_path)
    build_split("train", train_rows, args, extractor, source_views)
    if args.val_metadata_path:
        val_rows = load_jsonl(args.val_metadata_path)
        build_split("val", val_rows, args, extractor, source_views)
    config = {
        "geometry_cache_version": 3,
        "main_cache_path": str(Path(args.main_cache_path).resolve()),
        "lagernvs_root": str(Path(args.lagernvs_root).resolve()),
        "source_views": list(source_views),
        "target_view": int(args.cross_view_target_view),
        "num_frames": int(args.num_frames),
        "num_history_frames": int(args.num_history_frames),
        "camera_token_layout": "lagernvs_absT_quatxyzw_fovh_fovw_camera_scale_world_points_scale",
        "camera_token_normalization": "LagerNVS normalize_extrinsics with source first-frame cameras as conditioning views",
    }
    with open(Path(args.output_root) / "geometry_cache_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
