"""Utilities for cross-view keyframe anchor indexing.

The keyframe manifests contain one row per selected keyframe. Each training
clip has exactly ``num_keyframes`` rows, grouped by the main manifest row id
(``sample_id``) plus clip frame bounds.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


KEYFRAME_ANCHOR_LOOKUP_MODE = "sample_id_clip_offset"
_DIR_PATTERN = re.compile(r"^episode_(\d+)_clipstart_(\d+)_.*_frame_(\d+)$")
_DIRECT_PATH_TEMPLATE = (
    "episode_{episode:06d}_clipstart_{keyframe_frame:06d}_"
    "left_external_frame_{source_frame:06d}"
)


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        value = value.flatten()[0].item()
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    return int(value)


def _clip_key(
    episode_index: int,
    sample_id: int | None,
    clip_start_frame: int,
    clip_end_frame: int,
) -> tuple[int, int | None, int, int]:
    return (
        int(episode_index),
        None if sample_id is None else int(sample_id),
        int(clip_start_frame),
        int(clip_end_frame),
    )


def parse_keyframe_image_root(image_root: str | os.PathLike[str]) -> dict[tuple[int, int, int], str]:
    """Map (episode, keyframe_frame, left_source_frame) to ``000000_pred.png``."""
    root = Path(image_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Keyframe image root not found: {root}")

    index: dict[tuple[int, int, int], str] = {}
    malformed: list[str] = []
    missing_pred: list[str] = []
    duplicates: list[tuple[int, int, int]] = []

    for name in sorted(os.listdir(root)):
        dir_path = root / name
        if not dir_path.is_dir():
            continue
        match = _DIR_PATTERN.match(name)
        if match is None:
            malformed.append(name)
            continue
        key = tuple(int(part) for part in match.groups())
        pred_path = dir_path / "000000_pred.png"
        if not pred_path.is_file():
            missing_pred.append(name)
            continue
        if key in index:
            duplicates.append(key)
            continue
        index[key] = str(pred_path)

    if malformed:
        raise ValueError(
            "Malformed keyframe image directories, first examples: "
            f"{malformed[:5]}"
        )
    if missing_pred:
        raise FileNotFoundError(
            "Keyframe image directories missing 000000_pred.png, first examples: "
            f"{missing_pred[:5]}"
        )
    if duplicates:
        raise ValueError(
            "Duplicate keyframe image directory keys, first examples: "
            f"{duplicates[:5]}"
        )
    return index


def resolve_keyframe_image_path(
    image_root: str | os.PathLike[str],
    episode_index: int,
    keyframe_frame: int,
    source_frame: int,
    *,
    image_index: dict[tuple[int, int, int], str] | None = None,
) -> str | None:
    key = (int(episode_index), int(keyframe_frame), int(source_frame))
    if image_index is not None:
        return image_index.get(key)
    dirname = _DIRECT_PATH_TEMPLATE.format(
        episode=int(episode_index),
        keyframe_frame=int(keyframe_frame),
        source_frame=int(source_frame),
    )
    return str(Path(image_root) / dirname / "000000_pred.png")


def load_keyframe_anchor_index(
    manifest_path: str | os.PathLike[str],
    image_root: str | os.PathLike[str],
    *,
    num_keyframes: int = 3,
    num_frames: int = 81,
    require_paths: bool = True,
) -> dict[str, Any]:
    """Load keyframe manifest rows and resolve their synthesized image paths."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Keyframe manifest not found: {manifest_path}")
    image_root = Path(image_root)
    if not image_root.is_dir():
        raise FileNotFoundError(f"Keyframe image root not found: {image_root}")

    grouped: dict[tuple[int, int | None, int, int], list[dict[str, Any]]] = defaultdict(list)
    grouped_without_sample: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    missing_paths: list[str] = []
    invalid_offsets: list[tuple[int, int]] = []

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            episode_index = int(row["episode_index"])
            sample_id = _as_int(row.get("sample_id"))
            clip_start = int(row["clip_start_frame"])
            clip_end = int(row["clip_end_frame"])
            offset = int(row.get("selected_offset", row.get("effective_offset")))
            if offset < 0 or offset >= int(num_frames):
                invalid_offsets.append((episode_index, offset))
            keyframe_frame = int(row.get("start_frame", row["effective_timeline_frame"]))
            source_frame = int(row["video"][0]["start_frame"])
            path = resolve_keyframe_image_path(
                image_root,
                episode_index,
                keyframe_frame,
                source_frame,
            )
            if not os.path.isfile(path):
                missing_paths.append(path)
                if require_paths:
                    continue
                path = None
            item = {
                "offset": offset,
                "path": path,
                "selection_rank": int(row.get("selection_rank", len(grouped))),
                "frame_index": keyframe_frame,
                "source_frame": source_frame,
            }
            key = _clip_key(episode_index, sample_id, clip_start, clip_end)
            grouped[key].append(item)
            grouped_without_sample[(episode_index, clip_start, clip_end)].append(item)

    if invalid_offsets:
        raise ValueError(
            "Keyframe manifest contains offsets outside the clip, first examples: "
            f"{invalid_offsets[:8]}"
        )
    if missing_paths and require_paths:
        raise FileNotFoundError(
            "Keyframe manifest rows without matching synthesized images, first examples: "
            f"{missing_paths[:8]}"
        )

    bad_counts = [
        (key, len(items))
        for key, items in grouped.items()
        if len(items) != int(num_keyframes)
    ]
    if bad_counts:
        raise ValueError(
            f"Expected {int(num_keyframes)} keyframes per clip, first bad groups: "
            f"{bad_counts[:8]}"
        )

    for mapping in (grouped, grouped_without_sample):
        for key, items in list(mapping.items()):
            mapping[key] = sorted(
                items,
                key=lambda item: (int(item["selection_rank"]), int(item["offset"])),
            )

    return {
        "lookup_mode": KEYFRAME_ANCHOR_LOOKUP_MODE,
        "manifest_path": str(manifest_path),
        "image_root": str(image_root),
        "num_keyframes": int(num_keyframes),
        "by_key": dict(grouped),
        "by_clip": dict(grouped_without_sample),
    }


def resolve_keyframe_anchors(
    keyframe_index: dict[str, Any] | None,
    meta: dict | None,
    *,
    sample_id: int | None = None,
) -> list[dict[str, Any]]:
    if not keyframe_index or meta is None:
        return []
    episode_index = _as_int(meta.get("episode_index"))
    clip_start = _as_int(meta.get("clip_start_frame"), _as_int(meta.get("start_frame")))
    clip_end = _as_int(meta.get("clip_end_frame"), _as_int(meta.get("end_frame")))
    if sample_id is None:
        sample_id = _as_int(meta.get("sample_id"), _as_int(meta.get("__sample_id__")))
    if episode_index is None or clip_start is None or clip_end is None:
        return []

    by_key = keyframe_index.get("by_key", {})
    exact = by_key.get(_clip_key(episode_index, sample_id, clip_start, clip_end))
    if exact is not None:
        return list(exact)
    by_clip = keyframe_index.get("by_clip", {})
    fallback = by_clip.get((int(episode_index), int(clip_start), int(clip_end)))
    return [] if fallback is None else list(fallback)
