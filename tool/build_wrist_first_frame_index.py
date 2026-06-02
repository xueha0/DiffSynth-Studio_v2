#!/usr/bin/env python
"""
Build a frame-indexed wrist anchor JSON.

The output keeps the historical CLI name compatibility, but the key semantics
are now frame-index based:

  - "{episode_index}_{start_frame}" -> clip head wrist image
  - "{episode_index}_{end_frame}"   -> clip tail wrist image

For non-last clips, the tail key points to the next clip's first-frame cache
path. For last clips, the tail key points to the explicit end-frame cache path.
"""
import argparse
import json
import os
import re
from pathlib import Path


DIR_PATTERN = re.compile(r"^episode_(\d+)_clipstart_(\d+)_.*_frame_(\d+)$")
DEFAULT_META_ROOT = Path(
    "/data2/xuehao/datasets/droid_success_high_quality_crossview_meta/meta"
)
DEFAULT_TRAIN_MANIFEST = (
    DEFAULT_META_ROOT / "episodes_cross_view_train_81_small16567.jsonl"
)
DEFAULT_VAL_MANIFEST = DEFAULT_META_ROOT / "episodes_cross_view_val_81_small200.jsonl"
DEFAULT_TRAIN_FIRST_ROOT = Path(
    "/data2/xuehao/datasets/droid_success_wrist_first_frame_train/images_iter_060001"
)
DEFAULT_VAL_FIRST_ROOT = Path(
    "/data2/xuehao/datasets/droid_success_wrist_first_frame_val/images_iter_060001"
)
DEFAULT_TRAIN_END_ROOT = Path(
    "/data2/xuehao/datasets/droid_success_wrist_end_frame_train/images_iter_000000"
)
DEFAULT_VAL_END_ROOT = Path(
    "/data2/xuehao/datasets/droid_success_wrist_end_frame_val/images_iter_000000"
)
DEFAULT_OUTPUT = DEFAULT_META_ROOT / "wrist_frame_index_all.json"


def find_pred_image(dir_path: Path) -> str | None:
    for name in sorted(os.listdir(dir_path)):
        if name.endswith("_pred.png"):
            return str(dir_path / name)
    return None


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_cache_root(root: Path, label: str) -> dict[tuple[int, int], str]:
    if not root.is_dir():
        raise FileNotFoundError(f"{label} cache root not found: {root}")

    index: dict[tuple[int, int], str] = {}
    duplicate_keys: list[tuple[int, int]] = []
    malformed_dirs: list[str] = []
    missing_pred: list[str] = []

    for name in sorted(os.listdir(root)):
        dir_path = root / name
        if not dir_path.is_dir() or not name.startswith("episode_"):
            continue
        match = DIR_PATTERN.match(name)
        if match is None:
            malformed_dirs.append(name)
            continue
        episode_index = int(match.group(1))
        # Directory names still use "clipstart" for both cache types. The
        # second integer is the manifest-relative frame key we need.
        frame_index = int(match.group(2))
        pred_path = find_pred_image(dir_path)
        if pred_path is None:
            missing_pred.append(name)
            continue
        key = (episode_index, frame_index)
        if key in index:
            duplicate_keys.append(key)
            continue
        index[key] = pred_path

    if malformed_dirs:
        raise ValueError(
            f"{label} has malformed episode dirs, first examples: "
            f"{malformed_dirs[:5]}"
        )
    if missing_pred:
        raise FileNotFoundError(
            f"{label} dirs missing *_pred.png, first examples: {missing_pred[:5]}"
        )
    if duplicate_keys:
        raise ValueError(
            f"{label} has duplicate frame keys, first examples: "
            f"{duplicate_keys[:5]}"
        )
    return index


def split_last_clip_keys(rows: list[dict]) -> set[tuple[int, int]]:
    last_by_episode: dict[int, dict] = {}
    for row in rows:
        episode = int(row["episode_index"])
        current = last_by_episode.get(episode)
        if current is None or int(row["start_frame"]) > int(current["start_frame"]):
            last_by_episode[episode] = row
    return {
        (int(row["episode_index"]), int(row["end_frame"]))
        for row in last_by_episode.values()
    }


def rows_by_episode(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode_index"]), []).append(row)
    for episode_rows in grouped.values():
        episode_rows.sort(key=lambda item: int(item["start_frame"]))
    return grouped


def sorted_required_keys(rows: list[dict], frame_field: str) -> list[tuple[int, int]]:
    return sorted(
        {
            (int(row["episode_index"]), int(row[frame_field]))
            for row in rows
        }
    )


def check_required(
    label: str,
    cache_index: dict[tuple[int, int], str],
    required: set[tuple[int, int]],
) -> None:
    missing = sorted(required - set(cache_index))
    extra = sorted(set(cache_index) - required)
    if missing or extra:
        message = [f"{label} cache does not match manifest requirements."]
        if missing:
            message.append(f"missing={len(missing)}, examples={missing[:8]}")
        if extra:
            message.append(f"extra={len(extra)}, examples={extra[:8]}")
        raise ValueError(" ".join(message))


def add_entry(
    output: dict[str, str],
    source: dict[tuple[int, int], str],
    key: tuple[int, int],
    source_label: str,
    collision_policy: str,
    stats: dict[str, int],
    source_key: tuple[int, int] | None = None,
) -> None:
    out_key = f"{key[0]}_{key[1]}"
    path = source[source_key or key]
    existing = output.get(out_key)
    if existing is None:
        output[out_key] = path
        stats[f"added_{source_label}"] = stats.get(f"added_{source_label}", 0) + 1
        return
    if existing == path:
        stats[f"duplicate_same_{source_label}"] = (
            stats.get(f"duplicate_same_{source_label}", 0) + 1
        )
        return
    if collision_policy == "keep-head" and source_label == "head":
        raise ValueError(
            f"Unexpected head-key collision at {out_key}: existing={existing}, "
            f"head={path}"
        )
    if collision_policy == "error":
        raise ValueError(
            f"Frame key collision at {out_key}: existing={existing}, "
            f"{source_label}={path}"
        )
    if collision_policy == "overwrite":
        output[out_key] = path
    stats[f"collision_{source_label}"] = (
        stats.get(f"collision_{source_label}", 0) + 1
    )


def build_split_index(
    split_name: str,
    manifest_path: Path,
    first_root: Path,
    end_root: Path,
    output: dict[str, str],
    collision_policy: str,
) -> dict[str, int]:
    rows = load_manifest(manifest_path)
    first_index = parse_cache_root(first_root, f"{split_name} first-frame")
    end_index = parse_cache_root(end_root, f"{split_name} end-frame")

    head_keys = set(sorted_required_keys(rows, "start_frame"))
    last_tail_keys = split_last_clip_keys(rows)
    all_tail_keys = set(sorted_required_keys(rows, "end_frame"))
    grouped_rows = rows_by_episode(rows)

    # Existing first-frame caches are expected for every clip head. The new
    # end-frame caches are expected only for each episode's final clip.
    check_required(f"{split_name} first-frame", first_index, head_keys)
    check_required(f"{split_name} end-frame", end_index, last_tail_keys)

    stats: dict[str, int] = {
        "rows": len(rows),
        "episodes": len(last_tail_keys),
        "head_keys": len(head_keys),
        "tail_keys": len(all_tail_keys),
    }

    # Add all clip heads first. This intentionally makes keep-head collisions
    # deterministic for one-frame padded tail clips where start_frame==end_frame.
    for key in sorted(head_keys):
        add_entry(output, first_index, key, "head", collision_policy, stats)

    # Add tails. Non-last tail keys use the next clip's first-frame path as
    # their value while keeping the current clip's end_frame as the JSON key.
    # Last tail keys use the new explicit end-frame cache.
    missing_tail: list[tuple[int, int, str]] = []
    for episode, episode_rows in sorted(grouped_rows.items()):
        for row_index, row in enumerate(episode_rows):
            tail_key = (episode, int(row["end_frame"]))
            if row_index + 1 < len(episode_rows):
                next_row = episode_rows[row_index + 1]
                source_key = (episode, int(next_row["start_frame"]))
                if source_key not in first_index:
                    missing_tail.append((tail_key[0], tail_key[1], f"next_head={source_key}"))
                    continue
                add_entry(
                    output,
                    first_index,
                    tail_key,
                    "tail_from_next_head",
                    collision_policy,
                    stats,
                    source_key=source_key,
                )
            else:
                if tail_key not in end_index:
                    missing_tail.append((tail_key[0], tail_key[1], "end_frame"))
                    continue
                add_entry(output, end_index, tail_key, "tail_from_end", collision_policy, stats)
    if missing_tail:
        raise ValueError(
            f"{split_name} has tail keys without first/end cache paths, "
            f"missing={len(missing_tail)}, examples={missing_tail[:8]}"
        )

    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build frame-indexed wrist anchor JSON for cross-view clips."
    )
    parser.add_argument(
        "--train_meta_manifest",
        type=Path,
        default=DEFAULT_TRAIN_MANIFEST,
        help="Train manifest JSONL.",
    )
    parser.add_argument(
        "--val_meta_manifest",
        type=Path,
        default=DEFAULT_VAL_MANIFEST,
        help="Validation manifest JSONL.",
    )
    parser.add_argument(
        "--train_first_frame_root",
        type=Path,
        default=DEFAULT_TRAIN_FIRST_ROOT,
        help="Train first-frame cache root.",
    )
    parser.add_argument(
        "--val_first_frame_root",
        type=Path,
        default=DEFAULT_VAL_FIRST_ROOT,
        help="Validation first-frame cache root.",
    )
    parser.add_argument(
        "--train_end_frame_root",
        type=Path,
        default=DEFAULT_TRAIN_END_ROOT,
        help="Train end-frame cache root.",
    )
    parser.add_argument(
        "--val_end_frame_root",
        type=Path,
        default=DEFAULT_VAL_END_ROOT,
        help="Validation end-frame cache root.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output frame-index JSON.",
    )
    parser.add_argument(
        "--collision_policy",
        choices=("keep-head", "overwrite", "error"),
        default="keep-head",
        help=(
            "Policy for start_frame==end_frame collisions. keep-head preserves "
            "the first-frame cache path, which is the intended default."
        ),
    )
    parser.add_argument(
        "--meta_manifest",
        type=Path,
        default=None,
        help="Deprecated compatibility alias for --train_meta_manifest.",
    )
    parser.add_argument(
        "--first_frame_root",
        type=Path,
        default=None,
        help="Deprecated compatibility alias for --train_first_frame_root.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.meta_manifest is not None:
        args.train_meta_manifest = args.meta_manifest
    if args.first_frame_root is not None:
        args.train_first_frame_root = args.first_frame_root

    output: dict[str, str] = {}
    print("[wrist-index] Building frame-indexed wrist anchor JSON")
    print(f"[wrist-index] output: {args.output_json}")

    train_stats = build_split_index(
        "train",
        args.train_meta_manifest,
        args.train_first_frame_root,
        args.train_end_frame_root,
        output,
        args.collision_policy,
    )
    val_stats = build_split_index(
        "val",
        args.val_meta_manifest,
        args.val_first_frame_root,
        args.val_end_frame_root,
        output,
        args.collision_policy,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output_json.with_name(f".{args.output_json.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    os.replace(tmp_path, args.output_json)

    print(f"[wrist-index] train stats: {train_stats}")
    print(f"[wrist-index] val stats: {val_stats}")
    print(f"[wrist-index] saved {len(output)} entries to {args.output_json}")


if __name__ == "__main__":
    main()
