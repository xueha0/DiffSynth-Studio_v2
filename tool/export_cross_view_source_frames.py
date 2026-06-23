#!/usr/bin/env python3
"""Export source-view frames for a generated cross-view comparison video."""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_META_JSONL = (
    "/data2/xuehao/datasets/droid_success_high_quality_crossview_meta/"
    "meta/episodes_cross_view_val_81_small200.jsonl"
)
DEFAULT_OUTPUT_ROOT = "/data2/xuehao/datasets"
COMPARISON_RE = re.compile(r"^(?P<split>.+)_(?P<idx>\d+)_ep(?P<episode>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Given a comparison video named like val_015_ep9462.mp4, locate the "
            "matching JSONL row and export every frame from its source_views."
        )
    )
    parser.add_argument(
        "comparison_video",
        type=Path,
        help="Generated comparison video, e.g. .../comparisons/val/val_015_ep9462.mp4.",
    )
    parser.add_argument(
        "--meta-jsonl",
        type=Path,
        default=Path(DEFAULT_META_JSONL),
        help=f"Cross-view metadata JSONL. Default: {DEFAULT_META_JSONL}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(DEFAULT_OUTPUT_ROOT),
        help=(
            "Root directory used when --output-dir is not set. A folder named "
            "<comparison_stem>_source_frames will be created inside it."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact output directory. Overrides --output-root.",
    )
    parser.add_argument(
        "--views",
        default="source",
        help=(
            "Views to export. Use 'source' for row['source_views'] or a comma-separated "
            "list such as '0,1'. Default: source."
        ),
    )
    parser.add_argument(
        "--frame-scope",
        choices=["all", "valid"],
        default="all",
        help=(
            "'all' exports pad_to_frames frames using metadata pad_mode; "
            "'valid' exports only the source frame range. Default: all."
        ),
    )
    parser.add_argument(
        "--start-number",
        type=int,
        default=1,
        help="First exported frame number in frame_%%06d.png. Default: 1.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable. Default: ffmpeg.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing frame_*.png files in each target view directory before export.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved metadata and output plan without writing files.",
    )
    return parser.parse_args()


def load_rows(meta_jsonl: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with meta_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {meta_jsonl}: {exc}") from exc
    return rows


def parse_comparison_stem(comparison_video: Path) -> tuple[str, int, int]:
    match = COMPARISON_RE.match(comparison_video.stem)
    if match is None:
        raise ValueError(
            f"Cannot parse comparison filename '{comparison_video.name}'. "
            "Expected a stem like val_015_ep9462."
        )
    return match.group("split"), int(match.group("idx")), int(match.group("episode"))


def parse_views(views_arg: str, row: dict[str, Any]) -> list[int]:
    if views_arg.strip().lower() == "source":
        views = row.get("source_views")
        if not isinstance(views, list) or not views:
            raise ValueError("Metadata row does not contain a non-empty source_views list.")
        return [int(view) for view in views]

    views = [int(item.strip()) for item in views_arg.split(",") if item.strip()]
    if not views:
        raise ValueError("--views must be 'source' or contain at least one view index.")
    return views


def view_label(view_idx: int, video_path: str) -> str:
    match = re.search(r"observation\.images\.([^/]+)", video_path)
    if match is None:
        return f"view_{view_idx}"
    label = re.sub(r"[^A-Za-z0-9_]+", "_", match.group(1)).strip("_")
    return label or f"view_{view_idx}"


def expected_count(row: dict[str, Any], view_meta: dict[str, Any], frame_scope: str) -> int:
    valid_count = int(view_meta["end_frame"]) - int(view_meta["start_frame"]) + 1
    if valid_count <= 0:
        raise ValueError(f"Invalid frame range in view metadata: {view_meta}")
    if frame_scope == "valid":
        return valid_count
    return int(view_meta.get("pad_to_frames", row.get("length", valid_count)))


def frame_name(index: int) -> str:
    return f"frame_{index:06d}.png"


def list_frame_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("frame_*.png"))


def prepare_view_dir(view_dir: Path, overwrite: bool) -> None:
    view_dir.mkdir(parents=True, exist_ok=True)
    existing = list_frame_files(view_dir)
    if not existing:
        return
    if not overwrite:
        raise FileExistsError(
            f"{view_dir} already contains {len(existing)} frame files. "
            "Use --overwrite to replace them."
        )
    for path in existing:
        path.unlink()


def run_ffmpeg_extract(
    ffmpeg_bin: str,
    video_path: Path,
    start_frame: int,
    end_frame: int,
    out_dir: Path,
    start_number: int,
) -> None:
    filter_expr = f"select=between(n\\,{start_frame}\\,{end_frame})"
    output_pattern = str(out_dir / "frame_%06d.png")
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        filter_expr,
        "-vsync",
        "0",
        "-start_number",
        str(start_number),
        output_pattern,
    ]
    subprocess.run(command, check=True)


def pad_repeat_last(out_dir: Path, valid_count: int, total_count: int, start_number: int) -> None:
    if total_count < valid_count:
        raise ValueError(f"Requested {total_count} frames, but valid range has {valid_count} frames.")
    if total_count == valid_count:
        return

    last_valid_index = start_number + valid_count - 1
    last_valid = out_dir / frame_name(last_valid_index)
    if not last_valid.is_file():
        raise FileNotFoundError(f"Cannot pad because last valid frame is missing: {last_valid}")

    for index in range(last_valid_index + 1, start_number + total_count):
        shutil.copyfile(last_valid, out_dir / frame_name(index))


def export_view(
    *,
    row: dict[str, Any],
    view_idx: int,
    output_dir: Path,
    frame_scope: str,
    start_number: int,
    ffmpeg_bin: str,
    overwrite: bool,
) -> dict[str, Any]:
    videos = row.get("video")
    if not isinstance(videos, list) or view_idx >= len(videos) or view_idx < 0:
        raise IndexError(f"View {view_idx} is not available in metadata row.")

    view_meta = videos[view_idx]
    video_path = Path(view_meta["data"])
    if not video_path.is_file():
        raise FileNotFoundError(f"Source video not found for view {view_idx}: {video_path}")

    label = view_label(view_idx, str(video_path))
    view_dir = output_dir / f"source_view_{view_idx}_{label}"
    prepare_view_dir(view_dir, overwrite)

    start_frame = int(view_meta["start_frame"])
    end_frame = int(view_meta["end_frame"])
    valid_count = end_frame - start_frame + 1
    total_count = expected_count(row, view_meta, frame_scope)

    run_ffmpeg_extract(ffmpeg_bin, video_path, start_frame, end_frame, view_dir, start_number)
    actual_valid = len(list_frame_files(view_dir))
    if actual_valid != valid_count:
        raise RuntimeError(
            f"ffmpeg exported {actual_valid} frames for view {view_idx}, "
            f"but metadata range {start_frame}-{end_frame} expects {valid_count}."
        )

    if total_count > valid_count:
        pad_mode = str(view_meta.get("pad_mode", row.get("pad_mode", "repeat_last")))
        if pad_mode != "repeat_last":
            raise ValueError(
                f"Padding is needed for view {view_idx}, but pad_mode={pad_mode!r} is unsupported."
            )
        pad_repeat_last(view_dir, valid_count, total_count, start_number)

    final_count = len(list_frame_files(view_dir))
    if final_count != total_count:
        raise RuntimeError(
            f"View {view_idx} ended with {final_count} frame files, expected {total_count}."
        )

    return {
        "view_index": view_idx,
        "label": label,
        "source_video": str(video_path),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "valid_frames": valid_count,
        "saved_frames": final_count,
        "output_dir": str(view_dir),
    }


def write_manifest(
    output_dir: Path,
    *,
    comparison_video: Path,
    meta_jsonl: Path,
    split_name: str,
    row_index: int,
    row: dict[str, Any],
    frame_scope: str,
    exported: list[dict[str, Any]],
) -> None:
    manifest = {
        "comparison_video": str(comparison_video),
        "meta_jsonl": str(meta_jsonl),
        "split": split_name,
        "row_index": row_index,
        "jsonl_line": row_index + 1,
        "episode_index": int(row["episode_index"]),
        "clip_start_frame": int(row.get("start_frame", -1)),
        "clip_end_frame": int(row.get("end_frame", -1)),
        "row_valid_frames": int(row.get("valid_frames", -1)),
        "row_length": int(row.get("length", -1)),
        "source_views": [int(view) for view in row.get("source_views", [])],
        "target_view": int(row.get("target_view", -1)),
        "frame_scope": frame_scope,
        "exports": exported,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    if shutil.which(args.ffmpeg_bin) is None:
        raise FileNotFoundError(f"ffmpeg executable not found: {args.ffmpeg_bin}")
    if not args.comparison_video.is_file():
        raise FileNotFoundError(f"Comparison video not found: {args.comparison_video}")
    if not args.meta_jsonl.is_file():
        raise FileNotFoundError(f"Metadata JSONL not found: {args.meta_jsonl}")

    split_name, row_index, episode_index = parse_comparison_stem(args.comparison_video)
    rows = load_rows(args.meta_jsonl)
    if row_index >= len(rows):
        raise IndexError(f"Comparison row index {row_index} exceeds JSONL size {len(rows)}.")

    row = rows[row_index]
    row_episode = int(row["episode_index"])
    if row_episode != episode_index:
        raise ValueError(
            f"Episode mismatch: filename has ep{episode_index}, but JSONL row "
            f"{row_index} has episode_index={row_episode}. Check --meta-jsonl."
        )

    output_dir = args.output_dir or (args.output_root / f"{args.comparison_video.stem}_source_frames")
    views = parse_views(args.views, row)

    plan = {
        "comparison_video": str(args.comparison_video),
        "meta_jsonl": str(args.meta_jsonl),
        "row_index": row_index,
        "jsonl_line": row_index + 1,
        "episode_index": row_episode,
        "views": views,
        "output_dir": str(output_dir),
        "frame_scope": args.frame_scope,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    exported = [
        export_view(
            row=row,
            view_idx=view_idx,
            output_dir=output_dir,
            frame_scope=args.frame_scope,
            start_number=args.start_number,
            ffmpeg_bin=args.ffmpeg_bin,
            overwrite=args.overwrite,
        )
        for view_idx in views
    ]
    write_manifest(
        output_dir,
        comparison_video=args.comparison_video,
        meta_jsonl=args.meta_jsonl,
        split_name=split_name,
        row_index=row_index,
        row=row,
        frame_scope=args.frame_scope,
        exported=exported,
    )
    print(f"Saved source frames to: {output_dir}")


if __name__ == "__main__":
    main()
