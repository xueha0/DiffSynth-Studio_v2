#!/usr/bin/env python3
"""Extract three-view ground-truth validation clips from cross-view metadata."""

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save GT video clips for each cross-view validation metadata row."
    )
    parser.add_argument(
        "--meta-jsonl",
        default=(
            "/data2/xuehao/datasets/droid_success_high_quality_crossview_meta/"
            "meta/episodes_cross_view_val_81_small200.jsonl"
        ),
        help="Cross-view validation metadata JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/home/xuehao/xh/projects/DiffSynth-Studio_v2/Ckpt/"
            "clip_traj_iter_000000_gt_videos"
        ),
        help="Output directory. One subfolder per validation clip will be created.",
    )
    parser.add_argument(
        "--views",
        default="0,1,2",
        help="Comma-separated view indices to export. Default exports all three views.",
    )
    parser.add_argument(
        "--frame-scope",
        choices=["all", "valid"],
        default="all",
        help="`all` saves pad_to_frames frames; `valid` saves only metadata valid_frames.",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="OpenCV fourcc codec for mp4 output. Default: mp4v.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output FPS. Default uses source video FPS, falling back to 15.",
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=15.0,
        help="FPS used when source FPS is unavailable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing complete output videos.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel worker processes.",
    )
    return parser.parse_args()


def parse_views(views: str) -> list[int]:
    parsed = [int(item.strip()) for item in views.split(",") if item.strip()]
    if not parsed:
        raise ValueError("--views must contain at least one view index")
    return parsed


def load_rows(meta_jsonl: Path) -> list[dict[str, Any]]:
    rows = []
    with meta_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clip_dir_name(row: dict[str, Any]) -> str:
    return (
        f"episode_{int(row['episode_index']):06d}"
        f"_clipstart_{int(row['start_frame']):06d}"
    )


def view_label(view_idx: int, video_path: str) -> str:
    match = re.search(r"observation\.images\.([^/]+)", video_path)
    if match is None:
        return f"view_{view_idx}"
    label = re.sub(r"[^A-Za-z0-9_]+", "_", match.group(1)).strip("_")
    return label or f"view_{view_idx}"


def expected_frame_count(row: dict[str, Any], view_meta: dict[str, Any], frame_scope: str) -> int:
    if frame_scope == "valid":
        return int(row.get("valid_frames", view_meta["end_frame"] - view_meta["start_frame"] + 1))
    return int(view_meta.get("pad_to_frames", row.get("length", row.get("valid_frames", 0))))


def existing_complete(video_path: Path, frame_count: int) -> bool:
    if not video_path.is_file():
        return False
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    try:
        existing_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return existing_frames >= frame_count


def source_fps(cap: cv2.VideoCapture, override_fps: float | None, fallback_fps: float) -> float:
    if override_fps is not None:
        return float(override_fps)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0:
        return float(fallback_fps)
    return fps


def make_writer(path: Path, codec: str, fps: float, frame_shape) -> cv2.VideoWriter:
    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    return writer


def extract_view_video(
    row: dict[str, Any],
    view_idx: int,
    out_dir: Path,
    frame_scope: str,
    codec: str,
    fps_override: float | None,
    fallback_fps: float,
    overwrite: bool,
) -> tuple[str, int, Path]:
    view_meta = row["video"][view_idx]
    label = view_label(view_idx, str(view_meta["data"]))
    out_path = out_dir / f"view_{view_idx}_{label}.mp4"
    frame_count = expected_frame_count(row, view_meta, frame_scope)
    if frame_count <= 0:
        raise ValueError(f"Invalid frame_count={frame_count}")
    if not overwrite and existing_complete(out_path, frame_count):
        return "skipped_existing", frame_count, out_path

    video_path = Path(view_meta["data"])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open GT video: {video_path}")

    valid_frames = int(row.get("valid_frames", view_meta["end_frame"] - view_meta["start_frame"] + 1))
    frames_to_decode = min(valid_frames, frame_count)
    fps = source_fps(cap, fps_override, fallback_fps)
    written = 0
    writer = None
    last_frame = None

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(view_meta["start_frame"]))
        for _ in range(frames_to_decode):
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                writer = make_writer(out_path, codec, fps, frame.shape)
            writer.write(frame)
            last_frame = frame
            written += 1

        if written == 0 or last_frame is None:
            raise RuntimeError(
                f"No frames decoded from {video_path} at start_frame={view_meta['start_frame']}"
            )

        if written < frame_count:
            if str(view_meta.get("pad_mode", "repeat_last")) != "repeat_last":
                raise ValueError(f"Unsupported pad_mode={view_meta.get('pad_mode')!r}")
            if writer is None:
                writer = make_writer(out_path, codec, fps, last_frame.shape)
            for _ in range(written, frame_count):
                writer.write(last_frame)
            written = frame_count
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return "written", written, out_path


def extract_clip_task(task: dict[str, Any]) -> dict[str, Any]:
    row = task["row"]
    output_root = Path(task["output_root"])
    views = task["views"]
    clip_dir = output_root / clip_dir_name(row)
    clip_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "episode_index": int(row.get("episode_index", -1)),
        "clipstart": int(row.get("start_frame", -1)),
        "videos_written": 0,
        "videos_skipped_existing": 0,
        "frames": 0,
        "failed": [],
    }

    for view_idx in views:
        try:
            status, frames, _ = extract_view_video(
                row=row,
                view_idx=int(view_idx),
                out_dir=clip_dir,
                frame_scope=task["frame_scope"],
                codec=task["codec"],
                fps_override=task["fps_override"],
                fallback_fps=task["fallback_fps"],
                overwrite=task["overwrite"],
            )
            result["frames"] += int(frames)
            if status == "skipped_existing":
                result["videos_skipped_existing"] += 1
            else:
                result["videos_written"] += 1
        except Exception as exc:
            result["failed"].append(
                {
                    "episode_index": int(row.get("episode_index", -1)),
                    "clipstart": int(row.get("start_frame", -1)),
                    "view_idx": int(view_idx),
                    "reason": repr(exc),
                }
            )

    return result


def main() -> None:
    args = parse_args()
    meta_jsonl = Path(args.meta_jsonl)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = load_rows(meta_jsonl)
    views = parse_views(args.views)

    summary: dict[str, Any] = {
        "meta_jsonl": str(meta_jsonl),
        "output_dir": str(output_root),
        "views": views,
        "frame_scope": args.frame_scope,
        "codec": args.codec,
        "fps_override": args.fps,
        "clips": len(rows),
        "videos_written": 0,
        "videos_skipped_existing": 0,
        "frames": 0,
        "failed": [],
    }

    tasks = [
        {
            "row": row,
            "output_root": str(output_root),
            "views": views,
            "frame_scope": args.frame_scope,
            "codec": args.codec,
            "fps_override": args.fps,
            "fallback_fps": args.fallback_fps,
            "overwrite": args.overwrite,
        }
        for row in rows
    ]

    num_workers = max(1, int(args.num_workers))
    if num_workers == 1:
        iterator = (extract_clip_task(task) for task in tasks)
        for result in tqdm(iterator, total=len(tasks), desc="Extracting GT videos"):
            summary["videos_written"] += int(result["videos_written"])
            summary["videos_skipped_existing"] += int(result["videos_skipped_existing"])
            summary["frames"] += int(result["frames"])
            summary["failed"].extend(result["failed"])
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(extract_clip_task, task) for task in tasks]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Extracting GT videos ({num_workers} workers)",
            ):
                result = future.result()
                summary["videos_written"] += int(result["videos_written"])
                summary["videos_skipped_existing"] += int(result["videos_skipped_existing"])
                summary["frames"] += int(result["frames"])
                summary["failed"].extend(result["failed"])

    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Saved GT videos under: {output_root}")
    print(f"Saved summary to: {summary_path}")
    print(
        f"clips={summary['clips']}, videos_written={summary['videos_written']}, "
        f"videos_skipped_existing={summary['videos_skipped_existing']}, "
        f"frames={summary['frames']}, failed={len(summary['failed'])}"
    )
    if summary["failed"]:
        raise RuntimeError(f"Failed to extract {len(summary['failed'])} videos; see {summary_path}")


if __name__ == "__main__":
    main()
