#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from diffsynth.utils.data import save_video


VIEW_LABELS = ("src0", "src1", "tgt")
VIEW_COLORS = ((85, 205, 252), (255, 210, 63), (255, 99, 132))
HEADER_HEIGHT = 60
SHEET_COLUMNS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render side-by-side visualization examples from cross-view manifests."
    )
    parser.add_argument(
        "--dataset-root",
        default="/data2/xuehao/datasets/droid_success_high_quality_crossview_meta",
        help="Cross-view metadata root.",
    )
    parser.add_argument(
        "--train-manifest",
        default=None,
        help="Optional train manifest override.",
    )
    parser.add_argument(
        "--val-manifest",
        default=None,
        help="Optional val manifest override.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for videos and sheets. Defaults to <dataset-root>/visual_checks.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=180,
        help="Resize height for visualization.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Resize width for visualization.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Output video FPS.",
    )
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=3,
        help="Maximum rendered examples per split.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def shorten(text: str, limit: int = 110) -> str:
    text = " ".join(str(text).strip().split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def pick_examples(rows: list[dict], max_per_split: int) -> list[tuple[str, int, dict]]:
    if not rows:
        return []

    selected: list[tuple[str, int, dict]] = []
    used: set[int] = set()

    def add(index: int, tag: str) -> None:
        if index in used or len(selected) >= max_per_split:
            return
        selected.append((tag, index, rows[index]))
        used.add(index)

    full_indices = [i for i, row in enumerate(rows) if int(row.get("valid_frames", row.get("length", 0))) >= int(row.get("length", 0))]
    tail_indices = [i for i, row in enumerate(rows) if int(row.get("valid_frames", row.get("length", 0))) < int(row.get("length", 0))]

    if full_indices:
        add(full_indices[0], "full_a")
    if tail_indices:
        add(tail_indices[0], "tail")
    if full_indices:
        add(full_indices[len(full_indices) // 2], "full_b")
    if len(selected) < max_per_split:
        add(len(rows) - 1, "last")
    return selected[:max_per_split]


def resolve_resize_method():
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return resampling.BILINEAR
    return Image.BILINEAR


def load_clip_frames(video_item: dict, width: int, height: int) -> list[Image.Image]:
    resize_method = resolve_resize_method()
    start_frame = int(video_item["start_frame"])
    end_frame = int(video_item["end_frame"])
    pad_to_frames = int(video_item.get("pad_to_frames", end_frame - start_frame + 1))
    pad_mode = video_item.get("pad_mode", "repeat_last")

    frames: list[Image.Image] = []
    reader = imageio.get_reader(str(video_item["data"]))
    try:
        for frame_id in range(start_frame, end_frame + 1):
            frame = Image.fromarray(reader.get_data(frame_id)).convert("RGB")
            frames.append(frame.resize((width, height), resize_method))
    finally:
        reader.close()

    if len(frames) == 0:
        raise ValueError(f"Empty frame sequence for {video_item['data']}")
    if len(frames) < pad_to_frames:
        if pad_mode != "repeat_last":
            raise ValueError(f"Unsupported pad_mode={pad_mode!r}")
        frames.extend([frames[-1].copy() for _ in range(pad_to_frames - len(frames))])
    return frames[:pad_to_frames]


def compose_frame(
    split_name: str,
    tag: str,
    row: dict,
    frame_index: int,
    view_frames: list[Image.Image],
    width: int,
    height: int,
) -> Image.Image:
    canvas = Image.new("RGBA", (width * len(view_frames), HEADER_HEIGHT + height), (18, 18, 18, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    valid_frames = int(row.get("valid_frames", row.get("length", len(view_frames))))
    info = (
        f"{split_name} | {tag} | ep={row['episode_index']} | "
        f"clip={row['start_frame']}-{row['end_frame']} | valid={valid_frames}/{row['length']} | "
        f"frame={frame_index + 1}/{row['length']}"
    )
    draw.text((8, 8), info, fill=(245, 245, 245), font=font)
    draw.text((8, 28), shorten(row.get("prompt", "")), fill=(205, 205, 205), font=font)

    for view_id, frame in enumerate(view_frames):
        x0 = view_id * width
        canvas.paste(frame, (x0, HEADER_HEIGHT))
        draw.rectangle(
            [x0, HEADER_HEIGHT, x0 + width - 1, HEADER_HEIGHT + height - 1],
            outline=VIEW_COLORS[view_id],
            width=2,
        )
        draw.rectangle(
            [x0 + 6, HEADER_HEIGHT + 6, x0 + 58, HEADER_HEIGHT + 24],
            fill=(0, 0, 0, 170),
        )
        draw.text((x0 + 10, HEADER_HEIGHT + 9), VIEW_LABELS[view_id], fill=VIEW_COLORS[view_id], font=font)

    if frame_index >= valid_frames:
        overlay = Image.new("RGBA", (canvas.width, height), (180, 24, 24, 76))
        canvas.alpha_composite(overlay, (0, HEADER_HEIGHT))
        draw.rectangle([canvas.width - 40, 8, canvas.width - 8, 24], fill=(130, 20, 20, 255))
        draw.text((canvas.width - 35, 11), "PAD", fill=(255, 235, 235), font=font)
    return canvas.convert("RGB")


def build_sheet(frames: list[Image.Image], valid_frames: int) -> Image.Image:
    if not frames:
        raise ValueError("Cannot build sheet from empty frame list.")
    total = len(frames)
    candidate_indices = [0, total // 4, total // 2, (3 * total) // 4, max(0, valid_frames - 1), total - 1]
    indices = []
    for index in candidate_indices:
        index = max(0, min(total - 1, int(index)))
        if index not in indices:
            indices.append(index)
    chosen = [frames[index] for index in indices]
    thumb_w, thumb_h = chosen[0].size
    rows = (len(chosen) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
    sheet = Image.new("RGB", (thumb_w * SHEET_COLUMNS, thumb_h * rows), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for item_id, (index, frame) in enumerate(zip(indices, chosen)):
        row_id = item_id // SHEET_COLUMNS
        col_id = item_id % SHEET_COLUMNS
        x0 = col_id * thumb_w
        y0 = row_id * thumb_h
        sheet.paste(frame, (x0, y0))
        draw.rectangle([x0, y0, x0 + thumb_w - 1, y0 + thumb_h - 1], outline=(80, 80, 80), width=1)
        draw.rectangle([x0 + 8, y0 + 8, x0 + 86, y0 + 24], fill=(0, 0, 0))
        label = f"frame {index + 1}"
        if index >= valid_frames:
            label += " PAD"
        draw.text((x0 + 12, y0 + 11), label, fill=(255, 255, 255), font=font)
    return sheet


def render_sample(
    split_name: str,
    tag: str,
    row_index: int,
    row: dict,
    output_dir: Path,
    width: int,
    height: int,
    fps: int,
) -> dict:
    view_clips = [load_clip_frames(item, width=width, height=height) for item in row["video"]]
    clip_length = len(view_clips[0])
    if any(len(view_frames) != clip_length for view_frames in view_clips):
        raise ValueError("View clip lengths are inconsistent.")

    composed_frames = [
        compose_frame(
            split_name=split_name,
            tag=tag,
            row=row,
            frame_index=frame_index,
            view_frames=[view_frames[frame_index] for view_frames in view_clips],
            width=width,
            height=height,
        )
        for frame_index in range(clip_length)
    ]

    stem = (
        f"{split_name}_{row_index:03d}_{tag}_"
        f"ep{int(row['episode_index']):06d}_"
        f"s{int(row['start_frame']):03d}_e{int(row['end_frame']):03d}_"
        f"v{int(row.get('valid_frames', row['length'])):03d}"
    )
    video_path = output_dir / f"{stem}.mp4"
    sheet_path = output_dir / f"{stem}.png"
    save_video(composed_frames, str(video_path), fps=fps, quality=5)
    build_sheet(composed_frames, valid_frames=int(row.get("valid_frames", row["length"]))).save(sheet_path)
    return {
        "split": split_name,
        "tag": tag,
        "row_index": int(row_index),
        "episode_index": int(row["episode_index"]),
        "valid_frames": int(row.get("valid_frames", row["length"])),
        "start_frame": int(row["start_frame"]),
        "end_frame": int(row["end_frame"]),
        "prompt": row.get("prompt", ""),
        "video_path": str(video_path),
        "sheet_path": str(sheet_path),
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    train_manifest = (
        Path(args.train_manifest).resolve()
        if args.train_manifest is not None
        else dataset_root / "meta" / "episodes_cross_view_train_81_small200.jsonl"
    )
    val_manifest = (
        Path(args.val_manifest).resolve()
        if args.val_manifest is not None
        else dataset_root / "meta" / "episodes_cross_view_val_81_small15.jsonl"
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else dataset_root / "visual_checks"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    selections = []
    for split_name, manifest_path in (("train", train_manifest), ("val", val_manifest)):
        rows = load_jsonl(manifest_path)
        for tag, row_index, row in pick_examples(rows, max_per_split=int(args.max_per_split)):
            selections.append((split_name, tag, row_index, row))

    results = []
    for split_name, tag, row_index, row in selections:
        results.append(
            render_sample(
                split_name=split_name,
                tag=tag,
                row_index=row_index,
                row=row,
                output_dir=output_dir,
                width=int(args.width),
                height=int(args.height),
                fps=int(args.fps),
            )
        )

    summary = {
        "dataset_root": str(dataset_root),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "output_dir": str(output_dir),
        "num_examples": len(results),
        "examples": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
