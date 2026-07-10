#!/usr/bin/env python3
"""Export paper-ready frame assets for cross-view generation comparisons."""

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_META_JSONL = (
    "/data2/xuehao/datasets/droid_success_high_quality_crossview_meta/"
    "meta/episodes_cross_view_val_81_small200.jsonl"
)
DEFAULT_BASELINE_ROOT = "/data2/xuehao/datasets/baseline_val"
DEFAULT_OUTPUT_DIR = "/data2/xuehao/datasets/paper_comparison_frames"
DEFAULT_REL_FRAMES = "0.2,0.5,0.8"

VAL_STEM_RE = re.compile(r"^val_(?P<idx>\d+)_ep(?P<episode>\d+)$")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    directory: str
    filename_kind: str


MODEL_SPECS = [
    ModelSpec("ours", "ours_val", "val_stem"),
    ModelSpec("exoegov", "exoegov_val", "val_stem"),
    ModelSpec("lagernvs", "lagernvs_val", "val_stem"),
    ModelSpec("svdxt", "svdxt_val", "val_stem"),
    ModelSpec("vggt", "vggt_val", "val_stem"),
    ModelSpec("wristworld", "wristworld_val", "zero7"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export time-aligned PNG frames from two source videos, GT wrist video, "
            "and model predictions for selected validation samples."
        )
    )
    parser.add_argument(
        "--indices",
        default=None,
        help=(
            "Comma/space-separated validation indices or stems, e.g. "
            "'15,89' or 'val_015_ep9462'."
        ),
    )
    parser.add_argument(
        "--indices-file",
        type=Path,
        default=None,
        help="Text file containing validation indices or stems, separated by commas or whitespace.",
    )
    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument(
        "--rel-frames",
        default=DEFAULT_REL_FRAMES,
        help=(
            "Comma/space-separated relative clip positions in [0,1]. "
            f"Default: {DEFAULT_REL_FRAMES}."
        ),
    )
    frame_group.add_argument(
        "--frame-indices",
        default=None,
        help="Comma/space-separated 0-based clip frame offsets, e.g. '0,40,80'.",
    )
    parser.add_argument(
        "--meta-jsonl",
        type=Path,
        default=Path(DEFAULT_META_JSONL),
        help=f"Cross-view validation JSONL. Default: {DEFAULT_META_JSONL}",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path(DEFAULT_BASELINE_ROOT),
        help=f"Directory containing *_val result folders. Default: {DEFAULT_BASELINE_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Output root for frame assets. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "Models to export: 'all' or comma/space-separated names from "
            f"{', '.join(spec.name for spec in MODEL_SPECS)}. Default: all."
        ),
    )
    parser.add_argument(
        "--no-gt",
        action="store_true",
        help="Do not export GT wrist frames.",
    )
    parser.add_argument(
        "--image-size",
        default="320x180",
        help="Output image size as WIDTHxHEIGHT. Frames are directly resized to this size. Default: 320x180.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable. Default: ffmpeg.",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default="ffprobe",
        help="ffprobe executable. Default: ffprobe.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing per-sample output directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and print the export plan without writing files.",
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


def split_tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[\s,]+", value.strip()) if token]


def parse_index_token(token: str) -> tuple[int, int | None]:
    stem = Path(token).stem
    match = VAL_STEM_RE.match(stem)
    if match is not None:
        return int(match.group("idx")), int(match.group("episode"))
    try:
        return int(token), None
    except ValueError as exc:
        raise ValueError(f"Cannot parse validation index token: {token!r}") from exc


def parse_indices(
    indices_arg: str | None,
    indices_file: Path | None,
    rows: list[dict[str, Any]],
) -> list[int]:
    tokens: list[str] = []
    if indices_arg:
        tokens.extend(split_tokens(indices_arg))
    if indices_file is not None:
        with indices_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.split("#", 1)[0]
                tokens.extend(split_tokens(line))
    if not tokens:
        raise ValueError("Provide at least one sample via --indices or --indices-file.")

    indices: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        index, expected_episode = parse_index_token(token)
        if index < 0 or index >= len(rows):
            raise IndexError(f"Validation index {index} is outside JSONL range [0, {len(rows) - 1}].")
        row_episode = int(rows[index]["episode_index"])
        if expected_episode is not None and row_episode != expected_episode:
            raise ValueError(
                f"Index token {token!r} points to row {index}, but JSONL row has "
                f"episode_index={row_episode}."
            )
        if index not in seen:
            seen.add(index)
            indices.append(index)
    return indices


def parse_rel_frames(value: str) -> list[float]:
    rels: list[float] = []
    for token in split_tokens(value):
        rel = float(token)
        if rel < 0.0 or rel > 1.0:
            raise ValueError(f"Relative frame position must be in [0,1], got {rel}.")
        rels.append(rel)
    if not rels:
        raise ValueError("--rel-frames must contain at least one value.")
    return rels


def parse_abs_frames(value: str) -> list[int]:
    frames: list[int] = []
    for token in split_tokens(value):
        frame = int(token)
        if frame < 0:
            raise ValueError(f"Frame index must be non-negative, got {frame}.")
        frames.append(frame)
    if not frames:
        raise ValueError("--frame-indices must contain at least one value.")
    return frames


def parse_image_size(value: str) -> tuple[int, int]:
    match = re.match(r"^(?P<w>\d+)x(?P<h>\d+)$", value.strip().lower())
    if match is None:
        raise ValueError(f"Invalid --image-size {value!r}; expected WIDTHxHEIGHT.")
    width = int(match.group("w"))
    height = int(match.group("h"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid --image-size {value!r}; dimensions must be positive.")
    return width, height


def parse_models(value: str) -> list[ModelSpec]:
    by_name = {spec.name: spec for spec in MODEL_SPECS}
    if value.strip().lower() == "all":
        return MODEL_SPECS

    models: list[ModelSpec] = []
    seen: set[str] = set()
    for token in split_tokens(value):
        name = token.strip().lower()
        if name not in by_name:
            raise ValueError(f"Unknown model {token!r}; available: {', '.join(by_name)}.")
        if name not in seen:
            seen.add(name)
            models.append(by_name[name])
    if not models:
        raise ValueError("--models must be 'all' or contain at least one model name.")
    return models


def view_label(view_idx: int, video_path: str) -> str:
    match = re.search(r"observation\.images\.([^/]+)", video_path)
    if match is None:
        return f"view_{view_idx}"
    label = re.sub(r"[^A-Za-z0-9_]+", "_", match.group(1)).strip("_")
    return label or f"view_{view_idx}"


def val_stem(index: int, episode: int) -> str:
    return f"val_{index:03d}_ep{episode}"


def model_video_path(spec: ModelSpec, baseline_root: Path, index: int, stem: str) -> Path:
    if spec.filename_kind == "val_stem":
        return baseline_root / spec.directory / f"{stem}.mp4"
    if spec.filename_kind == "zero7":
        return baseline_root / spec.directory / f"{index:07d}.mp4"
    raise ValueError(f"Unsupported filename kind for {spec.name}: {spec.filename_kind}")


def run_json_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def probe_frame_count(ffprobe_bin: str, video_path: Path, cache: dict[Path, int]) -> int:
    if video_path in cache:
        return cache[video_path]

    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    data = run_json_command(command)
    streams = data.get("streams") or []
    if not streams:
        raise ValueError(f"No video stream found in {video_path}")

    stream = streams[0]
    for key in ("nb_read_frames", "nb_frames"):
        value = stream.get(key)
        if value is None or value == "N/A":
            continue
        count = int(value)
        if count > 0:
            cache[video_path] = count
            return count
    raise ValueError(f"Could not determine frame count for {video_path}")


def ffmpeg_resize_filter(width: int, height: int) -> str:
    return f"scale={width}:{height},setsar=1"


def export_frame(
    *,
    ffmpeg_bin: str,
    video_path: Path,
    frame_index: int,
    output_path: Path,
    image_size: tuple[int, int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = image_size
    filter_expr = f"select=eq(n\\,{frame_index}),{ffmpeg_resize_filter(width, height)}"
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        filter_expr,
        "-vsync",
        "0",
        "-frames:v",
        "1",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg did not create a valid output image: {output_path}")


def timepoint_specs(
    rel_frames: list[float] | None,
    abs_frames: list[int] | None,
    valid_frames: int,
) -> list[dict[str, Any]]:
    if valid_frames <= 0:
        raise ValueError(f"valid_frames must be positive, got {valid_frames}.")

    specs: list[dict[str, Any]] = []
    if abs_frames is not None:
        for pos, requested in enumerate(abs_frames):
            specs.append(
                {
                    "name": f"t{pos:02d}_f{requested:03d}",
                    "kind": "absolute",
                    "value": requested,
                    "requested_clip_frame": requested,
                    "valid_frame": min(requested, valid_frames - 1),
                    "clamped_to_valid": requested > valid_frames - 1,
                }
            )
        return specs

    assert rel_frames is not None
    for pos, rel in enumerate(rel_frames):
        requested = int(round(rel * (valid_frames - 1)))
        specs.append(
            {
                "name": f"t{pos:02d}_rel{int(round(rel * 100)):03d}",
                "kind": "relative",
                "value": rel,
                "requested_clip_frame": requested,
                "valid_frame": requested,
                "clamped_to_valid": False,
            }
        )
    return specs


def prepare_sample_dir(sample_dir: Path, overwrite: bool) -> None:
    if not sample_dir.exists():
        sample_dir.mkdir(parents=True)
        return
    if not overwrite:
        raise FileExistsError(f"{sample_dir} already exists. Use --overwrite to replace it.")
    shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True)


def source_record(
    row: dict[str, Any],
    view_idx: int,
    requested_clip_frame: int,
    output_name: str,
    output_path: Path,
) -> dict[str, Any]:
    videos = row.get("video")
    if not isinstance(videos, list) or view_idx < 0 or view_idx >= len(videos):
        raise IndexError(f"View {view_idx} is not available in metadata row.")
    view_meta = videos[view_idx]
    video_path = Path(view_meta["data"])
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found for view {view_idx}: {video_path}")

    start_frame = int(view_meta["start_frame"])
    end_frame = int(view_meta["end_frame"])
    valid_count = end_frame - start_frame + 1
    if valid_count <= 0:
        raise ValueError(f"Invalid source frame range for view {view_idx}: {view_meta}")
    clamped_frame = min(requested_clip_frame, valid_count - 1)
    absolute_frame = start_frame + clamped_frame

    return {
        "role": "source" if view_idx in row.get("source_views", []) else "target",
        "name": output_name,
        "view_index": view_idx,
        "view_name": view_label(view_idx, str(video_path)),
        "video": str(video_path),
        "requested_clip_frame": requested_clip_frame,
        "clip_frame": clamped_frame,
        "source_frame": absolute_frame,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "clamped": clamped_frame != requested_clip_frame,
        "output": str(output_path),
    }


def export_sample(
    *,
    row: dict[str, Any],
    index: int,
    sample_dir: Path,
    baseline_root: Path,
    selected_models: list[ModelSpec],
    rel_frames: list[float] | None,
    abs_frames: list[int] | None,
    image_size: tuple[int, int],
    ffmpeg_bin: str,
    ffprobe_bin: str,
    include_gt: bool,
    frame_count_cache: dict[Path, int],
    dry_run: bool,
) -> dict[str, Any]:
    episode = int(row["episode_index"])
    stem = val_stem(index, episode)
    valid_frames = int(row["valid_frames"])
    length = int(row.get("length", valid_frames))
    source_views = [int(view) for view in row.get("source_views", [0, 1])]
    target_view = int(row.get("target_view", 2))
    timepoints = timepoint_specs(rel_frames, abs_frames, valid_frames)

    sample_manifest: dict[str, Any] = {
        "sample": stem,
        "val_index": index,
        "jsonl_line": index + 1,
        "episode_index": episode,
        "task": row.get("task"),
        "prompt": row.get("prompt"),
        "row": {
            "length": length,
            "valid_frames": valid_frames,
            "clip_start_frame": int(row.get("start_frame", -1)),
            "clip_end_frame": int(row.get("end_frame", -1)),
            "source_views": source_views,
            "target_view": target_view,
        },
        "settings": {
            "image_size": list(image_size),
            "include_gt": include_gt,
            "models": [spec.name for spec in selected_models],
        },
        "timepoints": [],
    }

    for tp in timepoints:
        tp_dir = sample_dir / tp["name"]
        records: list[dict[str, Any]] = []

        for view_idx in source_views:
            label = view_label(view_idx, str(row["video"][view_idx]["data"]))
            output_name = f"source_{label}"
            output_path = tp_dir / f"{output_name}.png"
            record = source_record(
                row,
                view_idx,
                int(tp["requested_clip_frame"]),
                output_name,
                output_path,
            )
            records.append(record)
            if not dry_run:
                export_frame(
                    ffmpeg_bin=ffmpeg_bin,
                    video_path=Path(record["video"]),
                    frame_index=int(record["source_frame"]),
                    output_path=output_path,
                    image_size=image_size,
                )

        if include_gt:
            output_path = tp_dir / "gt_wrist.png"
            record = source_record(
                row,
                target_view,
                int(tp["requested_clip_frame"]),
                "gt_wrist",
                output_path,
            )
            record["role"] = "gt"
            records.append(record)
            if not dry_run:
                export_frame(
                    ffmpeg_bin=ffmpeg_bin,
                    video_path=Path(record["video"]),
                    frame_index=int(record["source_frame"]),
                    output_path=output_path,
                    image_size=image_size,
                )

        for spec in selected_models:
            video_path = model_video_path(spec, baseline_root, index, stem)
            if not video_path.is_file():
                raise FileNotFoundError(f"Model video not found for {spec.name}: {video_path}")
            frame_count = probe_frame_count(ffprobe_bin, video_path, frame_count_cache)
            requested_frame = int(tp["requested_clip_frame"])
            actual_frame = min(requested_frame, frame_count - 1)
            output_path = tp_dir / f"{spec.name}.png"
            record = {
                "role": "model",
                "name": spec.name,
                "video": str(video_path),
                "requested_frame": requested_frame,
                "decoded_frame": actual_frame,
                "video_frames": frame_count,
                "clamped": actual_frame != requested_frame,
                "output": str(output_path),
            }
            records.append(record)
            if not dry_run:
                export_frame(
                    ffmpeg_bin=ffmpeg_bin,
                    video_path=video_path,
                    frame_index=actual_frame,
                    output_path=output_path,
                    image_size=image_size,
                )

        timepoint_manifest = dict(tp)
        timepoint_manifest["assets"] = records
        sample_manifest["timepoints"].append(timepoint_manifest)

    if not dry_run:
        with (sample_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(sample_manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    return sample_manifest


def main() -> None:
    args = parse_args()
    if shutil.which(args.ffmpeg_bin) is None:
        raise FileNotFoundError(f"ffmpeg executable not found: {args.ffmpeg_bin}")
    if shutil.which(args.ffprobe_bin) is None:
        raise FileNotFoundError(f"ffprobe executable not found: {args.ffprobe_bin}")
    if not args.meta_jsonl.is_file():
        raise FileNotFoundError(f"Metadata JSONL not found: {args.meta_jsonl}")
    if not args.baseline_root.is_dir():
        raise FileNotFoundError(f"Baseline root not found: {args.baseline_root}")

    rows = load_rows(args.meta_jsonl)
    indices = parse_indices(args.indices, args.indices_file, rows)
    image_size = parse_image_size(args.image_size)
    selected_models = parse_models(args.models)
    rel_frames = None if args.frame_indices else parse_rel_frames(args.rel_frames)
    abs_frames = parse_abs_frames(args.frame_indices) if args.frame_indices else None
    include_gt = not args.no_gt

    output_dir = args.output_dir
    frame_count_cache: dict[Path, int] = {}
    manifests: list[dict[str, Any]] = []

    plan = {
        "meta_jsonl": str(args.meta_jsonl),
        "baseline_root": str(args.baseline_root),
        "output_dir": str(output_dir),
        "indices": indices,
        "image_size": list(image_size),
        "include_gt": include_gt,
        "models": [spec.name for spec in selected_models],
        "timepoint_mode": "absolute" if abs_frames is not None else "relative",
        "timepoints": abs_frames if abs_frames is not None else rel_frames,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for index in indices:
        row = rows[index]
        episode = int(row["episode_index"])
        stem = val_stem(index, episode)
        sample_dir = output_dir / stem
        if not args.dry_run:
            prepare_sample_dir(sample_dir, args.overwrite)
        manifest = export_sample(
            row=row,
            index=index,
            sample_dir=sample_dir,
            baseline_root=args.baseline_root,
            selected_models=selected_models,
            rel_frames=rel_frames,
            abs_frames=abs_frames,
            image_size=image_size,
            ffmpeg_bin=args.ffmpeg_bin,
            ffprobe_bin=args.ffprobe_bin,
            include_gt=include_gt,
            frame_count_cache=frame_count_cache,
            dry_run=args.dry_run,
        )
        manifests.append(manifest)

    if args.dry_run:
        print(json.dumps({"samples": manifests}, indent=2, ensure_ascii=False))
        return

    index_manifest = {
        "meta_jsonl": str(args.meta_jsonl),
        "baseline_root": str(args.baseline_root),
        "output_dir": str(output_dir),
        "samples": [
            {
                "sample": manifest["sample"],
                "val_index": manifest["val_index"],
                "episode_index": manifest["episode_index"],
                "manifest": str(output_dir / manifest["sample"] / "manifest.json"),
            }
            for manifest in manifests
        ],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(index_manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Saved frame assets for {len(manifests)} sample(s) to: {output_dir}")


if __name__ == "__main__":
    main()
