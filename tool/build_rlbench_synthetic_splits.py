#!/usr/bin/env python3
"""Build RLBench trainA/trainB-synthetic training splits without symlinks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
from PIL import Image
from tqdm import tqdm


DEFAULT_SOURCE_TRAIN_ROOT = Path("/data2/xuehao/datasets/rlbench/data/train")
DEFAULT_OUTPUT_DATA_ROOT = Path("/data2/xuehao/datasets/rlbench/data")
DEFAULT_SPLIT_JSON = Path(
    "/data2/xuehao/datasets/rlbench_meta/meta/trainA_trainB_split.json"
)
DEFAULT_TRAINB_MANIFEST = Path(
    "/data2/xuehao/datasets/rlbench_meta/meta/episodes_cross_view_trainB_81_textemb.jsonl"
)
DEFAULT_SYNTHETIC_WRIST_DIR = Path(
    "/data2/xuehao/datasets/rlbench_meta/wan_infer/"
    "rlbench_stage2_trainA_tail_key3_no3d_128_epoch25_trainB_wrist_only/"
    "wrist_pred/val"
)
DEFAULT_SUMMARY_PATH = Path(
    "/data2/xuehao/datasets/rlbench_meta/meta/"
    "rlbench_synthetic_split_build_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create RLBench traina/trainb_synthetic/"
            "traina_add_trainb_synthetic splits using physical copies."
        )
    )
    parser.add_argument("--source_train_root", type=Path, default=DEFAULT_SOURCE_TRAIN_ROOT)
    parser.add_argument("--output_data_root", type=Path, default=DEFAULT_OUTPUT_DATA_ROOT)
    parser.add_argument("--split_json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--trainb_manifest", type=Path, default=DEFAULT_TRAINB_MANIFEST)
    parser.add_argument("--synthetic_wrist_dir", type=Path, default=DEFAULT_SYNTHETIC_WRIST_DIR)
    parser.add_argument("--summary_path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output split directories and temporary build dirs first.",
    )
    return parser.parse_args()


def load_split(split_path: Path) -> dict[str, Any]:
    with split_path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    if "tasks" not in split or not isinstance(split["tasks"], dict):
        raise ValueError(f"Invalid split file: {split_path}")
    return split


def episode_dir(root: Path, task: str, episode_number: int) -> Path:
    return root / task / "all_variations" / "episodes" / f"episode{episode_number}"


def copy_files_only(src: Path, dst: Path, skip_names: set[str] | None = None) -> int:
    if not src.is_dir():
        raise FileNotFoundError(src)
    skip_names = skip_names or set()
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for child in src.iterdir():
        if child.name in skip_names:
            continue
        if child.is_file():
            shutil.copy2(child, dst / child.name, follow_symlinks=True)
            copied += 1
    return copied


def make_task_shell(src_train_root: Path, dst_split_root: Path, task: str) -> None:
    src_task = src_train_root / task
    dst_task = dst_split_root / task
    src_all = src_task / "all_variations"
    dst_all = dst_task / "all_variations"
    copy_files_only(src_task, dst_task, skip_names={"all_variations"})
    copy_files_only(src_all, dst_all, skip_names={"episodes"})
    (dst_all / "episodes").mkdir(parents=True, exist_ok=True)


def copy_episode(src_episode: Path, dst_episode: Path, ignore_wrist_rgb: bool = False) -> None:
    if not src_episode.is_dir():
        raise FileNotFoundError(src_episode)
    if dst_episode.exists():
        raise FileExistsError(dst_episode)
    if ignore_wrist_rgb:
        copy_episode_without_wrist_rgb(src_episode, dst_episode)
        return

    dst_episode.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cp", "-a", "--no-dereference", str(src_episode), str(dst_episode)],
        check=True,
    )


def copy_episode_without_wrist_rgb(src_episode: Path, dst_episode: Path) -> None:
    if not src_episode.is_dir():
        raise FileNotFoundError(src_episode)
    if dst_episode.exists():
        raise FileExistsError(dst_episode)
    dst_episode.mkdir(parents=True, exist_ok=False)

    for child in src_episode.iterdir():
        if child.name == "wrist_rgb":
            continue
        target = dst_episode / child.name
        if child.is_dir():
            subprocess.run(
                ["cp", "-a", "--no-dereference", str(child), str(target)],
                check=True,
            )
        elif child.is_file():
            shutil.copy2(child, target, follow_symlinks=True)


def load_trainb_manifest(
    manifest_path: Path,
    synthetic_wrist_dir: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    with manifest_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task = str(row["rlbench_task"])
            episode_number = int(row["rlbench_episode_number"])
            episode_index = int(row["episode_index"])
            video_path = synthetic_wrist_dir / f"val_{idx:03d}_ep{episode_index}.mp4"
            item = {
                "idx": idx,
                "task": task,
                "episode_number": episode_number,
                "episode_index": episode_index,
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "valid_frames": int(row["valid_frames"]),
                "video_path": str(video_path),
            }
            rows.append(item)
            grouped[(task, episode_number)].append(item)
    for items in grouped.values():
        items.sort(key=lambda row: int(row["start_frame"]))
    return rows, dict(grouped)


def list_png_names(path: Path) -> set[str]:
    return {child.name for child in path.iterdir() if child.is_file() and child.suffix == ".png"}


def decode_synthetic_wrist(rows: list[dict[str, Any]], dst_wrist_dir: Path) -> int:
    dst_wrist_dir.mkdir(parents=True, exist_ok=False)
    written: set[int] = set()
    for row in rows:
        start_frame = int(row["start_frame"])
        valid_frames = int(row["valid_frames"])
        video_path = Path(str(row["video_path"]))
        reader = imageio.get_reader(str(video_path))
        try:
            for local_idx, frame in enumerate(reader):
                if local_idx >= valid_frames:
                    break
                frame_index = start_frame + local_idx
                if frame_index in written:
                    raise ValueError(
                        f"Duplicate synthetic frame {frame_index} for {dst_wrist_dir}"
                    )
                Image.fromarray(frame).convert("RGB").save(dst_wrist_dir / f"{frame_index}.png")
                written.add(frame_index)
            if local_idx + 1 < valid_frames:
                raise RuntimeError(
                    f"Synthetic video too short: {video_path} has {local_idx + 1} "
                    f"frames, need {valid_frames}"
                )
        finally:
            reader.close()
    return len(written)


def build_trainb_synthetic_episode(
    src_train_root: Path,
    dst_split_root: Path,
    task: str,
    episode_number: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    src_episode = episode_dir(src_train_root, task, episode_number)
    dst_episode = episode_dir(dst_split_root, task, episode_number)
    copy_episode(src_episode, dst_episode, ignore_wrist_rgb=True)
    written = decode_synthetic_wrist(rows, dst_episode / "wrist_rgb")
    source_wrist_count = len(list_png_names(src_episode / "wrist_rgb"))
    if written != source_wrist_count:
        raise RuntimeError(
            f"Frame count mismatch for {task}/episode{episode_number}: "
            f"wrote {written}, source has {source_wrist_count}"
        )
    return {
        "task": task,
        "episode_number": episode_number,
        "frames": written,
        "clips": len(rows),
    }


def prepare_output_dirs(output_data_root: Path, split_names: list[str], overwrite: bool) -> dict[str, Path]:
    output_data_root.mkdir(parents=True, exist_ok=True)
    targets = {name: output_data_root / name for name in split_names}
    temps = {name: output_data_root / f".{name}.tmp.{os.getpid()}" for name in split_names}
    for path in [*targets.values(), *temps.values()]:
        if path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{path} already exists. Pass --overwrite to remove it first."
                )
            shutil.rmtree(path)
    return temps


def build_split_shells(src_train_root: Path, split_root: Path, tasks: list[str]) -> None:
    for task in tasks:
        make_task_shell(src_train_root, split_root, task)


def run_parallel(jobs: list[tuple], fn, workers: int, desc: str) -> list[Any]:
    if not jobs:
        return []
    results: list[Any] = []
    max_workers = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fn, *job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            results.append(future.result())
    return results


def validate_split_counts(root: Path, split: dict[str, Any], key: str, expected_per_task: int) -> None:
    for task, info in split["tasks"].items():
        eps_dir = root / task / "all_variations" / "episodes"
        episodes = sorted(path.name for path in eps_dir.iterdir() if path.is_dir())
        expected = {f"episode{int(ep)}" for ep in info[key]}
        if set(episodes) != expected:
            missing = sorted(expected - set(episodes))[:8]
            extra = sorted(set(episodes) - expected)[:8]
            raise RuntimeError(
                f"{root.name}/{task} episode mismatch: missing={missing}, extra={extra}"
            )
        if len(episodes) != expected_per_task:
            raise RuntimeError(
                f"{root.name}/{task} expected {expected_per_task} episodes, got {len(episodes)}"
            )


def validate_mixed_counts(root: Path, split: dict[str, Any]) -> None:
    for task, info in split["tasks"].items():
        eps_dir = root / task / "all_variations" / "episodes"
        episodes = sorted(path.name for path in eps_dir.iterdir() if path.is_dir())
        expected = {
            f"episode{int(ep)}"
            for ep in [*info["trainA"], *info["trainB"]]
        }
        if set(episodes) != expected or len(episodes) != 100:
            raise RuntimeError(
                f"{root.name}/{task} expected 100 mixed episodes, got {len(episodes)}"
            )


def count_symlinks(root: Path) -> int:
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames:
            if os.path.islink(os.path.join(dirpath, name)):
                count += 1
        for name in filenames:
            if os.path.islink(os.path.join(dirpath, name)):
                count += 1
    return count


def preflight(
    args: argparse.Namespace,
    split: dict[str, Any],
    trainb_rows: list[dict[str, Any]],
    trainb_grouped: dict[tuple[str, int], list[dict[str, Any]]],
) -> dict[str, Any]:
    if not args.source_train_root.is_dir():
        raise FileNotFoundError(args.source_train_root)
    if not args.synthetic_wrist_dir.is_dir():
        raise FileNotFoundError(args.synthetic_wrist_dir)
    tasks = sorted(split["tasks"].keys())
    if len(tasks) != 18:
        raise RuntimeError(f"Expected 18 tasks, found {len(tasks)}")

    traina_total = 0
    trainb_total = 0
    missing_sources: list[str] = []
    missing_videos: list[str] = []
    for task, info in split["tasks"].items():
        if len(info["trainA"]) != 50 or len(info["trainB"]) != 50:
            raise RuntimeError(
                f"{task}: expected trainA/trainB 50/50, got "
                f"{len(info['trainA'])}/{len(info['trainB'])}"
            )
        traina_total += len(info["trainA"])
        trainb_total += len(info["trainB"])
        for key in ("trainA", "trainB"):
            for ep in info[key]:
                src_episode = episode_dir(args.source_train_root, task, int(ep))
                if not src_episode.is_dir():
                    missing_sources.append(str(src_episode))

    trainb_keys = {
        (task, int(ep))
        for task, info in split["tasks"].items()
        for ep in info["trainB"]
    }
    manifest_keys = set(trainb_grouped.keys())
    if manifest_keys != trainb_keys:
        missing = sorted(trainb_keys - manifest_keys)[:8]
        extra = sorted(manifest_keys - trainb_keys)[:8]
        raise RuntimeError(f"TrainB manifest key mismatch: missing={missing}, extra={extra}")
    for row in trainb_rows:
        if not Path(row["video_path"]).is_file():
            missing_videos.append(row["video_path"])

    if missing_sources:
        raise FileNotFoundError(f"Missing source episodes: {missing_sources[:8]}")
    if missing_videos:
        raise FileNotFoundError(f"Missing synthetic videos: {missing_videos[:8]}")

    free_bytes = shutil.disk_usage(args.output_data_root).free
    return {
        "tasks": len(tasks),
        "trainA_episodes": traina_total,
        "trainB_episodes": trainb_total,
        "trainB_manifest_rows": len(trainb_rows),
        "synthetic_videos": len(trainb_rows),
        "free_bytes_before": free_bytes,
    }


def main() -> None:
    args = parse_args()
    split_names = ["traina", "trainb_synthetic", "traina_add_trainb_synthetic"]
    split = load_split(args.split_json)
    tasks = sorted(split["tasks"].keys())
    trainb_rows, trainb_grouped = load_trainb_manifest(
        args.trainb_manifest,
        args.synthetic_wrist_dir,
    )
    preflight_summary = preflight(args, split, trainb_rows, trainb_grouped)
    temp_roots = prepare_output_dirs(args.output_data_root, split_names, args.overwrite)

    print(json.dumps(preflight_summary, indent=2, ensure_ascii=False))
    print(f"Building temporary splits under {args.output_data_root}")

    for root in temp_roots.values():
        build_split_shells(args.source_train_root, root, tasks)

    traina_jobs = [
        (
            episode_dir(args.source_train_root, task, int(ep)),
            episode_dir(temp_roots["traina"], task, int(ep)),
            False,
        )
        for task, info in split["tasks"].items()
        for ep in info["trainA"]
    ]
    run_parallel(traina_jobs, copy_episode, args.workers, "copy traina")

    trainb_jobs = [
        (
            args.source_train_root,
            temp_roots["trainb_synthetic"],
            task,
            int(ep),
            trainb_grouped[(task, int(ep))],
        )
        for task, info in split["tasks"].items()
        for ep in info["trainB"]
    ]
    trainb_results = run_parallel(
        trainb_jobs,
        build_trainb_synthetic_episode,
        args.workers,
        "copy trainb_synthetic + decode wrist",
    )

    mixed_jobs: list[tuple[Path, Path, bool]] = []
    for task, info in split["tasks"].items():
        for ep in info["trainA"]:
            mixed_jobs.append(
                (
                    episode_dir(args.source_train_root, task, int(ep)),
                    episode_dir(temp_roots["traina_add_trainb_synthetic"], task, int(ep)),
                    False,
                )
            )
        for ep in info["trainB"]:
            mixed_jobs.append(
                (
                    episode_dir(temp_roots["trainb_synthetic"], task, int(ep)),
                    episode_dir(temp_roots["traina_add_trainb_synthetic"], task, int(ep)),
                    False,
                )
            )
    run_parallel(mixed_jobs, copy_episode, args.workers, "copy mixed")

    print("Running post-build validation...")
    validate_split_counts(temp_roots["traina"], split, "trainA", 50)
    validate_split_counts(temp_roots["trainb_synthetic"], split, "trainB", 50)
    validate_mixed_counts(temp_roots["traina_add_trainb_synthetic"], split)
    symlink_counts = {name: count_symlinks(root) for name, root in temp_roots.items()}
    bad_symlinks = {name: count for name, count in symlink_counts.items() if count}
    if bad_symlinks:
        raise RuntimeError(f"Symlinks found in outputs: {bad_symlinks}")

    target_roots = {name: args.output_data_root / name for name in split_names}
    for name in split_names:
        os.replace(temp_roots[name], target_roots[name])

    summary = {
        "source_train_root": str(args.source_train_root),
        "output_data_root": str(args.output_data_root),
        "split_json": str(args.split_json),
        "trainb_manifest": str(args.trainb_manifest),
        "synthetic_wrist_dir": str(args.synthetic_wrist_dir),
        "outputs": {name: str(path) for name, path in target_roots.items()},
        "preflight": preflight_summary,
        "trainb_synthetic": {
            "episodes": len(trainb_results),
            "clips": sum(int(item["clips"]) for item in trainb_results),
            "frames": sum(int(item["frames"]) for item in trainb_results),
        },
        "symlink_counts": symlink_counts,
        "free_bytes_after": shutil.disk_usage(args.output_data_root).free,
        "workers": int(args.workers),
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
