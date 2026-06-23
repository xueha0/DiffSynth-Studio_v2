#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import shutil
import subprocess
import sys
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


DEFAULT_DATASET_ROOT = "/data2/xuehao/datasets/rlbench"
DEFAULT_OUTPUT_ROOT = "/data2/xuehao/datasets/rlbench_meta"
DEFAULT_SOURCE_VIEWS = ("front_rgb", "overhead_rgb")
DEFAULT_TARGET_VIEW = "wrist_rgb"


class _StubObservation:
    pass


class _StubDemo:
    def __len__(self):
        return len(self._observations)

    def __getitem__(self, index):
        return self._observations[index]


@dataclass(frozen=True)
class EpisodeRef:
    task: str
    task_index: int
    episode_name: str
    episode_number: int
    path: Path
    split: str
    local_episode_index: int
    global_episode_index: int
    prompt: str
    length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare RLBench cross-view metadata and mp4 assets."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-views", default=",".join(DEFAULT_SOURCE_VIEWS))
    parser.add_argument("--target-view", default=DEFAULT_TARGET_VIEW)
    parser.add_argument("--clip-length", type=int, default=81)
    parser.add_argument("--stride", type=int, default=81)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--trainA-per-task", type=int, default=50)
    parser.add_argument("--trainB-per-task", type=int, default=50)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Episode-level worker processes for mp4/state generation.",
    )
    parser.add_argument(
        "--fourcc",
        default="mp4v",
        help="OpenCV VideoWriter fourcc. Default mp4v keeps dependencies minimal.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional dry-run cap on number of tasks.",
    )
    parser.add_argument(
        "--max-episodes-per-split",
        type=int,
        default=None,
        help="Optional dry-run cap per split after task split selection.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing mp4/parquet files when present.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mp4/parquet files.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Only write metadata/state files, do not encode mp4 assets.",
    )
    parser.add_argument(
        "--state-read-mode",
        choices=("auto", "stub", "rlbench"),
        default="auto",
        help="How to unpickle low_dim_obs.pkl. auto tries normal pickle then stub fallback.",
    )
    return parser.parse_args()


def install_rlbench_pickle_stubs() -> None:
    rlbench_mod = sys.modules.setdefault("rlbench", types.ModuleType("rlbench"))
    backend_mod = sys.modules.setdefault("rlbench.backend", types.ModuleType("rlbench.backend"))
    observation_mod = types.ModuleType("rlbench.backend.observation")
    observation_mod.Observation = _StubObservation
    demo_mod = types.ModuleType("rlbench.demo")
    demo_mod.Demo = _StubDemo
    sys.modules["rlbench.backend.observation"] = observation_mod
    sys.modules["rlbench.demo"] = demo_mod
    setattr(rlbench_mod, "backend", backend_mod)
    setattr(backend_mod, "observation", observation_mod)
    setattr(rlbench_mod, "demo", demo_mod)


def load_pickle(path: Path, mode: str = "auto") -> Any:
    if mode == "stub":
        install_rlbench_pickle_stubs()
        with path.open("rb") as handle:
            return pickle.load(handle)
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except ModuleNotFoundError:
        if mode == "rlbench":
            raise
        install_rlbench_pickle_stubs()
        with path.open("rb") as handle:
            return pickle.load(handle)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_views(source_views: str, target_view: str) -> list[str]:
    views = [item.strip() for item in source_views.split(",") if item.strip()]
    if len(views) != 2:
        raise ValueError(f"Expected exactly two source views, got {views!r}.")
    target = target_view.strip()
    if not target:
        raise ValueError("Missing target view.")
    if target in views:
        raise ValueError(f"Target view {target!r} must not be in source views {views!r}.")
    return [*views, target]


def numeric_pngs(directory: Path) -> list[Path]:
    files = []
    for path in directory.glob("*.png"):
        try:
            int(path.stem)
        except ValueError:
            continue
        files.append(path)
    files.sort(key=lambda item: int(item.stem))
    return files


def read_prompt(episode_dir: Path, task: str) -> str:
    path = episode_dir / "variation_descriptions.pkl"
    if path.exists():
        with path.open("rb") as handle:
            descriptions = pickle.load(handle)
        if isinstance(descriptions, (list, tuple)):
            for item in descriptions:
                prompt = " ".join(str(item).strip().split())
                if prompt:
                    return prompt
        prompt = " ".join(str(descriptions).strip().split())
        if prompt:
            return prompt
    return task.replace("_", " ")


def collect_episodes(dataset_root: Path, views: list[str], state_mode: str) -> list[EpisodeRef]:
    train_root = dataset_root / "data" / "train"
    tasks = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    episodes: list[EpisodeRef] = []
    global_episode_index = 0
    for task_index, task in enumerate(tasks):
        episodes_root = train_root / task / "all_variations" / "episodes"
        task_episodes = sorted(
            [
                path
                for path in episodes_root.iterdir()
                if path.is_dir()
                and path.name.startswith("episode")
                and path.name[len("episode") :].isdigit()
            ],
            key=lambda path: int(path.name[len("episode") :]),
        )
        for episode_dir in task_episodes:
            missing = [view for view in views if not (episode_dir / view).is_dir()]
            if missing:
                raise FileNotFoundError(f"{episode_dir} missing view directories: {missing}")
            frame_counts = [len(numeric_pngs(episode_dir / view)) for view in views]
            if any(count <= 0 for count in frame_counts):
                raise ValueError(f"{episode_dir} has empty view frame counts: {frame_counts}")
            if len(set(frame_counts)) != 1:
                raise ValueError(f"{episode_dir} has inconsistent view frame counts: {frame_counts}")
            state_path = episode_dir / "low_dim_obs.pkl"
            if not state_path.exists():
                raise FileNotFoundError(f"Missing state file: {state_path}")
            episode_number = int(episode_dir.name[len("episode") :])
            episodes.append(
                EpisodeRef(
                    task=task,
                    task_index=task_index,
                    episode_name=episode_dir.name,
                    episode_number=episode_number,
                    path=episode_dir,
                    split="",
                    local_episode_index=-1,
                    global_episode_index=global_episode_index,
                    prompt=read_prompt(episode_dir, task),
                    length=frame_counts[0],
                )
            )
            global_episode_index += 1
    return episodes


def split_episodes(
    episodes: list[EpisodeRef],
    train_a_per_task: int,
    train_b_per_task: int,
    seed: int,
    max_tasks: int | None,
    max_episodes_per_split: int | None,
) -> tuple[list[EpisodeRef], list[EpisodeRef], dict]:
    grouped: dict[str, list[EpisodeRef]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.task].append(episode)
    tasks = sorted(grouped.keys())
    if max_tasks is not None:
        tasks = tasks[: int(max_tasks)]

    train_a: list[EpisodeRef] = []
    train_b: list[EpisodeRef] = []
    split_manifest: dict[str, dict] = OrderedDict()
    rng = random.Random(int(seed))
    for task in tasks:
        task_items = sorted(grouped[task], key=lambda item: item.episode_number)
        shuffled = list(task_items)
        rng.shuffle(shuffled)
        required = int(train_a_per_task) + int(train_b_per_task)
        if len(shuffled) < required:
            raise ValueError(f"Task {task} has {len(shuffled)} episodes, need {required}.")
        a_items = shuffled[: int(train_a_per_task)]
        b_items = shuffled[int(train_a_per_task) : required]
        if max_episodes_per_split is not None:
            a_items = a_items[: int(max_episodes_per_split)]
            b_items = b_items[: int(max_episodes_per_split)]
        split_manifest[task] = {
            "task_index": int(task_items[0].task_index),
            "trainA": [int(item.episode_number) for item in a_items],
            "trainB": [int(item.episode_number) for item in b_items],
        }
        train_a.extend(
            EpisodeRef(**{**item.__dict__, "split": "trainA", "local_episode_index": idx})
            for idx, item in enumerate(a_items)
        )
        train_b.extend(
            EpisodeRef(**{**item.__dict__, "split": "trainB", "local_episode_index": idx})
            for idx, item in enumerate(b_items)
        )
    return train_a, train_b, split_manifest


def output_video_path(output_root: Path, episode: EpisodeRef, view: str) -> Path:
    return (
        output_root
        / "videos"
        / episode.split
        / view
        / episode.task
        / f"episode_{episode.global_episode_index:06d}.mp4"
    )


def output_state_path(output_root: Path, episode: EpisodeRef) -> Path:
    return (
        output_root
        / "data"
        / episode.split
        / episode.task
        / f"episode_{episode.global_episode_index:06d}.parquet"
    )


def encode_view_mp4(
    image_dir: Path,
    output_path: Path,
    fps: float,
    fourcc: str,
    skip_existing: bool,
    overwrite: bool,
) -> None:
    if output_path.exists():
        if overwrite:
            output_path.unlink()
        elif skip_existing:
            return
        else:
            return

    frames = numeric_pngs(image_dir)
    if not frames:
        raise ValueError(f"No PNG frames found in {image_dir}")
    stems = [int(path.stem) for path in frames]
    if stems == list(range(len(stems))) and shutil.which("ffmpeg"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp.mp4")
        if tmp_path.exists():
            tmp_path.unlink()
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(float(fps)),
            "-i",
            str(image_dir / "%d.png"),
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            str(tmp_path),
        ]
        try:
            subprocess.run(cmd, check=True)
            tmp_path.replace(output_path)
            return
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            # Fall back to OpenCV below; this keeps the script usable if the
            # local ffmpeg build lacks libx264.

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise OSError(f"Could not read {frames[0]}")
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*fourcc),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise OSError(f"Could not open VideoWriter for {output_path}")
    try:
        writer.write(first)
        for frame_path in frames[1:]:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise OSError(f"Could not read {frame_path}")
            if frame.shape[:2] != (height, width):
                raise ValueError(f"Inconsistent frame size in {frame_path}: {frame.shape[:2]}")
            writer.write(frame)
    finally:
        writer.release()


def quaternion_to_euler_xyz(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float32).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"Expected quaternion with 4 values, got shape {q.shape}")
    candidates = [
        q,  # scipy xyzw; most RLBench gripper_pose dumps follow xyzw.
        np.array([q[1], q[2], q[3], q[0]], dtype=np.float32),  # wxyz fallback.
    ]
    norms = [abs(float(np.linalg.norm(item)) - 1.0) for item in candidates]
    quat_xyzw = candidates[int(np.argmin(norms))]
    return Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False).astype(np.float32)


def extract_state_pose_7d(episode_dir: Path, state_mode: str) -> np.ndarray:
    demo = load_pickle(episode_dir / "low_dim_obs.pkl", mode=state_mode)
    rows = []
    for obs in demo:
        pose = np.asarray(getattr(obs, "gripper_pose"), dtype=np.float32).reshape(-1)
        if pose.shape[0] < 7:
            raise ValueError(f"{episode_dir} observation has gripper_pose shape {pose.shape}")
        xyz = pose[:3].astype(np.float32)
        euler = quaternion_to_euler_xyz(pose[3:7])
        gripper_open = np.asarray([float(getattr(obs, "gripper_open"))], dtype=np.float32)
        rows.append(np.concatenate([xyz, euler, gripper_open], axis=0))
    if not rows:
        raise ValueError(f"No low-dim observations in {episode_dir}")
    return np.stack(rows, axis=0).astype(np.float32)


def write_state_parquet(
    episode: EpisodeRef,
    output_root: Path,
    state_mode: str,
    skip_existing: bool,
    overwrite: bool,
) -> Path:
    path = output_state_path(output_root, episode)
    if path.exists():
        if overwrite:
            path.unlink()
        elif skip_existing:
            return path
        else:
            return path

    arr = extract_state_pose_7d(episode.path, state_mode)
    cartesian = arr[:, :6].astype(np.float32)
    gripper = arr[:, 6].astype(np.float32)
    table = pa.table(
        {
            "observation.state.cartesian_position": pa.array(cartesian.tolist()),
            "observation.state.gripper_position": pa.array(gripper.tolist()),
            "frame_index": pa.array(np.arange(arr.shape[0], dtype=np.int32)),
            "episode_index": pa.array(
                np.full(arr.shape[0], int(episode.global_episode_index), dtype=np.int32)
            ),
            "task_index": pa.array(
                np.full(arr.shape[0], int(episode.task_index), dtype=np.int32)
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def process_episode_assets(payload: dict) -> dict:
    episode = payload["episode"]
    output_root = payload["output_root"]
    views = payload["views"]
    encoded_videos = 0
    if not payload["no_video"]:
        for view in views:
            out_path = output_video_path(output_root, episode, view)
            existed = out_path.exists()
            encode_view_mp4(
                episode.path / view,
                out_path,
                fps=payload["fps"],
                fourcc=payload["fourcc"],
                skip_existing=payload["skip_existing"],
                overwrite=payload["overwrite"],
            )
            if payload["overwrite"] or not existed:
                encoded_videos += 1
    state_path = write_state_parquet(
        episode,
        output_root=output_root,
        state_mode=payload["state_mode"],
        skip_existing=payload["skip_existing"],
        overwrite=payload["overwrite"],
    )
    state_arr = extract_state_pose_7d(episode.path, payload["state_mode"])
    return {
        "episode": episode,
        "state_path": state_path,
        "state_arr": state_arr,
        "encoded_videos": encoded_videos,
    }


def build_clip_starts(length: int, clip_length: int, stride: int) -> list[int]:
    if length <= 0:
        return []
    starts = list(range(0, int(length), int(stride)))
    return starts or [0]


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def build_rows_for_split(
    episodes: list[EpisodeRef],
    output_root: Path,
    views: list[str],
    clip_length: int,
    stride: int,
    fps: float,
    fourcc: str,
    state_mode: str,
    skip_existing: bool,
    overwrite: bool,
    no_video: bool,
    workers: int,
) -> tuple[list[dict], list[np.ndarray], dict]:
    rows: list[dict] = []
    state_chunks: list[np.ndarray] = []
    stats = {
        "episodes": len(episodes),
        "clips": 0,
        "encoded_videos": 0,
        "state_files": 0,
        "min_episode_frames": None,
        "max_episode_frames": None,
    }
    lengths = []
    asset_results: dict[int, dict] = {}
    payloads = [
        {
            "episode": episode,
            "output_root": output_root,
            "views": views,
            "fps": float(fps),
            "fourcc": str(fourcc),
            "state_mode": state_mode,
            "skip_existing": bool(skip_existing),
            "overwrite": bool(overwrite),
            "no_video": bool(no_video),
        }
        for episode in episodes
    ]
    if int(workers) <= 1:
        for payload in payloads:
            result = process_episode_assets(payload)
            asset_results[int(result["episode"].global_episode_index)] = result
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = [executor.submit(process_episode_assets, payload) for payload in payloads]
            for future in as_completed(futures):
                result = future.result()
                asset_results[int(result["episode"].global_episode_index)] = result

    for episode in episodes:
        lengths.append(int(episode.length))
        result = asset_results[int(episode.global_episode_index)]
        stats["encoded_videos"] += int(result["encoded_videos"])
        state_path = result["state_path"]
        stats["state_files"] += 1
        state_chunks.append(result["state_arr"])

        for start in build_clip_starts(episode.length, clip_length, stride):
            valid_frames = min(int(clip_length), max(0, int(episode.length) - int(start)))
            if valid_frames <= 0:
                continue
            end = int(start) + int(valid_frames) - 1
            row = {
                "episode_index": int(episode.global_episode_index),
                "rlbench_episode_number": int(episode.episode_number),
                "rlbench_episode_name": episode.episode_name,
                "rlbench_task": episode.task,
                "length": int(clip_length),
                "valid_frames": int(valid_frames),
                "start_frame": int(start),
                "end_frame": int(end),
                "video": [
                    {
                        "data": rel(output_video_path(output_root, episode, view), output_root),
                        "start_frame": int(start),
                        "end_frame": int(end),
                        "pad_to_frames": int(clip_length),
                        "pad_mode": "repeat_last",
                    }
                    for view in views
                ],
                "state": {
                    "data": rel(state_path, output_root),
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "pad_to_frames": int(clip_length),
                    "pad_mode": "repeat_last",
                },
                "prompt": episode.prompt,
                "task": episode.task,
                "task_index": int(episode.task_index),
                "source_views": [0, 1],
                "target_view": 2,
                "data_type": episode.split,
                "state_type": "state_pose_7d",
                "source_view_names": views[:2],
                "target_view_name": views[2],
            }
            rows.append(row)
    stats["clips"] = len(rows)
    if lengths:
        stats["min_episode_frames"] = min(lengths)
        stats["max_episode_frames"] = max(lengths)
    return rows, state_chunks, stats


def summarize_array(arr: np.ndarray) -> dict:
    return {
        "shape": [int(arr.shape[1])],
        "min": np.min(arr, axis=0).astype(float).tolist(),
        "max": np.max(arr, axis=0).astype(float).tolist(),
        "p01": np.percentile(arr, 1, axis=0).astype(float).tolist(),
        "p99": np.percentile(arr, 99, axis=0).astype(float).tolist(),
        "mean": np.mean(arr, axis=0).astype(float).tolist(),
        "std": np.std(arr, axis=0).astype(float).tolist(),
    }


def validate_no_overlap(train_a: list[EpisodeRef], train_b: list[EpisodeRef]) -> None:
    a = {(item.task, item.episode_number) for item in train_a}
    b = {(item.task, item.episode_number) for item in train_b}
    overlap = sorted(a & b)
    if overlap:
        raise ValueError(f"trainA/trainB overlap: {overlap[:10]}")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    if args.skip_existing and args.overwrite:
        raise ValueError("--skip-existing and --overwrite are mutually exclusive.")
    if int(args.clip_length) <= 0 or int(args.stride) <= 0:
        raise ValueError("--clip-length and --stride must be positive.")
    if int(args.stride) != int(args.clip_length):
        raise ValueError("This script uses repeat_last tails; keep --stride == --clip-length.")
    views = parse_views(args.source_views, args.target_view)

    print(f"[rlbench-meta] dataset_root: {dataset_root}")
    print(f"[rlbench-meta] output_root : {output_root}")
    print(f"[rlbench-meta] views       : {views}")
    episodes = collect_episodes(dataset_root, views, state_mode=args.state_read_mode)
    train_a, train_b, split_manifest = split_episodes(
        episodes,
        train_a_per_task=int(args.trainA_per_task),
        train_b_per_task=int(args.trainB_per_task),
        seed=int(args.split_seed),
        max_tasks=args.max_tasks,
        max_episodes_per_split=args.max_episodes_per_split,
    )
    validate_no_overlap(train_a, train_b)
    print(f"[rlbench-meta] selected trainA episodes: {len(train_a)}")
    print(f"[rlbench-meta] selected trainB episodes: {len(train_b)}")

    train_a_rows, train_a_states, train_a_stats = build_rows_for_split(
        train_a,
        output_root=output_root,
        views=views,
        clip_length=int(args.clip_length),
        stride=int(args.stride),
        fps=float(args.fps),
        fourcc=str(args.fourcc),
        state_mode=args.state_read_mode,
        skip_existing=bool(args.skip_existing),
        overwrite=bool(args.overwrite),
        no_video=bool(args.no_video),
        workers=int(args.workers),
    )
    train_b_rows, train_b_states, train_b_stats = build_rows_for_split(
        train_b,
        output_root=output_root,
        views=views,
        clip_length=int(args.clip_length),
        stride=int(args.stride),
        fps=float(args.fps),
        fourcc=str(args.fourcc),
        state_mode=args.state_read_mode,
        skip_existing=bool(args.skip_existing),
        overwrite=bool(args.overwrite),
        no_video=bool(args.no_video),
        workers=int(args.workers),
    )

    meta_root = output_root / "meta"
    train_a_jsonl = meta_root / f"episodes_cross_view_trainA_{int(args.clip_length)}.jsonl"
    train_b_jsonl = meta_root / f"episodes_cross_view_trainB_{int(args.clip_length)}.jsonl"
    tasks_jsonl = meta_root / "tasks_cross_view.jsonl"
    split_json = meta_root / "trainA_trainB_split.json"
    stat_json = meta_root / "stat_state_pose_7d.json"
    summary_json = meta_root / "summary_rlbench_cross_view.json"

    dump_jsonl(train_a_jsonl, train_a_rows)
    dump_jsonl(train_b_jsonl, train_b_rows)
    task_rows = [
        {
            "task_index": index,
            "task": task,
            "prompt": task.replace("_", " "),
        }
        for index, task in enumerate(sorted({item.task for item in [*train_a, *train_b]}))
    ]
    dump_jsonl(tasks_jsonl, task_rows)
    dump_json(
        split_json,
        {
            "split_seed": int(args.split_seed),
            "trainA_per_task": int(args.trainA_per_task),
            "trainB_per_task": int(args.trainB_per_task),
            "source_train_split": "train",
            "tasks": split_manifest,
        },
    )

    if not train_a_states:
        raise ValueError("No trainA states collected for stats.")
    state_arr = np.concatenate(train_a_states, axis=0)
    dump_json(stat_json, {"state_pose_7d": summarize_array(state_arr)})
    dump_json(
        summary_json,
        {
            "dataset_root": str(dataset_root),
            "output_root": str(output_root),
            "source_views": views[:2],
            "target_view": views[2],
            "source_views_indices": [0, 1],
            "target_view_index": 2,
            "clip_length": int(args.clip_length),
            "stride": int(args.stride),
            "fps": float(args.fps),
            "fourcc": str(args.fourcc),
            "workers": int(args.workers),
            "split_seed": int(args.split_seed),
            "trainA_jsonl": str(train_a_jsonl),
            "trainB_jsonl": str(train_b_jsonl),
            "tasks_jsonl": str(tasks_jsonl),
            "stat_json": str(stat_json),
            "trainA": train_a_stats,
            "trainB": train_b_stats,
        },
    )
    print(f"[rlbench-meta] wrote {train_a_jsonl} rows={len(train_a_rows)}")
    print(f"[rlbench-meta] wrote {train_b_jsonl} rows={len(train_b_rows)}")
    print(f"[rlbench-meta] wrote {stat_json}")
    print(f"[rlbench-meta] wrote {summary_json}")


if __name__ == "__main__":
    main()
