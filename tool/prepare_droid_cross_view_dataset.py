#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

"""
python tool/prepare_droid_cross_view_dataset.py \
    --dataset-root /data2/xuehao/datasets/droid_success_high_quality \
    --output-root /data2/xuehao/datasets/droid_success_high_quality_crossview_meta \
    --dataset-format lerobot_v3 \
    --clip-length 81 \
    --train-stride 81 \
    --small-train-episodes 200 \
    --small-val-episodes 15 \
    --build-prompt-emb
"""

DROID_SOURCE_VIEWS_DEFAULT = (
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
)
DROID_TARGET_VIEW_DEFAULT = "observation.images.wrist_left"

LEROBOT_SOURCE_VIEWS_DEFAULT = (
    "observation.images.left_external",
    "observation.images.right_external",
)
LEROBOT_TARGET_VIEW_DEFAULT = "observation.images.wrist"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a metadata-only cross-view dataset for two source views "
            "to one wrist-view generation."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="/data2/xuehao/datasets/droid_success_high_quality",
        help="Root of the source dataset.",
    )
    parser.add_argument(
        "--output-root",
        default="/data2/xuehao/datasets/droid_success_high_quality_crossview_meta",
        help="Output directory for derived metadata/stat/prompt embeddings.",
    )
    parser.add_argument(
        "--dataset-format",
        choices=("auto", "droid_episode", "lerobot_v3"),
        default="auto",
        help="Dataset layout. `auto` infers from meta/info.json and directory layout.",
    )
    parser.add_argument(
        "--train-source-jsonl",
        default=None,
        help="Optional train episodes jsonl for droid_episode datasets.",
    )
    parser.add_argument(
        "--val-source-jsonl",
        default=None,
        help="Optional val episodes jsonl for droid_episode datasets.",
    )
    parser.add_argument(
        "--max-episode-frames",
        type=int,
        default=None,
        help="Optional cap on raw episode length. Omit to keep all lengths.",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=81,
        help="Output clip length after tail padding.",
    )
    parser.add_argument(
        "--train-stride",
        type=int,
        default=81,
        help="Clip stride. `pad_repeat_last` requires this to equal clip_length.",
    )
    parser.add_argument(
        "--tail-policy",
        choices=("pad_repeat_last", "drop_tail"),
        default="pad_repeat_last",
        help="How to handle episode tails shorter than clip_length.",
    )
    parser.add_argument(
        "--filter-success-only",
        type=int,
        choices=[0, 1],
        default=1,
        help="Keep only successful episodes when success metadata exists.",
    )
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=None,
        help="Optional total candidate cap for dry runs.",
    )
    parser.add_argument(
        "--small-train-episodes",
        type=int,
        default=200,
        help="Number of train episodes to keep after split selection.",
    )
    parser.add_argument(
        "--small-val-episodes",
        type=int,
        default=15,
        help="Number of val episodes to keep after split selection.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Deterministic seed used to sample train/val episodes.",
    )
    parser.add_argument(
        "--source-views",
        default=",".join(DROID_SOURCE_VIEWS_DEFAULT),
        help="Comma-separated source camera keys.",
    )
    parser.add_argument(
        "--target-view",
        default=DROID_TARGET_VIEW_DEFAULT,
        help="Target camera key.",
    )
    parser.add_argument(
        "--build-prompt-emb",
        action="store_true",
        help="Build prompt embeddings into <output-root>/prompt_emb.",
    )
    parser.add_argument(
        "--prompt-emb-model-root",
        default="/home/xuehao/xh/projects/DiffSynth-Studio-old/models/PAI/Wan2.1-Fun-V1.1-1.3B-InP",
        help="WAN model root used by tool/build_prompt_embeddings.py.",
    )
    parser.add_argument(
        "--prompt-emb-device",
        default="cpu",
        help="Device for prompt embedding generation.",
    )
    parser.add_argument(
        "--prompt-emb-torch-dtype",
        default="bfloat16",
        help="Torch dtype for prompt embedding generation.",
    )
    parser.add_argument(
        "--prompt-emb-skip-existing",
        action="store_true",
        help="Skip existing prompt embeddings.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def detect_dataset_format(dataset_root: Path, info: dict, requested: str) -> str:
    if requested != "auto":
        return requested
    if (dataset_root / "meta" / "episodes").exists():
        return "lerobot_v3"
    if "{file_index" in str(info.get("data_path", "")):
        return "lerobot_v3"
    return "droid_episode"


def resolve_view_keys(dataset_format: str, source_views_arg: str, target_view_arg: str) -> tuple[list[str], str]:
    source_views = [item.strip() for item in source_views_arg.split(",") if item.strip()]
    target_view = target_view_arg.strip()
    if dataset_format == "lerobot_v3":
        if tuple(source_views) == DROID_SOURCE_VIEWS_DEFAULT:
            source_views = list(LEROBOT_SOURCE_VIEWS_DEFAULT)
        if target_view == DROID_TARGET_VIEW_DEFAULT:
            target_view = LEROBOT_TARGET_VIEW_DEFAULT
    if len(source_views) != 2:
        raise ValueError(f"Expected exactly two source views, got {source_views!r}.")
    return source_views, target_view


def maybe_limit_length(length: int, max_episode_frames: int | None) -> bool:
    if max_episode_frames is None:
        return False
    if int(max_episode_frames) <= 0:
        return False
    return int(length) > int(max_episode_frames)


def resolve_droid_paths(
    info: dict,
    dataset_root: Path,
    episode_index: int,
    source_views: list[str],
    target_view: str,
) -> tuple[list[str], str]:
    chunk_size = int(info.get("chunks_size", 1000))
    chunk_id = episode_index // chunk_size
    episode_name = f"episode_{episode_index:06d}.mp4"
    parquet_name = f"episode_{episode_index:06d}.parquet"
    chunk_folder = f"chunk-{chunk_id:03d}"
    videos = []
    for view in [*source_views, target_view]:
        videos.append(
            str((dataset_root / "videos" / chunk_folder / view / episode_name).resolve())
        )
    parquet_path = str((dataset_root / "data" / chunk_folder / parquet_name).resolve())
    return videos, parquet_path


def resolve_old_droid_prompt(row: dict, parquet_path: str) -> str:
    tasks = row.get("tasks", [])
    if isinstance(tasks, list):
        for task in tasks:
            prompt = normalize_text(task)
            if prompt:
                return prompt

    table = pq.read_table(
        parquet_path,
        columns=[
            "language_instruction",
            "language_instruction_2",
            "language_instruction_3",
        ],
    )
    for key in table.schema.names:
        values = table[key].to_pylist()
        if not values:
            continue
        prompt = normalize_text(values[0])
        if prompt:
            return prompt
    return ""


def read_droid_episode_success(parquet_path: str) -> bool:
    table = pq.read_table(parquet_path, columns=["is_episode_successful"])
    values = table["is_episode_successful"].to_pylist()
    if len(values) == 0:
        return False
    return bool(values[-1])


def validate_paths(video_paths: list[str], parquet_path: str) -> bool:
    if not os.path.exists(parquet_path):
        return False
    return all(os.path.exists(path) for path in video_paths)


def collect_droid_candidates(
    source_jsonl: Path,
    info: dict,
    dataset_root: Path,
    source_views: list[str],
    target_view: str,
    max_episode_frames: int | None,
    filter_success_only: bool,
    episode_limit: int | None,
) -> tuple[list[dict], dict[str, int]]:
    rows = []
    stats = defaultdict(int)
    for row in iter_jsonl(source_jsonl):
        if episode_limit is not None and len(rows) >= episode_limit:
            break
        stats["episodes_total"] += 1
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        if maybe_limit_length(length, max_episode_frames):
            stats["dropped_too_long"] += 1
            continue
        video_paths, parquet_path = resolve_droid_paths(
            info, dataset_root, episode_index, source_views, target_view
        )
        if not validate_paths(video_paths, parquet_path):
            stats["dropped_missing_files"] += 1
            continue
        if filter_success_only and not read_droid_episode_success(parquet_path):
            stats["dropped_failed_episode"] += 1
            continue
        prompt = resolve_old_droid_prompt(row, parquet_path)
        if not prompt:
            stats["dropped_empty_prompt"] += 1
            continue
        rows.append(
            {
                "episode_index": episode_index,
                "length": length,
                "prompt": prompt,
                "video_segments": [
                    {"data": path, "base_start_frame": 0} for path in video_paths
                ],
                "state_segment": {"data": parquet_path, "base_start_frame": 0},
            }
        )
        stats["episodes_kept"] += 1
    return rows, dict(stats)


def load_lerobot_episode_rows(dataset_root: Path, source_views: list[str], target_view: str) -> list[dict]:
    meta_root = dataset_root / "meta" / "episodes"
    if not meta_root.exists():
        raise FileNotFoundError(f"Missing LeRobot episode metadata directory: {meta_root}")

    requested_columns = [
        "episode_index",
        "length",
        "tasks",
        "task",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    for view in [*source_views, target_view]:
        requested_columns.extend(
            [
                f"videos/{view}/chunk_index",
                f"videos/{view}/file_index",
                f"videos/{view}/from_timestamp",
                f"videos/{view}/to_timestamp",
            ]
        )

    rows: list[dict] = []
    for parquet_path in sorted(meta_root.rglob("*.parquet")):
        schema = pq.read_schema(parquet_path)
        missing = [col for col in requested_columns if col not in schema.names]
        if missing:
            raise KeyError(
                f"Missing required LeRobot episode metadata columns in {parquet_path}: {missing}"
            )
        table = pq.read_table(parquet_path, columns=requested_columns)
        columns = {name: table[name].to_pylist() for name in requested_columns}
        for row_index in range(table.num_rows):
            rows.append({name: columns[name][row_index] for name in requested_columns})
    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows


def build_data_file_base_offsets(rows: list[dict]) -> dict[tuple[int, int], int]:
    offsets: dict[tuple[int, int], int] = {}
    for row in rows:
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        start = int(row["dataset_from_index"])
        previous = offsets.get(key)
        if previous is None or start < previous:
            offsets[key] = start
    return offsets


def timestamp_to_frame(timestamp: float | int | None, fps: float) -> int:
    if timestamp is None:
        return 0
    return int(round(float(timestamp) * float(fps)))


def format_lerobot_path(dataset_root: Path, template: str, **kwargs) -> str:
    return str((dataset_root / template.format(**kwargs)).resolve())


def resolve_lerobot_prompt(row: dict, parquet_path: str, local_start: int) -> str:
    tasks = row.get("tasks")
    if isinstance(tasks, list):
        for task in tasks:
            prompt = normalize_text(task)
            if prompt:
                return prompt
    task = row.get("task")
    if isinstance(task, str):
        for chunk in task.split("|"):
            prompt = normalize_text(chunk)
            if prompt:
                return prompt

    try:
        table = pq.read_table(
            parquet_path,
            columns=[
                "language_instruction_1",
                "language_instruction_2",
                "language_instruction_3",
            ],
        )
    except Exception:
        return ""
    index = max(0, int(local_start))
    for key in table.schema.names:
        values = table[key].to_pylist()
        if not values:
            continue
        prompt = normalize_text(values[min(index, len(values) - 1)])
        if prompt:
            return prompt
    return ""


def collect_lerobot_candidates(
    dataset_root: Path,
    info: dict,
    source_views: list[str],
    target_view: str,
    max_episode_frames: int | None,
    filter_success_only: bool,
    episode_limit: int | None,
) -> tuple[list[dict], dict[str, int]]:
    rows = load_lerobot_episode_rows(dataset_root, source_views, target_view)
    fps = float(info.get("fps", 15))
    data_path_template = str(info.get("data_path", "data/chunk-{chunk_index:03d}/file_{file_index:03d}.parquet"))
    video_path_template = str(
        info.get(
            "video_path",
            "videos/{video_key}/chunk-{chunk_index:03d}/file_{file_index:03d}.mp4",
        )
    )
    data_file_base_offsets = build_data_file_base_offsets(rows)
    stats = defaultdict(int)
    candidates: list[dict] = []

    if filter_success_only:
        stats["success_filter_not_needed"] += 1

    for row in rows:
        if episode_limit is not None and len(candidates) >= episode_limit:
            break
        stats["episodes_total"] += 1
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        if maybe_limit_length(length, max_episode_frames):
            stats["dropped_too_long"] += 1
            continue

        data_chunk_index = int(row["data/chunk_index"])
        data_file_index = int(row["data/file_index"])
        data_path = format_lerobot_path(
            dataset_root,
            data_path_template,
            chunk_index=data_chunk_index,
            file_index=data_file_index,
        )
        if not os.path.exists(data_path):
            stats["dropped_missing_files"] += 1
            continue

        data_offset_key = (data_chunk_index, data_file_index)
        data_base_offset = data_file_base_offsets[data_offset_key]
        dataset_from_index = int(row["dataset_from_index"])
        dataset_to_index = int(row["dataset_to_index"])
        inferred_length = dataset_to_index - dataset_from_index
        actual_length = int(length if inferred_length <= 0 else min(length, inferred_length))
        if actual_length <= 0:
            stats["dropped_empty_episode"] += 1
            continue
        state_base_start = dataset_from_index - data_base_offset

        prompt = resolve_lerobot_prompt(row, data_path, state_base_start)
        if not prompt:
            stats["dropped_empty_prompt"] += 1
            continue

        video_segments = []
        missing_view_file = False
        for view in [*source_views, target_view]:
            view_chunk_index = int(row[f"videos/{view}/chunk_index"])
            view_file_index = int(row[f"videos/{view}/file_index"])
            video_path = format_lerobot_path(
                dataset_root,
                video_path_template,
                video_key=view,
                chunk_index=view_chunk_index,
                file_index=view_file_index,
            )
            if not os.path.exists(video_path):
                missing_view_file = True
                break
            local_start = timestamp_to_frame(
                row[f"videos/{view}/from_timestamp"],
                fps=fps,
            )
            video_segments.append(
                {
                    "data": video_path,
                    "base_start_frame": local_start,
                }
            )
        if missing_view_file:
            stats["dropped_missing_files"] += 1
            continue

        candidates.append(
            {
                "episode_index": episode_index,
                "length": actual_length,
                "prompt": prompt,
                "video_segments": video_segments,
                "state_segment": {
                    "data": data_path,
                    "base_start_frame": state_base_start,
                },
            }
        )
        stats["episodes_kept"] += 1

    return candidates, dict(stats)


def select_lerobot_splits(
    candidates: list[dict],
    split_seed: int,
    small_train_episodes: int,
    small_val_episodes: int,
) -> tuple[list[dict], list[dict]]:
    shuffled = list(candidates)
    random.Random(split_seed).shuffle(shuffled)
    required = int(small_train_episodes) + int(small_val_episodes)
    if len(shuffled) < required:
        raise ValueError(
            f"Not enough candidate episodes: need {required}, found {len(shuffled)}."
        )
    val_candidates = shuffled[: int(small_val_episodes)]
    train_candidates = shuffled[
        int(small_val_episodes) : int(small_val_episodes) + int(small_train_episodes)
    ]
    return train_candidates, val_candidates


def build_clip_starts(length: int, clip_length: int, stride: int, tail_policy: str) -> list[int]:
    if length <= 0:
        return []
    if tail_policy == "pad_repeat_last":
        if stride != clip_length:
            raise ValueError(
                "`pad_repeat_last` currently requires `train_stride == clip_length`."
            )
        return list(range(0, length, stride))
    if length < clip_length:
        return []
    return list(range(0, length - clip_length + 1, stride))


def build_clip_rows(
    rows: list[dict],
    split_name: str,
    clip_length: int,
    stride: int,
    tail_policy: str,
) -> tuple[list[dict], OrderedDict[str, int]]:
    clip_rows: list[dict] = []
    prompt_to_index: OrderedDict[str, int] = OrderedDict()
    for row in rows:
        length = int(row["length"])
        starts = build_clip_starts(length, clip_length, stride, tail_policy)
        for start in starts:
            valid_frames = min(int(clip_length), max(0, length - int(start)))
            if valid_frames <= 0:
                continue
            end = int(start) + valid_frames - 1
            prompt = row["prompt"]
            if prompt not in prompt_to_index:
                prompt_to_index[prompt] = len(prompt_to_index)
            task_index = prompt_to_index[prompt]

            video_payload = []
            for segment in row["video_segments"]:
                base_start = int(segment["base_start_frame"])
                video_payload.append(
                    {
                        "data": segment["data"],
                        "start_frame": base_start + int(start),
                        "end_frame": base_start + int(end),
                        "pad_to_frames": int(clip_length),
                        "pad_mode": "repeat_last",
                    }
                )

            state_base_start = int(row["state_segment"]["base_start_frame"])
            clip_rows.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "length": int(clip_length),
                    "valid_frames": int(valid_frames),
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "video": video_payload,
                    "state": {
                        "data": row["state_segment"]["data"],
                        "start_frame": state_base_start + int(start),
                        "end_frame": state_base_start + int(end),
                        "pad_to_frames": int(clip_length),
                        "pad_mode": "repeat_last",
                    },
                    "prompt": prompt,
                    "task": prompt,
                    "task_index": task_index,
                    "source_views": [0, 1],
                    "target_view": 2,
                    "data_type": split_name,
                    "state_type": "state_pose_7d",
                }
            )
    return clip_rows, prompt_to_index


def merge_prompt_indices(*mappings: OrderedDict[str, int]) -> OrderedDict[str, int]:
    merged: OrderedDict[str, int] = OrderedDict()
    for mapping in mappings:
        for prompt in mapping.keys():
            if prompt not in merged:
                merged[prompt] = len(merged)
    return merged


def rewrite_task_indices(rows: list[dict], merged_mapping: OrderedDict[str, int]) -> None:
    for row in rows:
        row["task_index"] = merged_mapping[row["prompt"]]


def build_tasks_rows(prompt_mapping: OrderedDict[str, int]) -> list[dict]:
    tasks = []
    for prompt, task_index in prompt_mapping.items():
        tasks.append(
            {
                "task_index": task_index,
                "task": prompt,
                "prompt": prompt,
            }
        )
    return tasks


def quaternion_wxyz_to_euler_xyz(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion_xyzw = np.concatenate(
        [quaternion_wxyz[:, 1:], quaternion_wxyz[:, :1]],
        axis=1,
    )
    return Rotation.from_quat(quaternion_xyzw).as_euler("xyz", degrees=False).astype(
        np.float32
    )


def load_state_pose_7d_from_droid_columns(table: pq.Table) -> np.ndarray:
    cartesian = np.asarray(
        table["observation.state.cartesian_position"].to_pylist(),
        dtype=np.float32,
    )
    gripper = np.asarray(
        table["observation.state.gripper_position"].to_pylist(),
        dtype=np.float32,
    ).reshape(-1, 1)
    return np.concatenate([cartesian, gripper], axis=1)


def load_state_pose_7d_from_lerobot_columns(table: pq.Table) -> np.ndarray:
    cartesian = np.asarray(
        table["observation.cartesian_position"].to_pylist(),
        dtype=np.float32,
    )
    if cartesian.ndim != 2 or cartesian.shape[1] != 7:
        raise ValueError(f"Unexpected observation.cartesian_position shape: {cartesian.shape}")
    euler = quaternion_wxyz_to_euler_xyz(cartesian[:, 3:7])
    gripper = np.asarray(
        table["observation.gripper_position"].to_pylist(),
        dtype=np.float32,
    ).reshape(-1, 1)
    return np.concatenate([cartesian[:, :3], euler, gripper], axis=1)


def load_state_pose_7d_from_observation_state(table: pq.Table) -> np.ndarray:
    arr = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 26:
        raise ValueError(f"Unexpected observation.state shape: {arr.shape}")
    indices = [19, 20, 21, 22, 23, 24, 25]
    return arr[:, indices]


def load_state_pose_7d_full(parquet_path: str) -> np.ndarray:
    try:
        table = pq.read_table(
            parquet_path,
            columns=[
                "observation.state.cartesian_position",
                "observation.state.gripper_position",
            ],
        )
        arr = load_state_pose_7d_from_droid_columns(table)
    except Exception:
        try:
            table = pq.read_table(
                parquet_path,
                columns=[
                    "observation.cartesian_position",
                    "observation.gripper_position",
                ],
            )
            arr = load_state_pose_7d_from_lerobot_columns(table)
        except Exception:
            table = pq.read_table(parquet_path, columns=["observation.state"])
            arr = load_state_pose_7d_from_observation_state(table)
    return arr


def load_state_pose_7d(parquet_path: str, start_frame: int, end_frame: int) -> np.ndarray:
    start = int(start_frame)
    end = int(end_frame) + 1
    arr = load_state_pose_7d_full(parquet_path)
    if end > arr.shape[0]:
        raise ValueError(
            f"Not enough rows in {parquet_path} for slice start={start_frame}, end={end_frame}."
        )
    return arr[start:end]


def summarize_array(arr: np.ndarray) -> dict:
    return {
        "shape": [int(arr.shape[1])],
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
        "p01": np.percentile(arr, 1, axis=0).tolist(),
        "p99": np.percentile(arr, 99, axis=0).tolist(),
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
    }


def compute_state_stats(train_rows: list[dict]) -> dict:
    grouped_segments: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in train_rows:
        state_payload = row["state"]
        grouped_segments[str(state_payload["data"])].append(
            (int(state_payload["start_frame"]), int(state_payload["end_frame"]))
        )
    chunks = []
    for parquet_path, segments in grouped_segments.items():
        arr = load_state_pose_7d_full(parquet_path)
        for start_frame, end_frame in segments:
            end_exclusive = int(end_frame) + 1
            if end_exclusive > arr.shape[0]:
                raise ValueError(
                    f"Not enough rows in {parquet_path} for slice start={start_frame}, end={end_frame}."
                )
            chunks.append(arr[int(start_frame) : end_exclusive])
    if len(chunks) == 0:
        raise ValueError("No train state slices collected for statistics.")
    state_pose = np.concatenate(chunks, axis=0)
    return {"state_pose_7d": summarize_array(state_pose)}


def build_prompt_embeddings(
    repo_root: Path,
    output_root: Path,
    model_root: str,
    device: str,
    torch_dtype: str,
    skip_existing: bool,
) -> None:
    tasks_jsonl = output_root / "meta" / "tasks_cross_view.jsonl"
    prompt_dir = output_root / "prompt_emb"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    build_script = repo_root / "tool" / "build_prompt_embeddings.py"
    if not build_script.exists():
        raise FileNotFoundError(f"Missing script: {build_script}")

    pos_cmd = [
        sys.executable,
        str(build_script),
        "--mode",
        "pos",
        "--pos-jsonl",
        str(tasks_jsonl),
        "--pos-output",
        str(prompt_dir),
        "--model-root",
        model_root,
        "--device",
        device,
        "--torch-dtype",
        torch_dtype,
    ]
    if skip_existing:
        pos_cmd.append("--skip-existing")
    subprocess.run(pos_cmd, check=True)

    neg_cmd = [
        sys.executable,
        str(build_script),
        "--mode",
        "neg",
        "--neg-output",
        str(prompt_dir / "neg_prompt.pt"),
        "--model-root",
        model_root,
        "--device",
        device,
        "--torch-dtype",
        torch_dtype,
    ]
    subprocess.run(neg_cmd, check=True)


def inject_prompt_emb(rows: list[dict], prompt_mapping: OrderedDict[str, int]) -> None:
    for row in rows:
        task_index = prompt_mapping[row["prompt"]]
        row["prompt_emb"] = f"prompt_emb/pos_{task_index}.pt"


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    meta_root = output_root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    info = load_json(dataset_root / "meta" / "info.json")
    dataset_format = detect_dataset_format(dataset_root, info, args.dataset_format)
    source_views, target_view = resolve_view_keys(
        dataset_format,
        args.source_views,
        args.target_view,
    )

    if args.tail_policy == "pad_repeat_last" and int(args.train_stride) != int(args.clip_length):
        raise ValueError("`pad_repeat_last` requires `--train-stride` to equal `--clip-length`.")

    if dataset_format == "lerobot_v3":
        candidates, candidate_stats = collect_lerobot_candidates(
            dataset_root=dataset_root,
            info=info,
            source_views=source_views,
            target_view=target_view,
            max_episode_frames=args.max_episode_frames,
            filter_success_only=bool(args.filter_success_only),
            episode_limit=args.episode_limit,
        )
        train_candidates, val_candidates = select_lerobot_splits(
            candidates,
            split_seed=int(args.split_seed),
            small_train_episodes=int(args.small_train_episodes),
            small_val_episodes=int(args.small_val_episodes),
        )
        source_stats = {
            "candidate_stats": candidate_stats,
            "selected_train_episodes": len(train_candidates),
            "selected_val_episodes": len(val_candidates),
        }
    else:
        train_source_jsonl = (
            Path(args.train_source_jsonl).resolve()
            if args.train_source_jsonl is not None
            else (dataset_root / "meta" / "episodes_train.jsonl")
        )
        val_source_jsonl = (
            Path(args.val_source_jsonl).resolve()
            if args.val_source_jsonl is not None
            else (dataset_root / "meta" / "episodes_val.jsonl")
        )
        train_candidates, train_stats = collect_droid_candidates(
            source_jsonl=train_source_jsonl,
            info=info,
            dataset_root=dataset_root,
            source_views=source_views,
            target_view=target_view,
            max_episode_frames=args.max_episode_frames,
            filter_success_only=bool(args.filter_success_only),
            episode_limit=args.small_train_episodes,
        )
        val_candidates, val_stats = collect_droid_candidates(
            source_jsonl=val_source_jsonl,
            info=info,
            dataset_root=dataset_root,
            source_views=source_views,
            target_view=target_view,
            max_episode_frames=args.max_episode_frames,
            filter_success_only=bool(args.filter_success_only),
            episode_limit=args.small_val_episodes,
        )
        source_stats = {
            "train_source_stats": train_stats,
            "val_source_stats": val_stats,
            "selected_train_episodes": len(train_candidates),
            "selected_val_episodes": len(val_candidates),
        }

    train_rows, train_prompts = build_clip_rows(
        rows=train_candidates,
        split_name="train",
        clip_length=int(args.clip_length),
        stride=int(args.train_stride),
        tail_policy=args.tail_policy,
    )
    val_rows, val_prompts = build_clip_rows(
        rows=val_candidates,
        split_name="val",
        clip_length=int(args.clip_length),
        stride=int(args.train_stride),
        tail_policy=args.tail_policy,
    )
    merged_prompts = merge_prompt_indices(train_prompts, val_prompts)
    rewrite_task_indices(train_rows, merged_prompts)
    rewrite_task_indices(val_rows, merged_prompts)
    tasks_rows = build_tasks_rows(merged_prompts)

    train_jsonl = (
        meta_root
        / f"episodes_cross_view_train_{int(args.clip_length)}_small{int(args.small_train_episodes)}.jsonl"
    )
    val_jsonl = (
        meta_root
        / f"episodes_cross_view_val_{int(args.clip_length)}_small{int(args.small_val_episodes)}.jsonl"
    )
    tasks_jsonl = meta_root / "tasks_cross_view.jsonl"
    dump_jsonl(train_jsonl, train_rows)
    dump_jsonl(val_jsonl, val_rows)
    dump_jsonl(tasks_jsonl, tasks_rows)

    stats = compute_state_stats(train_rows)
    stat_json = meta_root / "stat_state_pose_7d.json"
    write_summary(stat_json, stats)

    if args.build_prompt_emb:
        repo_root = Path(__file__).resolve().parent.parent
        build_prompt_embeddings(
            repo_root=repo_root,
            output_root=output_root,
            model_root=args.prompt_emb_model_root,
            device=args.prompt_emb_device,
            torch_dtype=args.prompt_emb_torch_dtype,
            skip_existing=args.prompt_emb_skip_existing,
        )
        inject_prompt_emb(train_rows, merged_prompts)
        inject_prompt_emb(val_rows, merged_prompts)
        dump_jsonl(train_jsonl, train_rows)
        dump_jsonl(val_jsonl, val_rows)

    summary = {
        "dataset_format": dataset_format,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "source_views": source_views,
        "target_view": target_view,
        "clip_length": int(args.clip_length),
        "train_stride": int(args.train_stride),
        "tail_policy": args.tail_policy,
        "max_episode_frames": (
            None if args.max_episode_frames is None else int(args.max_episode_frames)
        ),
        "filter_success_only": bool(args.filter_success_only),
        "split_seed": int(args.split_seed),
        "small_train_episodes": int(args.small_train_episodes),
        "small_val_episodes": int(args.small_val_episodes),
        "source_stats": source_stats,
        "train_clip_count": len(train_rows),
        "val_clip_count": len(val_rows),
        "task_count": len(tasks_rows),
        "state_stat_path": str(stat_json),
        "state_stat_source": "train_only_real_frames",
        "train_manifest": str(train_jsonl),
        "val_manifest": str(val_jsonl),
        "tasks_manifest": str(tasks_jsonl),
    }
    write_summary(meta_root / "summary_cross_view.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
