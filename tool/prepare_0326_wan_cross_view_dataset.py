#!/usr/bin/env python3
import argparse
import json
import shutil
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


RIGHT_POSE7_INDICES = [19, 20, 21, 22, 23, 24, 25]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a metadata-only three-view cross-view dataset from the "
            "0326_wan robot dataset by duplicating the third-person view."
        )
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="/data1/linzengrong/Code/DiffSynth-Studio/robot_data/0326_wan",
        help="Source dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/data1/xuehao/datasets/0326_wan_crossview_meta",
        help="Output derived dataset root.",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=81,
        help="Fixed clip length for train/val manifests.",
    )
    parser.add_argument(
        "--train-stride",
        type=int,
        default=32,
        help="Sliding-window stride for training clips.",
    )
    parser.add_argument(
        "--val-episodes",
        type=int,
        default=9,
        help="Number of held-out episodes for validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before writing.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt_mapping(rows: list[dict]) -> OrderedDict[str, int]:
    mapping: OrderedDict[str, int] = OrderedDict()
    for row in rows:
        prompt = str(row["prompt"])
        if prompt not in mapping:
            mapping[prompt] = len(mapping)
    return mapping


def build_tasks_rows(prompt_mapping: OrderedDict[str, int]) -> list[dict]:
    return [
        {"task_index": int(task_index), "task": prompt, "prompt": prompt}
        for prompt, task_index in prompt_mapping.items()
    ]


def build_video_triplet(video_paths: list[str], input_root: Path) -> list[str]:
    if len(video_paths) != 2:
        raise ValueError(f"Expected exactly 2 source videos, got {len(video_paths)}")
    third_person = input_root / video_paths[0]
    first_person = input_root / video_paths[1]
    if not third_person.is_file():
        raise FileNotFoundError(f"Missing third-person video: {third_person}")
    if not first_person.is_file():
        raise FileNotFoundError(f"Missing first-person video: {first_person}")
    return [
        str(third_person.resolve()),
        str(third_person.resolve()),
        str(first_person.resolve()),
    ]


def build_state_pose7_slice(parquet_path: Path, start_frame: int, end_frame: int) -> np.ndarray:
    table = pq.read_table(str(parquet_path), columns=["observation.state"])
    rows = table["observation.state"].to_pylist()
    arr = np.asarray(rows[start_frame : end_frame + 1], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < max(RIGHT_POSE7_INDICES) + 1:
        raise ValueError(
            f"Unexpected observation.state shape in {parquet_path}: {arr.shape}"
        )
    return arr[:, RIGHT_POSE7_INDICES]


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


def compute_state_stats(rows: list[dict]) -> dict:
    grouped_segments: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        grouped_segments[row["state"]].append(
            (int(row["start_frame"]), int(row["end_frame"]))
        )

    chunks = []
    for parquet_path, segments in grouped_segments.items():
        parquet = Path(parquet_path)
        table = pq.read_table(str(parquet), columns=["observation.state"])
        all_rows = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        if all_rows.ndim != 2 or all_rows.shape[1] < max(RIGHT_POSE7_INDICES) + 1:
            raise ValueError(
                f"Unexpected observation.state shape in {parquet}: {all_rows.shape}"
            )
        pose7 = all_rows[:, RIGHT_POSE7_INDICES]
        for start_frame, end_frame in segments:
            chunks.append(pose7[start_frame : end_frame + 1])

    state_pose = np.concatenate(chunks, axis=0)
    return {"state_pose_7d": summarize_array(state_pose)}


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_train_rows(
    base_rows: list[dict],
    input_root: Path,
    prompt_mapping: OrderedDict[str, int],
    clip_length: int,
    train_stride: int,
) -> list[dict]:
    rows: list[dict] = []
    for row in base_rows:
        episode_length = int(row["length"])
        if episode_length < clip_length:
            continue
        window_start = int(row["start_frame"])
        window_end = int(row["end_frame"])
        last_start = window_start + (episode_length - clip_length)
        for start_frame in range(window_start, last_start + 1, train_stride):
            end_frame = start_frame + clip_length - 1
            rows.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "length": clip_length,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "video": build_video_triplet(row["video"], input_root),
                    "state": str((input_root / row["action"]).resolve()),
                    "prompt": row["prompt"],
                    "task": row["prompt"],
                    "task_index": int(prompt_mapping[row["prompt"]]),
                    "source_views": [0, 1],
                    "target_view": 2,
                    "data_type": "train",
                    "state_type": "state_pose_7d",
                    "prompt_emb": row["prompt_emb"],
                }
            )
    return rows


def build_val_rows(
    base_rows: list[dict],
    input_root: Path,
    prompt_mapping: OrderedDict[str, int],
    clip_length: int,
) -> list[dict]:
    rows: list[dict] = []
    for row in base_rows:
        episode_length = int(row["length"])
        if episode_length < clip_length:
            continue
        window_start = int(row["start_frame"])
        centered_offset = max(0, (episode_length - clip_length) // 2)
        start_frame = window_start + centered_offset
        end_frame = start_frame + clip_length - 1
        rows.append(
            {
                "episode_index": int(row["episode_index"]),
                "length": clip_length,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "video": build_video_triplet(row["video"], input_root),
                "state": str((input_root / row["action"]).resolve()),
                "prompt": row["prompt"],
                "task": row["prompt"],
                "task_index": int(prompt_mapping[row["prompt"]]),
                "source_views": [0, 1],
                "target_view": 2,
                "data_type": "val",
                "state_type": "state_pose_7d",
                "prompt_emb": row["prompt_emb"],
            }
        )
    return rows


def copy_prompt_embeddings(input_root: Path, output_root: Path, rows: list[dict]) -> None:
    prompt_dir = output_root / "prompt_emb"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    needed = {row["prompt_emb"] for row in rows}
    needed.add("prompt_emb/neg_prompt.pt")
    for rel_path in sorted(needed):
        src = input_root / rel_path
        dst = output_root / rel_path
        if not src.is_file():
            raise FileNotFoundError(f"Missing prompt embedding: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. "
                "Pass --overwrite to recreate it."
            )
        shutil.rmtree(output_root)

    source_rows = load_jsonl(input_root / "meta" / "episodes.jsonl")
    source_rows = sorted(source_rows, key=lambda row: int(row["episode_index"]))
    if len(source_rows) <= args.val_episodes:
        raise ValueError(
            f"Not enough episodes ({len(source_rows)}) for val_episodes={args.val_episodes}"
        )

    val_base_rows = source_rows[-args.val_episodes :]
    train_base_rows = source_rows[: -args.val_episodes]
    prompt_mapping = build_prompt_mapping(source_rows)

    train_rows = build_train_rows(
        train_base_rows,
        input_root=input_root,
        prompt_mapping=prompt_mapping,
        clip_length=args.clip_length,
        train_stride=args.train_stride,
    )
    val_rows = build_val_rows(
        val_base_rows,
        input_root=input_root,
        prompt_mapping=prompt_mapping,
        clip_length=args.clip_length,
    )

    all_rows = train_rows + val_rows
    if not all_rows:
        raise ValueError("No cross-view rows were generated.")

    copy_prompt_embeddings(input_root, output_root, all_rows)
    tasks_rows = build_tasks_rows(prompt_mapping)
    stats = compute_state_stats(all_rows)

    meta_root = output_root / "meta"
    dump_jsonl(meta_root / "episodes_cross_view_train_81.jsonl", train_rows)
    dump_jsonl(meta_root / "episodes_cross_view_val_81.jsonl", val_rows)
    dump_jsonl(meta_root / "tasks_cross_view.jsonl", tasks_rows)
    write_summary(meta_root / "stat_state_pose_7d.json", stats)
    write_summary(
        meta_root / "summary_cross_view.json",
        {
            "input_root": str(input_root),
            "output_root": str(output_root),
            "clip_length": int(args.clip_length),
            "train_stride": int(args.train_stride),
            "val_episodes": int(args.val_episodes),
            "source_episode_count": len(source_rows),
            "train_episode_count": len(train_base_rows),
            "val_episode_count": len(val_base_rows),
            "train_clip_count": len(train_rows),
            "val_clip_count": len(val_rows),
            "task_count": len(tasks_rows),
            "source_views": [0, 1],
            "target_view": 2,
        },
    )

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "train_clip_count": len(train_rows),
                "val_clip_count": len(val_rows),
                "train_episode_indices": [
                    int(row["episode_index"]) for row in train_base_rows[:5]
                ],
                "val_episode_indices": [
                    int(row["episode_index"]) for row in val_base_rows
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
