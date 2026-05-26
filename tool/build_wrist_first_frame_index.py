#!/usr/bin/env python
"""
Build an index mapping (episode_index, start_frame) → wrist_first_frame PNG path.

核心逻辑：meta 按 (episode_index, start_frame) 排序后，与首帧目录的字典序排序一一对应。
"""
import argparse
import json
import os
import re
from pathlib import Path


DIR_PATTERN = re.compile(r"^episode_(\d+)_clipstart_(\d+)_.*_frame_(\d+)$")


def find_pred_image(dir_path: Path) -> str | None:
    for name in sorted(os.listdir(dir_path)):
        if name.endswith("_pred.png"):
            return str(dir_path / name)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_manifest", required=True)
    parser.add_argument("--first_frame_root", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    meta_records = []
    with open(args.meta_manifest) as f:
        for line in f:
            meta_records.append(json.loads(line))

    root = Path(args.first_frame_root)
    dirs = sorted(d for d in os.listdir(root) if d.startswith("episode_"))

    print(f"Meta records: {len(meta_records)}")
    print(f"FirstFrame dirs: {len(dirs)}")

    # 按 (episode_index, start_frame) 排序 meta，建立排序后索引 → 原始索引的映射
    meta_sorted = sorted(
        range(len(meta_records)),
        key=lambda i: (meta_records[i]["episode_index"], meta_records[i]["start_frame"]),
    )

    if len(dirs) != len(meta_sorted):
        print(f"[WARN] Count mismatch: dirs={len(dirs)}, meta={len(meta_sorted)}")

    index = {}
    hits, misses = 0, 0
    for rank, orig_idx in enumerate(meta_sorted):
        if rank >= len(dirs):
            misses += 1
            continue
        rec = meta_records[orig_idx]
        dir_path = root / dirs[rank]
        pred_path = find_pred_image(dir_path)
        if pred_path is None:
            misses += 1
            continue
        key = f"{rec['episode_index']}_{rec['start_frame']}"
        index[key] = pred_path
        hits += 1

    print(f"Hit rate: {hits}/{hits + misses}")
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(index, f)
    print(f"Saved index ({len(index)} entries) to {args.output_json}")


if __name__ == "__main__":
    main()
