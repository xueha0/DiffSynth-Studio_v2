"""增量构建 cross-view stage2 cache 的尾帧锚定字段 (target_tail_latents)。

设计要点
========
1. 每个 cache 文件 ``train/{idx:07d}.pth`` / ``val/{idx:07d}.pth`` 与 jsonl
   manifest 的第 idx 行一一对应。本脚本使用 manifest 提供的
   ``(episode_index, start_frame)`` 在同 episode 内寻找下一段。

2. 数据集 stride=81 严格成立 (已统计验证), 段 i+1 的 ``target_history_latents``
   即 LagerNVS(source@frame=a+81), 与段 i 期望的尾帧锚 LagerNVS(source@a+80)
   仅相差 1 帧 (~33ms). NVS 合成误差量级 ~20dB PSNR 的扰动远大于这 1 帧位移,
   因此**直接复用下一段的 head latent 作为本段的 tail latent**.

3. 末段 (episode 最后一段) 没有"下一段"可借, 由
   ``--tail-placeholder-mode`` 控制:
     * ``zero``        : 全零 tensor (默认; 模型训练时同时学习 "无尾锚" 与
                         "有尾锚" 两种模式, 推理可灵活降级)
     * ``repeat-head`` : 复制 head latent (尾锚永远 = 头锚)
     * ``gt-tail``     : 用 latent_views_gt 的最后一个 latent timestep
                         (引入训练时 GT 信息泄漏, 不推荐)

4. 改写策略
     * 默认 dry-run, 仅扫描并打印计划; ``--apply`` 才真正写盘.
     * 写盘使用 tmp file + rename, 中途崩溃不会留下半写文件.
     * 已存在 ``target_tail_latents`` 的 cache 默认跳过, ``--force`` 覆盖.
     * 失败列表写到 ``--failed-log`` 供后续重试.

5. 多进程加速
     * ``--num-workers``: torch.multiprocessing 启动 N 进程
     * 每个进程负责 idx % num_workers == worker_id 的样本
     * 主进程汇总进度并写 failed.json

用法
====
::

    # 1. 先 dry-run 验证逻辑 (扫描所有样本但不写盘)
    /env/conda/envs/studio/bin/python tool/build_cross_view_tail_cache.py \\
        --cache-root /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/cache_crossview_81f_180x320_lagernvs_iter060001_new \\
        --train-manifest /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/episodes_cross_view_train_81_small16567.jsonl \\
        --val-manifest /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/episodes_cross_view_val_81_small200.jsonl \\
        --tail-placeholder-mode zero \\
        --num-workers 8

    # 2. 确认无误后加 --apply 真正改写
    /env/conda/envs/studio/bin/python tool/build_cross_view_tail_cache.py \\
        --cache-root .../cache_crossview_81f_180x320_lagernvs_iter060001_new \\
        --train-manifest .../episodes_cross_view_train_81_small16567.jsonl \\
        --val-manifest .../episodes_cross_view_val_81_small200.jsonl \\
        --tail-placeholder-mode zero \\
        --num-workers 8 \\
        --apply
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.multiprocessing as mp

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cross_view_tail_helpers import (  # noqa: E402
    SegmentIndex,
    atomic_torch_save,
    make_tail_placeholder,
    validate_tail_shape,
)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--cache-root", type=str, required=True,
                   help="Root dir containing train/ val/ subdirs of .pth caches")
    p.add_argument("--train-manifest", type=str, required=True,
                   help="JSONL manifest aligned with cache-root/train/{idx:07d}.pth")
    p.add_argument("--val-manifest", type=str, default=None,
                   help="Optional val manifest aligned with cache-root/val/")
    p.add_argument("--target-view", type=int, default=2,
                   help="Target (wrist) view index. Default=2 matches DROID setup.")
    p.add_argument("--expected-stride", type=int, default=81,
                   help="Frames per segment. Must match dataset segmentation stride.")
    p.add_argument("--tail-placeholder-mode", type=str, default="zero",
                   choices=("zero", "repeat-head", "gt-tail"),
                   help="How to fill tail anchor for the LAST segment of each episode.")
    p.add_argument("--apply", action="store_true",
                   help="Actually write files. Without this flag the script is read-only (dry-run).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite even if target_tail_latents already exists in the cache.")
    p.add_argument("--num-workers", type=int, default=1,
                   help="Number of parallel processes. Each worker handles idx %% N == wid samples.")
    p.add_argument("--failed-log", type=str, default=None,
                   help="Where to write failed sample list. Default: <cache-root>/tail_cache_failed.json")
    p.add_argument("--limit", type=int, default=-1,
                   help="Process only the first N samples per split (for quick testing).")
    p.add_argument("--splits", type=str, default="train,val",
                   help="Comma list of splits to process. Empty -> nothing.")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Per-sample worker
# ----------------------------------------------------------------------------

def process_one_cache(
    cache_path: Path,
    next_cache_path: Optional[Path],
    target_view: int,
    tail_placeholder_mode: str,
    apply: bool,
    force: bool,
) -> Tuple[str, str]:
    """Returns (status, msg) where status in {"updated","skipped","placeholder","failed","dryrun"}."""
    sample = _torch_load_with_retry(cache_path)
    if "target_tail_latents" in sample and not force:
        return ("skipped", "already-has-target_tail_latents")

    head = sample.get("target_history_latents")
    if head is None:
        return ("failed", "missing-target_history_latents")

    if next_cache_path is not None:
        # Borrow next segment's head latent as this segment's tail latent.
        # Use retry-load because under multi-process apply mode the next file
        # might be in the middle of an atomic rename when we read it.
        next_sample = _torch_load_with_retry(next_cache_path)
        next_head = next_sample.get("target_history_latents")
        if next_head is None:
            return ("failed", f"next-segment-missing-target_history_latents:{next_cache_path.name}")
        tail = next_head.clone().to(dtype=head.dtype)
        validate_tail_shape(tail, head, str(cache_path))
        provenance = f"borrowed-from:{next_cache_path.name}"
    else:
        # Last segment of the episode -> placeholder.
        latent_views_gt = sample.get("latent_views_gt")
        if latent_views_gt is None and tail_placeholder_mode == "gt-tail":
            return ("failed", "missing-latent_views_gt-for-gt-tail-mode")
        tail = make_tail_placeholder(
            target_history_latents=head,
            latent_views_gt=latent_views_gt
            if latent_views_gt is not None
            else torch.zeros_like(head).expand(target_view + 1, *head.shape[1:]),
            target_view=target_view,
            mode=tail_placeholder_mode,
        )
        validate_tail_shape(tail, head, str(cache_path))
        provenance = f"placeholder:{tail_placeholder_mode}"

    if not apply:
        return ("dryrun", provenance)

    sample["target_tail_latents"] = tail
    # Provenance metadata for debugging/audit.
    meta = sample.get("__tail_cache_meta__", {})
    if not isinstance(meta, dict):
        meta = {}
    meta["provenance"] = provenance
    meta["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    sample["__tail_cache_meta__"] = meta

    atomic_torch_save(sample, cache_path)
    if next_cache_path is not None:
        return ("updated", provenance)
    return ("placeholder", provenance)


def _torch_load_with_retry(
    path: Path, max_retries: int = 5, base_sleep: float = 0.05
) -> dict:
    """Load .pth with brief retries to tolerate the short window where a
    concurrent worker is performing tmp+rename atomic write on this exact
    file. The rename itself is atomic, but file-existence checks issued
    *during* tmp removal can transiently fail.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        if path.exists():
            try:
                return torch.load(str(path), map_location="cpu", weights_only=False)
            except (FileNotFoundError, OSError, RuntimeError) as exc:
                last_exc = exc
        else:
            last_exc = FileNotFoundError(f"{path} not found")
        time.sleep(base_sleep * (2 ** attempt))
    raise last_exc if last_exc is not None else RuntimeError(f"failed to load {path}")


# ----------------------------------------------------------------------------
# Worker process: handles a stride of samples
# ----------------------------------------------------------------------------

def worker_main(
    worker_id: int,
    num_workers: int,
    split_dir: Path,
    seg_index_state: dict,
    target_view: int,
    tail_placeholder_mode: str,
    apply: bool,
    force: bool,
    expected_stride: int,
    limit: int,
    out_queue,
) -> None:
    seg_index = SegmentIndex(
        pair_to_idx=seg_index_state["pair_to_idx"],
        idx_to_pair=seg_index_state["idx_to_pair"],
        by_episode=seg_index_state["by_episode"],
    )
    total = len(seg_index)
    if limit > 0:
        total = min(total, limit)
    # Block-partition the index range across workers (instead of strided).
    # Reason: relevant `next` files are at idx+1 (same episode), so block
    # partitioning keeps both the read-target and next-target within the
    # same worker process, eliminating cross-worker rename races.
    block_size = (total + num_workers - 1) // num_workers
    start = worker_id * block_size
    end = min(total, start + block_size)
    processed = 0
    for idx in range(start, end):
        cache_path = split_dir / f"{idx:07d}.pth"
        if not cache_path.exists():
            out_queue.put(("failed", idx, f"cache-file-missing:{cache_path}"))
            continue
        next_idx = seg_index.next_segment_idx(idx, expected_stride=expected_stride)
        next_path = split_dir / f"{next_idx:07d}.pth" if next_idx is not None else None
        if next_path is not None and not next_path.exists():
            out_queue.put(("failed", idx, f"next-cache-missing:{next_path}"))
            continue
        try:
            status, msg = process_one_cache(
                cache_path=cache_path,
                next_cache_path=next_path,
                target_view=target_view,
                tail_placeholder_mode=tail_placeholder_mode,
                apply=apply,
                force=force,
            )
            out_queue.put((status, idx, msg))
        except Exception as exc:
            tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            out_queue.put(("failed", idx, f"exception:{tb}"))
        processed += 1
        if processed % 200 == 0:
            out_queue.put(("__heartbeat__", worker_id, processed))
    out_queue.put(("__done__", worker_id, processed))


# ----------------------------------------------------------------------------
# Per-split orchestrator
# ----------------------------------------------------------------------------

def run_split(
    split_name: str,
    split_dir: Path,
    manifest_path: Path,
    args: argparse.Namespace,
) -> dict:
    print(f"\n=== Processing split: {split_name} ===")
    print(f"  cache_dir : {split_dir}")
    print(f"  manifest  : {manifest_path}")
    if not split_dir.is_dir():
        print(f"  [skip] split dir does not exist: {split_dir}")
        return {"split": split_name, "skipped": True}

    seg_index = SegmentIndex.from_manifest(manifest_path)
    n_samples = len(seg_index)
    n_episodes = len(seg_index.by_episode)
    n_last_segments = n_episodes
    n_borrow = n_samples - n_last_segments
    print(f"  manifest rows  : {n_samples}")
    print(f"  episodes       : {n_episodes}")
    print(f"  borrow tail    : {n_borrow} ({100*n_borrow/n_samples:.1f}%)")
    print(f"  placeholder    : {n_last_segments} ({100*n_last_segments/n_samples:.1f}%)")

    if args.limit > 0:
        print(f"  limit          : first {args.limit} samples")

    seg_index_state = {
        "pair_to_idx": seg_index.pair_to_idx,
        "idx_to_pair": seg_index.idx_to_pair,
        "by_episode": seg_index.by_episode,
    }

    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue(maxsize=4096)
    procs: List[mp.Process] = []
    for wid in range(args.num_workers):
        p = ctx.Process(
            target=worker_main,
            args=(
                wid, args.num_workers, split_dir, seg_index_state,
                args.target_view, args.tail_placeholder_mode,
                args.apply, args.force, args.expected_stride, args.limit, out_queue,
            ),
        )
        p.start()
        procs.append(p)

    counter = {
        "updated": 0, "skipped": 0, "placeholder": 0, "failed": 0, "dryrun": 0,
    }
    failed_records: List[dict] = []
    done_workers = 0
    last_log_t = time.time()
    while done_workers < args.num_workers:
        msg = out_queue.get()
        tag = msg[0]
        if tag == "__done__":
            done_workers += 1
            print(f"  [worker {msg[1]}] finished, processed={msg[2]}")
            continue
        if tag == "__heartbeat__":
            now = time.time()
            if now - last_log_t > 5.0:
                total_done = sum(counter.values())
                print(f"  [progress] {total_done} / {len(seg_index)} samples seen")
                last_log_t = now
            continue
        status, idx, info = msg
        counter[status] = counter.get(status, 0) + 1
        if status == "failed":
            failed_records.append({"idx": idx, "split": split_name, "reason": info})

    for p in procs:
        p.join()

    print(f"  result: {counter}")
    if failed_records:
        print(f"  WARN: {len(failed_records)} failures (first few): {failed_records[:5]}")
    return {
        "split": split_name,
        "counter": counter,
        "failed_records": failed_records,
        "n_samples": n_samples,
        "n_episodes": n_episodes,
    }


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root).resolve()
    if not cache_root.is_dir():
        raise SystemExit(f"cache-root not found: {cache_root}")

    splits_to_run: List[Tuple[str, Path]] = []
    requested = [s.strip() for s in args.splits.split(",") if s.strip()]
    if "train" in requested and args.train_manifest:
        splits_to_run.append(("train", Path(args.train_manifest)))
    if "val" in requested and args.val_manifest:
        splits_to_run.append(("val", Path(args.val_manifest)))

    print("=" * 72)
    print(f"cache_root          : {cache_root}")
    print(f"target_view         : {args.target_view}")
    print(f"expected_stride     : {args.expected_stride}")
    print(f"tail_placeholder    : {args.tail_placeholder_mode}")
    print(f"apply               : {args.apply}  (dry-run if False)")
    print(f"force               : {args.force}")
    print(f"num_workers         : {args.num_workers}")
    print(f"splits              : {[s[0] for s in splits_to_run]}")
    print("=" * 72)

    all_results = []
    for split_name, manifest_path in splits_to_run:
        if not manifest_path.is_file():
            raise SystemExit(f"manifest not found for split={split_name}: {manifest_path}")
        split_dir = cache_root / split_name
        result = run_split(split_name, split_dir, manifest_path, args)
        all_results.append(result)

    failed_log_path = Path(
        args.failed_log
        if args.failed_log
        else cache_root / "tail_cache_failed.json"
    )
    failed_total = sum(len(r.get("failed_records", [])) for r in all_results)
    summary = {
        "cache_root": str(cache_root),
        "tail_placeholder_mode": args.tail_placeholder_mode,
        "apply": args.apply,
        "results": all_results,
        "failed_total": failed_total,
    }
    with failed_log_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary written to: {failed_log_path}")
    if failed_total:
        print(f"WARN: {failed_total} failures total. Inspect the JSON above.")
        sys.exit(2)
    print("All splits processed without failure.")


if __name__ == "__main__":
    main()
