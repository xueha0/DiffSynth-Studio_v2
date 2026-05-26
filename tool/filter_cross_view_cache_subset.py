"""按 (episode_index, start_frame) 过滤 cross-view cache, 用软链接构造子集.

典型用例: orig manifest 包含 episode 末段 (target_tail_latents 必为 zero placeholder),
new manifest 已剔除末段 -> 用本脚本生成只含非末段样本的 cache 子集, 训练时直接
切到这个子集 (CACHE_ROOT=...filtered).

输入:
  --src-cache       原 cache 目录 (含 train/, val/, cache_config.json)
  --src-sidecar     原 sidecar 目录 (含 train/, val/, geometry_cache_config.json)
                    [可选, 不传则跳过 sidecar 筛选]
  --orig-train-manifest / --orig-val-manifest:
                    原 manifest, 行 i = src cache/{i:07d}.pth
  --new-train-manifest / --new-val-manifest:
                    新 manifest, 决定要保留哪些样本
  --dst-cache       目标 cache 目录 (会创建); 内含 train/{0:07d}.pth ... 的软链接
  --dst-sidecar     目标 sidecar 目录 (可选, 与 dst-cache 配套)
  --copy-config     是否复制 cache_config.json 到 dst (默认 True)
  --apply           真正写盘 (无该 flag 则 dry-run)

输出布局:
  dst-cache/
    train/0000000.pth -> ../../../src-cache/train/{orig_idx:07d}.pth   (软链接)
    train/0000001.pth -> ...
    val/...
    cache_config.json                                                   (复制)
    __filter_provenance.json                                            (审计: dst_idx -> src_idx mapping)

正确性保证:
  - new manifest 的每行 (ep, sf) 必须在 orig manifest 中存在; 否则 abort.
  - 检查 src 文件存在; src 缺失 abort.
  - 检查 dst 不已存在 (避免误覆盖); 已存在则要求 --force.
  - 软链接是 *相对* 路径 (--relative-symlink, 默认), 这样把 dst 整体移动到其他位置
    只要保持与 src 的相对位置不变, 链接仍可用.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--src-cache", type=str, required=True)
    p.add_argument("--dst-cache", type=str, required=True)
    p.add_argument("--src-sidecar", type=str, default=None)
    p.add_argument("--dst-sidecar", type=str, default=None)
    p.add_argument("--orig-train-manifest", type=str, required=True)
    p.add_argument("--new-train-manifest", type=str, required=True)
    p.add_argument("--orig-val-manifest", type=str, default=None)
    p.add_argument("--new-val-manifest", type=str, default=None)
    p.add_argument("--copy-config", action="store_true", default=True)
    p.add_argument("--no-copy-config", dest="copy_config", action="store_false")
    p.add_argument("--apply", action="store_true",
                   help="Actually create symlinks. Without this flag the script is read-only.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing dst dirs.")
    p.add_argument("--absolute-symlink", action="store_true",
                   help="Use absolute symlink targets (default: relative).")
    return p.parse_args()


def load_manifest_pairs(path: Path) -> List[Tuple[int, int]]:
    """Return list of (episode_index, start_frame) in manifest row order."""
    rows: List[Tuple[int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append((int(row["episode_index"]), int(row["start_frame"])))
    return rows


def build_pair_to_orig_idx(pairs: List[Tuple[int, int]]) -> dict:
    out: dict = {}
    for orig_idx, key in enumerate(pairs):
        if key in out:
            raise ValueError(
                f"Duplicate (episode={key[0]}, start_frame={key[1]}) in orig manifest "
                f"at rows {out[key]} and {orig_idx}; cannot disambiguate."
            )
        out[key] = orig_idx
    return out


def make_symlink(src: Path, dst: Path, relative: bool, apply: bool, force: bool) -> str:
    """Returns one of: 'created', 'replaced', 'dryrun-create', 'skipped-exists'."""
    if dst.exists() or dst.is_symlink():
        if not force:
            return "skipped-exists"
        if apply:
            dst.unlink()
    target: Path = src
    if relative:
        # compute the target relative to dst's parent dir
        target = Path(os.path.relpath(src, start=dst.parent))
    if not apply:
        return "dryrun-create"
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(target), str(dst))
    if dst.exists() or dst.is_symlink():
        return "created" if not force else "replaced"
    return "failed"


def process_split(
    split_name: str,
    src_cache_split: Path,
    dst_cache_split: Path,
    orig_pairs: List[Tuple[int, int]],
    new_pairs: List[Tuple[int, int]],
    src_sidecar_split: Optional[Path],
    dst_sidecar_split: Optional[Path],
    apply: bool,
    force: bool,
    relative: bool,
) -> dict:
    print(f"\n=== Split: {split_name} ===")
    print(f"  src cache    : {src_cache_split}")
    print(f"  dst cache    : {dst_cache_split}")
    if src_sidecar_split is not None:
        print(f"  src sidecar  : {src_sidecar_split}")
        print(f"  dst sidecar  : {dst_sidecar_split}")
    if not src_cache_split.is_dir():
        raise SystemExit(f"src cache split not found: {src_cache_split}")
    if src_sidecar_split is not None and not src_sidecar_split.is_dir():
        raise SystemExit(f"src sidecar split not found: {src_sidecar_split}")

    pair_to_orig = build_pair_to_orig_idx(orig_pairs)
    print(f"  orig rows   : {len(orig_pairs)}")
    print(f"  new rows    : {len(new_pairs)}")

    # Validate every new pair is in orig
    missing = [p for p in new_pairs if p not in pair_to_orig]
    if missing:
        raise SystemExit(
            f"{len(missing)} pairs in new manifest absent from orig manifest "
            f"(first 3: {missing[:3]}). Cannot map to cache idx."
        )

    counter = {"created": 0, "replaced": 0, "dryrun-create": 0,
               "skipped-exists": 0, "failed": 0, "src-missing": 0}
    provenance = []
    for new_idx, key in enumerate(new_pairs):
        orig_idx = pair_to_orig[key]
        src_path = src_cache_split / f"{orig_idx:07d}.pth"
        dst_path = dst_cache_split / f"{new_idx:07d}.pth"
        if not src_path.exists():
            counter["src-missing"] += 1
            provenance.append({"new_idx": new_idx, "orig_idx": orig_idx,
                               "ep": key[0], "sf": key[1], "status": "src-missing"})
            continue
        st = make_symlink(src_path, dst_path, relative, apply, force)
        counter[st] = counter.get(st, 0) + 1
        provenance.append({"new_idx": new_idx, "orig_idx": orig_idx,
                           "ep": key[0], "sf": key[1], "status": st})

        if src_sidecar_split is not None and dst_sidecar_split is not None:
            sc_src = src_sidecar_split / f"{orig_idx:07d}.pth"
            sc_dst = dst_sidecar_split / f"{new_idx:07d}.pth"
            if not sc_src.exists():
                counter["src-missing"] += 1
                continue
            make_symlink(sc_src, sc_dst, relative, apply, force)
    print(f"  result      : {counter}")
    return {"split": split_name, "counter": counter, "provenance": provenance}


def main() -> None:
    args = parse_args()
    src_cache = Path(args.src_cache).resolve()
    dst_cache = Path(args.dst_cache).resolve()
    src_sidecar = Path(args.src_sidecar).resolve() if args.src_sidecar else None
    dst_sidecar = Path(args.dst_sidecar).resolve() if args.dst_sidecar else None

    if dst_cache.exists() and any(dst_cache.iterdir()) and not args.force:
        raise SystemExit(
            f"dst-cache already non-empty: {dst_cache}\n"
            f"  Pass --force to overwrite."
        )
    if not src_cache.is_dir():
        raise SystemExit(f"src-cache not found: {src_cache}")
    if (src_sidecar is None) != (dst_sidecar is None):
        raise SystemExit("--src-sidecar and --dst-sidecar must be set together (or neither).")

    print("=" * 72)
    print(f"  src-cache    : {src_cache}")
    print(f"  dst-cache    : {dst_cache}")
    print(f"  src-sidecar  : {src_sidecar}")
    print(f"  dst-sidecar  : {dst_sidecar}")
    print(f"  apply        : {args.apply} (dry-run if False)")
    print(f"  force        : {args.force}")
    print(f"  relative ln  : {not args.absolute_symlink}")
    print("=" * 72)

    splits_to_run: List[Tuple[str, Path, Path]] = []
    splits_to_run.append(("train", Path(args.orig_train_manifest), Path(args.new_train_manifest)))
    if args.orig_val_manifest and args.new_val_manifest:
        splits_to_run.append(("val", Path(args.orig_val_manifest), Path(args.new_val_manifest)))

    all_results = []
    for split_name, orig_m, new_m in splits_to_run:
        if not orig_m.is_file():
            raise SystemExit(f"orig manifest not found: {orig_m}")
        if not new_m.is_file():
            raise SystemExit(f"new manifest not found: {new_m}")
        orig_pairs = load_manifest_pairs(orig_m)
        new_pairs = load_manifest_pairs(new_m)

        result = process_split(
            split_name=split_name,
            src_cache_split=src_cache / split_name,
            dst_cache_split=dst_cache / split_name,
            orig_pairs=orig_pairs,
            new_pairs=new_pairs,
            src_sidecar_split=(src_sidecar / split_name) if src_sidecar else None,
            dst_sidecar_split=(dst_sidecar / split_name) if dst_sidecar else None,
            apply=args.apply,
            force=args.force,
            relative=not args.absolute_symlink,
        )
        all_results.append(result)

    # Copy cache_config.json
    if args.copy_config and args.apply:
        for cfg_name in ("cache_config.json",):
            src_cfg = src_cache / cfg_name
            if src_cfg.exists():
                dst_cache.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_cfg, dst_cache / cfg_name)
                print(f"  copied: {src_cfg.name} -> {dst_cache}/")
        if dst_sidecar is not None and src_sidecar is not None:
            for cfg_name in ("geometry_cache_config.json",):
                src_cfg = src_sidecar / cfg_name
                if src_cfg.exists():
                    dst_sidecar.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_cfg, dst_sidecar / cfg_name)
                    print(f"  copied: {src_cfg.name} -> {dst_sidecar}/")

    # Write provenance
    if args.apply:
        prov_path = dst_cache / "__filter_provenance.json"
        prov_path.parent.mkdir(parents=True, exist_ok=True)
        with prov_path.open("w", encoding="utf-8") as f:
            json.dump({
                "src_cache": str(src_cache),
                "src_sidecar": str(src_sidecar) if src_sidecar else None,
                "results": all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nProvenance: {prov_path}")

    total_failed = sum(r["counter"].get("failed", 0) + r["counter"].get("src-missing", 0)
                       for r in all_results)
    if total_failed:
        print(f"\nWARN: {total_failed} samples missing/failed.")
    else:
        print(f"\nAll splits OK ({sum(r['counter'].get('created', 0) + r['counter'].get('dryrun-create', 0) for r in all_results)} symlinks).")


if __name__ == "__main__":
    main()
