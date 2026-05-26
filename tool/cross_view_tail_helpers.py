"""Helper utilities for `build_cross_view_tail_cache.py`.

Kept in a separate module to keep the main entrypoint short and to make
unit-testing the manifest indexing logic easier.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# ---------- manifest indexing ----------

@dataclass
class SegmentIndex:
    """Maps (episode_index, start_frame) -> manifest row index.

    `cache_idx_for_pair` only returns rows that are inside the same split
    that produced the manifest (i.e. for a train manifest it indexes train
    cache files only).
    """

    pair_to_idx: Dict[Tuple[int, int], int]
    idx_to_pair: List[Tuple[int, int]]
    by_episode: Dict[int, List[int]]   # episode_index -> sorted list of (start_frame, manifest_idx)

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "SegmentIndex":
        pair_to_idx: Dict[Tuple[int, int], int] = {}
        idx_to_pair: List[Tuple[int, int]] = []
        by_episode_raw: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

        with manifest_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ep = int(row["episode_index"])
                sf = int(row["start_frame"])
                key = (ep, sf)
                if key in pair_to_idx:
                    raise ValueError(
                        f"Duplicate (episode_index={ep}, start_frame={sf}) "
                        f"at manifest rows {pair_to_idx[key]} and {idx}; "
                        "tail-cache logic assumes uniqueness."
                    )
                pair_to_idx[key] = idx
                idx_to_pair.append(key)
                by_episode_raw[ep].append((sf, idx))

        by_episode = {
            ep: sorted(items, key=lambda x: x[0])
            for ep, items in by_episode_raw.items()
        }
        return cls(
            pair_to_idx=pair_to_idx, idx_to_pair=idx_to_pair, by_episode=by_episode
        )

    def __len__(self) -> int:
        return len(self.idx_to_pair)

    def next_segment_idx(self, idx: int, expected_stride: int = 81) -> Optional[int]:
        """Return the manifest idx of the segment that immediately follows
        `idx` in the same episode, or None if `idx` is the last segment.

        Uses both the strict (ep, start+stride) lookup AND a sorted-list
        traversal so we tolerate small stride irregularities (logs a warning
        upstream if a fallback is used).
        """
        ep, sf = self.idx_to_pair[idx]
        # strict path
        next_pair = (ep, sf + expected_stride)
        if next_pair in self.pair_to_idx:
            return self.pair_to_idx[next_pair]
        # fallback: find the first segment in the same episode whose start_frame > sf
        ordered = self.by_episode[ep]
        for s, j in ordered:
            if s > sf:
                return j
        return None


# ---------- placeholder strategies ----------

def make_tail_placeholder(
    target_history_latents: torch.Tensor,
    latent_views_gt: torch.Tensor,
    target_view: int,
    mode: str,
) -> torch.Tensor:
    """Build a tail-anchor latent for the LAST segment of an episode.

    target_history_latents: (1, 16, T_hist, H_lat, W_lat) - existing head anchor
    latent_views_gt:        (V, 16, T_lat, H_lat, W_lat)
    target_view:            int (typically 2 for wrist)
    mode:
      "zero"        zero tensor with same shape as target_history_latents
      "repeat-head" clone of target_history_latents (no extra signal but well-formed)
      "gt-tail"     last latent timestep of the target view from latent_views_gt;
                    NOTE: this introduces train-only GT info that is NOT available
                    at inference time. Only use if you also strip it at inference.
    """
    if mode == "zero":
        return torch.zeros_like(target_history_latents)
    if mode == "repeat-head":
        return target_history_latents.clone()
    if mode == "gt-tail":
        if latent_views_gt.ndim != 5:
            raise ValueError(
                f"latent_views_gt must be (V,C,T,H,W), got {tuple(latent_views_gt.shape)}"
            )
        # take the LAST latent timestep of the target view, shape -> (1, 16, 1, H_lat, W_lat)
        tail = latent_views_gt[target_view, :, -1:, :, :].unsqueeze(0)
        return tail.to(dtype=target_history_latents.dtype)
    raise ValueError(
        f"Unsupported tail_placeholder_mode={mode!r}; "
        "expected one of {'zero', 'repeat-head', 'gt-tail'}"
    )


# ---------- atomic write ----------

def atomic_torch_save(payload: dict, dst: Path) -> None:
    """Write torch payload via tmp + rename so a crash mid-write never
    corrupts the cache file."""
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(payload, str(tmp))
    tmp.replace(dst)


# ---------- shape validation ----------

def validate_tail_shape(
    tail: torch.Tensor, head: torch.Tensor, cache_path: str
) -> None:
    if tail.shape != head.shape:
        raise ValueError(
            f"target_tail_latents shape {tuple(tail.shape)} != "
            f"target_history_latents shape {tuple(head.shape)} for {cache_path}"
        )
    if tail.dtype != head.dtype:
        raise ValueError(
            f"target_tail_latents dtype {tail.dtype} != "
            f"target_history_latents dtype {head.dtype} for {cache_path}"
        )
