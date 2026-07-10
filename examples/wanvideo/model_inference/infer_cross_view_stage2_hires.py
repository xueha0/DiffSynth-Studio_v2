#!/usr/bin/env python3
"""Stage2 cross-view inference entrypoint with inference-size overrides.

This wrapper intentionally leaves infer_cross_view_stage2.py unchanged. It
reuses that script's implementation and only adds CLI overrides for the
dataset/video size used during inference.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import infer_cross_view_stage2 as base


_BASE_PARSE_ARGS = base.parse_args
_BASE_BUILD_CONFIG = base.build_config


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    override_parser = argparse.ArgumentParser(add_help=False)
    override_parser.add_argument(
        "--height_override",
        type=_positive_int,
        default=None,
        help=(
            "Override config height for inference dataset preprocessing. With "
            "resize_mode=fit this is an area budget, so the effective frame "
            "height may be rounded down to a divisible size such as 352."
        ),
    )
    override_parser.add_argument(
        "--width_override",
        type=_positive_int,
        default=None,
        help="Override config width for inference dataset preprocessing.",
    )
    override_parser.add_argument(
        "--resize_mode_override",
        choices=("fit", "crop"),
        default=None,
        help=(
            "Override config resize_mode for inference. Use fit for the "
            "current training-style aspect-preserving behavior."
        ),
    )

    overrides, remaining = override_parser.parse_known_args()
    original_argv = sys.argv
    sys.argv = [original_argv[0], *remaining]
    try:
        args = _BASE_PARSE_ARGS()
    finally:
        sys.argv = original_argv

    args.height_override = overrides.height_override
    args.width_override = overrides.width_override
    args.resize_mode_override = overrides.resize_mode_override
    return args


def _maybe_warn_non_multiple(name: str, value: Optional[int], factor: int = 16) -> None:
    if value is not None and value % factor != 0:
        print(
            f"[hires] WARN: {name}={value} is not divisible by {factor}. "
            "resize_mode=fit may round the effective frame size; resize_mode=crop "
            "with non-divisible sizes can fail downstream."
        )


def build_config(args: argparse.Namespace) -> base.EvalConfig:
    config = _BASE_BUILD_CONFIG(args)

    old_height = int(config.height)
    old_width = int(config.width)
    old_resize_mode = str(config.resize_mode)

    if args.height_override is not None:
        config.height = int(args.height_override)
    if args.width_override is not None:
        config.width = int(args.width_override)
    if args.resize_mode_override is not None:
        config.resize_mode = str(args.resize_mode_override)

    if args.height_override is not None or args.width_override is not None:
        _maybe_warn_non_multiple("height_override", args.height_override)
        _maybe_warn_non_multiple("width_override", args.width_override)

    if (
        int(config.height) != old_height
        or int(config.width) != old_width
        or str(config.resize_mode) != old_resize_mode
    ):
        print(
            "[hires] Inference preprocessing override: "
            f"{old_width}x{old_height}/{old_resize_mode} -> "
            f"{int(config.width)}x{int(config.height)}/{config.resize_mode}"
        )
        if str(config.resize_mode) == "fit":
            print(
                "[hires] resize_mode=fit preserves aspect ratio under the target "
                "area budget; effective saved video size may be smaller than the "
                "requested height."
            )

    return config


def main() -> None:
    base.parse_args = parse_args
    base.build_config = build_config
    base.main()


if __name__ == "__main__":
    main()
