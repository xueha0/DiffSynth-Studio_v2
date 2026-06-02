#!/usr/bin/env python3
"""Evaluate wrist-target generated videos against cross-view metadata."""

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


PRED_NAME_RE = re.compile(
    r"^episode_(?P<episode>\d+)_clipstart_(?P<clipstart>\d+)_"
    r"(?P<source_view>.+)_frame_(?P<source_frame>\d+)_pred\.mp4$"
)
SIMPLE_PRED_NAME_RE = re.compile(
    r"^episode_(?P<episode>\d+)_clipstart_(?P<clipstart>\d+)_pred\.mp4$"
)


@dataclass(frozen=True)
class PredInfo:
    episode_index: int
    clipstart: int
    source_view: Optional[str]
    source_frame: Optional[int]


class AlexNetPerceptualDistance:
    def __init__(self, device: str, batch_size: int):
        from torchvision.models import AlexNet_Weights, alexnet

        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.model = alexnet(weights=AlexNet_Weights.DEFAULT).features.eval().to(self.device)
        self.layer_ids = {2, 5, 8, 10, 12}
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(frames_to_float(frames)).permute(0, 3, 1, 2).float()
        tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        tensor = tensor.to(self.device)
        return (tensor - self.mean) / self.std

    def forward_features(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for idx, layer in enumerate(self.model):
            tensor = layer(tensor)
            if idx in self.layer_ids:
                outputs.append(F.normalize(tensor, p=2, dim=1))
        return outputs

    def compute(self, gt_frames: np.ndarray, pred_frames: np.ndarray) -> float:
        total = 0.0
        count = 0
        with torch.inference_mode():
            for start in range(0, len(gt_frames), self.batch_size):
                gt = self.preprocess(gt_frames[start : start + self.batch_size])
                pred = self.preprocess(pred_frames[start : start + self.batch_size])
                gt_features = self.forward_features(gt)
                pred_features = self.forward_features(pred)
                value = 0.0
                for gt_feature, pred_feature in zip(gt_features, pred_features):
                    value = value + (gt_feature - pred_feature).pow(2).mean()
                batch_count = gt.shape[0]
                total += float(value.detach().cpu()) * batch_count
                count += batch_count
        return total / max(count, 1)


class LPIPSMetric:
    def __init__(
        self,
        backend: str,
        device: str,
        batch_size: int,
        net: str = "alex",
    ):
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.model = None
        self.name = ""

        if backend in ("auto", "lpips"):
            try:
                import lpips  # type: ignore

                self.model = lpips.LPIPS(net=net).eval().to(self.device)
                self.name = f"lpips_{net}"
            except ModuleNotFoundError as exc:
                if backend == "lpips":
                    raise ModuleNotFoundError(
                        "LPIPS metric requires the `lpips` package. Install it with "
                        "`pip install lpips`, or run with `--lpips-backend auto` for "
                        "the AlexNet perceptual fallback."
                    ) from exc

        if self.model is None:
            self.model = AlexNetPerceptualDistance(device=device, batch_size=batch_size)
            self.name = "alexnet_perceptual_fallback"

    def compute(self, gt_frames: np.ndarray, pred_frames: np.ndarray) -> float:
        if gt_frames.shape != pred_frames.shape:
            raise ValueError(
                f"LPIPS expects matching frame arrays, got {gt_frames.shape} and {pred_frames.shape}"
            )
        if len(gt_frames) == 0:
            return 0.0

        if self.name == "alexnet_perceptual_fallback":
            return self.model.compute(gt_frames, pred_frames)

        total = 0.0
        count = 0
        with torch.inference_mode():
            for start in range(0, len(gt_frames), self.batch_size):
                gt = frames_to_lpips_tensor(
                    gt_frames[start : start + self.batch_size], self.device
                )
                pred = frames_to_lpips_tensor(
                    pred_frames[start : start + self.batch_size], self.device
                )
                value = self.model(gt, pred).reshape(-1).mean()
                batch_count = gt.shape[0]
                total += float(value.detach().cpu()) * batch_count
                count += batch_count
        return total / max(count, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute PSNR/SSIM/LPIPS for generated wrist-target videos. "
            "Predictions are matched to metadata by episode index and clipstart."
        )
    )
    parser.add_argument(
        "--pred-dir",
        default="/home/xuehao/xh/projects/DiffSynth-Studio_v2/Ckpt/clip_traj_iter_000000",
        help="Directory containing `*_pred.mp4` wrist prediction videos.",
    )
    parser.add_argument(
        "--meta-jsonl",
        default=(
            "/data2/xuehao/datasets/droid_success_high_quality_crossview_meta/"
            "meta/episodes_cross_view_val_81_small200.jsonl"
        ),
        help="Cross-view metadata JSONL used to generate the validation clips.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Metrics JSON path. Defaults to `<pred-dir>/wrist_target_metrics.json`.",
    )
    parser.add_argument(
        "--metrics",
        default="psnr,ssim,lpips",
        help="Comma-separated metrics to compute. Supported: psnr,ssim,lpips.",
    )
    parser.add_argument(
        "--target-view",
        type=int,
        default=None,
        help="Override metadata target_view. Default uses each row's `target_view`.",
    )
    parser.add_argument(
        "--frame-scope",
        default="valid",
        choices=["valid", "all"],
        help=(
            "`valid` evaluates only real frames from metadata valid_frames; "
            "`all` evaluates padded clip length, repeating the last GT frame if needed."
        ),
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="Skip this many initial frames within each compared clip.",
    )
    parser.add_argument(
        "--resize-mode",
        default="gt_to_pred",
        choices=["gt_to_pred", "pred_to_gt", "none"],
        help="How to align GT/pred spatial sizes before metric computation.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for LPIPS.",
    )
    parser.add_argument(
        "--lpips-net",
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="Backbone passed to lpips.LPIPS.",
    )
    parser.add_argument(
        "--lpips-backend",
        default="lpips",
        choices=["auto", "lpips", "alexnet"],
        help=(
            "`lpips` requires the official lpips package; `auto` falls back to "
            "an AlexNet perceptual distance when lpips is unavailable."
        ),
    )
    parser.add_argument(
        "--lpips-batch-size",
        type=int,
        default=16,
        help="LPIPS frame batch size.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Evaluate only the first N sorted prediction videos.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on any missing match or decode issue instead of recording skipped samples.",
    )
    return parser.parse_args()


def parse_metrics(metrics_arg: str) -> set[str]:
    metrics = {item.strip().lower() for item in metrics_arg.split(",") if item.strip()}
    supported = {"psnr", "ssim", "lpips"}
    unknown = metrics - supported
    if unknown:
        raise ValueError(f"Unsupported metrics: {sorted(unknown)}. Supported: {sorted(supported)}")
    if not metrics:
        raise ValueError("At least one metric must be requested.")
    return metrics


def parse_pred_name(path: Path) -> PredInfo:
    match = PRED_NAME_RE.match(path.name)
    if match is not None:
        return PredInfo(
            episode_index=int(match.group("episode")),
            clipstart=int(match.group("clipstart")),
            source_view=match.group("source_view"),
            source_frame=int(match.group("source_frame")),
        )

    match = SIMPLE_PRED_NAME_RE.match(path.name)
    if match is not None:
        return PredInfo(
            episode_index=int(match.group("episode")),
            clipstart=int(match.group("clipstart")),
            source_view=None,
            source_frame=None,
        )

    raise ValueError(f"Prediction filename does not match expected pattern: {path.name}")


def load_metadata(meta_jsonl: Path) -> dict[tuple[int, int], tuple[int, dict[str, Any]]]:
    rows: dict[tuple[int, int], tuple[int, dict[str, Any]]] = {}
    with meta_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["episode_index"]), int(row["start_frame"]))
            if key in rows:
                raise ValueError(f"Duplicate metadata key {key} at line {line_no}")
            rows[key] = (line_no, row)
    return rows


def read_video_rgb(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    return np.stack(frames, axis=0)


def read_video_segment_rgb(
    path: Path,
    start_frame: int,
    valid_frames: int,
    output_frames: int,
    pad_mode: str,
) -> np.ndarray:
    if valid_frames <= 0:
        raise ValueError(f"valid_frames must be positive, got {valid_frames}")
    if output_frames <= 0:
        raise ValueError(f"output_frames must be positive, got {output_frames}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open GT video: {path}")

    frames = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        for _ in range(int(valid_frames)):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(
            f"No GT frames decoded from {path} starting at frame {start_frame}"
        )

    if len(frames) < output_frames:
        if pad_mode != "repeat_last":
            raise ValueError(f"Unsupported pad_mode={pad_mode!r}; only repeat_last is supported")
        last = frames[-1]
        frames.extend([last.copy() for _ in range(output_frames - len(frames))])

    return np.stack(frames[:output_frames], axis=0)


def align_video_sizes(
    gt_video: np.ndarray,
    pred_video: np.ndarray,
    resize_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    gt_h, gt_w = gt_video.shape[1:3]
    pred_h, pred_w = pred_video.shape[1:3]
    if (gt_h, gt_w) == (pred_h, pred_w):
        return gt_video, pred_video

    if resize_mode == "none":
        raise ValueError(
            f"GT size {(gt_w, gt_h)} does not match pred size {(pred_w, pred_h)}"
        )

    if resize_mode == "gt_to_pred":
        resized_gt = resize_video(gt_video, width=pred_w, height=pred_h)
        return resized_gt, pred_video

    if resize_mode == "pred_to_gt":
        resized_pred = resize_video(pred_video, width=gt_w, height=gt_h)
        return gt_video, resized_pred

    raise ValueError(f"Unknown resize_mode: {resize_mode}")


def resize_video(video: np.ndarray, width: int, height: int) -> np.ndarray:
    current_h, current_w = video.shape[1:3]
    interpolation = cv2.INTER_AREA if width * height < current_w * current_h else cv2.INTER_LINEAR
    return np.stack(
        [cv2.resize(frame, (int(width), int(height)), interpolation=interpolation) for frame in video],
        axis=0,
    )


def frames_to_float(frames: np.ndarray) -> np.ndarray:
    return frames.astype(np.float32) / 255.0


def frames_to_lpips_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(frames_to_float(frames)).permute(0, 3, 1, 2).contiguous()
    return tensor.to(device=device, dtype=torch.float32) * 2.0 - 1.0


def compute_basic_metrics(
    gt_video: np.ndarray,
    pred_video: np.ndarray,
    metrics: set[str],
) -> dict[str, Optional[float]]:
    gt = frames_to_float(gt_video)
    pred = frames_to_float(pred_video)
    values: dict[str, list[float]] = {"psnr": [], "ssim": [], "mse": []}

    for gt_frame, pred_frame in zip(gt, pred):
        mse = float(np.mean((pred_frame - gt_frame) ** 2))
        values["mse"].append(mse)
        if "psnr" in metrics:
            values["psnr"].append(
                float(peak_signal_noise_ratio(gt_frame, pred_frame, data_range=1.0))
            )
        if "ssim" in metrics:
            values["ssim"].append(
                float(
                    structural_similarity(
                        gt_frame,
                        pred_frame,
                        channel_axis=-1,
                        data_range=1.0,
                    )
                )
            )

    return {
        "psnr": safe_mean(values["psnr"]) if "psnr" in metrics else None,
        "ssim": safe_mean(values["ssim"]) if "ssim" in metrics else None,
        "mse": safe_mean(values["mse"]),
    }


def safe_mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def summarize_samples(samples: list[dict[str, Any]], metrics: set[str]) -> dict[str, Any]:
    total_frames = sum(int(sample["frames"]) for sample in samples)
    summary: dict[str, Any] = {
        "video_count": len(samples),
        "frame_count": total_frames,
        "overall": {},
        "per_video_mean": {},
    }

    metric_names = ["psnr", "ssim", "lpips", "mse"]
    for name in metric_names:
        if name not in metrics and name != "mse":
            continue
        weighted_values = []
        unweighted_values = []
        for sample in samples:
            value = sample["metrics"].get(name)
            if value is None:
                continue
            value = float(value)
            weighted_values.append((value, int(sample["frames"])))
            unweighted_values.append(value)

        if weighted_values and total_frames > 0:
            summary["overall"][name] = float(
                sum(value * frames for value, frames in weighted_values) / total_frames
            )
        else:
            summary["overall"][name] = None
        summary["per_video_mean"][name] = safe_mean(unweighted_values)

    return summary


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    return value


def evaluate_one(
    pred_path: Path,
    pred_info: PredInfo,
    meta_line: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    metrics: set[str],
    lpips_metric: Optional[LPIPSMetric],
) -> dict[str, Any]:
    target_view = int(row["target_view"] if args.target_view is None else args.target_view)
    target_meta = row["video"][target_view]
    pred_video = read_video_rgb(pred_path)

    if args.frame_scope == "valid":
        desired_frames = int(row.get("valid_frames", target_meta.get("pad_to_frames", len(pred_video))))
    else:
        desired_frames = int(target_meta.get("pad_to_frames", row.get("length", len(pred_video))))

    desired_frames = min(desired_frames, len(pred_video))
    if desired_frames <= args.frame_start:
        raise ValueError(
            f"No frames left after frame_start={args.frame_start}; desired_frames={desired_frames}"
        )

    gt_valid_frames = int(row.get("valid_frames", target_meta["end_frame"] - target_meta["start_frame"] + 1))
    gt_output_frames = desired_frames
    gt_video = read_video_segment_rgb(
        Path(target_meta["data"]),
        start_frame=int(target_meta["start_frame"]),
        valid_frames=min(gt_valid_frames, gt_output_frames),
        output_frames=gt_output_frames,
        pad_mode=str(target_meta.get("pad_mode", "repeat_last")),
    )

    pred_video = pred_video[:desired_frames]
    if args.frame_start:
        gt_video = gt_video[args.frame_start :]
        pred_video = pred_video[args.frame_start :]

    gt_video, pred_video = align_video_sizes(gt_video, pred_video, args.resize_mode)
    sample_metrics = compute_basic_metrics(gt_video, pred_video, metrics)
    if "lpips" in metrics:
        if lpips_metric is None:
            raise RuntimeError("LPIPS metric was requested but not initialized.")
        sample_metrics["lpips"] = lpips_metric.compute(gt_video, pred_video)

    return {
        "prediction": str(pred_path),
        "meta_line": int(meta_line),
        "episode_index": int(row["episode_index"]),
        "clipstart": int(row["start_frame"]),
        "target_view": target_view,
        "target_video": str(target_meta["data"]),
        "target_start_frame": int(target_meta["start_frame"]),
        "target_end_frame": int(target_meta["end_frame"]),
        "valid_frames": int(row.get("valid_frames", 0)),
        "frames": int(len(pred_video)),
        "pred_size": [int(pred_video.shape[2]), int(pred_video.shape[1])],
        "gt_size_after_resize": [int(gt_video.shape[2]), int(gt_video.shape[1])],
        "source_view_from_name": pred_info.source_view,
        "source_frame_from_name": pred_info.source_frame,
        "metrics": sample_metrics,
    }


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    meta_jsonl = Path(args.meta_jsonl)
    output_json = Path(args.output_json) if args.output_json else pred_dir / "wrist_target_metrics.json"
    metrics = parse_metrics(args.metrics)

    rows = load_metadata(meta_jsonl)
    pred_paths = sorted(pred_dir.glob("*_pred.mp4"))
    if args.sample_limit is not None:
        pred_paths = pred_paths[: int(args.sample_limit)]
    if not pred_paths:
        raise RuntimeError(f"No `*_pred.mp4` files found in {pred_dir}")

    lpips_metric = None
    if "lpips" in metrics:
        lpips_metric = LPIPSMetric(
            backend=args.lpips_backend,
            device=args.device,
            batch_size=args.lpips_batch_size,
            net=args.lpips_net,
        )

    samples = []
    skipped = []
    for pred_path in tqdm(pred_paths, desc="Evaluating wrist target videos"):
        try:
            pred_info = parse_pred_name(pred_path)
            key = (pred_info.episode_index, pred_info.clipstart)
            if key not in rows:
                raise KeyError(f"No metadata row for prediction key {key}")
            meta_line, row = rows[key]
            samples.append(
                evaluate_one(
                    pred_path=pred_path,
                    pred_info=pred_info,
                    meta_line=meta_line,
                    row=row,
                    args=args,
                    metrics=metrics,
                    lpips_metric=lpips_metric,
                )
            )
        except Exception as exc:
            if args.strict:
                raise
            skipped.append({"prediction": str(pred_path), "reason": repr(exc)})

    if not samples:
        raise RuntimeError("No samples were successfully evaluated.")

    payload = {
        "config": {
            "pred_dir": str(pred_dir),
            "meta_jsonl": str(meta_jsonl),
            "metrics": sorted(metrics),
            "target_view": args.target_view,
            "frame_scope": args.frame_scope,
            "frame_start": int(args.frame_start),
            "resize_mode": args.resize_mode,
            "device": args.device if "lpips" in metrics else None,
            "lpips_net": args.lpips_net if "lpips" in metrics else None,
            "lpips_backend_requested": args.lpips_backend if "lpips" in metrics else None,
            "lpips_backend_used": lpips_metric.name if lpips_metric is not None else None,
            "lpips_batch_size": int(args.lpips_batch_size) if "lpips" in metrics else None,
            "sample_limit": args.sample_limit,
        },
        "summary": summarize_samples(samples, metrics),
        "skipped": skipped,
        "samples": samples,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)

    summary = payload["summary"]["overall"]
    print(f"Saved metrics to: {output_json}")
    print(
        "Overall: "
        + ", ".join(
            f"{name}={summary.get(name)}"
            for name in ["psnr", "ssim", "lpips", "mse"]
            if name in summary
        )
    )
    if skipped:
        print(f"Skipped samples: {len(skipped)}")


if __name__ == "__main__":
    main()
