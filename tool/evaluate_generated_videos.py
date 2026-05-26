#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate already-generated comparison videos without regenerating them. "
            "Supports PSNR, SSIM, LPIPS-like distance, and FVD-style Fréchet distance."
        )
    )
    parser.add_argument(
        "--comparison-dir",
        required=True,
        help="Directory containing comparison videos. Supports recursive search.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to output metrics JSON.",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=3,
        help="Number of rows/views in each comparison video.",
    )
    parser.add_argument(
        "--metrics",
        default="fvd,lpips,ssim,psnr",
        help="Comma-separated metrics to compute.",
    )
    parser.add_argument(
        "--fvd-backbone",
        default="torchvision",
        choices=["torchvision", "i3d"],
        help="Feature backbone for FVD.",
    )
    parser.add_argument(
        "--i3d-weights",
        default=None,
        help="Optional I3D weights path for strict I3D-FVD mode.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional cap on number of videos for quick evaluation.",
    )
    parser.add_argument(
        "--lpips-backend",
        default="auto",
        choices=["auto", "lpips", "alexnet"],
        help="LPIPS backend: official package if available, else alexnet perceptual fallback.",
    )
    parser.add_argument(
        "--fvd-frames",
        type=int,
        default=16,
        help="Number of uniformly sampled frames used for FVD features.",
    )
    parser.add_argument(
        "--fvd-batch-size",
        type=int,
        default=8,
        help="Batch size for LPIPS/perceptual frame processing.",
    )
    return parser.parse_args()


def find_videos(root: Path, limit: Optional[int] = None) -> List[Path]:
    videos = []
    for current_root, _, files in os.walk(root):
        for file_name in files:
            if file_name.lower().endswith(".mp4"):
                videos.append(Path(current_root) / file_name)
    videos = sorted(videos)
    if limit is not None:
        videos = videos[: int(limit)]
    return videos


def find_split_dirs(root: Path) -> Dict[str, Path]:
    split_dirs = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if any(file.suffix.lower() == ".mp4" for file in child.rglob("*.mp4")):
            split_dirs[child.name] = child
    return split_dirs


def read_video(path: Path) -> np.ndarray:
    reader = imageio.get_reader(str(path))
    frames = []
    for frame in reader:
        frames.append(frame.astype(np.float32) / 255.0)
    reader.close()
    return np.asarray(frames, dtype=np.float32)


def split_comparison_grid(frames: np.ndarray, num_views: int) -> List[List[np.ndarray]]:
    if frames.ndim != 4:
        raise ValueError(f"Expected video frames with shape (T,H,W,C), got {frames.shape}")
    row_splits = np.array_split(frames, num_views, axis=1)
    grid = []
    for row in row_splits:
        cols = np.array_split(row, 2, axis=2)
        if len(cols) != 2:
            raise ValueError("Comparison video must have 2 columns (GT|Pred).")
        grid.append(cols)
    return grid


def crop_pair(gt_frame: np.ndarray, pred_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    min_h = min(gt_frame.shape[0], pred_frame.shape[0])
    min_w = min(gt_frame.shape[1], pred_frame.shape[1])
    if min_h <= 0 or min_w <= 0:
        raise ValueError("Invalid comparison frame size after cropping.")
    return gt_frame[:min_h, :min_w], pred_frame[:min_h, :min_w]


def uniform_sample_video(video: np.ndarray, num_frames: int) -> np.ndarray:
    if len(video) == 0:
        raise ValueError("Cannot sample an empty video.")
    if len(video) == num_frames:
        return video
    indices = np.linspace(0, len(video) - 1, num=num_frames)
    indices = np.clip(np.round(indices).astype(int), 0, len(video) - 1)
    return video[indices]


@dataclass
class MetricState:
    num_views: int
    metrics: set[str]
    psnr_sum: List[float]
    ssim_sum: List[float]
    lpips_sum: List[float]
    mse_sum: List[float]
    frames: List[int]
    fvd_gt_features: List[List[np.ndarray]]
    fvd_pred_features: List[List[np.ndarray]]
    overall_fvd_gt: List[np.ndarray]
    overall_fvd_pred: List[np.ndarray]
    failed_videos: List[dict]

    @classmethod
    def create(cls, num_views: int, metrics: set[str]):
        return cls(
            num_views=num_views,
            metrics=metrics,
            psnr_sum=[0.0] * num_views,
            ssim_sum=[0.0] * num_views,
            lpips_sum=[0.0] * num_views,
            mse_sum=[0.0] * num_views,
            frames=[0] * num_views,
            fvd_gt_features=[[] for _ in range(num_views)],
            fvd_pred_features=[[] for _ in range(num_views)],
            overall_fvd_gt=[],
            overall_fvd_pred=[],
            failed_videos=[],
        )


class TorchvisionFVDExtractor:
    def __init__(self, device: str = "cpu", num_frames: int = 16):
        from torchvision.models.video import R3D_18_Weights, r3d_18

        self.device = torch.device(device)
        self.num_frames = int(num_frames)
        self.weights = R3D_18_Weights.DEFAULT
        self.transforms = self.weights.transforms()
        self.model = r3d_18(weights=self.weights)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(self.device)

    def extract(self, video: np.ndarray) -> np.ndarray:
        video = uniform_sample_video(video, self.num_frames)
        tensor = torch.from_numpy(video).permute(0, 3, 1, 2).contiguous()
        tensor = self.transforms(tensor).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features = self.model(tensor)
        return features[0].detach().cpu().float().numpy()


class I3DFVDExtractor:
    def __init__(self, weights_path: str, device: str = "cpu", num_frames: int = 16):
        if weights_path is None:
            raise ValueError("`--i3d-weights` is required for i3d FVD mode.")
        try:
            from pytorch_i3d import InceptionI3d  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "I3D-FVD requested but `pytorch_i3d` is not installed."
            ) from exc
        self.device = torch.device(device)
        self.num_frames = int(num_frames)
        self.model = InceptionI3d(400, in_channels=3)
        state_dict = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().to(self.device)

    def extract(self, video: np.ndarray) -> np.ndarray:
        video = uniform_sample_video(video, self.num_frames)
        tensor = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0).float()
        tensor = tensor.to(self.device)
        tensor = tensor * 2.0 - 1.0
        with torch.inference_mode():
            if hasattr(self.model, "extract_features"):
                features = self.model.extract_features(tensor)
            else:
                raise RuntimeError(
                    "Loaded I3D model does not expose `extract_features`."
                )
        return features.flatten(1)[0].detach().cpu().float().numpy()


class AlexNetPerceptualDistance:
    def __init__(self, device: str = "cpu", batch_size: int = 8):
        from torchvision.models import AlexNet_Weights, alexnet

        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.model = alexnet(weights=AlexNet_Weights.DEFAULT).features.eval().to(self.device)
        self.layer_ids = {2, 5, 8, 10, 12}
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(self.device)
        tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        tensor = (tensor - self.mean) / self.std
        return tensor

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for idx, layer in enumerate(self.model):
            x = layer(x)
            if idx in self.layer_ids:
                outputs.append(F.normalize(x, p=2, dim=1))
        return outputs

    def compute(self, gt_frames: np.ndarray, pred_frames: np.ndarray) -> float:
        gt = self.preprocess(gt_frames)
        pred = self.preprocess(pred_frames)
        total = 0.0
        count = 0
        for start in range(0, gt.shape[0], self.batch_size):
            gt_batch = gt[start : start + self.batch_size]
            pred_batch = pred[start : start + self.batch_size]
            gt_feats = self.forward_features(gt_batch)
            pred_feats = self.forward_features(pred_batch)
            batch_value = 0.0
            for gt_feat, pred_feat in zip(gt_feats, pred_feats):
                batch_value = batch_value + (gt_feat - pred_feat).pow(2).mean()
            total += float(batch_value.detach().cpu()) * gt_batch.shape[0]
            count += gt_batch.shape[0]
        return total / max(count, 1)


class LPIPSComputer:
    def __init__(self, backend: str, device: str = "cpu", batch_size: int = 8):
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.backend = backend
        self.impl = None
        self.name = None
        if backend in ("auto", "lpips"):
            try:
                import lpips

                self.impl = lpips.LPIPS(net="alex").eval().to(self.device)
                self.name = "lpips_alex"
            except ModuleNotFoundError:
                if backend == "lpips":
                    raise
        if self.impl is None:
            self.impl = AlexNetPerceptualDistance(device=device, batch_size=batch_size)
            self.name = "alexnet_perceptual_fallback"

    def compute(self, gt_frames: np.ndarray, pred_frames: np.ndarray) -> float:
        if self.name == "lpips_alex":
            return self._compute_official_lpips(gt_frames, pred_frames)
        return self.impl.compute(gt_frames, pred_frames)

    def _compute_official_lpips(self, gt_frames: np.ndarray, pred_frames: np.ndarray) -> float:
        gt = torch.from_numpy(gt_frames).permute(0, 3, 1, 2).float().to(self.device)
        pred = torch.from_numpy(pred_frames).permute(0, 3, 1, 2).float().to(self.device)
        gt = F.interpolate(gt, size=(224, 224), mode="bilinear", align_corners=False)
        pred = F.interpolate(pred, size=(224, 224), mode="bilinear", align_corners=False)
        gt = gt * 2.0 - 1.0
        pred = pred * 2.0 - 1.0
        total = 0.0
        count = 0
        with torch.inference_mode():
            for start in range(0, gt.shape[0], self.batch_size):
                gt_batch = gt[start : start + self.batch_size]
                pred_batch = pred[start : start + self.batch_size]
                value = self.impl(gt_batch, pred_batch).mean()
                total += float(value.detach().cpu()) * gt_batch.shape[0]
                count += gt_batch.shape[0]
        return total / max(count, 1)


def compute_frechet_distance(features_a: List[np.ndarray], features_b: List[np.ndarray]) -> Optional[float]:
    if len(features_a) < 2 or len(features_b) < 2:
        return None
    feats_a = np.stack(features_a, axis=0)
    feats_b = np.stack(features_b, axis=0)
    mu_a = np.mean(feats_a, axis=0)
    mu_b = np.mean(feats_b, axis=0)
    sigma_a = np.cov(feats_a, rowvar=False)
    sigma_b = np.cov(feats_b, rowvar=False)
    eps = 1e-6
    covmean, _ = linalg.sqrtm((sigma_a + eps * np.eye(sigma_a.shape[0])) @ (sigma_b + eps * np.eye(sigma_b.shape[0])), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_a - mu_b
    return float(diff.dot(diff) + np.trace(sigma_a + sigma_b - 2.0 * covmean))


def process_video(
    video_path: Path,
    num_views: int,
    metrics: set[str],
    lpips_metric: Optional[LPIPSComputer],
    fvd_extractor,
    state: MetricState,
):
    frames = read_video(video_path)
    grid = split_comparison_grid(frames, num_views)
    for view_idx, row in enumerate(grid):
        gt_video, pred_video = row
        min_frames = min(len(gt_video), len(pred_video))
        if min_frames == 0:
            continue
        gt_video = gt_video[:min_frames]
        pred_video = pred_video[:min_frames]

        if "fvd" in metrics and fvd_extractor is not None:
            gt_feat = fvd_extractor.extract(gt_video)
            pred_feat = fvd_extractor.extract(pred_video)
            state.fvd_gt_features[view_idx].append(gt_feat)
            state.fvd_pred_features[view_idx].append(pred_feat)
            state.overall_fvd_gt.append(gt_feat)
            state.overall_fvd_pred.append(pred_feat)

        if "lpips" in metrics and lpips_metric is not None:
            state.lpips_sum[view_idx] += lpips_metric.compute(gt_video, pred_video) * min_frames

        for frame_idx in range(min_frames):
            gt_frame, pred_frame = crop_pair(gt_video[frame_idx], pred_video[frame_idx])
            if "psnr" in metrics:
                state.psnr_sum[view_idx] += peak_signal_noise_ratio(pred_frame, gt_frame, data_range=1.0)
            if "ssim" in metrics:
                state.ssim_sum[view_idx] += structural_similarity(
                    pred_frame,
                    gt_frame,
                    channel_axis=-1,
                    data_range=1.0,
                )
            state.mse_sum[view_idx] += float(np.mean((pred_frame - gt_frame) ** 2))
            state.frames[view_idx] += 1


def summarize_metrics(state: MetricState, lpips_backend: Optional[str], fvd_backbone: Optional[str]) -> dict:
    view_metrics = []
    total_psnr_sum = 0.0
    total_ssim_sum = 0.0
    total_lpips_sum = 0.0
    total_mse_sum = 0.0
    total_frames = 0

    for view_idx in range(state.num_views):
        frames = state.frames[view_idx]
        psnr = state.psnr_sum[view_idx] / frames if frames > 0 else None
        ssim = state.ssim_sum[view_idx] / frames if frames > 0 else None
        lpips_value = state.lpips_sum[view_idx] / frames if ("lpips" in state.metrics and frames > 0) else None
        mse = state.mse_sum[view_idx] / frames if frames > 0 else None
        fvd = compute_frechet_distance(
            state.fvd_gt_features[view_idx],
            state.fvd_pred_features[view_idx],
        ) if "fvd" in state.metrics else None
        if psnr is not None:
            total_psnr_sum += state.psnr_sum[view_idx]
        if ssim is not None:
            total_ssim_sum += state.ssim_sum[view_idx]
        if lpips_value is not None:
            total_lpips_sum += state.lpips_sum[view_idx]
        if mse is not None:
            total_mse_sum += state.mse_sum[view_idx]
        total_frames += frames
        view_metrics.append(
            {
                "view_index": view_idx,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips_value,
                "mse": mse,
                "fvd": fvd,
                "frames": frames,
                "video_count": len(state.fvd_gt_features[view_idx]) if "fvd" in state.metrics else None,
            }
        )

    overall = {
        "psnr": total_psnr_sum / total_frames if ("psnr" in state.metrics and total_frames > 0) else None,
        "ssim": total_ssim_sum / total_frames if ("ssim" in state.metrics and total_frames > 0) else None,
        "lpips": total_lpips_sum / total_frames if ("lpips" in state.metrics and total_frames > 0) else None,
        "mse": total_mse_sum / total_frames if total_frames > 0 else None,
        "fvd": compute_frechet_distance(state.overall_fvd_gt, state.overall_fvd_pred) if "fvd" in state.metrics else None,
        "frames": total_frames,
        "video_count": len(state.overall_fvd_gt) if "fvd" in state.metrics else sum(1 for frames in state.frames if frames > 0),
    }
    return {
        "overall": overall,
        "view_metrics": view_metrics,
        "failed_videos": state.failed_videos,
        "meta": {
            "lpips_backend": lpips_backend,
            "fvd_backbone": fvd_backbone,
            "num_views": state.num_views,
        },
    }


def evaluate_video_set(
    video_paths: Iterable[Path],
    num_views: int,
    metrics: set[str],
    lpips_metric: Optional[LPIPSComputer],
    fvd_extractor,
) -> dict:
    state = MetricState.create(num_views, metrics)
    for video_path in tqdm(list(video_paths), desc="Evaluating videos"):
        try:
            process_video(
                video_path=video_path,
                num_views=num_views,
                metrics=metrics,
                lpips_metric=lpips_metric,
                fvd_extractor=fvd_extractor,
                state=state,
            )
        except Exception as exc:  # pragma: no cover - runtime protection
            state.failed_videos.append({"path": str(video_path), "error": str(exc)})
    return summarize_metrics(
        state,
        lpips_backend=lpips_metric.name if lpips_metric is not None else None,
        fvd_backbone=getattr(fvd_extractor, "__class__", type(None)).__name__ if fvd_extractor is not None else None,
    )


def build_fvd_extractor(args: argparse.Namespace):
    if "fvd" not in args.metrics_set:
        return None
    if args.fvd_backbone == "torchvision":
        return TorchvisionFVDExtractor(device=args.device, num_frames=args.fvd_frames)
    return I3DFVDExtractor(
        weights_path=args.i3d_weights,
        device=args.device,
        num_frames=args.fvd_frames,
    )


def build_lpips_metric(args: argparse.Namespace):
    if "lpips" not in args.metrics_set:
        return None
    return LPIPSComputer(
        backend=args.lpips_backend,
        device=args.device,
        batch_size=args.fvd_batch_size,
    )


def main() -> None:
    args = parse_args()
    args.metrics_set = {item.strip().lower() for item in args.metrics.split(",") if item.strip()}
    comparison_dir = Path(args.comparison_dir).resolve()
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    lpips_metric = build_lpips_metric(args)
    fvd_extractor = build_fvd_extractor(args)

    overall_videos = find_videos(comparison_dir, limit=args.sample_limit)
    if len(overall_videos) == 0:
        raise RuntimeError(f"No comparison videos found in {comparison_dir}")
    payload = {
        "comparison_dir": str(comparison_dir),
        "video_count": len(overall_videos),
        "metrics_requested": sorted(args.metrics_set),
        "results": evaluate_video_set(
            overall_videos,
            num_views=args.num_views,
            metrics=args.metrics_set,
            lpips_metric=lpips_metric,
            fvd_extractor=fvd_extractor,
        ),
        "subdirs": {},
    }

    for name, split_dir in find_split_dirs(comparison_dir).items():
        videos = find_videos(split_dir, limit=args.sample_limit)
        if len(videos) == 0:
            continue
        payload["subdirs"][name] = evaluate_video_set(
            videos,
            num_views=args.num_views,
            metrics=args.metrics_set,
            lpips_metric=lpips_metric,
            fvd_extractor=fvd_extractor,
        )

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to: {output_json}")


if __name__ == "__main__":
    main()
