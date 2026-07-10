#!/usr/bin/env python3
"""Evaluate wrist-target generated videos against cross-view metadata."""

import argparse
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


PRED_NAME_RE = re.compile(
    r"^episode_(?P<episode>\d+)_clipstart_(?P<clipstart>\d+)_"
    r"(?P<source_view>.+)_frame_(?P<source_frame>\d+)_pred\.mp4$"
)
SIMPLE_PRED_NAME_RE = re.compile(
    r"^episode_(?P<episode>\d+)_clipstart_(?P<clipstart>\d+)_pred\.mp4$"
)
VAL_PRED_NAME_RE = re.compile(r"^val_(?P<row_index>\d+)_ep(?P<episode>\d+)\.mp4$")
INDEX_PRED_NAME_RE = re.compile(r"^(?P<row_index>\d+)\.mp4$")
PYAV_PREFERRED_DECODERS = {"av1", "libaom-av1", "libdav1d"}


@dataclass(frozen=True)
class PredInfo:
    episode_index: int
    clipstart: int
    source_view: Optional[str]
    source_frame: Optional[int]
    row_index: Optional[int] = None


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


class FIDMetric:
    def __init__(self, device: str, batch_size: int, dims: int = 2048):
        try:
            from pytorch_fid.inception import InceptionV3  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "FID requires the `pytorch-fid` package. Install it with "
                "`pip install pytorch-fid`."
            ) from exc

        if dims not in InceptionV3.BLOCK_INDEX_BY_DIM:
            raise ValueError(f"Unsupported FID dims={dims}.")
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.dims = int(dims)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(self.device).eval()

    def extract(self, frames: np.ndarray) -> np.ndarray:
        if len(frames) == 0:
            return np.empty((0, self.dims), dtype=np.float32)
        tensor = torch.from_numpy(frames_to_float(frames)).permute(0, 3, 1, 2).float()
        activations = []
        with torch.inference_mode():
            for start in range(0, tensor.shape[0], self.batch_size):
                batch = tensor[start : start + self.batch_size].to(self.device)
                pred = self.model(batch)[0]
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                pred = pred.squeeze(3).squeeze(2).detach().cpu().numpy()
                activations.append(pred)
        if not activations:
            return np.empty((0, self.dims), dtype=np.float32)
        return np.concatenate(activations, axis=0)


class FVDMetric:
    def __init__(self, device: str, i3d_path: str, frames: int):
        self.device = torch.device(device)
        self.frames = int(frames)
        path = Path(i3d_path)
        if not path.is_file():
            raise FileNotFoundError(f"FVD I3D TorchScript model not found: {path}")
        self.model = torch.jit.load(str(path), map_location=self.device).eval()

    def extract(self, video: np.ndarray) -> np.ndarray:
        if len(video) == 0:
            raise ValueError("Cannot extract FVD feature from an empty video.")
        video = uniform_sample_video(frames_to_float(video), self.frames)
        tensor = torch.from_numpy(video).permute(0, 3, 1, 2).contiguous().float()
        tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
        tensor = 2.0 * tensor - 1.0
        with torch.inference_mode():
            feature = self.model(tensor, rescale=False, resize=False, return_features=True)
        return feature.squeeze(0).detach().cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute PSNR/SSIM/LPIPS/FID/FVD for generated wrist-target videos. "
            "Predictions are matched to metadata by episode index and clipstart, "
            "or by `val_000_ep123.mp4` / `0000000.mp4` row index."
        )
    )
    parser.add_argument(
        "--pred-dir",
        default="/home/xuehao/xh/projects/DiffSynth-Studio_v2/Ckpt/clip_traj_iter_000000",
        help="Directory containing prediction videos.",
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
        default="psnr,ssim,lpips,fid,fvd,mse",
        help="Comma-separated metrics to compute. Supported: psnr,ssim,lpips,fid,fvd,mse.",
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
        "--pred-pad-mode",
        default="none",
        choices=["none", "repeat_last"],
        help=(
            "How to handle prediction videos shorter than the requested frame scope. "
            "`none` keeps the old behavior and evaluates only decoded prediction frames; "
            "`repeat_last` pads predictions in memory by repeating the last decoded frame."
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
        "--eval-size",
        default=None,
        help=(
            "Optional metric resolution as WIDTHxHEIGHT, e.g. 320x180. "
            "After GT/pred are aligned by --resize-mode, both are resized to this size."
        ),
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
        "--fid-batch-size",
        type=int,
        default=64,
        help="Frame batch size for InceptionV3 FID features.",
    )
    parser.add_argument(
        "--fid-dims",
        type=int,
        default=2048,
        help="Inception feature dimension for FID. Standard video/image papers use 2048.",
    )
    parser.add_argument(
        "--fvd-i3d-path",
        default=(
            str(Path(__file__).resolve().parents[1] / "diffsynth/core/metric/i3d_torchscript.pt")
        ),
        help="TorchScript I3D model used for standard FVD features.",
    )
    parser.add_argument(
        "--fvd-frames",
        type=int,
        default=16,
        help="Number of uniformly sampled frames per video for FVD.",
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
    supported = {"psnr", "ssim", "lpips", "fid", "fvd", "mse"}
    unknown = metrics - supported
    if unknown:
        raise ValueError(f"Unsupported metrics: {sorted(unknown)}. Supported: {sorted(supported)}")
    if not metrics:
        raise ValueError("At least one metric must be requested.")
    return metrics


def parse_eval_size(size_arg: Optional[str]) -> Optional[tuple[int, int]]:
    if size_arg is None:
        return None
    match = re.fullmatch(r"(?P<width>\d+)x(?P<height>\d+)", size_arg.strip().lower())
    if match is None:
        raise ValueError(f"--eval-size must be formatted as WIDTHxHEIGHT, got {size_arg!r}")
    width = int(match.group("width"))
    height = int(match.group("height"))
    if width <= 0 or height <= 0:
        raise ValueError(f"--eval-size dimensions must be positive, got {size_arg!r}")
    return width, height


def parse_pred_name(path: Path) -> PredInfo:
    match = PRED_NAME_RE.match(path.name)
    if match is not None:
        return PredInfo(
            episode_index=int(match.group("episode")),
            clipstart=int(match.group("clipstart")),
            source_view=match.group("source_view"),
            source_frame=int(match.group("source_frame")),
            row_index=None,
        )

    match = SIMPLE_PRED_NAME_RE.match(path.name)
    if match is not None:
        return PredInfo(
            episode_index=int(match.group("episode")),
            clipstart=int(match.group("clipstart")),
            source_view=None,
            source_frame=None,
            row_index=None,
        )

    match = VAL_PRED_NAME_RE.match(path.name)
    if match is not None:
        return PredInfo(
            episode_index=int(match.group("episode")),
            clipstart=-1,
            source_view=None,
            source_frame=None,
            row_index=int(match.group("row_index")),
        )

    match = INDEX_PRED_NAME_RE.match(path.name)
    if match is not None:
        return PredInfo(
            episode_index=-1,
            clipstart=-1,
            source_view=None,
            source_frame=None,
            row_index=int(match.group("row_index")),
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


def load_metadata_by_line(meta_jsonl: Path) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    with meta_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows.append((line_no, json.loads(line)))
    return rows


@lru_cache(maxsize=4096)
def video_decoder_name(path: str) -> Optional[str]:
    try:
        import av  # type: ignore
    except ModuleNotFoundError:
        return None

    try:
        with av.open(path) as container:
            if not container.streams.video:
                return None
            name = container.streams.video[0].codec_context.name
    except Exception:
        return None

    return str(name).lower() if name else None


def should_use_pyav(path: Path) -> bool:
    decoder_name = video_decoder_name(str(path))
    return decoder_name in PYAV_PREFERRED_DECODERS


def read_video_rgb_cv2(path: Path) -> np.ndarray:
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


def read_video_rgb_pyav(path: Path) -> np.ndarray:
    try:
        import av  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyAV is required to decode this video. Install it with `pip install av` "
            "or `conda install -c conda-forge av`."
        ) from exc

    frames = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise RuntimeError(f"No video stream found in: {path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))

    if not frames:
        raise RuntimeError(f"No frames decoded from video with PyAV: {path}")
    return np.stack(frames, axis=0)


def read_video_rgb(path: Path) -> np.ndarray:
    if should_use_pyav(path):
        return read_video_rgb_pyav(path)

    try:
        return read_video_rgb_cv2(path)
    except RuntimeError as cv2_exc:
        try:
            return read_video_rgb_pyav(path)
        except Exception as pyav_exc:
            raise RuntimeError(
                f"Failed to decode video {path}. OpenCV error: {cv2_exc!r}; "
                f"PyAV fallback error: {pyav_exc!r}"
            ) from pyav_exc


def pad_segment_frames(
    frames: list[np.ndarray],
    output_frames: int,
    pad_mode: str,
    context: str,
) -> np.ndarray:
    if not frames:
        raise RuntimeError(context)

    if len(frames) < output_frames:
        if pad_mode != "repeat_last":
            raise ValueError(f"Unsupported pad_mode={pad_mode!r}; only repeat_last is supported")
        last = frames[-1]
        frames.extend([last.copy() for _ in range(output_frames - len(frames))])

    return np.stack(frames[:output_frames], axis=0)


def read_video_segment_rgb_cv2(
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

    return pad_segment_frames(
        frames,
        output_frames=output_frames,
        pad_mode=pad_mode,
        context=f"No GT frames decoded from {path} starting at frame {start_frame}",
    )


def read_video_segment_rgb_pyav(
    path: Path,
    start_frame: int,
    valid_frames: int,
    output_frames: int,
    pad_mode: str,
) -> np.ndarray:
    try:
        import av  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyAV is required to decode this video. Install it with `pip install av` "
            "or `conda install -c conda-forge av`."
        ) from exc

    frames = []
    start_frame = int(start_frame)
    stop_frame = start_frame + int(valid_frames)
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise RuntimeError(f"No video stream found in: {path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index < start_frame:
                continue
            if frame_index >= stop_frame:
                break
            frames.append(frame.to_ndarray(format="rgb24"))

    return pad_segment_frames(
        frames,
        output_frames=output_frames,
        pad_mode=pad_mode,
        context=f"No GT frames decoded from {path} starting at frame {start_frame}",
    )


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

    if should_use_pyav(path):
        return read_video_segment_rgb_pyav(
            path=path,
            start_frame=start_frame,
            valid_frames=valid_frames,
            output_frames=output_frames,
            pad_mode=pad_mode,
        )

    try:
        return read_video_segment_rgb_cv2(
            path=path,
            start_frame=start_frame,
            valid_frames=valid_frames,
            output_frames=output_frames,
            pad_mode=pad_mode,
        )
    except RuntimeError as cv2_exc:
        try:
            return read_video_segment_rgb_pyav(
                path=path,
                start_frame=start_frame,
                valid_frames=valid_frames,
                output_frames=output_frames,
                pad_mode=pad_mode,
            )
        except Exception as pyav_exc:
            raise RuntimeError(
                f"Failed to decode GT video segment {path} from frame {start_frame}. "
                f"OpenCV error: {cv2_exc!r}; PyAV fallback error: {pyav_exc!r}"
            ) from pyav_exc


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


def pad_video_to_length(video: np.ndarray, output_frames: int, pad_mode: str) -> np.ndarray:
    if output_frames <= 0:
        raise ValueError(f"output_frames must be positive, got {output_frames}")
    if len(video) == 0:
        raise ValueError("Cannot pad an empty video.")
    if len(video) >= output_frames:
        return video[:output_frames]
    if pad_mode != "repeat_last":
        raise ValueError(f"Unsupported pred_pad_mode={pad_mode!r}; use repeat_last to pad predictions")
    pad_count = int(output_frames) - len(video)
    padding = np.repeat(video[-1:,...], pad_count, axis=0)
    return np.concatenate([video, padding], axis=0)


def frames_to_float(frames: np.ndarray) -> np.ndarray:
    return frames.astype(np.float32) / 255.0


def frames_to_lpips_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(frames_to_float(frames)).permute(0, 3, 1, 2).contiguous()
    return tensor.to(device=device, dtype=torch.float32) * 2.0 - 1.0


def uniform_sample_video(video: np.ndarray, num_frames: int) -> np.ndarray:
    if len(video) == 0:
        raise ValueError("Cannot sample an empty video.")
    if len(video) == num_frames:
        return video
    indices = np.linspace(0, len(video) - 1, num=int(num_frames))
    indices = np.clip(np.round(indices).astype(int), 0, len(video) - 1)
    return video[indices]


def compute_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D features, got {features.shape}")
    mu = np.mean(features, axis=0)
    if features.shape[0] == 1:
        sigma = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    else:
        sigma = np.cov(features, rowvar=False)
    return mu, np.atleast_2d(sigma)


def frechet_distance(
    features_a: list[np.ndarray],
    features_b: list[np.ndarray],
    eps: float = 1e-6,
) -> Optional[float]:
    if not features_a or not features_b:
        return None
    feats_a = np.concatenate([np.atleast_2d(x) for x in features_a], axis=0)
    feats_b = np.concatenate([np.atleast_2d(x) for x in features_b], axis=0)
    mu1, sigma1 = compute_stats(feats_a)
    mu2, sigma2 = compute_stats(feats_b)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0], dtype=np.float64) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return float(value)


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


def summarize_samples(
    samples: list[dict[str, Any]],
    metrics: set[str],
    fid: Optional[float] = None,
    fvd: Optional[float] = None,
) -> dict[str, Any]:
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

    if "fid" in metrics:
        summary["overall"]["fid"] = fid
        summary["per_video_mean"]["fid"] = None
    if "fvd" in metrics:
        summary["overall"]["fvd"] = fvd
        summary["per_video_mean"]["fvd"] = None

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
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    target_view = int(row["target_view"] if args.target_view is None else args.target_view)
    target_meta = row["video"][target_view]
    pred_video = read_video_rgb(pred_path)
    raw_pred_size = [int(pred_video.shape[2]), int(pred_video.shape[1])]
    pred_decoded_frames = int(len(pred_video))

    if args.frame_scope == "valid":
        requested_frames = int(row.get("valid_frames", target_meta.get("pad_to_frames", len(pred_video))))
    else:
        requested_frames = int(target_meta.get("pad_to_frames", row.get("length", len(pred_video))))

    if args.pred_pad_mode == "none":
        desired_frames = min(requested_frames, len(pred_video))
    else:
        desired_frames = requested_frames
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

    pred_video = pad_video_to_length(pred_video, desired_frames, args.pred_pad_mode)
    pred_padded_frames = max(0, int(desired_frames) - pred_decoded_frames)
    if args.frame_start:
        gt_video = gt_video[args.frame_start :]
        pred_video = pred_video[args.frame_start :]

    gt_video, pred_video = align_video_sizes(gt_video, pred_video, args.resize_mode)
    aligned_size = [int(pred_video.shape[2]), int(pred_video.shape[1])]
    eval_size = args.eval_size_tuple
    if eval_size is not None:
        eval_w, eval_h = eval_size
        gt_video = resize_video(gt_video, width=eval_w, height=eval_h)
        pred_video = resize_video(pred_video, width=eval_w, height=eval_h)
    sample_metrics = compute_basic_metrics(gt_video, pred_video, metrics)
    if "lpips" in metrics:
        if lpips_metric is None:
            raise RuntimeError("LPIPS metric was requested but not initialized.")
        sample_metrics["lpips"] = lpips_metric.compute(gt_video, pred_video)

    sample = {
        "prediction": str(pred_path),
        "meta_line": int(meta_line),
        "episode_index": int(row["episode_index"]),
        "clipstart": int(row["start_frame"]),
        "target_view": target_view,
        "target_video": str(target_meta["data"]),
        "target_start_frame": int(target_meta["start_frame"]),
        "target_end_frame": int(target_meta["end_frame"]),
        "valid_frames": int(row.get("valid_frames", 0)),
        "requested_frames": int(requested_frames),
        "pred_decoded_frames": pred_decoded_frames,
        "pred_padded_frames": pred_padded_frames,
        "pred_pad_mode": args.pred_pad_mode,
        "frames": int(len(pred_video)),
        "raw_pred_size": raw_pred_size,
        "aligned_size": aligned_size,
        "eval_size": [int(pred_video.shape[2]), int(pred_video.shape[1])],
        "source_view_from_name": pred_info.source_view,
        "source_frame_from_name": pred_info.source_frame,
        "row_index_from_name": pred_info.row_index,
        "metrics": sample_metrics,
    }
    return sample, gt_video, pred_video


def main() -> None:
    args = parse_args()
    args.eval_size_tuple = parse_eval_size(args.eval_size)
    pred_dir = Path(args.pred_dir)
    meta_jsonl = Path(args.meta_jsonl)
    output_json = Path(args.output_json) if args.output_json else pred_dir / "wrist_target_metrics.json"
    metrics = parse_metrics(args.metrics)

    rows = load_metadata(meta_jsonl)
    rows_by_line = load_metadata_by_line(meta_jsonl)
    pred_paths = sorted(
        {
            *pred_dir.glob("*_pred.mp4"),
            *pred_dir.glob("val_*.mp4"),
            *[path for path in pred_dir.glob("*.mp4") if INDEX_PRED_NAME_RE.match(path.name)],
        }
    )
    if args.sample_limit is not None:
        pred_paths = pred_paths[: int(args.sample_limit)]
    if not pred_paths:
        raise RuntimeError(f"No supported prediction mp4 files found in {pred_dir}")

    lpips_metric = None
    if "lpips" in metrics:
        lpips_metric = LPIPSMetric(
            backend=args.lpips_backend,
            device=args.device,
            batch_size=args.lpips_batch_size,
            net=args.lpips_net,
        )
    fid_metric = None
    if "fid" in metrics:
        fid_metric = FIDMetric(
            device=args.device,
            batch_size=args.fid_batch_size,
            dims=args.fid_dims,
        )
    fvd_metric = None
    if "fvd" in metrics:
        fvd_metric = FVDMetric(
            device=args.device,
            i3d_path=args.fvd_i3d_path,
            frames=args.fvd_frames,
        )

    samples = []
    skipped = []
    fid_gt_features: list[np.ndarray] = []
    fid_pred_features: list[np.ndarray] = []
    fvd_gt_features: list[np.ndarray] = []
    fvd_pred_features: list[np.ndarray] = []
    for pred_path in tqdm(pred_paths, desc="Evaluating wrist target videos"):
        try:
            pred_info = parse_pred_name(pred_path)
            if pred_info.row_index is not None:
                row_index = int(pred_info.row_index)
                if row_index < 0 or row_index >= len(rows_by_line):
                    raise IndexError(
                        f"Prediction row index {row_index} out of range for {len(rows_by_line)} metadata rows"
                    )
                meta_line, row = rows_by_line[row_index]
                if pred_info.episode_index >= 0 and int(row["episode_index"]) != pred_info.episode_index:
                    raise ValueError(
                        f"Prediction episode {pred_info.episode_index} does not match "
                        f"metadata row {row_index} episode {row['episode_index']}"
                    )
                pred_info = PredInfo(
                    episode_index=int(row["episode_index"]),
                    clipstart=int(row["start_frame"]),
                    source_view=pred_info.source_view,
                    source_frame=pred_info.source_frame,
                    row_index=pred_info.row_index,
                )
            else:
                key = (pred_info.episode_index, pred_info.clipstart)
                if key not in rows:
                    raise KeyError(f"No metadata row for prediction key {key}")
                meta_line, row = rows[key]
            sample, gt_video, pred_video = evaluate_one(
                pred_path=pred_path,
                pred_info=pred_info,
                meta_line=meta_line,
                row=row,
                args=args,
                metrics=metrics,
                lpips_metric=lpips_metric,
            )
            samples.append(sample)
            if fid_metric is not None:
                fid_gt_features.append(fid_metric.extract(gt_video))
                fid_pred_features.append(fid_metric.extract(pred_video))
            if fvd_metric is not None:
                fvd_gt_features.append(fvd_metric.extract(gt_video))
                fvd_pred_features.append(fvd_metric.extract(pred_video))
        except Exception as exc:
            if args.strict:
                raise
            skipped.append({"prediction": str(pred_path), "reason": repr(exc)})

    if not samples:
        raise RuntimeError("No samples were successfully evaluated.")

    fid_value = frechet_distance(fid_gt_features, fid_pred_features) if "fid" in metrics else None
    fvd_value = frechet_distance(fvd_gt_features, fvd_pred_features) if "fvd" in metrics else None

    payload = {
        "config": {
            "pred_dir": str(pred_dir),
            "meta_jsonl": str(meta_jsonl),
            "metrics": sorted(metrics),
            "target_view": args.target_view,
            "frame_scope": args.frame_scope,
            "pred_pad_mode": args.pred_pad_mode,
            "frame_start": int(args.frame_start),
            "resize_mode": args.resize_mode,
            "eval_size": list(args.eval_size_tuple) if args.eval_size_tuple is not None else None,
            "device": args.device if "lpips" in metrics else None,
            "lpips_net": args.lpips_net if "lpips" in metrics else None,
            "lpips_backend_requested": args.lpips_backend if "lpips" in metrics else None,
            "lpips_backend_used": lpips_metric.name if lpips_metric is not None else None,
            "lpips_batch_size": int(args.lpips_batch_size) if "lpips" in metrics else None,
            "fid_dims": int(args.fid_dims) if "fid" in metrics else None,
            "fid_batch_size": int(args.fid_batch_size) if "fid" in metrics else None,
            "fvd_i3d_path": str(args.fvd_i3d_path) if "fvd" in metrics else None,
            "fvd_frames": int(args.fvd_frames) if "fvd" in metrics else None,
            "sample_limit": args.sample_limit,
        },
        "summary": summarize_samples(samples, metrics, fid=fid_value, fvd=fvd_value),
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
            for name in ["psnr", "ssim", "lpips", "fid", "fvd", "mse"]
            if name in summary
        )
    )
    if skipped:
        print(f"Skipped samples: {len(skipped)}")


if __name__ == "__main__":
    main()
