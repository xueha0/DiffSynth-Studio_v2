import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.nn.functional import adaptive_avg_pool2d, interpolate
from tqdm import tqdm

try:
    import lpips
except ModuleNotFoundError:
    lpips = None

try:
    from decord import VideoReader, cpu
except ModuleNotFoundError:
    VideoReader = None
    cpu = None

try:
    from pytorch_fid.inception import InceptionV3
except ModuleNotFoundError:
    InceptionV3 = None


FID_DIMS = 2048
BATCH_SIZE = 512
DEFAULT_VIEW_NAMES = [
    "cam_high_rgb",
    "cam_left_wrist_rgb",
    "cam_right_wrist_rgb",
]
PBENCH_METRICS = (
    "aesthetic_quality",
    "imaging_quality",
    "motion_smoothness",
    "background_consistency",
    "subject_consistency",
    "overall_consistency",
    "i2v_background",
    "i2v_subject",
)


@dataclass(frozen=True)
class PreparedViewSample:
    video_path: str
    video_stem: str
    split_name: Optional[str]
    episode_index: Optional[int]
    prompt: Optional[str]
    view_name: str
    view_index: int
    gt_video: np.ndarray
    pred_video: np.ndarray
    frames: int


def _configure_local_torch_hub() -> None:
    metric_dir = Path(__file__).resolve().parent
    torch_hub_dir = metric_dir / "torch_hub"
    if torch_hub_dir.is_dir():
        torch.hub.set_dir(str(torch_hub_dir))


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _parse_epoch_num(path: Path) -> int:
    match = re.search(r"epoch[-_]?(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _resolve_stage2_eval_root(input_path: str) -> Optional[Path]:
    path = Path(input_path).resolve()

    if path.is_dir() and (path / "comparisons").is_dir() and (path / "config_eval.json").is_file():
        return path

    if path.is_dir() and path.name.startswith("epoch-") and (path / "stage2_eval").is_dir():
        return path / "stage2_eval"

    for candidate in [path] + list(path.parents):
        if candidate.is_dir() and (candidate / "comparisons").is_dir() and (candidate / "config_eval.json").is_file():
            return candidate

    if path.is_dir():
        candidates = sorted(
            [
                candidate / "stage2_eval"
                for candidate in path.iterdir()
                if candidate.is_dir()
                and candidate.name.startswith("epoch-")
                and (candidate / "stage2_eval").is_dir()
            ],
            key=lambda candidate: _parse_epoch_num(candidate.parent),
        )
        if candidates:
            return candidates[-1]

    return None


def _resolve_comparison_dir(input_path: str) -> Path:
    path = Path(input_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {path}")

    if any(path.glob("*.mp4")):
        return path

    if (path / "comparisons").is_dir():
        return path / "comparisons"

    stage2_eval_root = _resolve_stage2_eval_root(str(path))
    if stage2_eval_root is not None:
        return stage2_eval_root / "comparisons"

    if any(path.rglob("*.mp4")):
        return path

    raise RuntimeError(f"No comparison videos found under {path}")


def _infer_split_name(video_path: Path, comparison_dir: Path) -> Optional[str]:
    try:
        relative = video_path.relative_to(comparison_dir)
    except ValueError:
        relative = video_path
    if len(relative.parts) >= 2:
        return relative.parts[0]
    if comparison_dir.name in {"val", "train_preview"}:
        return comparison_dir.name
    parent_name = video_path.parent.name
    if parent_name in {"val", "train_preview"}:
        return parent_name
    return None


def _parse_episode_index(video_stem: str) -> Optional[int]:
    match = re.search(r"_ep(\d+)$", video_stem)
    return int(match.group(1)) if match else None


def _build_prompt_lookup(stage2_eval_root: Optional[Path]) -> Dict[Tuple[str, int], str]:
    if stage2_eval_root is None:
        return {}

    config_path = stage2_eval_root / "config_eval.json"
    if not config_path.is_file():
        return {}

    config = _load_json(config_path)
    metadata_sources = [
        ("val", config.get("dataset_metadata_path")),
        ("train_preview", config.get("train_metadata_path")),
    ]
    prompt_lookup: Dict[Tuple[str, int], str] = {}
    for split_name, metadata_path in metadata_sources:
        if not metadata_path:
            continue
        for record in _load_jsonl(Path(str(metadata_path))):
            episode_index = record.get("episode_index")
            prompt = record.get("prompt_en") or record.get("prompt")
            if episode_index is None or not prompt:
                continue
            prompt_lookup[(split_name, int(episode_index))] = str(prompt)
    return prompt_lookup


def _list_comparison_videos(comparison_dir: Path) -> List[str]:
    return sorted(str(path.resolve()) for path in comparison_dir.rglob("*.mp4"))


def read_video_decord(path: str) -> np.ndarray:
    if VideoReader is not None and cpu is not None:
        reader = VideoReader(path, ctx=cpu(0))
        frames = reader.get_batch(range(len(reader)))
        return frames.asnumpy().astype(np.float32) / 255.0

    reader = imageio.get_reader(path)
    try:
        frames = [frame.astype(np.float32) / 255.0 for frame in reader]
    finally:
        reader.close()
    return np.asarray(frames, dtype=np.float32)


def split_comparison_grid(frames: np.ndarray, rows: int = 3, cols: int = 2) -> List[List[np.ndarray]]:
    height = frames.shape[1]
    width = frames.shape[2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid comparison video size: {height}x{width}")

    row_splits = np.array_split(frames, rows, axis=1)
    grid: List[List[np.ndarray]] = []
    for row in row_splits:
        col_splits = np.array_split(row, cols, axis=2)
        if len(col_splits) != cols:
            raise ValueError(f"Failed to split comparison video into {cols} columns")
        grid.append(col_splits)
    if len(grid) != rows:
        raise ValueError(f"Failed to split comparison video into {rows} rows")
    return grid


def _crop_to_common_size(gt_frame: np.ndarray, pred_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    min_h = min(gt_frame.shape[0], pred_frame.shape[0])
    min_w = min(gt_frame.shape[1], pred_frame.shape[1])
    if min_h <= 0 or min_w <= 0:
        raise ValueError("Invalid frame size after cropping")
    return gt_frame[:min_h, :min_w], pred_frame[:min_h, :min_w]


def _safe_float(value, default: float = -1.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def _to_uint8_frames(frames: np.ndarray) -> np.ndarray:
    array = np.asarray(frames)
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    else:
        array = np.clip(array, 0, 255)
    return array.astype(np.uint8)


def _write_mp4(video_path: str, frames: np.ndarray, fps: int = 8) -> None:
    frames = _to_uint8_frames(frames)
    with imageio.get_writer(video_path, fps=fps, quality=5) as writer:
        for frame in frames:
            writer.append_data(frame)


def _write_image(image_path: str, frame: np.ndarray) -> None:
    imageio.imwrite(image_path, _to_uint8_frames(frame))


def _prepare_samples(
    comparison_dir: Path,
    num_views: int,
    frame_start: int,
    prompt_lookup: Dict[Tuple[str, int], str],
) -> List[PreparedViewSample]:
    video_files = _list_comparison_videos(comparison_dir)
    if not video_files:
        raise RuntimeError(f"No comparison videos found in {comparison_dir}")

    view_names = (
        list(DEFAULT_VIEW_NAMES)
        if num_views == len(DEFAULT_VIEW_NAMES)
        else [f"view_{view_idx + 1}" for view_idx in range(num_views)]
    )

    prepared_samples: List[PreparedViewSample] = []
    for video_path in tqdm(video_files, desc="Preparing comparison videos ...", leave=False):
        frames = read_video_decord(video_path)
        grid = split_comparison_grid(frames, rows=num_views, cols=2)
        video_stem = Path(video_path).stem
        episode_index = _parse_episode_index(video_stem)
        split_name = _infer_split_name(Path(video_path), comparison_dir)
        prompt = None
        if split_name is not None and episode_index is not None:
            prompt = prompt_lookup.get((split_name, episode_index))

        for view_idx, row in enumerate(grid[:num_views]):
            gt_video = row[0]
            pred_video = row[1]
            usable_frames = min(gt_video.shape[0], pred_video.shape[0])
            if usable_frames <= frame_start:
                continue

            gt_video = gt_video[frame_start:usable_frames]
            pred_video = pred_video[frame_start:usable_frames]
            min_h = min(gt_video.shape[1], pred_video.shape[1])
            min_w = min(gt_video.shape[2], pred_video.shape[2])
            if min_h <= 0 or min_w <= 0:
                continue

            prepared_samples.append(
                PreparedViewSample(
                    video_path=video_path,
                    video_stem=video_stem,
                    split_name=split_name,
                    episode_index=episode_index,
                    prompt=prompt,
                    view_name=view_names[view_idx],
                    view_index=view_idx,
                    gt_video=gt_video[:, :min_h, :min_w],
                    pred_video=pred_video[:, :min_h, :min_w],
                    frames=int(gt_video.shape[0]),
                )
            )

    if not prepared_samples:
        raise RuntimeError("No valid view pairs found in comparison videos.")
    return prepared_samples


def _compute_metrics_for_pair(gt_video: np.ndarray, pred_video: np.ndarray) -> Dict[str, float]:
    usable_frames = min(gt_video.shape[0], pred_video.shape[0])
    if usable_frames <= 0:
        raise RuntimeError("No frames found in comparison video")

    psnr_sum = 0.0
    ssim_sum = 0.0
    mse_sum = 0.0
    for frame_idx in range(usable_frames):
        gt_frame, pred_frame = _crop_to_common_size(gt_video[frame_idx], pred_video[frame_idx])
        psnr_sum += peak_signal_noise_ratio(pred_frame, gt_frame, data_range=1.0)
        ssim_sum += structural_similarity(pred_frame, gt_frame, channel_axis=-1, data_range=1.0)
        mse_sum += float(np.mean((pred_frame - gt_frame) ** 2))

    return {
        "frames": float(usable_frames),
        "psnr_sum": float(psnr_sum),
        "ssim_sum": float(ssim_sum),
        "mse_sum": float(mse_sum),
    }


def _compute_basic_metrics(
    prepared_samples: List[PreparedViewSample],
    num_views: int,
    device: torch.device,
    num_workers: int,
) -> Dict:
    lpips_available = lpips is not None
    view_totals = [
        {"psnr_sum": 0.0, "ssim_sum": 0.0, "mse_sum": 0.0, "frames": 0, "lpips_sum": 0.0, "lpips_frames": 0}
        for _ in range(num_views)
    ]

    max_workers = max(1, int(num_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {
            executor.submit(_compute_metrics_for_pair, sample.gt_video, sample.pred_video): sample
            for sample in prepared_samples
        }
        for future in tqdm(as_completed(future_to_sample), total=len(future_to_sample), desc="Computing PSNR/SSIM/MSE ..."):
            sample = future_to_sample[future]
            sample_metrics = future.result()
            totals = view_totals[sample.view_index]
            totals["psnr_sum"] += sample_metrics["psnr_sum"]
            totals["ssim_sum"] += sample_metrics["ssim_sum"]
            totals["mse_sum"] += sample_metrics["mse_sum"]
            totals["frames"] += int(sample_metrics["frames"])

    if lpips_available:
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
        for sample in tqdm(prepared_samples, desc="Computing LPIPS ...", leave=False):
            gt_tensor = torch.from_numpy(sample.gt_video).permute(0, 3, 1, 2).contiguous().float()
            pred_tensor = torch.from_numpy(sample.pred_video).permute(0, 3, 1, 2).contiguous().float()
            with torch.no_grad():
                for start in range(0, gt_tensor.shape[0], BATCH_SIZE):
                    gt_batch = gt_tensor[start:start + BATCH_SIZE].to(device)
                    pred_batch = pred_tensor[start:start + BATCH_SIZE].to(device)
                    lpips_batch = lpips_model(pred_batch, gt_batch, normalize=True)
                    view_totals[sample.view_index]["lpips_sum"] += float(lpips_batch.sum().item())
                    view_totals[sample.view_index]["lpips_frames"] += int(lpips_batch.shape[0])

    total_frames = sum(item["frames"] for item in view_totals)
    total_lpips_frames = sum(item["lpips_frames"] for item in view_totals)

    view_metrics = []
    for view_idx in range(num_views):
        totals = view_totals[view_idx]
        frames = int(totals["frames"])
        lpips_frames = int(totals["lpips_frames"])
        view_metrics.append(
            {
                "psnr": totals["psnr_sum"] / frames if frames > 0 else 0.0,
                "ssim": totals["ssim_sum"] / frames if frames > 0 else 0.0,
                "mse": totals["mse_sum"] / frames if frames > 0 else 0.0,
                "lpips": totals["lpips_sum"] / lpips_frames if lpips_frames > 0 else -1.0,
                "frames": frames,
            }
        )

    return {
        "avg_psnr": sum(item["psnr_sum"] for item in view_totals) / total_frames if total_frames > 0 else 0.0,
        "avg_ssim": sum(item["ssim_sum"] for item in view_totals) / total_frames if total_frames > 0 else 0.0,
        "avg_mse": sum(item["mse_sum"] for item in view_totals) / total_frames if total_frames > 0 else 0.0,
        "avg_lpips": sum(item["lpips_sum"] for item in view_totals) / total_lpips_frames if total_lpips_frames > 0 else -1.0,
        "lpips_skipped_reason": None if lpips_available else "lpips is not installed",
        "view_metrics": view_metrics,
    }


def _compute_stats(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if features.shape[0] <= 0:
        raise RuntimeError("No features to compute statistics")
    mu = np.mean(features, axis=0)
    if features.shape[0] == 1:
        sigma = np.eye(features.shape[1], dtype=np.float64) * 1e-6
    else:
        sigma = np.cov(features, rowvar=False)
    return mu, sigma


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0], dtype=np.float64) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def _extract_inception_features(frames: np.ndarray, model: InceptionV3, device: torch.device) -> np.ndarray:
    if frames.shape[0] <= 0:
        return np.empty((0, FID_DIMS), dtype=np.float32)

    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float()
    activations = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], BATCH_SIZE):
            batch = tensor[start:start + BATCH_SIZE].to(device)
            pred = model(batch)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            activations.append(pred)
    if not activations:
        return np.empty((0, FID_DIMS), dtype=np.float32)
    return np.concatenate(activations, axis=0)


def _compute_fid_target(target_samples: List[PreparedViewSample], device: torch.device) -> float:
    if not target_samples:
        return -1.0
    if InceptionV3 is None:
        return -1.0

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FID_DIMS]
    model = InceptionV3([block_idx]).to(device).eval()

    gt_acts = []
    pred_acts = []
    for sample in tqdm(target_samples, desc="Preparing FID features ...", leave=False):
        gt_feat = _extract_inception_features(sample.gt_video, model, device)
        pred_feat = _extract_inception_features(sample.pred_video, model, device)
        if gt_feat.shape[0] > 0 and pred_feat.shape[0] > 0:
            gt_acts.append(gt_feat)
            pred_acts.append(pred_feat)

    if not gt_acts or not pred_acts:
        return -1.0

    gt_acts = np.concatenate(gt_acts, axis=0)
    pred_acts = np.concatenate(pred_acts, axis=0)
    mu1, sigma1 = _compute_stats(gt_acts)
    mu2, sigma2 = _compute_stats(pred_acts)
    return _frechet_distance(mu1, sigma1, mu2, sigma2)


def _extract_i3d_feature(video: np.ndarray, i3d_model: torch.jit.ScriptModule, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(video).permute(0, 3, 1, 2).contiguous().float()
    tensor = interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
    tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device)
    tensor = 2.0 * tensor - 1.0
    with torch.no_grad():
        feature = i3d_model(tensor, rescale=False, resize=False, return_features=True)
    return feature.squeeze(0).cpu().numpy()


def _compute_fvd_target(
    target_samples: List[PreparedViewSample],
    device: torch.device,
    frame_chunk_size: int,
) -> Tuple[float, int]:
    if not target_samples:
        return -1.0, 0

    min_frames = min(sample.frames for sample in target_samples)
    if min_frames <= 0:
        return -1.0, 0

    clip_length = min(int(frame_chunk_size), int(min_frames)) if frame_chunk_size else int(min_frames)
    if clip_length <= 1:
        return -1.0, clip_length

    i3d_path = Path(__file__).resolve().parent / "i3d_torchscript.pt"
    if not i3d_path.is_file():
        raise FileNotFoundError(f"FVD model not found: {i3d_path}")
    i3d_model = torch.jit.load(str(i3d_path), map_location=device).eval()

    gt_feats = []
    pred_feats = []
    for sample in tqdm(target_samples, desc="Preparing FVD features ...", leave=False):
        gt_feats.append(_extract_i3d_feature(sample.gt_video[:clip_length], i3d_model, device))
        pred_feats.append(_extract_i3d_feature(sample.pred_video[:clip_length], i3d_model, device))

    gt_feats_array = np.asarray(gt_feats, dtype=np.float64)
    pred_feats_array = np.asarray(pred_feats, dtype=np.float64)
    mu1, sigma1 = _compute_stats(gt_feats_array)
    mu2, sigma2 = _compute_stats(pred_feats_array)
    return _frechet_distance(mu1, sigma1, mu2, sigma2), clip_length


def _compute_pbench_metrics(
    target_samples: List[PreparedViewSample],
    frame_chunk_size: int,
    device: torch.device,
) -> Dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="pbench_chunk_eval_") as tmp_dir:
        eval_root = Path(tmp_dir) / "video_quality"
        videos_dir = eval_root / "videos"
        images_dir = eval_root / "condition_images"
        output_dir = eval_root / "evaluation_results"
        videos_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        prompt_entries = []
        chunk_index = 0
        for sample in tqdm(target_samples, desc="Preparing PBench chunks ...", leave=False):
            if not sample.prompt:
                raise RuntimeError(f"Missing prompt for {sample.video_path}")
            for start in range(0, sample.frames, frame_chunk_size):
                pred_chunk = sample.pred_video[start:start + frame_chunk_size]
                if pred_chunk.shape[0] != frame_chunk_size:
                    continue
                sample_id = f"{sample.video_stem}_v{sample.view_index:02d}_c{chunk_index:06d}"
                video_out = videos_dir / f"{sample_id}.mp4"
                image_out = images_dir / f"{sample_id}.jpg"
                _write_mp4(str(video_out), pred_chunk)
                _write_image(str(image_out), sample.gt_video[start])
                prompt_entries.append(
                    {
                        "video_id": sample_id,
                        "prompt": sample.prompt,
                        "prompt_en": sample.prompt,
                        "custom_image_path": str(image_out),
                    }
                )
                chunk_index += 1

        if not prompt_entries:
            raise RuntimeError("No valid chunks prepared for PBench metrics")

        try:
            from .pbench import PBench
        except ImportError:
            from pbench import PBench

        full_json_dir = Path(__file__).resolve().parent / "pbench" / "VBench_full_info.json"
        evaluator = PBench(device, str(full_json_dir), str(output_dir))

        prev_force_single = os.environ.get("PBENCH_FORCE_SINGLE_PROCESS")
        os.environ["PBENCH_FORCE_SINGLE_PROCESS"] = "1"
        try:
            evaluator.evaluate(
                videos_path=str(videos_dir),
                name="results_chunked",
                prompt_list={f"{entry['video_id']}.mp4": entry for entry in prompt_entries},
                dimension_list=list(PBENCH_METRICS),
                local=True,
                read_frame=False,
                mode="custom_input",
                custom_image_folder=str(images_dir),
                enable_missing_videos=True,
            )
        finally:
            if prev_force_single is None:
                os.environ.pop("PBENCH_FORCE_SINGLE_PROCESS", None)
            else:
                os.environ["PBENCH_FORCE_SINGLE_PROCESS"] = prev_force_single

        result_files = [
            path for path in output_dir.iterdir()
            if path.name.startswith("results_") and path.name.endswith("_eval_results.json")
        ]
        if not result_files:
            return {metric_name: -1.0 for metric_name in PBENCH_METRICS}

        raw_results = _load_json(max(result_files, key=lambda path: path.stat().st_mtime))
        parsed = {}
        for metric_name in PBENCH_METRICS:
            metric_result = raw_results.get(metric_name)
            if isinstance(metric_result, list) and metric_result:
                parsed[metric_name] = _safe_float(metric_result[0])
            else:
                parsed[metric_name] = _safe_float(metric_result)
        return parsed


def _compute_extended_metrics(
    prepared_samples: List[PreparedViewSample],
    basic_results: Dict,
    target_view_index: int,
    device: torch.device,
    frame_chunk_size: int,
    enable_fid: bool,
    enable_fvd: bool,
    enable_pbench: bool,
) -> Dict:
    if target_view_index < 0 or target_view_index >= len(basic_results["view_metrics"]):
        raise IndexError(f"target_view_index out of range: {target_view_index}")

    target_samples = [sample for sample in prepared_samples if sample.view_index == target_view_index]
    target_lpips = _safe_float(basic_results["view_metrics"][target_view_index].get("lpips"))

    extended = {
        "scope": "target_view",
        "target_view_index": int(target_view_index),
        "target_view_name": target_samples[0].view_name if target_samples else f"view_{target_view_index + 1}",
        "lpips": target_lpips,
        "lpips_skipped_reason": basic_results.get("lpips_skipped_reason"),
        "fid": -1.0,
        "fid_skipped_reason": None if enable_fid else "disabled",
        "fvd": -1.0,
        "fvd_skipped_reason": None if enable_fvd else "disabled",
        "clip_length": 0,
        "pbench_enabled": bool(enable_pbench),
        "pbench_skipped_reason": "disabled by default for cross-view evaluation",
    }

    if enable_fid:
        if InceptionV3 is None:
            extended["fid_skipped_reason"] = "pytorch_fid is not installed"
        else:
            extended["fid"] = float(_compute_fid_target(target_samples, device))
            extended["fid_skipped_reason"] = None

    if enable_fvd:
        i3d_path = Path(__file__).resolve().parent / "i3d_torchscript.pt"
        if not i3d_path.is_file():
            extended["fvd_skipped_reason"] = f"FVD model not found: {i3d_path}"
        else:
            fvd_score, clip_length = _compute_fvd_target(target_samples, device, frame_chunk_size)
            extended["fvd"] = float(fvd_score)
            extended["clip_length"] = int(clip_length)
            extended["fvd_skipped_reason"] = None

    if enable_pbench:
        missing_prompts = [sample.video_path for sample in target_samples if not sample.prompt]
        if missing_prompts:
            extended["pbench_skipped_reason"] = "prompt mapping unavailable for one or more comparison videos"
        else:
            extended["pbench"] = _compute_pbench_metrics(target_samples, frame_chunk_size, device)
            extended["pbench_skipped_reason"] = None

    return extended


def evaluate(
    output_root: str = None,
    num_workers: int = 64,
    num_views: int = 3,
    frame_start: int = 0,
    target_view_index: Optional[int] = None,
    frame_chunk_size: int = 81,
    enable_fid: bool = True,
    enable_fvd: bool = True,
    enable_pbench: bool = False,
) -> Dict:
    """Evaluate comparison videos and return the legacy split-level structure.

    Args:
        output_root: Comparison directory, `stage2_eval`, `epoch-*`, or checkpoint root.
        num_workers: Worker threads for PSNR/SSIM/MSE.
        num_views: Number of camera views in each comparison video.
        frame_start: Start frame index for evaluation.
        target_view_index: View index used by extended metrics. Defaults to `num_views - 1`.
        frame_chunk_size: Clip length used by FVD and optional PBench.
        enable_fid: Whether to compute FID on the target view.
        enable_fvd: Whether to compute FVD on the target view.
        enable_pbench: Whether to compute PBench metrics on the target view.

    Returns:
        dict with legacy keys `avg_psnr`, `avg_ssim`, `avg_mse`, `view_metrics`,
        plus `avg_lpips` and `extended_metrics`.
    """
    if output_root is None:
        raise ValueError("`output_root` is required")

    _configure_local_torch_hub()
    comparison_dir = _resolve_comparison_dir(str(output_root))
    stage2_eval_root = _resolve_stage2_eval_root(str(output_root))
    prompt_lookup = _build_prompt_lookup(stage2_eval_root)

    num_views = int(num_views)
    frame_start = max(0, int(frame_start))
    frame_chunk_size = max(0, int(frame_chunk_size))
    if target_view_index is None:
        target_view_index = num_views - 1

    prepared_samples = _prepare_samples(
        comparison_dir=comparison_dir,
        num_views=num_views,
        frame_start=frame_start,
        prompt_lookup=prompt_lookup,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    basic_results = _compute_basic_metrics(
        prepared_samples=prepared_samples,
        num_views=num_views,
        device=device,
        num_workers=num_workers,
    )
    basic_results["extended_metrics"] = _compute_extended_metrics(
        prepared_samples=prepared_samples,
        basic_results=basic_results,
        target_view_index=int(target_view_index),
        device=device,
        frame_chunk_size=frame_chunk_size,
        enable_fid=enable_fid,
        enable_fvd=enable_fvd,
        enable_pbench=enable_pbench,
    )
    return basic_results


def _summarize_split_metrics(
    split_dir: Path,
    num_views: int,
    target_view_index: int,
    num_workers: int,
    frame_chunk_size: int,
    enable_fid: bool,
    enable_fvd: bool,
    enable_pbench: bool,
) -> Dict:
    metrics_all = evaluate(
        output_root=str(split_dir),
        num_workers=num_workers,
        num_views=num_views,
        frame_start=0,
        target_view_index=target_view_index,
        frame_chunk_size=frame_chunk_size,
        enable_fid=enable_fid,
        enable_fvd=enable_fvd,
        enable_pbench=enable_pbench,
    )
    metrics_exclude_first = evaluate(
        output_root=str(split_dir),
        num_workers=num_workers,
        num_views=num_views,
        frame_start=1,
        target_view_index=target_view_index,
        frame_chunk_size=frame_chunk_size,
        enable_fid=enable_fid,
        enable_fvd=enable_fvd,
        enable_pbench=enable_pbench,
    )
    return {
        "all_views": metrics_all,
        "all_views_exclude_first": metrics_exclude_first,
        "target_view_index": int(target_view_index),
        "target_view": metrics_all["view_metrics"][target_view_index],
        "target_view_exclude_first": metrics_exclude_first["view_metrics"][target_view_index],
        "target_view_extended": metrics_all.get("extended_metrics"),
        "target_view_exclude_first_extended": metrics_exclude_first.get("extended_metrics"),
        "evaluated_frame_start": {
            "all_views": 0,
            "all_views_exclude_first": 1,
            "target_view_exclude_first": 1,
        },
    }


def _infer_num_views_from_config(config: dict, default: int) -> int:
    target_view = int(config.get("cross_view_target_view", default - 1))
    source_views = []
    raw_source_views = config.get("cross_view_source_views")
    if isinstance(raw_source_views, str):
        source_views = [int(item) for item in raw_source_views.split(",") if item.strip()]
    elif isinstance(raw_source_views, Iterable):
        source_views = [int(item) for item in raw_source_views]
    return max([target_view] + source_views) + 1 if [target_view] + source_views else int(default)


def evaluate_and_write_report(
    output_root: str,
    metrics_output_path: Optional[str] = None,
    num_workers: int = 64,
    num_views: Optional[int] = None,
    target_view_index: Optional[int] = None,
    frame_chunk_size: int = 81,
    enable_fid: bool = True,
    enable_fvd: bool = True,
    enable_pbench: bool = False,
) -> Dict:
    stage2_eval_root = _resolve_stage2_eval_root(str(output_root))
    if stage2_eval_root is None:
        raise RuntimeError(f"Could not resolve a stage2_eval directory from {output_root}")

    config_path = stage2_eval_root / "config_eval.json"
    config = _load_json(config_path) if config_path.is_file() else {}
    resolved_num_views = _infer_num_views_from_config(config, default=num_views or len(DEFAULT_VIEW_NAMES))
    resolved_target_view = int(
        config.get("cross_view_target_view", resolved_num_views - 1)
        if target_view_index is None
        else target_view_index
    )

    comparisons_dir = stage2_eval_root / "comparisons"
    val_dir = comparisons_dir / "val"
    train_dir = comparisons_dir / "train_preview"

    payload = {
        "checkpoint_path": config.get("checkpoint_path"),
        "val_comparison_dir": str(val_dir.resolve()) if val_dir.is_dir() else None,
        "val_metrics": None,
        "train_preview_comparison_dir": str(train_dir.resolve()) if train_dir.is_dir() else None,
        "train_preview_metrics": None,
    }

    if val_dir.is_dir():
        payload["val_metrics"] = _summarize_split_metrics(
            split_dir=val_dir,
            num_views=resolved_num_views,
            target_view_index=resolved_target_view,
            num_workers=num_workers,
            frame_chunk_size=frame_chunk_size,
            enable_fid=enable_fid,
            enable_fvd=enable_fvd,
            enable_pbench=enable_pbench,
        )

    if train_dir.is_dir():
        payload["train_preview_metrics"] = _summarize_split_metrics(
            split_dir=train_dir,
            num_views=resolved_num_views,
            target_view_index=resolved_target_view,
            num_workers=num_workers,
            frame_chunk_size=frame_chunk_size,
            enable_fid=enable_fid,
            enable_fvd=enable_fvd,
            enable_pbench=enable_pbench,
        )

    output_path = Path(metrics_output_path).resolve() if metrics_output_path else (stage2_eval_root / "metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "--output_root", dest="input_path", type=str, required=True)
    parser.add_argument("--metrics_output_path", type=str, default=None)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--num_views", type=int, default=None)
    parser.add_argument("--target_view_index", type=int, default=None)
    parser.add_argument("--frame_chunk_size", type=int, default=81)
    parser.add_argument("--disable_fid", action="store_true")
    parser.add_argument("--disable_fvd", action="store_true")
    parser.add_argument("--enable_pbench", action="store_true")
    args = parser.parse_args()

    report = evaluate_and_write_report(
        output_root=args.input_path,
        metrics_output_path=args.metrics_output_path,
        num_workers=args.workers,
        num_views=args.num_views,
        target_view_index=args.target_view_index,
        frame_chunk_size=args.frame_chunk_size,
        enable_fid=not args.disable_fid,
        enable_fvd=not args.disable_fvd,
        enable_pbench=args.enable_pbench,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
