"""
  cd /data_ywj/data_xh/projects/DiffSynth-Studio_v2

  mkdir -p logs/cache_8shards

  for shard in $(seq 0 7); do
    gpu=$((shard % 8))

    CUDA_VISIBLE_DEVICES=$gpu \
    /env/conda/envs/studio/bin/python tool/build_cross_view_latent_cache.py \
      --dataset_base_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta \
      --train_metadata_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/episodes_cross_view_train_81_small16567.jsonl \
      --val_metadata_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/episodes_cross_view_val_81_small200.jsonl \
      --output_root /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/cache_crossview_81f_180x320_lagernvs_iter060001_new \
      --model_paths /data_ywj/data_xh/projects/datasets/PAI \
      --load_modules dit,text:emb,vae,image,action:noise \
      --state_type state_pose_7d \
      --state_stat_path /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/stat_state_pose_7d.json \
      --height 180 \
      --width 320 \
      --num_frames 81 \
      --num_history_frames 1 \
      --resize_mode fit \
      --cross_view_source_views 0,1 \
      --cross_view_target_view 2 \
      --cross_view_placeholder_mode zeros \
      --device cuda \
      --wrist_first_frame_index /data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta/meta/wrist_first_frame_index_all.json \
      --num_shards 8 \
      --shard_index $shard \
      --shard_mode contiguous \
      --cache_num_workers 1 \
      --skip-existing \
      > logs/cache_8shards/shard_${shard}_gpu_${gpu}.log 2>&1 &
  done

  wait

"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadDroidState, ResolvePromptEmbPath
from diffsynth.diffusion.parsers import prepare_wan_runtime
from examples.wanvideo.model_training.train import WanTrainingModule, set_global_seed
from diffsynth.pipelines.wan_video import (
    WanVideoUnit_ImageEmbedderCLIP,
    WanVideoUnit_ImageEmbedderVAE,
    WanVideoUnit_InputVideoEmbedder,
    WanVideoUnit_NoiseInitializer,
    WanVideoUnit_ShapeChecker,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build offline latent caches for cross-view WAN training."
    )
    parser.add_argument("--dataset_base_path", type=str, required=True)
    parser.add_argument("--train_metadata_path", type=str, required=True)
    parser.add_argument("--val_metadata_path", type=str, default=None)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--model_paths", type=str, required=True)
    parser.add_argument("--load_modules", type=str, default="dit,text:emb,vae,image,action:noise")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None)
    parser.add_argument("--state_type", type=str, default="state_pose_7d")
    parser.add_argument("--state_stat_path", type=str, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_history_frames", type=int, default=1)
    parser.add_argument("--max_pixels", type=int, default=4096 * 4096)
    parser.add_argument("--resize_mode", type=str, default="fit", choices=["crop", "fit"])
    parser.add_argument("--cross_view_source_views", type=str, default="0,1")
    parser.add_argument("--cross_view_target_view", type=int, default=2)
    parser.add_argument(
        "--cross_view_placeholder_mode",
        type=str,
        default="zeros",
        choices=["zeros", "source_mean"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--shard_mode",
        type=str,
        default="strided",
        choices=["strided", "contiguous"],
        help=(
            "How to split samples across shards. `contiguous` preserves episode/video "
            "locality and usually reduces duplicate video decoding."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--cache_num_workers",
        type=int,
        default=0,
        help="Number of CPU workers used to prefetch/decode raw samples. 0 keeps serial loading.",
    )
    parser.add_argument(
        "--cache_prefetch_factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor when --cache_num_workers > 0.",
    )
    parser.add_argument(
        "--cache_pin_memory",
        action="store_true",
        help="Pin prefetched CPU tensors before host-to-device transfer.",
    )
    parser.add_argument(
        "--vae_tiled_encode",
        action="store_true",
        help="Use VAE tiled encoding. Usually slower at 180x320; useful only for memory fallback.",
    )
    parser.add_argument(
        "--vae_tile_size",
        type=str,
        default="34,34",
        help="VAE tile size as H,W in latent-grid units.",
    )
    parser.add_argument(
        "--vae_tile_stride",
        type=str,
        default="18,16",
        help="VAE tile stride as H,W in latent-grid units.",
    )
    parser.add_argument(
        "--skip_legacy_branch",
        action="store_true",
        help=(
            "Do not cache legacy y/clip_feature tensors. Only use with training "
            "--cross_view_disable_legacy_image_branch=1."
        ),
    )
    parser.add_argument(
        "--scene_token_checkpoint",
        type=str,
        default=None,
        help="Path to LagerNVS checkpoint. If provided, scene tokens are pre-extracted and cached.",
    )
    parser.add_argument(
        "--wrist_first_frame_index",
        type=str,
        default=None,
        help=(
            "JSON index mapping (episode_index,start_frame) to wrist first-frame PNG. "
            "If provided, target history and legacy image conditions use the synthesized frame."
        ),
    )
    # ---- Plan A dual-end anchor: encode both head & tail synthesized frames
    # into the cached y channel via WanVideoUnit_ImageEmbedderVAE. The tail
    # frame is taken from the next segment's wrist first frame (DROID stride
    # = expected_stride). Last-segment falls back to a zero placeholder.
    parser.add_argument(
        "--cross_view_use_tail_anchor",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Plan A: when 1, encode wrist[..., -num_tail_frames:] as a known "
            "frame in the cached y channel (mask bits set on tail latent slot)."
        ),
    )
    parser.add_argument(
        "--num_tail_frames",
        type=int,
        default=1,
        help="Number of tail pixel frames marked as known when dual-end anchor is enabled.",
    )
    parser.add_argument(
        "--cross_view_tail_anchor_dropout",
        type=float,
        default=0.0,
        help=(
            "Probability of replacing the next-segment synth frame with a "
            "zero placeholder when building cache (data augmentation; usually "
            "kept at 0 for cache and applied at training time instead)."
        ),
    )
    parser.add_argument(
        "--tail_anchor_segment_stride",
        type=int,
        default=81,
        help=(
            "Frame stride between consecutive segments in DROID. Used to look "
            "up the next-segment wrist first frame at (episode, start_frame + stride)."
        ),
    )
    return parser


def create_dataset(args, metadata_path: str, modules, data_file_keys):
    module_bases = {str(item).partition(":")[0].strip().lower() for item in modules}
    special_operator_map = {}
    if "text" in module_bases and "prompt_emb" in data_file_keys:
        special_operator_map["prompt_emb"] = ResolvePromptEmbPath(
            base_path=args.dataset_base_path
        )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=metadata_path,
        repeat=1,
        data_file_keys=data_file_keys,
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            resize_mode=args.resize_mode,
        ),
        special_operator_map=special_operator_map,
        stat_path=args.state_stat_path,
    )
    if "state" in data_file_keys:
        dataset.special_operator_map["state"] = LoadDroidState(
            base_path=args.dataset_base_path,
            state_type=args.state_type,
            stat=dataset.stat,
            num_frames=args.num_frames,
        )
    if int(args.num_shards) > 1:
        if int(args.num_shards) < 1:
            raise ValueError(f"`num_shards` must be >= 1, got {args.num_shards}.")
        if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
            raise ValueError(
                f"`shard_index` must be in [0, {int(args.num_shards) - 1}], "
                f"got {args.shard_index}."
            )
        if args.shard_mode == "contiguous":
            shard_size = int(math.ceil(len(dataset.data) / int(args.num_shards)))
            start = int(args.shard_index) * shard_size
            end = min(len(dataset.data), start + shard_size)
            shard_indices = list(range(start, end))
        else:
            shard_indices = list(range(int(args.shard_index), len(dataset.data), int(args.num_shards)))
        dataset.data = [dataset.data[index] for index in shard_indices]
        dataset.sample_indices = shard_indices
        print(
            f"Using shard {int(args.shard_index) + 1}/{int(args.num_shards)} "
            f"({args.shard_mode}) with {len(shard_indices)} samples from {metadata_path}"
        )
        dataset._shard_index = int(args.shard_index)
    return dataset


def build_model(args, runtime):
    model = WanTrainingModule(
        model_paths=json.dumps(runtime["model_paths"]),
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=runtime["tokenizer_path"],
        trainable_models=None,
        modules=runtime["modules"],
        device=args.device,
        task="cross_view_stage1",
        num_history_frames=args.num_history_frames,
        cross_view_source_views=args.cross_view_source_views,
        cross_view_target_view=args.cross_view_target_view,
        cross_view_placeholder_mode=args.cross_view_placeholder_mode,
        state_type=args.state_type,
        # Plan A: keep model-instance dual-end flags consistent with CLI so
        # any code path that runs through `model.build_cross_view_condition_video`
        # also sees the tail anchor (currently the cache main loop builds
        # cond_video manually, but this future-proofs against refactors).
        cross_view_use_tail_anchor=int(getattr(args, "cross_view_use_tail_anchor", 0)),
        num_tail_frames=int(getattr(args, "num_tail_frames", 1)),
        # Cache construction always runs in eval mode; tail-anchor dropout is
        # a training-time augmentation and is applied by the cache main loop
        # itself (see cross_view_tail_anchor_dropout handling in the for-loop).
        cross_view_tail_anchor_dropout=0.0,
    )
    model.eval()
    model.requires_grad_(False)
    model.pipe.eval()
    return model


def to_cpu_tensor(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value).detach().cpu()


def _meta_int(raw: dict, key: str):
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.flatten()[0].item()
    return int(value)


def _lookup_wrist_first_frame_path(raw: dict, wrist_ff_index: dict | None):
    if wrist_ff_index is None:
        return None
    episode_index = _meta_int(raw, "episode_index")
    start_frame = _meta_int(raw, "start_frame")
    if episode_index is None or start_frame is None:
        return None
    path = wrist_ff_index.get(f"{episode_index}_{start_frame}")
    if path is None or not os.path.exists(path):
        return None
    return path


def _lookup_wrist_next_segment_first_frame_path(
    raw: dict, wrist_ff_index: dict | None, segment_stride: int = 81,
):
    """Look up the wrist first-frame PNG of the *next* segment in the same
    episode (key = (episode, start_frame + stride)). Returns None when the
    sample is the last segment of an episode or the index is missing.
    """
    if wrist_ff_index is None:
        return None
    episode_index = _meta_int(raw, "episode_index")
    start_frame = _meta_int(raw, "start_frame")
    if episode_index is None or start_frame is None:
        return None
    next_key = f"{episode_index}_{int(start_frame) + int(segment_stride)}"
    path = wrist_ff_index.get(next_key)
    if path is None or not os.path.exists(path):
        return None
    return path


def _parse_int_pair(value: str, name: str) -> tuple[int, int]:
    parts = str(value).replace("x", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"`{name}` must be formatted as H,W, got {value!r}.")
    try:
        parsed = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise ValueError(f"`{name}` must contain integers, got {value!r}.") from exc
    if parsed[0] <= 0 or parsed[1] <= 0:
        raise ValueError(f"`{name}` values must be positive, got {value!r}.")
    return parsed


def _load_first_frame_image_from_path(path, height, width, device, dtype):
    if path is None:
        return None
    from PIL import Image
    import numpy as np

    img = Image.open(path).convert("RGB")
    img = img.resize((width, height), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(
        device=device,
        dtype=dtype,
    )


def _load_first_frame_image(raw, wrist_ff_index, height, width, device, dtype):
    return _load_first_frame_image_from_path(
        _lookup_wrist_first_frame_path(raw, wrist_ff_index),
        height,
        width,
        device,
        dtype,
    )


def _encode_video_latents_by_view(
    model,
    video: torch.Tensor,
    tiled: bool,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
) -> torch.Tensor:
    model.pipe.load_models_to_device(["vae"])
    video = video.to(dtype=model.pipe.torch_dtype, device=model.pipe.device)
    latent_views = model.pipe.vae.encode(
        video,
        device=model.pipe.device,
        tiled=bool(tiled),
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    return latent_views.to(dtype=model.pipe.torch_dtype, device=model.pipe.device)


def _encode_first_frame_tensor_latent(
    model,
    first_frame_tensor,
    num_frames: int,
    height: int,
    width: int,
    tiled: bool,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
):
    if first_frame_tensor is None:
        return None
    video_1view = torch.zeros(
        1,
        3,
        int(num_frames),
        int(height),
        int(width),
        dtype=model.pipe.torch_dtype,
        device=model.pipe.device,
    )
    video_1view[0, :, 0] = first_frame_tensor.squeeze(0)
    return _encode_video_latents_by_view(
        model,
        video_1view,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )


def _target_history_latents_from_legacy_y(
    y: torch.Tensor | None,
    target_view: int,
    num_views: int,
    history_t: int,
    z_dim: int,
) -> torch.Tensor | None:
    if y is None:
        return None
    if y.ndim != 5 or y.shape[1] < 4 + z_dim:
        return None
    if y.shape[-2] % int(num_views) != 0:
        return None
    view_height = y.shape[-2] // int(num_views)
    start = int(target_view) * view_height
    end = start + view_height
    return y[:, 4:4 + z_dim, :history_t, start:end, :].contiguous()


def _build_legacy_image_branch(
    model,
    data,
    cond_video: torch.Tensor,
    tiled: bool,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
    num_tail_frames: int = 0,
) -> dict:
    inputs = model.build_cross_view_inputs(data, cond_video)
    inputs[0]["tiled"] = bool(tiled)
    inputs[0]["tile_size"] = tile_size
    inputs[0]["tile_stride"] = tile_stride
    # Plan A: ensure WanVideoUnit_ImageEmbedderVAE sees num_tail_frames so the
    # cached y channel includes the tail anchor mask + latent. Without this
    # override, build_cross_view_inputs's gated value
    # (cross_view_use_tail_anchor on the model instance) might be 0 when this
    # tool is invoked through a model loaded with use_tail_anchor=False.
    inputs[0]["num_tail_frames"] = int(num_tail_frames or 0)
    for unit in model.pipe.units:
        if isinstance(unit, WanVideoUnit_InputVideoEmbedder):
            continue
        if isinstance(unit, WanVideoUnit_NoiseInitializer):
            continue
        if not isinstance(
            unit,
            (
                WanVideoUnit_ShapeChecker,
                WanVideoUnit_ImageEmbedderVAE,
                WanVideoUnit_ImageEmbedderCLIP,
            ),
        ):
            continue
        inputs = model.pipe.unit_runner(unit, model.pipe, *inputs)
    inputs_shared, _, _ = inputs
    outputs = {}
    if "y" in inputs_shared:
        outputs["y"] = inputs_shared["y"]
    if "clip_feature" in inputs_shared:
        outputs["clip_feature"] = inputs_shared["clip_feature"]
    return outputs


def _build_target_history_latents(
    model,
    video_gt: torch.Tensor,
    first_frame_image: torch.Tensor | None,
    history_t: int,
    tiled: bool,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
) -> torch.Tensor:
    if first_frame_image is not None:
        first_frame_latent = _encode_first_frame_tensor_latent(
            model,
            first_frame_image,
            int(video_gt.shape[2]),
            int(video_gt.shape[-2]),
            int(video_gt.shape[-1]),
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return model.merge_view_latents(first_frame_latent[:, :, :history_t])

    history_video = model.build_target_history_condition_video(video_gt)
    history_latents = _encode_video_latents_by_view(
        model,
        history_video,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    return model.merge_view_latents(history_latents)


def _global_sample_id(dataset, sample_id: int) -> int:
    if getattr(dataset, "sample_indices", None) is not None:
        return int(dataset.sample_indices[sample_id])
    return int(sample_id)


class _CacheItemDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, output_dir: Path):
        self.dataset = dataset
        self.output_dir = output_dir

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, sample_id: int):
        global_sample_id = _global_sample_id(self.dataset, int(sample_id))
        save_path = self.output_dir / f"{global_sample_id:07d}.pth"
        if getattr(self.dataset, "_skip_existing", False) and save_path.is_file():
            return int(sample_id), global_sample_id, str(save_path), None
        return int(sample_id), global_sample_id, str(save_path), self.dataset[sample_id]


def _first_item_collate(batch):
    if len(batch) != 1:
        raise ValueError("Cache DataLoader uses batch_size=1.")
    return batch[0]


def _iter_cache_items(dataset, output_dir: Path, args):
    indexed_dataset = _CacheItemDataset(dataset, output_dir)
    if int(args.cache_num_workers) <= 0:
        for sample_id in range(len(indexed_dataset)):
            yield indexed_dataset[sample_id]
        return

    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": int(args.cache_num_workers),
        "collate_fn": _first_item_collate,
        "pin_memory": bool(args.cache_pin_memory),
        "persistent_workers": True,
        "prefetch_factor": int(args.cache_prefetch_factor),
    }
    loader = torch.utils.data.DataLoader(indexed_dataset, **loader_kwargs)
    yield from loader


@torch.no_grad()
def cache_split(
    model: WanTrainingModule,
    dataset,
    split_name: str,
    output_root: Path,
    args,
    scene_extractor=None,
    wrist_ff_index=None,
):
    output_dir = output_root / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    history_t = ((model.num_history_frames - 1) // 4) + 1
    vae_tile_size = _parse_int_pair(args.vae_tile_size, "vae_tile_size")
    vae_tile_stride = _parse_int_pair(args.vae_tile_stride, "vae_tile_stride")

    progress_desc = f"cache:{split_name}"
    if getattr(dataset, "sample_indices", None) is not None:
        progress_desc = f"cache:{split_name}:shard{int(getattr(dataset, '_shard_index', 0))}"

    iterator = _iter_cache_items(dataset, output_dir, args)
    for _, global_sample_id, save_path_str, raw in tqdm(
        iterator,
        total=len(dataset),
        desc=progress_desc,
    ):
        save_path = Path(save_path_str)
        if raw is None:
            continue

        # Keep metadata/path strings on CPU. Only tensors needed by VAE/CLIP/scene
        # extraction are moved to GPU.
        data = dict(raw)
        data["video"] = model.transfer_data_to_device(
            raw["video"],
            model.pipe.device,
            model.pipe.torch_dtype,
        )
        if raw.get("state") is not None:
            data["state"] = model.transfer_data_to_device(
                raw["state"],
                model.pipe.device,
                model.pipe.torch_dtype,
            )
        if raw.get("action") is not None:
            data["action"] = model.transfer_data_to_device(
                raw["action"],
                model.pipe.device,
                model.pipe.torch_dtype,
            )
        video_gt = data["video"]
        model.validate_cross_view_video(video_gt)

        latent_views_gt = _encode_video_latents_by_view(
            model,
            video_gt,
            tiled=bool(args.vae_tiled_encode),
            tile_size=vae_tile_size,
            tile_stride=vae_tile_stride,
        )

        target_view = model.cross_view_target_view
        first_frame_image = _load_first_frame_image(
            raw,
            wrist_ff_index,
            int(video_gt.shape[-2]),
            int(video_gt.shape[-1]),
            model.pipe.device,
            model.pipe.torch_dtype,
        )

        cond_video = video_gt.clone()
        if first_frame_image is None:
            cond_video[target_view, :, 0] = model.build_cross_view_placeholder(video_gt)
        else:
            cond_video[target_view, :, 0] = first_frame_image.squeeze(0)

        # Plan A dual-end anchor: also fill cond_video[target_view, :, -1] with
        # the next segment's wrist first frame (or zero placeholder for last
        # segments / dropout). Then WanVideoUnit_ImageEmbedderVAE will encode
        # the entire 81-frame sequence including the tail anchor, giving the
        # mask channel a slot-position-correct tail bit.
        tail_frames = int(getattr(args, "num_tail_frames", 1)) if bool(getattr(args, "cross_view_use_tail_anchor", 0)) else 0
        if tail_frames > 0:
            tail_dropout_prob = float(getattr(args, "cross_view_tail_anchor_dropout", 0.0))
            tail_dropout_fired = (
                tail_dropout_prob > 0.0
                and float(torch.rand(()).item()) < tail_dropout_prob
            )
            tail_frame_image = None
            if not tail_dropout_fired:
                next_seg_path = _lookup_wrist_next_segment_first_frame_path(
                    raw,
                    wrist_ff_index,
                    segment_stride=int(getattr(args, "tail_anchor_segment_stride", 81)),
                )
                tail_frame_image = _load_first_frame_image_from_path(
                    next_seg_path,
                    int(video_gt.shape[-2]),
                    int(video_gt.shape[-1]),
                    model.pipe.device,
                    model.pipe.torch_dtype,
                )
            if tail_frame_image is not None:
                cond_video[target_view, :, -1] = tail_frame_image.squeeze(0)
            else:
                # Last segment OR dropout fired OR index miss: zero placeholder.
                # The mask channel still marks this slot as "known"; the model
                # learns that "known but zero" is a valid input distribution.
                cond_video[target_view, :, -1] = torch.zeros_like(
                    cond_video[target_view, :, -1]
                )

        legacy_branch = {}
        if not bool(args.skip_legacy_branch):
            legacy_branch = _build_legacy_image_branch(
                model,
                data,
                cond_video,
                tiled=bool(args.vae_tiled_encode),
                tile_size=vae_tile_size,
                tile_stride=vae_tile_stride,
                num_tail_frames=tail_frames,
            )

        target_history_latents = None
        if int(model.num_history_frames) == 1:
            target_history_latents = _target_history_latents_from_legacy_y(
                legacy_branch.get("y"),
                target_view=target_view,
                num_views=int(latent_views_gt.shape[0]),
                history_t=history_t,
                z_dim=int(getattr(model.pipe.vae, "z_dim", 16)),
            )
        if target_history_latents is None:
            target_history_latents = _build_target_history_latents(
                model,
                video_gt,
                first_frame_image,
                history_t=history_t,
                tiled=bool(args.vae_tiled_encode),
                tile_size=vae_tile_size,
                tile_stride=vae_tile_stride,
            )

        cond_history_latents = model.overwrite_target_history_latents(
            model.merge_view_latents(latent_views_gt),
            target_history_latents,
            num_views=int(latent_views_gt.shape[0]),
            history_t=history_t,
        )[:, :, :history_t]

        scene_tokens_cpu = None
        source_views = list(model.cross_view_source_views)
        source_first_frames_cpu = video_gt[source_views, :, 0].detach().cpu()
        if scene_extractor is not None:
            source_first_frames = video_gt[source_views, :, 0].unsqueeze(0)
            cam_token = torch.zeros(
                1,
                len(source_views),
                11,
                device=model.pipe.device,
                dtype=model.pipe.torch_dtype,
            )
            scene_tokens_cpu = scene_extractor(source_first_frames, cam_token).detach().cpu()

        sample = {
            "latent_views_gt": latent_views_gt.detach().cpu(),
            "target_history_latents": target_history_latents.detach().cpu(),
            "cond_history_latents": cond_history_latents.detach().cpu(),
            "y": to_cpu_tensor(legacy_branch.get("y")),
            "clip_feature": to_cpu_tensor(legacy_branch.get("clip_feature")),
            "state": to_cpu_tensor(data.get("state")),
            "action": to_cpu_tensor(data.get("action")),
            "prompt_emb": data.get("prompt_emb"),
            "episode_index": int(raw.get("episode_index", global_sample_id)),
            "height": int(video_gt.shape[-2]),
            "width": int(video_gt.shape[-1]),
            "num_frames": int(video_gt.shape[2]),
        }
        if scene_tokens_cpu is not None:
            sample["scene_tokens"] = scene_tokens_cpu
        sample["source_first_frames"] = source_first_frames_cpu
        if "valid_frames" in raw:
            sample["valid_frames"] = int(raw["valid_frames"])
        if "start_frame" in raw:
            sample["start_frame"] = int(raw["start_frame"])
        if "end_frame" in raw:
            sample["end_frame"] = int(raw["end_frame"])
        if raw.get("prompt") is not None:
            sample["prompt"] = raw["prompt"]

        temp_path = save_path.with_suffix(".pth.tmp")
        torch.save({k: v for k, v in sample.items() if v is not None}, temp_path)
        os.replace(temp_path, save_path)


def main():
    parser = build_parser()
    args = parser.parse_args()
    set_global_seed(args.seed)
    if int(args.num_shards) < 1:
        raise ValueError(f"`num_shards` must be >= 1, got {args.num_shards}.")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError(
            f"`shard_index` must be in [0, {int(args.num_shards) - 1}], got {args.shard_index}."
        )
    if int(args.cache_num_workers) < 0:
        raise ValueError(f"`cache_num_workers` must be >= 0, got {args.cache_num_workers}.")
    if int(args.cache_prefetch_factor) < 1:
        raise ValueError(
            f"`cache_prefetch_factor` must be >= 1, got {args.cache_prefetch_factor}."
        )
    _parse_int_pair(args.vae_tile_size, "vae_tile_size")
    _parse_int_pair(args.vae_tile_stride, "vae_tile_stride")

    requested_data_file_keys = ["video", "state", "prompt_emb"]
    runtime = prepare_wan_runtime(
        args.model_paths,
        args.load_modules,
        requested_data_file_keys,
    )
    modules = runtime["modules"]
    module_bases = {str(item).partition(":")[0].strip().lower() for item in modules}
    missing_modules = {"dit", "vae", "image"} - module_bases
    if missing_modules:
        raise ValueError(
            "Cross-view latent caching requires WAN modules "
            f"{sorted(missing_modules)} to be loaded."
        )
    data_file_keys = runtime["data_file_keys"]

    if "state" in requested_data_file_keys:
        data_file_keys = [key for key in data_file_keys if key != "action"]
        if "state" not in data_file_keys:
            data_file_keys.append("state")

    scene_extractor = None
    if args.scene_token_checkpoint:
        from diffsynth.models.scene_token_extractor import SceneTokenExtractor

        scene_extractor = SceneTokenExtractor(
            checkpoint_path=args.scene_token_checkpoint,
            freeze=True,
            input_value_range="minus1_1",
        ).eval()
        print(f"[cache] SceneTokenExtractor loaded from {args.scene_token_checkpoint}")

    wrist_ff_index = None
    if args.wrist_first_frame_index:
        wrist_index_path = Path(args.wrist_first_frame_index)
        if not wrist_index_path.is_file():
            raise FileNotFoundError(f"Wrist first-frame index not found: {wrist_index_path}")
        with wrist_index_path.open("r", encoding="utf-8") as f:
            wrist_ff_index = json.load(f)
        print(f"[cache] Wrist first-frame index loaded: {len(wrist_ff_index)} entries")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_config = {
        "cache_format_version": 1,
        "num_frames": int(args.num_frames),
        "num_history_frames": int(args.num_history_frames),
        "height": int(args.height),
        "width": int(args.width),
        "cross_view_source_views": [int(item) for item in args.cross_view_source_views.split(",") if item.strip()],
        "cross_view_target_view": int(args.cross_view_target_view),
        "cross_view_placeholder_mode": str(args.cross_view_placeholder_mode),
        "state_type": args.state_type,
        "load_modules": [str(item) for item in modules],
        "has_scene_tokens": scene_extractor is not None,
        "has_wrist_first_frame": wrist_ff_index is not None,
        "skip_legacy_branch": bool(args.skip_legacy_branch),
        "vae_tiled_encode": bool(args.vae_tiled_encode),
        # Plan A dual-end anchor metadata: training-side validators can read
        # these to detect a head-only cache trying to be paired with a
        # dual-end training run (or vice versa).
        "cross_view_use_tail_anchor": bool(int(getattr(args, "cross_view_use_tail_anchor", 0))),
        "num_tail_frames": int(getattr(args, "num_tail_frames", 1)),
        "tail_anchor_segment_stride": int(getattr(args, "tail_anchor_segment_stride", 81)),
        "cross_view_tail_anchor_dropout": float(getattr(args, "cross_view_tail_anchor_dropout", 0.0)),
    }
    cache_config_tmp = output_root / f".cache_config.{os.getpid()}.tmp"
    with cache_config_tmp.open("w", encoding="utf-8") as f:
        json.dump(cache_config, f, indent=2)
    os.replace(cache_config_tmp, output_root / "cache_config.json")

    model = build_model(args, runtime)
    if scene_extractor is not None:
        scene_extractor = scene_extractor.to(
            dtype=model.pipe.torch_dtype,
            device=model.pipe.device,
        ).eval()

    train_dataset = create_dataset(args, args.train_metadata_path, modules, data_file_keys)
    train_dataset._skip_existing = bool(args.skip_existing)
    cache_split(
        model,
        train_dataset,
        "train",
        output_root,
        args,
        scene_extractor=scene_extractor,
        wrist_ff_index=wrist_ff_index,
    )
    if args.val_metadata_path:
        val_dataset = create_dataset(args, args.val_metadata_path, modules, data_file_keys)
        val_dataset._skip_existing = bool(args.skip_existing)
        cache_split(
            model,
            val_dataset,
            "val",
            output_root,
            args,
            scene_extractor=scene_extractor,
            wrist_ff_index=wrist_ff_index,
        )
    else:
        (output_root / "val").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
