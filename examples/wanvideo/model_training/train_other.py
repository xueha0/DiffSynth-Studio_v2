import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from tool.cross_view_keyframe_helpers import (
    KEYFRAME_ANCHOR_LOOKUP_MODE,
    load_keyframe_anchor_index,
    resolve_keyframe_anchors,
)

from diffsynth.core import UnifiedDataset, load_state_dict
from diffsynth.core.data.operators import (
    LoadCobotAction,
    LoadDroidCameraTokens,
    LoadDroidState,
    LoadPredictedDroidState,
    ResolvePromptEmbPath,
)
from diffsynth.diffusion import *
from diffsynth.diffusion.parsers import prepare_wan_runtime
from diffsynth.models.cross_view_projector import (
    CrossViewSourceVideoProjector3D,
    CrossViewSourceVideoProjector3DTemporal,
)
from diffsynth.models.wan_video_action_encoder import WanVideoActionEncoder
from diffsynth.pipelines.wan_video import (
    ModelConfig,
    WanVideoPipeline,
    WanVideoUnit_ImageEmbedderCLIP,
    WanVideoUnit_ImageEmbedderVAE,
    WanVideoUnit_InputVideoEmbedder,
    WanVideoUnit_NoiseInitializer,
    WanVideoUnit_ShapeChecker,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_global_seed(seed: int = 42) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_cross_view_source_views(value) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        items = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(items) == 0:
        raise ValueError("Expected at least one source view index.")
    return tuple(int(item) for item in items)


def normalize_module_specs(modules) -> list[str]:
    return [str(item).strip().lower() for item in modules if str(item).strip()]


def load_cross_view_cache_config(cache_root: str) -> dict:
    config_path = Path(cache_root) / "cache_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Cache config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Cache config must be a JSON object: {config_path}")
    return config


def validate_cross_view_cache_config(cache_config: dict, args, modules) -> None:
    if not isinstance(cache_config, dict):
        raise TypeError("`cache_config` must be a dict.")

    expected_pairs = (
        ("num_frames", int(args.num_frames)),
        ("num_history_frames", int(args.num_history_frames)),
        ("width", int(args.width)),
        ("cross_view_target_view", int(args.cross_view_target_view)),
        ("cross_view_placeholder_mode", str(args.cross_view_placeholder_mode)),
        ("state_type", args.state_type),
    )
    for key, expected in expected_pairs:
        actual = cache_config.get(key)
        if actual != expected:
            raise ValueError(
                f"Cached dataset config mismatch for `{key}`: "
                f"expected {expected!r}, got {actual!r}."
            )

    cached_height = cache_config.get("height")
    if cached_height != int(args.height):
        print(
            "[cache] Warning: cached config height "
            f"{cached_height!r} differs from requested height {int(args.height)!r}; "
            "cached tensor shapes will be treated as the source of truth."
        )

    cached_source_views = parse_cross_view_source_views(
        cache_config.get("cross_view_source_views", ())
    )
    expected_source_views = parse_cross_view_source_views(args.cross_view_source_views)
    if cached_source_views != expected_source_views:
        raise ValueError(
            "Cached dataset config mismatch for `cross_view_source_views`: "
            f"expected {expected_source_views}, got {cached_source_views}."
        )

    cached_modules = tuple(
        normalize_module_specs(cache_config.get("load_modules", ()))
    )
    expected_modules = tuple(normalize_module_specs(modules))
    if cached_modules != expected_modules:
        raise ValueError(
            "Cached dataset config mismatch for `load_modules`: "
            f"expected {expected_modules}, got {cached_modules}."
        )

    uses_geometry_sidecar = (
        getattr(args, "geometry_scene_token_source", "cached_zero_cam")
        == "camera_aware_sidecar"
        and getattr(args, "geometry_sidecar_cache_path", None) not in (None, "")
    )
    if (
        getattr(args, "scene_token_checkpoint", None)
        and not cache_config.get("has_scene_tokens", False)
        and not uses_geometry_sidecar
    ):
        raise ValueError(
            "Cached dataset was built without `--scene_token_checkpoint`, "
            "but training requested scene-token conditioning."
        )
    if getattr(args, "wrist_first_frame_index", None) and not cache_config.get("has_wrist_first_frame", False):
        raise ValueError(
            "Cached dataset was built without `--wrist_first_frame_index`, "
            "but training requested LagerNVS synthesized first-frame conditioning."
        )
    # Plan A dual-end anchor: training-time `cross_view_use_tail_anchor` MUST
    # match the cache builder's setting. Otherwise the cache's y channel
    # (read directly by attach_cached_legacy_image_branch) will not contain
    # the tail anchor's mask + latent, and the trainer will silently degrade
    # to single-end despite the CLI flag.
    train_use_tail = bool(int(getattr(args, "cross_view_use_tail_anchor", 0)))
    cache_use_tail = bool(cache_config.get("cross_view_use_tail_anchor", False))
    if train_use_tail and not cache_use_tail:
        raise ValueError(
            "Cached dataset was built with cross_view_use_tail_anchor=False "
            "(or pre-Plan-A cache without the field), but training requested "
            "dual-end anchoring. Rebuild the cache with "
            "--cross_view_use_tail_anchor 1 (and --num_tail_frames matching) "
            "before training Plan A. See userbook §7.3."
        )
    if cache_use_tail and not train_use_tail:
        # Cache has dual-end y but training disabled it. Not fatal (mask
        # bits and tail latent are simply ignored as extra signal in y),
        # but warn loudly so the user notices.
        print(
            "[cache] WARN: cache_config.cross_view_use_tail_anchor=True but "
            "training has cross_view_use_tail_anchor=False. The cached y "
            "channel will still contain the tail anchor signal -- this is "
            "the design's training-time ablation; if unintended, set "
            "CROSS_VIEW_USE_TAIL_ANCHOR=1 to use it."
        )
    if train_use_tail:
        train_n_tail = int(getattr(args, "num_tail_frames", 1))
        cache_n_tail = int(cache_config.get("num_tail_frames", 0))
        if train_n_tail != cache_n_tail:
            raise ValueError(
                f"num_tail_frames mismatch between training ({train_n_tail}) "
                f"and cache ({cache_n_tail}). Rebuild the cache with the "
                f"matching --num_tail_frames."
            )
        tail_lookup_mode = cache_config.get("tail_anchor_lookup_mode")
        if tail_lookup_mode != "end_frame_index":
            raise ValueError(
                "Cached dataset uses an old or unknown tail-anchor lookup mode "
                f"({tail_lookup_mode!r}). Rebuild the cache with the frame-indexed "
                "wrist anchor JSON so tail anchors are loaded from end_frame keys."
            )

    train_use_keyframes = bool(int(getattr(args, "cross_view_use_keyframe_anchor", 0)))
    cache_use_keyframes = bool(cache_config.get("cross_view_use_keyframe_anchor", False))
    if train_use_keyframes and not cache_use_keyframes:
        raise ValueError(
            "Cached dataset was built without keyframe anchors, but training requested "
            "cross_view_use_keyframe_anchor=1. Refresh/rebuild the cache y channel with "
            "keyframe anchors before training."
        )
    if cache_use_keyframes and not train_use_keyframes:
        print(
            "[cache] WARN: cache_config.cross_view_use_keyframe_anchor=True but "
            "training has cross_view_use_keyframe_anchor=False. The cached y channel "
            "will still contain keyframe anchor signal."
        )
    if train_use_keyframes:
        train_n_keyframes = int(getattr(args, "num_keyframe_anchors", 3))
        cache_n_keyframes = int(cache_config.get("num_keyframe_anchors", 0))
        if train_n_keyframes != cache_n_keyframes:
            raise ValueError(
                f"num_keyframe_anchors mismatch between training ({train_n_keyframes}) "
                f"and cache ({cache_n_keyframes})."
            )
        keyframe_lookup_mode = cache_config.get("keyframe_anchor_lookup_mode")
        if keyframe_lookup_mode != KEYFRAME_ANCHOR_LOOKUP_MODE:
            raise ValueError(
                "Cached dataset uses an old or unknown keyframe-anchor lookup mode "
                f"({keyframe_lookup_mode!r}). Expected {KEYFRAME_ANCHOR_LOOKUP_MODE!r}."
            )


class CrossViewSourceTemporalGate(nn.Module):
    def __init__(self, mode: str, condition_dim: int):
        super().__init__()
        self.mode = str(mode)
        if self.mode == "scalar":
            self.scalar_logit = nn.Parameter(torch.zeros(()))
        elif self.mode == "state_aware":
            hidden_dim = max(64, int(condition_dim) * 4)
            self.condition_mlp = nn.Sequential(
                nn.LayerNorm(condition_dim),
                nn.Linear(condition_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        source_memory: torch.Tensor | None,
        condition_sequence: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if source_memory is None or self.mode == "none":
            return source_memory

        if self.mode == "scalar":
            gate = torch.sigmoid(self.scalar_logit).to(
                dtype=source_memory.dtype,
                device=source_memory.device,
            )
            return source_memory * gate

        if condition_sequence is None:
            return source_memory

        condition_sequence = torch.as_tensor(
            condition_sequence,
            device=source_memory.device,
            dtype=source_memory.dtype,
        )
        if condition_sequence.ndim == 2:
            condition_sequence = condition_sequence.unsqueeze(0)

        if source_memory.ndim == 4:
            if condition_sequence.shape[1] != source_memory.shape[1]:
                raise ValueError(
                    "State-aware temporal gate expects condition sequence length "
                    f"{source_memory.shape[1]}, got {condition_sequence.shape[1]}."
                )
            gate = torch.sigmoid(self.condition_mlp(condition_sequence)).unsqueeze(-1)
            return source_memory * gate

        pooled_condition = condition_sequence.mean(dim=1)
        gate = torch.sigmoid(self.condition_mlp(pooled_condition)).view(
            pooled_condition.shape[0], 1, 1
        )
        return source_memory * gate.to(dtype=source_memory.dtype, device=source_memory.device)


class CrossViewTargetStateHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class CrossViewTargetCameraEncoder(nn.Module):
    """目标视角相机 token 编码器（瓶颈结构）。

    结构: camera_dim → bottleneck_dim → bottleneck_dim → dim
    相比 camera_dim → dim → dim 设计:
    - 低维 11D 信号在 bottleneck (256D) 内做非线性 refinement, 充分但不冗余
    - 去掉高维空间 (1536D) 内的 redundant projection
    - 参数量从 ~2.38M 降至 ~0.46M, 同时保持两层非线性深度
    """

    def __init__(self, dim: int, camera_dim: int = 11, bottleneck_dim: int = 256):
        super().__init__()
        bottleneck_dim = max(int(bottleneck_dim), int(camera_dim))
        self.proj = nn.Sequential(
            nn.LayerNorm(camera_dim),
            nn.Linear(camera_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, camera_tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(camera_tokens)


class Scene3DNoisePriorAdapter(nn.Module):
    def __init__(
        self,
        scene_dim: int = 1536,
        latent_channels: int = 16,
        hidden_dim: int = 512,
        num_heads: int = 8,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.scene_proj = nn.Sequential(
            nn.LayerNorm(scene_dim),
            nn.Linear(scene_dim, hidden_dim),
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_channels),
        )

    def forward(
        self,
        scene_tokens: torch.Tensor,
        latent_shape: tuple[int, int, int, int, int],
    ) -> torch.Tensor:
        batch_size, _, num_frames, height, width = latent_shape
        scene_tokens = self.scene_proj(scene_tokens)
        if scene_tokens.shape[0] == 1 and batch_size != 1:
            scene_tokens = scene_tokens.expand(batch_size, -1, -1)
        elif scene_tokens.shape[0] != batch_size:
            raise ValueError(
                "Scene3DNoisePriorAdapter scene-token batch size "
                f"{scene_tokens.shape[0]} does not match latent batch size {batch_size}."
            )
        y = torch.linspace(
            -1.0,
            1.0,
            height,
            device=scene_tokens.device,
            dtype=scene_tokens.dtype,
        )
        x = torch.linspace(
            -1.0,
            1.0,
            width,
            device=scene_tokens.device,
            dtype=scene_tokens.dtype,
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        queries = torch.stack([yy, xx], dim=-1).reshape(1, height * width, 2)
        queries = self.query_mlp(queries).expand(batch_size, -1, -1)
        noise_tokens, _ = self.cross_attn(queries, scene_tokens, scene_tokens, need_weights=False)
        noise = self.out(noise_tokens)
        noise = rearrange(noise, "b (h w) c -> b c 1 h w", h=height, w=width)
        noise = noise.expand(batch_size, self.latent_channels, num_frames, height, width)
        return normalize_noise_like(noise)


class DynamicViewActionScene3DNoisePriorAdapter(nn.Module):
    def __init__(
        self,
        scene_dim: int = 1536,
        condition_dim: int = 7,
        latent_channels: int = 16,
        hidden_dim: int = 512,
        num_heads: int = 8,
        max_views: int = 8,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.max_views = int(max_views)
        self.scene_proj = nn.Sequential(
            nn.LayerNorm(scene_dim),
            nn.Linear(scene_dim, hidden_dim),
        )
        self.grid_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.condition_mlp = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.wrist_camera_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))
        self.query_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_channels),
        )
        nn.init.normal_(self.wrist_camera_token, mean=0.0, std=0.02)

    def forward(
        self,
        scene_tokens: torch.Tensor,
        latent_shape: tuple[int, int, int, int, int],
        condition_sequence: torch.Tensor,
        source_view_ids=None,
        target_view_id: int | None = None,
    ) -> torch.Tensor:
        batch_size, _, num_frames, height, width = latent_shape
        scene_tokens = self.scene_proj(scene_tokens)
        if scene_tokens.shape[0] == 1 and batch_size != 1:
            scene_tokens = scene_tokens.expand(batch_size, -1, -1)
        elif scene_tokens.shape[0] != batch_size:
            raise ValueError(
                "DynamicViewActionScene3DNoisePriorAdapter scene-token batch size "
                f"{scene_tokens.shape[0]} does not match latent batch size {batch_size}."
            )

        if condition_sequence.ndim == 2:
            condition_sequence = condition_sequence.unsqueeze(0)
        if condition_sequence.shape[1] != num_frames:
            raise ValueError(
                "DynamicViewActionScene3DNoisePriorAdapter expects condition length "
                f"{num_frames}, got {condition_sequence.shape[1]}."
            )
        if condition_sequence.shape[0] == 1 and batch_size != 1:
            condition_sequence = condition_sequence.expand(batch_size, -1, -1)
        elif condition_sequence.shape[0] != batch_size:
            raise ValueError(
                "DynamicViewActionScene3DNoisePriorAdapter condition batch size "
                f"{condition_sequence.shape[0]} does not match latent batch size "
                f"{batch_size}."
            )
        condition_sequence = condition_sequence.to(
            device=scene_tokens.device,
            dtype=scene_tokens.dtype,
        )

        y = torch.linspace(
            -1.0,
            1.0,
            height,
            device=scene_tokens.device,
            dtype=scene_tokens.dtype,
        )
        x = torch.linspace(
            -1.0,
            1.0,
            width,
            device=scene_tokens.device,
            dtype=scene_tokens.dtype,
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([yy, xx], dim=-1).reshape(1, 1, height * width, 2)
        grid_emb = self.grid_mlp(grid).expand(batch_size, num_frames, -1, -1)

        condition_emb = self.condition_mlp(condition_sequence).unsqueeze(2)
        condition_emb = condition_emb.expand(-1, -1, height * width, -1)
        wrist_emb = self.wrist_camera_token.to(
            dtype=scene_tokens.dtype,
            device=scene_tokens.device,
        ).expand(batch_size, num_frames, height * width, -1)

        query_factors = torch.cat([grid_emb, condition_emb, wrist_emb], dim=-1)
        queries = self.query_norm(self.query_mlp(query_factors))
        queries = queries.reshape(batch_size * num_frames, height * width, self.hidden_dim)
        scene_memory = scene_tokens[:, None].expand(
            batch_size,
            num_frames,
            scene_tokens.shape[1],
            scene_tokens.shape[2],
        ).reshape(batch_size * num_frames, scene_tokens.shape[1], scene_tokens.shape[2])
        noise_tokens, _ = self.cross_attn(
            queries,
            scene_memory,
            scene_memory,
            need_weights=False,
        )
        noise = self.out(noise_tokens)
        noise = rearrange(
            noise,
            "(b t) (h w) c -> b c t h w",
            b=batch_size,
            t=num_frames,
            h=height,
            w=width,
        )
        return normalize_noise_like(noise)


class ActionNoiseModulator(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        latent_channels: int = 16,
        hidden_dim: int = 128,
        scale_strength: float = 0.1,
        bias_strength: float = 0.1,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.scale_strength = float(scale_strength)
        self.bias_strength = float(bias_strength)
        hidden_dim = max(int(hidden_dim), int(condition_dim) * 4)
        self.proj = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_channels * 2),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(
        self,
        base_noise: torch.Tensor,
        condition_sequence: torch.Tensor,
    ) -> torch.Tensor:
        if condition_sequence.ndim == 2:
            condition_sequence = condition_sequence.unsqueeze(0)
        if condition_sequence.shape[1] != base_noise.shape[2]:
            raise ValueError(
                "ActionNoiseModulator expects condition length "
                f"{base_noise.shape[2]}, got {condition_sequence.shape[1]}."
            )
        if condition_sequence.shape[0] == 1 and base_noise.shape[0] != 1:
            condition_sequence = condition_sequence.expand(base_noise.shape[0], -1, -1)
        elif condition_sequence.shape[0] != base_noise.shape[0]:
            raise ValueError(
                "ActionNoiseModulator condition batch size "
                f"{condition_sequence.shape[0]} does not match noise batch size "
                f"{base_noise.shape[0]}."
            )
        condition_sequence = condition_sequence.to(
            device=base_noise.device,
            dtype=base_noise.dtype,
        )
        scale_bias = self.proj(condition_sequence)
        scale, bias = scale_bias.chunk(2, dim=-1)
        scale = torch.tanh(scale).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        bias = torch.tanh(bias).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        modulated = base_noise * (1.0 + self.scale_strength * scale)
        modulated = modulated + self.bias_strength * bias
        return normalize_noise_like(modulated)


def normalize_noise_like(noise: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = noise.mean(dim=(1, 2, 3, 4), keepdim=True)
    std = noise.std(dim=(1, 2, 3, 4), keepdim=True).clamp_min(eps)
    return (noise - mean) / std


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        modules=("dit", "text", "vae", "image", "action"),
        fp8_models=None,
        offload_models=None,
        ckpt_path=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        num_history_frames=1,
        cross_view_source_views="0,1",
        cross_view_target_view=2,
        cross_view_placeholder_mode="zeros",
        cross_view_source_loss_weight=0.1,
        cross_view_old_branch_dropout=0.0,
        cross_view_projector_hidden_dim=512,
        cross_view_source_injection_mode="temporal_local",
        cross_view_source_branch_mode="sigma_matched_clamp",
        cross_view_source_window_radius=1,
        cross_view_source_gate_mode="scalar",
        cross_view_temp_loss_weight=0.1,
        cross_view_state_loss_weight=0.05,
        cross_view_global_source_tokens=0,
        cross_view_aux_loss_warmup_ratio=0.0,
        cross_view_old_branch_dropout_schedule="linear_warmup_to_high",
        cross_view_legacy_branch_schedule=None,
        cross_view_disable_legacy_image_branch=0,
        cross_view_use_tail_anchor=0,
        num_tail_frames=1,
        cross_view_tail_anchor_dropout=0.0,
        cross_view_use_keyframe_anchor=0,
        num_keyframe_anchors=3,
        keyframe_anchor_dropout=0.0,
        cross_view_3d_noise_prior_mode="none",
        cross_view_3d_noise_prior_weight=0.1,
        cross_view_3d_noise_anchor_attenuation=1.0,
        state_type=None,
        scene_token_checkpoint=None,
        scene_token_pool_size=512,
        geometry_gate_mode="learned",
        geometry_sidecar_cache_path=None,
        geometry_use_camera_tokens=0,
        geometry_target_camera_mode="none",
        geometry_scene_token_source="cached_zero_cam",
        cached_pred_state_root=None,
        alignment_loss_weight=0.1,
        alignment_loss_warmup_ratio=0.1,
    ):
        super().__init__()

        def module_base(name: str) -> str:
            return str(name).partition(":")[0].strip().lower()

        self.task = task
        self.cross_view_stage = self.resolve_cross_view_stage(task)
        self.cross_view_source_views = self.parse_view_indices(cross_view_source_views)
        self.cross_view_target_view = int(cross_view_target_view)
        self.cross_view_placeholder_mode = cross_view_placeholder_mode
        self.cross_view_source_loss_weight = float(cross_view_source_loss_weight)
        self.cross_view_old_branch_dropout = float(cross_view_old_branch_dropout)
        self.cross_view_projector_hidden_dim = int(cross_view_projector_hidden_dim)
        self.cross_view_source_injection_mode = str(cross_view_source_injection_mode)
        if self.cross_view_source_injection_mode not in (
            "none",
            "global_concat",
            "temporal_local",
        ):
            raise ValueError(
                "Unsupported cross_view_source_injection_mode="
                f"{self.cross_view_source_injection_mode!r}."
            )
        if (
            self.cross_view_stage == 2
            and self.cross_view_source_injection_mode == "temporal_local"
            and lora_base_model is not None
        ):
            raise ValueError(
                "Independent temporal source attention requires full DiT training; "
                "LoRA-only training is not supported."
            )
        self.cross_view_source_branch_mode = str(cross_view_source_branch_mode)
        self.cross_view_source_window_radius = int(cross_view_source_window_radius)
        self.cross_view_source_gate_mode = str(cross_view_source_gate_mode)
        self.cross_view_temp_loss_weight = float(cross_view_temp_loss_weight)
        self.cross_view_state_loss_weight = float(cross_view_state_loss_weight)
        self.cross_view_global_source_tokens = max(0, int(cross_view_global_source_tokens))
        if (
            self.cross_view_source_injection_mode == "temporal_local"
            and self.cross_view_global_source_tokens > 0
        ):
            print(
                "Independent temporal source attention ignores "
                "cross_view_global_source_tokens; disabling global source tokens."
            )
            self.cross_view_global_source_tokens = 0
        self.cross_view_aux_loss_warmup_ratio = max(
            0.0, min(1.0, float(cross_view_aux_loss_warmup_ratio))
        )
        self.cross_view_old_branch_dropout_schedule = str(
            cross_view_old_branch_dropout_schedule
        )
        self.cross_view_legacy_branch_schedule = (
            None
            if cross_view_legacy_branch_schedule in (None, "")
            else str(cross_view_legacy_branch_schedule)
        )
        self.cross_view_disable_legacy_image_branch = bool(
            int(cross_view_disable_legacy_image_branch)
        )
        self.cross_view_use_tail_anchor = bool(int(cross_view_use_tail_anchor))
        self.num_tail_frames = max(1, int(num_tail_frames))
        self.cross_view_tail_anchor_dropout = max(
            0.0, min(1.0, float(cross_view_tail_anchor_dropout))
        )
        self.cross_view_use_keyframe_anchor = bool(int(cross_view_use_keyframe_anchor))
        self.num_keyframe_anchors = max(0, int(num_keyframe_anchors))
        self.keyframe_anchor_dropout = max(
            0.0, min(1.0, float(keyframe_anchor_dropout))
        )
        self.cross_view_3d_noise_prior_mode = str(cross_view_3d_noise_prior_mode)
        if self.cross_view_3d_noise_prior_mode not in (
            "none",
            "scene_action_grid",
            "dynamic_view_action",
        ):
            raise ValueError(
                "Unsupported cross_view_3d_noise_prior_mode="
                f"{self.cross_view_3d_noise_prior_mode!r}."
            )
        self.cross_view_3d_noise_prior_weight = max(
            0.0, min(1.0, float(cross_view_3d_noise_prior_weight))
        )
        self.cross_view_3d_noise_anchor_attenuation = max(
            0.0, min(1.0, float(cross_view_3d_noise_anchor_attenuation))
        )
        self.state_type = state_type
        self.scene_token_checkpoint = scene_token_checkpoint
        self.scene_token_pool_size = int(scene_token_pool_size)
        self.geometry_gate_mode = str(geometry_gate_mode)
        self.geometry_sidecar_cache_path = (
            None
            if geometry_sidecar_cache_path in (None, "")
            else str(geometry_sidecar_cache_path)
        )
        self.geometry_use_camera_tokens = bool(int(geometry_use_camera_tokens))
        self.geometry_target_camera_mode = str(geometry_target_camera_mode)
        self.geometry_scene_token_source = str(geometry_scene_token_source)
        self.cached_pred_state_root = (
            None
            if cached_pred_state_root in (None, "")
            else str(cached_pred_state_root)
        )
        self.alignment_loss_weight = float(alignment_loss_weight)
        self.alignment_loss_warmup_ratio = float(alignment_loss_warmup_ratio)
        self.wrist_first_frame_index = None
        self.keyframe_anchor_index = None
        self.condition_dim = 7 if state_type == "state_pose_7d" else 14
        self.cross_view_total_training_steps = 0
        self.cross_view_current_training_step = 0
        if self.cross_view_stage > 0 and num_history_frames != 1:
            raise ValueError(
                "cross-view training currently requires --num_history_frames=1."
            )

        module_list = [str(item).strip().lower() for item in modules if str(item).strip()]
        module_bases = {module_base(item) for item in module_list}
        enable_text = "text" in module_bases
        trainable_models = self.extend_trainable_models(trainable_models)

        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        tokenizer_config = ModelConfig(tokenizer_path) if enable_text and tokenizer_path else None
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            modules=module_list,
        )
        if "action" in module_bases and self.condition_dim != 14:
            self.pipe.action_encoder = WanVideoActionEncoder(
                action_dim=self.condition_dim,
                dim=getattr(self.pipe.dit, "dim", 1536),
            ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        if self.cross_view_stage == 2:
            if self.cross_view_source_injection_mode != "none":
                projector_cls = (
                    CrossViewSourceVideoProjector3DTemporal
                    if self.cross_view_source_injection_mode == "temporal_local"
                    else CrossViewSourceVideoProjector3D
                )
                self.pipe.source_video_projector = projector_cls(
                    in_channels=getattr(self.pipe.vae, "z_dim", 16),
                    hidden_channels=self.cross_view_projector_hidden_dim,
                    out_channels=getattr(self.pipe.dit, "dim", 1536),
                    max_source_views=max(4, len(self.cross_view_source_views)),
                    max_time=64,
                ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                self.pipe.source_temporal_gate = CrossViewSourceTemporalGate(
                    mode=self.cross_view_source_gate_mode,
                    condition_dim=self.condition_dim,
                ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                if self.cross_view_source_injection_mode == "temporal_local":
                    if self.pipe.dit is None:
                        raise RuntimeError(
                            "Temporal source memory requires the WAN DiT to be loaded."
                        )
                    self.pipe.dit.enable_source_memory_attention()
            self.pipe.target_state_head = CrossViewTargetStateHead(
                hidden_dim=getattr(self.pipe.dit, "dim", 1536),
                out_dim=self.condition_dim,
            ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            if self.geometry_target_camera_mode == "add_time_mlp":
                self.pipe.target_camera_encoder = CrossViewTargetCameraEncoder(
                    dim=getattr(self.pipe.dit, "dim", 1536),
                ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            if self.scene_token_checkpoint is not None:
                from diffsynth.models.scene_token_extractor import (
                    SceneTokenExtractor,
                    SceneTokenAdapter,
                )
                from diffsynth.models.geometry_gate import TimestepAdaptiveGeometryGate

                dit_dim = getattr(self.pipe.dit, "dim", 1536)
                self.pipe.scene_token_extractor = SceneTokenExtractor(
                    checkpoint_path=self.scene_token_checkpoint,
                    freeze=True,
                ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                self.pipe.scene_token_adapter = SceneTokenAdapter(
                    in_dim=768,
                    out_dim=dit_dim,
                    pool_size=self.scene_token_pool_size,
                ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                num_blocks = len(self.pipe.dit.blocks)
                self.pipe.geometry_gates = nn.ModuleList([
                    TimestepAdaptiveGeometryGate(dim=dit_dim, mode=self.geometry_gate_mode)
                    for _ in range(num_blocks)
                ]).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            if self.is_3d_noise_prior_enabled():
                if self.scene_token_checkpoint is None:
                    raise ValueError(
                        f"cross_view_3d_noise_prior_mode={self.cross_view_3d_noise_prior_mode!r} requires "
                        "`scene_token_checkpoint` so scene tokens can be built."
                    )
                if self.cross_view_3d_noise_prior_mode == "dynamic_view_action":
                    self.pipe.scene_3d_noise_prior_adapter = (
                        DynamicViewActionScene3DNoisePriorAdapter(
                            scene_dim=getattr(self.pipe.dit, "dim", 1536),
                            condition_dim=self.condition_dim,
                            latent_channels=getattr(self.pipe.vae, "z_dim", 16),
                            max_views=max(
                                8,
                                max(self.cross_view_source_views + (self.cross_view_target_view,)) + 1,
                            ),
                        )
                    ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                    self.pipe.action_noise_modulator = None
                else:
                    self.pipe.scene_3d_noise_prior_adapter = Scene3DNoisePriorAdapter(
                        scene_dim=getattr(self.pipe.dit, "dim", 1536),
                        latent_channels=getattr(self.pipe.vae, "z_dim", 16),
                    ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                    self.pipe.action_noise_modulator = ActionNoiseModulator(
                        condition_dim=self.condition_dim,
                        latent_channels=getattr(self.pipe.vae, "z_dim", 16),
                    ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

        self.pipe = self.split_pipeline_units(
            task, self.pipe, trainable_models, lora_base_model
        )
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )
        if ckpt_path is not None:
            self.load_checkpoint_weights(ckpt_path)
        if not enable_text:
            self.freeze_unused_prompt_params()

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(
                pipe, **inputs_shared, **inputs_posi
            ),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(
                pipe, **inputs_shared, **inputs_posi
            ),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(
                pipe, **inputs_shared, **inputs_posi
            ),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(
                pipe, **inputs_shared, **inputs_posi
            ),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.num_history_frames = num_history_frames

    @staticmethod
    def resolve_cross_view_stage(task: str) -> int:
        if task == "cross_view_stage1" or task.startswith("cross_view_stage1:"):
            return 1
        if task == "cross_view_stage2" or task.startswith("cross_view_stage2:"):
            return 2
        return 0

    @staticmethod
    def parse_view_indices(value: str) -> tuple[int, ...]:
        items = [item.strip() for item in str(value).split(",") if item.strip()]
        if len(items) == 0:
            raise ValueError("Expected at least one source view index.")
        return tuple(int(item) for item in items)

    def extend_trainable_models(self, trainable_models):
        if trainable_models is None:
            return trainable_models
        models = [item.strip() for item in str(trainable_models).split(",") if item.strip()]
        models = [item for item in models if item != "geometry_gates"]
        if self.cross_view_source_injection_mode == "temporal_local":
            models = [item for item in models if item != "source_temporal_gate"]
            if self.cross_view_stage == 2 and "dit" not in models:
                models.append("dit")
        source_memory_enabled = (
            self.cross_view_stage == 2
            and self.cross_view_source_injection_mode != "none"
        )
        if source_memory_enabled and "source_video_projector" not in models:
            models.append("source_video_projector")
        if (
            source_memory_enabled
            and self.cross_view_source_injection_mode != "temporal_local"
            and self.cross_view_source_gate_mode != "none"
            and "source_temporal_gate" not in models
        ):
            models.append("source_temporal_gate")
        if (
            self.cross_view_stage == 2
            and self.cross_view_state_loss_weight > 0
            and "target_state_head" not in models
        ):
            models.append("target_state_head")
        if (
            self.cross_view_stage == 2
            and self.geometry_target_camera_mode == "add_time_mlp"
            and "target_camera_encoder" not in models
        ):
            models.append("target_camera_encoder")
        if (
            self.cross_view_stage == 2
            and self.scene_token_checkpoint is not None
            and "scene_token_adapter" not in models
        ):
            models.append("scene_token_adapter")
        # if (
        #     self.cross_view_stage == 2
        #     and self.scene_token_checkpoint is not None
        #     and "geometry_gates" not in models
        # ):
        #     models.append("geometry_gates")
        if self.cross_view_stage == 2 and self.is_3d_noise_prior_enabled():
            if "scene_3d_noise_prior_adapter" not in models:
                models.append("scene_3d_noise_prior_adapter")
            if (
                self.cross_view_3d_noise_prior_mode == "scene_action_grid"
                and "action_noise_modulator" not in models
            ):
                models.append("action_noise_modulator")
        return ",".join(models)

    def load_checkpoint_weights(self, ckpt_path: str):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        state_dict = load_state_dict(ckpt_path, torch_dtype=self.pipe.torch_dtype, device="cpu")
        dit_state, action_state, projector_state = {}, {}, {}
        gate_state, state_head_state, target_camera_state = {}, {}, {}
        scene_adapter_state, geometry_gates_state = {}, {}
        scene_noise_prior_state, action_noise_modulator_state = {}, {}
        ignored_key_count = 0
        for key, value in state_dict.items():
            if key.startswith("pipe.action_encoder."):
                action_state[key[len("pipe.action_encoder."):]] = value
            elif key.startswith("action_encoder."):
                action_state[key[len("action_encoder."):]] = value
            elif key.startswith("pipe.source_video_projector."):
                projector_state[key[len("pipe.source_video_projector."):]] = value
            elif key.startswith("source_video_projector."):
                projector_state[key[len("source_video_projector."):]] = value
            elif key.startswith("pipe.source_temporal_gate."):
                gate_state[key[len("pipe.source_temporal_gate."):]] = value
            elif key.startswith("source_temporal_gate."):
                gate_state[key[len("source_temporal_gate."):]] = value
            elif key.startswith("pipe.target_state_head."):
                state_head_state[key[len("pipe.target_state_head."):]] = value
            elif key.startswith("target_state_head."):
                state_head_state[key[len("target_state_head."):]] = value
            elif key.startswith("pipe.target_camera_encoder."):
                target_camera_state[key[len("pipe.target_camera_encoder."):]] = value
            elif key.startswith("target_camera_encoder."):
                target_camera_state[key[len("target_camera_encoder."):]] = value
            elif key.startswith("pipe.scene_token_adapter."):
                scene_adapter_state[key[len("pipe.scene_token_adapter."):]] = value
            elif key.startswith("scene_token_adapter."):
                scene_adapter_state[key[len("scene_token_adapter."):]] = value
            elif key.startswith("pipe.geometry_gates."):
                geometry_gates_state[key[len("pipe.geometry_gates."):]] = value
            elif key.startswith("geometry_gates."):
                geometry_gates_state[key[len("geometry_gates."):]] = value
            elif key.startswith("pipe.scene_3d_noise_prior_adapter."):
                scene_noise_prior_state[
                    key[len("pipe.scene_3d_noise_prior_adapter."):]
                ] = value
            elif key.startswith("scene_3d_noise_prior_adapter."):
                scene_noise_prior_state[
                    key[len("scene_3d_noise_prior_adapter."):]
                ] = value
            elif key.startswith("pipe.action_noise_modulator."):
                action_noise_modulator_state[
                    key[len("pipe.action_noise_modulator."):]
                ] = value
            elif key.startswith("action_noise_modulator."):
                action_noise_modulator_state[
                    key[len("action_noise_modulator."):]
                ] = value
            elif key.startswith("pipe.dit."):
                dit_state[key[len("pipe.dit."):]] = value
            elif key.startswith("dit."):
                dit_state[key[len("dit."):]] = value
            elif key.startswith("pipe."):
                ignored_key_count += 1
            else:
                dit_state[key] = value

        source_attention_parts = (
            ".source_norm.",
            ".source_cross_attn.",
            ".source_router.",
        )
        expected_source_attention_keys = set()
        checkpoint_source_attention_keys = {
            key
            for key in dit_state
            if any(part in key for part in source_attention_parts)
        }
        if self.pipe.dit is not None:
            expected_source_attention_keys = {
                name
                for name, _ in self.pipe.dit.named_parameters()
                if any(part in name for part in source_attention_parts)
            }
        if checkpoint_source_attention_keys and not expected_source_attention_keys:
            raise ValueError(
                "Checkpoint contains independent source-attention weights, but the "
                "current configuration does not enable temporal-local source memory."
            )
        if checkpoint_source_attention_keys:
            missing_source_attention_keys = (
                expected_source_attention_keys - checkpoint_source_attention_keys
            )
            if missing_source_attention_keys:
                preview = sorted(missing_source_attention_keys)[:8]
                raise ValueError(
                    "Checkpoint contains a partial source-memory attention branch. "
                    f"Missing {len(missing_source_attention_keys)} keys; examples: {preview}"
                )

        print(f"Loading training weights from checkpoint: {ckpt_path}")
        if self.pipe.dit is not None and len(dit_state) > 0:
            load_result = self.pipe.dit.load_state_dict(dit_state, strict=False)
            print(
                f"  - Loaded dit keys: {len(dit_state)} "
                f"(missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})"
            )
            if expected_source_attention_keys and not checkpoint_source_attention_keys:
                self.pipe.dit.initialize_source_memory_attention_from_base()
                print(
                    "  - WARNING: migrated a legacy DiT checkpoint to independent "
                    "source attention with a zero-initialized output projection. "
                    "Retrain this checkpoint before inference; direct evaluation "
                    "does not reproduce the legacy shared-attention behavior."
                )
        elif len(dit_state) > 0:
            print(f"  - Warning: dit weights found ({len(dit_state)} keys), but pipeline.dit is None")

        if self.pipe.action_encoder is not None and len(action_state) > 0:
            load_result = self.pipe.action_encoder.load_state_dict(action_state, strict=False)
            print(
                f"  - Loaded action_encoder keys: {len(action_state)} "
                f"(missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})"
            )
        elif len(action_state) > 0:
            print(
                f"  - Warning: action_encoder weights found ({len(action_state)} keys), "
                "but pipeline.action_encoder is None"
            )

        projector = getattr(self.pipe, "source_video_projector", None)
        if projector is not None and len(projector_state) > 0:
            load_result = projector.load_state_dict(projector_state, strict=False)
            print(
                f"  - Loaded source_video_projector keys: {len(projector_state)} "
                f"(missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})"
            )
        elif len(projector_state) > 0:
            print(
                f"  - Warning: source_video_projector weights found ({len(projector_state)} keys), "
                "but pipeline.source_video_projector is None"
            )

        source_temporal_gate = getattr(self.pipe, "source_temporal_gate", None)
        if source_temporal_gate is not None and len(gate_state) > 0:
            load_result = source_temporal_gate.load_state_dict(gate_state, strict=False)
            print(
                f"  - Loaded source_temporal_gate keys: {len(gate_state)} "
                f"(missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})"
            )
        elif len(gate_state) > 0:
            print(
                f"  - Warning: source_temporal_gate weights found ({len(gate_state)} keys), "
                "but pipeline.source_temporal_gate is None"
            )

        target_state_head = getattr(self.pipe, "target_state_head", None)
        if target_state_head is not None and len(state_head_state) > 0:
            load_result = target_state_head.load_state_dict(state_head_state, strict=False)
            print(
                f"  - Loaded target_state_head keys: {len(state_head_state)} "
                f"(missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})"
            )
        elif len(state_head_state) > 0:
            print(
                f"  - Warning: target_state_head weights found ({len(state_head_state)} keys), "
                "but pipeline.target_state_head is None"
            )

        target_camera_encoder = getattr(self.pipe, "target_camera_encoder", None)
        if target_camera_encoder is not None and len(target_camera_state) > 0:
            load_result = target_camera_encoder.load_state_dict(target_camera_state, strict=False)
            print(
                f"  - Loaded target_camera_encoder keys: {len(target_camera_state)} "
                f"(missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})"
            )
        elif len(target_camera_state) > 0:
            print(
                f"  - Warning: target_camera_encoder weights found ({len(target_camera_state)} keys), "
                "but pipeline.target_camera_encoder is None"
            )

        scene_token_adapter = getattr(self.pipe, "scene_token_adapter", None)
        if scene_token_adapter is not None and len(scene_adapter_state) > 0:
            load_result = scene_token_adapter.load_state_dict(scene_adapter_state, strict=False)
            print(
                f"  - Loaded scene_token_adapter keys: {len(scene_adapter_state)} "
                f"(missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})"
            )
        elif len(scene_adapter_state) > 0:
            print(
                f"  - Warning: scene_token_adapter weights found ({len(scene_adapter_state)} keys), "
                "but pipeline.scene_token_adapter is None"
            )

        geometry_gates = getattr(self.pipe, "geometry_gates", None)
        if geometry_gates is not None and len(geometry_gates_state) > 0:
            load_result = geometry_gates.load_state_dict(geometry_gates_state, strict=False)
            print(
                f"  - Loaded geometry_gates keys: {len(geometry_gates_state)} "
                f"(missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})"
            )
        elif len(geometry_gates_state) > 0:
            print(
                f"  - Warning: geometry_gates weights found ({len(geometry_gates_state)} keys), "
                "but pipeline.geometry_gates is None"
            )

        scene_3d_noise_prior_adapter = getattr(
            self.pipe,
            "scene_3d_noise_prior_adapter",
            None,
        )
        if scene_3d_noise_prior_adapter is not None and len(scene_noise_prior_state) > 0:
            current_state = scene_3d_noise_prior_adapter.state_dict()
            compatible_state = {
                key: value
                for key, value in scene_noise_prior_state.items()
                if key in current_state and tuple(current_state[key].shape) == tuple(value.shape)
            }
            skipped_count = len(scene_noise_prior_state) - len(compatible_state)
            load_result = scene_3d_noise_prior_adapter.load_state_dict(
                compatible_state,
                strict=False,
            )
            print(
                f"  - Loaded scene_3d_noise_prior_adapter keys: {len(compatible_state)} "
                f"(skipped_shape_mismatch={skipped_count}, "
                f"missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})"
            )
        elif len(scene_noise_prior_state) > 0:
            print(
                f"  - Warning: scene_3d_noise_prior_adapter weights found "
                f"({len(scene_noise_prior_state)} keys), "
                "but pipeline.scene_3d_noise_prior_adapter is None"
            )

        action_noise_modulator = getattr(self.pipe, "action_noise_modulator", None)
        if action_noise_modulator is not None and len(action_noise_modulator_state) > 0:
            load_result = action_noise_modulator.load_state_dict(
                action_noise_modulator_state,
                strict=False,
            )
            print(
                f"  - Loaded action_noise_modulator keys: {len(action_noise_modulator_state)} "
                f"(missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})"
            )
        elif len(action_noise_modulator_state) > 0:
            print(
                f"  - Warning: action_noise_modulator weights found "
                f"({len(action_noise_modulator_state)} keys), "
                "but pipeline.action_noise_modulator is None"
            )

        if ignored_key_count > 0:
            print(f"  - Ignored {ignored_key_count} keys with unsupported `pipe.*` prefixes")

    def freeze_unused_prompt_params(self):
        dit = getattr(self.pipe, "dit", None)
        if dit is None:
            return
        text_embedding = getattr(dit, "text_embedding", None)
        if text_embedding is None:
            return
        text_embedding.requires_grad_(False)
        text_embedding.eval()
        blocks = getattr(dit, "blocks", None)
        if blocks is None:
            return
        for block in blocks:
            cross_attn = getattr(block, "cross_attn", None)
            if cross_attn is None:
                continue
            for name in ("k", "v", "norm_k"):
                module = getattr(cross_attn, name, None)
                if module is None:
                    continue
                module.requires_grad_(False)
                module.eval()

    def get_pipeline_inputs(self, data):
        inputs_posi = {
            "prompt": data.get("prompt"),
            "prompt_emb": data.get("prompt_emb"),
        }
        inputs_nega = {
            "negative_prompt": data.get("negative_prompt"),
            "prompt_emb": data.get("negative_prompt_emb"),
        }
        condition = data.get("action")
        if condition is None:
            condition = data.get("state")
        inputs_shared = {
            "input_video": data["video"],
            "action": condition,
            "height": int(data["video"].shape[-2]),
            "width": int(data["video"].shape[-1]),
            "num_frames": int(data["video"].shape[2]),
            "num_history_frames": self.num_history_frames,
            # Non-cross-view paths never use the dual-end anchor; keep
            # num_tail_frames=0 so WanVideoUnit_ImageEmbedderVAE behaves
            # exactly as the original head-only InP encoder.
            "num_tail_frames": 0,
            "seed": data.get("seed"),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][:, :, 0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared, inputs_posi, inputs_nega

    def build_cross_view_condition_video(self, video_gt: torch.Tensor, meta: dict | None = None) -> torch.Tensor:
        self.validate_cross_view_video(video_gt)
        cond_video = video_gt.clone()
        first_frame = self._load_wrist_first_frame(
            meta,
            (1, 1, 3, video_gt.shape[-2], video_gt.shape[-1]),
        )
        if first_frame is not None:
            cond_video[self.cross_view_target_view, :, 0] = first_frame.squeeze(0)
        else:
            placeholder = self.build_cross_view_placeholder(video_gt)
            cond_video[self.cross_view_target_view, :, 0] = placeholder

        # Dual-end anchor: populate cond_video[wrist, :, -1] with the indexed
        # wrist end-frame anchor. At training time
        # this drives WanVideoUnit_ImageEmbedderVAE to encode it as a known
        # frame in the integrated 81-frame VAE pass, eliminating the slot-0
        # vs slot-20 mismatch of the previous latent-overwrite design.
        # The tail-anchor dropout happens here in pixel space: with
        # probability cross_view_tail_anchor_dropout, replace the indexed tail
        # synth with a zero placeholder. The VAE encoder still sees a valid
        # 81-frame sequence; only the meaning of the tail pixel changes.
        if self.cross_view_use_tail_anchor:
            wrist = self.cross_view_target_view
            tail_target_shape = (1, 1, 3, video_gt.shape[-2], video_gt.shape[-1])
            should_dropout = (
                self.training
                and self.cross_view_tail_anchor_dropout > 0.0
                and torch.rand((), device=self.pipe.device).item()
                < self.cross_view_tail_anchor_dropout
            )
            tail_frame = (
                None if should_dropout
                else self._load_wrist_next_segment_first_frame(meta, tail_target_shape)
            )
            if tail_frame is not None:
                cond_video[wrist, :, -1] = tail_frame.squeeze(0)
            else:
                # Dropout fired OR index miss: use a zero placeholder and keep
                # the y-channel encoder input shape valid.
                cond_video[wrist, :, -1] = torch.zeros_like(cond_video[wrist, :, -1])
        if self.cross_view_use_keyframe_anchor:
            wrist = self.cross_view_target_view
            anchors = resolve_keyframe_anchors(self.keyframe_anchor_index, meta)
            for anchor in anchors:
                offset = int(anchor["offset"])
                cond_video[wrist, :, offset] = torch.zeros_like(cond_video[wrist, :, offset])
                if (
                    self.training
                    and self.keyframe_anchor_dropout > 0.0
                    and torch.rand((), device=self.pipe.device).item()
                    < self.keyframe_anchor_dropout
                ):
                    continue
                frame = self._load_wrist_frame_path(
                    anchor.get("path"),
                    (1, 1, 3, video_gt.shape[-2], video_gt.shape[-1]),
                )
                if frame is not None:
                    cond_video[wrist, :, offset] = frame.squeeze(0)
        return cond_video

    def build_cross_view_placeholder(self, video_gt: torch.Tensor) -> torch.Tensor:
        target_frame = video_gt[self.cross_view_target_view, :, 0]
        if self.cross_view_placeholder_mode == "zeros":
            return torch.zeros_like(target_frame)
        if self.cross_view_placeholder_mode == "source_mean":
            source_video = video_gt[list(self.cross_view_source_views)]
            mean_rgb = source_video.mean(dim=(0, 2, 3, 4))
            return mean_rgb[:, None, None].expand_as(target_frame)
        raise ValueError(
            f"Unsupported cross_view_placeholder_mode={self.cross_view_placeholder_mode!r}"
        )

    def validate_cross_view_video(self, video: torch.Tensor) -> None:
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise TypeError("cross-view training expects `video` with shape (V,C,T,H,W).")
        num_views = int(video.shape[0])
        if self.cross_view_target_view >= num_views:
            raise ValueError(
                f"Target view index {self.cross_view_target_view} is out of range for num_views={num_views}."
            )
        for index in self.cross_view_source_views:
            if index >= num_views:
                raise ValueError(
                    f"Source view index {index} is out of range for num_views={num_views}."
                )
            if index == self.cross_view_target_view:
                raise ValueError("Source view indices must not include the target view index.")

    def build_cross_view_inputs(self, data, cond_video: torch.Tensor, seed=None):
        inputs_posi = {
            "prompt": data.get("prompt"),
            "prompt_emb": data.get("prompt_emb"),
        }
        inputs_nega = {
            "negative_prompt": data.get("negative_prompt"),
            "prompt_emb": data.get("negative_prompt_emb"),
        }
        condition = data.get("action")
        if condition is None:
            condition = data.get("state")
        inputs_shared = {
            "input_video": cond_video,
            "action": condition,
            "height": int(cond_video.shape[-2]),
            "width": int(cond_video.shape[-1]),
            "num_frames": int(cond_video.shape[2]),
            "num_history_frames": self.num_history_frames,
            # Dual-end anchoring: the WAN-Fun-InP y-channel will encode
            # input_video[:, :, :num_history_frames] and (when set)
            # input_video[:, :, -num_tail_frames:] as KNOWN frames, with the
            # mask channels marking those latent slots as conditioning.
            # When dual-end anchor is disabled, num_tail_frames=0 reduces the
            # behavior to the original head-only InP path bit-for-bit.
            "num_tail_frames": (
                int(self.num_tail_frames) if self.cross_view_use_tail_anchor else 0
            ),
            "anchor_frame_indices": self.resolve_keyframe_anchor_frame_indices(data),
            "seed": data.get("seed") if seed is None else seed,
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = cond_video[:, :, 0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared, inputs_posi, inputs_nega

    def build_cross_view_cached_inputs(self, data):
        latent_views_gt = data.get("latent_views_gt")
        if not isinstance(latent_views_gt, torch.Tensor) or latent_views_gt.ndim != 5:
            raise ValueError(
                "Cached cross-view samples must include `latent_views_gt` with shape (V,C,T,H,W)."
            )
        condition = data.get("action")
        if condition is None:
            condition = data.get("state")
        inputs_shared = {
            "action": condition,
            "height": int(data["height"]),
            "width": int(data["width"]),
            "num_frames": int(data["num_frames"]),
            "num_history_frames": self.num_history_frames,
            # Plan A: cached training does not re-run WanVideoUnit_ImageEmbedderVAE
            # (the y channel is read from the cache directly). Still surface
            # num_tail_frames here so that any downstream unit that does run
            # — or future code that decides to recompute y on the fly — picks
            # up the dual-end state consistently.
            "num_tail_frames": (
                int(self.num_tail_frames) if self.cross_view_use_tail_anchor else 0
            ),
            "anchor_frame_indices": data.get("anchor_frame_indices"),
            "seed": data.get("seed"),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_posi = {
            "prompt": data.get("prompt"),
            "prompt_emb": data.get("prompt_emb"),
        }
        inputs_nega = {
            "negative_prompt": data.get("negative_prompt"),
            "prompt_emb": data.get("negative_prompt_emb"),
        }
        inputs = (inputs_shared, inputs_posi, inputs_nega)
        for unit in self.iter_cross_view_units(
            include_noise_initializer=False,
            include_legacy_image_branch=False,
        ):
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return inputs

    def resolve_keyframe_anchor_frame_indices(self, data: dict | None) -> list[int]:
        if not self.cross_view_use_keyframe_anchor:
            return []
        anchors = resolve_keyframe_anchors(self.keyframe_anchor_index, data)
        offsets = [int(anchor["offset"]) for anchor in anchors]
        if not offsets and data is not None and data.get("anchor_frame_indices") is not None:
            value = data["anchor_frame_indices"]
            if isinstance(value, torch.Tensor):
                offsets = [int(item) for item in value.detach().cpu().flatten().tolist()]
            elif isinstance(value, (list, tuple, set)):
                offsets = [int(item) for item in value]
            else:
                offsets = [int(value)]
        return sorted(set(offsets))

    def iter_cross_view_units(
        self,
        include_noise_initializer: bool = True,
        include_legacy_image_branch: bool = True,
    ):
        for unit in self.pipe.units:
            if isinstance(unit, WanVideoUnit_InputVideoEmbedder):
                continue
            if (not include_noise_initializer) and isinstance(
                unit, WanVideoUnit_NoiseInitializer
            ):
                continue
            if (not include_legacy_image_branch) and isinstance(
                unit,
                (WanVideoUnit_ImageEmbedderVAE, WanVideoUnit_ImageEmbedderCLIP),
            ):
                continue
            yield unit

    def build_cross_view_legacy_image_branch(
        self,
        data,
        cond_video: torch.Tensor,
    ) -> dict:
        inputs = self.build_cross_view_inputs(data, cond_video)
        for unit in self.pipe.units:
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
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        inputs_shared, _, _ = inputs
        outputs = {}
        if "y" in inputs_shared:
            outputs["y"] = inputs_shared["y"]
        if "clip_feature" in inputs_shared:
            outputs["clip_feature"] = inputs_shared["clip_feature"]
        return outputs

    def encode_video_latents_by_view(self, video: torch.Tensor) -> torch.Tensor:
        self.pipe.load_models_to_device(["vae"])
        video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        latent_views = self.pipe.vae.encode(video, device=self.pipe.device, tiled=False)
        return latent_views.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

    def merge_view_latents(self, latent_views: torch.Tensor) -> torch.Tensor:
        if latent_views.ndim == 5:
            return rearrange(latent_views, "v c t h w -> 1 c t (v h) w")
        if latent_views.ndim == 6:
            return rearrange(latent_views, "b v c t h w -> b c t (v h) w")
        raise ValueError(
            "Expected latent views with shape (V,C,T,H,W) or (B,V,C,T,H,W)."
        )

    def encode_joint_video_latents(self, video: torch.Tensor) -> torch.Tensor:
        return self.merge_view_latents(self.encode_video_latents_by_view(video))

    def build_target_history_condition_video(self, video_gt: torch.Tensor, meta: dict | None = None) -> torch.Tensor:
        target_video = torch.zeros_like(
            video_gt[self.cross_view_target_view : self.cross_view_target_view + 1]
        )
        placeholder = self.build_cross_view_placeholder(video_gt).unsqueeze(0)
        target_video[:, :, 0] = placeholder
        first_frame = self._load_wrist_first_frame(meta, target_video.shape)
        if first_frame is not None:
            target_video[:, :, 0] = first_frame
        return target_video

    def _load_wrist_first_frame(self, meta: dict | None, target_shape) -> torch.Tensor | None:
        return self._load_wrist_indexed_frame(meta, target_shape, start_frame_offset=0)

    def _load_wrist_next_segment_first_frame(
        self, meta: dict | None, target_shape, segment_stride: int = 81,
    ) -> torch.Tensor | None:
        """Load wrist tail anchor for the current segment.

        New frame-indexed wrist indexes store this at (episode, end_frame).
        For older first-frame-only indexes, fall back to (episode,
        start_frame+segment_stride). Returns None when:
          - meta missing episode_index and usable frame keys
          - wrist_first_frame_index not loaded
          - neither the end-frame key nor the compatibility key is available
        """
        tail_frame = self._load_wrist_indexed_frame(meta, target_shape, frame_field="end_frame")
        if tail_frame is not None:
            return tail_frame
        return self._load_wrist_indexed_frame(
            meta,
            target_shape,
            frame_field="start_frame",
            frame_offset=int(segment_stride),
        )

    def _load_wrist_indexed_frame(
        self,
        meta: dict | None,
        target_shape,
        frame_field: str = "start_frame",
        frame_offset: int = 0,
        start_frame_offset: int | None = None,
    ) -> torch.Tensor | None:
        if meta is None or self.wrist_first_frame_index is None:
            return None
        if start_frame_offset is not None:
            frame_field = "start_frame"
            frame_offset = int(start_frame_offset)
        episode_index = meta.get("episode_index")
        frame_index = meta.get(frame_field)
        if isinstance(episode_index, torch.Tensor):
            episode_index = int(episode_index.item()) if episode_index.numel() == 1 else int(episode_index.flatten()[0].item())
        if isinstance(frame_index, torch.Tensor):
            frame_index = int(frame_index.item()) if frame_index.numel() == 1 else int(frame_index.flatten()[0].item())
        if episode_index is None or frame_index is None:
            return None
        key = f"{episode_index}_{int(frame_index) + int(frame_offset)}"
        path = self.wrist_first_frame_index.get(key)
        if path is None or not os.path.exists(path):
            return None
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(path).convert("RGB")
            _, _, _, H, W = target_shape
            img = img.resize((W, H), Image.BICUBIC)
            arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            return tensor.to(
                device=self.pipe.device, dtype=self.pipe.torch_dtype
            )
        except Exception as exc:
            print(f"[wrist_first_frame] load failed for {key}: {exc}")
            return None

    def _load_wrist_frame_path(self, path: str | None, target_shape) -> torch.Tensor | None:
        if path is None or not os.path.exists(path):
            return None
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(path).convert("RGB")
            _, _, _, H, W = target_shape
            img = img.resize((W, H), Image.BICUBIC)
            arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            return tensor.to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
        except Exception as exc:
            print(f"[keyframe_anchor] load failed for {path}: {exc}")
            return None

    def _scalar_int_from_data(self, data: dict, key: str, default: int) -> int:
        value = data.get(key, default)
        if isinstance(value, torch.Tensor):
            value = value.flatten()[0].item()
        elif isinstance(value, (list, tuple)):
            value = value[0]
        return int(value)

    def _maybe_overwrite_target_history_with_wrist_first_frame(
        self,
        target_history_latents: torch.Tensor,
        data: dict,
    ) -> torch.Tensor:
        if self.wrist_first_frame_index is None:
            return target_history_latents
        history_t = int(target_history_latents.shape[2])
        if history_t != 1:
            return target_history_latents
        height = self._scalar_int_from_data(data, "height", 180)
        width = self._scalar_int_from_data(data, "width", 320)
        num_frames = self._scalar_int_from_data(data, "num_frames", 1)
        first_frame = self._load_wrist_first_frame(data, (1, 1, 3, height, width))
        if first_frame is None:
            return target_history_latents
        vae = getattr(self.pipe, "vae", None)
        if vae is None:
            return target_history_latents
        try:
            video_1view = torch.zeros(
                1,
                3,
                num_frames,
                height,
                width,
                dtype=self.pipe.torch_dtype,
                device=self.pipe.device,
            )
            video_1view[:, :, 0] = first_frame
            latent = vae.encode(video_1view, device=self.pipe.device, tiled=False)
            latent = latent.to(dtype=target_history_latents.dtype, device=target_history_latents.device)
            updated = target_history_latents.clone()
            updated[:, :, :1] = latent[:, :, :1]
            return updated
        except Exception as exc:
            print(f"[wrist_first_frame] VAE encode failed: {exc}")
        return target_history_latents

    def _resolve_target_tail_latents(
        self, data: dict, tail_t: int
    ) -> torch.Tensor | None:
        """读取 cache 中的 target_tail_latents 并应用训练时的 dropout augmentation.

        - tail_t <= 0 或未启用 dual-end anchor: 返回 None, 调用方退化为单端
        - cache 中无该字段:                       返回 None, 兼容旧 cache
        - 训练时按概率 dropout:                   返回 None (相当于 zero placeholder)
        - 否则                                     返回 (B,C,T,H,W) 的 latent
        """
        if tail_t <= 0 or not self.cross_view_use_tail_anchor:
            return None
        tail = data.get("target_tail_latents")
        if tail is None:
            return None
        # Augmentation: drop the tail anchor with probability p during training.
        if (
            self.training
            and self.cross_view_tail_anchor_dropout > 0.0
            and torch.rand((), device=self.pipe.device).item()
            < self.cross_view_tail_anchor_dropout
        ):
            return None
        return tail.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

    def build_target_history_latents(self, video_gt: torch.Tensor, meta: dict | None = None) -> torch.Tensor:
        history_video = self.build_target_history_condition_video(video_gt, meta=meta)
        history_latents = self.encode_video_latents_by_view(history_video)
        return self.merge_view_latents(history_latents)

    def to_condition_tensor(self, condition) -> torch.Tensor | None:
        if condition is None:
            return None
        condition = torch.as_tensor(
            condition,
            device=self.pipe.device,
            dtype=self.pipe.torch_dtype,
        )
        if condition.ndim == 2:
            condition = condition.unsqueeze(0)
        return condition

    def downsample_condition_sequence(
        self,
        condition,
        num_frames: int,
    ) -> torch.Tensor | None:
        condition = self.to_condition_tensor(condition)
        if condition is None:
            return None
        latent_length = ((int(num_frames) - 1) // 4) + 1
        condition = torch.cat(
            [torch.repeat_interleave(condition[:, 0:1], repeats=4, dim=1), condition[:, 1:]],
            dim=1,
        )
        target_length = latent_length * 4
        if condition.shape[1] < target_length:
            padding = condition[:, -1:].repeat(1, target_length - condition.shape[1], 1)
            condition = torch.cat([condition, padding], dim=1)
        else:
            condition = condition[:, :target_length]
        return condition.reshape(condition.shape[0], latent_length, 4, condition.shape[-1]).mean(dim=2)

    def get_cross_view_condition_sequence(self, data, num_frames: int) -> torch.Tensor | None:
        condition = data.get("state")
        if condition is None:
            condition = data.get("action")
        return self.downsample_condition_sequence(condition, num_frames)

    def build_cross_view_source_condition(
        self,
        video_gt: torch.Tensor | None = None,
        condition_sequence: torch.Tensor | None = None,
        source_latents: torch.Tensor | None = None,
    ) -> dict:
        if self.cross_view_source_injection_mode == "none":
            return {}
        projector = getattr(self.pipe, "source_video_projector", None)
        if projector is None:
            return {}
        if source_latents is None:
            if video_gt is None:
                raise ValueError(
                    "Either `video_gt` or `source_latents` must be provided for "
                    "cross-view source conditioning."
                )
            source_video = video_gt[list(self.cross_view_source_views)]
            source_latents = self.encode_video_latents_by_view(source_video).unsqueeze(0)
        else:
            if source_latents.ndim == 5:
                source_latents = source_latents.unsqueeze(0)
            source_latents = source_latents.to(
                dtype=self.pipe.torch_dtype,
                device=self.pipe.device,
            )
        source_features = projector(source_latents)
        source_temporal_gate = getattr(self.pipe, "source_temporal_gate", None)
        if (
            self.cross_view_source_injection_mode != "temporal_local"
            and source_temporal_gate is not None
        ):
            source_features = source_temporal_gate(source_features, condition_sequence)
        if self.cross_view_source_injection_mode == "temporal_local":
            return {
                "source_memory_by_time": source_features,
                "source_window_radius": self.cross_view_source_window_radius,
            }
        return {"source_tokens": source_features}

    def build_cross_view_source_tokens(self, video_gt: torch.Tensor) -> torch.Tensor | None:
        return self.build_cross_view_source_condition(video_gt).get("source_tokens")

    def build_cross_view_global_source_tokens(
        self,
        source_features: torch.Tensor | None,
    ) -> torch.Tensor | None:
        num_tokens = int(self.cross_view_global_source_tokens)
        if source_features is None or num_tokens <= 0:
            return None
        if source_features.ndim == 4:
            pooled = source_features.mean(dim=2)
            if pooled.shape[1] == num_tokens:
                return pooled.contiguous()
            pooled = pooled.transpose(1, 2)
            pooled = F.interpolate(
                pooled,
                size=num_tokens,
                mode="linear",
                align_corners=False,
            )
            return pooled.transpose(1, 2).contiguous()
        if source_features.ndim == 3:
            if source_features.shape[1] == num_tokens:
                return source_features.contiguous()
            pooled = source_features.transpose(1, 2)
            pooled = F.interpolate(
                pooled,
                size=num_tokens,
                mode="linear",
                align_corners=False,
            )
            return pooled.transpose(1, 2).contiguous()
        raise ValueError(
            "Expected source features with shape (B,T,N,D) or (B,N,D) when "
            "building global source tokens."
        )

    def _build_geometry_aware_inputs(self, data) -> dict:
        extractor = getattr(self.pipe, "scene_token_extractor", None)
        adapter = getattr(self.pipe, "scene_token_adapter", None)
        gates = getattr(self.pipe, "geometry_gates", None)
        if extractor is None or adapter is None or gates is None:
            return {}
        if self.geometry_scene_token_source == "camera_aware_sidecar":
            cached_scene_tokens = data.get("scene_tokens_camera_aware")
            if cached_scene_tokens is None:
                raise KeyError(
                    "Geometry scene token source is `camera_aware_sidecar`, but "
                    "`scene_tokens_camera_aware` is missing. Build or attach the geometry sidecar cache."
                )
            raw_tokens = cached_scene_tokens.to(
                dtype=self.pipe.torch_dtype, device=self.pipe.device
            )
            scene_tokens = adapter(raw_tokens)
            return {"scene_tokens": scene_tokens, "geometry_gates": gates}
        cached_scene_tokens = (
            None if self.geometry_scene_token_source == "runtime" else data.get("scene_tokens")
        )
        if cached_scene_tokens is not None:
            raw_tokens = cached_scene_tokens.to(
                dtype=self.pipe.torch_dtype, device=self.pipe.device
            )
        else:
            # Fallback: 从 cache 中保存的 source 首帧像素实时提取
            source_first_frames = data.get("source_first_frames")
            if source_first_frames is None:
                return {}
            source_first_frames = source_first_frames.to(
                dtype=self.pipe.torch_dtype, device=self.pipe.device
            ).unsqueeze(0)  # (1, V, 3, H, W)
            cam_tokens = self.select_source_camera_tokens(data) if self.geometry_use_camera_tokens else None
            raw_tokens = extractor(source_first_frames, cam_tokens)
        scene_tokens = adapter(raw_tokens)
        return {"scene_tokens": scene_tokens, "geometry_gates": gates}

    def _build_geometry_aware_inputs_from_video(self, video_gt: torch.Tensor, data: dict | None = None) -> dict:
        extractor = getattr(self.pipe, "scene_token_extractor", None)
        adapter = getattr(self.pipe, "scene_token_adapter", None)
        gates = getattr(self.pipe, "geometry_gates", None)
        if extractor is None or adapter is None or gates is None:
            return {}
        source_first_frames = video_gt[list(self.cross_view_source_views), :, 0]
        source_first_frames = source_first_frames.unsqueeze(0)
        cam_tokens = self.select_source_camera_tokens(data) if self.geometry_use_camera_tokens else None
        if cam_tokens is None:
            cam_tokens = torch.zeros(
                1, len(self.cross_view_source_views), 11,
                device=self.pipe.device, dtype=self.pipe.torch_dtype,
            )
        raw_tokens = extractor(source_first_frames, cam_tokens)
        scene_tokens = adapter(raw_tokens)
        return {"scene_tokens": scene_tokens, "geometry_gates": gates}

    def select_source_camera_tokens(self, data: dict | None) -> torch.Tensor | None:
        if data is None:
            return None
        tokens = data.get("source_cam_tokens")
        if tokens is None:
            tokens = data.get("source_camera_tokens")
        if tokens is None:
            return None
        tokens = torch.as_tensor(tokens, device=self.pipe.device, dtype=self.pipe.torch_dtype)
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        return tokens

    def select_target_camera_tokens(
        self,
        data: dict | None,
        latent_length: int | None = None,
    ) -> torch.Tensor | None:
        if data is None:
            return None
        tokens = data.get("target_cam_tokens_latent")
        if tokens is not None:
            tokens = torch.as_tensor(tokens, device=self.pipe.device, dtype=self.pipe.torch_dtype)
            if tokens.ndim == 2:
                tokens = tokens.unsqueeze(0)
            if latent_length is not None and tokens.shape[1] != latent_length:
                tokens = self.downsample_camera_sequence(tokens, latent_length)
            return tokens
        tokens = data.get("target_cam_tokens")
        if tokens is None:
            tokens = data.get("target_camera_tokens")
        if tokens is None:
            return None
        tokens = torch.as_tensor(tokens, device=self.pipe.device, dtype=self.pipe.torch_dtype)
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        if latent_length is not None and tokens.shape[1] != latent_length:
            tokens = self.downsample_camera_sequence(tokens, latent_length)
        return tokens

    def downsample_camera_sequence(
        self,
        camera_tokens: torch.Tensor,
        latent_length: int,
    ) -> torch.Tensor:
        if camera_tokens.shape[1] == latent_length:
            return camera_tokens
        target_length = int(latent_length) * 4
        tokens = torch.cat(
            [
                torch.repeat_interleave(camera_tokens[:, 0:1], repeats=4, dim=1),
                camera_tokens[:, 1:],
            ],
            dim=1,
        )
        if tokens.shape[1] < target_length:
            tokens = torch.cat(
                [tokens, tokens[:, -1:].repeat(1, target_length - tokens.shape[1], 1)],
                dim=1,
            )
        else:
            tokens = tokens[:, :target_length]
        return tokens.reshape(tokens.shape[0], latent_length, 4, tokens.shape[-1]).mean(dim=2)

    def build_target_camera_condition(
        self,
        data: dict | None,
        latent_length: int | None,
    ) -> dict:
        if self.geometry_target_camera_mode != "add_time_mlp":
            return {}
        encoder = getattr(self.pipe, "target_camera_encoder", None)
        if encoder is None:
            return {}
        camera_tokens = self.select_target_camera_tokens(data, latent_length=latent_length)
        if camera_tokens is None:
            return {}
        target_camera_emb = encoder(camera_tokens)
        return {"target_camera_emb": target_camera_emb}

    def is_3d_noise_prior_enabled(self) -> bool:
        return (
            self.cross_view_stage == 2
            and self.cross_view_3d_noise_prior_mode != "none"
            and self.cross_view_3d_noise_prior_weight > 0
        )

    def build_anchor_noise_prior_mask(
        self,
        y: torch.Tensor | None,
        latent_shape: tuple[int, int, int, int, int],
    ) -> torch.Tensor | None:
        if y is None or self.cross_view_3d_noise_anchor_attenuation <= 0:
            return None
        if not isinstance(y, torch.Tensor) or y.ndim != 5 or y.shape[1] < 4:
            return None
        batch_size = int(latent_shape[0])
        latent_length = int(latent_shape[2])
        mask = y[:, :4].amax(dim=(1, 3, 4)).float().clamp(0.0, 1.0)
        mask = mask[:, None, :, None, None]
        if mask.shape[2] != latent_length:
            mask_1d = mask.squeeze(-1).squeeze(-1)
            mask_1d = F.interpolate(mask_1d, size=latent_length, mode="nearest")
            mask = mask_1d.unsqueeze(-1).unsqueeze(-1)
        if mask.shape[0] == 1 and batch_size != 1:
            mask = mask.expand(batch_size, -1, -1, -1, -1)
        elif mask.shape[0] != batch_size:
            raise ValueError(
                "Anchor mask batch size "
                f"{mask.shape[0]} does not match latent batch size {batch_size}."
            )
        return mask.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

    def build_stage2_3d_noise_prior(
        self,
        gaussian_noise: torch.Tensor,
        scene_tokens: torch.Tensor | None,
        condition_sequence: torch.Tensor | None,
        y: torch.Tensor | None = None,
        source_view_ids=None,
        target_view_id: int | None = None,
    ) -> torch.Tensor:
        if not self.is_3d_noise_prior_enabled():
            return gaussian_noise
        if scene_tokens is None:
            raise ValueError(
                f"cross_view_3d_noise_prior_mode={self.cross_view_3d_noise_prior_mode!r} requires "
                "`scene_tokens` in the stage2 inputs."
            )
        if condition_sequence is None:
            raise ValueError(
                f"cross_view_3d_noise_prior_mode={self.cross_view_3d_noise_prior_mode!r} requires "
                "an action/state condition sequence."
            )
        scene_adapter = getattr(self.pipe, "scene_3d_noise_prior_adapter", None)
        if scene_adapter is None:
            raise ValueError(
                "3D noise prior is enabled, but the scene noise-prior adapter "
                "was not initialized."
            )

        scene_tokens = scene_tokens.to(
            dtype=gaussian_noise.dtype,
            device=gaussian_noise.device,
        )
        if self.cross_view_3d_noise_prior_mode == "dynamic_view_action":
            structured_noise = scene_adapter(
                scene_tokens,
                tuple(gaussian_noise.shape),
                condition_sequence,
                source_view_ids=(
                    self.cross_view_source_views
                    if source_view_ids is None else source_view_ids
                ),
                target_view_id=(
                    self.cross_view_target_view
                    if target_view_id is None else target_view_id
                ),
            )
        else:
            action_modulator = getattr(self.pipe, "action_noise_modulator", None)
            if action_modulator is None:
                raise ValueError(
                    "scene_action_grid 3D noise prior is enabled, but the action "
                    "noise modulator was not initialized."
                )
            structured_noise = scene_adapter(scene_tokens, tuple(gaussian_noise.shape))
        if structured_noise.shape[0] == 1 and gaussian_noise.shape[0] != 1:
            structured_noise = structured_noise.expand_as(gaussian_noise)
        if structured_noise.shape != gaussian_noise.shape:
            raise ValueError(
                "3D structured noise shape "
                f"{tuple(structured_noise.shape)} does not match Gaussian noise shape "
                f"{tuple(gaussian_noise.shape)}."
            )

        if self.cross_view_3d_noise_prior_mode == "scene_action_grid":
            structured_noise = action_modulator(structured_noise, condition_sequence)
        lambda_eff = torch.full(
            (1, 1, 1, 1, 1),
            self.cross_view_3d_noise_prior_weight,
            dtype=gaussian_noise.dtype,
            device=gaussian_noise.device,
        )
        anchor_mask = self.build_anchor_noise_prior_mask(y, tuple(gaussian_noise.shape))
        if anchor_mask is not None:
            anchor_mask = anchor_mask.to(
                dtype=gaussian_noise.dtype,
                device=gaussian_noise.device,
            )
            attenuation = self.cross_view_3d_noise_anchor_attenuation
            lambda_eff = lambda_eff * (1.0 - attenuation * anchor_mask).clamp(0.0, 1.0)

        gaussian_coeff = (1.0 - lambda_eff.square()).clamp_min(0.0).sqrt()
        mixed_noise = gaussian_coeff * gaussian_noise + lambda_eff * structured_noise
        return normalize_noise_like(mixed_noise)

    def detach_3d_noise_target_if_needed(self, noise: torch.Tensor) -> torch.Tensor:
        if self.is_3d_noise_prior_enabled():
            return noise.detach()
        return noise

    def compute_geometry_alignment_loss(
        self,
        hidden_by_time: torch.Tensor | None,
        scene_tokens: torch.Tensor | None,
        num_views: int,
        history_t: int,
        target_only: bool = False,
    ) -> torch.Tensor | None:
        if hidden_by_time is None or scene_tokens is None:
            return None
        target_hidden = hidden_by_time if target_only else self.select_target_hidden_by_view(hidden_by_time, num_views)
        if target_hidden is None or target_hidden.shape[1] <= history_t:
            return None
        target_hidden = target_hidden[:, history_t:]
        pooled_hidden = target_hidden.mean(dim=(1, 2, 3))
        pooled_scene = scene_tokens.mean(dim=1)
        cos_sim = F.cosine_similarity(pooled_hidden.float(), pooled_scene.float(), dim=-1)
        return (1.0 - cos_sim).mean()

    def sample_training_timestep(self):
        total_steps = len(self.pipe.scheduler.timesteps)
        max_boundary = int(self.max_timestep_boundary * total_steps)
        min_boundary = int(self.min_timestep_boundary * total_steps)
        max_boundary = max(min(total_steps, max_boundary), min_boundary + 1)
        timestep_id = torch.randint(min_boundary, max_boundary, (1,))
        return self.pipe.scheduler.timesteps[timestep_id].to(
            dtype=self.pipe.torch_dtype,
            device=self.pipe.device,
        )

    def split_joint_views(self, tensor: torch.Tensor, num_views: int) -> torch.Tensor:
        if tensor.shape[-2] % num_views != 0:
            raise ValueError(
                f"Joint latent height {tensor.shape[-2]} is not divisible by num_views={num_views}."
            )
        view_height = tensor.shape[-2] // num_views
        return rearrange(
            tensor,
            "b c t (v h) w -> b v c t h w",
            v=num_views,
            h=view_height,
        )

    def combine_joint_views(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 6:
            raise ValueError("Expected view tensor with shape (B,V,C,T,H,W).")
        return rearrange(tensor, "b v c t h w -> b c t (v h) w")

    def overwrite_joint_view_latents(
        self,
        joint_latents: torch.Tensor,
        view_latents: torch.Tensor,
        view_indices: tuple[int, ...],
        num_views: int,
    ) -> torch.Tensor:
        if view_latents.ndim == 5:
            view_latents = view_latents.unsqueeze(0)
        joint_views = self.split_joint_views(joint_latents, num_views).clone()
        if view_latents.ndim != 6:
            raise ValueError("Expected replacement view latents with shape (B,V,C,T,H,W).")
        if view_latents.shape[0] != joint_views.shape[0]:
            raise ValueError("Batch size mismatch when overwriting joint view latents.")
        if view_latents.shape[1] != len(view_indices):
            raise ValueError(
                "Replacement view count does not match the number of target view indices."
            )
        joint_views[:, list(view_indices)] = view_latents.to(
            dtype=joint_views.dtype,
            device=joint_views.device,
        )
        return self.combine_joint_views(joint_views)

    def overwrite_target_history_latents(
        self,
        joint_latents: torch.Tensor,
        target_history_latents: torch.Tensor,
        num_views: int,
        history_t: int,
    ) -> torch.Tensor:
        if target_history_latents.ndim != 5:
            raise ValueError("Expected target history latents with shape (B,C,T,H,W).")
        joint_views = self.split_joint_views(joint_latents, num_views).clone()
        if target_history_latents.shape[0] != joint_views.shape[0]:
            raise ValueError("Batch size mismatch when overwriting target history latents.")
        joint_views[:, self.cross_view_target_view, :, :history_t] = target_history_latents[
            :, :, :history_t
        ].to(dtype=joint_views.dtype, device=joint_views.device)
        return self.combine_joint_views(joint_views)

    def build_stage2_source_branch_latents(
        self,
        source_x0_latents: torch.Tensor,
        source_noise_latents: torch.Tensor,
        timestep: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.cross_view_source_branch_mode == "none":
            return source_noise_latents
        if self.cross_view_source_branch_mode != "sigma_matched_clamp":
            raise ValueError(
                f"Unsupported cross_view_source_branch_mode="
                f"{self.cross_view_source_branch_mode!r}"
            )
        if timestep is None:
            return source_x0_latents
        return self.pipe.scheduler.add_noise(
            source_x0_latents,
            source_noise_latents,
            timestep,
        )

    def apply_cross_view_stage2_constraints(
        self,
        latents: torch.Tensor,
        num_views: int,
        history_t: int,
        target_history_latents: torch.Tensor,
        source_x0_latents: torch.Tensor | None = None,
        source_noise_latents: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        constrained = latents
        if source_x0_latents is not None and source_noise_latents is not None:
            source_branch_latents = self.build_stage2_source_branch_latents(
                source_x0_latents,
                source_noise_latents,
                timestep,
            )
            constrained = self.overwrite_joint_view_latents(
                constrained,
                source_branch_latents,
                self.cross_view_source_views,
                num_views=num_views,
            )
        return self.overwrite_target_history_latents(
            constrained,
            target_history_latents,
            num_views=num_views,
            history_t=history_t,
        )

    def timestep_to_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep_value = timestep.detach().to(self.pipe.scheduler.timesteps.device)
        timestep_id = torch.argmin((self.pipe.scheduler.timesteps - timestep_value).abs())
        sigma = self.pipe.scheduler.sigmas[timestep_id]
        return sigma.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

    def get_old_branch_dropout_probability(self) -> float:
        base_probability = max(0.0, min(1.0, self.cross_view_old_branch_dropout))
        schedule = self.cross_view_legacy_branch_schedule
        if schedule == "anchor_then_dropout":
            total_steps = max(1, int(getattr(self, "cross_view_total_training_steps", 0)))
            current_step = max(0, int(getattr(self, "cross_view_current_training_step", 0)))
            progress = min(1.0, current_step / total_steps)
            if progress <= 0.2:
                return 0.0
            if progress >= 0.8:
                return base_probability
            return base_probability * ((progress - 0.2) / 0.6)
        if self.cross_view_old_branch_dropout_schedule != "linear_warmup_to_high":
            return base_probability
        total_steps = max(1, int(getattr(self, "cross_view_total_training_steps", 0)))
        current_step = max(0, int(getattr(self, "cross_view_current_training_step", 0)))
        progress = min(1.0, current_step / total_steps)
        target_probability = max(base_probability, 0.9)
        return min(0.95, base_probability + (target_probability - base_probability) * progress)

    def get_cross_view_aux_loss_scale(self) -> float:
        warmup_ratio = float(self.cross_view_aux_loss_warmup_ratio)
        if warmup_ratio <= 0:
            return 1.0
        total_steps = max(1, int(getattr(self, "cross_view_total_training_steps", 0)))
        current_step = max(0, int(getattr(self, "cross_view_current_training_step", 0)))
        progress = min(1.0, current_step / total_steps)
        if progress <= warmup_ratio:
            return 0.0
        remaining = max(1e-6, 1.0 - warmup_ratio)
        return min(1.0, (progress - warmup_ratio) / remaining)

    def get_alignment_loss_scale(self) -> float:
        warmup_ratio = max(0.0, min(1.0, float(self.alignment_loss_warmup_ratio)))
        if warmup_ratio <= 0:
            return 1.0
        total_steps = max(1, int(getattr(self, "cross_view_total_training_steps", 0)))
        current_step = max(0, int(getattr(self, "cross_view_current_training_step", 0)))
        progress = min(1.0, current_step / total_steps)
        return min(1.0, progress / max(1e-6, warmup_ratio))

    def select_target_hidden_by_view(
        self,
        hidden_by_time: torch.Tensor | None,
        num_views: int,
    ) -> torch.Tensor | None:
        if hidden_by_time is None:
            return None
        if hidden_by_time.shape[2] % num_views != 0:
            raise ValueError(
                f"Hidden token height {hidden_by_time.shape[2]} is not divisible by num_views={num_views}."
            )
        view_token_height = hidden_by_time.shape[2] // num_views
        hidden_views = hidden_by_time.reshape(
            hidden_by_time.shape[0],
            hidden_by_time.shape[1],
            num_views,
            view_token_height,
            hidden_by_time.shape[3],
            hidden_by_time.shape[4],
        )
        return hidden_views[:, :, self.cross_view_target_view]

    def compute_cross_view_temporal_loss(
        self,
        latents_future: torch.Tensor,
        noise_pred_future: torch.Tensor,
        input_latents_gt: torch.Tensor,
        timestep: torch.Tensor,
        num_views: int,
        history_t: int,
        target_only: bool = False,
        tail_t: int = 0,
    ) -> torch.Tensor | None:
        if noise_pred_future.shape[2] < 2:
            return None
        sigma = self.timestep_to_sigma(timestep).reshape(
            1, *([1] * (latents_future.ndim - 1))
        )
        pred_x0 = latents_future - sigma * noise_pred_future
        # When dual-end anchoring is enabled (tail_t > 0), `latents_future` and
        # `noise_pred_future` already exclude the tail anchor; we must apply the
        # same exclusion to `input_latents_gt` so that gt_target shape matches
        # pred_target shape after delta computation.
        gt_future_end = -tail_t if tail_t > 0 else None
        if target_only:
            pred_target = pred_x0
            gt_target = input_latents_gt[:, :, history_t:gt_future_end]
        else:
            pred_views = self.split_joint_views(pred_x0, num_views)
            gt_views = self.split_joint_views(input_latents_gt[:, :, history_t:gt_future_end], num_views)
            pred_target = pred_views[:, self.cross_view_target_view]
            gt_target = gt_views[:, self.cross_view_target_view]
        if pred_target.shape[2] < 2:
            return None
        pred_delta = pred_target[:, :, 1:] - pred_target[:, :, :-1]
        gt_delta = gt_target[:, :, 1:] - gt_target[:, :, :-1]
        return F.mse_loss(pred_delta.float(), gt_delta.float())

    def compute_cross_view_state_loss(
        self,
        hidden_by_time: torch.Tensor | None,
        condition_sequence: torch.Tensor | None,
        num_views: int,
        history_t: int,
        target_only: bool = False,
    ) -> torch.Tensor | None:
        target_state_head = getattr(self.pipe, "target_state_head", None)
        if target_state_head is None or hidden_by_time is None or condition_sequence is None:
            return None
        target_hidden = hidden_by_time if target_only else self.select_target_hidden_by_view(hidden_by_time, num_views)
        if target_hidden is None or target_hidden.shape[1] <= history_t:
            return None
        target_hidden = target_hidden[:, history_t:]
        target_state = condition_sequence[:, history_t:]
        if target_state.shape[1] == 0:
            return None
        valid_length = min(target_hidden.shape[1], target_state.shape[1])
        pooled_hidden = target_hidden[:, :valid_length].mean(dim=(2, 3))
        target_state = target_state[:, :valid_length]
        pred_state = target_state_head(pooled_hidden)
        return F.mse_loss(pred_state.float(), target_state.float())

    def cross_view_weighted_loss(
        self,
        noise_pred: torch.Tensor,
        training_target: torch.Tensor,
        timestep: torch.Tensor,
        num_views: int,
        target_only: bool,
    ) -> torch.Tensor:
        pred_views = self.split_joint_views(noise_pred, num_views)
        target_views = self.split_joint_views(training_target, num_views)
        target_loss = F.mse_loss(
            pred_views[:, self.cross_view_target_view].float(),
            target_views[:, self.cross_view_target_view].float(),
        )
        if target_only:
            loss = target_loss
        else:
            source_indices = [
                index for index in range(num_views) if index != self.cross_view_target_view
            ]
            if len(source_indices) == 0 or self.cross_view_source_loss_weight <= 0:
                loss = target_loss
            else:
                source_loss = 0.0
                for index in source_indices:
                    source_loss = source_loss + F.mse_loss(
                        pred_views[:, index].float(),
                        target_views[:, index].float(),
                    )
                source_loss = source_loss / len(source_indices)
                loss = target_loss + self.cross_view_source_loss_weight * source_loss
        return loss * self.pipe.scheduler.training_weight(timestep)

    def get_cross_view_latent_grid(
        self,
        num_views: int,
        num_frames: int,
        height: int,
        width: int,
    ) -> tuple[int, int, int]:
        latent_length = ((int(num_frames) - 1) // 4) + 1
        latent_height = (int(height) * int(num_views)) // int(self.pipe.vae.upsampling_factor)
        latent_width = int(width) // int(self.pipe.vae.upsampling_factor)
        return latent_length, latent_height, latent_width

    def build_zero_legacy_image_branch_tensors(
        self,
        num_views: int,
        num_frames: int,
        height: int,
        width: int,
        batch_size: int = 1,
        latent_view_height: int | None = None,
        target_only: bool = False,
    ) -> dict:
        outputs = {}
        dit = getattr(self.pipe, "dit", None)
        if dit is None or not getattr(dit, "has_image_input", False):
            return outputs
        if latent_view_height is None:
            latent_length, latent_height, latent_width = self.get_cross_view_latent_grid(
                num_views=num_views,
                num_frames=num_frames,
                height=height,
                width=width,
            )
            if target_only:
                latent_height = latent_height // int(num_views)
        else:
            latent_length = ((int(num_frames) - 1) // 4) + 1
            latent_height = int(latent_view_height)
            latent_width = int(width) // int(self.pipe.vae.upsampling_factor)
        if getattr(dit, "require_vae_embedding", False):
            y_channels = 4 + int(getattr(self.pipe.vae, "z_dim", 16))
            outputs["y"] = torch.zeros(
                (batch_size, y_channels, latent_length, latent_height, latent_width),
                dtype=self.pipe.torch_dtype,
                device=self.pipe.device,
            )
        if getattr(dit, "require_clip_embedding", False):
            outputs["clip_feature"] = torch.zeros(
                (batch_size, 257, 1280),
                dtype=self.pipe.torch_dtype,
                device=self.pipe.device,
            )
        return outputs

    def select_target_latents(self, latent_views: torch.Tensor) -> torch.Tensor:
        if latent_views.ndim == 5:
            return latent_views[self.cross_view_target_view].unsqueeze(0)
        if latent_views.ndim == 6:
            return latent_views[:, self.cross_view_target_view]
        raise ValueError("Expected latent views with shape (V,C,T,H,W) or (B,V,C,T,H,W).")

    def select_source_latents(self, latent_views: torch.Tensor) -> torch.Tensor:
        if latent_views.ndim == 5:
            return latent_views[list(self.cross_view_source_views)].unsqueeze(0)
        if latent_views.ndim == 6:
            return latent_views[:, list(self.cross_view_source_views)]
        raise ValueError("Expected latent views with shape (V,C,T,H,W) or (B,V,C,T,H,W).")

    def select_target_legacy_y(self, y: torch.Tensor | None, num_views: int) -> torch.Tensor | None:
        if y is None:
            return None
        if y.shape[-2] % int(num_views) != 0:
            return y
        view_height = y.shape[-2] // int(num_views)
        start = int(self.cross_view_target_view) * view_height
        end = start + view_height
        return y[..., start:end, :]

    def overwrite_target_only_history_latents(
        self,
        latents: torch.Tensor,
        target_history_latents: torch.Tensor,
        history_t: int,
    ) -> torch.Tensor:
        if target_history_latents.ndim != 5:
            raise ValueError("Expected target history latents with shape (B,C,T,H,W).")
        updated = latents.clone()
        updated[:, :, :history_t] = target_history_latents[:, :, :history_t].to(
            dtype=updated.dtype,
            device=updated.device,
        )
        return updated

    def overwrite_target_only_anchor_latents(
        self,
        latents: torch.Tensor,
        target_history_latents: torch.Tensor | None,
        target_tail_latents: torch.Tensor | None,
        history_t: int,
        tail_t: int,
    ) -> torch.Tensor:
        """Dual-end anchor overwrite for stage2 target-only path.

        - 当 target_tail_latents is None 或 tail_t<=0 时退化为单端 head-only 锚定，
          与 overwrite_target_only_history_latents 行为一致。
        - 尾锚 dropout 在调用方控制：调用方决定是否把 target_tail_latents 设为 None
          来跳过尾端覆盖。
        """
        if target_history_latents is None and target_tail_latents is None:
            return latents
        updated = latents.clone()
        if target_history_latents is not None and history_t > 0:
            if target_history_latents.ndim != 5:
                raise ValueError(
                    "Expected target_history_latents with shape (B,C,T,H,W)."
                )
            updated[:, :, :history_t] = target_history_latents[:, :, :history_t].to(
                dtype=updated.dtype, device=updated.device,
            )
        if target_tail_latents is not None and tail_t > 0:
            if target_tail_latents.ndim != 5:
                raise ValueError(
                    "Expected target_tail_latents with shape (B,C,T,H,W)."
                )
            updated[:, :, -tail_t:] = target_tail_latents[:, :, :tail_t].to(
                dtype=updated.dtype, device=updated.device,
            )
        return updated

    def attach_geometry_sidecar(self, data: dict) -> dict:
        if not self.geometry_sidecar_cache_path:
            return data
        cache_file = data.get("__cache_file__")
        if not cache_file:
            return data
        split_name = Path(cache_file).parent.name
        sidecar_path = Path(self.geometry_sidecar_cache_path) / split_name / Path(cache_file).name
        if not sidecar_path.is_file():
            if self.geometry_scene_token_source == "camera_aware_sidecar":
                raise FileNotFoundError(f"Geometry sidecar not found: {sidecar_path}")
            return data
        sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        if not isinstance(sidecar, dict):
            raise TypeError(f"Geometry sidecar must contain a dict: {sidecar_path}")
        merged = dict(data)
        merged.update(sidecar)
        return merged

    def _resolve_cached_pred_state_path(self, cache_file: str) -> Path:
        root = Path(self.cached_pred_state_root)
        cache_path = Path(cache_file)
        split_name = cache_path.parent.name
        stem = cache_path.stem
        candidates = [
            root / split_name / f"{stem}.npy",
            root / f"{stem}.npy",
            root / split_name / f"{stem}.pt",
            root / f"{stem}.pt",
            root / split_name / f"{stem}.pth",
            root / f"{stem}.pth",
        ]
        for path in candidates:
            if path.is_file():
                return path
        joined = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Predicted state file not found for cached sample {cache_file}. Tried: {joined}"
        )

    def _load_cached_pred_state_tensor(self, path: Path, num_frames: int) -> torch.Tensor:
        ext = path.suffix.lower()
        if ext == ".npy":
            arr = np.load(path)
        elif ext in (".pt", ".pth"):
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(loaded, dict):
                loaded = loaded.get("pred_state", loaded.get("state", loaded.get("action")))
            if loaded is None:
                raise KeyError(
                    f"Expected one of pred_state/state/action in predicted state file: {path}"
                )
            if isinstance(loaded, torch.Tensor):
                loaded = loaded.detach().cpu().numpy()
            arr = np.asarray(loaded)
        else:
            raise ValueError(f"Unsupported predicted state file extension: {path}")

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[-1] != 7:
            raise ValueError(
                f"Expected predicted state shape (T,7), got {arr.shape} from {path}"
            )
        if arr.shape[0] == 0:
            raise ValueError(f"Predicted state file has zero frames: {path}")
        if arr.shape[0] < int(num_frames):
            pad = np.repeat(arr[-1:, :], int(num_frames) - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        arr = np.clip(arr[: int(num_frames)], -1.0, 1.0)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor.to(device=self.pipe.device, dtype=self.pipe.torch_dtype)

    def attach_cached_predicted_state(self, data: dict) -> dict:
        if not self.cached_pred_state_root:
            return data
        cache_file = data.get("__cache_file__")
        if not cache_file:
            return data
        num_frames = self._scalar_int_from_data(data, "num_frames", 81)
        pred_state_path = self._resolve_cached_pred_state_path(str(cache_file))
        pred_state = self._load_cached_pred_state_tensor(pred_state_path, num_frames)
        merged = dict(data)
        merged["state"] = pred_state
        # Cached training prefers `action` over `state`; overwrite both so the
        # action-conditioning path uses the predicted state sequence.
        merged["action"] = pred_state
        return merged

    def attach_cached_legacy_image_branch(
        self,
        inputs_shared: dict,
        data,
        num_views: int,
    ) -> dict:
        updated = dict(inputs_shared)
        target_only = self.cross_view_stage == 2
        latent_views_gt = data.get("latent_views_gt")
        latent_view_height = None
        if target_only and isinstance(latent_views_gt, torch.Tensor):
            latent_view_height = int(latent_views_gt.shape[-2])
        if self.cross_view_stage == 2 and self.cross_view_disable_legacy_image_branch:
            updated.update(
                self.build_zero_legacy_image_branch_tensors(
                    num_views=num_views,
                    num_frames=int(updated["num_frames"]),
                    height=int(updated["height"]),
                    width=int(updated["width"]),
                    latent_view_height=latent_view_height,
                    target_only=True,
                )
            )
            return updated

        if "y" in data:
            y = data["y"]
            if target_only:
                y = self.select_target_legacy_y(y, num_views)
            updated["y"] = y
        if "clip_feature" in data:
            updated["clip_feature"] = data["clip_feature"]

        missing = []
        dit = getattr(self.pipe, "dit", None)
        if dit is not None and getattr(dit, "has_image_input", False):
            if getattr(dit, "require_vae_embedding", False) and "y" not in updated:
                missing.append("y")
            if getattr(dit, "require_clip_embedding", False) and "clip_feature" not in updated:
                missing.append("clip_feature")
        if missing:
            raise KeyError(
                "Cached cross-view sample is missing required legacy image branch tensors: "
                f"{', '.join(missing)}."
            )
        return updated

    def zero_legacy_image_branch(self, inputs_shared: dict) -> dict:
        zeroed = dict(inputs_shared)
        if "y" in zeroed:
            zeroed["y"] = torch.zeros_like(zeroed["y"])
        if "clip_feature" in zeroed:
            zeroed["clip_feature"] = torch.zeros_like(zeroed["clip_feature"])
        return zeroed

    def validate_cross_view_cached_batch(self, data) -> None:
        required_keys = ("latent_views_gt", "target_history_latents", "height", "width", "num_frames")
        missing = [key for key in required_keys if key not in data]
        if missing:
            raise KeyError(
                "Cached cross-view batch is missing required keys: "
                f"{', '.join(missing)}."
            )
        latent_views_gt = data["latent_views_gt"]
        if not isinstance(latent_views_gt, torch.Tensor) or latent_views_gt.ndim != 5:
            raise ValueError(
                "`latent_views_gt` must be a tensor with shape (V,C,T,H,W) in cached batches."
            )
        target_history_latents = data["target_history_latents"]
        if (
            not isinstance(target_history_latents, torch.Tensor)
            or target_history_latents.ndim != 5
        ):
            raise ValueError(
                "`target_history_latents` must be a tensor with shape (B,C,T,H,W) in cached batches."
            )
        num_views = int(latent_views_gt.shape[0])
        if self.cross_view_target_view >= num_views:
            raise ValueError(
                f"Target view index {self.cross_view_target_view} is out of range for cached num_views={num_views}."
            )
        for index in self.cross_view_source_views:
            if index >= num_views:
                raise ValueError(
                    f"Source view index {index} is out of range for cached num_views={num_views}."
                )

    def build_stage1_cached_history_latents(
        self,
        data,
        latent_views_gt: torch.Tensor,
        history_t: int,
    ) -> torch.Tensor:
        cond_history_latents = data.get("cond_history_latents")
        if cond_history_latents is not None:
            if not isinstance(cond_history_latents, torch.Tensor) or cond_history_latents.ndim != 5:
                raise ValueError(
                    "`cond_history_latents` must be a tensor with shape (B,C,T,H,W) when provided."
                )
            cond_history_latents = cond_history_latents[:, :, :history_t].to(
                dtype=self.pipe.torch_dtype,
                device=self.pipe.device,
            )
            if self.wrist_first_frame_index is None:
                return cond_history_latents
            target_history_latents = self._maybe_overwrite_target_history_with_wrist_first_frame(
                data["target_history_latents"].to(
                    dtype=self.pipe.torch_dtype,
                    device=self.pipe.device,
                ),
                data,
            )
            return self.overwrite_target_history_latents(
                cond_history_latents,
                target_history_latents,
                num_views=int(latent_views_gt.shape[0]),
                history_t=history_t,
            )[:, :, :history_t]
        input_latents_cond = self.merge_view_latents(latent_views_gt)
        target_history_latents = data["target_history_latents"].to(
            dtype=self.pipe.torch_dtype,
            device=self.pipe.device,
        )
        target_history_latents = self._maybe_overwrite_target_history_with_wrist_first_frame(
            target_history_latents,
            data,
        )
        input_latents_cond = self.overwrite_target_history_latents(
            input_latents_cond,
            target_history_latents,
            num_views=int(latent_views_gt.shape[0]),
            history_t=history_t,
        )
        return input_latents_cond[:, :, :history_t]

    def maybe_drop_old_branch(self, inputs_shared: dict, allow_dropout: bool = True) -> dict:
        if self.cross_view_stage != 2:
            return inputs_shared
        if self.cross_view_disable_legacy_image_branch:
            return self.zero_legacy_image_branch(inputs_shared)
        if (
            (not allow_dropout)
            or (not self.training)
            or self.cross_view_old_branch_dropout <= 0
        ):
            return inputs_shared
        if torch.rand((), device=self.pipe.device) >= self.get_old_branch_dropout_probability():
            return inputs_shared
        return self.zero_legacy_image_branch(inputs_shared)

    def forward_cross_view(self, data):
        data = self.transfer_data_to_device(data, self.pipe.device, self.pipe.torch_dtype)
        video_gt = data["video"]
        self.validate_cross_view_video(video_gt)
        cond_video = self.build_cross_view_condition_video(video_gt, meta=data)
        inputs = self.build_cross_view_inputs(data, cond_video)
        skip_legacy_branch_compute = (
            self.cross_view_stage == 2 and self.cross_view_disable_legacy_image_branch
        )
        for unit in self.iter_cross_view_units(
            include_legacy_image_branch=not skip_legacy_branch_compute
        ):
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        inputs_shared, inputs_posi, _ = inputs
        if skip_legacy_branch_compute:
            inputs_shared.update(
                self.build_zero_legacy_image_branch_tensors(
                    num_views=int(video_gt.shape[0]),
                    num_frames=int(inputs_shared["num_frames"]),
                    height=int(inputs_shared["height"]),
                    width=int(inputs_shared["width"]),
                    target_only=(self.cross_view_stage == 2),
                )
            )
        condition_sequence = self.get_cross_view_condition_sequence(
            data,
            num_frames=int(video_gt.shape[2]),
        )

        num_views = int(video_gt.shape[0])
        history_t = ((self.num_history_frames - 1) // 4) + 1
        # Plan A: dual-end anchor in raw forward is now active too -- the
        # cond_video already has wrist[..., -1] = indexed wrist end-frame
        # anchor (set in build_cross_view_condition_video) and the y channel encodes
        # both anchors via the integrated 81-frame VAE pass. We no longer
        # build target_history_latents / target_tail_latents nor overwrite
        # any latent slot. The full-sequence loss below also makes tail_t
        # slicing unnecessary (kept history_t for stage1 head-only path).
        if self.cross_view_stage == 2:
            latent_views_gt = self.encode_video_latents_by_view(video_gt)
            input_latents_gt = self.select_target_latents(latent_views_gt)
            source_x0_latents = self.select_source_latents(latent_views_gt)
            if "y" in inputs_shared:
                inputs_shared["y"] = self.select_target_legacy_y(inputs_shared["y"], num_views)
            inputs_shared.update(
                self.build_cross_view_source_condition(
                    condition_sequence=condition_sequence,
                    source_latents=source_x0_latents,
                )
            )
            inputs_shared.update(self.build_target_camera_condition(data, latent_length=input_latents_gt.shape[2]))
            inputs_shared.update(self._build_geometry_aware_inputs_from_video(video_gt, data=data))
        else:
            input_latents_gt = self.encode_joint_video_latents(video_gt)
            input_latents_cond = self.encode_joint_video_latents(cond_video)
            source_x0_latents = None
        inputs_shared = self.maybe_drop_old_branch(inputs_shared, allow_dropout=True)

        timestep = self.sample_training_timestep()
        gaussian_noise = torch.randn_like(input_latents_gt)
        if self.cross_view_stage == 2:
            noise = self.build_stage2_3d_noise_prior(
                gaussian_noise,
                inputs_shared.get("scene_tokens"),
                condition_sequence,
                inputs_shared.get("y"),
                source_view_ids=self.cross_view_source_views,
                target_view_id=self.cross_view_target_view,
            )
        else:
            noise = gaussian_noise
        latents = self.pipe.scheduler.add_noise(input_latents_gt, noise, timestep)
        # Stage1 keeps head-only latent overwrite; stage2 relies entirely on
        # the y-channel signal.
        if self.cross_view_stage != 2:
            latents[:, :, :history_t] = input_latents_cond[:, :, :history_t]
        target_noise = self.detach_3d_noise_target_if_needed(noise)
        training_target = self.pipe.scheduler.training_target(
            input_latents_gt,
            target_noise,
            timestep,
        )

        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        model_output = self.pipe.model_fn(
            **models,
            **inputs_shared,
            **inputs_posi,
            latents=latents,
            timestep=timestep,
            return_hidden_by_time=(
                self.cross_view_stage == 2 and (
                    self.cross_view_state_loss_weight > 0
                    or self.alignment_loss_weight > 0
                )
            ),
        )
        if isinstance(model_output, tuple):
            noise_pred, hidden_by_time = model_output
        else:
            noise_pred, hidden_by_time = model_output, None

        # Plan A: stage2 supervises the entire denoising trajectory.
        # head/tail anchor positions are noised like any other slot and the
        # model learns to denoise them with the y-channel signal as guidance.
        # Stage1 keeps its head-anchor-mask behavior (slice [history_t:]).
        if self.cross_view_stage == 2:
            noise_pred_future = noise_pred
            training_target_future = training_target
            main_loss = F.mse_loss(noise_pred_future.float(), training_target_future.float())
        else:
            noise_pred_future = noise_pred[:, :, history_t:]
            training_target_future = training_target[:, :, history_t:]
            main_loss = self.cross_view_weighted_loss(
                noise_pred_future,
                training_target_future,
                timestep,
                num_views=num_views,
                target_only=False,
            )
        if self.cross_view_stage != 2:
            return main_loss

        total_loss = main_loss
        loss_weight = self.pipe.scheduler.training_weight(timestep)
        aux_loss_scale = self.get_cross_view_aux_loss_scale()

        if self.cross_view_temp_loss_weight > 0:
            temporal_loss = self.compute_cross_view_temporal_loss(
                latents,
                noise_pred_future,
                input_latents_gt,
                timestep,
                num_views=num_views,
                history_t=0,
                target_only=True,
                tail_t=0,
            )
            if temporal_loss is not None:
                total_loss = total_loss + (
                    loss_weight
                    * aux_loss_scale
                    * self.cross_view_temp_loss_weight
                    * temporal_loss
                )

        if self.cross_view_state_loss_weight > 0:
            state_loss = self.compute_cross_view_state_loss(
                hidden_by_time,
                condition_sequence,
                num_views=num_views,
                history_t=0,
                target_only=True,
            )
            if state_loss is not None:
                total_loss = total_loss + (
                    loss_weight
                    * aux_loss_scale
                    * self.cross_view_state_loss_weight
                    * state_loss
                )

        if self.alignment_loss_weight > 0:
            scene_tokens = inputs_shared.get("scene_tokens")
            align_loss = self.compute_geometry_alignment_loss(
                hidden_by_time, scene_tokens, num_views, history_t=0, target_only=True
            )
            if align_loss is not None:
                alignment_loss_scale = self.get_alignment_loss_scale()
                total_loss = total_loss + (
                    loss_weight
                    * alignment_loss_scale
                    * self.alignment_loss_weight
                    * align_loss
                )

        return total_loss

    def forward_cross_view_cached(self, data):
        data = self.transfer_data_to_device(data, self.pipe.device, self.pipe.torch_dtype)
        data = self.attach_geometry_sidecar(data)
        data = self.attach_cached_predicted_state(data)
        self.validate_cross_view_cached_batch(data)
        latent_views_gt = data["latent_views_gt"]
        num_views = int(latent_views_gt.shape[0])
        inputs_shared, inputs_posi, _ = self.build_cross_view_cached_inputs(data)
        inputs_shared = self.attach_cached_legacy_image_branch(
            inputs_shared,
            data,
            num_views=num_views,
        )
        condition_sequence = self.get_cross_view_condition_sequence(
            data,
            num_frames=int(inputs_shared["num_frames"]),
        )

        history_t = ((self.num_history_frames - 1) // 4) + 1
        # Plan A (WAN-Fun-InP aligned): dual-end anchoring no longer overwrites
        # latent slots. Anchor pixels (head + tail) are placed in cond_video at
        # build_cross_view_condition_video time and integrally encoded by
        # WanVideoUnit_ImageEmbedderVAE into the y channel together with mask
        # bits. The DiT's 36-channel input concatenates [noisy_latent, y],
        # giving the model a slot-position-correct prior on which timesteps
        # are known anchors. Loss now supervises the entire sequence (no
        # history_t / tail_t slicing) — anchor positions get noised like
        # everything else, and the model learns to denoise them too, exactly
        # matching the original WAN-Fun-InP training objective. tail_t is
        # therefore not needed in this forward path; we keep history_t for
        # the stage1 head-only branch.
        if self.cross_view_stage == 2:
            input_latents_gt = self.select_target_latents(latent_views_gt)
            source_x0_latents = self.select_source_latents(latent_views_gt)
            inputs_shared.update(
                self.build_cross_view_source_condition(
                    condition_sequence=condition_sequence,
                    source_latents=source_x0_latents,
                )
            )
            inputs_shared.update(self._build_geometry_aware_inputs(data))
            inputs_shared.update(self.build_target_camera_condition(data, latent_length=input_latents_gt.shape[2]))
        else:
            input_latents_gt = self.merge_view_latents(latent_views_gt)
            source_x0_latents = None
            cond_history_latents = self.build_stage1_cached_history_latents(
                data,
                latent_views_gt=latent_views_gt,
                history_t=history_t,
            )
        inputs_shared = self.maybe_drop_old_branch(inputs_shared, allow_dropout=True)

        timestep = self.sample_training_timestep()
        gaussian_noise = torch.randn_like(input_latents_gt)
        if self.cross_view_stage == 2:
            noise = self.build_stage2_3d_noise_prior(
                gaussian_noise,
                inputs_shared.get("scene_tokens"),
                condition_sequence,
                inputs_shared.get("y"),
                source_view_ids=self.cross_view_source_views,
                target_view_id=self.cross_view_target_view,
            )
        else:
            noise = gaussian_noise
        latents = self.pipe.scheduler.add_noise(input_latents_gt, noise, timestep)
        # Stage1 still uses head-only latent overwrite (its joint-views design
        # depends on it). Stage2 now relies entirely on the y-channel for
        # anchor signal -- no slot overwrite.
        if self.cross_view_stage != 2:
            latents[:, :, :history_t] = cond_history_latents[:, :, :history_t]
        target_noise = self.detach_3d_noise_target_if_needed(noise)
        training_target = self.pipe.scheduler.training_target(
            input_latents_gt,
            target_noise,
            timestep,
        )

        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        model_output = self.pipe.model_fn(
            **models,
            **inputs_shared,
            **inputs_posi,
            latents=latents,
            timestep=timestep,
            return_hidden_by_time=(
                self.cross_view_stage == 2 and (
                    self.cross_view_state_loss_weight > 0
                    or self.alignment_loss_weight > 0
                )
            ),
        )
        if isinstance(model_output, tuple):
            noise_pred, hidden_by_time = model_output
        else:
            noise_pred, hidden_by_time = model_output, None

        # Plan A: stage2 supervises the entire denoising trajectory.
        # head/tail anchor positions are noised like any other slot and the
        # model learns to denoise them with the y-channel signal as guidance.
        # Stage1 keeps its head-anchor-mask behavior (slice [history_t:]).
        if self.cross_view_stage == 2:
            noise_pred_future = noise_pred
            training_target_future = training_target
            main_loss = F.mse_loss(noise_pred_future.float(), training_target_future.float())
        else:
            noise_pred_future = noise_pred[:, :, history_t:]
            training_target_future = training_target[:, :, history_t:]
            main_loss = self.cross_view_weighted_loss(
                noise_pred_future,
                training_target_future,
                timestep,
                num_views=num_views,
                target_only=False,
            )
        if self.cross_view_stage != 2:
            return main_loss

        total_loss = main_loss
        loss_weight = self.pipe.scheduler.training_weight(timestep)
        aux_loss_scale = self.get_cross_view_aux_loss_scale()

        if self.cross_view_temp_loss_weight > 0:
            temporal_loss = self.compute_cross_view_temporal_loss(
                latents,
                noise_pred_future,
                input_latents_gt,
                timestep,
                num_views=num_views,
                history_t=0,
                target_only=True,
                tail_t=0,
            )
            if temporal_loss is not None:
                total_loss = total_loss + (
                    loss_weight
                    * aux_loss_scale
                    * self.cross_view_temp_loss_weight
                    * temporal_loss
                )

        if self.cross_view_state_loss_weight > 0:
            state_loss = self.compute_cross_view_state_loss(
                hidden_by_time,
                condition_sequence,
                num_views=num_views,
                history_t=0,
                target_only=True,
            )
            if state_loss is not None:
                total_loss = total_loss + (
                    loss_weight
                    * aux_loss_scale
                    * self.cross_view_state_loss_weight
                    * state_loss
                )

        if self.alignment_loss_weight > 0:
            scene_tokens = inputs_shared.get("scene_tokens")
            align_loss = self.compute_geometry_alignment_loss(
                hidden_by_time, scene_tokens, num_views, history_t=0, target_only=True
            )
            if align_loss is not None:
                alignment_loss_scale = self.get_alignment_loss_scale()
                total_loss = total_loss + (
                    loss_weight
                    * alignment_loss_scale
                    * self.alignment_loss_weight
                    * align_loss
                )

        return total_loss

    def forward(self, data, inputs=None):
        if self.cross_view_stage > 0:
            if isinstance(inputs, dict) and "latent_views_gt" in inputs:
                return self.forward_cross_view_cached(inputs)
            return self.forward_cross_view(data)
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser = add_action_config(parser)
    parser = add_state_config(parser)
    parser.add_argument(
        "--state_loader",
        type=str,
        choices=["droid", "predicted"],
        default="droid",
        help="droid reads parquet state; predicted reads pre-normalized .npy/.pt/.pth state.",
    )
    parser.add_argument(
        "--cached_pred_state_root",
        type=str,
        default=None,
        help=(
            "Optional predicted-state root for cached training. The loader tries "
            "<root>/<split>/<cache_stem>.npy and <root>/<cache_stem>.npy, then .pt/.pth."
        ),
    )
    parser = add_cross_view_config(parser)

    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    set_global_seed(args.seed)
    requested_data_file_keys = [
        key.strip() for key in args.data_file_keys.split(",") if key.strip()
    ]
    runtime = prepare_wan_runtime(
        args.model_paths,
        args.load_modules,
        requested_data_file_keys,
    )
    modules = runtime["modules"]
    data_file_keys = runtime["data_file_keys"]

    # Cross-view DROID training uses `state` as the conditioning sequence while
    # still reusing the existing action-conditioning path in the model. In that
    # case we must not force-load a missing `action` field from the dataset.
    if args.state_type is not None and "state" in requested_data_file_keys:
        data_file_keys = [key for key in data_file_keys if key != "action"]
        if "state" not in data_file_keys:
            data_file_keys.append("state")

    if (
        WanTrainingModule.resolve_cross_view_stage(args.task) == 2
        and bool(int(getattr(args, "geometry_use_camera_tokens", 0)))
        and getattr(args, "cached_dataset_path", None) in (None, "")
    ):
        for key in ("source_camera_tokens", "target_camera_tokens"):
            if key not in data_file_keys:
                data_file_keys.append(key)

    def module_base(name: str) -> str:
        return str(name).partition(":")[0].strip().lower()

    module_bases = {module_base(item) for item in modules}
    action_enabled = "action" in module_bases
    cached_dataset_path = getattr(args, "cached_dataset_path", None)
    use_cached_dataset = cached_dataset_path not in (None, "")
    if use_cached_dataset and args.dataset_metadata_path is not None:
        raise ValueError(
            "`--cached_dataset_path` and `--dataset_metadata_path` are mutually exclusive."
        )
    if use_cached_dataset and WanTrainingModule.resolve_cross_view_stage(args.task) == 0:
        raise ValueError(
            "`--cached_dataset_path` is currently supported only for cross-view training tasks."
        )

    trainable_models = args.trainable_models
    models = [m.strip() for m in str(trainable_models).split(",") if m.strip()] if trainable_models else []
    models = [item for item in models if item != "geometry_gates"]
    if len(models) == 0:
        models = ["dit"]
    if action_enabled and all(m == "dit" for m in models):
        models.append("action_encoder")
    if WanTrainingModule.resolve_cross_view_stage(args.task) == 2:
        source_memory_enabled = args.cross_view_source_injection_mode != "none"
        if args.cross_view_source_injection_mode == "temporal_local":
            models = [item for item in models if item != "source_temporal_gate"]
            if "dit" not in models:
                models.append("dit")
        if source_memory_enabled:
            if "source_video_projector" not in models:
                models.append("source_video_projector")
            if (
                args.cross_view_source_injection_mode != "temporal_local"
                and args.cross_view_source_gate_mode != "none"
                and "source_temporal_gate" not in models
            ):
                models.append("source_temporal_gate")
        if (
            args.cross_view_state_loss_weight > 0
            and "target_state_head" not in models
        ):
            models.append("target_state_head")
        if (
            getattr(args, "geometry_target_camera_mode", "none") == "add_time_mlp"
            and "target_camera_encoder" not in models
        ):
            models.append("target_camera_encoder")
        if getattr(args, "scene_token_checkpoint", None) is not None:
            if "scene_token_adapter" not in models:
                models.append("scene_token_adapter")
            # if "geometry_gates" not in models:
            #     models.append("geometry_gates")
        if (
            getattr(args, "cross_view_3d_noise_prior_mode", "none") != "none"
            and float(getattr(args, "cross_view_3d_noise_prior_weight", 0.1)) > 0
        ):
            if "scene_3d_noise_prior_adapter" not in models:
                models.append("scene_3d_noise_prior_adapter")
            if (
                getattr(args, "cross_view_3d_noise_prior_mode", "none") == "scene_action_grid"
                and "action_noise_modulator" not in models
            ):
                models.append("action_noise_modulator")
    trainable_models = ",".join(models)
    args.trainable_models = trainable_models
    args.data_file_keys = ",".join(data_file_keys)

    model_paths_json = json.dumps(runtime["model_paths"])
    tokenizer_path = runtime["tokenizer_path"]
    log_with = []
    if getattr(args, "use_wandb", False):
        log_with.append("wandb")
    if getattr(args, "use_swanlab", False):
        log_with.append("swanlab")
    log_with = log_with if len(log_with) > 0 else None
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )

    special_operator_map = {}
    cache_config = None
    if use_cached_dataset:
        cache_config = load_cross_view_cache_config(cached_dataset_path)
        if args.height is None:
            args.height = int(cache_config["height"])
        if args.width is None:
            args.width = int(cache_config["width"])
        if args.num_frames is None:
            args.num_frames = int(cache_config["num_frames"])
        if args.num_history_frames is None:
            args.num_history_frames = int(cache_config["num_history_frames"])
        if args.state_type is None:
            args.state_type = cache_config.get("state_type")
        validate_cross_view_cache_config(cache_config, args, modules)

    if "text" in module_bases and "prompt_emb" in data_file_keys and (not use_cached_dataset):
        special_operator_map["prompt_emb"] = ResolvePromptEmbPath(
            base_path=args.dataset_base_path
        )
    state_loader = getattr(args, "state_loader", "droid")
    use_predicted_state = state_loader == "predicted"

    stat_path = args.action_stat_path
    if stat_path is None and "state" in data_file_keys and not use_predicted_state:
        stat_path = args.state_stat_path
    if "state" in data_file_keys and stat_path is not None:
        args.state_stat_path = stat_path
    if "action" in data_file_keys and stat_path is not None:
        args.action_stat_path = stat_path
    if "state" in data_file_keys and args.state_type is None:
        raise ValueError("`--state_type` is required when `state` is included in `--data_file_keys`.")
    if (
        not use_cached_dataset
        and "state" in data_file_keys
        and not use_predicted_state
        and stat_path is None
    ):
        raise ValueError("`--state_stat_path` is required when loading normalized `state` inputs.")
    if use_cached_dataset:
        train_cache_dir = os.path.join(cached_dataset_path, "train")
        if not os.path.isdir(train_cache_dir):
            raise FileNotFoundError(f"Cached train split not found: {train_cache_dir}")
        dataset = UnifiedDataset(
            base_path=train_cache_dir,
            metadata_path=None,
            repeat=args.dataset_repeat,
        )
    else:
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
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
            stat_path=stat_path,
            action_type=args.action_type,
        )
        if "action" in data_file_keys:
            dataset.special_operator_map["action"] = LoadCobotAction(
                base_path=args.dataset_base_path,
                action_type=args.action_type,
                stat=dataset.stat,
                num_frames=args.num_frames,
            )
        if "state" in data_file_keys:
            if use_predicted_state:
                dataset.special_operator_map["state"] = LoadPredictedDroidState(
                    base_path=args.dataset_base_path,
                    num_frames=args.num_frames,
                    state_dim=7,
                )
            else:
                dataset.special_operator_map["state"] = LoadDroidState(
                    base_path=args.dataset_base_path,
                    state_type=args.state_type,
                    stat=dataset.stat,
                    num_frames=args.num_frames,
                )
        if "source_camera_tokens" in data_file_keys:
            dataset.special_operator_map["source_camera_tokens"] = LoadDroidCameraTokens(
                base_path=args.dataset_base_path,
                role="source",
                view_indices=WanTrainingModule.parse_view_indices(args.cross_view_source_views),
                num_frames=args.num_frames,
            )
        if "target_camera_tokens" in data_file_keys:
            dataset.special_operator_map["target_camera_tokens"] = LoadDroidCameraTokens(
                base_path=args.dataset_base_path,
                role="target",
                view_indices=(int(args.cross_view_target_view),),
                num_frames=args.num_frames,
            )
    model = WanTrainingModule(
        model_paths=model_paths_json,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=tokenizer_path,
        trainable_models=trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        modules=modules,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        ckpt_path=args.ckpt_path,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        num_history_frames=args.num_history_frames,
        cross_view_source_views=args.cross_view_source_views,
        cross_view_target_view=args.cross_view_target_view,
        cross_view_placeholder_mode=args.cross_view_placeholder_mode,
        cross_view_source_loss_weight=args.cross_view_source_loss_weight,
        cross_view_old_branch_dropout=args.cross_view_old_branch_dropout,
        cross_view_projector_hidden_dim=args.cross_view_projector_hidden_dim,
        cross_view_source_injection_mode=args.cross_view_source_injection_mode,
        cross_view_source_branch_mode=args.cross_view_source_branch_mode,
        cross_view_source_window_radius=args.cross_view_source_window_radius,
        cross_view_source_gate_mode=args.cross_view_source_gate_mode,
        cross_view_temp_loss_weight=args.cross_view_temp_loss_weight,
        cross_view_state_loss_weight=args.cross_view_state_loss_weight,
        cross_view_global_source_tokens=args.cross_view_global_source_tokens,
        cross_view_aux_loss_warmup_ratio=args.cross_view_aux_loss_warmup_ratio,
        cross_view_old_branch_dropout_schedule=args.cross_view_old_branch_dropout_schedule,
        cross_view_legacy_branch_schedule=args.cross_view_legacy_branch_schedule,
        cross_view_disable_legacy_image_branch=args.cross_view_disable_legacy_image_branch,
        cross_view_use_tail_anchor=getattr(args, "cross_view_use_tail_anchor", 0),
        num_tail_frames=getattr(args, "num_tail_frames", 1),
        cross_view_tail_anchor_dropout=getattr(args, "cross_view_tail_anchor_dropout", 0.0),
        cross_view_use_keyframe_anchor=getattr(args, "cross_view_use_keyframe_anchor", 0),
        num_keyframe_anchors=getattr(args, "num_keyframe_anchors", 3),
        keyframe_anchor_dropout=getattr(args, "keyframe_anchor_dropout", 0.0),
        cross_view_3d_noise_prior_mode=getattr(args, "cross_view_3d_noise_prior_mode", "none"),
        cross_view_3d_noise_prior_weight=getattr(args, "cross_view_3d_noise_prior_weight", 0.1),
        cross_view_3d_noise_anchor_attenuation=getattr(
            args,
            "cross_view_3d_noise_anchor_attenuation",
            1.0,
        ),
        state_type=args.state_type,
        scene_token_checkpoint=getattr(args, "scene_token_checkpoint", None),
        scene_token_pool_size=getattr(args, "scene_token_pool_size", 512),
        geometry_gate_mode=getattr(args, "geometry_gate_mode", "learned"),
        geometry_sidecar_cache_path=getattr(args, "geometry_sidecar_cache_path", None),
        geometry_use_camera_tokens=getattr(args, "geometry_use_camera_tokens", 0),
        geometry_target_camera_mode=getattr(args, "geometry_target_camera_mode", "none"),
        geometry_scene_token_source=getattr(args, "geometry_scene_token_source", "cached_zero_cam"),
        cached_pred_state_root=getattr(args, "cached_pred_state_root", None),
        alignment_loss_weight=getattr(args, "alignment_loss_weight", 0.1),
        alignment_loss_warmup_ratio=getattr(args, "alignment_loss_warmup_ratio", 0.1),
    )
    wrist_first_frame_index_path = getattr(args, "wrist_first_frame_index", None)
    if wrist_first_frame_index_path and os.path.exists(wrist_first_frame_index_path):
        import json as _json
        with open(wrist_first_frame_index_path) as _f:
            model.wrist_first_frame_index = _json.load(_f)
        print(f"[wrist_first_frame] loaded {len(model.wrist_first_frame_index)} entries from {wrist_first_frame_index_path}")
    if (
        bool(int(getattr(args, "cross_view_use_keyframe_anchor", 0)))
        and not use_cached_dataset
    ):
        keyframe_manifest = getattr(args, "keyframe_anchor_manifest_train", None)
        keyframe_root = getattr(args, "keyframe_anchor_image_root_train", None)
        if not keyframe_manifest or not keyframe_root:
            raise ValueError(
                "--keyframe_anchor_manifest_train and --keyframe_anchor_image_root_train "
                "are required for raw keyframe-anchor training."
            )
        model.keyframe_anchor_index = load_keyframe_anchor_index(
            keyframe_manifest,
            keyframe_root,
            num_keyframes=int(getattr(args, "num_keyframe_anchors", 3)),
            num_frames=int(args.num_frames),
        )
        print(
            f"[keyframe_anchor] loaded {len(model.keyframe_anchor_index['by_key'])} train clips"
        )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        config=build_grouped_config(parser, args),
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
        "cross_view_stage1": launch_training_task,
        "cross_view_stage1:train": launch_training_task,
        "cross_view_stage2": launch_training_task,
        "cross_view_stage2:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
