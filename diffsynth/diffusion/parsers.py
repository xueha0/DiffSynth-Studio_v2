import argparse
import os
from typing import Literal, Optional

_DEFAULT_GROUP_TITLES = {"positional arguments", "optional arguments", "options"}

_WAN_DEFAULT_MODULES = ("dit", "text", "vae", "image", "action")
_WAN_MODULE_FILES = {
    "dit": ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.pth"),
    "text": ("models_t5_umt5-xxl-enc-bf16.pth", "models_t5_umt5-xxl-enc-bf16.safetensors"),
    "vae": ("Wan2.1_VAE.pth", "Wan2.1_VAE.safetensors"),
    "image": (
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors",
    ),
}
_WAN_TOKENIZER_SUBDIR = os.path.join("google", "umt5-xxl")


def _wan_module_base(name: str) -> str:
    return str(name).partition(":")[0].strip().lower()


def resolve_wan_text_mode(modules: Optional[list[str]]) -> Literal["none", "t5", "emb"]:
    if modules is None:
        return "t5"
    for spec in reversed(modules):
        if _wan_module_base(spec) != "text":
            continue
        spec = str(spec).strip().lower()
        _, sep, variant = spec.partition(":")
        if not sep:
            return "t5"
        mode = variant.strip()
        if mode in ("none", "off", "disable", "disabled"):
            return "none"
        if mode == "emb":
            return "emb"
        return "t5"
    return "none"


def resolve_wan_action_injection_mode(modules: Optional[list[str]]) -> Literal["none", "noise", "adaln"]:
    if modules is None:
        return "noise"
    for spec in reversed(modules):
        if _wan_module_base(spec) != "action":
            continue
        spec = str(spec).strip().lower()
        _, sep, variant = spec.partition(":")
        mode = variant.strip() if sep else "noise"
        if mode in ("none", "off", "disable", "disabled"):
            return "none"
        return "adaln" if mode == "adaln" else "noise"
    return "none"


def normalize_wan_modules(load_modules):
    # Normalize a module list (lowercased, deduplicated) while preserving
    # the last specified action/text variant.
    if not load_modules:
        modules = list(_WAN_DEFAULT_MODULES)
    elif isinstance(load_modules, str):
        modules = [item.strip() for item in load_modules.split(",") if item.strip()]
    else:
        modules = [str(item).strip() for item in load_modules if str(item).strip()]

    order: list[str] = []
    spec_by_base: dict[str, str] = {}

    def maybe_add_base(base: str):
        if base not in spec_by_base:
            order.append(base)

    def drop_base(base: str):
        if base in spec_by_base:
            del spec_by_base[base]
            order[:] = [item for item in order if item != base]

    for module in modules:
        base, sep, variant = module.partition(":")
        key = base.lower().strip()

        if key == "action" and sep:
            mode = variant.lower().strip()
            if mode in ("none", "off"):
                drop_base("action")
                continue
            maybe_add_base("action")
            spec_by_base["action"] = f"action:{mode}"
            continue

        if key == "text" and sep:
            mode = variant.lower().strip()
            if mode in ("none", "off"):
                drop_base("text")
                continue
            maybe_add_base("text")
            spec_by_base["text"] = "text" if mode in ("t5", "default") else f"text:{mode}"
            continue

        maybe_add_base(key)
        spec_by_base[key] = key

    return [spec_by_base[base] for base in order if base in spec_by_base]


def _pick_wan_candidate(model_root, candidates):
    for name in candidates:
        path = os.path.join(model_root, name)
        if os.path.isfile(path):
            return path
    return os.path.join(model_root, candidates[0])


def resolve_wan_model_paths(model_root, modules):
    if not model_root:
        raise ValueError("`--model_paths` is required.")
    paths = []
    for module in modules:
        candidates = _WAN_MODULE_FILES.get(module)
        if not candidates:
            continue
        paths.append(_pick_wan_candidate(model_root, candidates))
    return paths


def resolve_wan_tokenizer_path(model_root):
    if not model_root:
        raise ValueError("`--model_paths` is required.")
    return os.path.join(model_root, _WAN_TOKENIZER_SUBDIR)


def prepare_wan_runtime(model_root, load_modules, data_file_keys):
    modules = normalize_wan_modules(load_modules)

    def _ensure_key(keys: list[str], name: str) -> list[str]:
        if name not in keys:
            keys.append(name)
        return keys

    filtered_keys = [key for key in data_file_keys if key]
    text_mode = resolve_wan_text_mode(list(modules))
    action_injection_mode = resolve_wan_action_injection_mode(list(modules))
    action_enabled = action_injection_mode != "none"

    if text_mode == "none":
        filtered_keys = [
            key
            for key in filtered_keys
            if key not in ("prompt_emb", "negative_prompt_emb")
        ]
    elif text_mode == "emb":
        filtered_keys = _ensure_key(filtered_keys, "prompt_emb")

    if action_enabled:
        filtered_keys = _ensure_key(filtered_keys, "action")
    else:
        filtered_keys = [key for key in filtered_keys if key != "action"]

    weight_modules: list[str] = []
    for module in modules:
        base = _wan_module_base(module)
        if base == "text" and text_mode != "t5":
            continue
        weight_modules.append(base)

    model_paths = resolve_wan_model_paths(model_root, weight_modules)
    tokenizer_path = resolve_wan_tokenizer_path(model_root) if text_mode == "t5" else None
    return {
        "modules": tuple(modules),
        "data_file_keys": filtered_keys,
        "model_paths": model_paths,
        "tokenizer_path": tokenizer_path,
    }

def _get_group(parser: argparse.ArgumentParser, title: str):
    for group in parser._action_groups:
        if group.title == title:
            return group
    return parser.add_argument_group(title)

def build_grouped_config(parser: argparse.ArgumentParser, args):
    if args is None:
        return None
    args_dict = args if isinstance(args, dict) else vars(args)
    grouped = {}
    used = set()
    for group in parser._action_groups:
        if group.title in _DEFAULT_GROUP_TITLES:
            continue
        values = {}
        for action in group._group_actions:
            dest = getattr(action, "dest", None)
            if dest in args_dict:
                values[dest] = args_dict[dest]
                used.add(dest)
        if values:
            grouped[group.title] = values
    other = {k: args_dict[k] for k in args_dict if k not in used}
    if other:
        grouped["other"] = other
    return grouped

def add_dataset_base_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "dataset")
    group.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    group.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    group.add_argument("--cached_dataset_path", type=str, default=None, help="Optional root directory of precomputed cache samples. When set, training loads `.pth` samples from this cache instead of reading raw dataset metadata.")
    group.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    group.add_argument("--dataset_num_workers", type=int, default=8, help="Number of workers for data loading.")
    group.add_argument("--data_file_keys", type=str, default="video", help="Data file keys in the metadata. Comma-separated.")
    return parser

def add_image_size_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "image")
    group.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    group.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    group.add_argument("--max_pixels", type=int, default=4096*4096, help="Maximum number of pixels per frame, used for dynamic resolution.")
    return parser

def add_video_size_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "video")
    group.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    group.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    group.add_argument("--max_pixels", type=int, default=4096*4096, help="Maximum number of pixels per frame, used for dynamic resolution.")
    group.add_argument("--resize_mode", type=str, default="fit", choices=["crop", "fit"], help="Resize behavior: crop (center crop), fit (no crop), short (scale by short edge).")
    group.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    group.add_argument("--num_history_frames", type=int, default=1, help="Number of conditioning history frames. Must satisfy 1 <= num_history_frames < num_frames.")
    return parser

def add_model_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "model")
    group.add_argument("--model_paths", type=str, default=None, help="Root path of the WAN pretrained weights.")
    group.add_argument("--load_modules", type=str, default=None, help="Comma-separated modules to load: dit,text,vae,image,action. Supported variants: action:noise|adaln|none and text:t5|emb|none. You can also set the default via env LOAD_MODULES.")
    group.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    group.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    group.add_argument("--extra_inputs", default="input_image", help="Additional model inputs, comma-separated.")
    group.add_argument("--fp8_models", default=None, help="Models with FP8 precision, comma-separated.")
    group.add_argument("--offload_models", default=None, help="Models with offload, comma-separated. Only used in splited training.")
    return parser

def add_training_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "training")
    group.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    group.add_argument("--seed", type=int, default=42, help="Random seed for python/numpy/torch.")
    group.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    group.add_argument("--trainable_models", type=str, default="dit", help="Models to train, e.g., dit, vae, text_encoder.")
    group.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    group.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    group.add_argument("--task", type=str, default="sft", required=False, help="Task type.")
    group.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision mode.")
    group.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    group.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    return parser

def add_output_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "output")
    group.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    group.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    group.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    group.add_argument("--ckpt_path", type=str, default=None, help="Path to model checkpoint (.safetensors) used to initialize training weights (model-only resume).")
    group.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint directory saved by accelerator (e.g., output_path/epoch-0).")
    return parser

def add_lora_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "lora")
    group.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    group.add_argument("--lora_target_modules", type=str, default=None, help="Which layers LoRA is added to (default: q,k,v,o,ffn.0,ffn.2).")
    group.add_argument("--lora_rank", type=int, default=None, help="Rank of LoRA.")
    group.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    group.add_argument("--preset_lora_path", type=str, default=None, help="Path to the preset LoRA checkpoint. If provided, this LoRA will be fused to the base model.")
    group.add_argument("--preset_lora_model", type=str, default=None, help="Which model the preset LoRA is fused to.")
    return parser

def add_gradient_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "gradient")
    group.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    group.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    group.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    group.add_argument("--max_grad_norm", type=float, default=0.5, help="Maximum gradient norm for clipping.")
    return parser

def add_tracking_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "tracking")
    group.add_argument("--use_wandb", type=int, choices=[0, 1], default=0, help="Enable Weights & Biases tracking (1 启用，0 关闭).")
    group.add_argument("--use_swanlab", type=int, choices=[0, 1], default=0, help="Enable SwanLab tracking (1 启用，0 关闭).")
    group.add_argument("--swanlab_experiment_name", type=str, default=None, help="SwanLab experiment name. Defaults to output_path.")
    return parser

def add_action_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "action")
    group.add_argument("--action_type", type=str, choices=["state_joint", "state_pose", "action_joint", "action_pose"], default=None, help="Which action/state slice to load from parquet.")
    group.add_argument("--action_stat_path", type=str, default=None, help="Path to action/state normalization stats (stat.json). Defaults to dataset_base_path/meta/stat.json if present.")
    return parser

def add_state_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "state")
    group.add_argument("--state_type", type=str, choices=["state_pose_7d"], default=None, help="State slice to load from parquet for DROID-style datasets.")
    group.add_argument("--state_stat_path", type=str, default=None, help="Path to state normalization stats JSON.")
    return parser

def add_cross_view_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "cross_view")
    group.add_argument("--cross_view_source_views", type=str, default="0,1", help="Comma-separated source view indices used for conditioning.")
    group.add_argument("--cross_view_target_view", type=int, default=2, help="Target first-person view index in the joint video tensor.")
    group.add_argument("--cross_view_placeholder_mode", type=str, choices=["zeros", "source_mean"], default="zeros", help="How to mask the target first frame in stage-1/2 cross-view training.")
    group.add_argument("--cross_view_source_loss_weight", type=float, default=0.1, help="Auxiliary loss weight for source-view reconstruction in cross_view_stage1.")
    group.add_argument("--cross_view_old_branch_dropout", type=float, default=0.0, help="Dropout probability for the legacy image-conditioning branch in cross_view_stage2.")
    group.add_argument("--cross_view_projector_hidden_dim", type=int, default=512, help="Hidden dimension of the 3D source-video projector in cross_view_stage2.")
    group.add_argument("--cross_view_source_injection_mode", type=str, choices=["none", "global_concat", "temporal_local"], default="temporal_local", help="How source-view temporal memory is injected in cross_view_stage2. Set to none to disable source-memory cross-attention.")
    group.add_argument("--cross_view_source_branch_mode", type=str, choices=["none", "sigma_matched_clamp"], default="sigma_matched_clamp", help="How source-view latent branches are constrained in cross_view_stage2.")
    group.add_argument("--cross_view_source_window_radius", type=int, default=1, help="Temporal radius used by local source-memory injection in cross_view_stage2.")
    group.add_argument("--cross_view_source_gate_mode", type=str, choices=["none", "scalar", "state_aware"], default="scalar", help="How to gate source-view memory in cross_view_stage2.")
    group.add_argument("--cross_view_temp_loss_weight", type=float, default=0.1, help="Temporal consistency loss weight for target-view latent prediction in cross_view_stage2.")
    group.add_argument("--cross_view_state_loss_weight", type=float, default=0.05, help="Auxiliary target-state prediction loss weight in cross_view_stage2.")
    group.add_argument("--cross_view_global_source_tokens", type=int, default=0, help="Number of pooled global source tokens appended to context in cross_view_stage2. Set to 0 to disable.")
    group.add_argument("--cross_view_aux_loss_warmup_ratio", type=float, default=0.0, help="Fraction of training progress during which stage2 auxiliary losses stay at zero before linearly ramping up.")
    group.add_argument("--cross_view_old_branch_dropout_schedule", type=str, choices=["fixed", "linear_warmup_to_high"], default="linear_warmup_to_high", help="Scheduling mode for legacy image-branch dropout in cross_view_stage2.")
    group.add_argument("--cross_view_legacy_branch_schedule", type=str, choices=["anchor_then_dropout"], default=None, help="Optional explicit schedule for the legacy image-conditioning branch. When set, it overrides cross_view_old_branch_dropout_schedule.")
    group.add_argument("--cross_view_disable_legacy_image_branch", type=int, default=0, choices=[0, 1], help="Force-zero the legacy y/clip image-conditioning branch in cross_view_stage2 while keeping tensor shapes unchanged.")
    group.add_argument("--cross_view_use_tail_anchor", type=int, default=0, choices=[0, 1], help="Enable dual-end anchoring in cross_view_stage2: in addition to the head anchor (target_history_latents), also overwrite the last tail_t latent timesteps with target_tail_latents from the cache. Requires cache built with tool/build_cross_view_tail_cache.py.")
    group.add_argument("--num_tail_frames", type=int, default=1, help="Number of conditioning tail frames (mirror of num_history_frames). Each tail frame collapses into ((N-1)//4)+1 latent timesteps via VAE temporal downsampling. Only effective when --cross_view_use_tail_anchor=1.")
    group.add_argument("--cross_view_tail_anchor_dropout", type=float, default=0.0, help="Probability to randomly zero-out the tail anchor during training (augmentation for inference robustness when tail anchor is unavailable). 0.0 disables.")
    group.add_argument("--cross_view_use_keyframe_anchor", type=int, default=0, choices=[0, 1], help="Enable Plan-A keyframe anchors in the legacy y channel. Requires cache refreshed/rebuilt with matching keyframe anchors.")
    group.add_argument("--num_keyframe_anchors", type=int, default=3, help="Number of synthesized keyframe anchors per clip.")
    group.add_argument("--keyframe_anchor_dropout", type=float, default=0.0, help="Probability to drop synthesized keyframe anchors while building/refeshing cache. Cached training reads y directly.")
    group.add_argument("--keyframe_anchor_manifest_train", type=str, default=None, help="Keyframe anchor manifest for train clips.")
    group.add_argument("--keyframe_anchor_manifest_val", type=str, default=None, help="Keyframe anchor manifest for val clips.")
    group.add_argument("--keyframe_anchor_image_root_train", type=str, default=None, help="Directory containing synthesized train keyframe image folders.")
    group.add_argument("--keyframe_anchor_image_root_val", type=str, default=None, help="Directory containing synthesized val keyframe image folders.")
    group.add_argument("--cross_view_3d_noise_prior_mode", type=str, choices=["none", "scene_action_grid", "dynamic_view_action"], default="none", help="Stage2-only 3D-token structured noise prior mode.")
    group.add_argument("--cross_view_3d_noise_prior_weight", type=float, default=0.1, help="Mixing weight lambda for the stage2 3D structured noise prior.")
    group.add_argument("--cross_view_3d_noise_anchor_attenuation", type=float, default=1.0, help="How strongly y-channel anchor slots suppress the 3D noise prior (1.0 disables it at anchor slots).")
    group.add_argument("--scene_token_checkpoint", type=str, default=None, help="Path to LagerNVS checkpoint for scene token extraction.")
    group.add_argument("--scene_token_pool_size", type=int, default=512, help="Number of scene tokens after spatial pooling (0=no pooling).")
    group.add_argument("--geometry_gate_mode", type=str, default="learned", choices=["learned", "constant"], help="Mode for timestep-adaptive geometry gate.")
    group.add_argument("--geometry_sidecar_cache_path", type=str, default=None, help="Optional cache root containing camera-aware geometry sidecar .pth files.")
    group.add_argument("--geometry_use_camera_tokens", type=int, default=0, choices=[0, 1], help="Use real DROID camera tokens for geometry-aware conditioning when available.")
    group.add_argument("--geometry_target_camera_mode", type=str, default="none", choices=["none", "add_time_mlp"], help="How target-view camera tokens are injected into target-only stage2 DiT tokens.")
    group.add_argument("--geometry_scene_token_source", type=str, default="cached_zero_cam", choices=["camera_aware_sidecar", "cached_zero_cam", "runtime"], help="Source of scene tokens for geometry-aware cross attention.")
    group.add_argument("--alignment_loss_weight", type=float, default=0.1, help="Weight for geometry alignment loss between DiT hidden states and scene tokens.")
    group.add_argument("--alignment_loss_warmup_ratio", type=float, default=0.1, help="Fraction of training progress used to linearly ramp alignment loss from zero to full weight.")
    group.add_argument("--wrist_first_frame_index", type=str, default=None, help="JSON index mapping (episode_index,start_frame) → wrist first-frame PNG path. Used to replace FP placeholder latent at frame 0.")
    return parser

def add_infer_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "infer")
    group.add_argument("--ckpt_path", dest="checkpoint_path", type=str, default=None, help=("Path to checkpoint file or directory (optional; merged onto pretrained WAN weights). " "Supports checkpoints that include dit/action_encoder keys."))
    group.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale for generation")
    group.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps.")
    group.add_argument("--negative_prompt", type=str, default=("The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion"), help="Negative prompt for generation")
    group.add_argument("--negative_prompt_emb", type=str, default=None, help="Path to the pre-extracted negative prompt embedding.")
    group.add_argument("--quality", type=int, default=5, help="Output video quality.")
    group.add_argument("--metrics", dest="enable_metrics", type=int, default=1, choices=[0, 1], help="Enable (1) or disable (0) evaluation metrics")
    group.add_argument("--chunk_infer", type=int, default=1, choices=[0, 1], help="Enable chunked inference with 81-frame segments (0=off, 1=on).")
    group.add_argument("--fps", type=int, default=30, help="Output video FPS")
    return parser

def add_general_config(parser: argparse.ArgumentParser):
    parser = add_dataset_base_config(parser)
    parser = add_model_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_lora_config(parser)
    parser = add_gradient_config(parser)
    parser = add_tracking_config(parser)
    return parser
