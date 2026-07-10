#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuehao/.conda/envs/studio/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

TAG="${TAG:-stage2_3d_noise_prior}"
TASK="${TASK:-cross_view_stage2}"
NUM_FRAMES="${NUM_FRAMES:-81}"
HEIGHT="${HEIGHT:-180}"
WIDTH="${WIDTH:-320}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

MODEL_DIR="${MODEL_DIR:-/home/xuehao/xh/projects/DiffSynth-Studio-old/models/PAI}"
DATASET_META_ROOT="${DATASET_META_ROOT:-/data2/xuehao/datasets/droid_success_high_quality_crossview_meta}"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-$DATASET_META_ROOT/meta/episodes_cross_view_train_81_small16567.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$DATASET_META_ROOT/meta/episodes_cross_view_val_81_small200.jsonl}"

CACHED_DATASET_PATH="${CACHED_DATASET_PATH:-$DATASET_META_ROOT/cache_crossview_81f_180x320_lagernvs_iter060001_keyframe}"
STATE_STAT_PATH="${STATE_STAT_PATH:-$DATASET_META_ROOT/meta/stat_state_pose_7d.json}"
NEG_PROMPT_EMB="${NEG_PROMPT_EMB:-$DATASET_META_ROOT/prompt_emb/neg_prompt.pt}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/Ckpt/${TAG}}"

LOAD_MODULES="${LOAD_MODULES:-dit,text:emb,vae,image,action:noise}"

CROSS_VIEW_SOURCE_VIEWS="${CROSS_VIEW_SOURCE_VIEWS:-0,1}"
CROSS_VIEW_TARGET_VIEW="${CROSS_VIEW_TARGET_VIEW:-2}"
CROSS_VIEW_PLACEHOLDER_MODE="${CROSS_VIEW_PLACEHOLDER_MODE:-zeros}"

CROSS_VIEW_SOURCE_LOSS_WEIGHT="${CROSS_VIEW_SOURCE_LOSS_WEIGHT:-0.8}"
CROSS_VIEW_OLD_BRANCH_DROPOUT="${CROSS_VIEW_OLD_BRANCH_DROPOUT:-0.0}"

CROSS_VIEW_PROJECTOR_HIDDEN_DIM="${CROSS_VIEW_PROJECTOR_HIDDEN_DIM:-512}"
CROSS_VIEW_SOURCE_INJECTION_MODE="${CROSS_VIEW_SOURCE_INJECTION_MODE:-temporal_local}"
CROSS_VIEW_SOURCE_BRANCH_MODE="${CROSS_VIEW_SOURCE_BRANCH_MODE:-sigma_matched_clamp}"
CROSS_VIEW_SOURCE_WINDOW_RADIUS="${CROSS_VIEW_SOURCE_WINDOW_RADIUS:-1}"
CROSS_VIEW_SOURCE_GATE_MODE="${CROSS_VIEW_SOURCE_GATE_MODE:-scalar}"

CROSS_VIEW_TEMP_LOSS_WEIGHT="${CROSS_VIEW_TEMP_LOSS_WEIGHT:-0.1}"
CROSS_VIEW_STATE_LOSS_WEIGHT="${CROSS_VIEW_STATE_LOSS_WEIGHT:-0.05}"
CROSS_VIEW_GLOBAL_SOURCE_TOKENS="${CROSS_VIEW_GLOBAL_SOURCE_TOKENS:-0}"
CROSS_VIEW_AUX_LOSS_WARMUP_RATIO="${CROSS_VIEW_AUX_LOSS_WARMUP_RATIO:-0.0}"

CROSS_VIEW_OLD_BRANCH_DROPOUT_SCHEDULE="${CROSS_VIEW_OLD_BRANCH_DROPOUT_SCHEDULE:-linear_warmup_to_high}"
CROSS_VIEW_LEGACY_BRANCH_SCHEDULE="${CROSS_VIEW_LEGACY_BRANCH_SCHEDULE:-}"
CROSS_VIEW_DISABLE_LEGACY_IMAGE_BRANCH="${CROSS_VIEW_DISABLE_LEGACY_IMAGE_BRANCH:-0}"

CROSS_VIEW_USE_TAIL_ANCHOR="${CROSS_VIEW_USE_TAIL_ANCHOR:-1}"
NUM_TAIL_FRAMES="${NUM_TAIL_FRAMES:-1}"
CROSS_VIEW_TAIL_ANCHOR_DROPOUT="${CROSS_VIEW_TAIL_ANCHOR_DROPOUT:-0.0}"

CROSS_VIEW_USE_KEYFRAME_ANCHOR="${CROSS_VIEW_USE_KEYFRAME_ANCHOR:-1}"
NUM_KEYFRAME_ANCHORS="${NUM_KEYFRAME_ANCHORS:-3}"
KEYFRAME_ANCHOR_DROPOUT="${KEYFRAME_ANCHOR_DROPOUT:-0.0}"

KEYFRAME_ANCHOR_MANIFEST_TRAIN="${KEYFRAME_ANCHOR_MANIFEST_TRAIN:-/data2/xuehao/datasets/droid_success_wrist_key_frame_train/lagernvs_keyframe_train_manifest.jsonl}"
KEYFRAME_ANCHOR_MANIFEST_VAL="${KEYFRAME_ANCHOR_MANIFEST_VAL:-/data2/xuehao/datasets/droid_success_wrist_key_frame_val/lagernvs_keyframe_eval_manifest.jsonl}"

KEYFRAME_ANCHOR_IMAGE_ROOT_TRAIN="${KEYFRAME_ANCHOR_IMAGE_ROOT_TRAIN:-/data2/xuehao/datasets/droid_success_wrist_key_frame_train/lagernvs_keyframe_val/lagernvs_keyframe_train/images_iter_000000}"
KEYFRAME_ANCHOR_IMAGE_ROOT_VAL="${KEYFRAME_ANCHOR_IMAGE_ROOT_VAL:-/data2/xuehao/datasets/droid_success_wrist_key_frame_val/lagernvs_keyframe_val/lagernvs_keyframe_val/images_iter_000000}"

SCENE_TOKEN_CHECKPOINT="${SCENE_TOKEN_CHECKPOINT-/home/xuehao/xh/projects/DiffSynth-Studio_v2/lagernvs/ckpt/droid_base_stage2/checkpoint_0060000.pt}"
SCENE_TOKEN_POOL_SIZE="${SCENE_TOKEN_POOL_SIZE:-512}"

GEOMETRY_GATE_MODE="${GEOMETRY_GATE_MODE:-learned}"
GEOMETRY_SIDECAR_CACHE_PATH="${GEOMETRY_SIDECAR_CACHE_PATH-/data2/xuehao/datasets/droid_success_high_quality_crossview_meta/geometry_sidecar_lagernvs_strict_iter060000}"
GEOMETRY_USE_CAMERA_TOKENS="${GEOMETRY_USE_CAMERA_TOKENS:-1}"
GEOMETRY_TARGET_CAMERA_MODE="${GEOMETRY_TARGET_CAMERA_MODE:-add_time_mlp}"
GEOMETRY_SCENE_TOKEN_SOURCE="${GEOMETRY_SCENE_TOKEN_SOURCE:-camera_aware_sidecar}"

ALIGNMENT_LOSS_WEIGHT="${ALIGNMENT_LOSS_WEIGHT:-0.0}"
ALIGNMENT_LOSS_WARMUP_RATIO="${ALIGNMENT_LOSS_WARMUP_RATIO:-0}"

WRIST_FIRST_FRAME_INDEX="${WRIST_FIRST_FRAME_INDEX:-/data2/xuehao/datasets/droid_success_high_quality_crossview_meta/meta/wrist_frame_index_all.json}"

USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-1}"

TRAINABLE_MODELS="${TRAINABLE_MODELS:-dit}"

CKPT_PATH="${CKPT_PATH:-/ssdfs/datahome/usersht/dev/xh/projects/DiffSynth-Studio_v2/Ckpt/droid_success_high_quality_crossview_stage2_sidecar/epoch-5/epoch-5.safetensors}"
#/ssdfs/datahome/usersht/dev/xh/projects/DiffSynth-Studio_v2/Ckpt/droid_success_high_quality_crossview_stage2_sidecar/epoch-5/epoch-5.safetensors
#
# 3D noise prior
CROSS_VIEW_3D_NOISE_PRIOR_MODE="${CROSS_VIEW_3D_NOISE_PRIOR_MODE:-scene_action_grid}"
CROSS_VIEW_3D_NOISE_PRIOR_WEIGHT="${CROSS_VIEW_3D_NOISE_PRIOR_WEIGHT:-0.5}"
CROSS_VIEW_3D_NOISE_ANCHOR_ATTENUATION="${CROSS_VIEW_3D_NOISE_ANCHOR_ATTENUATION:-1.0}"


check_manifest_refs() {
  local manifest_path="$1"
  "$PYTHON_BIN" - "$DATASET_META_ROOT" "$manifest_path" <<'PY'
import json
import os
import sys

root, manifest = sys.argv[1:3]
counts = {
    "rows": 0,
    "missing_video": 0,
    "missing_state": 0,
    "missing_prompt": 0,
}

with open(manifest, "r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        counts["rows"] += 1
        row = json.loads(line)

        for video in row.get("video", []):
            data_path = video.get("data")
            if not data_path or not os.path.exists(os.path.join(root, data_path)):
                counts["missing_video"] += 1

        state = row.get("state")
        if state:
            data_path = state.get("data")
            if not data_path or not os.path.exists(os.path.join(root, data_path)):
                counts["missing_state"] += 1

        prompt_path = row.get("prompt_emb")
        if not prompt_path or not os.path.exists(os.path.join(root, prompt_path)):
            counts["missing_prompt"] += 1

print(
    f"[CHECK] {os.path.basename(manifest)} "
    f"rows={counts['rows']} "
    f"missing_video_refs={counts['missing_video']} "
    f"missing_state_refs={counts['missing_state']} "
    f"missing_prompt_refs={counts['missing_prompt']}"
)

if counts["missing_video"] or counts["missing_state"] or counts["missing_prompt"]:
    sys.exit(1)
PY
}

echo "REPO_ROOT=$REPO_ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "TASK=$TASK"
echo "MODEL_DIR=$MODEL_DIR"
echo "TRAIN_MANIFEST=$TRAIN_MANIFEST"
echo "VAL_MANIFEST=$VAL_MANIFEST"
echo "CACHED_DATASET_PATH=$CACHED_DATASET_PATH"
echo "STATE_STAT_PATH=$STATE_STAT_PATH"
echo "NEG_PROMPT_EMB=$NEG_PROMPT_EMB"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "SCENE_TOKEN_CHECKPOINT=$SCENE_TOKEN_CHECKPOINT"
echo "GEOMETRY_SIDECAR_CACHE_PATH=$GEOMETRY_SIDECAR_CACHE_PATH"
echo "GEOMETRY_USE_CAMERA_TOKENS=$GEOMETRY_USE_CAMERA_TOKENS"
echo "GEOMETRY_TARGET_CAMERA_MODE=$GEOMETRY_TARGET_CAMERA_MODE"
echo "GEOMETRY_SCENE_TOKEN_SOURCE=$GEOMETRY_SCENE_TOKEN_SOURCE"
echo "WRIST_FIRST_FRAME_INDEX=$WRIST_FIRST_FRAME_INDEX"
echo "USE_GRADIENT_CHECKPOINTING=$USE_GRADIENT_CHECKPOINTING"

if [[ -n "$CACHED_DATASET_PATH" ]]; then
  if [[ ! -d "$CACHED_DATASET_PATH/train" ]]; then
    echo "[ERROR] cached train split not found: $CACHED_DATASET_PATH/train"
    exit 1
  fi
  if [[ ! -f "$CACHED_DATASET_PATH/cache_config.json" ]]; then
    echo "[ERROR] cache_config.json not found: $CACHED_DATASET_PATH/cache_config.json"
    exit 1
  fi
else
  if [[ ! -f "$TRAIN_MANIFEST" ]]; then
    echo "[ERROR] train manifest not found: $TRAIN_MANIFEST"
    exit 1
  fi

  if [[ ! -f "$STATE_STAT_PATH" ]]; then
    echo "[ERROR] state stat file not found: $STATE_STAT_PATH"
    exit 1
  fi

  if ! check_manifest_refs "$TRAIN_MANIFEST"; then
    echo "[ERROR] train manifest references missing assets"
    exit 1
  fi
fi

if [[ ! -f "$NEG_PROMPT_EMB" ]]; then
  echo "[WARN] negative prompt embedding not found: $NEG_PROMPT_EMB"
fi

if [[ "$TASK" == "cross_view_stage2" && -z "$CKPT_PATH" ]]; then
  echo "[ERROR] CKPT_PATH is required for cross_view_stage2"
  exit 1
fi

if [[ -n "$SCENE_TOKEN_CHECKPOINT" && ! -f "$SCENE_TOKEN_CHECKPOINT" ]]; then
  echo "[ERROR] scene token checkpoint not found: $SCENE_TOKEN_CHECKPOINT"
  exit 1
fi

if [[ -n "$WRIST_FIRST_FRAME_INDEX" && ! -f "$WRIST_FIRST_FRAME_INDEX" ]]; then
  echo "[ERROR] wrist first-frame index not found: $WRIST_FIRST_FRAME_INDEX"
  exit 1
fi

cd "$REPO_ROOT"
CMD=(
  "$PYTHON_BIN" -m accelerate.commands.launch
  examples/wanvideo/model_training/train.py
  --dataset_base_path "$DATASET_META_ROOT"
  --data_file_keys "video,state,prompt_emb"
  --state_type "state_pose_7d"
  --height "$HEIGHT"
  --width "$WIDTH"
  --num_frames "$NUM_FRAMES"
  --num_history_frames 1
  --dataset_repeat "$DATASET_REPEAT"
  --model_paths "$MODEL_DIR"
  --learning_rate "$LEARNING_RATE"
  --num_epochs "$NUM_EPOCHS"
  --output_path "$OUTPUT_PATH"
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS"
  --mixed_precision "$MIXED_PRECISION"
  --use_swanlab 0
  --load_modules "$LOAD_MODULES"
  --task "$TASK"
  --trainable_models "$TRAINABLE_MODELS"
  --cross_view_source_views "$CROSS_VIEW_SOURCE_VIEWS"
  --cross_view_target_view "$CROSS_VIEW_TARGET_VIEW"
  --cross_view_placeholder_mode "$CROSS_VIEW_PLACEHOLDER_MODE"
  --cross_view_source_loss_weight "$CROSS_VIEW_SOURCE_LOSS_WEIGHT"
  --cross_view_old_branch_dropout "$CROSS_VIEW_OLD_BRANCH_DROPOUT"
  --cross_view_projector_hidden_dim "$CROSS_VIEW_PROJECTOR_HIDDEN_DIM"
  --cross_view_source_injection_mode "$CROSS_VIEW_SOURCE_INJECTION_MODE"
  --cross_view_source_branch_mode "$CROSS_VIEW_SOURCE_BRANCH_MODE"
  --cross_view_source_window_radius "$CROSS_VIEW_SOURCE_WINDOW_RADIUS"
  --cross_view_source_gate_mode "$CROSS_VIEW_SOURCE_GATE_MODE"
  --cross_view_temp_loss_weight "$CROSS_VIEW_TEMP_LOSS_WEIGHT"
  --cross_view_state_loss_weight "$CROSS_VIEW_STATE_LOSS_WEIGHT"
  --cross_view_global_source_tokens "$CROSS_VIEW_GLOBAL_SOURCE_TOKENS"
  --cross_view_aux_loss_warmup_ratio "$CROSS_VIEW_AUX_LOSS_WARMUP_RATIO"
  --cross_view_old_branch_dropout_schedule "$CROSS_VIEW_OLD_BRANCH_DROPOUT_SCHEDULE"
  --cross_view_disable_legacy_image_branch "$CROSS_VIEW_DISABLE_LEGACY_IMAGE_BRANCH"
  --cross_view_use_tail_anchor "$CROSS_VIEW_USE_TAIL_ANCHOR"
  --num_tail_frames "$NUM_TAIL_FRAMES"
  --cross_view_tail_anchor_dropout "$CROSS_VIEW_TAIL_ANCHOR_DROPOUT"
  --cross_view_use_keyframe_anchor "$CROSS_VIEW_USE_KEYFRAME_ANCHOR"
  --num_keyframe_anchors "$NUM_KEYFRAME_ANCHORS"
  --keyframe_anchor_dropout "$KEYFRAME_ANCHOR_DROPOUT"
  --scene_token_pool_size "$SCENE_TOKEN_POOL_SIZE"
  --geometry_gate_mode "$GEOMETRY_GATE_MODE"
  --geometry_use_camera_tokens "$GEOMETRY_USE_CAMERA_TOKENS"
  --geometry_target_camera_mode "$GEOMETRY_TARGET_CAMERA_MODE"
  --geometry_scene_token_source "$GEOMETRY_SCENE_TOKEN_SOURCE"
  --alignment_loss_weight "$ALIGNMENT_LOSS_WEIGHT"
  --alignment_loss_warmup_ratio "$ALIGNMENT_LOSS_WARMUP_RATIO"
  --cross_view_3d_noise_prior_mode "$CROSS_VIEW_3D_NOISE_PRIOR_MODE"
  --cross_view_3d_noise_prior_weight "$CROSS_VIEW_3D_NOISE_PRIOR_WEIGHT"
  --cross_view_3d_noise_anchor_attenuation 1.0
)

if [[ -n "$CACHED_DATASET_PATH" ]]; then
  CMD+=(--cached_dataset_path "$CACHED_DATASET_PATH")
else
  CMD+=(--dataset_metadata_path "$TRAIN_MANIFEST")
  CMD+=(--state_stat_path "$STATE_STAT_PATH")
fi

if [[ -n "$CKPT_PATH" ]]; then
  CMD+=(--ckpt_path "$CKPT_PATH")
fi

if [[ -n "$CROSS_VIEW_LEGACY_BRANCH_SCHEDULE" ]]; then
  CMD+=(--cross_view_legacy_branch_schedule "$CROSS_VIEW_LEGACY_BRANCH_SCHEDULE")
fi

if [[ -n "$SCENE_TOKEN_CHECKPOINT" ]]; then
  CMD+=(--scene_token_checkpoint "$SCENE_TOKEN_CHECKPOINT")
fi

if [[ -n "$GEOMETRY_SIDECAR_CACHE_PATH" ]]; then
  CMD+=(--geometry_sidecar_cache_path "$GEOMETRY_SIDECAR_CACHE_PATH")
fi

if [[ -n "$WRIST_FIRST_FRAME_INDEX" ]]; then
  CMD+=(--wrist_first_frame_index "$WRIST_FIRST_FRAME_INDEX")
fi

if [[ -n "$KEYFRAME_ANCHOR_MANIFEST_TRAIN" ]]; then
  CMD+=(--keyframe_anchor_manifest_train "$KEYFRAME_ANCHOR_MANIFEST_TRAIN")
fi

if [[ -n "$KEYFRAME_ANCHOR_MANIFEST_VAL" ]]; then
  CMD+=(--keyframe_anchor_manifest_val "$KEYFRAME_ANCHOR_MANIFEST_VAL")
fi

if [[ -n "$KEYFRAME_ANCHOR_IMAGE_ROOT_TRAIN" ]]; then
  CMD+=(--keyframe_anchor_image_root_train "$KEYFRAME_ANCHOR_IMAGE_ROOT_TRAIN")
fi

if [[ -n "$KEYFRAME_ANCHOR_IMAGE_ROOT_VAL" ]]; then
  CMD+=(--keyframe_anchor_image_root_val "$KEYFRAME_ANCHOR_IMAGE_ROOT_VAL")
fi

if [[ "$USE_GRADIENT_CHECKPOINTING" == "1" ]]; then
  CMD+=(--use_gradient_checkpointing)
fi

printf 'Running command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
