#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/env/conda/envs/studio/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a CACHE_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#CACHE_GPU_IDS[@]}" -eq 0 || -z "${CACHE_GPU_IDS[0]}" ]]; then
  CACHE_GPU_IDS=("0")
fi

TASK="${TASK:-cross_view_stage1}"
if [[ -z "${TAG:-}" ]]; then
  if [[ "$TASK" == "cross_view_stage2" ]]; then
    TAG="droid_success_high_quality_crossview_cache_stage2"
  else
    TAG="droid_success_high_quality_crossview_cache_stage1"
  fi
fi

NUM_FRAMES="${NUM_FRAMES:-81}"
HEIGHT="${HEIGHT:-180}"
WIDTH="${WIDTH:-320}"
NUM_HISTORY_FRAMES="${NUM_HISTORY_FRAMES:-1}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/Ckpt/${TAG}}"

MODEL_DIR="${MODEL_DIR:-/root/autodl-fs/models/PAI}"
DATASET_META_ROOT="${DATASET_META_ROOT:-/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$DATASET_META_ROOT/meta/episodes_cross_view_train_81_small16567.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$DATASET_META_ROOT/meta/episodes_cross_view_val_81_small200.jsonl}"
STATE_STAT_PATH="${STATE_STAT_PATH:-$DATASET_META_ROOT/meta/stat_state_pose_7d.json}"
NEG_PROMPT_EMB="${NEG_PROMPT_EMB:-$DATASET_META_ROOT/prompt_emb/neg_prompt.pt}"
CACHE_ROOT="${CACHE_ROOT:-$DATASET_META_ROOT/cache_crossview_81f_180x320_lagernvs_iter060001}"

BUILD_CACHE="${BUILD_CACHE:-1}"
FORCE_REBUILD_CACHE="${FORCE_REBUILD_CACHE:-0}"
CACHE_DEVICE="${CACHE_DEVICE:-cuda}"
CACHE_NUM_SHARDS="${CACHE_NUM_SHARDS:-8}"
CACHE_SHARD_MODE="${CACHE_SHARD_MODE:-strided}"
CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-0}"
CACHE_PREFETCH_FACTOR="${CACHE_PREFETCH_FACTOR:-2}"
CACHE_PIN_MEMORY="${CACHE_PIN_MEMORY:-0}"
CACHE_VAE_TILED_ENCODE="${CACHE_VAE_TILED_ENCODE:-0}"
CACHE_SKIP_LEGACY_BRANCH="${CACHE_SKIP_LEGACY_BRANCH:-0}"
RUN_TRAIN_AFTER_CACHE="${RUN_TRAIN_AFTER_CACHE:-0}"
RESIZE_MODE="${RESIZE_MODE:-fit}"
MODEL_ID_WITH_ORIGIN_PATHS="${MODEL_ID_WITH_ORIGIN_PATHS:-}"

LOAD_MODULES="${LOAD_MODULES:-dit,text:emb,vae,image,action:noise}"
CROSS_VIEW_SOURCE_VIEWS="${CROSS_VIEW_SOURCE_VIEWS:-0,1}"
CROSS_VIEW_TARGET_VIEW="${CROSS_VIEW_TARGET_VIEW:-2}"
CROSS_VIEW_PLACEHOLDER_MODE="${CROSS_VIEW_PLACEHOLDER_MODE:-zeros}"
CROSS_VIEW_SOURCE_LOSS_WEIGHT="${CROSS_VIEW_SOURCE_LOSS_WEIGHT:-0.8}"
CROSS_VIEW_OLD_BRANCH_DROPOUT="${CROSS_VIEW_OLD_BRANCH_DROPOUT:-0.2}"
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
CROSS_VIEW_USE_TAIL_ANCHOR="${CROSS_VIEW_USE_TAIL_ANCHOR:-0}"
NUM_TAIL_FRAMES="${NUM_TAIL_FRAMES:-1}"
CROSS_VIEW_TAIL_ANCHOR_DROPOUT="${CROSS_VIEW_TAIL_ANCHOR_DROPOUT:-0.0}"
SCENE_TOKEN_CHECKPOINT="${SCENE_TOKEN_CHECKPOINT:-}"
SCENE_TOKEN_POOL_SIZE="${SCENE_TOKEN_POOL_SIZE:-512}"
GEOMETRY_GATE_MODE="${GEOMETRY_GATE_MODE:-learned}"
GEOMETRY_SIDECAR_CACHE_PATH="${GEOMETRY_SIDECAR_CACHE_PATH:-}"
GEOMETRY_USE_CAMERA_TOKENS="${GEOMETRY_USE_CAMERA_TOKENS:-0}"
GEOMETRY_TARGET_CAMERA_MODE="${GEOMETRY_TARGET_CAMERA_MODE:-none}"
GEOMETRY_SCENE_TOKEN_SOURCE="${GEOMETRY_SCENE_TOKEN_SOURCE:-cached_zero_cam}"
ALIGNMENT_LOSS_WEIGHT="${ALIGNMENT_LOSS_WEIGHT:-0.1}"
ALIGNMENT_LOSS_WARMUP_RATIO="${ALIGNMENT_LOSS_WARMUP_RATIO:-0.1}"
WRIST_FIRST_FRAME_INDEX="${WRIST_FIRST_FRAME_INDEX:-}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-0}"

TRAINABLE_MODELS="${TRAINABLE_MODELS:-dit}"
CKPT_PATH="${CKPT_PATH:-}"

run_cmd() {
  printf 'Running command:\n'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

sanitize_thread_env() {
  if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]] || [[ "${OMP_NUM_THREADS:-0}" -lt 1 ]]; then
    export OMP_NUM_THREADS=1
  fi
  if ! [[ "${MKL_NUM_THREADS:-}" =~ ^[0-9]+$ ]] || [[ "${MKL_NUM_THREADS:-0}" -lt 1 ]]; then
    export MKL_NUM_THREADS=1
  fi
}

count_manifest_rows() {
  local manifest_path="$1"
  "$PYTHON_BIN" - "$manifest_path" <<'PY'
import sys

count = 0
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            count += 1
print(count)
PY
}

count_cache_files() {
  local split_dir="$1"
  if [[ ! -d "$split_dir" ]]; then
    echo 0
    return
  fi
  # -L makes find follow symlinks so that files (or symlinks pointing to files)
  # both count. This is required when the cache root is a symlink-only subset
  # produced by tool/filter_cross_view_cache_subset.py.
  find -L "$split_dir" -maxdepth 1 -type f -name '*.pth' | wc -l
}

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
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CACHE_GPU_IDS=${CACHE_GPU_IDS[*]}"
echo "TASK=$TASK"
echo "TAG=$TAG"
echo "MODEL_DIR=$MODEL_DIR"
echo "DATASET_META_ROOT=$DATASET_META_ROOT"
echo "TRAIN_MANIFEST=$TRAIN_MANIFEST"
echo "VAL_MANIFEST=$VAL_MANIFEST"
echo "STATE_STAT_PATH=$STATE_STAT_PATH"
echo "NEG_PROMPT_EMB=$NEG_PROMPT_EMB"
echo "CACHE_ROOT=$CACHE_ROOT"
echo "BUILD_CACHE=$BUILD_CACHE"
echo "FORCE_REBUILD_CACHE=$FORCE_REBUILD_CACHE"
echo "CACHE_NUM_SHARDS=$CACHE_NUM_SHARDS"
echo "CACHE_SHARD_MODE=$CACHE_SHARD_MODE"
echo "CACHE_NUM_WORKERS=$CACHE_NUM_WORKERS"
echo "CACHE_PREFETCH_FACTOR=$CACHE_PREFETCH_FACTOR"
echo "CACHE_PIN_MEMORY=$CACHE_PIN_MEMORY"
echo "CACHE_VAE_TILED_ENCODE=$CACHE_VAE_TILED_ENCODE"
echo "CACHE_SKIP_LEGACY_BRANCH=$CACHE_SKIP_LEGACY_BRANCH"
echo "RUN_TRAIN_AFTER_CACHE=$RUN_TRAIN_AFTER_CACHE"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "SCENE_TOKEN_CHECKPOINT=$SCENE_TOKEN_CHECKPOINT"
echo "GEOMETRY_SIDECAR_CACHE_PATH=$GEOMETRY_SIDECAR_CACHE_PATH"
echo "GEOMETRY_USE_CAMERA_TOKENS=$GEOMETRY_USE_CAMERA_TOKENS"
echo "GEOMETRY_TARGET_CAMERA_MODE=$GEOMETRY_TARGET_CAMERA_MODE"
echo "GEOMETRY_SCENE_TOKEN_SOURCE=$GEOMETRY_SCENE_TOKEN_SOURCE"
echo "WRIST_FIRST_FRAME_INDEX=$WRIST_FIRST_FRAME_INDEX"

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

if [[ "$BUILD_CACHE" != "0" && "$BUILD_CACHE" != "1" ]]; then
  echo "[ERROR] BUILD_CACHE must be 0 or 1"
  exit 1
fi

if [[ "$FORCE_REBUILD_CACHE" != "0" && "$FORCE_REBUILD_CACHE" != "1" ]]; then
  echo "[ERROR] FORCE_REBUILD_CACHE must be 0 or 1"
  exit 1
fi

if ! [[ "$CACHE_NUM_SHARDS" =~ ^[0-9]+$ ]] || [[ "$CACHE_NUM_SHARDS" -lt 1 ]]; then
  echo "[ERROR] CACHE_NUM_SHARDS must be a positive integer"
  exit 1
fi

if [[ "$CACHE_SHARD_MODE" != "strided" && "$CACHE_SHARD_MODE" != "contiguous" ]]; then
  echo "[ERROR] CACHE_SHARD_MODE must be strided or contiguous"
  exit 1
fi

if ! [[ "$CACHE_NUM_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] CACHE_NUM_WORKERS must be a non-negative integer"
  exit 1
fi

if ! [[ "$CACHE_PREFETCH_FACTOR" =~ ^[0-9]+$ ]] || [[ "$CACHE_PREFETCH_FACTOR" -lt 1 ]]; then
  echo "[ERROR] CACHE_PREFETCH_FACTOR must be a positive integer"
  exit 1
fi

if [[ "$CACHE_PIN_MEMORY" != "0" && "$CACHE_PIN_MEMORY" != "1" ]]; then
  echo "[ERROR] CACHE_PIN_MEMORY must be 0 or 1"
  exit 1
fi

if [[ "$CACHE_VAE_TILED_ENCODE" != "0" && "$CACHE_VAE_TILED_ENCODE" != "1" ]]; then
  echo "[ERROR] CACHE_VAE_TILED_ENCODE must be 0 or 1"
  exit 1
fi

if [[ "$CACHE_SKIP_LEGACY_BRANCH" != "0" && "$CACHE_SKIP_LEGACY_BRANCH" != "1" ]]; then
  echo "[ERROR] CACHE_SKIP_LEGACY_BRANCH must be 0 or 1"
  exit 1
fi

if [[ "$RUN_TRAIN_AFTER_CACHE" != "0" && "$RUN_TRAIN_AFTER_CACHE" != "1" ]]; then
  echo "[ERROR] RUN_TRAIN_AFTER_CACHE must be 0 or 1"
  exit 1
fi

if [[ "$FORCE_REBUILD_CACHE" == "1" && "$BUILD_CACHE" != "1" ]]; then
  echo "[ERROR] FORCE_REBUILD_CACHE=1 requires BUILD_CACHE=1"
  exit 1
fi

if [[ ! -d "$DATASET_META_ROOT" ]]; then
  echo "[ERROR] dataset root not found: $DATASET_META_ROOT"
  exit 1
fi

if [[ ! -f "$TRAIN_MANIFEST" ]]; then
  echo "[ERROR] train manifest not found: $TRAIN_MANIFEST"
  exit 1
fi

if [[ ! -f "$VAL_MANIFEST" ]]; then
  echo "[ERROR] val manifest not found: $VAL_MANIFEST"
  exit 1
fi

if [[ ! -f "$STATE_STAT_PATH" ]]; then
  echo "[ERROR] state stat file not found: $STATE_STAT_PATH"
  exit 1
fi

if [[ ! -f "$NEG_PROMPT_EMB" ]]; then
  echo "[WARN] negative prompt embedding not found: $NEG_PROMPT_EMB"
fi

if ! check_manifest_refs "$TRAIN_MANIFEST"; then
  echo "[ERROR] train manifest references missing assets"
  exit 1
fi

if ! check_manifest_refs "$VAL_MANIFEST"; then
  echo "[ERROR] val manifest references missing assets"
  exit 1
fi

sanitize_thread_env
expected_train_rows="$(count_manifest_rows "$TRAIN_MANIFEST")"
expected_val_rows="$(count_manifest_rows "$VAL_MANIFEST")"
echo "[INFO] Expected cache rows: train=$expected_train_rows val=$expected_val_rows"

cache_ready=0
if [[ -d "$CACHE_ROOT/train" && -d "$CACHE_ROOT/val" && -f "$CACHE_ROOT/cache_config.json" && "$FORCE_REBUILD_CACHE" == "0" ]]; then
  actual_train="$(count_cache_files "$CACHE_ROOT/train")"
  actual_val="$(count_cache_files "$CACHE_ROOT/val")"
  if [[ "$actual_train" == "$expected_train_rows" && "$actual_val" == "$expected_val_rows" ]]; then
    cache_ready=1
    echo "[INFO] Reusing existing complete cache: $CACHE_ROOT"
  else
    echo "[WARN] Existing cache is incomplete: train=$actual_train/$expected_train_rows val=$actual_val/$expected_val_rows"
  fi
fi

if [[ "$FORCE_REBUILD_CACHE" == "1" ]]; then
  echo "[INFO] Removing existing cache root: $CACHE_ROOT"
  rm -rf "$CACHE_ROOT"
fi

if [[ "$cache_ready" == "0" ]]; then
  if [[ "$BUILD_CACHE" != "1" ]]; then
    echo "[ERROR] Cache is missing and BUILD_CACHE=0: $CACHE_ROOT"
    exit 1
  fi

  mkdir -p "$CACHE_ROOT"
  BUILD_CMD=(
    "$PYTHON_BIN"
    tool/build_cross_view_latent_cache.py
    --dataset_base_path "$DATASET_META_ROOT"
    --train_metadata_path "$TRAIN_MANIFEST"
    --val_metadata_path "$VAL_MANIFEST"
    --output_root "$CACHE_ROOT"
    --model_paths "$MODEL_DIR"
    --load_modules "$LOAD_MODULES"
    --state_type state_pose_7d
    --state_stat_path "$STATE_STAT_PATH"
    --height "$HEIGHT"
    --width "$WIDTH"
    --num_frames "$NUM_FRAMES"
    --num_history_frames "$NUM_HISTORY_FRAMES"
    --resize_mode "$RESIZE_MODE"
    --cross_view_source_views "$CROSS_VIEW_SOURCE_VIEWS"
    --cross_view_target_view "$CROSS_VIEW_TARGET_VIEW"
    --cross_view_placeholder_mode "$CROSS_VIEW_PLACEHOLDER_MODE"
    --device "$CACHE_DEVICE"
    --shard_mode "$CACHE_SHARD_MODE"
    --cache_num_workers "$CACHE_NUM_WORKERS"
    --cache_prefetch_factor "$CACHE_PREFETCH_FACTOR"
    --skip-existing
  )
  if [[ "$CACHE_PIN_MEMORY" == "1" ]]; then
    BUILD_CMD+=(--cache_pin_memory)
  fi
  if [[ "$CACHE_VAE_TILED_ENCODE" == "1" ]]; then
    BUILD_CMD+=(--vae_tiled_encode)
  fi
  if [[ "$CACHE_SKIP_LEGACY_BRANCH" == "1" ]]; then
    BUILD_CMD+=(--skip_legacy_branch)
  fi
  if [[ -n "$MODEL_ID_WITH_ORIGIN_PATHS" ]]; then
    BUILD_CMD+=(--model_id_with_origin_paths "$MODEL_ID_WITH_ORIGIN_PATHS")
  fi
  if [[ -n "$SCENE_TOKEN_CHECKPOINT" ]]; then
    BUILD_CMD+=(--scene_token_checkpoint "$SCENE_TOKEN_CHECKPOINT")
  fi
  if [[ -n "$WRIST_FIRST_FRAME_INDEX" ]]; then
    BUILD_CMD+=(--wrist_first_frame_index "$WRIST_FIRST_FRAME_INDEX")
  fi
  # Plan A dual-end anchor: pass dual-end flags to the cache builder so the
  # cached y channel encodes both head and tail synthesized frames. Without
  # these flags the builder defaults to head-only and the runtime training
  # path (which reads cache["y"] directly) will silently miss the tail anchor
  # signal regardless of CROSS_VIEW_USE_TAIL_ANCHOR being set on the trainer.
  BUILD_CMD+=(--cross_view_use_tail_anchor "$CROSS_VIEW_USE_TAIL_ANCHOR")
  BUILD_CMD+=(--num_tail_frames "$NUM_TAIL_FRAMES")
  BUILD_CMD+=(--cross_view_tail_anchor_dropout "$CROSS_VIEW_TAIL_ANCHOR_DROPOUT")

  if [[ "$CACHE_NUM_SHARDS" == "1" ]]; then
    (
      cd "$REPO_ROOT"
      run_cmd "${BUILD_CMD[@]}"
    )
  else
    pids=()
    for shard_index in $(seq 0 $((CACHE_NUM_SHARDS - 1))); do
      gpu_id="${CACHE_GPU_IDS[$((shard_index % ${#CACHE_GPU_IDS[@]}))]}"
      (
        cd "$REPO_ROOT"
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        run_cmd "${BUILD_CMD[@]}" --num_shards "$CACHE_NUM_SHARDS" --shard_index "$shard_index"
      ) &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
  fi

  actual_train="$(count_cache_files "$CACHE_ROOT/train")"
  actual_val="$(count_cache_files "$CACHE_ROOT/val")"
  echo "[INFO] Cache file counts: train=$actual_train val=$actual_val"
  if [[ "$actual_train" != "$expected_train_rows" ]]; then
    echo "[ERROR] cache train file count mismatch: expected $expected_train_rows got $actual_train"
    exit 1
  fi
  if [[ "$actual_val" != "$expected_val_rows" ]]; then
    echo "[ERROR] cache val file count mismatch: expected $expected_val_rows got $actual_val"
    exit 1
  fi
fi

if [[ ! -d "$CACHE_ROOT/train" || ! -f "$CACHE_ROOT/cache_config.json" ]]; then
  echo "[ERROR] Cache build did not produce a valid cache root: $CACHE_ROOT"
  exit 1
fi

if [[ "$RUN_TRAIN_AFTER_CACHE" != "1" ]]; then
  echo "[INFO] Cache is ready. RUN_TRAIN_AFTER_CACHE=0, skipping training."
  exit 0
fi

TRAIN_CMD=(
  env
  "PYTHON_BIN=$PYTHON_BIN"
  "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  "TAG=$TAG"
  "TASK=$TASK"
  "NUM_FRAMES=$NUM_FRAMES"
  "HEIGHT=$HEIGHT"
  "WIDTH=$WIDTH"
  "DATASET_REPEAT=$DATASET_REPEAT"
  "NUM_EPOCHS=$NUM_EPOCHS"
  "LEARNING_RATE=$LEARNING_RATE"
  "GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS"
  "MIXED_PRECISION=$MIXED_PRECISION"
  "MODEL_DIR=$MODEL_DIR"
  "DATASET_META_ROOT=$DATASET_META_ROOT"
  "TRAIN_MANIFEST=$TRAIN_MANIFEST"
  "VAL_MANIFEST=$VAL_MANIFEST"
  "STATE_STAT_PATH=$STATE_STAT_PATH"
  "NEG_PROMPT_EMB=$NEG_PROMPT_EMB"
  "CACHED_DATASET_PATH=$CACHE_ROOT"
  "OUTPUT_PATH=$OUTPUT_PATH"
  "LOAD_MODULES=$LOAD_MODULES"
  "CROSS_VIEW_SOURCE_VIEWS=$CROSS_VIEW_SOURCE_VIEWS"
  "CROSS_VIEW_TARGET_VIEW=$CROSS_VIEW_TARGET_VIEW"
  "CROSS_VIEW_PLACEHOLDER_MODE=$CROSS_VIEW_PLACEHOLDER_MODE"
  "CROSS_VIEW_SOURCE_LOSS_WEIGHT=$CROSS_VIEW_SOURCE_LOSS_WEIGHT"
  "CROSS_VIEW_OLD_BRANCH_DROPOUT=$CROSS_VIEW_OLD_BRANCH_DROPOUT"
  "CROSS_VIEW_PROJECTOR_HIDDEN_DIM=$CROSS_VIEW_PROJECTOR_HIDDEN_DIM"
  "CROSS_VIEW_SOURCE_INJECTION_MODE=$CROSS_VIEW_SOURCE_INJECTION_MODE"
  "CROSS_VIEW_SOURCE_BRANCH_MODE=$CROSS_VIEW_SOURCE_BRANCH_MODE"
  "CROSS_VIEW_SOURCE_WINDOW_RADIUS=$CROSS_VIEW_SOURCE_WINDOW_RADIUS"
  "CROSS_VIEW_SOURCE_GATE_MODE=$CROSS_VIEW_SOURCE_GATE_MODE"
  "CROSS_VIEW_TEMP_LOSS_WEIGHT=$CROSS_VIEW_TEMP_LOSS_WEIGHT"
  "CROSS_VIEW_STATE_LOSS_WEIGHT=$CROSS_VIEW_STATE_LOSS_WEIGHT"
  "CROSS_VIEW_GLOBAL_SOURCE_TOKENS=$CROSS_VIEW_GLOBAL_SOURCE_TOKENS"
  "CROSS_VIEW_AUX_LOSS_WARMUP_RATIO=$CROSS_VIEW_AUX_LOSS_WARMUP_RATIO"
  "CROSS_VIEW_OLD_BRANCH_DROPOUT_SCHEDULE=$CROSS_VIEW_OLD_BRANCH_DROPOUT_SCHEDULE"
  "CROSS_VIEW_LEGACY_BRANCH_SCHEDULE=$CROSS_VIEW_LEGACY_BRANCH_SCHEDULE"
  "CROSS_VIEW_DISABLE_LEGACY_IMAGE_BRANCH=$CROSS_VIEW_DISABLE_LEGACY_IMAGE_BRANCH"
  "CROSS_VIEW_USE_TAIL_ANCHOR=$CROSS_VIEW_USE_TAIL_ANCHOR"
  "NUM_TAIL_FRAMES=$NUM_TAIL_FRAMES"
  "CROSS_VIEW_TAIL_ANCHOR_DROPOUT=$CROSS_VIEW_TAIL_ANCHOR_DROPOUT"
  "SCENE_TOKEN_CHECKPOINT=$SCENE_TOKEN_CHECKPOINT"
  "SCENE_TOKEN_POOL_SIZE=$SCENE_TOKEN_POOL_SIZE"
  "GEOMETRY_GATE_MODE=$GEOMETRY_GATE_MODE"
  "GEOMETRY_SIDECAR_CACHE_PATH=$GEOMETRY_SIDECAR_CACHE_PATH"
  "GEOMETRY_USE_CAMERA_TOKENS=$GEOMETRY_USE_CAMERA_TOKENS"
  "GEOMETRY_TARGET_CAMERA_MODE=$GEOMETRY_TARGET_CAMERA_MODE"
  "GEOMETRY_SCENE_TOKEN_SOURCE=$GEOMETRY_SCENE_TOKEN_SOURCE"
  "ALIGNMENT_LOSS_WEIGHT=$ALIGNMENT_LOSS_WEIGHT"
  "ALIGNMENT_LOSS_WARMUP_RATIO=$ALIGNMENT_LOSS_WARMUP_RATIO"
  "WRIST_FIRST_FRAME_INDEX=$WRIST_FIRST_FRAME_INDEX"
  "USE_GRADIENT_CHECKPOINTING=$USE_GRADIENT_CHECKPOINTING"
  "TRAINABLE_MODELS=$TRAINABLE_MODELS"
  "CKPT_PATH=$CKPT_PATH"
  bash
  "$REPO_ROOT/bash/train_droid_crossview_small200.sh"
)

run_cmd "${TRAIN_CMD[@]}"
