#!/usr/bin/env bash
# Multi-GPU stage2 inference launcher.
#
# Strategy:
#   - Spawn one process per GPU; each process owns sample indices i where
#     i % NUM_SHARDS == SHARD_INDEX. All processes write to the SAME
#     output dir (split-based filenames are unique by global idx).
#   - All shards run with --skip_metrics to avoid each shard recomputing
#     metrics on a partial set.
#   - After all shards finish, run ONE final aggregator pass with
#     num_shards=1 and skip_metrics=False. The aggregator skips already-
#     existing videos (idempotent) and computes metrics on the union.
#
# Usage:
#   bash bash/infer_stage2_multi_gpu.sh
#
# Env overrides (with defaults):
#   CKPT_PATH, CONFIG_JSON, OUTPUT_DIR, GEOMETRY_SIDECAR_CACHE_PATH,
#   DATASET_BASE_PATH, DATASET_METADATA_PATH, SAMPLE_LIMIT,
#   GPUS (e.g. "0,1,2,3,4,5,6,7"), CFG_SCALE, NUM_INFERENCE_STEPS,
#   PYTHON_BIN, SKIP_TRAIN_PREVIEW (1/0).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/xuehao/xh/projects/DiffSynth-Studio_v2}"
PYTHON_BIN="${PYTHON_BIN:-/env/conda/envs/studio/bin/python}"

CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/Ckpt/droid_success_high_quality_crossview_stage2_sidecar/epoch-0/epoch-0.safetensors}"
CONFIG_JSON="${CONFIG_JSON:-${REPO_ROOT}/Ckpt/droid_success_high_quality_crossview_stage2_sidecar/epoch-0/config.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/Ckpt/droid_success_high_quality_crossview_stage2_sidecar/epoch-0/stage2_eval_8gpu}"
DATASET_BASE_PATH="${DATASET_BASE_PATH:-/data_ywj/data_xh/projects/datasets/droid_success_high_quality_crossview_meta}"
DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-${DATASET_BASE_PATH}/meta/episodes_cross_view_val_81_small200.jsonl}"
GEOMETRY_SIDECAR_CACHE_PATH="${GEOMETRY_SIDECAR_CACHE_PATH:-${DATASET_BASE_PATH}/geometry_sidecar_lagernvs_strict_iter060000}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-2000}"
CFG_SCALE="${CFG_SCALE:-1.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SKIP_TRAIN_PREVIEW="${SKIP_TRAIN_PREVIEW:-1}"
# state_stat_path: training config may have it as null. Resolution order:
#   explicit env > <dataset_base>/meta/stat_state_pose_7d.json > unset (script will error early if dataset_base/meta is also missing)
STATE_STAT_PATH="${STATE_STAT_PATH:-${DATASET_BASE_PATH}/meta/stat_state_pose_7d.json}"
# wrist_first_frame_index: same shape of bug as state_stat_path. When the
# training cmd omitted --wrist_first_frame_index, the saved config.json has
# it as null and the wrist anchor frame silently degrades to a zero
# placeholder (visible as a gray first frame at inference). The cache itself
# may already contain VAE(synth) for target_history_latents, but the runtime
# rebuild path in cond_video[wrist, 0] still needs the index dict.
WRIST_FIRST_FRAME_INDEX="${WRIST_FIRST_FRAME_INDEX:-${DATASET_BASE_PATH}/meta/wrist_frame_index_all.json}"

INFER_SCRIPT="${REPO_ROOT}/examples/wanvideo/model_inference/infer_cross_view_stage2.py"
mkdir -p "${OUTPUT_DIR}"
LOG_DIR="${OUTPUT_DIR}/shard_logs"
mkdir -p "${LOG_DIR}"

# Parse GPU list -> array
IFS=',' read -ra GPU_ARR <<< "${GPUS}"
NUM_SHARDS=${#GPU_ARR[@]}
echo "[multi-gpu-infer] using ${NUM_SHARDS} shards on GPUs: ${GPUS}"
echo "[multi-gpu-infer] ckpt        : ${CKPT_PATH}"
echo "[multi-gpu-infer] config_json : ${CONFIG_JSON}"
echo "[multi-gpu-infer] output_dir  : ${OUTPUT_DIR}"
echo "[multi-gpu-infer] sidecar     : ${GEOMETRY_SIDECAR_CACHE_PATH}"
echo "[multi-gpu-infer] manifest    : ${DATASET_METADATA_PATH}"
echo "[multi-gpu-infer] sample_limit: ${SAMPLE_LIMIT}"
echo "[multi-gpu-infer] cfg_scale   : ${CFG_SCALE}"
echo "[multi-gpu-infer] steps       : ${NUM_INFERENCE_STEPS}"
echo "[multi-gpu-infer] state_stat  : ${STATE_STAT_PATH}"
echo "[multi-gpu-infer] wrist_index : ${WRIST_FIRST_FRAME_INDEX}"
echo

EXTRA_FLAGS=()
if [[ "${SKIP_TRAIN_PREVIEW}" == "1" ]]; then
  EXTRA_FLAGS+=(--skip_train_preview)
fi
if [[ -n "${STATE_STAT_PATH}" ]]; then
  EXTRA_FLAGS+=(--state_stat_path "${STATE_STAT_PATH}")
fi
if [[ -n "${WRIST_FIRST_FRAME_INDEX}" && -f "${WRIST_FIRST_FRAME_INDEX}" ]]; then
  EXTRA_FLAGS+=(--wrist_first_frame_index "${WRIST_FIRST_FRAME_INDEX}")
elif [[ -n "${WRIST_FIRST_FRAME_INDEX}" ]]; then
  echo "[multi-gpu-infer] WARN: WRIST_FIRST_FRAME_INDEX path does not exist: ${WRIST_FIRST_FRAME_INDEX}"
  echo "                  inference will fall back to zero placeholder for wrist frame 0."
fi
# Inference-time ablation: disable tail anchor (force num_tail_frames=0 in
# the y-channel encoder) even when the ckpt was trained dual-end. Set
# DISABLE_TAIL_ANCHOR_AT_INFERENCE=1 to enable.
if [[ "${DISABLE_TAIL_ANCHOR_AT_INFERENCE:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(--disable_tail_anchor_at_inference)
fi

PIDS=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_ARR[$shard]}"
  log_file="${LOG_DIR}/shard_${shard}_gpu_${gpu}.log"
  echo "[multi-gpu-infer] spawn shard=${shard}/${NUM_SHARDS} gpu=${gpu} -> ${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${INFER_SCRIPT}" \
    --ckpt_path "${CKPT_PATH}" \
    --config_json "${CONFIG_JSON}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --dataset_metadata_path "${DATASET_METADATA_PATH}" \
    --geometry_sidecar_cache_path "${GEOMETRY_SIDECAR_CACHE_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --cfg_scale "${CFG_SCALE}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --sample_limit "${SAMPLE_LIMIT}" \
    --num_shards "${NUM_SHARDS}" \
    --shard_index "${shard}" \
    --skip_metrics \
    "${EXTRA_FLAGS[@]}" \
    > "${log_file}" 2>&1 &
  PIDS+=($!)
done

echo "[multi-gpu-infer] waiting for ${#PIDS[@]} shards to finish..."
FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    echo "[multi-gpu-infer] shard pid=${pid} FAILED"
    FAILED=$((FAILED + 1))
  fi
done

if [[ "${FAILED}" -gt 0 ]]; then
  echo "[multi-gpu-infer] ${FAILED}/${NUM_SHARDS} shards failed; check logs in ${LOG_DIR}"
  exit 1
fi
echo "[multi-gpu-infer] all shards finished successfully."
echo

# Final aggregator pass: re-run with num_shards=1 to compute metrics on the
# union of generated videos. Existing videos are skipped (idempotent), so only
# metric computation actually runs.
echo "[multi-gpu-infer] running aggregator pass to compute metrics..."
AGG_GPU="${GPU_ARR[0]}"
CUDA_VISIBLE_DEVICES="${AGG_GPU}" "${PYTHON_BIN}" "${INFER_SCRIPT}" \
  --ckpt_path "${CKPT_PATH}" \
  --config_json "${CONFIG_JSON}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --geometry_sidecar_cache_path "${GEOMETRY_SIDECAR_CACHE_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --cfg_scale "${CFG_SCALE}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --sample_limit "${SAMPLE_LIMIT}" \
  --num_shards 1 \
  --shard_index 0 \
  "${EXTRA_FLAGS[@]}" \
  2>&1 | tee "${LOG_DIR}/aggregator.log"

echo
echo "[multi-gpu-infer] DONE. metrics  : ${OUTPUT_DIR}/metrics.json"
echo "                  comparisons : ${OUTPUT_DIR}/comparisons/val/"
