#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TAG="${TAG:-3_14_real_robot_history1}"
EPOCH="${EPOCH:-149}"


CKPT_PATH="${CKPT_PATH:-Ckpt/${TAG}/epoch-${EPOCH}/epoch-${EPOCH}.safetensors}"
CONFIG_JSON="${CONFIG_JSON:-Ckpt/${TAG}/epoch-${EPOCH}/config.json}"

DATASET_DIR="${DATASET_DIR:-/data1/linzengrong/Code/DiffSynth-Studio/robot_data/piper}"
DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-$DATASET_DIR/meta/episodes_val.jsonl}"
ACTION_STAT_PATH="${ACTION_STAT_PATH:-$DATASET_DIR/meta/stat.json}"

MODEL_PATHS="${MODEL_PATHS:-/data1/linzengrong/Models/wan2.1/Wan2.1-Fun-V1.1-1.3B-InP}"

VLA_HOST="${VLA_HOST:-100.64.147.46}"
VLA_PORT="${VLA_PORT:-6666}"
ROBOT_UID="${ROBOT_UID:-piper_real}"

FPS="${FPS:-15}"
NEGATIVE_PROMPT_EMB="${NEGATIVE_PROMPT_EMB:-prompt_emb/neg_prompt.pt}"
PYTHON_BIN="${PYTHON_BIN:-/data1/linzengrong/.conda/envs/diff/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/linzengrong/Code/DiffSynth-Studio/Ckpt/${TAG}/epoch-${EPOCH}/output_vla_lerobot}"

PROCESS_ALL_EPISODES="${PROCESS_ALL_EPISODES:-1}"
REPEAT_PER_EPISODE="${REPEAT_PER_EPISODE:-1}"
START_JITTER="${START_JITTER:-5}"
MAX_LENGTH_MULTIPLIER="${MAX_LENGTH_MULTIPLIER:-1.3}"
CHUNK_SIZE="${CHUNK_SIZE:-1000}"
FOLLOW_METADATA="${FOLLOW_METADATA:-1}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-1}"
IDLE_TIMEOUT_SEC="${IDLE_TIMEOUT_SEC:-300}"
INSTANCES_PER_GPU="${INSTANCES_PER_GPU:-2}"
LAUNCH_STAGGER_SEC="${LAUNCH_STAGGER_SEC:-3}"
STRICT_LEROBOT_V21="${STRICT_LEROBOT_V21:-1}"


GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-2}}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if [[ ${#GPU_LIST[@]} -lt 1 ]]; then
  echo "[ERROR] No GPUs specified. Set GPUS or CUDA_VISIBLE_DEVICES."
  exit 1
fi
for i in "${!GPU_LIST[@]}"; do
  GPU_LIST[$i]="$(echo "${GPU_LIST[$i]}" | xargs)"
  if [[ -z "${GPU_LIST[$i]}" ]]; then
    echo "[ERROR] Empty GPU id at position $i in GPUS='$GPUS'"
    exit 1
  fi
done

if [[ "$INSTANCES_PER_GPU" -lt 1 ]]; then
  echo "[ERROR] INSTANCES_PER_GPU must be >= 1."
  exit 1
fi
if [[ "$LAUNCH_STAGGER_SEC" -lt 0 ]]; then
  echo "[ERROR] LAUNCH_STAGGER_SEC must be >= 0."
  exit 1
fi

GPU_COUNT="${#GPU_LIST[@]}"
if [[ -n "${NUM_WORKERS:-}" ]]; then
  echo "[WARN] NUM_WORKERS is set explicitly ($NUM_WORKERS), overriding INSTANCES_PER_GPU=$INSTANCES_PER_GPU."
else
  NUM_WORKERS="$((GPU_COUNT * INSTANCES_PER_GPU))"
fi

if [[ "$NUM_WORKERS" -lt 1 ]]; then
  echo "[ERROR] NUM_WORKERS must be >= 1."
  exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

COMMON_CMD=(
  "$PYTHON_BIN" examples/wanvideo/model_inference/infer_vla_wm_closed_loop.py
  --tag "$TAG"
  --epoch "$EPOCH"
  --ckpt_path "$CKPT_PATH"
  --config_json "$CONFIG_JSON"
  --model_paths "$MODEL_PATHS"
  --dataset_base_path "$DATASET_DIR"
  --dataset_metadata_path "$DATASET_METADATA_PATH"
  --action_stat_path "$ACTION_STAT_PATH"
  --vla_host "$VLA_HOST"
  --vla_port "$VLA_PORT"
  --robot_uid "$ROBOT_UID"
  --fps "$FPS"
  --negative_prompt_emb "$NEGATIVE_PROMPT_EMB"
  --output_dir "$OUTPUT_DIR"
  --process_all_episodes "$PROCESS_ALL_EPISODES"
  --repeat_per_episode "$REPEAT_PER_EPISODE"
  --start_jitter "$START_JITTER"
  --max_length_multiplier "$MAX_LENGTH_MULTIPLIER"
  --chunk_size "$CHUNK_SIZE"
  --run_id "$RUN_ID"
  --follow_metadata "$FOLLOW_METADATA"
  --poll_interval_sec "$POLL_INTERVAL_SEC"
  --idle_timeout_sec "$IDLE_TIMEOUT_SEC"
  --strict_lerobot_v21 "$STRICT_LEROBOT_V21"
)

echo "Run ID: $RUN_ID"
echo "GPUs: ${GPU_LIST[*]}"
echo "Workers: $NUM_WORKERS"
echo "Instances per GPU (requested): $INSTANCES_PER_GPU"
echo "Launch stagger seconds: $LAUNCH_STAGGER_SEC"
echo "[launch] multi-worker mode"

PIDS=()
for ((i=0; i<NUM_WORKERS; i++)); do
  gpu_idx="$((i % GPU_COUNT))"
  gpu="${GPU_LIST[$gpu_idx]}"
  slot="$((i / GPU_COUNT + 1))"
  CMD=("${COMMON_CMD[@]}" --worker_id "$i" --num_workers "$NUM_WORKERS")
  echo "[launch] worker=$i gpu=$gpu slot=$slot"
  CUDA_VISIBLE_DEVICES="$gpu" "${CMD[@]}" &
  PIDS+=("$!")
  if [[ "$LAUNCH_STAGGER_SEC" -gt 0 && "$i" -lt $((NUM_WORKERS - 1)) ]]; then
    sleep "$LAUNCH_STAGGER_SEC"
  fi
done

FAIL=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  if ! wait "$pid"; then
    echo "[error] worker $i failed (pid=$pid)"
    FAIL=1
  fi
done
if [[ "$FAIL" -ne 0 ]]; then
  exit 1
fi

echo "[done] all workers finished. output under: $OUTPUT_DIR/batch_${RUN_ID}"
