#!/usr/bin/env bash
set -euo pipefail

PYTHON="/data1/linzengrong/.conda/envs/vllm/bin/python"
MODEL="/data1/linzengrong/Models/Qwen3.5-27B"
# MODEL="/data2/Model/Qwen3-VL-30B-A3B-Instruct"
# API 中显示/调用的模型名
MODEL_NAME="qwen"

HOST="0.0.0.0"
PORT="5454"
MAX_MODEL_LEN="128000"
GPU_MEM_UTIL="0.8"
DEFAULT_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
ALLOWED_LOCAL_MEDIA_PATH="/data1/linzengrong/Code/DiffSynth-Studio/Ckpt/3_14_real_robot_history1/epoch-149/output_vla_lerobot/batch_20260314_201604/videos/chunk-000/comparison"
MEDIA_IO_KWARGS='{"video":{"num_frames":-1}}'
DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'


export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
# 自动根据 CUDA_VISIBLE_DEVICES 推断并行卡数（去掉空格后按逗号分割）
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES// /}"
IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
TP_SIZE="${#CUDA_DEVICE_ARRAY[@]}"

if [[ "$TP_SIZE" -lt 1 || -z "${CUDA_DEVICE_ARRAY[0]}" ]]; then
  echo "[ERROR] CUDA_VISIBLE_DEVICES 无效: '$CUDA_VISIBLE_DEVICES'" >&2
  exit 1
fi

"$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --tensor-parallel-size "$TP_SIZE" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --media-io-kwargs "$MEDIA_IO_KWARGS" \
  --default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS"
