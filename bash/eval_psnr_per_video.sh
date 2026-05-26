#!/bin/bash
set -euo pipefail

REPO_ROOT="/data1/linzengrong/Code/DiffSynth-Studio"
PYTHON_BIN="/data1/linzengrong/.conda/envs/diff/bin/python"

# 用法:
#   bash bash/eval_psnr_per_video.sh <推理输出目录> [workers]
# 例子:
#   bash bash/eval_psnr_per_video.sh \
#     ckpt/Ckpt/ckpt/2月3日_1w条_state_pose/epoch-19/epoch-19/03M09D_10H47Min 8

OUTPUT_DIR="${1:-$REPO_ROOT/ckpt/Ckpt/ckpt/2月3日_1w条_state_pose/epoch-19/epoch-19/03M09D_10H47Min}"
WORKERS="${2:-8}"
OUTPUT_JSONL="${OUTPUT_DIR}/psnr_per_video.jsonl"

cd "$REPO_ROOT"

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "[ERROR] output dir not found: $OUTPUT_DIR"
  exit 1
fi

echo "[INFO] output dir: $OUTPUT_DIR"
echo "[INFO] output jsonl: $OUTPUT_JSONL"
echo "[INFO] workers: $WORKERS"

"$PYTHON_BIN" tool/eval_psnr_per_video.py \
  --comparison-dir "$OUTPUT_DIR" \
  --output-jsonl "$OUTPUT_JSONL" \
  --workers "$WORKERS"

echo "[DONE] PSNR results saved to: $OUTPUT_JSONL"
