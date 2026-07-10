#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REALBOT_META_ROOT="${REALBOT_META_ROOT:-/data2/xuehao/datasets/realbot/realbot_crossview_81_pad_meta}"
LAGERNVS_ROOT="${LAGERNVS_ROOT:-/data2/xuehao/datasets/realbot/realbot_lagernvs_state}"
EPOCH_DIR="${EPOCH_DIR:-$REPO_ROOT/Ckpt/realbot_stage2_tail_key3_state_no3d_480x640/epoch-39}"

# env \
#   "REPO_ROOT=$REPO_ROOT" \
#   "PYTHON_BIN=${PYTHON_BIN:-/home/xuehao/.conda/envs/studio/bin/python}" \
#   "GPUS=${GPUS:-4,5,6,7}" \
#   "CKPT_PATH=${CKPT_PATH:-$EPOCH_DIR/epoch-39.safetensors}" \
#   "CONFIG_JSON=${CONFIG_JSON:-$EPOCH_DIR/config.json}" \
#   "OUTPUT_DIR=${OUTPUT_DIR:-$EPOCH_DIR/stage2_eval_realbot}" \
#   "DATASET_BASE_PATH=$REALBOT_META_ROOT" \
#   "DATASET_METADATA_PATH=${DATASET_METADATA_PATH:-$REALBOT_META_ROOT/meta/episodes_cross_view_val_81_pad.jsonl}" \
#   "STATE_STAT_PATH=${STATE_STAT_PATH:-$REALBOT_META_ROOT/meta/stat_state_pose_7d.json}" \
#   "WRIST_FIRST_FRAME_INDEX=${WRIST_FIRST_FRAME_INDEX:-$LAGERNVS_ROOT/meta/wrist_frame_index_all.json}" \
#   "GEOMETRY_SIDECAR_CACHE_PATH=${GEOMETRY_SIDECAR_CACHE_PATH-}" \
#   "SAMPLE_LIMIT=${SAMPLE_LIMIT:-628}" \
#   "CFG_SCALE=${CFG_SCALE:-1.0}" \
#   "NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}" \
#   "SKIP_TRAIN_PREVIEW=${SKIP_TRAIN_PREVIEW:-1}" \
#   "SAVE_WRIST_ONLY=${SAVE_WRIST_ONLY:-0}" \
#   bash "$REPO_ROOT/bash/infer_stage2_multi_gpu.sh"


env \
  "REPO_ROOT=$REPO_ROOT" \
  "PYTHON_BIN=${PYTHON_BIN:-/home/xuehao/.conda/envs/studio/bin/python}" \
  "GPUS=${GPUS:-4,5,6,7}" \
  "CKPT_PATH=${CKPT_PATH:-$EPOCH_DIR/epoch-39.safetensors}" \
  "CONFIG_JSON=${CONFIG_JSON:-$EPOCH_DIR/config.json}" \
  "OUTPUT_DIR=${OUTPUT_DIR:-$EPOCH_DIR/stage2_eval_realbot_onlypred}" \
  "DATASET_BASE_PATH=$REALBOT_META_ROOT" \
  "DATASET_METADATA_PATH=${DATASET_METADATA_PATH:-$REALBOT_META_ROOT/meta/episodes_cross_view_val_81_pad.jsonl}" \
  "STATE_STAT_PATH=${STATE_STAT_PATH:-$REALBOT_META_ROOT/meta/stat_state_pose_7d.json}" \
  "WRIST_FIRST_FRAME_INDEX=${WRIST_FIRST_FRAME_INDEX:-$LAGERNVS_ROOT/meta/wrist_frame_index_all.json}" \
  "GEOMETRY_SIDECAR_CACHE_PATH=${GEOMETRY_SIDECAR_CACHE_PATH-}" \
  "SAMPLE_LIMIT=${SAMPLE_LIMIT:-628}" \
  "CFG_SCALE=${CFG_SCALE:-1.0}" \
  "NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}" \
  "SKIP_TRAIN_PREVIEW=${SKIP_TRAIN_PREVIEW:-1}" \
  "SAVE_WRIST_ONLY=${SAVE_WRIST_ONLY:-1}" \
  bash "$REPO_ROOT/bash/infer_stage2_multi_gpu.sh"
