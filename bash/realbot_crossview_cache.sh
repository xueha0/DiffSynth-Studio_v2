#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

REALBOT_ROOT="${REALBOT_ROOT:-/data2/xuehao/datasets/realbot}"
DATASET_META_ROOT="${DATASET_META_ROOT:-$REALBOT_ROOT/realbot_crossview_81_pad_meta}"
LAGERNVS_ROOT="${LAGERNVS_ROOT:-$REALBOT_ROOT/realbot_lagernvs_state}"

env \
  "REPO_ROOT=$REPO_ROOT" \
  "PYTHON_BIN=${PYTHON_BIN:-/home/xuehao/.conda/envs/studio/bin/python}" \
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}" \
  "MODEL_DIR=${MODEL_DIR:-/home/xuehao/xh/projects/DiffSynth-Studio-old/models/PAI/Wan2.1-Fun-V1.1-1.3B-InP}" \
  "DATASET_META_ROOT=$DATASET_META_ROOT" \
  "TRAIN_MANIFEST=${TRAIN_MANIFEST:-$DATASET_META_ROOT/meta/episodes_cross_view_train_81_pad.jsonl}" \
  "VAL_MANIFEST=${VAL_MANIFEST:-$DATASET_META_ROOT/meta/episodes_cross_view_val_81_pad.jsonl}" \
  "STATE_STAT_PATH=${STATE_STAT_PATH:-$DATASET_META_ROOT/meta/stat_state_pose_7d.json}" \
  "NEG_PROMPT_EMB=${NEG_PROMPT_EMB:-$DATASET_META_ROOT/prompt_emb/neg_prompt.pt}" \
  "CACHE_ROOT=${CACHE_ROOT:-$DATASET_META_ROOT/cache_realbot_81f_180x320_tail_key3}" \
  "BUILD_CACHE=${BUILD_CACHE:-1}" \
  "FORCE_REBUILD_CACHE=${FORCE_REBUILD_CACHE:-0}" \
  "RUN_TRAIN_AFTER_CACHE=0" \
  "CACHE_NUM_SHARDS=${CACHE_NUM_SHARDS:-2}" \
  "CACHE_SHARD_MODE=${CACHE_SHARD_MODE:-contiguous}" \
  "CACHE_NUM_WORKERS=${CACHE_NUM_WORKERS:-1}" \
  "CACHE_PREFETCH_FACTOR=${CACHE_PREFETCH_FACTOR:-2}" \
  "CACHE_PIN_MEMORY=${CACHE_PIN_MEMORY:-0}" \
  "NUM_FRAMES=${NUM_FRAMES:-81}" \
  "HEIGHT=${HEIGHT:-480}" \
  "WIDTH=${WIDTH:-640}" \
  "RESIZE_MODE=${RESIZE_MODE:-crop}" \
  "LOAD_MODULES=${LOAD_MODULES:-dit,text:emb,vae,image,action:noise}" \
  "CROSS_VIEW_SOURCE_VIEWS=${CROSS_VIEW_SOURCE_VIEWS:-0,1}" \
  "CROSS_VIEW_TARGET_VIEW=${CROSS_VIEW_TARGET_VIEW:-2}" \
  "CROSS_VIEW_PLACEHOLDER_MODE=${CROSS_VIEW_PLACEHOLDER_MODE:-zeros}" \
  "WRIST_FIRST_FRAME_INDEX=${WRIST_FIRST_FRAME_INDEX:-$LAGERNVS_ROOT/meta/wrist_frame_index_all.json}" \
  "CROSS_VIEW_USE_TAIL_ANCHOR=${CROSS_VIEW_USE_TAIL_ANCHOR:-1}" \
  "NUM_TAIL_FRAMES=${NUM_TAIL_FRAMES:-1}" \
  "CROSS_VIEW_TAIL_ANCHOR_DROPOUT=${CROSS_VIEW_TAIL_ANCHOR_DROPOUT:-0.0}" \
  "CROSS_VIEW_USE_KEYFRAME_ANCHOR=${CROSS_VIEW_USE_KEYFRAME_ANCHOR:-1}" \
  "NUM_KEYFRAME_ANCHORS=${NUM_KEYFRAME_ANCHORS:-3}" \
  "KEYFRAME_ANCHOR_DROPOUT=${KEYFRAME_ANCHOR_DROPOUT:-0.0}" \
  "KEYFRAME_ANCHOR_MANIFEST_TRAIN=${KEYFRAME_ANCHOR_MANIFEST_TRAIN:-$LAGERNVS_ROOT/meta/realbot_lagernvs_keyframe_train_manifest.jsonl}" \
  "KEYFRAME_ANCHOR_MANIFEST_VAL=${KEYFRAME_ANCHOR_MANIFEST_VAL:-$LAGERNVS_ROOT/meta/realbot_lagernvs_keyframe_eval_manifest.jsonl}" \
  "KEYFRAME_ANCHOR_IMAGE_ROOT_TRAIN=${KEYFRAME_ANCHOR_IMAGE_ROOT_TRAIN:-$LAGERNVS_ROOT/keyframe_train/realbot_lagernvs_keyframe_train/realbot_lagernvs_keyframe_train/images_iter_035001}" \
  "KEYFRAME_ANCHOR_IMAGE_ROOT_VAL=${KEYFRAME_ANCHOR_IMAGE_ROOT_VAL:-$LAGERNVS_ROOT/keyframe_eval/realbot_lagernvs_keyframe_eval/realbot_lagernvs_keyframe_eval/images_iter_035001}" \
  bash "$REPO_ROOT/bash/train_droid_success_high_quality_crossview_cache.sh"
