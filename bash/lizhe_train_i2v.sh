#!/bin/bash

TAG="temp"  # 修改3_17_lizhe_train_i2v
LOAD_MODULES="dit,text:emb,vae,image,action:noise"  #
export CUDA_VISIBLE_DEVICES="0"  # 修改
NUM_FRAMES=81

CKPT_PATH="Ckpt/3_9_15sets_21frame_clip/epoch-19/epoch-19.safetensors"  #从这个权重开始训练
TRAIN_OUTPUT_PATH="Ckpt/${TAG}"

MODEL_DIR="/data1/blm/Baseline-Pi0/DiffSynth-Studio/models/PAI/Wan2.1-Fun-V1.1-1.3B-InP"  #预训练模型
DATASET_DIR="/data2/linzengrong/Dataset/Cobot_Magic_all_extracted/"  #数据路径

/data/linzengrong/.conda/envs/diff/bin/python -m accelerate.commands.launch examples/wanvideo/model_training/train.py \
  --dataset_base_path "$DATASET_DIR" \
  --use_gradient_checkpointing \
  --dataset_metadata_path "$DATASET_DIR/episodes_clipped_state_pose_train.jsonl" \   #数据总jsonl
  --action_stat_path "$DATASET_DIR/stat.json" \   #action统计信息
  --action_type "state_pose" \
  --height 240 \
  --width 320 \
  --num_frames "$NUM_FRAMES" \
  --dataset_repeat 10 \
  --model_paths "$MODEL_DIR" \
  --learning_rate 5e-5 \
  --num_epochs 3 \
  --output_path "$TRAIN_OUTPUT_PATH" \
  --gradient_accumulation_steps 8 \
  --use_swanlab 0 \
  --swanlab_experiment_name "$TAG" \
  --load_modules "$LOAD_MODULES" \
  --ckpt_path "$CKPT_PATH"
