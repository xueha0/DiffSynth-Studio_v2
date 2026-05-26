#!/bin/bash

# ====================== 请修改这里的配置 ======================
# 所有 epoch 根目录（epoch-xx 文件夹的上一级）
BASE_CKPT_DIR="/home/xuehao/xh/projects/DiffSynth-Studio/Ckpt/droid_crossview_10000_stage1"

# GPU 编号
CUDA_DEVICE="7"

# 数据集固定路径
DATASET_BASE_PATH="/data1/xuehao/datasets/droid_1.0.1_crossview_meta"
DATASET_METADATA_PATH="/data1/xuehao/datasets/droid_1.0.1_crossview_meta/meta/episodes_cross_view_val_81.jsonl"
TRAIN_METADATA_PATH="/data1/xuehao/datasets/droid_1.0.1_crossview_meta/meta/episodes_cross_view_train_81_10000.jsonl"

# 推理脚本路径
INFER_SCRIPT="/home/xuehao/xh/projects/DiffSynth-Studio/examples/wanvideo/model_inference/infer_cross_view_stage1.py"
# ==============================================================

# 遍历所有 epoch-xx 文件夹
for EPOCH_DIR in "${BASE_CKPT_DIR}"/epoch-*; do
    # 提取 epoch 名称
    EPOCH_NAME=$(basename "${EPOCH_DIR}")
    EPOCH_NUM=${EPOCH_NAME#epoch-}

    echo -e "\n=================================================="
    echo "正在评估 Stage1：${EPOCH_NAME}"
    echo "目录：${EPOCH_DIR}"
    echo "=================================================="

    # 输出目录（每个 epoch 独立文件夹）
    OUTPUT_DIR="${BASE_CKPT_DIR}/${EPOCH_NAME}/stage1_eval"

    # 模型文件 & 配置文件
    CKPT_PATH="${EPOCH_DIR}/${EPOCH_NAME}.safetensors"
    CONFIG_JSON="${BASE_CKPT_DIR}/config.json"  # 共用根目录 config

    # 文件检查
    if [[ ! -f "${CKPT_PATH}" ]]; then
        echo "⚠️  缺失模型文件，跳过"
        continue
    fi
    if [[ ! -f "${CONFIG_JSON}" ]]; then
        echo "⚠️  缺失配置文件，跳过"
        continue
    fi

    # 执行评估命令
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    python "${INFER_SCRIPT}" \
      --ckpt_path "${CKPT_PATH}" \
      --config_json "${CONFIG_JSON}" \
      --dataset_base_path "${DATASET_BASE_PATH}" \
      --dataset_metadata_path "${DATASET_METADATA_PATH}" \
      --train_metadata_path "${TRAIN_METADATA_PATH}" \
      --output_dir "${OUTPUT_DIR}"

    echo -e "✅ ${EPOCH_NAME} Stage1 评估完成\n"
done

echo -e "\n🎉 所有 epoch Stage1 评估全部完成！"