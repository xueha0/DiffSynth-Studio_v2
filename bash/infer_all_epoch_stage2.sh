#!/bin/bash

# ====================== 请修改这里的配置 ======================
# 你的所有 epoch 根目录（所有 epoch-xx 文件夹的上一级）
BASE_CKPT_DIR="/home/xuehao/xh/projects/DiffSynth-Studio/Ckpt/droid_crossview_stage2_hybrid"

# GPU 编号
CUDA_DEVICE="7"

# 推理参数
CFG_SCALE="1.0"
NUM_INFERENCE_STEPS="50"

# 数据集路径（固定不变）
DATASET_METADATA_PATH="/data1/xuehao/datasets/droid_1.0.1_crossview_meta/meta/episodes_cross_view_val_81_small50.jsonl"
TRAIN_METADATA_PATH="/data1/xuehao/datasets/droid_1.0.1_crossview_meta/meta/episodes_cross_view_train_81_small200.jsonl"
# ==============================================================

# 遍历所有 epoch-xx 文件夹
for EPOCH_DIR in "${BASE_CKPT_DIR}"/epoch-*; do
    # 提取 epoch 编号（如 epoch-9 → 9）
    EPOCH_NAME=$(basename "${EPOCH_DIR}")
    EPOCH_NUM=${EPOCH_NAME#epoch-}

    echo -e "\n=================================================="
    echo "正在评估：${EPOCH_NAME}"
    echo "目录：${EPOCH_DIR}"
    echo "=================================================="

    # 输出目录（每个 epoch 单独一个文件夹）
    OUTPUT_DIR="${BASE_CKPT_DIR}/${EPOCH_NAME}/stage2_eval"

    # 模型与配置路径
    CKPT_PATH="${EPOCH_DIR}/${EPOCH_NAME}.safetensors"
    CONFIG_JSON="${EPOCH_DIR}/config.json"

    # 检查必要文件是否存在
    if [[ ! -f "${CKPT_PATH}" ]]; then
        echo "⚠️  缺失模型文件：${CKPT_PATH}，跳过"
        continue
    fi
    if [[ ! -f "${CONFIG_JSON}" ]]; then
        echo "⚠️  缺失配置文件：${CONFIG_JSON}，跳过"
        continue
    fi

    # 执行评估命令
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    python examples/wanvideo/model_inference/infer_cross_view_stage2.py \
      --dataset_metadata_path "${DATASET_METADATA_PATH}" \
      --output_dir "${OUTPUT_DIR}" \
      --cfg_scale "${CFG_SCALE}" \
      --num_inference_steps "${NUM_INFERENCE_STEPS}" \
      --ckpt_path "${CKPT_PATH}" \
      --config_json "${CONFIG_JSON}" \
      --train_metadata_path "${TRAIN_METADATA_PATH}"

    echo -e "✅ ${EPOCH_NAME} 评估完成\n"
done

echo -e "\n🎉 所有 epoch 评估全部完成！"