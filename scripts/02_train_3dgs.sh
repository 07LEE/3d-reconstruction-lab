#!/bin/bash

# [Step 02] Inria 3D Gaussian Splatting Training
# Blackwell (RTX 50) optimization and high-density parameters

CONFIG_PATH="configs/default_config.sh"
if [ -f "$CONFIG_PATH" ]; then
    source "$CONFIG_PATH"
fi

CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate 3drc

# 1. Update SfM data link
echo "Updating SfM data links in ${DATA_DIR}/sparse/0..."
rm -rf "${DATA_DIR}/sparse/0"
mkdir -p "${DATA_DIR}/sparse/0"
cp -r "${HLOC_RECON}/sfm/models/0/"* "${DATA_DIR}/sparse/0/"

# 2. Training Execution
if [ "$TRAIN_METHOD" = "planar" ]; then
    echo "Starting Planar-GS (Planar Gaussian Splatting) Training..."
    python third_party/planar-gs/train.py \
        -s "$DATA_DIR" \
        --model_path "${OUTPUT_DIR}/gs_final_precision" \
        -r "$DOWNSAMPLE_RATE" \
        --data_device "$DATA_DEVICE" \
        --densify_grad_threshold "$DENSIFY_GRAD_THRESHOLD" \
        --planar_weight "$PLANAR_REG_WEIGHT"
else
    echo "Starting High-Density Original 3DGS Training..."
    python third_party/gaussian-splatting/train.py \
        -s "$DATA_DIR" \
        --model_path "${OUTPUT_DIR}/gs_final_precision" \
        -r "$DOWNSAMPLE_RATE" \
        --data_device "$DATA_DEVICE" \
        --densify_grad_threshold "$DENSIFY_GRAD_THRESHOLD"
fi

echo "Training (${TRAIN_METHOD}) Completed! Results saved in ${OUTPUT_DIR}/gs_final_precision"
