#!/bin/bash
set -euo pipefail

# [Step 02] Inria 3D Gaussian Splatting Training
# Blackwell (RTX 50) optimization and high-density parameters

CONFIG_PATH="configs/default_config.sh"
if [ -f "$CONFIG_PATH" ]; then
    source "$CONFIG_PATH"
fi

CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
set +u
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate gs_train
set -u

# Run patch & environment verification inside activated gs_train environment
"$(dirname "$0")/utils/verify_patches.sh" || { echo "[FATAL] Patch verification failed!"; exit 1; }

INPUT_DATASET="${1:-$DATA_DIR}"

if [ -d "${INPUT_DATASET}/sparse/0" ] || [ -d "${INPUT_DATASET}/sparse" ]; then
    echo "Using existing pre-computed dataset structure at: ${INPUT_DATASET}"
    TARGET_DATA_DIR="${INPUT_DATASET}"
else
    # Guard: Ensure SfM reconstruction model exists before cleaning/updating
    SRC="${HLOC_RECON}/sfm/models/0"
    if [ ! -f "$SRC/cameras.bin" ] && [ ! -f "$SRC/cameras.txt" ] && [ ! -f "data/vi_sfm_reconstruction/sparse/0/cameras.bin" ]; then
        echo "[FATAL] SfM model not found: $SRC"
        exit 1
    fi

    # Update SfM data link after verification succeeds
    echo "Updating SfM data links in ${DATA_DIR}/sparse/0..."
    rm -rf "${DATA_DIR}/sparse/0"
    mkdir -p "${DATA_DIR}/sparse/0"

    if [ -f "${HLOC_RECON}/sfm/models/0/cameras.bin" ] || [ -f "${HLOC_RECON}/sfm/models/0/cameras.txt" ]; then
        cp -r "${HLOC_RECON}/sfm/models/0/"* "${DATA_DIR}/sparse/0/"
    elif [ -f "${HLOC_RECON}/sfm/cameras.bin" ] || [ -f "${HLOC_RECON}/sfm/cameras.txt" ]; then
        cp -r "${HLOC_RECON}/sfm/"*.bin "${DATA_DIR}/sparse/0/" 2>/dev/null || true
        cp -r "${HLOC_RECON}/sfm/"*.txt "${DATA_DIR}/sparse/0/" 2>/dev/null || true
    elif [ -f "data/vi_sfm_reconstruction/sparse/0/cameras.bin" ]; then
        cp -r "data/vi_sfm_reconstruction/sparse/0/"* "${DATA_DIR}/sparse/0/"
    fi

    # Ensure images link exists in DATA_DIR
    if [ ! -d "${DATA_DIR}/images" ]; then
        echo "Creating images link in ${DATA_DIR}/images..."
        ln -s "$(pwd)/${IMAGE_DIR}" "${DATA_DIR}/images" 2>/dev/null || cp -r "${IMAGE_DIR}" "${DATA_DIR}/images"
    fi
    TARGET_DATA_DIR="${DATA_DIR}"
fi

SCENE_NAME=$(basename "$TARGET_DATA_DIR")
MODEL_OUTPUT="${OUTPUT_DIR}/${SCENE_NAME}/3dgs/inria_30k"
mkdir -p "$MODEL_OUTPUT"

# 2. Training Execution
echo "Starting High-Density Original 3DGS Training (Scene: $SCENE_NAME)..."
python third_party/gaussian-splatting/train.py \
    -s "$TARGET_DATA_DIR" \
    --model_path "$MODEL_OUTPUT" \
    -r "$DOWNSAMPLE_RATE" \
    --densify_grad_threshold "$DENSIFY_GRAD_THRESHOLD" \
    --data_device "$DATA_DEVICE"

echo "3DGS Training Completed! Results saved to $MODEL_OUTPUT"
