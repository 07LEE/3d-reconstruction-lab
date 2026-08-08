#!/bin/bash
set -euo pipefail

# [Step 02b] PlanarGS Indoor Planar-Regularized 3DGS Pipeline
# Selective planar priors for indoor scene geometry optimization

CONFIG_PATH="configs/default_config.sh"
if [ -f "$CONFIG_PATH" ]; then
    source "$CONFIG_PATH"
else
    echo "[FATAL] Configuration file not found at $CONFIG_PATH!"
    exit 1
fi

CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    set +u
    conda activate gs_train
    set -u
    [ "${CONDA_DEFAULT_ENV:-}" = "gs_train" ] || { echo "[FATAL] Conda environment 'gs_train' activation failed!"; exit 1; }
fi

# Source CUDA build environment (GCC 12 / NVCC 12.8)
source "$(dirname "$0")/utils/setup_build_env.sh"

PROJECT_ROOT=$(pwd)
INPUT_DATASET="${1:-}"
if [ -z "$INPUT_DATASET" ]; then
    if [ -d "$PROJECT_ROOT/data/undistorted/sparse/0" ]; then
        INPUT_DATASET="data/undistorted"
    else
        INPUT_DATASET="$DATA_DIR"
    fi
fi

mkdir -p "$PROJECT_ROOT/${OUTPUT_DIR}/planargs"

# Execute PlanarGS Training
echo "Starting PlanarGS Training (Dataset: $INPUT_DATASET)..."
cd third_party/PlanarGS || exit 1

python train.py \
    -s "$PROJECT_ROOT/$INPUT_DATASET" \
    -m "$PROJECT_ROOT/${OUTPUT_DIR}/planargs" \
    -r "$DOWNSAMPLE_RATE" \
    --data_device "$DATA_DEVICE"

echo "PlanarGS Training Completed! Models saved at $PROJECT_ROOT/${OUTPUT_DIR}/planargs"
