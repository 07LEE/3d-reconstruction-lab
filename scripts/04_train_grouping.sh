#!/bin/bash
set -euo pipefail

# [Step 04] Gaussian Grouping Training
# Joint Reconstruction and Segmentation Lifted from 2D SAM

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
    conda activate gs_group
    set -u
    [ "${CONDA_DEFAULT_ENV:-}" = "gs_group" ] || { echo "[FATAL] Conda environment 'gs_group' not found or activation failed. Run: ./scripts/00_setup_environment.sh --env gs_group"; exit 1; }
fi

PROJECT_ROOT=$(pwd)

# Determine dataset source
TARGET_DATASET="${1:-$GROUPING_DATASET}"
SCENE_NAME=$(basename "$TARGET_DATASET")

# Resolve absolute dataset path
if [ -d "$TARGET_DATASET" ]; then
    DATASET_ABS_PATH="$(cd "$TARGET_DATASET" && pwd)"
else
    DATASET_ABS_PATH="$(pwd)/data/$TARGET_DATASET"
fi

# Move to gaussian-grouping directory
cd third_party/gaussian-grouping || exit 1

# Execute Training
echo "Starting Gaussian Grouping Training for dataset: ${SCENE_NAME}..."
ITER_PARAM="${GS_ITERATIONS:-30000}"
python train.py -s "$DATASET_ABS_PATH" -r "$GROUPING_DOWNSAMPLE" -m "output/${SCENE_NAME}" --config_file config/gaussian_dataset/train.json --iterations "$ITER_PARAM"

echo "Gaussian Grouping Training Completed!"
