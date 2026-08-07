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
    [ "${CONDA_DEFAULT_ENV:-}" = "gs_group" ] || { echo "[FATAL] Conda environment 'gs_group' activation failed!"; exit 1; }
fi

PROJECT_ROOT=$(pwd)

# Move to gaussian-grouping directory
cd third_party/gaussian-grouping || exit 1

# Execute Training
echo "Starting Gaussian Grouping Training for dataset: ${GROUPING_DATASET}..."
bash script/train.sh "$GROUPING_DATASET" "$GROUPING_DOWNSAMPLE"

echo "Gaussian Grouping Training Completed!"
