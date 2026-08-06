#!/bin/bash

# [Step 05] Gaussian Grouping Training
# Joint Reconstruction and Segmentation Lifted from 2D SAM

CONFIG_PATH="configs/default_config.sh"
if [ -f "$CONFIG_PATH" ]; then
    source "$CONFIG_PATH"
fi

CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate 3drc

PROJECT_ROOT=$(pwd)

# Move to gaussian-grouping directory
cd third_party/gaussian-grouping || exit 1

# Execute Training
echo "Starting Gaussian Grouping Training for dataset: ${GROUPING_DATASET}..."
bash script/train.sh "$GROUPING_DATASET" "$GROUPING_DOWNSAMPLE"

echo "Gaussian Grouping Training Completed!"
