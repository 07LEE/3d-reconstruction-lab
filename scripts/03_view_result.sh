#!/bin/bash

# [Step 03] Visualizing 3D Gaussian Splatting Results
# View PLY output using the 3D viewer

CONFIG_PATH="configs/default_config.sh"
if [ -f "$CONFIG_PATH" ]; then
    source "$CONFIG_PATH"
fi

CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate 3drc

DEFAULT_PATH="${OUTPUT_DIR}/gs_final_precision/point_cloud/iteration_30000/point_cloud.ply"

# Use provided path argument or default path
PLY_PATH=${1:-$DEFAULT_PATH}

if [ ! -f "$PLY_PATH" ]; then
    echo "Error: File not found at $PLY_PATH"
    exit 1
fi

echo "Launching 3D Viewer for: $PLY_PATH"
python third_party/gaussian-splatting/view_ply.py --path "$PLY_PATH"
