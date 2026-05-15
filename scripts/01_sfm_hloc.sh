#!/bin/bash

# [Step 01] hloc SuperPoint+SuperGlue SfM Pipeline
# High-precision camera pose estimation for large-scale high-resolution datasets

CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate 3drc

echo "Starting High-Precision hloc SfM Pipeline..."

# Hardware optimization
export TORCH_CUDA_ARCH_LIST="12.0"

# Execution (Default: strategy sequential, overlap 100)
python src/hloc_pipeline.py \
    --image_dir data/images \
    --output_dir data/hloc_reconstruction \
    --strategy sequential \
    --overlap 100

echo "SfM Reconstruction Completed!"
