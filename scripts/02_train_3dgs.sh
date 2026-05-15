#!/bin/bash

# [Step 02] Inria 3D Gaussian Splatting Training
# Blackwell (RTX 50) optimization and high-density parameters

CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate gs_original

# 1. Hardware Optimization (Blackwell Architecture)
export TORCH_CUDA_ARCH_LIST="12.0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. Update SfM data link
echo "Updating SfM data links in nerfstudio_data/sparse/0..."
rm -rf data/nerfstudio_data/sparse/0
mkdir -p data/nerfstudio_data/sparse/0
cp -r data/hloc_reconstruction/sfm/models/0/* data/nerfstudio_data/sparse/0/

# 3. Training Execution
echo "Starting High-Density 3DGS Training..."
python third_party/gaussian-splatting/train.py \
    -s data/nerfstudio_data \
    --model_path outputs/gs_final_precision \
    -r 2 \
    --data_device cpu \
    --densify_grad_threshold 0.0002

echo "3DGS Training Completed! Results saved in outputs/gs_final_precision"
