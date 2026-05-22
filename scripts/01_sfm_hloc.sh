#!/bin/bash

# [Step 01] hloc SuperPoint+SuperGlue SfM Pipeline
# High-precision camera pose estimation for large-scale high-resolution datasets

CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate 3drc

METHOD=${1:-hloc}

# Hardware optimization
export TORCH_CUDA_ARCH_LIST="12.0"

if [ "$METHOD" = "sfm" ]; then
    echo "Starting Traditional SIFT SfM Pipeline..."
    python scripts/sfm_pipeline.py \
        --image_dir data/images \
        --output_dir data/sfm_reconstruction
elif [ "$METHOD" = "fastmap" ]; then
    echo "Starting Super-Fast FastMap GPU SfM Pipeline..."
    python scripts/fastmap_pipeline.py \
        --image_dir data/images \
        --output_dir data/fastmap_reconstruction
else
    echo "Starting High-Precision hloc SfM Pipeline..."
    python scripts/hloc_pipeline.py \
        --image_dir data/images \
        --output_dir data/hloc_reconstruction \
        --strategy sequential \
        --overlap 100
fi

echo "SfM Reconstruction (${METHOD}) Completed!"
