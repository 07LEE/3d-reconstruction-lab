#!/bin/bash

# [Step 01] hloc SuperPoint+SuperGlue SfM Pipeline
# High-precision camera pose estimation for large-scale high-resolution datasets

CONFIG_PATH="configs/default_config.sh"
if [ -f "$CONFIG_PATH" ]; then
    source "$CONFIG_PATH"
fi

CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate 3drc

METHOD=${1:-$SFM_METHOD}

if [ "$METHOD" = "sfm" ]; then
    echo "Starting Traditional SIFT SfM Pipeline..."
    python scripts/sfm_pipeline.py \
        --image_dir "$IMAGE_DIR" \
        --output_dir data/sfm_reconstruction
elif [ "$METHOD" = "fastmap" ]; then
    echo "Starting Super-Fast FastMap GPU SfM Pipeline..."
    python scripts/fastmap_pipeline.py \
        --image_dir "$IMAGE_DIR" \
        --output_dir data/fastmap_reconstruction
elif [ "$METHOD" = "vggt" ]; then
    echo "Starting VGGT-Omega feed-forward pose estimation..."
    python third_party/vggt/demo_colmap.py \
        --scene_dir "$DATA_DIR" \
        --use_ba
else
    echo "Starting High-Precision hloc SfM Pipeline..."
    python scripts/hloc_pipeline.py \
        --image_dir "$IMAGE_DIR" \
        --output_dir "$HLOC_RECON" \
        --strategy sequential \
        --overlap 100
fi

echo "SfM Reconstruction (${METHOD}) Completed!"
