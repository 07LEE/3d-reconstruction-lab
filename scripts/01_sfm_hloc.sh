#!/bin/bash
set -euo pipefail

# [Step 01] hloc SuperPoint+SuperGlue SfM Pipeline
# High-precision camera pose estimation for large-scale high-resolution datasets

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
    conda activate 3drc
    set -u
    [ "${CONDA_DEFAULT_ENV:-}" = "3drc" ] || { echo "[FATAL] Conda environment '3drc' activation failed!"; exit 1; }
fi

METHOD=${1:-$SFM_METHOD}

if [ "$METHOD" = "sfm" ]; then
    echo "Starting Traditional SIFT SfM Pipeline..."
    python -m src.sfm.sfm_pipeline \
        --image_dir "$IMAGE_DIR" \
        --output_dir data/sfm_reconstruction
elif [ "$METHOD" = "fastmap" ]; then
    echo "Starting Super-Fast FastMap GPU SfM Pipeline..."
    python -m src.sfm.fastmap_pipeline \
        --image_dir "$IMAGE_DIR" \
        --output_dir data/fastmap_reconstruction
elif [ "$METHOD" = "vggt" ]; then
    echo "Starting VGGT-Omega feed-forward pose estimation..."
    python third_party/vggt/demo_colmap.py \
        --scene_dir "$DATA_DIR" \
        --use_ba
elif [ "$METHOD" = "vi_sfm" ]; then
    echo "Starting Visual-Inertial (RGB + IMU) SfM Pipeline..."
    python -m src.sfm.vi_sfm_pipeline \
        --image_dir "$IMAGE_DIR" \
        --imu_path "$IMU_DATA_PATH" \
        --output_dir data/vi_sfm_reconstruction \
        --format "$IMU_FORMAT"
else
    echo "Starting High-Precision hloc SfM Pipeline..."
    python -m src.sfm.hloc_pipeline \
        --image_dir "$IMAGE_DIR" \
        --output_dir "$HLOC_RECON" \
        --strategy sequential \
        --overlap 100
fi

echo "SfM Reconstruction (${METHOD}) Completed! Models saved at ${DATA_DIR}/sparse/0"
