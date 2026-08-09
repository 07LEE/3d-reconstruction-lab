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
INPUT_DATASET="${2:-$DATA_DIR}"

IMAGE_DIR_TARGET="${INPUT_DATASET}/raw_images"
if [ ! -d "$IMAGE_DIR_TARGET" ]; then
    IMAGE_DIR_TARGET="$IMAGE_DIR"
fi

SPARSE_BASE_DIR="${INPUT_DATASET}/sparse"
SPARSE_METHOD_DIR="${SPARSE_BASE_DIR}/${METHOD}"
mkdir -p "$SPARSE_METHOD_DIR"

if [ "$METHOD" = "sfm" ]; then
    echo "Starting Traditional SIFT SfM Pipeline..."
    python -m src.sfm.sfm_pipeline \
        --image_dir "$IMAGE_DIR_TARGET" \
        --output_dir "${INPUT_DATASET}"
elif [ "$METHOD" = "fastmap" ]; then
    echo "Starting Super-Fast FastMap GPU SfM Pipeline..."
    python -m src.sfm.fastmap_pipeline \
        --image_dir "$IMAGE_DIR_TARGET" \
        --output_dir "${INPUT_DATASET}"
elif [ "$METHOD" = "vggt" ]; then
    echo "Starting VGGT-Omega feed-forward pose estimation..."
    if [ ! -d "${INPUT_DATASET}/images" ] && [ -d "${INPUT_DATASET}/raw_images" ]; then
        ln -s raw_images "${INPUT_DATASET}/images"
    fi
    python third_party/vggt/demo_colmap.py \
        --scene_dir "$INPUT_DATASET"
elif [ "$METHOD" = "vi_sfm" ]; then
    echo "Starting Visual-Inertial (RGB + IMU) SfM Pipeline..."
    python -m src.sfm.vi_sfm_pipeline \
        --image_dir "$IMAGE_DIR_TARGET" \
        --imu_path "$IMU_DATA_PATH" \
        --output_dir "${INPUT_DATASET}" \
        --format "$IMU_FORMAT"
else
    echo "Starting High-Precision hloc SfM Pipeline..."
    python -m src.sfm.hloc_pipeline \
        --image_dir "$IMAGE_DIR_TARGET" \
        --output_dir "${INPUT_DATASET}/cache" \
        --strategy sequential \
        --overlap 100
fi

# Move outputs to method specific directory
if [ -d "${INPUT_DATASET}/0" ]; then
    mv "${INPUT_DATASET}/0"/* "$SPARSE_METHOD_DIR/" 2>/dev/null || true
    rmdir "${INPUT_DATASET}/0" 2>/dev/null || true
fi
if [ -f "${SPARSE_BASE_DIR}/cameras.bin" ]; then
    mv "${SPARSE_BASE_DIR}"/*.bin "$SPARSE_METHOD_DIR/" 2>/dev/null || true
    mv "${SPARSE_BASE_DIR}"/*.ply "$SPARSE_METHOD_DIR/" 2>/dev/null || true
fi

# Write metadata json
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
python3 -c "
import json, os
metadata = {
    'method': '$METHOD',
    'timestamp': '$TIMESTAMP'
}
with open('$SPARSE_METHOD_DIR/sfm_info.json', 'w') as f:
    json.dump(metadata, f, indent=2)
"

# Create / Update active symlink sparse/0 -> method
cd "$SPARSE_BASE_DIR"
rm -rf 0
ln -s "$METHOD" 0
cd - > /dev/null

echo "SfM Reconstruction (${METHOD}) Completed! Saved to ${SPARSE_METHOD_DIR} (Active symlink: ${SPARSE_BASE_DIR}/0 -> ${METHOD})"
