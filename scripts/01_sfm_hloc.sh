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

ARG1="${1:-}"
ARG2="${2:-}"
ARG_DATASET=""
ARG_METHOD=""

if [ -n "$ARG1" ]; then
    if [ -d "$ARG1" ] || [[ "$ARG1" == data/* ]]; then
        ARG_DATASET="$ARG1"
        [ -n "$ARG2" ] && ARG_METHOD="$ARG2"
    else
        ARG_METHOD="$ARG1"
        [ -n "$ARG2" ] && ARG_DATASET="$ARG2"
    fi
fi

METHOD="${ARG_METHOD:-$SFM_METHOD}"
INPUT_DATASET="${ARG_DATASET:-$DATA_DIR}"

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
        --strategy "${SFM_STRATEGY:-sequential}" \
        --overlap "${SFM_OVERLAP:-100}" \
        --retrieval_k "${SFM_RETRIEVAL_K:-30}"
fi

# Move/copy outputs to method specific directory
if [ -d "${INPUT_DATASET}/cache/sfm" ]; then
    cp -r "${INPUT_DATASET}/cache/sfm"/* "$SPARSE_METHOD_DIR/" 2>/dev/null || true
fi
if [ -d "${INPUT_DATASET}/0" ]; then
    mv "${INPUT_DATASET}/0"/* "$SPARSE_METHOD_DIR/" 2>/dev/null || true
    rmdir "${INPUT_DATASET}/0" 2>/dev/null || true
fi
if [ -f "${SPARSE_BASE_DIR}/cameras.bin" ]; then
    mv "${SPARSE_BASE_DIR}"/*.bin "$SPARSE_METHOD_DIR/" 2>/dev/null || true
    mv "${SPARSE_BASE_DIR}"/*.ply "$SPARSE_METHOD_DIR/" 2>/dev/null || true
fi

# Verify reconstruction integrity before updating symlink
HAS_CAMERAS=false
if [ -f "$SPARSE_METHOD_DIR/cameras.bin" ] || [ -f "$SPARSE_METHOD_DIR/cameras.txt" ]; then
    HAS_CAMERAS=true
fi

HAS_IMAGES=false
if [ -f "$SPARSE_METHOD_DIR/images.bin" ] || [ -f "$SPARSE_METHOD_DIR/images.txt" ]; then
    HAS_IMAGES=true
fi

HAS_POINTS=false
if [ -f "$SPARSE_METHOD_DIR/points3D.bin" ] || [ -f "$SPARSE_METHOD_DIR/points3D.txt" ]; then
    HAS_POINTS=true
fi

if [ "$HAS_CAMERAS" = false ] || [ "$HAS_IMAGES" = false ] || [ "$HAS_POINTS" = false ]; then
    echo "[FATAL] SfM (${METHOD}) failed: Required reconstruction files (cameras, images, points3D) missing in $SPARSE_METHOD_DIR" >&2
    exit 1
fi

# Write metadata json
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
python3 -c "
import json, os
from pathlib import Path

img_dir = Path('$IMAGE_DIR_TARGET')
img_files = sorted([f.name for f in img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]) if img_dir.is_dir() else []
prefix_counts = {}
for name in img_files:
    stem = Path(name).stem
    if '_' in stem:
        parts = stem.rsplit('_', 1)
        p = parts[0] if parts[1].isdigit() else stem.split('_', 1)[0]
    else:
        p = 'single'
    prefix_counts[p] = prefix_counts.get(p, 0) + 1

metadata = {
    'method': '$METHOD',
    'timestamp': '$TIMESTAMP',
    'strategy': '$SFM_STRATEGY',
    'overlap': int('$SFM_OVERLAP') if '$SFM_OVERLAP'.isdigit() else '$SFM_OVERLAP',
    'retrieval_k': int('$SFM_RETRIEVAL_K') if '$SFM_RETRIEVAL_K'.isdigit() else '$SFM_RETRIEVAL_K',
    'camera_model': os.environ.get('CAMERA_MODEL', 'SIMPLE_RADIAL'),
    'camera_mode': os.environ.get('CAMERA_MODE', 'SINGLE'),
    'num_images_total': len(img_files),
    'video_prefixes': prefix_counts
}
with open('$SPARSE_METHOD_DIR/sfm_info.json', 'w') as f:
    json.dump(metadata, f, indent=2)
"

# Create / Update active symlink sparse/0 -> method only after verification succeeds
cd "$SPARSE_BASE_DIR"
rm -rf 0
ln -s "$METHOD" 0
cd - > /dev/null

echo "SfM Reconstruction (${METHOD}) Completed! Saved to ${SPARSE_METHOD_DIR} (Active symlink: ${SPARSE_BASE_DIR}/0 -> ${METHOD})"
