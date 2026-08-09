#!/bin/bash
set -euo pipefail

# [Step 03c] 2D Gaussian Splatting TSDF Mesh Extraction
# Renders unbiased depth/normals from surfels and fuses via Open3D TSDF Integration

CONFIG_PATH="configs/default_config.sh"
if [ -f "$CONFIG_PATH" ]; then
    source "$CONFIG_PATH"
fi

CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
set +u
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate gs_train
set -u

INPUT_DATASET="${1:-$DATA_DIR}"
SCENE_NAME=$(basename "$INPUT_DATASET")

MODEL_DIR="${OUTPUT_DIR}/${SCENE_NAME}/2dgs"
MESH_OUTPUT_DIR="${OUTPUT_DIR}/${SCENE_NAME}/mesh/2dgs"
mkdir -p "$MESH_OUTPUT_DIR"

if [ ! -d "$MODEL_DIR" ]; then
    echo "[FATAL] Trained 2DGS model directory not found at $MODEL_DIR"
    echo "Please run Step 02c (./scripts/02c_train_2dgs.sh) first."
    exit 1
fi

echo "Starting 2DGS TSDF Mesh Extraction (Scene: $SCENE_NAME)..."
python third_party/2d-gaussian-splatting/render.py \
    -m "$MODEL_DIR" \
    --skip_train \
    --skip_test

# Locate and sync exported TSDF mesh to canonical 3DRC mesh path
EXPORTED_MESH=$(find "$MODEL_DIR/train" -name "*fuse_post.ply" | head -n 1)
if [ -z "$EXPORTED_MESH" ]; then
    EXPORTED_MESH=$(find "$MODEL_DIR/train" -name "*fuse*.ply" | head -n 1)
fi

if [ -n "$EXPORTED_MESH" ] && [ -f "$EXPORTED_MESH" ]; then
    cp "$EXPORTED_MESH" "${MESH_OUTPUT_DIR}/tsdf_mesh.ply"
    echo "2DGS TSDF Mesh Extraction Completed!"
    echo "Canonical Mesh saved to: ${MESH_OUTPUT_DIR}/tsdf_mesh.ply"
else
    echo "[WARN] Could not find extracted fuse mesh in $MODEL_DIR/train"
fi
