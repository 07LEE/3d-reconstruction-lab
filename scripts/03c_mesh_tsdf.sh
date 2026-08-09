#!/bin/bash
set -euo pipefail

# [Step 03c] TSDF Mesh Extraction Engine (Universal Depth/Normal Fusion)
# Renders depth/normals from trained Gaussian/Surfel models and fuses via Open3D TSDF Integration

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

# Dynamic source model selection (default: 2dgs, supports planargs, 3dgs, or custom model directory)
SOURCE_MODEL_NAME="${2:-2dgs}"
if [ -d "$SOURCE_MODEL_NAME" ]; then
    MODEL_DIR="$SOURCE_MODEL_NAME"
    SOURCE_MODEL_NAME=$(basename "$MODEL_DIR")
else
    MODEL_DIR="${OUTPUT_DIR}/${SCENE_NAME}/${SOURCE_MODEL_NAME}"
fi

MESH_OUTPUT_DIR="${OUTPUT_DIR}/${SCENE_NAME}/mesh/${SOURCE_MODEL_NAME}"
mkdir -p "$MESH_OUTPUT_DIR"

if [ ! -d "$MODEL_DIR" ]; then
    echo "[FATAL] Source Gaussian model directory not found at: $MODEL_DIR"
    echo "Usage: ./scripts/03c_mesh_tsdf.sh [dataset_path] [source_model_subdir_or_path]"
    exit 1
fi

# Detect active SfM pose source dynamically
ACTIVE_SFM="unknown"
if [ -L "${INPUT_DATASET}/sparse/0" ]; then
    ACTIVE_SFM=$(readlink "${INPUT_DATASET}/sparse/0" | xargs basename)
fi

echo "Starting TSDF Mesh Extraction (Scene: $SCENE_NAME, Source Model: $MODEL_DIR, Active SfM: $ACTIVE_SFM)..."
VOXEL_SIZE="${TSDF_VOXEL_SIZE:-0.005}"
DEPTH_TRUNC="${TSDF_DEPTH_TRUNC:-6.0}"

python third_party/2d-gaussian-splatting/render.py \
    -m "$MODEL_DIR" \
    --data_device cpu \
    --skip_train \
    --skip_test \
    --voxel_size "$VOXEL_SIZE" \
    --depth_trunc "$DEPTH_TRUNC"

# Locate and sync exported TSDF mesh to canonical 3DRC mesh path
EXPORTED_MESH=$(find "$MODEL_DIR/train" -name "*fuse_post.ply" | head -n 1)
if [ -z "$EXPORTED_MESH" ]; then
    EXPORTED_MESH=$(find "$MODEL_DIR/train" -name "*fuse*.ply" | head -n 1)
fi

if [ -n "$EXPORTED_MESH" ] && [ -f "$EXPORTED_MESH" ]; then
    cp "$EXPORTED_MESH" "${MESH_OUTPUT_DIR}/tsdf_mesh.ply"
    echo "TSDF Mesh Extraction Completed!"
    echo "Canonical Mesh saved to: ${MESH_OUTPUT_DIR}/tsdf_mesh.ply"

    # Dynamically record execution provenance metadata
    cat <<EOF > "$MESH_OUTPUT_DIR/pipeline_meta.json"
{
  "stage": "Stage 3 (Mesh Extraction)",
  "engine": "Open3D TSDF Voxel Integration",
  "source_model_path": "$MODEL_DIR",
  "source_model_type": "$SOURCE_MODEL_NAME",
  "source_dataset": "$INPUT_DATASET",
  "active_sfm": "$ACTIVE_SFM",
  "voxel_size": $VOXEL_SIZE,
  "depth_trunc": $DEPTH_TRUNC,
  "timestamp": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF
else
    echo "[WARN] Could not find extracted fuse mesh in $MODEL_DIR/train"
fi
