#!/bin/bash
set -euo pipefail

# [Step 03b] MILo 3D Mesh Reconstruction Pipeline
# Multi-Scale Implicit Layer Optimization for compact and sharp surface extraction

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
    conda activate gs_milo
    set -u
    [ "${CONDA_DEFAULT_ENV:-}" = "gs_milo" ] || { echo "[FATAL] Conda environment 'gs_milo' activation failed!"; exit 1; }
fi

# Source CUDA & host compiler build environment (GCC 12 / NVCC 12.8)
source "$(dirname "$0")/utils/setup_build_env.sh"

# Ensure Conda environment site-packages is explicitly first in PYTHONPATH
export PYTHONPATH="${CONDA_PREFIX}/lib/python3.10/site-packages"

# Get project root
PROJECT_ROOT=$(pwd)

# Determine dataset source
INPUT_DATASET="${1:-$DATA_DIR}"
SCENE_NAME=$(basename "$INPUT_DATASET")

MILO_MODEL_DIR="$PROJECT_ROOT/${OUTPUT_DIR}/${SCENE_NAME}/mesh/milo"
mkdir -p "$MILO_MODEL_DIR"

# Ensure undistorted PINHOLE dataset workspace exists for MILo
DENSE_DATASET="${INPUT_DATASET}/dense"
if [ ! -d "$DENSE_DATASET/sparse" ]; then
    echo "Preparing undistorted PINHOLE dataset for MILo via src/dense_undistort.py..."
    SPARSE_IN="${INPUT_DATASET}/sparse/0"
    IMG_IN="${INPUT_DATASET}/raw_images"
    if [ ! -d "$IMG_IN" ]; then
        IMG_IN="${INPUT_DATASET}/images"
    fi
    PYTHON_3DRC="$CONDA_PATH/envs/3drc/bin/python"
    if [ ! -x "$PYTHON_3DRC" ]; then
        PYTHON_3DRC="python3"
    fi
    "$PYTHON_3DRC" src/dense_undistort.py \
        --input_dir "$SPARSE_IN" \
        --image_dir "$IMG_IN" \
        --output_dir "$DENSE_DATASET"
fi

# Move to MILo directory
cd third_party/milo/milo || exit 1

# Execute MILo Training & Mesh Extraction
echo "Starting MILo Differentiable Mesh Training & Extraction (Scene: $SCENE_NAME, Dataset: $DENSE_DATASET)..."
python train.py \
    -s "$PROJECT_ROOT/$DENSE_DATASET" \
    -m "$MILO_MODEL_DIR" \
    --imp_metric indoor \
    --rasterizer radegs \
    --data_device cpu

echo "Extracting final MILo surface mesh..."
python mesh_extract_sdf.py \
    -s "$PROJECT_ROOT/$DENSE_DATASET" \
    -m "$MILO_MODEL_DIR" \
    --rasterizer radegs \
    --data_device cpu

# Clean floater components and generate viewer files
echo "Cleaning floater components and generating viewer files in $MILO_MODEL_DIR..."
cd "$PROJECT_ROOT"
if [ -f "$MILO_MODEL_DIR/mesh_learnable_sdf.ply" ]; then
    python3 src/clean_milo_mesh.py \
        --input "$MILO_MODEL_DIR/mesh_learnable_sdf.ply" \
        --output "$MILO_MODEL_DIR/mesh_cleaned_largest.ply" || true
fi

echo "MILo Pipeline Completed! Results saved to $MILO_MODEL_DIR"
