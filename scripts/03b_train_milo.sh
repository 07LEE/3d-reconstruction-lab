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

# Create main outputs subdirectories
mkdir -p "$PROJECT_ROOT/${OUTPUT_DIR}/milo"
mkdir -p "$PROJECT_ROOT/${OUTPUT_DIR}/milo_mesh"

# Move to MILo directory
cd third_party/milo/milo || exit 1

# Determine dataset source (prefer data/undistorted if available)
INPUT_DATASET="${1:-}"
if [ -z "$INPUT_DATASET" ]; then
    if [ -d "$PROJECT_ROOT/data/undistorted/sparse/0" ]; then
        INPUT_DATASET="data/undistorted"
    else
        INPUT_DATASET="$DATA_DIR"
    fi
fi

# Execute MILo Training & Mesh Extraction
echo "Starting MILo Differentiable Mesh Training & Extraction (Dataset: $INPUT_DATASET)..."
python train.py \
    -s "$PROJECT_ROOT/$INPUT_DATASET" \
    -m "$PROJECT_ROOT/${OUTPUT_DIR}/milo" \
    --imp_metric outdoor \
    --rasterizer radegs \
    --data_device cpu

echo "Extracting final MILo surface mesh..."
python mesh_extract_sdf.py \
    -s "$PROJECT_ROOT/$INPUT_DATASET" \
    -m "$PROJECT_ROOT/${OUTPUT_DIR}/milo" \
    --rasterizer radegs \
    --data_device cpu

# Sync extracted results to main outputs directory
echo "Syncing extracted MILo results to main $OUTPUT_DIR directory..."
if [ -d "$PROJECT_ROOT/${OUTPUT_DIR}/milo" ]; then
    find "$PROJECT_ROOT/${OUTPUT_DIR}/milo" -name "*.obj" -exec cp {} "$PROJECT_ROOT/${OUTPUT_DIR}/milo_mesh/" \; 2>/dev/null || true
    find "$PROJECT_ROOT/${OUTPUT_DIR}/milo" -name "*.ply" -exec cp {} "$PROJECT_ROOT/${OUTPUT_DIR}/milo_mesh/" \; 2>/dev/null || true
fi

echo "MILo Pipeline Completed! Results saved to $PROJECT_ROOT/${OUTPUT_DIR}/milo_mesh"
