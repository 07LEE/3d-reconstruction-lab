#!/bin/bash
set -euo pipefail

# [Step 03] SuGaR Mesh Reconstruction Pipeline
# Surface-Aligned Gaussian Splatting and high-quality mesh extraction

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
    conda activate gs_sugar
    set -u
    [ "${CONDA_DEFAULT_ENV:-}" = "gs_sugar" ] || { echo "[FATAL] Conda environment 'gs_sugar' activation failed!"; exit 1; }
fi

# Get project root
PROJECT_ROOT=$(pwd)

# Move to sugar directory
cd third_party/sugar || exit 1

# Create main outputs subdirectories
mkdir -p "$PROJECT_ROOT/${OUTPUT_DIR}/sugar_mesh"
mkdir -p "$PROJECT_ROOT/${OUTPUT_DIR}/sugar"

# Determine dataset source (prefer data/undistorted if available)
INPUT_DATASET="${1:-}"
if [ -z "$INPUT_DATASET" ]; then
    if [ -d "$PROJECT_ROOT/data/undistorted/sparse/0" ]; then
        INPUT_DATASET="data/undistorted"
    else
        INPUT_DATASET="$DATA_DIR"
    fi
fi

SCENE_NAME=$(basename "$INPUT_DATASET")
GS_CHECKPOINT="$PROJECT_ROOT/${OUTPUT_DIR}/${SCENE_NAME}/3dgs/inria_30k"
SUGAR_MESH_DIR="$PROJECT_ROOT/${OUTPUT_DIR}/${SCENE_NAME}/mesh/sugar"
mkdir -p "$SUGAR_MESH_DIR"

# Fallback to legacy path if inria_30k not found
if [ ! -d "$GS_CHECKPOINT" ] && [ -d "$PROJECT_ROOT/${OUTPUT_DIR}/gs_final_precision" ]; then
    GS_CHECKPOINT="$PROJECT_ROOT/${OUTPUT_DIR}/gs_final_precision"
fi

# Execute SuGaR Pipeline
echo "Starting SuGaR Mesh Extraction (Scene: $SCENE_NAME, Dataset: $INPUT_DATASET)..."
python train_full_pipeline.py \
    -s "$PROJECT_ROOT/$INPUT_DATASET" \
    --gs_output_dir "$GS_CHECKPOINT" \
    -r "$SUGAR_REGULARIZATION" \
    --high_poly "$SUGAR_HIGH_POLY" \
    --export_obj True \
    --refinement_time "$SUGAR_REFINEMENT"

# Sync output files to main outputs directory
echo "Syncing extracted SuGaR results to main $OUTPUT_DIR directory..."
if [ -d "output" ]; then
    cp -r output/* "$PROJECT_ROOT/${OUTPUT_DIR}/sugar/" 2>/dev/null || true
    if [ -d "output/refined_mesh" ]; then
        cp -r output/refined_mesh/* "$PROJECT_ROOT/${OUTPUT_DIR}/sugar_mesh/" 2>/dev/null || true
    fi
fi

echo "SuGaR Pipeline Completed! Results synced to $PROJECT_ROOT/${OUTPUT_DIR}/sugar_mesh"
