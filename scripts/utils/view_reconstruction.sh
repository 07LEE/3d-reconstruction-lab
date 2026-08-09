#!/bin/bash
set -euo pipefail

CONFIG_PATH="configs/default_config.sh"
if [ -f "$CONFIG_PATH" ]; then
    source "$CONFIG_PATH"
fi

TYPE=${1:-"hloc"}
INPUT_DATASET="${2:-$DATA_DIR}"

IMAGE_DIR_TARGET="${INPUT_DATASET}/raw_images"
if [ ! -d "$IMAGE_DIR_TARGET" ]; then
    IMAGE_DIR_TARGET="${INPUT_DATASET}/images"
fi

# Dynamic Conda environment activation for COLMAP GUI
CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    set +u
    conda activate 3drc
    set -u
    [ "${CONDA_DEFAULT_ENV:-}" = "3drc" ] || { echo "[FATAL] Conda environment '3drc' activation failed!"; exit 1; }
fi

if ! command -v colmap >/dev/null 2>&1 && [ -x "$CONDA_PATH/bin/colmap" ]; then
    export PATH="$CONDA_PATH/bin:$PATH"
fi

# Resolve sparse import directory
IMPORT_DIR="${INPUT_DATASET}/sparse/${TYPE}"
if [ ! -d "$IMPORT_DIR" ]; then
    IMPORT_DIR="${INPUT_DATASET}/sparse/0"
fi
if [ -d "$IMPORT_DIR/models/0" ] && [ -f "$IMPORT_DIR/models/0/points3D.bin" ]; then
    IMPORT_DIR="$IMPORT_DIR/models/0"
fi

# Resolve database path if exists
DB_ARG=()
if [ -f "${INPUT_DATASET}/database.db" ]; then
    DB_ARG=(--database_path "${INPUT_DATASET}/database.db")
elif [ -f "${INPUT_DATASET}/cache/database.db" ]; then
    DB_ARG=(--database_path "${INPUT_DATASET}/cache/database.db")
elif [ -f "${INPUT_DATASET}/sparse/${TYPE}/database.db" ]; then
    DB_ARG=(--database_path "${INPUT_DATASET}/sparse/${TYPE}/database.db")
fi

echo "Starting COLMAP GUI for reconstruction (Type: $TYPE, Dataset: $INPUT_DATASET)..."
colmap gui \
    "${DB_ARG[@]}" \
    --image_path "$IMAGE_DIR_TARGET" \
    --import_path "$IMPORT_DIR"
