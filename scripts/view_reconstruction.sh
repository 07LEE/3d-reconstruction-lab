#!/bin/bash
set -euo pipefail

TYPE=${1:-"hloc"}

if [ -z "$TYPE" ]; then
    echo "Usage: ./scripts/view_reconstruction.sh [fastmap|hloc]"
    exit 1
fi

# Dynamic Conda environment activation
CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    set +u
    conda activate gs_train 2>/dev/null || true
    set -u
fi

if [ "$TYPE" = "fastmap" ]; then
    echo "Starting COLMAP GUI for FastMap reconstruction..."
    colmap gui \
        --database_path data/database_fastmap.db \
        --image_path data/images \
        --import_path data/fastmap_reconstruction/sparse/0
elif [ "$TYPE" = "hloc" ]; then
    echo "Starting COLMAP GUI for Hloc (standard) reconstruction..."
    IMPORT_DIR="data/hloc_reconstruction/sfm"
    if [ -d "$IMPORT_DIR/models/0" ] && [ -f "$IMPORT_DIR/models/0/points3D.bin" ]; then
        echo "Detected multi-model reconstruction. Using models/0 for importing."
        IMPORT_DIR="$IMPORT_DIR/models/0"
    fi
    colmap gui \
        --database_path data/hloc_reconstruction/sfm/database.db \
        --image_path data/images \
        --import_path "$IMPORT_DIR"
else
    echo "Unknown reconstruction type: $TYPE"
    echo "Please choose either 'fastmap' or 'hloc'."
    exit 1
fi
