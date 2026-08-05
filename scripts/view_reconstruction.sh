#!/bin/bash
# scripts/view_reconstruction.sh

TYPE=$1

if [ -z "$TYPE" ]; then
    echo "Usage: ./scripts/view_reconstruction.sh [fastmap|hloc]"
    exit 1
fi

# Load and activate Conda environment
CONDA_PATH="/home/lee/miniconda3/bin/conda"
if [ -f "$CONDA_PATH" ]; then
    eval "$($CONDA_PATH 'shell.bash' 'hook')"
    conda activate 3drc
fi

if [ "$TYPE" == "fastmap" ]; then
    echo "Starting COLMAP GUI for FastMap reconstruction..."
    colmap gui \
        --database_path data/database_fastmap.db \
        --image_path data/images \
        --import_path data/fastmap_reconstruction/sparse/0
elif [ "$TYPE" == "hloc" ]; then
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
