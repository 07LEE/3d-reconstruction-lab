#!/bin/bash

# 3DRC Unified Command Line Interface (CLI)
# Manages execution of all 3D reconstruction pipeline steps from a single entry point.

set -euo pipefail

CONFIG_PATH="configs/default_config.sh"
if [ -f "$CONFIG_PATH" ]; then
    source "$CONFIG_PATH"
else
    echo "[FATAL] Configuration file not found at $CONFIG_PATH!"
    exit 1
fi

show_help() {
    echo "3DRC Pipeline Unified CLI Interface"
    echo "Usage: ./scripts/run_3drc.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  sfm [method]     Run Step 1 Camera Pose Estimation (hloc, vi_sfm, vggt, fastmap, sfm)"
    echo "  train            Run Step 2 High-Density 3DGS Training"
    echo "  view [type]      Run COLMAP GUI Visualization (hloc, fastmap)"
    echo "  sugar            Run Step 3 SuGaR Mesh Reconstruction"
    echo "  grouping         Run Step 4 Gaussian Grouping Object Segmentation"
    echo "  pipeline [method] Run End-to-End Pipeline (SfM -> Train)"
    echo "  help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run_3drc.sh sfm vi_sfm"
    echo "  ./scripts/run_3drc.sh train"
    echo "  ./scripts/run_3drc.sh view hloc"
    echo "  ./scripts/run_3drc.sh pipeline vi_sfm"
}

COMMAND=${1:-"help"}

case "$COMMAND" in
    sfm)
        METHOD=${2:-$SFM_METHOD}
        echo "[3DRC CLI] Executing Step 1: Camera Pose Estimation (Method: ${METHOD})..."
        ./scripts/01_sfm_hloc.sh "$METHOD"
        ;;
    train)
        echo "[3DRC CLI] Executing Step 2: 3DGS Training..."
        ./scripts/02_train_3dgs.sh
        ;;
    view)
        TYPE=${2:-"hloc"}
        echo "[3DRC CLI] Launching Reconstruction Viewer (Type: ${TYPE})..."
        ./scripts/view_reconstruction.sh "$TYPE"
        ;;
    sugar)
        echo "[3DRC CLI] Executing Step 3: SuGaR Mesh Reconstruction..."
        ./scripts/03_train_sugar.sh
        ;;
    grouping)
        echo "[3DRC CLI] Executing Step 4: Gaussian Grouping..."
        ./scripts/04_train_grouping.sh
        ;;
    pipeline)
        SFM_OPT=${2:-"hloc"}
        echo "[3DRC CLI] Starting End-to-End Automated Pipeline (SfM: ${SFM_OPT})..."
        ./scripts/01_sfm_hloc.sh "$SFM_OPT"
        ./scripts/02_train_3dgs.sh
        echo "[3DRC CLI] End-to-End Automated Pipeline Execution Finished!"
        ;;
    help|*)
        show_help
        ;;
esac
