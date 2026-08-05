#!/bin/bash

# 3DRC Unified Command Line Interface (CLI)
# Manages execution of all 3D reconstruction pipeline steps from a single entry point.

set -e

CONFIG_PATH="configs/default_config.sh"
if [ -f "$CONFIG_PATH" ]; then
    source "$CONFIG_PATH"
fi

show_help() {
    echo "3DRC Pipeline Unified CLI Interface"
    echo "Usage: ./scripts/run_3drc.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  sfm [method]     Run Step 1 Camera Pose Estimation (hloc, vi_sfm, vggt, fastmap, sfm)"
    echo "  train [method]   Run Step 2 High-Density 3DGS Training (3dgs, planar)"
    echo "  view             Run Step 3 Result Visualization"
    echo "  sugar            Run Step 4 SuGaR Mesh Reconstruction"
    echo "  grouping         Run Step 5 Gaussian Grouping Object Segmentation"
    echo "  pipeline [method] Run End-to-End Pipeline (SfM -> Train -> View)"
    echo "  help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run_3drc.sh sfm vi_sfm"
    echo "  ./scripts/run_3drc.sh train 3dgs"
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
        TRAIN_OPT=${2:-$TRAIN_METHOD}
        echo "[3DRC CLI] Executing Step 2: 3DGS Training (Method: ${TRAIN_OPT})..."
        if [ "$TRAIN_OPT" != "$TRAIN_METHOD" ]; then
            export TRAIN_METHOD="$TRAIN_OPT"
        fi
        ./scripts/02_train_3dgs.sh
        ;;
    view)
        echo "[3DRC CLI] Executing Step 3: Result Visualization..."
        ./scripts/03_view_result.sh
        ;;
    sugar)
        echo "[3DRC CLI] Executing Step 4: SuGaR Mesh Reconstruction..."
        ./scripts/04_train_sugar.sh
        ;;
    grouping)
        echo "[3DRC CLI] Executing Step 5: Gaussian Grouping..."
        ./scripts/05_train_grouping.sh
        ;;
    pipeline)
        SFM_OPT=${2:-"hloc"}
        echo "[3DRC CLI] Starting End-to-End Automated Pipeline (SfM: ${SFM_OPT})..."
        ./scripts/01_sfm_hloc.sh "$SFM_OPT"
        ./scripts/02_train_3dgs.sh
        ./scripts/03_view_result.sh
        echo "[3DRC CLI] End-to-End Automated Pipeline Execution Finished!"
        ;;
    help|*)
        show_help
        ;;
esac
