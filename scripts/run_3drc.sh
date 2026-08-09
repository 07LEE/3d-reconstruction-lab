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

CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
PYTHON_BIN="$CONDA_PATH/envs/3drc/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

show_help() {
    echo "3DRC Pipeline Unified CLI Interface"
    echo "Usage: ./scripts/run_3drc.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  sfm [method]     Run Step 1 Camera Pose Estimation (hloc, vi_sfm, vggt, fastmap, sfm)"
    echo "  train [backend]  Run Step 2 3DGS Training (3dgs, planargs)"
    echo "  planargs         Run Step 2b PlanarGS Training"
    echo "  view [type]      Run COLMAP GUI Visualization (hloc, fastmap)"
    echo "  sugar            Run Step 3 SuGaR Mesh Reconstruction"
    echo "  milo             Run Step 3b MILo Mesh Reconstruction"
    echo "  grouping         Run Step 4 Gaussian Grouping Object Segmentation"
    echo "  eval [mesh] [gt] Run Step 5 Mesh Evaluation against LiDAR GT PointCloud"
    echo "  outputs          Inspect and summarize all generated output artifacts"
    echo "  pipeline [method] Run End-to-End Pipeline (SfM -> Train -> Mesh)"
    echo "  help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run_3drc.sh sfm vi_sfm"
    echo "  ./scripts/run_3drc.sh train 3dgs"
    echo "  ./scripts/run_3drc.sh train planargs"
    echo "  ./scripts/run_3drc.sh sugar"
    echo "  ./scripts/run_3drc.sh milo"
    echo "  ./scripts/run_3drc.sh eval <mesh_path> [gt_pcd_path]"
    echo "  ./scripts/run_3drc.sh outputs"
    echo "  ./scripts/run_3drc.sh view hloc"
    echo "  ./scripts/run_3drc.sh pipeline vi_sfm"
}

COMMAND=${1:-"help"}

case "$COMMAND" in
    outputs)
        "$PYTHON_BIN" src/inspect_outputs.py
        ;;
    sfm)
        METHOD=${2:-$SFM_METHOD}
        echo "[3DRC CLI] Executing Step 1: Camera Pose Estimation (Method: ${METHOD})..."
        ./scripts/01_sfm_hloc.sh "$METHOD"
        ;;
    train)
        BACKEND=${2:-"3dgs"}
        if [ "$BACKEND" = "planargs" ]; then
            echo "[3DRC CLI] Executing Step 2b: PlanarGS Training..."
            ./scripts/02b_train_planargs.sh
        else
            echo "[3DRC CLI] Executing Step 2: 3DGS Training..."
            ./scripts/02_train_3dgs.sh
        fi
        ;;
    planargs)
        echo "[3DRC CLI] Executing Step 2b: PlanarGS Training..."
        ./scripts/02b_train_planargs.sh
        ;;
    view)
        TYPE=${2:-"hloc"}
        echo "[3DRC CLI] Launching Reconstruction Viewer (Type: ${TYPE})..."
        ./scripts/utils/view_reconstruction.sh "$TYPE"
        ;;
    sugar)
        echo "[3DRC CLI] Executing Step 3: SuGaR Mesh Reconstruction..."
        ./scripts/03_train_sugar.sh
        ;;
    milo)
        echo "[3DRC CLI] Executing Step 3b: MILo Mesh Reconstruction..."
        ./scripts/03b_train_milo.sh
        ;;
    grouping)
        echo "[3DRC CLI] Executing Step 4: Gaussian Grouping..."
        ./scripts/04_train_grouping.sh
        ;;
    eval)
        MESH_PATH="${2:-}"
        GT_PATH="${3:-}"
        if [ -z "$MESH_PATH" ]; then
            echo "Error: Mesh file path required for evaluation."
            echo "Usage: ./scripts/run_3drc.sh eval <mesh_path> [gt_pointcloud_path]"
            exit 1
        fi
        EVAL_ARGS=("--mesh" "$MESH_PATH")
        if [ -n "$GT_PATH" ]; then
            EVAL_ARGS+=("--gt" "$GT_PATH")
        fi
        echo "[3DRC CLI] Executing Mesh Evaluation on ${MESH_PATH}..."
        "$PYTHON_BIN" src/eval_mesh.py "${EVAL_ARGS[@]}"
        ;;
    pipeline)
        SFM_OPT=${2:-"hloc"}
        echo "[3DRC CLI] Starting End-to-End Automated Pipeline (SfM: ${SFM_OPT})..."
        ./scripts/01_sfm_hloc.sh "$SFM_OPT"
        ./scripts/02_train_3dgs.sh
        ./scripts/03_train_sugar.sh
        echo "[3DRC CLI] End-to-End Automated Pipeline Execution Finished!"
        ;;
    help|*)
        show_help
        ;;
esac
