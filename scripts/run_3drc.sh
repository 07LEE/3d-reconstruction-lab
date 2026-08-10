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
    echo "  sfm [method]     Run Step 1 Camera Pose Estimation (hloc, vggt, fastmap, sfm)"
    echo "  train [backend]  Run Step 2 Gaussian Training (3dgs, planargs, 2dgs)"
    echo "  planargs         Run Step 2b PlanarGS Training"
    echo "  2dgs             Run Step 2c 2D Gaussian Splatting Training"
    echo "  view [type]      Run COLMAP GUI Visualization (hloc, fastmap)"
    echo "  sugar            Run Step 3 SuGaR Mesh Reconstruction"
    echo "  milo             Run Step 3b MILo Mesh Reconstruction"
    echo "  tsdf             Run Step 3c TSDF Mesh Extraction"
    echo "  grouping         Run Step 4 Gaussian Grouping Object Segmentation"
    echo "  eval [mesh] [gt] Run Step 5 Mesh Evaluation against LiDAR GT PointCloud"
    echo "  outputs          Inspect and summarize all generated output artifacts"
    echo "  pipeline [method] Run End-to-End Pipeline (SfM -> Train -> Mesh)"
    echo "  help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run_3drc.sh sfm hloc"
    echo "  ./scripts/run_3drc.sh train 3dgs"
    echo "  ./scripts/run_3drc.sh train 2dgs"
    echo "  ./scripts/run_3drc.sh train planargs"
    echo "  ./scripts/run_3drc.sh sugar"
    echo "  ./scripts/run_3drc.sh milo"
    echo "  ./scripts/run_3drc.sh tsdf"
    echo "  ./scripts/run_3drc.sh eval <mesh_path> [gt_pcd_path]"
    echo "  ./scripts/run_3drc.sh outputs"
    echo "  ./scripts/run_3drc.sh view hloc"
    echo "  ./scripts/run_3drc.sh pipeline vi_sfm"
}

COMMAND=${1:-"help"}

case "$COMMAND" in
    outputs)
        "$PYTHON_BIN" src/utils/inspect_outputs.py
        ;;
    sfm)
        # Smart argument parsing for sfm: supports (method, dataset) or (dataset, method)
        ARG1="${2:-}"
        ARG2="${3:-}"
        METHOD="$SFM_METHOD"
        ARG_DATASET=""
        if [ -n "$ARG1" ]; then
            if [ -d "$ARG1" ] || [[ "$ARG1" == data/* ]]; then
                ARG_DATASET="$ARG1"
                [ -n "$ARG2" ] && METHOD="$ARG2"
            else
                METHOD="$ARG1"
                [ -n "$ARG2" ] && ARG_DATASET="$ARG2"
            fi
        fi
        echo "[3DRC CLI] Executing Step 1: Camera Pose Estimation (Method: ${METHOD})..."
        ./scripts/01_sfm_hloc.sh "$METHOD" ${ARG_DATASET:+"$ARG_DATASET"}
        ;;
    train)
        # Smart argument parsing for train: supports (backend, dataset) or (dataset, backend)
        ARG1="${2:-}"
        ARG2="${3:-}"
        BACKEND="3dgs"
        ARG_DATASET=""
        if [ -n "$ARG1" ]; then
            if [ -d "$ARG1" ] || [[ "$ARG1" == data/* ]]; then
                ARG_DATASET="$ARG1"
                [ -n "$ARG2" ] && BACKEND="$ARG2"
            else
                BACKEND="$ARG1"
                [ -n "$ARG2" ] && ARG_DATASET="$ARG2"
            fi
        fi
        if [ "$BACKEND" = "planargs" ]; then
            echo "[3DRC CLI] Executing Step 2b: PlanarGS Training..."
            ./scripts/02b_train_planargs.sh ${ARG_DATASET:+"$ARG_DATASET"}
        elif [ "$BACKEND" = "2dgs" ]; then
            echo "[3DRC CLI] Executing Step 2c: 2D Gaussian Splatting Training..."
            ./scripts/02c_train_2dgs.sh ${ARG_DATASET:+"$ARG_DATASET"}
        else
            echo "[3DRC CLI] Executing Step 2: 3DGS Training..."
            ./scripts/02_train_3dgs.sh ${ARG_DATASET:+"$ARG_DATASET"}
        fi
        ;;
    2dgs)
        ARG_DATASET="${2:-}"
        echo "[3DRC CLI] Executing Step 2c: 2D Gaussian Splatting Training..."
        ./scripts/02c_train_2dgs.sh ${ARG_DATASET:+"$ARG_DATASET"}
        ;;
    planargs)
        ARG_DATASET="${2:-}"
        echo "[3DRC CLI] Executing Step 2b: PlanarGS Training..."
        ./scripts/02b_train_planargs.sh ${ARG_DATASET:+"$ARG_DATASET"}
        ;;
    view)
        TYPE=${2:-"hloc"}
        echo "[3DRC CLI] Launching Reconstruction Viewer (Type: ${TYPE})..."
        ./scripts/utils/view_reconstruction.sh "$TYPE"
        ;;
    sugar)
        ARG1="${2:-}"
        ARG2="${3:-}"
        ARG_DATASET=""
        ARG_SOURCE_MODEL=""
        if [ -n "$ARG1" ]; then
            if [ -d "$ARG1" ] || [[ "$ARG1" == data/* ]]; then
                ARG_DATASET="$ARG1"
                [ -n "$ARG2" ] && ARG_SOURCE_MODEL="$ARG2"
            else
                ARG_SOURCE_MODEL="$ARG1"
                [ -n "$ARG2" ] && ARG_DATASET="$ARG2"
            fi
        fi
        echo "[3DRC CLI] Executing Step 3: SuGaR Mesh Reconstruction..."
        ./scripts/03_train_sugar.sh ${ARG_DATASET:+"$ARG_DATASET"} ${ARG_SOURCE_MODEL:+"$ARG_SOURCE_MODEL"}
        ;;
    milo)
        ARG_DATASET="${2:-}"
        echo "[3DRC CLI] Executing Step 3b: MILo Mesh Reconstruction..."
        ./scripts/03b_train_milo.sh ${ARG_DATASET:+"$ARG_DATASET"}
        ;;
    tsdf|mesh_tsdf)
        ARG1="${2:-}"
        ARG2="${3:-}"
        ARG_DATASET=""
        ARG_SOURCE_MODEL=""
        if [ -n "$ARG1" ]; then
            if [ -d "$ARG1" ] || [[ "$ARG1" == data/* ]]; then
                ARG_DATASET="$ARG1"
                [ -n "$ARG2" ] && ARG_SOURCE_MODEL="$ARG2"
            else
                ARG_SOURCE_MODEL="$ARG1"
                [ -n "$ARG2" ] && ARG_DATASET="$ARG2"
            fi
        fi
        echo "[3DRC CLI] Executing Step 3c: TSDF Mesh Extraction..."
        ./scripts/03c_mesh_tsdf.sh ${ARG_DATASET:+"$ARG_DATASET"} ${ARG_SOURCE_MODEL:+"$ARG_SOURCE_MODEL"}
        ;;
    grouping)
        ARG_DATASET="${2:-}"
        echo "[3DRC CLI] Executing Step 4: Gaussian Grouping..."
        ./scripts/04_train_grouping.sh ${ARG_DATASET:+"$ARG_DATASET"}
        ;;
    eval)
        MESH_PATH="${2:-}"
        GT_PATH="${3:-}"
        if [ -z "$MESH_PATH" ]; then
            echo "[Error] Mesh path required for eval."
            echo "Usage: ./scripts/run_3drc.sh eval <mesh_path> [gt_pcd_path]"
            exit 1
        fi
        EVAL_ARGS=("--mesh" "$MESH_PATH")
        if [ -n "$GT_PATH" ]; then
            EVAL_ARGS+=("--gt" "$GT_PATH")
        fi
        echo "[3DRC CLI] Executing Step 5: Mesh Quantitative Evaluation..."
        "$PYTHON_BIN" src/mesh/eval_mesh.py "${EVAL_ARGS[@]}"
        ;;
    pipeline)
        ARG1="${2:-}"
        ARG2="${3:-}"
        ARG_DATASET=""
        ARG_METHOD=""
        if [ -n "$ARG1" ]; then
            if [ -d "$ARG1" ] || [[ "$ARG1" == data/* ]]; then
                ARG_DATASET="$ARG1"
                [ -n "$ARG2" ] && ARG_METHOD="$ARG2"
            else
                ARG_METHOD="$ARG1"
                [ -n "$ARG2" ] && ARG_DATASET="$ARG2"
            fi
        fi
        SFM_OPT="${ARG_METHOD:-$SFM_METHOD}"
        TARGET_DATASET="${ARG_DATASET:-$DATA_DIR}"
        echo "[3DRC CLI] Starting End-to-End Automated Pipeline (SfM: ${SFM_OPT}, Dataset: ${TARGET_DATASET})..."
        ./scripts/01_sfm_hloc.sh "$SFM_OPT" "$TARGET_DATASET"
        ./scripts/02_train_3dgs.sh "$TARGET_DATASET"
        ./scripts/03_train_sugar.sh "$TARGET_DATASET"
        echo "[3DRC CLI] End-to-End Automated Pipeline Execution Finished!"
        ;;
    help|*)
        show_help
        ;;
esac
