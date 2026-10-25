#!/bin/bash

# 3DRC Unified Command Line Interface (CLI)
# Manages execution of all 3D reconstruction pipeline steps from a single entry point.

set -euo pipefail

# Pass the dataset argument (if any) to the config so scene resolution matches the step
CONFIG_SCENE_ARG=""
for arg in "${@:2}"; do
    if [[ "$arg" == data/* ]] || [ -d "$arg/sparse" ] || [ -d "$arg/raw_images" ] || [ -d "$arg/images" ]; then
        CONFIG_SCENE_ARG="$arg"
        break
    fi
done

CONFIG_PATH="configs/default_config.sh"
if [ -f "$CONFIG_PATH" ]; then
    case "${1:-help}" in
        help|outputs|results|eval|view)
            # These commands do not use the scene, so skip the fallback notice and REQUIRE_SCENE
            QUIET_CONFIG=true REQUIRE_SCENE=false source "$CONFIG_PATH" ${CONFIG_SCENE_ARG:+"$CONFIG_SCENE_ARG"}
            ;;
        *)
            source "$CONFIG_PATH" ${CONFIG_SCENE_ARG:+"$CONFIG_SCENE_ARG"}
            ;;
    esac
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
    echo "  data [dataset]   Run Step 0b YUV In-Memory Data Load (raw YUV session -> in-memory tensors)"
    echo "  sfm [method]     Run Step 1 Camera Pose Estimation (hloc, vggt, fastmap, sfm)"
    echo "  train [backend]  Run Step 2 Gaussian Training (3dgs, planargs, 2dgs, scaffoldgs)"
    echo "  planargs         Run Step 2b PlanarGS Training"
    echo "  2dgs             Run Step 2c 2D Gaussian Splatting Training"
    echo "  scaffoldgs       Run Step 2d Scaffold-GS Training"
    echo "  view [type]      Run COLMAP GUI Visualization (hloc, fastmap)"
    echo "  sugar            Run Step 3 SuGaR Mesh Reconstruction"
    echo "  milo             Run Step 3b MILo Mesh Reconstruction"
    echo "  tsdf             Run Step 3c TSDF Mesh Extraction"
    echo "  grouping         Run Step 4 Gaussian Grouping Object Segmentation"
    echo "  eval [mesh] [gt] Run Step 5 Mesh Evaluation against LiDAR GT PointCloud"
    echo "  nvs [dataset] [model] Run Step 5b NVS Evaluation (PSNR/SSIM/LPIPS on held-out views)"
    echo "  results [--publish PATH] Aggregate SfM/NVS/mesh/timing results into one report"
    echo "  outputs          Inspect and summarize all generated output artifacts"
    echo "  pipeline [method] Run End-to-End Pipeline (SfM -> Train -> Mesh)"
    echo "  help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run_3drc.sh data data/session_419864820"
    echo "  ./scripts/run_3drc.sh sfm hloc"
    echo "  ./scripts/run_3drc.sh train 3dgs"
    echo "  ./scripts/run_3drc.sh train 2dgs"
    echo "  ./scripts/run_3drc.sh train planargs"
    echo "  ./scripts/run_3drc.sh train scaffoldgs"
    echo "  ./scripts/run_3drc.sh sugar"
    echo "  ./scripts/run_3drc.sh milo"
    echo "  ./scripts/run_3drc.sh tsdf"
    echo "  ./scripts/run_3drc.sh eval <mesh_path> [gt_pcd_path]"
    echo "  ./scripts/run_3drc.sh nvs data/<scene> 3dgs/inria_30k"
    echo "  ./scripts/run_3drc.sh results --publish docs/results.md"
    echo "  ./scripts/run_3drc.sh outputs"
    echo "  ./scripts/run_3drc.sh view hloc"
    echo "  ./scripts/run_3drc.sh pipeline hloc"
}

COMMAND=${1:-"help"}

case "$COMMAND" in
    data)
        ARG_DATASET="${2:-}"
        log_header "Step 0b: YUV In-Memory Data Load"
        ./scripts/00b_data_yuv.sh ${ARG_DATASET:+"$ARG_DATASET"}
        ;;
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
        log_header "Step 1: Camera Pose Estimation (Method: ${METHOD})"
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
            log_header "Step 2b: PlanarGS Training"
            ./scripts/02b_train_planargs.sh ${ARG_DATASET:+"$ARG_DATASET"}
        elif [ "$BACKEND" = "2dgs" ]; then
            log_header "Step 2c: 2D Gaussian Splatting Training"
            ./scripts/02c_train_2dgs.sh ${ARG_DATASET:+"$ARG_DATASET"}
        elif [ "$BACKEND" = "scaffoldgs" ]; then
            log_header "Step 2d: Scaffold-GS Training"
            ./scripts/02d_train_scaffoldgs.sh ${ARG_DATASET:+"$ARG_DATASET"}
        elif [ "$BACKEND" = "mipsplatting" ]; then
            log_header "Step 2e: Mip-Splatting Training"
            ./scripts/02e_train_mipsplatting.sh ${ARG_DATASET:+"$ARG_DATASET"}
        elif [ "$BACKEND" = "3dgs" ]; then
            log_header "Step 2: 3DGS Training"
            ./scripts/02_train_3dgs.sh ${ARG_DATASET:+"$ARG_DATASET"}
        else
            log_fatal "Unknown training backend: '${BACKEND}' (expected: 3dgs, planargs, 2dgs, scaffoldgs, mipsplatting)"
        fi
        ;;
    2dgs)
        ARG_DATASET="${2:-}"
        log_header "Step 2c: 2D Gaussian Splatting Training"
        ./scripts/02c_train_2dgs.sh ${ARG_DATASET:+"$ARG_DATASET"}
        ;;
    mipsplatting)
        ARG_DATASET="${2:-}"
        log_header "Step 2e: Mip-Splatting Training"
        ./scripts/02e_train_mipsplatting.sh ${ARG_DATASET:+"$ARG_DATASET"}
        ;;
    planargs)
        ARG_DATASET="${2:-}"
        log_header "Step 2b: PlanarGS Training"
        ./scripts/02b_train_planargs.sh ${ARG_DATASET:+"$ARG_DATASET"}
        ;;
    scaffoldgs)
        ARG_DATASET="${2:-}"
        log_header "Step 2d: Scaffold-GS Training"
        ./scripts/02d_train_scaffoldgs.sh ${ARG_DATASET:+"$ARG_DATASET"}
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
        ./scripts/03_train_sugar.sh ${ARG_DATASET:+"$ARG_DATASET"} ${ARG_SOURCE_MODEL:+"$ARG_SOURCE_MODEL"} "${GS_ITERATIONS:-30000}"
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
    nvs)
        ARG1="${2:-}"
        ARG2="${3:-}"
        ARG_DATASET=""
        ARG_MODEL=""
        if [ -n "$ARG1" ]; then
            # Model directories are also directories, so detect datasets by their layout
            if [[ "$ARG1" == data/* ]] || [ -d "$ARG1/sparse" ] || [ -d "$ARG1/raw_images" ]; then
                ARG_DATASET="$ARG1"
                [ -n "$ARG2" ] && ARG_MODEL="$ARG2"
            else
                ARG_MODEL="$ARG1"
                [ -n "$ARG2" ] && ARG_DATASET="$ARG2"
            fi
        fi
        log_header "Step 5b: NVS Evaluation"
        ./scripts/05_eval_nvs.sh "${ARG_DATASET:-$DATA_DIR}" ${ARG_MODEL:+"$ARG_MODEL"}
        ;;
    results)
        shift
        "$PYTHON_BIN" src/utils/collect_results.py "$@"
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
