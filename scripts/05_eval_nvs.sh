#!/bin/bash
set -euo pipefail

# [Step 05b] Novel View Synthesis (NVS) Evaluation
# Renders held-out test views of a trained Gaussian model and computes PSNR / SSIM / LPIPS
# with the backend's own render.py and metrics.py (writes results.json in the model directory).
# Requires the model to be trained with GS_EVAL_MODE="true" (LLFF-style 1/8 holdout split).

CONFIG_PATH="configs/default_config.sh"
if [ -f "$CONFIG_PATH" ]; then
    source "$CONFIG_PATH"
else
    echo "[FATAL] Configuration file not found at $CONFIG_PATH!"
    exit 1
fi

PROJECT_ROOT=$(pwd)
INPUT_DATASET="${1:-$DATA_DIR}"
SCENE_NAME=$(basename "$INPUT_DATASET")

# Source model: exact path or subdirectory under outputs/<scene>/ (default: 3dgs/inria_30k)
MODEL_ARG="${2:-3dgs/inria_30k}"
if [ -d "$MODEL_ARG" ]; then
    MODEL_DIR="$MODEL_ARG"
elif [ -d "${OUTPUT_DIR}/${SCENE_NAME}/${MODEL_ARG}" ]; then
    MODEL_DIR="${OUTPUT_DIR}/${SCENE_NAME}/${MODEL_ARG}"
else
    log_fatal "Model directory not found: '${MODEL_ARG}' (checked as exact path and in ${OUTPUT_DIR}/${SCENE_NAME}/${MODEL_ARG})"
fi
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"

# Backend resolution from the output layout; override with NVS_BACKEND=3dgs|planargs|2dgs|scaffoldgs|mipsplatting
BACKEND="${NVS_BACKEND:-}"
if [ -z "$BACKEND" ]; then
    case "$MODEL_DIR" in
        */3dgs/planargs*) BACKEND="planargs" ;;
        */2dgs*)          BACKEND="2dgs" ;;
        */scaffoldgs*)    BACKEND="scaffoldgs" ;;
        */mipsplatting*)  BACKEND="mipsplatting" ;;
        */3dgs/*)         BACKEND="3dgs" ;;
        *) log_fatal "Cannot infer backend from '${MODEL_DIR}'. Set NVS_BACKEND=3dgs|planargs|2dgs|scaffoldgs|mipsplatting." ;;
    esac
fi

RENDER_EXTRA_ARGS=()
METRICS_EXTRA_ARGS=()
case "$BACKEND" in
    3dgs)       ENV_NAME="gs_train";    REPO_DIR="third_party/gaussian-splatting" ;;
    planargs)   ENV_NAME="gs_train";    REPO_DIR="third_party/PlanarGS" ;;
    2dgs)       ENV_NAME="gs_train";    REPO_DIR="third_party/2d-gaussian-splatting"; RENDER_EXTRA_ARGS+=("--skip_mesh") ;;
    scaffoldgs) ENV_NAME="gs_scaffold"; REPO_DIR="third_party/scaffold-gs" ;;
    mipsplatting) ENV_NAME="gs_mipsplatting"; REPO_DIR="third_party/mip-splatting" ;;
    *) log_fatal "Unknown NVS backend: ${BACKEND}" ;;
esac

# A model trained without --eval has no test cameras; its numbers would be training-view fits.
CFG_ARGS="${MODEL_DIR}/cfg_args"
[ -f "$CFG_ARGS" ] || log_fatal "cfg_args not found in ${MODEL_DIR}; is this a trained model directory?"
if ! grep -q "eval=True" "$CFG_ARGS"; then
    log_fatal "Model at ${MODEL_DIR} was trained without a holdout split. Retrain with GS_EVAL_MODE=\"true\" to evaluate NVS metrics."
fi

# Mip-Splatting names its render folders by training resolution (test_preds_<r>), so metrics.py needs it
if [ "$BACKEND" = "mipsplatting" ]; then
    TRAIN_RES=$(grep -o "resolution=-\?[0-9]\+" "$CFG_ARGS" | head -n 1 | cut -d= -f2)
    [ -n "$TRAIN_RES" ] || log_fatal "Training resolution not found in ${CFG_ARGS}"
    METRICS_EXTRA_ARGS+=("-r" "$TRAIN_RES")
fi

CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
set +u
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u
[ "${CONDA_DEFAULT_ENV:-}" = "$ENV_NAME" ] || log_fatal "Conda environment '${ENV_NAME}' activation failed!"

"$(dirname "$0")/utils/verify_patches.sh" || { echo "[FATAL] Patch verification failed!"; exit 1; }

log_info "Evaluating NVS metrics (Scene: ${SCENE_NAME}, Backend: ${BACKEND}, Model: ${MODEL_DIR})..."

# Run from the repository directory: cfg_args stores the source path as given at training time,
# and PlanarGS trains from its own directory with absolute paths.
if [ "$BACKEND" = "planargs" ]; then
    RUN_DIR="$PROJECT_ROOT/$REPO_DIR"
else
    RUN_DIR="$PROJECT_ROOT"
fi

bench_start
(
    cd "$RUN_DIR"
    python "$PROJECT_ROOT/$REPO_DIR/render.py" \
        -m "$MODEL_DIR" \
        --skip_train \
        --data_device "${DATA_DEVICE:-cpu}" \
        "${RENDER_EXTRA_ARGS[@]}"
    python "$PROJECT_ROOT/$REPO_DIR/metrics.py" -m "$MODEL_DIR" "${METRICS_EXTRA_ARGS[@]}"
)

[ -f "$MODEL_DIR/results.json" ] || log_fatal "metrics.py did not produce ${MODEL_DIR}/results.json"

bench_record "nvs_eval_${BACKEND}" "$SCENE_NAME" "$MODEL_DIR" "backend=$BACKEND"

log_ok "NVS evaluation completed. Metrics saved to ${MODEL_DIR}/results.json"
