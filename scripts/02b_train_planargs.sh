#!/bin/bash
set -euo pipefail

# [Step 02b] PlanarGS Indoor Planar-Regularized 3DGS Pipeline
# Selective planar priors for indoor scene geometry optimization

source "$(dirname "$0")/utils/train_stage.sh"

train_env planargs "$@"

# Source CUDA build environment (nvcc and host compiler)
source "$(dirname "$0")/utils/setup_build_env.sh"

PROJECT_ROOT=$(pwd)
train_dataset 3dgs/planargs "$@"
# PlanarGS runs from its own directory, so the model path it receives must be absolute
MODEL_OUTPUT="$PROJECT_ROOT/$MODEL_OUTPUT"

# Ensure PlanarGS dataset_readers finds images.bin and cameras.bin directly under sparse/
if [ -f "${TARGET_DATA_DIR}/sparse/0/images.bin" ]; then
    cp "${TARGET_DATA_DIR}/sparse/0/"*.bin "${TARGET_DATA_DIR}/sparse/" 2>/dev/null || true
fi

train_extra_args

# Execute PlanarGS Training
log_info "Starting PlanarGS Training (Scene: $SCENE_NAME, Dataset: $INPUT_DATASET)..."

PYTHON_BIN="${CONDA_PREFIX:-}/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

bench_start
(
    cd third_party/PlanarGS
    "$PYTHON_BIN" train.py \
        -s "$PROJECT_ROOT/$TARGET_DATA_DIR" \
        -m "$MODEL_OUTPUT" \
        -r "$DOWNSAMPLE_RATE" \
        --data_device "$DATA_DEVICE" \
        --densify_grad_threshold "${DENSIFY_GRAD_THRESHOLD:-0.0002}" \
        --iterations "${GS_ITERATIONS:-30000}" \
        "${EXTRA_TRAIN_ARGS[@]}"
)

train_finish 3dgs_planargs_train "PlanarGS (Planar-Regularized 3DGS)"
