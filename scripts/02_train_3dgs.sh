#!/bin/bash
set -euo pipefail

# [Step 02] Inria 3D Gaussian Splatting Training
# Blackwell (RTX 50) optimization and high-density parameters

source "$(dirname "$0")/utils/train_stage.sh"

train_env 3dgs "$@"
train_dataset 3dgs/inria_30k "$@"
train_extra_args

log_info "Starting High-Density Original 3DGS Training (Scene: $SCENE_NAME)..."
bench_start
python third_party/gaussian-splatting/train.py \
    -s "$TARGET_DATA_DIR" \
    --model_path "$MODEL_OUTPUT" \
    -r "$DOWNSAMPLE_RATE" \
    --densify_grad_threshold "$DENSIFY_GRAD_THRESHOLD" \
    --data_device "$DATA_DEVICE" \
    --iterations "${GS_ITERATIONS:-30000}" \
    "${EXTRA_TRAIN_ARGS[@]}"

train_finish 3dgs_inria_train "Original 3DGS (inria_30k)"
