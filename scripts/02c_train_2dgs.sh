#!/bin/bash
set -euo pipefail

# [Step 02c] 2D Gaussian Splatting Training (Surfel Representation)
# Blackwell (RTX 50) optimized with diff-surfel-rasterization

source "$(dirname "$0")/utils/train_stage.sh"

train_env 2dgs "$@"
train_dataset 2dgs "$@"
train_extra_args

log_info "Starting 2D Gaussian Splatting Training (Scene: $SCENE_NAME)..."
bench_start
python third_party/2d-gaussian-splatting/train.py \
    -s "$TARGET_DATA_DIR" \
    -m "$MODEL_OUTPUT" \
    -r "$DOWNSAMPLE_RATE" \
    --data_device "$DATA_DEVICE" \
    --iterations "${GS_ITERATIONS:-30000}" \
    "${EXTRA_TRAIN_ARGS[@]}"

train_finish 2dgs_train "2D Gaussian Splatting (Surfels)"
