#!/bin/bash
set -euo pipefail

# [Step 02e] Mip-Splatting Training
# 3D Smoothing Filter 3DGS extension with Blackwell (RTX 50) optimization

source "$(dirname "$0")/utils/train_stage.sh"

train_env mipsplatting "$@"
train_dataset mipsplatting "$@"
train_extra_args

log_info "Starting Mip-Splatting Training (Scene: $SCENE_NAME)..."
bench_start
python third_party/mip-splatting/train.py \
    -s "$TARGET_DATA_DIR" \
    --model_path "$MODEL_OUTPUT" \
    --data_device "${DATA_DEVICE:-cpu}" \
    -r "$DOWNSAMPLE_RATE" \
    --iterations "${GS_ITERATIONS:-30000}" \
    "${EXTRA_TRAIN_ARGS[@]}"

train_finish mipsplatting_train "Mip-Splatting (3D Smoothing Filter 3DGS)" "Stage 2e (Training)"
