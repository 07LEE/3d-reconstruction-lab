#!/bin/bash
set -euo pipefail

# [Step 02d] Scaffold-GS (Anchor-based 3D Gaussian Splatting) Training
# Blackwell (RTX 50) optimization and anchor-based high-fidelity parameters

source "$(dirname "$0")/utils/train_stage.sh"

train_env scaffoldgs "$@"
train_dataset scaffoldgs "$@"
train_extra_args

log_info "Starting Scaffold-GS Anchor-based Training (Scene: $SCENE_NAME)..."
bench_start
python third_party/scaffold-gs/train.py \
    -s "$TARGET_DATA_DIR" \
    --model_path "$MODEL_OUTPUT" \
    --data_device "${DATA_DEVICE:-cpu}" \
    -r "$DOWNSAMPLE_RATE" \
    --iterations "${GS_ITERATIONS:-30000}" \
    "${EXTRA_TRAIN_ARGS[@]}"

train_finish scaffoldgs_train "Scaffold-GS (Anchor-based 3D Gaussian Splatting)" "Stage 2d (Training)"
