#!/bin/bash
set -euo pipefail

# [Step 04] Gaussian Grouping Training
# Joint Reconstruction and Segmentation Lifted from 2D SAM

CONFIG_PATH="configs/default_config.sh"
if [ -f "$CONFIG_PATH" ]; then
    source "$CONFIG_PATH"
else
    echo "[FATAL] Configuration file not found at $CONFIG_PATH!"
    exit 1
fi

source "$(dirname "$0")/utils/activate_backend.sh"
activate_backend grouping

PROJECT_ROOT=$(pwd)

# Determine dataset source
TARGET_DATASET="${1:-$GROUPING_DATASET}"
SCENE_NAME=$(basename "$TARGET_DATASET")

# Resolve absolute dataset path
if [ -d "$TARGET_DATASET" ]; then
    DATASET_ABS_PATH="$(cd "$TARGET_DATASET" && pwd)"
else
    DATASET_ABS_PATH="$(pwd)/data/$TARGET_DATASET"
fi

# Store results with the other stages under outputs/<scene>/ instead of inside the submodule
MODEL_OUTPUT="$PROJECT_ROOT/${OUTPUT_DIR}/${SCENE_NAME}/grouping"
mkdir -p "$MODEL_OUTPUT"

LEGACY_OUTPUT="$PROJECT_ROOT/third_party/gaussian-grouping/output/${SCENE_NAME}"
if [ -d "$LEGACY_OUTPUT" ]; then
    log_warn "Found previous Gaussian Grouping results at ${LEGACY_OUTPUT}; new results are written to ${MODEL_OUTPUT}."
fi

ACTIVE_SFM="unknown"
if [ -L "${DATASET_ABS_PATH}/sparse/0" ]; then
    ACTIVE_SFM=$(readlink "${DATASET_ABS_PATH}/sparse/0" | xargs basename)
fi

# Move to gaussian-grouping directory
cd third_party/gaussian-grouping || exit 1

# Execute Training
log_info "Starting Gaussian Grouping Training (Scene: ${SCENE_NAME}, Active SfM: ${ACTIVE_SFM})..."
ITER_PARAM="${GS_ITERATIONS:-30000}"
bench_start
python train.py -s "$DATASET_ABS_PATH" -r "$GROUPING_DOWNSAMPLE" -m "$MODEL_OUTPUT" --config_file config/gaussian_dataset/train.json --iterations "$ITER_PARAM"

cd "$PROJECT_ROOT"
bench_record "grouping_train" "$SCENE_NAME" "$MODEL_OUTPUT" \
    "iterations=$ITER_PARAM" "downsample_rate=$GROUPING_DOWNSAMPLE" "active_sfm=$ACTIVE_SFM"

# Dynamically record execution provenance metadata
cat <<EOF > "$MODEL_OUTPUT/pipeline_meta.json"
{
  "stage": "Stage 4 (Segmentation)",
  "engine": "Gaussian Grouping (SAM Identity Encoding)",
  "source_dataset": "$DATASET_ABS_PATH",
  "active_sfm": "$ACTIVE_SFM",
  "downsample_rate": $GROUPING_DOWNSAMPLE,
  "iterations": $ITER_PARAM,
  "timestamp": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF

log_ok "Gaussian Grouping Training Completed! Results saved to $MODEL_OUTPUT"
