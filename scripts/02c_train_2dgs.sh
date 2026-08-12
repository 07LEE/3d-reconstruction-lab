#!/bin/bash
set -euo pipefail

# [Step 02c] 2D Gaussian Splatting Training (Surfel Representation)
# Blackwell (RTX 50) optimized with diff-surfel-rasterization

CONFIG_PATH="configs/default_config.sh"
if [ -f "$CONFIG_PATH" ]; then
    source "$CONFIG_PATH"
fi

CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
set +u
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate gs_train
set -u

# Run patch & environment verification inside activated environment
"$(dirname "$0")/utils/verify_patches.sh" || { echo "[FATAL] Patch verification failed!"; exit 1; }

INPUT_DATASET="${1:-$DATA_DIR}"

if [ -f "${INPUT_DATASET}/sparse/0/cameras.bin" ] || [ -f "${INPUT_DATASET}/sparse/0/cameras.txt" ] || \
   [ -f "${INPUT_DATASET}/sparse/hloc/cameras.bin" ] || [ -f "${INPUT_DATASET}/sparse/hloc/cameras.txt" ] || \
   [ -f "${INPUT_DATASET}/sparse/cameras.bin" ] || [ -f "${INPUT_DATASET}/sparse/cameras.txt" ]; then
    echo "Using existing pre-computed dataset structure at: ${INPUT_DATASET}"
    TARGET_DATA_DIR="${INPUT_DATASET}"
    if [ -f "${INPUT_DATASET}/sparse/hloc/cameras.bin" ] && [ ! -d "${INPUT_DATASET}/sparse/0" ]; then
        mkdir -p "${INPUT_DATASET}/sparse/0"
        cp -r "${INPUT_DATASET}/sparse/hloc/"* "${INPUT_DATASET}/sparse/0/"
    fi
else
    # Guard: Ensure SfM reconstruction model exists before proceeding
    SRC="${INPUT_DATASET}/cache/sfm"
    if [ ! -f "$SRC/cameras.bin" ] && [ ! -f "$SRC/cameras.txt" ] && [ ! -f "$SRC/models/0/cameras.bin" ]; then
        echo "[FATAL] SfM model not found in ${INPUT_DATASET}/sparse/0, ${INPUT_DATASET}/sparse/hloc, or $SRC"
        exit 1
    fi

    echo "Updating SfM data in ${INPUT_DATASET}/sparse/0..."
    mkdir -p "${INPUT_DATASET}/sparse/0"

    if [ -d "$SRC/models/0" ]; then
        cp -r "$SRC/models/0/"* "${INPUT_DATASET}/sparse/0/"
    else
        cp -r "$SRC/"* "${INPUT_DATASET}/sparse/0/" 2>/dev/null || true
    fi

    TARGET_DATA_DIR="${INPUT_DATASET}"
fi

# Ensure images directory/link exists for 2DGS dataloader
if [ ! -d "${TARGET_DATA_DIR}/images" ] && [ -d "${TARGET_DATA_DIR}/raw_images" ]; then
    echo "Creating images symlink in ${TARGET_DATA_DIR}/images..."
    ln -s raw_images "${TARGET_DATA_DIR}/images" 2>/dev/null || cp -r "${TARGET_DATA_DIR}/raw_images" "${TARGET_DATA_DIR}/images"
fi

SCENE_NAME=$(basename "$TARGET_DATA_DIR")
MODEL_OUTPUT="${OUTPUT_DIR}/${SCENE_NAME}/2dgs"
mkdir -p "$MODEL_OUTPUT"

# 2DGS Training Execution
echo "Starting 2D Gaussian Splatting Training (Scene: $SCENE_NAME)..."

ACTIVE_SFM="unknown"
if [ -L "${TARGET_DATA_DIR}/sparse/0" ]; then
    ACTIVE_SFM=$(readlink "${TARGET_DATA_DIR}/sparse/0" | xargs basename)
fi

EXTRA_TRAIN_ARGS=()
if [ "${GS_EVAL_MODE:-false}" = "true" ]; then
    EXTRA_TRAIN_ARGS+=("--eval")
fi

python third_party/2d-gaussian-splatting/train.py \
    -s "$TARGET_DATA_DIR" \
    -m "$MODEL_OUTPUT" \
    -r "$DOWNSAMPLE_RATE" \
    --data_device "$DATA_DEVICE" \
    "${EXTRA_TRAIN_ARGS[@]}" \
    --iterations "${GS_ITERATIONS:-30000}"

# Dynamically record execution provenance metadata
cat <<EOF > "$MODEL_OUTPUT/pipeline_meta.json"
{
  "stage": "Stage 2 (Training)",
  "engine": "2D Gaussian Splatting (Surfels)",
  "source_dataset": "$TARGET_DATA_DIR",
  "active_sfm": "$ACTIVE_SFM",
  "downsample_rate": $DOWNSAMPLE_RATE,
  "iterations": ${GS_ITERATIONS:-30000},
  "timestamp": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF

echo "2DGS Training Completed! Results saved to ${MODEL_OUTPUT}"
