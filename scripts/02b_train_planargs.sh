#!/bin/bash
set -euo pipefail

# [Step 02b] PlanarGS Indoor Planar-Regularized 3DGS Pipeline
# Selective planar priors for indoor scene geometry optimization

CONFIG_PATH="configs/default_config.sh"
if [ -f "$CONFIG_PATH" ]; then
    source "$CONFIG_PATH"
else
    echo "[FATAL] Configuration file not found at $CONFIG_PATH!"
    exit 1
fi

CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    set +u
    conda activate gs_train
    set -u
    [ "${CONDA_DEFAULT_ENV:-}" = "gs_train" ] || { echo "[FATAL] Conda environment 'gs_train' activation failed!"; exit 1; }
fi

# Run patch & environment verification inside activated gs_train environment
"$(dirname "$0")/utils/verify_patches.sh" || { echo "[FATAL] Patch verification failed!"; exit 1; }

# Source CUDA build environment (GCC 12 / NVCC 12.8)
source "$(dirname "$0")/utils/setup_build_env.sh"

PROJECT_ROOT=$(pwd)
INPUT_DATASET="${1:-$DATA_DIR}"

if [ -f "${INPUT_DATASET}/sparse/0/cameras.bin" ] || [ -f "${INPUT_DATASET}/sparse/0/cameras.txt" ] || \
   [ -f "${INPUT_DATASET}/sparse/cameras.bin" ] || [ -f "${INPUT_DATASET}/sparse/cameras.txt" ]; then
    echo "Using existing pre-computed dataset structure at: ${INPUT_DATASET}"
    TARGET_DATA_DIR="${INPUT_DATASET}"
else
    # Guard: Ensure SfM reconstruction model exists before cleaning/updating
    SRC="${INPUT_DATASET}/cache/sfm"
    if [ ! -f "$SRC/cameras.bin" ] && [ ! -f "$SRC/cameras.txt" ] && [ ! -f "$SRC/models/0/cameras.bin" ]; then
        echo "[FATAL] SfM model not found in ${INPUT_DATASET}/sparse/0 or $SRC"
        exit 1
    fi

    # Update SfM data link after verification succeeds
    echo "Updating SfM data in ${INPUT_DATASET}/sparse/0..."
    mkdir -p "${INPUT_DATASET}/sparse/0"

    if [ -d "$SRC/models/0" ]; then
        cp -r "$SRC/models/0/"* "${INPUT_DATASET}/sparse/0/"
    else
        cp -r "$SRC/"* "${INPUT_DATASET}/sparse/0/" 2>/dev/null || true
    fi

    TARGET_DATA_DIR="${INPUT_DATASET}"
fi

# Ensure images directory/link exists for PlanarGS dataloader
if [ -d "${TARGET_DATA_DIR}/raw_images" ]; then
    if [ -d "${TARGET_DATA_DIR}/images" ] && [ ! -L "${TARGET_DATA_DIR}/images" ]; then
        echo "[INFO] Removing non-symlink images directory to enforce strict symlink..."
        rm -rf "${TARGET_DATA_DIR}/images"
    fi
    if [ ! -L "${TARGET_DATA_DIR}/images" ]; then
        echo "[INFO] Creating strict images symlink in ${TARGET_DATA_DIR}/images..."
        ln -sf raw_images "${TARGET_DATA_DIR}/images" || { echo "[FATAL] Failed to create symlink ${TARGET_DATA_DIR}/images -> raw_images"; exit 1; }
    fi
fi

SCENE_NAME=$(basename "$INPUT_DATASET")
MODEL_OUTPUT="$PROJECT_ROOT/${OUTPUT_DIR}/${SCENE_NAME}/3dgs/planargs"
mkdir -p "$MODEL_OUTPUT"

ACTIVE_SFM="unknown"
if [ -L "${TARGET_DATA_DIR}/sparse/0" ]; then
    ACTIVE_SFM=$(readlink "${TARGET_DATA_DIR}/sparse/0" | xargs basename)
fi

# Execute PlanarGS Training
echo "Starting PlanarGS Training (Scene: $SCENE_NAME, Dataset: $INPUT_DATASET)..."
cd third_party/PlanarGS || exit 1

EXTRA_TRAIN_ARGS=()
if [ "${GS_EVAL_MODE:-false}" = "true" ]; then
    EXTRA_TRAIN_ARGS+=("--eval")
fi

CHECKPOINTS_LIST=""
if [ -n "${GS_CHECKPOINT_ITERATIONS:-}" ]; then
    CHECKPOINTS_LIST="$GS_CHECKPOINT_ITERATIONS"
elif [ -n "${GS_CHECKPOINT_INTERVAL:-}" ] && [ "${GS_CHECKPOINT_INTERVAL:-0}" -gt 0 ]; then
    MAX_ITER="${GS_ITERATIONS:-30000}"
    INTERVAL="${GS_CHECKPOINT_INTERVAL:-10000}"
    CHECKPOINTS_LIST=$(seq "$INTERVAL" "$INTERVAL" "$MAX_ITER" | tr '\n' ' ')
fi

if [ -n "$CHECKPOINTS_LIST" ]; then
    EXTRA_TRAIN_ARGS+=("--checkpoint_iterations" $CHECKPOINTS_LIST)
fi

python train.py \
    -s "$PROJECT_ROOT/$TARGET_DATA_DIR" \
    -m "$MODEL_OUTPUT" \
    -r "$DOWNSAMPLE_RATE" \
    --data_device "$DATA_DEVICE" \
    --densify_grad_threshold "${DENSIFY_GRAD_THRESHOLD:-0.0002}" \
    --iterations "${GS_ITERATIONS:-30000}" \
    "${EXTRA_TRAIN_ARGS[@]}"

# Dynamically record execution provenance metadata
cat <<EOF > "$MODEL_OUTPUT/pipeline_meta.json"
{
  "stage": "Stage 2 (Training)",
  "engine": "PlanarGS (Planar-Regularized 3DGS)",
  "source_dataset": "$TARGET_DATA_DIR",
  "active_sfm": "$ACTIVE_SFM",
  "downsample_rate": $DOWNSAMPLE_RATE,
  "iterations": ${GS_ITERATIONS:-30000},
  "timestamp": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF

echo "PlanarGS Training Completed! Models saved at $MODEL_OUTPUT"
