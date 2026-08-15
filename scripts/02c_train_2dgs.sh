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
[ "${CONDA_DEFAULT_ENV:-}" = "gs_train" ] || { echo "[FATAL] Conda environment 'gs_train' not found or activation failed. Run: ./scripts/00_setup_environment.sh --env gs_train"; exit 1; }

# Run patch & environment verification inside activated environment
"$(dirname "$0")/utils/verify_patches.sh" || { echo "[FATAL] Patch verification failed!"; exit 1; }

INPUT_DATASET="${1:-$DATA_DIR}"

if [ -f "${INPUT_DATASET}/sparse/0/cameras.bin" ] || [ -f "${INPUT_DATASET}/sparse/0/cameras.txt" ] || \
   [ -f "${INPUT_DATASET}/sparse/cameras.bin" ] || [ -f "${INPUT_DATASET}/sparse/cameras.txt" ]; then
    echo "Using existing pre-computed dataset structure at: ${INPUT_DATASET}"
    TARGET_DATA_DIR="${INPUT_DATASET}"
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

CHECKPOINTS_LIST=""
if [ -n "${GS_CHECKPOINT_ITERATIONS:-}" ]; then
    CHECKPOINTS_LIST="$GS_CHECKPOINT_ITERATIONS"
elif [ -n "${GS_CHECKPOINT_INTERVAL:-}" ] && [ "${GS_CHECKPOINT_INTERVAL:-0}" -gt 0 ]; then
    MAX_ITER="${GS_ITERATIONS:-30000}"
    INTERVAL="${GS_CHECKPOINT_INTERVAL:-10000}"
    CHECKPOINTS_LIST=$(seq "$INTERVAL" "$INTERVAL" "$MAX_ITER" | tr '\n' ' ')
fi

if [ -n "$CHECKPOINTS_LIST" ]; then
    mapfile -t CK < <(echo "$CHECKPOINTS_LIST" | tr ' ' '\n' | grep -v '^$')
    if [ "${#CK[@]}" -gt 0 ]; then
        EXTRA_TRAIN_ARGS+=("--checkpoint_iterations" "${CK[@]}")
    fi
fi

# Deduplicate & upper-bound filter --save_iterations against GS_ITERATIONS
MAX_ITER="${GS_ITERATIONS:-30000}"
mapfile -t SAVE_ITERS < <(printf '%s\n' "$MAX_ITER" | sort -un)
EXTRA_TRAIN_ARGS+=("--save_iterations" "${SAVE_ITERS[@]}")



# Auto-resume from latest checkpoint if available (with staleness invalidation guard)
LATEST_CHKPNT=$(ls -v "${MODEL_OUTPUT}"/checkpoints/chkpnt*.pth "${MODEL_OUTPUT}"/chkpnt*.pth 2>/dev/null | tail -n 1 || true)
SPARSE_PTS="${TARGET_DATA_DIR}/sparse/0/points3D.bin"
if [ ! -f "$SPARSE_PTS" ]; then
    SPARSE_PTS="${TARGET_DATA_DIR}/sparse/0/points3D.txt"
fi

if [ -n "$LATEST_CHKPNT" ]; then
    if [ -f "$SPARSE_PTS" ] && [ "$SPARSE_PTS" -nt "$LATEST_CHKPNT" ]; then
        log_warn "Sparse point cloud ($SPARSE_PTS) is newer than latest checkpoint ($(basename "$LATEST_CHKPNT")). Invalidating outdated checkpoint and starting fresh training."
        LATEST_CHKPNT=""
    else
        log_info "Auto-resuming 2DGS training from latest checkpoint: $(basename "$LATEST_CHKPNT")"
        EXTRA_TRAIN_ARGS+=("--start_checkpoint" "$LATEST_CHKPNT")
    fi
fi

python third_party/2d-gaussian-splatting/train.py \
    -s "$TARGET_DATA_DIR" \
    -m "$MODEL_OUTPUT" \
    -r "$DOWNSAMPLE_RATE" \
    --data_device "$DATA_DEVICE" \
    --iterations "${GS_ITERATIONS:-30000}" \
    "${EXTRA_TRAIN_ARGS[@]}"

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
