#!/bin/bash
set -euo pipefail

# [Step 01] hloc SuperPoint+SuperGlue SfM Pipeline
# High-precision camera pose estimation for large-scale high-resolution datasets

# Usage: 01_sfm_hloc.sh [method] [dataset]; the config picks the scene from the dataset
CONFIG_PATH="configs/default_config.sh"
if [ -f "$CONFIG_PATH" ]; then
    source "$CONFIG_PATH" ${2:+"$2"}
else
    echo "[FATAL] Configuration file not found at $CONFIG_PATH!"
    exit 1
fi

CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    set +u
    conda activate 3drc
    set -u
    [ "${CONDA_DEFAULT_ENV:-}" = "3drc" ] || { echo "[FATAL] Conda environment '3drc' not found or activation failed. Run: ./scripts/00_setup_environment.sh --env 3drc"; exit 1; }
fi

METHOD="${1:-$SFM_METHOD}"
INPUT_DATASET="${2:-$DATA_DIR}"

IMAGE_DIR_TARGET="${INPUT_DATASET}/raw_images"
if [ ! -d "$IMAGE_DIR_TARGET" ]; then
    IMAGE_DIR_TARGET="$IMAGE_DIR"
fi

SPARSE_BASE_DIR="${INPUT_DATASET}/sparse"
SPARSE_METHOD_DIR="${SPARSE_BASE_DIR}/${METHOD}"

bench_start
case "$METHOD" in
    sfm)
        mkdir -p "$SPARSE_METHOD_DIR"
        echo "Starting Traditional SIFT SfM Pipeline..."
        python -m src.sfm.sfm_pipeline \
            --image_dir "$IMAGE_DIR_TARGET" \
            --output_dir "${INPUT_DATASET}"
        ;;
    fastmap)
        mkdir -p "$SPARSE_METHOD_DIR"
        echo "Starting Super-Fast FastMap GPU SfM Pipeline..."
        python -m src.sfm.fastmap_pipeline \
            --image_dir "$IMAGE_DIR_TARGET" \
            --output_dir "${INPUT_DATASET}"
        ;;
    vggt)
        mkdir -p "$SPARSE_METHOD_DIR"
        echo "Starting VGGT-Omega feed-forward pose estimation..."
        if [ ! -d "${INPUT_DATASET}/images" ] && [ -d "${INPUT_DATASET}/raw_images" ]; then
            ln -s raw_images "${INPUT_DATASET}/images"
        fi
        python third_party/vggt/demo_colmap.py \
            --scene_dir "$INPUT_DATASET"
        ;;
    hloc)
        mkdir -p "$SPARSE_METHOD_DIR"
        echo "Starting High-Precision hloc SfM Pipeline..."
        OVERWRITE_FLAG=""
        if [ "${OVERWRITE:-0}" = "1" ]; then
            OVERWRITE_FLAG="--overwrite"
            echo "[INFO] Cache overwrite enabled for fresh SfM reconstruction."
        fi
        RECONSTRUCT_ONLY_FLAG=""
        if [ "${RECONSTRUCT_ONLY:-0}" = "1" ]; then
            RECONSTRUCT_ONLY_FLAG="--reconstruct-only"
            echo "[INFO] Reconstruct-only mode enabled for sparse model re-generation."
        fi
        python -m src.sfm.hloc_pipeline \
            --image_dir "$IMAGE_DIR_TARGET" \
            --output_dir "${INPUT_DATASET}/cache" \
            --strategy "${SFM_STRATEGY:-sequential}" \
            --overlap "${SFM_OVERLAP:-100}" \
            --retrieval_k "${SFM_RETRIEVAL_K:-30}" \
            ${OVERWRITE_FLAG} \
            ${RECONSTRUCT_ONLY_FLAG}
        ;;
    *)
        echo "[FATAL] Unknown SfM method: $METHOD" >&2
        exit 1
        ;;
esac

# Move/copy outputs to method specific directory
if [ -d "${INPUT_DATASET}/cache/sfm" ]; then
    cp -r "${INPUT_DATASET}/cache/sfm"/* "$SPARSE_METHOD_DIR/" 2>/dev/null || true
    # If COLMAP created submodels in models/, promote the largest submodel by registered image count (num_reg_images)
    if [ -d "${INPUT_DATASET}/cache/sfm/models" ]; then
        PROMOTION_INFO=$(python -m src.sfm.sparse_outputs largest-model "${INPUT_DATASET}/cache/sfm" 2>/dev/null || true)

        if [ -n "$PROMOTION_INFO" ]; then
            BEST_MODEL=$(echo "$PROMOTION_INFO" | cut -d'|' -f1)
            REG_COUNT=$(echo "$PROMOTION_INFO" | cut -d'|' -f2)
            if [ "$BEST_MODEL" = "root" ]; then
                log_info "Largest model is the sfm root ($REG_COUNT registered images); keeping it in ${SPARSE_METHOD_DIR}."
            elif [ -n "$BEST_MODEL" ] && [ -d "$BEST_MODEL" ]; then
                log_info "Promoting submodel $(basename "$BEST_MODEL") ($REG_COUNT registered images) to root of ${SPARSE_METHOD_DIR}..."
                # Clear residual sparse files before promoting largest model
                rm -f "${SPARSE_METHOD_DIR}/cameras.bin" "${SPARSE_METHOD_DIR}/images.bin" "${SPARSE_METHOD_DIR}/points3D.bin" "${SPARSE_METHOD_DIR}/points3D.ply" 2>/dev/null || true
                cp -f "$BEST_MODEL"/* "$SPARSE_METHOD_DIR/" || log_fatal "Failed to promote model from $BEST_MODEL to $SPARSE_METHOD_DIR"
            fi
        fi
    fi
fi
if [ -d "${INPUT_DATASET}/0" ]; then
    mv "${INPUT_DATASET}/0"/* "$SPARSE_METHOD_DIR/" 2>/dev/null || true
    rmdir "${INPUT_DATASET}/0" 2>/dev/null || true
fi
# hloc writes only under cache/ and ${SPARSE_METHOD_DIR}; *.bin left in sparse/ come from training stages and are stale.
if [ "$METHOD" != "hloc" ] && [ -f "${SPARSE_BASE_DIR}/cameras.bin" ]; then
    mv "${SPARSE_BASE_DIR}"/*.bin "$SPARSE_METHOD_DIR/" 2>/dev/null || true
    mv "${SPARSE_BASE_DIR}"/*.ply "$SPARSE_METHOD_DIR/" 2>/dev/null || true
fi

# Verify reconstruction integrity before updating symlink
HAS_CAMERAS=false
if [ -f "$SPARSE_METHOD_DIR/cameras.bin" ] || [ -f "$SPARSE_METHOD_DIR/cameras.txt" ]; then
    HAS_CAMERAS=true
fi

HAS_IMAGES=false
if [ -f "$SPARSE_METHOD_DIR/images.bin" ] || [ -f "$SPARSE_METHOD_DIR/images.txt" ]; then
    HAS_IMAGES=true
fi

HAS_POINTS=false
if [ -f "$SPARSE_METHOD_DIR/points3D.bin" ] || [ -f "$SPARSE_METHOD_DIR/points3D.txt" ]; then
    HAS_POINTS=true
fi

if [ "$HAS_CAMERAS" = false ] || [ "$HAS_IMAGES" = false ] || [ "$HAS_POINTS" = false ]; then
    echo "[FATAL] SfM (${METHOD}) failed: Required reconstruction files (cameras, images, points3D) missing in $SPARSE_METHOD_DIR" >&2
    exit 1
fi

# Write metadata json
python -m src.sfm.sparse_outputs write-info "$SPARSE_METHOD_DIR" --image_dir "$IMAGE_DIR_TARGET" --method "$METHOD" \
    --strategy "${SFM_STRATEGY:-}" --overlap "${SFM_OVERLAP:-}" --retrieval_k "${SFM_RETRIEVAL_K:-}"

# Create / Update active symlink sparse/0 -> method only after verification succeeds
cd "$SPARSE_BASE_DIR"
rm -rf 0
ln -s "$METHOD" 0
cd - > /dev/null

bench_record "sfm_${METHOD}" "$(basename "$INPUT_DATASET")" "$SPARSE_METHOD_DIR" \
    "method=${METHOD}" "strategy=${SFM_STRATEGY}"

# Trajectory and per-image quality report; a plotting failure must not fail a finished reconstruction
python -m src.sfm.sfm_report "$SPARSE_METHOD_DIR" --image_dir "$IMAGE_DIR_TARGET" || \
    log_warn "SfM report generation failed; rerun with: python -m src.sfm.sfm_report $INPUT_DATASET --method $METHOD"

echo "SfM Reconstruction (${METHOD}) Completed! Saved to ${SPARSE_METHOD_DIR} (Active symlink: ${SPARSE_BASE_DIR}/0 -> ${METHOD})"
