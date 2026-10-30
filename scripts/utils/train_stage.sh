#!/usr/bin/env bash
# 3DRC Gaussian Training Stage Helpers
#
# Usage (sourced): source scripts/utils/train_stage.sh
# The 02*_train_*.sh scripts call these in order and run their trainer in between:
#   train_env <backend> [dataset]                      config, backend environment, patch verification
#   train_dataset <output_subdir> [dataset]            SfM model, images link and model output directory
#   train_extra_args                                   holdout, checkpoint, save and resume arguments; stops on a model
#                                                      trained on the other holdout split, exits when it is finished
#   train_finish <bench_stage> <engine> [stage_label]  timing record and pipeline_meta.json
#
# Results are left in globals for the calling script: TRAIN_BACKEND, INPUT_DATASET, TARGET_DATA_DIR,
# SCENE_NAME, MODEL_OUTPUT, ACTIVE_SFM, EXTRA_TRAIN_ARGS and RESUMED_FROM.

_TRAIN_STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_TRAIN_STAGE_DIR/common_logger.sh"
source "$_TRAIN_STAGE_DIR/activate_backend.sh"

# The config picks the scene from its first argument, so the dataset is always passed explicitly
train_env() {
    local backend="$1" dataset="${2:-}"
    local config_path="configs/default_config.sh"

    [ -f "$config_path" ] || log_fatal "Configuration file not found at $config_path!"
    source "$config_path" "$dataset"

    activate_backend "$backend"
    TRAIN_BACKEND="$backend"

    # Run patch & environment verification inside the activated environment
    "$_TRAIN_STAGE_DIR/verify_patches.sh" || log_fatal "Patch verification failed!"
}

train_dataset() {
    local output_subdir="$1" dataset="${2:-}"
    local sfm_src

    INPUT_DATASET="${dataset:-$DATA_DIR}"

    if [ -f "${INPUT_DATASET}/sparse/0/cameras.bin" ] || [ -f "${INPUT_DATASET}/sparse/0/cameras.txt" ] || \
       [ -f "${INPUT_DATASET}/sparse/cameras.bin" ] || [ -f "${INPUT_DATASET}/sparse/cameras.txt" ]; then
        log_info "Using existing pre-computed dataset structure at: ${INPUT_DATASET}"
    else
        # Guard: Ensure SfM reconstruction model exists before cleaning/updating
        sfm_src="${INPUT_DATASET}/cache/sfm"
        if [ ! -f "$sfm_src/cameras.bin" ] && [ ! -f "$sfm_src/cameras.txt" ] && [ ! -f "$sfm_src/models/0/cameras.bin" ]; then
            log_fatal "SfM model not found in ${INPUT_DATASET}/sparse/0 or $sfm_src"
        fi

        # Update SfM data link after verification succeeds
        log_info "Updating SfM data in ${INPUT_DATASET}/sparse/0..."
        mkdir -p "${INPUT_DATASET}/sparse/0"

        if [ -d "$sfm_src/models/0" ]; then
            cp -r "$sfm_src/models/0/"* "${INPUT_DATASET}/sparse/0/"
        else
            cp -r "$sfm_src/"* "${INPUT_DATASET}/sparse/0/" 2>/dev/null || true
        fi
    fi
    TARGET_DATA_DIR="${INPUT_DATASET}"

    # Ensure images directory/link exists for the trainer's dataloader
    if [ -d "${TARGET_DATA_DIR}/raw_images" ]; then
        if [ -d "${TARGET_DATA_DIR}/images" ] && [ ! -L "${TARGET_DATA_DIR}/images" ]; then
            log_info "Removing non-symlink images directory to enforce strict symlink..."
            rm -rf "${TARGET_DATA_DIR}/images"
        fi
        if [ ! -L "${TARGET_DATA_DIR}/images" ]; then
            log_info "Creating strict images symlink in ${TARGET_DATA_DIR}/images..."
            ln -sf raw_images "${TARGET_DATA_DIR}/images" || log_fatal "Failed to create symlink ${TARGET_DATA_DIR}/images -> raw_images"
        fi
    fi

    SCENE_NAME=$(basename "$TARGET_DATA_DIR")
    MODEL_OUTPUT="${OUTPUT_DIR}/${SCENE_NAME}/${output_subdir}"
    mkdir -p "$MODEL_OUTPUT"

    ACTIVE_SFM="unknown"
    if [ -L "${TARGET_DATA_DIR}/sparse/0" ]; then
        ACTIVE_SFM=$(readlink "${TARGET_DATA_DIR}/sparse/0" | xargs basename)
    fi
}

# The holdout flag a trainer recorded in a model directory's cfg_args (True or False), nothing when there is none
_train_cfg_eval() {
    [ -f "$1/cfg_args" ] || return 0
    grep -o 'eval=[A-Za-z]*' "$1/cfg_args" | head -n 1 | cut -d= -f2 || true
}

train_extra_args() {
    local max_iter="${GS_ITERATIONS:-30000}" checkpoints="" interval latest sparse_pts want="False" trained done_iter
    local -a ck

    EXTRA_TRAIN_ARGS=()
    if [ "${GS_EVAL_MODE:-false}" = "true" ]; then
        EXTRA_TRAIN_ARGS+=("--eval")
        want="True"
    fi

    if [ -n "${GS_CHECKPOINT_ITERATIONS:-}" ]; then
        checkpoints="$GS_CHECKPOINT_ITERATIONS"
    elif [ -n "${GS_CHECKPOINT_INTERVAL:-}" ] && [ "${GS_CHECKPOINT_INTERVAL:-0}" -gt 0 ]; then
        interval="${GS_CHECKPOINT_INTERVAL:-10000}"
        checkpoints=$(seq "$interval" "$interval" "$max_iter" | tr '\n' ' ')
    fi

    if [ -n "$checkpoints" ]; then
        mapfile -t ck < <(echo "$checkpoints" | tr ' ' '\n' | grep -v '^$')
        if [ "${#ck[@]}" -gt 0 ]; then
            EXTRA_TRAIN_ARGS+=("--checkpoint_iterations" "${ck[@]}")
        fi
    fi

    # Only the final iteration is saved as a model
    EXTRA_TRAIN_ARGS+=("--save_iterations" "$max_iter")

    RESUMED_FROM=""
    # Auto-resume from latest checkpoint if available (with staleness invalidation guard)
    latest=$(ls -v "${MODEL_OUTPUT}"/checkpoints/chkpnt*.pth "${MODEL_OUTPUT}"/chkpnt*.pth 2>/dev/null | tail -n 1 || true)
    sparse_pts="${TARGET_DATA_DIR}/sparse/0/points3D.bin"
    if [ ! -f "$sparse_pts" ]; then
        sparse_pts="${TARGET_DATA_DIR}/sparse/0/points3D.txt"
    fi

    # The trainers rewrite cfg_args on every start and keep restoring the checkpoint they are given, so a model
    # trained on one holdout split and continued on the other would pass the NVS check with test views in training
    trained=$(_train_cfg_eval "$MODEL_OUTPUT")
    if [ -n "$trained" ] && [ "$trained" != "$want" ] && { [ -n "$latest" ] || compgen -G "${MODEL_OUTPUT}/point_cloud/iteration_*" > /dev/null; }; then
        log_fatal "${MODEL_OUTPUT} already holds a model trained with eval=${trained}, but GS_EVAL_MODE asks for eval=${want}. Training on top of it would mix holdout and training views. Move or delete the directory to train from scratch."
    fi

    if [ -n "$latest" ]; then
        if [ -f "$sparse_pts" ] && [ "$sparse_pts" -nt "$latest" ]; then
            log_warn "Sparse point cloud ($sparse_pts) is newer than latest checkpoint ($(basename "$latest")). Invalidating outdated checkpoint and starting fresh training."
        else
            # A trainer resumed at its last iteration does no training but still rewrites cfg_args
            done_iter=$(basename "$latest" .pth)
            done_iter="${done_iter#chkpnt}"
            if [[ "$done_iter" =~ ^[0-9]+$ ]] && [ "$done_iter" -ge "$max_iter" ]; then
                log_ok "${TRAIN_BACKEND} model in ${MODEL_OUTPUT} is already trained to ${done_iter} iterations ($(basename "$latest")). Nothing to do."
                exit 0
            fi
            log_info "Auto-resuming ${TRAIN_BACKEND} training from latest checkpoint: $(basename "$latest")"
            EXTRA_TRAIN_ARGS+=("--start_checkpoint" "$latest")
            RESUMED_FROM="$(basename "$latest")"
            # torch 2.6+ refuses these checkpoints (they hold numpy scalars) unless loading is allowed; the pipeline wrote them
            export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        fi
    fi
}

train_finish() {
    local bench_stage="$1" engine="$2" stage_label="${3:-Stage 2 (Training)}"

    bench_record "$bench_stage" "$SCENE_NAME" "$MODEL_OUTPUT" \
        "iterations=${GS_ITERATIONS:-30000}" "downsample_rate=$DOWNSAMPLE_RATE" \
        "eval_split=${GS_EVAL_MODE:-false}" "active_sfm=$ACTIVE_SFM" "resumed_from=$RESUMED_FROM"

    # Dynamically record execution provenance metadata
    cat <<EOF > "$MODEL_OUTPUT/pipeline_meta.json"
{
  "stage": "$stage_label",
  "engine": "$engine",
  "source_dataset": "$TARGET_DATA_DIR",
  "active_sfm": "$ACTIVE_SFM",
  "downsample_rate": $DOWNSAMPLE_RATE,
  "iterations": ${GS_ITERATIONS:-30000},
  "timestamp": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF

    log_ok "Training completed (${engine}). Results saved to $MODEL_OUTPUT"
}
