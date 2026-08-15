#!/bin/bash
set -euo pipefail

# [Step 03] SuGaR Mesh Reconstruction Pipeline
# Surface-Aligned Gaussian Splatting and high-quality mesh extraction

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
    conda activate gs_sugar
    set -u
    [ "${CONDA_DEFAULT_ENV:-}" = "gs_sugar" ] || { echo "[FATAL] Conda environment 'gs_sugar' not found or activation failed. Run: ./scripts/00_setup_environment.sh --env gs_sugar"; exit 1; }
fi

# Get project root
PROJECT_ROOT=$(pwd)

# Move to sugar directory
cd third_party/sugar || exit 1

# Determine dataset source
INPUT_DATASET="${1:-$DATA_DIR}"
SCENE_NAME=$(basename "$INPUT_DATASET")

if [ -n "${2:-}" ]; then
    GS_SUBDIR="$2"
else
    # Auto-detect existing 3DGS model output directory or fallback
    LATEST_3DGS=$(ls -d "$PROJECT_ROOT/${OUTPUT_DIR}/${SCENE_NAME}/3dgs/"* 2>/dev/null | tail -n 1 || true)
    if [ -n "$LATEST_3DGS" ]; then
        GS_SUBDIR="3dgs/$(basename "$LATEST_3DGS")"
    else
        GS_SUBDIR="3dgs/inria_30k"
    fi
fi

if [ -d "$GS_SUBDIR" ]; then
    GS_CHECKPOINT="$GS_SUBDIR"
    MODEL_LABEL=$(basename "$GS_CHECKPOINT")
else
    GS_CHECKPOINT="$PROJECT_ROOT/${OUTPUT_DIR}/${SCENE_NAME}/${GS_SUBDIR}"
    MODEL_LABEL=$(echo "$GS_SUBDIR" | tr '/' '_')
fi

SUGAR_MESH_DIR="$PROJECT_ROOT/${OUTPUT_DIR}/${SCENE_NAME}/mesh/sugar_${MODEL_LABEL}"
mkdir -p "$SUGAR_MESH_DIR"

if [ ! -d "$GS_CHECKPOINT" ]; then
    echo "[FATAL] Source Gaussian checkpoint directory not found at: $GS_CHECKPOINT"
    echo "Usage: ./scripts/03_train_sugar.sh [dataset_path] [source_checkpoint_dir_or_subdir]"
    exit 1
fi

# Detect active SfM pose source dynamically
ACTIVE_SFM="unknown"
if [ -L "$PROJECT_ROOT/${INPUT_DATASET}/sparse/0" ]; then
    ACTIVE_SFM=$(readlink "$PROJECT_ROOT/${INPUT_DATASET}/sparse/0" | xargs basename)
fi

ITERATION_TO_LOAD="${3:-${GS_ITERATIONS:-30000}}"

# Execute SuGaR Pipeline
echo "Starting SuGaR Mesh Extraction (Scene: $SCENE_NAME, Source Checkpoint: $GS_CHECKPOINT, Iteration: $ITERATION_TO_LOAD, Active SfM: $ACTIVE_SFM)..."
python train_full_pipeline.py \
    -s "$PROJECT_ROOT/$INPUT_DATASET" \
    --gs_output_dir "$GS_CHECKPOINT" \
    -i "$ITERATION_TO_LOAD" \
    -r "$SUGAR_REGULARIZATION" \
    --high_poly "$SUGAR_HIGH_POLY" \
    --export_obj True \
    --refinement_time "$SUGAR_REFINEMENT"

# Sync output files to scene specific mesh directory
echo "Syncing extracted SuGaR results to $SUGAR_MESH_DIR..."
if [ -d "output" ] && compgen -G "output/*" > /dev/null; then
    cp -r output/* "$SUGAR_MESH_DIR/"
else
    echo "[FATAL] SuGaR pipeline failed: No exported meshes found in output/" >&2
    exit 1
fi


# Dynamically record execution provenance metadata
cat <<EOF > "$SUGAR_MESH_DIR/pipeline_meta.json"
{
  "stage": "Stage 3 (Mesh Reconstruction)",
  "engine": "SuGaR (Surface-Aligned Gaussians)",
  "source_model_path": "$GS_CHECKPOINT",
  "source_dataset": "$INPUT_DATASET",
  "active_sfm": "$ACTIVE_SFM",
  "regularization": "$SUGAR_REGULARIZATION",
  "timestamp": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF

echo "SuGaR Pipeline Completed! Results saved to $SUGAR_MESH_DIR"
