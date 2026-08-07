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
    [ "${CONDA_DEFAULT_ENV:-}" = "gs_sugar" ] || { echo "[FATAL] Conda environment 'gs_sugar' activation failed!"; exit 1; }
fi

# Get project root
PROJECT_ROOT=$(pwd)

# Move to sugar directory
cd third_party/sugar || exit 1

# Execute SuGaR Pipeline
echo "Starting SuGaR Mesh Extraction..."
python train_full_pipeline.py \
    -s "$PROJECT_ROOT/$DATA_DIR" \
    --gs_output_dir "$PROJECT_ROOT/${OUTPUT_DIR}/gs_final_precision" \
    -r "$SUGAR_REGULARIZATION" \
    --high_poly "$SUGAR_HIGH_POLY" \
    --export_obj True \
    --refinement_time "$SUGAR_REFINEMENT"

echo "SuGaR Pipeline Completed! Results are saved in third_party/sugar/output"
