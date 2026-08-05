#!/bin/bash

# [Step 04] SuGaR Mesh Reconstruction Pipeline
# Surface-Aligned Gaussian Splatting and high-quality mesh extraction

CONFIG_PATH="configs/default_config.sh"
if [ -f "$CONFIG_PATH" ]; then
    source "$CONFIG_PATH"
fi

CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate gs_original

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
