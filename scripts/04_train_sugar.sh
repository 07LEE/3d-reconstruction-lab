#!/bin/bash

# [Step 04] SuGaR Mesh Reconstruction Pipeline
# Surface-Aligned Gaussian Splatting and high-quality mesh extraction

CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Activate the conda environment for SuGaR
if conda env list | grep -q "^sugar "; then
    conda activate sugar
else
    echo "Conda environment 'sugar' not found. Trying 'gs_original'..."
    conda activate gs_original
fi

# Hardware optimization
export TORCH_CUDA_ARCH_LIST="12.0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Get project root
PROJECT_ROOT=$(pwd)

# Move to sugar directory
cd third_party/sugar || exit 1

# Execute SuGaR Pipeline
echo "Starting SuGaR Mesh Extraction..."
python train_full_pipeline.py \
    -s "$PROJECT_ROOT/data/nerfstudio_data" \
    --gs_output_dir "$PROJECT_ROOT/outputs/gs_final_precision" \
    -r dn_consistency \
    --high_poly True \
    --export_obj True \
    --refinement_time short

echo "SuGaR Pipeline Completed! Results are saved in third_party/sugar/output"
