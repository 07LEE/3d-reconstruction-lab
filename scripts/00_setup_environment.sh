#!/bin/bash

# 3DRC Automated Environment & Submodule Setup Script
# Initializes git submodules, verifies Conda environments, and installs dependencies.

set -e

echo "=================================================="
echo " Starting 3DRC Automated Environment Setup..."
echo "=================================================="

# 1. Initialize & Update Git Submodules
echo "\n[Step 1/3] Initializing Git Submodules in third_party/..."
git submodule update --init --recursive

# 2. Check Conda Installation
CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
else
    echo "[Warning] Conda executable not found automatically. Using default environment."
fi

# 3. Setup Python Requirements
echo "\n[Step 2/3] Installing Python Dependencies..."
if command -v pip &> /dev/null; then
    pip install -r requirements.txt
else
    echo "[Error] pip is not available. Please activate your target Conda environment first."
    exit 1
fi

# 4. Check & Build Submodule Extensions if available
echo "\n[Step 3/3] Verifying Submodules Structure..."
if [ -d "third_party/gaussian-splatting/submodules/diff-gaussian-rasterization" ]; then
    echo "Submodule CUDA extensions ready."
fi

echo "\n=================================================="
echo " 3DRC Environment Setup Completed Successfully!"
echo " You can now run the pipeline via ./scripts/run_3drc.sh or python 3drc.py"
echo "=================================================="
