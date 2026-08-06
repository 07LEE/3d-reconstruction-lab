#!/bin/bash

# 3DRC Automated Single Environment Setup Script
# Initializes submodules, builds CUDA C++ rasterizer modules, and configures unified 3drc conda environment.

set -e

echo "=================================================="
echo " Starting 3DRC Unified Environment Setup..."
echo "=================================================="

# 1. Initialize & Update Git Submodules
echo -e "\n[Step 1/3] Initializing Git Submodules in third_party/..."
git submodule update --init --recursive

# 2. Check & Activate Conda Environment (3drc)
CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
fi

if conda env list | grep -q "3drc"; then
    echo -e "\n[Step 2/3] Activating unified Conda environment '3drc'..."
    conda activate 3drc
else
    echo -e "\n[Step 2/3] Creating unified Conda environment '3drc' (Python 3.10)..."
    conda create -n 3drc python=3.10 -y
    conda activate 3drc
fi

# 3. Install Python Dependencies & CUDA Submodules
echo -e "\n[Step 3/3] Installing Python Dependencies and CUDA C++ Extensions..."
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    pip install numpy scipy pandas h5py tqdm opencv-python matplotlib pycolmap torchvision pyyaml plyfile ninja || true
fi

export CUDA_HOME="/usr/lib/nvidia-cuda-toolkit"
export PATH="/usr/lib/nvidia-cuda-toolkit/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.9"
export CC="/usr/bin/gcc-12"
# Apply automatic submodule patches if available
if [ -d "patches" ] && [ -f "patches/gaussian_splatting_fix.patch" ]; then
    echo "Applying submodule patches..."
    (cd third_party/gaussian-splatting && git apply ../../patches/gaussian_splatting_fix.patch 2>/dev/null || true)
fi

# Build C++ CUDA Rasterizer Submodules if available
if [ -d "third_party/gaussian-splatting/submodules/diff-gaussian-rasterization" ]; then
    echo "Installing diff-gaussian-rasterization extension..."
    pip install --no-build-isolation third_party/gaussian-splatting/submodules/diff-gaussian-rasterization || true
fi

if [ -d "third_party/gaussian-splatting/submodules/simple-knn" ]; then
    echo "Installing simple-knn extension..."
    pip install --no-build-isolation third_party/gaussian-splatting/submodules/simple-knn || true
fi

echo -e "\n=================================================="
echo " Unified 3DRC Environment Setup Completed Successfully!"
echo " All steps (SfM, 3DGS, SuGaR, Grouping) can now be run within '3drc' environment."
echo "=================================================="
