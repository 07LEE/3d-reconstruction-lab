#!/bin/bash

# 3DRC Automated Environment Setup Script
# Initializes submodules, applies patches, and configures split Conda environments (gs_train, gs_sugar, gs_group).

set -e

echo "=================================================="
echo " Starting 3DRC Environment Setup..."
echo "=================================================="

# Guard: Check for uncommitted submodule changes before update
if ! git submodule foreach --recursive 'git diff --quiet' >/dev/null 2>&1; then
    echo "[WARN] Submodules contain uncommitted local changes."
    echo "       Continuing will overwrite them. Run export_patch.sh first to save changes."
    read -p "Continue? [y/N] " a; [ "$a" = "y" ] || exit 1
fi

# 1. Initialize & Update Git Submodules
echo -e "\n[Step 1/4] Initializing Git Submodules in third_party/..."
git submodule update --init --recursive

# 2. Apply Submodule Patches
echo -e "\n[Step 2/4] Applying Submodule Patches..."
chmod +x scripts/*.sh scripts/utils/*.sh 2>/dev/null || true
./scripts/utils/apply_patches.sh

# 3. Check Conda Environments
CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
fi

echo -e "\n[Step 3/4] Ensuring Conda Environments (gs_train, gs_sugar, gs_group)..."
for env_name in gs_train gs_sugar gs_group; do
    if ! conda env list | awk '{print $1}' | grep -qx "$env_name"; then
        echo "Creating Conda environment '$env_name'..."
        if [ -f "envs/$env_name.yml" ]; then
            conda env create -f "envs/$env_name.yml"
        else
            conda create -n "$env_name" python=3.10 -y
        fi
    fi
done

# 4. Build CUDA Extensions in gs_train
echo -e "\n[Step 4/4] Building CUDA Extensions in 'gs_train'..."
source scripts/utils/setup_build_env.sh || { echo "[FATAL] Build environment setup failed."; exit 1; }
conda activate gs_train

if [ -d "third_party/gaussian-splatting/submodules/diff-gaussian-rasterization" ]; then
    echo "Installing diff-gaussian-rasterization extension in gs_train..."
    (cd third_party/gaussian-splatting/submodules/diff-gaussian-rasterization && pip install --no-build-isolation -e .)
fi

if [ -d "third_party/gaussian-splatting/submodules/simple-knn" ]; then
    echo "Installing simple-knn extension in gs_train..."
    (cd third_party/gaussian-splatting/submodules/simple-knn && pip install --no-build-isolation -e .)
fi

# Verify patch & environment integrity
echo -e "\nVerifying Patch and Environment Integrity..."
./scripts/utils/verify_patches.sh

echo -e "\n=================================================="
echo " 3DRC Environment Setup Completed Successfully!"
echo " Environments (gs_train, gs_sugar, gs_group) are ready."
echo "=================================================="
