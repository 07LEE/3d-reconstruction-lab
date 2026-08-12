#!/bin/bash

# 3DRC Automated Environment Setup Script
# Initializes submodules, applies patches, and configures split Conda environments (3drc, gs_train, gs_scaffold, gs_sugar, gs_group, gs_milo).

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

echo -e "\n[Step 3/4] Ensuring Conda Environments (3drc, gs_train, gs_sugar, gs_group, gs_milo, gs_scaffold)..."
for env_name in 3drc gs_train gs_sugar gs_group gs_milo gs_scaffold; do
    if ! conda env list | awk '{print $1}' | grep -qx "$env_name"; then
        echo "Creating Conda environment '$env_name'..."
        if [ -f "envs/$env_name.yml" ]; then
            conda env create -f "envs/$env_name.yml"
        else
            conda create -n "$env_name" python=3.10 -y
        fi
    fi
done

# 4. Build CUDA Extensions & Submodules across Environments
echo -e "\n[Step 4/4] Building CUDA Extensions and Submodules..."
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
source scripts/utils/setup_build_env.sh || { echo "[FATAL] Build environment setup failed."; exit 1; }

# Helper function for absolute isolated pip install
install_ext() {
    local env_target="$1"
    local module_path="$2"
    local pip_bin="$CONDA_PATH/envs/$env_target/bin/pip"

    if [ -d "$module_path" ] && [ -x "$pip_bin" ]; then
        echo "Installing $module_path into '$env_target'..."
        "$pip_bin" install --no-cache-dir --no-build-isolation "$module_path"
    fi
}

# 4a. 3drc submodules (hloc)
install_ext "3drc" "third_party/hloc"

# 4b. gs_train extensions
install_ext "gs_train" "third_party/gaussian-splatting/submodules/diff-gaussian-rasterization"
install_ext "gs_train" "third_party/gaussian-splatting/submodules/simple-knn"
install_ext "gs_train" "third_party/2d-gaussian-splatting/submodules/diff-surfel-rasterization"

# 4c. gs_scaffold extensions
install_ext "gs_scaffold" "third_party/scaffold-gs/submodules/diff-gaussian-rasterization"
install_ext "gs_scaffold" "third_party/scaffold-gs/submodules/simple-knn"

# 4d. gs_sugar extensions
install_ext "gs_sugar" "third_party/gaussian-splatting/submodules/diff-gaussian-rasterization"
install_ext "gs_sugar" "third_party/gaussian-splatting/submodules/simple-knn"

# 4e. gs_group extensions
if [ -d "third_party/gaussian-grouping/submodules/diff-gaussian-rasterization" ]; then
    install_ext "gs_group" "third_party/gaussian-grouping/submodules/diff-gaussian-rasterization"
else
    install_ext "gs_group" "third_party/gaussian-splatting/submodules/diff-gaussian-rasterization"
fi
install_ext "gs_group" "third_party/gaussian-splatting/submodules/simple-knn"

# 4f. gs_milo extensions
install_ext "gs_milo" "third_party/milo/submodules/diff-gaussian-rasterization"
install_ext "gs_milo" "third_party/milo/submodules/diff-gaussian-rasterization_ms"
install_ext "gs_milo" "third_party/milo/submodules/diff-gaussian-rasterization_gof"
install_ext "gs_milo" "third_party/milo/submodules/simple-knn"
install_ext "gs_milo" "third_party/milo/submodules/fused-ssim"

# Reset active conda environment to primary training environment (gs_train)
conda activate gs_train

# Verify patch & environment integrity
echo -e "\nVerifying Patch and Environment Integrity..."
./scripts/utils/verify_patches.sh

echo -e "\n=================================================="
echo " 3DRC Environment Setup Completed Successfully!"
echo " Environments (3drc, gs_train, gs_sugar, gs_group, gs_milo, gs_scaffold) are ready."
echo "=================================================="
