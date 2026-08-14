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

echo -e "\n[Step 3/5] Ensuring Conda Environments (3drc, gs_train, gs_sugar, gs_group, gs_milo, gs_scaffold, gs_mipsplatting)..."
for env_name in 3drc gs_train gs_sugar gs_group gs_milo gs_scaffold gs_mipsplatting; do
    if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
        echo "  • Conda env '$env_name': [OK]"
    else
        echo "  • Creating Conda env '$env_name'..."
        if [ -f "envs/${env_name}.yml" ]; then
            conda env create -f "envs/${env_name}.yml"
        else
            conda create -n "$env_name" python=3.10 -y
        fi
    fi
done

# 4. Build CUDA Extensions & Submodules across Environments
echo -e "\n[Step 4/5] Building CUDA Extensions and Submodules..."
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
source scripts/utils/setup_build_env.sh || { echo "[FATAL] Build environment setup failed."; exit 1; }

# Helper function for absolute isolated pip install
install_ext() {
    local target_env="$1"
    local ext_path="$2"
    local pip_bin="$CONDA_PATH/envs/$target_env/bin/pip"

    [ -d "$ext_path" ] || { echo "[FATAL] Missing extension path: $ext_path"; exit 1; }
    [ -x "$pip_bin" ] || { echo "[FATAL] No pip binary found in environment: $target_env"; exit 1; }

    echo "Building $ext_path -> $target_env..."
    "$pip_bin" install --no-cache-dir --no-build-isolation -e "$ext_path"
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

# 4d. gs_mipsplatting extensions
install_ext "gs_mipsplatting" "third_party/mip-splatting/submodules/diff-gaussian-rasterization"
install_ext "gs_mipsplatting" "third_party/mip-splatting/submodules/simple-knn"

# 4e. gs_sugar extensions
install_ext "gs_sugar" "third_party/gaussian-splatting/submodules/diff-gaussian-rasterization"
install_ext "gs_sugar" "third_party/gaussian-splatting/submodules/simple-knn"

# 4f. gs_group extensions
if [ -d "third_party/gaussian-grouping/submodules/diff-gaussian-rasterization" ]; then
    install_ext "gs_group" "third_party/gaussian-grouping/submodules/diff-gaussian-rasterization"
else
    install_ext "gs_group" "third_party/gaussian-splatting/submodules/diff-gaussian-rasterization"
fi
install_ext "gs_group" "third_party/gaussian-splatting/submodules/simple-knn"

# 4g. gs_milo extensions
install_ext "gs_milo" "third_party/milo/submodules/diff-gaussian-rasterization"
install_ext "gs_milo" "third_party/milo/submodules/diff-gaussian-rasterization_ms"
install_ext "gs_milo" "third_party/milo/submodules/diff-gaussian-rasterization_gof"
install_ext "gs_milo" "third_party/milo/submodules/simple-knn"
install_ext "gs_milo" "third_party/milo/submodules/fused-ssim"

# Reset active conda environment to primary training environment (gs_train)
conda activate gs_train

# 5. Verify patch & environment integrity
echo -e "\n[Step 5/5] Verifying Patch and Environment Integrity..."
./scripts/utils/verify_patches.sh

echo -e "\n=================================================="
echo " 3DRC Environment Setup Completed Successfully!"
echo " Environments (3drc, gs_train, gs_sugar, gs_group, gs_milo, gs_scaffold, gs_mipsplatting) are ready."
echo "=================================================="
