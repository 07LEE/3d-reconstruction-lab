#!/bin/bash

# 3DRC Automated Environment Setup Script
# Initializes submodules, applies patches, and configures split Conda environments (gs_train, gs_scaffold, gs_sugar, gs_group, gs_milo).

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

# 4. Build CUDA Extensions across all 5 Environments
echo -e "\n[Step 4/4] Building CUDA Extensions in all environments..."
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
source scripts/utils/setup_build_env.sh || { echo "[FATAL] Build environment setup failed."; exit 1; }

# 4a. gs_train extensions
echo "Installing extensions in 'gs_train'..."
conda activate gs_train
source scripts/utils/setup_build_env.sh

if [ -d "third_party/gaussian-splatting/submodules/diff-gaussian-rasterization" ]; then
    (cd third_party/gaussian-splatting/submodules/diff-gaussian-rasterization && pip install --no-build-isolation -e .)
fi
if [ -d "third_party/gaussian-splatting/submodules/simple-knn" ]; then
    (cd third_party/gaussian-splatting/submodules/simple-knn && pip install --no-build-isolation -e .)
fi
if [ -d "third_party/2d-gaussian-splatting/submodules/diff-surfel-rasterization" ]; then
    (cd third_party/2d-gaussian-splatting/submodules/diff-surfel-rasterization && pip install --no-build-isolation -e .)
fi

# 4b. gs_scaffold extensions
if conda env list | awk '{print $1}' | grep -qx "gs_scaffold"; then
    echo "Installing extensions in 'gs_scaffold'..."
    conda activate gs_scaffold
    source scripts/utils/setup_build_env.sh

    if [ -d "third_party/scaffold-gs/submodules/diff-gaussian-rasterization" ]; then
        (cd third_party/scaffold-gs/submodules/diff-gaussian-rasterization && pip install --no-build-isolation -e .)
    fi
    if [ -d "third_party/scaffold-gs/submodules/simple-knn" ]; then
        (cd third_party/scaffold-gs/submodules/simple-knn && pip install --no-build-isolation -e .)
    fi
fi

# 4c. gs_sugar extensions
if conda env list | awk '{print $1}' | grep -qx "gs_sugar"; then
    echo "Installing extensions in 'gs_sugar'..."
    conda activate gs_sugar
    source scripts/utils/setup_build_env.sh

    if [ -d "third_party/gaussian-splatting/submodules/diff-gaussian-rasterization" ]; then
        (cd third_party/gaussian-splatting/submodules/diff-gaussian-rasterization && pip install --no-build-isolation -e .)
    fi
    if [ -d "third_party/gaussian-splatting/submodules/simple-knn" ]; then
        (cd third_party/gaussian-splatting/submodules/simple-knn && pip install --no-build-isolation -e .)
    fi
fi

# 4d. gs_group extensions
if conda env list | awk '{print $1}' | grep -qx "gs_group"; then
    echo "Installing extensions in 'gs_group'..."
    conda activate gs_group
    source scripts/utils/setup_build_env.sh

    if [ -d "third_party/gaussian-grouping/submodules/diff-gaussian-rasterization" ]; then
        (cd third_party/gaussian-grouping/submodules/diff-gaussian-rasterization && pip install --no-build-isolation -e .)
    elif [ -d "third_party/gaussian-splatting/submodules/diff-gaussian-rasterization" ]; then
        (cd third_party/gaussian-splatting/submodules/diff-gaussian-rasterization && pip install --no-build-isolation -e .)
    fi
    if [ -d "third_party/gaussian-splatting/submodules/simple-knn" ]; then
        (cd third_party/gaussian-splatting/submodules/simple-knn && pip install --no-build-isolation -e .)
    fi
fi

# 4e. gs_milo extensions
if conda env list | awk '{print $1}' | grep -qx "gs_milo"; then
    echo "Installing extensions in 'gs_milo'..."
    conda activate gs_milo
    source scripts/utils/setup_build_env.sh

    if [ -d "third_party/milo/submodules/diff-gaussian-rasterization" ]; then
        (cd third_party/milo/submodules/diff-gaussian-rasterization && pip install --no-build-isolation -e .)
    fi
    if [ -d "third_party/milo/submodules/diff-gaussian-rasterization_ms" ]; then
        (cd third_party/milo/submodules/diff-gaussian-rasterization_ms && pip install --no-build-isolation -e .)
    fi
    if [ -d "third_party/milo/submodules/diff-gaussian-rasterization_gof" ]; then
        (cd third_party/milo/submodules/diff-gaussian-rasterization_gof && pip install --no-build-isolation -e .)
    fi
    if [ -d "third_party/milo/submodules/simple-knn" ]; then
        (cd third_party/milo/submodules/simple-knn && pip install --no-build-isolation -e .)
    fi
    if [ -d "third_party/milo/submodules/fused-ssim" ]; then
        (cd third_party/milo/submodules/fused-ssim && pip install --no-build-isolation -e .)
    fi
fi

# Reset active conda environment to primary training environment (gs_train)
conda activate gs_train

# Verify patch & environment integrity
echo -e "\nVerifying Patch and Environment Integrity..."
./scripts/utils/verify_patches.sh

echo -e "\n=================================================="
echo " 3DRC Environment Setup Completed Successfully!"
echo " Environments (3drc, gs_train, gs_sugar, gs_group, gs_milo, gs_scaffold) are ready."
echo "=================================================="
