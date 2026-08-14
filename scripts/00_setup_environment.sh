#!/bin/bash

# 3DRC Automated Environment Setup Script
# Initializes submodules, applies patches, and configures Conda environments.

set -e

# Argument Parsing
TARGET_ENVS=()
BUILD_ALL=0

while [ $# -gt 0 ]; do
    case "$1" in
        --env)
            if [ $# -lt 2 ]; then
                echo "[FATAL] --env requires an environment name argument."
                exit 1
            fi
            TARGET_ENVS+=("$2")
            shift 2
            ;;
        --all)
            BUILD_ALL=1
            shift
            ;;
        *)
            echo "[FATAL] Unknown option: $1"
            exit 1
            ;;
    esac
done

ALL_ENVS=(3drc gs_train gs_sugar gs_group gs_milo gs_scaffold gs_mipsplatting)

if [ "$BUILD_ALL" = "1" ]; then
    TARGET_ENVS=("${ALL_ENVS[@]}")
elif [ "${#TARGET_ENVS[@]}" -eq 0 ]; then
    TARGET_ENVS=(3drc gs_train)
fi

# Validate target environment names
for e in "${TARGET_ENVS[@]}"; do
    printf '%s\n' "${ALL_ENVS[@]}" | grep -qx "$e" \
        || { echo "[FATAL] Unknown environment: $e"; exit 1; }
done

# Ensure '3drc' is always included in target environments (SfM is required step 1)
has_3drc=0
for e in "${TARGET_ENVS[@]}"; do
    if [ "$e" = "3drc" ]; then
        has_3drc=1
        break
    fi
done

if [ "$has_3drc" -eq 0 ]; then
    echo "[INFO] Environment '3drc' is required for SfM pipeline. Automatically adding '3drc' to build targets."
    TARGET_ENVS=("3drc" "${TARGET_ENVS[@]}")
fi

# Deduplicate target environments while preserving order
UNIQUE_TARGET_ENVS=()
for e in "${TARGET_ENVS[@]}"; do
    if [[ ! " ${UNIQUE_TARGET_ENVS[*]} " =~ " ${e} " ]]; then
        UNIQUE_TARGET_ENVS+=("$e")
    fi
done
TARGET_ENVS=("${UNIQUE_TARGET_ENVS[@]}")

# Environment extensions mapping
declare -A ENV_EXTS
ENV_EXTS[3drc]="third_party/hloc"

ENV_EXTS[gs_train]="third_party/gaussian-splatting/submodules/diff-gaussian-rasterization
third_party/gaussian-splatting/submodules/simple-knn
third_party/2d-gaussian-splatting/submodules/diff-surfel-rasterization"

ENV_EXTS[gs_scaffold]="third_party/scaffold-gs/submodules/diff-gaussian-rasterization
third_party/scaffold-gs/submodules/simple-knn"

ENV_EXTS[gs_mipsplatting]="third_party/mip-splatting/submodules/diff-gaussian-rasterization
third_party/mip-splatting/submodules/simple-knn"

ENV_EXTS[gs_sugar]="third_party/gaussian-splatting/submodules/diff-gaussian-rasterization
third_party/gaussian-splatting/submodules/simple-knn"

ENV_EXTS[gs_group]="third_party/gaussian-splatting/submodules/simple-knn"

ENV_EXTS[gs_milo]="third_party/milo/submodules/diff-gaussian-rasterization
third_party/milo/submodules/diff-gaussian-rasterization_ms
third_party/milo/submodules/diff-gaussian-rasterization_gof
third_party/milo/submodules/simple-knn
third_party/milo/submodules/fused-ssim"

echo "=================================================="
echo " Starting 3DRC Environment Setup..."
echo " Target Environments: ${TARGET_ENVS[*]}"
echo "=================================================="

# Guard: Check for uncommitted submodule changes before update
if ! git submodule foreach --recursive 'git diff --quiet' >/dev/null 2>&1; then
    echo "[WARN] Submodules contain uncommitted local changes."
    echo "       Continuing will overwrite them. Run export_patch.sh first to save changes."
    read -p "Continue? [y/N] " a; [ "$a" = "y" ] || exit 1
fi

# 1. Initialize & Update Git Submodules
echo -e "\n[Step 1/5] Initializing Git Submodules in third_party/..."
git submodule update --init --recursive

# 2. Apply Submodule Patches
echo -e "\n[Step 2/5] Applying Submodule Patches..."
chmod +x scripts/*.sh scripts/utils/*.sh 2>/dev/null || true
./scripts/utils/apply_patches.sh

# 3. Check Conda Environments
CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
fi

echo -e "\n[Step 3/5] Ensuring Conda Environments (${TARGET_ENVS[*]})..."
for env_name in "${TARGET_ENVS[@]}"; do
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

# 4. Build CUDA Extensions & Submodules across Target Environments
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

build_env() {
    local env_name="$1"
    
    if [ "$env_name" = "gs_group" ]; then
        if [ -d "third_party/gaussian-grouping/submodules/diff-gaussian-rasterization" ]; then
            install_ext "gs_group" "third_party/gaussian-grouping/submodules/diff-gaussian-rasterization"
        else
            install_ext "gs_group" "third_party/gaussian-splatting/submodules/diff-gaussian-rasterization"
        fi
    fi

    while IFS= read -r ext; do
        [ -n "$ext" ] && install_ext "$env_name" "$ext"
    done <<< "${ENV_EXTS[$env_name]}"
}

for env_name in "${TARGET_ENVS[@]}"; do
    build_env "$env_name"
done

# Activate primary target environment
active_env="gs_train"
if ! conda env list | awk '{print $1}' | grep -qx "gs_train"; then
    active_env="${TARGET_ENVS[0]}"
fi
echo "Activating Conda env '$active_env'..."
conda activate "$active_env"

# 5. Verify patch & environment integrity
echo -e "\n[Step 5/5] Verifying Patch and Environment Integrity..."
./scripts/utils/verify_patches.sh

echo -e "\n=================================================="
echo " 3DRC Environment Setup Completed Successfully!"
echo " Environments (${TARGET_ENVS[*]}) are ready."
echo "=================================================="
