#!/bin/bash

# 3DRC Automated Environment Setup Script
# Initializes submodules, applies patches, and configures Conda environments.

set -e

# Argument Parsing
TARGET_ENVS=()

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
        *)
            echo "[FATAL] Unknown option: $1"
            exit 1
            ;;
    esac
done

# The single Gaussian environment (ADR 0018); GS_ENV_NAME overrides its name
UNIFIED_ENV="${GS_ENV_NAME:-gs}"

if [ "${#TARGET_ENVS[@]}" -eq 0 ]; then
    TARGET_ENVS=(3drc "$UNIFIED_ENV")
fi

# Validate target environment names
for e in "${TARGET_ENVS[@]}"; do
    printf '%s\n' 3drc "$UNIFIED_ENV" | grep -qx "$e" \
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

# Unified env: extensions with unique module names are shared. All simple-knn copies have
# identical sources (patches only add includes), so one build serves every backend.
ENV_EXTS["$UNIFIED_ENV"]="third_party/2d-gaussian-splatting/submodules/diff-surfel-rasterization
third_party/PlanarGS/submodules/diff-plane-rasterization
third_party/milo/submodules/diff-gaussian-rasterization_ms
third_party/milo/submodules/diff-gaussian-rasterization_gof
third_party/milo/submodules/fused-ssim"

# Unified env: diff_gaussian_rasterization differs per backend under one module name, so each
# build goes to .ext/<variant>/ (or $GS_EXT_ROOT/<variant>/) and scripts/utils/activate_backend.sh
# puts it on PYTHONPATH.
GS_EXT_ROOT="${GS_EXT_ROOT:-$(pwd)/.ext}"
declare -A GS_RASTERIZER_VARIANTS
GS_RASTERIZER_VARIANTS[inria]="third_party/gaussian-splatting/submodules/diff-gaussian-rasterization"
GS_RASTERIZER_VARIANTS[scaffold]="third_party/scaffold-gs/submodules/diff-gaussian-rasterization"
GS_RASTERIZER_VARIANTS[mipsplatting]="third_party/mip-splatting/submodules/diff-gaussian-rasterization"
GS_RASTERIZER_VARIANTS[milo]="third_party/milo/submodules/diff-gaussian-rasterization"
GS_RASTERIZER_VARIANTS[grouping]="third_party/gaussian-grouping/submodules/diff-gaussian-rasterization"
if [ ! -d "${GS_RASTERIZER_VARIANTS[grouping]}" ]; then
    GS_RASTERIZER_VARIANTS[grouping]="${GS_RASTERIZER_VARIANTS[inria]}"
fi

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
# torch (cu130) and the build toolkit must share a CUDA major version, so build with CUDA 13.2 from 'cuda132'
BUILD_TOOLKIT_ENV="${BUILD_TOOLKIT_ENV:-cuda132}"
export BUILD_TOOLKIT_ENV
if [ "$BUILD_TOOLKIT_ENV" = "cuda132" ] && ! conda env list | awk '{print $1}' | grep -qx "cuda132"; then
    echo "  • Creating build toolkit env 'cuda132' (CUDA 13.2)..."
    conda env create -f envs/cuda132.yml || { echo "[FATAL] Failed to create the cuda132 toolkit env."; exit 1; }
fi
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
source scripts/utils/setup_build_env.sh || { echo "[FATAL] Build environment setup failed."; exit 1; }

# Installs into site-packages (non-editable) so a rebuild never overwrites a .so that is imported elsewhere
install_ext() {
    local ext_path="$1"
    local pip_bin="$CONDA_PATH/envs/$UNIFIED_ENV/bin/pip"

    [ -d "$ext_path" ] || { echo "[FATAL] Missing extension path: $ext_path"; exit 1; }
    [ -x "$pip_bin" ] || { echo "[FATAL] No pip binary found in environment: $UNIFIED_ENV"; exit 1; }

    echo "Building $ext_path -> $UNIFIED_ENV..."
    "$pip_bin" install --no-cache-dir --no-build-isolation --no-deps "$ext_path"
}

# Builds one rasterizer variant into .ext/<variant>/ (not site-packages) for the unified env
install_rasterizer_variant() {
    local variant="$1"
    local ext_path="${GS_RASTERIZER_VARIANTS[$variant]}"
    local pip_bin="$CONDA_PATH/envs/$UNIFIED_ENV/bin/pip"

    [ -d "$ext_path" ] || { echo "[FATAL] Missing extension path: $ext_path"; exit 1; }
    echo "Building $ext_path -> $GS_EXT_ROOT/$variant..."
    rm -rf "${GS_EXT_ROOT:?}/$variant"
    "$pip_bin" install --no-cache-dir --no-build-isolation --no-deps --target "$GS_EXT_ROOT/$variant" "$ext_path"
}

# MILo's tetra-nerf: the pip package is Python-only, so the C++ extension is built with CMake
# against the target env's torch (CMake's Torch config forces ABI=0 while the cu128 wheels use ABI=1).
install_tetra_nerf() {
    local target_env="$1"
    local env_prefix="$CONDA_PATH/envs/$target_env"
    local src="third_party/milo/submodules/tetra_triangulation"
    local build_dir site_ext abi torch_cmake

    [ -d "$src" ] || { echo "[FATAL] Missing extension path: $src"; exit 1; }
    build_dir="$(mktemp -d)"
    abi="$("$env_prefix/bin/python" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
    torch_cmake="$("$env_prefix/bin/python" -c 'import torch; print(torch.utils.cmake_prefix_path)')"

    # The conda ld shipped with the CUDA toolkit cannot link against a newer system glibc, so the system
    # linker goes first for g++ (PATH) and for the nvcc host-link step (-B).
    echo "Building $src -> $target_env (tetranerf C++ extension)..."
    "$env_prefix/bin/pip" install --no-cache-dir --no-build-isolation --no-deps "$src"
    PATH="/usr/bin:$env_prefix/bin:$PATH" NVCC_APPEND_FLAGS="-Xcompiler -B/usr/bin" "$env_prefix/bin/cmake" -S "$src" -B "$build_dir" \
        -DCMAKE_PREFIX_PATH="$torch_cmake;$env_prefix" -DCONDA_PREFIX="$env_prefix" \
        -DPython_EXECUTABLE="$env_prefix/bin/python" -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
        -DCMAKE_CUDA_HOST_COMPILER="$CXX" -DCMAKE_CUDA_ARCHITECTURES=120 \
        "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=$abi" \
        || { echo "[FATAL] tetranerf CMake configure failed"; exit 1; }
    PATH="/usr/bin:$env_prefix/bin:$PATH" NVCC_APPEND_FLAGS="-Xcompiler -B/usr/bin" "$env_prefix/bin/cmake" --build "$build_dir" -j"$(nproc)" \
        || { echo "[FATAL] tetranerf build failed"; exit 1; }
    site_ext="$("$env_prefix/bin/python" -c 'import tetranerf.utils.extension as e, os; print(os.path.dirname(e.__file__))' 2>/dev/null || true)"
    [ -n "$site_ext" ] || site_ext="$env_prefix/lib/python3.10/site-packages/tetranerf/utils/extension"
    cp "$build_dir"/tetranerf/utils/extension/*.so "$site_ext"/
    rm -rf "${build_dir:?}"
}

build_env() {
    local env_name="$1"

    if [ "$env_name" = "$UNIFIED_ENV" ]; then
        # nvdiffrast has no Python 3.10 wheel on PyPI; install from the milo submodule
        # non-editable: milo puts submodules/ on sys.path, where an editable nvdiffrast resolves to a bare namespace dir
        "$CONDA_PATH/envs/$UNIFIED_ENV/bin/pip" install --no-cache-dir --no-build-isolation --no-deps third_party/milo/submodules/nvdiffrast
        # scaffold-gs imports torch_scatter, which has no wheel for this torch/CUDA combination
        "$CONDA_PATH/envs/$UNIFIED_ENV/bin/pip" install --no-cache-dir --no-build-isolation torch_scatter==2.1.2
        # simple_knn is a namespace package (no __init__.py); an editable install cannot be imported
        "$CONDA_PATH/envs/$UNIFIED_ENV/bin/pip" install --no-cache-dir --no-build-isolation --no-deps third_party/gaussian-splatting/submodules/simple-knn
        for variant in inria scaffold mipsplatting milo grouping; do
            install_rasterizer_variant "$variant"
        done
        # PlanarGS and SuGaR import pytorch3d; no wheel exists for this torch/CUDA combination (pinned to the commit SuGaR was validated with)
        "$CONDA_PATH/envs/$UNIFIED_ENV/bin/pip" install --no-cache-dir --no-build-isolation --no-deps \
            "git+https://github.com/facebookresearch/pytorch3d.git@9381c4016376345bb795b97c45a6c2de66db354a"
        install_tetra_nerf "$UNIFIED_ENV"
    fi

    if [ "$env_name" = "3drc" ] && [ -f "requirements.txt" ]; then
        echo "Installing external Python dependencies (requirements.txt) -> 3drc..."
        "$CONDA_PATH/envs/3drc/bin/pip" install --no-cache-dir -r requirements.txt
    fi

    while IFS= read -r ext; do
        [ -z "$ext" ] && continue
        install_ext "$ext"
    done <<< "${ENV_EXTS[$env_name]}"
}

for env_name in "${TARGET_ENVS[@]}"; do
    build_env "$env_name"
done

# Activate primary target environment
active_env="${TARGET_ENVS[0]}"
if printf '%s\n' "${TARGET_ENVS[@]}" | grep -qx "$UNIFIED_ENV"; then
    active_env="$UNIFIED_ENV"
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
