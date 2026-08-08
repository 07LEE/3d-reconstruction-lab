# Dynamic CUDA build environment setup script
CONDA_BASE_DIR=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
BUILD_TOOLKIT_ENV="${BUILD_TOOLKIT_ENV:-gs_train}"
TARGET_TOOLKIT_DIR="${CONDA_BASE_DIR}/envs/${BUILD_TOOLKIT_ENV}"

if [ -d "$TARGET_TOOLKIT_DIR" ]; then
    export CUDA_HOME="$TARGET_TOOLKIT_DIR"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"

if [ -x /usr/bin/g++-12 ]; then
    export NVCC_CCBIN=/usr/bin/g++-12
    export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-12"
fi

# Pre-flight check for compute_120 support in NVCC
if ! nvcc --list-gpu-arch 2>/dev/null | grep -q compute_120; then
    echo "[FATAL] NVCC compiler at CUDA_HOME=${CUDA_HOME:-none} lacks compute_120 support."
    echo "[FATAL] Ensure CUDA Toolkit 12.8+ is installed in $TARGET_TOOLKIT_DIR."
    return 1 2>/dev/null || exit 1
fi

echo "[INFO] CUDA build environment initialized cleanly with NVCC $(nvcc --version | grep -o 'release [0-9.]*')"
