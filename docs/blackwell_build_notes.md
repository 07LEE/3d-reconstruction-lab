# Blackwell (sm_120) CUDA C++ Build & Optimization Notes

This document provides technical diagnosis, root causes, and verified resolution strategies for building and running 3D reconstruction C++ CUDA extensions on NVIDIA Blackwell GPUs (sm_120, RTX 50 Series).

## Performance Benchmark

| Build Mode | Kernel Execution Engine | Throughput (it/s) | Relative Speed |
|---|---|---|---|
| PTX JIT Fallback | sm_90 SASS + Runtime JIT | ~37.18 it/s | Baseline (1.0x) |
| Native sm_120 CUBIN | Native Blackwell sm_120 SASS | ~66.09 it/s | 1.78x (78% faster) |

Benchmark Environment: NVIDIA GeForce RTX 5070 Ti (16GB VRAM), 1,112-frame indoor scene, PyTorch 2.12.0+cu128, CUDA 12.8 Toolkit.

## Troubleshooting Matrix

### 1. nvcc fatal: Unsupported gpu architecture 'compute_120'

- Symptom: nvcc compilation fails immediately with an unsupported GPU architecture error when passing sm_120 or compute_120.
- Root Cause: The active nvcc in PATH belongs to CUDA Toolkit 12.4 or older (e.g. system /usr/lib/nvidia-cuda-toolkit or Conda environment cuda-nvcc=12.4), which lacks native definition for Compute Capability 12.0 (sm_120).
- Solution: Set CUDA_HOME to a validated CUDA Toolkit 12.8+ environment (such as $(conda info --base)/envs/gs_train) and prepend $CUDA_HOME/bin to PATH via scripts/utils/setup_build_env.sh.

```bash
source scripts/utils/setup_build_env.sh
```

### 2. cudaErrorNoKernelImageForDevice (Runtime Crash)

- Symptom: Build completes without error, but model training crashes instantly upon launching CUDA kernels with cudaErrorNoKernelImageForDevice.
- Root Cause: The built .so extension binary contains SASS binaries only for older architectures (e.g. sm_89 Ada or sm_90 Hopper) without native sm_120 CUBIN or +PTX fallback. Changing TORCH_CUDA_ARCH_LIST without clearing build/ directory causes Ninja/setuptools to reuse cached .o object files.
- Solution: Wipe build/ directory and force re-build with TORCH_CUDA_ARCH_LIST="12.0+PTX". Verify binary architectures using cuobjdump:

```bash
cuobjdump --list-elf <path_to_extension>/_C*.so | grep -o "sm_[0-9]*"
```

### 3. GCC 13/14 _Float32 Type Compilation Errors

- Symptom: CUDA compilation errors in PyTorch/NVCC header files referencing _Float32 or narrowing conversion warnings.
- Root Cause: Host compiler GCC 13/14 introduces _Float32 type definitions that conflict with PyTorch and CUDA 12.8 C++ header standards.
- Solution: Explicitly set -ccbin /usr/bin/g++-12 in NVCC_PREPEND_FLAGS to enforce GCC 12 as the NVCC host compiler backend.

```bash
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-12"
```

### 4. Dead Lock File Hangs (torch.utils.cpp_extension.load)

- Symptom: JIT compilation (e.g. nvdiffrast_plugin) hangs indefinitely without throwing an exception or timing out.
- Root Cause: Previous build processes interrupted by Ctrl+C, OOM kill, or crash leave orphaned .lock files in ~/.cache/torch_extensions/, causing FileBaton.wait() to wait indefinitely.
- Solution: Wipe the torch_extensions cache directory to clear stale Baton lock files:

```bash
rm -rf ~/.cache/torch_extensions/
```

### 5. nvdiffrast Overriding Global TORCH_CUDA_ARCH_LIST

- Symptom: Global TORCH_CUDA_ARCH_LIST="12.0+PTX" is set, but nvdiffrast_plugin.so compiles with older fallback architectures.
- Root Cause: nvdiffrast/torch/ops.py internally rewrites TORCH_CUDA_ARCH_LIST if unset or if it detects unsupported older strings.
- Solution: Apply patches/third_party/milo/ or pre-warm nvdiffrast_plugin.so via scripts/utils/setup_build_env.sh prior to training execution.
