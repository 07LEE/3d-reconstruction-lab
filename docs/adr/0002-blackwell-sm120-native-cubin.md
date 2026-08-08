# 0002. Blackwell sm_120 Native SASS Build (12.0+PTX)

Status: Accepted (2026-08-07)

Context:
Extension binaries compiled with TORCH_CUDA_ARCH_LIST="8.9;9.0" ran under PTX JIT fallback on sm_120. Active NVCC 12.4 in gs_milo rejected compute_120 generation.

Decision:
Compile native sm_120 SASS CUBIN binaries with TORCH_CUDA_ARCH_LIST="12.0+PTX" by setting CUDA_HOME to gs_train via setup_build_env.sh to access CUDA Toolkit 12.8 (CUDART_VERSION 12080). Pass -ccbin /usr/bin/g++-12 to avoid GCC 13 _Float32 header compilation errors.

Empirical Benchmark Evidence (Decision-time snapshot, tqdm peak):

- sm_90 SASS + PTX JIT: ~37.18 it/s
- Native sm_120 SASS: ~66.09 it/s (~1.78x speed boost)

Benchmark fingerprint: RTX 5070 Ti / CUDA 12.8 / PyTorch 2.12.0+cu128 / MILo / 1,112-frame indoor scene / ~3k iterations (tqdm peak instantaneous).
Note: These numbers serve as fixed historical decision evidence. Refer to README.md for final run average metrics.

Alternatives:

- "12.0" standalone without PTX: Rejected. While "12.0" standalone functions across minor CUDA 12.x architectures (e.g. sm_121), it breaks on the next major CUDA architecture generation (sm_130+). +PTX adds zero runtime overhead when SASS matches.

Consequences:
Requires CUDA 12.8 Toolkit and GCC 12 host compiler. Referencing gs_train as CUDA_HOME via setup_build_env.sh supplies both NVCC 12.8 and CUDA 12.8 headers/libcudart (CUDART_VERSION 12080), avoiding Thrust/CUB version mismatches across environments.

Verification:
cuobjdump --list-elf confirms native sm_120 SASS across all 9 CUDA C++ modules in gs_milo, gs_train, and gs_sugar.
