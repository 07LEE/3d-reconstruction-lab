# 0012. PyTorch Stable cu128 Standardization and Nightly Deprecation

Status: Accepted (2026-08-12)

Context:
3DRC previously specified PyTorch Nightly (`2.12.0.dev20260407+cu128`) across multiple training environments (`gs_train`, `gs_sugar`, `gs_group`). However, PyTorch Nightly build wheels are transient and pruned periodically from PyTorch indices, leading to broken environment reproduction and build failures. PyTorch Stable 2.11.0+cu128 natively includes CUDA 12.8 runtime and NVIDIA Blackwell RTX 50 (`sm_120`) SASS CUBIN support without relying on unpinned nightly distributions.

Decision:
1. Standardize all 3DRC conda environments (`gs_train`, `gs_scaffold`, `gs_sugar`, `gs_group`, `gs_milo`) on PyTorch Stable `2.11.0+cu128` with explicit `--extra-index-url https://download.pytorch.org/whl/cu128`.
2. Deprecate and remove all PyTorch Nightly (`.dev*`) package references.
3. Rebuild all 12 C++ CUDA extension modules against PyTorch Stable 2.11.0+cu128 and verify `sm_120` SASS CUBIN binaries via `scripts/utils/verify_patches.sh`.
4. Archive pre-migration environment files in `envs/archive/`.

Consequences:
Guarantees long-term environment reproducibility across fresh environment installations and ensures native NVIDIA Blackwell `sm_120` GPU execution across all 12 CUDA extensions.
