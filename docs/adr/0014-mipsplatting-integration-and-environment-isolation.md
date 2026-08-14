# 0014. Mip-Splatting Integration for 3D Smoothing Filter and Multi-Scale Rendering

Status: Accepted (2026-08-15)

Context:
Mip-Splatting introduces 3D smoothing filters and 2D anti-aliasing filters into 3D Gaussian Splatting to eliminate aliasing and popping artifacts during multi-scale scene rendering. Because Mip-Splatting modifies the CUDA rasterizer kernel under the same C++ module name (`diff_gaussian_rasterization`), installing it into the primary `gs_train` environment would overwrite `dr_aa` (antialiasing) rasterizer binaries, causing rasterizer collision failures.

Decision:
1. Isolate Mip-Splatting into a dedicated Conda environment (`gs_mipsplatting`) with PyTorch Stable `2.11.0+cu128` and CUDA 12.8 runtime.
2. Track all Mip-Splatting specific fixes (GCC-12 toolchain compatibility, PyTorch 2.6+ `weights_only=False` unpickling, and ADR-0013 `checkpoints/` & `events/` subdirectory isolation) in `patches/third_party/mip-splatting/0001-cstdint-and-sm120-compat.patch`.
3. Wire Mip-Splatting runner (`scripts/02e_train_mipsplatting.sh`) and CLI (`mipsplatting`) into the pipeline.

Consequences:
Enables high-fidelity multi-scale rendering without corrupting `gs_train`, `gs_scaffold`, or `gs_sugar` rasterizer binaries.
