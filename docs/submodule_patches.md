# Submodule Patches and Environment Integrity Guide

This document details the tracked submodule patches in `patches/` and the environment verification system in `scripts/utils/`.

## Tracked Patches

### Patch Inventory

| Target Submodule | Patch File Path | Fix & Purpose Description |
|---|---|---|
| `gaussian-splatting` | `patches/third_party/gaussian-splatting/0001-colmap-camera-models.patch` | Support for `SIMPLE_RADIAL`, `RADIAL`, and `OPENCV` COLMAP camera models |
| `gaussian-splatting` | `patches/third_party/gaussian-splatting/0002-distcuda2-scipy-fallback.patch` | `scipy.spatial.KDTree` fallback if CUDA nearest-neighbor fails |
| `diff-gaussian-rasterization` | `patches/third_party/gaussian-splatting/submodules/diff-gaussian-rasterization/0001-zero-init-state-structs.patch` | Zero-initialization of CUDA state structs in `rasterizer_impl.cu` |
| `diff-gaussian-rasterization` | `patches/third_party/gaussian-splatting/submodules/diff-gaussian-rasterization/0002-cstdint-include.patch` | `<cstdint>` header include for GCC 13+ / GCC 14 toolchains |
| `sugar` | `patches/third_party/sugar/0001-sugar-extension-dup-and-cpu-device.patch` | Extension fix, CPU `data_device` VRAM OOM fix, `weights_only=False` PyTorch 2.6+ fix |

## Utility Scripts

### Integrity Scripts

- `./scripts/utils/apply_patches.sh`: Idempotently applies all tracked `.patch` files to submodules upon repository setup.
- `./scripts/utils/verify_patches.sh`: Audits source code files and active Python environments to verify that all 5 patches are intact.
