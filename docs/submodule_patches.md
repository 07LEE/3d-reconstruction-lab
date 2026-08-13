---
title: Submodule Patches and Environment Integrity Guide
description: Tracked submodule patches in patches/ and environment verification system details.
category: guide
last_updated: 2026-08-14
---

# Submodule Patches and Environment Integrity Guide

This document details the tracked submodule patches in `patches/` and the environment verification system in `scripts/utils/`.

## Upstream Repository Policy

All submodules in `third_party/` (including `gaussian-splatting`) reference official upstream repositories (e.g., `graphdeco-inria/gaussian-splatting.git`) directly at official release or main commits. Personal fork repositories are not used or referenced. All project-specific bug fixes, feature extensions, and environment compatibility modifications are stored as versioned `.patch` files under `patches/third_party/` and applied automatically during setup.

## Tracked Patches

### Patch Inventory

| Target Submodule | Patch File Path | Fix & Purpose Description |
| --- | --- | --- |
| `gaussian-splatting` | `patches/third_party/gaussian-splatting/0001-colmap-camera-models.patch` | Support for `SIMPLE_RADIAL`, `RADIAL`, and `OPENCV` COLMAP camera models |
| `gaussian-splatting` | `patches/third_party/gaussian-splatting/0002-distcuda2-scipy-fallback.patch` | `scipy.spatial.KDTree` fallback if CUDA nearest-neighbor fails |
| `diff-gaussian-rasterization` | `patches/third_party/gaussian-splatting/submodules/diff-gaussian-rasterization/0001-zero-init-state-structs.patch` | Zero-initialization of CUDA state structs in `rasterizer_impl.cu` |
| `diff-gaussian-rasterization` | `patches/third_party/gaussian-splatting/submodules/diff-gaussian-rasterization/0002-cstdint-include.patch` | `<cstdint>` header include for GCC 13+ / GCC 14 toolchains |
| `sugar` | `patches/third_party/sugar/0001-sugar-extension-dup-and-cpu-device.patch` | Extension fix, CPU `data_device` VRAM OOM fix, `weights_only=False` PyTorch 2.6+ fix |
| `milo` | `patches/third_party/milo/0001-cstdint-and-cmake-fixes.patch` | `<cstdint>` includes for GCC 13+ in rasterizers and CMake 4.4 CXX standard / pybind11 tag fixes |
| `vggt` | `patches/third_party/vggt/0001-pycolmap-313-compat.patch` | PyCOLMAP 3.13+ API compatibility and temporary directory handling |
| `2d-gaussian-splatting` | `patches/third_party/2d-gaussian-splatting/0001-colmap-camera-models.patch` | Support for `SIMPLE_RADIAL` and `RADIAL` camera models in 2DGS |
| `2d-gaussian-splatting` | `patches/third_party/2d-gaussian-splatting/0002-mesh-utils-bounds-and-empty-fix.patch` | Empty cluster bounds guard in mesh postprocessing |
| `diff-surfel-rasterization` | `patches/third_party/2d-gaussian-splatting/submodules/diff-surfel-rasterization/0001-cstdint-and-zero-init.patch` | `<cstdint>` includes, zero-init structs, and backward SH gradient dimension match |
| `scaffold-gs` | `patches/third_party/scaffold-gs/0001-rasterizer-antialiasing-compat.patch` | `antialiasing` kwarg guard and `visible_filter` fallback for dr_aa rasterizer |
| `scaffold-gs` | `patches/third_party/scaffold-gs/0002-checkpoint-capture-and-getattr-fix.patch` | PyTorch 2.6+ `weights_only=False` fix, `[INFO]` CleanStreamFormatter logger, 10/11-element checkpoint tuple compatibility, `checkpoints/` & `events/` subdirectory isolation, and `getattr` guard for deleted `offset_denom` |

## Utility Scripts

### Integrity Scripts

- `./scripts/utils/apply_patches.sh`: Idempotently applies all tracked `.patch` files to submodules upon repository setup.
- `./scripts/utils/verify_patches.sh`: Audits source code files and active Python environments to verify that all patches are intact.
