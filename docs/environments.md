---
title: Conda Environments
description: The 3drc, unified gs and split gs_* Conda environments, how to build them, and the environment variables that select them.
category: guide
last_updated: 2026-10-29
---

# Conda Environments

## Overview

- `3drc`: Main pipeline orchestrator, Step 1 SfM (COLMAP, hloc, VGGT, FastMap), preprocessing (`dense_undistort`), and Step 5 geometric evaluation (`eval_mesh`).
- `gs`: Unified environment for every Gaussian backend (Inria 3DGS, PlanarGS, 2DGS, Scaffold-GS, Mip-Splatting, Gaussian Grouping, SuGaR, MILo, TSDF). Default.
- `cuda132`: Build-time CUDA 13.2 toolkit, used only to compile extensions for `gs`.
- Split `gs_*` environments (`GS_ENV_MODE=split`), kept for existing installs:
  - `gs_train`: Inria 3DGS and 2DGS training (`dr_aa` and `diff-surfel-rasterization`), plus Open3D TSDF mesh extraction.
  - `gs_sugar`: SuGaR surface-aligned mesh extraction and PlanarGS training (`diff-plane-rasterization`).
  - `gs_milo`: MILo differentiable mesh-in-the-loop training and extraction.
  - `gs_scaffold`: Scaffold-GS anchor-based training (isolated rasterizer build).
  - `gs_mipsplatting`: Mip-Splatting 3D smoothing filter training (isolated rasterizer build).
  - `gs_group`: Gaussian Grouping 3D SAM instance segmentation.

## Unified Gaussian Environment

The six `gs_*` environments differ only in which `diff_gaussian_rasterization` build they carry. `gs` replaces them with one torch install: shared extensions go into `gs`, and each backend's rasterizer build goes into `.ext/<variant>/`, which `scripts/utils/activate_backend.sh` puts first on `PYTHONPATH` for that backend. See [ADR 0018](adr/0018-unified-gs-environment-with-per-backend-rasterizers.md).

```bash
./scripts/00_setup_environment.sh               # 3drc + gs (creates cuda132 when missing), builds .ext/{inria,scaffold,mipsplatting,milo,grouping}
./scripts/utils/verify_patches.sh --verbose     # checks each backend loads its own rasterizer
./scripts/run_3drc.sh train 3dgs data/<scene_name>
```

`gs` is built with torch `2.11.0+cu130` and the CUDA 13.2 toolkit from `cuda132`. Build `gs` and the split `gs_*` environments in separate runs, because they use different CUDA major versions. See [ADR 0020](adr/0020-unified-gs-environment-on-cuda-13-2.md) and [GPU, CUDA and PyTorch Support](gpu_support.md).

## Environment Variables

| Variable | Default | Effect |
| --- | --- | --- |
| `GS_ENV_MODE` | `unified` | `unified` uses `gs`; `split` uses the per-backend `gs_*` environments |
| `GS_ENV_NAME` | `gs` | Name of the unified environment, to keep a second one next to the first |
| `GS_EXT_ROOT` | `.ext` | Where the per-backend rasterizer builds live |
| `BUILD_TOOLKIT_ENV` | `cuda132` (unified), `gs_train` (split) | Conda environment whose CUDA toolkit builds the extensions |
| `TORCH_CUDA_ARCH_LIST` | `12.0+PTX` | GPU architectures the extensions are built for |
