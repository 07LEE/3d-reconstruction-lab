---
title: Conda Environments
description: The 3drc and unified gs Conda environments, how to build them, and the environment variables that select them.
category: guide
last_updated: 2026-10-29
---

# Conda Environments

## Overview

- `3drc`: Main pipeline orchestrator, Step 1 SfM (COLMAP, hloc, VGGT, FastMap), preprocessing (`dense_undistort`), and Step 5 geometric evaluation (`eval_mesh`).
- `gs`: Unified environment for every Gaussian backend (Inria 3DGS, PlanarGS, 2DGS, Scaffold-GS, Mip-Splatting, Gaussian Grouping, SuGaR, MILo, TSDF).
- `cuda132`: Build-time CUDA 13.2 toolkit, used only to compile extensions for `gs`.

## Unified Gaussian Environment

The seven per-backend `gs_*` environments of ADR 0001 differed mostly in which `diff_gaussian_rasterization` build they carried. `gs` replaced them with one torch install: shared extensions go into `gs`, and each backend's rasterizer build goes into `.ext/<variant>/`, which `scripts/utils/activate_backend.sh` puts first on `PYTHONPATH` for that backend. See [ADR 0018](adr/0018-unified-gs-environment-with-per-backend-rasterizers.md).

```bash
./scripts/00_setup_environment.sh               # 3drc + gs (creates cuda132 when missing), builds .ext/{inria,scaffold,mipsplatting,milo,grouping}
./scripts/utils/verify_patches.sh --verbose     # checks each backend loads its own rasterizer
./scripts/run_3drc.sh train 3dgs data/<scene_name>
```

`gs` is built with torch `2.11.0+cu130` and the CUDA 13.2 toolkit from `cuda132`. See [ADR 0020](adr/0020-unified-gs-environment-on-cuda-13-2.md) and [GPU, CUDA and PyTorch Support](gpu_support.md).

## Environment Variables

| Variable | Default | Effect |
| --- | --- | --- |
| `GS_ENV_NAME` | `gs` | Name of the unified environment, to keep a second one next to the first |
| `GS_EXT_ROOT` | `.ext` | Where the per-backend rasterizer builds live |
| `BUILD_TOOLKIT_ENV` | `cuda132` | Conda environment whose CUDA toolkit builds the extensions |
| `TORCH_CUDA_ARCH_LIST` | `12.0+PTX` | GPU architectures the extensions are built for |
