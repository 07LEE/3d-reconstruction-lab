---
title: Large-Scale Scene Training Research Notes
description: Published approaches for Gaussian scenes that do not fit on one 16GB GPU, and a small-scale validation plan for 3DRC.
category: guide
last_updated: 2026-10-30
---

# Large-Scale Scene Training Research Notes

This note lists published approaches to training scenes that are too large for one 16GB GPU (RTX 5070 Ti, 60GB system RAM), and what to try in 3DRC. Nothing here is implemented yet.

## Starting Point in 3DRC

Meshes from the 3000-iteration test runs lack detail. Those runs used a quarter-resolution training image (`DOWNSAMPLE_RATE=4`) and 78,852 Gaussians, and densification only ends at iteration 15000, so they say little about the algorithms.

GPU memory goes mostly to the Gaussians and their optimizer state, which grow with their number, and to rasterizer buffers, which grow with image resolution. `outputs/benchmark_timings.json` does not record peak GPU memory, so the limit of this host is not yet known.

## Approaches in the Literature

Dividing the scene into parts that are trained separately and merged:

- [VastGaussian](https://arxiv.org/abs/2402.17427) (CVPR 2024): cells on the ground plane balanced by camera count, with expanded boundaries and a decoupled appearance model.
- [CityGaussian](https://arxiv.org/abs/2404.01133) (ECCV 2024): a global model as a prior, blocks on a uniform grid, and levels of detail for rendering.
- [Hierarchical 3D Gaussians](https://arxiv.org/abs/2406.12080) (ACM TOG 2024): fixed-size chunks merged into a hierarchy. It reports training on one V100 (32GB) with peak memory up to 16GB.

Fitting more Gaussians on one GPU:

- [CLM](https://arxiv.org/abs/2511.04951) (ASPLOS 2026): keeps position, scale and rotation on the GPU and the other attributes in CPU memory. It reports 47 million Gaussians on an 11GB GPU with 256GB of RAM. It is built on Grendel-GS and gsplat.
- [Taming 3DGS](https://arxiv.org/abs/2406.15643) (SIGGRAPH Asia 2024): densification that follows a set budget of Gaussians.
- [EAGLES](https://arxiv.org/abs/2312.04564) (ECCV 2024) and [Opti3DGS](https://arxiv.org/abs/2503.14475): coarse-to-fine training by image resolution or image frequency, with fewer Gaussians and lower memory.

Using several GPUs:

- [Grendel](https://arxiv.org/abs/2406.18533) (ICLR 2025): spreads the Gaussians over many GPUs. It does not apply to a single-GPU host.

## Parts of 3DRC That Help

- Multi-video merged SfM (ADR 0010): a starting point for solving poses per segment and merging.
- Scene-level Sim(3) alignment (ADR 0017): can bring block reconstructions into one frame.
- `data_device=cpu`: images already stay off the GPU.

SfM itself also has to scale. Solving thousands of frames in one hloc or COLMAP run is a separate limit that has not been measured.

## Evaluation Without LiDAR

The captures come from a mobile device and no LiDAR scan exists, so mesh accuracy against ground truth cannot be computed. What can be measured is novel-view PSNR, SSIM and LPIPS on held-out frames (training with `GS_EVAL_MODE="true"`), SfM statistics such as registered images and reprojection error, mesh self-consistency such as connected components, holes and normal consistency, and cost in time, peak GPU memory and model size. A few tape-measured distances would give a real-scale check.

## Open Questions

1. At what resolution and Gaussian count does one run fail on 16GB?
2. How much quality does dividing a scene cost compared with training it whole, and which division rule suits a mobile capture?
3. Is CPU offloading usable with 16GB of GPU memory and 60GB of RAM, and what would it take to bring it into the 3DRC backends, which use the Inria rasterizer?
4. Does a stopping rule based on the change in held-out PSNR carry over to other scenes?
5. Do iteration count and the schedule (`position_lr_max_steps`, `densify_until_iter`) need to scale with the number of frames? The training script changes only `--iterations`.

## Validation Plan at Small Scale

Test the division pipeline on the existing 371-frame scene before a large capture is available.

1. Record peak GPU memory and Gaussian count per run, for `DOWNSAMPLE_RATE` 4 and 2, at 30000 iterations.
2. Train once with `--eval` and several `--test_iterations` (for example 3000, 7000, 15000, 30000, 45000, 60000) to get a curve of held-out PSNR over iterations.
3. Split the cameras into 2 to 4 overlapping blocks by position, train each block, merge, and compare with the whole-scene run on held-out PSNR, Gaussian count and seams.
4. Extract a mesh from the merged result and from the whole-scene run and compare them.

## Related

- [Mesh Evaluation Methods](mesh_evaluation_methods.md)
- [GPU, CUDA and PyTorch Support](gpu_support.md)
- [ADR 0010](adr/0010-multi-video-merged-sfm-and-cache-integrity.md), [ADR 0017](adr/0017-scene-level-sim3-alignment-and-mesh-metrics-v2.md)
