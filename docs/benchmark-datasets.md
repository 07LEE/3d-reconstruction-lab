---
title: 3D Reconstruction Benchmark Datasets Guide
description: Standard benchmark datasets (Mip-NeRF 360, Tanks and Temples, Deep Blending, Synthetic NeRF), formats, and metrics.
category: datasets
last_updated: 2026-08-09
---

# 3D Reconstruction Benchmark Datasets Guide

This document details the standard 3D reconstruction and radiance field benchmark datasets used in the 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) evaluation framework, dataset formatting standards, camera parameter specifications, and integration workflows within the 3DRC repository.

## 1. Supported Benchmark Datasets

### 1.1 Mip-NeRF 360 Dataset

- Publication: Barron et al., "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields", CVPR 2022.
- Characteristics: Unbounded indoor and outdoor real-world scenes captured with complex backgrounds and varying lighting conditions.
- Outdoor Scenes: bicycle, flowers, garden, stump, treehill
- Indoor Scenes: bonsai, counter, kitchen, room
- Resolution Levels: Original 4K/8K images, with downsampled versions (images_2, images_4, images_8) commonly used for 3DGS optimization.

### 1.2 Tanks and Temples Dataset

- Publication: Knapitsch et al., "Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction", ACM TOG 2017.
- Characteristics: High-resolution video and photo sequences of outdoor and indoor objects/scenes.
- Key Benchmark Scenes: truck, train, barn, caterpillar, ignatius

### 1.3 Deep Blending Dataset

- Publication: Hedman et al., "Deep Blending for Free-Viewpoint Image Synthesis", ACM TOG 2018.
- Characteristics: Real-world indoor spaces captured with high geometric detail and view-dependent reflections.
- Key Benchmark Scenes: playroom, drjohnson

### 1.4 Synthetic NeRF (Blender) Dataset

- Publication: Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis", ECCV 2020.
- Characteristics: Synthetic 360-degree object captures with perfect ground-truth camera poses and transparent/reflective materials.
- Scenes: chair, drums, ficus, hotdog, lego, materials, mic, ship

## 2. Dataset Directory Structure

For 3DGS optimization and COLMAP pose estimation within 3DRC, datasets must be organized in the standard COLMAP / Inria dataset format:

```text
data/benchmark/<dataset_name>/<scene_name>/
├── images/                  # Primary input images (or raw images for SfM)
├── images_2/                # 2x downsampled images (optional)
├── images_4/                # 4x downsampled images (optional)
├── images_8/                # 8x downsampled images (optional)
├── sparse/
│   └── 0/                   # Pre-computed COLMAP sparse reconstruction
│       ├── cameras.bin      # Camera intrinsic parameters
│       ├── images.bin       # Camera extrinsic poses
│       └── points3D.bin     # Sparse point cloud initialization
└── cameras.json             # Synthetic NeRF camera configuration (if applicable)
```

## 3. Camera Models & Intrinsic Specs

The Inria 3DGS dataset loader requires specific camera model support:

1. PINHOLE / SIMPLE_PINHOLE: Standard focal length (fx, fy) and principal point (cx, cy) without radial distortion parameters.
2. SIMPLE_RADIAL / RADIAL / OPENCV: Extended camera models with radial distortion parameters k1, k2, p1, p2. Supported in 3DRC via gaussian-splatting patch 0001-colmap-camera-models.patch.

## 4. Benchmark Download & Setup

Use the included helper script to fetch and extract benchmark scenes:

```bash
./scripts/download_benchmark.sh mipnerf360 garden
./scripts/download_benchmark.sh tanks_and_temples truck
```

## 5. Quantitative Evaluation Methodology

When evaluating models trained on benchmark datasets, use the following metrics:

| Metric | Target Aspect | Evaluation Tool |
| --- | --- | --- |
| PSNR (Peak Signal-to-Noise Ratio) | Rendering color accuracy | gaussian-splatting/render.py |
| SSIM (Structural Similarity) | Structural texture similarity | gaussian-splatting/render.py |
| LPIPS (Learned Perceptual Image Patch Similarity) | Perceptual realism | gaussian-splatting/render.py |
| Chamfer Distance / Normal Consistency | Polygon mesh geometry quality | python src/eval_mesh.py --mesh <mesh.obj> |
