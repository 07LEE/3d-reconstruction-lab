---
title: Pipeline Architecture Guide
description: Technical architecture, stage workflows, and execution guide for the 3DRC reconstruction pipeline.
category: guide
last_updated: 2026-08-10
---

# 3DRC Pipeline Architecture Guide

This document provides technical details, configuration parameters, and execution guides for all pipeline steps in the 3DRC Lab environment.

## Pipeline Architecture Overview

```text
[ Raw Captures / Benchmark Datasets ]
            │
            ▼
┌─────────────────────────┐
│   Step 0: Prep & Format │  (Download Benchmark / Undistortion / Stereo)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Step 1: SfM & Pose    │  (hloc, VGGT-Omega, COLMAP, FastMap)
└───────────┬─────────────┘
            │  (Sparse Reconstruction & Intrinsic/Extrinsic Cameras)
            ▼
┌─────────────────────────┐
│   Step 2: 3DGS Training │  (Inria 3DGS, PlanarGS, 2DGS Surfels)
└───────────┬─────────────┘
            │  (Trained 3D/2D Gaussian Point Cloud Checkpoint)
            ▼
┌─────────────────────────┐
│  Step 3: Mesh Extraction│  (SuGaR: Surface-Aligned GS / MILo: MT SDF / TSDF: Open3D Voxel Fusion)
└───────────┬─────────────┘
            │  (UV-Textured OBJ / Compact Collision Mesh PLY / TSDF Mesh PLY)
            ▼
┌─────────────────────────┐
│   Step 4: Segmentation  │  (Gaussian Grouping 3D Instance Segmentation)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Step 5: 3D Mesh Eval  │  (3-Axis Topology Hygiene, Chamfer/Hausdorff LiDAR Metric)
└─────────────────────────┘
```

## Step 0: Benchmark Datasets & Preparation

### Supported Datasets & Structure

3DRC supports standard 3D reconstruction and radiance field benchmark datasets:

- Mip-NeRF 360: bicycle, flowers, garden, stump, treehill, bonsai, counter, kitchen, room
- Tanks and Temples: truck, train, barn, caterpillar, ignatius
- Deep Blending: playroom, drjohnson
- Synthetic NeRF / Blender: chair, drums, ficus, hotdog, lego, materials, mic, ship

Datasets must follow the standard COLMAP / Inria directory structure (`images/`, `sparse/0/` with `cameras.bin`, `images.bin`, `points3D.bin`).

### Download & Preprocessing Commands

```bash
# Fetch and extract standard benchmark dataset
./scripts/download_benchmark.sh mipnerf360 garden
./scripts/download_benchmark.sh tanks_and_temples truck

# High-throughput dense undistortion (zero-IO hardlinks)
python src/prep/dense_undistort.py --input_path data/<scene_name> --output_path data/<scene_name>_undistorted
```

## Step 1: Camera Pose Estimation (SfM)

### Step 1 Overview & Engines

Determines camera intrinsics, extrinsics, and sparse 3D point clouds from uncalibrated image sequences.

| Method | Script | Input Requirements | Key Strengths |
| --- | --- | --- | --- |
| hloc (Default) | `01_sfm_hloc.sh` | RGB Images | Deep SuperPoint + SuperGlue (Sequential, Exhaustive, Sequential+Retrieval for multi-video) |
| VGGT-Omega | `01_sfm_hloc.sh vggt` | RGB Images | Feed-forward transformer pose estimation |
| FastMap | `01_sfm_hloc.sh fastmap` | RGB Images | Fast keypoint matching for large scenes |
| COLMAP | `01_sfm_hloc.sh sfm` | RGB Images | Standard SIFT + incremental SfM |

### Step 1 Execution Commands

```bash
# Standard deep learning SfM (hloc)
./scripts/01_sfm_hloc.sh

# Multi-video merged SfM with NetVLAD global retrieval
SFM_STRATEGY="sequential+retrieval" ./scripts/01_sfm_hloc.sh hloc data/<multi_video_scene>

# Transformer-based instant pose estimation (VGGT)
./scripts/01_sfm_hloc.sh vggt
```

## Step 2: 3D & 2D Gaussian Training

### Step 2 Overview & Engines

Optimizes 3D/2D Gaussian Splatting scene representations using differentiable rasterization.

| Method | Script | Input Requirements | Key Strengths |
| --- | --- | --- | --- |
| Inria 3DGS (Default) | `02_train_3dgs.sh` | Sparse COLMAP Poses | Reference implementation baseline |
| PlanarGS | `02b_train_planargs.sh` | Sparse COLMAP Poses | Planar regularization on detected indoor surfaces |
| 2DGS | `02c_train_2dgs.sh` | Sparse COLMAP Poses | 2D planar surfel representation with analytical ray-splat intersection |
| Scaffold-GS | `02d_train_scaffoldgs.sh` | Sparse COLMAP Poses | Anchor-based representation for view-dependent object detail |

### Step 2 Execution Commands

```bash
# Reference 3DGS training
./scripts/02_train_3dgs.sh data/<scene_name>

# Planar-regularized 3DGS training (PlanarGS)
./scripts/02b_train_planargs.sh data/<scene_name>

# 2D Gaussian Splatting surfel training (2DGS)
./scripts/02c_train_2dgs.sh data/<scene_name>

# Anchor-based Scaffold-GS training (Scaffold-GS)
./scripts/02d_train_scaffoldgs.sh data/<scene_name>
```

- Output Checkpoint (Inria 3DGS): `outputs/<scene_name>/3dgs/inria_30k/`
- Output Checkpoint (PlanarGS): `outputs/<scene_name>/3dgs/planargs/`
- Output Checkpoint (2DGS): `outputs/<scene_name>/2dgs/`
- Output Checkpoint (Scaffold-GS): `outputs/<scene_name>/scaffoldgs/`

## Step 3: 3D Mesh Reconstruction (SuGaR, MILo & TSDF)

### Step 3 Overview & Engines

Extracts polygon meshes for rendering, collision checking, and geometry analysis.

- SuGaR (`./scripts/03_train_sugar.sh`): Surface-Aligned Gaussian Regularization for UV-textured OBJ models.
- MILo (`./scripts/03b_train_milo.sh`): Mesh-in-the-loop optimization for compact collision meshes.
- TSDF (`./scripts/03c_mesh_tsdf.sh`): Open3D TSDF voxel integration from rendered depth and normal maps.

### Step 3 Execution Commands

```bash
# SuGaR Mesh Reconstruction
./scripts/03_train_sugar.sh data/<scene_name>

# MILo Compact Mesh Reconstruction
./scripts/03b_train_milo.sh data/<scene_name>

# TSDF Mesh Extraction (from 2DGS or custom model)
./scripts/03c_mesh_tsdf.sh data/<scene_name> 2dgs
```

## Step 4: Gaussian Grouping Object Segmentation

### Step 4 Overview & Engine

Learns 3D Identity Embeddings lifted from SAM 2D masks for object-level segmentation.

### Step 4 Execution Commands

```bash
./scripts/04_train_grouping.sh
```

## Step 5: Mesh Topology and Geometric Evaluation (`src/mesh/eval_mesh.py`)

### Step 5 Overview & Metrics

Evaluates mesh surface topology and computes geometric distance against reference LiDAR point clouds.

1. Topology Audit: Watertightness (is_watertight), 2-manifold consistency, self-intersecting triangle count, and Euler characteristic (chi = V - E + F).
2. Geometric Distance: Symmetric Chamfer Distance (CD, 2M sampled points) and Hausdorff Distance (HD).
3. Sim(3) 7-DoF Alignment: Umeyama RANSAC scale and rigid alignment against reference coordinate systems.

### Limitations & Evaluation Notes

> [!NOTE]
>
> - Chamfer Distance Bias in Room-Scale Scenes: In typical indoor environments, walls, floors, and ceilings constitute the vast majority of surface area. As a result, global Chamfer Distance predominantly reflects wall flatness and may penalize models that preserve fine object geometry.
> - Recommended Complementary Metrics: For holistic benchmarking, combine geometric distance with Novel View Synthesis (NVS) rendering metrics (PSNR, SSIM, LPIPS) on held-out views.

### Step 5 Execution Commands

```bash
# 1. Topology hygiene audit on reconstructed mesh
python src/mesh/eval_mesh.py --mesh outputs/<scene_name>/mesh/milo/mesh_cleaned_largest.ply

# 2. Chamfer & Hausdorff accuracy against LiDAR Ground Truth PCD
python src/mesh/eval_mesh.py --mesh outputs/<scene_name>/mesh/2dgs/tsdf_mesh.ply --gt data/<scene_name>/lidar.pcd

# 3. Evaluation with Sim(3) transform JSON
python src/mesh/eval_mesh.py --mesh outputs/<scene_name>/mesh/sugar/sugarfine_mesh.obj --gt data/<scene_name>/lidar.pcd --transform outputs/<scene_name>/eval/sim3_transform.json
```
