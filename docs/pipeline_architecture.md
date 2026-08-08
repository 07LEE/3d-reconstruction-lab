# 3DRC Pipeline Architecture Guide

This document provides technical details, configuration parameters, and execution guides for all pipeline steps in the 3DRC workspace.

## Pipeline Architecture Overview

```text
[ Input Images / RGB-D / IMU ]
            │
            ▼
┌─────────────────────────┐
│   Step 1: SfM & Pose    │  (hloc, VGGT-Omega, vi_sfm, COLMAP, FastMap)
└───────────┬─────────────┘
            │  (Sparse Reconstruction & Intrinsic/Extrinsic Cameras)
            ▼
┌─────────────────────────┐
│   Step 2: 3DGS Training │  (Inria 3DGS with antialiasing rasterizer)
└───────────┬─────────────┘
            │  (Trained 3D Gaussian Point Cloud Checkpoint)
            ▼
┌─────────────────────────┐
│  Step 3: Mesh Extraction│  (SuGaR: Surface-Aligned GS / MILo: Mesh-In-the-Loop)
└───────────┬─────────────┘
            │  (UV-Textured OBJ / Compact Collision Mesh PLY)
            ▼
┌─────────────────────────┐
│   Step 4: Segmentation  │  (Gaussian Grouping 3D Instance Segmentation)
└─────────────────────────┘
```

## Step 1: Camera Pose Estimation (SfM)

### Step 1 Overview & Engines

Determines camera intrinsics, extrinsics, and sparse 3D point clouds from uncalibrated image sequences.

| Method | Script | Input Requirements | Key Strengths |
| --- | --- | --- | --- |
| hloc (Default) | `01_sfm_hloc.sh` | RGB Images | Deep SuperPoint + SuperGlue matching |
| vi_sfm | `01_sfm_hloc.sh vi_sfm` | RGB Images + IMU `imu_data.csv` | Metric scale, gravity alignment |
| VGGT-Omega | `01_sfm_hloc.sh vggt` | RGB Images | Feed-forward transformer pose estimation |
| FastMap | `01_sfm_hloc.sh fastmap` | RGB Images | Fast keypoint matching for large scenes |
| COLMAP | `01_sfm_hloc.sh sfm` | RGB Images | Standard SIFT + incremental SfM |

### Step 1 Execution Commands

```bash
# Standard deep learning SfM (hloc)
./scripts/01_sfm_hloc.sh

# Visual-Inertial SfM (RGB + IMU with gravity vector alignment)
./scripts/01_sfm_hloc.sh vi_sfm

# Transformer-based instant pose estimation (VGGT)
./scripts/01_sfm_hloc.sh vggt
```

## Step 2: High-Density 3DGS Training

### Step 2 Overview & Engine

Optimizes a 3D Gaussian Splatting scene representation using differentiable rasterization.

### Step 2 Execution Commands

```bash
./scripts/02_train_3dgs.sh
```

- Output Checkpoint: `outputs/gs_final_precision/`
- Output PointCloud: `outputs/gs_final_precision/point_cloud/iteration_30000/point_cloud.ply`

## Step 3: 3D Mesh Reconstruction (SuGaR & MILo)

### Step 3 Overview & Engines

Extracts polygon meshes for 3D rendering, physics collision, and downstream applications.

- SuGaR (`./scripts/03_train_sugar.sh`): Surface-Aligned Gaussian Regularization for high-poly UV-textured OBJ models.
- MILo (`./scripts/03b_train_milo.sh`): Mesh-in-the-loop optimization for compact 3D collision meshes (10x fewer vertices).

### Step 3 Execution Commands

```bash
# SuGaR Mesh Reconstruction
./scripts/03_train_sugar.sh

# MILo Compact Mesh Reconstruction
./scripts/03b_train_milo.sh
```

## Step 4: Gaussian Grouping Object Segmentation

### Step 4 Overview & Engine

Learns 3D Identity Embeddings lifted from SAM 2D masks for object-level 3D manipulation.

### Step 4 Execution Commands

```bash
./scripts/04_train_grouping.sh
```
