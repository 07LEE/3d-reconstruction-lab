---
title: Pipeline Backends
description: Catalog of wired implementations across pipeline stages, rationale, and operational status.
category: architecture
last_updated: 2026-08-09
---

# Backends

Which implementations are wired into the pipeline, why each was added, and whether it currently runs.

This is a living document — update it when a backend is added, promoted, or dropped. (Unlike DECISIONS.md, whose entries are frozen once written.)

Versions and commit SHAs deliberately live in envs/*.yml and submodule gitlinks, not here. This file answers why and does it work, nothing else.

## Status Legend

- active: Used in the default pipeline path
- optional: Works, selected via config, not on the default path
- untested: Wired in, never successfully run end-to-end
- dropped: Removed — kept in this table so the reason survives
- candidate: Under evaluation, not yet integrated

## Stage 1 — SfM / Pose Estimation

| Backend | Status | Why It's Here | Notes |
| --- | --- | --- | --- |
| hloc | active | Default path. Learned features hold up better than classic COLMAP on low-texture indoor scenes. | CameraMode.SINGLE SIMPLE_RADIAL, undistorted to PINHOLE — see ADR 0005 |
| COLMAP | active | Used as hloc's backend for triangulation / bundle adjustment | Not an independent path |
| vi_sfm | untested | IMU gravity alignment — produces a Z-up reconstruction so downstream meshes land in a usable world frame | match_features.main kwargs fixed and camera extrinsics now rotate with points3D, but no successful full run yet |
| vggt | optional | Feed-forward camera pose and point estimation via deep network priors without iterative bundle adjustment, enabling rapid coarse pose initialization. | PyCOLMAP 3.13 adapter patched; optional fast pose baseline |
| fastmap | optional | GPU-accelerated fast keypoint matching and mapping alternative to standard COLMAP, drastically reducing pose estimation time on large datasets. | Dedicated viewer script supported in scripts/utils/view_reconstruction.sh |

## Stage 2 — Gaussian Representation

| Backend | Status | Why It's Here | Notes |
| --- | --- | --- | --- |
| Inria 3DGS | active | Reference implementation. Baseline everything else is compared against. | dr_aa variant (2024-10); requires patches 0001/0002 + zero-init |
| PGSR | active | Planar constraints improve wall/floor geometry, which TSDF fusion depends on | Current default for the mesh path |
| MILo | active | Jointly optimizes a mesh with the Gaussians — claims ~10x fewer vertices than SuGaR | Motivation is collision-mesh practicality, not visual fidelity |
| Gaussian Grouping | optional | Per-object segmentation masks — prerequisite for instance-separated mesh extraction | Requires gs_group environment |
| PlanarGS | optional | Selective planar priors on detected indoor wall/floor regions | Integrated in third_party/PlanarGS (SJTU-ViSYS-team/PlanarGS). Primary candidate against PGSR. |
| 2DGS | optional | 2D surfel planar disks with exact ray-splat intersection; renders unbiased depth and surface normals | Wired in scripts/02c_train_2dgs.sh with Blackwell sm_120 support |

## Stage 3 — Mesh Extraction

| Backend | Status | Why It's Here | Notes |
| --- | --- | --- | --- |
| TSDF fusion | active | Default extraction from PGSR depth maps | Voxel-size sweep not yet done |
| SuGaR | active | Comparison baseline for MILo's vertex-count claim | Needs CPU data_device patch (14.5 GiB -> 1.4 GiB VRAM) |
| MILo (SDF) | active | mesh_extract_sdf.py — Marching Tetrahedra SDF isosurface extraction | Requires PINHOLE undistorted input dataset |
| 2DGS (TSDF) | optional | Open3D TSDF integration from 2DGS surfel depth/normal renders | Wired in scripts/03c_mesh_2dgs.sh |
| GOF / SDFRaster | candidate | Fallback if instance-separated TSDF and MILo both fall short on object detail | Under consideration for fine detail recovery |

## Stage 4 — Evaluation

| Backend | Status | Why It's Here | Notes |
| --- | --- | --- | --- |
| eval_mesh.py | active | Chamfer distance against LiDAR GT point cloud | Known limitation: global Chamfer on a room-scale scene is close to a wall-flatness score. A method that destroys object detail can still win. Planar/object region-split aggregation (RANSAC plane fit on the GT cloud) is needed before these numbers mean anything. |

## Open Questions

- vi_sfm has never completed a run. Decide whether to finish it or mark it dropped — an untested entry that stays untested for months is just noise.
- PlanarGS vs PGSR planar regularization quality on indoor scenes remains to be benchmarked with eval_mesh.py.
