---
title: Architecture Decision Records (ADR)
description: Index and operational rules for Architecture Decision Records in 3DRC.
category: adr
last_updated: 2026-10-26
---

# Architecture Decision Records (ADR)

This directory documents key architectural decisions, rationale, and context for the 3DRC pipeline.

## Writing Rules

- Do not edit existing entries. If a decision changes, add a new numbered entry below and update the status line of the old entry (e.g. `Superseded by 0007 (2026-08-08)`).
- Do not log trivial changes. Record only decisions that meet at least one criterion: hard to revert, took over 2 hours to diagnose, or will make future maintainers ask "why was this designed this way?".
- Include environment fingerprints for all performance numbers (GPU, CUDA version, commit hash, dataset).

## Index

- 0001-split-conda-envs.md: Split Conda Environments by Component
- 0002-blackwell-sm120-native-cubin.md: Blackwell sm_120 Native SASS Build (12.0+PTX)
- 0003-decoupling-buildtime-and-runtime-config.md: Decouple Buildtime and Runtime Configurations
- 0004-submodule-patches-over-fork.md: Manage Submodule Changes via patches/ Instead of Fork Branches
- 0005-colmap-camera-models.md: Expand COLMAP Camera Model Support (SIMPLE_RADIAL / RADIAL / OPENCV)
- 0006-complete-removal-of-fork-repositories.md: Complete Removal of Personal Fork Repositories in Favor of Direct Upstream Submodules and Patches
- 0007-scene-based-output-hierarchy.md: Scene-Based Output Hierarchy for Multi-Dataset Pipeline Isolation
- 0008-2d-gaussian-splatting-integration.md: 2D Gaussian Splatting Integration for Unbiased Depth and TSDF Mesh Reconstruction
- 0009-dynamic-provenance-and-subpackage-architecture.md: Dynamic Upstream Provenance Tracking and Subpackage Architecture
- 0010-multi-video-merged-sfm-and-cache-integrity.md: Multi-Video Merged SfM via Hybrid Sequential-Retrieval Matching and Cache Integrity
- 0011-scaffoldgs-integration.md: Scaffold-GS Integration for Anchor-Based Scene Reconstruction
- 0012-pytorch-stable-cu128-standardization.md: PyTorch Stable cu128 Standardization and Nightly Deprecation
- 0013-clean-output-hierarchy-and-subdirectories.md: Clean Output Hierarchy and Subdirectory Isolation Policy
- 0014-mipsplatting-integration-and-environment-isolation.md: Mip-Splatting Integration for 3D Smoothing Filter and Multi-Scale Rendering
- 0015-results-recording-and-aggregation.md: Results Recording and Aggregation for Cross-Engine Comparison
- 0016-gaussian-grouping-output-under-scene-outputs.md: Gaussian Grouping Output Under Scene Outputs
- 0017-scene-level-sim3-alignment-and-mesh-metrics-v2.md: Scene-Level Sim(3) Alignment and Mesh Metrics v2
- 0018-unified-gs-environment-with-per-backend-rasterizers.md: Unified Gaussian Environment with Per-Backend Rasterizer Builds
- 0019-defer-in-memory-yuv-ingestion.md: Defer In-Memory YUV Ingestion, Use Disk Extraction as Interim Path
- 0020-unified-gs-environment-on-cuda-13-2.md: Build the Unified gs Environment on CUDA 13.2 (torch cu130)
- 0021-remove-split-gs-environment-mode.md: Remove the Split gs_* Environment Mode
