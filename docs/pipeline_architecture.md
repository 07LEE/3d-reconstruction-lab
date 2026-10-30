---
title: Pipeline Architecture Guide
description: Technical architecture, stage workflows, and execution guide for the 3DRC reconstruction pipeline.
category: guide
last_updated: 2026-10-30
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
│   Step 2: 3DGS Training │  (Inria 3DGS, PlanarGS, 2DGS Surfels, Scaffold-GS, Mip-Splatting)
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
│   Step 5: Evaluation    │  (Mesh: Topology Hygiene, Chamfer/F-score/HD95 vs LiDAR / 5b NVS: PSNR, SSIM, LPIPS)
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

### Scene Selection

Each step resolves the scene in this order: the dataset path argument (`data/<scene_name>`), the `SCENE_NAME` environment variable, then the most recently modified directory in `data/`. The last fallback prints a warning because any write into `data/<scene>/` (e.g. the SfM cache) changes which scene it picks. Set `REQUIRE_SCENE=true` to make a missing scene an error instead.

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
| Mip-Splatting | `02e_train_mipsplatting.sh` | Sparse COLMAP Poses | 3D smoothing filter and 2D anti-aliasing for multi-scale rendering |

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

# 3D smoothing filter Mip-Splatting training (Mip-Splatting)
./scripts/02e_train_mipsplatting.sh data/<scene_name>
```

- Output Checkpoint (Inria 3DGS): `outputs/<scene_name>/3dgs/inria_30k/`
- Output Checkpoint (PlanarGS): `outputs/<scene_name>/3dgs/planargs/`
- Output Checkpoint (2DGS): `outputs/<scene_name>/2dgs/`
- Output Checkpoint (Scaffold-GS): `outputs/<scene_name>/scaffoldgs/`
- Output Checkpoint (Mip-Splatting): `outputs/<scene_name>/mipsplatting/`

### Dynamic Training Checkpoint Configuration

By default, training scripts save intermediate model checkpoints at `10000`, `20000`, and `30000` iterations via `GS_CHECKPOINT_INTERVAL=10000` defined in `configs/default_config.sh`. To adjust the saving frequency:

```bash
# Custom checkpoint interval (e.g. save every 5,000 iterations)
GS_CHECKPOINT_INTERVAL=5000 ./scripts/02_train_3dgs.sh data/<scene_name>
```

### Reusing a Model Directory

Running a training stage again on the same scene works in the same model directory.

- When an earlier run left a checkpoint, the stage resumes from the latest one. A checkpoint that is already at the target iteration means the model is finished, so the stage reports that and ends without starting the trainer.
- A checkpoint older than the SfM sparse points is ignored and the stage trains from scratch.
- A directory that holds a model trained with the other `GS_EVAL_MODE` setting is refused. The trainers rewrite cfg_args on every start and restore whatever checkpoint they are given, so continuing such a model would keep the test views in training while cfg_args claims a holdout split, and the NVS evaluation would accept it. Move or delete the directory to train from scratch.

### Adding a Training Backend

A training backend is one row in configs/backends.txt plus one stage script. The row drives the train command, the shortcut and the help text of run_3drc.sh, the model directory and evaluation repository used by the NVS evaluation, and the rasterizer build chosen by activate_backend.sh. The columns are explained at the top of the file.

The stage scripts share their dataset, checkpoint and bookkeeping steps through scripts/utils/train_stage.sh, so a backend script only declares its trainer command:

```bash
source "$(dirname "$0")/utils/train_stage.sh"

train_env <backend> "$@"                # config, backend environment, patch verification
train_dataset <output_subdir> "$@"      # SfM model, images link, outputs/<scene_name>/<output_subdir>/
train_extra_args                        # EXTRA_TRAIN_ARGS: --eval, checkpoint, save and resume arguments

bench_start
python third_party/<repo>/train.py -s "$TARGET_DATA_DIR" -m "$MODEL_OUTPUT" ... "${EXTRA_TRAIN_ARGS[@]}"

train_finish <bench_stage> "<engine name>"   # timing record and pipeline_meta.json
```

- The trainer command stays in each script because its flags differ per backend. 02c_train_2dgs.sh is a minimal example.
- `train_env` and `train_dataset` are separate so a backend can prepare its toolchain in between (PlanarGS sources the CUDA build environment there).
- The repo column must be a submodule under third_party/, and the variant column must be one of the rasterizer builds made by scripts/00_setup_environment.sh. A backend with a rasterizer build of its own also needs entries in that script and in scripts/utils/verify_patches.sh. Submodules and patches are covered in the [Submodule Patches Guide](submodule_patches.md).
- The Python CLI (src/cli/main.py) keeps its own backend lists, and tests/test_cli.py fails when they drift from the registry. The mesh rules in run_3drc.sh (SuGaR needs a 3dgs or planargs model, TSDF needs a 2dgs model) are not in the registry either.
- The backend tables in tests/test_training_stages.py, tests/test_activate_backend.py and tests/test_nvs_backend.py pin each backend's command lines, so a new backend gets a row there. tests/smoke_test/run_smoke_test.sh runs every backend end to end on a GPU.

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

- Output: `outputs/<scene_name>/grouping/` (previously `third_party/gaussian-grouping/output/<scene_name>/`, which is no longer written).

### Step 4 Execution Commands

```bash
./scripts/04_train_grouping.sh data/<scene_name>
```

## Step 5: Mesh Topology and Geometric Evaluation (`src/mesh/eval_mesh.py`)

### Step 5 Overview & Metrics

Evaluates mesh surface topology and computes geometric distance against reference LiDAR point clouds.

1. Topology Audit: Watertightness (is_watertight), 2-manifold consistency, self-intersecting triangle count, and Euler characteristic (chi = V - E + F).
2. Geometric Distance: Chamfer L1 (mean of accuracy and completeness, 2M sampled points), F-score with precision / recall at a distance threshold (default 0.05 m, curve at 0.01 / 0.02 / 0.05 / 0.1 m), 95th percentile Hausdorff Distance (HD95), and GT coverage. Accuracy and completeness clip each distance at `--coverage-radius` (default 0.5 m), so GT regions the mesh misses count at the clip distance instead of being dropped.
3. Sim(3) 7-DoF Alignment: Estimated once per scene from the SfM sparse points the meshes were trained on (`data/<scene>/sparse/<active_sfm>/points3D.bin`), not from each mesh, so every mesh of a scene is scored in the same frame. The 24 PCA axis hypotheses are each refined with scale-aware ICP and the best fit is kept; the transform is cached in `outputs/<scene>/eval/alignment/` and reused until the sparse model or GT changes. See [ADR 0017](adr/0017-scene-level-sim3-alignment-and-mesh-metrics-v2.md).

### Limitations & Evaluation Notes

> [!NOTE]
>
> - Chamfer Distance Bias in Room-Scale Scenes: In typical indoor environments, walls, floors, and ceilings constitute the vast majority of surface area. As a result, global Chamfer Distance predominantly reflects wall flatness and may penalize models that preserve fine object geometry.
> - Alignment Check: `eval_results.json` records the alignment `fitness` (fraction of SfM points within `--align-threshold` of the GT after alignment) and an `ambiguous` flag set when a different rotation fits nearly as well (symmetric rooms). Low fitness or an ambiguous flag means the transform should be checked visually before the numbers are trusted.
> - Recommended Complementary Metrics: For holistic benchmarking, combine geometric distance with Novel View Synthesis (NVS) rendering metrics (PSNR, SSIM, LPIPS) on held-out views.

### Step 5 Execution Commands

```bash
# 1. Topology hygiene audit on reconstructed mesh
python -m src.mesh.eval_mesh --mesh outputs/<scene_name>/mesh/milo/mesh_cleaned_largest.ply

# 2. Accuracy against LiDAR Ground Truth PCD (scene-level alignment from the SfM sparse points)
python -m src.mesh.eval_mesh --mesh outputs/<scene_name>/mesh/2dgs/tsdf_mesh.ply --gt data/<scene_name>/lidar.pcd

# 3. Recompute the cached scene transform (e.g. after a manual check)
python -m src.mesh.eval_mesh --mesh outputs/<scene_name>/mesh/2dgs/tsdf_mesh.ply --gt data/<scene_name>/lidar.pcd --compute-transform

# 4. Evaluation with an explicit Sim(3) transform JSON (e.g. hand-verified)
python -m src.mesh.eval_mesh --mesh outputs/<scene_name>/mesh/sugar/sugarfine_mesh.obj --gt data/<scene_name>/lidar.pcd --transform outputs/<scene_name>/eval/alignment/sim3_hloc_lidar.json
```

`./scripts/run_3drc.sh pipeline` runs this step after mesh extraction on the mesh it produced (SuGaR `refined_mesh/<scene>/*.obj`, TSDF `tsdf_mesh.ply`, MILo `mesh_cleaned_largest.ply`). It uses `--gt <file>` (or `PIPELINE_GT`), else `data/<scene_name>/lidar.pcd` or `lidar.ply`; without GT it records topology only. A missing `--gt` file stops the pipeline before SfM.

`--align-ref` selects the alignment source: `auto` (default, scene SfM sparse points; falls back to the mesh with a warning when none is found), `mesh` (the previous per-mesh behavior), or a path to a sparse model / point cloud in the mesh frame. Results are written to `<mesh_dir>/eval/<mesh_stem>/eval_results.json` unless `--out` is given; the scene transform is written to `outputs/<scene_name>/eval/alignment/sim3_<sfm>_<gt_stem>.json`.

## Step 5b: Novel View Synthesis Evaluation (`scripts/05_eval_nvs.sh`)

Renders held-out test views with the backend's own `render.py --skip_train` and computes PSNR / SSIM / LPIPS with its `metrics.py`. Metrics are written to `<model_dir>/results.json` and `per_view.json`.

- The model must be trained with `GS_EVAL_MODE="true"` (every 8th view held out). Models without `eval=True` in `cfg_args` are rejected.
- A model trained without the holdout cannot be switched over by training again in its directory, because the stage refuses that (see Reusing a Model Directory). Move or delete the directory and train with `GS_EVAL_MODE="true"` from scratch.
- The backend is inferred from the model directory below outputs/<scene_name>/ (3dgs/inria_30k, 3dgs/planargs, 2dgs, scaffoldgs, mipsplatting, as listed in configs/backends.txt), so a scene or parent directory that carries a backend's name does not matter. Override with `NVS_BACKEND`.

```bash
GS_EVAL_MODE=true ./scripts/run_3drc.sh train 2dgs data/<scene_name>   # or set GS_EVAL_MODE in configs/default_config.sh
./scripts/run_3drc.sh nvs data/<scene_name> 2dgs
```

## Results Aggregation (`src/utils/collect_results.py`)

Stage scripts append runtime entries to `outputs/benchmark_timings.json` (stage, output directory, parameters, `resumed_from`, GPU / driver / torch / git commit fingerprint). The collector joins them with:

| Source | Metrics |
| --- | --- |
| `data/<scene>/sparse/<method>/` | Registered / total images, sparse point count |
| `outputs/<scene>/<model>/results.json`, `cfg_args`, `point_cloud/` | PSNR, SSIM, LPIPS, holdout flag, primitive count, model size |
| `outputs/<scene>/mesh/<engine>/eval/<mesh>/eval_results.json` | Vertices, faces, watertightness, sliver ratio, Chamfer L1, F-score / precision / recall, HD95, GT coverage, alignment fitness |

```bash
./scripts/run_3drc.sh results                          # outputs/results_summary.md + outputs/results_runs.csv
./scripts/run_3drc.sh results --publish docs/results.md  # also write a tracked copy
```
