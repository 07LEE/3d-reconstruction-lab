# 3D Reconstruction Lab

A personal lab environment for hands-on experimentation and benchmarking of 3D reconstruction pipelines — SfM, Gaussian Splatting variants, and mesh extraction engines — on Blackwell hardware.

## System Requirements

- NVIDIA GPU with CUDA support (16GB+ VRAM recommended)
- CUDA Toolkit 12.8+ (required for `sm_120`; 12.4+ for older architectures)
- PyTorch 2.11.0+cu128 (Stable, `sm_120` native CUBIN required for Blackwell)

## Verified Configuration

- GPU: NVIDIA GeForce RTX 5070 Ti (Blackwell Architecture, `sm_120`, 16GB VRAM)
- CUDA: CUDA 12.8 Toolkit / PyTorch 2.11.0+cu128
- Host OS: Ubuntu 24.04 LTS (GCC 12 host compiler via `-ccbin /usr/bin/g++-12`)
- CUBIN Binaries: Native `sm_120` SASS CUBIN with `12.0+PTX` fallback
- Rasterizer Performance Benchmark (RTX 5070 Ti, 1,112-frame indoor scene):
  - PTX JIT Fallback (`sm_90` SASS + JIT on `sm_120`): `~37.18 it/s`
  - Native `sm_120` SASS CUBIN: `~66.09 it/s` (`~1.78x` throughput performance gain)

## Quick Start

```bash
# 1. Apply tracked patches and verify environment
./scripts/utils/apply_patches.sh
./scripts/utils/verify_patches.sh

# 2. Run Step 1 Camera Pose Estimation (COLMAP, hloc, VGGT, FastMap)
./scripts/run_3drc.sh sfm

# 3. Run Step 2 Gaussian Training (Inria 3DGS / PlanarGS / 2DGS / Scaffold-GS / Mip-Splatting)
./scripts/run_3drc.sh train 3dgs
./scripts/run_3drc.sh train planargs
./scripts/run_3drc.sh train 2dgs
./scripts/run_3drc.sh train scaffoldgs
./scripts/run_3drc.sh train mipsplatting

# 4. Run Step 3 Mesh Reconstruction (SuGaR / MILo / TSDF)
./scripts/run_3drc.sh sugar
./scripts/run_3drc.sh milo
./scripts/run_3drc.sh tsdf

# 5. Evaluate 3D Mesh Topology & Geometric Accuracy
./scripts/run_3drc.sh eval outputs/<scene_name>/mesh/milo/mesh_cleaned_largest.ply
./scripts/run_3drc.sh eval outputs/<scene_name>/mesh/2dgs/tsdf_mesh.ply data/<scene_name>/lidar.pcd

# 5b. Evaluate NVS metrics on held-out views (model trained with GS_EVAL_MODE="true")
./scripts/run_3drc.sh nvs data/<scene_name> 3dgs/inria_30k

# 6. Inspect Generated Artifacts
./scripts/run_3drc.sh outputs

# 7. Aggregate results into one comparison report
./scripts/run_3drc.sh results
./scripts/run_3drc.sh results --publish docs/results.md
```

## Results Recording

Each stage records its runtime and environment fingerprint to `outputs/benchmark_timings.json`. `./scripts/run_3drc.sh results` joins SfM statistics, NVS metrics, mesh metrics, model sizes, and timings into `outputs/results_summary.md` and `outputs/results_runs.csv`; `--publish docs/results.md` keeps a tracked copy. NVS metrics are reported only for models trained with `GS_EVAL_MODE="true"`. See [ADR 0015](docs/adr/0015-results-recording-and-aggregation.md).

The collector and timing helpers have unit tests that run without a GPU:

```bash
pytest tests
```

## Environment Architecture

- `3drc`: Main pipeline orchestrator, Step 1 SfM (COLMAP, hloc, VGGT, FastMap), preprocessing (`dense_undistort`), and Step 5 geometric evaluation (`eval_mesh`).
- `gs_train`: Inria 3DGS, PlanarGS, and 2DGS training (`dr_aa` & `diff-surfel-rasterization`), plus Open3D TSDF mesh extraction.
- `gs_sugar`: SuGaR surface-aligned mesh extraction environment.
- `gs_milo`: MILo differentiable mesh-in-the-loop training and extraction environment.
- `gs_scaffold`: Scaffold-GS anchor-based training environment (isolated rasterizer build).
- `gs_mipsplatting`: Mip-Splatting 3D smoothing filter training environment (isolated rasterizer build).
- `gs_group`: Gaussian Grouping 3D SAM instance segmentation environment.

## Pipeline Model Mapping

| Tool | Pipeline Step | Core Role | Methodology |
| --- | --- | --- | --- |
| [COLMAP](https://github.com/colmap/colmap) | Step 1 (SfM) | Camera pose estimation | SIFT keypoint extraction + Incremental SfM BA |
| [hloc](https://github.com/cvg/Hierarchical-Localization) | Step 1 (SfM) | Feature-matched camera pose estimation | SuperPoint + SuperGlue with COLMAP BA |
| [VGGT-Omega](https://github.com/facebookresearch/vggt) | Step 1 (SfM) | Feed-forward pose estimation | Visual Geometry Transformer pose initialization |
| [FastMap](https://github.com/pals-ttic/fastmap) | Step 1 (SfM) | GPU-accelerated pose estimation | Fast keypoint matching and mapping |
| [3DGS (Inria)](https://github.com/graphdeco-inria/gaussian-splatting) | Step 2 (Training) | 3D Gaussian scene optimization | Differentiable 3D Gaussian rasterization |
| [PlanarGS](https://github.com/SJTU-ViSYS-team/PlanarGS) | Step 2b (Training) | Planar-regularized 3DGS | Planar priors on detected indoor surfaces |
| [2DGS](https://github.com/hbb1/2d-gaussian-splatting) | Step 2c (Training) | Planar surfel representation | Exact ray-splat intersection with 2D oriented disks |
| [Scaffold-GS](https://github.com/city-super/scaffold-gs) | Step 2d (Training) | Anchor-based 3DGS representation | Learnable anchors for view-dependent attribute prediction |
| [Mip-Splatting](https://github.com/autonomousvision/mip-splatting) | Step 2e (Training) | Alias-free 3DGS rendering | 3D smoothing filter and 2D Mip filter |
| [SuGaR](https://github.com/Anttwo/SuGaR) | Step 3 (Mesh Extraction) | UV-textured OBJ polygon mesh | Surface-Aligned Gaussian Regularization |
| [MILo](https://github.com/Anttwo/MILo) | Step 3b (Mesh Extraction) | Compact collision mesh | Differentiable Mesh-in-the-loop MT SDF |
| [TSDF (Open3D)](https://github.com/isl-org/Open3D) | Step 3c (Mesh Extraction) | Volumetric polygon mesh | Volumetric Open3D TSDF Integration from depth/normals |
| [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping) | Step 4 (Segmentation) | 3D object instance segmentation | 3D Identity Embedding from SAM masks |

## Submodule Patches

All submodules in `third_party/` reference official upstream repositories directly. Custom modifications (e.g. Blackwell sm_120 CUBIN build fixes, camera model extensions) are maintained as versioned patch files in `patches/`.

For the complete patch catalog, patch application, and verification workflows, refer to the [Submodule Patches Guide](docs/submodule_patches.md).

## Detailed Documentation Guides

For in-depth technical guides, execution options, and evaluation methodologies, refer to the documentation in [`docs/`](docs/):

- [Pipeline Architecture Guide](docs/pipeline_architecture.md): Complete end-to-end workflow from Dataset Preparation (Step 0) to SfM, Gaussian Training, Mesh Extraction, and Quantitative Evaluation (Step 5).
- [Blackwell (sm_120) Build Notes](docs/blackwell_build_notes.md): Troubleshooting matrix and native sm_120 CUBIN build guide for RTX 50 Series.
- [Submodule Patches Guide](docs/submodule_patches.md): Patch maintenance, `apply_patches.sh`, and `verify_patches.sh` mechanisms for official upstream submodules.
- [Architecture Decision Records (ADR)](docs/adr/): Architecture decision records with operational writing rules.
