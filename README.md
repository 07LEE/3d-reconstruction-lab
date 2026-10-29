# 3D Reconstruction Lab

3DRC turns photos or video frames of a scene into a 3D model. It estimates the camera poses, trains a Gaussian Splatting scene, extracts a mesh, and measures how good the result is. Several research implementations sit behind one command line so they can be compared on the same input.

It is a personal lab for experimenting and benchmarking, developed on an NVIDIA RTX 5070 Ti (Blackwell, 16GB).

## What it does

1. **Camera poses (SfM)**: hloc, COLMAP, VGGT, or FastMap.
2. **Gaussian training**: Inria 3DGS, PlanarGS, 2DGS, Scaffold-GS, Mip-Splatting.
3. **Mesh extraction**: SuGaR, MILo, or TSDF.
4. **Segmentation** (optional): Gaussian Grouping.
5. **Evaluation**: mesh accuracy against a LiDAR scan, novel-view quality (PSNR/SSIM/LPIPS), and one comparison report.

## Requirements

- Linux with Conda and an NVIDIA GPU (16GB+ VRAM recommended)
- A driver that supports CUDA 13.2

CUDA, PyTorch, and GPU generation details are in [GPU, CUDA and PyTorch Support](docs/gpu_support.md).

## Quick Start

```bash
# 1. Build the environments (compiles CUDA extensions, takes a while)
./scripts/00_setup_environment.sh
./scripts/utils/verify_patches.sh

# 2. Put your scene images in data/<scene_name>/raw_images/

# 3. Run everything: SfM -> training -> mesh -> mesh evaluation -> results report
./scripts/run_3drc.sh pipeline data/<scene_name> hloc --train 2dgs --mesh tsdf
```

For an Android YUV capture session, extract the frames with the `yuv-sensor` CLI and point `raw_images` at its `extracted_frames_undistorted` output ([ADR 0019](docs/adr/0019-defer-in-memory-yuv-ingestion.md)).

`pipeline` options: SuGaR needs `--train 3dgs|planargs`, TSDF needs `--train 2dgs`, MILo works with any backend. Mesh evaluation uses `data/<scene_name>/lidar.pcd` (or `lidar.ply`) as ground truth when present; `--gt` overrides it.

## Step by Step

```bash
# Step 1: camera poses
./scripts/run_3drc.sh sfm hloc data/<scene_name>

# Step 2: Gaussian training
./scripts/run_3drc.sh train 3dgs data/<scene_name>        # also: planargs, 2dgs, scaffoldgs, mipsplatting

# Step 3: mesh
./scripts/run_3drc.sh sugar data/<scene_name>
./scripts/run_3drc.sh milo data/<scene_name>
./scripts/run_3drc.sh tsdf data/<scene_name>

# Step 4: segmentation
./scripts/run_3drc.sh grouping data/<scene_name>

# Step 5: evaluation
./scripts/run_3drc.sh eval outputs/<scene_name>/mesh/milo/mesh_cleaned_largest.ply
./scripts/run_3drc.sh eval outputs/<scene_name>/mesh/2dgs/tsdf_mesh.ply data/<scene_name>/lidar.pcd
./scripts/run_3drc.sh nvs data/<scene_name> 3dgs/inria_30k   # model trained with GS_EVAL_MODE="true"

# Results
./scripts/run_3drc.sh outputs                                 # list generated artifacts
./scripts/run_3drc.sh results                                 # one comparison report
./scripts/run_3drc.sh results --publish docs/results.md
```

Each stage records its runtime and environment fingerprint to `outputs/benchmark_timings.json`; `results` joins SfM statistics, NVS metrics, mesh metrics, model sizes, and timings into `outputs/results_summary.md` and `outputs/results_runs.csv` ([ADR 0015](docs/adr/0015-results-recording-and-aggregation.md)). Unit tests run without a GPU: `pytest tests`.

## Methods

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

## Documentation

All submodules in `third_party/` reference official upstream repositories directly; custom changes are versioned patch files in `patches/`.

- [Pipeline Architecture Guide](docs/pipeline_architecture.md): End-to-end workflow from dataset preparation (Step 0) to evaluation (Step 5).
- [Conda Environments](docs/environments.md): The unified `gs` environment, the split `gs_*` environments, and the variables that select them.
- [GPU, CUDA and PyTorch Support](docs/gpu_support.md): Verified configuration, supported GPU generations, CUDA toolkit and driver requirements.
- [Blackwell (sm_120) Build Notes](docs/blackwell_build_notes.md): Troubleshooting matrix and native `sm_120` CUBIN build guide for RTX 50 Series.
- [Submodule Patches Guide](docs/submodule_patches.md): Patch catalog, `apply_patches.sh`, and `verify_patches.sh`.
- [Mesh Evaluation Methods](docs/mesh_evaluation_methods.md): How published papers evaluate meshes and obtain ground truth, compared with 3DRC.
- [Architecture Decision Records (ADR)](docs/adr/): Architecture decisions with operational writing rules.
