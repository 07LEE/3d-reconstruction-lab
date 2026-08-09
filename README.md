# 3D Reconstruction Test Workspace

A personal workspace for hands-on experimentation with 3D reconstruction pipelines — SfM, Gaussian Splatting, and mesh extraction — on Blackwell hardware.

## System Requirements

- NVIDIA GPU with CUDA support (16GB+ VRAM recommended)
- CUDA Toolkit 12.8+ (required for `sm_120`; 12.4+ for older architectures)
- PyTorch 2.6+ with matching CUDA build

## Verified Configuration

- GPU: NVIDIA GeForce RTX 5070 Ti (Blackwell Architecture, `sm_120`, 16GB VRAM)
- CUDA: CUDA 12.8 Toolkit / PyTorch 2.12.0+cu128
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

# 2. Run Step 1 Camera Pose Estimation (COLMAP, hloc, vi_sfm, VGGT)
./scripts/run_3drc.sh sfm

# 3. Run Step 2 Gaussian Training (Inria 3DGS / PlanarGS / 2DGS)
./scripts/run_3drc.sh train 3dgs
./scripts/run_3drc.sh train planargs
./scripts/run_3drc.sh train 2dgs

# 4. Run Step 3 Mesh Reconstruction (SuGaR / MILo / 2DGS TSDF)
./scripts/run_3drc.sh sugar
./scripts/run_3drc.sh milo
./scripts/run_3drc.sh tsdf

# 5. Evaluate 3D Mesh Topology & Quantitative Accuracy
./scripts/run_3drc.sh eval outputs/<scene_name>/mesh/milo/mesh_cleaned_largest.ply
./scripts/run_3drc.sh eval outputs/<scene_name>/mesh/2dgs/tsdf_mesh.ply data/<scene_name>/lidar.pcd

# 6. Inspect Generated Artifacts
./scripts/run_3drc.sh outputs
```

## Environment Architecture

- `gs_train`: Inria 3DGS and 2DGS training environment (`dr_aa` & `diff-surfel-rasterization`).
- `gs_sugar`: SuGaR mesh extraction environment.
- `gs_milo`: MILo differentiable mesh-in-the-loop training and extraction environment.
- `gs_group`: Gaussian Grouping segmentation environment.

## Pipeline Model Mapping

| Tool | Pipeline Step | Core Role | Methodology |
| --- | --- | --- | --- |
| COLMAP (Default) | Step 1 (SfM) | Classic camera pose estimation | SIFT keypoint extraction + Incremental SfM BA |
| hloc | Step 1 (SfM) | Deep learning camera pose estimation | SuperPoint + SuperGlue with COLMAP BA |
| VGGT-Omega | Step 1 (SfM) | Immediate pose estimation | Feed-forward Visual Geometry Transformer |
| vi_sfm | Step 1 (SfM) | Visual-Inertial pose estimation | RGB + IMU fusion with gravity alignment |
| 3DGS (Inria) | Step 2 (Training) | High-fidelity 3D scene optimization | Differentiable 3D Gaussian rasterization |
| PlanarGS | Step 2 (Training) | Indoor planar-regularized 3DGS | Selective planar priors on walls/floors |
| 2DGS | Step 2c (Training) | Planar surfel representation | Exact ray-splat intersection with 2D oriented disks |
| SuGaR | Step 3 (Mesh Extraction) | UV-textured OBJ polygon mesh | Surface-Aligned Gaussian Regularization |
| MILo | Step 3b (Mesh Extraction) | Compact 3D collision mesh | Differentiable Mesh-in-the-loop Optimization |
| TSDF (2DGS) | Step 3c (Mesh Extraction) | Crisp, floater-free polygon mesh | Volumetric Open3D TSDF Integration from surfel depth/normals |
| Gaussian Grouping | Step 4 (Segmentation) | 3D object instance segmentation | 3D Identity Embedding from SAM masks |

## Submodule Patches Summary

All submodules in `third_party/` reference official upstream repositories directly. Custom modifications are maintained as versioned patch files:

| Target Submodule | Patch File | Purpose |
| --- | --- | --- |
| `gaussian-splatting` | `0001-colmap-camera-models.patch` | Support for `SIMPLE_RADIAL`, `RADIAL`, `OPENCV` camera models |
| `gaussian-splatting` | `0002-distcuda2-scipy-fallback.patch` | `scipy.spatial.KDTree` fallback for PCD initialization |
| `diff-gaussian-rasterization` | `0001-zero-init-state-structs.patch` | Zero-initialization of CUDA state structs |
| `diff-gaussian-rasterization` | `0002-cstdint-include.patch` | `<cstdint>` header include for GCC 13+/14 |
| `sugar` | `0001-sugar-extension-dup-and-cpu-device.patch` | Extension fix, CPU `data_device` VRAM OOM fix, `weights_only=False` PyTorch 2.6+ fix |
| `milo` | `0001-cstdint-and-cmake-fixes.patch` | `<cstdint>` includes for GCC 13+ in rasterizers and CMake 4.4 CXX standard / pybind11 tag fixes |
| `vggt` | `0001-pycolmap-313-compat.patch` | PyCOLMAP 3.13 text export compatibility fix |
| `2d-gaussian-splatting` | `0001-colmap-camera-models.patch` | Support for `SIMPLE_RADIAL` and `RADIAL` camera models in 2DGS |
| `2d-gaussian-splatting` | `0002-mesh-utils-bounds-and-empty-fix.patch` | Empty cluster bounds guard in mesh postprocessing |
| `diff-surfel-rasterization` | `0001-cstdint-and-zero-init.patch` | `<cstdint>` includes, zero-init structs, and backward SH gradient dimension match |

## Detailed Documentation Guides

For in-depth technical guides, execution options, and evaluation methodologies, refer to the documentation in [`docs/`](docs/):

- [Pipeline Architecture Guide](docs/pipeline_architecture.md): Detailed workflow from SfM (Step 1) to Segmentation (Step 4).
- [Pipeline Backends Guide](docs/backends.md): Living record of wired pipeline backends, integration status, and trade-offs.
- [Blackwell (sm_120) Build Notes](docs/blackwell_build_notes.md): Troubleshooting matrix and native sm_120 CUBIN build guide.
- [Mesh Reconstruction & Evaluation Guide](docs/mesh_reconstruction_and_eval.md): SuGaR/MILo/TSDF mesh extraction and `src/eval_mesh.py` 3-axis quantitative evaluation.
- [Submodule Patches Guide](docs/submodule_patches.md): Patch maintenance, `apply_patches.sh`, and `verify_patches.sh` mechanisms.
- [Architecture Decision Records (ADR)](docs/adr/): Individual architecture decision records (0001-0008) with writing rules.
