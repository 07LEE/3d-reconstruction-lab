# 3D Reconstruction Test Workspace

A research workspace dedicated to testing 3D reconstruction algorithms, from state-of-the-art SfM to 3D Gaussian Splatting (3DGS), SuGaR, MILo, and Gaussian Grouping.

## Quick Start

```bash
# 1. Apply tracked patches and verify environment
./scripts/utils/apply_patches.sh
./scripts/utils/verify_patches.sh

# 2. Run Step 1 Camera Pose Estimation (hloc, vi_sfm, VGGT)
./scripts/run_3drc.sh sfm vi_sfm

# 3. Run Step 2 3DGS Training
./scripts/run_3drc.sh train

# 4. Run Step 3 Mesh Reconstruction (SuGaR / MILo)
./scripts/03_train_sugar.sh
./scripts/03b_train_milo.sh

# 5. Evaluate 3D Mesh Topology & Accuracy
python src/eval_mesh.py --mesh outputs/sugar_mesh/.../mesh.obj
```

## Environment Architecture

- `gs_train`: Inria 3DGS training environment (`dr_aa` rasterizer).
- `gs_sugar`: SuGaR mesh extraction environment.
- `gs_milo`: MILo differentiable mesh-in-the-loop training and extraction environment.
- `gs_group`: Gaussian Grouping segmentation environment.

## Pipeline Model Mapping

| Tool | Pipeline Step | Core Role | Methodology |
|---|---|---|---|
| COLMAP / hloc | Step 1 (SfM) | Camera pose estimation | SIFT / SuperPoint+SuperGlue SfM |
| VGGT-Omega | Step 1 (SfM) | Immediate pose estimation | Feed-forward Visual Geometry Transformer |
| vi_sfm | Step 1 (SfM) | Visual-Inertial pose estimation | RGB + IMU fusion with gravity alignment |
| 3DGS (Inria) | Step 2 (Training) | High-fidelity 3D scene optimization | Differentiable 3D Gaussian rasterization |
| SuGaR | Step 3 (Mesh Extraction) | UV-textured OBJ polygon mesh | Surface-Aligned Gaussian Regularization |
| MILo | Step 3b (Mesh Extraction) | Compact 3D collision mesh | Differentiable Mesh-in-the-loop Optimization |
| Gaussian Grouping | Step 4 (Segmentation) | 3D object instance segmentation | 3D Identity Embedding from SAM masks |

## Submodule Patches Summary

| Target Submodule | Patch File | Purpose |
|---|---|---|
| `gaussian-splatting` | `0001-colmap-camera-models.patch` | Support for `SIMPLE_RADIAL`, `RADIAL`, `OPENCV` camera models |
| `gaussian-splatting` | `0002-distcuda2-scipy-fallback.patch` | `scipy.spatial.KDTree` fallback for PCD initialization |
| `diff-gaussian-rasterization` | `0001-zero-init-state-structs.patch` | Zero-initialization of CUDA state structs |
| `diff-gaussian-rasterization` | `0002-cstdint-include.patch` | `<cstdint>` header include for GCC 13+/14 |
| `sugar` | `0001-sugar-extension-dup-and-cpu-device.patch` | Extension fix, CPU `data_device` VRAM OOM fix, `weights_only=False` PyTorch 2.6+ fix |

## Detailed Documentation Guides

For in-depth technical guides, execution options, and evaluation methodologies, refer to the documentation in [`docs/`](docs/):

- [Pipeline Architecture Guide](docs/pipeline_architecture.md): Detailed workflow from SfM (Step 1) to Segmentation (Step 4).
- [Mesh Reconstruction & Evaluation Guide](docs/mesh_reconstruction_and_eval.md): SuGaR/MILo mesh extraction and `src/eval_mesh.py` 3-axis quantitative evaluation.
- [Submodule Patches Guide](docs/submodule_patches.md): Patch maintenance, `apply_patches.sh`, and `verify_patches.sh` mechanisms.
