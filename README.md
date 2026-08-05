# 3D Reconstruction Test Workspace

A research workspace dedicated to testing 3D reconstruction algorithms, from state-of-the-art SfM to 3D Gaussian Splatting (3DGS).

## Environment Architecture

This project utilizes virtual environments to ensure stability and compatibility:

1. `3drc` (SfM & Rendering Environment): Integrated environment for camera registration, sparse reconstruction (COLMAP, hloc, vggt) and neural rendering (nerfstudio).
2. `gs_original` (Unified 3DGS & Optimization Environment): Unified environment for all 3D Gaussian Splatting algorithms, including Inria 3DGS, Planar-GS, SuGaR mesh extraction, and Gaussian Grouping segmentation.

### Quick Setup

Run the automated setup script to synchronize submodules and install dependencies:

```bash
./scripts/00_setup_environment.sh
```

## Pipeline Model Mapping

| Tool | Pipeline Step | Core Role | Methodology |
| --- | --- | --- | --- |
| COLMAP | Step 1 (SfM) | Camera pose estimation and sparse reconstruction | SIFT feature extraction and incremental triangulation SfM |
| hloc | Step 1 (SfM) | Deep learning-based camera pose estimation | SuperPoint feature extraction and SuperGlue graph matching |
| VGGT-Omega | Step 1 (SfM) | Immediate feed-forward camera pose and point cloud generation | Feed-forward Visual Geometry Grounded Transformer model |
| vi_sfm | Step 1 (SfM) | Visual-Inertial (RGB + IMU) camera pose estimation | SuperPoint/Glue matching with IMU gravity alignment |
| 3DGS (Inria) | Step 2 (Training) | High-fidelity 3D scene representation optimization | Differentiable 3D Gaussian rasterization |
| Planar-GS | Step 2 (Training) | Planar constraint optimization for textureless flat surfaces | Planar Regularization Loss guided 3DGS training |
| SuGaR | Step 4 (Mesh Extraction) | Polygon mesh and OBJ file extraction from points | Surface-Aligned Gaussian Regularization and Poisson reconstruction |
| Gaussian Grouping | Step 5 (Segmentation) | 3D object instance grouping and segmentation | Identity Embedding learning lifted from 2D SAM masks |

## Execution Pipeline

The project provides unified CLI launchers (`./scripts/run_3drc.sh` and `python 3drc.py`) as well as individual scripts for each pipeline step.

### Quick Start with Unified CLI

```bash
# Run Step 1 SfM (Visual-Inertial or hloc)
./scripts/run_3drc.sh sfm vi_sfm
python 3drc.py sfm --method vi_sfm

# Run Step 2 3DGS Training
./scripts/run_3drc.sh train 3dgs

# Run End-to-End Automated Pipeline (Step 1 -> 2 -> 3)
./scripts/run_3drc.sh pipeline vi_sfm
```

### Step 1: Camera Pose Estimation (SfM)

High-precision image matching to determine camera positions in 3D space. Supports hloc, sfm, fastmap, vggt, and vi_sfm (RGB+IMU) methods.

```bash
# Default hloc method
./scripts/01_sfm_hloc.sh

# Or run with Visual-Inertial (RGB + IMU) method:
./scripts/01_sfm_hloc.sh vi_sfm

# Or run with alternative methods: sfm, fastmap, vggt
./scripts/01_sfm_hloc.sh vggt
```

### Step 2: High-Density 3DGS Training

Performs training with Blackwell GPU (RTX 50) optimization. Supports original 3DGS and Planar-GS models via TRAIN_METHOD variable.

```bash
# Default original 3DGS training
./scripts/02_train_3dgs.sh

# Or switch to Planar-GS in configs/default_config.sh (set TRAIN_METHOD="planar")
./scripts/02_train_3dgs.sh
```

### Step 3: Result Visualization

View the trained 3D model using the specialized viewer.

```bash
./scripts/03_view_result.sh
```

### Step 4: SuGaR Mesh Reconstruction

Extract a high-quality 3D mesh from the optimized 3DGS checkpoint using surface alignment regularization.

```bash
./scripts/04_train_sugar.sh
```

### Step 5: Gaussian Grouping Object Segmentation

Perform joint reconstruction and segmentation from SAM-based 2D masks.

```bash
./scripts/05_train_grouping.sh
```

## Advanced Features

### Visual-Inertial (RGB + IMU) Integration

For datasets containing IMU sensor measurements alongside RGB images, the `vi_sfm` pipeline provides key advantages:

- **Metric Scale Recovery**: Absolute meter-scale 3D reconstruction.
- **Gravity Alignment**: Automatic $Z$-axis alignment with the world gravity vector.
- **Motion Blur Robustness**: Stable trajectory tracking during rapid sensor movements.

```bash
# Execute Step 1 Visual-Inertial SfM Pipeline
./scripts/01_sfm_hloc.sh vi_sfm
```

### Nerfstudio Integration

Supports testing various Nerfstudio-based engines like Splatfacto.

```bash
# Example: Train using splatfacto
conda activate nerfstudio
ns-train splatfacto --data data/nerfstudio_data
```

### Original 3D Gaussian Splatting (Inria)

Highly optimized for performance and quality, specifically patched for Blackwell (sm_120) hardware.

- Environment: `gs_original`
- Location: `third_party/gaussian-splatting`
- Hardware Patch: `rasterizer_impl.h` modified to include `<cstdint>` for CUDA 12.8 compatibility.
- Execution:

  ```bash
  # Standard Training (2x/4x downsampled for memory efficiency)
  python train.py -s ../../data/nerfstudio_data -r 2 --model_path ../../outputs/gs_result
  
  # High-Resolution / Large Dataset Training (Memory Swap Mode)
  # Uses system RAM instead of VRAM for image storage
  python train.py -s ../../data/nerfstudio_data -r 1 --data_device cpu --model_path ../../outputs/gs_original_res
  ```

### Nerfstudio (Splatfacto & Nerfacto)

Comprehensive framework for neural rendering experiments.

- Environment: `nerfstudio`
- Execution:
  - `./ns_run.sh train splatfacto --data data/nerfstudio_data`

### SuGaR Mesh Extraction (CVPR 2024)

Enforces surface alignment constraints on 3D Gaussians to enable fast and clean mesh reconstruction via Poisson reconstruction.

- Environment: `gs_original`
- Location: `third_party/sugar`
- Execution:
  
  ```bash
  ./scripts/04_train_sugar.sh
  ```

### VGGT (Visual Geometry Grounded Transformer)

Feed-forward neural network for camera pose estimation and immediate point cloud generation.

- Environment: `3drc`
- Location: `third_party/vggt`
- Execution: Enables exporting depth, point maps, and cameras directly into COLMAP-compatible structures.

### Gaussian Grouping (ECCV 2024)

Joint reconstruction and segmentation of 3D objects using identity embeddings lifted from 2D SAM masks.

- Environment: `gs_original`
- Location: `third_party/gaussian-grouping`
- Execution:
  
  ```bash
  ./scripts/05_train_grouping.sh
  ```

### Planar-GS

Enforces geometry planar regularization on flat surfaces to reconstruct flat objects (walls, desks, floors) without holes or noise.

- Environment: `gs_original`
- Location: `third_party/planar-gs`
- Execution: Configure TRAIN_METHOD="planar" in configs/default_config.sh, then run ./scripts/02_train_3dgs.sh

## Directory Structure

- `src/`: Core pipeline scripts and utilities.
- `configs/`: Central configuration scripts containing pipeline hyperparameter parameters.
- `data/`: Datasets and SfM outputs.
- `data/nerfstudio_data/`: Unified data for rendering engines (contains `sparse` and `images`).
- `third_party/`: Unified directory for external tools (`hloc`, `gaussian-splatting`, `sugar`, `vggt`, `gaussian-grouping`).
- `outputs/`: 3DGS/NeRF trained models and point clouds.

## Hardware Insights (Blackwell RTX 50-series)

- CUDA Compatibility: Patched `diff-gaussian-rasterization` to fix header issues in CUDA 12.x.
- Memory Management:
  - `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is used to prevent fragmentation.
  - `--data_device cpu` is recommended for large-scale datasets on limited VRAM hardware to enable high-resolution training without OOM errors.

## Setup Instructions for Unified Environment

To optimize disk space, all Gaussian Splatting tasks are unified under the `gs_original` environment. Run the following setup steps to compile the CUDA extensions for SuGaR and Gaussian Grouping inside the `gs_original` environment:

```bash
# 1. Activate the unified environment
conda activate gs_original

# 2. Build and install SuGaR extensions
cd third_party/sugar
pip install -e .

# 3. Build and install Gaussian Grouping extensions
cd ../gaussian-grouping
pip install -e .

# 4. Build and install DEVA tracking module inside grouping
cd Tracking-Anything-with-DEVA
pip install -e .
```

## Third-Party Submodules

This project enforces a highly-structured and clean dependency architecture. All external packages are tracked as official Git submodules in the `third_party/` directory to preserve structural integrity:

- `third_party/fastmap`: Fast image feature matching framework.
- `third_party/gaussian-grouping`: Identity embedding framework for object segmentation.
- `third_party/gaussian-splatting`: Customized 3DGS training engine.
  - *Fork Integration*: Linked to the personal workspace repository [07LEE/gaussian-splatting](https://github.com/07LEE/gaussian-splatting) for custom patches and cloud backups.
  - *Workflow*: When making custom code edits in `gaussian-splatting/`, developers must commit internally and `git push` to their personal fork.
- `third_party/hloc`: Visual localization toolbox for structure-from-motion pipelines.
- `third_party/planar-gs`: Planar-constrained 3D Gaussian Splatting engine for flat surfaces.
- `third_party/sugar`: Surface-Aligned Gaussian Splatting tool for 3D mesh reconstruction.
- `third_party/vggt`: Visual Geometry Grounded Transformer model for visual geometry prediction.

### Build Pollution Defense (ignore = dirty)

To prevent CUDA/C++ build artifacts and compilations from polluting the parent workspace git status, all submodules are registered with the `ignore = dirty` attribute in `.gitmodules`. This guarantees that `git status` on the parent repository `3DRC` remains clean (`working tree clean`) even during high-intensity training and compilation.
