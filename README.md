# 3D Reconstruction Test Workspace (3DRC)

A research workspace dedicated to testing 3D reconstruction algorithms, from state-of-the-art SfM to 3D Gaussian Splatting (3DGS).

## Environment Architecture

This project utilizes two specialized virtual environments to ensure stability and compatibility:

1. **`3drc` (SfM Environment)**: Focused on camera registration and sparse reconstruction using COLMAP and hloc.
2. **`nerfstudio` (Nerfstudio Environment)**: Advanced neural rendering framework for large scenes.
3. **`gs_original` (Inria 3DGS Environment)**: Optimized environment for the original 3D Gaussian Splatting implementation with Blackwell (RTX 50) support.

## Execution Pipeline

The project provides automated scripts for each step from data processing to visualization.

### Step 1: Camera Pose Estimation (SfM)

High-precision image matching to determine camera positions in 3D space.

```bash
./scripts/01_sfm_hloc.sh
```

### Step 2: High-Density 3DGS Training

Performs training with Blackwell GPU (RTX 50) optimization and high-density parameters.

```bash
./scripts/02_train_3dgs.sh
```

### Step 3: Result Visualization

View the trained 3D model using the specialized viewer.

```bash
./scripts/03_view_result.sh
```

## Advanced Features

### Nerfstudio Integration

Supports testing various Nerfstudio-based engines like Splatfacto.

```bash
# Example: Train using splatfacto
conda activate nerfstudio
ns-train splatfacto --data data/nerfstudio_data
```

### Original 3D Gaussian Splatting (Inria)

Highly optimized for performance and quality, specifically patched for Blackwell (sm_120) hardware.

- **Environment**: `gs_original`
- **Location**: `third_party/gaussian-splatting`
- **Hardware Patch**: `rasterizer_impl.h` modified to include `<cstdint>` for CUDA 12.8 compatibility.
- **Execution**:

  ```bash
  # Standard Training (2x/4x downsampled for memory efficiency)
  python train.py -s ../../data/nerfstudio_data -r 2 --model_path ../../outputs/gs_result

  # High-Resolution / Large Dataset Training (Memory Swap Mode)
  # Uses system RAM instead of VRAM for image storage
  python train.py -s ../../data/nerfstudio_data -r 1 --data_device cpu --model_path ../../outputs/gs_original_res
  ```

### Nerfstudio (Splatfacto & Nerfacto)

Comprehensive framework for neural rendering experiments.

- **Environment**: `nerfstudio`
- **Execution**:
  - `./ns_run.sh train splatfacto --data data/nerfstudio_data`

## Directory Structure

- `src/`: Core pipeline scripts and utilities.
- `data/`: Datasets and SfM outputs.
  - `data/nerfstudio_data/`: Unified data for rendering engines (contains `sparse` and `images`).
- `third_party/`: Unified directory for external tools (`hloc`, `gaussian-splatting`).
- `outputs/`: 3DGS/NeRF trained models and point clouds.

## Hardware Insights (Blackwell RTX 50-series)

- **CUDA Compatibility**: Patched `diff-gaussian-rasterization` to fix header issues in CUDA 12.x.
- **Memory Management**:
  - `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is used to prevent fragmentation.
  - `--data_device cpu` is recommended for large-scale datasets on limited VRAM hardware to enable high-resolution training without OOM errors.
