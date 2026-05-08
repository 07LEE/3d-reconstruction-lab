# 3D Reconstruction Test Workspace (3DRC)

A research workspace dedicated to testing 3D reconstruction algorithms and data processing workflows.

## Directory Structure

- `src/`: Source codes and scripts related to 3D reconstruction.
- `data/`: Input images/videos and reconstruction outputs (Point Cloud, Mesh, etc.).
- `configs/`: Configuration files and hyperparameters for algorithm execution.

## Core Objectives

- Validating and optimizing the 3D reconstruction pipeline performance.
- Analyzing the effectiveness of various frame sampling strategies (e.g., Smart Sampling).
- Evaluating the precision and accuracy of reconstruction results.

## Environment Setup

### Execution (Using run.sh)

It is recommended to use `run.sh`, which handles environment activation and script execution in a single step.

```bash
./run.sh --image_dir [image_path] --output_dir [output_path]
```

### Manual Environment Activation

This project uses `conda` for environment management.

```bash
conda activate 3drc
```

### Required Libraries

- `pycolmap`: Python interface for COLMAP algorithms.
- `opencv-python`: Image preprocessing and analysis.
- `numpy`, `scipy`, `matplotlib`: Data processing and visualization.

## SfM (Structure from Motion) Overview

SfM is a technique that analyzes multiple 2D images to reconstruct 3D structures and camera poses.

1. **Feature Extraction & Matching**: Identifying and connecting common points across images.
2. **Sparse Reconstruction**: Calculating 3D coordinates of features and camera paths.
3. **Dense Reconstruction**: Generating more detailed 3D models (Mesh, Depth Maps).
