#!/bin/bash

# 3DRC Pipeline Configuration File

# Paths
DATA_DIR="data/nerfstudio_data"
IMAGE_DIR="data/images"
OUTPUT_DIR="outputs"
HLOC_RECON="data/hloc_reconstruction"

# Hardware Setup (Blackwell Compatibility sm_120 / sm_90)
export TORCH_CUDA_ARCH_LIST="12.0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SfM parameters
SFM_METHOD="hloc"  # Options: sfm, fastmap, hloc

# 3DGS training parameters
TRAIN_METHOD="3dgs"  # Options: 3dgs, planar
DOWNSAMPLE_RATE=2  # Equivalent to -r 2
DATA_DEVICE="cpu"  # VRAM optimization: cpu or cuda
DENSIFY_GRAD_THRESHOLD=0.0002

# Planar-GS parameters
PLANAR_REG_WEIGHT=0.1

# SuGaR parameters
SUGAR_REGULARIZATION="dn_consistency"
SUGAR_HIGH_POLY="True"
SUGAR_REFINEMENT="short"

# Gaussian Grouping parameters
GROUPING_DATASET="bear"  # Default test dataset
GROUPING_DOWNSAMPLE=1
