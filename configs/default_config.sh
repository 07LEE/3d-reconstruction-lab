#!/bin/bash

# 3DRC Pipeline Configuration File

# Paths
DATA_DIR="data/nerfstudio_data"
IMAGE_DIR="data/images"
OUTPUT_DIR="outputs"
HLOC_RECON="data/hloc_reconstruction"

# Hardware Setup (Dynamic detection with fallback guards)
if [ -z "$CUDA_HOME" ] && [ -d "/usr/lib/nvidia-cuda-toolkit" ]; then
    export CUDA_HOME="/usr/lib/nvidia-cuda-toolkit"
    export PATH="/usr/lib/nvidia-cuda-toolkit/bin:$PATH"
fi

if command -v /usr/bin/gcc-12 >/dev/null 2>&1; then
    export CC="/usr/bin/gcc-12"
    export CXX="/usr/bin/g++-12"
fi
export TORCH_CUDA_ARCH_LIST="8.9"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SfM parameters
SFM_METHOD="hloc"  # Options: sfm, fastmap, hloc, vi_sfm
IMU_DATA_PATH="data/imu_data.csv"
IMU_FORMAT="euroc"  # Options: euroc, tum, custom_csv


# 3DGS training parameters
TRAIN_METHOD="3dgs"  # Options: 3dgs, planar
DOWNSAMPLE_RATE=1  # Full original resolution (-r 1)
DATA_DEVICE="cpu"  # Keep images in System RAM to prevent VRAM OOM for 1,000+ frames
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
