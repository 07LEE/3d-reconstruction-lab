#!/bin/bash

# 3DRC Pipeline Configuration File

# Dynamic Scene Name Resolution
# Priority: 1) $SCENE_NAME environment variable, 2) First positional arg if directory, 3) Default scene "20260429_140922"
if [ -n "${1:-}" ] && [ -d "$1" ]; then
    SCENE_NAME=$(basename "$1")
else
    SCENE_NAME="${SCENE_NAME:-20260429_140922}"
fi

# Dynamic Paths Derived from SCENE_NAME
DATA_DIR="data/${SCENE_NAME}"
IMAGE_DIR="${DATA_DIR}/raw_images"
HLOC_RECON="${DATA_DIR}/cache"
OUTPUT_DIR="outputs"

# Hardware Environment Flags
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SfM parameters
SFM_METHOD="sfm"  # Options: sfm, fastmap, hloc, vi_sfm
export CAMERA_MODE="SINGLE"  # Options: SINGLE, PER_FOLDER, PER_IMAGE, AUTO
IMU_DATA_PATH="${DATA_DIR}/imu_data.csv"
IMU_FORMAT="euroc"  # Options: euroc, tum, custom_csv

# 3DGS training parameters
DOWNSAMPLE_RATE=4
DATA_DEVICE="cpu"
DENSIFY_GRAD_THRESHOLD=0.0002

# SuGaR parameters
SUGAR_REGULARIZATION="dn_consistency"
SUGAR_HIGH_POLY="True"
SUGAR_REFINEMENT="short"

# Gaussian Grouping parameters
GROUPING_DATASET="${SCENE_NAME}"
GROUPING_DOWNSAMPLE=1
