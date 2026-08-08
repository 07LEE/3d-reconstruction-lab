#!/bin/bash

# 3DRC Pipeline Configuration File

# Paths
DATA_DIR="data/nerfstudio_data"
IMAGE_DIR="data/images"
OUTPUT_DIR="outputs"
HLOC_RECON="data/hloc_reconstruction"

# Hardware & Build Environment Setup
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${CONFIG_DIR}/../scripts/utils/setup_build_env.sh" ]; then
    source "${CONFIG_DIR}/../scripts/utils/setup_build_env.sh"
else
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SfM parameters
SFM_METHOD="hloc"  # Options: sfm, fastmap, hloc, vi_sfm
export CAMERA_MODE="SINGLE"  # Options: SINGLE, PER_FOLDER, PER_IMAGE, AUTO
IMU_DATA_PATH="data/imu_data.csv"
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
GROUPING_DATASET="bear"  # Default test dataset
GROUPING_DOWNSAMPLE=1
