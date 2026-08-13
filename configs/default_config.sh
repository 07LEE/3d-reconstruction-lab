#!/bin/bash

# 3DRC Pipeline Configuration File

# Source Common Logger Library if available
SCRIPT_DIR_CFG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_CFG="$(cd "$SCRIPT_DIR_CFG/.." && pwd)"
if [ -f "$PROJECT_ROOT_CFG/scripts/utils/common_logger.sh" ]; then
    source "$PROJECT_ROOT_CFG/scripts/utils/common_logger.sh"
fi

# Dynamic Scene Name Resolution
# Priority: 1) First positional arg if directory, 2) $SCENE_NAME environment variable, 3) Default scene "20260429_140922" (with informational notice)
if [ -n "${1:-}" ] && [ -d "$1" ]; then
    SCENE_NAME=$(basename "$1")
elif [ -n "${SCENE_NAME:-}" ]; then
    SCENE_NAME="${SCENE_NAME}"
else
    SCENE_NAME="20260429_140922"
    if [ "${QUIET_CONFIG:-false}" != "true" ] && [ "${_3DRC_SCENE_NOTICE_SHOWN:-0}" = "0" ]; then
        export _3DRC_SCENE_NOTICE_SHOWN=1
        if type log_info >/dev/null 2>&1; then
            log_info "No dataset path or SCENE_NAME provided. Defaulting to '${SCENE_NAME}'."
        else
            echo "[INFO]  No dataset path or SCENE_NAME provided. Defaulting to '${SCENE_NAME}'."
        fi
    fi
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
SFM_METHOD="hloc"  # Options: hloc, sfm, fastmap, vggt
SFM_STRATEGY="sequential"  # Options: sequential, exhaustive, sequential+retrieval
SFM_OVERLAP=100
SFM_RETRIEVAL_K=30
export CAMERA_MODE="SINGLE"  # Options: SINGLE, PER_FOLDER, PER_IMAGE, AUTO
export CAMERA_MODEL="SIMPLE_RADIAL"  # Options: SIMPLE_RADIAL, PINHOLE, OPENCV

# 3DGS training parameters
DOWNSAMPLE_RATE=4
DATA_DEVICE="cpu"
DENSIFY_GRAD_THRESHOLD=0.0002
GS_ITERATIONS=30000
export GS_CHECKPOINT_INTERVAL=10000  # Automatically generates checkpoints every N iterations up to GS_ITERATIONS
GS_EVAL_MODE="false"  # Options: false (full reconstruction), true (holdout evaluation)

# SuGaR parameters
SUGAR_REGULARIZATION="dn_consistency"
SUGAR_HIGH_POLY="True"
SUGAR_REFINEMENT="short"

# TSDF Mesh Extraction parameters
TSDF_VOXEL_SIZE=0.005
TSDF_DEPTH_TRUNC=6.0

# MILo parameters
MILO_IMP_METRIC="indoor"  # Options: indoor, outdoor
MILO_RASTERIZER="radegs"  # Options: radegs, 2dgs, 3dgs

# Gaussian Grouping parameters
GROUPING_DATASET="${SCENE_NAME}"
GROUPING_DOWNSAMPLE=1
