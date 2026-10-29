#!/bin/bash
set -euo pipefail

# [Step 00b] YUV In-Memory Data Pipeline
# Loads a raw YUV_420_888 capture session (yuv_sensor package) into an in-memory
# PyTorch dataset for zero-disk-write ingestion by the SfM / 3DGS pipelines.

CONFIG_PATH="configs/default_config.sh"
if [ -f "$CONFIG_PATH" ]; then
    source "$CONFIG_PATH"
else
    echo "[FATAL] Configuration file not found at $CONFIG_PATH!"
    exit 1
fi

CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    set +u
    conda activate 3drc
    set -u
    [ "${CONDA_DEFAULT_ENV:-}" = "3drc" ] || { echo "[FATAL] Conda environment '3drc' not found or activation failed. Run: ./scripts/00_setup_environment.sh --env 3drc"; exit 1; }
fi

INPUT_DATASET="${1:-$DATA_DIR}"

log_header "Step 0b: YUV In-Memory Data Load (${INPUT_DATASET})"
python -m src.data.yuv_pipeline "$INPUT_DATASET"
