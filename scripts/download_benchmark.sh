#!/usr/bin/env bash
# ==============================================================================
# 3DRC Benchmark Dataset Downloader
# ==============================================================================
# Usage:
#   ./scripts/download_benchmark.sh <dataset_name> [scene_name]
# Example:
#   ./scripts/download_benchmark.sh tanks_and_temples truck
#   ./scripts/download_benchmark.sh mipnerf360 garden
#   ./scripts/download_benchmark.sh deep_blending
#   ./scripts/download_benchmark.sh synthetic_nerf lego
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="${WORKSPACE_DIR}/data/benchmark"

DATASET="${1:-help}"
SCENE="${2:-all}"

show_help() {
    echo "Usage: ./scripts/download_benchmark.sh <dataset_name> [scene_name]"
    echo ""
    echo "Available Datasets:"
    echo "  mipnerf360         - Mip-NeRF 360 Dataset (garden, bicycle, bonsai, counter, kitchen, room, stump, treehill)"
    echo "  tanks_and_temples  - Tanks and Temples Dataset (truck, train)"
    echo "  deep_blending      - Deep Blending Dataset (playroom, drjohnson)"
    echo "  synthetic_nerf     - Synthetic NeRF Dataset (chair, drums, ficus, hotdog, lego, materials, mic, ship)"
    echo ""
    echo "Example:"
    echo "  ./scripts/download_benchmark.sh tanks_and_temples truck"
    exit 0
}

if [ "$DATASET" = "help" ] || [ "$DATASET" = "-h" ] || [ "$DATASET" = "--help" ]; then
    show_help
fi

mkdir -p "${BENCHMARK_DIR}"

download_and_extract() {
    local url="$1"
    local target_zip="$2"
    local extract_dir="$3"

    echo "==> Target Directory: ${extract_dir}"
    mkdir -p "${extract_dir}"

    if [ -f "${target_zip}" ]; then
        echo "==> Found existing archive: ${target_zip}. Skipping download."
    else
        echo "==> Downloading dataset archive from ${url}..."
        curl -L -C - --progress-bar -o "${target_zip}" "${url}"
    fi

    echo "==> Extracting dataset archive..."
    unzip -q -o "${target_zip}" -d "${extract_dir}"
    echo "==> Extraction complete."
}

case "${DATASET}" in
    mipnerf360)
        URL="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/360_v2.zip"
        ZIP_FILE="${BENCHMARK_DIR}/360_v2.zip"
        EXTRACT_DIR="${BENCHMARK_DIR}/mipnerf360"
        download_and_extract "${URL}" "${ZIP_FILE}" "${EXTRACT_DIR}"
        ;;
    tanks_and_temples|tandt)
        URL="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tanks_and_temples.zip"
        ZIP_FILE="${BENCHMARK_DIR}/tanks_and_temples.zip"
        EXTRACT_DIR="${BENCHMARK_DIR}/tanks_and_temples"
        download_and_extract "${URL}" "${ZIP_FILE}" "${EXTRACT_DIR}"
        ;;
    deep_blending)
        URL="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/deep_blending.zip"
        ZIP_FILE="${BENCHMARK_DIR}/deep_blending.zip"
        EXTRACT_DIR="${BENCHMARK_DIR}/deep_blending"
        download_and_extract "${URL}" "${ZIP_FILE}" "${EXTRACT_DIR}"
        ;;
    synthetic_nerf|blender)
        URL="https://storage.googleapis.com/nerfblender/projects/nerf_synthetic.zip"
        ZIP_FILE="${BENCHMARK_DIR}/nerf_synthetic.zip"
        EXTRACT_DIR="${BENCHMARK_DIR}/synthetic_nerf"
        download_and_extract "${URL}" "${ZIP_FILE}" "${EXTRACT_DIR}"
        ;;
    *)
        echo "[ERROR] Unknown dataset: ${DATASET}"
        show_help
        ;;
esac

echo "=========================================================================="
echo "[SUCCESS] Dataset '${DATASET}' setup complete in ${BENCHMARK_DIR}/${DATASET}"
echo "=========================================================================="
