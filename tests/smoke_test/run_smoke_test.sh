#!/usr/bin/env bash
# ==============================================================================
# 3DRC Fast End-to-End Pipeline Smoke Test Script
# ==============================================================================
# Location: tests/smoke_test/run_smoke_test.sh
# Runs SfM and 7 training/mesh backends with reduced iterations (GS_ITERATIONS=3000)
# on a subsampled dataset in data/test/smoke_test_100.
# Features strict error reporting, automated artifact assertion, and TSDF mesh verification.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORKSPACE_DIR"

TARGET_DATASET="${1:-data/test/smoke_test_100}"
SCENE_NAME="$(basename "$TARGET_DATASET")"
export GS_ITERATIONS="${GS_ITERATIONS:-3000}"
export TSDF_VOXEL_SIZE="${TSDF_VOXEL_SIZE:-0.02}"
export TSDF_DEPTH_TRUNC="${TSDF_DEPTH_TRUNC:-3.0}"

FAILED=()

run_stage() {
    local label="$1"; shift
    local artifact_pattern="$1"; shift

    # Incremental optimization: Skip stage if valid artifact already exists
    if [ "${CLEAN:-0}" -ne 1 ] && [ -n "$artifact_pattern" ] && compgen -G "$artifact_pattern" > /dev/null; then
        local existing_file
        existing_file=$(compgen -G "$artifact_pattern" | head -n 1)
        if [ -s "$existing_file" ]; then
            echo "=========================================================================="
            echo " ==> [STAGE] ${label}"
            echo "=========================================================================="
            echo "[SKIP] Stage '${label}' already completed (Artifact verified: $(basename "$existing_file"))"
            return 0
        fi
    fi

    echo "=========================================================================="
    echo " ==> [STAGE] ${label}"
    echo "=========================================================================="
    
    local rc=0
    "$@" || rc=$?

    if [ "$rc" -eq 0 ]; then
        if [ -n "$artifact_pattern" ]; then
            # Verify that expected output artifact exists and size > 0
            if compgen -G "$artifact_pattern" > /dev/null; then
                local first_file
                first_file=$(compgen -G "$artifact_pattern" | head -n 1)
                if [ -s "$first_file" ]; then
                    echo "[PASS] Stage '${label}' completed and artifact verified: $(basename "$first_file")"
                    return 0
                fi
            fi
            echo "[FAIL] Stage '${label}' completed but output artifact missing/empty: ${artifact_pattern}" >&2
            FAILED+=("${label} (Artifact Missing: ${artifact_pattern})")
        else
            echo "[PASS] Stage '${label}' completed successfully"
        fi
    else
        echo "[FAIL] Stage '${label}' failed with exit code ${rc}" >&2
        FAILED+=("${label} (Exit Code: ${rc})")
    fi
    return 0
}

echo "=========================================================================="
echo " Starting 3DRC Smoke Test Pipeline"
echo " Target Dataset : ${TARGET_DATASET}"
echo " GS Iterations  : ${GS_ITERATIONS}"
echo "=========================================================================="

# 0. Ensure subsampled dataset exists
if [ ! -d "${TARGET_DATASET}/raw_images" ]; then
    echo "==> Subsampling 100-frame dataset at ${TARGET_DATASET}..."
    python3 tests/smoke_test/subsample_dataset.py --src data/20260429_140922 --dst "${TARGET_DATASET}" --step 10
fi

# Clean outputs ONLY if CLEAN=1 is explicitly requested
if [ "${CLEAN:-0}" -eq 1 ]; then
    echo "==> Cleaning previous SfM cache and outputs for fresh smoke test..."
    rm -rf "${TARGET_DATASET}/sparse" "${TARGET_DATASET}/cache" "outputs/${SCENE_NAME}"
fi

# 1. Step 1: SfM Camera Pose Estimation (hloc)
run_stage "Step 1: SfM Pose Estimation (hloc)" \
    "${TARGET_DATASET}/sparse/0/cameras.*" \
    ./scripts/run_3drc.sh sfm hloc "${TARGET_DATASET}"

# 2. Step 2 & 3: Mesh Backends (Longer execution path priority)

# 2a. 2DGS Training & TSDF Mesh Extraction
run_stage "Step 2c: 2DGS Training" \
    "outputs/${SCENE_NAME}/2dgs/point_cloud/iteration_${GS_ITERATIONS}/point_cloud.ply" \
    ./scripts/run_3drc.sh train 2dgs "${TARGET_DATASET}"

run_stage "Step 2c Mesh: 2DGS TSDF Mesh Extraction" \
    "outputs/${SCENE_NAME}/mesh/2dgs/tsdf_mesh.ply" \
    ./scripts/run_3drc.sh tsdf "${TARGET_DATASET}" 2dgs

# 2b. PlanarGS Training & TSDF Mesh Extraction
run_stage "Step 2b: PlanarGS Training" \
    "outputs/${SCENE_NAME}/3dgs/planargs/point_cloud/iteration_${GS_ITERATIONS}/point_cloud.ply" \
    ./scripts/run_3drc.sh train planargs "${TARGET_DATASET}"

run_stage "Step 2b Mesh: PlanarGS TSDF Mesh Extraction" \
    "outputs/${SCENE_NAME}/mesh/planargs/tsdf_mesh.ply" \
    ./scripts/run_3drc.sh tsdf "${TARGET_DATASET}" planargs

# 2c. Inria 3DGS Training & SuGaR Mesh Reconstruction
run_stage "Step 2: Inria 3DGS Training" \
    "outputs/${SCENE_NAME}/3dgs/inria_30k/point_cloud/iteration_${GS_ITERATIONS}/point_cloud.ply" \
    ./scripts/run_3drc.sh train 3dgs "${TARGET_DATASET}"

run_stage "Step 3: SuGaR Mesh Reconstruction" \
    "outputs/${SCENE_NAME}/mesh/sugar_*/*.ply" \
    ./scripts/run_3drc.sh sugar "${TARGET_DATASET}"

# 3b. Scaffold-GS Training
run_stage "Step 2d: Scaffold-GS Training" \
    "outputs/${SCENE_NAME}/scaffoldgs/point_cloud/iteration_${GS_ITERATIONS}/point_cloud.ply" \
    ./scripts/run_3drc.sh train scaffoldgs "${TARGET_DATASET}"

# 3c. Mip-Splatting Training
run_stage "Step 2e: Mip-Splatting Training" \
    "outputs/${SCENE_NAME}/mipsplatting/point_cloud/iteration_${GS_ITERATIONS}/point_cloud.ply" \
    ./scripts/run_3drc.sh train mipsplatting "${TARGET_DATASET}"

# 4. Summary & Verification Output
echo "=========================================================================="
echo " Smoke Test Pipeline Artifact Summary:"
echo "=========================================================================="
./scripts/run_3drc.sh outputs

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "=========================================================================="
    echo "[FATAL] Smoke Test Failed! ${#FAILED[@]} stage(s) failed:"
    for f in "${FAILED[@]}"; do
        echo "  - ${f}"
    done
    echo "=========================================================================="
    exit 1
else
    echo "=========================================================================="
    echo "[SUCCESS] All Smoke Test stages completed and verified successfully!"
    echo "=========================================================================="
    exit 0
fi
