#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=== Applying Patches ==="

apply_patch_dir() {
  local target_dir="$1"
  local patch_dir="$2"

  if [ -d "${target_dir}" ] && [ -d "${patch_dir}" ]; then
    cd "${target_dir}"
    for patch in "${patch_dir}"/*.patch; do
      if [ -f "${patch}" ]; then
        local pbasename="$(basename "${patch}")"
        echo "Checking ${pbasename} in ${target_dir#${REPO_ROOT}/}..."
        if git apply --reverse --check "${patch}" >/dev/null 2>&1; then
          echo "  -> Already applied: ${pbasename}"
        elif git apply --check "${patch}" >/dev/null 2>&1; then
          git apply "${patch}"
          echo "  -> Successfully applied: ${pbasename}"
        else
          echo "  -> [FAIL] Patch ${pbasename} cannot be checked or applied"
          exit 1
        fi
      fi
    done
    cd "${REPO_ROOT}"
  fi
}

apply_patch_dir "${REPO_ROOT}/third_party/gaussian-splatting" "${REPO_ROOT}/patches/third_party/gaussian-splatting"
apply_patch_dir "${REPO_ROOT}/third_party/gaussian-splatting/submodules/diff-gaussian-rasterization" "${REPO_ROOT}/patches/third_party/gaussian-splatting/submodules/diff-gaussian-rasterization"
apply_patch_dir "${REPO_ROOT}/third_party/sugar" "${REPO_ROOT}/patches/third_party/sugar"
apply_patch_dir "${REPO_ROOT}/third_party/milo" "${REPO_ROOT}/patches/third_party/milo"
apply_patch_dir "${REPO_ROOT}/third_party/vggt" "${REPO_ROOT}/patches/third_party/vggt"
apply_patch_dir "${REPO_ROOT}/third_party/2d-gaussian-splatting" "${REPO_ROOT}/patches/third_party/2d-gaussian-splatting"
apply_patch_dir "${REPO_ROOT}/third_party/2d-gaussian-splatting/submodules/diff-surfel-rasterization" "${REPO_ROOT}/patches/third_party/2d-gaussian-splatting/submodules/diff-surfel-rasterization"
apply_patch_dir "${REPO_ROOT}/third_party/scaffold-gs" "${REPO_ROOT}/patches/third_party/scaffold-gs"

echo "=== Apply Patches Completed ==="
