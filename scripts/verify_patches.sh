#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Verifying Patches & Environment ==="

# 1. Camera model patch check
READERS_PY="${REPO_ROOT}/third_party/gaussian-splatting/scene/dataset_readers.py"
if grep -q "SIMPLE_RADIAL" "${READERS_PY}"; then
  echo "[ok] Camera models patch (SIMPLE_RADIAL) in dataset_readers.py"
else
  echo "[LOST] Camera models patch (SIMPLE_RADIAL) missing in dataset_readers.py"
fi

# 2. distCUDA2 scipy fallback check
MODEL_PY="${REPO_ROOT}/third_party/gaussian-splatting/scene/gaussian_model.py"
if grep -q "KDTree" "${MODEL_PY}"; then
  echo "[ok] distCUDA2 scipy fallback in gaussian_model.py"
else
  echo "[LOST] distCUDA2 scipy fallback missing in gaussian_model.py"
fi

# 3. Rasterizer zero-init check
RAST_CU="${REPO_ROOT}/third_party/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu"
if grep -q "GeometryState geom = {}" "${RAST_CU}"; then
  echo "[ok] Rasterizer zero-init patch in rasterizer_impl.cu"
else
  echo "[LOST] Rasterizer zero-init patch missing in rasterizer_impl.cu"
fi

# 4. Imported rasterizer check in current python env
python -c "
import os, sys
try:
    import diff_gaussian_rasterization as d
    sp_path = os.path.dirname(d.__file__)
    init_py = os.path.join(sp_path, '__init__.py')
    with open(init_py, 'r') as f:
        content = f.read()
    if 'antialiasing' in content or 'invdepths' in content or 'dr_aa' in sp_path or 'third_party' in sp_path:
        print('[ok] Active python env has dr_aa rasterizer installed:', sp_path)
    else:
        print('[WARNING] Active python env has non-dr_aa rasterizer installed:', sp_path)
except Exception as e:
    print('[FAIL] Failed to import diff_gaussian_rasterization:', e)
"

echo "=== Verification Completed ==="
