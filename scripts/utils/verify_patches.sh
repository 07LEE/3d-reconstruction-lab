#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=== Verifying Patches & Environment ==="

fail=0

# 1. Camera model patch check
READERS_PY="${REPO_ROOT}/third_party/gaussian-splatting/scene/dataset_readers.py"
if grep -q "SIMPLE_RADIAL" "${READERS_PY}"; then
  echo "[ok] Camera models patch (SIMPLE_RADIAL) in dataset_readers.py"
else
  echo "[LOST] Camera models patch (SIMPLE_RADIAL) missing in dataset_readers.py"
  fail=1
fi

# 2. distCUDA2 scipy fallback check
MODEL_PY="${REPO_ROOT}/third_party/gaussian-splatting/scene/gaussian_model.py"
if grep -q "KDTree" "${MODEL_PY}"; then
  echo "[ok] distCUDA2 scipy fallback in gaussian_model.py"
else
  echo "[LOST] distCUDA2 scipy fallback missing in gaussian_model.py"
  fail=1
fi

# 3. Rasterizer zero-init check
RAST_CU="${REPO_ROOT}/third_party/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu"
if grep -q "GeometryState geom = {}" "${RAST_CU}"; then
  echo "[ok] Rasterizer zero-init patch in rasterizer_impl.cu"
else
  echo "[LOST] Rasterizer zero-init patch missing in rasterizer_impl.cu"
  fail=1
fi

# 4. SuGaR CPU data_device OOM patch check
SUGAR_MODEL="${REPO_ROOT}/third_party/sugar/sugar_scene/gs_model.py"
if [ -f "${SUGAR_MODEL}" ] && grep -q 'self.data_device = "cpu"' "${SUGAR_MODEL}"; then
  echo "[ok] SuGaR CPU data_device OOM patch in gs_model.py"
else
  echo "[LOST] SuGaR CPU data_device OOM patch missing in gs_model.py"
  fail=1
fi

# 5. Imported rasterizer check in active python env via NamedTuple _fields inspection
if python - <<'PY'
import sys
try:
    import diff_gaussian_rasterization as d
    from diff_gaussian_rasterization import GaussianRasterizationSettings as S
except Exception as e:
    print("[FAIL] import failed: %s: %s" % (type(e).__name__, e))
    sys.exit(1)

if "antialiasing" not in getattr(S, "_fields", ()):
    print("[WRONG] non-dr_aa rasterizer installed:", getattr(d, "__file__", "unknown"))
    sys.exit(1)

print("[ok] Active python env has dr_aa rasterizer installed:", getattr(d, "__file__", "unknown"))
PY
then :; else fail=1; fi

if [ "$fail" -ne 0 ]; then
    echo "[verify] NOT safe to train."
    exit 1
fi

echo "=== Verification Completed ==="
exit 0
