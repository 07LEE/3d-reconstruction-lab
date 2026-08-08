#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE_DIR=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")

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
PYTHON_BIN="${CONDA_BASE_DIR}/envs/gs_train/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then PYTHON_BIN="python3"; fi

if "$PYTHON_BIN" - <<'PY'
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

# 6. Blackwell sm_120 CUBIN SASS binary verification across C++ CUDA extensions
echo "=== Verifying sm_120 CUBIN Binaries ==="
CONDA_BASE_DIR=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
CUOBJDUMP_BIN="${CONDA_BASE_DIR}/envs/gs_train/bin/cuobjdump"

if [ -x "$CUOBJDUMP_BIN" ]; then
    checked_count=0
    for env_name in gs_milo gs_train gs_sugar; do
        env_dir="${CONDA_BASE_DIR}/envs/${env_name}"
        if [ -d "$env_dir" ]; then
            for so in $(find "$env_dir/lib/python3.10/site-packages/" \( -path "*/diff_gaussian_rasterization/*" -o -path "*/diff_gaussian_rasterization_ms/*" -o -path "*/diff_gaussian_rasterization_gof/*" -o -path "*/simple_knn/*" -o -path "*/fused_ssim*" -o -path "*/sugar_ext*" \) -name "*.so" 2>/dev/null || true); do
                checked_count=$((checked_count + 1))
                archs=$("$CUOBJDUMP_BIN" --list-elf "$so" 2>/dev/null | grep -o "sm_[0-9]*" | sort -u | tr '\n' ' ' || true)
                parent_dir=$(basename "$(dirname "$so")")
                if [ "$parent_dir" = "site-packages" ]; then
                    mod_label="$(basename "$so")"
                else
                    mod_label="${parent_dir}/$(basename "$so")"
                fi
                case "$archs" in
                    *sm_120*)
                        echo "  [ok]    [${env_name}] ${mod_label}: ${archs}"
                        ;;
                    "")
                        echo "  [WARN]  [${env_name}] ${mod_label}: no CUBIN (PTX JIT fallback)"
                        ;;
                    *)
                        echo "  [WRONG] [${env_name}] ${mod_label}: ${archs} (native sm_120 missing)"
                        fail=1
                        ;;
                esac
            done
        fi
    done
    echo "[INFO] Total C++ CUDA extension modules audited: ${checked_count}"
fi

if [ "$fail" -ne 0 ]; then
    echo "[verify] NOT safe to train."
    exit 1
fi

echo "=== Verification Completed ==="
exit 0
