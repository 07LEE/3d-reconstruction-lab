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
export REPO_ROOT
export CONDA_BASE_DIR
PYTHON_AUDIT_BIN="${CONDA_BASE_DIR}/envs/gs_milo/bin/python"
if [ ! -x "$PYTHON_AUDIT_BIN" ]; then PYTHON_AUDIT_BIN="python3"; fi

if "$PYTHON_AUDIT_BIN" - <<'PY'
import os, sys, glob, subprocess, re

CONDA_BASE = os.environ.get("CONDA_BASE_DIR", os.path.expanduser("~/miniconda3"))
REPO_ROOT = os.environ.get("REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
CUOBJDUMP = os.path.join(CONDA_BASE, "envs/gs_train/bin/cuobjdump")

ENVS = {
    "gs_milo": (["diff_gaussian_rasterization", "diff_gaussian_rasterization_ms", "diff_gaussian_rasterization_gof", "simple_knn", "fused_ssim"], "milo"),
    "gs_train": (["diff_gaussian_rasterization", "simple_knn"], "gaussian-splatting"),
    "gs_sugar": (["diff_gaussian_rasterization", "simple_knn"], "sugar")
}

fail = 0
total_checked = 0

for env, (mods, repo_sub) in ENVS.items():
    env_dir = os.path.join(CONDA_BASE, "envs", env)
    if not os.path.exists(env_dir):
        continue
    
    site_pkg = os.path.join(env_dir, "lib/python3.10/site-packages")
    search_dirs = [site_pkg, os.path.join(REPO_ROOT, "third_party", repo_sub)]
    
    for mod in mods:
        so_files = []
        mod_pat = mod.replace("_", "*")
        for sd in search_dirs:
            found = glob.glob(os.path.join(sd, f"**/{mod}/_C*.so"), recursive=True) or \
                    glob.glob(os.path.join(sd, f"**/{mod_pat}/_C*.so"), recursive=True) or \
                    glob.glob(os.path.join(sd, f"**/{mod}*.so"), recursive=True) or \
                    glob.glob(os.path.join(sd, f"**/{mod_pat}*.so"), recursive=True)
            for f in found:
                if "/build/" not in f and not f.endswith(".py"):
                    so_files.append(f)
        
        valid_sos = [f for f in so_files if site_pkg in f or repo_sub in f]
        if valid_sos:
            total_checked += 1
            so = valid_sos[0]
            try:
                out = subprocess.check_output([CUOBJDUMP, "--list-elf", so], stderr=subprocess.DEVNULL).decode()
                archs = sorted(list(set(re.findall(r'sm_\d+', out))))
                sm_str = " ".join(archs) if archs else "no CUBIN"
                if "sm_120" in archs:
                    print(f"  [ok]    [{env}] {mod} -> {sm_str}")
                else:
                    print(f"  [WRONG] [{env}] {mod} -> {sm_str} (native sm_120 missing)")
                    fail = 1
            except Exception as e:
                print(f"  [WARN]  [{env}] {mod}: cuobjdump failed ({e})")
        else:
            print(f"  [MISS]  [{env}] {mod}: .so file not found")
            fail = 1

print(f"[INFO] Total C++ CUDA extension modules audited: {total_checked}")
if fail != 0:
    sys.exit(1)
PY
then :; else fail=1; fi

if [ "$fail" -ne 0 ]; then
    echo "[verify] NOT safe to train."
    exit 1
fi

echo "=== Verification Completed ==="
exit 0
