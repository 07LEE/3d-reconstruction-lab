#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE_DIR=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")

VERBOSE="${VERIFY_VERBOSE:-0}"
if [ "${1:-}" = "--verbose" ] || [ "${1:-}" = "-v" ]; then
    VERBOSE=1
fi

LOG_BUFFER=""
fail=0

log_check() {
    local status="$1"
    local msg="$2"
    local line="[$status] $msg"
    LOG_BUFFER="${LOG_BUFFER}${line}\n"
    if [ "$VERBOSE" = "1" ]; then
        echo "$line"
    fi
}

if [ "$VERBOSE" = "1" ]; then
    echo ""
    echo "=== Verifying Patches & Environment (Verbose) ==="
fi

patch_count=0
patch_passed=0

# 1. Camera model patch check
READERS_PY="${REPO_ROOT}/third_party/gaussian-splatting/scene/dataset_readers.py"
patch_count=$((patch_count + 1))
if grep -q "SIMPLE_RADIAL" "${READERS_PY}"; then
  log_check "ok" "Camera models patch (SIMPLE_RADIAL) in dataset_readers.py"
  patch_passed=$((patch_passed + 1))
else
  log_check "LOST" "Camera models patch (SIMPLE_RADIAL) missing in dataset_readers.py"
  fail=1
fi

# 2. distCUDA2 scipy fallback check
MODEL_PY="${REPO_ROOT}/third_party/gaussian-splatting/scene/gaussian_model.py"
patch_count=$((patch_count + 1))
if grep -q "KDTree" "${MODEL_PY}"; then
  log_check "ok" "distCUDA2 scipy fallback in gaussian_model.py"
  patch_passed=$((patch_passed + 1))
else
  log_check "LOST" "distCUDA2 scipy fallback missing in gaussian_model.py"
  fail=1
fi

# 3. Rasterizer zero-init check
RAST_CU="${REPO_ROOT}/third_party/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu"
patch_count=$((patch_count + 1))
if grep -q "GeometryState geom = {}" "${RAST_CU}"; then
  log_check "ok" "Rasterizer zero-init patch in rasterizer_impl.cu"
  patch_passed=$((patch_passed + 1))
else
  log_check "LOST" "Rasterizer zero-init patch missing in rasterizer_impl.cu"
  fail=1
fi

# 4. Inria diff-gaussian-rasterization cstdint include check
DIFF_GAUSS_H="${REPO_ROOT}/third_party/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"
patch_count=$((patch_count + 1))
if [ -f "${DIFF_GAUSS_H}" ] && grep -q "#include <cstdint>" "${DIFF_GAUSS_H}"; then
  log_check "ok" "diff-gaussian-rasterization cstdint include patch in rasterizer_impl.h"
  patch_passed=$((patch_passed + 1))
else
  log_check "LOST" "diff-gaussian-rasterization cstdint include patch missing in rasterizer_impl.h"
  fail=1
fi

# 5. 2DGS camera model patch check (SIMPLE_RADIAL)
READERS_2D_PY="${REPO_ROOT}/third_party/2d-gaussian-splatting/scene/dataset_readers.py"
patch_count=$((patch_count + 1))
if [ -f "${READERS_2D_PY}" ] && grep -q "SIMPLE_RADIAL" "${READERS_2D_PY}"; then
  log_check "ok" "2DGS camera models patch (SIMPLE_RADIAL) in dataset_readers.py"
  patch_passed=$((patch_passed + 1))
else
  log_check "LOST" "2DGS camera models patch (SIMPLE_RADIAL) missing in 2d-gaussian-splatting dataset_readers.py"
  fail=1
fi

# 6. 2DGS mesh_utils bounds & empty cluster patch check
MESH_UTILS_PY="${REPO_ROOT}/third_party/2d-gaussian-splatting/utils/mesh_utils.py"
patch_count=$((patch_count + 1))
if [ -f "${MESH_UTILS_PY}" ] && grep -q "len(cluster_n_triangles) == 0" "${MESH_UTILS_PY}"; then
  log_check "ok" "2DGS mesh_utils empty cluster and bounds patch in mesh_utils.py"
  patch_passed=$((patch_passed + 1))
else
  log_check "LOST" "2DGS mesh_utils patch missing in mesh_utils.py"
  fail=1
fi

# 7. 2DGS diff-surfel-rasterization cstdint and zero-init check
SURFEL_RAST_CU="${REPO_ROOT}/third_party/2d-gaussian-splatting/submodules/diff-surfel-rasterization/cuda_rasterizer/rasterizer_impl.cu"
SURFEL_RAST_H="${REPO_ROOT}/third_party/2d-gaussian-splatting/submodules/diff-surfel-rasterization/cuda_rasterizer/rasterizer_impl.h"
patch_count=$((patch_count + 1))
if [ -f "${SURFEL_RAST_CU}" ] && grep -q "GeometryState geom = {}" "${SURFEL_RAST_CU}" && \
   [ -f "${SURFEL_RAST_H}" ] && grep -q "#include <cstdint>" "${SURFEL_RAST_H}"; then
  log_check "ok" "diff-surfel-rasterization cstdint & zero-init patch"
  patch_passed=$((patch_passed + 1))
else
  log_check "LOST" "diff-surfel-rasterization patch missing in rasterizer_impl.cu/h"
  fail=1
fi

# 8. SuGaR CPU data_device OOM patch check
SUGAR_MODEL="${REPO_ROOT}/third_party/sugar/sugar_scene/gs_model.py"
patch_count=$((patch_count + 1))
if [ -f "${SUGAR_MODEL}" ] && grep -q 'self.data_device = "cpu"' "${SUGAR_MODEL}"; then
  log_check "ok" "SuGaR CPU data_device OOM patch in gs_model.py"
  patch_passed=$((patch_passed + 1))
else
  log_check "LOST" "SuGaR CPU data_device OOM patch missing in gs_model.py"
  fail=1
fi

# 9. Milo KDTree fallback & cstdint check
MILO_MODEL="${REPO_ROOT}/third_party/milo/milo/scene/gaussian_model.py"
MILO_RAST_H="${REPO_ROOT}/third_party/milo/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"
patch_count=$((patch_count + 1))
if [ -f "${MILO_MODEL}" ] && grep -q "KDTree" "${MILO_MODEL}" && \
   [ -f "${MILO_RAST_H}" ] && grep -q "#include <cstdint>" "${MILO_RAST_H}"; then
  log_check "ok" "Milo KDTree fallback and cstdint patch"
  patch_passed=$((patch_passed + 1))
else
  log_check "LOST" "Milo patch missing in gaussian_model.py or rasterizer_impl.h"
  fail=1
fi

# 10. VGGT PyCOLMAP 3.13 patch check
VGGT_PY="${REPO_ROOT}/third_party/vggt/vggt/dependency/np_to_pycolmap.py"
patch_count=$((patch_count + 1))
if [ -f "${VGGT_PY}" ] && grep -q 'tmp_dir = tempfile.mkdtemp' "${VGGT_PY}"; then
  log_check "ok" "VGGT PyCOLMAP 3.13 patch in np_to_pycolmap.py"
  patch_passed=$((patch_passed + 1))
else
  log_check "LOST" "VGGT PyCOLMAP 3.13 patch missing in np_to_pycolmap.py"
  fail=1
fi

# Scaffold-GS rasterizer & visible_filter compatibility patch check
SCAFFOLD_INIT="${REPO_ROOT}/third_party/scaffold-gs/gaussian_renderer/__init__.py"
patch_count=$((patch_count + 1))
if [ -f "${SCAFFOLD_INIT}" ] && grep -q "_create_raster_settings" "${SCAFFOLD_INIT}"; then
  log_check "ok" "Scaffold-GS rasterizer & visible_filter patch in gaussian_renderer/__init__.py"
  patch_passed=$((patch_passed + 1))
else
  log_check "LOST" "Scaffold-GS rasterizer patch missing in gaussian_renderer/__init__.py"
  fail=1
fi

# Scaffold-GS checkpoint capture & getattr safeguard patch check
SCAFFOLD_MODEL="${REPO_ROOT}/third_party/scaffold-gs/scene/gaussian_model.py"
patch_count=$((patch_count + 1))
if [ -f "${SCAFFOLD_MODEL}" ] && grep -q "getattr(self, 'offset_denom', None)" "${SCAFFOLD_MODEL}"; then
  log_check "ok" "Scaffold-GS checkpoint capture & getattr patch in scene/gaussian_model.py"
  patch_passed=$((patch_passed + 1))
else
  log_check "LOST" "Scaffold-GS checkpoint capture patch missing in scene/gaussian_model.py"
  fail=1
fi

# Multi-environment explicit rasterizer verification
env_count=0
env_passed=0
env_skipped=0

TRAIN_PYTHON="${CONDA_BASE_DIR}/envs/gs_train/bin/python"
if [ -x "$TRAIN_PYTHON" ]; then
    env_count=$((env_count + 1))
    diag_out=""
    if diag_out=$("$TRAIN_PYTHON" - <<'PY' 2>&1
import sys
try:
    import diff_gaussian_rasterization as d
    from diff_gaussian_rasterization import GaussianRasterizationSettings as S
except Exception as e:
    print(f"[FAIL] [gs_train] import failed: {type(e).__name__}: {e}")
    sys.exit(1)
if "antialiasing" not in getattr(S, "_fields", ()):
    print(f"[WRONG] [gs_train] non-dr_aa rasterizer installed (path: {getattr(d, '__file__', 'unknown')})")
    sys.exit(1)
PY
    ); then
        log_check "ok" "Explicit check: gs_train has dr_aa rasterizer installed"
        env_passed=$((env_passed + 1))
    else
        log_check "FAIL" "[gs_train] rasterizer verification failed: ${diag_out:-unknown error}"
        fail=1
    fi
else
    env_skipped=$((env_skipped + 1))
    log_check "skip" "Explicit check: gs_train environment not installed"
fi

SCAFFOLD_PYTHON="${CONDA_BASE_DIR}/envs/gs_scaffold/bin/python"
if [ -x "$SCAFFOLD_PYTHON" ]; then
    env_count=$((env_count + 1))
    diag_out=""
    if diag_out=$("$SCAFFOLD_PYTHON" - <<'PY' 2>&1
import sys
try:
    import diff_gaussian_rasterization as d
except Exception as e:
    print(f"[FAIL] [gs_scaffold] import failed: {type(e).__name__}: {e}")
    sys.exit(1)
PY
    ); then
        log_check "ok" "Explicit check: gs_scaffold has dedicated rasterizer installed"
        env_passed=$((env_passed + 1))
    else
        log_check "FAIL" "[gs_scaffold] rasterizer verification failed: ${diag_out:-unknown error}"
        fail=1
    fi
else
    env_skipped=$((env_skipped + 1))
    log_check "skip" "Explicit check: gs_scaffold environment not installed"
fi

MIPSPLATTING_PYTHON="${CONDA_BASE_DIR}/envs/gs_mipsplatting/bin/python"
if [ -x "$MIPSPLATTING_PYTHON" ]; then
    env_count=$((env_count + 1))
    diag_out=""
    if diag_out=$("$MIPSPLATTING_PYTHON" - <<'PY' 2>&1
import sys
try:
    import diff_gaussian_rasterization as d
except Exception as e:
    print(f"[FAIL] [gs_mipsplatting] import failed: {type(e).__name__}: {e}")
    sys.exit(1)
PY
    ); then
        log_check "ok" "Explicit check: gs_mipsplatting has dedicated rasterizer installed"
        env_passed=$((env_passed + 1))
    else
        log_check "FAIL" "[gs_mipsplatting] rasterizer verification failed: ${diag_out:-unknown error}"
        fail=1
    fi
else
    env_skipped=$((env_skipped + 1))
    log_check "skip" "Explicit check: gs_mipsplatting environment not installed"
fi

# Blackwell sm_120 CUBIN SASS binary verification across C++ CUDA extensions
export REPO_ROOT
export CONDA_BASE_DIR
PYTHON_AUDIT_BIN=""
for e in 3drc gs_train gs_sugar gs_group gs_milo gs_scaffold gs_mipsplatting; do
    if [ -x "${CONDA_BASE_DIR}/envs/${e}/bin/python" ]; then
        PYTHON_AUDIT_BIN="${CONDA_BASE_DIR}/envs/${e}/bin/python"
        break
    fi
done
if [ -z "$PYTHON_AUDIT_BIN" ]; then PYTHON_AUDIT_BIN="python3"; fi

cubin_output=""
if cubin_output=$("$PYTHON_AUDIT_BIN" - <<'PY' 2>&1
import os, sys, glob, subprocess, re

CONDA_BASE = os.environ.get("CONDA_BASE_DIR", os.path.expanduser("~/miniconda3"))
REPO_ROOT = os.environ.get("REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

CUOBJDUMP = None
for env_name in ["gs_train", "3drc", "gs_sugar", "gs_group", "gs_milo", "gs_scaffold", "gs_mipsplatting"]:
    cand = os.path.join(CONDA_BASE, "envs", env_name, "bin/cuobjdump")
    if os.path.exists(cand):
        CUOBJDUMP = cand
        break
if not CUOBJDUMP:
    CUOBJDUMP = "cuobjdump"

ENVS = {
    "gs_milo": (["diff_gaussian_rasterization", "diff_gaussian_rasterization_ms", "diff_gaussian_rasterization_gof", "simple_knn", "fused_ssim"], "milo"),
    "gs_train": (["diff_gaussian_rasterization", "simple_knn", "diff_surfel_rasterization"], "gaussian-splatting"),
    "gs_sugar": (["diff_gaussian_rasterization", "simple_knn"], "sugar"),
    "gs_scaffold": (["diff_gaussian_rasterization", "simple_knn"], "scaffold-gs"),
    "gs_mipsplatting": (["diff_gaussian_rasterization", "simple_knn"], "mip-splatting")
}

fail = 0
total_checked = 0
total_skipped = 0

for env, (mods, repo_sub) in ENVS.items():
    env_dir = os.path.join(CONDA_BASE, "envs", env)
    if not os.path.exists(env_dir):
        print(f"  [skip]  [{env}] environment not installed")
        total_skipped += 1
        continue
    
    site_pkg = os.path.join(env_dir, "lib/python3.10/site-packages")
    search_dirs = [site_pkg, os.path.join(REPO_ROOT, "third_party", repo_sub), os.path.join(REPO_ROOT, "third_party/2d-gaussian-splatting")]
    
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

print(f"SUMMARY:{total_checked}:{fail}:{total_skipped}")
if fail != 0:
    sys.exit(1)
PY
); then
    cubin_ok=1
else
    cubin_ok=0
    fail=1
fi

cubin_total=0
cubin_skipped=0
if echo "$cubin_output" | grep -q "SUMMARY:"; then
    summary_line=$(echo "$cubin_output" | grep "SUMMARY:" | tail -n 1)
    cubin_total=$(echo "$summary_line" | cut -d':' -f2)
    cubin_skipped=$(echo "$summary_line" | cut -d':' -f4)
fi

if [ "$VERBOSE" = "1" ]; then
    echo "=== Verifying sm_120 CUBIN Binaries ==="
    echo "$cubin_output" | grep -v "SUMMARY:"
fi

# Print clean formatted summary box if non-verbose and all checks passed
if [ "$fail" -eq 0 ]; then
    if [ "$VERBOSE" = "0" ]; then
        echo "[VERIFY] Environment & Patch Verification Summary"
        printf "  • Source Code Patches (%2d/%2d)                : [ OK ]\n" "$patch_passed" "$patch_count"
        if [ "$env_skipped" -gt 0 ]; then
            printf "  • Conda Rasterizer Installs (%2d/%2d, %d skipped): [ OK ]\n" "$env_passed" "$env_count" "$env_skipped"
        else
            printf "  • Conda Rasterizer Installs (%2d/%2d)          : [ OK ]\n" "$env_passed" "$env_count"
        fi
        if [ "$cubin_skipped" -gt 0 ]; then
            printf "  • CUDA Extensions sm_120 CUBIN (%2d checked, %d skipped): [ OK ]\n" "$cubin_total" "$cubin_skipped"
        else
            printf "  • CUDA Extensions sm_120 CUBIN (%2d/%2d)        : [ OK ]\n" "$cubin_total" "$cubin_total"
        fi
    fi
else
    echo ""
    echo "[FATAL] 3DRC Environment verification failed! Detailed logs:"
    echo -e "$LOG_BUFFER"
    if [ "$VERBOSE" = "0" ]; then
        echo "$cubin_output" | grep -v "SUMMARY:"
    fi
    echo "[verify] NOT safe to train."
    echo ""
    exit 1
fi

exit 0
