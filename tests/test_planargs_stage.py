import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def build_workspace(root: Path) -> Path:
    for name in ("scripts/utils", "configs", "third_party/PlanarGS", "ext/inria", "bin", "conda/etc/profile.d"):
        (root / name).mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / "scripts" / "02b_train_planargs.sh", root / "scripts")
    for name in ("activate_backend.sh", "common_logger.sh"):
        shutil.copy(PROJECT_ROOT / "scripts" / "utils" / name, root / "scripts" / "utils")
    shutil.copy(PROJECT_ROOT / "configs" / "default_config.sh", root / "configs")
    # Patch verification and the CUDA build environment need a real toolchain, so they are skipped
    for name in ("verify_patches.sh", "setup_build_env.sh"):
        (root / "scripts" / "utils" / name).write_text(":\n")
        (root / "scripts" / "utils" / name).chmod(0o755)

    # A conda stub that activates any environment, and a trainer stub that records its arguments
    (root / "conda" / "etc" / "profile.d" / "conda.sh").write_text(
        'conda() {\n'
        f'  export CONDA_DEFAULT_ENV="$2" CONDA_PREFIX="{root}/conda/envs/$2"\n'
        '}\n'
    )
    (root / "bin" / "conda").write_text(f'#!/bin/bash\n[ "$1" = "info" ] && echo "{root}/conda"\n')
    (root / "bin" / "python").write_text(f'#!/bin/bash\necho "$*" >> "{root}/train_args.log"\n')
    for name in ("conda", "python"):
        (root / "bin" / name).chmod(0o755)

    (root / "data" / "room" / "raw_images").mkdir(parents=True)
    return root

def add_sfm_model(root: Path) -> None:
    sparse = root / "data" / "room" / "sparse" / "0"
    sparse.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse / name).touch()

def run_stage(root: Path, **env_overrides: str) -> subprocess.CompletedProcess:
    env = {"PATH": f"{root}/bin:/usr/bin:/bin", "HOME": str(root), "GS_EXT_ROOT": str(root / "ext"), **env_overrides}
    return subprocess.run(["bash", "scripts/02b_train_planargs.sh", "data/room"], cwd=root, env=env,
                          capture_output=True, text=True)

@pytest.fixture
def workspace(tmp_path):
    return build_workspace(tmp_path)

def test_missing_sfm_model_stops_with_a_clear_message(workspace):
    result = run_stage(workspace)
    assert result.returncode != 0
    assert "SfM model not found" in result.stdout + result.stderr

@pytest.mark.parametrize("env", [
    {},
    {"GS_CHECKPOINT_ITERATIONS": "5000"},
    {"GS_CHECKPOINT_INTERVAL": "0"},
])
def test_final_iteration_is_always_saved(workspace, env):
    add_sfm_model(workspace)
    result = run_stage(workspace, **env)
    assert result.returncode == 0, result.stderr
    assert "--save_iterations 30000" in (workspace / "train_args.log").read_text()
