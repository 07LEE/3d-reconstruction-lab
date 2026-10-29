import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Mesh stage scripts are replaced by stubs that log the arguments they receive
STAGE_SCRIPTS = {"sugar": "03_train_sugar.sh", "tsdf": "03c_mesh_tsdf.sh"}

@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "scripts" / "utils").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    shutil.copy(PROJECT_ROOT / "scripts" / "run_3drc.sh", tmp_path / "scripts")
    shutil.copy(PROJECT_ROOT / "scripts" / "utils" / "common_logger.sh", tmp_path / "scripts" / "utils")
    shutil.copy(PROJECT_ROOT / "configs" / "default_config.sh", tmp_path / "configs")
    for name in STAGE_SCRIPTS.values():
        stub = tmp_path / "scripts" / name
        stub.write_text(f'#!/bin/bash\necho "{name} $*" >> calls.log\n')
        stub.chmod(0o755)
    (tmp_path / "data" / "room" / "raw_images").mkdir(parents=True)
    return tmp_path

def run_step(root: Path, *args: str, **env_overrides: str):
    # No conda on PATH and an empty HOME, so run_3drc.sh falls back to python3
    env = {"PATH": "/usr/bin:/bin", "HOME": str(root), **env_overrides}
    result = subprocess.run(["bash", "scripts/run_3drc.sh", *args], cwd=root, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return (root / "calls.log").read_text().splitlines()

@pytest.mark.parametrize("command", STAGE_SCRIPTS)
@pytest.mark.parametrize("argv, expected", [
    ([], "data/room"),
    (["data/room"], "data/room"),
    (["3dgs/planargs"], "data/room 3dgs/planargs"),
    (["data/room", "3dgs/planargs"], "data/room 3dgs/planargs"),
    (["3dgs/planargs", "data/room"], "data/room 3dgs/planargs"),
])
def test_mesh_step_receives_dataset_then_source_model(workspace, command, argv, expected):
    assert run_step(workspace, command, *argv) == [f"{STAGE_SCRIPTS[command]} {expected}"]

def test_sugar_iterations_do_not_shift_positional_arguments(workspace):
    assert run_step(workspace, "sugar", "data/room", GS_ITERATIONS="3000") == ["03_train_sugar.sh data/room"]
