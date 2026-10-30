import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Training backend -> stage script, the header run_3drc.sh prints, and the model directory under outputs/<scene>/
BACKENDS = {
    "3dgs": ("02_train_3dgs.sh", "Step 2: 3DGS Training", "3dgs/inria_30k"),
    "planargs": ("02b_train_planargs.sh", "Step 2b: PlanarGS Training", "3dgs/planargs"),
    "2dgs": ("02c_train_2dgs.sh", "Step 2c: 2D Gaussian Splatting Training", "2dgs"),
    "scaffoldgs": ("02d_train_scaffoldgs.sh", "Step 2d: Scaffold-GS Training", "scaffoldgs"),
    "mipsplatting": ("02e_train_mipsplatting.sh", "Step 2e: Mip-Splatting Training", "mipsplatting"),
}
SHORTCUTS = [name for name in BACKENDS if name != "3dgs"]

# Stage scripts are replaced by stubs that log their call; the mesh stubs create the outputs the pipeline evaluates next
STAGE_STUBS = {
    **{script: "" for script, _, _ in BACKENDS.values()},
    "01_sfm_hloc.sh": "",
    "05_eval_nvs.sh": "",
    "03_train_sugar.sh": 'd="outputs/room/mesh/sugar_${2//\\//_}/refined_mesh/room"; mkdir -p "$d"; touch "$d/sugarfine_mesh.obj"',
    "03c_mesh_tsdf.sh": 'd="outputs/room/mesh/$(basename "$2")"; mkdir -p "$d"; touch "$d/tsdf_mesh.ply"',
}

@pytest.fixture
def workspace(tmp_path):
    root = tmp_path
    for name in ("scripts/utils", "configs", "bin", "src/mesh", "src/utils", "data/room/raw_images"):
        (root / name).mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / "scripts" / "run_3drc.sh", root / "scripts")
    for helper in (PROJECT_ROOT / "scripts" / "utils").glob("*.sh"):
        shutil.copy(helper, root / "scripts" / "utils")
    for config in (PROJECT_ROOT / "configs").iterdir():
        shutil.copy(config, root / "configs")
    for name, body in STAGE_STUBS.items():
        stub = root / "scripts" / name
        stub.write_text(f'#!/bin/bash\nset -e\necho "{name} $*" >> calls.log\n{body}\n')
        stub.chmod(0o755)

    # Python entry points the pipeline calls, recording their arguments
    for module in ("src/mesh/eval_mesh.py", "src/utils/collect_results.py"):
        (root / module).write_text(f'import sys\nopen("calls.log", "a").write("{module} " + " ".join(sys.argv[1:]) + "\\n")\n')
    for package in ("src", "src/mesh", "src/utils"):
        (root / package / "__init__.py").touch()
    (root / "bin" / "python3").symlink_to(sys.executable)
    return root

def run(root: Path, *args: str, **env_overrides: str) -> subprocess.CompletedProcess:
    # No conda on PATH and an empty HOME, so run_3drc.sh falls back to python3
    env = {"PATH": f"{root}/bin:/usr/bin:/bin", "HOME": str(root), **env_overrides}
    return subprocess.run(["bash", "scripts/run_3drc.sh", *args], cwd=root, env=env, capture_output=True, text=True)

def calls(root: Path) -> list:
    log = root / "calls.log"
    return [line.rstrip() for line in log.read_text().splitlines()] if log.exists() else []

@pytest.mark.parametrize("backend", BACKENDS)
def test_train_runs_the_backend_script(workspace, backend):
    script, header, _ = BACKENDS[backend]
    result = run(workspace, "train", backend, "data/room")
    assert result.returncode == 0, result.stderr
    assert calls(workspace) == [f"{script} data/room"]
    assert header in result.stdout

@pytest.mark.parametrize("backend", BACKENDS)
def test_train_accepts_the_dataset_before_the_backend(workspace, backend):
    assert run(workspace, "train", "data/room", backend).returncode == 0
    assert calls(workspace) == [f"{BACKENDS[backend][0]} data/room"]

def test_train_defaults_to_3dgs(workspace):
    assert run(workspace, "train").returncode == 0
    assert calls(workspace) == ["02_train_3dgs.sh"]

@pytest.mark.parametrize("command", SHORTCUTS)
def test_shortcut_command_runs_the_backend_script(workspace, command):
    script, header, _ = BACKENDS[command]
    result = run(workspace, command, "data/room")
    assert result.returncode == 0, result.stderr
    assert calls(workspace) == [f"{script} data/room"]
    assert header in result.stdout

@pytest.mark.parametrize("command", SHORTCUTS)
def test_shortcut_command_without_a_dataset(workspace, command):
    assert run(workspace, command).returncode == 0
    assert calls(workspace) == [BACKENDS[command][0]]

@pytest.mark.parametrize("command", ["3dgs", "nonsense"])
def test_commands_without_a_shortcut_show_the_help(workspace, command):
    result = run(workspace, command, "data/room")
    assert result.returncode == 0
    assert calls(workspace) == []
    assert "Usage: ./scripts/run_3drc.sh" in result.stdout

def test_train_rejects_an_unknown_backend(workspace):
    result = run(workspace, "train", "nerf", "data/room")
    message = result.stdout + result.stderr
    assert result.returncode != 0
    assert calls(workspace) == []
    assert "Unknown training backend: 'nerf'" in message
    assert all(name in message for name in BACKENDS)

@pytest.mark.parametrize("backend", BACKENDS)
def test_pipeline_trains_with_the_selected_backend(workspace, backend):
    result = run(workspace, "pipeline", "data/room", "hloc", "--train", backend, "--mesh", "none")
    assert result.returncode == 0, result.stderr
    assert calls(workspace) == ["01_sfm_hloc.sh hloc data/room", f"{BACKENDS[backend][0]} data/room", "src/utils/collect_results.py"]

def test_pipeline_backend_can_come_from_the_environment(workspace):
    result = run(workspace, "pipeline", "data/room", "hloc", "--mesh", "none", PIPELINE_TRAIN="2dgs")
    assert result.returncode == 0, result.stderr
    assert "02c_train_2dgs.sh data/room" in calls(workspace)

@pytest.mark.parametrize("backend", BACKENDS)
def test_pipeline_evaluates_the_model_directory_of_the_backend(workspace, backend):
    result = run(workspace, "pipeline", "data/room", "hloc", "--train", backend, "--mesh", "none", GS_EVAL_MODE="true")
    assert result.returncode == 0, result.stderr
    assert f"05_eval_nvs.sh data/room {BACKENDS[backend][2]}" in calls(workspace)

@pytest.mark.parametrize("backend, mesh, script", [
    ("3dgs", "sugar", "03_train_sugar.sh data/room 3dgs/inria_30k"),
    ("planargs", "sugar", "03_train_sugar.sh data/room 3dgs/planargs"),
    ("2dgs", "tsdf", "03c_mesh_tsdf.sh data/room 2dgs"),
])
def test_pipeline_hands_the_model_directory_to_the_mesh_step(workspace, backend, mesh, script):
    result = run(workspace, "pipeline", "data/room", "hloc", "--train", backend, "--mesh", mesh)
    assert result.returncode == 0, result.stderr
    assert script in calls(workspace)

@pytest.mark.parametrize("backend, mesh, message", [
    ("2dgs", "sugar", "SuGaR requires a 3DGS-format model"),
    ("scaffoldgs", "sugar", "SuGaR requires a 3DGS-format model"),
    ("mipsplatting", "sugar", "SuGaR requires a 3DGS-format model"),
    ("3dgs", "tsdf", "TSDF extraction requires a 2DGS model"),
    ("planargs", "tsdf", "TSDF extraction requires a 2DGS model"),
    ("scaffoldgs", "tsdf", "TSDF extraction requires a 2DGS model"),
    ("mipsplatting", "tsdf", "TSDF extraction requires a 2DGS model"),
])
def test_pipeline_rejects_a_mesh_engine_the_backend_cannot_feed(workspace, backend, mesh, message):
    result = run(workspace, "pipeline", "data/room", "hloc", "--train", backend, "--mesh", mesh)
    assert result.returncode != 0
    assert calls(workspace) == []
    assert message in result.stdout + result.stderr
    assert f"got '{backend}'" in result.stdout + result.stderr

def test_pipeline_rejects_an_unknown_backend(workspace):
    result = run(workspace, "pipeline", "data/room", "hloc", "--train", "nerf", "--mesh", "none")
    message = result.stdout + result.stderr
    assert result.returncode != 0
    assert calls(workspace) == []
    assert "Unknown training backend: 'nerf'" in message
    assert all(name in message for name in BACKENDS)
