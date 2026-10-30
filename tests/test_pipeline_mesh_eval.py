import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Stage scripts are replaced by stubs that log their call and create the canonical mesh outputs
STAGE_STUBS = {
    "01_sfm_hloc.sh": "",
    "02_train_3dgs.sh": "",
    "02c_train_2dgs.sh": "",
    "05_eval_nvs.sh": "",
    "03_train_sugar.sh": 'mkdir -p outputs/room/mesh/sugar_3dgs_inria_30k/refined_mesh/room && touch outputs/room/mesh/sugar_3dgs_inria_30k/refined_mesh/room/sugarfine_mesh.obj',
    "03c_mesh_tsdf.sh": 'mkdir -p outputs/room/mesh/2dgs && touch outputs/room/mesh/2dgs/tsdf_mesh.ply',
    "03b_train_milo.sh": ': "${MILO_SKIP_OUTPUT:=}"; [ -n "$MILO_SKIP_OUTPUT" ] || { mkdir -p outputs/room/mesh/milo && touch outputs/room/mesh/milo/mesh_cleaned_largest.ply; }',
}

def build_workspace(root: Path) -> None:
    (root / "scripts" / "utils").mkdir(parents=True)
    (root / "configs").mkdir()
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
        path = root / module
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'import sys\nopen("calls.log", "a").write("{module} " + " ".join(sys.argv[1:]) + "\\n")\n')
    for package in ("src", "src/mesh", "src/utils"):
        (root / package / "__init__.py").touch()

    (root / "data" / "room" / "raw_images").mkdir(parents=True)

def run_pipeline(root: Path, *args: str, **env_overrides: str) -> subprocess.CompletedProcess:
    bin_dir = root / "bin"
    bin_dir.mkdir(exist_ok=True)
    python = bin_dir / "python3"
    if not python.exists():
        python.symlink_to(sys.executable)
    # No conda on PATH and an empty HOME, so run_3drc.sh falls back to python3
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(root), **env_overrides}
    return subprocess.run(["bash", "scripts/run_3drc.sh", "pipeline", *args], cwd=root, env=env,
                          capture_output=True, text=True)

def calls(root: Path):
    log = root / "calls.log"
    return log.read_text().splitlines() if log.exists() else []

@pytest.fixture
def workspace(tmp_path):
    build_workspace(tmp_path)
    return tmp_path

def test_tsdf_mesh_is_evaluated_against_scene_lidar(workspace):
    (workspace / "data" / "room" / "lidar.pcd").touch()
    result = run_pipeline(workspace, "data/room", "hloc", "--train", "2dgs", "--mesh", "tsdf")
    assert result.returncode == 0, result.stderr

    log = calls(workspace)
    assert "src/mesh/eval_mesh.py --mesh outputs/room/mesh/2dgs/tsdf_mesh.ply --gt data/room/lidar.pcd" in log
    # Evaluation runs after mesh extraction and before the results report
    order = [line.split()[0] for line in log]
    assert order.index("03c_mesh_tsdf.sh") < order.index("src/mesh/eval_mesh.py") < order.index("src/utils/collect_results.py")

def test_sugar_mesh_without_gt_is_evaluated_for_topology(workspace):
    result = run_pipeline(workspace, "data/room", "hloc", "--train", "3dgs", "--mesh", "sugar")
    assert result.returncode == 0, result.stderr
    assert "src/mesh/eval_mesh.py --mesh outputs/room/mesh/sugar_3dgs_inria_30k/refined_mesh/room/sugarfine_mesh.obj" in calls(workspace)
    assert "topology only" in result.stdout + result.stderr

def test_explicit_gt_overrides_scene_lidar(workspace):
    (workspace / "data" / "room" / "lidar.pcd").touch()
    (workspace / "scan.ply").touch()
    result = run_pipeline(workspace, "data/room", "hloc", "--mesh", "milo", "--gt", "scan.ply")
    assert result.returncode == 0, result.stderr
    assert "src/mesh/eval_mesh.py --mesh outputs/room/mesh/milo/mesh_cleaned_largest.ply --gt scan.ply" in calls(workspace)

def test_missing_gt_fails_before_sfm(workspace):
    result = run_pipeline(workspace, "data/room", "hloc", "--mesh", "milo", "--gt", "missing.pcd")
    assert result.returncode != 0
    assert calls(workspace) == []

def test_missing_mesh_output_fails(workspace):
    result = run_pipeline(workspace, "data/room", "hloc", "--mesh", "milo", MILO_SKIP_OUTPUT="1")
    assert result.returncode != 0
    assert not any(line.startswith("src/utils/collect_results.py") for line in calls(workspace))

def test_no_mesh_engine_skips_evaluation(workspace):
    result = run_pipeline(workspace, "data/room", "hloc", "--mesh", "none")
    assert result.returncode == 0, result.stderr
    assert not any(line.startswith("src/mesh/eval_mesh.py") for line in calls(workspace))
