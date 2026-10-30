import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = PROJECT_ROOT / "configs" / "backends.txt"

def registry_rows() -> list:
    lines = (line.strip() for line in REGISTRY.read_text().splitlines())
    return [line.split() for line in lines if line and not line.startswith("#")]

def shell(call: str, root: Path = PROJECT_ROOT) -> subprocess.CompletedProcess:
    script = f'set -euo pipefail; source "{root}/scripts/utils/backends.sh"; {call}'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)

def test_every_row_has_the_documented_columns():
    rows = registry_rows()
    names = [row[0] for row in rows]
    assert rows
    assert all(len(row) == 7 for row in rows)
    assert all(row[3] in ("yes", "no") for row in rows)
    assert len(set(names)) == len(names)

def test_registered_scripts_exist():
    for name, script, *_ in registry_rows():
        assert (PROJECT_ROOT / "scripts" / script).is_file(), name

def test_variants_are_built_by_the_setup_script():
    setup = (PROJECT_ROOT / "scripts" / "00_setup_environment.sh").read_text()
    built = set(re.findall(r"GS_RASTERIZER_VARIANTS\[(\w+)\]=", setup))
    assert {row[5] for row in registry_rows()} <= built

def test_evaluation_repos_are_submodules():
    modules = (PROJECT_ROOT / ".gitmodules").read_text()
    for row in registry_rows():
        assert f"path = {row[6]}" in modules, row[0]

def test_registered_backends_can_be_activated():
    names = " ".join(row[0] for row in registry_rows())
    script = (f'set -euo pipefail; source "{PROJECT_ROOT}/scripts/utils/activate_backend.sh"; '
              f'for name in {names}; do backend_rasterizer_variant "$name" > /dev/null; done')
    assert subprocess.run(["bash", "-c", script]).returncode == 0

def test_lookups_read_the_registry():
    result = shell('backend_names ", "; backend_script 2dgs; backend_model_subdir planargs; backend_step 3dgs; '
                   'backend_title 2dgs; backend_header 2dgs; backend_header mipsplatting')
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "3dgs, planargs, 2dgs, scaffoldgs, mipsplatting",
        "02c_train_2dgs.sh",
        "3dgs/planargs",
        "2",
        "2D Gaussian Splatting",
        "Step 2c: 2D Gaussian Splatting Training",
        "Step 2e: Mip-Splatting Training",
    ]

def test_variant_and_repo_lookups():
    result = shell('backend_variant planargs; backend_variant scaffoldgs; backend_repo 2dgs; backend_repo mipsplatting')
    assert result.stdout.splitlines() == ["inria", "scaffold", "third_party/2d-gaussian-splatting", "third_party/mip-splatting"]

@pytest.mark.parametrize("path, backend", [
    ("/ws/outputs/room/3dgs/inria_30k", "3dgs"),
    ("/ws/outputs/room/3dgs/planargs", "planargs"),
    ("/ws/outputs/room/3dgs/another_run", "3dgs"),
    ("/ws/outputs/room/2dgs", "2dgs"),
    ("/ws/outputs/room/2dgs_run2", "2dgs"),
    ("/ws/outputs/room/scaffoldgs", "scaffoldgs"),
    ("/ws/outputs/room/mipsplatting", "mipsplatting"),
    ("/elsewhere/models/2dgs", "2dgs"),
])
def test_backend_is_inferred_from_a_model_path(path, backend):
    assert shell(f'backend_infer "{path}"').stdout.strip() == backend

def test_paths_without_a_model_directory_are_not_inferred():
    assert shell('backend_infer "/ws/outputs/room/custom_run"').returncode != 0

def test_names_are_space_separated_by_default():
    assert shell("backend_names").stdout.split() == ["3dgs", "planargs", "2dgs", "scaffoldgs", "mipsplatting"]

def test_only_the_default_backend_lacks_a_shortcut():
    result = shell('for name in $(backend_names); do backend_has_shortcut "$name" || echo "$name"; done')
    assert result.stdout.split() == ["3dgs"]

@pytest.mark.parametrize("call", [
    "backend_exists nerf", "backend_script nerf", "backend_model_subdir nerf", "backend_step nerf",
    "backend_title nerf", "backend_header nerf", "backend_has_shortcut nerf", "backend_variant nerf",
    "backend_repo nerf",
])
def test_unknown_backends_fail_every_lookup(call):
    assert shell(call).returncode != 0

def test_comments_and_blank_lines_are_ignored(tmp_path):
    (tmp_path / "scripts" / "utils").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    shutil.copy(PROJECT_ROOT / "scripts" / "utils" / "backends.sh", tmp_path / "scripts" / "utils")
    (tmp_path / "configs" / "backends.txt").write_text(
        "# comment\n\n   # indented comment\nalpha  01_a.sh  a  yes  Alpha_One\n\nbeta  02b_b.sh  b  no  Beta\n"
    )
    result = shell('backend_names; backend_header alpha; backend_header beta', root=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["alpha beta", "Step 1: Alpha One Training", "Step 2b: Beta Training"]
