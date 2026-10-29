import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELPER = PROJECT_ROOT / "scripts" / "utils" / "activate_backend.sh"
BACKENDS = ["3dgs", "2dgs", "tsdf", "planargs", "sugar", "scaffoldgs", "mipsplatting", "milo", "grouping"]

def fake_conda(tmp_path: Path, envs) -> Path:
    """A conda stub: `conda info --base` prints the base and `conda activate` succeeds for known envs."""
    base = tmp_path / "conda"
    (base / "etc" / "profile.d").mkdir(parents=True)
    known = " ".join(envs)
    (base / "etc" / "profile.d" / "conda.sh").write_text(
        'conda() {\n'
        '  if [ "$1" = "activate" ]; then\n'
        f'    case " {known} " in *" $2 "*) export CONDA_DEFAULT_ENV="$2"; export CONDA_PREFIX="{base}/envs/$2" ;; *) return 1 ;; esac\n'
        '  fi\n'
        '}\n'
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "conda").write_text(f'#!/bin/bash\n[ "$1" = "info" ] && echo "{base}"\n')
    (bin_dir / "conda").chmod(0o755)
    return bin_dir

def activate(tmp_path: Path, backend: str, envs, **env_overrides: str) -> subprocess.CompletedProcess:
    bin_dir = fake_conda(tmp_path, envs)
    script = f'set -euo pipefail; source "{HELPER}"; activate_backend {backend}; echo "ENV=$CONDA_DEFAULT_ENV"; echo "EXT=$BACKEND_EXT_PATH"; echo "PYTHONPATH=${{PYTHONPATH:-}}"'
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), **env_overrides}
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)

def parse(result: subprocess.CompletedProcess):
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)

@pytest.mark.parametrize("backend, variant", [
    ("3dgs", "inria"), ("2dgs", "inria"), ("tsdf", "inria"), ("planargs", "inria"), ("sugar", "inria"),
    ("scaffoldgs", "scaffold"), ("mipsplatting", "mipsplatting"), ("milo", "milo"), ("grouping", "grouping"),
])
def test_puts_backend_rasterizer_first(tmp_path, backend, variant):
    ext_root = tmp_path / "ext"
    (ext_root / variant).mkdir(parents=True)
    result = activate(tmp_path, backend, ["gs"], GS_EXT_ROOT=str(ext_root), PYTHONPATH="/existing")
    assert result.returncode == 0, result.stderr
    out = parse(result)
    assert out["ENV"] == "gs"
    assert out["EXT"] == str(ext_root / variant)
    assert out["PYTHONPATH"] == f"{ext_root / variant}:/existing"

def test_env_name_override(tmp_path):
    ext_root = tmp_path / "ext"
    (ext_root / "inria").mkdir(parents=True)
    result = activate(tmp_path, "3dgs", ["gs_custom"], GS_ENV_NAME="gs_custom", GS_EXT_ROOT=str(ext_root))
    assert result.returncode == 0, result.stderr
    assert parse(result)["ENV"] == "gs_custom"

def test_without_build_fails(tmp_path):
    result = activate(tmp_path, "milo", ["gs"], GS_EXT_ROOT=str(tmp_path / "empty"))
    assert result.returncode != 0
    assert "--env gs" in result.stderr

def test_missing_env_reports_setup_command(tmp_path):
    result = activate(tmp_path, "scaffoldgs", [])
    assert result.returncode != 0
    assert "./scripts/00_setup_environment.sh --env gs" in result.stderr

def test_unknown_backend_fails(tmp_path):
    assert activate(tmp_path, "nerf", ["gs"]).returncode != 0

def test_every_stage_script_activates_a_known_backend():
    for script in sorted((PROJECT_ROOT / "scripts").glob("0[2-4]*.sh")):
        lines = [l.split() for l in script.read_text().splitlines() if l.startswith(("activate_backend ", "train_env "))]
        assert len(lines) == 1, script.name
        assert lines[0][1] in BACKENDS, script.name
        assert "conda activate" not in script.read_text(), script.name
