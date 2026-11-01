"""Stubbed project copy for running the CLI end to end; the workspace fixture is in conftest.py."""

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Training backend -> stage script, the header the CLI prints, and the model directory under outputs/<scene>/
BACKENDS = {
    "3dgs": ("02_train_3dgs.sh", "Step 2: 3DGS Training", "3dgs/inria_30k"),
    "planargs": ("02b_train_planargs.sh", "Step 2b: PlanarGS Training", "3dgs/planargs"),
    "2dgs": ("02c_train_2dgs.sh", "Step 2c: 2D Gaussian Splatting Training", "2dgs"),
    "scaffoldgs": ("02d_train_scaffoldgs.sh", "Step 2d: Scaffold-GS Training", "scaffoldgs"),
    "mipsplatting": ("02e_train_mipsplatting.sh", "Step 2e: Mip-Splatting Training", "mipsplatting"),
}

# Stage scripts are replaced by stubs that log their call; the mesh stubs create the outputs the pipeline evaluates next
STAGE_STUBS = {
    **{script: "" for script, _, _ in BACKENDS.values()},
    "00b_data_yuv.sh": "",
    "01_sfm_hloc.sh": "",
    "03b_train_milo.sh": '[ -n "${MILO_SKIP_OUTPUT:-}" ] || { mkdir -p outputs/room/mesh/milo && touch outputs/room/mesh/milo/mesh_cleaned_largest.ply; }',
    "04_train_grouping.sh": "",
    "05_eval_nvs.sh": "",
    "03_train_sugar.sh": 'm="${2:-3dgs/inria_30k}"; d="outputs/room/mesh/sugar_${m//\\//_}/refined_mesh/room"; mkdir -p "$d"; touch "$d/sugarfine_mesh.obj"',
    "03c_mesh_tsdf.sh": 'd="outputs/room/mesh/$(basename "${2:-2dgs}")"; mkdir -p "$d"; touch "$d/tsdf_mesh.ply"',
    "utils/view_reconstruction.sh": "",
}


def build_workspace(root: Path) -> Path:
    """Copies the CLI, the config and the script helpers into root, stubs every stage script, and adds data/room."""
    for name in ("scripts/utils", "configs", "bin", "src/cli", "src/mesh", "src/utils", "data/room/raw_images"):
        (root / name).mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / "3drc.py", root)
    shutil.copy(PROJECT_ROOT / "scripts" / "run_3drc.sh", root / "scripts")
    for helper in (PROJECT_ROOT / "scripts" / "utils").glob("*.sh"):
        shutil.copy(helper, root / "scripts" / "utils")
    for config in (PROJECT_ROOT / "configs").iterdir():
        shutil.copy(config, root / "configs")
    shutil.copy(PROJECT_ROOT / "src" / "cli" / "main.py", root / "src" / "cli")
    for name, body in STAGE_STUBS.items():
        stub = root / "scripts" / name
        stub.write_text(f'#!/bin/bash\nset -e\necho "{name} $*" >> calls.log\n{body}\n')
        stub.chmod(0o755)

    # Python entry points the CLI calls, recording their arguments
    for module in ("src/mesh/eval_mesh.py", "src/utils/collect_results.py", "src/utils/inspect_outputs.py"):
        (root / module).write_text(f'import sys\nopen("calls.log", "a").write(("{module} " + " ".join(sys.argv[1:])).strip() + "\\n")\n')
    for package in ("src", "src/cli", "src/mesh", "src/utils"):
        (root / package / "__init__.py").touch()
    (root / "bin" / "python3").symlink_to(sys.executable)
    return root


def run(root: Path, *args: str, **env_overrides: str) -> subprocess.CompletedProcess:
    """Runs scripts/run_3drc.sh in the workspace."""
    # No conda on PATH and an empty HOME, so the CLI falls back to python3
    env = {"PATH": f"{root}/bin:/usr/bin:/bin", "HOME": str(root), **env_overrides}
    return subprocess.run(["bash", "scripts/run_3drc.sh", *args], cwd=root, env=env, capture_output=True, text=True)


def calls(root: Path) -> list:
    log = root / "calls.log"
    return [line.rstrip() for line in log.read_text().splitlines()] if log.exists() else []
