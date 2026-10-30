import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Backend -> model directory under outputs/<scene>/, the repository whose render.py and metrics.py run,
# the rasterizer build it activates, its extra arguments, and the directory the evaluation runs from
BACKENDS = {
    "3dgs": {"model": "3dgs/inria_30k", "repo": "third_party/gaussian-splatting", "variant": "inria",
             "render": [], "metrics": [], "cwd": "."},
    "planargs": {"model": "3dgs/planargs", "repo": "third_party/PlanarGS", "variant": "inria",
                 "render": [], "metrics": [], "cwd": "third_party/PlanarGS"},
    "2dgs": {"model": "2dgs", "repo": "third_party/2d-gaussian-splatting", "variant": "inria",
             "render": ["--skip_mesh"], "metrics": [], "cwd": "."},
    "scaffoldgs": {"model": "scaffoldgs", "repo": "third_party/scaffold-gs", "variant": "scaffold",
                   "render": [], "metrics": [], "cwd": "."},
    "mipsplatting": {"model": "mipsplatting", "repo": "third_party/mip-splatting", "variant": "mipsplatting",
                     "render": [], "metrics": ["-r", "4"], "cwd": "."},
}

def build_workspace(root: Path) -> Path:
    for name in ("scripts/utils", "configs", "bin", "src/utils", "conda/etc/profile.d", "ext/inria", "ext/scaffold",
                 "ext/mipsplatting", "third_party/PlanarGS", "data/room"):
        (root / name).mkdir(parents=True)
    root = root.resolve()
    shutil.copy(PROJECT_ROOT / "scripts" / "05_eval_nvs.sh", root / "scripts")
    for helper in (PROJECT_ROOT / "scripts" / "utils").glob("*.sh"):
        shutil.copy(helper, root / "scripts" / "utils")
    for config in (PROJECT_ROOT / "configs").iterdir():
        shutil.copy(config, root / "configs")
    # Patch verification needs a real toolchain, so it is skipped
    (root / "scripts" / "utils" / "verify_patches.sh").write_text(":\n")
    (root / "scripts" / "utils" / "verify_patches.sh").chmod(0o755)

    # A conda stub that activates any environment, a python stub and a benchmark logger stub that record their calls
    (root / "conda" / "etc" / "profile.d" / "conda.sh").write_text(
        'conda() {\n'
        f'  export CONDA_DEFAULT_ENV="$2" CONDA_PREFIX="{root}/conda/envs/$2"\n'
        '}\n'
    )
    (root / "bin" / "conda").write_text(f'#!/bin/bash\n[ "$1" = "info" ] && echo "{root}/conda"\n')
    (root / "bin" / "python").write_text(
        f'#!{sys.executable}\n'
        'import json, os, sys\n'
        f'open("{root}/python.jsonl", "a").write(json.dumps({{"cwd": os.getcwd(), "args": sys.argv[1:], '
        '"pythonpath": os.environ.get("PYTHONPATH", "")}) + "\\n")\n'
    )
    for name in ("conda", "python"):
        (root / "bin" / name).chmod(0o755)
    (root / "src" / "utils" / "log_benchmark.py").write_text(
        'import json, sys\n'
        f'open("{root}/bench.jsonl", "a").write(json.dumps(sys.argv[1:]) + "\\n")\n'
    )
    return root

@pytest.fixture
def workspace(tmp_path):
    return build_workspace(tmp_path / "ws")

def add_model(root: Path, subdir: str, scene: str = "room", cfg: str = "Namespace(eval=True, resolution=4)") -> Path:
    model = root / "outputs" / scene / subdir
    model.mkdir(parents=True)
    (model / "cfg_args").write_text(cfg)
    (model / "results.json").write_text("{}")
    return model

def run_nvs(root: Path, *args: str, **env_overrides: str) -> subprocess.CompletedProcess:
    env = {"PATH": f"{root}/bin:/usr/bin:/bin", "HOME": str(root), "GS_EXT_ROOT": str(root / "ext"), **env_overrides}
    return subprocess.run(["bash", "scripts/05_eval_nvs.sh", *args], cwd=root, env=env, capture_output=True, text=True)

def jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

def python_calls(root: Path) -> list:
    return jsonl(root / "python.jsonl")

def bench_options(root: Path) -> list:
    options = []
    for call in jsonl(root / "bench.jsonl"):
        parsed = {"set": []}
        for flag, value in zip(call[::2], call[1::2]):
            if flag == "--set":
                parsed["set"].append(value)
            else:
                parsed[flag.lstrip("-")] = value
        options.append(parsed)
    return options

def expected_calls(root: Path, backend: str, model: Path) -> list:
    spec = BACKENDS[backend]
    cwd, repo = str(root / spec["cwd"]), root / spec["repo"]
    return [
        {"cwd": cwd, "args": [f"{repo}/render.py", "-m", str(model), "--skip_train", "--data_device", "cpu", *spec["render"]]},
        {"cwd": cwd, "args": [f"{repo}/metrics.py", "-m", str(model), *spec["metrics"]]},
    ]

def without_pythonpath(calls: list) -> list:
    return [{"cwd": call["cwd"], "args": call["args"]} for call in calls]

@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_is_inferred_from_the_model_directory(workspace, backend):
    spec = BACKENDS[backend]
    model = add_model(workspace, spec["model"])
    result = run_nvs(workspace, "data/room", spec["model"])
    assert result.returncode == 0, result.stderr
    calls = python_calls(workspace)
    assert without_pythonpath(calls) == expected_calls(workspace, backend, model)
    assert [call["pythonpath"].split(":")[0] for call in calls] == [str(workspace / "ext" / spec["variant"])] * 2
    [bench] = bench_options(workspace)
    assert (bench["stage"], bench["dataset"], bench["set"]) == (f"nvs_eval_{backend}", "room", [f"backend={backend}"])

@pytest.mark.parametrize("backend", BACKENDS)
def test_exact_model_path_is_accepted(workspace, backend):
    model = add_model(workspace, BACKENDS[backend]["model"])
    result = run_nvs(workspace, "data/room", f"outputs/room/{BACKENDS[backend]['model']}")
    assert result.returncode == 0, result.stderr
    assert without_pythonpath(python_calls(workspace)) == expected_calls(workspace, backend, model)

def test_defaults_to_the_inria_model(workspace):
    model = add_model(workspace, "3dgs/inria_30k")
    result = run_nvs(workspace, "data/room")
    assert result.returncode == 0, result.stderr
    assert without_pythonpath(python_calls(workspace)) == expected_calls(workspace, "3dgs", model)

def test_backend_can_be_set_explicitly(workspace):
    model = add_model(workspace, "custom_run")
    result = run_nvs(workspace, "data/room", "custom_run", NVS_BACKEND="scaffoldgs")
    assert result.returncode == 0, result.stderr
    assert without_pythonpath(python_calls(workspace)) == expected_calls(workspace, "scaffoldgs", model)

def test_unknown_layout_asks_for_the_backend(workspace):
    add_model(workspace, "custom_run")
    result = run_nvs(workspace, "data/room", "custom_run")
    message = result.stdout + result.stderr
    assert result.returncode != 0
    assert python_calls(workspace) == []
    assert "Cannot infer backend" in message
    assert all(name in message for name in BACKENDS)

def test_unknown_backend_override_is_rejected(workspace):
    add_model(workspace, "custom_run")
    result = run_nvs(workspace, "data/room", "custom_run", NVS_BACKEND="nerf")
    assert result.returncode != 0
    assert python_calls(workspace) == []
    assert "Unknown NVS backend: nerf" in result.stdout + result.stderr

def test_missing_model_directory_is_reported(workspace):
    result = run_nvs(workspace, "data/room", "2dgs")
    assert result.returncode != 0
    assert "Model directory not found" in result.stdout + result.stderr

def test_model_trained_without_a_holdout_split_is_rejected(workspace):
    add_model(workspace, "2dgs", cfg="Namespace(eval=False, resolution=4)")
    result = run_nvs(workspace, "data/room", "2dgs")
    assert result.returncode != 0
    assert python_calls(workspace) == []
    assert "without a holdout split" in result.stdout + result.stderr
    assert "Move or delete" in result.stdout + result.stderr

def test_mip_splatting_needs_the_training_resolution(workspace):
    add_model(workspace, "mipsplatting", cfg="Namespace(eval=True)")
    result = run_nvs(workspace, "data/room", "mipsplatting")
    assert result.returncode != 0
    assert python_calls(workspace) == []
    assert "Training resolution not found" in result.stdout + result.stderr

@pytest.mark.parametrize("scene, subdir, backend", [
    ("2dgs_room", "3dgs/inria_30k", "3dgs"),
    ("2dgs_room", "scaffoldgs", "scaffoldgs"),
    ("2dgs_room", "mipsplatting", "mipsplatting"),
    ("scaffoldgs_room", "mipsplatting", "mipsplatting"),
])
def test_scene_names_do_not_change_the_inferred_backend(workspace, scene, subdir, backend):
    (workspace / "data" / scene).mkdir()
    model = add_model(workspace, subdir, scene=scene)
    result = run_nvs(workspace, f"data/{scene}", subdir)
    assert result.returncode == 0, result.stderr
    assert without_pythonpath(python_calls(workspace)) == expected_calls(workspace, backend, model)

def test_parent_directories_do_not_change_the_inferred_backend(tmp_path):
    root = build_workspace(tmp_path / "2dgs_lab" / "ws")
    model = add_model(root, "3dgs/inria_30k")
    result = run_nvs(root, "data/room", "3dgs/inria_30k")
    assert result.returncode == 0, result.stderr
    assert without_pythonpath(python_calls(root)) == expected_calls(root, "3dgs", model)
