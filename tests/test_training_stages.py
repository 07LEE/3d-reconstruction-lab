import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Per stage script: where the model goes and the trainer command it runs.
# {root}, {model} and {iters} are filled in per test; PlanarGS runs from its own directory with absolute paths.
STAGES = {
    "3dgs": {
        "script": "02_train_3dgs.sh",
        "model": "outputs/room/3dgs/inria_30k",
        "cwd": ".",
        "trainer": "third_party/gaussian-splatting/train.py",
        "args": "-s data/room --model_path {model} -r 4 --densify_grad_threshold 0.0002 --data_device cpu --iterations {iters}",
        "bench": "3dgs_inria_train",
        "engine": "Original 3DGS (inria_30k)",
        "label": "Stage 2 (Training)",
    },
    "planargs": {
        "script": "02b_train_planargs.sh",
        "model": "{root}/outputs/room/3dgs/planargs",
        "cwd": "third_party/PlanarGS",
        "trainer": "train.py",
        "args": "-s {root}/data/room -m {model} -r 4 --data_device cpu --densify_grad_threshold 0.0002 --iterations {iters}",
        "bench": "3dgs_planargs_train",
        "engine": "PlanarGS (Planar-Regularized 3DGS)",
        "label": "Stage 2 (Training)",
    },
    "2dgs": {
        "script": "02c_train_2dgs.sh",
        "model": "outputs/room/2dgs",
        "cwd": ".",
        "trainer": "third_party/2d-gaussian-splatting/train.py",
        "args": "-s data/room -m {model} -r 4 --data_device cpu --iterations {iters}",
        "bench": "2dgs_train",
        "engine": "2D Gaussian Splatting (Surfels)",
        "label": "Stage 2 (Training)",
    },
    "scaffoldgs": {
        "script": "02d_train_scaffoldgs.sh",
        "model": "outputs/room/scaffoldgs",
        "cwd": ".",
        "trainer": "third_party/scaffold-gs/train.py",
        "args": "-s data/room --model_path {model} --data_device cpu -r 4 --iterations {iters}",
        "bench": "scaffoldgs_train",
        "engine": "Scaffold-GS (Anchor-based 3D Gaussian Splatting)",
        "label": "Stage 2d (Training)",
    },
    "mipsplatting": {
        "script": "02e_train_mipsplatting.sh",
        "model": "outputs/room/mipsplatting",
        "cwd": ".",
        "trainer": "third_party/mip-splatting/train.py",
        "args": "-s data/room --model_path {model} --data_device cpu -r 4 --iterations {iters}",
        "bench": "mipsplatting_train",
        "engine": "Mip-Splatting (3D Smoothing Filter 3DGS)",
        "label": "Stage 2e (Training)",
    },
}

SFM_FILES = ("cameras.bin", "images.bin", "points3D.bin")
DEFAULT_EXTRA = ["--checkpoint_iterations", "10000", "20000", "30000", "--save_iterations", "30000"]

# Patch verification and the CUDA build environment need a real toolchain, so they are replaced by stubs
STUBBED_HELPERS = {
    "verify_patches.sh": '[ "${VERIFY_FAIL:-0}" = "0" ]\n',
    "setup_build_env.sh": ":\n",
}

def build_workspace(root: Path, script: str) -> Path:
    for name in ("scripts/utils", "configs", "src/utils", "bin", "conda/etc/profile.d",
                 "ext/inria", "ext/scaffold", "ext/mipsplatting", "third_party/PlanarGS", "data/room/raw_images"):
        (root / name).mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / "scripts" / script, root / "scripts")
    for config in (PROJECT_ROOT / "configs").iterdir():
        shutil.copy(config, root / "configs")
    for helper in (PROJECT_ROOT / "scripts" / "utils").glob("*.sh"):
        if helper.name not in STUBBED_HELPERS:
            shutil.copy(helper, root / "scripts" / "utils")
    for name, body in STUBBED_HELPERS.items():
        (root / "scripts" / "utils" / name).write_text(body)
        (root / "scripts" / "utils" / name).chmod(0o755)

    # A conda stub that activates any environment, a trainer stub and a benchmark logger stub that record their calls
    (root / "conda" / "etc" / "profile.d" / "conda.sh").write_text(
        'conda() {\n'
        f'  export CONDA_DEFAULT_ENV="$2" CONDA_PREFIX="{root}/conda/envs/$2"\n'
        '}\n'
    )
    (root / "bin" / "conda").write_text(f'#!/bin/bash\n[ "$1" = "info" ] && echo "{root}/conda"\n')
    (root / "bin" / "python").write_text(
        f'#!{sys.executable}\n'
        'import json, os, sys\n'
        f'open("{root}/trainer.jsonl", "a").write(json.dumps({{"cwd": os.getcwd(), "args": sys.argv[1:]}}) + "\\n")\n'
        'sys.exit(int(os.environ.get("TRAINER_EXIT", "0")))\n'
    )
    for name in ("conda", "python"):
        (root / "bin" / name).chmod(0o755)
    (root / "src" / "utils" / "log_benchmark.py").write_text(
        'import json, sys\n'
        f'open("{root}/bench.jsonl", "a").write(json.dumps(sys.argv[1:]) + "\\n")\n'
    )
    return root

def add_sfm_model(root: Path) -> None:
    sparse = root / "data" / "room" / "sparse" / "0"
    sparse.mkdir(parents=True)
    for name in SFM_FILES:
        (sparse / name).touch()

def run_stage(root: Path, stage: str, dataset: str | None = "data/room", **env_overrides: str) -> subprocess.CompletedProcess:
    env = {"PATH": f"{root}/bin:/usr/bin:/bin", "HOME": str(root), "GS_EXT_ROOT": str(root / "ext"), **env_overrides}
    args = ["bash", f"scripts/{STAGES[stage]['script']}", *([dataset] if dataset else [])]
    return subprocess.run(args, cwd=root, env=env, capture_output=True, text=True)

def jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

def trainer_calls(root: Path) -> list:
    return jsonl(root / "trainer.jsonl")

def bench_calls(root: Path) -> list:
    return jsonl(root / "bench.jsonl")

def parse_bench(args: list) -> dict:
    parsed = {"set": []}
    for flag, value in zip(args[::2], args[1::2]):
        if flag == "--set":
            parsed["set"].append(value)
        else:
            parsed[flag.lstrip("-")] = value
    return parsed

def model_arg(root: Path, stage: str) -> str:
    return STAGES[stage]["model"].format(root=root)

def model_path(root: Path, stage: str) -> Path:
    return Path(root, model_arg(root, stage))

def expected_trainer(root: Path, stage: str, extra: list, iters: str = "30000") -> dict:
    spec = STAGES[stage]
    args = spec["args"].format(root=root, model=model_arg(root, stage), iters=iters).split()
    return {"cwd": str(root / spec["cwd"]), "args": [spec["trainer"], *args, *extra]}

def set_age(path: Path, seconds_ago: int) -> None:
    moment = time.time() - seconds_ago
    os.utime(path, (moment, moment))

@pytest.fixture(params=list(STAGES))
def stage(request):
    return request.param

@pytest.fixture
def workspace(tmp_path, stage):
    return build_workspace(tmp_path.resolve(), STAGES[stage]["script"])

def test_runs_the_trainer_with_the_backend_arguments(workspace, stage):
    add_sfm_model(workspace)
    result = run_stage(workspace, stage)
    assert result.returncode == 0, result.stderr
    assert trainer_calls(workspace) == [expected_trainer(workspace, stage, DEFAULT_EXTRA)]

def test_dataset_defaults_to_the_latest_scene(workspace, stage):
    add_sfm_model(workspace)
    result = run_stage(workspace, stage, dataset=None)
    assert result.returncode == 0, result.stderr
    assert trainer_calls(workspace) == [expected_trainer(workspace, stage, DEFAULT_EXTRA)]

def test_records_timing_and_provenance(workspace, stage):
    add_sfm_model(workspace)
    assert run_stage(workspace, stage).returncode == 0
    spec = STAGES[stage]
    [bench] = [parse_bench(call) for call in bench_calls(workspace)]
    assert (bench["stage"], bench["dataset"], bench["output-dir"]) == (spec["bench"], "room", str(model_path(workspace, stage)))
    assert bench["set"] == ["iterations=30000", "downsample_rate=4", "eval_split=false", "active_sfm=unknown", "resumed_from="]
    meta = json.loads((model_path(workspace, stage) / "pipeline_meta.json").read_text())
    meta.pop("timestamp")
    assert meta == {
        "stage": spec["label"],
        "engine": spec["engine"],
        "source_dataset": "data/room",
        "active_sfm": "unknown",
        "downsample_rate": 4,
        "iterations": 30000,
    }

def test_active_sfm_follows_the_sparse_link(workspace, stage):
    sparse = workspace / "data" / "room" / "sparse"
    (sparse / "hloc").mkdir(parents=True)
    for name in SFM_FILES:
        (sparse / "hloc" / name).touch()
    (sparse / "0").symlink_to("hloc")
    assert run_stage(workspace, stage).returncode == 0
    assert "active_sfm=hloc" in parse_bench(bench_calls(workspace)[0])["set"]

def test_missing_sfm_model_stops_with_a_clear_message(workspace, stage):
    result = run_stage(workspace, stage)
    assert result.returncode != 0
    assert "SfM model not found" in result.stdout + result.stderr
    assert trainer_calls(workspace) == []

@pytest.mark.parametrize("layout", ["sfm", "sfm/models/0"])
def test_sfm_model_is_taken_from_the_cache_when_sparse_is_missing(workspace, stage, layout):
    cache = workspace / "data" / "room" / "cache" / layout
    cache.mkdir(parents=True)
    for name in SFM_FILES:
        (cache / name).touch()
    result = run_stage(workspace, stage)
    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in (workspace / "data" / "room" / "sparse" / "0").iterdir()) == sorted(SFM_FILES)

@pytest.mark.parametrize("env, checkpoints, iters", [
    ({}, ["10000", "20000", "30000"], "30000"),
    ({"GS_CHECKPOINT_ITERATIONS": "5000"}, ["5000"], "30000"),
    ({"GS_CHECKPOINT_INTERVAL": "0"}, [], "30000"),
    ({"GS_CHECKPOINT_INTERVAL": "5000", "GS_ITERATIONS": "12000"}, ["5000", "10000"], "12000"),
])
def test_final_iteration_is_always_saved(workspace, stage, env, checkpoints, iters):
    add_sfm_model(workspace)
    result = run_stage(workspace, stage, **env)
    assert result.returncode == 0, result.stderr
    extra = [*(["--checkpoint_iterations", *checkpoints] if checkpoints else []), "--save_iterations", iters]
    assert trainer_calls(workspace) == [expected_trainer(workspace, stage, extra, iters)]

def test_holdout_flag_follows_the_eval_mode(workspace, stage):
    add_sfm_model(workspace)
    assert run_stage(workspace, stage, GS_EVAL_MODE="true").returncode == 0
    assert trainer_calls(workspace) == [expected_trainer(workspace, stage, ["--eval", *DEFAULT_EXTRA])]
    assert "eval_split=true" in parse_bench(bench_calls(workspace)[0])["set"]

def test_images_link_replaces_a_real_directory(workspace, stage):
    add_sfm_model(workspace)
    images = workspace / "data" / "room" / "images"
    images.mkdir()
    (images / "stale.jpg").touch()
    assert run_stage(workspace, stage).returncode == 0
    assert images.is_symlink() and os.readlink(images) == "raw_images"

def test_resumes_from_the_latest_checkpoint(workspace, stage):
    add_sfm_model(workspace)
    model = model_path(workspace, stage)
    model.mkdir(parents=True)
    for name in ("chkpnt10000.pth", "chkpnt30000.pth"):
        (model / name).touch()
    set_age(workspace / "data" / "room" / "sparse" / "0" / "points3D.bin", 1000)
    assert run_stage(workspace, stage).returncode == 0
    resume = ["--start_checkpoint", f"{model_arg(workspace, stage)}/chkpnt30000.pth"]
    assert trainer_calls(workspace) == [expected_trainer(workspace, stage, [*DEFAULT_EXTRA, *resume])]
    assert "resumed_from=chkpnt30000.pth" in parse_bench(bench_calls(workspace)[0])["set"]

def test_stale_checkpoint_is_not_resumed(workspace, stage):
    add_sfm_model(workspace)
    model = model_path(workspace, stage)
    model.mkdir(parents=True)
    (model / "chkpnt30000.pth").touch()
    set_age(model / "chkpnt30000.pth", 1000)
    assert run_stage(workspace, stage).returncode == 0
    assert trainer_calls(workspace) == [expected_trainer(workspace, stage, DEFAULT_EXTRA)]
    assert "resumed_from=" in parse_bench(bench_calls(workspace)[0])["set"]

def test_trainer_failure_stops_the_stage(workspace, stage):
    add_sfm_model(workspace)
    result = run_stage(workspace, stage, TRAINER_EXIT="3")
    assert result.returncode != 0
    assert bench_calls(workspace) == []
    assert not (model_path(workspace, stage) / "pipeline_meta.json").exists()

def test_patch_verification_failure_stops_before_training(workspace, stage):
    add_sfm_model(workspace)
    result = run_stage(workspace, stage, VERIFY_FAIL="1")
    assert result.returncode != 0
    assert trainer_calls(workspace) == []
