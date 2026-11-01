import pytest

from tests.cli.cli_workspace import BACKENDS, calls, run


def pipeline(root, *args, **env):
    return run(root, "pipeline", "--dataset", "data/room", "--method", "hloc", *args, **env)


@pytest.mark.parametrize("backend", BACKENDS)
def test_pipeline_trains_with_the_selected_backend(workspace, backend):
    result = pipeline(workspace, "--train", backend, "--mesh", "none")
    assert result.returncode == 0, result.stderr
    assert calls(workspace) == ["01_sfm_hloc.sh hloc data/room", f"{BACKENDS[backend][0]} data/room", "src/utils/collect_results.py"]


def test_pipeline_backend_can_come_from_the_environment(workspace):
    result = pipeline(workspace, "--mesh", "none", PIPELINE_TRAIN="2dgs")
    assert result.returncode == 0, result.stderr
    assert "02c_train_2dgs.sh data/room" in calls(workspace)


def test_pipeline_uses_the_config_scene_and_method(workspace):
    result = run(workspace, "pipeline", "--mesh", "none")
    assert result.returncode == 0, result.stderr
    assert calls(workspace)[:2] == ["01_sfm_hloc.sh hloc data/room", "02_train_3dgs.sh data/room"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_pipeline_evaluates_the_model_directory_of_the_backend(workspace, backend):
    result = pipeline(workspace, "--train", backend, "--mesh", "none", GS_EVAL_MODE="true")
    assert result.returncode == 0, result.stderr
    assert f"05_eval_nvs.sh data/room {BACKENDS[backend][2]}" in calls(workspace)


@pytest.mark.parametrize("backend, mesh, script", [
    ("3dgs", "sugar", "03_train_sugar.sh data/room 3dgs/inria_30k"),
    ("planargs", "sugar", "03_train_sugar.sh data/room 3dgs/planargs"),
    ("2dgs", "tsdf", "03c_mesh_tsdf.sh data/room 2dgs"),
])
def test_pipeline_hands_the_model_directory_to_the_mesh_step(workspace, backend, mesh, script):
    result = pipeline(workspace, "--train", backend, "--mesh", mesh)
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
    result = pipeline(workspace, "--train", backend, "--mesh", mesh)
    assert result.returncode != 0
    assert calls(workspace) == []
    assert message in result.stderr
    assert f"got '{backend}'" in result.stderr


@pytest.mark.parametrize("option, value, message", [
    ("--train", "nerf", "Unknown training backend: 'nerf'"),
    ("--mesh", "poisson", "Unknown mesh engine: 'poisson'"),
])
def test_pipeline_rejects_an_unknown_choice(workspace, option, value, message):
    result = pipeline(workspace, option, value)
    assert result.returncode != 0
    assert calls(workspace) == []
    assert message in result.stderr


def test_tsdf_mesh_is_evaluated_against_scene_lidar(workspace):
    (workspace / "data" / "room" / "lidar.pcd").touch()
    result = pipeline(workspace, "--train", "2dgs", "--mesh", "tsdf")
    assert result.returncode == 0, result.stderr

    log = calls(workspace)
    assert "src/mesh/eval_mesh.py --mesh outputs/room/mesh/2dgs/tsdf_mesh.ply --gt data/room/lidar.pcd" in log
    # Evaluation runs after mesh extraction and before the results report
    order = [line.split()[0] for line in log]
    assert order.index("03c_mesh_tsdf.sh") < order.index("src/mesh/eval_mesh.py") < order.index("src/utils/collect_results.py")


def test_sugar_mesh_without_gt_is_evaluated_for_topology(workspace):
    result = pipeline(workspace, "--train", "3dgs", "--mesh", "sugar")
    assert result.returncode == 0, result.stderr
    assert "src/mesh/eval_mesh.py --mesh outputs/room/mesh/sugar_3dgs_inria_30k/refined_mesh/room/sugarfine_mesh.obj" in calls(workspace)
    assert "topology only" in result.stdout


def test_explicit_gt_overrides_scene_lidar(workspace):
    (workspace / "data" / "room" / "lidar.pcd").touch()
    (workspace / "scan.ply").touch()
    result = pipeline(workspace, "--mesh", "milo", "--gt", "scan.ply")
    assert result.returncode == 0, result.stderr
    assert "src/mesh/eval_mesh.py --mesh outputs/room/mesh/milo/mesh_cleaned_largest.ply --gt scan.ply" in calls(workspace)


def test_missing_gt_fails_before_sfm(workspace):
    result = pipeline(workspace, "--mesh", "milo", "--gt", "missing.pcd")
    assert result.returncode != 0
    assert calls(workspace) == []


def test_missing_mesh_output_fails(workspace):
    result = pipeline(workspace, "--mesh", "milo", MILO_SKIP_OUTPUT="1")
    assert result.returncode != 0
    assert not any(line.startswith("src/utils/collect_results.py") for line in calls(workspace))


def test_no_mesh_engine_skips_evaluation(workspace):
    result = pipeline(workspace, "--mesh", "none")
    assert result.returncode == 0, result.stderr
    assert not any(line.startswith("src/mesh/eval_mesh.py") for line in calls(workspace))


def test_a_failing_stage_stops_the_pipeline(workspace):
    (workspace / "scripts" / "01_sfm_hloc.sh").write_text('#!/bin/bash\necho "01_sfm_hloc.sh $*" >> calls.log\nexit 3\n')
    result = pipeline(workspace, "--mesh", "none")
    assert result.returncode == 3
    assert calls(workspace) == ["01_sfm_hloc.sh hloc data/room"]
