import pytest

from src.cli import main as cli
from tests.cli.cli_workspace import BACKENDS, calls, run

SHORTCUTS = [name for name in BACKENDS if name != "3dgs"]


def test_registry_backends_match_the_expected_scripts_and_headers():
    backends = cli.load_backends()
    assert list(backends) == list(BACKENDS)
    for name, (script, header, model_subdir) in BACKENDS.items():
        assert (backends[name].script, backends[name].header, backends[name].model_subdir) == (script, header, model_subdir)


@pytest.mark.parametrize("backend", BACKENDS)
def test_train_runs_the_backend_script(workspace, backend):
    script, header, _ = BACKENDS[backend]
    result = run(workspace, "train", backend, "--dataset", "data/room")
    assert result.returncode == 0, result.stderr
    assert calls(workspace) == [f"{script} data/room"]
    assert header in result.stdout


def test_train_defaults_to_3dgs_on_the_config_scene(workspace):
    result = run(workspace, "train")
    assert result.returncode == 0, result.stderr
    assert calls(workspace) == ["02_train_3dgs.sh data/room"]


@pytest.mark.parametrize("command", SHORTCUTS)
def test_shortcut_command_runs_the_backend_script(workspace, command):
    script, header, _ = BACKENDS[command]
    result = run(workspace, command, "-d", "data/room")
    assert result.returncode == 0, result.stderr
    assert calls(workspace) == [f"{script} data/room"]
    assert header in result.stdout


@pytest.mark.parametrize("command", ["3dgs", "nonsense"])
def test_unknown_commands_are_rejected(workspace, command):
    result = run(workspace, command)
    assert result.returncode != 0
    assert calls(workspace) == []
    assert "invalid choice" in result.stderr


def test_train_rejects_an_unknown_backend(workspace):
    result = run(workspace, "train", "nerf", "--dataset", "data/room")
    assert result.returncode != 0
    assert calls(workspace) == []
    assert "Unknown training backend: 'nerf'" in result.stderr
    assert all(name in result.stderr for name in BACKENDS)


def test_a_missing_dataset_directory_is_rejected(workspace):
    result = run(workspace, "train", "--dataset", "data/missing")
    assert result.returncode != 0
    assert calls(workspace) == []
    assert "Dataset directory not found: 'data/missing'" in result.stderr


def test_a_dataset_given_as_a_positional_is_rejected(workspace):
    # The dataset is always --dataset; a path in the backend slot is not guessed to be one
    result = run(workspace, "train", "data/room")
    assert result.returncode != 0
    assert calls(workspace) == []


@pytest.mark.parametrize("argv, expected", [
    (["sfm"], "01_sfm_hloc.sh hloc data/room"),
    (["sfm", "fastmap", "--dataset", "data/room"], "01_sfm_hloc.sh fastmap data/room"),
    (["data"], "00b_data_yuv.sh data/room"),
    (["sugar"], "03_train_sugar.sh data/room"),
    (["sugar", "--source-model", "3dgs/planargs"], "03_train_sugar.sh data/room 3dgs/planargs"),
    (["tsdf", "--dataset", "data/room", "--source-model", "2dgs"], "03c_mesh_tsdf.sh data/room 2dgs"),
    (["mesh_tsdf"], "03c_mesh_tsdf.sh data/room"),
    (["milo"], "03b_train_milo.sh data/room"),
    (["grouping"], "04_train_grouping.sh data/room"),
    (["nvs", "--model", "2dgs"], "05_eval_nvs.sh data/room 2dgs"),
    (["view"], "utils/view_reconstruction.sh hloc data/room"),
    (["outputs"], "src/utils/inspect_outputs.py"),
])
def test_stage_receives_its_arguments_in_a_fixed_order(workspace, argv, expected):
    result = run(workspace, *argv)
    assert result.returncode == 0, result.stderr
    assert calls(workspace) == [expected]


def test_sfm_method_defaults_to_the_config(workspace):
    config = workspace / "configs" / "default_config.sh"
    config.write_text(config.read_text().replace('SFM_METHOD="hloc"', 'SFM_METHOD="vggt"'))
    assert run(workspace, "sfm").returncode == 0
    assert calls(workspace) == ["01_sfm_hloc.sh vggt data/room"]


@pytest.mark.parametrize("argv, expected", [
    (["eval", "--mesh", "m.ply", "--gt", "gt.pcd", "--align-ref", "mesh"], "src/mesh/eval_mesh.py --mesh m.ply --gt gt.pcd --align-ref mesh"),
    (["results", "--publish", "docs/results.md"], "src/utils/collect_results.py --publish docs/results.md"),
    (["results"], "src/utils/collect_results.py"),
])
def test_eval_and_results_forward_their_arguments(workspace, argv, expected):
    result = run(workspace, *argv)
    assert result.returncode == 0, result.stderr
    assert calls(workspace) == [expected]


@pytest.mark.parametrize("argv", [[], ["help"], ["--help"]])
def test_help_lists_the_commands(workspace, argv):
    result = run(workspace, *argv)
    assert result.returncode == 0
    assert calls(workspace) == []
    assert all(command in result.stdout for command in ("sfm", "train", "pipeline", *SHORTCUTS))


def test_registry_comments_and_blank_lines_are_ignored(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "backends.txt").write_text(
        "# comment\n\n   # indented comment\nalpha  01_a.sh  a  yes  Alpha_One  va  repo_a\n\nbeta  02b_b.sh  b  no  Beta  vb  repo_b\n"
    )
    backends = cli.load_backends(tmp_path)
    assert list(backends) == ["alpha", "beta"]
    assert [b.header for b in backends.values()] == ["Step 1: Alpha One Training", "Step 2b: Beta Training"]
    assert [b.shortcut for b in backends.values()] == [True, False]


def test_only_the_default_backend_lacks_a_shortcut():
    assert [b.name for b in cli.load_backends().values() if not b.shortcut] == ["3dgs"]
