import sys

import pytest

from src.cli import main as cli

@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake_run(cmd, check, cwd):
        calls.append((cmd[2:], cwd))

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    return calls

def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["3drc.py", *argv])
    cli.main()

@pytest.mark.parametrize("argv, expected", [
    (["sfm", "--method", "fastmap", "--dataset", "data/room"], ["sfm", "fastmap", "data/room"]),
    (["train", "--method", "2dgs"], ["train", "2dgs"]),
    (["planargs", "--dataset", "data/room"], ["planargs", "data/room"]),
    (["scaffoldgs"], ["scaffoldgs"]),
    (["sugar", "--source-model", "3dgs/planargs"], ["sugar", "3dgs/planargs"]),
    (["tsdf", "--dataset", "data/room", "--source-model", "2dgs"], ["tsdf", "data/room", "2dgs"]),
    (["eval", "--mesh", "m.ply", "--gt", "gt.pcd"], ["eval", "m.ply", "gt.pcd"]),
    (["nvs", "--model", "2dgs"], ["nvs", "2dgs"]),
    (["results", "--publish", "docs/results.md"], ["results", "--publish", "docs/results.md"]),
    (["pipeline", "--dataset", "data/room"], ["pipeline", "hloc", "data/room"]),
    (["pipeline", "--train", "2dgs", "--mesh", "tsdf"], ["pipeline", "hloc", "--train", "2dgs", "--mesh", "tsdf"]),
    (["pipeline", "--mesh", "milo", "--gt", "gt.pcd"], ["pipeline", "hloc", "--mesh", "milo", "--gt", "gt.pcd"]),
])
def test_cli_maps_to_run_3drc_arguments(monkeypatch, captured, argv, expected):
    run_cli(monkeypatch, *argv)
    assert captured == [(expected, cli.PROJECT_ROOT)]

# The CLI keeps its own backend lists, so these tests fail when it drifts from configs/backends.txt
REGISTRY_ROWS = [line.split() for line in (cli.PROJECT_ROOT / "configs" / "backends.txt").read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
TRAINING_BACKENDS = [row[0] for row in REGISTRY_ROWS]
SHORTCUT_BACKENDS = [row[0] for row in REGISTRY_ROWS if row[3] == "yes"]

@pytest.mark.parametrize("backend", TRAINING_BACKENDS)
def test_cli_accepts_every_training_backend(monkeypatch, captured, backend):
    run_cli(monkeypatch, "train", "--method", backend)
    run_cli(monkeypatch, "pipeline", "--train", backend)
    assert [args for args, _ in captured] == [["train", backend], ["pipeline", "hloc", "--train", backend]]

@pytest.mark.parametrize("backend", SHORTCUT_BACKENDS)
def test_cli_has_a_shortcut_command_per_backend(monkeypatch, captured, backend):
    run_cli(monkeypatch, backend, "--dataset", "data/room")
    assert captured == [([backend, "data/room"], cli.PROJECT_ROOT)]

@pytest.mark.parametrize("argv", [["train", "--method", "nerf"], ["pipeline", "--train", "nerf"]])
def test_cli_rejects_an_unknown_training_backend(monkeypatch, captured, argv):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, *argv)
    assert captured == []
