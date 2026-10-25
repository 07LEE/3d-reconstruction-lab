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
])
def test_cli_maps_to_run_3drc_arguments(monkeypatch, captured, argv, expected):
    run_cli(monkeypatch, *argv)
    assert captured == [(expected, cli.PROJECT_ROOT)]
