import pytest

from tests.cli.cli_workspace import build_workspace


@pytest.fixture
def workspace(tmp_path):
    return build_workspace(tmp_path)
