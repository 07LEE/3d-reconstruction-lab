import sys
import types

from src.sfm.sparse_outputs import build_sfm_info, largest_model


def fake_pycolmap(monkeypatch, counts):
    """pycolmap stand-in whose Reconstruction reads its registered image count from `counts` by path."""

    class Reconstruction:
        def __init__(self, path):
            if path not in counts:
                raise ValueError(f"no model at {path}")
            self.n = counts[path]

        def num_reg_images(self):
            return self.n

    monkeypatch.setitem(sys.modules, "pycolmap", types.SimpleNamespace(Reconstruction=Reconstruction))


def make_sfm_dir(tmp_path, submodels):
    sfm = tmp_path / "sfm"
    for name in submodels:
        (sfm / "models" / name).mkdir(parents=True)
    return sfm


def test_root_wins_when_largest(tmp_path, monkeypatch):
    sfm = make_sfm_dir(tmp_path, ["0", "1"])
    fake_pycolmap(monkeypatch, {str(sfm): 50, str(sfm / "models" / "0"): 50, str(sfm / "models" / "1"): 10})
    assert largest_model(sfm) == (sfm, 50)


def test_larger_submodel_is_chosen(tmp_path, monkeypatch):
    sfm = make_sfm_dir(tmp_path, ["0", "1"])
    fake_pycolmap(monkeypatch, {str(sfm): 20, str(sfm / "models" / "1"): 80})
    assert largest_model(sfm) == (sfm / "models" / "1", 80)


def test_no_loadable_model_returns_none(tmp_path, monkeypatch):
    sfm = make_sfm_dir(tmp_path, [])
    fake_pycolmap(monkeypatch, {})
    assert largest_model(sfm) is None


def test_sfm_info_counts_images_per_video(tmp_path, monkeypatch):
    monkeypatch.delenv("CAMERA_MODEL", raising=False)
    monkeypatch.setenv("CAMERA_MODE", "PER_FOLDER")
    for name in ["a_0001.jpg", "a_0002.JPG", "b_0001.png", "123.jpg", "notes.txt"]:
        (tmp_path / name).touch()
    info = build_sfm_info(tmp_path, "hloc", "sequential", "100", "auto")
    assert info["num_images_total"] == 4
    assert info["video_prefixes"] == {"single": 1, "a": 2, "b": 1}
    assert info["overlap"] == 100
    assert info["retrieval_k"] == "auto"
    assert info["camera_model"] == "SIMPLE_RADIAL"
    assert info["camera_mode"] == "PER_FOLDER"


def test_sfm_info_without_image_dir(tmp_path):
    info = build_sfm_info(tmp_path / "missing", "vggt", "", "", "")
    assert info["num_images_total"] == 0
    assert info["video_prefixes"] == {}
