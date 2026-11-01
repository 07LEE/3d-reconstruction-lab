import pytest

pytest.importorskip("torch")

from src.data import yuv_pipeline


class FakeLoader:
    """Stands in for yuv_sensor.SessionDataLoader, which keeps session.json in session_config."""

    def __init__(self, session_dir):
        self.frames_df = [0, 1, 2]
        self.session_config = {
            "intrinsics": [1648.25, 1650.4, 2033.89, 1538.02, 0],
            "distortion": [0.0158913, -0.00116444, -0.00231631, 0, 0],
            "sensor_orientation": 90,
            "device": "samsung SM-S938N",
        }


def test_camera_intrinsics_come_from_session_config(monkeypatch, tmp_path):
    monkeypatch.setattr(yuv_pipeline, "SessionDataLoader", FakeLoader)
    dataset = yuv_pipeline.YUVInMemoryDataset(tmp_path)

    assert len(dataset) == 3
    assert dataset.get_camera_intrinsics() == {
        "intrinsics": [1648.25, 1650.4, 2033.89, 1538.02, 0],
        "distortion": [0.0158913, -0.00116444, -0.00231631, 0, 0],
        "sensor_orientation": 90,
        "device": "samsung SM-S938N",
    }


def test_missing_yuv_sensor_raises_import_error(monkeypatch, tmp_path):
    monkeypatch.setattr(yuv_pipeline, "SessionDataLoader", None)
    with pytest.raises(ImportError, match="yuv_sensor"):
        yuv_pipeline.YUVInMemoryDataset(tmp_path)
