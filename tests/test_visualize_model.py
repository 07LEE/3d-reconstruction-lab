import numpy as np
import pytest

o3d = pytest.importorskip("open3d")

from src.sfm.sfm_report import CameraTrack, sort_track
from src.viz.visualize_model import camera_geometries


def line_track(names):
    n = len(names)
    return sort_track(CameraTrack(
        names=list(names),
        centers=np.stack([np.arange(n, dtype=float), np.zeros(n), np.zeros(n)], axis=1),
        forward=np.tile([0.0, 1.0, 0.0], (n, 1)),
        up=np.tile([0.0, 0.0, 1.0], (n, 1)),
        reproj_errors=np.ones(n),
        num_points=np.full(n, 100),
    ))


def test_trajectory_then_one_frustum_per_camera():
    geometries = camera_geometries(line_track([f"{i:03d}.jpg" for i in range(5)]), size=0.2)

    assert len(geometries) == 1 + 5
    assert np.asarray(geometries[0].lines).tolist() == [[0, 1], [1, 2], [2, 3], [3, 4]]
    for frustum in geometries[1:]:
        assert len(frustum.points) == 5 and len(frustum.lines) == 8


def test_frustum_points_along_viewing_direction():
    frustum = camera_geometries(line_track(["000.jpg", "001.jpg"]), size=0.2)[1]
    apex, *corners = np.asarray(frustum.points)
    np.testing.assert_allclose(np.mean(corners, axis=0) - apex, [0.0, 0.2, 0.0], atol=1e-9)


def test_trajectory_is_not_drawn_between_videos():
    geometries = camera_geometries(line_track(["a_000.jpg", "a_001.jpg", "b_000.jpg", "b_001.jpg"]), size=0.2)
    assert np.asarray(geometries[0].lines).tolist() == [[0, 1], [2, 3]]
