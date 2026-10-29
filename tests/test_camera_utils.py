import json

import pytest

pytest.importorskip("PIL")
from PIL import Image

from src.sfm.camera_utils import parse_camera_prior_from_dataset

SESSION = {"intrinsics": [1648.25, 1650.4, 2033.89, 1538.02, 0], "distortion": [0.0158913, -0.00116444, -0.00231631, 0, 0]}


def make_scene(tmp_path, orientation, undistorted, size):
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", size).save(frames / "0.jpg")
    (tmp_path / "session.json").write_text(json.dumps(SESSION))
    (frames / "extraction_info.json").write_text(json.dumps({"sensor_orientation": orientation, "undistorted": undistorted}))
    return frames


def test_undistorted_rotated_frames_get_rotated_pinhole(tmp_path):
    prior = parse_camera_prior_from_dataset(make_scene(tmp_path, 90, True, (3060, 4080)))
    assert prior["camera_model"] == "PINHOLE"
    fx, fy, cx, cy = (float(v) for v in prior["camera_params"].split(","))
    assert (fx, fy) == pytest.approx((1650.4, 1648.25))
    assert (cx, cy) == pytest.approx((3060 - 1538.02, 2033.89))


def test_raw_rotated_frames_keep_distortion(tmp_path):
    prior = parse_camera_prior_from_dataset(make_scene(tmp_path, 90, False, (3060, 4080)))
    assert prior["camera_model"] == "OPENCV"
    params = [float(v) for v in prior["camera_params"].split(",")]
    assert params[:4] == pytest.approx([1650.4, 1648.25, 3060 - 1538.02, 2033.89])
    assert params[4:6] == pytest.approx([0.015891, -0.001164])


def test_180_rotation_flips_principal_point(tmp_path):
    prior = parse_camera_prior_from_dataset(make_scene(tmp_path, 180, False, (4080, 3060)))
    params = [float(v) for v in prior["camera_params"].split(",")]
    assert params[2:4] == pytest.approx([4080 - 2033.89, 3060 - 1538.02])


def test_dataset_without_extraction_info_is_unchanged(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    (tmp_path / "session.json").write_text(json.dumps(SESSION))
    prior = parse_camera_prior_from_dataset(frames)
    assert prior["camera_model"] == "OPENCV"
    assert prior["camera_params"].startswith("1648.250000,1650.400000,2033.890000,1538.020000")
