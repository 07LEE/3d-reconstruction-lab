import numpy as np
import pytest

from src.sfm.sfm_report import (
    CameraTrack,
    SparseModel,
    build_report,
    capture_order_key,
    image_group,
    plot_axes,
    plot_per_image,
    plot_trajectory,
    resolve_paths,
    sort_track,
    trajectory_stats,
)


def circle_track(names, radius=2.0, height=1.5):
    """Cameras walking a horizontal circle with world +z up, looking at the center."""
    n = len(names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n, height)], axis=1)
    forward = -centers * [1, 1, 0]
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    up = np.tile([0.0, 0.0, 1.0], (n, 1))
    return CameraTrack(
        names=list(names),
        centers=centers,
        forward=forward,
        up=up,
        reproj_errors=np.linspace(1.0, 2.0, n),
        num_points=np.arange(100, 100 + n),
    )


def sparse_model(track, n_points=500):
    rng = np.random.default_rng(0)
    return SparseModel(
        track=track,
        points=rng.uniform(-3, 3, size=(n_points, 3)),
        colors=np.zeros((n_points, 3), dtype=np.uint8),
        cameras=[{"camera_id": 1, "model": "PINHOLE", "width": 300, "height": 400, "params": [1.0, 1.0, 150.0, 200.0]}],
        mean_reproj_error=1.5,
        mean_track_length=4.0,
        mean_observations=120.0,
    )


def test_capture_order_is_numeric_not_lexical():
    names = ["frame_10.jpg", "frame_9.jpg", "frame_100.jpg"]
    assert sorted(names, key=capture_order_key) == ["frame_9.jpg", "frame_10.jpg", "frame_100.jpg"]


@pytest.mark.parametrize("name, group", [
    ("112540788259710.jpg", "single"),
    ("video1_0003.jpg", "video1"),
    ("cam_a_0003.jpg", "cam_a"),
])
def test_image_group_matches_sfm_info_prefix_rule(name, group):
    assert image_group(name) == group


def test_sort_track_reorders_every_field_together():
    names = [f"{i:03d}.jpg" for i in range(6)]
    track = circle_track(names)
    perm = [3, 0, 5, 1, 4, 2]
    shuffled = CameraTrack(
        names=[names[i] for i in perm],
        centers=track.centers[perm],
        forward=track.forward[perm],
        up=track.up[perm],
        reproj_errors=track.reproj_errors[perm],
        num_points=track.num_points[perm],
    )
    restored = sort_track(shuffled)
    assert restored.names == names
    np.testing.assert_allclose(restored.centers, track.centers)
    np.testing.assert_array_equal(restored.num_points, track.num_points)
    assert restored.groups == ["single"] * 6


def test_trajectory_stats_flags_a_pose_jump():
    track = sort_track(circle_track([f"{i:03d}.jpg" for i in range(40)]))
    track.centers[20:] += [10.0, 0.0, 0.0]
    stats = trajectory_stats(track)
    assert [(j["from"], j["to"]) for j in stats["jumps"]] == [("019.jpg", "020.jpg")]
    assert stats["jumps"][0]["ratio"] > 5


def test_trajectory_stats_does_not_link_separate_videos():
    a = circle_track([f"a_{i:03d}.jpg" for i in range(20)])
    b = circle_track([f"b_{i:03d}.jpg" for i in range(20)])
    b.centers += [50.0, 0.0, 0.0]
    track = sort_track(CameraTrack(
        names=a.names + b.names,
        centers=np.vstack([a.centers, b.centers]),
        forward=np.vstack([a.forward, b.forward]),
        up=np.vstack([a.up, b.up]),
        reproj_errors=np.concatenate([a.reproj_errors, b.reproj_errors]),
        num_points=np.concatenate([a.num_points, b.num_points]),
    ))
    stats = trajectory_stats(track)
    assert stats["groups"] == {"a": 20, "b": 20}
    assert stats["jumps"] == []


def test_plot_axes_use_camera_up_as_vertical():
    track = sort_track(circle_track([f"{i:03d}.jpg" for i in range(30)]))
    axes = plot_axes(track)
    np.testing.assert_allclose(axes[2], [0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(axes @ axes.T, np.eye(3), atol=1e-9)


def test_report_lists_unregistered_images_and_writes_plots(tmp_path):
    names = [f"{i:03d}.jpg" for i in range(30)]
    registered = [n for n in names if n not in ("010.jpg", "011.jpg")]
    model = sparse_model(sort_track(circle_track(registered)))
    stats = trajectory_stats(model.track)

    text = build_report(model, stats, names, "room (hloc)", {"method": "hloc", "strategy": "sequential"})
    assert "| Registered images | 28 / 30 (93.3%) |" in text
    assert "2: `010.jpg`, `011.jpg`" in text
    assert "| SfM strategy | sequential |" in text
    assert "PINHOLE 300x400" in text

    plot_trajectory(model, stats, tmp_path / "trajectory.png")
    plot_per_image(model, names, tmp_path / "per_image.png")
    assert (tmp_path / "trajectory.png").stat().st_size > 0
    assert (tmp_path / "per_image.png").stat().st_size > 0


def test_resolve_paths_from_dataset_and_model_dir(tmp_path):
    model = tmp_path / "scene" / "sparse" / "hloc"
    model.mkdir(parents=True)
    (model / "images.bin").touch()
    (tmp_path / "scene" / "raw_images").mkdir()

    assert resolve_paths(tmp_path / "scene", "hloc", None) == (model, tmp_path / "scene" / "raw_images")
    assert resolve_paths(model, None, None) == (model, tmp_path / "scene" / "raw_images")
    with pytest.raises(FileNotFoundError):
        resolve_paths(tmp_path / "scene", "fastmap", None)
