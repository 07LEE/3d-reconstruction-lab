import struct

import numpy as np
import pytest

from src.mesh.alignment import (
    apply_transform,
    global_align_sim3,
    pca_initial_transforms,
    read_colmap_points3d,
    rotation_angle_deg,
    umeyama_sim3,
)

def random_sim3(rng, scale):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    transform = np.identity(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = rng.normal(scale=3.0, size=3)
    return transform

def box_surface(rng, lo, hi, n):
    """Samples n points on the faces of an axis-aligned box."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    pts = rng.uniform(lo, hi, size=(n, 3))
    axis = rng.integers(0, 3, size=n)
    side = rng.integers(0, 2, size=n)
    pts[np.arange(n), axis] = np.where(side == 1, hi[axis], lo[axis])
    return pts

def room_scene(rng, n=40000):
    """An asymmetric room: walls plus furniture blocks placed off-center."""
    room = box_surface(rng, [0, 0, 0], [6, 4, 2.7], n)
    sofa = box_surface(rng, [0.3, 0.3, 0], [2.3, 1.2, 0.8], n // 4)
    shelf = box_surface(rng, [5.2, 2.5, 0], [5.9, 3.9, 2.0], n // 4)
    return np.vstack([room, sofa, shelf])

def test_umeyama_recovers_exact_sim3():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(500, 3))
    true = random_sim3(rng, 2.5)
    est = umeyama_sim3(src, apply_transform(true, src))
    np.testing.assert_allclose(est, true, atol=1e-9)

def test_pca_hypotheses_are_proper_rotations():
    rng = np.random.default_rng(1)
    pts = room_scene(rng, 4000)
    hypotheses = pca_initial_transforms(pts, pts * 2.0)
    assert len(hypotheses) == 24
    for t in hypotheses:
        assert np.linalg.det(t[:3, :3]) > 0
        assert np.cbrt(np.linalg.det(t[:3, :3])) == pytest.approx(2.0, rel=0.05)

def test_global_align_recovers_arbitrary_sfm_frame():
    rng = np.random.default_rng(2)
    gt = room_scene(rng)
    # SfM frame: arbitrary rotation, scale 0.3, offset; sparse, noisy points with outliers
    gt_to_sfm = random_sim3(rng, 0.3)
    sfm = apply_transform(gt_to_sfm, gt[rng.choice(len(gt), 5000, replace=False)] + rng.normal(scale=0.01, size=(5000, 3)))
    sfm = np.vstack([sfm, rng.normal(scale=3.0, size=(150, 3))])

    result = global_align_sim3(sfm, gt, threshold=0.1)

    expected = np.linalg.inv(gt_to_sfm)
    assert result.scale == pytest.approx(1.0 / 0.3, rel=0.01)
    assert rotation_angle_deg(result.transform, expected) < 1.0
    probe = gt[:1000]
    residual = np.linalg.norm(apply_transform(result.transform, apply_transform(gt_to_sfm, probe)) - probe, axis=1)
    assert residual.max() < 0.05
    assert result.fitness > 0.9
    assert not result.ambiguous

def test_read_colmap_points3d_bin_and_txt(tmp_path):
    points = [(1, (0.0, 1.0, 2.0), 3), (2, (3.0, 4.0, 5.0), 2), (3, (6.0, 7.0, 8.0), 4)]

    with open(tmp_path / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pid, xyz, track in points:
            f.write(struct.pack("<QdddBBBdQ", pid, *xyz, 10, 20, 30, 0.5, track))
            for i in range(track):
                f.write(struct.pack("<ii", i, i))
    np.testing.assert_allclose(read_colmap_points3d(tmp_path), [[0, 1, 2], [6, 7, 8]])

    txt_dir = tmp_path / "txt"
    txt_dir.mkdir()
    lines = ["# 3D point list"]
    for pid, xyz, track in points:
        lines.append(" ".join([str(pid), *map(str, xyz), "10 20 30 0.5"] + [f"{i} {i}" for i in range(track)]))
    (txt_dir / "points3D.txt").write_text("\n".join(lines) + "\n")
    np.testing.assert_allclose(read_colmap_points3d(txt_dir / "points3D.txt", min_track_length=2), [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
