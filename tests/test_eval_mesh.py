import pytest

pytest.importorskip("open3d")
trimesh = pytest.importorskip("trimesh")

from src.mesh.eval_mesh import compute_mesh_hygiene_and_topology


def test_single_box_has_no_self_intersections(tmp_path):
    path = tmp_path / "box.ply"
    trimesh.creation.box().export(path)

    result = compute_mesh_hygiene_and_topology(path, check_self_intersect=True)
    assert result["num_faces"] == 12
    assert result["self_intersecting_faces"] == 0


def test_overlapping_boxes_report_self_intersecting_faces(tmp_path):
    box = trimesh.creation.box()
    shifted = box.copy()
    shifted.apply_translation([0.5, 0.3, 0.2])
    path = tmp_path / "overlap.ply"
    trimesh.util.concatenate([box, shifted]).export(path)

    result = compute_mesh_hygiene_and_topology(path, check_self_intersect=True)
    assert result["self_intersecting_faces"] > 0


def test_self_intersection_check_is_off_by_default(tmp_path):
    path = tmp_path / "box.ply"
    trimesh.creation.box().export(path)

    assert compute_mesh_hygiene_and_topology(path)["self_intersecting_faces"] == -1
