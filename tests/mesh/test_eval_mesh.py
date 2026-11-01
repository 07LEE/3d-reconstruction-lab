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


def write_textured_obj(path, mesh):
    """OBJ with a separate texture coordinate per face corner, as SuGaR writes it."""
    lines = [f"v {x} {y} {z}" for x, y, z in mesh.vertices]
    lines += [f"vt {(i % 7) / 7} {(i % 5) / 5}" for i in range(3 * len(mesh.faces))]
    for f, (a, b, c) in enumerate(mesh.faces):
        lines.append(f"f {a + 1}/{3 * f + 1} {b + 1}/{3 * f + 2} {c + 1}/{3 * f + 3}")
    path.write_text("\n".join(lines) + "\n")


def test_textured_obj_is_measured_on_merged_vertices(tmp_path):
    path = tmp_path / "box.obj"
    write_textured_obj(path, trimesh.creation.box())

    result = compute_mesh_hygiene_and_topology(path)
    assert result["num_vertices"] == 8
    assert result["num_faces"] == 12
    assert result["is_watertight"]
    assert result["num_connected_components"] == 1
    assert result["euler_characteristic"] == 2


def test_separate_parts_are_counted_as_components(tmp_path):
    box = trimesh.creation.box()
    far = box.copy()
    far.apply_translation([5.0, 0.0, 0.0])
    path = tmp_path / "two.ply"
    trimesh.util.concatenate([box, far]).export(path)

    assert compute_mesh_hygiene_and_topology(path)["num_connected_components"] == 2
