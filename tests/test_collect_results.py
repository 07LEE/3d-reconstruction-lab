import csv
import json
import struct
from pathlib import Path

from src.utils.collect_results import collect_results, write_reports

def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))

def write_ply(path: Path, vertices: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {vertices}\nproperty float x\nend_header\n"
    path.write_bytes(header.encode("ascii") + b"\x00" * (4 * vertices))

def build_workspace(root: Path) -> None:
    scene = "room"

    # Step 1: two SfM models, hloc active
    sparse = root / "data" / scene / "sparse"
    for method, reg, pts in (("hloc", 95, 40000), ("fastmap", 80, 30000)):
        (sparse / method).mkdir(parents=True)
        (sparse / method / "images.bin").write_bytes(struct.pack("<Q", reg))
        (sparse / method / "points3D.bin").write_bytes(struct.pack("<Q", pts))
        write_json(sparse / method / "sfm_info.json", {"method": method, "num_images_total": 100})
    (sparse / "0").symlink_to("hloc")

    # Step 2: one holdout model with metrics, one full-reconstruction model with stale metrics
    gs = root / "outputs" / scene / "3dgs" / "inria_30k"
    write_json(gs / "pipeline_meta.json", {"stage": "Stage 2 (Training)", "engine": "Original 3DGS (inria_30k)", "active_sfm": "hloc", "iterations": 30000})
    (gs / "cfg_args").write_text("Namespace(eval=True, resolution=4)")
    write_json(gs / "results.json", {"ours_7000": {"PSNR": 20.0, "SSIM": 0.7, "LPIPS": 0.4}, "ours_30000": {"PSNR": 27.5, "SSIM": 0.85, "LPIPS": 0.2}})
    write_ply(gs / "point_cloud" / "iteration_7000" / "point_cloud.ply", 10)
    write_ply(gs / "point_cloud" / "iteration_30000" / "point_cloud.ply", 25)

    tdgs = root / "outputs" / scene / "2dgs"
    write_json(tdgs / "pipeline_meta.json", {"stage": "Stage 2 (Training)", "engine": "2D Gaussian Splatting (Surfels)", "active_sfm": "hloc", "iterations": 30000})
    (tdgs / "cfg_args").write_text("Namespace(eval=False)")
    write_json(tdgs / "results.json", {"ours_30000": {"PSNR": 99.0, "SSIM": 1.0, "LPIPS": 0.0}})

    # Step 3/5: evaluated mesh and an unevaluated one
    mesh = root / "outputs" / scene / "mesh" / "2dgs"
    write_json(mesh / "pipeline_meta.json", {"stage": "Stage 3 (Mesh Extraction)", "engine": "Open3D TSDF Voxel Integration (2DGS)"})
    write_json(mesh / "eval" / "tsdf_mesh" / "eval_results.json", {
        "mesh_path": "outputs/room/mesh/2dgs/tsdf_mesh.ply",
        "hygiene": {"num_vertices": 1000, "num_faces": 2000, "is_watertight": False, "sliver_ratio_pct": 1.5},
        "accuracy": {"chamfer_l1_m": 0.0123, "hausdorff_95_m": 0.05, "gt_coverage_pct": 91.2},
    })
    sugar = root / "outputs" / scene / "mesh" / "sugar"
    write_json(sugar / "pipeline_meta.json", {"stage": "Stage 3 (Mesh Extraction)", "engine": "SuGaR", "active_sfm": "hloc"})
    write_json(sugar / "eval" / "sugarfine_mesh" / "eval_results.json", {
        "mesh_path": "outputs/room/mesh/sugar/sugarfine_mesh.obj",
        "metrics_version": 2,
        "hygiene": {"num_vertices": 500, "num_faces": 900, "is_watertight": True, "sliver_ratio_pct": 0.5},
        "accuracy": {"chamfer_l1_m": 0.0345, "hausdorff_95_m": 0.08, "gt_coverage_pct": 88.0,
                     "fscore_threshold_m": 0.05, "precision": 0.8, "recall": 0.7, "fscore": 0.7467},
        "alignment": {"source": "reference", "fitness": 0.912, "ambiguous": True},
    })
    write_json(root / "outputs" / scene / "mesh" / "milo" / "pipeline_meta.json", {"stage": "Stage 3 (Mesh Reconstruction)", "engine": "MILo (Mesh-in-the-Loop SDF)"})

    env = {"gpu": "NVIDIA GeForce RTX 5070 Ti", "driver": "580.0", "torch": "2.11.0+cu128", "conda_env": "gs_train", "git_commit": "abc1234", "git_dirty": False}
    write_json(root / "outputs" / "benchmark_timings.json", [
        {"stage": "3dgs_inria_train", "output_dir": "outputs/room/3dgs/inria_30k", "elapsed_seconds": 100.0, "metadata": {}, "env": env},
        {"stage": "3dgs_inria_train", "output_dir": "outputs/room/3dgs/inria_30k", "elapsed_seconds": 3725.0, "metadata": {"resumed_from": "chkpnt20000.pth"}, "env": env},
        {"stage": "nvs_eval_3dgs", "output_dir": "outputs/room/3dgs/inria_30k", "elapsed_seconds": 61.0, "metadata": {}, "env": env},
        {"stage": "sfm_hloc", "output_dir": "data/room/sparse/hloc", "elapsed_seconds": 600.0, "metadata": {}, "env": env},
    ])

def test_collect_results_reads_all_sources(tmp_path):
    build_workspace(tmp_path)
    sfm, runs = collect_results(tmp_path)

    hloc = next(r for r in sfm["room"] if r.method == "hloc")
    assert (hloc.registered_images, hloc.total_images, hloc.points, hloc.active) == (95, 100, 40000, True)
    assert hloc.timing["elapsed_seconds"] == 600.0

    by_dir = {r.rel_dir: r for r in runs["room"]}
    gs = by_dir["outputs/room/3dgs/inria_30k"]
    assert gs.eval_split is True
    assert gs.nvs == {"iteration": 30000, "PSNR": 27.5, "SSIM": 0.85, "LPIPS": 0.2}
    assert gs.primitives == 25
    assert gs.timing["elapsed_seconds"] == 3725.0  # latest stage entry wins
    assert gs.nvs_timing["elapsed_seconds"] == 61.0

    assert by_dir["outputs/room/2dgs"].eval_split is False
    assert by_dir["outputs/room/mesh/2dgs"].mesh_evals[0]["mesh"] == "tsdf_mesh.ply"

def test_write_reports_hides_metrics_without_holdout(tmp_path):
    build_workspace(tmp_path)
    publish = tmp_path / "docs" / "results.md"
    md_path, csv_path = write_reports(tmp_path, publish)

    markdown = md_path.read_text()
    assert publish.read_text() == markdown
    assert "| 27.50 | 0.8500 | 0.2000 |" in markdown
    assert "99.00" not in markdown  # training-view fit from a no-holdout model is not reported
    assert "1:02:05 (resumed)" in markdown
    assert "RTX 5070 Ti" in markdown
    assert "(not evaluated)" in markdown
    assert "| 0.0123 |" in markdown
    assert "| 0.7467 @ 0.05 m |" in markdown
    assert "| 0.912 (ambiguous) |" in markdown
    assert "(v1)" in markdown  # the legacy tsdf_mesh eval is flagged as not comparable

    rows = list(csv.DictReader(csv_path.open()))
    steps = sorted(r["step"] for r in rows)
    assert steps == ["mesh", "mesh", "mesh", "sfm", "sfm", "train", "train"]
    sugar_row = next(r for r in rows if r["output_dir"] == "outputs/room/mesh/sugar")
    assert (sugar_row["fscore"], sugar_row["align_fitness"], sugar_row["metrics_version"]) == ("0.7467", "0.912", "2")
    no_holdout = next(r for r in rows if r["output_dir"] == "outputs/room/2dgs")
    assert no_holdout["psnr"] == ""

def test_empty_workspace(tmp_path):
    md_path, _ = write_reports(tmp_path)
    assert "No results found" in md_path.read_text()
