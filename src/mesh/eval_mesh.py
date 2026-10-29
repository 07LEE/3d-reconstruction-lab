"""3D Mesh Evaluation and Ground Truth Comparison Module.

Computes 3D Mesh Topology Hygiene, Geometry Complexity, Sliver Triangles,
and Sim(3) 7-DoF Aligned Chamfer / Hausdorff / F-score Accuracy against LiDAR GT PointClouds.

The Sim(3) transform is estimated once per scene from the SfM sparse points the meshes were
built from (see src/mesh/alignment.py), so all meshes of a scene are scored in the same frame.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import open3d as o3d
import trimesh

if __package__ in (None, ""):
    # Allow `python src/mesh/eval_mesh.py` in addition to `python -m src.mesh.eval_mesh`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mesh.alignment import global_align_sim3, read_colmap_points3d
from src.mesh.metrics import METRICS_VERSION, surface_distance_metrics

def compute_mesh_hygiene_and_topology(mesh_path: str | Path, check_self_intersect: bool = False) -> Dict[str, Any]:
    """Computes mesh topology hygiene and triangle quality metrics via Trimesh.

    Args:
        mesh_path: Path to the input 3D mesh (.obj or .ply).
        check_self_intersect: If True, executes expensive triangle-triangle self-intersection check.

    Returns:
        Dict containing topology, manifold, sliver, and watertight metrics.
    """
    path_obj = Path(mesh_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Input mesh not found: {path_obj}")

    print(f"[Eval] Loading mesh for hygiene analysis: {path_obj}")
    tm = trimesh.load(str(path_obj), process=False)

    if isinstance(tm, trimesh.Scene):
        if len(tm.geometry) > 0:
            valid_geoms = [g for g in tm.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if valid_geoms:
                tm = trimesh.util.concatenate(valid_geoms)
            else:
                raise ValueError(f"No valid Trimesh polygon geometries found in scene: {path_obj}")
        else:
            raise ValueError(f"Empty Trimesh scene: {path_obj}")

    if isinstance(tm, trimesh.PointCloud):
        return {
            "num_vertices": len(tm.vertices),
            "num_faces": 0,
            "is_watertight": False,
            "is_manifold": False,
            "num_connected_components": 0,
            "euler_characteristic": len(tm.vertices),
            "degenerate_faces": 0,
            "sliver_ratio_pct": 0.0,
            "self_intersecting_faces": 0,
        }

    num_vertices = len(tm.vertices)
    num_faces = len(tm.faces)
    is_watertight = bool(tm.is_watertight)
    is_manifold = bool(tm.is_winding_consistent)

    # Connected components and Euler characteristic
    try:
        num_connected_components = len(tm.split(only_watertight=False))
    except Exception:
        num_connected_components = -1

    if hasattr(tm, "edges_unique"):
        euler_characteristic = int(num_vertices - len(tm.edges_unique) + num_faces)
    else:
        euler_characteristic = int(num_vertices - (num_faces // 2))

    # Degenerate faces (zero area)
    face_areas = tm.area_faces
    degenerate_faces = int(np.sum(face_areas <= 1e-12)) if len(face_areas) > 0 else 0

    # Sliver triangles (min angle < 5 deg)
    angles = tm.face_angles
    if len(angles) > 0:
        min_angles_deg = np.rad2deg(np.min(angles, axis=1))
        sliver_ratio = float(np.mean(min_angles_deg < 5.0))
    else:
        sliver_ratio = 0.0

    # Optional self-intersection inspection
    self_intersecting_faces = -1
    if check_self_intersect:
        print("[Eval] Checking self-intersecting faces (may take extra time)...")
        try:
            self_intersecting_faces = len(trimesh.invalid.inspect_self_intersections(tm))
        except Exception as e:
            print(f"[Warn] Self-intersection check failed: {e}")
            self_intersecting_faces = -1

    return {
        "num_vertices": num_vertices,
        "num_faces": num_faces,
        "is_watertight": is_watertight,
        "is_manifold": is_manifold,
        "num_connected_components": num_connected_components,
        "euler_characteristic": euler_characteristic,
        "degenerate_faces": degenerate_faces,
        "sliver_ratio_pct": round(sliver_ratio * 100.0, 3),
        "self_intersecting_faces": self_intersecting_faces,
    }


def load_point_set(path: str | Path, n_samples: int) -> np.ndarray:
    """Loads a point cloud (.pcd/.ply), COLMAP sparse model, or mesh (sampled) as an Nx3 array."""
    path = Path(path)
    if path.is_dir() or path.name.startswith("points3D."):
        return read_colmap_points3d(path)
    # A .ply may hold a mesh; reading it as a point cloud would return only its vertices
    if path.suffix.lower() != ".pcd":
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles) > 0:
            return np.asarray(mesh.sample_points_uniformly(number_of_points=n_samples).points)
    return np.asarray(o3d.io.read_point_cloud(str(path)).points)

def file_fingerprint(path: Path) -> str:
    if path.is_dir():
        path = next((path / name for name in ("points3D.bin", "points3D.txt") if (path / name).is_file()), path)
    stat = path.stat()
    return f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"

def resolve_scene_alignment_reference(mesh_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Finds the SfM sparse model a mesh under outputs/<scene>/ was built from.

    Reads `active_sfm` and `source_dataset` from the nearest pipeline_meta.json and falls back
    to data/<scene>/sparse/0. Returns (sparse model dir, scene-level alignment cache dir).
    """
    parts = mesh_path.parts
    if "outputs" not in parts:
        return None, None
    idx = len(parts) - 1 - parts[::-1].index("outputs")
    if idx + 1 >= len(parts) - 1:
        return None, None
    root = Path(*parts[:idx])
    scene_dir = root / "outputs" / parts[idx + 1]

    meta: Dict[str, Any] = {}
    for directory in [mesh_path.parent, *mesh_path.parent.parents]:
        meta_file = directory / "pipeline_meta.json"
        if meta_file.is_file():
            try:
                meta = json.loads(meta_file.read_text())
            except (OSError, ValueError):
                meta = {}
            break
        if directory == scene_dir:
            break

    dataset = Path(meta["source_dataset"]) if meta.get("source_dataset") else Path("data") / scene_dir.name
    dataset_dir = dataset if dataset.is_absolute() else root / dataset
    candidates = []
    if meta.get("active_sfm") not in (None, "", "unknown"):
        candidates.append(dataset_dir / "sparse" / str(meta["active_sfm"]))
    candidates.append(dataset_dir / "sparse" / "0")

    for sparse_dir in candidates:
        if (sparse_dir / "points3D.bin").is_file() or (sparse_dir / "points3D.txt").is_file():
            return sparse_dir, scene_dir / "eval" / "alignment"
    return None, scene_dir / "eval" / "alignment"

def estimate_alignment(src_points: np.ndarray, gt_points: np.ndarray, threshold: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    print(f"[Eval] Computing 7-DoF Sim(3) registration ({len(src_points):,} source points, threshold {threshold} m)...")
    result = global_align_sim3(src_points, gt_points, threshold=threshold)
    info = {
        "scale": round(result.scale, 6),
        "fitness": round(result.fitness, 4),
        "inlier_rmse_m": round(result.inlier_rmse, 5),
        "threshold_m": threshold,
        "ambiguous": result.ambiguous,
    }
    if result.ambiguous:
        print("[Warn] Alignment is ambiguous: another rotation fits nearly as well (symmetric scene?). Verify it visually.")
    return result.transform, info

def resolve_transform(args: argparse.Namespace, mesh_file: Path, out_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads or computes the mesh-to-GT Sim(3) transform.

    Order: explicit --transform file, then the scene-level transform estimated from the SfM
    sparse points (shared by every mesh of the scene and cached), then the mesh itself.
    """
    if args.transform and Path(args.transform).is_file() and not args.compute_transform:
        print(f"[Eval] Loading precomputed Sim(3) transform matrix from {args.transform}")
        data = json.loads(Path(args.transform).read_text())
        info = {k: v for k, v in data.items() if k != "transform_matrix"}
        return np.array(data["transform_matrix"]), {**info, "transform_path": args.transform}

    gt_path = Path(args.gt)
    ref_path: Optional[Path] = None
    cache_file: Optional[Path] = Path(args.transform) if args.transform else None
    if args.align_ref == "auto":
        ref_path, cache_dir = resolve_scene_alignment_reference(Path(args.mesh).absolute())
        if ref_path is None:
            print("[Warn] No SfM sparse model found for this mesh; aligning the mesh itself to GT. "
                  "Meshes aligned this way are not in a shared frame. Pass --align-ref to fix.")
        elif cache_file is None:
            cache_file = cache_dir / f"sim3_{ref_path.resolve().name}_{gt_path.stem}.json"
    elif args.align_ref != "mesh":
        ref_path = Path(args.align_ref)

    fingerprint = {"gt_fingerprint": file_fingerprint(gt_path)}
    if ref_path is not None:
        fingerprint["ref_fingerprint"] = file_fingerprint(ref_path)

    if cache_file is not None and cache_file.is_file() and not args.compute_transform:
        data = json.loads(cache_file.read_text())
        if all(data.get(k) == v for k, v in fingerprint.items()):
            print(f"[Eval] Reusing scene-level Sim(3) transform from {cache_file}")
            info = {k: v for k, v in data.items() if k != "transform_matrix"}
            return np.array(data["transform_matrix"]), {**info, "transform_path": str(cache_file)}
        print(f"[Eval] Cached transform {cache_file} is stale (reference or GT changed); recomputing.")

    gt_points = load_point_set(gt_path, args.n_samples)
    if ref_path is not None:
        print(f"[Eval] Aligning SfM reference {ref_path} to GT")
        src_points = load_point_set(ref_path, args.n_samples)
        source = "reference"
    else:
        src_points = load_point_set(mesh_file, 200000)
        source = "mesh"
    transform, info = estimate_alignment(src_points, gt_points, args.align_threshold)
    info = {"source": source, "ref_path": str(ref_path) if ref_path else str(mesh_file), "gt_path": str(gt_path), **info, **fingerprint}

    save_path = cache_file if cache_file is not None else out_dir / "sim3_transform.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps({"transform_matrix": transform.tolist(), **info}, indent=2))
    print(f"[Eval] Saved Sim(3) transform to {save_path}")
    return transform, {**info, "transform_path": str(save_path)}

def evaluate_gt_accuracy(mesh_path: str | Path, gt_path: str | Path, transform_matrix: np.ndarray,
                         n_samples: int = 2000000, truncation: float = 0.5,
                         fscore_thresholds: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.1),
                         fscore_threshold: float = 0.05) -> Dict[str, Any]:
    """Evaluates accuracy, completeness, Chamfer, HD95, and F-score against Ground Truth.

    Args:
        mesh_path: Path to reconstructed mesh file.
        gt_path: Path to Ground Truth PCD or mesh file.
        transform_matrix: 4x4 Sim(3) transform from the mesh frame to the GT frame.
        n_samples: Number of surface sample points (default: 2M).
        truncation: Distance clip and GT coverage radius in meters.
        fscore_thresholds: Thresholds in meters for the F-score curve.
        fscore_threshold: Primary F-score threshold in meters.
    """
    path_mesh = Path(mesh_path)
    print(f"[Eval] Loading Open3D Mesh: {path_mesh}")
    mesh = o3d.io.read_triangle_mesh(str(path_mesh))

    print(f"[Eval] Uniformly sampling {n_samples:,} surface points from mesh...")
    mesh_pcd = mesh.sample_points_uniformly(number_of_points=n_samples)
    mesh_pcd.transform(transform_matrix)

    print(f"[Eval] Loading Ground Truth: {gt_path}")
    gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(load_point_set(gt_path, n_samples)))

    dists_mesh_to_gt = np.asarray(mesh_pcd.compute_point_cloud_distance(gt_pcd))
    dists_gt_to_mesh = np.asarray(gt_pcd.compute_point_cloud_distance(mesh_pcd))
    return surface_distance_metrics(dists_mesh_to_gt, dists_gt_to_mesh, truncation, fscore_thresholds, fscore_threshold)

def parse_thresholds(value: str) -> Tuple[float, ...]:
    return tuple(float(v) for v in value.split(",") if v.strip())

def main() -> None:
    parser = argparse.ArgumentParser(description="3D Mesh Topology Hygiene & Sim(3) GT Evaluation Tool")
    parser.add_argument("--mesh", type=str, required=True, help="Path to input .obj or .ply mesh file")
    parser.add_argument("--gt", type=str, default=None, help="Path to Ground Truth LiDAR PCD or Mesh file")
    parser.add_argument("--transform", type=str, default=None, help="Sim(3) transform JSON: loaded if it exists, otherwise computed and saved there")
    parser.add_argument("--compute-transform", action="store_true", help="Force recomputing the Sim(3) transform even if one is cached")
    parser.add_argument("--align-ref", type=str, default="auto",
                        help="Alignment source: 'auto' (scene SfM sparse points, shared by all meshes), 'mesh' (the mesh itself), "
                             "or a path to a COLMAP sparse model / point cloud in the mesh frame (default: auto)")
    parser.add_argument("--align-threshold", type=float, default=0.1, help="Final ICP correspondence threshold in meters (default: 0.1m)")
    parser.add_argument("--n-samples", type=int, default=2000000, help="Number of uniform surface sample points (default: 2M)")
    parser.add_argument("--coverage-radius", type=float, default=0.5, help="Distance clip and GT coverage radius in meters (default: 0.5m)")
    parser.add_argument("--fscore-thresholds", type=parse_thresholds, default=(0.01, 0.02, 0.05, 0.1), help="F-score curve thresholds in meters (default: 0.01,0.02,0.05,0.1)")
    parser.add_argument("--fscore-threshold", type=float, default=0.05, help="Primary F-score threshold in meters (default: 0.05m)")
    parser.add_argument("--check-self-intersect", action="store_true", help="Enable computationally expensive self-intersection check")
    parser.add_argument("--out", type=str, default=None, help="Output directory for eval results (default: <mesh_dir>/eval/<mesh_stem>)")

    args = parser.parse_args()

    mesh_file = Path(args.mesh).resolve()
    out_dir = Path(args.out) if args.out else mesh_file.parent / "eval" / mesh_file.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print(" 3DRC 3D Mesh & Geometry Evaluation Pipeline")
    print("==================================================")

    # 1. Topology Hygiene Analysis
    hygiene_res = compute_mesh_hygiene_and_topology(args.mesh, check_self_intersect=args.check_self_intersect)

    # 2. Sim(3) GT Accuracy Analysis
    gt_res = None
    alignment = None
    if args.gt:
        transform_matrix, alignment = resolve_transform(args, mesh_file, out_dir)
        gt_res = evaluate_gt_accuracy(
            args.mesh, args.gt, transform_matrix, n_samples=args.n_samples, truncation=args.coverage_radius,
            fscore_thresholds=args.fscore_thresholds, fscore_threshold=args.fscore_threshold,
        )

    # Combine Results
    results = {
        "mesh_path": args.mesh,
        "gt_path": args.gt,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics_version": METRICS_VERSION,
        "hygiene": hygiene_res,
        "accuracy": gt_res,
        "alignment": alignment,
    }

    # Save Results
    json_path = out_dir / "eval_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n==================================================")
    print(" Evaluation Summary Table")
    print("==================================================")
    print(f"Mesh: {Path(args.mesh).name}")
    print(f"Vertices: {hygiene_res['num_vertices']:,} | Faces: {hygiene_res['num_faces']:,}")
    print(f"Watertight: {hygiene_res['is_watertight']} | Manifold: {hygiene_res['is_manifold']}")
    print(f"Connected Components: {hygiene_res.get('num_connected_components', -1):,} | Euler Characteristic: {hygiene_res.get('euler_characteristic', 0):,}")
    print(f"Degenerate Faces: {hygiene_res['degenerate_faces']:,} | Sliver Ratio: {hygiene_res['sliver_ratio_pct']}%")
    if hygiene_res['self_intersecting_faces'] >= 0:
        print(f"Self-Intersecting Faces: {hygiene_res['self_intersecting_faces']:,}")

    if gt_res:
        print("--------------------------------------------------")
        print(f"Alignment: {alignment.get('source', 'file')} | Fitness: {alignment.get('fitness', '-')} "
              f"| Inlier RMSE: {alignment.get('inlier_rmse_m', '-')} m | Ambiguous: {alignment.get('ambiguous', '-')}")
        print(f"Sim(3) Aligned Chamfer L1: {gt_res['chamfer_l1_m']} m (distances clipped at {gt_res['truncation_m']} m)")
        print(f"  - Accuracy (Mesh -> GT):   {gt_res['accuracy_mean_m']} m (RMSE: {gt_res['accuracy_rmse_m']} m)")
        print(f"  - Completeness (GT -> Mesh): {gt_res['completeness_mean_m']} m (RMSE: {gt_res['completeness_rmse_m']} m)")
        print(f"F-score @ {gt_res['fscore_threshold_m']} m: {gt_res['fscore']} (Precision: {gt_res['precision']} | Recall: {gt_res['recall']})")
        print(f"Hausdorff 95%: {gt_res['hausdorff_95_m']} m | GT Coverage: {gt_res['gt_coverage_pct']}%")

    print("==================================================")
    print(f"Full results saved to {json_path}")

if __name__ == "__main__":
    main()
