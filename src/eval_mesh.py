"""3D Mesh Evaluation and Ground Truth Comparison Module.

Computes 3D Mesh Topology Hygiene, Geometry Complexity, Sliver Triangles,
and Sim(3) 7-DoF Aligned Chamfer/Hausdorff Accuracy against LiDAR GT PointClouds.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh

def compute_mesh_hygiene_and_topology(mesh_path: str, check_self_intersect: bool = False):
    """Computes mesh topology hygiene and triangle quality metrics via Trimesh."""
    print(f"[Eval] Loading mesh for hygiene analysis: {mesh_path}")
    tm = trimesh.load(mesh_path, process=False)
    
    if isinstance(tm, trimesh.Scene):
        if len(tm.geometry) > 0:
            tm = trimesh.util.concatenate([g for g in tm.geometry.values() if isinstance(g, trimesh.Trimesh)])
        else:
            raise ValueError(f"No valid Trimesh geometry found in scene: {mesh_path}")

    if isinstance(tm, trimesh.PointCloud):
        return {
            "num_vertices": len(tm.vertices),
            "num_faces": 0,
            "is_watertight": False,
            "is_manifold": False,
            "degenerate_faces": 0,
            "sliver_ratio_pct": 0.0,
            "self_intersecting_faces": 0
        }

    num_vertices = len(tm.vertices)
    num_faces = len(tm.faces)
    is_watertight = bool(tm.is_watertight)
    is_manifold = bool(tm.is_winding_consistent)

    # Connected components and boundary edges
    num_connected_components = len(tm.split(only_watertight=False))
    num_boundary_edges = len(tm.edges_unique) - len(tm.edges_unique_length) if hasattr(tm, 'edges_unique_length') else len(tm.outline().entities) if hasattr(tm, 'outline') else -1
    euler_characteristic = num_vertices - len(tm.edges_unique) + num_faces if hasattr(tm, 'edges_unique') else num_vertices - (num_faces // 2)

    # Degenerate faces (zero area)
    face_areas = tm.area_faces
    degenerate_faces = int(np.sum(face_areas <= 1e-12))

    # Sliver triangles (min angle < 5 deg or high aspect ratio)
    angles = tm.face_angles
    if len(angles) > 0:
        min_angles_deg = np.rad2deg(np.min(angles, axis=1))
        sliver_ratio = float(np.mean(min_angles_deg < 5.0))
    else:
        sliver_ratio = 0.0

    # Self-intersection check (optional due to compute cost)
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
        "self_intersecting_faces": self_intersecting_faces
    }

def align_sim3_point_to_point(source_pcd, target_pcd, max_correspondence_distance=0.5):
    """Computes Sim(3) 7-DoF registration with scale estimation using Multi-Scale PointToPoint ICP."""
    print("[Eval] Computing 7-DoF Sim(3) registration (Scale + Rigid Transformation)...")
    
    # Downsample for faster and robust coarse registration
    src_down = source_pcd.voxel_down_sample(voxel_size=0.05) if len(source_pcd.points) > 50000 else source_pcd
    tgt_down = target_pcd.voxel_down_sample(voxel_size=0.05) if len(target_pcd.points) > 50000 else target_pcd

    # Initial coarse alignment via centroids and bounding box extents
    source_center = src_down.get_center()
    target_center = tgt_down.get_center()
    
    source_extent = np.linalg.norm(src_down.get_max_bound() - src_down.get_min_bound())
    target_extent = np.linalg.norm(tgt_down.get_max_bound() - tgt_down.get_min_bound())
    
    init_scale = target_extent / (source_extent + 1e-8)
    current_transform = np.identity(4)
    current_transform[:3, :3] *= init_scale
    current_transform[:3, 3] = target_center - (source_center * init_scale)

    # Multi-scale coarse-to-fine ICP stages
    thresholds = [max_correspondence_distance * 4.0, max_correspondence_distance * 2.0, max_correspondence_distance]
    for stage_idx, thresh in enumerate(thresholds):
        reg = o3d.pipelines.registration.registration_icp(
            src_down, tgt_down, thresh, current_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )
        current_transform = reg.transformation

    return current_transform

def evaluate_gt_accuracy(mesh_path: str, gt_path: str, transform_matrix=None, compute_transform: bool = False,
                         n_samples: int = 2000000, coverage_radius: float = 0.5):
    """Evaluates Chamfer accuracy/completeness and 95% Hausdorff against Ground Truth LiDAR PCD."""
    print(f"[Eval] Loading Open3D Mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    
    print(f"[Eval] Uniformly sampling {n_samples:,} surface points from mesh...")
    mesh_pcd = mesh.sample_points_uniformly(number_of_points=n_samples)

    print(f"[Eval] Loading Ground Truth PCD: {gt_path}")
    if gt_path.lower().endswith(".pcd") or gt_path.lower().endswith(".ply"):
        gt_pcd = o3d.io.read_point_cloud(gt_path)
    else:
        # Fallback for mesh GT
        gt_mesh = o3d.io.read_triangle_mesh(gt_path)
        gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=n_samples)

    # Sim(3) 7-DoF Alignment
    if compute_transform or transform_matrix is None:
        transform_matrix = align_sim3_point_to_point(mesh_pcd, gt_pcd, max_correspondence_distance=coverage_radius)
    
    mesh_pcd.transform(transform_matrix)

    # Accuracy: Mesh -> GT (nearest distance from mesh sample to GT)
    dists_mesh_to_gt = np.asarray(mesh_pcd.compute_point_cloud_distance(gt_pcd))
    accuracy_mean = np.mean(dists_mesh_to_gt)
    accuracy_rmse = np.sqrt(np.mean(dists_mesh_to_gt ** 2))
    
    # Completeness: GT -> Mesh (filtered by coverage radius)
    dists_gt_to_mesh = np.asarray(gt_pcd.compute_point_cloud_distance(mesh_pcd))
    valid_gt_mask = dists_gt_to_mesh <= coverage_radius
    valid_dists_gt_to_mesh = dists_gt_to_mesh[valid_gt_mask] if np.any(valid_gt_mask) else dists_gt_to_mesh

    completeness_mean = np.mean(valid_dists_gt_to_mesh)
    completeness_rmse = np.sqrt(np.mean(valid_dists_gt_to_mesh ** 2))

    chamfer_l1 = (accuracy_mean + completeness_mean) / 2.0
    hausdorff_95 = float(np.percentile(dists_mesh_to_gt, 95))

    return {
        "accuracy_mean_m": round(accuracy_mean, 5),
        "accuracy_rmse_m": round(accuracy_rmse, 5),
        "completeness_mean_m": round(completeness_mean, 5),
        "completeness_rmse_m": round(completeness_rmse, 5),
        "chamfer_l1_m": round(chamfer_l1, 5),
        "hausdorff_95_m": round(hausdorff_95, 5),
        "gt_coverage_pct": round(float(np.mean(valid_gt_mask)) * 100.0, 2)
    }, transform_matrix

def main():
    parser = argparse.ArgumentParser(description="3D Mesh Topology Hygiene & Sim(3) GT Evaluation Tool")
    parser.add_argument("--mesh", type=str, required=True, help="Path to input .obj or .ply mesh file")
    parser.add_argument("--gt", type=str, default=None, help="Path to Ground Truth LiDAR PCD or Mesh file")
    parser.add_argument("--transform", type=str, default=None, help="Path to precomputed Sim(3) transform JSON file")
    parser.add_argument("--compute-transform", action="store_true", help="Force compute new Sim(3) transform and save")
    parser.add_argument("--n-samples", type=int, default=2000000, help="Number of uniform surface sample points (default: 2M)")
    parser.add_argument("--coverage-radius", type=float, default=0.5, help="GT coverage mask radius in meters (default: 0.5m)")
    parser.add_argument("--check-self-intersect", action="store_true", help="Enable computationally expensive self-intersection check")
    parser.add_argument("--out", type=str, default="outputs/eval", help="Output directory for eval results")

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print(" 3DRC 3D Mesh & Geometry Evaluation Pipeline")
    print("==================================================")

    # 1. Topology Hygiene Analysis
    hygiene_res = compute_mesh_hygiene_and_topology(args.mesh, check_self_intersect=args.check_self_intersect)

    # 2. Sim(3) GT Accuracy Analysis
    gt_res = None
    transform_matrix = None
    if args.gt:
        if args.transform and os.path.exists(args.transform) and not args.compute_transform:
            print(f"[Eval] Loading precomputed Sim(3) transform matrix from {args.transform}")
            with open(args.transform, "r") as f:
                transform_matrix = np.array(json.load(f)["transform_matrix"])
        
        gt_res, transform_matrix = evaluate_gt_accuracy(
            args.mesh, args.gt, transform_matrix=transform_matrix,
            compute_transform=args.compute_transform, n_samples=args.n_samples,
            coverage_radius=args.coverage_radius
        )

        # Save transform matrix for reused fair comparison
        transform_save_path = out_dir / "sim3_transform.json"
        with open(transform_save_path, "w") as f:
            json.dump({"transform_matrix": transform_matrix.tolist()}, f, indent=2)
        print(f"[Eval] Saved Sim(3) transform matrix to {transform_save_path}")

    # Combine Results
    results = {
        "mesh_path": args.mesh,
        "gt_path": args.gt,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hygiene": hygiene_res,
        "accuracy": gt_res
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
        print(f"Sim(3) Aligned Chamfer L1: {gt_res['chamfer_l1_m']} m")
        print(f"  - Accuracy (Mesh -> GT):   {gt_res['accuracy_mean_m']} m (RMSE: {gt_res['accuracy_rmse_m']} m)")
        print(f"  - Completeness (GT -> Mesh): {gt_res['completeness_mean_m']} m (RMSE: {gt_res['completeness_rmse_m']} m)")
        print(f"Hausdorff 95%: {gt_res['hausdorff_95_m']} m | GT Coverage: {gt_res['gt_coverage_pct']}%")

    print("==================================================")
    print(f"Full results saved to {json_path}")

if __name__ == "__main__":
    main()
