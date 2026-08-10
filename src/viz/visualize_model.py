from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import numpy as np
import open3d as o3d
import pycolmap

def visualize_reconstruction(model_path: str | Path) -> None:
    """Visualizes a COLMAP sparse reconstruction using Open3D.

    Loads camera poses and 3D point cloud landmarks, visualizes them with 
    coordinate frames, and launches an interactive 3D rendering window.

    Args:
        model_path: Path to the directory containing binary/text reconstruction files 
            (cameras, images, points3D).

    Returns:
        None
    """
    path = Path(model_path).resolve()
    has_sparse = (path / "points3D.bin").is_file() or (path / "points3D.txt").is_file() or (path / "points.ply").is_file()
    if not has_sparse:
        print(f"[Error] Could not find sparse point cloud files in {path}")
        return

    print(f"--- Loading reconstruction from: {path} ---")
    reconstruction = pycolmap.Reconstruction(path)

    # 1. Prepare 3D Points
    points3d = reconstruction.points3D
    xyz = []
    rgb = []

    for point in points3d.values():
        xyz.append(point.xyz)
        rgb.append(point.color / 255.0)

    if not xyz:
        print(f"[Warn] No 3D points found in reconstruction: {path}")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
    pcd.colors = o3d.utility.Vector3dVector(np.array(rgb))

    # 2. Prepare Coordinate Axes
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries: List[o3d.geometry.Geometry] = [pcd, origin]

    print(f"Total Points: {len(xyz):,}")
    print(f"Total Registered Images: {reconstruction.num_reg_images():,}")

    # 3. Launch Visualization
    print("\nTip: Use mouse to rotate, scroll to zoom.")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"3DRC Visualization - {path.name}",
        width=1280,
        height=720,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize COLMAP/hloc sparse reconstruction results.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to sparse reconstruction directory containing .bin files")
    args = parser.parse_args()
    visualize_reconstruction(args.model_path)
