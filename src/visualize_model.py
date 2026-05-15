import pycolmap
import open3d as o3d
import numpy as np
import argparse
from pathlib import Path

def visualize_reconstruction(model_path):
    """Visualizes a COLMAP reconstruction using Open3D.

    Args:
        model_path (str): Path to the directory containing binary files (cameras, images, points3D).
    """
    path = Path(model_path)
    if not (path / "points3D.bin").exists():
        print(f"Error: Could not find points3D.bin in {model_path}")
        return

    print(f"--- Loading reconstruction from: {model_path} ---")
    reconstruction = pycolmap.Reconstruction(path)

    # 1. Prepare 3D Points
    points3d = reconstruction.points3D
    xyz = []
    rgb = []

    for point_id, point in points3d.items():
        xyz.append(point.xyz)
        # Normalize color to [0, 1] for Open3D
        rgb.append(point.color / 255.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
    pcd.colors = o3d.utility.Vector3dVector(np.array(rgb))

    # 2. Prepare Camera Poses (Optional visualization)
    geometries = [pcd]
    
    # Create a coordinate frame for the origin
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(origin)

    print(f"Total Points: {len(xyz)}")
    print(f"Total Registered Images: {reconstruction.num_reg_images()}")
    
    # 3. Launch Visualization
    print("\n💡 Tip: Use mouse to rotate, scroll to zoom.")
    o3d.visualization.draw_geometries(geometries, 
                                      window_name=f"3DRC Visualization - {path.name}",
                                      width=1280, height=720)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize COLMAP/hloc sparse reconstruction results.")
    parser.add_argument("--model_path", type=str, default="data/hloc_reconstruction/sfm/models/0", 
                        help="Path to the sparse reconstruction directory")

    args = parser.parse_args()
    visualize_reconstruction(args.model_path)
