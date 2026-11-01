from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np
import open3d as o3d

from src.sfm.sfm_report import CameraTrack, load_model, resolve_paths, trajectory_stats

def camera_geometries(track: CameraTrack, size: float) -> List[o3d.geometry.LineSet]:
    """Builds the capture-order trajectory and one frustum per camera, colored from first to last frame.

    Args:
        track: Registered cameras in capture order.
        size: Frustum depth in world units.

    Returns:
        The trajectory line set followed by the camera frustums.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    n = len(track.names)
    mapper = ScalarMappable(norm=Normalize(0, max(n - 1, 1)), cmap="viridis")
    order_colors = np.asarray(mapper.to_rgba(np.arange(n)))[:, :3]

    lines = [[i - 1, i] for i in range(1, n) if track.groups[i] == track.groups[i - 1]]
    trajectory = o3d.geometry.LineSet(o3d.utility.Vector3dVector(track.centers), o3d.utility.Vector2iVector(lines))
    trajectory.colors = o3d.utility.Vector3dVector(order_colors[[i for _, i in lines]])
    geometries = [trajectory]

    for i in range(n):
        apex = track.centers[i]
        f = track.forward[i] * size
        u = track.up[i] * size * 0.75
        r = np.cross(track.forward[i], track.up[i]) * size
        corners = [apex + f + r + u, apex + f - r + u, apex + f - r - u, apex + f + r - u]
        frustum = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector([apex] + corners),
            o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]),
        )
        frustum.paint_uniform_color(order_colors[i])
        geometries.append(frustum)
    return geometries

def visualize_reconstruction(model_path: str | Path, method: Optional[str] = None, show_cameras: bool = True) -> None:
    """Visualizes a COLMAP sparse reconstruction using Open3D.

    Loads the colored 3D points, camera poses and the camera path in capture order, and launches
    an interactive 3D rendering window.

    Args:
        model_path: Directory containing binary/text reconstruction files (cameras, images, points3D),
            or a dataset directory (data/<scene>) whose sparse/<method> model is shown.
        method: SfM method subdirectory under sparse/ when model_path is a dataset directory
            (default: active sparse/0).
        show_cameras: Draw camera frustums and the capture-order trajectory.

    Returns:
        None
    """
    try:
        path, _ = resolve_paths(model_path, method, None)
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        return
    path = path.resolve()
    has_sparse = (path / "points3D.bin").is_file() or (path / "points3D.txt").is_file()
    if not has_sparse:
        print(f"[Error] Could not find sparse point cloud files in {path}")
        return

    print(f"--- Loading reconstruction from: {path} ---")
    model = load_model(path)
    if len(model.points) == 0:
        print(f"[Warn] No 3D points found in reconstruction: {path}")
        return

    # 1. Prepare 3D Points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(model.points)
    pcd.colors = o3d.utility.Vector3dVector(model.colors / 255.0)

    # 2. Prepare Coordinate Axes
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries: List[o3d.geometry.Geometry] = [pcd, origin]

    print(f"Total Points: {len(model.points):,}")
    print(f"Total Registered Images: {len(model.track.names):,}")

    # 3. Prepare Cameras and Trajectory
    if show_cameras and len(model.track.names) > 0:
        stats = trajectory_stats(model.track)
        size = 2 * stats["median_step"] if stats["median_step"] > 0 else 0.05
        geometries.extend(camera_geometries(model.track, size))
        print(f"Trajectory: {stats['path_length']:.3f} SfM units, {len(stats['jumps'])} pose jumps "
              "(cameras colored from the first frame, purple, to the last, yellow)")

    # 4. Launch Visualization
    print("\nTip: Use mouse to rotate, scroll to zoom.")
    # draw_geometries comes from open3d's compiled bindings, which Pylance cannot see
    o3d.visualization.draw_geometries(  # pyright: ignore[reportAttributeAccessIssue]
        geometries,
        window_name=f"3DRC Visualization - {path.name}",
        width=1280,
        height=720,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize COLMAP/hloc sparse reconstruction results with cameras and trajectory.")
    parser.add_argument("--model_path", type=str, required=True, help="Sparse reconstruction directory, or a dataset directory (data/<scene>)")
    parser.add_argument("--method", type=str, default=None, help="SfM method under sparse/ when --model_path is a dataset directory (default: sparse/0)")
    parser.add_argument("--no_cameras", action="store_true", help="Show only the point cloud")
    args = parser.parse_args()
    visualize_reconstruction(args.model_path, method=args.method, show_cameras=not args.no_cameras)
