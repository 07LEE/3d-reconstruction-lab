import argparse
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import open3d as o3d

def clean_mesh(input_path: str | Path, output_path: str | Path, keep_largest: bool = True, min_triangles: int = 500) -> bool:
    """Cleans a raw mesh by filtering out disconnected floater components.

    Args:
        input_path: Path to the input raw polygon mesh (.ply or .obj).
        output_path: Path where the cleaned mesh will be saved.
        keep_largest: If True, keeps only the single largest connected component.
        min_triangles: Minimum triangle count threshold for components to keep (if keep_largest=False).

    Returns:
        bool: True if cleaning succeeded and mesh was saved, False otherwise.
    """
    in_p = Path(input_path)
    out_p = Path(output_path)

    if not in_p.is_file():
        print(f"[Error] Input mesh file not found: {in_p}")
        return False

    print(f"[CleanMesh] Loading raw mesh from: {in_p}")
    mesh = o3d.io.read_triangle_mesh(str(in_p))

    orig_v = len(mesh.vertices)
    orig_f = len(mesh.triangles)
    print(f"[CleanMesh] Original Mesh - Vertices: {orig_v:,}, Triangles: {orig_f:,}")

    if orig_f == 0:
        print("[Warn] Mesh contains 0 triangles. Skipping component clustering.")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(out_p), mesh)
        return True

    # Cluster connected triangle components
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if len(cluster_n_triangles) == 0:
        print("[Warn] No connected components identified in mesh.")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(out_p), mesh)
        return True

    if keep_largest:
        largest_cluster_idx = int(np.argmax(cluster_n_triangles))
        largest_triangle_count = int(cluster_n_triangles[largest_cluster_idx])
        pct = (largest_triangle_count / orig_f) * 100.0
        print(f"[CleanMesh] Largest component: {largest_triangle_count:,} triangles ({pct:.2f}% of total)")
        triangles_to_remove = triangle_clusters != largest_cluster_idx
    else:
        print(f"[CleanMesh] Filtering components with triangle count >= {min_triangles}...")
        valid_cluster_indices = set(np.where(cluster_n_triangles >= min_triangles)[0])
        triangles_to_remove = np.isin(triangle_clusters, list(valid_cluster_indices), invert=True)

    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()

    clean_v = len(mesh.vertices)
    clean_f = len(mesh.triangles)

    print(f"[CleanMesh] Cleaned Mesh - Vertices: {clean_v:,}, Triangles: {clean_f:,}")
    print(f"[CleanMesh] Strict Edge Manifold: {mesh.is_edge_manifold(allow_boundary_edges=False)}")
    print(f"[CleanMesh] Edge Manifold (Allow Boundary): {mesh.is_edge_manifold(allow_boundary_edges=True)}")
    print(f"[CleanMesh] Is Watertight: {mesh.is_watertight()}")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    success = o3d.io.write_triangle_mesh(str(out_p), mesh)
    if success:
        print(f"[CleanMesh] Saved cleaned mesh to: {out_p}")
    else:
        print(f"[Error] Failed to write mesh to: {out_p}")
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw 3D mesh by extracting largest component or removing floaters.")
    parser.add_argument("--input", type=str, required=True, help="Input raw PLY or OBJ mesh file path")
    parser.add_argument("--output", type=str, required=True, help="Output cleaned PLY or OBJ mesh file path")
    parser.add_argument("--min_triangles", type=int, default=500, help="Min triangle threshold when not keep_largest")
    parser.add_argument("--all_large", action="store_true", help="Keep all components larger than min_triangles instead of only the single largest")

    args = parser.parse_args()

    ok = clean_mesh(
        input_path=args.input,
        output_path=args.output,
        keep_largest=not args.all_large,
        min_triangles=args.min_triangles
    )
    if not ok:
        sys.exit(1)
