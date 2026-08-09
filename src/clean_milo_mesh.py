import os
import argparse
import numpy as np
import open3d as o3d

def clean_mesh(input_path, output_path, min_triangles=1000):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input mesh not found: {input_path}")
    
    print(f"Loading raw mesh from {input_path}...")
    mesh = o3d.io.read_triangle_mesh(input_path)
    
    orig_v = len(mesh.vertices)
    orig_f = len(mesh.triangles)
    print(f"Original Mesh - Vertices: {orig_v}, Triangles: {orig_f}")
    
    # Cluster connected components
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    largest_cluster_idx = np.argmax(cluster_n_triangles)
    largest_triangle_count = cluster_n_triangles[largest_cluster_idx]
    print(f"Largest component triangle count: {largest_triangle_count} ({largest_triangle_count/orig_f*100:.2f}% of total)")
    
    # Keep only the largest component
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    
    clean_v = len(mesh.vertices)
    clean_f = len(mesh.triangles)
    
    print(f"Cleaned Mesh - Vertices: {clean_v}, Triangles: {clean_f}")
    print(f"Strict Edge Manifold: {mesh.is_edge_manifold(allow_boundary_edges=False)}")
    print(f"Edge Manifold (Allow Boundary): {mesh.is_edge_manifold(allow_boundary_edges=True)}")
    print(f"Is Watertight: {mesh.is_watertight()}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Saved cleaned mesh to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean MILo mesh by extracting largest connected component.")
    parser.add_argument("--input", type=str, required=True, help="Input raw PLY mesh file path")
    parser.add_argument("--output", type=str, required=True, help="Output cleaned PLY mesh file path")
    args = parser.parse_args()
    
    clean_mesh(args.input, args.output)
