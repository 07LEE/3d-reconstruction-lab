import pycolmap
import argparse
from pathlib import Path

def get_reconstruction_stats(path):
    """Extracts key statistics from a COLMAP reconstruction.

    Args:
        path (Path): Path to the sparse reconstruction directory (containing bin files).

    Returns:
        dict: A dictionary containing reconstruction statistics.
    """
    if not (path / "cameras.bin").exists() and not (path / "cameras.txt").exists():
        return None

    try:
        reconstruction = pycolmap.Reconstruction(path)
        reconstruction.update_point_3d_errors()

        stats = {
            "reg_images": reconstruction.num_reg_images(),
            "num_points3D": reconstruction.num_points3D(),
            "mean_reproj_error": reconstruction.compute_mean_reprojection_error(),
            "mean_track_length": reconstruction.compute_mean_track_length(),
            "mean_obs_per_image": reconstruction.compute_mean_observations_per_reg_image()
        }
        return stats
    except Exception as e:
        print(f"Error loading reconstruction at {path}: {e}")
        return None

def compare_models(colmap_path, hloc_path):
    """Compares statistics between COLMAP and hloc reconstructions and prints a table."""
    colmap_stats = get_reconstruction_stats(Path(colmap_path))
    hloc_stats = get_reconstruction_stats(Path(hloc_path))

    print("\n" + "="*60)
    print(f"{'Metric':<30} | {'COLMAP (SIFT)':<12} | {'hloc (SP+SG)':<12}")
    print("-" * 60)

    metrics = [
        ("Registered Images", "reg_images", "{:.0f}"),
        ("Total 3D Points", "num_points3D", "{:.0f}"),
        ("Mean Reproj. Error (px)", "mean_reproj_error", "{:.4f}"),
        ("Mean Track Length", "mean_track_length", "{:.2f}"),
        ("Mean Obs per Image", "mean_obs_per_image", "{:.2f}")
    ]

    for label, key, fmt in metrics:
        val1 = fmt.format(colmap_stats[key]) if colmap_stats else "N/A"
        val2 = fmt.format(hloc_stats[key]) if hloc_stats else "N/A"
        print(f"{label:<30} | {val1:<12} | {val2:<12}")

    print("="*60)
    
    if colmap_stats and hloc_stats:
        improvement = (hloc_stats['num_points3D'] - colmap_stats['num_points3D']) / colmap_stats['num_points3D'] * 100
        print(f"\n💡 Insight: hloc generated {improvement:+.1f}% more 3D points than COLMAP.")
    
    print("\n* Note: Comparison is read-only and does not modify any data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare COLMAP and hloc reconstruction results.")
    parser.add_argument("--colmap_path", type=str, default="data/reconstruction", help="Path to standard COLMAP sparse dir")
    parser.add_argument("--hloc_path", type=str, default="data/hloc_reconstruction/sfm", help="Path to hloc sparse dir")

    args = parser.parse_args()
    compare_models(args.colmap_path, args.hloc_path)
