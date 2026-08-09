from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional
import pycolmap

def get_reconstruction_stats(path: str | Path) -> Optional[Dict[str, Any]]:
    """Extracts key statistics from a COLMAP reconstruction.

    Args:
        path: Path to the sparse reconstruction directory containing binary or text files.

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing registered images, 3D point counts,
            mean reprojection error, mean track length, and mean observations per image,
            or None if loading failed.
    """
    p = Path(path)
    if not (p / "cameras.bin").is_file() and not (p / "cameras.txt").is_file():
        return None

    try:
        reconstruction = pycolmap.Reconstruction(p)
        reconstruction.update_point_3d_errors()

        return {
            "reg_images": reconstruction.num_reg_images(),
            "num_points3D": reconstruction.num_points3D(),
            "mean_reproj_error": reconstruction.compute_mean_reprojection_error(),
            "mean_track_length": reconstruction.compute_mean_track_length(),
            "mean_obs_per_image": reconstruction.compute_mean_observations_per_reg_image(),
        }
    except Exception as e:
        print(f"[Error] Failed loading reconstruction at {p}: {e}")
        return None

def compare_models(colmap_path: str | Path, hloc_path: str | Path) -> None:
    """Compares statistics between COLMAP and hloc reconstructions and prints a comparison table.

    Args:
        colmap_path: Path to the baseline COLMAP sparse model directory.
        hloc_path: Path to the hloc deep learning sparse model directory.

    Returns:
        None
    """
    colmap_stats = get_reconstruction_stats(colmap_path)
    hloc_stats = get_reconstruction_stats(hloc_path)

    print("\n" + "=" * 60)
    print(f"{'Metric':<30} | {'COLMAP (SIFT)':<12} | {'hloc (SP+SG)':<12}")
    print("-" * 60)

    metrics = [
        ("Registered Images", "reg_images", "{:.0f}"),
        ("Total 3D Points", "num_points3D", "{:.0f}"),
        ("Mean Reproj. Error (px)", "mean_reproj_error", "{:.4f}"),
        ("Mean Track Length", "mean_track_length", "{:.2f}"),
        ("Mean Obs per Image", "mean_obs_per_image", "{:.2f}"),
    ]

    for label, key, fmt in metrics:
        val1 = fmt.format(colmap_stats[key]) if colmap_stats else "N/A"
        val2 = fmt.format(hloc_stats[key]) if hloc_stats else "N/A"
        print(f"{label:<30} | {val1:<12} | {val2:<12}")

    print("=" * 60)

    if colmap_stats and hloc_stats and colmap_stats["num_points3D"] > 0:
        improvement = (hloc_stats["num_points3D"] - colmap_stats["num_points3D"]) / colmap_stats["num_points3D"] * 100.0
        print(f"\nInsight: hloc generated {improvement:+.1f}% more 3D points than COLMAP.")

    print("\n* Note: Comparison is read-only and does not modify any data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare COLMAP and hloc reconstruction results.")
    parser.add_argument("--colmap_path", type=str, required=True, help="Path to standard COLMAP sparse directory")
    parser.add_argument("--hloc_path", type=str, required=True, help="Path to hloc sparse directory")

    args = parser.parse_args()
    compare_models(args.colmap_path, args.hloc_path)
