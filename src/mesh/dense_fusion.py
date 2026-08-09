from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

def run_stereo_fusion(workspace_dir: str | Path, max_image_size: int = 2000) -> bool:
    """Fuses depth and normal maps into a unified dense point cloud using COLMAP.

    Args:
        workspace_dir: Path to the dense workspace.
        max_image_size: Max resolution limit during fusion for memory optimization.

    Returns:
        bool: True if fusion succeeded and fused.ply was created, False otherwise.
    """
    workspace_path = Path(workspace_dir).resolve()
    output_path = workspace_path / "fused.ply"

    print("--- Stereo Fusion (Dense Point Cloud Generation) Started ---")
    print(f"Workspace Path: {workspace_path}")
    print(f"Output File: {output_path}")

    # Verify input directories
    if not (workspace_path / "stereo").is_dir():
        print(f"[Error] Stereo directory not found in: {workspace_path}")
        return False

    colmap_bin = shutil.which("colmap") or "colmap"

    try:
        print("\n[Step] Fusing depth maps into a unified point cloud...")
        command = [
            colmap_bin, "stereo_fusion",
            "--workspace_path", str(workspace_path),
            "--output_path", str(output_path),
            "--StereoFusion.max_image_size", str(max_image_size),
        ]

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        if process.stdout:
            for line in process.stdout:
                print(line, end="")

        process.wait()

    except Exception as e:
        print(f"\n[Error] Stereo fusion execution failed: {e}")
        return False

    if output_path.is_file():
        print("\n[Success] Stereo fusion completed successfully!")
        print(f"Dense point cloud saved at: {output_path}")
        return True
    else:
        print(f"\n[Error] Stereo fusion completed but failed to generate output file: {output_path}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo Fusion Script for Point Cloud Generation")
    parser.add_argument("--workspace_dir", type=str, required=True, help="Path to dense workspace directory")
    parser.add_argument("--max_image_size", type=int, default=2000, help="Max image size limit for memory optimization")

    args = parser.parse_args()
    success = run_stereo_fusion(args.workspace_dir, max_image_size=args.max_image_size)
    if not success:
        sys.exit(1)
