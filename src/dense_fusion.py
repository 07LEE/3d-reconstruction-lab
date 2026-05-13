import subprocess
import argparse
from pathlib import Path
import sys

def run_stereo_fusion(workspace_dir):
    """Fuses depth maps into a dense point cloud using COLMAP.

    Args:
        workspace_dir (str): Path to the dense workspace.

    Returns:
        bool: True if successful, False otherwise.
    """
    workspace_path = Path(workspace_dir)
    output_path = workspace_path / "fused.ply"

    print(f"--- Stereo Fusion (Dense Point Cloud Generation) Started ---")
    print(f"Workspace Path: {workspace_path}")
    print(f"Output File: {output_path}")

    # Verify input directories
    if not (workspace_path / "stereo").exists():
        print(f"Error: Stereo directory not found in {workspace_path}")
        return False

    try:
        print("\n[Step] Fusing depth maps into a unified point cloud...")
        
        # Build the CLI command
        command = [
            "colmap", "stereo_fusion",
            "--workspace_path", str(workspace_path),
            "--output_path", str(output_path),
            "--StereoFusion.max_image_size", "2000"  # Optimization for memory/speed
        ]
        
        # Execute and stream output
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        for line in process.stdout:
            print(line, end="")

        process.wait()

    except Exception as e:
        print(f"\n❌ Stereo fusion failed.")
        print(f"Error: {str(e)}")
        return False

    if output_path.exists():
        print(f"\n✅ Stereo fusion successfully completed!")
        print(f"Dense point cloud saved at: {output_path}")
        return True
    else:
        print(f"\n❌ Stereo fusion failed to generate the output file.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo Fusion Script for Point Cloud Generation")
    parser.add_argument("--workspace_dir", type=str, default="data/reconstruction/dense", help="Path to dense workspace")

    args = parser.parse_args()
    run_stereo_fusion(args.workspace_dir)
