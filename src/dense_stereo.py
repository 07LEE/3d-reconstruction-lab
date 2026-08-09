import subprocess
import argparse
from pathlib import Path
import sys

def run_stereo_matching(workspace_dir):
    """Performs Patch Match Stereo to estimate depth and normal maps.

    This script computes the depth of each pixel in the undistorted images
    using the sparse model as a geometric constraint. 

    NOTE: This version uses the COLMAP CLI directly via subprocess to 
    ensure CUDA/GPU support, bypassing issues in the pycolmap library.

    Args:
        workspace_dir (str): Path to the dense workspace (created by undistort script).

    Returns:
        bool: True if successful, False otherwise.
    """
    workspace_path = Path(workspace_dir)

    print(f"--- Patch Match Stereo (GPU Accelerated) Started ---")
    print(f"Workspace Path: {workspace_path}")

    # Verify workspace existence
    if not (workspace_path / "sparse").exists():
        print(f"Error: Dense workspace not found at {workspace_path}")
        print("Please run 'src/dense_undistort.py' first.")
        return False

    try:
        # Perform Stereo Matching using COLMAP CLI to force CUDA usage
        print("\n[Step] Estimating depth and normal maps using GPU...")
        
        # Build the CLI command
        command = [
            "colmap", "patch_match_stereo",
            "--workspace_path", str(workspace_path),
            "--PatchMatchStereo.gpu_index", "0"  # Use the first GPU
        ]
        
        # Execute the command and stream output
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Print output in real-time
        for line in process.stdout:
            print(line, end="")

        process.wait()

        if process.returncode == 0:
            print(f"\nStereo matching successfully completed!")
            print(f"Depth maps and normal maps generated in: {workspace_path / 'stereo'}")
            return True
        else:
            print(f"\nStereo matching failed with exit code {process.returncode}.")
            return False

    except Exception as e:
        print(f"\nStereo matching failed.")
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch Match Stereo Script for Depth Estimation")
    parser.add_argument("--workspace_dir", type=str, required=True, help="Path to dense workspace directory")

    args = parser.parse_args()

    # Path verification
    if not Path(args.workspace_dir).exists():
        print(f"Error: Workspace directory '{args.workspace_dir}' not found.")
        sys.exit(1)

    success = run_stereo_matching(args.workspace_dir)
    if not success:
        sys.exit(1)
