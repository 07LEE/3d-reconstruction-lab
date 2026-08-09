from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

def run_stereo_matching(workspace_dir: str | Path, gpu_index: str = "0") -> bool:
    """Performs Patch Match Stereo to estimate depth and normal maps using COLMAP CUDA CLI.

    Args:
        workspace_dir: Path to the dense workspace (created by dense_undistort.py).
        gpu_index: GPU device index to execute stereo matching on (default: '0').

    Returns:
        bool: True if successful, False otherwise.
    """
    workspace_path = Path(workspace_dir).resolve()

    print("--- Patch Match Stereo (GPU Accelerated) Started ---")
    print(f"Workspace Path: {workspace_path}")

    # Verify workspace existence
    if not (workspace_path / "sparse").is_dir():
        print(f"[Error] Dense workspace not found at: {workspace_path}")
        print("Please run 'src/dense_undistort.py' first.")
        return False

    colmap_bin = shutil.which("colmap") or "colmap"

    try:
        print("\n[Step] Estimating depth and normal maps using GPU...")
        command = [
            colmap_bin, "patch_match_stereo",
            "--workspace_path", str(workspace_path),
            "--PatchMatchStereo.gpu_index", str(gpu_index),
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

        if process.returncode == 0:
            print("\n[Success] Stereo matching completed successfully!")
            print(f"Depth maps and normal maps generated in: {workspace_path / 'stereo'}")
            return True
        else:
            print(f"\n[Error] Stereo matching failed with exit code: {process.returncode}")
            return False

    except Exception as e:
        print(f"\n[Error] Stereo matching execution failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch Match Stereo Script for Depth Estimation")
    parser.add_argument("--workspace_dir", type=str, required=True, help="Path to dense workspace directory")
    parser.add_argument("--gpu_index", type=str, default="0", help="GPU index for CUDA acceleration")

    args = parser.parse_args()

    if not Path(args.workspace_dir).exists():
        print(f"[Error] Workspace directory '{args.workspace_dir}' not found.")
        sys.exit(1)

    success = run_stereo_matching(args.workspace_dir, gpu_index=args.gpu_index)
    if not success:
        sys.exit(1)
