import pycolmap
import argparse
from pathlib import Path
import sys

def run_undistortion(input_dir, image_dir, output_dir):
    """Undistorts images and creates a workspace for dense reconstruction.

    This script takes a sparse reconstruction model and original images,
    removes lens distortion, and prepares the data for stereo matching.

    Args:
        input_dir (str): Path to the sparse reconstruction folder (containing .bin files).
        image_dir (str): Path to the original image directory.
        output_dir (str): Path where undistorted images and dense workspace will be saved.

    Returns:
        bool: True if successful, False otherwise.
    """
    input_path = Path(input_dir)
    image_path = Path(image_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"--- Image Undistortion Started ---")
    print(f"Sparse Model Path: {input_path}")
    print(f"Original Image Path: {image_path}")
    print(f"Output Workspace Path: {output_path}")

    # Verify input sparse model exists
    if not (input_path / "cameras.bin").exists():
        print(f"Error: Sparse model not found in {input_path}")
        return False

    try:
        # Perform undistortion
        # This creates a workspace with undistorted images and updated camera models.
        print("\n[Step] Processing image undistortion...")
        pycolmap.undistort_images(output_path, input_path, image_path)
        
        print(f"\nUndistortion successfully completed!")
        print(f"Dense workspace prepared at: {output_path}")
        return True
    except Exception as e:
        print(f"\nUndistortion failed.")
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Undistortion Script for Dense Reconstruction")
    parser.add_argument("--input_dir", type=str, default="data/reconstruction/1", help="Path to sparse model")
    parser.add_argument("--image_dir", type=str, default="data/images", help="Original image directory")
    parser.add_argument("--output_dir", type=str, default="data/reconstruction/dense", help="Dense workspace output path")

    args = parser.parse_args()

    # Basic path verification
    if not Path(args.input_dir).exists():
        print(f"Error: Sparse model directory '{args.input_dir}' not found.")
        sys.exit(1)
    
    if not Path(args.image_dir).exists():
        print(f"Error: Image directory '{args.image_dir}' not found.")
        sys.exit(1)

    success = run_undistortion(args.input_dir, args.image_dir, args.output_dir)
    if not success:
        sys.exit(1)
