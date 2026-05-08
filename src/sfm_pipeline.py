import pycolmap
import argparse
from pathlib import Path
import sys

def run_sfm_pipeline(image_dir, output_dir):
    """Performs basic Incremental SfM (Sparse Reconstruction) using Pycolmap.

    This function executes the standard COLMAP pipeline including feature
    extraction, exhaustive matching, and incremental mapping.

    Args:
        image_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory where results will be saved.

    Returns:
        None
    """
    # 1. Path Setup
    image_path = Path(image_dir)
    output_path = Path(output_dir)
    database_path = output_path / "database.db"

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"--- 3D Reconstruction Pipeline Started ---")
    print(f"Input Path: {image_path}")
    print(f"Output Path: {output_path}")

    # 2. Feature Extraction
    # Identify keypoints and generate descriptors for each image.
    print("\n[Step 1/3] Feature Extraction in progress...")
    pycolmap.extract_features(database_path, image_path)

    # 3. Feature Matching
    # Find correspondences between features in different images.
    # Exhaustive matching compares all image pairs.
    print("\n[Step 2/3] Feature Matching in progress...")
    pycolmap.match_exhaustive(database_path)

    # 4. Incremental Mapping
    # Calculate camera poses and 3D point cloud based on matches.
    print("\n[Step 3/3] Incremental Mapping in progress...")
    reconstructions = pycolmap.incremental_mapping(database_path, image_path, output_path)

    # 5. Save and Verify Results
    if reconstructions:
        # Select and save the largest (first) reconstructed model
        reconstructions[0].write(output_path)
        print(f"\n✅ Reconstruction successfully completed!")
        print(f"Results saved at: {output_path}")
        print(f"Generated files:")
        print(f"  - points3D.bin: 3D point cloud coordinates")
        print(f"  - images.bin: Camera poses and image metadata")
        print(f"  - cameras.bin: Camera intrinsic parameters")
    else:
        print("\n❌ Reconstruction failed.")
        print("Reason: Insufficient common features between images for triangulation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pycolmap SfM Pipeline Script")
    parser.add_argument("--image_dir", type=str, default="data/images", help="Input image directory")
    parser.add_argument("--output_dir", type=str, default="data/reconstruction", help="Output results directory")

    args = parser.parse_args()

    # Verify input directory
    if not Path(args.image_dir).exists():
        print(f"Error: Path '{args.image_dir}' not found.")
        print("Please create 'data/images' folder and add test images.")
        sys.exit(1)

    run_sfm_pipeline(args.image_dir, args.output_dir)
