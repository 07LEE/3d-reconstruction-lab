import pycolmap
import argparse
from pathlib import Path
import sys
import time
from datetime import datetime

def log_performance(image_dir, output_dir, time_extract, time_match, time_map, time_total):
    log_dir = Path("outputs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "performance_log.txt"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"\n========================================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Pipeline: Pycolmap SfM\n")
        f.write(f"Input: {image_dir}\n")
        f.write(f"Output: {output_dir}\n")
        f.write(f"----------------------------------------\n")
        f.write(f"Feature Extraction: {time_extract:.2f} s\n")
        f.write(f"Feature Matching: {time_match:.2f} s\n")
        f.write(f"Incremental Mapping: {time_map:.2f} s\n")
        f.write(f"Total Elapsed Time: {time_total:.2f} s\n")
        f.write(f"========================================\n")

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

    start_total = time.time()

    # 2. Feature Extraction
    # Identify keypoints and generate descriptors for each image.
    print("\n[Step 1/3] Feature Extraction in progress...")
    start_extract = time.time()
    pycolmap.extract_features(database_path, image_path)
    end_extract = time.time()
    time_extract = end_extract - start_extract
    print(f"Feature Extraction elapsed: {time_extract:.2f} seconds")

    # 3. Feature Matching
    # Find correspondences between features in different images.
    # Exhaustive matching compares all image pairs.
    print("\n[Step 2/3] Feature Matching in progress...")
    start_match = time.time()
    pycolmap.match_exhaustive(database_path)
    end_match = time.time()
    time_match = end_match - start_match
    print(f"Feature Matching elapsed: {time_match:.2f} seconds")

    # 4. Incremental Mapping
    # Calculate camera poses and 3D point cloud based on matches.
    print("\n[Step 3/3] Incremental Mapping in progress...")
    start_map = time.time()
    reconstructions = pycolmap.incremental_mapping(database_path, image_path, output_path)
    end_map = time.time()
    time_map = end_map - start_map
    print(f"Incremental Mapping elapsed: {time_map:.2f} seconds")

    end_total = time.time()
    time_total = end_total - start_total

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
        
        # Log to file
        log_performance(image_dir, output_dir, time_extract, time_match, time_map, time_total)
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

