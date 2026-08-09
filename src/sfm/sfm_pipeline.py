"""Pycolmap Traditional SfM Pipeline Module."""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import pycolmap

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

def run_sfm_pipeline(image_dir: str, output_dir: str):
    """Performs basic Incremental SfM using Pycolmap."""
    image_path = Path(image_dir)
    output_path = Path(output_dir)
    database_path = output_path / "database.db"

    output_path.mkdir(parents=True, exist_ok=True)

    mode_str = os.environ.get("CAMERA_MODE", "SINGLE").upper()
    if not hasattr(pycolmap.CameraMode, mode_str):
        valid = list(pycolmap.CameraMode.__members__.keys())
        raise ValueError(f"Invalid CAMERA_MODE '{mode_str}'. Valid choices: {valid}")
    cam_mode = getattr(pycolmap.CameraMode, mode_str)

    camera_model = os.environ.get("CAMERA_MODEL", "SIMPLE_RADIAL").upper()

    print(f"--- 3D Reconstruction Pipeline Started ---")
    print(f"Input Path: {image_path}")
    print(f"Output Path: {output_path}")
    print(f"Camera Mode: {cam_mode.name}, Camera Model: {camera_model}")

    start_total = time.time()

    print("\n[Step 1/3] Feature Extraction in progress...")
    start_extract = time.time()
    pycolmap.extract_features(
        database_path,
        image_path,
        camera_mode=cam_mode,
        camera_model=camera_model
    )
    time_extract = time.time() - start_extract
    print(f"Feature Extraction elapsed: {time_extract:.2f} seconds")

    print("\n[Step 2/3] Feature Matching in progress...")
    start_match = time.time()
    pycolmap.match_exhaustive(database_path)
    time_match = time.time() - start_match
    print(f"Feature Matching elapsed: {time_match:.2f} seconds")

    print("\n[Step 3/3] Incremental Mapping in progress...")
    start_map = time.time()
    reconstructions = pycolmap.incremental_mapping(database_path, image_path, output_path)
    time_map = time.time() - start_map
    print(f"Incremental Mapping elapsed: {time_map:.2f} seconds")

    time_total = time.time() - start_total

    if reconstructions:
        reconstructions[0].write(output_path)
        print(f"\nReconstruction successfully completed!")
        print(f"Results saved at: {output_path}")
        log_performance(image_dir, output_dir, time_extract, time_match, time_map, time_total)
    else:
        print("\nReconstruction failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pycolmap SfM Pipeline Script")
    parser.add_argument("--image_dir", type=str, default="data/images", help="Input image directory")
    parser.add_argument("--output_dir", type=str, default="data/reconstruction", help="Output results directory")

    args = parser.parse_args()

    if not Path(args.image_dir).exists():
        print(f"Error: Path '{args.image_dir}' not found.")
        sys.exit(1)

    run_sfm_pipeline(args.image_dir, args.output_dir)
