"""Pycolmap Traditional SfM Pipeline Module."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import pycolmap

def log_performance(image_dir: str | Path, output_dir: str | Path, time_extract: float, time_match: float, time_map: float, time_total: float) -> None:
    """Logs execution metrics for traditional COLMAP SfM to outputs/performance_log.txt.

    Args:
        image_dir: Input images directory.
        output_dir: Output sparse model directory.
        time_extract: Duration of feature extraction in seconds.
        time_match: Duration of feature matching in seconds.
        time_map: Duration of incremental mapping in seconds.
        time_total: Total pipeline execution duration in seconds.

    Returns:
        None
    """
    log_dir = Path("outputs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "performance_log.txt"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
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

def run_sfm_pipeline(image_dir: str | Path, output_dir: str | Path) -> bool:
    """Performs traditional Incremental SfM using Pycolmap and SIFT features.

    Args:
        image_dir: Path to directory containing input frame images.
        output_dir: Target directory where SQLite database and sparse models will be written.

    Returns:
        bool: True if reconstruction generated a valid sparse model, False otherwise.
    """
    image_path = Path(image_dir).resolve()
    output_path = Path(output_dir).resolve()
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
    pycolmap.extract_features(database_path, image_path, camera_mode=cam_mode, camera_model=camera_model)
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

    if reconstructions and len(reconstructions) > 0:
        reconstructions[0].write(output_path)
        print(f"\n[Success] Reconstruction completed successfully! Model saved at: {output_path}")
        log_performance(image_path, output_path, time_extract, time_match, time_map, time_total)
        return True
    else:
        print("\n[Error] Reconstruction failed to reconstruct sparse points.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pycolmap SfM Pipeline Script")
    parser.add_argument("--image_dir", type=str, default="data/images", help="Input image directory")
    parser.add_argument("--output_dir", type=str, default="data/reconstruction", help="Output results directory")

    args = parser.parse_args()

    if not Path(args.image_dir).is_dir():
        print(f"[Error] Image directory not found: '{args.image_dir}'")
        sys.exit(1)

    ok = run_sfm_pipeline(args.image_dir, args.output_dir)
    if not ok:
        sys.exit(1)
