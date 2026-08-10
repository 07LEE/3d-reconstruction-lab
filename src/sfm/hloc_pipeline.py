"""hloc (SuperPoint + SuperGlue) SfM Pipeline Module."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import torch

# Add third_party/hloc to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "third_party" / "hloc"))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "third_party"))

import pycolmap
from hloc import extract_features, match_features, pairs_from_exhaustive, pairs_from_retrieval, reconstruction

def log_performance(image_dir: str | Path, output_dir: str | Path, strategy: str, time_extract: float, time_pairs: float, time_match: float, time_sfm: float, time_total: float) -> None:
    """Logs detailed execution timing metrics for hloc pipeline to outputs/performance_log.txt.

    Args:
        image_dir: Input images directory.
        output_dir: Output sparse model directory.
        strategy: Matching pair strategy ('sequential' or 'exhaustive').
        time_extract: Duration of SuperPoint feature extraction in seconds.
        time_pairs: Duration of pair list generation in seconds.
        time_match: Duration of SuperGlue feature matching in seconds.
        time_sfm: Duration of incremental reconstruction in seconds.
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
        f.write(f"Pipeline: hloc SfM ({strategy})\n")
        f.write(f"Input: {image_dir}\n")
        f.write(f"Output: {output_dir}\n")
        f.write(f"----------------------------------------\n")
        f.write(f"Feature Extraction (SuperPoint): {time_extract:.2f} s\n")
        f.write(f"Pairs Generation: {time_pairs:.2f} s\n")
        f.write(f"Feature Matching (SuperGlue): {time_match:.2f} s\n")
        f.write(f"Sparse Reconstruction: {time_sfm:.2f} s\n")
        f.write(f"Total Elapsed Time: {time_total:.2f} s\n")
        f.write(f"========================================\n")

def generate_sequential_pairs(image_dir: Path, output_pairs_path: Path, overlap: int = 10) -> None:
    """Generates an image pair list based on sequential video frame proximity.

    Args:
        image_dir: Path to directory containing sequentially named frame images.
        output_pairs_path: Path to text file where pair list will be saved.
        overlap: Maximum sequential distance between paired frames (default: 10).

    Returns:
        None
    """
    images = sorted([f.name for f in image_dir.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    print(f"Found {len(images)} images. Generating sequential pairs with overlap {overlap}...")

    pairs: List[Tuple[str, str]] = []
    for i in range(len(images)):
        for j in range(1, overlap + 1):
            if i + j < len(images):
                pairs.append((images[i], images[i + j]))

    with open(output_pairs_path, "w", encoding="utf-8") as f:
        for p1, p2 in pairs:
            f.write(f"{p1} {p2}\n")
    print(f"Generated {len(pairs)} pairs.")

def run_hloc_pipeline(image_dir: str | Path, output_dir: str | Path, weights_dir: str | Path, strategy: str = "sequential", overlap: int = 10) -> bool:
    """Executes the complete hloc SfM pipeline with SuperPoint and SuperGlue.

    Args:
        image_dir: Path to input images directory.
        output_dir: Path to output directory where sparse models and H5 caches are saved.
        weights_dir: Path to pretrained deep model weights directory (sets TORCH_HOME).
        strategy: Matching pair strategy ('sequential' or 'exhaustive').
        overlap: Sequential matching window size.

    Returns:
        bool: True if reconstruction model was successfully generated, False otherwise.
    """
    images = Path(image_dir).resolve()
    outputs = Path(output_dir).resolve()
    weights = Path(weights_dir).resolve()

    os.environ["TORCH_HOME"] = str(weights)
    outputs.mkdir(parents=True, exist_ok=True)

    sfm_pairs = outputs / f"pairs-{strategy}.txt"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"
    sfm_dir = outputs / "sfm"

    feature_conf = extract_features.confs["superpoint_aachen"]
    feature_conf["preprocessing"]["resize_max"] = 2400
    matcher_conf = match_features.confs["superglue"]

    print(f"--- hloc 3D Reconstruction Pipeline ({strategy}) ---")
    start_total = time.time()

    # 1. Feature Extraction
    start_time = time.time()
    if not features.is_file():
        print(f"\n[Step 1/4] Extracting features (SuperPoint, resize_max={feature_conf['preprocessing']['resize_max']})...")
        extract_features.main(feature_conf, images, feature_path=features)
        time_extract = time.time() - start_time
        print(f"Feature Extraction elapsed: {time_extract:.2f} seconds")
    else:
        print("\n[Step 1/4] Features already exist. Skipping extraction.")
        time_extract = 0.0

    # 2. Pairs Generation
    print(f"\n[Step 2/4] Generating image pairs ({strategy})...")
    start_time = time.time()
    if strategy == "exhaustive":
        pairs_from_exhaustive.main(sfm_pairs, image_list=None, features=features)
    elif strategy == "sequential":
        generate_sequential_pairs(images, sfm_pairs, overlap=overlap)
    else:
        print(f"[Error] Unknown strategy: {strategy}")
        return False
    time_pairs = time.time() - start_time
    print(f"Pairs Generation elapsed: {time_pairs:.2f} seconds")

    # 3. Feature Matching
    print("\n[Step 3/4] Matching features (SuperGlue)...")
    start_time = time.time()
    if not matches.is_file():
        match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
        time_match = time.time() - start_time
        print(f"Feature Matching elapsed: {time_match:.2f} seconds")
    else:
        print("Matches already exist. Skipping matching.")
        time_match = 0.0

    # 4. Sparse Reconstruction
    print("\n[Step 4/4] Running 3D Reconstruction (COLMAP Incremental Mapper)...")
    start_time = time.time()
    camera_model = os.environ.get("CAMERA_MODEL", "SIMPLE_RADIAL").upper()

    camera_options = pycolmap.IncrementalPipelineOptions()
    reconstruction.main(
        sfm_dir,
        images,
        sfm_pairs,
        features,
        matches,
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model=camera_model,
        options=camera_options,
    )
    time_sfm = time.time() - start_time
    print(f"Sparse Reconstruction elapsed: {time_sfm:.2f} seconds")

    time_total = time.time() - start_total
    print(f"\nTotal Pipeline Elapsed Time: {time_total:.2f} seconds")

    # Log performance
    log_performance(images, outputs, strategy, time_extract, time_pairs, time_match, time_sfm, time_total)

    target_model = sfm_dir / "0" if (sfm_dir / "0").is_dir() else sfm_dir
    success = (target_model / "cameras.bin").is_file() or (target_model / "cameras.txt").is_file()
    if success:
        print(f"\n[Success] Reconstruction completed successfully! Model saved at: {target_model}")
    else:
        print(f"\n[Error] Reconstruction failed to generate sparse model files.")
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hloc SuperPoint + SuperGlue Structure from Motion Pipeline")
    parser.add_argument("--image_dir", type=str, default="data/images", help="Path to input images directory")
    parser.add_argument("--output_dir", type=str, default="data/reconstruction_hloc", help="Path to output directory")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Path to deep model weights directory")
    parser.add_argument("--strategy", type=str, default="sequential", choices=["sequential", "exhaustive"], help="Pair matching strategy")
    parser.add_argument("--overlap", type=int, default=10, help="Overlap window for sequential matching")

    args = parser.parse_args()

    if not Path(args.image_dir).is_dir():
        print(f"[Error] Image directory not found: '{args.image_dir}'")
        sys.exit(1)

    ok = run_hloc_pipeline(
        args.image_dir,
        args.output_dir,
        args.weights_dir,
        strategy=args.strategy,
        overlap=args.overlap,
    )
    if not ok:
        sys.exit(1)
