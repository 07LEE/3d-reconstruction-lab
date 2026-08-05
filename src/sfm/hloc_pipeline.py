"""hloc (SuperPoint + SuperGlue) SfM Pipeline Module."""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import torch

# Add third_party/hloc to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "third_party" / "hloc"))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "third_party"))

from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive, pairs_from_retrieval

def log_performance(image_dir, output_dir, strategy, time_extract, time_pairs, time_match, time_sfm, time_total):
    log_dir = Path("outputs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "performance_log.txt"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
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

def generate_sequential_pairs(image_dir, output_pairs_path, overlap=10):
    images = sorted([f.name for f in image_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
    print(f"Found {len(images)} images. Generating sequential pairs with overlap {overlap}...")
    
    pairs = []
    for i in range(len(images)):
        for j in range(1, overlap + 1):
            if i + j < len(images):
                pairs.append((images[i], images[i+j]))
    
    with open(output_pairs_path, 'w') as f:
        for p1, p2 in pairs:
            f.write(f"{p1} {p2}\n")
    print(f"Generated {len(pairs)} pairs.")

def run_hloc_pipeline(image_dir, output_dir, weights_dir, strategy='sequential', overlap=10):
    images = Path(image_dir)
    outputs = Path(output_dir)
    weights = Path(weights_dir)
    
    os.environ['TORCH_HOME'] = str(weights)
    outputs.mkdir(parents=True, exist_ok=True)
    
    sfm_pairs = outputs / f'pairs-{strategy}.txt'
    features = outputs / 'features.h5'
    matches = outputs / 'matches.h5'
    sfm_dir = outputs / 'sfm'
    
    feature_conf = extract_features.confs['superpoint_aachen']
    feature_conf['preprocessing']['resize_max'] = 2400
    matcher_conf = match_features.confs['superglue']
    
    print(f"--- hloc 3D Reconstruction Pipeline ({strategy}) ---")
    
    start_total = time.time()
    
    # 1. Feature Extraction
    start_time = time.time()
    if not features.exists():
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
    if strategy == 'exhaustive':
        pairs_from_exhaustive.main(sfm_pairs, image_list=None, features=features)
    elif strategy == 'sequential':
        generate_sequential_pairs(images, sfm_pairs, overlap=overlap)
    else:
        print(f"Error: Unknown strategy {strategy}")
        sys.exit(1)
    time_pairs = time.time() - start_time
    print(f"Pairs Generation elapsed: {time_pairs:.2f} seconds")

    # 3. Feature Matching
    print("\n[Step 3/4] Matching features (SuperGlue)...")
    start_time = time.time()
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
    time_match = time.time() - start_time
    print(f"Feature Matching elapsed: {time_match:.2f} seconds")

    # 4. Sparse Reconstruction
    print("\n[Step 4/4] Performing sparse reconstruction...")
    start_time = time.time()
    reconstruction.main(sfm_dir, images, sfm_pairs, features, matches)
    time_sfm = time.time() - start_time
    print(f"Sparse Reconstruction elapsed: {time_sfm:.2f} seconds")

    time_total = time.time() - start_total

    print(f"\nhloc Reconstruction ({strategy}) completed!")
    print(f"Results saved at: {sfm_dir}")
    
    log_performance(image_dir, output_dir, strategy, time_extract, time_pairs, time_match, time_sfm, time_total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hloc SfM Pipeline Script")
    parser.add_argument("--image_dir", type=str, default="data/images", help="Input image directory")
    parser.add_argument("--output_dir", type=str, default="data/hloc_reconstruction", help="Output results directory")
    parser.add_argument("--weights_dir", type=str, default="weights/hloc", help="Model weights directory")
    parser.add_argument("--strategy", type=str, default="sequential", choices=['exhaustive', 'sequential'], help="Matching strategy")
    parser.add_argument("--overlap", type=int, default=10, help="Overlap for sequential matching")

    args = parser.parse_args()

    if not Path(args.image_dir).exists():
        print(f"Error: Path '{args.image_dir}' not found.")
        sys.exit(1)

    run_hloc_pipeline(args.image_dir, args.output_dir, args.weights_dir, args.strategy, args.overlap)
