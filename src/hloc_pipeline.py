import argparse
import os
import sys
from pathlib import Path
from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive, pairs_from_retrieval
import torch

def generate_sequential_pairs(image_dir, output_pairs_path, overlap=10):
    """Generates sequential pairs for video frames.
    
    Args:
        image_dir (Path): Path to images.
        output_pairs_path (Path): Path to save pairs.txt.
        overlap (int): Number of subsequent frames to match with.
    """
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
    """Performs SfM reconstruction using hloc.

    Args:
        image_dir (str): Path to images.
        output_dir (str): Path to output.
        weights_dir (str): Path to weights.
        strategy (str): 'exhaustive' or 'sequential'.
        overlap (int): Overlap for sequential matching.
    """
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
    feature_conf['preprocessing']['resize_max'] = 2400  # 원본 디테일을 살리기 위해 해상도 상향
    matcher_conf = match_features.confs['superglue']
    
    print(f"--- hloc 3D Reconstruction Pipeline ({strategy}) ---")
    
    # 1. Feature Extraction
    if not features.exists():
        print(f"\n[Step 1/4] Extracting features (SuperPoint, resize_max={feature_conf['preprocessing']['resize_max']})...")
        extract_features.main(feature_conf, images, feature_path=features)
    else:
        print("\n[Step 1/4] Features already exist. Skipping extraction.")

    # 2. Pairs Generation
    print(f"\n[Step 2/4] Generating image pairs ({strategy})...")
    if strategy == 'exhaustive':
        pairs_from_exhaustive.main(sfm_pairs, image_list=None, features=features)
    elif strategy == 'sequential':
        generate_sequential_pairs(images, sfm_pairs, overlap=overlap)
    else:
        print(f"Error: Unknown strategy {strategy}")
        sys.exit(1)

    # 3. Feature Matching
    print("\n[Step 3/4] Matching features (SuperGlue)...")
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

    # 4. Sparse Reconstruction
    print("\n[Step 4/4] Performing sparse reconstruction...")
    # Increase the number of registered images threshold for better results if needed
    reconstruction.main(sfm_dir, images, sfm_pairs, features, matches)

    print(f"\n✅ hloc Reconstruction ({strategy}) completed!")
    print(f"Results saved at: {sfm_dir}")

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
