import argparse
import os
import shutil
import subprocess
import sys

# Add fastmap directory to system path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "third_party", "fastmap"))

from fastmap.config import Config
from fastmap.engine import engine

def main():
    parser = argparse.ArgumentParser(description="FastMap Structure from Motion Pipeline")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for reconstruction")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device index")
    args = parser.parse_args()

    # Pre-clean output directory to avoid FastMap file existence error
    if os.path.exists(args.output_dir):
        print(f"Cleaning existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    # database_path should be outside output_dir to prevent deletion by shutil.rmtree
    db_dir = os.path.dirname(args.output_dir.rstrip("/"))
    db_path = os.path.join(db_dir, "database_fastmap.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    # Step 1: COLMAP Feature Extraction
    print("Running COLMAP feature extraction...")
    extractor_cmd = [
        "colmap", "feature_extractor",
        "--database_path", db_path,
        "--image_path", args.image_dir,
        "--ImageReader.single_camera", "1"
    ]
    subprocess.run(extractor_cmd, check=True)

    # Step 2: COLMAP Exhaustive Matcher
    print("Running COLMAP exhaustive matching...")
    matcher_cmd = [
        "colmap", "exhaustive_matcher",
        "--database_path", db_path
    ]
    subprocess.run(matcher_cmd, check=True)

    # Step 3: FastMap Pose Estimation & Sparse Reconstruction
    print("Running FastMap GPU-accelerated pose estimation...")
    cfg = Config()
    
    # FastMap engine handles output directory creation internally
    # Let's delete the directory just before engine start so it can create it
    shutil.rmtree(args.output_dir)

    engine(
        cfg=cfg,
        device=args.device,
        database_path=db_path,
        output_dir=args.output_dir,
        pinhole=False,
        headless=True,
        calibrated=False,
        image_dir=args.image_dir,
        gt_model_path=None
    )

    print(f"FastMap Structure from Motion completed. Outputs saved to {args.output_dir}/sparse/0")

if __name__ == "__main__":
    main()
