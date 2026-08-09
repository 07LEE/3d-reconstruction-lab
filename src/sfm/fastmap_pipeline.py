"""FastMap GPU-accelerated SfM Pipeline Module."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add fastmap directory to system path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "third_party" / "fastmap"))

from fastmap.config import Config
from fastmap.engine import engine

def run_fastmap_pipeline(image_dir: str, output_dir: str, device: str = "cuda:0"):
    """Executes FastMap GPU-accelerated SfM pipeline."""
    if os.path.exists(output_dir):
        print(f"Cleaning existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    db_dir = os.path.dirname(output_dir.rstrip("/"))
    db_path = os.path.join(db_dir, "database_fastmap.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    # Step 1: COLMAP Feature Extraction
    print("Running COLMAP feature extraction...")
    extractor_cmd = [
        "colmap", "feature_extractor",
        "--database_path", db_path,
        "--image_path", image_dir,
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
    os.makedirs(output_dir, exist_ok=True)

    engine(
        cfg=cfg,
        device=device,
        database_path=db_path,
        output_dir=output_dir,
        pinhole=False,
        headless=True,
        calibrated=False,
        image_dir=image_dir,
        gt_model_path=None
    )

    print(f"FastMap Structure from Motion completed. Outputs saved to {output_dir}/sparse/0")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastMap Structure from Motion Pipeline")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for reconstruction")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device index")
    args = parser.parse_args()

    run_fastmap_pipeline(args.image_dir, args.output_dir, args.device)
