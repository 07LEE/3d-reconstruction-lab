"""FastMap GPU-accelerated SfM Pipeline Module."""

from __future__ import annotations

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

def run_fastmap_pipeline(image_dir: str | Path, output_dir: str | Path, device: str = "cuda:0", overwrite: bool = False) -> bool:
    """Executes FastMap GPU-accelerated SfM pipeline.

    Args:
        image_dir: Directory containing input frame images.
        output_dir: Target output directory for reconstructed models.
        device: GPU device string (default: 'cuda:0').
        overwrite: If True, clears existing database and model outputs.

    Returns:
        bool: True if FastMap sparse reconstruction completed successfully, False otherwise.
    """
    img_path = Path(image_dir).resolve()
    out_path = Path(output_dir).resolve()

    target_model = out_path / "0" if (out_path / "0").is_dir() else out_path
    has_model = (target_model / "cameras.bin").is_file() or (target_model / "cameras.txt").is_file()
    if has_model and not overwrite:
        print(f"[FastMap] Sparse reconstruction already exists at {target_model}. Skipping computation.")
        return True

    if out_path.is_dir():
        print(f"[FastMap] Cleaning existing output directory: {out_path}")
        shutil.rmtree(out_path)

    out_path.mkdir(parents=True, exist_ok=True)
    db_dir = out_path.parent
    db_path = db_dir / "database_fastmap.db"
    if db_path.is_file():
        db_path.unlink()

    colmap_bin = shutil.which("colmap") or "colmap"

    try:
        # Step 1: COLMAP Feature Extraction
        print("[FastMap] Running COLMAP feature extraction...")
        camera_model = os.environ.get("CAMERA_MODEL", "SIMPLE_RADIAL").upper()
        single_camera = "1" if os.environ.get("CAMERA_MODE", "SINGLE").upper() == "SINGLE" else "0"
        extractor_cmd = [
            colmap_bin, "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(img_path),
            "--ImageReader.single_camera", single_camera,
            "--ImageReader.camera_model", camera_model,
        ]
        subprocess.run(extractor_cmd, check=True)

        # Step 2: COLMAP Exhaustive Matcher
        print("[FastMap] Running COLMAP exhaustive matching...")
        matcher_cmd = [
            colmap_bin, "exhaustive_matcher",
            "--database_path", str(db_path),
        ]
        subprocess.run(matcher_cmd, check=True)

        # Step 3: FastMap Pose Estimation & Sparse Reconstruction
        print("[FastMap] Running FastMap GPU-accelerated pose estimation...")
        cfg = Config()

        engine(
            cfg=cfg,
            device=device,
            database_path=str(db_path),
            output_dir=str(out_path),
            pinhole=(camera_model == "PINHOLE"),
            headless=True,
            calibrated=False,
            image_dir=str(img_path),
            gt_model_path=None,
        )

        print(f"[Success] FastMap Structure from Motion completed. Outputs saved to {out_path}/sparse/0")
        return True

    except Exception as e:
        print(f"[Error] FastMap execution failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastMap Structure from Motion Pipeline")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for reconstruction")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device index")
    parser.add_argument("--overwrite", action="store_true", help="Force complete re-extraction and reconstruction")

    args = parser.parse_args()

    if not Path(args.image_dir).is_dir():
        print(f"[Error] Image directory not found: '{args.image_dir}'")
        sys.exit(1)

    ok = run_fastmap_pipeline(args.image_dir, args.output_dir, device=args.device, overwrite=args.overwrite)
    if not ok:
        sys.exit(1)
