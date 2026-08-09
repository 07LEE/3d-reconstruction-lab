from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

def prepare_nerfstudio_data(image_dir: str | Path,
                            sfm_model_dir: str | Path,
                            output_dir: str | Path) -> bool:
    """Prepares standard dataset hierarchy expected by Nerfstudio / Splatfacto.

    Target Structure:
        output_dir/
          images/ -> link/copy to image_dir
          colmap/
            sparse/
              0/ -> link/copy to sfm_model_dir (.bin or .txt files)

    Args:
        image_dir: Path to the input images directory.
        sfm_model_dir: Path to the sparse SfM model directory.
        output_dir: Target output directory for Nerfstudio workspace.

    Returns:
        bool: True if dataset setup succeeded, False otherwise.
    """
    img_p = Path(image_dir).resolve()
    sfm_p = Path(sfm_model_dir).resolve()
    out_p = Path(output_dir).resolve()

    if not img_p.is_dir():
        print(f"[Error] Image directory not found: {img_p}")
        return False
    if not sfm_p.is_dir():
        print(f"[Error] Sparse SfM model directory not found: {sfm_p}")
        return False

    colmap_sparse_path = out_p / "colmap" / "sparse" / "0"
    images_dest_path = out_p / "images"

    colmap_sparse_path.mkdir(parents=True, exist_ok=True)

    # 1. Link or Copy Images
    if not images_dest_path.exists():
        print(f"[PrepNerfstudio] Linking images: {img_p} -> {images_dest_path}")
        try:
            images_dest_path.symlink_to(img_p, target_is_directory=True)
        except OSError:
            print("[PrepNerfstudio] Symlink failed; copying image directory...")
            shutil.copytree(img_p, images_dest_path)

    # 2. Link or Copy Sparse Reconstruction Files
    print(f"[PrepNerfstudio] Linking sparse model: {sfm_p} -> {colmap_sparse_path}")
    for file in sfm_p.iterdir():
        if file.is_file():
            dest_file = colmap_sparse_path / file.name
            if dest_file.exists() or dest_file.is_symlink():
                dest_file.unlink()
            try:
                dest_file.symlink_to(file)
            except OSError:
                shutil.copy2(file, dest_file)

    print(f"\n[Success] Nerfstudio dataset prepared at: {out_p}")
    print(f"To train with Nerfstudio, run: ns-train splatfacto --data {out_p}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset structure for Nerfstudio / Splatfacto")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to input images directory")
    parser.add_argument("--sfm_model_dir", type=str, required=True, help="Path to sparse SfM model directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for nerfstudio dataset")

    args = parser.parse_args()
    ok = prepare_nerfstudio_data(args.image_dir, args.sfm_model_dir, args.output_dir)
    if not ok:
        sys.exit(1)
