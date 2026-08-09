import os
from pathlib import Path

def prepare_nerfstudio_data(image_dir, sfm_model_dir, output_dir):
    """Prepares data in the format expected by nerfstudio using symbolic links.
    
    Structure:
    output_dir/
      images/ -> link to image_dir
      colmap/
        sparse/
          0/ -> link to sfm_model_dir
    """
    output_path = Path(output_dir)
    colmap_sparse_path = output_path / "colmap" / "sparse" / "0"
    images_dest_path = output_path / "images"
    
    # Create directories
    colmap_sparse_path.mkdir(parents=True, exist_ok=True)
    
    # Create symbolic link for images
    if not images_dest_path.exists():
        print(f"Linking images: {image_dir} -> {images_dest_path}")
        os.symlink(Path(image_dir).absolute(), images_dest_path.absolute(), target_is_directory=True)
    
    # Create symbolic link for sparse model (link the contents of models/0 to colmap/sparse/0)
    print(f"Linking sparse model: {sfm_model_dir} -> {colmap_sparse_path}")
    for file in Path(sfm_model_dir).iterdir():
        dest_file = colmap_sparse_path / file.name
        if dest_file.exists():
            dest_file.unlink()
        os.symlink(file.absolute(), dest_file.absolute())

    print(f"\nData prepared at: {output_dir}")
    print(f"To train with nerfstudio, run: ns-train splatfacto --data {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare dataset structure for Nerfstudio / Splatfacto")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to input images directory")
    parser.add_argument("--sfm_model_dir", type=str, required=True, help="Path to sparse SfM model directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for nerfstudio dataset")

    args = parser.parse_args()
    prepare_nerfstudio_data(args.image_dir, args.sfm_model_dir, args.output_dir)
