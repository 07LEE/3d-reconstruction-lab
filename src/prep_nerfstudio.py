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

    print(f"\n✅ Data prepared at: {output_dir}")
    print(f"Now run: ./ns_run.sh train splatfacto --data {output_dir}")

if __name__ == "__main__":
    # Based on your current directory structure
    prepare_nerfstudio_data(
        image_dir="data/images", 
        sfm_model_dir="data/hloc_reconstruction/sfm/models/0", 
        output_dir="data/nerfstudio_data"
    )
