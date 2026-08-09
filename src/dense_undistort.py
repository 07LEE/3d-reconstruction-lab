import pycolmap
import argparse
import shutil
from pathlib import Path
import sys
import time

def run_undistortion(input_dir: str, image_dir: str, output_dir: str,
                     copy_policy: str = "hardlink",
                     num_patch_match_src_images: int = 0,
                     max_image_size: int = -1,
                     blank_pixels: float = 0.0) -> bool:
    """Undistorts images and creates a workspace for dense reconstruction with high-throughput optimizations.

    Args:
        input_dir: Path to the sparse reconstruction folder (containing .bin/.ply files).
        image_dir: Path to the original image directory.
        output_dir: Path where undistorted images and dense workspace will be saved.
        copy_policy: Image linkage policy ('hardlink', 'softlink', 'copy').
        num_patch_match_src_images: Number of overlapping source images for dense stereo.
                                   Setting to 0 bypasses costly MVS pair graph construction for 3DGS/MILo.
        max_image_size: Maximum resolution bound (-1 for native resolution).
        blank_pixels: Ratio of blank pixels to tolerate in undistorted frames.

    Returns:
        bool: True if successful, False otherwise.
    """
    input_path = Path(input_dir)
    image_path = Path(image_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    print("--- High-Throughput Image Undistortion Started ---")
    print(f"Sparse Model Path: {input_path}")
    print(f"Original Image Path: {image_path}")
    print(f"Output Workspace Path: {output_path}")
    print(f"Copy Policy: {copy_policy} | PatchMatch Overlap: {num_patch_match_src_images} | Max Size: {max_image_size}")

    # Verify input sparse model exists
    has_sparse_files = (input_path / "cameras.bin").exists() or (input_path / "cameras.txt").exists()
    if not has_sparse_files:
        print(f"[Error] Sparse reconstruction files not found in {input_path}")
        return False

    policy_map = {
        "hardlink": pycolmap.CopyType.hardlink,
        "softlink": pycolmap.CopyType.softlink,
        "copy": pycolmap.CopyType.copy,
    }
    selected_policy = policy_map.get(copy_policy.lower(), pycolmap.CopyType.hardlink)

    undistort_opts = pycolmap.UndistortCameraOptions()
    undistort_opts.max_image_size = max_image_size
    undistort_opts.blank_pixels = blank_pixels

    t0 = time.time()
    try:
        print("\n[Step] Processing image undistortion via PyCOLMAP...")
        try:
            pycolmap.undistort_images(
                output_path=str(output_path),
                input_path=str(input_path),
                image_path=str(image_path),
                copy_policy=selected_policy,
                num_patch_match_src_images=num_patch_match_src_images,
                undistort_options=undistort_opts,
            )
        except Exception as link_err:
            if selected_policy != pycolmap.CopyType.copy:
                print(f"[Warn] {copy_policy} failed ({link_err}), falling back to copy policy...")
                pycolmap.undistort_images(
                    output_path=str(output_path),
                    input_path=str(input_path),
                    image_path=str(image_path),
                    copy_policy=pycolmap.CopyType.copy,
                    num_patch_match_src_images=num_patch_match_src_images,
                    undistort_options=undistort_opts,
                )
            else:
                raise link_err

        # Standardize sparse/0 directory hierarchy for 3DGS, MILo, and 2DGS readers
        sparse_dir = output_path / "sparse"
        sparse_zero = sparse_dir / "0"
        if sparse_dir.exists() and not sparse_zero.exists():
            sparse_zero.mkdir(parents=True, exist_ok=True)
            for f in sparse_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, sparse_zero / f.name)

        elapsed = time.time() - t0
        print(f"\n[Success] Undistortion completed in {elapsed:.2f} seconds!")
        print(f"Dense workspace prepared at: {output_path}")
        return True

    except Exception as e:
        print(f"\n[Error] Undistortion failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Throughput Image Undistortion for Dense 3D Reconstruction")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to sparse model directory containing .bin/.txt files")
    parser.add_argument("--image_dir", type=str, required=True, help="Original image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Dense workspace output path")
    parser.add_argument("--copy_policy", type=str, default="hardlink", choices=["hardlink", "softlink", "copy"],
                        help="Policy for linking unchanged frames (default: hardlink for zero-IO latency)")
    parser.add_argument("--num_patch_match", type=int, default=0,
                        help="Overlapping frames for MVS dense stereo (default: 0 to bypass MVS graph for 3DGS/MILo)")
    parser.add_argument("--max_image_size", type=int, default=-1, help="Max image dimension bound (default: -1)")
    parser.add_argument("--blank_pixels", type=float, default=0.0, help="Tolerated blank pixels ratio")

    args = parser.parse_args()

    if not Path(args.input_dir).exists():
        print(f"[Error] Sparse model directory '{args.input_dir}' not found.")
        sys.exit(1)

    if not Path(args.image_dir).exists():
        print(f"[Error] Image directory '{args.image_dir}' not found.")
        sys.exit(1)

    success = run_undistortion(
        input_dir=args.input_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        copy_policy=args.copy_policy,
        num_patch_match_src_images=args.num_patch_match,
        max_image_size=args.max_image_size,
        blank_pixels=args.blank_pixels
    )
    if not success:
        sys.exit(1)
