"""Visual-Inertial (RGB + IMU) SfM Pipeline Module."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

# Add third_party/hloc to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "third_party" / "hloc"))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "third_party"))

import pycolmap
from hloc import extract_features, match_features, reconstruction
from hloc.utils.read_write_model import Camera, Image, Point3D, qvec2rotmat, read_model, rotmat2qvec, write_model

def parse_imu_data(imu_path: Path, imu_format: str = "euroc") -> Optional[Dict[str, np.ndarray]]:
    """Parses raw IMU measurements file into structured numpy arrays.

    Args:
        imu_path: Path to raw CSV/text IMU measurements file.
        imu_format: Format layout identifier (default: 'euroc').

    Returns:
        Optional[Dict[str, np.ndarray]]: Dictionary with 'timestamps', 'gyro', and 'acc' arrays,
            or None if parsing failed.
    """
    if not imu_path.is_file():
        print(f"[Warning] IMU data file not found at {imu_path}. Proceeding with synthetic IMU prior...")
        return None

    print(f"[VI-SfM] Parsing IMU data from {imu_path} (Format: {imu_format})...")
    try:
        timestamps, gyro, acc = [], [], []
        with open(imu_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#") or row[0].isalpha():
                    continue
                values = [float(val.strip()) for val in row if val.strip()]
                if len(values) >= 7:
                    timestamps.append(values[0])
                    gyro.append(values[1:4])
                    acc.append(values[4:7])

        if timestamps:
            print(f"[VI-SfM] Successfully loaded {len(timestamps):,} IMU measurement frames.")
            return {"timestamps": np.array(timestamps), "gyro": np.array(gyro), "acc": np.array(acc)}
        else:
            print(f"[Warning] Empty or unparseable IMU file at {imu_path}.")
            return None
    except Exception as e:
        print(f"[Error] Failed to parse IMU file: {e}")
        return None

def compute_gravity_vector(acc_data: Optional[np.ndarray]) -> np.ndarray:
    """Estimates gravity vector direction from static accelerometer measurements.

    Args:
        acc_data: Nx3 accelerometer measurements array.

    Returns:
        np.ndarray: 3D gravity acceleration vector in m/s^2.
    """
    if acc_data is None or len(acc_data) == 0:
        return np.array([0.0, 0.0, -9.81])

    mean_acc = np.mean(acc_data, axis=0)
    gravity_norm = np.linalg.norm(mean_acc)
    print(f"[VI-SfM] Estimated Gravity Vector Magnitude: {gravity_norm:.3f} m/s^2")
    return mean_acc

def align_reconstruction_to_gravity(model_path: Path, gravity_vec: np.ndarray) -> None:
    """Aligns COLMAP sparse reconstruction model (points AND cameras) to gravity vector.

    Args:
        model_path: Path to COLMAP sparse model directory containing cameras/images/points3D.
        gravity_vec: 3D gravity vector used as world vertical reference.

    Returns:
        None
    """
    has_model = (model_path / "cameras.bin").is_file() or (model_path / "cameras.txt").is_file()
    if not has_model:
        print(f"[VI-SfM] COLMAP model files not found at {model_path}, skipping gravity alignment.")
        return

    print(f"[VI-SfM] Aligning COLMAP 3D model with IMU Gravity Vector: {gravity_vec}")
    ext = ".bin" if (model_path / "cameras.bin").is_file() else ".txt"
    cameras, images, points3D = read_model(str(model_path), ext=ext)
    print(f"[VI-SfM] Loaded {len(cameras)} cameras, {len(images)} images, and {len(points3D):,} 3D points for alignment.")

    z_world = np.array([0.0, 0.0, -1.0])
    g_unit = gravity_vec / (np.linalg.norm(gravity_vec) + 1e-8)

    v = np.cross(g_unit, z_world)
    c = np.dot(g_unit, z_world)
    if np.linalg.norm(v) > 1e-6:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_align = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (np.linalg.norm(v) ** 2))
    elif c < 0:
        R_align = np.diag([1.0, -1.0, -1.0])
    else:
        R_align = np.eye(3)

    # 1. Rotate 3D points
    for p_id in points3D:
        points3D[p_id] = points3D[p_id]._replace(xyz=np.dot(R_align, points3D[p_id].xyz))

    # 2. Rotate camera orientations (world-to-camera rotation matrix R_cw' = R_cw @ R_align.T)
    for img_id in images:
        im = images[img_id]
        R_cw_new = qvec2rotmat(im.qvec) @ R_align.T
        images[img_id] = im._replace(qvec=rotmat2qvec(R_cw_new))

    write_model(cameras, images, points3D, str(model_path), ext=ext)
    print(f"[VI-SfM] Gravity alignment successfully applied to points and cameras, saved to {model_path}.")

def run_vi_sfm_pipeline(image_dir: str | Path, imu_path: str | Path, output_dir: str | Path, imu_format: str = "euroc", overwrite: bool = False) -> bool:
    """Executes Visual-Inertial (RGB + IMU) SfM reconstruction pipeline.

    Args:
        image_dir: Path to input images directory.
        imu_path: Path to input IMU data CSV/text file.
        output_dir: Target output directory for reconstructed models.
        imu_format: IMU data layout format identifier.
        overwrite: If True, clears existing features/matches and sparse model.

    Returns:
        bool: True if reconstruction and gravity alignment succeeded, False otherwise.
    """
    start_time = time.time()
    img_path = Path(image_dir).resolve()
    imu_file = Path(imu_path).resolve()
    out_path = Path(output_dir).resolve()

    out_path.mkdir(parents=True, exist_ok=True)
    sfm_dir = out_path / "sfm"

    features_path = out_path / "features.h5"
    matches_path = out_path / "matches.h5"
    pairs_path = out_path / "pairs-sequential.h5"
    hash_file = out_path / "matches.pairs.sha256"

    if overwrite:
        print("[VI-SfM] Overwrite flag set. Clearing existing H5 caches and sparse reconstruction...")
        for cache_file in [features_path, matches_path, pairs_path, hash_file]:
            if cache_file.is_file():
                cache_file.unlink()
        if sfm_dir.is_dir():
            import shutil
            shutil.rmtree(sfm_dir)

    print("==================================================")
    print(" 3DRC Visual-Inertial (RGB + IMU) SfM Pipeline")
    print("==================================================")
    print(f"Images Directory:   {img_path}")
    print(f"IMU Data File:      {imu_file}")
    print(f"Output Directory:   {out_path}")

    # 1. Parse IMU Data & Gravity Direction
    imu_data = parse_imu_data(imu_file, imu_format=imu_format)
    gravity_vec = compute_gravity_vector(imu_data["acc"] if imu_data else None)

    # 2. Run Visual Feature Extraction & Deep Matching (SuperPoint + SuperGlue)
    feature_conf = extract_features.confs["superpoint_aachen"]
    feature_conf["preprocessing"]["resize_max"] = 2400
    matcher_conf = match_features.confs["superglue"]

    if not features_path.is_file():
        print(f"\n[Step 1/3] Extracting SuperPoint Features...")
        extract_features.main(feature_conf, img_path, feature_path=features_path)

    # Generate sequential pairs for high-overlap trajectory
    image_names = sorted([f.name for f in img_path.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    pairs = []
    overlap = 10
    for i in range(len(image_names)):
        for j in range(1, overlap + 1):
            if i + j < len(image_names):
                pairs.append((image_names[i], image_names[i + j]))

    with open(pairs_path, "w", encoding="utf-8") as f:
        for p1, p2 in pairs:
            f.write(f"{p1} {p2}\n")

    pairs_hash = hashlib.sha256(pairs_path.read_bytes()).hexdigest()
    stale = (not hash_file.is_file()) or (hash_file.read_text(encoding="utf-8").strip() != pairs_hash)

    if matches_path.is_file() and not stale:
        print("\n[Step 2/3] Matches already exist and pair list unchanged. Skipping matching.")
    else:
        print(f"\n[Step 2/3] Matching Features with SuperGlue...")
        match_features.main(matcher_conf, pairs_path, features=features_path, matches=matches_path)
        hash_file.write_text(pairs_hash, encoding="utf-8")

    # 3. Incremental Visual Mapping
    target_model = sfm_dir / "0" if (sfm_dir / "0").is_dir() else sfm_dir
    has_model = (target_model / "cameras.bin").is_file() or (target_model / "cameras.txt").is_file()

    if has_model and not overwrite and not stale:
        print(f"\n[Step 3/3] Sparse reconstruction already exists at {target_model} and pair list unchanged. Skipping mapping.")
    else:
        print(f"\n[Step 3/3] Running Incremental Visual Mapping...")
        camera_model = os.environ.get("CAMERA_MODEL", "SIMPLE_RADIAL").upper()
        camera_options = pycolmap.IncrementalPipelineOptions()

        reconstruction.main(
            sfm_dir,
            img_path,
            pairs_path,
            features_path,
            matches_path,
            camera_mode=pycolmap.CameraMode.SINGLE,
            camera_model=camera_model,
            options=camera_options,
        )

    # 4. Gravity Alignment
    target_model = sfm_dir / "0" if (sfm_dir / "0").is_dir() else sfm_dir
    if (target_model / "cameras.bin").is_file() or (target_model / "cameras.txt").is_file():
        print("\n[Step 4/4] Aligning Visual Reconstruction to World Gravity Vector...")
        align_reconstruction_to_gravity(target_model, gravity_vec)

        # Write metadata record
        info_path = target_model / "sfm_info.json"
        import json
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump({
                "method": "vi_sfm",
                "gravity_vector": gravity_vec.tolist(),
                "num_images": len(image_names),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)

        elapsed = time.time() - start_time
        print(f"\n[Success] Visual-Inertial SfM Pipeline completed in {elapsed:.2f} seconds.")
        print(f"Aligned Model saved to: {target_model}")
        return True
    else:
        print("\n[Error] Visual mapping failed to generate initial reconstruction.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual-Inertial (RGB + IMU) SfM Pipeline")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to input images directory")
    parser.add_argument("--imu_path", type=str, default="data/imu.csv", help="Path to raw IMU measurements CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory for reconstruction")
    parser.add_argument("--imu_format", type=str, default="euroc", help="IMU data format (euroc, tum, or custom)")
    parser.add_argument("--overwrite", action="store_true", help="Force complete re-extraction and reconstruction")

    if not Path(args.image_dir).is_dir():
        print(f"[Error] Image directory not found: '{args.image_dir}'")
        sys.exit(1)

    ok = run_vi_sfm_pipeline(
        args.image_dir,
        args.imu_path,
        args.output_dir,
        imu_format=args.imu_format,
        overwrite=args.overwrite,
    )
    if not ok:
        sys.exit(1)
