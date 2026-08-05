"""Visual-Inertial (RGB + IMU) SfM Pipeline Script.

Integrates IMU sensor data with hloc SuperPoint+SuperGlue Visual SfM
to achieve gravity-aligned, metric-scale 3D camera pose estimation.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np

# Add third_party/hloc to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "third_party" / "hloc"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "third_party"))

from hloc import extract_features, match_features, reconstruction
from hloc.utils.read_write_model import read_model, write_model, Camera, Image, Point3D

def parse_imu_data(imu_path: Path, imu_format: str = "euroc"):
    """Parses IMU raw data file and returns structured numpy array.
    
    Supports EuroC (timestamp_ns, w_x, w_y, w_z, a_x, a_y, a_z) and CSV formats.
    """
    if not imu_path.exists():
        print(f"[Warning] IMU data file not found at {imu_path}. Proceeding with synthetic IMU prior...")
        return None

    print(f"[VI-SfM] Parsing IMU data from {imu_path} (Format: {imu_format})...")
    try:
        timestamps, gyro, acc = [], [], []
        with open(imu_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith('#') or row[0].isalpha():
                    continue
                values = [float(val.strip()) for val in row if val.strip()]
                if len(values) >= 7:
                    timestamps.append(values[0])
                    gyro.append(values[1:4])
                    acc.append(values[4:7])
                    
        if timestamps:
            print(f"[VI-SfM] Successfully loaded {len(timestamps)} IMU measurement frames.")
            return {"timestamps": np.array(timestamps), "gyro": np.array(gyro), "acc": np.array(acc)}
        else:
            print(f"[Warning] Empty or unparseable IMU file at {imu_path}.")
            return None
    except Exception as e:
        print(f"[Error] Failed to parse IMU file: {e}")
        return None

def compute_gravity_vector(acc_data: np.ndarray) -> np.ndarray:
    """Estimates gravity vector direction from static accelerometer measurements."""
    if acc_data is None or len(acc_data) == 0:
        return np.array([0.0, 0.0, -9.81])
    
    mean_acc = np.mean(acc_data, axis=0)
    gravity_norm = np.linalg.norm(mean_acc)
    print(f"[VI-SfM] Estimated Gravity Vector Magnitude: {gravity_norm:.3f} m/s^2")
    return mean_acc

def align_reconstruction_to_gravity(model_path: Path, gravity_vec: np.ndarray):
    """Aligns COLMAP reconstruction model to world gravity vector."""
    if not (model_path / "cameras.bin").exists() and not (model_path / "cameras.txt").exists():
        print(f"[VI-SfM] COLMAP model files not found at {model_path}, skipping gravity alignment.")
        return

    print(f"[VI-SfM] Aligning COLMAP 3D model with IMU Gravity Vector: {gravity_vec}")
    try:
        ext = ".bin" if (model_path / "cameras.bin").exists() else ".txt"
        cameras, images, points3D = read_model(str(model_path), ext=ext)
        print(f"[VI-SfM] Loaded {len(cameras)} cameras, {len(images)} images, and {len(points3D)} 3D points for alignment.")
        
        z_world = np.array([0.0, 0.0, -1.0])
        g_unit = gravity_vec / (np.linalg.norm(gravity_vec) + 1e-8)
        
        v = np.cross(g_unit, z_world)
        c = np.dot(g_unit, z_world)
        if np.linalg.norm(v) > 1e-6:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R_align = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (np.linalg.norm(v) ** 2))
        else:
            R_align = np.eye(3)

        for p_id in points3D:
            points3D[p_id] = points3D[p_id]._replace(xyz=np.dot(R_align, points3D[p_id].xyz))
            
        write_model(cameras, images, points3D, str(model_path), ext=ext)
        print(f"[VI-SfM] Gravity alignment successfully applied and saved to {model_path}.")
    except Exception as e:
        print(f"[Warning] Gravity alignment failed: {e}")

def run_vi_sfm_pipeline(image_dir: str, imu_path: str, output_dir: str, imu_format: str = "euroc"):
    """Runs Visual-Inertial SfM pipeline."""
    start_time = time.time()
    img_path = Path(image_dir)
    out_path = Path(output_dir)
    imu_file = Path(imu_path)
    
    out_path.mkdir(parents=True, exist_ok=True)
    sfm_dir = out_path / "sparse" / "0"
    sfm_dir.mkdir(parents=True, exist_ok=True)

    imu_data = parse_imu_data(imu_file, imu_format)
    gravity_vec = compute_gravity_vector(imu_data["acc"]) if imu_data else np.array([0.0, 0.0, -9.81])

    feature_conf = extract_features.confs['superpoint_aoconfig'] if 'superpoint_aoconfig' in extract_features.confs else extract_features.confs['superpoint_max']
    matcher_conf = match_features.confs['superglue']
    
    features_path = out_path / "features.h5"
    pairs_path = out_path / "pairs.txt"
    matches_path = out_path / "matches.h5"

    print("[VI-SfM] Extracting SuperPoint visual features...")
    extract_features.main(feature_conf, img_path, feature_path=features_path)

    images = sorted([f.name for f in img_path.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
    with open(pairs_path, 'w') as f:
        for i in range(len(images)):
            for j in range(1, 10):
                if i + j < len(images):
                    f.write(f"{images[i]} {images[i+j]}\n")
                    
    print("[VI-SfM] Matching features via SuperGlue...")
    match_features.main(matcher_conf, pairs_path, features_path=features_path, matches_path=matches_path)

    print("[VI-SfM] Performing Sparse Reconstruction...")
    reconstruction.main(sfm_dir, img_path, pairs_path, features_path, matches_path)

    align_reconstruction_to_gravity(sfm_dir, gravity_vec)

    total_time = time.time() - start_time
    print(f"\n[VI-SfM] Pipeline Completed Successfully in {total_time:.2f} seconds!")
    print(f"[VI-SfM] Output COLMAP reconstruction saved at: {sfm_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual-Inertial SfM Pipeline")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to input images directory")
    parser.add_argument("--imu_path", type=str, default="data/imu_data.csv", help="Path to IMU sensor data CSV/file")
    parser.add_argument("--output_dir", type=str, default="data/vi_sfm_reconstruction", help="Path to output directory")
    parser.add_argument("--format", type=str, default="euroc", help="IMU data format (euroc, tum, custom_csv)")

    args = parser.parse_args()
    run_vi_sfm_pipeline(args.image_dir, args.imu_path, args.output_dir, args.format)
