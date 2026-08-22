"""Universal Camera Calibration Prior Parser Module.

Provides flexible detection and parsing of various camera intrinsic and distortion metadata formats
(e.g., session.json, calibration.json, transforms.json, intrinsics.json, meta.json).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def parse_camera_prior_from_dataset(dataset_dir: str | Path) -> Optional[Dict[str, str]]:
    """Inspects a dataset directory for camera intrinsic & distortion calibration files.

    Supports multiple file naming conventions and JSON schemas (e.g. Android session.json,
    OpenCV calibration.json, NeRF transforms.json, ARKit intrinsics.json).

    Args:
        dataset_dir: Path to the dataset directory (or image directory inside dataset).

    Returns:
        Optional[Dict[str, str]]: Dictionary containing camera_model and camera_params string
            if valid priors are found, or None if no valid calibration metadata exists.
    """
    path = Path(dataset_dir).resolve()
    target_dirs = [path]
    if path.parent.is_dir():
        target_dirs.append(path.parent)

    candidate_filenames = [
        "session.json",
        "calibration.json",
        "intrinsics.json",
        "transforms.json",
        "meta.json",
        "camera_intrinsics.json",
    ]

    for d in target_dirs:
        for fname in candidate_filenames:
            calib_path = d / fname
            if calib_path.is_file():
                prior = _extract_prior_from_json(calib_path)
                if prior is not None:
                    print(f"[CameraUtils] Successfully extracted camera prior from '{calib_path.name}': {prior}")
                    return prior

    return None


def _extract_prior_from_json(json_path: Path) -> Optional[Dict[str, str]]:
    """Extracts camera intrinsics and distortion from a JSON file using flexible schema matching.

    Args:
        json_path: Path to JSON calibration file.

    Returns:
        Optional[Dict[str, str]]: Dictionary with camera_model and camera_params, or None.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return None

        fx, fy, cx, cy = None, None, None, None
        k1, k2, p1, p2 = 0.0, 0.0, 0.0, 0.0

        # Schema 1: Direct intrinsics array [fx, fy, cx, cy, ...]
        if "intrinsics" in data and isinstance(data["intrinsics"], (list, tuple)) and len(data["intrinsics"]) >= 4:
            fx, fy, cx, cy = (
                float(data["intrinsics"][0]),
                float(data["intrinsics"][1]),
                float(data["intrinsics"][2]),
                float(data["intrinsics"][3]),
            )

        # Schema 2: Individual focal length & principal point keys
        elif "fl_x" in data or "fx" in data:
            fx = float(data.get("fl_x", data.get("fx", 0)))
            fy = float(data.get("fl_y", data.get("fy", fx)))
            cx = float(data.get("cx", 0))
            cy = float(data.get("cy", 0))

        # Schema 3: 3x3 Camera matrix
        elif "camera_matrix" in data or "K" in data:
            matrix = data.get("camera_matrix", data.get("K"))
            if isinstance(matrix, list) and len(matrix) == 3:
                fx = float(matrix[0][0])
                fy = float(matrix[1][1])
                cx = float(matrix[0][2])
                cy = float(matrix[1][2])

        if fx is None or fy is None or fx <= 0 or fy <= 0:
            return None

        # Extract Distortion Coefficients
        if "distortion" in data and isinstance(data["distortion"], (list, tuple)):
            dist = data["distortion"]
            if len(dist) > 0:
                k1 = float(dist[0])
            if len(dist) > 1:
                k2 = float(dist[1])
            if len(dist) > 3:
                p1 = float(dist[3])
            if len(dist) > 4:
                p2 = float(dist[4])

        elif "distortion_coefficients" in data and isinstance(data["distortion_coefficients"], (list, tuple)):
            dist = data["distortion_coefficients"]
            if len(dist) > 0:
                k1 = float(dist[0])
            if len(dist) > 1:
                k2 = float(dist[1])
            if len(dist) > 2:
                p1 = float(dist[2])
            if len(dist) > 3:
                p2 = float(dist[3])

        elif "k1" in data:
            k1 = float(data.get("k1", 0.0))
            k2 = float(data.get("k2", 0.0))
            p1 = float(data.get("p1", 0.0))
            p2 = float(data.get("p2", 0.0))

        camera_params_str = f"{fx:.6f},{fy:.6f},{cx:.6f},{cy:.6f},{k1:.6f},{k2:.6f},{p1:.6f},{p2:.6f}"
        return {
            "camera_model": "OPENCV",
            "camera_params": camera_params_str,
        }

    except Exception:
        return None
