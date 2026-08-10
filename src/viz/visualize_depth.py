from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np

def read_colmap_depth_map(path: str | Path) -> Optional[np.ndarray]:
    """Reads a COLMAP binary depth map file with ASCII header.

    Args:
        path: Path to the binary depth map file (.geometric.bin or .photometric.bin).

    Returns:
        np.ndarray: Loaded depth array of shape (height, width, channels), 
            or None if parsing failed.
    """
    file_path = Path(path)
    try:
        with open(file_path, "rb") as fid:
            header = b""
            while True:
                char = fid.read(1)
                if not char:
                    break
                header += char
                if header.count(b"&") == 3:
                    break

            parts = header.decode().strip().split("&")
            width, height, channels = map(int, parts[:3])
            array = np.fromfile(fid, dtype=np.float32)

        if array.size != width * height * channels:
            return None

        return array.reshape((height, width, channels), order="C")
    except Exception as e:
        print(f"[Error] Failed to read {file_path}: {e}")
        return None

def visualize_depth_maps(depth_dir: str | Path, output_dir: str | Path) -> bool:
    """Converts COLMAP binary depth maps to normalized colorized PNG images.

    Args:
        depth_dir: Directory containing input COLMAP .bin depth maps.
        output_dir: Target directory where rendered PNG color maps will be saved.

    Returns:
        bool: True if at least one depth map was successfully converted, False otherwise.
    """
    depth_path = Path(depth_dir)
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    bin_files = list(depth_path.glob("*.geometric.bin"))
    if not bin_files:
        bin_files = list(depth_path.glob("*.bin"))
    print(f"[DepthViz] Found {len(bin_files)} depth maps to visualize.")

    converted = 0
    for bin_file in bin_files:
        depth = read_colmap_depth_map(bin_file)
        if depth is None:
            continue

        depth_2d = depth.squeeze()
        valid_depth = depth_2d[depth_2d > 0]
        if valid_depth.size == 0:
            continue

        min_depth = float(np.percentile(valid_depth, 2))
        max_depth = float(np.percentile(valid_depth, 98))
        if max_depth <= min_depth:
            max_depth = min_depth + 1e-6

        depth_norm = np.clip((depth_2d - min_depth) / (max_depth - min_depth), 0, 1)
        depth_viz = plt.get_cmap("magma")(1.0 - depth_norm)
        depth_viz = (depth_viz[:, :, :3] * 255).astype(np.uint8)
        depth_viz = cv2.cvtColor(depth_viz, cv2.COLOR_RGB2BGR)

        output_file = save_path / f"{bin_file.stem}.png"
        cv2.imwrite(str(output_file), depth_viz)
        converted += 1

    print(f"[Success] Converted {converted}/{len(bin_files)} depth maps to: {save_path}")
    return converted > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize COLMAP depth maps (.bin) to images (.png)")
    parser.add_argument("--depth_dir", type=str, required=True, help="Path to COLMAP depth maps directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save visualized PNG images")

    args = parser.parse_args()
    visualize_depth_maps(args.depth_dir, args.output_dir)
