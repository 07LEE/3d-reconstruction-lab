import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def read_colmap_depth_map(path):
    """Reads a COLMAP binary depth map with ASCII header."""
    try:
        with open(path, "rb") as fid:
            header = b""
            while True:
                char = fid.read(1)
                if not char: break
                header += char
                if header.count(b"&") == 3: break
            
            parts = header.decode().strip().split("&")
            width, height, channels = map(int, parts[:3])
            array = np.fromfile(fid, dtype=np.float32)
            
        if array.size != width * height * channels:
            return None
            
        return array.reshape((height, width, channels), order="C")
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def visualize_depth_maps(depth_dir, output_dir):
    depth_path = Path(depth_dir)
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of geometric.bin files
    bin_files = list(depth_path.glob("*.geometric.bin"))
    print(f"Found {len(bin_files)} depth maps to visualize.")
    
    for bin_file in bin_files:
        depth = read_colmap_depth_map(bin_file)
        if depth is None: continue
        
        # Flatten depth to 2D
        depth_2d = depth.squeeze()
        
        # Filter invalid values (0 or very large values)
        # Use 2nd and 98th percentile for robust normalization
        valid_depth = depth_2d[depth_2d > 0]
        if valid_depth.size == 0: continue
        
        min_depth = np.percentile(valid_depth, 2)
        max_depth = np.percentile(valid_depth, 98)
        
        # Normalize to [0, 1]
        depth_norm = (depth_2d - min_depth) / (max_depth - min_depth)
        depth_norm = np.clip(depth_norm, 0, 1)
        
        # Map to colormap (Magma or Jet)
        # We use matplotlib's colormap for better visualization
        depth_viz = plt.get_cmap('magma')(1.0 - depth_norm) # Closer is brighter
        depth_viz = (depth_viz[:, :, :3] * 255).astype(np.uint8)
        depth_viz = cv2.cvtColor(depth_viz, cv2.COLOR_RGB2BGR)
        
        # Save as PNG
        output_file = save_path / f"{bin_file.stem}.png"
        cv2.imwrite(str(output_file), depth_viz)
        
    print(f"Successfully visualized depth maps to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize COLMAP depth maps (.bin) to images (.png)")
    parser.add_argument("--depth_dir", type=str, default="data/reconstruction/dense/stereo/depth_maps", help="Path to COLMAP depth maps")
    parser.add_argument("--output_dir", type=str, default="data/reconstruction/dense/stereo/depth_visualized", help="Path to save visualized images")
    
    args = parser.parse_args()
    
    visualize_depth_maps(args.depth_dir, args.output_dir)
