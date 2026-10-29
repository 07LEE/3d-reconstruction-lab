"""In-Memory YUV Pipeline Orchestrator for 3DRC.

Integrates yuv_sensor package directly into PyTorch SfM and 3DGS pipelines,
enabling zero-disk-writing in-memory frame decoding and feature extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from yuv_sensor import SessionDataLoader, CameraCalibration
    HAS_YUV_SENSOR = True
except ImportError:
    HAS_YUV_SENSOR = False
    SessionDataLoader = None
    CameraCalibration = None


class YUVInMemoryDataset(Dataset):
    """PyTorch Dataset that streams decoded YUV frames directly into in-memory RGB Tensors."""

    def __init__(
        self,
        session_dir: str | Path,
        apply_undistort: bool = True,
        transform: Optional[Any] = None,
    ) -> None:
        if not HAS_YUV_SENSOR:
            raise ImportError(
                "yuv_sensor package is not installed. Install via: "
                "pip install git+https://github.com/07LEE/yuv-sensor-data-processor.git"
            )

        self.session_dir = Path(session_dir).resolve()
        self.apply_undistort = apply_undistort
        self.transform = transform

        self.loader = SessionDataLoader(self.session_dir)
        self.num_frames = len(self.loader.frames_df)

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Frame index {idx} out of range (0-{self.num_frames - 1})")

        rgb_array = self.loader.get_decoded_frame(idx, apply_undistort=self.apply_undistort)
        frame_row = self.loader.frames_df.iloc[idx]
        timestamp_ns = int(frame_row["timestamp_ns"])
        filename = str(frame_row["filename"])

        tensor_img = torch.from_numpy(rgb_array).permute(2, 0, 1).float() / 255.0

        if self.transform is not None:
            tensor_img = self.transform(tensor_img)

        return {
            "image": tensor_img,
            "timestamp_ns": timestamp_ns,
            "frame_idx": idx,
            "filename": filename,
        }

    def get_camera_intrinsics(self) -> Dict[str, Any]:
        return {
            "intrinsics": self.loader.session_meta.get("intrinsics", []),
            "distortion": self.loader.session_meta.get("distortion", []),
            "sensor_orientation": self.loader.session_meta.get("sensor_orientation", 0),
            "device": self.loader.session_meta.get("device", "unknown"),
        }


class YUVInMemoryPipelineManager:
    """Orchestrates zero-disk-I/O in-memory YUV frame streaming for SfM and 3DGS."""

    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir).resolve()
        print(f"[YUV Pipeline] Initializing In-Memory Dataset for session: {self.session_dir.name}...")
        self.dataset = YUVInMemoryDataset(self.session_dir, apply_undistort=True)
        print(f"[YUV Pipeline] Loaded {len(self.dataset)} frames into memory streaming buffer.")

    def get_frame_tensor(self, idx: int) -> torch.Tensor:
        sample = self.dataset[idx]
        return sample["image"]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        session_path = sys.argv[1]
    else:
        session_path = "data/session_419864820"

    manager = YUVInMemoryPipelineManager(session_path)
    sample = manager.dataset[0]
    print(f"[Test OK] Sample 0 loaded from GitHub yuv_sensor: Tensor Shape = {sample['image'].shape}, Timestamp = {sample['timestamp_ns']} ns")
