#!/usr/bin/env python3
"""Subsample dataset images by picking every Nth frame for quick smoke testing."""

import argparse
import os
import shutil
import sys

def subsample(src_dir: str, dst_dir: str, step: int = 10):
    src_raw = os.path.join(src_dir, "raw_images")
    if not os.path.exists(src_raw):
        src_raw = os.path.join(src_dir, "images")
    if not os.path.exists(src_raw):
        print(f"[Error] No images directory found in {src_dir}")
        sys.exit(1)

    dst_raw = os.path.join(dst_dir, "raw_images")
    # Clean destination raw_images to avoid stale frames from previous runs
    if os.path.exists(dst_raw):
        shutil.rmtree(dst_raw)
    os.makedirs(dst_raw, exist_ok=True)

    files = sorted([f for f in os.listdir(src_raw) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not files:
        print(f"[Error] No image files found in {src_raw}")
        sys.exit(1)

    selected = files[::step]
    print(f"==> Subsampling {len(files)} total frames with step {step} -> {len(selected)} frames selected.")

    for f in selected:
        src_file = os.path.join(src_raw, f)
        dst_file_raw = os.path.join(dst_raw, f)
        shutil.copy2(src_file, dst_file_raw)

    print(f"[SUCCESS] Subsampled dataset created at {dst_dir} ({len(selected)} frames in raw_images)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample dataset images for fast smoke testing.")
    parser.add_argument("--src", default="data/20260429_140922", help="Source dataset directory")
    parser.add_argument("--dst", default="data/test/smoke_test_100", help="Destination dataset directory")
    parser.add_argument("--step", type=int, default=10, help="Subsampling step interval (default: 10)")
    args = parser.parse_args()

    subsample(args.src, args.dst, args.step)
