"""Post-processing of an SfM run: pick the largest sparse model and write sfm_info.json.

Called by scripts/01_sfm_hloc.sh after the SfM method has finished.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from src.sfm.sfm_report import IMAGE_SUFFIXES, image_group


def largest_model(sfm_dir: Path) -> Optional[Tuple[Path, int]]:
    """Finds the model with the most registered images among sfm_dir and sfm_dir/models/*.

    hloc writes the largest model to the sfm root and only the smaller ones to models/,
    so the root is checked first and wins ties.

    Args:
        sfm_dir: hloc sfm output directory.

    Returns:
        Optional[Tuple[Path, int]]: Model directory and its registered image count, or None if no model loads.
    """
    import pycolmap

    models_dir = sfm_dir / "models"
    candidates = [sfm_dir] + (sorted(p for p in models_dir.iterdir() if p.is_dir()) if models_dir.is_dir() else [])
    best: Optional[Tuple[Path, int]] = None
    for model in candidates:
        try:
            num_imgs = pycolmap.Reconstruction(str(model)).num_reg_images()
        except Exception:
            continue
        if best is None or num_imgs > best[1]:
            best = (model, num_imgs)
    return best


def _int_or_str(value: str) -> Union[int, str]:
    return int(value) if value.isdigit() else value


def build_sfm_info(image_dir: Path, method: str, strategy: str, overlap: str, retrieval_k: str) -> Dict[str, Any]:
    """Builds the sfm_info.json content for one SfM run.

    Args:
        image_dir: Input image directory of the run.
        method: SfM method name (hloc, sfm, fastmap, vggt).
        strategy: hloc pair strategy.
        overlap: hloc sequential overlap, kept as a string if it is not a number.
        retrieval_k: hloc retrieval neighbors, kept as a string if it is not a number.

    Returns:
        Dict[str, Any]: Metadata with image count and per-video image counts.
    """
    img_files = sorted(f.name for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES) if image_dir.is_dir() else []
    prefix_counts: Dict[str, int] = {}
    for name in img_files:
        prefix = image_group(name)
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

    return {
        "method": method,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": strategy,
        "overlap": _int_or_str(overlap),
        "retrieval_k": _int_or_str(retrieval_k),
        "camera_model": os.environ.get("CAMERA_MODEL", "SIMPLE_RADIAL"),
        "camera_mode": os.environ.get("CAMERA_MODE", "SINGLE"),
        "num_images_total": len(img_files),
        "video_prefixes": prefix_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SfM output post-processing")
    sub = parser.add_subparsers(dest="command", required=True)

    p_largest = sub.add_parser("largest-model", help="Print '<model_dir>|<registered images>', or 'root|<n>' when the sfm root is largest")
    p_largest.add_argument("sfm_dir", help="hloc sfm output directory (data/<scene>/cache/sfm)")

    p_info = sub.add_parser("write-info", help="Write <model_dir>/sfm_info.json")
    p_info.add_argument("model_dir", help="Method sparse directory (data/<scene>/sparse/<method>)")
    p_info.add_argument("--image_dir", required=True, help="Input image directory of the run")
    p_info.add_argument("--method", required=True, help="SfM method name")
    p_info.add_argument("--strategy", default="", help="hloc pair strategy")
    p_info.add_argument("--overlap", default="", help="hloc sequential overlap")
    p_info.add_argument("--retrieval_k", default="", help="hloc retrieval neighbors")
    args = parser.parse_args()

    if args.command == "largest-model":
        sfm_dir = Path(args.sfm_dir)
        best = largest_model(sfm_dir)
        if best is not None:
            model, num_imgs = best
            print(f"{'root' if model == sfm_dir else model}|{num_imgs}")
    else:
        info = build_sfm_info(Path(args.image_dir), args.method, args.strategy, args.overlap, args.retrieval_k)
        with open(Path(args.model_dir) / "sfm_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
