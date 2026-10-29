"""hloc (SuperPoint + SuperGlue) SfM Pipeline Module."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import torch

# Add third_party/hloc and its nested third_party to sys.path
hloc_dir = Path(__file__).resolve().parent.parent.parent / "third_party" / "hloc"
sys.path.append(str(hloc_dir))
sys.path.append(str(hloc_dir / "third_party"))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "third_party"))

import pycolmap
from hloc import extract_features, match_features, pairs_from_exhaustive, pairs_from_retrieval, reconstruction
from src.sfm.camera_utils import parse_camera_prior_from_dataset

def log_performance(image_dir: str | Path, output_dir: str | Path, strategy: str, time_extract: float, time_pairs: float, time_match: float, time_sfm: float, time_total: float) -> None:
    """Logs detailed execution timing metrics for hloc pipeline to outputs/performance_log.txt.

    Args:
        image_dir: Input images directory.
        output_dir: Output sparse model directory.
        strategy: Matching pair strategy ('sequential', 'exhaustive', or 'sequential+retrieval').
        time_extract: Duration of SuperPoint feature extraction in seconds.
        time_pairs: Duration of pair list generation in seconds.
        time_match: Duration of SuperGlue feature matching in seconds.
        time_sfm: Duration of incremental reconstruction in seconds.
        time_total: Total pipeline execution duration in seconds.

    Returns:
        None
    """
    log_dir = Path("outputs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "performance_log.txt"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n========================================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Pipeline: hloc SfM ({strategy})\n")
        f.write(f"Input: {image_dir}\n")
        f.write(f"Output: {output_dir}\n")
        f.write(f"----------------------------------------\n")
        f.write(f"Feature Extraction (SuperPoint): {time_extract:.2f} s\n")
        f.write(f"Pairs Generation: {time_pairs:.2f} s\n")
        f.write(f"Feature Matching (SuperGlue): {time_match:.2f} s\n")
        f.write(f"Sparse Reconstruction: {time_sfm:.2f} s\n")
        f.write(f"Total Elapsed Time: {time_total:.2f} s\n")
        f.write(f"========================================\n")

def group_images_by_prefix(images: List[str]) -> Dict[str, List[str]]:
    """Groups image filenames by video prefix.

    Prefix is determined by splitting before the last underscore if followed by digits,
    or before the first underscore, or grouped as 'default' if no underscore exists.

    Args:
        images: List of image file names.

    Returns:
        Dict[str, List[str]]: Dictionary mapping prefix to sorted list of image names.
    """
    groups: Dict[str, List[str]] = {}
    for img in images:
        stem = Path(img).stem
        if "_" in stem:
            parts = stem.rsplit("_", 1)
            if parts[1].isdigit():
                prefix = parts[0]
            else:
                prefix = stem.split("_", 1)[0]
        else:
            prefix = "default"
        groups.setdefault(prefix, []).append(img)

    for prefix in groups:
        groups[prefix].sort()
    return groups

def generate_sequential_pairs(image_dir: Path, output_pairs_path: Path, overlap: int = 10) -> List[Tuple[str, str]]:
    """Generates an image pair list based on sequential video frame proximity within each prefix group.

    Args:
        image_dir: Path to directory containing sequentially named frame images.
        output_pairs_path: Path to text file where pair list will be saved.
        overlap: Maximum sequential distance between paired frames (default: 10).

    Returns:
        List[Tuple[str, str]]: List of generated sequential pairs.
    """
    images = [f.name for f in image_dir.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    groups = group_images_by_prefix(images)
    print(f"Found {len(images)} images across {len(groups)} prefix group(s): {list(groups.keys())}. Generating sequential pairs with overlap {overlap}...")

    pairs: List[Tuple[str, str]] = []
    for prefix, img_list in groups.items():
        for i in range(len(img_list)):
            for j in range(1, overlap + 1):
                if i + j < len(img_list):
                    pairs.append((img_list[i], img_list[i + j]))

    with open(output_pairs_path, "w", encoding="utf-8") as f:
        for p1, p2 in pairs:
            f.write(f"{p1} {p2}\n")
    print(f"Generated {len(pairs)} sequential pairs.")
    return pairs

def generate_retrieval_pairs(images: Path, global_features_path: Path, output_pairs_path: Path, retrieval_k: int = 30) -> List[Tuple[str, str]]:
    """Generates global descriptor retrieval pairs using NetVLAD.

    Args:
        images: Path to image directory.
        global_features_path: Path to global features HDF5 file.
        output_pairs_path: Path to save temporary retrieval pair list.
        retrieval_k: Number of nearest neighbors to retrieve per image (default: 30).

    Returns:
        List[Tuple[str, str]]: List of retrieved image pairs.
    """
    global_conf = extract_features.confs["netvlad"]
    if not global_features_path.is_file():
        print(f"Extracting global descriptors (NetVLAD)...")
        extract_features.main(global_conf, images, feature_path=global_features_path)

    pairs_from_retrieval.main(
        descriptors=global_features_path,
        output=output_pairs_path,
        num_matched=retrieval_k,
    )

    pairs: List[Tuple[str, str]] = []
    if output_pairs_path.is_file():
        with open(output_pairs_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
    return pairs

def merge_and_deduplicate_pairs(pair_lists: List[List[Tuple[str, str]]], output_pairs_path: Path) -> List[Tuple[str, str]]:
    """Merges multiple lists of image pairs, normalizes pair ordering, removes duplicates, and saves to file.

    Args:
        pair_lists: List of pair lists to merge.
        output_pairs_path: Path to destination text file.

    Returns:
        List[Tuple[str, str]]: Canonical, deduplicated list of image pairs.
    """
    seen = set()
    deduped: List[Tuple[str, str]] = []
    for pairs in pair_lists:
        for p1, p2 in pairs:
            if p1 == p2:
                continue
            canonical = (min(p1, p2), max(p1, p2))
            if canonical not in seen:
                seen.add(canonical)
                deduped.append(canonical)

    with open(output_pairs_path, "w", encoding="utf-8") as f:
        for p1, p2 in deduped:
            f.write(f"{p1} {p2}\n")
    print(f"Merged and deduplicated total pairs: {len(deduped)}")
    return deduped

def run_hloc_pipeline(image_dir: str | Path, output_dir: str | Path, weights_dir: str | Path, strategy: str = "sequential", overlap: int = 10, retrieval_k: int = 30, overwrite: bool = False, reconstruct_only: bool = False) -> bool:
    """Executes the complete hloc SfM pipeline with SuperPoint and SuperGlue.

    Args:
        image_dir: Path to input images directory.
        output_dir: Path to output directory where sparse models and H5 caches are saved.
        weights_dir: Path to pretrained deep model weights directory (sets TORCH_HOME).
        strategy: Matching pair strategy ('sequential', 'exhaustive', or 'sequential+retrieval').
        overlap: Sequential matching window size.
        retrieval_k: Top-k nearest neighbors for global descriptor retrieval.
        overwrite: If True, clears existing feature/match caches and sparse models.
        reconstruct_only: If True, clears existing sfm output directory while preserving feature/match caches.

    Returns:
        bool: True if reconstruction model was successfully generated, False otherwise.
    """
    images = Path(image_dir).resolve()
    outputs = Path(output_dir).resolve()
    weights = Path(weights_dir).resolve()

    os.environ["TORCH_HOME"] = str(weights)
    outputs.mkdir(parents=True, exist_ok=True)

    sfm_pairs = outputs / f"pairs-{strategy}.txt"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"
    sfm_dir = outputs / "sfm"

    if overwrite:
        print("[hloc] Overwrite flag set. Clearing existing H5 caches and sparse reconstruction...")
        for cache_file in [features, matches, outputs / "global_features.h5", outputs / "matches.pairs.sha256"]:
            if cache_file.is_file():
                cache_file.unlink()
        if sfm_dir.is_dir():
            import shutil
            shutil.rmtree(sfm_dir)

    feature_conf = extract_features.confs["superpoint_aachen"]
    feature_conf["preprocessing"]["resize_max"] = 2400
    matcher_conf = match_features.confs["superglue"]

    print(f"--- hloc 3D Reconstruction Pipeline ({strategy}) ---")
    start_total = time.time()

    # 1. Feature Extraction
    start_time = time.time()
    if not features.is_file():
        print(f"\n[Step 1/4] Extracting features (SuperPoint, resize_max={feature_conf['preprocessing']['resize_max']})...")
        extract_features.main(feature_conf, images, feature_path=features)
        time_extract = time.time() - start_time
        print(f"Feature Extraction elapsed: {time_extract:.2f} seconds")
    else:
        print("\n[Step 1/4] Features already exist. Skipping extraction.")
        time_extract = 0.0

    # 2. Pairs Generation
    print(f"\n[Step 2/4] Generating image pairs ({strategy})...")
    start_time = time.time()
    if strategy == "exhaustive":
        pairs_from_exhaustive.main(sfm_pairs, image_list=None, features=features)
    elif strategy == "sequential":
        generate_sequential_pairs(images, sfm_pairs, overlap=overlap)
    elif strategy == "sequential+retrieval":
        temp_seq_file = outputs / "pairs-seq-temp.txt"
        temp_ret_file = outputs / "pairs-ret-temp.txt"
        global_features = outputs / "global_features.h5"

        seq_pairs = generate_sequential_pairs(images, temp_seq_file, overlap=overlap)
        ret_pairs = generate_retrieval_pairs(images, global_features, temp_ret_file, retrieval_k=retrieval_k)

        merge_and_deduplicate_pairs([seq_pairs, ret_pairs], sfm_pairs)

        temp_seq_file.unlink(missing_ok=True)
        temp_ret_file.unlink(missing_ok=True)
    else:
        print(f"[Error] Unknown strategy: {strategy}")
        return False
    time_pairs = time.time() - start_time
    print(f"Pairs Generation elapsed: {time_pairs:.2f} seconds")

    # 3. Feature Matching
    print("\n[Step 3/4] Matching features (SuperGlue)...")
    start_time = time.time()
    pairs_hash = hashlib.sha256(sfm_pairs.read_bytes()).hexdigest()
    hash_file = outputs / "matches.pairs.sha256"
    stale = (not hash_file.is_file()) or (hash_file.read_text(encoding="utf-8").strip() != pairs_hash)

    if matches.is_file() and not stale:
        print("Matches already exist and pair list unchanged. Skipping matching.")
        time_match = 0.0
    else:
        match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
        hash_file.write_text(pairs_hash, encoding="utf-8")
        time_match = time.time() - start_time
        print(f"Feature Matching elapsed: {time_match:.2f} seconds")

    # 4. Sparse Reconstruction
    is_reconstruct_only = reconstruct_only or os.environ.get("RECONSTRUCT_ONLY", "0") == "1"
    if is_reconstruct_only and sfm_dir.is_dir():
        print("[hloc] Reconstruct-only flag set. Clearing existing sfm output directory prior to mapping...")
        import shutil
        shutil.rmtree(sfm_dir)

    target_model = sfm_dir / "0" if (sfm_dir / "0").is_dir() else sfm_dir
    has_model = (target_model / "cameras.bin").is_file() or (target_model / "cameras.txt").is_file()

    if has_model and not overwrite and not stale and not is_reconstruct_only:
        print(f"\n[Step 4/4] Sparse reconstruction already exists at {target_model} and pair list unchanged. Skipping mapping.")
        time_sfm = 0.0
    else:
        print("\n[Step 4/4] Running 3D Reconstruction (COLMAP Incremental Mapper)...")
        (sfm_dir / "models").mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        mode_str = os.environ.get("CAMERA_MODE", "SINGLE").upper()
        if not hasattr(pycolmap.CameraMode, mode_str):
            valid = list(pycolmap.CameraMode.__members__.keys())
            raise ValueError(f"Invalid CAMERA_MODE '{mode_str}'. Valid choices: {valid}")
        cam_mode = getattr(pycolmap.CameraMode, mode_str)
        camera_model = os.environ.get("CAMERA_MODEL", "SIMPLE_RADIAL").upper()
        image_options = {"camera_model": camera_model}

        camera_prior = parse_camera_prior_from_dataset(images)
        if camera_prior is not None:
            image_options.update(camera_prior)
            print(f"[hloc] Injected camera prior: {camera_prior}")

        reconstruction.main(
            sfm_dir,
            images,
            sfm_pairs,
            features,
            matches,
            camera_mode=cam_mode,
            image_options=image_options,
        )
        time_sfm = time.time() - start_time
        print(f"Sparse Reconstruction elapsed: {time_sfm:.2f} seconds")

    time_total = time.time() - start_total
    print(f"\nTotal Pipeline Elapsed Time: {time_total:.2f} seconds")

    # Log performance
    log_performance(images, outputs, strategy, time_extract, time_pairs, time_match, time_sfm, time_total)

    target_model = sfm_dir / "0" if (sfm_dir / "0").is_dir() else sfm_dir
    success = (target_model / "cameras.bin").is_file() or (target_model / "cameras.txt").is_file()
    if success:
        print(f"\n[Success] Reconstruction completed successfully! Model saved at: {target_model}")
    else:
        print(f"\n[Error] Reconstruction failed to generate sparse model files.")
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hloc SuperPoint + SuperGlue Structure from Motion Pipeline")
    parser.add_argument("--image_dir", type=str, default="data/images", help="Path to input images directory")
    parser.add_argument("--output_dir", type=str, default="data/reconstruction_hloc", help="Path to output directory")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Path to deep model weights directory")
    parser.add_argument("--strategy", type=str, default="sequential", choices=["sequential", "exhaustive", "sequential+retrieval"], help="Pair matching strategy")
    parser.add_argument("--overlap", type=int, default=10, help="Overlap window for sequential matching")
    parser.add_argument("--retrieval_k", type=int, default=30, help="Top-K nearest neighbors for global descriptor retrieval matching")
    parser.add_argument("--overwrite", action="store_true", help="Force complete re-extraction and reconstruction")
    parser.add_argument("--reconstruct-only", action="store_true", help="Force re-running reconstruction while keeping feature and match caches")

    args = parser.parse_args()

    if not Path(args.image_dir).is_dir():
        print(f"[Error] Image directory not found: '{args.image_dir}'")
        sys.exit(1)

    ok = run_hloc_pipeline(
        args.image_dir,
        args.output_dir,
        args.weights_dir,
        strategy=args.strategy,
        overlap=args.overlap,
        retrieval_k=args.retrieval_k,
        overwrite=args.overwrite,
        reconstruct_only=args.reconstruct_only,
    )
    if not ok:
        sys.exit(1)
