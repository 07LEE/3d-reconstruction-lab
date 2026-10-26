"""Sim(3) Alignment Between a Reconstruction Frame and a Ground Truth Frame.

SfM reconstructions live in an arbitrary frame (rotation, translation, and scale) relative to
LiDAR ground truth. This module estimates one 7-DoF transform per scene from a reference point
set (COLMAP sparse points by default), so every mesh of the scene is scored in the same frame.

Global initialization enumerates the 24 proper rotations that map the source PCA axes onto the
target PCA axes, refines each with scale-aware ICP, and keeps the best-fitting hypothesis.
Pure NumPy/SciPy so it runs without Open3D or a GPU.
"""

from __future__ import annotations

import itertools
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

@dataclass
class AlignmentResult:
    transform: np.ndarray
    scale: float
    fitness: float
    inlier_rmse: float
    threshold: float
    ambiguous: bool = False
    candidates: List[Tuple[float, float]] = field(default_factory=list)

def read_colmap_points3d(path: str | Path, min_track_length: int = 3) -> np.ndarray:
    """Reads COLMAP sparse points from a model directory, points3D.bin, or points3D.txt.

    Points observed in fewer than `min_track_length` images are dropped as likely outliers.
    """
    path = Path(path)
    if path.is_dir():
        for name in ("points3D.bin", "points3D.txt"):
            if (path / name).is_file():
                path = path / name
                break
        else:
            raise FileNotFoundError(f"No points3D.bin or points3D.txt in {path}")

    points = []
    if path.suffix == ".bin":
        with open(path, "rb") as f:
            num_points = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_points):
                # id (uint64), xyz (3 x double), rgb (3 x uint8), error (double), track length (uint64)
                _, x, y, z, _, _, _, _, track_length = struct.unpack("<QdddBBBdQ", f.read(51))
                f.seek(8 * track_length, 1)
                if track_length >= min_track_length:
                    points.append((x, y, z))
    else:
        with open(path, "r") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                tokens = line.split()
                track_length = (len(tokens) - 8) // 2
                if track_length >= min_track_length:
                    points.append(tuple(float(v) for v in tokens[1:4]))
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)

def apply_transform(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]

def transform_scale(transform: np.ndarray) -> float:
    return float(np.cbrt(np.linalg.det(transform[:3, :3])))

def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Closed-form least-squares Sim(3) mapping src onto dst (Umeyama, 1991)."""
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = dst_c.T @ src_c / len(src)
    u, d, vt = np.linalg.svd(cov)
    s = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1] = -1.0
    rotation = u @ np.diag(s) @ vt
    var_src = np.mean(np.sum(src_c ** 2, axis=1))
    scale = float(np.sum(d * s) / var_src) if var_src > 0 else 1.0

    transform = np.identity(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = mu_dst - scale * rotation @ mu_src
    return transform

def subsample(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), max_points, replace=False)]

def trim_outliers(points: np.ndarray, keep_quantile: float = 0.95) -> np.ndarray:
    """Drops points far from the median, which would otherwise inflate the PCA extent."""
    dists = np.linalg.norm(points - np.median(points, axis=0), axis=1)
    return points[dists <= np.quantile(dists, keep_quantile)]

def pca_frame(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (centroid, axes as columns sorted by descending variance, variances)."""
    centroid = points.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov((points - centroid).T))
    order = np.argsort(eigvals)[::-1]
    return centroid, eigvecs[:, order], eigvals[order]

def pca_initial_transforms(src: np.ndarray, tgt: np.ndarray) -> List[np.ndarray]:
    """Enumerates the 24 Sim(3) hypotheses mapping source PCA axes onto target PCA axes.

    Axis permutations are included because near-equal variances (e.g. a square room)
    make the eigenvector order unreliable.
    """
    c_src, axes_src, var_src = pca_frame(trim_outliers(src))
    c_tgt, axes_tgt, var_tgt = pca_frame(trim_outliers(tgt))
    scale = float(np.sqrt(var_tgt.sum() / var_src.sum()))

    transforms = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            mapping = np.zeros((3, 3))
            mapping[np.arange(3), perm] = signs
            rotation = axes_tgt @ mapping @ axes_src.T
            if np.linalg.det(rotation) < 0:
                continue
            transform = np.identity(4)
            transform[:3, :3] = scale * rotation
            transform[:3, 3] = c_tgt - scale * rotation @ c_src
            transforms.append(transform)
    return transforms

def score_alignment(tree: cKDTree, src: np.ndarray, transform: np.ndarray, threshold: float) -> Tuple[float, float]:
    """Returns (fitness, inlier RMSE): the fraction of source points within threshold of the target."""
    dists, _ = tree.query(apply_transform(transform, src))
    inliers = dists <= threshold
    fitness = float(np.mean(inliers))
    rmse = float(np.sqrt(np.mean(dists[inliers] ** 2))) if np.any(inliers) else float("inf")
    return fitness, rmse

def icp_sim3(src: np.ndarray, tree: cKDTree, init: np.ndarray, thresholds: Sequence[float],
             max_iterations: int = 30, tolerance: float = 1e-7) -> np.ndarray:
    """Scale-aware point-to-point ICP over a coarse-to-fine schedule of correspondence thresholds."""
    transform = init.copy()
    tgt = tree.data
    for threshold in thresholds:
        for _ in range(max_iterations):
            moved = apply_transform(transform, src)
            dists, idx = tree.query(moved, distance_upper_bound=threshold)
            valid = np.isfinite(dists)
            if valid.sum() < 3:
                break
            delta = umeyama_sim3(moved[valid], tgt[idx[valid]])
            transform = delta @ transform
            if np.linalg.norm(delta - np.identity(4)) < tolerance:
                break
    return transform

def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    ra = a[:3, :3] / transform_scale(a)
    rb = b[:3, :3] / transform_scale(b)
    cos = (np.trace(ra.T @ rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

def global_align_sim3(src: np.ndarray, tgt: np.ndarray, threshold: float,
                      init: Optional[np.ndarray] = None, max_src_points: int = 20000,
                      max_tgt_points: int = 200000, num_refine: int = 3) -> AlignmentResult:
    """Estimates the Sim(3) transform mapping src onto tgt without an initial guess.

    Args:
        src: Reconstruction-frame points (e.g. SfM sparse points), arbitrary scale.
        tgt: Ground truth points in metric units.
        threshold: Final correspondence threshold in target units, also used for scoring.
        init: Optional initial transform; skips the PCA hypothesis search when given.
        num_refine: Number of best coarse hypotheses refined at full resolution.

    Returns:
        AlignmentResult. `ambiguous` is set when a hypothesis with a clearly different rotation
        fits nearly as well as the chosen one (symmetric scenes); verify such alignments visually.
    """
    src_full = subsample(src, max_src_points)
    tgt_full = subsample(tgt, max_tgt_points)
    tree = cKDTree(tgt_full)

    # Coarse thresholds scale with the target size so a rough PCA guess can still converge
    _, _, var_tgt = pca_frame(trim_outliers(tgt_full))
    radius = float(np.sqrt(var_tgt.sum()))
    coarse = [t for t in (0.5 * radius, 0.2 * radius, 0.1 * radius) if t > threshold]
    fine = [2.0 * threshold, threshold]

    hypotheses = [init] if init is not None else pca_initial_transforms(src_full, tgt_full)
    src_coarse = subsample(src_full, 3000, seed=1)
    coarse_results = []
    for hypothesis in hypotheses:
        transform = icp_sim3(src_coarse, tree, hypothesis, coarse + [2.0 * threshold], max_iterations=15)
        coarse_results.append((score_alignment(tree, src_coarse, transform, 2.0 * threshold), transform))
    coarse_results.sort(key=lambda r: (-r[0][0], r[0][1]))

    refined = []
    for _, transform in coarse_results[:num_refine]:
        transform = icp_sim3(src_full, tree, transform, fine)
        refined.append((score_alignment(tree, src_full, transform, threshold), transform))
    refined.sort(key=lambda r: (-r[0][0], r[0][1]))

    (fitness, rmse), best = refined[0]
    ambiguous = any(
        other_fitness >= fitness - 0.02 and rotation_angle_deg(best, other) > 10.0
        for (other_fitness, _), other in refined[1:]
    )
    return AlignmentResult(
        transform=best,
        scale=transform_scale(best),
        fitness=fitness,
        inlier_rmse=rmse,
        threshold=threshold,
        ambiguous=ambiguous,
        candidates=[score for score, _ in refined],
    )
