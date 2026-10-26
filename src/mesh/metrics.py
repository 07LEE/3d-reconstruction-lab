"""Surface Distance Metrics Between a Reconstruction and Ground Truth.

Pure NumPy so the metric definitions can be unit tested without Open3D.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

METRICS_VERSION = 2

def precision_recall_fscore(d_pred_to_gt: np.ndarray, d_gt_to_pred: np.ndarray, threshold: float) -> Dict[str, float]:
    """Tanks and Temples style precision / recall / F-score at one distance threshold."""
    precision = float(np.mean(d_pred_to_gt <= threshold))
    recall = float(np.mean(d_gt_to_pred <= threshold))
    fscore = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"threshold_m": threshold, "precision": round(precision, 4), "recall": round(recall, 4), "fscore": round(fscore, 4)}

def surface_distance_metrics(d_pred_to_gt: np.ndarray, d_gt_to_pred: np.ndarray, truncation: float,
                             fscore_thresholds: Sequence[float], fscore_threshold: float) -> Dict[str, Any]:
    """Computes accuracy, completeness, Chamfer, HD95, coverage, and F-scores from nearest distances.

    Accuracy and completeness are means of distances clipped at `truncation`: GT regions the
    reconstruction misses count as `truncation` instead of being dropped, so holes raise
    completeness rather than vanishing from it. HD95 uses the raw prediction-to-GT distances.

    Args:
        d_pred_to_gt: Distance from each reconstruction sample to the nearest GT point.
        d_gt_to_pred: Distance from each GT point to the nearest reconstruction sample.
        truncation: Clipping distance in meters (also the GT coverage radius).
        fscore_thresholds: Thresholds in meters reported in `fscore_curve`.
        fscore_threshold: Primary threshold in meters reported as top-level precision/recall/fscore.
    """
    acc = np.minimum(d_pred_to_gt, truncation)
    comp = np.minimum(d_gt_to_pred, truncation)
    accuracy_mean = float(np.mean(acc))
    completeness_mean = float(np.mean(comp))
    primary = precision_recall_fscore(d_pred_to_gt, d_gt_to_pred, fscore_threshold)

    return {
        "truncation_m": truncation,
        "accuracy_mean_m": round(accuracy_mean, 5),
        "accuracy_rmse_m": round(float(np.sqrt(np.mean(acc ** 2))), 5),
        "completeness_mean_m": round(completeness_mean, 5),
        "completeness_rmse_m": round(float(np.sqrt(np.mean(comp ** 2))), 5),
        "chamfer_l1_m": round((accuracy_mean + completeness_mean) / 2.0, 5),
        "hausdorff_95_m": round(float(np.percentile(d_pred_to_gt, 95)), 5),
        "gt_coverage_pct": round(float(np.mean(d_gt_to_pred <= truncation)) * 100.0, 2),
        "fscore_threshold_m": fscore_threshold,
        "precision": primary["precision"],
        "recall": primary["recall"],
        "fscore": primary["fscore"],
        "fscore_curve": [precision_recall_fscore(d_pred_to_gt, d_gt_to_pred, t) for t in sorted(set(fscore_thresholds))],
    }
