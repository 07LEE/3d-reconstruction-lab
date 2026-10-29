import numpy as np
import pytest

from src.mesh.metrics import precision_recall_fscore, surface_distance_metrics

def test_fscore_from_precision_and_recall():
    d_pred = np.array([0.01, 0.01, 0.01, 0.2])  # precision 0.75 at 0.05
    d_gt = np.array([0.01, 0.3])                 # recall 0.5 at 0.05
    res = precision_recall_fscore(d_pred, d_gt, 0.05)
    assert res["precision"] == 0.75
    assert res["recall"] == 0.5
    assert res["fscore"] == pytest.approx(0.6)

def test_fscore_is_zero_when_nothing_matches():
    assert precision_recall_fscore(np.array([1.0]), np.array([1.0]), 0.05)["fscore"] == 0.0

def test_missing_gt_regions_raise_completeness():
    d_pred = np.full(100, 0.01)
    covered = surface_distance_metrics(d_pred, np.full(100, 0.01), 0.5, [0.05], 0.05)
    # Half of the GT is far from the mesh: those points count at the truncation distance
    holes = surface_distance_metrics(d_pred, np.r_[np.full(50, 0.01), np.full(50, 3.0)], 0.5, [0.05], 0.05)

    assert covered["completeness_mean_m"] == pytest.approx(0.01)
    assert holes["completeness_mean_m"] == pytest.approx((0.01 + 0.5) / 2)
    assert holes["chamfer_l1_m"] > covered["chamfer_l1_m"]
    assert holes["gt_coverage_pct"] == 50.0
    assert holes["recall"] == 0.5

def test_floaters_are_clipped_but_hd95_uses_raw_distances():
    d_pred = np.r_[np.full(90, 0.01), np.full(10, 10.0)]
    res = surface_distance_metrics(d_pred, np.full(100, 0.01), 0.5, [0.02, 0.05], 0.05)
    assert res["accuracy_mean_m"] == pytest.approx(0.9 * 0.01 + 0.1 * 0.5)
    assert res["hausdorff_95_m"] == pytest.approx(10.0)
    assert [c["threshold_m"] for c in res["fscore_curve"]] == [0.02, 0.05]
