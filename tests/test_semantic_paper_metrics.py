import numpy as np
import pytest


def test_binary_segmentation_metrics_matches_hand_computed_counts():
    from utils.semantic_paper_metrics import binary_segmentation_metrics

    pred = np.array([[1, 1], [0, 0]], dtype=bool)
    target = np.array([[1, 0], [1, 0]], dtype=bool)

    metrics = binary_segmentation_metrics(pred, target)

    assert metrics["intersection"] == 1
    assert metrics["union"] == 3
    assert metrics["predicted"] == 2
    assert metrics["target"] == 2
    assert metrics["iou"] == pytest.approx(1.0 / 3.0)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)


def test_soft_iou_uses_fuzzy_min_max_intersection_union():
    from utils.semantic_paper_metrics import soft_iou

    pred = np.array([0.2, 0.8], dtype=np.float32)
    target = np.array([1.0, 0.0], dtype=np.float32)

    assert soft_iou(pred, target) == pytest.approx(0.2 / 1.8)


def test_boundary_metrics_identical_masks_are_perfect():
    from utils.semantic_paper_metrics import boundary_metrics

    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True

    metrics = boundary_metrics(mask, mask, tolerance=2)

    assert metrics["boundary_precision"] == pytest.approx(1.0)
    assert metrics["boundary_recall"] == pytest.approx(1.0)
    assert metrics["boundary_f1"] == pytest.approx(1.0)
    assert metrics["boundary_iou"] == pytest.approx(1.0)


def test_boundary_metrics_separated_masks_have_zero_match():
    from utils.semantic_paper_metrics import boundary_metrics

    pred = np.zeros((12, 12), dtype=bool)
    target = np.zeros((12, 12), dtype=bool)
    pred[1:3, 1:3] = True
    target[9:11, 9:11] = True

    metrics = boundary_metrics(pred, target, tolerance=1)

    assert metrics["boundary_f1"] == pytest.approx(0.0)
    assert metrics["boundary_iou"] == pytest.approx(0.0)


def test_aggregate_part_metrics_excludes_empty_targets_from_macro_miou():
    from utils.semantic_paper_metrics import aggregate_part_metrics

    rows = [
        {"part": "face", "intersection": 3, "union": 4, "iou": 0.75, "target": 4},
        {"part": "shoes", "intersection": 0, "union": 2, "iou": 0.0, "target": 0},
        {"part": "upper", "intersection": 1, "union": 4, "iou": 0.25, "target": 2},
    ]

    summary = aggregate_part_metrics(rows)

    assert summary["macro_miou"] == pytest.approx(0.5)
    assert summary["micro_iou"] == pytest.approx(4.0 / 10.0)
    assert summary["evaluated_part_count"] == 2
    assert summary["empty_target_part_count"] == 1


def test_interpolate_curve_at_retention_interpolates_inside_range():
    from utils.semantic_paper_metrics import interpolate_curve_at_retention

    rows = [
        {"retention": 0.4, "actionable_leakage": 0.20, "raw_leakage": 0.40},
        {"retention": 0.8, "actionable_leakage": 0.10, "raw_leakage": 0.20},
    ]

    row = interpolate_curve_at_retention(rows, 0.6)

    assert row["retention"] == pytest.approx(0.6)
    assert row["actionable_leakage"] == pytest.approx(0.15)
    assert row["raw_leakage"] == pytest.approx(0.30)


def test_interpolate_curve_at_retention_forbids_extrapolation():
    from utils.semantic_paper_metrics import interpolate_curve_at_retention

    rows = [
        {"retention": 0.4, "actionable_leakage": 0.20},
        {"retention": 0.8, "actionable_leakage": 0.10},
    ]

    with pytest.raises(ValueError, match="outside observed retention range"):
        interpolate_curve_at_retention(rows, 0.9)


def test_shared_retention_targets_keeps_only_values_supported_by_all_curves():
    from utils.semantic_paper_metrics import shared_retention_targets

    curves = {
        "hard": [{"retention": 0.5}, {"retention": 1.0}],
        "ours": [{"retention": 0.3}, {"retention": 0.8}],
    }

    targets = shared_retention_targets(curves, [0.3, 0.4, 0.5, 0.6, 0.8, 0.9])

    assert targets == [0.5, 0.6, 0.8]
