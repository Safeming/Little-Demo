import numpy as np
import pytest


def test_screen_selection_metrics_separate_inside_and_outside_mass():
    from utils.semantic_temporal_stability import compute_screen_selection_metrics

    selection = np.array(
        [
            [0.0, 0.4, 0.0],
            [0.3, 0.8, 0.2],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    target = np.zeros((3, 3), dtype=np.float32)
    target[1, 1] = 1.0
    valid = np.ones((3, 3), dtype=np.float32)

    metrics = compute_screen_selection_metrics(selection, target, valid, threshold=0.2)

    assert metrics["target_pixel_count"] == 1
    assert metrics["valid_pixel_count"] == 9
    assert metrics["inside_selection_mass"] == pytest.approx(0.8)
    assert metrics["outside_selection_mass"] == pytest.approx(0.9)
    assert metrics["selection_leakage_ratio"] == pytest.approx(1.125)
    assert metrics["screen_soft_iou"] == pytest.approx(0.8 / 1.9)
    assert metrics["screen_hard_iou"] == pytest.approx(0.25)
    assert metrics["screen_precision"] == pytest.approx(0.25)
    assert metrics["screen_recall"] == pytest.approx(1.0)


def test_screen_selection_metrics_respect_valid_mask_and_reject_shape_mismatch():
    from utils.semantic_temporal_stability import compute_screen_selection_metrics

    selection = np.array([[0.8, 0.9]], dtype=np.float32)
    target = np.array([[1.0, 0.0]], dtype=np.float32)
    valid = np.array([[1.0, 0.0]], dtype=np.float32)
    metrics = compute_screen_selection_metrics(selection, target, valid)

    assert metrics["inside_selection_mass"] == pytest.approx(0.8)
    assert metrics["outside_selection_mass"] == 0.0
    with pytest.raises(ValueError, match="matching shape"):
        compute_screen_selection_metrics(selection, np.ones((2, 2)), valid)


def test_temporal_signal_summary_reports_std_cv_and_adjacent_flicker():
    from utils.semantic_temporal_stability import summarize_temporal_signal

    summary = summarize_temporal_signal([1.0, 2.0, 1.0])

    assert summary["mean"] == pytest.approx(4.0 / 3.0)
    assert summary["std"] == pytest.approx(np.std([1.0, 2.0, 1.0], ddof=1))
    assert summary["cv"] == pytest.approx(summary["std"] / summary["mean"])
    assert summary["adjacent_flicker"] == pytest.approx(1.0 / summary["mean"])


def test_temporal_signal_summary_is_finite_for_zero_and_singleton_sequences():
    from utils.semantic_temporal_stability import summarize_temporal_signal

    zeros = summarize_temporal_signal([0.0, 0.0])
    singleton = summarize_temporal_signal([2.0])

    assert zeros == {"mean": 0.0, "std": 0.0, "cv": 0.0, "adjacent_flicker": 0.0}
    assert singleton == {"mean": 2.0, "std": 0.0, "cv": 0.0, "adjacent_flicker": 0.0}


def test_temporal_signal_summary_rejects_non_finite_values():
    from utils.semantic_temporal_stability import summarize_temporal_signal

    with pytest.raises(ValueError, match="finite"):
        summarize_temporal_signal([1.0, np.nan])
