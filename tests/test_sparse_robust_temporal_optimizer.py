import numpy as np
import pytest


def _synthetic_sequences(high_target_on_unstable: bool = False):
    # Two cameras, two frames each, two carriers, one part.
    unstable_target = 10.0 if high_target_on_unstable else 0.1
    target = np.array(
        [
            [[unstable_target], [10.0]],
            [[unstable_target], [10.0]],
            [[unstable_target], [10.0]],
            [[unstable_target], [10.0]],
        ],
        dtype=np.float32,
    )
    unstable = np.array([0.0, 2.0, 0.0, 2.0], dtype=np.float32)
    stable = np.ones(4, dtype=np.float32)
    outer = np.stack((unstable, stable), axis=1)[:, :, None]
    boundary = outer.copy()
    return target, outer, boundary


def test_camera_metrics_reset_adjacent_difference_between_cameras():
    from utils.sparse_robust_temporal_optimizer import summarize_camera_signal

    result = summarize_camera_signal(
        np.array([1.0, 3.0, 10.0, 14.0]),
        camera_index=np.array([0, 0, 1, 1], dtype=np.int16),
        camera_ids=(0, 1),
    )

    np.testing.assert_allclose(result["mean_response"], [2.0, 12.0])
    np.testing.assert_allclose(result["adjacent_absolute_change"], [2.0, 4.0])
    np.testing.assert_allclose(result["normalized_flicker"], [1.0, 1.0 / 3.0])


def test_sparse_optimizer_selects_one_unstable_carrier_deterministically():
    from utils.sparse_robust_temporal_optimizer import optimize_sparse_part

    target, outer, boundary = _synthetic_sequences()
    kwargs = dict(
        a5_weights=np.ones(2, dtype=np.float32),
        target_sequence=target[:, :, 0],
        outer_sequence=outer[:, :, 0],
        boundary_sequence=boundary[:, :, 0],
        camera_index=np.array([0, 0, 1, 1], dtype=np.int16),
        optimization_camera_ids=(0, 1),
        eligible_indices=np.array([0, 1], dtype=np.int64),
        reduction_fractions=(0.05, 0.1),
        maximum_changed_count=1,
        minimum_camera_target_ratio=0.98,
        objective_mean_weight=0.25,
        objective_absolute_adjacent_weight=0.05,
    )

    first = optimize_sparse_part(**kwargs)
    second = optimize_sparse_part(**kwargs)

    np.testing.assert_array_equal(first["weights"], second["weights"])
    assert first["accepted_moves"] == second["accepted_moves"]
    assert first["changed_indices"] == [0]
    assert first["accepted_moves"][0]["reduction_fraction"] == pytest.approx(0.1)
    assert first["final_ratios"]["outer_normalized_flicker"].max() < 1.0
    assert first["final_ratios"]["boundary_normalized_flicker"].max() < 1.0
    assert first["final_ratios"]["target_mean_response"].min() >= 0.98


def test_sparse_optimizer_rejects_move_that_breaks_target_constraint():
    from utils.sparse_robust_temporal_optimizer import optimize_sparse_part

    target, outer, boundary = _synthetic_sequences(high_target_on_unstable=True)
    result = optimize_sparse_part(
        a5_weights=np.ones(2, dtype=np.float32),
        target_sequence=target[:, :, 0],
        outer_sequence=outer[:, :, 0],
        boundary_sequence=boundary[:, :, 0],
        camera_index=np.array([0, 0, 1, 1], dtype=np.int16),
        optimization_camera_ids=(0, 1),
        eligible_indices=np.array([0], dtype=np.int64),
        reduction_fractions=(0.1,),
        maximum_changed_count=1,
        minimum_camera_target_ratio=0.98,
        objective_mean_weight=0.25,
        objective_absolute_adjacent_weight=0.05,
    )

    assert result["changed_indices"] == []
    np.testing.assert_array_equal(result["weights"], np.ones(2, dtype=np.float32))


def test_loco_capacity_aggregates_processed_parts_equally():
    from utils.sparse_robust_temporal_optimizer import run_loco_sparse_capacity

    target, outer, boundary = _synthetic_sequences()
    sequences = {
        "target": np.repeat(target, 2, axis=2),
        "outer": np.repeat(outer, 2, axis=2),
        "boundary": np.repeat(boundary, 2, axis=2),
    }
    result = run_loco_sparse_capacity(
        a5_weights=np.ones((2, 2), dtype=np.float32),
        sequences=sequences,
        camera_index=np.array([0, 0, 1, 1], dtype=np.int16),
        consecutive_visible_count=np.full((2, 2), 10, dtype=np.int32),
        processed_part_indices=(0, 1),
        selection_threshold=0.2,
        min_pair_support=8,
        reduction_fractions=(0.05, 0.1),
        maximum_changed_fraction=0.5,
        minimum_camera_target_ratio=0.98,
        objective_mean_weight=0.25,
        objective_absolute_adjacent_weight=0.05,
    )

    assert result["all_folds_passed"] is True
    assert len(result["folds"]) == 2
    assert all(fold["held_out_aggregate"]["outer_normalized_flicker"] < 1.0 for fold in result["folds"])
    assert all(fold["held_out_aggregate"]["boundary_normalized_flicker"] < 1.0 for fold in result["folds"])
    assert result["final"]["per_part"]["0"]["changed_indices"] == [0]
    assert result["final"]["per_part"]["1"]["changed_indices"] == [0]
