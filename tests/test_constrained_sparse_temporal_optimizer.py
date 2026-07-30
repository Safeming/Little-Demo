import numpy as np


def _camera_index():
    return np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int16)


def test_consecutive_support_uses_only_requested_cameras():
    from utils.constrained_sparse_temporal_optimizer import (
        consecutive_support_from_sequences,
    )

    target = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)
    outer = np.zeros_like(target)
    cameras = np.array([0, 0, 1, 1], dtype=np.int16)

    training_only = consecutive_support_from_sequences(
        selection_target_sequence=target,
        selection_outer_sequence=outer,
        camera_index=cameras,
        camera_ids=(0,),
    )
    all_cameras = consecutive_support_from_sequences(
        selection_target_sequence=target,
        selection_outer_sequence=outer,
        camera_index=cameras,
        camera_ids=(0, 1),
    )

    assert training_only.tolist() == [0]
    assert all_cameras.tolist() == [1]


def test_hair_compensation_opens_only_for_temporal_shortfall():
    from utils.constrained_sparse_temporal_optimizer import (
        should_open_hair_compensation,
    )

    assert should_open_hair_compensation(
        {"constraints_passed": True, "temporal_passed": False}, 3
    ) is True
    assert should_open_hair_compensation(
        {"constraints_passed": False, "temporal_passed": False}, 3
    ) is False
    assert should_open_hair_compensation(
        {"constraints_passed": True, "temporal_passed": True}, 3
    ) is False
    assert should_open_hair_compensation(
        {"constraints_passed": True, "temporal_passed": False}, 0
    ) is False


def test_visibility_limits_require_construction_margin_not_weaker_than_audit():
    from utils.constrained_sparse_temporal_optimizer import resolve_visibility_limits

    assert resolve_visibility_limits(
        maximum_training_visibility_response_ratio=0.9995,
        maximum_audit_visibility_response_ratio=1.0,
    ) == (0.9995, 1.0)

    import pytest

    with pytest.raises(ValueError, match="training visibility"):
        resolve_visibility_limits(
            maximum_training_visibility_response_ratio=1.0,
            maximum_audit_visibility_response_ratio=0.9995,
        )


def test_target_limits_require_construction_floor_not_weaker_than_audit():
    from utils.constrained_sparse_temporal_optimizer import resolve_target_limits

    assert resolve_target_limits(
        minimum_training_target_response_ratio=0.9975,
        minimum_audit_target_response_ratio=0.99,
    ) == (0.9975, 0.99)

    import pytest

    with pytest.raises(ValueError, match="training target"):
        resolve_target_limits(
            minimum_training_target_response_ratio=0.99,
            minimum_audit_target_response_ratio=0.9975,
        )


def test_temporal_gain_limits_allow_directional_held_out_gate():
    from utils.constrained_sparse_temporal_optimizer import resolve_temporal_gain_limits

    assert resolve_temporal_gain_limits(
        minimum_construction_temporal_gain=0.005,
        minimum_held_out_temporal_gain=0.0,
    ) == (0.005, 0.0)

    import pytest

    with pytest.raises(ValueError, match="held-out temporal"):
        resolve_temporal_gain_limits(
            minimum_construction_temporal_gain=0.005,
            minimum_held_out_temporal_gain=0.006,
        )


def test_capacity_requires_both_construction_and_audit_evaluations():
    from utils.constrained_sparse_temporal_optimizer import capacity_candidate_passes

    assert capacity_candidate_passes(
        {"passed": True}, {"passed": True}
    ) is True
    assert capacity_candidate_passes(
        {"passed": False}, {"passed": True}
    ) is False
    assert capacity_candidate_passes(
        {"passed": True}, {"passed": False}
    ) is False


def test_constraint_evaluation_reports_soft_iou_drop_per_camera():
    from utils.constrained_sparse_temporal_optimizer import (
        evaluate_constrained_part_weights,
    )

    a5 = np.ones(2, dtype=np.float32)
    candidate = np.array([0.9, 1.0], dtype=np.float32)
    samples = 8
    edit_target = np.tile(np.array([[0.1, 10.0]], dtype=np.float32), (samples, 1))
    edit_outer = np.ones((samples, 2), dtype=np.float32)
    selection_target = np.tile(
        np.array([[0.6, 0.3]], dtype=np.float32), (samples, 1)
    )
    selection_outer = np.tile(
        np.array([[0.0, 0.1]], dtype=np.float32), (samples, 1)
    )

    result = evaluate_constrained_part_weights(
        a5_weights=a5,
        candidate_weights=candidate,
        edit_target_sequence=edit_target,
        edit_outer_sequence=edit_outer,
        edit_boundary_sequence=edit_outer,
        selection_target_sequence=selection_target,
        selection_outer_sequence=selection_outer,
        target_pixel_count_sequence=np.ones(samples, dtype=np.float32),
        camera_index=_camera_index(),
        camera_ids=(0, 1),
    )

    np.testing.assert_allclose(
        result["soft_iou_drop"],
        np.full(2, (0.9 - 0.84) / 1.1),
        rtol=1e-6,
    )
    assert result["target_mean_response"].min() > 0.99


def test_signal_evaluation_matches_weight_evaluation_without_reprojection():
    from utils.constrained_sparse_temporal_optimizer import (
        evaluate_constrained_part_signals,
        evaluate_constrained_part_weights,
    )

    a5 = np.ones(2, dtype=np.float32)
    candidate = np.array([0.9, 1.0], dtype=np.float32)
    samples = 8
    sequences = {
        "target": np.tile(np.array([[0.1, 10.0]], dtype=np.float32), (samples, 1)),
        "outer": np.tile(np.array([[1.0, 2.0]], dtype=np.float32), (samples, 1)),
        "boundary": np.tile(np.array([[2.0, 1.0]], dtype=np.float32), (samples, 1)),
        "selection_target": np.tile(
            np.array([[0.6, 0.3]], dtype=np.float32), (samples, 1)
        ),
        "selection_outer": np.tile(
            np.array([[0.0, 0.1]], dtype=np.float32), (samples, 1)
        ),
    }
    weighted = evaluate_constrained_part_weights(
        a5_weights=a5,
        candidate_weights=candidate,
        edit_target_sequence=sequences["target"],
        edit_outer_sequence=sequences["outer"],
        edit_boundary_sequence=sequences["boundary"],
        selection_target_sequence=sequences["selection_target"],
        selection_outer_sequence=sequences["selection_outer"],
        target_pixel_count_sequence=np.ones(samples, dtype=np.float32),
        camera_index=_camera_index(),
        camera_ids=(0, 1),
    )
    signals = evaluate_constrained_part_signals(
        base_signals={
            key: value.astype(np.float64) @ a5.astype(np.float64)
            for key, value in sequences.items()
        },
        candidate_signals={
            key: value.astype(np.float64) @ candidate.astype(np.float64)
            for key, value in sequences.items()
        },
        target_pixel_count_sequence=np.ones(samples, dtype=np.float32),
        camera_index=_camera_index(),
        camera_ids=(0, 1),
    )

    assert signals.keys() == weighted.keys()
    for key in signals:
        np.testing.assert_allclose(signals[key], weighted[key])


def test_optimizer_rejects_temporal_move_that_breaks_soft_iou_constraint():
    from utils.constrained_sparse_temporal_optimizer import (
        optimize_constrained_sparse_part,
    )

    samples = 8
    unstable = np.array([0.0, 2.0, 0.0, 2.0] * 2, dtype=np.float32)
    result = optimize_constrained_sparse_part(
        a5_weights=np.ones(2, dtype=np.float32),
        initial_weights=np.ones(2, dtype=np.float32),
        edit_target_sequence=np.tile(
            np.array([[0.1, 10.0]], dtype=np.float32), (samples, 1)
        ),
        edit_outer_sequence=np.stack((unstable, np.ones(samples)), axis=1),
        edit_boundary_sequence=np.stack((unstable, np.ones(samples)), axis=1),
        selection_target_sequence=np.tile(
            np.array([[0.6, 0.3]], dtype=np.float32), (samples, 1)
        ),
        selection_outer_sequence=np.tile(
            np.array([[0.0, 0.1]], dtype=np.float32), (samples, 1)
        ),
        target_pixel_count_sequence=np.ones(samples, dtype=np.float32),
        camera_index=_camera_index(),
        optimization_camera_ids=(0, 1),
        eligible_indices=np.array([0], dtype=np.int64),
        reduction_fractions=(0.05, 0.1),
        maximum_changed_count=1,
        minimum_camera_target_ratio=0.99,
        maximum_camera_soft_iou_drop=0.005,
        maximum_camera_visibility_response_ratio=1.0,
        objective_mean_weight=0.25,
        objective_absolute_adjacent_weight=0.05,
    )

    assert result["changed_indices"] == []
    assert result["accepted_moves"] == []


def test_optimizer_rejects_move_that_increases_visibility_response_flicker():
    from utils.constrained_sparse_temporal_optimizer import (
        optimize_constrained_sparse_part,
    )

    stabilizer = np.array([2.0, 0.0, 2.0, 0.0] * 2, dtype=np.float32)
    complement = np.array([20.0, 22.0, 20.0, 22.0] * 2, dtype=np.float32)
    unstable = np.array([0.0, 2.0, 0.0, 2.0] * 2, dtype=np.float32)
    result = optimize_constrained_sparse_part(
        a5_weights=np.ones(2, dtype=np.float32),
        initial_weights=np.ones(2, dtype=np.float32),
        edit_target_sequence=np.stack((stabilizer, complement), axis=1),
        edit_outer_sequence=np.stack((unstable, np.ones(8)), axis=1),
        edit_boundary_sequence=np.stack((unstable, np.ones(8)), axis=1),
        selection_target_sequence=np.tile(
            np.array([[0.01, 0.8]], dtype=np.float32), (8, 1)
        ),
        selection_outer_sequence=np.tile(
            np.array([[0.0, 0.1]], dtype=np.float32), (8, 1)
        ),
        target_pixel_count_sequence=np.ones(8, dtype=np.float32),
        camera_index=_camera_index(),
        optimization_camera_ids=(0, 1),
        eligible_indices=np.array([0], dtype=np.int64),
        reduction_fractions=(0.1,),
        maximum_changed_count=1,
        minimum_camera_target_ratio=0.99,
        maximum_camera_soft_iou_drop=0.005,
        maximum_camera_visibility_response_ratio=1.0,
        objective_mean_weight=0.25,
        objective_absolute_adjacent_weight=0.05,
    )

    assert result["changed_indices"] == []
    assert result["accepted_moves"] == []
