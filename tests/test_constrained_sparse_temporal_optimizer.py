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


def test_assign_temporal_blocks_balances_each_camera_independently():
    from utils.constrained_sparse_temporal_optimizer import assign_temporal_blocks

    cameras = np.repeat(np.array([0, 1], dtype=np.int16), 6)

    blocks = assign_temporal_blocks(cameras, block_count=3)

    assert blocks.tolist() == [0, 0, 1, 1, 2, 2] * 2


def test_consecutive_support_does_not_bridge_removed_temporal_segment():
    from utils.constrained_sparse_temporal_optimizer import (
        consecutive_support_from_sequences,
    )

    visible = np.ones((4, 1), dtype=np.float32)
    cameras = np.zeros(4, dtype=np.int16)
    segments = np.array([0, 0, 1, 1], dtype=np.int16)

    support = consecutive_support_from_sequences(
        selection_target_sequence=visible,
        selection_outer_sequence=np.zeros_like(visible),
        camera_index=cameras,
        camera_ids=(0,),
        segment_index=segments,
    )

    assert support.tolist() == [2]


def test_temporal_block_robustness_reports_positive_fraction_quantile_and_worst():
    from utils.constrained_sparse_temporal_optimizer import (
        evaluate_temporal_block_robustness,
    )

    cameras = np.zeros(8, dtype=np.int16)
    blocks = np.repeat(np.arange(4, dtype=np.int16), 2)
    base = np.tile(np.array([1.0, 2.0], dtype=np.float64), 4)
    candidate_outer = np.array(
        [1.0, 1.5, 1.0, 1.2, 1.0, 3.0, 1.0, 1.1], dtype=np.float64
    )
    candidate_boundary = np.array(
        [1.0, 1.5, 1.0, 1.4, 1.0, 1.3, 1.0, 1.2], dtype=np.float64
    )

    result = evaluate_temporal_block_robustness(
        base_signals={"outer": base, "boundary": base},
        candidate_signals={
            "outer": candidate_outer,
            "boundary": candidate_boundary,
        },
        camera_index=cameras,
        block_index=blocks,
        camera_ids=(0,),
        block_ids=(0, 1, 2, 3),
        gain_quantile=0.25,
    )

    assert result["outer_positive_fraction"] == 0.75
    assert result["boundary_positive_fraction"] == 1.0
    assert result["outer_worst_gain"] < 0.0
    assert result["boundary_worst_gain"] > 0.0
    assert result["outer_gain_quantile"] <= result["outer_gain_median"]


def test_block_rank_rejects_spiky_candidate_with_better_mean_gain():
    from utils.constrained_sparse_temporal_optimizer import rank_temporal_block_metrics

    spiky = {
        "outer_gains": np.array([0.04, 0.04, 0.04, -0.01]),
        "boundary_gains": np.array([0.03, 0.03, 0.03, -0.008]),
        "outer_positive_fraction": 0.75,
        "boundary_positive_fraction": 0.75,
        "outer_gain_quantile": -0.001,
        "boundary_gain_quantile": -0.0005,
        "outer_worst_gain": -0.01,
        "boundary_worst_gain": -0.008,
    }
    stable = {
        "outer_gains": np.array([0.006, 0.007, 0.008, 0.009]),
        "boundary_gains": np.array([0.005, 0.006, 0.007, 0.008]),
        "outer_positive_fraction": 1.0,
        "boundary_positive_fraction": 1.0,
        "outer_gain_quantile": 0.0063,
        "boundary_gain_quantile": 0.0053,
        "outer_worst_gain": 0.006,
        "boundary_worst_gain": 0.005,
    }

    spiky_rank = rank_temporal_block_metrics(
        spiky,
        minimum_positive_fraction=0.9,
        minimum_gain_quantile=0.0,
        maximum_worst_regression=0.005,
        cvar_fraction=0.25,
    )
    stable_rank = rank_temporal_block_metrics(
        stable,
        minimum_positive_fraction=0.9,
        minimum_gain_quantile=0.0,
        maximum_worst_regression=0.005,
        cvar_fraction=0.25,
    )

    assert stable_rank < spiky_rank
    assert stable_rank[0] == 0.0
    assert spiky_rank[0] > 0.0


def test_camera_time_fold_specs_form_cartesian_product():
    from utils.constrained_sparse_temporal_optimizer import camera_time_fold_specs

    folds = camera_time_fold_specs(range(8), range(6))

    assert len(folds) == 48
    assert folds[0] == {"held_out_camera": 0, "held_out_block": 0}
    assert folds[-1] == {"held_out_camera": 7, "held_out_block": 5}
    assert len({(row["held_out_camera"], row["held_out_block"]) for row in folds}) == 48


def test_camera_time_fold_pass_uses_camera_constraints_and_block_temporal_only():
    from utils.constrained_sparse_temporal_optimizer import camera_time_fold_passes

    assert camera_time_fold_passes(
        {"passed": True},
        {"constraints_passed": False, "temporal_passed": True, "passed": False},
    ) is True
    assert camera_time_fold_passes(
        {"passed": False},
        {"constraints_passed": True, "temporal_passed": True, "passed": True},
    ) is False
    assert camera_time_fold_passes(
        {"passed": True},
        {"constraints_passed": True, "temporal_passed": False, "passed": False},
    ) is False


def test_consensus_fold_weights_requires_stable_frequency_and_uses_modal_level():
    from utils.constrained_sparse_temporal_optimizer import consensus_fold_weights

    a5 = np.ones(4, dtype=np.float32)
    fold_weights = np.array(
        [
            [0.9, 0.95, 0.9, 1.0],
            [0.9, 0.95, 0.9, 1.0],
            [0.9, 0.95, 1.0, 1.0],
            [0.9, 1.0, 1.0, 0.9],
        ],
        dtype=np.float32,
    )

    result = consensus_fold_weights(
        a5_weights=a5,
        fold_weights=fold_weights,
        minimum_fold_count=3,
        selection_threshold=0.2,
    )

    np.testing.assert_allclose(result["weights"], [0.9, 0.95, 1.0, 1.0])
    assert result["selected_indices"] == [0, 1]
    assert result["selection_frequency"] == [4, 3, 2, 1]
    assert result["minimum_fold_count"] == 3


def test_camera_time_stability_capacity_builds_fold_consensus_without_full_data_append():
    from utils.constrained_sparse_temporal_optimizer import (
        run_camera_time_stability_capacity,
    )

    camera_index = np.repeat(np.array([0, 1], dtype=np.int16), 8)
    frame_index = np.tile(np.arange(8, dtype=np.int32), 2)
    unstable = np.tile(np.array([0.0, 2.0], dtype=np.float32), 8)
    constant = np.ones(16, dtype=np.float32)
    point_count = 2
    part_count = 2

    def sequence(first, second):
        output = np.zeros((16, point_count, part_count), dtype=np.float32)
        output[:, 0, 1] = first
        output[:, 1, 1] = second
        output[:, :, 0] = 1.0
        return output

    target = sequence(np.full(16, 0.1, dtype=np.float32), np.full(16, 10.0, dtype=np.float32))
    outer = sequence(unstable, constant)
    selection_target = sequence(np.full(16, 0.2, dtype=np.float32), np.full(16, 0.8, dtype=np.float32))
    selection_outer = sequence(np.zeros(16, dtype=np.float32), np.full(16, 0.1, dtype=np.float32))
    a5 = np.ones((point_count, part_count), dtype=np.float32)
    v4 = a5.copy()
    v4[0, 1] = 0.9

    result = run_camera_time_stability_capacity(
        a5_weights=a5,
        v4_weights=v4,
        sequences={
            "target": target,
            "outer": outer,
            "boundary": outer,
            "selection_target": selection_target,
            "selection_outer": selection_outer,
            "selection_boundary": selection_outer,
        },
        target_pixel_count=np.ones((16, part_count), dtype=np.float32),
        camera_index=camera_index,
        frame_index=frame_index,
        hair_index=0,
        lower_index=1,
        selection_threshold=0.2,
        min_pair_support=1,
        reduction_fractions=(0.1,),
        maximum_changed_fraction=1.0,
        minimum_camera_target_ratio=0.99,
        maximum_camera_soft_iou_drop=0.5,
        maximum_camera_visibility_response_ratio=1.1,
        objective_mean_weight=0.25,
        objective_absolute_adjacent_weight=0.05,
        temporal_block_count=2,
        minimum_stability_fold_count=3,
        minimum_positive_block_fraction=0.5,
        minimum_block_gain_quantile=-0.1,
        maximum_worst_block_regression=0.1,
        block_gain_quantile=0.1,
        block_cvar_fraction=0.25,
        minimum_aggregate_temporal_gain=0.0,
        minimum_lower_temporal_gain=0.0,
        maximum_changed_count=1,
    )

    assert result["fold_count"] == 4
    assert len(result["folds"]) == 4
    assert all("fold_v4_lower_seed_changed_indices" in row for row in result["folds"])
    assert result["consensus"]["selected_indices"] == [0]
    assert result["consensus"]["selection_frequency"][0] == 4
    assert result["weights"][0, 1] == np.float32(0.9)


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
