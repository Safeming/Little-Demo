import numpy as np


def _inputs():
    return {
        "a5_weights": np.array(
            [[0.8, 0.7], [0.4, 0.6], [0.1, 0.5]], dtype=np.float32
        ),
        "target_contribution_mean": np.array(
            [[1.0, 0.8], [0.7, 0.6], [0.2, 0.4]], dtype=np.float32
        ),
        "temporal_reliability": np.array(
            [[0.0, 0.2], [1.0, 0.8], [0.1, 1.0]], dtype=np.float32
        ),
        "consecutive_visible_count": np.full((3, 2), 100, dtype=np.int32),
    }


def test_bounded_damping_never_exceeds_a5_and_preserves_topology():
    from utils.bounded_temporal_calibration import calibrate_bounded_a7_weights

    inputs = _inputs()
    weights, summary = calibrate_bounded_a7_weights(
        **inputs,
        rho=0.9,
        min_pair_support=8,
        minimum_weight_ratio_from_a5=0.9,
        restore_target_mass=False,
        maximum_part_weight_l1_from_a5=12.0,
        selection_threshold=0.2,
        preserve_selection_topology=True,
    )

    assert np.all(weights <= inputs["a5_weights"])
    np.testing.assert_array_equal(weights >= 0.2, inputs["a5_weights"] >= 0.2)
    assert summary["selection_crossing_count"] == 0
    assert summary["maximum_weight_above_a5"] == 0.0


def test_bounded_retention_restores_only_toward_a5_target_floor():
    from utils.bounded_temporal_calibration import calibrate_bounded_a7_weights

    inputs = _inputs()
    weights, summary = calibrate_bounded_a7_weights(
        **inputs,
        rho=0.95,
        min_pair_support=8,
        minimum_weight_ratio_from_a5=0.8,
        restore_target_mass=True,
        maximum_part_weight_l1_from_a5=12.0,
        selection_threshold=0.2,
        preserve_selection_topology=True,
    )

    assert summary["valid"] is True
    assert np.all(weights <= inputs["a5_weights"])
    for part in summary["per_part"]:
        assert part["restored_target_mass"] >= part["target_floor"] - 1e-6
        assert all(
            weights[index, part["part_index"]]
            <= inputs["a5_weights"][index, part["part_index"]]
            for index in part["restored_gaussian_indices"]
        )


def test_bounded_calibration_enforces_part_l1_limit_and_frozen_column():
    from utils.bounded_temporal_calibration import calibrate_bounded_a7_weights

    inputs = _inputs()
    weights, summary = calibrate_bounded_a7_weights(
        **inputs,
        rho=0.5,
        min_pair_support=8,
        minimum_weight_ratio_from_a5=0.0,
        restore_target_mass=False,
        maximum_part_weight_l1_from_a5=0.1,
        frozen_part_indices=(1,),
        selection_threshold=0.2,
        preserve_selection_topology=True,
    )

    assert summary["per_part"][0]["weight_l1_from_a5"] <= 0.100001
    np.testing.assert_array_equal(weights[:, 1], inputs["a5_weights"][:, 1])
    assert summary["per_part"][1]["frozen"] is True


def test_bounded_retention_marks_unreachable_target_floor_invalid():
    from utils.bounded_temporal_calibration import calibrate_bounded_a7_weights

    weights, summary = calibrate_bounded_a7_weights(
        a5_weights=np.array([[0.8], [0.7]], dtype=np.float32),
        target_contribution_mean=np.ones((2, 1), dtype=np.float32),
        temporal_reliability=np.zeros((2, 1), dtype=np.float32),
        consecutive_visible_count=np.zeros((2, 1), dtype=np.int32),
        rho=0.95,
        min_pair_support=8,
        minimum_weight_ratio_from_a5=0.5,
        restore_target_mass=True,
        maximum_part_weight_l1_from_a5=12.0,
        selection_threshold=0.2,
        preserve_selection_topology=True,
    )

    assert np.all(weights <= np.array([[0.8], [0.7]], dtype=np.float32))
    assert summary["valid"] is False
    assert summary["per_part"][0]["remaining_deficit"] > 0.0
    assert "target_floor_unreachable:0" in summary["invalid_reasons"]


def test_target_floor_deficit_uses_scale_aware_float_tolerance():
    from utils.bounded_temporal_calibration import (
        target_floor_deficit_is_significant,
    )

    assert target_floor_deficit_is_significant(5.9e-6, 340.0) is False
    assert target_floor_deficit_is_significant(1.0e-3, 340.0) is True
