import copy

import numpy as np
import pytest


EXPECTED_EVIDENCE_KEYS = {
    "temporal_visible_count",
    "temporal_consecutive_visible_count",
    "temporal_target_ratio_mean",
    "temporal_target_ratio_std",
    "temporal_target_flicker",
    "temporal_outer_ratio_mean",
    "temporal_outer_ratio_std",
    "temporal_outer_flicker",
    "temporal_boundary_crossing_rate",
    "temporal_visibility_transition_rate",
}


def _accumulate_sequence():
    from utils.temporal_reliability_calibration import (
        accumulate_temporal_footprint_frame,
    )

    state = {}
    frames = [
        (
            0,
            [[True], [False]],
            [[0.2], [0.0]],
            [[0.1], [0.0]],
            [[1], [0]],
        ),
        (
            1,
            [[True], [True]],
            [[0.4], [0.8]],
            [[0.2], [0.1]],
            [[1], [1]],
        ),
        (
            2,
            [[True], [False]],
            [[0.1], [0.0]],
            [[0.5], [0.0]],
            [[3], [0]],
        ),
    ]
    for frame_index, visible, target, outer, boundary in frames:
        accumulate_temporal_footprint_frame(
            state,
            frame_index=frame_index,
            visible=np.asarray(visible, dtype=np.bool_),
            target_ratio=np.asarray(target, dtype=np.float32),
            outer_ratio=np.asarray(outer, dtype=np.float32),
            boundary_state=np.asarray(boundary, dtype=np.int8),
        )
    return state


def test_temporal_evidence_matches_fixed_metric_definitions():
    from utils.temporal_reliability_calibration import (
        finalize_temporal_footprint_evidence,
    )

    evidence = finalize_temporal_footprint_evidence(_accumulate_sequence())

    assert set(evidence) == EXPECTED_EVIDENCE_KEYS
    np.testing.assert_array_equal(
        evidence["temporal_visible_count"], np.array([[3], [1]], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        evidence["temporal_consecutive_visible_count"],
        np.array([[2], [0]], dtype=np.int32),
    )
    np.testing.assert_allclose(
        evidence["temporal_target_ratio_mean"][:, 0],
        [np.mean([0.2, 0.4, 0.1]), 0.8],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        evidence["temporal_target_ratio_std"][:, 0],
        [np.std([0.2, 0.4, 0.1], ddof=0), 0.0],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        evidence["temporal_target_flicker"][:, 0],
        [0.25, 0.0],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        evidence["temporal_outer_ratio_mean"][:, 0],
        [np.mean([0.1, 0.2, 0.5]), 0.1],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        evidence["temporal_outer_ratio_std"][:, 0],
        [np.std([0.1, 0.2, 0.5], ddof=0), 0.0],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        evidence["temporal_outer_flicker"][:, 0],
        [0.2, 0.0],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        evidence["temporal_boundary_crossing_rate"][:, 0],
        [0.5, 0.0],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        evidence["temporal_visibility_transition_rate"][:, 0],
        [0.0, 1.0],
        rtol=0,
        atol=1e-7,
    )


def test_temporal_evidence_uses_required_shapes_dtypes_and_zero_support_values():
    from utils.temporal_reliability_calibration import (
        accumulate_temporal_footprint_frame,
        finalize_temporal_footprint_evidence,
    )

    state = {}
    zeros = np.zeros((3, 2), dtype=np.float32)
    accumulate_temporal_footprint_frame(
        state,
        frame_index=5,
        visible=np.zeros((3, 2), dtype=np.bool_),
        target_ratio=zeros,
        outer_ratio=zeros,
        boundary_state=np.zeros((3, 2), dtype=np.int8),
    )
    evidence = finalize_temporal_footprint_evidence(state)

    for key, value in evidence.items():
        assert value.shape == (3, 2), key
        assert np.all(np.isfinite(value)), key
        assert np.all(value == 0), key
        expected_dtype = np.int32 if key.endswith("count") else np.float32
        assert value.dtype == expected_dtype, key


def test_temporal_evidence_is_bitwise_deterministic():
    from utils.temporal_reliability_calibration import (
        finalize_temporal_footprint_evidence,
    )

    first = finalize_temporal_footprint_evidence(_accumulate_sequence())
    second = finalize_temporal_footprint_evidence(_accumulate_sequence())

    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("target_ratio", np.array([[np.nan]], dtype=np.float32), "finite"),
        ("outer_ratio", np.array([[np.inf]], dtype=np.float32), "finite"),
        ("visible", np.array([[1]], dtype=np.int32), "boolean"),
        ("boundary_state", np.array([[4]], dtype=np.int8), "boundary_state"),
    ],
)
def test_accumulator_rejects_invalid_inputs(field, replacement, message):
    from utils.temporal_reliability_calibration import (
        accumulate_temporal_footprint_frame,
    )

    values = {
        "visible": np.array([[True]], dtype=np.bool_),
        "target_ratio": np.array([[0.5]], dtype=np.float32),
        "outer_ratio": np.array([[0.2]], dtype=np.float32),
        "boundary_state": np.array([[1]], dtype=np.int8),
    }
    values[field] = replacement

    with pytest.raises(ValueError, match=message):
        accumulate_temporal_footprint_frame({}, frame_index=0, **values)


def test_accumulator_rejects_shape_mismatch():
    from utils.temporal_reliability_calibration import (
        accumulate_temporal_footprint_frame,
    )

    with pytest.raises(ValueError, match="shape"):
        accumulate_temporal_footprint_frame(
            {},
            frame_index=0,
            visible=np.ones((2, 1), dtype=np.bool_),
            target_ratio=np.ones((2, 2), dtype=np.float32),
            outer_ratio=np.ones((2, 1), dtype=np.float32),
            boundary_state=np.ones((2, 1), dtype=np.int8),
        )


def test_accumulator_requires_strictly_increasing_frame_indices():
    from utils.temporal_reliability_calibration import (
        accumulate_temporal_footprint_frame,
    )

    state = _accumulate_sequence()
    kwargs = {
        "visible": np.ones((2, 1), dtype=np.bool_),
        "target_ratio": np.ones((2, 1), dtype=np.float32),
        "outer_ratio": np.zeros((2, 1), dtype=np.float32),
        "boundary_state": np.ones((2, 1), dtype=np.int8),
    }
    for frame_index in (2, 1):
        with pytest.raises(ValueError, match="strictly increasing"):
            accumulate_temporal_footprint_frame(
                copy.deepcopy(state), frame_index=frame_index, **kwargs
            )


def test_temporal_reliability_matches_fixed_formula_and_support_gate():
    from utils.temporal_reliability_calibration import compute_temporal_reliability

    support = np.array([[10, 2], [8, 7]], dtype=np.int32)
    outer = np.array([[0.2, 0.3], [0.1, 0.4]], dtype=np.float32)
    boundary = np.array([[0.1, 0.2], [0.3, 0.1]], dtype=np.float32)
    target = np.array([[0.4, 0.1], [0.2, 0.5]], dtype=np.float32)

    reliability, summary = compute_temporal_reliability(
        consecutive_visible_count=support,
        temporal_outer_flicker=outer,
        temporal_boundary_crossing_rate=boundary,
        temporal_target_flicker=target,
        lambda_outer=0.5,
        lambda_boundary=0.25,
        lambda_target=1.0,
        min_pair_support=8,
    )
    expected = (support >= 8) * np.exp(-0.5 * outer - 0.25 * boundary - target)

    assert reliability.dtype == np.float32
    np.testing.assert_allclose(reliability, expected, rtol=1e-6, atol=0)
    assert np.all((reliability >= 0) & (reliability <= 1))
    assert summary["supported_entry_count"] == 2
    assert summary["total_entry_count"] == 4
    assert summary["min_pair_support"] == 8


def test_temporal_reliability_is_one_when_lambdas_are_zero_and_support_is_sufficient():
    from utils.temporal_reliability_calibration import compute_temporal_reliability

    values = np.array([[0.9, 0.2]], dtype=np.float32)
    reliability, _ = compute_temporal_reliability(
        consecutive_visible_count=np.array([[8, 9]], dtype=np.int32),
        temporal_outer_flicker=values,
        temporal_boundary_crossing_rate=values,
        temporal_target_flicker=values,
        lambda_outer=0.0,
        lambda_boundary=0.0,
        lambda_target=0.0,
        min_pair_support=8,
    )

    np.testing.assert_array_equal(reliability, np.ones((1, 2), dtype=np.float32))


def _water_filling_inputs():
    return {
        "a5_weights": np.array(
            [[1.0, 0.2], [0.8, 0.4], [0.5, 0.6], [0.4, 0.8]],
            dtype=np.float32,
        ),
        "semantic_probs": np.array(
            [[1.0, 0.3], [0.9, 0.5], [0.6, 0.7], [0.9, 0.9]],
            dtype=np.float32,
        ),
        "temporal_target_ratio_mean": np.array(
            [[0.9, 0.0], [0.8, 0.0], [0.7, 0.0], [0.1, 0.0]],
            dtype=np.float32,
        ),
        "temporal_outer_ratio_mean": np.array(
            [[0.1, 0.0], [0.2, 0.0], [0.8, 0.0], [0.4, 0.0]],
            dtype=np.float32,
        ),
        "temporal_reliability": np.array(
            [[0.5, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        ),
        "consecutive_visible_count": np.array(
            [[10, 0], [10, 0], [10, 0], [10, 0]], dtype=np.int32
        ),
        "rho": 0.9,
        "min_pair_support": 8,
        "max_weight_scale_from_posterior": 1.0,
    }


def test_water_filling_reaches_target_floor_in_stable_candidate_order():
    from utils.temporal_reliability_calibration import calibrate_a7_soft_edit_weights

    inputs = _water_filling_inputs()
    calibrated, summary = calibrate_a7_soft_edit_weights(**inputs)
    part = summary["per_part"][0]

    assert calibrated.shape == inputs["a5_weights"].shape
    assert calibrated.dtype == np.float32
    assert np.all(np.isfinite(calibrated))
    assert np.all((calibrated >= 0) & (calibrated <= 1))
    assert calibrated[0, 0] == pytest.approx(1.0)
    assert calibrated[1, 0] > inputs["a5_weights"][1, 0] * 0.5
    assert calibrated[2, 0] == pytest.approx(0.5)
    assert calibrated[3, 0] == pytest.approx(0.4)
    assert part["candidate_gaussian_indices"] == [0, 1]
    assert part["redistributed_gaussian_count"] == 2
    assert part["cap_saturated_count"] == 1
    assert part["remaining_deficit"] == pytest.approx(0.0, abs=1e-6)
    assert part["restored_target_mass"] >= part["target_floor"] - 1e-6

    ceiling = np.minimum(
        1.0,
        inputs["semantic_probs"] * inputs["max_weight_scale_from_posterior"],
    )
    assert np.all(calibrated[:, 0] <= ceiling[:, 0] + 1e-7)


def test_water_filling_preserves_part_without_temporal_evidence():
    from utils.temporal_reliability_calibration import calibrate_a7_soft_edit_weights

    inputs = _water_filling_inputs()
    calibrated, summary = calibrate_a7_soft_edit_weights(**inputs)

    np.testing.assert_array_equal(calibrated[:, 1], inputs["a5_weights"][:, 1])
    assert summary["per_part"][1]["processed"] is False


def test_water_filling_reports_deficit_when_all_carriers_are_unstable():
    from utils.temporal_reliability_calibration import calibrate_a7_soft_edit_weights

    a5 = np.array([[0.8], [0.7]], dtype=np.float32)
    calibrated, summary = calibrate_a7_soft_edit_weights(
        a5_weights=a5,
        semantic_probs=np.ones_like(a5),
        temporal_target_ratio_mean=np.array([[0.8], [0.7]], dtype=np.float32),
        temporal_outer_ratio_mean=np.array([[0.1], [0.2]], dtype=np.float32),
        temporal_reliability=np.zeros_like(a5),
        consecutive_visible_count=np.array([[2], [3]], dtype=np.int32),
        rho=0.95,
        min_pair_support=8,
        max_weight_scale_from_posterior=1.0,
    )
    part = summary["per_part"][0]

    np.testing.assert_array_equal(calibrated, np.zeros_like(a5))
    assert part["stable_candidate_count"] == 0
    assert part["redistributed_gaussian_count"] == 0
    assert part["remaining_deficit"] == pytest.approx(part["target_floor"])


def test_water_filling_reports_capacity_shortfall_without_exceeding_ceiling():
    from utils.temporal_reliability_calibration import calibrate_a7_soft_edit_weights

    calibrated, summary = calibrate_a7_soft_edit_weights(
        a5_weights=np.array([[1.0], [1.0]], dtype=np.float32),
        semantic_probs=np.array([[0.2], [0.1]], dtype=np.float32),
        temporal_target_ratio_mean=np.array([[1.0], [1.0]], dtype=np.float32),
        temporal_outer_ratio_mean=np.zeros((2, 1), dtype=np.float32),
        temporal_reliability=np.zeros((2, 1), dtype=np.float32),
        consecutive_visible_count=np.full((2, 1), 10, dtype=np.int32),
        rho=0.9,
        min_pair_support=8,
        max_weight_scale_from_posterior=1.0,
    )
    part = summary["per_part"][0]

    np.testing.assert_allclose(calibrated[:, 0], [0.2, 0.1], rtol=0, atol=1e-7)
    assert part["remaining_deficit"] == pytest.approx(1.5, abs=1e-6)
    assert part["cap_saturated_count"] == 2


@pytest.mark.parametrize("seed", range(20))
def test_water_filling_randomized_invariants_are_deterministic(seed):
    from utils.temporal_reliability_calibration import calibrate_a7_soft_edit_weights

    rng = np.random.default_rng(seed)
    shape = (17, 3)
    inputs = {
        "a5_weights": rng.random(shape, dtype=np.float32),
        "semantic_probs": rng.random(shape, dtype=np.float32),
        "temporal_target_ratio_mean": rng.random(shape, dtype=np.float32),
        "temporal_outer_ratio_mean": rng.random(shape, dtype=np.float32),
        "temporal_reliability": rng.random(shape, dtype=np.float32),
        "consecutive_visible_count": rng.integers(0, 20, size=shape, dtype=np.int32),
        "rho": 0.9,
        "min_pair_support": 8,
        "max_weight_scale_from_posterior": 1.5,
    }
    first_weights, first_summary = calibrate_a7_soft_edit_weights(**inputs)
    second_weights, second_summary = calibrate_a7_soft_edit_weights(**inputs)
    ceiling = np.minimum(1.0, inputs["semantic_probs"] * 1.5)

    np.testing.assert_array_equal(first_weights, second_weights)
    assert first_summary == second_summary
    assert first_weights.dtype == np.float32
    assert np.all(np.isfinite(first_weights))
    assert np.all((first_weights >= 0) & (first_weights <= 1))
    for part_index, part in enumerate(first_summary["per_part"]):
        if not part["processed"]:
            continue
        assert np.all(first_weights[:, part_index] <= ceiling[:, part_index] + 1e-7)
        if part["remaining_deficit"] <= 1e-6:
            assert part["restored_target_mass"] >= part["target_floor"] - 1e-6
