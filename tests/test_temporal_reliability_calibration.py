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
