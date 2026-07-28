from __future__ import annotations

from collections.abc import MutableMapping

import numpy as np


def _validate_frame_inputs(
    *,
    frame_index: int,
    visible: np.ndarray,
    target_ratio: np.ndarray,
    outer_ratio: np.ndarray,
    boundary_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)):
        raise ValueError("frame_index must be an integer")

    visible_array = np.asarray(visible)
    target_array = np.asarray(target_ratio)
    outer_array = np.asarray(outer_ratio)
    boundary_array = np.asarray(boundary_state)
    if visible_array.dtype != np.bool_:
        raise ValueError("visible must be a boolean array")
    if visible_array.ndim != 2:
        raise ValueError("temporal footprint inputs must have shape [N, C]")
    if any(
        value.shape != visible_array.shape
        for value in (target_array, outer_array, boundary_array)
    ):
        raise ValueError("all temporal footprint inputs must have the same shape")
    if not np.issubdtype(boundary_array.dtype, np.integer):
        raise ValueError("boundary_state must use an integer dtype")

    target64 = target_array.astype(np.float64, copy=False)
    outer64 = outer_array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(target64)) or not np.all(np.isfinite(outer64)):
        raise ValueError("target_ratio and outer_ratio must be finite")
    if np.any((target64 < 0.0) | (target64 > 1.0)):
        raise ValueError("target_ratio must be in [0, 1]")
    if np.any((outer64 < 0.0) | (outer64 > 1.0)):
        raise ValueError("outer_ratio must be in [0, 1]")
    if np.any((boundary_array < 0) | (boundary_array > 3)):
        raise ValueError("boundary_state must be in [0, 3]")
    if np.any(boundary_array[~visible_array] != 0):
        raise ValueError("boundary_state must be 0 when not visible")
    if np.any(boundary_array[visible_array] == 0):
        raise ValueError("visible entries require a nonzero boundary_state")

    return visible_array, target64, outer64, boundary_array.astype(np.int8, copy=False)


def _initialize_state(state: MutableMapping, shape: tuple[int, int]) -> None:
    state["shape"] = shape
    state["last_frame_index"] = None
    state["visible_count"] = np.zeros(shape, dtype=np.int64)
    state["target_mean"] = np.zeros(shape, dtype=np.float64)
    state["target_m2"] = np.zeros(shape, dtype=np.float64)
    state["outer_mean"] = np.zeros(shape, dtype=np.float64)
    state["outer_m2"] = np.zeros(shape, dtype=np.float64)
    state["consecutive_visible_count"] = np.zeros(shape, dtype=np.int64)
    state["target_flicker_sum"] = np.zeros(shape, dtype=np.float64)
    state["outer_flicker_sum"] = np.zeros(shape, dtype=np.float64)
    state["boundary_crossing_count"] = np.zeros(shape, dtype=np.int64)
    state["visibility_transition_count"] = np.zeros(shape, dtype=np.int64)
    state["visibility_pair_count"] = np.zeros(shape, dtype=np.int64)
    state["previous_visible"] = None
    state["previous_target_ratio"] = None
    state["previous_outer_ratio"] = None
    state["previous_boundary_state"] = None


def accumulate_temporal_footprint_frame(
    state,
    *,
    frame_index: int,
    visible: np.ndarray,
    target_ratio: np.ndarray,
    outer_ratio: np.ndarray,
    boundary_state: np.ndarray,
) -> None:
    """Accumulate one [N, C] frame into deterministic float64 running state."""
    if not isinstance(state, MutableMapping):
        raise ValueError("state must be a mutable mapping")
    visible_array, target64, outer64, boundary_array = _validate_frame_inputs(
        frame_index=frame_index,
        visible=visible,
        target_ratio=target_ratio,
        outer_ratio=outer_ratio,
        boundary_state=boundary_state,
    )
    if not state:
        _initialize_state(state, visible_array.shape)
    if tuple(state.get("shape", ())) != visible_array.shape:
        raise ValueError("frame shape does not match accumulator state shape")

    last_frame_index = state["last_frame_index"]
    if last_frame_index is not None and int(frame_index) <= int(last_frame_index):
        raise ValueError("frame_index must be strictly increasing")

    previous_visible = state["previous_visible"]
    if previous_visible is not None:
        both_visible = previous_visible & visible_array
        state["consecutive_visible_count"] += both_visible
        state["target_flicker_sum"][both_visible] += np.abs(
            target64[both_visible] - state["previous_target_ratio"][both_visible]
        )
        state["outer_flicker_sum"][both_visible] += np.abs(
            outer64[both_visible] - state["previous_outer_ratio"][both_visible]
        )
        state["boundary_crossing_count"][both_visible] += (
            boundary_array[both_visible]
            != state["previous_boundary_state"][both_visible]
        )
        state["visibility_transition_count"] += previous_visible != visible_array
        state["visibility_pair_count"] += 1

    count = state["visible_count"]
    new_count = count + visible_array
    target_delta = target64 - state["target_mean"]
    state["target_mean"][visible_array] += (
        target_delta[visible_array] / new_count[visible_array]
    )
    target_delta2 = target64 - state["target_mean"]
    state["target_m2"][visible_array] += (
        target_delta[visible_array] * target_delta2[visible_array]
    )
    outer_delta = outer64 - state["outer_mean"]
    state["outer_mean"][visible_array] += (
        outer_delta[visible_array] / new_count[visible_array]
    )
    outer_delta2 = outer64 - state["outer_mean"]
    state["outer_m2"][visible_array] += (
        outer_delta[visible_array] * outer_delta2[visible_array]
    )
    state["visible_count"] = new_count

    state["last_frame_index"] = int(frame_index)
    state["previous_visible"] = visible_array.copy()
    state["previous_target_ratio"] = target64.copy()
    state["previous_outer_ratio"] = outer64.copy()
    state["previous_boundary_state"] = boundary_array.copy()


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros(numerator.shape, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def finalize_temporal_footprint_evidence(state) -> dict[str, np.ndarray]:
    """Return float32 evidence arrays and integer support arrays with shape [N, C]."""
    if not isinstance(state, MutableMapping) or "shape" not in state:
        raise ValueError("cannot finalize an empty temporal footprint state")

    visible_count = np.asarray(state["visible_count"], dtype=np.int64)
    consecutive_count = np.asarray(
        state["consecutive_visible_count"], dtype=np.int64
    )
    target_variance = _safe_ratio(state["target_m2"], visible_count)
    outer_variance = _safe_ratio(state["outer_m2"], visible_count)
    target_variance = np.maximum(target_variance, 0.0)
    outer_variance = np.maximum(outer_variance, 0.0)

    return {
        "temporal_visible_count": visible_count.astype(np.int32),
        "temporal_consecutive_visible_count": consecutive_count.astype(np.int32),
        "temporal_target_ratio_mean": np.asarray(
            state["target_mean"], dtype=np.float32
        ),
        "temporal_target_ratio_std": np.sqrt(target_variance).astype(np.float32),
        "temporal_target_flicker": _safe_ratio(
            state["target_flicker_sum"], consecutive_count
        ).astype(np.float32),
        "temporal_outer_ratio_mean": np.asarray(
            state["outer_mean"], dtype=np.float32
        ),
        "temporal_outer_ratio_std": np.sqrt(outer_variance).astype(np.float32),
        "temporal_outer_flicker": _safe_ratio(
            state["outer_flicker_sum"], consecutive_count
        ).astype(np.float32),
        "temporal_boundary_crossing_rate": _safe_ratio(
            state["boundary_crossing_count"], consecutive_count
        ).astype(np.float32),
        "temporal_visibility_transition_rate": _safe_ratio(
            state["visibility_transition_count"], state["visibility_pair_count"]
        ).astype(np.float32),
    }
