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


def _validate_metric_matrix(value, *, name: str, shape=None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [N, C]")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape must match {shape}")
    array64 = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array64)):
        raise ValueError(f"{name} must be finite")
    return array64


def compute_temporal_reliability(
    *,
    consecutive_visible_count,
    temporal_outer_flicker,
    temporal_boundary_crossing_rate,
    temporal_target_flicker,
    lambda_outer: float,
    lambda_boundary: float,
    lambda_target: float,
    min_pair_support: int,
) -> tuple[np.ndarray, dict]:
    support_array = np.asarray(consecutive_visible_count)
    if support_array.ndim != 2 or not np.issubdtype(support_array.dtype, np.integer):
        raise ValueError("consecutive_visible_count must be an integer [N, C] array")
    if np.any(support_array < 0):
        raise ValueError("consecutive_visible_count must be non-negative")
    if not isinstance(min_pair_support, int) or min_pair_support <= 0:
        raise ValueError("min_pair_support must be a positive integer")

    shape = support_array.shape
    outer = _validate_metric_matrix(
        temporal_outer_flicker, name="temporal_outer_flicker", shape=shape
    )
    boundary = _validate_metric_matrix(
        temporal_boundary_crossing_rate,
        name="temporal_boundary_crossing_rate",
        shape=shape,
    )
    target = _validate_metric_matrix(
        temporal_target_flicker, name="temporal_target_flicker", shape=shape
    )
    for name, array in (
        ("temporal_outer_flicker", outer),
        ("temporal_boundary_crossing_rate", boundary),
        ("temporal_target_flicker", target),
    ):
        if np.any(array < 0):
            raise ValueError(f"{name} must be non-negative")
    lambdas = {
        "lambda_outer": float(lambda_outer),
        "lambda_boundary": float(lambda_boundary),
        "lambda_target": float(lambda_target),
    }
    if any(not np.isfinite(value) or value < 0 for value in lambdas.values()):
        raise ValueError("temporal reliability lambdas must be finite and non-negative")

    supported = support_array >= min_pair_support
    exponent = -(
        lambdas["lambda_outer"] * outer
        + lambdas["lambda_boundary"] * boundary
        + lambdas["lambda_target"] * target
    )
    reliability64 = supported.astype(np.float64) * np.exp(exponent)
    reliability = np.clip(reliability64, 0.0, 1.0).astype(np.float32)
    supported_count = int(np.count_nonzero(supported))
    total_count = int(supported.size)
    summary = {
        **lambdas,
        "min_pair_support": int(min_pair_support),
        "supported_entry_count": supported_count,
        "total_entry_count": total_count,
        "support_coverage": float(supported_count / total_count) if total_count else 0.0,
        "mean_reliability": float(np.mean(reliability, dtype=np.float64))
        if total_count
        else 0.0,
    }
    return reliability, summary


def _validate_weight_inputs(
    *,
    a5_weights,
    semantic_probs,
    temporal_target_ratio_mean,
    temporal_outer_ratio_mean,
    temporal_reliability,
    consecutive_visible_count,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a5 = _validate_metric_matrix(a5_weights, name="a5_weights")
    shape = a5.shape
    posterior = _validate_metric_matrix(
        semantic_probs, name="semantic_probs", shape=shape
    )
    target = _validate_metric_matrix(
        temporal_target_ratio_mean,
        name="temporal_target_ratio_mean",
        shape=shape,
    )
    outer = _validate_metric_matrix(
        temporal_outer_ratio_mean,
        name="temporal_outer_ratio_mean",
        shape=shape,
    )
    reliability = _validate_metric_matrix(
        temporal_reliability, name="temporal_reliability", shape=shape
    )
    support = np.asarray(consecutive_visible_count)
    if support.shape != shape or not np.issubdtype(support.dtype, np.integer):
        raise ValueError(
            "consecutive_visible_count must be an integer array matching a5_weights"
        )
    if np.any(support < 0):
        raise ValueError("consecutive_visible_count must be non-negative")
    for name, array in (
        ("a5_weights", a5),
        ("semantic_probs", posterior),
        ("temporal_target_ratio_mean", target),
        ("temporal_outer_ratio_mean", outer),
        ("temporal_reliability", reliability),
    ):
        if np.any((array < 0.0) | (array > 1.0)):
            raise ValueError(f"{name} must be in [0, 1]")
    return a5, posterior, target, outer, reliability, support.astype(np.int64)


def _part_summary(
    *,
    part_index: int,
    processed: bool,
    a5_target_mass: float,
    damped_target_mass: float,
    target_floor: float,
    restored_target_mass: float,
    remaining_deficit: float,
    redistributed_indices: list[int],
    cap_saturated_count: int,
    candidate_indices: list[int],
    carrier_min_pair_support: int,
    frozen: bool,
    selection_crossing_count: int,
    weight_l1_from_a5: float,
) -> dict:
    return {
        "part_index": int(part_index),
        "processed": bool(processed),
        "a5_target_mass": float(a5_target_mass),
        "damped_target_mass": float(damped_target_mass),
        "target_floor": float(target_floor),
        "restored_target_mass": float(restored_target_mass),
        "remaining_deficit": float(remaining_deficit),
        "redistributed_gaussian_count": len(redistributed_indices),
        "redistributed_gaussian_indices": redistributed_indices,
        "cap_saturated_count": int(cap_saturated_count),
        "stable_candidate_count": len(candidate_indices),
        "candidate_gaussian_indices": candidate_indices,
        "carrier_min_pair_support": int(carrier_min_pair_support),
        "frozen": bool(frozen),
        "selection_crossing_count": int(selection_crossing_count),
        "weight_l1_from_a5": float(weight_l1_from_a5),
    }


def calibrate_a7_soft_edit_weights(
    *,
    a5_weights,
    semantic_probs,
    temporal_target_ratio_mean,
    temporal_outer_ratio_mean,
    temporal_reliability,
    consecutive_visible_count,
    rho: float,
    min_pair_support: int,
    max_weight_scale_from_posterior: float,
    minimum_carrier_support_ratio: float = 0.0,
    minimum_carrier_existing_weight: float = 0.0,
    carrier_ranking: str = "posterior_target_reliability_support",
    frozen_part_indices: tuple[int, ...] | list[int] = (),
    selection_threshold: float = 0.0,
    preserve_selection_topology: bool = False,
) -> tuple[np.ndarray, dict]:
    """Dampen A5 weights and deterministically restore target mass per part."""
    a5, posterior, target, outer, reliability, support = _validate_weight_inputs(
        a5_weights=a5_weights,
        semantic_probs=semantic_probs,
        temporal_target_ratio_mean=temporal_target_ratio_mean,
        temporal_outer_ratio_mean=temporal_outer_ratio_mean,
        temporal_reliability=temporal_reliability,
        consecutive_visible_count=consecutive_visible_count,
    )
    rho_value = float(rho)
    scale = float(max_weight_scale_from_posterior)
    if not np.isfinite(rho_value) or not 0.0 <= rho_value <= 1.0:
        raise ValueError("rho must be finite and in [0, 1]")
    if not isinstance(min_pair_support, int) or min_pair_support <= 0:
        raise ValueError("min_pair_support must be a positive integer")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("max_weight_scale_from_posterior must be finite and positive")
    carrier_support_ratio = float(minimum_carrier_support_ratio)
    carrier_existing_weight = float(minimum_carrier_existing_weight)
    if not np.isfinite(carrier_support_ratio) or not 0.0 <= carrier_support_ratio <= 1.0:
        raise ValueError("minimum_carrier_support_ratio must be in [0, 1]")
    if not np.isfinite(carrier_existing_weight) or not 0.0 <= carrier_existing_weight <= 1.0:
        raise ValueError("minimum_carrier_existing_weight must be in [0, 1]")
    ranking = str(carrier_ranking)
    supported_rankings = {
        "posterior_target_reliability_support",
        "reliability_support_target_posterior",
    }
    if ranking not in supported_rankings:
        raise ValueError(f"unsupported carrier_ranking: {ranking}")
    frozen_parts = {int(index) for index in frozen_part_indices}
    if any(index < 0 or index >= a5.shape[1] for index in frozen_parts):
        raise ValueError("frozen_part_indices contains an invalid part index")
    threshold = float(selection_threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("selection_threshold must be in [0, 1]")
    preserve_topology = bool(preserve_selection_topology)

    output = a5.copy()
    per_part = []
    for part_index in range(a5.shape[1]):
        a5_part = a5[:, part_index]
        posterior_part = posterior[:, part_index]
        target_part = target[:, part_index]
        outer_part = outer[:, part_index]
        reliability_part = reliability[:, part_index]
        support_part = support[:, part_index]
        selected_a5 = a5_part >= threshold if threshold > 0.0 else a5_part > 0.0
        has_evidence = bool(
            np.any(target_part > 0.0)
            or np.any(outer_part > 0.0)
            or np.any(support_part > 0)
        )
        a5_target_mass = float(np.sum(a5_part * target_part, dtype=np.float64))
        if part_index in frozen_parts:
            per_part.append(
                _part_summary(
                    part_index=part_index,
                    processed=False,
                    a5_target_mass=a5_target_mass,
                    damped_target_mass=a5_target_mass,
                    target_floor=rho_value * a5_target_mass,
                    restored_target_mass=a5_target_mass,
                    remaining_deficit=0.0,
                    redistributed_indices=[],
                    cap_saturated_count=0,
                    candidate_indices=[],
                    carrier_min_pair_support=min_pair_support,
                    frozen=True,
                    selection_crossing_count=0,
                    weight_l1_from_a5=0.0,
                )
            )
            continue
        if not has_evidence:
            per_part.append(
                _part_summary(
                    part_index=part_index,
                    processed=False,
                    a5_target_mass=a5_target_mass,
                    damped_target_mass=a5_target_mass,
                    target_floor=rho_value * a5_target_mass,
                    restored_target_mass=a5_target_mass,
                    remaining_deficit=0.0,
                    redistributed_indices=[],
                    cap_saturated_count=0,
                    candidate_indices=[],
                    carrier_min_pair_support=min_pair_support,
                    frozen=False,
                    selection_crossing_count=0,
                    weight_l1_from_a5=0.0,
                )
            )
            continue

        ceiling = np.minimum(1.0, posterior_part * scale)
        calibrated_part = np.minimum(a5_part * reliability_part, ceiling)
        if preserve_topology:
            if np.any(selected_a5 & (ceiling < threshold)):
                raise ValueError("posterior ceiling cannot preserve A5 selection topology")
            calibrated_part[selected_a5] = np.maximum(
                calibrated_part[selected_a5], threshold
            )
            calibrated_part[~selected_a5] = np.minimum(
                calibrated_part[~selected_a5],
                np.nextafter(threshold, -np.inf),
            )
        damped_target_mass = float(
            np.sum(calibrated_part * target_part, dtype=np.float64)
        )
        target_floor = rho_value * a5_target_mass
        deficit = max(0.0, target_floor - damped_target_mass)
        maximum_support = int(np.max(support_part)) if support_part.size else 0
        carrier_min_pair_support = max(
            min_pair_support,
            int(np.ceil(maximum_support * carrier_support_ratio)),
        )
        eligible = np.flatnonzero(
            (support_part >= carrier_min_pair_support)
            & (target_part > outer_part)
            & (a5_part >= carrier_existing_weight)
            & (selected_a5 if preserve_topology else True)
        )
        if ranking == "reliability_support_target_posterior":
            rank_key = lambda index: (
                -reliability_part[index],
                -support_part[index],
                -target_part[index],
                -posterior_part[index],
                index,
            )
        else:
            rank_key = lambda index: (
                -posterior_part[index],
                -target_part[index],
                -reliability_part[index],
                -support_part[index],
                index,
            )
        candidate_indices = sorted(
            (int(index) for index in eligible),
            key=rank_key,
        )
        redistributed_indices = []
        saturated_indices = []
        for index in candidate_indices:
            if deficit <= 0.0:
                break
            capacity = max(0.0, ceiling[index] - calibrated_part[index])
            if capacity <= 0.0 or target_part[index] <= 0.0:
                continue
            increment = min(capacity, deficit / target_part[index])
            if increment <= 0.0:
                continue
            calibrated_part[index] += increment
            deficit = max(0.0, deficit - increment * target_part[index])
            redistributed_indices.append(index)
            if calibrated_part[index] >= ceiling[index] - 1e-12:
                saturated_indices.append(index)

        output[:, part_index] = calibrated_part
        output_part32 = calibrated_part.astype(np.float32)
        selection_crossing_count = int(
            np.count_nonzero((output_part32 >= threshold) != selected_a5)
        ) if preserve_topology else 0
        restored_target_mass = float(
            np.sum(output_part32.astype(np.float64) * target_part, dtype=np.float64)
        )
        remaining_deficit = max(0.0, target_floor - restored_target_mass)
        per_part.append(
            _part_summary(
                part_index=part_index,
                processed=True,
                a5_target_mass=a5_target_mass,
                damped_target_mass=damped_target_mass,
                target_floor=target_floor,
                restored_target_mass=restored_target_mass,
                remaining_deficit=remaining_deficit,
                redistributed_indices=redistributed_indices,
                cap_saturated_count=len(saturated_indices),
                candidate_indices=candidate_indices,
                carrier_min_pair_support=carrier_min_pair_support,
                frozen=False,
                selection_crossing_count=selection_crossing_count,
                weight_l1_from_a5=float(
                    np.sum(np.abs(output_part32.astype(np.float64) - a5_part))
                ),
            )
        )

    output32 = np.clip(output, 0.0, 1.0).astype(np.float32)
    summary = {
        "rho": rho_value,
        "min_pair_support": int(min_pair_support),
        "max_weight_scale_from_posterior": scale,
        "minimum_carrier_support_ratio": carrier_support_ratio,
        "minimum_carrier_existing_weight": carrier_existing_weight,
        "carrier_ranking": ranking,
        "frozen_part_indices": sorted(frozen_parts),
        "selection_threshold": threshold,
        "preserve_selection_topology": preserve_topology,
        "per_part": per_part,
        "weight_l1_from_a5": float(
            np.sum(np.abs(output32.astype(np.float64) - a5), dtype=np.float64)
        ),
    }
    return output32, summary
