from __future__ import annotations

import math

import numpy as np

from utils.sparse_robust_temporal_optimizer import (
    _response_ratios,
    _robust_score,
    optimize_sparse_part,
    summarize_camera_signal,
)


def _camera_means(values, *, camera_index, camera_ids) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float64).reshape(-1)
    cameras = np.asarray(camera_index).reshape(-1)
    return np.asarray(
        [float(np.mean(signal[cameras == int(camera)])) for camera in camera_ids],
        dtype=np.float64,
    )


def _safe_ratio(numerator, denominator) -> np.ndarray:
    top = np.asarray(numerator, dtype=np.float64)
    bottom = np.asarray(denominator, dtype=np.float64)
    result = np.ones_like(top)
    supported = np.abs(bottom) > 1.0e-12
    np.divide(top, bottom, out=result, where=supported)
    result[(~supported) & (np.abs(top) > 1.0e-12)] = np.inf
    return result


def consecutive_support_from_sequences(
    *,
    selection_target_sequence,
    selection_outer_sequence,
    camera_index,
    camera_ids,
    visibility_epsilon: float = 1.0e-8,
) -> np.ndarray:
    target = np.asarray(selection_target_sequence, dtype=np.float64)
    outer = np.asarray(selection_outer_sequence, dtype=np.float64)
    cameras = np.asarray(camera_index).reshape(-1)
    if target.shape != outer.shape or target.ndim != 2:
        raise ValueError("selection sequences must have matching shape [S, N]")
    if cameras.shape != (target.shape[0],):
        raise ValueError("camera_index must match the selection sample count")
    support = np.zeros((target.shape[1],), dtype=np.int32)
    for camera in camera_ids:
        visible = (target[cameras == int(camera)] + outer[cameras == int(camera)]) > float(
            visibility_epsilon
        )
        if visible.shape[0] > 1:
            support += np.sum(visible[1:] & visible[:-1], axis=0, dtype=np.int32)
    return support


def should_open_hair_compensation(active_evaluation: dict, maximum_count: int) -> bool:
    return bool(
        active_evaluation.get("constraints_passed")
        and not active_evaluation.get("temporal_passed")
        and int(maximum_count) > 0
    )


def evaluate_constrained_part_signals(
    *,
    base_signals,
    candidate_signals,
    target_pixel_count_sequence,
    camera_index,
    camera_ids,
) -> dict[str, np.ndarray]:
    target_pixels = np.asarray(target_pixel_count_sequence, dtype=np.float64).reshape(-1)
    required = {"target", "outer", "boundary", "selection_target", "selection_outer"}
    if set(base_signals) != required or set(candidate_signals) != required:
        raise ValueError("constrained signals must contain edit and selection fields")
    base = {
        key: np.asarray(value, dtype=np.float64).reshape(-1)
        for key, value in base_signals.items()
    }
    current = {
        key: np.asarray(value, dtype=np.float64).reshape(-1)
        for key, value in candidate_signals.items()
    }
    sample_count = base["target"].size
    if any(value.shape != (sample_count,) for value in (*base.values(), *current.values())):
        raise ValueError("constrained signals must have matching sample counts")
    if target_pixels.shape != (sample_count,):
        raise ValueError("target_pixel_count_sequence must match the sample count")

    ratios = _response_ratios(
        {key: base[key] for key in ("target", "outer", "boundary")},
        {key: current[key] for key in ("target", "outer", "boundary")},
        camera_index=camera_index,
        camera_ids=camera_ids,
    )
    base_inside = base["selection_target"]
    candidate_inside = current["selection_target"]
    base_outside = base["selection_outer"]
    candidate_outside = current["selection_outer"]
    base_soft_iou = np.clip(base_inside, 0.0, target_pixels) / np.maximum(
        target_pixels + np.maximum(base_outside, 0.0), 1.0e-12
    )
    candidate_soft_iou = np.clip(candidate_inside, 0.0, target_pixels) / np.maximum(
        target_pixels + np.maximum(candidate_outside, 0.0), 1.0e-12
    )
    base_soft_mean = _camera_means(
        base_soft_iou, camera_index=camera_index, camera_ids=camera_ids
    )
    candidate_soft_mean = _camera_means(
        candidate_soft_iou, camera_index=camera_index, camera_ids=camera_ids
    )

    base_response = base["target"] / np.maximum(target_pixels, 1.0)
    candidate_response = current["target"] / np.maximum(target_pixels, 1.0)
    base_visibility = summarize_camera_signal(
        base_response, camera_index=camera_index, camera_ids=camera_ids
    )["normalized_flicker"]
    candidate_visibility = summarize_camera_signal(
        candidate_response, camera_index=camera_index, camera_ids=camera_ids
    )["normalized_flicker"]
    ratios.update(
        {
            "soft_iou_drop": base_soft_mean - candidate_soft_mean,
            "soft_iou_mean": candidate_soft_mean,
            "visibility_response_ratio": _safe_ratio(
                candidate_visibility, base_visibility
            ),
        }
    )
    return ratios


def evaluate_constrained_part_weights(
    *,
    a5_weights,
    candidate_weights,
    edit_target_sequence,
    edit_outer_sequence,
    edit_boundary_sequence,
    selection_target_sequence,
    selection_outer_sequence,
    target_pixel_count_sequence,
    camera_index,
    camera_ids,
) -> dict[str, np.ndarray]:
    a5 = np.asarray(a5_weights, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_weights, dtype=np.float64).reshape(-1)
    if candidate.shape != a5.shape:
        raise ValueError("candidate_weights must match a5_weights")
    sequences = {
        "target": np.asarray(edit_target_sequence, dtype=np.float64),
        "outer": np.asarray(edit_outer_sequence, dtype=np.float64),
        "boundary": np.asarray(edit_boundary_sequence, dtype=np.float64),
        "selection_target": np.asarray(selection_target_sequence, dtype=np.float64),
        "selection_outer": np.asarray(selection_outer_sequence, dtype=np.float64),
    }
    if any(value.ndim != 2 or value.shape[1] != a5.size for value in sequences.values()):
        raise ValueError("contribution sequences must have shape [S, N]")
    return evaluate_constrained_part_signals(
        base_signals={key: value @ a5 for key, value in sequences.items()},
        candidate_signals={key: value @ candidate for key, value in sequences.items()},
        target_pixel_count_sequence=target_pixel_count_sequence,
        camera_index=camera_index,
        camera_ids=camera_ids,
    )


def _constraint_violation(
    metrics: dict[str, np.ndarray],
    *,
    minimum_camera_target_ratio: float,
    maximum_camera_soft_iou_drop: float,
    maximum_camera_visibility_response_ratio: float,
) -> float:
    target = max(
        0.0,
        float(minimum_camera_target_ratio)
        - float(np.min(metrics["target_mean_response"])),
    )
    soft = max(
        0.0,
        float(np.max(metrics["soft_iou_drop"]))
        - float(maximum_camera_soft_iou_drop),
    )
    visibility_max = float(np.max(metrics["visibility_response_ratio"]))
    visibility = (
        1.0e6
        if not np.isfinite(visibility_max)
        else max(0.0, visibility_max - float(maximum_camera_visibility_response_ratio))
    )
    return target + soft + visibility


def optimize_constrained_sparse_part(
    *,
    a5_weights,
    initial_weights,
    edit_target_sequence,
    edit_outer_sequence,
    edit_boundary_sequence,
    selection_target_sequence,
    selection_outer_sequence,
    target_pixel_count_sequence,
    camera_index,
    optimization_camera_ids,
    eligible_indices,
    reduction_fractions,
    maximum_changed_count: int,
    minimum_camera_target_ratio: float,
    maximum_camera_soft_iou_drop: float,
    maximum_camera_visibility_response_ratio: float,
    objective_mean_weight: float,
    objective_absolute_adjacent_weight: float,
    selection_threshold: float = 0.2,
) -> dict:
    a5 = np.asarray(a5_weights, dtype=np.float64).reshape(-1)
    weights = np.asarray(initial_weights, dtype=np.float64).reshape(-1).copy()
    if weights.shape != a5.shape:
        raise ValueError("initial_weights must match a5_weights")
    cameras = tuple(int(value) for value in optimization_camera_ids)
    eligible = sorted({int(value) for value in np.asarray(eligible_indices).reshape(-1)})
    fractions = tuple(sorted({float(value) for value in reduction_fractions}))
    if not cameras or not fractions:
        raise ValueError("optimization cameras and reduction fractions are required")

    sequences = {
        "target": np.asarray(edit_target_sequence, dtype=np.float64),
        "outer": np.asarray(edit_outer_sequence, dtype=np.float64),
        "boundary": np.asarray(edit_boundary_sequence, dtype=np.float64),
        "selection_target": np.asarray(selection_target_sequence, dtype=np.float64),
        "selection_outer": np.asarray(selection_outer_sequence, dtype=np.float64),
    }
    if any(value.ndim != 2 or value.shape[1] != a5.size for value in sequences.values()):
        raise ValueError("contribution sequences must have shape [S, N]")
    base_signals = {key: value @ a5 for key, value in sequences.items()}
    current_signals = {key: value @ weights for key, value in sequences.items()}

    def evaluate(signals):
        return evaluate_constrained_part_signals(
            base_signals=base_signals,
            candidate_signals=signals,
            target_pixel_count_sequence=target_pixel_count_sequence,
            camera_index=camera_index,
            camera_ids=cameras,
        )

    def rank(metrics):
        violation = _constraint_violation(
            metrics,
            minimum_camera_target_ratio=minimum_camera_target_ratio,
            maximum_camera_soft_iou_drop=maximum_camera_soft_iou_drop,
            maximum_camera_visibility_response_ratio=maximum_camera_visibility_response_ratio,
        )
        score = _robust_score(
            metrics,
            mean_weight=objective_mean_weight,
            adjacent_weight=objective_absolute_adjacent_weight,
        )
        return violation, score

    metrics = evaluate(current_signals)
    current_rank = rank(metrics)
    accepted_moves = []
    used = set()
    maximum_steps = max(0, int(maximum_changed_count)) + int(
        np.count_nonzero(np.abs(weights - a5) > 1.0e-8)
    )
    for _step in range(maximum_steps):
        best = None
        for index in eligible:
            if index in used:
                continue
            levels = [a5[index]] + [a5[index] * (1.0 - value) for value in fractions]
            for level in levels:
                if abs(level - weights[index]) <= 1.0e-10:
                    continue
                if a5[index] >= selection_threshold and level < selection_threshold:
                    continue
                trial = weights.copy()
                trial[index] = level
                changed_count = int(np.count_nonzero(np.abs(trial - a5) > 1.0e-8))
                if changed_count > int(maximum_changed_count):
                    continue
                delta = weights[index] - level
                trial_signals = {
                    key: current_signals[key] - delta * sequence[:, index]
                    for key, sequence in sequences.items()
                }
                trial_metrics = evaluate(trial_signals)
                trial_rank = rank(trial_metrics)
                key = (trial_rank[0], trial_rank[1], index, float(level))
                improves = (
                    trial_rank[0] < current_rank[0] - 1.0e-10
                    or (
                        trial_rank[0] <= current_rank[0] + 1.0e-10
                        and trial_rank[1] < current_rank[1] - 1.0e-7
                    )
                )
                if improves and (best is None or key < best[0]):
                    best = (
                        key,
                        index,
                        level,
                        trial,
                        trial_signals,
                        trial_metrics,
                        trial_rank,
                    )
        if best is None:
            break
        _key, index, level, weights, current_signals, metrics, current_rank = best
        used.add(index)
        accepted_moves.append(
            {
                "gaussian_index": int(index),
                "a5_weight": float(a5[index]),
                "input_weight": float(initial_weights[index]),
                "output_weight": float(level),
                "constraint_violation": float(current_rank[0]),
                "score": float(current_rank[1]),
            }
        )

    changed = np.flatnonzero(np.abs(weights - a5) > 1.0e-8)
    return {
        "weights": weights.astype(np.float32),
        "changed_indices": [int(value) for value in changed],
        "accepted_moves": accepted_moves,
        "constraint_violation": float(current_rank[0]),
        "final_score": float(current_rank[1]),
        "final_metrics": metrics,
    }


def _jsonable_metrics(metrics: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {
        key: [float(value) for value in np.asarray(values).reshape(-1)]
        for key, values in metrics.items()
    }


def _evaluate_active_candidate(
    *,
    a5_weights,
    candidate_weights,
    sequences,
    target_pixel_count,
    camera_index,
    camera_ids,
    part_indices,
    minimum_camera_target_ratio: float,
    maximum_camera_soft_iou_drop: float,
    maximum_camera_visibility_response_ratio: float,
    minimum_active_temporal_gain: float,
) -> dict:
    per_part = {}
    for part_index in part_indices:
        metrics = evaluate_constrained_part_weights(
            a5_weights=a5_weights[:, part_index],
            candidate_weights=candidate_weights[:, part_index],
            edit_target_sequence=sequences["target"][:, :, part_index],
            edit_outer_sequence=sequences["outer"][:, :, part_index],
            edit_boundary_sequence=sequences["boundary"][:, :, part_index],
            selection_target_sequence=sequences["selection_target"][:, :, part_index],
            selection_outer_sequence=sequences["selection_outer"][:, :, part_index],
            target_pixel_count_sequence=target_pixel_count[:, part_index],
            camera_index=camera_index,
            camera_ids=camera_ids,
        )
        per_part[str(part_index)] = metrics
    aggregate_outer = np.mean(
        [per_part[str(index)]["outer_normalized_flicker"] for index in part_indices],
        axis=0,
    )
    aggregate_boundary = np.mean(
        [per_part[str(index)]["boundary_normalized_flicker"] for index in part_indices],
        axis=0,
    )
    constraints_ok = all(
        float(np.min(per_part[str(index)]["target_mean_response"]))
        >= float(minimum_camera_target_ratio) - 1.0e-7
        and float(np.max(per_part[str(index)]["soft_iou_drop"]))
        <= float(maximum_camera_soft_iou_drop) + 1.0e-7
        and float(np.max(per_part[str(index)]["visibility_response_ratio"]))
        <= float(maximum_camera_visibility_response_ratio) + 1.0e-7
        for index in part_indices
    )
    temporal_ok = (
        float(np.mean(aggregate_outer))
        <= 1.0 - float(minimum_active_temporal_gain) + 1.0e-7
        and float(np.mean(aggregate_boundary))
        <= 1.0 - float(minimum_active_temporal_gain) + 1.0e-7
        and float(np.max(aggregate_outer)) < 1.0
        and float(np.max(aggregate_boundary)) < 1.0
    )
    return {
        "camera_ids": [int(value) for value in camera_ids],
        "per_part": {
            key: _jsonable_metrics(value) for key, value in per_part.items()
        },
        "aggregate_outer_normalized_flicker": [
            float(value) for value in aggregate_outer
        ],
        "aggregate_boundary_normalized_flicker": [
            float(value) for value in aggregate_boundary
        ],
        "constraints_passed": bool(constraints_ok),
        "temporal_passed": bool(temporal_ok),
        "passed": bool(constraints_ok and temporal_ok),
    }


def _optimize_v5_candidate(
    *,
    a5_weights,
    v4_weights,
    sequences,
    target_pixel_count,
    camera_index,
    optimization_camera_ids,
    consecutive_visible_count,
    hair_index: int,
    lower_index: int,
    selection_threshold: float,
    min_pair_support: int,
    reduction_fractions,
    maximum_changed_fraction: float,
    maximum_hair_changed_count: int,
    minimum_camera_target_ratio: float,
    maximum_camera_soft_iou_drop: float,
    maximum_camera_visibility_response_ratio: float,
    objective_mean_weight: float,
    objective_absolute_adjacent_weight: float,
    minimum_active_temporal_gain: float,
) -> tuple[np.ndarray, dict]:
    weights = np.asarray(a5_weights, dtype=np.float32).copy()
    summaries = {}

    lower_eligible = np.flatnonzero(
        (a5_weights[:, lower_index] >= float(selection_threshold))
        & (consecutive_visible_count[:, lower_index] >= int(min_pair_support))
    )
    lower_maximum = int(
        math.floor(float(maximum_changed_fraction) * len(lower_eligible))
    )
    lower = optimize_constrained_sparse_part(
        a5_weights=a5_weights[:, lower_index],
        initial_weights=v4_weights[:, lower_index],
        edit_target_sequence=sequences["target"][:, :, lower_index],
        edit_outer_sequence=sequences["outer"][:, :, lower_index],
        edit_boundary_sequence=sequences["boundary"][:, :, lower_index],
        selection_target_sequence=sequences["selection_target"][:, :, lower_index],
        selection_outer_sequence=sequences["selection_outer"][:, :, lower_index],
        target_pixel_count_sequence=target_pixel_count[:, lower_index],
        camera_index=camera_index,
        optimization_camera_ids=optimization_camera_ids,
        eligible_indices=lower_eligible,
        reduction_fractions=reduction_fractions,
        maximum_changed_count=lower_maximum,
        minimum_camera_target_ratio=minimum_camera_target_ratio,
        maximum_camera_soft_iou_drop=maximum_camera_soft_iou_drop,
        maximum_camera_visibility_response_ratio=maximum_camera_visibility_response_ratio,
        objective_mean_weight=objective_mean_weight,
        objective_absolute_adjacent_weight=objective_absolute_adjacent_weight,
        selection_threshold=selection_threshold,
    )
    weights[:, lower_index] = lower["weights"]
    summaries[str(lower_index)] = {
        **{key: value for key, value in lower.items() if key not in {"weights", "final_metrics"}},
        "final_metrics": _jsonable_metrics(lower["final_metrics"]),
    }

    active = _evaluate_active_candidate(
        a5_weights=a5_weights,
        candidate_weights=weights,
        sequences=sequences,
        target_pixel_count=target_pixel_count,
        camera_index=camera_index,
        camera_ids=optimization_camera_ids,
        part_indices=(hair_index, lower_index),
        minimum_camera_target_ratio=minimum_camera_target_ratio,
        maximum_camera_soft_iou_drop=maximum_camera_soft_iou_drop,
        maximum_camera_visibility_response_ratio=maximum_camera_visibility_response_ratio,
        minimum_active_temporal_gain=minimum_active_temporal_gain,
    )
    if should_open_hair_compensation(active, maximum_hair_changed_count):
        hair_eligible = np.flatnonzero(
            (a5_weights[:, hair_index] >= float(selection_threshold))
            & (consecutive_visible_count[:, hair_index] >= int(min_pair_support))
        )
        hair = optimize_constrained_sparse_part(
            a5_weights=a5_weights[:, hair_index],
            initial_weights=a5_weights[:, hair_index],
            edit_target_sequence=sequences["target"][:, :, hair_index],
            edit_outer_sequence=sequences["outer"][:, :, hair_index],
            edit_boundary_sequence=sequences["boundary"][:, :, hair_index],
            selection_target_sequence=sequences["selection_target"][:, :, hair_index],
            selection_outer_sequence=sequences["selection_outer"][:, :, hair_index],
            target_pixel_count_sequence=target_pixel_count[:, hair_index],
            camera_index=camera_index,
            optimization_camera_ids=optimization_camera_ids,
            eligible_indices=hair_eligible,
            reduction_fractions=reduction_fractions,
            maximum_changed_count=int(maximum_hair_changed_count),
            minimum_camera_target_ratio=minimum_camera_target_ratio,
            maximum_camera_soft_iou_drop=maximum_camera_soft_iou_drop,
            maximum_camera_visibility_response_ratio=maximum_camera_visibility_response_ratio,
            objective_mean_weight=objective_mean_weight,
            objective_absolute_adjacent_weight=objective_absolute_adjacent_weight,
            selection_threshold=selection_threshold,
        )
        weights[:, hair_index] = hair["weights"]
        summaries[str(hair_index)] = {
            **{key: value for key, value in hair.items() if key not in {"weights", "final_metrics"}},
            "final_metrics": _jsonable_metrics(hair["final_metrics"]),
        }
    else:
        summaries[str(hair_index)] = {
            "changed_indices": [],
            "accepted_moves": [],
            "constraint_violation": 0.0,
            "initial_policy": "a5",
        }
    return weights, summaries


def run_constrained_v5_capacity(
    *,
    a5_weights,
    v4_weights,
    sequences,
    target_pixel_count,
    camera_index,
    consecutive_visible_count,
    hair_index: int,
    lower_index: int,
    selection_threshold: float,
    min_pair_support: int,
    reduction_fractions,
    maximum_changed_fraction: float,
    maximum_hair_changed_count: int,
    minimum_camera_target_ratio: float,
    maximum_camera_soft_iou_drop: float,
    maximum_camera_visibility_response_ratio: float,
    objective_mean_weight: float,
    objective_absolute_adjacent_weight: float,
    minimum_active_temporal_gain: float,
    source_v4_minimum_camera_target_ratio: float = 0.98,
) -> dict:
    a5 = np.asarray(a5_weights, dtype=np.float32)
    v4 = np.asarray(v4_weights, dtype=np.float32)
    cameras = np.asarray(camera_index).reshape(-1)
    unique_cameras = tuple(int(value) for value in np.unique(cameras))
    kwargs = {
        "a5_weights": a5,
        "v4_weights": v4,
        "sequences": sequences,
        "target_pixel_count": target_pixel_count,
        "camera_index": cameras,
        "hair_index": int(hair_index),
        "lower_index": int(lower_index),
        "selection_threshold": float(selection_threshold),
        "min_pair_support": int(min_pair_support),
        "reduction_fractions": reduction_fractions,
        "maximum_changed_fraction": float(maximum_changed_fraction),
        "maximum_hair_changed_count": int(maximum_hair_changed_count),
        "minimum_camera_target_ratio": float(minimum_camera_target_ratio),
        "maximum_camera_soft_iou_drop": float(maximum_camera_soft_iou_drop),
        "maximum_camera_visibility_response_ratio": float(maximum_camera_visibility_response_ratio),
        "objective_mean_weight": float(objective_mean_weight),
        "objective_absolute_adjacent_weight": float(objective_absolute_adjacent_weight),
        "minimum_active_temporal_gain": float(minimum_active_temporal_gain),
    }
    folds = []
    for held_out in unique_cameras:
        train = tuple(value for value in unique_cameras if value != held_out)
        fold_support = np.zeros_like(np.asarray(consecutive_visible_count), dtype=np.int32)
        for part_index in (hair_index, lower_index):
            fold_support[:, part_index] = consecutive_support_from_sequences(
                selection_target_sequence=np.asarray(sequences["selection_target"])[
                    :, :, part_index
                ],
                selection_outer_sequence=np.asarray(sequences["selection_outer"])[
                    :, :, part_index
                ],
                camera_index=cameras,
                camera_ids=train,
            )
        fold_v4 = a5.copy()
        lower_eligible = np.flatnonzero(
            (a5[:, lower_index] >= float(selection_threshold))
            & (fold_support[:, lower_index] >= int(min_pair_support))
        )
        lower_maximum = int(
            math.floor(float(maximum_changed_fraction) * len(lower_eligible))
        )
        fold_seed = optimize_sparse_part(
            a5_weights=a5[:, lower_index],
            target_sequence=np.asarray(sequences["target"])[:, :, lower_index],
            outer_sequence=np.asarray(sequences["outer"])[:, :, lower_index],
            boundary_sequence=np.asarray(sequences["boundary"])[:, :, lower_index],
            camera_index=cameras,
            optimization_camera_ids=train,
            eligible_indices=lower_eligible,
            reduction_fractions=reduction_fractions,
            maximum_changed_count=lower_maximum,
            minimum_camera_target_ratio=float(source_v4_minimum_camera_target_ratio),
            objective_mean_weight=objective_mean_weight,
            objective_absolute_adjacent_weight=objective_absolute_adjacent_weight,
            selection_threshold=selection_threshold,
        )
        fold_v4[:, lower_index] = fold_seed["weights"]
        fold_kwargs = dict(kwargs)
        fold_kwargs["v4_weights"] = fold_v4
        fold_kwargs["consecutive_visible_count"] = fold_support
        fold_weights, optimization = _optimize_v5_candidate(
            optimization_camera_ids=train, **fold_kwargs
        )
        held = _evaluate_active_candidate(
            a5_weights=a5,
            candidate_weights=fold_weights,
            sequences=sequences,
            target_pixel_count=target_pixel_count,
            camera_index=cameras,
            camera_ids=(held_out,),
            part_indices=(hair_index, lower_index),
            minimum_camera_target_ratio=minimum_camera_target_ratio,
            maximum_camera_soft_iou_drop=maximum_camera_soft_iou_drop,
            maximum_camera_visibility_response_ratio=maximum_camera_visibility_response_ratio,
            minimum_active_temporal_gain=minimum_active_temporal_gain,
        )
        folds.append(
            {
                "held_out_camera": held_out,
                "training_cameras": list(train),
                "fold_v4_lower_seed_changed_indices": fold_seed["changed_indices"],
                "optimization": optimization,
                "held_out": held,
                "passed": bool(held["passed"]),
            }
        )

    final_support = np.zeros_like(np.asarray(consecutive_visible_count), dtype=np.int32)
    for part_index in (hair_index, lower_index):
        final_support[:, part_index] = consecutive_support_from_sequences(
            selection_target_sequence=np.asarray(sequences["selection_target"])[
                :, :, part_index
            ],
            selection_outer_sequence=np.asarray(sequences["selection_outer"])[
                :, :, part_index
            ],
            camera_index=cameras,
            camera_ids=unique_cameras,
        )
    final_kwargs = dict(kwargs)
    final_kwargs["consecutive_visible_count"] = final_support
    final_weights, final_optimization = _optimize_v5_candidate(
        optimization_camera_ids=unique_cameras, **final_kwargs
    )
    final_evaluation = _evaluate_active_candidate(
        a5_weights=a5,
        candidate_weights=final_weights,
        sequences=sequences,
        target_pixel_count=target_pixel_count,
        camera_index=cameras,
        camera_ids=unique_cameras,
        part_indices=(hair_index, lower_index),
        minimum_camera_target_ratio=minimum_camera_target_ratio,
        maximum_camera_soft_iou_drop=maximum_camera_soft_iou_drop,
        maximum_camera_visibility_response_ratio=maximum_camera_visibility_response_ratio,
        minimum_active_temporal_gain=minimum_active_temporal_gain,
    )
    return {
        "weights": final_weights,
        "camera_ids": list(unique_cameras),
        "folds": folds,
        "all_folds_passed": all(fold["passed"] for fold in folds),
        "final": {
            "optimization": final_optimization,
            "evaluation": final_evaluation,
        },
    }
