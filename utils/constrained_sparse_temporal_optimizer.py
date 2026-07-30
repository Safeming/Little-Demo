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


def assign_temporal_blocks(camera_index, *, block_count: int) -> np.ndarray:
    cameras = np.asarray(camera_index).reshape(-1)
    count = int(block_count)
    if count <= 0:
        raise ValueError("block_count must be positive")
    blocks = np.full(cameras.shape, -1, dtype=np.int16)
    for camera in np.unique(cameras):
        positions = np.flatnonzero(cameras == camera)
        if positions.size < count:
            raise ValueError("each camera must have at least block_count samples")
        for block, members in enumerate(np.array_split(positions, count)):
            blocks[members] = int(block)
    return blocks


def _normalized_adjacent(values) -> float:
    signal = np.asarray(values, dtype=np.float64).reshape(-1)
    if signal.size <= 1:
        return 0.0
    mean = abs(float(np.mean(signal)))
    adjacent = float(np.mean(np.abs(np.diff(signal))))
    return adjacent / max(mean, 1.0e-12)


def evaluate_temporal_block_robustness(
    *,
    base_signals,
    candidate_signals,
    camera_index,
    block_index,
    camera_ids,
    block_ids,
    gain_quantile: float,
) -> dict:
    cameras = np.asarray(camera_index).reshape(-1)
    blocks = np.asarray(block_index).reshape(-1)
    if cameras.shape != blocks.shape:
        raise ValueError("camera_index and block_index must match")
    gains = {signal: [] for signal in ("outer", "boundary")}
    for camera in camera_ids:
        for block in block_ids:
            selected = (cameras == int(camera)) & (blocks == int(block))
            if np.count_nonzero(selected) <= 1:
                raise ValueError("each camera-time block must contain at least two samples")
            for signal in gains:
                base = _normalized_adjacent(np.asarray(base_signals[signal])[selected])
                candidate = _normalized_adjacent(
                    np.asarray(candidate_signals[signal])[selected]
                )
                gains[signal].append(1.0 - float(_safe_ratio(candidate, base)))
    output = {}
    quantile = float(gain_quantile)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("gain_quantile must be in [0, 1]")
    for signal, values in gains.items():
        array = np.asarray(values, dtype=np.float64)
        output[f"{signal}_gains"] = array
        output[f"{signal}_positive_fraction"] = float(np.mean(array > 0.0))
        output[f"{signal}_gain_quantile"] = float(np.quantile(array, quantile))
        output[f"{signal}_gain_median"] = float(np.median(array))
        output[f"{signal}_worst_gain"] = float(np.min(array))
    return output


def rank_temporal_block_metrics(
    metrics,
    *,
    minimum_positive_fraction: float,
    minimum_gain_quantile: float,
    maximum_worst_regression: float,
    cvar_fraction: float,
) -> tuple[float, float]:
    fraction = float(cvar_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("cvar_fraction must be in (0, 1]")
    violation = 0.0
    cvar_losses = []
    for signal in ("outer", "boundary"):
        violation += max(
            0.0,
            float(minimum_positive_fraction)
            - float(metrics[f"{signal}_positive_fraction"]),
        )
        violation += max(
            0.0,
            float(minimum_gain_quantile)
            - float(metrics[f"{signal}_gain_quantile"]),
        )
        violation += max(
            0.0,
            -float(maximum_worst_regression)
            - float(metrics[f"{signal}_worst_gain"]),
        )
        gains = np.sort(np.asarray(metrics[f"{signal}_gains"], dtype=np.float64))
        count = max(1, int(math.ceil(fraction * gains.size)))
        cvar_losses.append(-float(np.mean(gains[:count])))
    return float(violation), float(max(cvar_losses))


def camera_time_fold_specs(camera_ids, block_ids) -> list[dict[str, int]]:
    return [
        {"held_out_camera": int(camera), "held_out_block": int(block)}
        for camera in camera_ids
        for block in block_ids
    ]


def camera_time_fold_passes(camera_evaluation, block_evaluation) -> bool:
    return bool(
        camera_evaluation.get("passed")
        and block_evaluation.get("temporal_passed")
    )


def consensus_fold_weights(
    *,
    a5_weights,
    fold_weights,
    minimum_fold_count: int,
    selection_threshold: float,
) -> dict:
    a5 = np.asarray(a5_weights, dtype=np.float64).reshape(-1)
    folds = np.asarray(fold_weights, dtype=np.float64)
    if folds.ndim != 2 or folds.shape[1] != a5.size:
        raise ValueError("fold_weights must have shape [F, N]")
    minimum = int(minimum_fold_count)
    if minimum <= 0 or minimum > folds.shape[0]:
        raise ValueError("minimum_fold_count must be in [1, fold_count]")
    changed = np.abs(folds - a5[None, :]) > 1.0e-8
    frequency = np.sum(changed, axis=0, dtype=np.int32)
    weights = a5.copy()
    selected = []
    levels = {}
    for index in np.flatnonzero(frequency >= minimum):
        values = folds[changed[:, index], index]
        rounded = np.round(values, decimals=8)
        unique, counts = np.unique(rounded, return_counts=True)
        modal = unique[counts == np.max(counts)]
        level = float(np.max(modal))
        if a5[index] >= float(selection_threshold) and level < float(selection_threshold):
            raise ValueError("consensus level crosses the selection threshold")
        weights[index] = level
        selected.append(int(index))
        levels[str(int(index))] = level
    return {
        "weights": weights.astype(np.float32),
        "selected_indices": selected,
        "selection_frequency": [int(value) for value in frequency],
        "selected_levels": levels,
        "minimum_fold_count": minimum,
        "fold_count": int(folds.shape[0]),
    }


def _rank_improves(trial, current, *, tolerance: float = 1.0e-10) -> bool:
    for trial_value, current_value in zip(trial, current):
        if float(trial_value) < float(current_value) - float(tolerance):
            return True
        if float(trial_value) > float(current_value) + float(tolerance):
            return False
    return False


def consecutive_support_from_sequences(
    *,
    selection_target_sequence,
    selection_outer_sequence,
    camera_index,
    camera_ids,
    visibility_epsilon: float = 1.0e-8,
    segment_index=None,
) -> np.ndarray:
    target = np.asarray(selection_target_sequence, dtype=np.float64)
    outer = np.asarray(selection_outer_sequence, dtype=np.float64)
    cameras = np.asarray(camera_index).reshape(-1)
    if target.shape != outer.shape or target.ndim != 2:
        raise ValueError("selection sequences must have matching shape [S, N]")
    if cameras.shape != (target.shape[0],):
        raise ValueError("camera_index must match the selection sample count")
    segments = None
    if segment_index is not None:
        segments = np.asarray(segment_index).reshape(-1)
        if segments.shape != cameras.shape:
            raise ValueError("segment_index must match the selection sample count")
    support = np.zeros((target.shape[1],), dtype=np.int32)
    for camera in camera_ids:
        selected = cameras == int(camera)
        visible = (target[selected] + outer[selected]) > float(
            visibility_epsilon
        )
        if visible.shape[0] > 1:
            pairs = visible[1:] & visible[:-1]
            if segments is not None:
                camera_segments = segments[selected]
                pairs &= (camera_segments[1:] == camera_segments[:-1])[:, None]
            support += np.sum(pairs, axis=0, dtype=np.int32)
    return support


def should_open_hair_compensation(active_evaluation: dict, maximum_count: int) -> bool:
    return bool(
        active_evaluation.get("constraints_passed")
        and not active_evaluation.get("temporal_passed")
        and int(maximum_count) > 0
    )


def resolve_visibility_limits(
    *,
    maximum_training_visibility_response_ratio: float,
    maximum_audit_visibility_response_ratio: float,
) -> tuple[float, float]:
    training = float(maximum_training_visibility_response_ratio)
    audit = float(maximum_audit_visibility_response_ratio)
    if training > audit:
        raise ValueError("training visibility ratio must not exceed the audit ratio")
    return training, audit


def resolve_target_limits(
    *,
    minimum_training_target_response_ratio: float,
    minimum_audit_target_response_ratio: float,
) -> tuple[float, float]:
    training = float(minimum_training_target_response_ratio)
    audit = float(minimum_audit_target_response_ratio)
    if training < audit:
        raise ValueError("training target ratio must not be below the audit ratio")
    return training, audit


def resolve_temporal_gain_limits(
    *,
    minimum_construction_temporal_gain: float,
    minimum_held_out_temporal_gain: float,
) -> tuple[float, float]:
    construction = float(minimum_construction_temporal_gain)
    held_out = float(minimum_held_out_temporal_gain)
    if held_out > construction:
        raise ValueError("held-out temporal gain must not exceed construction gain")
    return construction, held_out


def capacity_candidate_passes(
    construction_evaluation: dict, audit_evaluation: dict
) -> bool:
    return bool(
        construction_evaluation.get("passed") and audit_evaluation.get("passed")
    )


def _changed_part_indices(a5_weights, candidate_weights, part_indices) -> tuple[int, ...]:
    a5 = np.asarray(a5_weights)
    candidate = np.asarray(candidate_weights)
    return tuple(
        int(index)
        for index in part_indices
        if np.any(np.abs(candidate[:, int(index)] - a5[:, int(index)]) > 1.0e-8)
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
    temporal_block_index=None,
    temporal_block_ids=None,
    minimum_positive_block_fraction: float = 0.0,
    minimum_block_gain_quantile: float = -1.0,
    maximum_worst_block_regression: float = 1.0,
    block_gain_quantile: float = 0.1,
    block_cvar_fraction: float = 0.1,
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

    block_metrics = None

    def rank(metrics, signals):
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
        if temporal_block_index is None:
            return (violation, 0.0, 0.0, score), None
        current_block_metrics = evaluate_temporal_block_robustness(
            base_signals=base_signals,
            candidate_signals=signals,
            camera_index=camera_index,
            block_index=temporal_block_index,
            camera_ids=cameras,
            block_ids=temporal_block_ids,
            gain_quantile=block_gain_quantile,
        )
        block_rank = rank_temporal_block_metrics(
            current_block_metrics,
            minimum_positive_fraction=minimum_positive_block_fraction,
            minimum_gain_quantile=minimum_block_gain_quantile,
            maximum_worst_regression=maximum_worst_block_regression,
            cvar_fraction=block_cvar_fraction,
        )
        return (violation, block_rank[0], block_rank[1], score), current_block_metrics

    metrics = evaluate(current_signals)
    current_rank, block_metrics = rank(metrics, current_signals)
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
                trial_rank, trial_block_metrics = rank(trial_metrics, trial_signals)
                key = (*trial_rank, index, float(level))
                improves = _rank_improves(trial_rank, current_rank)
                if improves and (best is None or key < best[0]):
                    best = (
                        key,
                        index,
                        level,
                        trial,
                        trial_signals,
                        trial_metrics,
                        trial_rank,
                        trial_block_metrics,
                    )
        if best is None:
            break
        (
            _key,
            index,
            level,
            weights,
            current_signals,
            metrics,
            current_rank,
            block_metrics,
        ) = best
        used.add(index)
        accepted_moves.append(
            {
                "gaussian_index": int(index),
                "a5_weight": float(a5[index]),
                "input_weight": float(initial_weights[index]),
                "output_weight": float(level),
                "constraint_violation": float(current_rank[0]),
                "score": float(current_rank[-1]),
            }
        )

    changed = np.flatnonzero(np.abs(weights - a5) > 1.0e-8)
    return {
        "weights": weights.astype(np.float32),
        "changed_indices": [int(value) for value in changed],
        "accepted_moves": accepted_moves,
        "constraint_violation": float(current_rank[0]),
        "block_violation": float(current_rank[1]),
        "block_cvar_loss": float(current_rank[2]),
        "final_score": float(current_rank[-1]),
        "final_metrics": metrics,
        "final_block_metrics": block_metrics,
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
    constraint_part_indices=None,
    minimum_camera_target_ratio: float,
    maximum_camera_soft_iou_drop: float,
    maximum_camera_visibility_response_ratio: float,
    minimum_active_temporal_gain: float,
) -> dict:
    constrained_parts = (
        tuple(int(index) for index in part_indices)
        if constraint_part_indices is None
        else tuple(int(index) for index in constraint_part_indices)
    )
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
        for index in constrained_parts
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
        "constraint_part_indices": [int(index) for index in constrained_parts],
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
        constraint_part_indices=_changed_part_indices(
            a5_weights, weights, (hair_index, lower_index)
        ),
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
    maximum_training_visibility_response_ratio: float | None = None,
    maximum_audit_visibility_response_ratio: float | None = None,
    minimum_training_target_response_ratio: float | None = None,
    minimum_audit_target_response_ratio: float | None = None,
    minimum_held_out_temporal_gain: float | None = None,
) -> dict:
    a5 = np.asarray(a5_weights, dtype=np.float32)
    v4 = np.asarray(v4_weights, dtype=np.float32)
    cameras = np.asarray(camera_index).reshape(-1)
    unique_cameras = tuple(int(value) for value in np.unique(cameras))
    training_visibility, audit_visibility = resolve_visibility_limits(
        maximum_training_visibility_response_ratio=(
            maximum_camera_visibility_response_ratio
            if maximum_training_visibility_response_ratio is None
            else maximum_training_visibility_response_ratio
        ),
        maximum_audit_visibility_response_ratio=(
            maximum_camera_visibility_response_ratio
            if maximum_audit_visibility_response_ratio is None
            else maximum_audit_visibility_response_ratio
        ),
    )
    training_target, audit_target = resolve_target_limits(
        minimum_training_target_response_ratio=(
            minimum_camera_target_ratio
            if minimum_training_target_response_ratio is None
            else minimum_training_target_response_ratio
        ),
        minimum_audit_target_response_ratio=(
            minimum_camera_target_ratio
            if minimum_audit_target_response_ratio is None
            else minimum_audit_target_response_ratio
        ),
    )
    construction_gain, held_out_gain = resolve_temporal_gain_limits(
        minimum_construction_temporal_gain=minimum_active_temporal_gain,
        minimum_held_out_temporal_gain=(
            minimum_active_temporal_gain
            if minimum_held_out_temporal_gain is None
            else minimum_held_out_temporal_gain
        ),
    )
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
        "minimum_camera_target_ratio": training_target,
        "maximum_camera_soft_iou_drop": float(maximum_camera_soft_iou_drop),
        "maximum_camera_visibility_response_ratio": training_visibility,
        "objective_mean_weight": float(objective_mean_weight),
        "objective_absolute_adjacent_weight": float(objective_absolute_adjacent_weight),
        "minimum_active_temporal_gain": construction_gain,
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
        construction = _evaluate_active_candidate(
            a5_weights=a5,
            candidate_weights=fold_weights,
            sequences=sequences,
            target_pixel_count=target_pixel_count,
            camera_index=cameras,
            camera_ids=train,
            part_indices=(hair_index, lower_index),
            constraint_part_indices=_changed_part_indices(
                a5, fold_weights, (hair_index, lower_index)
            ),
            minimum_camera_target_ratio=training_target,
            maximum_camera_soft_iou_drop=maximum_camera_soft_iou_drop,
            maximum_camera_visibility_response_ratio=training_visibility,
            minimum_active_temporal_gain=construction_gain,
        )
        held = _evaluate_active_candidate(
            a5_weights=a5,
            candidate_weights=fold_weights,
            sequences=sequences,
            target_pixel_count=target_pixel_count,
            camera_index=cameras,
            camera_ids=(held_out,),
            part_indices=(hair_index, lower_index),
            minimum_camera_target_ratio=audit_target,
            maximum_camera_soft_iou_drop=maximum_camera_soft_iou_drop,
            maximum_camera_visibility_response_ratio=audit_visibility,
            minimum_active_temporal_gain=held_out_gain,
        )
        folds.append(
            {
                "held_out_camera": held_out,
                "training_cameras": list(train),
                "fold_v4_lower_seed_changed_indices": fold_seed["changed_indices"],
                "optimization": optimization,
                "construction": construction,
                "held_out": held,
                "passed": capacity_candidate_passes(construction, held),
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
    final_construction = _evaluate_active_candidate(
        a5_weights=a5,
        candidate_weights=final_weights,
        sequences=sequences,
        target_pixel_count=target_pixel_count,
        camera_index=cameras,
        camera_ids=unique_cameras,
        part_indices=(hair_index, lower_index),
        constraint_part_indices=_changed_part_indices(
            a5, final_weights, (hair_index, lower_index)
        ),
        minimum_camera_target_ratio=training_target,
        maximum_camera_soft_iou_drop=maximum_camera_soft_iou_drop,
        maximum_camera_visibility_response_ratio=training_visibility,
        minimum_active_temporal_gain=construction_gain,
    )
    final_evaluation = _evaluate_active_candidate(
        a5_weights=a5,
        candidate_weights=final_weights,
        sequences=sequences,
        target_pixel_count=target_pixel_count,
        camera_index=cameras,
        camera_ids=unique_cameras,
        part_indices=(hair_index, lower_index),
        minimum_camera_target_ratio=audit_target,
        maximum_camera_soft_iou_drop=maximum_camera_soft_iou_drop,
        maximum_camera_visibility_response_ratio=audit_visibility,
        minimum_active_temporal_gain=construction_gain,
    )
    return {
        "weights": final_weights,
        "camera_ids": list(unique_cameras),
        "maximum_training_visibility_response_ratio": training_visibility,
        "maximum_audit_visibility_response_ratio": audit_visibility,
        "minimum_training_target_response_ratio": training_target,
        "minimum_audit_target_response_ratio": audit_target,
        "minimum_held_out_temporal_gain": held_out_gain,
        "folds": folds,
        "all_folds_passed": all(fold["passed"] for fold in folds),
        "final": {
            "optimization": final_optimization,
            "construction_evaluation": final_construction,
            "evaluation": final_evaluation,
        },
    }


def run_camera_time_stability_capacity(
    *,
    a5_weights,
    v4_weights,
    sequences,
    target_pixel_count,
    camera_index,
    frame_index,
    hair_index: int,
    lower_index: int,
    selection_threshold: float,
    min_pair_support: int,
    reduction_fractions,
    maximum_changed_fraction: float,
    minimum_camera_target_ratio: float,
    maximum_camera_soft_iou_drop: float,
    maximum_camera_visibility_response_ratio: float,
    objective_mean_weight: float,
    objective_absolute_adjacent_weight: float,
    temporal_block_count: int,
    minimum_stability_fold_count: int,
    minimum_positive_block_fraction: float,
    minimum_block_gain_quantile: float,
    maximum_worst_block_regression: float,
    block_gain_quantile: float,
    block_cvar_fraction: float,
    minimum_aggregate_temporal_gain: float,
    minimum_lower_temporal_gain: float,
    maximum_changed_count: int,
    source_v4_minimum_camera_target_ratio: float = 0.98,
) -> dict:
    a5 = np.asarray(a5_weights, dtype=np.float32)
    v4 = np.asarray(v4_weights, dtype=np.float32)
    cameras = np.asarray(camera_index).reshape(-1)
    frames = np.asarray(frame_index).reshape(-1)
    if cameras.shape != frames.shape:
        raise ValueError("camera_index and frame_index must match")
    blocks = assign_temporal_blocks(cameras, block_count=temporal_block_count)
    camera_ids = tuple(int(value) for value in np.unique(cameras))
    block_ids = tuple(range(int(temporal_block_count)))
    arrays = {key: np.asarray(value) for key, value in sequences.items()}
    pixels = np.asarray(target_pixel_count)
    fold_rows = []
    lower_fold_weights = []

    for spec in camera_time_fold_specs(camera_ids, block_ids):
        held_camera = int(spec["held_out_camera"])
        held_block = int(spec["held_out_block"])
        train_mask = (cameras != held_camera) & (blocks != held_block)
        train_cameras = tuple(value for value in camera_ids if value != held_camera)
        train_blocks = tuple(value for value in block_ids if value != held_block)
        train_sequences = {key: value[train_mask] for key, value in arrays.items()}
        train_camera_index = cameras[train_mask]
        train_block_index = blocks[train_mask]
        fold_support = consecutive_support_from_sequences(
            selection_target_sequence=train_sequences["selection_target"][:, :, lower_index],
            selection_outer_sequence=train_sequences["selection_outer"][:, :, lower_index],
            camera_index=train_camera_index,
            camera_ids=train_cameras,
            segment_index=train_block_index,
        )
        eligible = np.flatnonzero(
            (a5[:, lower_index] >= float(selection_threshold))
            & (fold_support >= int(min_pair_support))
        )
        fold_seed = optimize_sparse_part(
            a5_weights=a5[:, lower_index],
            target_sequence=train_sequences["target"][:, :, lower_index],
            outer_sequence=train_sequences["outer"][:, :, lower_index],
            boundary_sequence=train_sequences["boundary"][:, :, lower_index],
            camera_index=train_camera_index,
            optimization_camera_ids=train_cameras,
            eligible_indices=eligible,
            reduction_fractions=reduction_fractions,
            maximum_changed_count=int(maximum_changed_count),
            minimum_camera_target_ratio=float(source_v4_minimum_camera_target_ratio),
            objective_mean_weight=objective_mean_weight,
            objective_absolute_adjacent_weight=objective_absolute_adjacent_weight,
            selection_threshold=selection_threshold,
        )
        optimized = optimize_constrained_sparse_part(
            a5_weights=a5[:, lower_index],
            initial_weights=fold_seed["weights"],
            edit_target_sequence=train_sequences["target"][:, :, lower_index],
            edit_outer_sequence=train_sequences["outer"][:, :, lower_index],
            edit_boundary_sequence=train_sequences["boundary"][:, :, lower_index],
            selection_target_sequence=train_sequences["selection_target"][:, :, lower_index],
            selection_outer_sequence=train_sequences["selection_outer"][:, :, lower_index],
            target_pixel_count_sequence=pixels[train_mask, lower_index],
            camera_index=train_camera_index,
            optimization_camera_ids=train_cameras,
            eligible_indices=eligible,
            reduction_fractions=reduction_fractions,
            maximum_changed_count=int(maximum_changed_count),
            minimum_camera_target_ratio=minimum_camera_target_ratio,
            maximum_camera_soft_iou_drop=maximum_camera_soft_iou_drop,
            maximum_camera_visibility_response_ratio=maximum_camera_visibility_response_ratio,
            objective_mean_weight=objective_mean_weight,
            objective_absolute_adjacent_weight=objective_absolute_adjacent_weight,
            selection_threshold=selection_threshold,
            temporal_block_index=train_block_index,
            temporal_block_ids=train_blocks,
            minimum_positive_block_fraction=minimum_positive_block_fraction,
            minimum_block_gain_quantile=minimum_block_gain_quantile,
            maximum_worst_block_regression=maximum_worst_block_regression,
            block_gain_quantile=block_gain_quantile,
            block_cvar_fraction=block_cvar_fraction,
        )
        fold_weights = a5.copy()
        fold_weights[:, lower_index] = optimized["weights"]
        lower_fold_weights.append(np.asarray(optimized["weights"], dtype=np.float32))
        held_camera_evaluation = _evaluate_active_candidate(
            a5_weights=a5,
            candidate_weights=fold_weights,
            sequences=arrays,
            target_pixel_count=pixels,
            camera_index=cameras,
            camera_ids=(held_camera,),
            part_indices=(hair_index, lower_index),
            constraint_part_indices=(lower_index,),
            minimum_camera_target_ratio=minimum_camera_target_ratio,
            maximum_camera_soft_iou_drop=maximum_camera_soft_iou_drop,
            maximum_camera_visibility_response_ratio=1.0,
            minimum_active_temporal_gain=0.0,
        )
        held_block_mask = (cameras != held_camera) & (blocks == held_block)
        held_block_cameras = tuple(value for value in camera_ids if value != held_camera)
        held_block_evaluation = _evaluate_active_candidate(
            a5_weights=a5,
            candidate_weights=fold_weights,
            sequences={key: value[held_block_mask] for key, value in arrays.items()},
            target_pixel_count=pixels[held_block_mask],
            camera_index=cameras[held_block_mask],
            camera_ids=held_block_cameras,
            part_indices=(hair_index, lower_index),
            constraint_part_indices=(lower_index,),
            minimum_camera_target_ratio=minimum_camera_target_ratio,
            maximum_camera_soft_iou_drop=maximum_camera_soft_iou_drop,
            maximum_camera_visibility_response_ratio=1.0,
            minimum_active_temporal_gain=0.0,
        )
        fold_rows.append(
            {
                **spec,
                "training_cameras": list(train_cameras),
                "training_blocks": list(train_blocks),
                "fold_v4_lower_seed_changed_indices": fold_seed["changed_indices"],
                "changed_indices": optimized["changed_indices"],
                "block_violation": float(optimized["block_violation"]),
                "held_out_camera_evaluation": held_camera_evaluation,
                "held_out_block_evaluation": held_block_evaluation,
                "passed": camera_time_fold_passes(
                    held_camera_evaluation, held_block_evaluation
                ),
            }
        )

    consensus = consensus_fold_weights(
        a5_weights=a5[:, lower_index],
        fold_weights=np.stack(lower_fold_weights, axis=0),
        minimum_fold_count=minimum_stability_fold_count,
        selection_threshold=selection_threshold,
    )
    final_weights = a5.copy()
    final_weights[:, lower_index] = consensus["weights"]
    full_evaluation = _evaluate_active_candidate(
        a5_weights=a5,
        candidate_weights=final_weights,
        sequences=arrays,
        target_pixel_count=pixels,
        camera_index=cameras,
        camera_ids=camera_ids,
        part_indices=(hair_index, lower_index),
        constraint_part_indices=(lower_index,),
        minimum_camera_target_ratio=minimum_camera_target_ratio,
        maximum_camera_soft_iou_drop=maximum_camera_soft_iou_drop,
        maximum_camera_visibility_response_ratio=maximum_camera_visibility_response_ratio,
        minimum_active_temporal_gain=minimum_aggregate_temporal_gain,
    )
    lower_metrics = full_evaluation["per_part"][str(lower_index)]
    lower_outer_gain = 1.0 - float(np.mean(lower_metrics["outer_normalized_flicker"]))
    lower_boundary_gain = 1.0 - float(
        np.mean(lower_metrics["boundary_normalized_flicker"])
    )
    base_lower = {
        signal: arrays[signal][:, :, lower_index] @ a5[:, lower_index]
        for signal in ("outer", "boundary")
    }
    candidate_lower = {
        signal: arrays[signal][:, :, lower_index] @ final_weights[:, lower_index]
        for signal in ("outer", "boundary")
    }
    block_metrics = evaluate_temporal_block_robustness(
        base_signals=base_lower,
        candidate_signals=candidate_lower,
        camera_index=cameras,
        block_index=blocks,
        camera_ids=camera_ids,
        block_ids=block_ids,
        gain_quantile=block_gain_quantile,
    )
    block_rank = rank_temporal_block_metrics(
        block_metrics,
        minimum_positive_fraction=minimum_positive_block_fraction,
        minimum_gain_quantile=minimum_block_gain_quantile,
        maximum_worst_regression=maximum_worst_block_regression,
        cvar_fraction=block_cvar_fraction,
    )
    lower_gate = (
        lower_outer_gain >= float(minimum_lower_temporal_gain) - 1.0e-7
        and lower_boundary_gain >= float(minimum_lower_temporal_gain) - 1.0e-7
    )
    valid = bool(
        all(row["passed"] for row in fold_rows)
        and full_evaluation["passed"]
        and lower_gate
        and block_rank[0] <= 1.0e-10
    )
    return {
        "weights": final_weights,
        "camera_ids": list(camera_ids),
        "block_ids": list(block_ids),
        "fold_count": len(fold_rows),
        "folds": fold_rows,
        "all_folds_passed": all(row["passed"] for row in fold_rows),
        "consensus": {key: value for key, value in consensus.items() if key != "weights"},
        "final": {
            "evaluation": full_evaluation,
            "lower_outer_gain": lower_outer_gain,
            "lower_boundary_gain": lower_boundary_gain,
            "lower_gate_passed": bool(lower_gate),
            "block_metrics": _jsonable_metrics(block_metrics),
            "block_violation": float(block_rank[0]),
            "block_cvar_loss": float(block_rank[1]),
            "passed": valid,
        },
        "valid": valid,
    }
