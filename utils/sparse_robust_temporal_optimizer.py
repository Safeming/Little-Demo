from __future__ import annotations

import math

import numpy as np


SIGNALS = ("target", "outer", "boundary")


def summarize_camera_signal(values, *, camera_index, camera_ids) -> dict[str, np.ndarray]:
    signal = np.asarray(values, dtype=np.float64).reshape(-1)
    cameras = np.asarray(camera_index).reshape(-1)
    if signal.shape != cameras.shape:
        raise ValueError("values and camera_index must have matching shape")
    outputs = {key: [] for key in ("mean_response", "adjacent_absolute_change", "normalized_flicker")}
    for camera in camera_ids:
        selected = signal[cameras == int(camera)]
        if selected.size == 0:
            raise ValueError(f"camera {camera} has no sequence samples")
        mean = float(np.mean(selected, dtype=np.float64))
        adjacent = float(np.mean(np.abs(np.diff(selected)), dtype=np.float64)) if selected.size > 1 else 0.0
        outputs["mean_response"].append(mean)
        outputs["adjacent_absolute_change"].append(adjacent)
        outputs["normalized_flicker"].append(adjacent / max(abs(mean), 1.0e-12))
    return {key: np.asarray(value, dtype=np.float64) for key, value in outputs.items()}


def _response_ratios(base: dict, current: dict, camera_index, camera_ids) -> dict[str, np.ndarray]:
    ratios = {}
    for signal in SIGNALS:
        baseline = summarize_camera_signal(base[signal], camera_index=camera_index, camera_ids=camera_ids)
        candidate = summarize_camera_signal(current[signal], camera_index=camera_index, camera_ids=camera_ids)
        for metric in ("mean_response", "adjacent_absolute_change", "normalized_flicker"):
            denominator = baseline[metric]
            numerator = candidate[metric]
            ratio = np.ones_like(numerator)
            np.divide(numerator, denominator, out=ratio, where=np.abs(denominator) > 1.0e-12)
            ratios[f"{signal}_{metric}"] = ratio
    return ratios


def _robust_score(ratios: dict[str, np.ndarray], *, mean_weight: float, adjacent_weight: float) -> float:
    outer = ratios["outer_normalized_flicker"]
    boundary = ratios["boundary_normalized_flicker"]
    outer_adjacent = ratios["outer_adjacent_absolute_change"]
    boundary_adjacent = ratios["boundary_adjacent_absolute_change"]
    return float(
        max(float(np.max(outer)), float(np.max(boundary)))
        + float(mean_weight) * float(np.mean(np.concatenate((outer, boundary))))
        + float(adjacent_weight)
        * float(np.mean(np.concatenate((outer_adjacent, boundary_adjacent))))
    )


def evaluate_sparse_part_weights(
    *,
    a5_weights,
    candidate_weights,
    target_sequence,
    outer_sequence,
    boundary_sequence,
    camera_index,
    camera_ids,
) -> dict[str, np.ndarray]:
    a5 = np.asarray(a5_weights, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_weights, dtype=np.float64).reshape(-1)
    sequences = {
        "target": np.asarray(target_sequence, dtype=np.float64),
        "outer": np.asarray(outer_sequence, dtype=np.float64),
        "boundary": np.asarray(boundary_sequence, dtype=np.float64),
    }
    if candidate.shape != a5.shape or any(value.ndim != 2 or value.shape[1] != a5.size for value in sequences.values()):
        raise ValueError("weights and sequences must have compatible [S, N] shapes")
    base = {signal: sequence @ a5 for signal, sequence in sequences.items()}
    current = {signal: sequence @ candidate for signal, sequence in sequences.items()}
    return _response_ratios(base, current, camera_index, camera_ids)


def optimize_sparse_part(
    *,
    a5_weights,
    target_sequence,
    outer_sequence,
    boundary_sequence,
    camera_index,
    optimization_camera_ids,
    eligible_indices,
    reduction_fractions,
    maximum_changed_count: int,
    minimum_camera_target_ratio: float,
    objective_mean_weight: float,
    objective_absolute_adjacent_weight: float,
    selection_threshold: float = 0.2,
) -> dict:
    a5 = np.asarray(a5_weights, dtype=np.float64).reshape(-1)
    sequences = {
        "target": np.asarray(target_sequence, dtype=np.float64),
        "outer": np.asarray(outer_sequence, dtype=np.float64),
        "boundary": np.asarray(boundary_sequence, dtype=np.float64),
    }
    if any(value.ndim != 2 or value.shape[1] != a5.size for value in sequences.values()):
        raise ValueError("renderer sequences must have shape [S, N]")
    cameras = tuple(int(value) for value in optimization_camera_ids)
    if not cameras:
        raise ValueError("optimization_camera_ids must not be empty")
    fractions = tuple(sorted({float(value) for value in reduction_fractions}))
    if not fractions or any(value <= 0.0 or value > 1.0 for value in fractions):
        raise ValueError("reduction_fractions must be in (0, 1]")
    eligible = sorted({int(value) for value in np.asarray(eligible_indices).reshape(-1)})
    weights = a5.copy()
    base = {signal: sequence @ a5 for signal, sequence in sequences.items()}
    current = {signal: value.copy() for signal, value in base.items()}
    ratios = _response_ratios(base, current, camera_index, cameras)
    score = _robust_score(
        ratios,
        mean_weight=objective_mean_weight,
        adjacent_weight=objective_absolute_adjacent_weight,
    )
    accepted_moves = []
    used: set[int] = set()
    for _step in range(max(0, int(maximum_changed_count))):
        best = None
        for index in eligible:
            if index in used:
                continue
            for fraction in fractions:
                new_weight = a5[index] * (1.0 - fraction)
                if a5[index] >= selection_threshold and new_weight < selection_threshold:
                    continue
                delta = weights[index] - new_weight
                if delta <= 0.0:
                    continue
                trial = {
                    signal: current[signal] - delta * sequence[:, index]
                    for signal, sequence in sequences.items()
                }
                trial_ratios = _response_ratios(base, trial, camera_index, cameras)
                if float(np.min(trial_ratios["target_mean_response"])) < float(minimum_camera_target_ratio):
                    continue
                trial_score = _robust_score(
                    trial_ratios,
                    mean_weight=objective_mean_weight,
                    adjacent_weight=objective_absolute_adjacent_weight,
                )
                key = (trial_score, index, -fraction)
                if trial_score < score - 1.0e-7 and (best is None or key < best[0]):
                    best = (key, index, fraction, new_weight, trial, trial_ratios)
        if best is None:
            break
        key, index, fraction, new_weight, current, ratios = best
        score = float(key[0])
        weights[index] = new_weight
        used.add(index)
        accepted_moves.append(
            {
                "gaussian_index": index,
                "reduction_fraction": float(fraction),
                "a5_weight": float(a5[index]),
                "output_weight": float(new_weight),
                "score": score,
            }
        )
    return {
        "weights": weights.astype(np.float32),
        "changed_indices": sorted(used),
        "accepted_moves": accepted_moves,
        "final_score": score,
        "final_ratios": ratios,
    }


def _jsonable_ratios(ratios: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {key: [float(value) for value in np.asarray(values).reshape(-1)] for key, values in ratios.items()}


def run_loco_sparse_capacity(
    *,
    a5_weights,
    sequences,
    camera_index,
    consecutive_visible_count,
    processed_part_indices,
    selection_threshold: float,
    min_pair_support: int,
    reduction_fractions,
    maximum_changed_fraction: float,
    minimum_camera_target_ratio: float,
    objective_mean_weight: float,
    objective_absolute_adjacent_weight: float,
) -> dict:
    a5 = np.asarray(a5_weights, dtype=np.float32)
    support = np.asarray(consecutive_visible_count)
    cameras = np.asarray(camera_index).reshape(-1)
    unique_cameras = [int(value) for value in np.unique(cameras)]
    parts = [int(value) for value in processed_part_indices]
    sequence_arrays = {signal: np.asarray(sequences[signal]) for signal in SIGNALS}
    folds = []
    for held_out in unique_cameras:
        train = tuple(value for value in unique_cameras if value != held_out)
        held_part_ratios = {}
        for part_index in parts:
            eligible = np.flatnonzero(
                (a5[:, part_index] >= float(selection_threshold))
                & (support[:, part_index] >= int(min_pair_support))
            )
            maximum_changed = int(math.floor(float(maximum_changed_fraction) * len(eligible)))
            optimized = optimize_sparse_part(
                a5_weights=a5[:, part_index],
                target_sequence=sequence_arrays["target"][:, :, part_index],
                outer_sequence=sequence_arrays["outer"][:, :, part_index],
                boundary_sequence=sequence_arrays["boundary"][:, :, part_index],
                camera_index=cameras,
                optimization_camera_ids=train,
                eligible_indices=eligible,
                reduction_fractions=reduction_fractions,
                maximum_changed_count=maximum_changed,
                minimum_camera_target_ratio=minimum_camera_target_ratio,
                objective_mean_weight=objective_mean_weight,
                objective_absolute_adjacent_weight=objective_absolute_adjacent_weight,
                selection_threshold=selection_threshold,
            )
            held = evaluate_sparse_part_weights(
                a5_weights=a5[:, part_index],
                candidate_weights=optimized["weights"],
                target_sequence=sequence_arrays["target"][:, :, part_index],
                outer_sequence=sequence_arrays["outer"][:, :, part_index],
                boundary_sequence=sequence_arrays["boundary"][:, :, part_index],
                camera_index=cameras,
                camera_ids=(held_out,),
            )
            held_part_ratios[str(part_index)] = _jsonable_ratios(held)
        aggregate = {
            metric: float(
                np.mean(
                    [held_part_ratios[str(part)][metric][0] for part in parts]
                )
            )
            for metric in (
                "target_mean_response",
                "outer_normalized_flicker",
                "boundary_normalized_flicker",
            )
        }
        target_ok = all(
            held_part_ratios[str(part)]["target_mean_response"][0]
            >= float(minimum_camera_target_ratio)
            for part in parts
        )
        folds.append(
            {
                "held_out_camera": held_out,
                "training_cameras": list(train),
                "held_out_per_part": held_part_ratios,
                "held_out_aggregate": aggregate,
                "passed": bool(
                    target_ok
                    and aggregate["outer_normalized_flicker"] < 1.0
                    and aggregate["boundary_normalized_flicker"] < 1.0
                ),
            }
        )

    final_weights = a5.copy()
    final_parts = {}
    for part_index in parts:
        eligible = np.flatnonzero(
            (a5[:, part_index] >= float(selection_threshold))
            & (support[:, part_index] >= int(min_pair_support))
        )
        maximum_changed = int(math.floor(float(maximum_changed_fraction) * len(eligible)))
        optimized = optimize_sparse_part(
            a5_weights=a5[:, part_index],
            target_sequence=sequence_arrays["target"][:, :, part_index],
            outer_sequence=sequence_arrays["outer"][:, :, part_index],
            boundary_sequence=sequence_arrays["boundary"][:, :, part_index],
            camera_index=cameras,
            optimization_camera_ids=tuple(unique_cameras),
            eligible_indices=eligible,
            reduction_fractions=reduction_fractions,
            maximum_changed_count=maximum_changed,
            minimum_camera_target_ratio=minimum_camera_target_ratio,
            objective_mean_weight=objective_mean_weight,
            objective_absolute_adjacent_weight=objective_absolute_adjacent_weight,
            selection_threshold=selection_threshold,
        )
        final_weights[:, part_index] = optimized["weights"]
        final_parts[str(part_index)] = {
            "changed_indices": optimized["changed_indices"],
            "accepted_moves": optimized["accepted_moves"],
            "final_score": optimized["final_score"],
            "final_ratios": _jsonable_ratios(optimized["final_ratios"]),
        }
    return {
        "weights": final_weights,
        "camera_ids": unique_cameras,
        "processed_part_indices": parts,
        "folds": folds,
        "all_folds_passed": all(fold["passed"] for fold in folds),
        "final": {"per_part": final_parts},
    }
