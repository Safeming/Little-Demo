from __future__ import annotations

import numpy as np


def target_floor_deficit_is_significant(
    remaining_deficit: float, a5_target_mass: float
) -> bool:
    tolerance = max(1.0e-6, 1.0e-6 * abs(float(a5_target_mass)))
    return float(remaining_deficit) > tolerance


def _matrix(value, *, name: str, shape=None, integer: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or (shape is not None and array.shape != shape):
        raise ValueError(f"{name} must have shape {shape or '[N, C]'}")
    if integer and not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must use an integer dtype")
    array64 = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array64)) or np.any(array64 < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    return array


def calibrate_bounded_a7_weights(
    *,
    a5_weights,
    target_contribution_mean,
    temporal_reliability,
    consecutive_visible_count,
    rho: float,
    min_pair_support: int,
    minimum_weight_ratio_from_a5: float,
    restore_target_mass: bool,
    maximum_part_weight_l1_from_a5: float,
    frozen_part_indices=(),
    selection_threshold: float = 0.2,
    preserve_selection_topology: bool = True,
) -> tuple[np.ndarray, dict]:
    a5 = _matrix(a5_weights, name="a5_weights").astype(np.float64)
    shape = a5.shape
    target = _matrix(
        target_contribution_mean, name="target_contribution_mean", shape=shape
    ).astype(np.float64)
    reliability = _matrix(
        temporal_reliability, name="temporal_reliability", shape=shape
    ).astype(np.float64)
    support = _matrix(
        consecutive_visible_count,
        name="consecutive_visible_count",
        shape=shape,
        integer=True,
    ).astype(np.int64)
    if np.any(a5 > 1.0) or np.any(reliability > 1.0):
        raise ValueError("weights and reliability must be in [0, 1]")
    values = {
        "rho": float(rho),
        "minimum_weight_ratio_from_a5": float(minimum_weight_ratio_from_a5),
        "maximum_part_weight_l1_from_a5": float(maximum_part_weight_l1_from_a5),
        "selection_threshold": float(selection_threshold),
    }
    if not 0.0 <= values["rho"] <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    if not 0.0 <= values["minimum_weight_ratio_from_a5"] <= 1.0:
        raise ValueError("minimum_weight_ratio_from_a5 must be in [0, 1]")
    if not np.isfinite(values["maximum_part_weight_l1_from_a5"]) or values[
        "maximum_part_weight_l1_from_a5"
    ] < 0.0:
        raise ValueError("maximum_part_weight_l1_from_a5 must be non-negative")
    if not 0.0 <= values["selection_threshold"] <= 1.0:
        raise ValueError("selection_threshold must be in [0, 1]")
    if not isinstance(min_pair_support, int) or min_pair_support <= 0:
        raise ValueError("min_pair_support must be a positive integer")

    frozen = {int(index) for index in frozen_part_indices}
    if any(index < 0 or index >= shape[1] for index in frozen):
        raise ValueError("frozen_part_indices contains an invalid index")

    output = a5.copy()
    invalid_reasons: list[str] = []
    per_part = []
    total_crossings = 0
    l1_limit = values["maximum_part_weight_l1_from_a5"]
    for part_index in range(shape[1]):
        a5_part = a5[:, part_index]
        target_part = target[:, part_index]
        selected = a5_part >= values["selection_threshold"]
        a5_target_mass = float(np.sum(a5_part * target_part, dtype=np.float64))
        target_floor = values["rho"] * a5_target_mass
        if part_index in frozen:
            per_part.append(
                {
                    "part_index": part_index,
                    "frozen": True,
                    "a5_target_mass": a5_target_mass,
                    "target_floor": target_floor,
                    "restored_target_mass": a5_target_mass,
                    "remaining_deficit": 0.0,
                    "restored_gaussian_indices": [],
                    "weight_l1_from_a5": 0.0,
                    "selection_crossing_count": 0,
                }
            )
            continue

        lower = a5_part * values["minimum_weight_ratio_from_a5"]
        if preserve_selection_topology:
            lower[selected] = np.maximum(
                lower[selected], values["selection_threshold"]
            )
        calibrated = np.maximum(a5_part * reliability[:, part_index], lower)
        calibrated = np.minimum(calibrated, a5_part)

        l1 = float(np.sum(a5_part - calibrated, dtype=np.float64))
        if l1 > l1_limit and l1 > 0.0:
            calibrated = a5_part - (a5_part - calibrated) * (l1_limit / l1)

        restored_indices: list[int] = []
        current_target_mass = float(
            np.sum(calibrated * target_part, dtype=np.float64)
        )
        deficit = max(0.0, target_floor - current_target_mass)
        if restore_target_mass and deficit > 0.0:
            eligible = np.flatnonzero(
                (support[:, part_index] >= min_pair_support)
                & selected
                & (target_part > 0.0)
            )
            ranked = sorted(
                (int(index) for index in eligible),
                key=lambda index: (
                    -reliability[index, part_index],
                    -support[index, part_index],
                    -target_part[index],
                    index,
                ),
            )
            for index in ranked:
                if deficit <= 0.0:
                    break
                capacity = max(0.0, a5_part[index] - calibrated[index])
                if capacity <= 0.0:
                    continue
                increment = min(capacity, deficit / target_part[index])
                calibrated[index] += increment
                deficit = max(0.0, deficit - increment * target_part[index])
                restored_indices.append(index)

        output[:, part_index] = calibrated
        output32 = calibrated.astype(np.float32)
        crossings = int(
            np.count_nonzero(
                (output32 >= values["selection_threshold"]) != selected
            )
        ) if preserve_selection_topology else 0
        total_crossings += crossings
        restored_target_mass = float(
            np.sum(output32.astype(np.float64) * target_part, dtype=np.float64)
        )
        remaining = max(0.0, target_floor - restored_target_mass)
        if target_floor_deficit_is_significant(remaining, a5_target_mass):
            invalid_reasons.append(f"target_floor_unreachable:{part_index}")
        if crossings:
            invalid_reasons.append(f"selection_topology_crossing:{part_index}")
        per_part.append(
            {
                "part_index": part_index,
                "frozen": False,
                "a5_target_mass": a5_target_mass,
                "target_floor": target_floor,
                "restored_target_mass": restored_target_mass,
                "remaining_deficit": remaining,
                "restored_gaussian_indices": restored_indices,
                "weight_l1_from_a5": float(
                    np.sum(a5_part - output32.astype(np.float64), dtype=np.float64)
                ),
                "selection_crossing_count": crossings,
            }
        )

    output32 = output.astype(np.float32)
    maximum_above = float(np.max(output32.astype(np.float64) - a5)) if output32.size else 0.0
    if maximum_above > 1.0e-7:
        invalid_reasons.append("weight_above_a5")
    return output32, {
        "valid": not invalid_reasons,
        "invalid_reasons": sorted(set(invalid_reasons)),
        "rho": values["rho"],
        "minimum_weight_ratio_from_a5": values[
            "minimum_weight_ratio_from_a5"
        ],
        "restore_target_mass": bool(restore_target_mass),
        "maximum_part_weight_l1_from_a5": l1_limit,
        "selection_threshold": values["selection_threshold"],
        "preserve_selection_topology": bool(preserve_selection_topology),
        "selection_crossing_count": total_crossings,
        "maximum_weight_above_a5": max(0.0, maximum_above),
        "weight_l1_from_a5": float(
            np.sum(a5 - output32.astype(np.float64), dtype=np.float64)
        ),
        "per_part": per_part,
    }
