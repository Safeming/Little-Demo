from __future__ import annotations

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from utils.a7c_temporal_joint_projection import (
    _append_row,
    _finite_array,
    _solver_options,
)


def normalized_flicker(values, epsilon: float = 1.0e-12) -> float:
    signal = _finite_array("signal", values).reshape(-1)
    if signal.size < 2:
        raise ValueError("normalized flicker needs at least two frames")
    return float(
        np.mean(np.abs(np.diff(signal)))
        / max(abs(float(np.mean(signal))), float(epsilon))
    )


def _validated_stream(stream, frames: int, carriers: int, name: str):
    base = _finite_array(f"{name}.base", stream["base"]).reshape(-1)
    point = _finite_array(f"{name}.point", stream["point"])
    if base.shape != (frames,) or point.shape != (frames, carriers):
        raise ValueError(f"{name} stream shape differs from gates")
    return base, point


def _compose(base: np.ndarray, point: np.ndarray, gates: np.ndarray):
    return base - point.sum(axis=1) + np.sum(point * gates, axis=1)


def soft_iou_linear_slack(
    candidate_target,
    candidate_outer,
    base_target,
    base_outer,
    maximum_drop: float,
) -> np.ndarray:
    candidate_target = _finite_array(
        "candidate_target", candidate_target
    ).reshape(-1)
    candidate_outer = _finite_array(
        "candidate_outer", candidate_outer
    ).reshape(-1)
    base_target = _finite_array("base_target", base_target).reshape(-1)
    base_outer = _finite_array("base_outer", base_outer).reshape(-1)
    if not (
        candidate_target.shape
        == candidate_outer.shape
        == base_target.shape
        == base_outer.shape
    ):
        raise ValueError("soft-IoU arrays must share shape")
    base_denominator = base_target + base_outer
    candidate_denominator = candidate_target + candidate_outer
    if np.any(base_denominator <= 0.0) or np.any(candidate_denominator <= 0.0):
        raise ValueError("soft-IoU denominators must be positive")
    minimum_iou = base_target / base_denominator - float(maximum_drop)
    return (1.0 - minimum_iou) * candidate_target - (
        minimum_iou * candidate_outer
    )


def evaluate_oracle_gates(gates, streams) -> dict:
    gate_values = _finite_array("gates", gates)
    if gate_values.ndim != 2 or gate_values.shape[0] < 2:
        raise ValueError("oracle gates need shape [frames, carriers]")
    frames, carriers = gate_values.shape
    objective = {}
    for signal in ("outer", "boundary"):
        base, point = _validated_stream(
            streams["objective"][signal], frames, carriers, f"objective.{signal}"
        )
        candidate = _compose(base, point, gate_values)
        objective[signal] = {
            "base": base,
            "candidate": candidate,
            "gain": 1.0
            - normalized_flicker(candidate)
            / max(normalized_flicker(base), 1.0e-12),
        }
    target_base, target_point = _validated_stream(
        streams["guard"]["target"], frames, carriers, "guard.target"
    )
    outer_base, outer_point = _validated_stream(
        streams["guard"]["outer"], frames, carriers, "guard.outer"
    )
    target_candidate = _compose(target_base, target_point, gate_values)
    outer_candidate = _compose(outer_base, outer_point, gate_values)
    if np.any(target_base <= 0.0):
        raise ValueError("guard target base must be positive")
    base_denominator = target_base + outer_base
    candidate_denominator = target_candidate + outer_candidate
    if np.any(base_denominator <= 0.0) or np.any(candidate_denominator <= 0.0):
        raise ValueError("guard soft-IoU denominator must be positive")
    base_iou = target_base / base_denominator
    candidate_iou = target_candidate / candidate_denominator
    jumps = np.abs(np.diff(gate_values, axis=0))
    return {
        "outer_gain": float(objective["outer"]["gain"]),
        "boundary_gain": float(objective["boundary"]["gain"]),
        "minimum_target_response": float(
            np.min(target_candidate / target_base)
        ),
        "maximum_soft_iou_drop": float(np.max(base_iou - candidate_iou)),
        "maximum_adjacent_gate_change": float(np.max(jumps)),
    }


def solve_fixed_gain_oracle(
    *,
    runtime_mass,
    a5_weight,
    streams,
    minimum_gate: float,
    maximum_gate: float,
    selection_threshold: float,
    proxy_target_response: float,
    maximum_gate_jump: float,
    minimum_target_response: float,
    maximum_soft_iou_drop: float,
    minimum_outer_gain,
    minimum_boundary_gain,
    primal_tolerance: float = 1.0e-9,
    residual_tolerance: float = 1.0e-7,
) -> dict:
    mass = _finite_array("runtime_mass", runtime_mass)
    weights = _finite_array("a5_weight", a5_weight).reshape(-1)
    if mass.ndim != 2 or mass.shape[0] < 2 or mass.size == 0:
        raise ValueError("runtime_mass needs shape [frames, carriers]")
    frames, carriers = mass.shape
    if weights.shape != (carriers,):
        raise ValueError("a5_weight must match carrier count")
    if np.any(mass < 0.0) or np.any(weights < 0.0):
        raise ValueError("runtime mass and A5 weight must be nonnegative")
    if not 0.0 <= minimum_gate <= maximum_gate <= 1.0:
        raise ValueError("gate bounds are invalid")

    topology_floor = np.maximum(
        float(minimum_gate),
        float(selection_threshold) / np.maximum(weights, 1.0e-8),
    )
    if np.any(topology_floor > maximum_gate + 1.0e-12):
        raise ValueError("topology floor exceeds maximum gate")
    topology_floor = np.minimum(topology_floor, maximum_gate)

    objective_streams = {}
    for signal in ("outer", "boundary"):
        objective_streams[signal] = _validated_stream(
            streams["objective"][signal],
            frames,
            carriers,
            f"objective.{signal}",
        )
    guard_target = _validated_stream(
        streams["guard"]["target"], frames, carriers, "guard.target"
    )
    guard_outer = _validated_stream(
        streams["guard"]["outer"], frames, carriers, "guard.outer"
    )

    gate_count = frames * carriers
    difference_count = frames - 1
    outer_difference_offset = gate_count
    boundary_difference_offset = gate_count + difference_count
    variable_count = gate_count + 2 * difference_count
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    upper: list[float] = []

    for frame in range(frames):
        offset = frame * carriers
        _append_row(
            rows,
            columns,
            values,
            upper,
            [
                (offset + carrier, -mass[frame, carrier])
                for carrier in range(carriers)
            ],
            -float(proxy_target_response) * float(mass[frame].sum()),
        )
    for frame in range(1, frames):
        current = frame * carriers
        previous = (frame - 1) * carriers
        for carrier in range(carriers):
            _append_row(
                rows,
                columns,
                values,
                upper,
                ((current + carrier, 1.0), (previous + carrier, -1.0)),
                maximum_gate_jump,
            )
            _append_row(
                rows,
                columns,
                values,
                upper,
                ((current + carrier, -1.0), (previous + carrier, 1.0)),
                maximum_gate_jump,
            )

    target_base, target_point = guard_target
    target_fixed = target_base - target_point.sum(axis=1)
    outer_guard_base, outer_guard_point = guard_outer
    outer_guard_fixed = outer_guard_base - outer_guard_point.sum(axis=1)
    if np.any(target_base <= 0.0) or np.any(
        target_base + outer_guard_base <= 0.0
    ):
        raise ValueError("guard base denominators must be positive")
    base_iou = target_base / (target_base + outer_guard_base)
    for frame in range(frames):
        offset = frame * carriers
        required_target = float(minimum_target_response) * target_base[frame]
        _append_row(
            rows,
            columns,
            values,
            upper,
            [
                (offset + carrier, -target_point[frame, carrier])
                for carrier in range(carriers)
            ],
            target_fixed[frame] - required_target,
        )
        minimum_iou = base_iou[frame] - float(maximum_soft_iou_drop)
        if minimum_iou > 0.0:
            fixed_slack = (
                (1.0 - minimum_iou) * target_fixed[frame]
                - minimum_iou * outer_guard_fixed[frame]
            )
            coefficients = (
                (1.0 - minimum_iou) * target_point[frame]
                - minimum_iou * outer_guard_point[frame]
            )
            _append_row(
                rows,
                columns,
                values,
                upper,
                [
                    (offset + carrier, -coefficients[carrier])
                    for carrier in range(carriers)
                ],
                fixed_slack,
            )

    requested = {
        "outer": minimum_outer_gain,
        "boundary": minimum_boundary_gain,
    }
    for signal_index, signal in enumerate(("outer", "boundary")):
        base, point = objective_streams[signal]
        fixed = base - point.sum(axis=1)
        difference_offset = (
            outer_difference_offset
            if signal_index == 0
            else boundary_difference_offset
        )
        for frame in range(1, frames):
            difference_index = difference_offset + frame - 1
            fixed_delta = fixed[frame] - fixed[frame - 1]
            coefficients = []
            for carrier in range(carriers):
                coefficients.append(
                    (frame * carriers + carrier, point[frame, carrier])
                )
                coefficients.append(
                    (
                        (frame - 1) * carriers + carrier,
                        -point[frame - 1, carrier],
                    )
                )
            coefficients.append((difference_index, -1.0))
            _append_row(
                rows,
                columns,
                values,
                upper,
                coefficients,
                -fixed_delta,
            )
            reverse = [
                (column, -value)
                for column, value in coefficients[:-1]
            ] + [(difference_index, -1.0)]
            _append_row(
                rows,
                columns,
                values,
                upper,
                reverse,
                fixed_delta,
            )
        if requested[signal] is not None:
            scale = (
                (1.0 - float(requested[signal]))
                * normalized_flicker(base)
                * difference_count
                / frames
            )
            coefficients = [
                (difference_offset + index, 1.0)
                for index in range(difference_count)
            ]
            for frame in range(frames):
                for carrier in range(carriers):
                    coefficients.append(
                        (
                            frame * carriers + carrier,
                            -scale * point[frame, carrier],
                        )
                    )
            _append_row(
                rows,
                columns,
                values,
                upper,
                coefficients,
                scale * float(fixed.sum()),
            )

    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(len(upper), variable_count),
        dtype=np.float64,
    ).tocsr()
    upper_array = np.asarray(upper, dtype=np.float64)
    gate_bounds = [
        (float(topology_floor[index % carriers]), float(maximum_gate))
        for index in range(gate_count)
    ]
    result = linprog(
        np.zeros(variable_count, dtype=np.float64),
        A_ub=matrix,
        b_ub=upper_array,
        bounds=gate_bounds + [(0.0, None)] * (2 * difference_count),
        method="highs",
        options=_solver_options(primal_tolerance),
    )
    if not result.success:
        raise RuntimeError(f"oracle linear program failed: {result.message}")

    gates = result.x[:gate_count].reshape(frames, carriers)
    metrics = evaluate_oracle_gates(gates, streams)
    achieved_proxy = np.sum(mass * gates, axis=1)
    required_proxy = float(proxy_target_response) * np.sum(mass, axis=1)
    matrix_violation = float(
        max(np.max(matrix @ result.x - upper_array), 0.0)
    )
    direct_violations = [
        float(np.maximum(topology_floor[None, :] - gates, 0.0).max()),
        float(np.maximum(gates - maximum_gate, 0.0).max()),
        float(np.maximum(required_proxy - achieved_proxy, 0.0).max()),
        max(
            metrics["maximum_adjacent_gate_change"] - maximum_gate_jump,
            0.0,
        ),
        max(minimum_target_response - metrics["minimum_target_response"], 0.0),
        max(metrics["maximum_soft_iou_drop"] - maximum_soft_iou_drop, 0.0),
    ]
    if minimum_outer_gain is not None:
        direct_violations.append(
            max(float(minimum_outer_gain) - metrics["outer_gain"], 0.0)
        )
    if minimum_boundary_gain is not None:
        direct_violations.append(
            max(float(minimum_boundary_gain) - metrics["boundary_gain"], 0.0)
        )
    maximum_violation = max([matrix_violation] + direct_violations)
    if not np.isfinite(gates).all() or maximum_violation > residual_tolerance:
        raise RuntimeError(
            f"oracle residual {maximum_violation} exceeds tolerance"
        )
    return {
        "success": True,
        "gates": gates,
        "metrics": metrics,
        "certificate": {
            "solver": "scipy.optimize.linprog:highs",
            "scipy_version": str(scipy.__version__),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "maximum_matrix_violation": matrix_violation,
            "maximum_primal_violation": float(maximum_violation),
        },
    }


def bisect_feasible_gain(
    is_feasible,
    *,
    lower: float = -0.01,
    upper: float = 1.00001,
    tolerance: float = 1.0e-5,
) -> dict:
    feasible_lower = float(lower)
    infeasible_upper = float(upper)
    if tolerance <= 0.0 or feasible_lower >= infeasible_upper:
        raise ValueError("bisection interval and tolerance are invalid")
    if not bool(is_feasible(feasible_lower)):
        raise RuntimeError("oracle lower endpoint is infeasible")
    if bool(is_feasible(infeasible_upper)):
        raise RuntimeError("oracle upper endpoint must be infeasible")
    iterations = 0
    while infeasible_upper - feasible_lower > float(tolerance):
        middle = 0.5 * (feasible_lower + infeasible_upper)
        if bool(is_feasible(middle)):
            feasible_lower = middle
        else:
            infeasible_upper = middle
        iterations += 1
    return {
        "feasible_lower": feasible_lower,
        "infeasible_upper": infeasible_upper,
        "interval_width": infeasible_upper - feasible_lower,
        "iterations": iterations,
    }


def promotion_summary_passes(summary: dict, contract: dict) -> bool:
    improves = bool(
        float(summary["outer_gain"])
        > float(contract["r1_1_f1_outer_gain"])
        and float(summary["boundary_gain"])
        > float(contract["r1_1_f1_boundary_gain"])
    )
    distribution = all(
        float(summary[f"{signal}_positive_block_fraction"])
        >= float(contract["minimum_positive_block_fraction"])
        and float(summary[f"{signal}_block_gain_quantile"])
        >= float(contract["minimum_block_gain_quantile"]) - 1.0e-9
        and float(summary[f"{signal}_worst_block_gain"])
        >= -float(contract["maximum_worst_block_regression"]) - 1.0e-9
        for signal in ("outer", "boundary")
    )
    return bool(
        float(summary["outer_gain"])
        >= float(contract["minimum_outer_gain"])
        and float(summary["boundary_gain"])
        >= float(contract["minimum_boundary_gain"])
        and improves
        and distribution
    )


def classify_oracle(
    *,
    sufficient_audit_passed: bool,
    optimistic_summary: dict,
    contract: dict,
) -> str:
    if bool(sufficient_audit_passed):
        return "CERTIFIED_FEASIBLE"
    if not promotion_summary_passes(optimistic_summary, contract):
        return "CERTIFIED_INFEASIBLE"
    return "UNRESOLVED"
