from __future__ import annotations

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from utils.a7c_feasibility_oracle import (
    _validated_stream,
    bisect_feasible_gain,
    evaluate_oracle_gates,
    normalized_flicker,
    solve_fixed_gain_oracle,
)
from utils.a7c_temporal_joint_projection import (
    _append_row,
    _finite_array,
    _solver_options,
)


def _validate_problem(
    *,
    runtime_mass,
    a5_weight,
    streams,
    anchor_gates,
    minimum_gate: float,
    maximum_gate: float,
    selection_threshold: float,
):
    mass = _finite_array("runtime_mass", runtime_mass)
    weights = _finite_array("a5_weight", a5_weight).reshape(-1)
    anchor = _finite_array("anchor_gates", anchor_gates)
    if mass.ndim != 2 or mass.shape[0] < 2 or mass.size == 0:
        raise ValueError("runtime_mass needs shape [frames, carriers]")
    frames, carriers = mass.shape
    if anchor.shape != mass.shape:
        raise ValueError("anchor gates must match runtime mass")
    if weights.shape != (carriers,):
        raise ValueError("a5_weight must match carrier count")
    if np.any(mass < 0.0) or np.any(weights < 0.0):
        raise ValueError("runtime mass and A5 weight must be nonnegative")
    if not 0.0 <= minimum_gate <= maximum_gate <= 1.0:
        raise ValueError("gate bounds are invalid")
    if np.any(anchor < minimum_gate - 1.0e-7) or np.any(
        anchor > maximum_gate + 1.0e-7
    ):
        raise ValueError("anchor gates violate frozen bounds")
    topology_floor = np.maximum(
        float(minimum_gate),
        float(selection_threshold) / np.maximum(weights, 1.0e-8),
    )
    if np.any(topology_floor > maximum_gate + 1.0e-12):
        raise ValueError("topology floor exceeds maximum gate")
    topology_floor = np.minimum(topology_floor, maximum_gate)
    validated = {
        "objective": {
            signal: _validated_stream(
                streams["objective"][signal],
                frames,
                carriers,
                f"objective.{signal}",
            )
            for signal in ("outer", "boundary")
        },
        "guard": {
            signal: _validated_stream(
                streams["guard"][signal],
                frames,
                carriers,
                f"guard.{signal}",
            )
            for signal in ("target", "outer")
        },
    }
    return mass, weights, anchor, topology_floor, validated


def _base_fixed_gain_problem(
    *,
    mass,
    topology_floor,
    streams,
    minimum_gate: float,
    maximum_gate: float,
    proxy_target_response: float,
    maximum_gate_jump: float,
    minimum_target_response: float,
    maximum_soft_iou_drop: float,
    minimum_outer_gain: float,
    minimum_boundary_gain: float,
):
    frames, carriers = mass.shape
    gate_count = frames * carriers
    difference_count = frames - 1
    outer_offset = gate_count
    boundary_offset = gate_count + difference_count
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

    target_base, target_point = streams["guard"]["target"]
    target_fixed = target_base - target_point.sum(axis=1)
    outer_base, outer_point = streams["guard"]["outer"]
    outer_fixed = outer_base - outer_point.sum(axis=1)
    if np.any(target_base <= 0.0) or np.any(target_base + outer_base <= 0.0):
        raise ValueError("guard base denominators must be positive")
    base_iou = target_base / (target_base + outer_base)
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
                - minimum_iou * outer_fixed[frame]
            )
            coefficients = (
                (1.0 - minimum_iou) * target_point[frame]
                - minimum_iou * outer_point[frame]
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
        "outer": float(minimum_outer_gain),
        "boundary": float(minimum_boundary_gain),
    }
    for signal, difference_offset in (
        ("outer", outer_offset),
        ("boundary", boundary_offset),
    ):
        base, point = streams["objective"][signal]
        fixed = base - point.sum(axis=1)
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
                (column, -value) for column, value in coefficients[:-1]
            ] + [(difference_index, -1.0)]
            _append_row(
                rows,
                columns,
                values,
                upper,
                reverse,
                fixed_delta,
            )
        scale = (
            (1.0 - requested[signal])
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
    bounds = [
        (float(topology_floor[index % carriers]), float(maximum_gate))
        for index in range(gate_count)
    ] + [(0.0, None)] * (2 * difference_count)
    return matrix, np.asarray(upper), bounds, gate_count


def _augment_problem(matrix, upper, variable_count: int, extra_rows):
    source = matrix.tocoo()
    rows = source.row.tolist()
    columns = source.col.tolist()
    values = source.data.tolist()
    bounds = np.asarray(upper, dtype=np.float64).tolist()
    for coefficients, bound in extra_rows:
        _append_row(rows, columns, values, bounds, coefficients, bound)
    output = coo_matrix(
        (values, (rows, columns)),
        shape=(len(bounds), int(variable_count)),
        dtype=np.float64,
    ).tocsr()
    return output, np.asarray(bounds, dtype=np.float64)


def _solve_stage(
    objective,
    matrix,
    upper,
    bounds,
    primal_tolerance: float,
    *,
    presolve: bool = True,
):
    options = _solver_options(primal_tolerance)
    options["presolve"] = bool(presolve)
    result = linprog(
        np.asarray(objective, dtype=np.float64),
        A_ub=matrix,
        b_ub=upper,
        bounds=bounds,
        method="highs",
        options=options,
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"teacher linear program failed: {result.message}")
    violation = float(max(np.max(matrix @ result.x - upper), 0.0))
    return result, violation


def solve_lexicographic_fixed_gain_oracle(
    *,
    runtime_mass,
    a5_weight,
    streams,
    anchor_gates,
    minimum_gate: float,
    maximum_gate: float,
    selection_threshold: float,
    proxy_target_response: float,
    maximum_gate_jump: float,
    minimum_target_response: float,
    maximum_soft_iou_drop: float,
    minimum_outer_gain: float,
    minimum_boundary_gain: float,
    lexicographic_tolerance: float = 1.0e-9,
    primal_tolerance: float = 1.0e-9,
    residual_tolerance: float = 1.0e-7,
) -> dict:
    if min(
        lexicographic_tolerance, primal_tolerance, residual_tolerance
    ) <= 0.0:
        raise ValueError("solver tolerances must be positive")
    mass, _, anchor, topology_floor, validated = _validate_problem(
        runtime_mass=runtime_mass,
        a5_weight=a5_weight,
        streams=streams,
        anchor_gates=anchor_gates,
        minimum_gate=minimum_gate,
        maximum_gate=maximum_gate,
        selection_threshold=selection_threshold,
    )
    base_matrix, base_upper, base_bounds, gate_count = _base_fixed_gain_problem(
        mass=mass,
        topology_floor=topology_floor,
        streams=validated,
        minimum_gate=minimum_gate,
        maximum_gate=maximum_gate,
        proxy_target_response=proxy_target_response,
        maximum_gate_jump=maximum_gate_jump,
        minimum_target_response=minimum_target_response,
        maximum_soft_iou_drop=maximum_soft_iou_drop,
        minimum_outer_gain=minimum_outer_gain,
        minimum_boundary_gain=minimum_boundary_gain,
    )
    anchor_flat = anchor.reshape(-1)
    base_count = base_matrix.shape[1]

    rho_index = base_count
    stage_one_rows = []
    for index, anchor_value in enumerate(anchor_flat):
        stage_one_rows.extend(
            (
                (((index, 1.0), (rho_index, -1.0)), anchor_value),
                (((index, -1.0), (rho_index, -1.0)), -anchor_value),
            )
        )
    stage_one_matrix, stage_one_upper = _augment_problem(
        base_matrix, base_upper, base_count + 1, stage_one_rows
    )
    stage_one_objective = np.zeros(base_count + 1)
    stage_one_objective[rho_index] = 1.0
    stage_one, stage_one_violation = _solve_stage(
        stage_one_objective,
        stage_one_matrix,
        stage_one_upper,
        base_bounds + [(0.0, None)],
        primal_tolerance,
    )
    rho_star = float(stage_one.x[rho_index])

    deviation_offset = base_count
    stage_two_rows = []
    for index, anchor_value in enumerate(anchor_flat):
        deviation = deviation_offset + index
        stage_two_rows.extend(
            (
                (((index, 1.0),), anchor_value + rho_star + lexicographic_tolerance),
                (((index, -1.0),), -anchor_value + rho_star + lexicographic_tolerance),
                (((index, 1.0), (deviation, -1.0)), anchor_value),
                (((index, -1.0), (deviation, -1.0)), -anchor_value),
            )
        )
    stage_two_count = base_count + gate_count
    stage_two_matrix, stage_two_upper = _augment_problem(
        base_matrix, base_upper, stage_two_count, stage_two_rows
    )
    stage_two_objective = np.zeros(stage_two_count)
    stage_two_objective[deviation_offset:] = 1.0
    stage_two, stage_two_violation = _solve_stage(
        stage_two_objective,
        stage_two_matrix,
        stage_two_upper,
        base_bounds + [(0.0, None)] * gate_count,
        primal_tolerance,
    )
    total_deviation_star = float(
        np.sum(stage_two.x[deviation_offset:stage_two_count])
    )

    frames, carriers = mass.shape
    change_count = (frames - 1) * carriers
    change_offset = stage_two_count
    stage_three_rows = list(stage_two_rows)
    stage_three_rows.append(
        (
            tuple(
                (deviation_offset + index, 1.0) for index in range(gate_count)
            ),
            total_deviation_star + lexicographic_tolerance,
        )
    )
    for frame in range(1, frames):
        for carrier in range(carriers):
            current = frame * carriers + carrier
            previous = (frame - 1) * carriers + carrier
            change = change_offset + (frame - 1) * carriers + carrier
            stage_three_rows.extend(
                (
                    (
                        ((current, 1.0), (previous, -1.0), (change, -1.0)),
                        0.0,
                    ),
                    (
                        ((current, -1.0), (previous, 1.0), (change, -1.0)),
                        0.0,
                    ),
                )
            )
    stage_three_count = stage_two_count + change_count
    stage_three_matrix, stage_three_upper = _augment_problem(
        base_matrix, base_upper, stage_three_count, stage_three_rows
    )
    stage_three_objective = np.zeros(stage_three_count)
    stage_three_objective[change_offset:] = 1.0
    stage_three, stage_three_violation = _solve_stage(
        stage_three_objective,
        stage_three_matrix,
        stage_three_upper,
        base_bounds
        + [(0.0, None)] * gate_count
        + [(0.0, None)] * change_count,
        primal_tolerance,
        presolve=False,
    )
    gates = stage_three.x[:gate_count].reshape(frames, carriers)
    metrics = evaluate_oracle_gates(gates, streams)
    achieved_proxy = np.sum(mass * gates, axis=1)
    required_proxy = float(proxy_target_response) * np.sum(mass, axis=1)
    direct_violations = (
        float(np.maximum(topology_floor[None, :] - gates, 0.0).max()),
        float(np.maximum(gates - maximum_gate, 0.0).max()),
        float(np.maximum(required_proxy - achieved_proxy, 0.0).max()),
        max(metrics["maximum_adjacent_gate_change"] - maximum_gate_jump, 0.0),
        max(minimum_target_response - metrics["minimum_target_response"], 0.0),
        max(metrics["maximum_soft_iou_drop"] - maximum_soft_iou_drop, 0.0),
        max(minimum_outer_gain - metrics["outer_gain"], 0.0),
        max(minimum_boundary_gain - metrics["boundary_gain"], 0.0),
    )
    maximum_violation = max(
        stage_one_violation,
        stage_two_violation,
        stage_three_violation,
        *direct_violations,
    )
    if not np.isfinite(gates).all() or maximum_violation > residual_tolerance:
        raise RuntimeError(
            f"teacher residual {maximum_violation} exceeds tolerance"
        )
    return {
        "gates": gates,
        "metrics": metrics,
        "certificate": {
            "solver": "scipy.optimize.linprog:highs",
            "scipy_version": str(scipy.__version__),
            "stage_one_status": int(stage_one.status),
            "stage_two_status": int(stage_two.status),
            "stage_three_status": int(stage_three.status),
            "stage_one_presolve": True,
            "stage_two_presolve": True,
            "stage_three_presolve": False,
            "stage_one_maximum_deviation": rho_star,
            "stage_two_total_deviation": total_deviation_star,
            "stage_three_total_gate_change": float(
                np.sum(stage_three.x[change_offset:])
            ),
            "maximum_primal_violation": float(maximum_violation),
        },
    }


def solve_fit_teacher(
    *,
    runtime_mass,
    a5_weight,
    streams,
    anchor_gates,
    minimum_gate: float,
    maximum_gate: float,
    selection_threshold: float,
    proxy_target_response: float,
    maximum_gate_jump: float,
    minimum_target_response: float,
    maximum_soft_iou_drop: float,
    minimum_outer_gain: float,
    boundary_margin: float,
    bisection_tolerance: float,
    lexicographic_tolerance: float,
    primal_tolerance: float,
    residual_tolerance: float,
) -> dict:
    if boundary_margin < 0.0 or bisection_tolerance <= 0.0:
        raise ValueError("capacity margin and tolerance are invalid")
    _validate_problem(
        runtime_mass=runtime_mass,
        a5_weight=a5_weight,
        streams=streams,
        anchor_gates=anchor_gates,
        minimum_gate=minimum_gate,
        maximum_gate=maximum_gate,
        selection_threshold=selection_threshold,
    )
    common = {
        "runtime_mass": runtime_mass,
        "a5_weight": a5_weight,
        "streams": streams,
        "minimum_gate": minimum_gate,
        "maximum_gate": maximum_gate,
        "selection_threshold": selection_threshold,
        "proxy_target_response": proxy_target_response,
        "maximum_gate_jump": maximum_gate_jump,
        "minimum_target_response": minimum_target_response,
        "maximum_soft_iou_drop": maximum_soft_iou_drop,
        "minimum_outer_gain": minimum_outer_gain,
        "primal_tolerance": primal_tolerance,
        "residual_tolerance": residual_tolerance,
    }

    def is_feasible(boundary_gain: float) -> bool:
        try:
            solve_fixed_gain_oracle(
                **common, minimum_boundary_gain=float(boundary_gain)
            )
        except RuntimeError:
            return False
        return True

    capacity = bisect_feasible_gain(
        is_feasible, tolerance=float(bisection_tolerance)
    )
    boundary_request = float(capacity["feasible_lower"]) - float(boundary_margin)
    solved = solve_lexicographic_fixed_gain_oracle(
        runtime_mass=runtime_mass,
        a5_weight=a5_weight,
        streams=streams,
        anchor_gates=anchor_gates,
        minimum_gate=minimum_gate,
        maximum_gate=maximum_gate,
        selection_threshold=selection_threshold,
        proxy_target_response=proxy_target_response,
        maximum_gate_jump=maximum_gate_jump,
        minimum_target_response=minimum_target_response,
        maximum_soft_iou_drop=maximum_soft_iou_drop,
        minimum_outer_gain=minimum_outer_gain,
        minimum_boundary_gain=boundary_request,
        lexicographic_tolerance=lexicographic_tolerance,
        primal_tolerance=primal_tolerance,
        residual_tolerance=residual_tolerance,
    )
    return {
        "capacity": capacity,
        "request": {
            "outer_gain": float(minimum_outer_gain),
            "boundary_gain": boundary_request,
        },
        **solved,
    }


def insert_teacher_segment(
    teacher_gates,
    teacher_mask,
    selected,
    solved,
    *,
    residual_tolerance: float,
) -> None:
    gates = np.asarray(teacher_gates)
    mask = np.asarray(teacher_mask)
    segment = np.asarray(selected, dtype=bool).reshape(-1)
    if gates.ndim != 2 or mask.shape != (gates.shape[0],):
        raise ValueError("teacher arrays are not aligned")
    if mask.dtype != np.bool_ or segment.shape != mask.shape:
        raise ValueError("teacher masks are invalid")
    if not np.any(segment):
        raise ValueError("teacher segment must not be empty")
    if np.any(mask[segment]) or np.any(np.isfinite(gates[segment])):
        raise ValueError("teacher segment overlaps an existing segment")
    certificate = solved.get("certificate", {})
    residual = float(certificate.get("maximum_primal_violation", np.inf))
    if not np.isfinite(residual) or residual > float(residual_tolerance):
        raise ValueError("teacher certificate residual exceeds tolerance")
    values = _finite_array("solved teacher gates", solved["gates"])
    if values.shape != (int(np.count_nonzero(segment)), gates.shape[1]):
        raise ValueError("solved teacher gate shape differs from segment")
    gates[segment] = values
    mask[segment] = True
