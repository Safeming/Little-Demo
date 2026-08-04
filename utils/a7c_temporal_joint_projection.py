from __future__ import annotations

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _finite_array(name: str, value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _append_row(rows, columns, values, upper, coefficients, bound) -> None:
    row = len(upper)
    for column, value in coefficients:
        if value != 0.0:
            rows.append(row)
            columns.append(int(column))
            values.append(float(value))
    upper.append(float(bound))


def _common_constraints(
    *,
    variable_count: int,
    raw: np.ndarray,
    mass: np.ndarray,
    proxy_target_response: float,
    maximum_gate_jump: float,
    stage: int,
    rho_bound: float | None = None,
):
    frames, carriers = raw.shape
    gate_count = raw.size
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    upper: list[float] = []

    for index, raw_value in enumerate(raw.reshape(-1)):
        if stage == 1:
            rho_index = gate_count
            _append_row(
                rows,
                columns,
                values,
                upper,
                ((index, 1.0), (rho_index, -1.0)),
                raw_value,
            )
            _append_row(
                rows,
                columns,
                values,
                upper,
                ((index, -1.0), (rho_index, -1.0)),
                -raw_value,
            )
        else:
            deviation_index = gate_count + index
            _append_row(
                rows,
                columns,
                values,
                upper,
                ((index, 1.0), (deviation_index, -1.0)),
                raw_value,
            )
            _append_row(
                rows,
                columns,
                values,
                upper,
                ((index, -1.0), (deviation_index, -1.0)),
                -raw_value,
            )
            _append_row(
                rows,
                columns,
                values,
                upper,
                ((index, 1.0),),
                raw_value + float(rho_bound),
            )
            _append_row(
                rows,
                columns,
                values,
                upper,
                ((index, -1.0),),
                -raw_value + float(rho_bound),
            )

    for frame in range(frames):
        offset = frame * carriers
        coefficients = [
            (offset + carrier, -mass[frame, carrier])
            for carrier in range(carriers)
        ]
        required = float(proxy_target_response) * float(mass[frame].sum())
        _append_row(
            rows,
            columns,
            values,
            upper,
            coefficients,
            -required,
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

    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(len(upper), variable_count),
        dtype=np.float64,
    ).tocsr()
    return matrix, np.asarray(upper, dtype=np.float64)


def _solver_options(primal_tolerance: float) -> dict[str, float]:
    return {
        "primal_feasibility_tolerance": float(primal_tolerance),
        "dual_feasibility_tolerance": float(primal_tolerance),
        "ipm_optimality_tolerance": min(float(primal_tolerance), 1.0e-10),
    }


def solve_temporal_joint_projection(
    *,
    raw_gates,
    runtime_mass,
    a5_weight,
    minimum_gate: float,
    maximum_gate: float,
    selection_threshold: float,
    proxy_target_response: float,
    maximum_gate_jump: float,
    rho_tolerance: float = 1.0e-9,
    primal_tolerance: float = 1.0e-9,
    residual_tolerance: float = 1.0e-7,
) -> dict:
    raw = _finite_array("raw_gates", raw_gates)
    mass = _finite_array("runtime_mass", runtime_mass)
    weights = _finite_array("a5_weight", a5_weight).reshape(-1)
    if raw.ndim != 2 or raw.size == 0:
        raise ValueError("raw_gates must have non-empty shape [frames, carriers]")
    if mass.shape != raw.shape:
        raise ValueError("runtime_mass must match raw_gates")
    if weights.shape != (raw.shape[1],):
        raise ValueError("a5_weight must match the carrier dimension")
    if np.any(mass < 0.0) or np.any(weights < 0.0):
        raise ValueError("runtime mass and A5 weight must be nonnegative")
    if not 0.0 <= minimum_gate <= maximum_gate <= 1.0:
        raise ValueError("gate bounds are invalid")
    if not minimum_gate <= proxy_target_response <= maximum_gate:
        raise ValueError("proxy target response is outside gate bounds")
    if selection_threshold < 0.0 or maximum_gate_jump < 0.0:
        raise ValueError("selection threshold and gate jump must be nonnegative")
    if min(rho_tolerance, primal_tolerance, residual_tolerance) <= 0.0:
        raise ValueError("solver tolerances must be positive")
    if np.any(raw < minimum_gate - 1.0e-7) or np.any(
        raw > maximum_gate + 1.0e-7
    ):
        raise ValueError("raw gates violate frozen bounds")

    topology_floor = np.maximum(
        float(minimum_gate),
        float(selection_threshold) / np.maximum(weights, 1.0e-8),
    )
    if np.any(topology_floor > maximum_gate + 1.0e-12):
        raise ValueError("topology floor exceeds maximum gate")
    topology_floor = np.minimum(topology_floor, maximum_gate)
    gate_count = raw.size
    gate_bounds = [
        (float(topology_floor[index % raw.shape[1]]), float(maximum_gate))
        for index in range(gate_count)
    ]
    options = _solver_options(primal_tolerance)

    stage_one_matrix, stage_one_upper = _common_constraints(
        variable_count=gate_count + 1,
        raw=raw,
        mass=mass,
        proxy_target_response=proxy_target_response,
        maximum_gate_jump=maximum_gate_jump,
        stage=1,
    )
    stage_one_objective = np.zeros(gate_count + 1, dtype=np.float64)
    stage_one_objective[-1] = 1.0
    stage_one = linprog(
        stage_one_objective,
        A_ub=stage_one_matrix,
        b_ub=stage_one_upper,
        bounds=gate_bounds + [(0.0, None)],
        method="highs",
        options=options,
    )
    if not stage_one.success or not np.isfinite(stage_one.fun):
        raise RuntimeError(f"stage-one projection failed: {stage_one.message}")
    rho_star = float(stage_one.x[-1])

    rho_bound = rho_star + float(rho_tolerance)
    stage_two_matrix, stage_two_upper = _common_constraints(
        variable_count=2 * gate_count,
        raw=raw,
        mass=mass,
        proxy_target_response=proxy_target_response,
        maximum_gate_jump=maximum_gate_jump,
        stage=2,
        rho_bound=rho_bound,
    )
    stage_two_objective = np.zeros(2 * gate_count, dtype=np.float64)
    stage_two_objective[gate_count:] = 1.0
    stage_two = linprog(
        stage_two_objective,
        A_ub=stage_two_matrix,
        b_ub=stage_two_upper,
        bounds=gate_bounds + [(0.0, None)] * gate_count,
        method="highs",
        options=options,
    )
    if not stage_two.success or not np.isfinite(stage_two.fun):
        raise RuntimeError(f"stage-two projection failed: {stage_two.message}")

    gates = stage_two.x[:gate_count].reshape(raw.shape)
    displacement = np.abs(gates - raw)
    achieved_proxy = np.sum(mass * gates, axis=1)
    required_proxy = float(proxy_target_response) * np.sum(mass, axis=1)
    jumps = np.abs(np.diff(gates, axis=0))
    observed_jump = float(np.max(jumps)) if jumps.size else 0.0
    if jumps.size:
        jump_flat = int(np.argmax(jumps))
        jump_frame, jump_carrier = np.unravel_index(jump_flat, jumps.shape)
        jump_location = {
            "previous_frame_offset": int(jump_frame),
            "current_frame_offset": int(jump_frame + 1),
            "carrier_offset": int(jump_carrier),
        }
    else:
        jump_location = None
    violations = {
        "lower_bound": float(
            np.maximum(topology_floor[None, :] - gates, 0.0).max()
        ),
        "upper_bound": float(np.maximum(gates - maximum_gate, 0.0).max()),
        "proxy_target": float(
            np.maximum(required_proxy - achieved_proxy, 0.0).max()
        ),
        "gate_jump": max(observed_jump - float(maximum_gate_jump), 0.0),
        "rho": float(np.maximum(displacement - rho_bound, 0.0).max()),
    }
    maximum_violation = max(violations.values())
    if not np.isfinite(gates).all() or maximum_violation > residual_tolerance:
        raise RuntimeError(
            f"projection residual {maximum_violation} exceeds tolerance"
        )

    certificate = {
        "solver": "scipy.optimize.linprog:highs",
        "scipy_version": str(scipy.__version__),
        "highs_version": f"bundled-with-scipy-{scipy.__version__}",
        "stage_one_status": int(stage_one.status),
        "stage_one_message": str(stage_one.message),
        "stage_one_iterations": int(stage_one.nit),
        "stage_one_objective": float(stage_one.fun),
        "stage_two_status": int(stage_two.status),
        "stage_two_message": str(stage_two.message),
        "stage_two_iterations": int(stage_two.nit),
        "stage_two_objective": float(stage_two.fun),
        "rho_star": rho_star,
        "rho_bound": rho_bound,
        "maximum_displacement": float(displacement.max()),
        "mean_displacement": float(displacement.mean()),
        "total_displacement": float(displacement.sum()),
        "minimum_lower_bound_slack": float(
            np.min(gates - topology_floor[None, :])
        ),
        "minimum_upper_bound_slack": float(maximum_gate - gates.max()),
        "minimum_proxy_target_slack": float(
            np.min(achieved_proxy - required_proxy)
        ),
        "maximum_adjacent_gate_change": observed_jump,
        "maximum_jump_location": jump_location,
        "maximum_primal_violation": float(maximum_violation),
        "primal_violations": violations,
    }
    return {"gates": gates, "certificate": certificate}
