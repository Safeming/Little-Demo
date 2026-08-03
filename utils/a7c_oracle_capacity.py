from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


def _artifact_fingerprint(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        if key == "output_fingerprint":
            continue
        value = np.ascontiguousarray(np.asarray(arrays[key]))
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def assemble_teacher_gate_matrix(camera_index, gates_by_camera) -> np.ndarray:
    cameras = np.asarray(camera_index).reshape(-1)
    if cameras.size == 0:
        raise ValueError("camera_index cannot be empty")
    output = None
    for camera in np.unique(cameras):
        selected = np.flatnonzero(cameras == camera)
        if int(camera) not in gates_by_camera:
            raise ValueError(f"missing teacher gates for camera {int(camera)}")
        values = np.asarray(gates_by_camera[int(camera)], dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != selected.size:
            raise ValueError("camera teacher gates do not match source sample order")
        if output is None:
            output = np.empty((cameras.size, values.shape[1]), dtype=np.float32)
        if values.shape[1] != output.shape[1]:
            raise ValueError("teacher carrier count changed across cameras")
        output[selected] = values.astype(np.float32)
    return output


def save_teacher_artifact(
    path,
    *,
    gates,
    carrier_ids,
    camera_index,
    frame_index,
    minimum_gate: float,
    maximum_gate: float,
    source_fingerprints: dict[str, str],
) -> str:
    gate_values = np.asarray(gates, dtype=np.float32)
    carriers = np.asarray(carrier_ids, dtype=np.int64).reshape(-1)
    cameras = np.asarray(camera_index, dtype=np.int16).reshape(-1)
    frames = np.asarray(frame_index, dtype=np.int32).reshape(-1)
    if gate_values.shape != (cameras.size, carriers.size) or frames.shape != cameras.shape:
        raise ValueError("teacher arrays have inconsistent shapes")
    tolerance = 1.0e-6
    if np.any(~np.isfinite(gate_values)):
        raise ValueError("teacher gates must be finite")
    if np.min(gate_values) < minimum_gate - tolerance or np.max(gate_values) > maximum_gate + tolerance:
        raise ValueError("teacher gates violate frozen bounds")
    arrays = {
        "schema_version": np.array(1, dtype=np.int32),
        "gates": gate_values,
        "carrier_ids": carriers,
        "camera_index": cameras,
        "frame_index": frames,
        "minimum_gate": np.array(minimum_gate, dtype=np.float32),
        "maximum_gate": np.array(maximum_gate, dtype=np.float32),
        "paper_test_eligible": np.array(0, dtype=np.uint8),
    }
    for name, fingerprint in sorted(source_fingerprints.items()):
        arrays[f"source_{name}_fingerprint"] = np.array(str(fingerprint))
    fingerprint = _artifact_fingerprint(arrays)
    arrays["output_fingerprint"] = np.array(fingerprint)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)
    load_teacher_artifact(output)
    return fingerprint


def load_teacher_artifact(path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    expected = str(arrays.get("output_fingerprint", ""))
    if not expected or _artifact_fingerprint(arrays) != expected:
        raise ValueError("teacher artifact fingerprint mismatch")
    return arrays


def interpolation_basis(frame_count: int, knot_count: int) -> np.ndarray:
    frames = int(frame_count)
    knots = int(knot_count)
    if frames < 2 or knots < 2 or knots > frames:
        raise ValueError("knot_count must be in [2, frame_count]")
    positions = np.linspace(0.0, frames - 1.0, knots)
    basis = np.zeros((frames, knots), dtype=np.float64)
    for frame in range(frames):
        right = int(np.searchsorted(positions, frame, side="right"))
        if right == 0:
            basis[frame, 0] = 1.0
        elif right >= knots:
            basis[frame, -1] = 1.0
        else:
            left = right - 1
            span = positions[right] - positions[left]
            amount = (frame - positions[left]) / span
            basis[frame, left] = 1.0 - amount
            basis[frame, right] = amount
    return basis


def normalized_flicker(values) -> float:
    signal = np.asarray(values, dtype=np.float64).reshape(-1)
    if signal.size <= 1:
        return 0.0
    return float(np.mean(np.abs(np.diff(signal))) / max(abs(np.mean(signal)), 1.0e-12))


def _feature_matrix(values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, None]
    return np.concatenate(
        [signal[:, index, None] * basis for index in range(signal.shape[1])],
        axis=1,
    )


def solve_fractional_gate(
    *,
    variable_values,
    fixed_values,
    minimum_gate: float,
    maximum_gate: float,
    knot_count: int,
    target_variable=None,
    target_fixed=None,
    target_base=None,
    minimum_target_response: float = 0.0,
    temporal_block_count: int | None = None,
    minimum_block_gain: float = 0.0,
    block_guard_values=(),
    block_base_values=None,
) -> dict:
    variable = np.asarray(variable_values, dtype=np.float64)
    if variable.ndim == 1:
        variable = variable[:, None]
    fixed = np.asarray(fixed_values, dtype=np.float64).reshape(-1)
    if variable.ndim != 2 or fixed.shape != (variable.shape[0],):
        raise ValueError("variable and fixed values must share the frame dimension")
    frames, channels = variable.shape
    basis = interpolation_basis(frames, knot_count)
    matrix = _feature_matrix(variable, basis)
    coefficient_count = matrix.shape[1]
    difference_count = frames - 1
    if block_base_values is None:
        block_base = fixed + np.sum(variable, axis=1)
    else:
        block_base = np.asarray(block_base_values, dtype=np.float64).reshape(-1)
        if block_base.shape != fixed.shape:
            raise ValueError("block_base_values must match the frame dimension")
    signal_specs = [(variable, fixed, matrix, block_base)]
    for guard_variable, guard_fixed in block_guard_values:
        guard_variable = np.asarray(guard_variable, dtype=np.float64)
        if guard_variable.ndim == 1:
            guard_variable = guard_variable[:, None]
        guard_fixed = np.asarray(guard_fixed, dtype=np.float64).reshape(-1)
        if guard_variable.shape != variable.shape or guard_fixed.shape != fixed.shape:
            raise ValueError("block guard values must match variable_values")
        signal_specs.append(
            (
                guard_variable,
                guard_fixed,
                _feature_matrix(guard_variable, basis),
                guard_fixed + np.sum(guard_variable, axis=1),
            )
        )
    q_index = coefficient_count + difference_count * len(signal_specs)
    variable_count = q_index + 1

    block_indices = []
    if temporal_block_count is not None:
        block_indices = [
            indices
            for indices in np.array_split(np.arange(frames), int(temporal_block_count))
            if len(indices) > 1
        ]
    row_count = (
        2 * coefficient_count
        + 2 * difference_count * len(signal_specs)
        + len(block_indices) * len(signal_specs)
    )
    has_target = target_variable is not None
    if has_target:
        target_values = np.asarray(target_variable, dtype=np.float64)
        if target_values.ndim == 1:
            target_values = target_values[:, None]
        target_fixed_values = np.asarray(target_fixed, dtype=np.float64).reshape(-1)
        target_base_values = np.asarray(target_base, dtype=np.float64).reshape(-1)
        if target_values.shape != variable.shape:
            raise ValueError("target_variable must match variable_values")
        target_matrix = _feature_matrix(target_values, basis)
        row_count += frames
    constraints = lil_matrix((row_count, variable_count), dtype=np.float64)
    upper = np.zeros(row_count, dtype=np.float64)
    row = 0
    for index in range(coefficient_count):
        constraints[row, index] = 1.0
        constraints[row, q_index] = -float(maximum_gate)
        row += 1
        constraints[row, index] = -1.0
        constraints[row, q_index] = float(minimum_gate)
        row += 1
    for signal_index, (_, signal_fixed, signal_matrix, signal_base) in enumerate(signal_specs):
        difference_offset = coefficient_count + signal_index * difference_count
        for frame in range(1, frames):
            delta = signal_matrix[frame] - signal_matrix[frame - 1]
            fixed_delta = signal_fixed[frame] - signal_fixed[frame - 1]
            difference_index = difference_offset + frame - 1
            constraints[row, :coefficient_count] = delta
            constraints[row, difference_index] = -1.0
            constraints[row, q_index] = fixed_delta
            row += 1
            constraints[row, :coefficient_count] = -delta
            constraints[row, difference_index] = -1.0
            constraints[row, q_index] = -fixed_delta
            row += 1
        for indices in block_indices:
            base_flicker = normalized_flicker(signal_base[indices])
            flicker_limit = base_flicker * (1.0 - float(minimum_block_gain))
            mean_scale = flicker_limit * (len(indices) - 1) / len(indices)
            for frame in indices[1:]:
                constraints[row, difference_offset + int(frame) - 1] = 1.0
            constraints[row, :coefficient_count] = -mean_scale * np.sum(
                signal_matrix[indices], axis=0
            )
            constraints[row, q_index] = -mean_scale * float(
                np.sum(signal_fixed[indices])
            )
            row += 1
    if has_target:
        for frame in range(frames):
            constraints[row, :coefficient_count] = -target_matrix[frame]
            constraints[row, q_index] = (
                float(minimum_target_response) * target_base_values[frame]
                - target_fixed_values[frame]
            )
            row += 1
    equality = lil_matrix((1, variable_count), dtype=np.float64)
    equality[0, :coefficient_count] = np.sum(matrix, axis=0)
    equality[0, q_index] = float(np.sum(fixed))
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[coefficient_count:coefficient_count + difference_count] = (
        frames / max(frames - 1, 1)
    )
    result = linprog(
        objective,
        A_ub=constraints.tocsr(),
        b_ub=upper,
        A_eq=equality.tocsr(),
        b_eq=np.ones(1, dtype=np.float64),
        bounds=(0.0, None),
        method="highs",
        options={
            "primal_feasibility_tolerance": 1.0e-9,
            "dual_feasibility_tolerance": 1.0e-9,
            "ipm_optimality_tolerance": 1.0e-10,
        },
    )
    if not result.success or result.x[q_index] <= 0.0:
        raise RuntimeError(f"oracle linear program failed: {result.message}")
    q = float(result.x[q_index])
    knot_values = result.x[:coefficient_count].reshape(channels, knot_count) / q
    gates = basis @ knot_values.T
    candidate = fixed + np.sum(variable * gates, axis=1)
    base = fixed + np.sum(variable, axis=1)
    target_minimum = 1.0
    if has_target:
        target_candidate = target_fixed_values + np.sum(target_values * gates, axis=1)
        target_minimum = float(
            np.min(target_candidate / np.maximum(target_base_values, 1.0e-12))
        )
    return {
        "success": True,
        "gates": gates,
        "candidate": candidate,
        "base": base,
        "normalized_flicker_gain": 1.0
        - normalized_flicker(candidate) / max(normalized_flicker(base), 1.0e-12),
        "minimum_target_response": target_minimum,
        "maximum_adjacent_gate_change": float(np.max(np.abs(np.diff(gates, axis=0)))),
    }


def _evaluate_gate(target, outer, boundary, point_target, point_outer, point_boundary, solved):
    gates = solved["gates"]
    target_fixed = target - np.sum(point_target, axis=1)
    outer_fixed = outer - np.sum(point_outer, axis=1)
    boundary_fixed = boundary - np.sum(point_boundary, axis=1)
    candidate_target = target_fixed + np.sum(point_target * gates, axis=1)
    candidate_outer = outer_fixed + np.sum(point_outer * gates, axis=1)
    candidate_boundary = boundary_fixed + np.sum(point_boundary * gates, axis=1)
    return {
        "target": candidate_target,
        "outer": candidate_outer,
        "boundary": candidate_boundary,
        "outer_gain": 1.0 - normalized_flicker(candidate_outer) / max(normalized_flicker(outer), 1e-12),
        "boundary_gain": 1.0 - normalized_flicker(candidate_boundary) / max(normalized_flicker(boundary), 1e-12),
        "minimum_target_response": float(np.min(candidate_target / np.maximum(target, 1e-12))),
        "gates": gates,
    }


def _best(candidates):
    return max(candidates, key=lambda row: (min(row["outer_gain"], row["boundary_gain"]), row["outer_gain"] + row["boundary_gain"]))


def evaluate_camera_oracles(
    *, target, outer, boundary, point_target, point_outer, point_boundary,
    minimum_gate: float, minimum_target_response: float, knot_count: int,
    temporal_block_count: int | None = None,
) -> dict:
    target = np.asarray(target, dtype=np.float64)
    outer = np.asarray(outer, dtype=np.float64)
    boundary = np.asarray(boundary, dtype=np.float64)
    point_target = np.asarray(point_target, dtype=np.float64)
    point_outer = np.asarray(point_outer, dtype=np.float64)
    point_boundary = np.asarray(point_boundary, dtype=np.float64)
    ones = lambda x: np.asarray(x, dtype=np.float64)[:, None]

    global_candidates = []
    for signal, guard in ((outer, boundary), (boundary, outer)):
        solved = solve_fractional_gate(
            variable_values=ones(signal), fixed_values=np.zeros_like(signal),
            target_variable=ones(target), target_fixed=np.zeros_like(target),
            target_base=target, minimum_target_response=minimum_target_response,
            minimum_gate=minimum_gate, maximum_gate=1.0, knot_count=knot_count,
            temporal_block_count=temporal_block_count,
            block_guard_values=((ones(guard), np.zeros_like(guard)),),
        )
        global_candidates.append(_evaluate_gate(target, outer, boundary, ones(target), ones(outer), ones(boundary), solved))
    global_result = _best(global_candidates)

    point_candidates = []
    for signal_points, fixed_signal, guard_points, guard_fixed in (
        (
            point_outer,
            outer - point_outer.sum(axis=1),
            point_boundary,
            boundary - point_boundary.sum(axis=1),
        ),
        (
            point_boundary,
            boundary - point_boundary.sum(axis=1),
            point_outer,
            outer - point_outer.sum(axis=1),
        ),
    ):
        solved = solve_fractional_gate(
            variable_values=signal_points, fixed_values=fixed_signal,
            target_variable=point_target, target_fixed=target - point_target.sum(axis=1),
            target_base=target, minimum_target_response=minimum_target_response,
            minimum_gate=minimum_gate, maximum_gate=1.0, knot_count=knot_count,
            temporal_block_count=temporal_block_count,
            block_guard_values=((guard_points, guard_fixed),),
        )
        point_candidates.append(_evaluate_gate(target, outer, boundary, point_target, point_outer, point_boundary, solved))
    point_result = _best(point_candidates)

    nonboundary = np.maximum(outer - boundary, 0.0)
    ray_outer_values = np.stack([boundary, nonboundary], axis=1)
    ray_boundary_values = np.stack([boundary, np.zeros_like(boundary)], axis=1)
    ray_candidates = []
    for objective_values, objective_base, guard_values in (
        (ray_outer_values, outer, ray_boundary_values),
        (ray_boundary_values, boundary, ray_outer_values),
    ):
        solved = solve_fractional_gate(
            variable_values=objective_values,
            fixed_values=np.zeros_like(outer),
            block_base_values=objective_base,
            minimum_gate=minimum_gate,
            maximum_gate=1.0,
            knot_count=knot_count,
            temporal_block_count=temporal_block_count,
            block_guard_values=((guard_values, np.zeros_like(outer)),),
        )
        gates = solved["gates"]
        boundary_candidate = boundary * gates[:, 0]
        outer_candidate = boundary_candidate + nonboundary * gates[:, 1]
        ray_candidates.append(
            {
                "target": target.copy(),
                "outer": outer_candidate,
                "boundary": boundary_candidate,
                "outer_gain": 1.0
                - normalized_flicker(outer_candidate)
                / max(normalized_flicker(outer), 1e-12),
                "boundary_gain": 1.0
                - normalized_flicker(boundary_candidate)
                / max(normalized_flicker(boundary), 1e-12),
                "minimum_target_response": 1.0,
                "gates": gates,
            }
        )
    ray_result = _best(ray_candidates)
    return {"global": global_result, "point": point_result, "ray": ray_result}
