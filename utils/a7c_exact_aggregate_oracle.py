from __future__ import annotations

import numpy as np


def extract_replay_requests(
    records,
    *,
    replay_margin: float,
    maximum_interval_width: float,
) -> list[dict]:
    rows = list(records)
    if len(rows) != 24:
        raise ValueError("replay source requires exactly 24 records")
    margin = float(replay_margin)
    width_limit = float(maximum_interval_width)
    if not np.isfinite([margin, width_limit]).all() or margin <= 0.0:
        raise ValueError("replay margin and interval width must be finite")
    if width_limit <= 0.0:
        raise ValueError("maximum interval width must be positive")

    expected = {(fold, camera) for fold in range(6) for camera in range(4)}
    observed = [
        (int(row["fold"]), int(row["camera_index"])) for row in rows
    ]
    if len(set(observed)) != len(observed):
        raise ValueError("duplicate fold-camera replay record")
    if set(observed) != expected:
        raise ValueError("replay source fold-camera grid differs")

    output = []
    for row in sorted(
        rows,
        key=lambda value: (value["fold"], value["camera_index"]),
    ):
        endpoint = row["boundary_conditioned"]
        if endpoint.get("status") != "bracketed":
            raise ValueError("every replay endpoint must be bracketed")
        lower = float(endpoint["feasible_lower"])
        upper = float(endpoint["infeasible_upper"])
        width = float(endpoint["interval_width"])
        if not np.isfinite([lower, upper, width]).all() or width < 0.0:
            raise ValueError("replay endpoint values must be finite")
        if (
            width > width_limit
            or upper <= lower
            or abs((upper - lower) - width) > 1.0e-12
        ):
            raise ValueError("replay endpoint interval width differs")
        output.append(
            {
                "fold": int(row["fold"]),
                "camera_index": int(row["camera_index"]),
                "source_feasible_lower": lower,
                "source_infeasible_upper": upper,
                "source_interval_width": width,
                "minimum_outer_gain": 0.005,
                "minimum_boundary_gain": lower - margin,
            }
        )
    return output


def insert_replay_segment(
    *,
    replay_gates,
    replay_mask,
    selected,
    solved,
    request,
    frame_index,
    block_ids,
    carrier_ids,
    residual_tolerance,
) -> dict:
    output_gates = np.asarray(replay_gates)
    output_mask = np.asarray(replay_mask)
    mask = np.asarray(selected, dtype=bool).reshape(-1)
    frames = np.asarray(frame_index).reshape(-1)
    blocks = np.asarray(block_ids).reshape(-1)
    carriers = np.asarray(carrier_ids).reshape(-1)
    if output_gates.ndim != 2:
        raise ValueError("replay gates need shape [samples, carriers]")
    if not (
        output_mask.shape
        == mask.shape
        == frames.shape
        == blocks.shape
        == (output_gates.shape[0],)
    ):
        raise ValueError("replay manifest arrays differ")
    if carriers.shape != (output_gates.shape[1],):
        raise ValueError("replay carrier manifest differs")
    if not np.any(mask):
        raise ValueError("replay segment is empty")
    if np.any(output_mask[mask]):
        raise ValueError("replay segments overlap")

    values = np.asarray(solved["gates"], dtype=np.float64)
    if values.shape != (int(mask.sum()), output_gates.shape[1]):
        raise ValueError("replay gate shape differs from selected segment")
    certificate = dict(solved["certificate"])
    if int(certificate["status"]) != 0:
        raise RuntimeError("replay solver is not optimal")
    violation = float(certificate["maximum_primal_violation"])
    if not np.isfinite(values).all() or not np.isfinite(violation):
        raise RuntimeError("replay witness certificate is non-finite")
    if violation > float(residual_tolerance):
        raise RuntimeError("replay witness certificate failed")
    selected_blocks = np.unique(blocks[mask])
    if selected_blocks.size != 1:
        raise ValueError("replay segment crosses temporal blocks")

    metrics = {key: float(value) for key, value in solved["metrics"].items()}
    slacks = {
        key: float(value) for key, value in solved.get("slack", {}).items()
    }
    if any(not np.isfinite(value) for value in metrics.values()):
        raise RuntimeError("replay metrics are non-finite")
    if any(not np.isfinite(value) for value in slacks.values()):
        raise RuntimeError("replay certificate slack is non-finite")
    indices = np.flatnonzero(mask)
    record = {
        **request,
        **metrics,
        **certificate,
        **dict(solved.get("locations", {})),
        **slacks,
        "block_index": int(selected_blocks[0]),
        "first_frame": int(frames[indices[0]]),
        "last_frame": int(frames[indices[-1]]),
        "sample_count": int(mask.sum()),
        "carrier_count": int(carriers.size),
        "source_fingerprints": dict(solved.get("source_fingerprints", {})),
        "sample_order_fingerprint": str(solved["sample_order_fingerprint"]),
        "carrier_order_fingerprint": str(solved["carrier_order_fingerprint"]),
    }

    output_gates[mask] = values
    output_mask[mask] = True
    return record
