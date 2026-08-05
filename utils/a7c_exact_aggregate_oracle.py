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
