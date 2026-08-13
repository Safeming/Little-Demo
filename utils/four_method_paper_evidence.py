from __future__ import annotations

from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np


TARGET_RETENTION = 0.60
GG_377_RETENTION = 0.40
SUBJECTS = ("377", "386", "394")
METHODS = ("saga", "gaussian_grouping", "sggs", "a5")

_BASELINES = {
    "a5": "A5",
    "saga": "B4",
    "gaussian_grouping": "B4",
    "sggs": "B4",
}
_FIGURE_LABELS = {
    "a5": "Ours",
    "saga": "SAGA",
    "gaussian_grouping": "GG",
    "sggs": "SG-GS",
}


def build_temporal_windows(
    *,
    cameras: Sequence[int] = (21, 22, 23),
    anchors: Sequence[int] = (180, 420, 540),
    radius: int = 10,
) -> list[dict]:
    radius = int(radius)
    if radius < 0:
        raise ValueError("temporal window radius must be non-negative")
    windows: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for camera in cameras:
        camera = int(camera)
        for anchor in anchors:
            anchor = int(anchor)
            frames = list(range(anchor - radius, anchor + radius + 1))
            if frames[0] < 0:
                raise ValueError("temporal window contains a negative frame")
            keys = {(camera, frame) for frame in frames}
            if seen & keys:
                raise ValueError("temporal windows overlap")
            seen.update(keys)
            windows.append(
                {
                    "camera": camera,
                    "anchor": anchor,
                    "window": f"c{camera:02d}_a{anchor:06d}",
                    "frames": frames,
                }
            )
    if not windows:
        raise ValueError("no temporal windows requested")
    return windows


def resolve_frozen_operating_point(
    rows: Iterable[Mapping],
    *,
    method: str,
    subject: str,
) -> dict:
    method = str(method)
    subject = str(subject)
    if method not in _BASELINES:
        raise ValueError(f"unsupported method: {method}")
    is_gg_377 = method == "gaussian_grouping" and subject == "377"
    retention = GG_377_RETENTION if is_gg_377 else TARGET_RETENTION
    baseline = _BASELINES[method]
    matches = [
        dict(row)
        for row in rows
        if str(row.get("baseline", "")) == baseline
        and np.isclose(float(row.get("retention", "nan")), retention, rtol=0.0, atol=1e-12)
    ]
    if not matches:
        raise ValueError(
            f"missing frozen operating point for method={method} subject={subject} retention={retention}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"frozen operating point is not unique for method={method} subject={subject} retention={retention}"
        )
    result = matches[0]
    strength = float(result.get("edit_strength", "nan"))
    if not np.isfinite(strength) or not 0.0 < strength <= 1.0:
        raise ValueError(f"edit_strength must be finite and in (0, 1], got {strength}")
    result.update(
        {
            "method": method,
            "subject": subject,
            "baseline": baseline,
            "retention": float(retention),
            "edit_strength": strength,
            "target_retention": TARGET_RETENTION,
            "target_retention_feasible": not is_gg_377,
            "figure_label": "GG\N{DAGGER}" if is_gg_377 else _FIGURE_LABELS[method],
        }
    )
    return result


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def aggregate_frame(rows: Iterable[Mapping]) -> dict:
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError("cannot aggregate an empty frame")
    target_activation = float(sum(float(row["target_activation"]) for row in rows))
    outer_activation = float(sum(float(row["outer_activation"]) for row in rows))
    actionable = float(sum(float(row["actionable_outer_activation"]) for row in rows))
    reference_target = float(
        sum(float(row.get("reference_target_activation", row["target_activation"])) for row in rows)
    )
    valid = [row for row in rows if not _as_bool(row.get("target_empty", False))]
    if not valid:
        raise ValueError("frame contains no valid target parts")
    iou = np.asarray([float(row["iou"]) for row in valid], dtype=np.float64)
    boundary = np.asarray([float(row["boundary_f1"]) for row in valid], dtype=np.float64)
    if not np.isfinite(iou).all() or not np.isfinite(boundary).all():
        raise ValueError("frame quality metrics must be finite")
    return {
        "target_activation": target_activation,
        "outer_activation": outer_activation,
        "actionable_outer_activation": actionable,
        "reference_target_activation": reference_target,
        "view_retention": _safe_ratio(target_activation, reference_target),
        "raw_leakage": _safe_ratio(outer_activation, reference_target),
        "actionable_leakage": _safe_ratio(actionable, reference_target),
        "raw_leakage_ratio": _safe_ratio(outer_activation, target_activation),
        "actionable_leakage_ratio": _safe_ratio(actionable, target_activation),
        "macro_miou": float(iou.mean()),
        "mean_boundary_f1": float(boundary.mean()),
        "valid_part_count": int(len(valid)),
    }


def _group_values(rows: Iterable[Mapping], keys: Sequence[str], value_key: str) -> dict[tuple, list[float]]:
    grouped: dict[tuple, list[float]] = {}
    for row in rows:
        key = tuple(str(row[name]) for name in keys)
        value = float(row[value_key])
        if not np.isfinite(value):
            raise ValueError(f"{value_key} must contain finite values")
        grouped.setdefault(key, []).append(value)
    if not grouped:
        raise ValueError("statistical input is empty")
    return grouped


def exact_block_sign_flip(
    rows: Iterable[Mapping],
    *,
    value_key: str,
    block_keys: Sequence[str] = ("subject", "camera"),
) -> dict:
    grouped = _group_values(rows, block_keys, value_key)
    block_values = np.asarray(
        [np.mean(grouped[key], dtype=np.float64) for key in sorted(grouped)], dtype=np.float64
    )
    observed = float(block_values.mean())
    extreme = 0
    total = 0
    tolerance = 1e-15
    for signs in product((-1.0, 1.0), repeat=block_values.size):
        permuted = float(np.mean(block_values * np.asarray(signs, dtype=np.float64)))
        total += 1
        if abs(permuted) + tolerance >= abs(observed):
            extreme += 1
    return {
        "observed": observed,
        "block_count": int(block_values.size),
        "permutation_count": int(total),
        "extreme_count": int(extreme),
        "p_value": float(extreme / total),
        "block_keys": list(block_keys),
    }


def hierarchical_bootstrap_paired(
    rows: Iterable[Mapping],
    *,
    value_key: str,
    iterations: int = 20_000,
    seed: int = 20260813,
) -> dict:
    rows = [dict(row) for row in rows]
    if int(iterations) <= 0:
        raise ValueError("bootstrap iterations must be positive")
    subjects = sorted({str(row["subject"]) for row in rows})
    if not subjects:
        raise ValueError("bootstrap input is empty")
    hierarchy: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        subject = str(row["subject"])
        camera = str(row["camera"])
        value = float(row[value_key])
        if not np.isfinite(value):
            raise ValueError(f"{value_key} must contain finite values")
        hierarchy.setdefault(subject, {}).setdefault(camera, []).append(value)
    estimate = float(np.mean([float(row[value_key]) for row in rows], dtype=np.float64))
    rng = np.random.default_rng(int(seed))
    samples = np.empty((int(iterations),), dtype=np.float64)
    for iteration in range(int(iterations)):
        subject_means = []
        for sampled_subject in rng.choice(subjects, size=len(subjects), replace=True):
            cameras = sorted(hierarchy[str(sampled_subject)])
            camera_means = []
            for sampled_camera in rng.choice(cameras, size=len(cameras), replace=True):
                values = np.asarray(
                    hierarchy[str(sampled_subject)][str(sampled_camera)], dtype=np.float64
                )
                sampled_values = rng.choice(values, size=values.size, replace=True)
                camera_means.append(float(sampled_values.mean()))
            subject_means.append(float(np.mean(camera_means, dtype=np.float64)))
        samples[iteration] = float(np.mean(subject_means, dtype=np.float64))
    ci_low, ci_high = np.quantile(samples, (0.025, 0.975))
    return {
        "estimate": estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "iterations": int(iterations),
        "seed": int(seed),
        "subject_count": int(len(subjects)),
    }


def holm_adjust(p_values_by_method: Mapping[str, float]) -> dict[str, float]:
    if not p_values_by_method:
        return {}
    ordered = sorted(
        ((str(method), float(value)) for method, value in p_values_by_method.items()),
        key=lambda item: (item[1], item[0]),
    )
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for _, value in ordered):
        raise ValueError("p-values must be finite and within [0, 1]")
    count = len(ordered)
    running = 0.0
    adjusted: dict[str, float] = {}
    for index, (method, value) in enumerate(ordered):
        running = max(running, min(1.0, float(count - index) * value))
        adjusted[method] = float(running)
    return adjusted


def summarize_temporal_window(
    frame_rows: Iterable[Mapping],
    *,
    metric_names: Sequence[str],
) -> dict:
    rows = sorted((dict(row) for row in frame_rows), key=lambda row: int(row["frame"]))
    if not rows:
        raise ValueError("temporal window is empty")
    frames = [int(row["frame"]) for row in rows]
    if len(set(frames)) != len(frames) or any(right - left != 1 for left, right in zip(frames, frames[1:])):
        raise ValueError("temporal window frames must be unique and consecutive")
    result = {
        "frame_count": int(len(rows)),
        "first_frame": int(frames[0]),
        "last_frame": int(frames[-1]),
    }
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"temporal metric {metric} must be finite")
        deltas = np.abs(np.diff(values))
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = float(values.std(ddof=0))
        result[f"{metric}_mean_abs_delta"] = float(deltas.mean()) if deltas.size else 0.0
        result[f"{metric}_p95_abs_delta"] = (
            float(np.quantile(deltas, 0.95)) if deltas.size else 0.0
        )
    return result
