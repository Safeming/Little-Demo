from __future__ import annotations

from collections import defaultdict
from itertools import combinations


def real_edit_record_key(row: dict) -> tuple[str, str, str, str]:
    return tuple(str(row[key]) for key in ("subject", "view", "part", "task"))


def collapse_curve_points(points: list[dict], *, tolerance_digits: int = 12) -> list[dict]:
    collapsed: dict[float, dict] = {}
    for point in points:
        retention = float(point["retention"])
        token = round(retention, int(tolerance_digits))
        if token not in collapsed:
            collapsed[token] = dict(point)
            continue
        current = collapsed[token]
        current["outer_burden"] = min(float(current["outer_burden"]), float(point["outer_burden"]))
        current["boundary_burden"] = min(
            float(current["boundary_burden"]),
            float(point["boundary_burden"]),
        )
        current["retention"] = min(float(current["retention"]), retention)
    return sorted(collapsed.values(), key=lambda item: float(item["retention"]))


def build_matched_strength_curves(
    rows: list[dict],
    *,
    reference_method: str = "voting",
    reference_strength: float = 1.0,
    epsilon: float = 1.0e-8,
) -> dict:
    references = {}
    all_keys = {real_edit_record_key(row) for row in rows}
    for row in rows:
        if str(row["method"]) != str(reference_method):
            continue
        if abs(float(row["edit_strength"]) - float(reference_strength)) > 1.0e-8:
            continue
        references[real_edit_record_key(row)] = row

    supported = {
        key
        for key, row in references.items()
        if int(float(row["target_pixel_count"])) > 0 and float(row["target_delta_sum"]) > float(epsilon)
    }
    points = defaultdict(list)
    for row in rows:
        key = real_edit_record_key(row)
        if key not in supported:
            continue
        reference_target = float(references[key]["target_delta_sum"])
        method = str(row["method"])
        points[(key, method)].append(
            {
                "retention": float(row["target_delta_sum"]) / reference_target,
                "outer_burden": float(row["outer_delta_sum"]) / reference_target,
                "boundary_burden": float(row["boundary_outer_delta_sum"]) / reference_target,
                "edit_strength": float(row["edit_strength"]),
            }
        )

    curves = {}
    for owner, method_points in points.items():
        origin = {
            "retention": 0.0,
            "outer_burden": 0.0,
            "boundary_burden": 0.0,
            "edit_strength": 0.0,
        }
        curves[owner] = collapse_curve_points([origin, *method_points])
    return {
        "references": references,
        "reference_supported": supported,
        "unsupported_reference_count": len(all_keys - supported),
        "curves": curves,
    }


def interpolate_curve(curve: list[dict], retention: float, *, epsilon: float = 1.0e-8) -> dict | None:
    target = float(retention)
    if not curve or target < -float(epsilon):
        return None
    ordered = sorted(curve, key=lambda item: float(item["retention"]))
    if target > float(ordered[-1]["retention"]) + float(epsilon):
        return None
    for point in ordered:
        if abs(float(point["retention"]) - target) <= float(epsilon):
            return {
                "retention": target,
                "outer_burden": float(point["outer_burden"]),
                "boundary_burden": float(point["boundary_burden"]),
            }
    for left, right in zip(ordered, ordered[1:]):
        left_r = float(left["retention"])
        right_r = float(right["retention"])
        if left_r <= target <= right_r and right_r - left_r > float(epsilon):
            alpha = (target - left_r) / (right_r - left_r)
            return {
                "retention": target,
                "outer_burden": float(left["outer_burden"]) + alpha * (
                    float(right["outer_burden"]) - float(left["outer_burden"])
                ),
                "boundary_burden": float(left["boundary_burden"]) + alpha * (
                    float(right["boundary_burden"]) - float(left["boundary_burden"])
                ),
            }
    return None


def match_curves_at_retention(curves: dict, *, reference_keys: set, retention: float) -> dict:
    methods = sorted({method for key, method in curves if key in reference_keys})
    matched = {}
    coverage = {method: 0 for method in methods}
    covered_keys = {method: set() for method in methods}
    for key in reference_keys:
        for method in methods:
            point = interpolate_curve(curves.get((key, method), []), float(retention))
            if point is None:
                continue
            matched[(key, method)] = point
            coverage[method] += 1
            covered_keys[method].add(key)
    common = {
        pair: covered_keys[pair[0]] & covered_keys[pair[1]]
        for pair in combinations(methods, 2)
    }
    return {
        "matched": matched,
        "coverage": coverage,
        "covered_keys": covered_keys,
        "common_coverage": common,
        "reference_count": len(reference_keys),
        "retention": float(retention),
    }
