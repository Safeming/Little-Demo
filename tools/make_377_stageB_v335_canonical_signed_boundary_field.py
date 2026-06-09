#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def _safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def _safe_int(row: dict[str, str], key: str, default: int = -1) -> int:
    try:
        return int(float(row.get(key, default) or default))
    except Exception:
        return default


def _parse_ids(value: str) -> list[int]:
    out = []
    for token in str(value or "").replace(",", ";").split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out


def _parse_scores(value: str, count: int) -> list[float]:
    out = []
    for token in str(value or "").replace(",", ";").split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            out.append(1.0)
    if len(out) < count:
        out.extend([1.0] * (count - len(out)))
    return out[:count]


def _parse_allowed(value: str) -> set[int]:
    text = str(value or "").strip()
    if not text or text.lower() in {"all", "*", "none", "null"}:
        return set()
    out = set()
    for token in text.replace("[", "").replace("]", "").replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.add(int(float(token)))
        except ValueError:
            continue
    return out


def _load_point_stats(path: Path | None) -> dict[int, dict[str, float]]:
    if path is None or not path.exists():
        return {}
    stats: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            point_idx = _safe_int(row, "point_idx", -1)
            if point_idx < 0:
                continue
            stats[point_idx] = {
                "layer_id": float(_safe_int(row, "layer_id", -1)),
                "region_id": float(_safe_int(row, "region_id", -1)),
                "dominant_joint": float(_safe_int(row, "dominant_joint", -1)),
                "surface_distance": _safe_float(row, "surface_distance", 0.0),
                "thin_score": _safe_float(row, "thin_score", 0.0),
                "boundary_score": _safe_float(row, "boundary_score", 0.0),
                "visible_frame_hits": _safe_float(row, "visible_frame_hits", 0.0),
            }
    return stats


def _passes_filters(point_id: int, stats: dict[int, dict[str, float]], args: argparse.Namespace, prefix: str) -> bool:
    if not stats:
        return True
    item = stats.get(int(point_id))
    if item is None:
        return False
    min_boundary = float(getattr(args, f"{prefix}_min_boundary"))
    min_thin = float(getattr(args, f"{prefix}_min_thin"))
    min_visible = float(getattr(args, f"{prefix}_min_visible_hits"))
    surface_min = getattr(args, f"{prefix}_surface_min")
    surface_max = getattr(args, f"{prefix}_surface_max")
    if min_boundary >= 0.0 and item["boundary_score"] < min_boundary:
        return False
    if min_thin >= 0.0 and item["thin_score"] < min_thin:
        return False
    if min_visible >= 0.0 and item["visible_frame_hits"] < min_visible:
        return False
    if surface_min is not None and item["surface_distance"] < float(surface_min):
        return False
    if surface_max is not None and item["surface_distance"] > float(surface_max):
        return False
    for key, arg_name in (
        ("layer_id", f"{prefix}_allowed_layers"),
        ("region_id", f"{prefix}_allowed_regions"),
        ("dominant_joint", f"{prefix}_allowed_joints"),
    ):
        allowed = _parse_allowed(str(getattr(args, arg_name) or ""))
        if allowed and int(item.get(key, -1)) not in allowed:
            return False
    return True


def _row_weight(row: dict[str, str]) -> float:
    area = max(_safe_float(row, "area", 0.0), 1.0)
    near = max(_safe_float(row, "near_score_sum", 0.0), 1.0)
    return math.log1p(area) * math.log1p(near)


def _rank(scores: dict[int, float], max_count: int, min_score: float) -> list[dict[str, float]]:
    ranked = [
        {"point_idx": int(point_id), "score": float(score)}
        for point_id, score in scores.items()
        if float(score) >= float(min_score)
    ]
    ranked.sort(key=lambda item: (item["score"], item["point_idx"]), reverse=True)
    return ranked[: max(int(max_count), 0)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a canonical point-level signed boundary field from selected residual components.")
    parser.add_argument("--component-csv", required=True, type=Path)
    parser.add_argument("--point-csv", default=None, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--max-shrink", type=int, default=256)
    parser.add_argument("--max-grow", type=int, default=256)
    parser.add_argument("--top-points-per-component", type=int, default=8)
    parser.add_argument("--min-component-area", type=float, default=20.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--balance", choices=("none", "min", "grow", "shrink"), default="none")
    parser.add_argument("--shrink-min-boundary", type=float, default=-1.0)
    parser.add_argument("--shrink-min-thin", type=float, default=-1.0)
    parser.add_argument("--shrink-min-visible-hits", type=float, default=-1.0)
    parser.add_argument("--shrink-surface-min", type=float, default=None)
    parser.add_argument("--shrink-surface-max", type=float, default=None)
    parser.add_argument("--shrink-allowed-layers", default="")
    parser.add_argument("--shrink-allowed-regions", default="")
    parser.add_argument("--shrink-allowed-joints", default="")
    parser.add_argument("--grow-min-boundary", type=float, default=-1.0)
    parser.add_argument("--grow-min-thin", type=float, default=-1.0)
    parser.add_argument("--grow-min-visible-hits", type=float, default=-1.0)
    parser.add_argument("--grow-surface-min", type=float, default=None)
    parser.add_argument("--grow-surface-max", type=float, default=None)
    parser.add_argument("--grow-allowed-layers", default="")
    parser.add_argument("--grow-allowed-regions", default="")
    parser.add_argument("--grow-allowed-joints", default="")
    args = parser.parse_args()

    point_stats = _load_point_stats(args.point_csv)
    shrink_scores: dict[int, float] = defaultdict(float)
    grow_scores: dict[int, float] = defaultdict(float)
    direction_rows = defaultdict(int)

    with args.component_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            direction = str(row.get("direction", "")).strip().lower()
            if direction not in {"outer", "inner"}:
                continue
            if _safe_float(row, "area", 0.0) < float(args.min_component_area):
                continue
            ids = _parse_ids(row.get("top_point_ids", ""))
            if not ids:
                continue
            scores = _parse_scores(row.get("top_point_scores", ""), len(ids))
            weight = _row_weight(row)
            direction_rows[direction] += 1
            for point_id, local_score in zip(ids[: args.top_points_per_component], scores[: args.top_points_per_component]):
                if point_id < 0:
                    continue
                score = weight * max(float(local_score), 1.0e-4)
                if direction == "outer":
                    if _passes_filters(point_id, point_stats, args, "shrink"):
                        shrink_scores[int(point_id)] += score
                else:
                    if _passes_filters(point_id, point_stats, args, "grow"):
                        grow_scores[int(point_id)] += score

    max_shrink = int(args.max_shrink)
    max_grow = int(args.max_grow)
    if args.balance == "min":
        bound = min(max_shrink, max_grow)
        max_shrink = bound
        max_grow = bound
    elif args.balance == "grow":
        max_shrink = min(max_shrink, max_grow)
    elif args.balance == "shrink":
        max_grow = min(max_grow, max_shrink)

    shrink = _rank(shrink_scores, max_shrink, args.min_score)
    grow = _rank(grow_scores, max_grow, args.min_score)
    shrink_ids = {int(item["point_idx"]) for item in shrink}
    grow = [item for item in grow if int(item["point_idx"]) not in shrink_ids]

    payload = {
        "type": "canonical_signed_boundary_field",
        "source_component_csv": str(args.component_csv),
        "source_point_csv": str(args.point_csv or ""),
        "selection": {
            "max_shrink": int(max_shrink),
            "max_grow": int(max_grow),
            "top_points_per_component": int(args.top_points_per_component),
            "min_component_area": float(args.min_component_area),
            "min_score": float(args.min_score),
            "balance": str(args.balance),
            "direction_rows": {key: int(value) for key, value in direction_rows.items()},
        },
        "shrink_point_ids": [int(item["point_idx"]) for item in shrink],
        "grow_point_ids": [int(item["point_idx"]) for item in grow],
        "shrink_records": shrink,
        "grow_records": grow,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_json": str(args.out_json),
        "shrink_count": len(payload["shrink_point_ids"]),
        "grow_count": len(payload["grow_point_ids"]),
        "top_shrink": payload["shrink_point_ids"][:8],
        "top_grow": payload["grow_point_ids"][:8],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
