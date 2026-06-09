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
                "over_score_sum": _safe_float(row, "over_score_sum", 0.0),
                "under_score_sum": _safe_float(row, "under_score_sum", 0.0),
                "over_frame_hits": _safe_float(row, "over_frame_hits", 0.0),
                "under_frame_hits": _safe_float(row, "under_frame_hits", 0.0),
                "visible_frame_hits": _safe_float(row, "visible_frame_hits", 0.0),
                "layer_id": float(_safe_int(row, "layer_id", -1)),
                "region_id": float(_safe_int(row, "region_id", -1)),
                "dominant_joint": float(_safe_int(row, "dominant_joint", -1)),
                "surface_distance": _safe_float(row, "surface_distance", 0.0),
                "thin_score": _safe_float(row, "thin_score", 0.0),
                "boundary_score": _safe_float(row, "boundary_score", 0.0),
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


def _point_direction_ok(point_id: int, stats: dict[int, dict[str, float]], direction: str, margin: float) -> bool:
    if float(margin) < 0.0:
        return True
    if not stats or point_id not in stats:
        return True
    item = stats[point_id]
    if direction == "outer":
        return item["over_score_sum"] + float(margin) >= item["under_score_sum"]
    return item["under_score_sum"] + float(margin) >= item["over_score_sum"]


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


def _image_filter(image_name: str, include_images: set[str]) -> bool:
    return not include_images or image_name in include_images


def main() -> int:
    parser = argparse.ArgumentParser(description="Build per-image group-paired signed boundary fields for v336.")
    parser.add_argument("--component-csv", required=True, type=Path)
    parser.add_argument("--point-csv", default=None, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--max-shrink-per-image", type=int, default=96)
    parser.add_argument("--max-grow-per-image", type=int, default=96)
    parser.add_argument("--max-components-per-direction", type=int, default=12)
    parser.add_argument("--top-points-per-component", type=int, default=6)
    parser.add_argument("--min-component-area", type=float, default=30.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--require-paired-image", action="store_true")
    parser.add_argument("--protect-grow-from-shrink", action="store_true")
    parser.add_argument("--protect-inner-points", type=int, default=0)
    parser.add_argument("--direction-score-margin", type=float, default=0.0)
    parser.add_argument("--include-images", default="")
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

    include_images = {token.strip() for token in str(args.include_images or "").replace(";", ",").split(",") if token.strip()}
    point_stats = _load_point_stats(args.point_csv)
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: {"outer": [], "inner": []})
    with args.component_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_name = str(row.get("image_name", "")).strip()
            direction = str(row.get("direction", "")).strip().lower()
            if not image_name or direction not in {"outer", "inner"}:
                continue
            if not _image_filter(image_name, include_images):
                continue
            if _safe_float(row, "area", 0.0) < float(args.min_component_area):
                continue
            grouped[image_name][direction].append(row)

    by_image = {}
    totals = {"images": 0, "paired_images": 0, "shrink": 0, "grow": 0}
    for image_name, buckets in sorted(grouped.items()):
        for direction in ("outer", "inner"):
            buckets[direction].sort(key=lambda row: (_safe_float(row, "area", 0.0), _safe_float(row, "near_score_sum", 0.0)), reverse=True)
            buckets[direction] = buckets[direction][: max(int(args.max_components_per_direction), 0)]
        if args.require_paired_image and (not buckets["outer"] or not buckets["inner"]):
            continue

        shrink_scores: dict[int, float] = defaultdict(float)
        grow_scores: dict[int, float] = defaultdict(float)
        protect_scores: dict[int, float] = defaultdict(float)
        for direction, target in (("outer", shrink_scores), ("inner", grow_scores)):
            for row in buckets[direction]:
                ids = _parse_ids(row.get("top_point_ids", ""))
                scores = _parse_scores(row.get("top_point_scores", ""), len(ids))
                weight = _row_weight(row)
                for point_id, local_score in zip(ids[: args.top_points_per_component], scores[: args.top_points_per_component]):
                    if point_id < 0:
                        continue
                    prefix = "shrink" if direction == "outer" else "grow"
                    if not _passes_filters(point_id, point_stats, args, prefix):
                        continue
                    if not _point_direction_ok(point_id, point_stats, direction, args.direction_score_margin):
                        continue
                    target[int(point_id)] += weight * max(float(local_score), 1.0e-4)
                    if direction == "inner":
                        protect_scores[int(point_id)] += weight * max(float(local_score), 1.0e-4)

        protected = {item["point_idx"] for item in _rank(protect_scores, args.protect_inner_points, args.min_score)}
        grow = _rank(grow_scores, args.max_grow_per_image, args.min_score)
        if args.protect_grow_from_shrink:
            protected |= {int(item["point_idx"]) for item in grow}
        shrink = _rank(shrink_scores, args.max_shrink_per_image, args.min_score)
        shrink = [item for item in shrink if int(item["point_idx"]) not in protected]
        if not shrink and not grow:
            continue
        by_image[image_name] = {
            "shrink_point_ids": [int(item["point_idx"]) for item in shrink],
            "grow_point_ids": [int(item["point_idx"]) for item in grow],
            "shrink_records": shrink,
            "grow_records": grow,
            "protected_point_ids": sorted(int(point_id) for point_id in protected),
            "component_counts": {"outer": len(buckets["outer"]), "inner": len(buckets["inner"])},
        }
        totals["images"] += 1
        if buckets["outer"] and buckets["inner"]:
            totals["paired_images"] += 1
        totals["shrink"] += len(shrink)
        totals["grow"] += len(grow)

    payload = {
        "type": "group_paired_signed_boundary_field",
        "source_component_csv": str(args.component_csv),
        "source_point_csv": str(args.point_csv or ""),
        "selection": {
            "max_shrink_per_image": int(args.max_shrink_per_image),
            "max_grow_per_image": int(args.max_grow_per_image),
            "max_components_per_direction": int(args.max_components_per_direction),
            "top_points_per_component": int(args.top_points_per_component),
            "min_component_area": float(args.min_component_area),
            "min_score": float(args.min_score),
            "require_paired_image": bool(args.require_paired_image),
            "protect_grow_from_shrink": bool(args.protect_grow_from_shrink),
            "protect_inner_points": int(args.protect_inner_points),
            "direction_score_margin": float(args.direction_score_margin),
            "include_images": sorted(include_images),
            "totals": totals,
        },
        "by_image": by_image,
        "shrink_point_ids": [],
        "grow_point_ids": [],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_json": str(args.out_json),
        "image_count": totals["images"],
        "paired_image_count": totals["paired_images"],
        "total_shrink_assignments": totals["shrink"],
        "total_grow_assignments": totals["grow"],
        "sample_images": list(by_image.keys())[:5],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
