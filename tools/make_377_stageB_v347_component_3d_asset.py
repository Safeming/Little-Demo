#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


METRICS = ("fg", "boundary", "edge", "inner", "outer", "hard", "opacity_inner", "opacity_outer")
SAMPLE_FILES = (
    ("contours/contour_samples.csv", {
        "fg": "fg_l1",
        "boundary": "boundary_l1",
        "edge": "edge_symmetric_dist_px",
    }),
    ("boundary_residuals/boundary_residual_samples.csv", {
        "inner": "inner_missing_pixels",
        "outer": "outer_leak_pixels",
        "hard": "hard_residual_score",
    }),
    ("opacity_footprint/opacity_footprint_samples.csv", {
        "opacity_inner": "primary_opacity_inner_missing_pixels",
        "opacity_outer": "primary_opacity_outer_leak_pixels",
    }),
)
REGION_NAMES = {0: "body", 1: "soft", 2: "cloth"}
LAYER_NAMES = {0: "rigid", 1: "soft", 2: "free"}


def _image_name(row: dict[str, str]) -> str:
    cam = int(float(row.get("cam", 0) or 0))
    frame = int(float(row.get("frame", 0) or 0))
    return f"c{cam:02d}_f{frame:06d}"


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return float(default)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default) or default))
    except Exception:
        return int(default)


def _parse_ids(text: str) -> list[int]:
    ids: list[int] = []
    for token in str(text or "").replace(",", ";").split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            ids.append(int(token))
        except Exception:
            continue
    return ids


def _load_samples(render_exp: Path) -> dict[str, dict[str, float]]:
    records: dict[str, dict[str, float]] = {}
    diag = render_exp / "diagnostics"
    for rel_path, mapping in SAMPLE_FILES:
        path = diag / rel_path
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                item = records.setdefault(_image_name(row), {})
                for metric, column in mapping.items():
                    item[metric] = _float(row, column)
    return records


def _load_component_rows(path: Path) -> dict[str, dict[str, list[dict[str, object]]]]:
    by_image: dict[str, dict[str, list[dict[str, object]]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            image_name = str(row.get("image_name", "") or "").strip()
            direction = str(row.get("direction", "") or "").strip().lower()
            if not image_name or direction not in ("inner", "outer"):
                continue
            record = {
                "row_index": row_index,
                "image_name": image_name,
                "direction": direction,
                "component_id": _int(row, "component_id", -1),
                "area": _float(row, "area"),
                "bbox_x": _float(row, "bbox_x"),
                "bbox_y": _float(row, "bbox_y"),
                "bbox_w": _float(row, "bbox_w"),
                "bbox_h": _float(row, "bbox_h"),
                "centroid_x": _float(row, "centroid_x"),
                "centroid_y": _float(row, "centroid_y"),
                "near_score_sum": _float(row, "near_score_sum"),
                "top_point_ids": _parse_ids(str(row.get("top_point_ids", "") or "")),
            }
            by_image.setdefault(image_name, {"inner": [], "outer": []})[direction].append(record)
    for item in by_image.values():
        for direction in ("inner", "outer"):
            item[direction].sort(
                key=lambda rec: (float(rec.get("area", 0.0)), float(rec.get("near_score_sum", 0.0))),
                reverse=True,
            )
    return by_image


def _load_point_stats(path: Path) -> dict[int, dict[str, object]]:
    stats: dict[int, dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            point_idx = _int(row, "point_idx", -1)
            if point_idx < 0:
                continue
            layer_id = _int(row, "layer_id", -1)
            region_id = _int(row, "region_id", -1)
            stats[point_idx] = {
                "point_idx": point_idx,
                "layer_id": layer_id,
                "layer_name": str(row.get("layer_name", "") or LAYER_NAMES.get(layer_id, "")),
                "region_id": region_id,
                "region_name": str(row.get("region_name", "") or REGION_NAMES.get(region_id, "")),
                "dominant_joint": _int(row, "dominant_joint", -1),
                "boundary_score": _float(row, "boundary_score"),
                "surface_distance": _float(row, "surface_distance"),
                "thin_score": _float(row, "thin_score"),
                "scale_mean": _float(row, "scale_mean"),
                "radius_px_mean": _float(row, "radius_px_mean"),
                "canonical": (
                    _float(row, "canonical_x"),
                    _float(row, "canonical_y"),
                    _float(row, "canonical_z"),
                ),
            }
    return stats


def _load_drop_images(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(item).strip() for item in data.get("drop_images", []) if str(item).strip()}
    text = path.read_text(encoding="utf-8")
    return {token.strip() for token in text.replace(";", ",").split(",") if token.strip()}


def _load_signed_temporal_sources(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    by_image = data.get("by_image", {}) if isinstance(data, dict) else {}
    sources = {}
    for image_name, item in by_image.items():
        if isinstance(item, dict):
            sources[str(image_name)] = str(item.get("source_image_name", image_name) or image_name)
    return sources


def _owner(top_ids: list[int], point_stats: dict[int, dict[str, object]]) -> dict[str, object]:
    stats = [point_stats[idx] for idx in top_ids if idx in point_stats]
    if not stats:
        return {
            "owner_layer": "",
            "owner_layer_id": "",
            "owner_region": "",
            "owner_region_id": "",
            "owner_joint": "",
            "owner_consistency": 0.0,
        }
    layers = Counter((int(item.get("layer_id", -1)), str(item.get("layer_name", ""))) for item in stats)
    regions = Counter((int(item.get("region_id", -1)), str(item.get("region_name", ""))) for item in stats)
    joints = Counter(int(item.get("dominant_joint", -1)) for item in stats)
    (layer_id, layer_name), layer_count = layers.most_common(1)[0]
    (region_id, region_name), region_count = regions.most_common(1)[0]
    joint, joint_count = joints.most_common(1)[0]
    consistency = min(layer_count, region_count, joint_count) / max(len(stats), 1)
    return {
        "owner_layer": layer_name,
        "owner_layer_id": int(layer_id),
        "owner_region": region_name,
        "owner_region_id": int(region_id),
        "owner_joint": int(joint),
        "owner_consistency": float(consistency),
    }


def _canonical_support(
    top_ids: list[int],
    point_stats: dict[int, dict[str, object]],
    *,
    radius_floor: float,
    radius_pad: float,
    radius_scale: float,
    max_top_ids: int,
    cluster_enable: bool = False,
    cluster_min_points: int = 4,
    cluster_owner_gate: bool = True,
    cluster_radius_max: float = 0.18,
) -> dict[str, object] | None:
    ids = [idx for idx in top_ids[: max(max_top_ids, 0)] if idx in point_stats]
    if not ids:
        return None
    source_ids = list(ids)
    if cluster_enable and len(ids) > max(int(cluster_min_points), 1):
        grouped_ids = ids
        if cluster_owner_gate:
            grouped = Counter(
                (
                    int(point_stats[idx].get("region_id", -1)),
                    int(point_stats[idx].get("dominant_joint", -1)),
                )
                for idx in ids
            )
            if grouped:
                owner_key, _ = grouped.most_common(1)[0]
                owner_ids = [
                    idx for idx in ids
                    if (
                        int(point_stats[idx].get("region_id", -1)),
                        int(point_stats[idx].get("dominant_joint", -1)),
                    ) == owner_key
                ]
                if len(owner_ids) >= max(int(cluster_min_points), 1):
                    grouped_ids = owner_ids
        best_ids = grouped_ids
        best_radius = None
        min_points = min(max(int(cluster_min_points), 1), len(grouped_ids))
        for seed in grouped_ids:
            seed_point = point_stats[seed]["canonical"]
            ordered = sorted(
                grouped_ids,
                key=lambda idx: sum((point_stats[idx]["canonical"][k] - seed_point[k]) ** 2 for k in range(3)),
            )
            for count in range(min_points, len(ordered) + 1):
                candidate = ordered[:count]
                candidate_points = [point_stats[idx]["canonical"] for idx in candidate]
                center = tuple(sum(point[k] for point in candidate_points) / len(candidate_points) for k in range(3))
                radius = max(
                    math.sqrt(sum((point[k] - center[k]) ** 2 for k in range(3)))
                    for point in candidate_points
                )
                if best_radius is None or radius < best_radius or (
                    math.isclose(radius, best_radius) and len(candidate) > len(best_ids)
                ):
                    best_radius = radius
                    best_ids = candidate
                if radius <= float(cluster_radius_max):
                    break
        if best_radius is not None:
            ids = best_ids
    points = [point_stats[idx]["canonical"] for idx in ids]
    center = tuple(sum(point[k] for point in points) / len(points) for k in range(3))
    dists = [
        math.sqrt(sum((point[k] - center[k]) ** 2 for k in range(3)))
        for point in points
    ]
    scale_values = [float(point_stats[idx].get("scale_mean", 0.0) or 0.0) for idx in ids]
    scale_hint = sorted(scale_values)[len(scale_values) // 2] if scale_values else 0.0
    radius = max(max(dists or [0.0]) * radius_scale + radius_pad, scale_hint * 2.5, radius_floor)
    return {
        "canonical_center": [float(center[0]), float(center[1]), float(center[2])],
        "canonical_radius": float(radius),
        "canonical_radius_source": {
            "max_top_id_dist": float(max(dists or [0.0])),
            "median_scale_mean": float(scale_hint),
            "radius_floor": float(radius_floor),
            "radius_pad": float(radius_pad),
            "radius_scale": float(radius_scale),
            "cluster_enable": bool(cluster_enable),
            "cluster_min_points": int(cluster_min_points),
            "cluster_radius_max": float(cluster_radius_max),
            "source_top_point_count": int(len(source_ids)),
            "cluster_top_point_count": int(len(ids)),
        },
        "top_point_ids": ids,
        "source_top_point_ids": source_ids,
    }


def _metric_delta(current: dict[str, float], base: dict[str, float], metric: str) -> float:
    return float(current.get(metric, 0.0)) - float(base.get(metric, 0.0))


def _candidate_directions(delta: dict[str, float], rows: dict[str, list[dict[str, object]]]) -> list[str]:
    directions = []
    if max(delta.get("outer", 0.0), delta.get("opacity_outer", 0.0), delta.get("hard", 0.0) * 1000.0) > 0.0:
        if rows.get("inner"):
            directions.append("inner")
        if rows.get("outer"):
            directions.append("outer")
    if max(delta.get("inner", 0.0), delta.get("opacity_inner", 0.0)) > 0.0:
        if rows.get("outer"):
            directions.append("outer")
        if rows.get("inner"):
            directions.append("inner")
    if not directions:
        if rows.get("inner"):
            directions.append("inner")
        elif rows.get("outer"):
            directions.append("outer")
    deduped = []
    for direction in directions:
        if direction not in deduped:
            deduped.append(direction)
    return deduped


def _action_mode(direction: str, policy: str) -> str:
    if policy != "auto":
        return policy
    return "local_3d_replace" if direction == "inner" else "local_3d_intersect"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a v347 canonical 3D component-local asset from residual samples."
    )
    parser.add_argument("--baseline-render-exp", required=True, type=Path)
    parser.add_argument("--current-render-exp", required=True, type=Path)
    parser.add_argument("--component-csv", default="assets/adopted_geometry/377/v320_selected_components.csv", type=Path)
    parser.add_argument("--point-csv", default="assets/adopted_geometry/377/v304_point_contributors_all.csv", type=Path)
    parser.add_argument("--signed-point-json", type=Path, default=None)
    parser.add_argument("--exclude-drop-json", type=Path, default=None)
    parser.add_argument("--min-positive", default=1.0, type=float)
    parser.add_argument("--min-hard-positive", default=0.00005, type=float)
    parser.add_argument("--min-edge-positive", default=0.004, type=float)
    parser.add_argument("--top-frames", default=80, type=int)
    parser.add_argument("--components-per-frame", default=2, type=int)
    parser.add_argument("--max-actions", default=160, type=int)
    parser.add_argument("--max-top-ids", default=8, type=int)
    parser.add_argument("--min-owner-consistency", default=0.50, type=float)
    parser.add_argument("--owner-gate", action="store_true")
    parser.add_argument("--mode-policy", default="auto", choices=(
        "auto",
        "local_3d_replace",
        "local_3d_intersect",
        "local_3d_union",
    ))
    parser.add_argument("--radius-floor", default=0.010, type=float)
    parser.add_argument("--radius-pad", default=0.006, type=float)
    parser.add_argument("--radius-scale", default=1.80, type=float)
    parser.add_argument("--cluster-enable", action="store_true")
    parser.add_argument("--cluster-min-points", default=4, type=int)
    parser.add_argument("--cluster-radius-max", default=0.18, type=float)
    parser.add_argument("--cluster-owner-gate", action="store_true")
    parser.add_argument("--virtual-grow-clone-enable", action="store_true")
    parser.add_argument("--virtual-grow-clone-min-inner-gain", default=0.0, type=float)
    parser.add_argument("--virtual-grow-clone-max-outer-regress", default=0.0, type=float)
    parser.add_argument("--virtual-grow-clone-min-opacity-outer-regress", default=0.0, type=float)
    parser.add_argument("--virtual-grow-clone-max-opacity-outer-regress", default=1.0e9, type=float)
    parser.add_argument("--virtual-grow-clone-max-hard-regress", default=0.0, type=float)
    parser.add_argument("--virtual-grow-clone-max-radius", default=0.12, type=float)
    parser.add_argument("--virtual-grow-clone-min-owner-consistency", default=0.75, type=float)
    parser.add_argument("--virtual-grow-clone-opacity-scale", default=None, type=float)
    parser.add_argument("--include-temporal-source", action="store_true")
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-row-guard-json", required=True, type=Path)
    parser.add_argument("--out-candidates-tsv", required=True, type=Path)
    args = parser.parse_args()

    baseline = _load_samples(args.baseline_render_exp)
    current = _load_samples(args.current_render_exp)
    components = _load_component_rows(args.component_csv)
    point_stats = _load_point_stats(args.point_csv)
    excluded = _load_drop_images(args.exclude_drop_json)
    temporal_source = _load_signed_temporal_sources(args.signed_point_json)

    frame_rows = []
    for image_name, cur in current.items():
        base = baseline.get(image_name)
        if not base or image_name in excluded:
            continue
        comp_rows = components.get(image_name, {"inner": [], "outer": []})
        if not comp_rows.get("inner") and not comp_rows.get("outer"):
            continue
        delta = {metric: _metric_delta(cur, base, metric) for metric in METRICS}
        max_count_delta = max(
            delta.get("inner", 0.0),
            delta.get("outer", 0.0),
            delta.get("opacity_inner", 0.0),
            delta.get("opacity_outer", 0.0),
        )
        if (
            max_count_delta < float(args.min_positive)
            and delta.get("hard", 0.0) < float(args.min_hard_positive)
            and delta.get("edge", 0.0) < float(args.min_edge_positive)
        ):
            continue
        score = (
            max(delta.get("opacity_outer", 0.0), 0.0)
            + max(delta.get("opacity_inner", 0.0), 0.0)
            + max(delta.get("outer", 0.0), 0.0)
            + max(delta.get("inner", 0.0), 0.0)
            + 1000.0 * max(delta.get("hard", 0.0), 0.0)
            + 10.0 * max(delta.get("edge", 0.0), 0.0)
        )
        frame_rows.append({
            "image_name": image_name,
            "score": score,
            **{f"{metric}_delta": delta[metric] for metric in METRICS},
        })
    frame_rows.sort(key=lambda row: (float(row["score"]), str(row["image_name"])), reverse=True)
    if args.top_frames > 0:
        frame_rows = frame_rows[: args.top_frames]

    actions = []
    row_guard_drops = []
    audit_rows = []
    seen = set()

    def emit_action(image_name: str, rec: dict[str, object], delta: dict[str, float], frame_score: float) -> None:
        nonlocal actions
        direction = str(rec["direction"])
        owner = _owner(list(rec.get("top_point_ids", [])), point_stats)
        if float(owner["owner_consistency"]) < float(args.min_owner_consistency):
            return
        support = _canonical_support(
            list(rec.get("top_point_ids", [])),
            point_stats,
            radius_floor=float(args.radius_floor),
            radius_pad=float(args.radius_pad),
            radius_scale=float(args.radius_scale),
            max_top_ids=int(args.max_top_ids),
            cluster_enable=bool(args.cluster_enable),
            cluster_min_points=int(args.cluster_min_points),
            cluster_owner_gate=bool(args.cluster_owner_gate),
            cluster_radius_max=float(args.cluster_radius_max),
        )
        if not support:
            return
        key = (image_name, direction, int(rec["row_index"]))
        if key in seen:
            return
        seen.add(key)
        mode = _action_mode(direction, args.mode_policy)
        action = {
            "component_key": f"{image_name}:{direction}:row{int(rec['row_index'])}",
            "image_name": image_name,
            "direction": direction,
            "row_index": int(rec["row_index"]),
            "component_id": int(rec["component_id"]),
            "mode": mode,
            "bbox_x": float(rec["bbox_x"]),
            "bbox_y": float(rec["bbox_y"]),
            "bbox_w": float(rec["bbox_w"]),
            "bbox_h": float(rec["bbox_h"]),
            "centroid_x": float(rec["centroid_x"]),
            "centroid_y": float(rec["centroid_y"]),
            "area": float(rec["area"]),
            "near_score_sum": float(rec["near_score_sum"]),
            "top_ids_enable": False,
            "top_ids_only": False,
            "max_top_ids": int(args.max_top_ids),
            "owner_gate": bool(args.owner_gate),
            "local_3d_fallback_top_ids": True,
            "reason": "v347_component_canonical_3d_residual_asset",
            **support,
            **owner,
            **{f"{metric}_delta_base": delta[metric] for metric in METRICS},
        }
        if bool(args.virtual_grow_clone_enable) and direction == "inner":
            inner_gain = max(-float(delta.get("inner", 0.0)), 0.0) + max(-float(delta.get("opacity_inner", 0.0)), 0.0)
            outer_regress = max(float(delta.get("outer", 0.0)), 0.0)
            opacity_outer_regress = max(float(delta.get("opacity_outer", 0.0)), 0.0)
            hard_regress = max(float(delta.get("hard", 0.0)), 0.0)
            if (
                inner_gain >= float(args.virtual_grow_clone_min_inner_gain)
                and outer_regress <= float(args.virtual_grow_clone_max_outer_regress)
                and opacity_outer_regress >= float(args.virtual_grow_clone_min_opacity_outer_regress)
                and opacity_outer_regress <= float(args.virtual_grow_clone_max_opacity_outer_regress)
                and hard_regress <= float(args.virtual_grow_clone_max_hard_regress)
                and float(support.get("canonical_radius", 0.0) or 0.0) <= float(args.virtual_grow_clone_max_radius)
                and float(owner.get("owner_consistency", 0.0) or 0.0) >= float(args.virtual_grow_clone_min_owner_consistency)
            ):
                action["virtual_grow_clone_enable"] = True
                action["clone_role"] = "inner_grow_support"
                action["virtual_grow_clone_reason"] = "guarded_inner_gain_no_outer_regress"
                if args.virtual_grow_clone_opacity_scale is not None:
                    action["virtual_grow_clone_opacity_scale"] = float(args.virtual_grow_clone_opacity_scale)
        actions.append(action)
        row_guard_drops.append({
            "image_name": image_name,
            "direction": direction,
            "row_index": int(rec["row_index"]),
            "component_id": int(rec["component_id"]),
            "bbox_x": float(rec["bbox_x"]),
            "bbox_y": float(rec["bbox_y"]),
            "bbox_w": float(rec["bbox_w"]),
            "bbox_h": float(rec["bbox_h"]),
            "centroid_x": float(rec["centroid_x"]),
            "centroid_y": float(rec["centroid_y"]),
            "tol_px": 1.0,
            "reason": "v347_row_guard_upper_bound",
        })
        audit_rows.append({
            "frame_score": frame_score,
            **action,
        })

    for frame in frame_rows:
        image_name = str(frame["image_name"])
        delta = {metric: float(frame[f"{metric}_delta"]) for metric in METRICS}
        directions = _candidate_directions(delta, components.get(image_name, {"inner": [], "outer": []}))
        for direction in directions:
            ranked_rows = sorted(
                components.get(image_name, {}).get(direction, []),
                key=lambda rec: (
                    float(rec.get("near_score_sum", 0.0)) * max(float(rec.get("area", 0.0)), 1.0),
                    float(rec.get("area", 0.0)),
                ),
                reverse=True,
            )
            kept = 0
            for rec in ranked_rows:
                emit_action(image_name, rec, delta, float(frame["score"]))
                kept += 1
                if int(args.components_per_frame) > 0 and kept >= int(args.components_per_frame):
                    break
                if int(args.max_actions) > 0 and len(actions) >= int(args.max_actions):
                    break
            if int(args.max_actions) > 0 and len(actions) >= int(args.max_actions):
                break

        if args.include_temporal_source:
            source_image = temporal_source.get(image_name, "")
            if source_image and source_image != image_name and source_image in components:
                for direction in directions:
                    ranked_rows = components[source_image].get(direction, [])[:1]
                    for rec in ranked_rows:
                        copied = dict(rec)
                        copied["image_name"] = image_name
                        emit_action(image_name, copied, delta, float(frame["score"]))
        if int(args.max_actions) > 0 and len(actions) >= int(args.max_actions):
            break

    payload = {
        "version": "v347_component_canonical_3d_asset",
        "policy": (
            "Discover residual component rows from dense no-train diagnostics, then express fixes as "
            "canonical 3D local neighborhoods with optional semantic owner gates."
        ),
        "source": {
            "baseline_render_exp": str(args.baseline_render_exp),
            "current_render_exp": str(args.current_render_exp),
            "component_csv": str(args.component_csv),
            "point_csv": str(args.point_csv),
            "signed_point_json": str(args.signed_point_json or ""),
            "exclude_drop_json": str(args.exclude_drop_json or ""),
        },
        "thresholds": {
            "min_positive": float(args.min_positive),
            "min_hard_positive": float(args.min_hard_positive),
            "min_edge_positive": float(args.min_edge_positive),
            "min_owner_consistency": float(args.min_owner_consistency),
            "radius_floor": float(args.radius_floor),
            "radius_pad": float(args.radius_pad),
            "radius_scale": float(args.radius_scale),
            "cluster_enable": bool(args.cluster_enable),
            "cluster_min_points": int(args.cluster_min_points),
            "cluster_radius_max": float(args.cluster_radius_max),
            "cluster_owner_gate": bool(args.cluster_owner_gate),
            "virtual_grow_clone_enable": bool(args.virtual_grow_clone_enable),
            "virtual_grow_clone_min_inner_gain": float(args.virtual_grow_clone_min_inner_gain),
            "virtual_grow_clone_max_outer_regress": float(args.virtual_grow_clone_max_outer_regress),
            "virtual_grow_clone_min_opacity_outer_regress": float(args.virtual_grow_clone_min_opacity_outer_regress),
            "virtual_grow_clone_max_opacity_outer_regress": float(args.virtual_grow_clone_max_opacity_outer_regress),
            "virtual_grow_clone_max_hard_regress": float(args.virtual_grow_clone_max_hard_regress),
            "virtual_grow_clone_max_radius": float(args.virtual_grow_clone_max_radius),
            "virtual_grow_clone_min_owner_consistency": float(args.virtual_grow_clone_min_owner_consistency),
        },
        "frame_count": len(frame_rows),
        "action_count": len(actions),
        "actions": actions,
    }
    row_guard_payload = {
        "version": "v347_component_canonical_3d_row_guard_upper_bound",
        "source_asset": str(args.out_json),
        "drop_count": len(row_guard_drops),
        "drops": row_guard_drops,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_row_guard_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_candidates_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_row_guard_json.write_text(json.dumps(row_guard_payload, indent=2, sort_keys=True), encoding="utf-8")

    fieldnames = []
    for row in audit_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with args.out_candidates_tsv.open("w", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(audit_rows)
        else:
            handle.write("component_key\n")

    mode_counts = Counter(str(action.get("mode", "")) for action in actions)
    print(f"wrote {args.out_json} actions={len(actions)} frames={len(frame_rows)} modes={dict(mode_counts)}")
    print(f"wrote {args.out_row_guard_json} drops={len(row_guard_drops)}")
    print(f"wrote {args.out_candidates_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
