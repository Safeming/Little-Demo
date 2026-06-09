#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from make_377_stageB_v347_component_3d_asset import (
    METRICS,
    _canonical_support,
    _load_component_rows,
    _load_drop_images,
    _load_point_stats,
    _load_samples,
    _metric_delta,
    _owner,
)


def _rank_component(row: dict[str, object]) -> tuple[float, float, float]:
    area = float(row.get("area", 0.0) or 0.0)
    near = float(row.get("near_score_sum", 0.0) or 0.0)
    return near * max(area, 1.0), area, near


def _center(support: dict[str, object]) -> tuple[float, float, float]:
    value = support.get("canonical_center", [0.0, 0.0, 0.0])
    return float(value[0]), float(value[1]), float(value[2])


def _dist(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> float:
    return math.sqrt(sum((lhs[idx] - rhs[idx]) ** 2 for idx in range(3)))


def _owner_distance(lhs: dict[str, object], rhs: dict[str, object]) -> int:
    distance = 0
    for key in ("owner_region_id", "owner_joint", "owner_layer_id"):
        if lhs.get(key, "") != "" and rhs.get(key, "") != "" and lhs.get(key) != rhs.get(key):
            distance += 1
    return distance


def _frame_score(delta: dict[str, float]) -> float:
    return (
        max(delta.get("opacity_outer", 0.0), 0.0)
        + max(delta.get("opacity_inner", 0.0), 0.0)
        + max(delta.get("outer", 0.0), 0.0)
        + max(delta.get("inner", 0.0), 0.0)
        + 1000.0 * max(delta.get("hard", 0.0), 0.0)
        + 10.0 * max(delta.get("edge", 0.0), 0.0)
    )


def _needs(delta: dict[str, float]) -> tuple[float, float]:
    outer_need = (
        max(delta.get("outer", 0.0), 0.0)
        + max(delta.get("opacity_outer", 0.0), 0.0)
        + 1000.0 * max(delta.get("hard", 0.0), 0.0)
        + 5.0 * max(delta.get("edge", 0.0), 0.0)
    )
    inner_need = max(delta.get("inner", 0.0), 0.0) + max(delta.get("opacity_inner", 0.0), 0.0)
    return inner_need, outer_need


def _score_scales(delta: dict[str, float], boost: float) -> tuple[float, float]:
    inner_need, outer_need = _needs(delta)
    total = max(inner_need + outer_need, 1.0e-6)
    boost = max(float(boost), 0.0)
    return 1.0 + boost * inner_need / total, 1.0 + boost * outer_need / total


def _row_action(
    image_name: str,
    pair_id: str,
    pair_role: str,
    row: dict[str, object],
    owner: dict[str, object],
    support: dict[str, object] | None,
    mode: str,
    score_scale: float,
) -> dict[str, object]:
    direction = str(row["direction"])
    action = {
        "component_key": f"{image_name}:{direction}:row{int(row['row_index'])}",
        "image_name": image_name,
        "direction": direction,
        "pair_id": pair_id,
        "pair_role": pair_role,
        "row_index": int(row["row_index"]),
        "component_id": int(row["component_id"]),
        "mode": mode,
        "bbox_x": float(row["bbox_x"]),
        "bbox_y": float(row["bbox_y"]),
        "bbox_w": float(row["bbox_w"]),
        "bbox_h": float(row["bbox_h"]),
        "centroid_x": float(row["centroid_x"]),
        "centroid_y": float(row["centroid_y"]),
        "area": float(row["area"]),
        "near_score_sum": float(row["near_score_sum"]),
        "top_ids_enable": False,
        "top_ids_only": False,
        "score_scale": float(score_scale),
        "targeted_only": True,
        "semantic_override": True,
        "local_3d_fallback_top_ids": True,
        **owner,
    }
    if support:
        action.update(support)
    return action


def _shared_support(
    inner_support: dict[str, object],
    outer_support: dict[str, object],
    *,
    radius_pad: float,
    radius_scale: float,
    radius_floor: float,
    radius_max: float,
) -> dict[str, object] | None:
    inner_center = _center(inner_support)
    outer_center = _center(outer_support)
    center = tuple((inner_center[idx] + outer_center[idx]) * 0.5 for idx in range(3))
    inner_radius = float(inner_support.get("canonical_radius", 0.0) or 0.0)
    outer_radius = float(outer_support.get("canonical_radius", 0.0) or 0.0)
    radius = max(
        _dist(center, inner_center) + inner_radius,
        _dist(center, outer_center) + outer_radius,
    )
    radius = max(radius * float(radius_scale) + float(radius_pad), float(radius_floor))
    if float(radius_max) > 0.0 and radius > float(radius_max):
        return None
    top_ids = []
    for support in (inner_support, outer_support):
        for point_id in support.get("top_point_ids", []):
            if point_id not in top_ids:
                top_ids.append(point_id)
    return {
        "canonical_center": [float(center[0]), float(center[1]), float(center[2])],
        "canonical_radius": float(radius),
        "canonical_radius_inner": float(radius),
        "canonical_radius_outer": float(radius),
        "canonical_radius_source": {
            "inner_radius": inner_radius,
            "outer_radius": outer_radius,
            "center_distance": _dist(inner_center, outer_center),
            "pair_radius_pad": float(radius_pad),
            "pair_radius_scale": float(radius_scale),
            "pair_radius_floor": float(radius_floor),
            "pair_radius_max": float(radius_max),
        },
        "top_point_ids": top_ids,
        "source_top_point_ids": top_ids,
    }


def _support_for_row(
    row: dict[str, object],
    point_stats: dict[int, dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object] | None:
    return _canonical_support(
        list(row.get("top_point_ids", [])),
        point_stats,
        radius_floor=float(args.radius_floor),
        radius_pad=float(args.radius_pad),
        radius_scale=float(args.row_radius_scale),
        max_top_ids=int(args.max_top_ids),
        cluster_enable=bool(args.cluster_enable),
        cluster_min_points=int(args.cluster_min_points),
        cluster_owner_gate=bool(args.cluster_owner_gate),
        cluster_radius_max=float(args.cluster_radius_max),
    )


def _eligible_frames(
    baseline: dict[str, dict[str, float]],
    current: dict[str, dict[str, float]],
    components: dict[str, dict[str, list[dict[str, object]]]],
    excluded: set[str],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    frames = []
    for image_name, cur in current.items():
        base = baseline.get(image_name)
        if not base or image_name in excluded:
            continue
        rows = components.get(image_name, {"inner": [], "outer": []})
        if not rows.get("inner") or not rows.get("outer"):
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
        frames.append({
            "image_name": image_name,
            "score": _frame_score(delta),
            **{f"{metric}_delta": delta[metric] for metric in METRICS},
        })
    frames.sort(key=lambda row: (float(row["score"]), str(row["image_name"])), reverse=True)
    if int(args.top_frames) > 0:
        frames = frames[: int(args.top_frames)]
    return frames


def _pair_rows(
    image_name: str,
    delta: dict[str, float],
    rows: dict[str, list[dict[str, object]]],
    point_stats: dict[int, dict[str, object]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    inner_rows = sorted(rows.get("inner", []), key=_rank_component, reverse=True)[: max(int(args.rows_per_direction), 1)]
    outer_rows = sorted(rows.get("outer", []), key=_rank_component, reverse=True)[: max(int(args.rows_per_direction), 1)]
    inner_items = []
    outer_items = []
    for row in inner_rows:
        owner = _owner(list(row.get("top_point_ids", [])), point_stats)
        if float(owner["owner_consistency"]) < float(args.min_owner_consistency):
            continue
        support = _support_for_row(row, point_stats, args)
        if support:
            inner_items.append((row, owner, support))
    for row in outer_rows:
        owner = _owner(list(row.get("top_point_ids", [])), point_stats)
        if float(owner["owner_consistency"]) < float(args.min_owner_consistency):
            continue
        support = _support_for_row(row, point_stats, args)
        if support:
            outer_items.append((row, owner, support))

    candidates = []
    for inner_row, inner_owner, inner_support in inner_items:
        for outer_row, outer_owner, outer_support in outer_items:
            center_distance = _dist(_center(inner_support), _center(outer_support))
            if float(args.pair_center_max) > 0.0 and center_distance > float(args.pair_center_max):
                continue
            owner_distance = _owner_distance(inner_owner, outer_owner)
            if bool(args.require_owner_match) and owner_distance > 0:
                continue
            shared = _shared_support(
                inner_support,
                outer_support,
                radius_pad=float(args.pair_radius_pad),
                radius_scale=float(args.pair_radius_scale),
                radius_floor=float(args.radius_floor),
                radius_max=float(args.pair_radius_max),
            )
            if not shared:
                continue
            pair_score = (
                _rank_component(inner_row)[0]
                + _rank_component(outer_row)[0]
                - 1000.0 * center_distance
                - 250.0 * owner_distance
            )
            candidates.append({
                "score": pair_score,
                "center_distance": center_distance,
                "owner_distance": owner_distance,
                "inner_row": inner_row,
                "outer_row": outer_row,
                "inner_owner": inner_owner,
                "outer_owner": outer_owner,
                "inner_support": inner_support,
                "outer_support": outer_support,
                "shared": shared,
            })
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    used_inner = set()
    used_outer = set()
    pairs = []
    inner_scale, outer_scale = _score_scales(delta, float(args.score_scale_boost))
    for candidate in candidates:
        inner_key = int(candidate["inner_row"]["row_index"])
        outer_key = int(candidate["outer_row"]["row_index"])
        if inner_key in used_inner or outer_key in used_outer:
            continue
        pair_id = f"{image_name}:pair{len(pairs):02d}:i{inner_key}:o{outer_key}"
        owner = candidate["inner_owner"] if candidate["owner_distance"] <= 0 else candidate["outer_owner"]
        shared = dict(candidate["shared"])
        shared.update(owner)
        shared.update({
            "owner_gate": bool(args.owner_gate),
            "max_top_ids": int(args.max_top_ids),
            "pair_center_distance": float(candidate["center_distance"]),
            "pair_owner_distance": int(candidate["owner_distance"]),
            "reason": "v349_paired_component_canonical_3d_residual_asset",
            **{f"{metric}_delta_base": delta[metric] for metric in METRICS},
        })
        inner_action = _row_action(
            image_name,
            pair_id,
            "inner_grow",
            candidate["inner_row"],
            candidate["inner_owner"],
            candidate["inner_support"] if bool(args.row_local_actions) else None,
            str(args.inner_mode),
            inner_scale,
        )
        outer_action = _row_action(
            image_name,
            pair_id,
            "outer_shrink",
            candidate["outer_row"],
            candidate["outer_owner"],
            candidate["outer_support"] if bool(args.row_local_actions) else None,
            str(args.outer_mode),
            outer_scale,
        )
        pairs.append({
            "pair_id": pair_id,
            "image_name": image_name,
            "shared": shared,
            "inner_action": inner_action,
            "outer_action": outer_action,
        })
        used_inner.add(inner_key)
        used_outer.add(outer_key)
        if int(args.pairs_per_frame) > 0 and len(pairs) >= int(args.pairs_per_frame):
            break
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a v349 paired canonical 3D component-local asset from dense residual diagnostics."
    )
    parser.add_argument("--baseline-render-exp", required=True, type=Path)
    parser.add_argument("--current-render-exp", required=True, type=Path)
    parser.add_argument("--component-csv", default="assets/adopted_geometry/377/v320_selected_components.csv", type=Path)
    parser.add_argument("--point-csv", default="assets/adopted_geometry/377/v304_point_contributors_all.csv", type=Path)
    parser.add_argument("--exclude-drop-json", type=Path, default=None)
    parser.add_argument("--min-positive", default=1.0, type=float)
    parser.add_argument("--min-hard-positive", default=0.00005, type=float)
    parser.add_argument("--min-edge-positive", default=0.004, type=float)
    parser.add_argument("--top-frames", default=80, type=int)
    parser.add_argument("--rows-per-direction", default=5, type=int)
    parser.add_argument("--pairs-per-frame", default=2, type=int)
    parser.add_argument("--max-pairs", default=120, type=int)
    parser.add_argument("--max-top-ids", default=8, type=int)
    parser.add_argument("--min-owner-consistency", default=0.50, type=float)
    parser.add_argument("--owner-gate", action="store_true")
    parser.add_argument("--require-owner-match", action="store_true")
    parser.add_argument("--row-local-actions", action="store_true")
    parser.add_argument("--inner-mode", default="paired_local_3d_replace")
    parser.add_argument("--outer-mode", default="paired_local_3d_intersect")
    parser.add_argument("--score-scale-boost", default=0.35, type=float)
    parser.add_argument("--radius-floor", default=0.010, type=float)
    parser.add_argument("--radius-pad", default=0.006, type=float)
    parser.add_argument("--row-radius-scale", default=1.15, type=float)
    parser.add_argument("--pair-radius-pad", default=0.004, type=float)
    parser.add_argument("--pair-radius-scale", default=1.05, type=float)
    parser.add_argument("--pair-radius-max", default=0.24, type=float)
    parser.add_argument("--pair-center-max", default=0.18, type=float)
    parser.add_argument("--cluster-enable", action="store_true")
    parser.add_argument("--cluster-min-points", default=4, type=int)
    parser.add_argument("--cluster-radius-max", default=0.18, type=float)
    parser.add_argument("--cluster-owner-gate", action="store_true")
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-row-guard-json", required=True, type=Path)
    parser.add_argument("--out-candidates-tsv", required=True, type=Path)
    args = parser.parse_args()

    baseline = _load_samples(args.baseline_render_exp)
    current = _load_samples(args.current_render_exp)
    components = _load_component_rows(args.component_csv)
    point_stats = _load_point_stats(args.point_csv)
    excluded = _load_drop_images(args.exclude_drop_json)

    frames = _eligible_frames(baseline, current, components, excluded, args)
    pairs = []
    actions = []
    row_guard_drops = []
    audit_rows = []
    for frame in frames:
        image_name = str(frame["image_name"])
        delta = {metric: float(frame[f"{metric}_delta"]) for metric in METRICS}
        for pair in _pair_rows(image_name, delta, components.get(image_name, {}), point_stats, args):
            if int(args.max_pairs) > 0 and len(pairs) >= int(args.max_pairs):
                break
            flattened = []
            for key in ("inner_action", "outer_action"):
                action = dict(pair["shared"])
                action.update(pair[key])
                actions.append(action)
                flattened.append(action)
                row_guard_drops.append({
                    "image_name": image_name,
                    "direction": action["direction"],
                    "row_index": int(action["row_index"]),
                    "component_id": int(action["component_id"]),
                    "bbox_x": float(action["bbox_x"]),
                    "bbox_y": float(action["bbox_y"]),
                    "bbox_w": float(action["bbox_w"]),
                    "bbox_h": float(action["bbox_h"]),
                    "centroid_x": float(action["centroid_x"]),
                    "centroid_y": float(action["centroid_y"]),
                    "tol_px": 1.0,
                    "reason": "v349_pair_row_guard_upper_bound",
                })
            pairs.append(pair)
            for action in flattened:
                audit_rows.append({
                    "frame_score": float(frame["score"]),
                    **action,
                })
        if int(args.max_pairs) > 0 and len(pairs) >= int(args.max_pairs):
            break

    payload = {
        "version": "v349_paired_component_canonical_3d_asset",
        "policy": (
            "Pair inner grow and outer shrink component rows in the same residual frame. "
            "When row_local_actions is enabled, the pair is used for coupled selection and audit, "
            "while each action keeps its own canonical 3D support."
        ),
        "source": {
            "baseline_render_exp": str(args.baseline_render_exp),
            "current_render_exp": str(args.current_render_exp),
            "component_csv": str(args.component_csv),
            "point_csv": str(args.point_csv),
            "exclude_drop_json": str(args.exclude_drop_json or ""),
        },
        "thresholds": {
            "min_positive": float(args.min_positive),
            "min_hard_positive": float(args.min_hard_positive),
            "min_edge_positive": float(args.min_edge_positive),
            "min_owner_consistency": float(args.min_owner_consistency),
            "owner_gate": bool(args.owner_gate),
            "require_owner_match": bool(args.require_owner_match),
            "pair_center_max": float(args.pair_center_max),
            "pair_radius_max": float(args.pair_radius_max),
            "row_local_actions": bool(args.row_local_actions),
            "cluster_enable": bool(args.cluster_enable),
            "cluster_min_points": int(args.cluster_min_points),
            "cluster_radius_max": float(args.cluster_radius_max),
            "cluster_owner_gate": bool(args.cluster_owner_gate),
        },
        "frame_count": len(frames),
        "pair_count": len(pairs),
        "action_count": len(actions),
        "pairs": pairs,
        "actions": actions,
    }
    row_guard_payload = {
        "version": "v349_pair_row_guard_upper_bound",
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

    print(f"wrote {args.out_json} pairs={len(pairs)} actions={len(actions)} frames={len(frames)}")
    print(f"wrote {args.out_row_guard_json} drops={len(row_guard_drops)}")
    print(f"wrote {args.out_candidates_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
