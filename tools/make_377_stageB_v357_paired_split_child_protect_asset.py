#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.make_377_stageB_v347_component_3d_asset import (
    METRICS,
    _canonical_support,
    _load_component_rows,
    _load_point_stats,
    _owner,
)


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def _center3(support: dict[str, object]) -> tuple[float, float, float]:
    value = support.get("canonical_center", [0.0, 0.0, 0.0])
    return float(value[0]), float(value[1]), float(value[2])


def _dist3(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> float:
    return math.sqrt(sum((lhs[idx] - rhs[idx]) ** 2 for idx in range(3)))


def _screen_dist(lhs: dict[str, object], rhs: dict[str, object]) -> float:
    lx = float(lhs.get("centroid_x", 0.0) or 0.0)
    ly = float(lhs.get("centroid_y", 0.0) or 0.0)
    rx = float(rhs.get("centroid_x", 0.0) or 0.0)
    ry = float(rhs.get("centroid_y", 0.0) or 0.0)
    return math.hypot(lx - rx, ly - ry)


def _owner_distance(lhs: dict[str, object], rhs: dict[str, object]) -> int:
    distance = 0
    for key in ("owner_region_id", "owner_joint", "owner_layer_id"):
        if lhs.get(key, "") != "" and rhs.get(key, "") != "" and lhs.get(key) != rhs.get(key):
            distance += 1
    return distance


def _row_rank(row: dict[str, object]) -> float:
    area = float(row.get("area", 0.0) or 0.0)
    near = float(row.get("near_score_sum", 0.0) or 0.0)
    return near * max(area, 1.0)


def _load_v356_children(path: Path) -> dict[str, dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("children") or data.get("actions") or []
    out = {}
    for row in rows:
        if isinstance(row, dict) and row.get("component_key"):
            out[str(row["component_key"])] = row
    return out


def _load_keep_candidates(path: Path, *, min_target_gain: float, min_outer_hurt: float) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("status") == "keep":
                continue
            inner_gain = max(-_float(row, "inner_delta_control"), -_float(row, "opacity_inner_delta_control"), 0.0)
            outer_hurt = max(
                _float(row, "outer_delta_control"),
                _float(row, "opacity_outer_delta_control"),
                1000.0 * _float(row, "hard_delta_control"),
                10.0 * _float(row, "edge_delta_control"),
                0.0,
            )
            if inner_gain < float(min_target_gain) or outer_hurt < float(min_outer_hurt):
                continue
            row["_inner_gain"] = str(inner_gain)
            row["_outer_hurt"] = str(outer_hurt)
            rows.append(row)
    rows.sort(key=lambda r: (_float(r, "_inner_gain") + _float(r, "_outer_hurt"), r.get("component_key", "")), reverse=True)
    return rows


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


def _outer_action(
    *,
    pair_id: str,
    outer_row: dict[str, object],
    owner: dict[str, object],
    support: dict[str, object],
    mode: str,
    score_scale: float,
) -> dict[str, object]:
    image_name = str(outer_row["image_name"])
    row_index = int(outer_row["row_index"])
    action = {
        "component_key": f"{image_name}:outer:row{row_index}",
        "image_name": image_name,
        "direction": "outer",
        "pair_id": pair_id,
        "pair_role": "outer_parent_protect",
        "row_index": row_index,
        "component_id": int(outer_row["component_id"]),
        "mode": str(mode),
        "bbox_x": float(outer_row["bbox_x"]),
        "bbox_y": float(outer_row["bbox_y"]),
        "bbox_w": float(outer_row["bbox_w"]),
        "bbox_h": float(outer_row["bbox_h"]),
        "centroid_x": float(outer_row["centroid_x"]),
        "centroid_y": float(outer_row["centroid_y"]),
        "area": float(outer_row["area"]),
        "near_score_sum": float(outer_row["near_score_sum"]),
        "top_ids_enable": False,
        "top_ids_only": False,
        "score_scale": float(score_scale),
        "targeted_only": True,
        "semantic_override": True,
        "local_3d_fallback_top_ids": True,
        **support,
        **owner,
    }
    return action


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build v357 paired split-child + parent outer protect asset from v356 action validation."
    )
    parser.add_argument("--v356-seed-json", required=True, type=Path)
    parser.add_argument("--v356-action-validation-tsv", required=True, type=Path)
    parser.add_argument("--component-csv", default="assets/adopted_geometry/377/v320_selected_components.csv", type=Path)
    parser.add_argument("--point-csv", default="assets/adopted_geometry/377/v304_point_contributors_all.csv", type=Path)
    parser.add_argument("--min-target-gain", default=0.25, type=float)
    parser.add_argument("--min-outer-hurt", default=1.0, type=float)
    parser.add_argument("--outer-candidates-per-child", default=2, type=int)
    parser.add_argument("--max-pairs", default=12, type=int)
    parser.add_argument("--max-top-ids", default=8, type=int)
    parser.add_argument("--min-owner-consistency", default=0.50, type=float)
    parser.add_argument("--require-owner-match", action="store_true")
    parser.add_argument("--outer-mode", default="paired_local_3d_intersect")
    parser.add_argument("--outer-score-scale", default=1.35, type=float)
    parser.add_argument("--radius-floor", default=0.010, type=float)
    parser.add_argument("--radius-pad", default=0.006, type=float)
    parser.add_argument("--row-radius-scale", default=1.15, type=float)
    parser.add_argument("--cluster-enable", action="store_true")
    parser.add_argument("--cluster-min-points", default=4, type=int)
    parser.add_argument("--cluster-radius-max", default=0.18, type=float)
    parser.add_argument("--cluster-owner-gate", action="store_true")
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-candidates-tsv", required=True, type=Path)
    args = parser.parse_args()

    children_by_key = _load_v356_children(args.v356_seed_json)
    validation_rows = _load_keep_candidates(
        args.v356_action_validation_tsv,
        min_target_gain=float(args.min_target_gain),
        min_outer_hurt=float(args.min_outer_hurt),
    )
    components = _load_component_rows(args.component_csv)
    point_stats = _load_point_stats(args.point_csv)

    pairs = []
    actions = []
    children = []
    audit_rows = []
    used_pairs = set()
    for validation in validation_rows:
        component_key = str(validation.get("component_key", "") or "")
        child_src = children_by_key.get(component_key)
        if not child_src:
            continue
        image_name = str(child_src.get("image_name", "") or "")
        outer_rows = list(components.get(image_name, {}).get("outer", []))
        if not image_name or not outer_rows:
            continue
        child_owner = {
            key: child_src.get(key, "")
            for key in ("owner_layer", "owner_layer_id", "owner_region", "owner_region_id", "owner_joint")
        }
        child_support = {
            "canonical_center": child_src.get("canonical_center", [0.0, 0.0, 0.0]),
            "canonical_radius": child_src.get("canonical_radius", 0.0),
        }
        ranked = []
        for outer in outer_rows:
            owner = _owner(list(outer.get("top_point_ids", [])), point_stats)
            if float(owner["owner_consistency"]) < float(args.min_owner_consistency):
                continue
            owner_dist = _owner_distance(child_owner, owner)
            if bool(args.require_owner_match) and owner_dist > 0:
                continue
            support = _support_for_row(outer, point_stats, args)
            if not support:
                continue
            dist3 = _dist3(_center3(child_support), _center3(support))
            dist2 = _screen_dist(child_src, outer)
            score = _row_rank(outer) - 800.0 * dist3 - 10.0 * dist2 - 250.0 * owner_dist
            ranked.append((score, outer, owner, support, dist2, dist3, owner_dist))
        ranked.sort(key=lambda item: item[0], reverse=True)
        for local_index, (score, outer, owner, support, dist2, dist3, owner_dist) in enumerate(
            ranked[: max(int(args.outer_candidates_per_child), 1)]
        ):
            pair_id = f"{image_name}:v357:i{int(child_src['row_index'])}:o{int(outer['row_index'])}:k{local_index}"
            if pair_id in used_pairs:
                continue
            used_pairs.add(pair_id)
            child = dict(child_src)
            child["pair_id"] = pair_id
            child["split_child_enable"] = True
            child["split_child_reason"] = "v357_candidate_child_from_v356_failed_action"
            outer_action = _outer_action(
                pair_id=pair_id,
                outer_row=outer,
                owner=owner,
                support=support,
                mode=str(args.outer_mode),
                score_scale=float(args.outer_score_scale),
            )
            pair = {
                "pair_id": pair_id,
                "image_name": image_name,
                "child_component_key": component_key,
                "outer_component_key": outer_action["component_key"],
                "child_action": child,
                "outer_action": outer_action,
                "diagnostic": {
                    "v356_inner_gain": _float(validation, "_inner_gain"),
                    "v356_outer_hurt": _float(validation, "_outer_hurt"),
                    "pair_score": float(score),
                    "screen_center_distance": float(dist2),
                    "canonical_center_distance": float(dist3),
                    "owner_distance": int(owner_dist),
                    **{f"v356_{metric}_delta_control": _float(validation, f"{metric}_delta_control") for metric in METRICS},
                },
            }
            pairs.append(pair)
            children.append(child)
            actions.append(outer_action)
            audit_rows.append({
                "pair_id": pair_id,
                "image_name": image_name,
                "child_component_key": component_key,
                "outer_component_key": outer_action["component_key"],
                "child_row_index": child.get("row_index", ""),
                "outer_row_index": outer_action.get("row_index", ""),
                "v356_inner_gain": _float(validation, "_inner_gain"),
                "v356_outer_hurt": _float(validation, "_outer_hurt"),
                "screen_center_distance": float(dist2),
                "canonical_center_distance": float(dist3),
                "owner_distance": int(owner_dist),
                "pair_score": float(score),
            })
            if int(args.max_pairs) > 0 and len(pairs) >= int(args.max_pairs):
                break
        if int(args.max_pairs) > 0 and len(pairs) >= int(args.max_pairs):
            break

    payload = {
        "version": "v357_paired_split_child_parent_protect_asset",
        "policy": (
            "Pair only v356 split-child actions that had target inner/opacity gain but failed do-no-harm "
            "because of outer/opacity/edge/hard damage, with same-frame parent outer row-local protect/shrink."
        ),
        "source": {
            "v356_seed_json": str(args.v356_seed_json),
            "v356_action_validation_tsv": str(args.v356_action_validation_tsv),
            "component_csv": str(args.component_csv),
            "point_csv": str(args.point_csv),
        },
        "thresholds": {
            "min_target_gain": float(args.min_target_gain),
            "min_outer_hurt": float(args.min_outer_hurt),
            "outer_candidates_per_child": int(args.outer_candidates_per_child),
            "max_pairs": int(args.max_pairs),
            "min_owner_consistency": float(args.min_owner_consistency),
            "require_owner_match": bool(args.require_owner_match),
            "outer_mode": str(args.outer_mode),
            "outer_score_scale": float(args.outer_score_scale),
            "cluster_enable": bool(args.cluster_enable),
            "cluster_min_points": int(args.cluster_min_points),
            "cluster_radius_max": float(args.cluster_radius_max),
            "cluster_owner_gate": bool(args.cluster_owner_gate),
        },
        "pair_count": len(pairs),
        "child_count": len(children),
        "action_count": len(actions),
        "pairs": pairs,
        "children": children,
        "actions": actions,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_candidates_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

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
            handle.write("pair_id\n")
    print(f"wrote {args.out_json} pairs={len(pairs)} children={len(children)} actions={len(actions)}")
    print(f"wrote {args.out_candidates_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
