#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.make_377_stageB_v347_component_3d_asset import (
    METRICS,
    _canonical_support,
    _float,
    _int,
    _load_component_rows,
    _load_drop_images,
    _load_point_stats,
    _load_samples,
    _metric_delta,
    _owner,
)


def _frame_score(delta: dict[str, float]) -> float:
    return (
        max(delta.get("opacity_outer", 0.0), 0.0)
        + max(delta.get("opacity_inner", 0.0), 0.0)
        + max(delta.get("outer", 0.0), 0.0)
        + max(delta.get("inner", 0.0), 0.0)
        + 1000.0 * max(delta.get("hard", 0.0), 0.0)
        + 10.0 * max(delta.get("edge", 0.0), 0.0)
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a v356 render-time split-child component asset from residual diagnostics."
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
    parser.add_argument("--children-per-frame", default=2, type=int)
    parser.add_argument("--max-children", default=120, type=int)
    parser.add_argument("--max-top-ids", default=8, type=int)
    parser.add_argument("--min-owner-consistency", default=0.50, type=float)
    parser.add_argument("--radius-floor", default=0.010, type=float)
    parser.add_argument("--radius-pad", default=0.006, type=float)
    parser.add_argument("--radius-scale", default=1.15, type=float)
    parser.add_argument("--child-radius-scale", default=0.38, type=float)
    parser.add_argument("--child-opacity", default=0.18, type=float)
    parser.add_argument("--cluster-enable", action="store_true")
    parser.add_argument("--cluster-min-points", default=4, type=int)
    parser.add_argument("--cluster-radius-max", default=0.18, type=float)
    parser.add_argument("--cluster-owner-gate", action="store_true")
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-candidates-tsv", required=True, type=Path)
    args = parser.parse_args()

    baseline = _load_samples(args.baseline_render_exp)
    current = _load_samples(args.current_render_exp)
    components = _load_component_rows(args.component_csv)
    point_stats = _load_point_stats(args.point_csv)
    excluded = _load_drop_images(args.exclude_drop_json)

    frames = []
    for image_name, cur in current.items():
        base = baseline.get(image_name)
        rows = components.get(image_name, {})
        if not base or image_name in excluded or not rows.get("inner"):
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

    children = []
    audit_rows = []
    seen = set()
    for frame in frames:
        image_name = str(frame["image_name"])
        delta = {metric: float(frame[f"{metric}_delta"]) for metric in METRICS}
        ranked = sorted(
            components.get(image_name, {}).get("inner", []),
            key=lambda rec: (
                float(rec.get("near_score_sum", 0.0)) * max(float(rec.get("area", 0.0)), 1.0),
                float(rec.get("area", 0.0)),
            ),
            reverse=True,
        )
        kept_for_frame = 0
        for rec in ranked:
            key = (image_name, int(rec["row_index"]))
            if key in seen:
                continue
            owner = _owner(list(rec.get("top_point_ids", [])), point_stats)
            if float(owner["owner_consistency"]) < float(args.min_owner_consistency):
                continue
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
                continue
            seen.add(key)
            child = {
                "component_key": f"{image_name}:inner:row{int(rec['row_index'])}",
                "image_name": image_name,
                "direction": "inner",
                "row_index": int(rec["row_index"]),
                "component_id": int(rec["component_id"]),
                "bbox_x": float(rec["bbox_x"]),
                "bbox_y": float(rec["bbox_y"]),
                "bbox_w": float(rec["bbox_w"]),
                "bbox_h": float(rec["bbox_h"]),
                "centroid_x": float(rec["centroid_x"]),
                "centroid_y": float(rec["centroid_y"]),
                "area": float(rec["area"]),
                "near_score_sum": float(rec["near_score_sum"]),
                "split_child_enable": True,
                "child_role": "inner_supplement",
                "child_pose_mode": "top_ids_translation",
                "child_color_source": "top_ids_mean",
                "child_radius_scale": float(args.child_radius_scale),
                "child_opacity": float(args.child_opacity),
                "reason": "v356_split_child_component_asset",
                **support,
                **owner,
                **{f"{metric}_delta_base": delta[metric] for metric in METRICS},
            }
            children.append(child)
            audit_rows.append({"frame_score": float(frame["score"]), **child})
            kept_for_frame += 1
            if int(args.max_children) > 0 and len(children) >= int(args.max_children):
                break
            if int(args.children_per_frame) > 0 and kept_for_frame >= int(args.children_per_frame):
                break
        if int(args.max_children) > 0 and len(children) >= int(args.max_children):
            break

    payload = {
        "version": "v356_split_child_component_asset",
        "policy": (
            "Render-time child Gaussian component asset. Children supplement inner residual components "
            "with independent child covariance and top-id canonical-to-posed translation anchoring."
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
            "radius_floor": float(args.radius_floor),
            "radius_pad": float(args.radius_pad),
            "radius_scale": float(args.radius_scale),
            "child_radius_scale": float(args.child_radius_scale),
            "child_opacity": float(args.child_opacity),
            "cluster_enable": bool(args.cluster_enable),
            "cluster_min_points": int(args.cluster_min_points),
            "cluster_radius_max": float(args.cluster_radius_max),
            "cluster_owner_gate": bool(args.cluster_owner_gate),
        },
        "frame_count": len(frames),
        "child_count": len(children),
        "children": children,
        "actions": children,
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
            handle.write("component_key\n")
    print(f"wrote {args.out_json} children={len(children)} frames={len(frames)}")
    print(f"wrote {args.out_candidates_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
