#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
                "top_point_scores": str(row.get("top_point_scores", "") or ""),
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
    if not path or not path.exists():
        return {}
    stats: dict[int, dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            point_idx = _int(row, "point_idx", -1)
            if point_idx < 0:
                continue
            region_id = _int(row, "region_id", -1)
            layer_id = _int(row, "layer_id", -1)
            stats[point_idx] = {
                "layer_id": layer_id,
                "layer_name": str(row.get("layer_name", "") or LAYER_NAMES.get(layer_id, "")),
                "region_id": region_id,
                "region_name": str(row.get("region_name", "") or REGION_NAMES.get(region_id, "")),
                "dominant_joint": _int(row, "dominant_joint", -1),
                "boundary_score": _float(row, "boundary_score"),
                "surface_distance": _float(row, "surface_distance"),
                "thin_score": _float(row, "thin_score"),
            }
    return stats


def _load_drop_images(path: Path | None) -> set[str]:
    if not path:
        return set()
    if not path.exists():
        return set()
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(item).strip() for item in data.get("drop_images", []) if str(item).strip()}
    text = path.read_text(encoding="utf-8")
    return {token.strip() for token in text.replace(";", ",").split(",") if token.strip()}


def _owner(top_ids: list[int], point_stats: dict[int, dict[str, object]]) -> dict[str, object]:
    stats = [point_stats[idx] for idx in top_ids if idx in point_stats]
    if not stats:
        return {
            "owner_layer": "",
            "owner_region": "",
            "owner_joint": "",
            "owner_consistency": 0.0,
        }
    layers = Counter(str(item.get("layer_name", "")) for item in stats)
    regions = Counter(str(item.get("region_name", "")) for item in stats)
    joints = Counter(int(item.get("dominant_joint", -1)) for item in stats)
    layer, layer_count = layers.most_common(1)[0]
    region, region_count = regions.most_common(1)[0]
    joint, joint_count = joints.most_common(1)[0]
    consistency = min(layer_count, region_count, joint_count) / max(len(stats), 1)
    return {
        "owner_layer": layer,
        "owner_region": region,
        "owner_joint": joint,
        "owner_consistency": float(consistency),
    }


def _metric_delta(current: dict[str, float], base: dict[str, float], metric: str) -> float:
    return float(current.get(metric, 0.0)) - float(base.get(metric, 0.0))


def _candidate_direction(delta: dict[str, float], rows: dict[str, list[dict[str, object]]]) -> str:
    outer_like = max(delta.get("outer", 0.0), 0.0) + max(delta.get("opacity_outer", 0.0), 0.0)
    inner_like = max(delta.get("inner", 0.0), 0.0) + max(delta.get("opacity_inner", 0.0), 0.0)
    if outer_like >= inner_like and rows.get("inner"):
        return "inner"
    if rows.get("outer"):
        return "outer"
    if rows.get("inner"):
        return "inner"
    return ""


def _action_mode(direction: str, delta: dict[str, float], mode_policy: str) -> str:
    if mode_policy != "auto":
        return mode_policy
    if direction == "inner" and (delta.get("opacity_outer", 0.0) > 0.0 or delta.get("hard", 0.0) > 0.0):
        return "tight_top_ids"
    if direction == "outer" and (delta.get("inner", 0.0) > 0.0 or delta.get("opacity_inner", 0.0) > 0.0):
        return "tight_bbox"
    return "tight_top_ids"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a generic v346 component-local residual asset from dense contour gate samples."
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
    parser.add_argument("--components-per-frame", default=2, type=int)
    parser.add_argument("--max-actions", default=160, type=int)
    parser.add_argument("--min-owner-consistency", default=0.50, type=float)
    parser.add_argument("--mode-policy", default="auto", choices=("auto", "tight_top_ids", "tight_bbox", "drop_component"))
    parser.add_argument("--tight-bbox-pad-px", default=1.0, type=float)
    parser.add_argument("--tight-bbox-ellipse-scale", default=1.02, type=float)
    parser.add_argument("--max-top-ids", default=8, type=int)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-row-guard-json", required=True, type=Path)
    parser.add_argument("--out-candidates-tsv", required=True, type=Path)
    args = parser.parse_args()

    baseline = _load_samples(args.baseline_render_exp)
    current = _load_samples(args.current_render_exp)
    components = _load_component_rows(args.component_csv)
    point_stats = _load_point_stats(args.point_csv)
    excluded = _load_drop_images(args.exclude_drop_json)

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
        direction = _candidate_direction(delta, comp_rows)
        if not direction:
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
            "direction": direction,
            "score": score,
            **{f"{metric}_delta": delta[metric] for metric in METRICS},
        })
    frame_rows.sort(key=lambda row: (float(row["score"]), str(row["image_name"])), reverse=True)
    if args.top_frames > 0:
        frame_rows = frame_rows[: args.top_frames]

    actions = []
    audit_rows = []
    seen = set()
    for frame in frame_rows:
        image_name = str(frame["image_name"])
        direction = str(frame["direction"])
        delta = {metric: float(frame[f"{metric}_delta"]) for metric in METRICS}
        rows = components.get(image_name, {}).get(direction, [])
        ranked_rows = sorted(
            rows,
            key=lambda rec: (
                float(rec.get("near_score_sum", 0.0)) * max(float(rec.get("area", 0.0)), 1.0),
                float(rec.get("area", 0.0)),
            ),
            reverse=True,
        )
        kept = 0
        for rec in ranked_rows:
            owner = _owner(list(rec.get("top_point_ids", [])), point_stats)
            if float(owner["owner_consistency"]) < float(args.min_owner_consistency):
                continue
            key = (image_name, direction, int(rec["row_index"]))
            if key in seen:
                continue
            seen.add(key)
            mode = _action_mode(direction, delta, args.mode_policy)
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
                "top_point_ids": list(rec.get("top_point_ids", []))[: max(int(args.max_top_ids), 0)],
                "top_ids_enable": mode in ("tight_top_ids", "tight_bbox"),
                "top_ids_only": mode == "tight_top_ids",
                "max_top_ids": int(args.max_top_ids),
                "pad_px": float(args.tight_bbox_pad_px),
                "ellipse_scale": float(args.tight_bbox_ellipse_scale),
                "reason": "v346_component_local_residual_asset",
                **owner,
                **{f"{metric}_delta_base": delta[metric] for metric in METRICS},
            }
            actions.append(action)
            audit_rows.append({
                "frame_score": frame["score"],
                **action,
            })
            kept += 1
            if int(args.components_per_frame) > 0 and kept >= int(args.components_per_frame):
                break
            if int(args.max_actions) > 0 and len(actions) >= int(args.max_actions):
                break
        if int(args.max_actions) > 0 and len(actions) >= int(args.max_actions):
            break

    row_guard_drops = [
        {
            "image_name": action["image_name"],
            "direction": action["direction"],
            "row_index": action["row_index"],
            "component_id": action["component_id"],
            "bbox_x": action["bbox_x"],
            "bbox_y": action["bbox_y"],
            "bbox_w": action["bbox_w"],
            "bbox_h": action["bbox_h"],
            "centroid_x": action["centroid_x"],
            "centroid_y": action["centroid_y"],
            "tol_px": 1.0,
            "reason": action["reason"],
        }
        for action in actions
    ]

    payload = {
        "version": "v346_component_local_residual_asset",
        "policy": (
            "Generate component-local actions from dense residual deltas. This excludes no-local-component "
            "screen-guard images and keeps actions tied to component rows and semantic owners."
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
        },
        "frame_count": len(frame_rows),
        "action_count": len(actions),
        "actions": actions,
    }
    row_guard_payload = {
        "version": "v346_component_local_row_guard_probe",
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

    print(f"wrote {args.out_json} actions={len(actions)} frames={len(frame_rows)}")
    print(f"wrote {args.out_row_guard_json} drops={len(row_guard_drops)}")
    print(f"wrote {args.out_candidates_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
