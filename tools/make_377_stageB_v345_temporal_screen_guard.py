#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _image_name(row: dict) -> str:
    return f"c{int(float(row.get('cam', 0))):02d}_f{int(float(row.get('frame', 0))):06d}"


def _load_opacity(path: Path) -> dict[str, dict[str, float]]:
    out = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out[_image_name(row)] = {
                "opacity_outer": float(row["primary_opacity_outer_leak_pixels"]),
                "opacity_inner": float(row["primary_opacity_inner_missing_pixels"]),
                "rgb_outer": float(row["rgb_outer_leak_pixels"]),
                "rgb_inner": float(row["rgb_inner_missing_pixels"]),
            }
    return out


def _component_counts(path: Path) -> dict[str, dict[str, int]]:
    counts = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = str(row.get("image_name", "")).strip()
            direction = str(row.get("direction", "")).strip().lower()
            if not name or direction not in ("inner", "outer"):
                continue
            item = counts.setdefault(name, {"inner": 0, "outer": 0})
            item[direction] += 1
    return counts


def _signed_stats(path: Path) -> dict[str, dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_image = data.get("by_image", {}) if isinstance(data, dict) else {}
    out = {}
    for name, item in by_image.items():
        if not isinstance(item, dict):
            continue
        shrink = item.get("shrink_point_ids", []) or []
        grow = item.get("grow_point_ids", []) or []
        out[name] = {
            "shrink": len(shrink),
            "grow": len(grow),
            "source": item.get("source_image_name", ""),
            "distance": item.get("temporal_distance", ""),
            "mode": item.get("temporal_mode", ""),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate v345 point-screen drop list for temporal signed-point frames without local components."
    )
    parser.add_argument("--baseline-opacity-samples", required=True, type=Path)
    parser.add_argument("--candidate-opacity-samples", required=True, type=Path)
    parser.add_argument("--component-csv", default="assets/adopted_geometry/377/v320_selected_components.csv", type=Path)
    parser.add_argument("--signed-point-json", default="assets/adopted_geometry/377/v338_temporal_selector_grow_only_guard.json", type=Path)
    parser.add_argument("--min-opacity-outer-delta", default=1.0, type=float)
    parser.add_argument("--require-no-local-components", action="store_true", default=True)
    parser.add_argument("--include-self-source", action="store_true")
    parser.add_argument(
        "--seed-drop-images",
        default="",
        help="Comma/semicolon separated images from prior accepted point-screen guards to keep in the output.",
    )
    parser.add_argument("--topk", default=-1, type=int)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-list", required=True, type=Path)
    args = parser.parse_args()

    baseline = _load_opacity(args.baseline_opacity_samples)
    candidate = _load_opacity(args.candidate_opacity_samples)
    components = _component_counts(args.component_csv)
    signed = _signed_stats(args.signed_point_json)

    rows = []
    for image_name, cand in candidate.items():
        base = baseline.get(image_name)
        stat = signed.get(image_name)
        if base is None or stat is None:
            continue
        delta = cand["opacity_outer"] - base["opacity_outer"]
        if delta < float(args.min_opacity_outer_delta):
            continue
        comp = components.get(image_name, {"inner": 0, "outer": 0})
        if bool(args.require_no_local_components) and (comp["inner"] + comp["outer"]) > 0:
            continue
        if int(stat["shrink"]) + int(stat["grow"]) <= 0:
            continue
        source = str(stat.get("source", "") or "")
        if not args.include_self_source and source == image_name:
            continue
        rows.append({
            "image_name": image_name,
            "opacity_outer_delta": delta,
            "opacity_outer_baseline": base["opacity_outer"],
            "opacity_outer_candidate": cand["opacity_outer"],
            "opacity_inner_delta": cand["opacity_inner"] - base["opacity_inner"],
            "rgb_outer_delta": cand["rgb_outer"] - base["rgb_outer"],
            "rgb_inner_delta": cand["rgb_inner"] - base["rgb_inner"],
            "inner_component_rows": comp["inner"],
            "outer_component_rows": comp["outer"],
            "signed_shrink_points": stat["shrink"],
            "signed_grow_points": stat["grow"],
            "temporal_source": source,
            "temporal_distance": stat.get("distance", ""),
            "temporal_mode": stat.get("mode", ""),
            "reason": "v345_temporal_no_component_screen_guard",
        })
    rows.sort(key=lambda item: (float(item["opacity_outer_delta"]), item["image_name"]), reverse=True)
    if int(args.topk) > 0:
        rows = rows[: int(args.topk)]

    existing = {row["image_name"] for row in rows}
    for token in str(args.seed_drop_images or "").replace(";", ",").split(","):
        image_name = token.strip()
        if not image_name or image_name in existing:
            continue
        stat = signed.get(image_name, {})
        comp = components.get(image_name, {"inner": 0, "outer": 0})
        base = baseline.get(image_name, {})
        cand = candidate.get(image_name, {})
        rows.append({
            "image_name": image_name,
            "opacity_outer_delta": float(cand.get("opacity_outer", 0.0)) - float(base.get("opacity_outer", 0.0)),
            "opacity_outer_baseline": base.get("opacity_outer", ""),
            "opacity_outer_candidate": cand.get("opacity_outer", ""),
            "opacity_inner_delta": float(cand.get("opacity_inner", 0.0)) - float(base.get("opacity_inner", 0.0)),
            "rgb_outer_delta": float(cand.get("rgb_outer", 0.0)) - float(base.get("rgb_outer", 0.0)),
            "rgb_inner_delta": float(cand.get("rgb_inner", 0.0)) - float(base.get("rgb_inner", 0.0)),
            "inner_component_rows": comp["inner"],
            "outer_component_rows": comp["outer"],
            "signed_shrink_points": stat.get("shrink", ""),
            "signed_grow_points": stat.get("grow", ""),
            "temporal_source": stat.get("source", ""),
            "temporal_distance": stat.get("distance", ""),
            "temporal_mode": stat.get("mode", ""),
            "reason": "v345_seed_point_screen_guard",
        })
        existing.add(image_name)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v345_temporal_no_component_screen_guard",
        "policy": (
            "Drop point screen actuator only for temporal signed-point frames with positive opacity_outer "
            "delta and no local adopted component rows. The signed point JSON remains enabled."
        ),
        "min_opacity_outer_delta": float(args.min_opacity_outer_delta),
        "drop_count": len(rows),
        "drop_images": [row["image_name"] for row in rows],
        "drops": rows,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_list.write_text(",".join(row["image_name"] for row in rows), encoding="utf-8")
    print(f"wrote {args.out_json} drops={len(rows)}")
    print(f"wrote {args.out_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
