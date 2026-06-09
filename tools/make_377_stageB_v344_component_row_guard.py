#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def _parse_targets(text):
    targets = []
    for item in str(text or "").split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 1:
            image_name, direction, topk = parts[0], "inner", 1
        elif len(parts) == 2:
            image_name, direction = parts
            topk = 1
        else:
            image_name, direction, topk = parts[:3]
        try:
            topk = int(topk)
        except ValueError:
            topk = 1
        targets.append((image_name.strip(), direction.strip().lower(), max(topk, 0)))
    return targets


def _score(row, mode):
    area = float(row.get("area", 0.0) or 0.0)
    near = float(row.get("near_score_sum", 0.0) or 0.0)
    if mode == "near":
        return near
    if mode == "area":
        return area
    return area * max(near, 1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate v344 component-row guard JSON from adopted component CSV."
    )
    parser.add_argument(
        "--component-csv",
        default="assets/adopted_geometry/377/v320_selected_components.csv",
    )
    parser.add_argument(
        "--targets",
        required=True,
        help="Semicolon list: image[:direction[:topk]], e.g. c21_f000480:inner:1;c22_f000240:inner:2",
    )
    parser.add_argument("--rank", choices=("area_near", "area", "near"), default="area_near")
    parser.add_argument("--reason", default="v344_component_row_guard_probe")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    component_csv = Path(args.component_csv)
    rows = []
    with component_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            row = dict(row)
            row["row_index"] = row_index
            rows.append(row)

    drops = []
    for image_name, direction, topk in _parse_targets(args.targets):
        if topk <= 0:
            continue
        candidates = [
            row for row in rows
            if str(row.get("image_name", "")).strip() == image_name
            and str(row.get("direction", "")).strip().lower() == direction
        ]
        candidates.sort(key=lambda row: _score(row, args.rank), reverse=True)
        for row in candidates[:topk]:
            drops.append({
                "image_name": image_name,
                "direction": direction,
                "row_index": int(row["row_index"]),
                "component_id": int(float(row.get("component_id", -1) or -1)),
                "area": float(row.get("area", 0.0) or 0.0),
                "bbox_x": float(row.get("bbox_x", 0.0) or 0.0),
                "bbox_y": float(row.get("bbox_y", 0.0) or 0.0),
                "bbox_w": float(row.get("bbox_w", 0.0) or 0.0),
                "bbox_h": float(row.get("bbox_h", 0.0) or 0.0),
                "centroid_x": float(row.get("centroid_x", 0.0) or 0.0),
                "centroid_y": float(row.get("centroid_y", 0.0) or 0.0),
                "near_score_sum": float(row.get("near_score_sum", 0.0) or 0.0),
                "tol_px": 1.0,
                "reason": args.reason,
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v344_component_row_guard",
        "component_csv": str(component_csv),
        "rank": args.rank,
        "targets": [
            {"image_name": image, "direction": direction, "topk": topk}
            for image, direction, topk in _parse_targets(args.targets)
        ],
        "drops": drops,
        "drop_count": len(drops),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out} drops={len(drops)}")


if __name__ == "__main__":
    main()
