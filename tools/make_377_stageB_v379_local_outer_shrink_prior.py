#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _ids(value) -> list[int]:
    if isinstance(value, list):
        src = value
    else:
        src = str(value or "").replace(";", ",").replace("[", "").replace("]", "").split(",")
    out = []
    for item in src:
        try:
            out.append(int(float(item)))
        except Exception:
            pass
    return out


def _bad_images(path: Path, max_images: int) -> set[str]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if not str(row.get("variant", "")).startswith("candidate"):
                continue
            try:
                outer = float(row.get("outer_delta", 0.0) or 0.0)
                opacity_outer = float(row.get("opacity_outer_delta", 0.0) or 0.0)
                worsen = float(row.get("worsen_score", 0.0) or 0.0)
            except ValueError:
                continue
            if outer > 0.0 or opacity_outer > 0.0:
                rows.append((worsen + max(outer, 0.0) + max(opacity_outer, 0.0), row.get("image", "")))
    rows.sort(reverse=True)
    return {image for _, image in rows[:max_images] if image}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local outer-shrink signed point prior from generic worst-frame asset actions.")
    parser.add_argument("--asset-json", required=True, type=Path)
    parser.add_argument("--worst-frames-tsv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--max-points", type=int, default=96)
    args = parser.parse_args()

    asset = json.loads(args.asset_json.read_text(encoding="utf-8"))
    bad = _bad_images(args.worst_frames_tsv, int(args.max_images))
    scored: dict[int, float] = {}
    records = []
    for action in asset.get("actions", []):
        image = str(action.get("source_image_name", action.get("image_name", "")) or "")
        direction = str(action.get("direction", "") or "").lower()
        if image not in bad or direction not in ("outer", "over"):
            continue
        ids = _ids(action.get("anchor_point_ids") or action.get("top_point_ids") or action.get("source_top_point_ids"))
        for rank, point_id in enumerate(ids):
            score = float(len(ids) - rank)
            scored[point_id] = max(scored.get(point_id, 0.0), score)
            records.append({
                "point_idx": int(point_id),
                "score": score,
                "image_name": image,
                "pair_id": action.get("pair_id", ""),
                "component_key": action.get("component_key", ""),
            })
    selected = sorted(scored, key=lambda point_id: scored[point_id], reverse=True)[: max(0, int(args.max_points))]
    payload = {
        "source_asset_json": str(args.asset_json),
        "source_worst_frames_tsv": str(args.worst_frames_tsv),
        "bad_images": sorted(bad),
        "selection": {"max_images": int(args.max_images), "max_points": int(args.max_points)},
        "shrink_point_ids": [int(x) for x in selected],
        "grow_point_ids": [],
        "shrink_records": [r for r in records if int(r["point_idx"]) in set(selected)],
        "grow_records": [],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "bad_images": len(bad), "shrink_count": len(selected), "top": selected[:8]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
