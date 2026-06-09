#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default) or default))
    except (TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description="Build signed point prior JSON from contributor audit CSV.")
    parser.add_argument("--point-csv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--max-shrink", type=int, default=96)
    parser.add_argument("--max-grow", type=int, default=96)
    parser.add_argument("--min-abs-signed", type=float, default=0.0)
    parser.add_argument("--min-hit-gap", type=int, default=0)
    args = parser.parse_args()

    rows = []
    with args.point_csv.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            point_idx = _int(row, "point_idx", -1)
            if point_idx < 0:
                continue
            over_hits = _int(row, "over_frame_hits", 0)
            under_hits = _int(row, "under_frame_hits", 0)
            signed = _float(row, "signed_score", _float(row, "over_score_sum") - _float(row, "under_score_sum"))
            rows.append({
                "point_idx": point_idx,
                "signed_score": signed,
                "over_score_sum": _float(row, "over_score_sum"),
                "under_score_sum": _float(row, "under_score_sum"),
                "over_frame_hits": over_hits,
                "under_frame_hits": under_hits,
                "hit_gap": over_hits - under_hits,
                "layer_id": _int(row, "layer_id", -1),
                "region_id": _int(row, "region_id", -1),
                "dominant_joint": _int(row, "dominant_joint", -1),
                "boundary_score": _float(row, "boundary_score"),
                "surface_distance": _float(row, "surface_distance"),
                "thin_score": _float(row, "thin_score"),
            })

    min_abs = float(args.min_abs_signed)
    min_hit_gap = int(args.min_hit_gap)
    shrink_candidates = [
        row for row in rows
        if row["signed_score"] > min_abs and row["hit_gap"] >= min_hit_gap
    ]
    grow_candidates = [
        row for row in rows
        if row["signed_score"] < -min_abs and row["hit_gap"] <= -min_hit_gap
    ]
    shrink_candidates.sort(key=lambda r: (r["signed_score"], r["over_score_sum"], r["over_frame_hits"]), reverse=True)
    grow_candidates.sort(key=lambda r: (abs(r["signed_score"]), r["under_score_sum"], r["under_frame_hits"]), reverse=True)

    shrink = shrink_candidates[: max(int(args.max_shrink), 0)]
    grow = grow_candidates[: max(int(args.max_grow), 0)]
    payload = {
        "source_point_csv": str(args.point_csv),
        "selection": {
            "max_shrink": int(args.max_shrink),
            "max_grow": int(args.max_grow),
            "min_abs_signed": min_abs,
            "min_hit_gap": min_hit_gap,
        },
        "shrink_point_ids": [int(row["point_idx"]) for row in shrink],
        "grow_point_ids": [int(row["point_idx"]) for row in grow],
        "shrink_records": shrink,
        "grow_records": grow,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_json": str(args.out_json),
        "rows": len(rows),
        "shrink_count": len(shrink),
        "grow_count": len(grow),
        "top_shrink": payload["shrink_point_ids"][:8],
        "top_grow": payload["grow_point_ids"][:8],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
