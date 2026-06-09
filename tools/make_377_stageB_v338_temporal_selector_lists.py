#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = {
    "fg": "fg_l1",
    "boundary": "boundary_l1",
    "edge": "edge_symmetric_dist_px",
    "inner": "inner_missing_pixels",
    "outer": "outer_leak_pixels",
    "hard": "hard_residual_score",
    "opacity_inner": "primary_opacity_inner_missing_pixels",
    "opacity_outer": "primary_opacity_outer_leak_pixels",
}


def _key(row: dict[str, str]) -> str:
    return f"c{int(float(row.get('cam', 0))):02d}_f{int(float(row.get('frame', 0))):06d}"


def _load_records(render_exp: Path) -> dict[str, dict[str, float]]:
    records: dict[str, dict[str, float]] = {}
    paths = [
        render_exp / "diagnostics/contours/contour_samples.csv",
        render_exp / "diagnostics/boundary_residuals/boundary_residual_samples.csv",
        render_exp / "diagnostics/opacity_footprint/opacity_footprint_samples.csv",
    ]
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rec = records.setdefault(_key(row), {})
                for key, value in row.items():
                    if value is None or value == "":
                        continue
                    try:
                        rec[key] = float(value)
                    except ValueError:
                        pass
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v338 drop/grow-only frame lists from v337 per-frame metrics.")
    parser.add_argument("--baseline-exp", required=True, type=Path)
    parser.add_argument("--formal-exp", required=True, type=Path)
    parser.add_argument("--temporal-exp", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--strict-vs", choices=("baseline", "formal"), default="formal")
    parser.add_argument("--drop-opacity-inner-delta", type=float, default=0.0)
    parser.add_argument("--drop-inner-delta", type=float, default=0.0)
    parser.add_argument("--drop-outer-delta", type=float, default=0.0)
    parser.add_argument("--drop-hard-delta", type=float, default=0.0)
    parser.add_argument("--drop-fg-delta", type=float, default=0.0)
    parser.add_argument("--drop-boundary-delta", type=float, default=0.0)
    parser.add_argument("--drop-edge-delta", type=float, default=0.0)
    parser.add_argument("--grow-only-opacity-inner-delta", type=float, default=0.0)
    parser.add_argument("--grow-only-inner-delta", type=float, default=0.0)
    parser.add_argument("--grow-only-hard-delta", type=float, default=0.0)
    args = parser.parse_args()

    baseline = _load_records(args.baseline_exp)
    formal = _load_records(args.formal_exp)
    temporal = _load_records(args.temporal_exp)
    reference = baseline if args.strict_vs == "baseline" else formal

    drop: set[str] = set()
    grow_only: set[str] = set()
    reasons: dict[str, list[str]] = {}
    improve_counts = {"temporal_better_outer": 0, "temporal_better_opacity_outer": 0, "temporal_better_hard": 0}
    for key, temp in temporal.items():
        ref = reference.get(key)
        base = baseline.get(key)
        if ref is None or base is None:
            continue
        deltas = {name: temp[col] - ref[col] for name, col in METRICS.items() if col in temp and col in ref}
        formal_delta = {}
        if key in formal:
            formal_delta = {name: temp[col] - formal[key][col] for name, col in METRICS.items() if col in temp and col in formal[key]}
            if formal_delta.get("outer", 0.0) < 0:
                improve_counts["temporal_better_outer"] += 1
            if formal_delta.get("opacity_outer", 0.0) < 0:
                improve_counts["temporal_better_opacity_outer"] += 1
            if formal_delta.get("hard", 0.0) < 0:
                improve_counts["temporal_better_hard"] += 1

        frame_reasons = []
        checks = [
            ("opacity_inner", args.drop_opacity_inner_delta),
            ("inner", args.drop_inner_delta),
            ("outer", args.drop_outer_delta),
            ("hard", args.drop_hard_delta),
            ("fg", args.drop_fg_delta),
            ("boundary", args.drop_boundary_delta),
            ("edge", args.drop_edge_delta),
        ]
        for name, threshold in checks:
            if name in deltas and deltas[name] > float(threshold):
                frame_reasons.append(f"{name}_delta={deltas[name]:.8f}>{threshold}")
        if frame_reasons:
            drop.add(key)
            reasons[key] = frame_reasons
            continue

        grow_reasons = []
        grow_checks = [
            ("opacity_inner", args.grow_only_opacity_inner_delta),
            ("inner", args.grow_only_inner_delta),
            ("hard", args.grow_only_hard_delta),
        ]
        for name, threshold in grow_checks:
            if name in deltas and deltas[name] > float(threshold):
                grow_reasons.append(f"{name}_delta={deltas[name]:.8f}>{threshold}")
        if grow_reasons:
            grow_only.add(key)
            reasons[key] = grow_reasons

    payload = {
        "strict_vs": args.strict_vs,
        "drop_images": sorted(drop),
        "grow_only_images": sorted(grow_only - drop),
        "counts": {
            "total_temporal_records": len(temporal),
            "drop": len(drop),
            "grow_only": len(grow_only - drop),
            **improve_counts,
        },
        "reasons": {key: reasons[key] for key in sorted(reasons)[:500]},
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2, sort_keys=True))
    print("drop_images=" + ",".join(payload["drop_images"]))
    print("grow_only_images=" + ",".join(payload["grow_only_images"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
