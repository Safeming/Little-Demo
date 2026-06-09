#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


BASELINE = {
    "mean_inner_missing_pixels": 361.23,
    "mean_outer_leak_pixels": 1203.83,
    "mean_fg_l1": 0.04460,
    "mean_boundary_l1": 0.06626,
    "mean_edge_symmetric_dist_px": 3.1613,
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_baseline(path: Path | None) -> dict:
    if path is None:
        return dict(BASELINE)
    data = _read_json(path)
    baseline = dict(BASELINE)
    for key in baseline:
        if key in data and data[key] is not None:
            baseline[key] = float(data[key])
    return baseline


def _get(summary: dict, key: str, default=None):
    value = summary.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description="v261 do-no-harm gate against fixed v233d baseline.")
    parser.add_argument("--candidate-summary", type=Path, default=None)
    parser.add_argument("--residual-summary", type=Path, default=None)
    parser.add_argument("--contour-summary", type=Path, default=None)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument("--require-candidate-ok", action="store_true")
    parser.add_argument("--inner-epsilon", type=float, default=0.0)
    parser.add_argument("--outer-tolerance", type=float, default=0.0)
    parser.add_argument("--rgb-tolerance", type=float, default=0.0)
    parser.add_argument("--edge-tolerance", type=float, default=0.0)
    args = parser.parse_args()
    baseline = _load_baseline(args.baseline_json)

    failures = []
    report = {
        "baseline": baseline,
        "status": "ok",
        "failures": failures,
    }

    if args.candidate_summary is not None:
        candidate = _read_json(args.candidate_summary)
        report["candidate"] = {
            "path": str(args.candidate_summary),
            "status": candidate.get("status"),
            "accepted_candidate_count": candidate.get("accepted_candidate_count"),
            "inner_dominant_candidate_count": candidate.get("inner_dominant_candidate_count"),
            "mean_projected_candidates_on_inner_points": candidate.get("mean_projected_candidates_on_inner_points"),
            "mean_projected_candidates_on_outer_points": candidate.get("mean_projected_candidates_on_outer_points"),
            "projected_inner_outer_ratio": candidate.get("projected_inner_outer_ratio"),
            "reasons": candidate.get("reasons", []),
        }
        if args.require_candidate_ok and candidate.get("status") != "ok":
            failures.append("candidate_validator_blocked")

    if args.residual_summary is not None:
        residual = _read_json(args.residual_summary)
        inner = _get(residual, "mean_inner_missing_pixels")
        outer = _get(residual, "mean_outer_leak_pixels")
        report["residual"] = {
            "path": str(args.residual_summary),
            "mean_inner_missing_pixels": inner,
            "mean_outer_leak_pixels": outer,
        }
        if inner is None:
            failures.append("missing_inner_metric")
        elif inner > baseline["mean_inner_missing_pixels"] - float(args.inner_epsilon):
            failures.append("inner_missing_not_improved_vs_v233d")
        if outer is None:
            failures.append("missing_outer_metric")
        elif outer > baseline["mean_outer_leak_pixels"] + float(args.outer_tolerance):
            failures.append("outer_leak_worse_than_v233d")

    if args.contour_summary is not None:
        contour = _read_json(args.contour_summary)
        fg_l1 = _get(contour, "mean_fg_l1")
        boundary_l1 = _get(contour, "mean_boundary_l1")
        edge = _get(contour, "mean_edge_symmetric_dist_px")
        report["contour"] = {
            "path": str(args.contour_summary),
            "mean_fg_l1": fg_l1,
            "mean_boundary_l1": boundary_l1,
            "mean_edge_symmetric_dist_px": edge,
        }
        rgb_tol = float(args.rgb_tolerance)
        edge_tol = float(args.edge_tolerance)
        if fg_l1 is None:
            failures.append("missing_fg_l1")
        elif fg_l1 > baseline["mean_fg_l1"] + rgb_tol:
            failures.append("fg_l1_worse_than_v233d")
        if boundary_l1 is None:
            failures.append("missing_boundary_l1")
        elif boundary_l1 > baseline["mean_boundary_l1"] + rgb_tol:
            failures.append("boundary_l1_worse_than_v233d")
        if edge is None:
            failures.append("missing_edge_dist")
        elif edge > baseline["mean_edge_symmetric_dist_px"] + edge_tol:
            failures.append("edge_dist_worse_than_v233d")

    if failures:
        report["status"] = "blocked"
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
