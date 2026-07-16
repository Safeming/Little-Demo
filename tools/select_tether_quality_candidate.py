#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DEFAULT_THRESHOLDS = {
    "same30_lpips_max": 0.1261183471,
    "same30_psnr_min": 21.9559345,
    "original57_lpips_max": 0.1288665946,
    "original57_psnr_min": 21.7641456,
    "edge_px_max": 2.90,
    "boundary_l1_max": 0.06720,
}


def select_continuation(candidates, lpips_tolerance=0.001):
    if not candidates:
        raise ValueError("at least one candidate is required")
    best_lpips = min(float(row["lpips_fg"]) for row in candidates)
    eligible = [
        row
        for row in candidates
        if float(row["lpips_fg"]) <= best_lpips + float(lpips_tolerance)
    ]
    return min(
        eligible,
        key=lambda row: (
            float(row["edge_px"]),
            float(row["boundary_l1"]),
            -float(row["psnr_fg"]),
            float(row["lpips_fg"]),
        ),
    )


def evaluate_final_candidate(*, same30, original57, contour, thresholds=None):
    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        limits.update(thresholds)
    gates = {
        "same30_lpips": float(same30["lpips_fg"]) <= limits["same30_lpips_max"],
        "same30_psnr": float(same30["psnr_fg"]) >= limits["same30_psnr_min"],
        "original57_lpips": float(original57["lpips_fg"])
        <= limits["original57_lpips_max"],
        "original57_psnr": float(original57["psnr_fg"])
        >= limits["original57_psnr_min"],
        "contour_edge": float(contour["edge_px"]) <= limits["edge_px_max"],
        "boundary_l1": float(contour["boundary_l1"])
        <= limits["boundary_l1_max"],
    }
    return {
        "accepted": all(gates.values()),
        "gates": gates,
        "thresholds": limits,
        "same30": dict(same30),
        "original57": dict(original57),
        "contour": dict(contour),
    }


def _read_json(path):
    return json.loads(Path(path).read_text())


def _metric_payload(path):
    payload = _read_json(path)
    return payload.get("best_eval") or payload.get("test") or payload


def _contour_payload(path):
    payload = _read_json(path)
    return {
        "edge_px": float(payload["mean_edge_symmetric_dist_px"]),
        "boundary_l1": float(payload["mean_boundary_l1"]),
    }


def _write_json(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select-continuation")
    select_parser.add_argument("--manifest", required=True)
    select_parser.add_argument("--output", required=True)
    select_parser.add_argument("--lpips-tolerance", type=float, default=0.001)

    gate_parser = subparsers.add_parser("evaluate-final")
    gate_parser.add_argument("--same30", required=True)
    gate_parser.add_argument("--original57", required=True)
    gate_parser.add_argument("--contour", required=True)
    gate_parser.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "select-continuation":
        manifest = _read_json(args.manifest)
        candidates = manifest.get("candidates", manifest)
        selected = select_continuation(candidates, args.lpips_tolerance)
        result = {"selected": selected, "candidates": candidates}
    else:
        result = evaluate_final_candidate(
            same30=_metric_payload(args.same30),
            original57=_metric_payload(args.original57),
            contour=_contour_payload(args.contour),
        )
    _write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
