#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.assess_voting_posterior_candidate import assess_candidate
from utils.semantic_eval_protocol import load_protocol, protocol_fingerprint


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def select_guarded_candidate(candidates: list[dict], *, max_miou_gap: float = 0.02) -> dict:
    eligible = []
    for candidate in candidates:
        miou_gap = float(candidate["b1_macro_miou"]) - float(candidate["b5_macro_miou"])
        if miou_gap > float(max_miou_gap) + 1.0e-12:
            continue
        checks = list(candidate.get("retention_checks", []))
        if not checks or any(
            float(row["b5_actionable_leakage"]) > float(row["b1_actionable_leakage"]) + 1.0e-12
            for row in checks
        ):
            continue
        eligible.append(dict(candidate))
    if not eligible:
        raise ValueError("no guarded validation candidate satisfies the mIoU and leakage gates")
    eligible.sort(
        key=lambda row: (
            sum(float(item["b5_actionable_leakage"]) for item in row["retention_checks"])
            / len(row["retention_checks"]),
            -float(row["b5_macro_miou"]),
            -float(row.get("b5_mean_boundary_f1", 0.0)),
            float(row["soft_threshold"]),
            float(row.get("support_threshold", 0.0)),
            int(row.get("boundary_radius", 0)),
        )
    )
    selected = eligible[0]
    selected["miou_gap"] = float(selected["b1_macro_miou"]) - float(selected["b5_macro_miou"])
    selected["mean_b5_actionable_leakage"] = sum(
        float(item["b5_actionable_leakage"]) for item in selected["retention_checks"]
    ) / len(selected["retention_checks"])
    return selected


def derive_fallback_parts(b5_rows: list[dict], b3_rows: list[dict]) -> list[str]:
    b3_by_part = {str(row["part"]): row for row in b3_rows}
    fallback = []
    for row in b5_rows:
        part = str(row["part"])
        b3 = b3_by_part.get(part)
        if b3 is None:
            continue
        if (
            float(row.get("target", 0.0)) > 0.0
            and float(row.get("predicted", 0.0)) <= 0.0
            and float(b3.get("predicted", 0.0)) > 0.0
        ):
            fallback.append(part)
    return sorted(fallback)


def write_guarded_config(
    path: Path | str,
    *,
    candidate: dict,
    fallback_parts: list[str],
    fallback_threshold: float,
    protocol_name: str,
    protocol_fingerprint: str,
    checkpoint_fingerprint: str,
    bank_fingerprint: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = dict(candidate)
    selected["b5_fallback_parts"] = list(fallback_parts)
    selected["b5_fallback_threshold"] = float(fallback_threshold)
    payload = {
        "protocol_name": str(protocol_name),
        "protocol_fingerprint": str(protocol_fingerprint),
        "checkpoint_fingerprint": str(checkpoint_fingerprint),
        "bank_fingerprint": str(bank_fingerprint),
        "selection_objective": [
            "require_b5_miou_gap_at_most_maximum",
            "require_b5_leakage_not_above_b1_at_required_retentions",
            "min_mean_b5_actionable_leakage",
            "max_b5_macro_miou",
            "max_b5_mean_boundary_f1",
        ],
        "selected": selected,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_candidate_report(
    report_dir: Path,
    *,
    soft_threshold: float,
    support_threshold: float,
    boundary_radius: int,
    required_retentions: tuple[float, ...],
    max_miou_gap: float,
) -> tuple[dict, list[dict], list[dict]]:
    baseline_rows = _read_csv(report_dir / "baseline_summary.csv")
    curve_rows = _read_csv(report_dir / "leakage_retention_curve.csv")
    per_part_rows = _read_csv(report_dir / "per_part_metrics.csv")
    assessment = assess_candidate(
        baseline_rows,
        curve_rows,
        required_retentions=required_retentions,
        max_miou_gap=max_miou_gap,
    )
    baseline_by_name = {str(row["baseline"]): row for row in baseline_rows}
    candidate = {
        "soft_threshold": float(soft_threshold),
        "support_threshold": float(support_threshold),
        "boundary_radius": int(boundary_radius),
        "b1_macro_miou": float(assessment["b1_macro_miou"]),
        "b5_macro_miou": float(assessment["b5_macro_miou"]),
        "b5_mean_boundary_f1": float(baseline_by_name["B5"]["mean_boundary_f1"]),
        "retention_checks": list(assessment["retention_checks"]),
        "report_dir": str(report_dir.resolve()),
    }
    b5_rows = [row for row in per_part_rows if str(row.get("baseline")) == "B5"]
    b3_rows = [row for row in per_part_rows if str(row.get("baseline")) == "B3"]
    return candidate, b5_rows, b3_rows


def _candidate_spec(value: str) -> tuple[float, Path]:
    threshold, separator, path = str(value).partition(":")
    if not separator or not path:
        raise argparse.ArgumentTypeError("candidate must be SOFT_THRESHOLD:REPORT_DIR")
    return float(threshold), Path(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a validation-gated semantic configuration.")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--candidate", action="append", required=True, type=_candidate_spec)
    parser.add_argument("--b3-per-part-metrics", type=Path, default=None)
    parser.add_argument("--checkpoint-fingerprint", required=True)
    parser.add_argument("--bank-fingerprint", required=True)
    parser.add_argument("--required-retention", nargs="+", type=float, default=[0.5, 0.6])
    parser.add_argument("--max-miou-gap", type=float, default=0.02)
    parser.add_argument("--support-threshold", type=float, default=0.1)
    parser.add_argument("--boundary-radius", type=int, default=6)
    parser.add_argument("--b5-fallback-threshold", type=float, default=0.5)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    candidates = []
    parts_by_threshold = {}
    fallback_b3_rows = []
    if args.b3_per_part_metrics is not None:
        fallback_b3_rows = [
            row
            for row in _read_csv(args.b3_per_part_metrics)
            if str(row.get("baseline")) == "B3"
        ]
    for threshold, report_dir in args.candidate:
        candidate, b5_rows, b3_rows = load_candidate_report(
            report_dir,
            soft_threshold=threshold,
            support_threshold=float(args.support_threshold),
            boundary_radius=int(args.boundary_radius),
            required_retentions=tuple(float(value) for value in args.required_retention),
            max_miou_gap=float(args.max_miou_gap),
        )
        candidates.append(candidate)
        parts_by_threshold[float(threshold)] = (b5_rows, b3_rows or fallback_b3_rows)
    selected = select_guarded_candidate(candidates, max_miou_gap=float(args.max_miou_gap))
    b5_rows, b3_rows = parts_by_threshold[float(selected["soft_threshold"])]
    fallback_parts = derive_fallback_parts(b5_rows, b3_rows)
    output = write_guarded_config(
        args.output,
        candidate=selected,
        fallback_parts=fallback_parts,
        fallback_threshold=float(args.b5_fallback_threshold),
        protocol_name=protocol["protocol_name"],
        protocol_fingerprint=protocol_fingerprint(protocol),
        checkpoint_fingerprint=args.checkpoint_fingerprint,
        bank_fingerprint=args.bank_fingerprint,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
