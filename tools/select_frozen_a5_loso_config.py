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

from utils.frozen_semantic_method import load_frozen_semantic_method
from utils.semantic_eval_protocol import load_protocol, protocol_fingerprint


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _row_at_retention(rows: list[dict], *, method: str, retention: float) -> dict:
    matches = [
        row
        for row in rows
        if str(row.get("baseline")) == str(method)
        and abs(float(row.get("retention", -1.0)) - float(retention)) <= 1.0e-6
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {method} row at retention {retention}, found {len(matches)}"
        )
    return matches[0]


def _optional_row_at_retention(
    rows: list[dict], *, method: str, retention: float
) -> dict | None:
    matches = [
        row
        for row in rows
        if str(row.get("baseline")) == str(method)
        and abs(float(row.get("retention", -1.0)) - float(retention)) <= 1.0e-6
    ]
    if len(matches) > 1:
        raise ValueError(
            f"expected at most one {method} row at retention {retention}, found {len(matches)}"
        )
    return matches[0] if matches else None


def load_a5_candidate_report(
    report_dir: Path,
    *,
    donor_subject: str,
    soft_threshold: float,
    required_retentions: tuple[float, ...],
) -> dict:
    report_dir = Path(report_dir)
    baseline_rows = _read_csv(report_dir / "baseline_summary.csv")
    baseline_by_name = {str(row["baseline"]): row for row in baseline_rows}
    missing = sorted({"B1", "A5"} - set(baseline_by_name))
    if missing:
        raise ValueError(f"A5 LOSO report is missing baselines: {missing}")
    if "B5" in baseline_by_name and "A5" not in baseline_by_name:
        raise ValueError("historical B5 report cannot select the frozen A5 method")

    b1_curve = _read_csv(report_dir / "leakage_retention_curve.csv")
    a5_matched = _read_csv(report_dir / "matched_retention.csv")
    checks = []
    coverage_complete = True
    for retention in required_retentions:
        b1 = _optional_row_at_retention(b1_curve, method="B1", retention=retention)
        a5 = _optional_row_at_retention(a5_matched, method="A5", retention=retention)
        if b1 is None or a5 is None:
            coverage_complete = False
            continue
        checks.append(
            {
                "retention": float(retention),
                "b1_actionable_leakage": float(b1["actionable_leakage"]),
                "a5_actionable_leakage": float(a5["actionable_leakage"]),
                "b1_raw_leakage": float(b1["raw_leakage"]),
                "a5_raw_leakage": float(a5["raw_leakage"]),
            }
        )
    return {
        "donor_subject": str(donor_subject),
        "soft_threshold": float(soft_threshold),
        "coverage_complete": coverage_complete,
        "report_dir": str(report_dir.resolve()),
        "b1_macro_miou": float(baseline_by_name["B1"]["macro_miou"]),
        "a5_macro_miou": float(baseline_by_name["A5"]["macro_miou"]),
        "a5_mean_boundary_f1": float(
            baseline_by_name["A5"]["mean_boundary_f1"]
        ),
        "retention_checks": checks,
    }


def _candidate_passes(candidate: dict, *, max_miou_gap: float) -> bool:
    if not bool(candidate.get("coverage_complete", True)):
        return False
    gap = float(candidate["b1_macro_miou"]) - float(candidate["a5_macro_miou"])
    checks = list(candidate.get("retention_checks", []))
    return gap <= float(max_miou_gap) + 1.0e-12 and bool(checks) and all(
        float(row["a5_actionable_leakage"])
        <= float(row["b1_actionable_leakage"]) + 1.0e-12
        for row in checks
    )


def select_a5_loso_candidate(
    reports_by_threshold: dict[float, list[dict]],
    *,
    expected_donor_count: int = 4,
    max_miou_gap: float = 0.02,
) -> dict:
    if not reports_by_threshold:
        raise ValueError("at least one A5 LOSO threshold candidate is required")
    donor_sets = [
        {str(row["donor_subject"]) for row in reports}
        for reports in reports_by_threshold.values()
    ]
    if any(len(values) != int(expected_donor_count) for values in donor_sets):
        raise ValueError(f"every threshold requires exactly {expected_donor_count} unique donor subjects")
    if any(values != donor_sets[0] for values in donor_sets[1:]):
        raise ValueError("every A5 LOSO threshold must contain the same donor subjects")

    eligible = []
    for threshold, reports in reports_by_threshold.items():
        if len(reports) != int(expected_donor_count):
            raise ValueError(f"every threshold requires exactly {expected_donor_count} unique donor subjects")
        if not all(_candidate_passes(row, max_miou_gap=max_miou_gap) for row in reports):
            continue
        leakage_values = [
            float(check["a5_actionable_leakage"])
            for row in reports
            for check in row["retention_checks"]
        ]
        eligible.append(
            {
                "soft_threshold": float(threshold),
                "support_threshold": 0.1,
                "boundary_radius": 6,
                "donor_subjects": sorted(donor_sets[0]),
                "mean_a5_actionable_leakage": sum(leakage_values)
                / len(leakage_values),
                "mean_a5_macro_miou": sum(
                    float(row["a5_macro_miou"]) for row in reports
                )
                / len(reports),
                "mean_a5_boundary_f1": sum(
                    float(row["a5_mean_boundary_f1"]) for row in reports
                )
                / len(reports),
                "donor_reports": [dict(row) for row in reports],
            }
        )
    if not eligible:
        raise ValueError("no A5 LOSO threshold satisfies every donor mIoU and leakage gate")
    eligible.sort(
        key=lambda row: (
            float(row["mean_a5_actionable_leakage"]),
            -float(row["mean_a5_macro_miou"]),
            -float(row["mean_a5_boundary_f1"]),
            float(row["soft_threshold"]),
        )
    )
    return eligible[0]


def write_a5_loso_config(
    path: Path,
    *,
    selected: dict,
    held_out_subject: str,
    protocol: dict,
    checkpoint_fingerprint: str,
    bank_fingerprint: str,
    footprint_bank_fingerprint: str,
    method_freeze: dict,
) -> Path:
    threshold = float(selected["soft_threshold"])
    payload = {
        "protocol_name": str(protocol["protocol_name"]),
        "protocol_fingerprint": protocol_fingerprint(protocol),
        "checkpoint_fingerprint": str(checkpoint_fingerprint),
        "bank_fingerprint": str(bank_fingerprint),
        "footprint_bank_fingerprint": str(footprint_bank_fingerprint),
        "method_freeze_id": str(method_freeze["freeze_id"]),
        "method_freeze_fingerprint": str(method_freeze["_fingerprint"]),
        "selection_mode": "leave_one_subject_out_a5",
        "candidate_method": "A5",
        "held_out_subject": str(held_out_subject),
        "donor_subjects": list(selected["donor_subjects"]),
        "selection_objective": [
            "require_every_donor_a5_miou_gap_at_most_maximum",
            "require_every_donor_a5_leakage_not_above_b1",
            "min_donor_mean_a5_actionable_leakage",
            "max_donor_mean_a5_macro_miou",
            "max_donor_mean_a5_boundary_f1",
        ],
        "selected": {
            **dict(selected),
            "soft_threshold": threshold,
            "support_threshold": float(selected.get("support_threshold", 0.1)),
            "boundary_radius": int(selected.get("boundary_radius", 6)),
            "b5_fallback_parts": [],
            "b5_fallback_threshold": threshold,
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _donor_candidate_spec(value: str) -> tuple[str, float, Path]:
    subject, separator, remainder = str(value).partition(":")
    threshold, separator2, report_dir = remainder.partition(":")
    if not separator or not separator2 or not subject or not report_dir:
        raise argparse.ArgumentTypeError(
            "donor candidate must be SUBJECT:SOFT_THRESHOLD:REPORT_DIR"
        )
    return subject, float(threshold), Path(report_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a five-subject A5 LOSO config.")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--held-out-subject", required=True)
    parser.add_argument("--donor-candidate", action="append", required=True, type=_donor_candidate_spec)
    parser.add_argument("--checkpoint-fingerprint", required=True)
    parser.add_argument("--bank-fingerprint", required=True)
    parser.add_argument("--footprint-bank-fingerprint", required=True)
    parser.add_argument("--method-freeze", required=True, type=Path)
    parser.add_argument("--required-retention", nargs="+", type=float, default=[0.5, 0.6])
    parser.add_argument("--max-miou-gap", type=float, default=0.02)
    parser.add_argument("--expected-donor-count", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports_by_threshold: dict[float, list[dict]] = {}
    for subject, threshold, report_dir in args.donor_candidate:
        candidate = load_a5_candidate_report(
            report_dir,
            donor_subject=subject,
            soft_threshold=threshold,
            required_retentions=tuple(float(value) for value in args.required_retention),
        )
        reports_by_threshold.setdefault(float(threshold), []).append(candidate)
    selected = select_a5_loso_candidate(
        reports_by_threshold,
        expected_donor_count=int(args.expected_donor_count),
        max_miou_gap=float(args.max_miou_gap),
    )
    output = write_a5_loso_config(
        args.output,
        selected=selected,
        held_out_subject=args.held_out_subject,
        protocol=load_protocol(args.protocol),
        checkpoint_fingerprint=args.checkpoint_fingerprint,
        bank_fingerprint=args.bank_fingerprint,
        footprint_bank_fingerprint=args.footprint_bank_fingerprint,
        method_freeze=load_frozen_semantic_method(args.method_freeze),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
