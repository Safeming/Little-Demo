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

from utils.semantic_paper_metrics import interpolate_curve_at_retention


def _baseline_metric(rows: list[dict], baseline: str, key: str) -> float:
    matches = [row for row in rows if str(row.get("baseline")) == baseline]
    if len(matches) != 1:
        raise ValueError(f"expected one {baseline} baseline summary row, found {len(matches)}")
    return float(matches[0][key])


def _numeric_curve(rows: list[dict], baseline: str) -> list[dict]:
    result = []
    for row in rows:
        if str(row.get("baseline")) != baseline:
            continue
        result.append(
            {
                "baseline": baseline,
                "retention": float(row["retention"]),
                "actionable_leakage": float(row["actionable_leakage"]),
            }
        )
    return result


def assess_candidate(
    baseline_rows: list[dict],
    curve_rows: list[dict],
    *,
    required_retentions: tuple[float, ...] = (0.5, 0.6),
    max_miou_gap: float = 0.02,
) -> dict:
    b1_miou = _baseline_metric(baseline_rows, "B1", "macro_miou")
    b5_miou = _baseline_metric(baseline_rows, "B5", "macro_miou")
    miou_gap = b1_miou - b5_miou
    failure_reasons = []
    retention_checks = []

    curves = {baseline: _numeric_curve(curve_rows, baseline) for baseline in ("B1", "B5")}
    for retention in required_retentions:
        try:
            b1 = interpolate_curve_at_retention(curves["B1"], float(retention))
            b5 = interpolate_curve_at_retention(curves["B5"], float(retention))
        except ValueError as error:
            failure_reasons.append(f"retention {float(retention):.4f}: {error}")
            continue
        b1_leakage = float(b1["actionable_leakage"])
        b5_leakage = float(b5["actionable_leakage"])
        leakage_passed = b5_leakage <= b1_leakage + 1.0e-12
        retention_checks.append(
            {
                "retention": float(retention),
                "b1_actionable_leakage": b1_leakage,
                "b5_actionable_leakage": b5_leakage,
                "passed": leakage_passed,
            }
        )
        if not leakage_passed:
            failure_reasons.append(
                f"retention {float(retention):.4f}: B5 actionable leakage "
                f"{b5_leakage:.8f} exceeds B1 {b1_leakage:.8f}"
            )

    if miou_gap > float(max_miou_gap) + 1.0e-12:
        failure_reasons.append(
            f"B1-B5 mIoU gap {miou_gap:.8f} exceeds maximum {float(max_miou_gap):.8f}"
        )

    return {
        "passed": not failure_reasons,
        "required_retentions": [float(value) for value in required_retentions],
        "max_miou_gap": float(max_miou_gap),
        "b1_macro_miou": b1_miou,
        "b5_macro_miou": b5_miou,
        "miou_gap": miou_gap,
        "retention_checks": retention_checks,
        "failure_reasons": failure_reasons,
    }


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assess a validation-only voting posterior candidate.")
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--curve", required=True, type=Path)
    parser.add_argument("--required-retention", nargs="+", type=float, default=[0.5, 0.6])
    parser.add_argument("--max-miou-gap", type=float, default=0.02)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    result = assess_candidate(
        _read_csv(args.baseline_summary),
        _read_csv(args.curve),
        required_retentions=tuple(args.required_retention),
        max_miou_gap=float(args.max_miou_gap),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
