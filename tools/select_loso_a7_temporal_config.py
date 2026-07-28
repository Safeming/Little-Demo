#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REJECTION_CODE = "A7_REJECTED_FOR_HELD_OUT_SUBJECT"
GATE_ORDER = (
    "formal_eligible_parts",
    "matched_target_coverage",
    "pooled_outer_burden",
    "pooled_boundary_burden",
    "macro_miou",
    "micro_iou",
)


class A7SelectionRejected(ValueError):
    pass


def _as_float(mapping: dict, key: str) -> float:
    try:
        return float(mapping[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid metric: {key}") from error


def _as_int(mapping: dict, key: str) -> int:
    try:
        return int(mapping[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid metric: {key}") from error


def evaluate_donor_report(report: dict) -> dict:
    if str(report.get("split", "")) != "validation":
        raise ValueError("A7 LOSO selection accepts validation reports only")
    a5 = dict(report.get("a5", {}))
    a7 = dict(report.get("a7", {}))
    gates = {
        "formal_eligible_parts": {
            "a5": _as_int(a5, "formal_eligible_parts"),
            "a7": _as_int(a7, "formal_eligible_parts"),
        },
        "matched_target_coverage": {
            "a5": _as_float(a5, "matched_target_coverage"),
            "a7": _as_float(a7, "matched_target_coverage"),
        },
        "pooled_outer_burden": {
            "a5": _as_float(a5, "pooled_outer_burden"),
            "a7": _as_float(a7, "pooled_outer_burden"),
        },
        "pooled_boundary_burden": {
            "a5": _as_float(a5, "pooled_boundary_burden"),
            "a7": _as_float(a7, "pooled_boundary_burden"),
        },
        "macro_miou": {
            "a5": _as_float(a5, "macro_miou"),
            "a7": _as_float(a7, "macro_miou"),
        },
        "micro_iou": {
            "a5": _as_float(a5, "micro_iou"),
            "a7": _as_float(a7, "micro_iou"),
        },
    }
    gates["formal_eligible_parts"]["threshold"] = gates["formal_eligible_parts"]["a5"]
    gates["formal_eligible_parts"]["passed"] = (
        gates["formal_eligible_parts"]["a7"]
        >= gates["formal_eligible_parts"]["threshold"]
    )
    gates["matched_target_coverage"]["threshold"] = (
        gates["matched_target_coverage"]["a5"] - 0.02
    )
    gates["matched_target_coverage"]["passed"] = (
        gates["matched_target_coverage"]["a7"] + 1.0e-12
        >= gates["matched_target_coverage"]["threshold"]
    )
    for name in ("pooled_outer_burden", "pooled_boundary_burden"):
        gates[name]["threshold"] = 1.02 * gates[name]["a5"]
        gates[name]["passed"] = gates[name]["a7"] <= gates[name]["threshold"] + 1.0e-12
    gates["macro_miou"]["threshold"] = gates["macro_miou"]["a5"] - 0.01
    gates["macro_miou"]["passed"] = (
        gates["macro_miou"]["a7"] + 1.0e-12 >= gates["macro_miou"]["threshold"]
    )
    gates["micro_iou"]["threshold"] = gates["micro_iou"]["a5"] - 0.005
    gates["micro_iou"]["passed"] = (
        gates["micro_iou"]["a7"] + 1.0e-12 >= gates["micro_iou"]["threshold"]
    )
    return {
        "donor_subject": str(report.get("donor_subject", "")),
        "candidate_id": str(report.get("candidate_id", "")),
        "source_report": str(report.get("source_report", "")),
        "eligible": all(bool(gates[name]["passed"]) for name in GATE_ORDER),
        "gates": gates,
    }


def _candidate_identity(report: dict) -> tuple:
    parameters = dict(report.get("parameters", {}))
    return (
        str(report.get("candidate_fingerprint", "")),
        str(report.get("a7_bank_fingerprint", "")),
        str(report.get("a5_method_freeze_fingerprint", "")),
        str(report.get("a7_contract_fingerprint", "")),
        json.dumps(parameters, sort_keys=True, separators=(",", ":")),
        float(report.get("weight_l1_from_a5", 0.0)),
    )


def _mean(values) -> float:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("cannot average an empty metric sequence")
    return sum(values) / len(values)


def _candidate_trace(candidate_id: str, reports: list[dict]) -> dict:
    identities = {_candidate_identity(report) for report in reports}
    if len(identities) != 1:
        raise ValueError(f"candidate {candidate_id} has inconsistent fingerprints or parameters")
    donor_results = [evaluate_donor_report(report) for report in reports]
    failed_donors = sorted(
        row["donor_subject"] for row in donor_results if not bool(row["eligible"])
    )
    parameters = dict(reports[0].get("parameters", {}))
    lambda_sum = sum(
        float(parameters.get(name, 0.0))
        for name in ("lambda_outer", "lambda_boundary", "lambda_target")
    )
    mean_fixed_temporal_burden = _mean(
        float(report["a7"]["fixed_outer_flicker"])
        + float(report["a7"]["fixed_boundary_flicker"])
        for report in reports
    )
    mean_spatial_burden = _mean(
        float(report["a7"]["pooled_outer_burden"])
        + float(report["a7"]["pooled_boundary_burden"])
        for report in reports
    )
    return {
        "candidate_id": candidate_id,
        "candidate_fingerprint": str(reports[0]["candidate_fingerprint"]),
        "a7_bank_fingerprint": str(reports[0]["a7_bank_fingerprint"]),
        "a5_method_freeze_fingerprint": str(reports[0]["a5_method_freeze_fingerprint"]),
        "a7_contract_fingerprint": str(reports[0]["a7_contract_fingerprint"]),
        "parameters": parameters,
        "weight_l1_from_a5": float(reports[0].get("weight_l1_from_a5", 0.0)),
        "lambda_sum": float(lambda_sum),
        "donor_subjects": sorted(str(report["donor_subject"]) for report in reports),
        "eligible": not failed_donors,
        "failed_donors": failed_donors,
        "mean_fixed_outer_plus_boundary_flicker": mean_fixed_temporal_burden,
        "mean_pooled_outer_plus_boundary_burden": mean_spatial_burden,
        "donor_results": donor_results,
    }


def select_loso_a7_candidate(
    reports: list[dict],
    *,
    held_out_subject: str,
    expected_donor_count: int = 4,
) -> tuple[dict, dict]:
    if not reports:
        raise ValueError("at least one A7 donor validation report is required")
    held_out_subject = str(held_out_subject)
    for report in reports:
        donor = str(report.get("donor_subject", ""))
        if donor == held_out_subject:
            raise ValueError(
                f"held-out validation report is forbidden for subject {held_out_subject}"
            )
        if str(report.get("split", "")) != "validation":
            raise ValueError("A7 LOSO selection accepts validation reports only")

    grouped: dict[str, list[dict]] = {}
    for report in reports:
        candidate_id = str(report.get("candidate_id", ""))
        if not candidate_id:
            raise ValueError("donor report is missing candidate_id")
        grouped.setdefault(candidate_id, []).append(report)

    candidate_rows = []
    expected_donors = None
    for candidate_id, candidate_reports in sorted(grouped.items()):
        donors = [str(report.get("donor_subject", "")) for report in candidate_reports]
        if len(donors) != len(set(donors)):
            raise ValueError(f"candidate {candidate_id} contains duplicate donor subjects")
        if len(donors) != int(expected_donor_count):
            raise ValueError(
                f"candidate {candidate_id} requires exactly {expected_donor_count} unique donor subjects"
            )
        donor_set = frozenset(donors)
        if expected_donors is None:
            expected_donors = donor_set
        elif donor_set != expected_donors:
            raise ValueError("every A7 candidate must contain the same donor subjects")
        candidate_rows.append(_candidate_trace(candidate_id, candidate_reports))

    candidate_rows.sort(
        key=lambda row: (
            not bool(row["eligible"]),
            float(row["mean_fixed_outer_plus_boundary_flicker"]),
            float(row["mean_pooled_outer_plus_boundary_burden"]),
            float(row["weight_l1_from_a5"]),
            float(row["lambda_sum"]),
            str(row["candidate_id"]),
        )
    )
    eligible = [row for row in candidate_rows if bool(row["eligible"])]
    trace = {
        "selection_mode": "leave_one_subject_out_a7_temporal_reliability",
        "held_out_subject": held_out_subject,
        "expected_donor_count": int(expected_donor_count),
        "fallback_policy": "none",
        "selection_priority": [
            "min_donor_subject_equal_mean_fixed_outer_plus_boundary_flicker",
            "min_donor_subject_equal_mean_pooled_spatial_burden",
            "min_weight_l1_from_a5",
            "min_lambda_outer_plus_boundary_plus_target",
            "candidate_id_lexicographic",
        ],
        "candidates": candidate_rows,
    }
    if not eligible:
        raise A7SelectionRejected(f"{REJECTION_CODE}:{held_out_subject}")
    selected = dict(eligible[0])
    selected.pop("donor_results", None)
    return selected, trace


def _eligibility_rows(trace: dict) -> list[dict]:
    rows = []
    for candidate in trace["candidates"]:
        for donor in candidate["donor_results"]:
            row = {
                "candidate_id": candidate["candidate_id"],
                "donor_subject": donor["donor_subject"],
                "eligible": donor["eligible"],
            }
            for gate_name in GATE_ORDER:
                gate = donor["gates"][gate_name]
                row[f"{gate_name}_a5"] = gate["a5"]
                row[f"{gate_name}_a7"] = gate["a7"]
                row[f"{gate_name}_threshold"] = gate["threshold"]
                row[f"{gate_name}_passed"] = gate["passed"]
            rows.append(row)
    return rows


def write_selection_outputs(
    output_dir: Path,
    *,
    selected: dict,
    trace: dict,
    held_out_subject: str,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_config = {
        "schema_version": 1,
        "selection_mode": trace["selection_mode"],
        "fallback_policy": "none",
        "held_out_subject": str(held_out_subject),
        "donor_subjects": list(selected["donor_subjects"]),
        "candidate_id": str(selected["candidate_id"]),
        "candidate_fingerprint": str(selected["candidate_fingerprint"]),
        "a7_bank_fingerprint": str(selected["a7_bank_fingerprint"]),
        "a5_method_freeze_fingerprint": str(selected["a5_method_freeze_fingerprint"]),
        "a7_contract_fingerprint": str(selected["a7_contract_fingerprint"]),
        "parameters": dict(selected["parameters"]),
        "selection_metrics": {
            "mean_fixed_outer_plus_boundary_flicker": selected[
                "mean_fixed_outer_plus_boundary_flicker"
            ],
            "mean_pooled_outer_plus_boundary_burden": selected[
                "mean_pooled_outer_plus_boundary_burden"
            ],
            "weight_l1_from_a5": selected["weight_l1_from_a5"],
            "lambda_sum": selected["lambda_sum"],
        },
    }
    selected_path = output_dir / "selected_config.json"
    selected_path.write_text(
        json.dumps(selected_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    trace_path = output_dir / "selection_trace.json"
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
    matrix_rows = _eligibility_rows(trace)
    matrix_path = output_dir / "eligibility_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0]))
        writer.writeheader()
        writer.writerows(matrix_rows)
    return selected_path


def load_donor_report(path: Path) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"donor report must be a JSON object: {path}")
    payload["source_report"] = str(path.resolve())
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a frozen A7 configuration with four-donor LOSO hard gates."
    )
    parser.add_argument("--held-out-subject", required=True)
    parser.add_argument("--donor-report", action="append", required=True, type=Path)
    parser.add_argument("--expected-donor-count", type=int, default=4)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports = [load_donor_report(path) for path in args.donor_report]
    try:
        selected, trace = select_loso_a7_candidate(
            reports,
            held_out_subject=args.held_out_subject,
            expected_donor_count=args.expected_donor_count,
        )
    except A7SelectionRejected as error:
        print(str(error), file=sys.stderr)
        return 2
    output = write_selection_outputs(
        args.output_dir,
        selected=selected,
        trace=trace,
        held_out_subject=args.held_out_subject,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
