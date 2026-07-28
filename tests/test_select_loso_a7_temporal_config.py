import csv
import json

import pytest


DONORS = ("377", "386", "387", "393")


def _report(subject, candidate_id, *, flicker=0.2, **overrides):
    report = {
        "schema_version": 1,
        "split": "validation",
        "donor_subject": subject,
        "candidate_id": candidate_id,
        "candidate_fingerprint": f"candidate-{candidate_id}",
        "a7_bank_fingerprint": f"bank-{candidate_id}",
        "a5_method_freeze_fingerprint": "a5-freeze",
        "a7_contract_fingerprint": "a7-contract",
        "parameters": {
            "lambda_outer": 0.25,
            "lambda_boundary": 0.25,
            "lambda_target": 0.0,
            "rho": 0.9,
            "min_pair_support": 8,
        },
        "weight_l1_from_a5": 2.0,
        "a5": {
            "formal_eligible_parts": 5,
            "matched_target_coverage": 0.90,
            "pooled_outer_burden": 0.20,
            "pooled_boundary_burden": 0.10,
            "macro_miou": 0.50,
            "micro_iou": 0.70,
            "fixed_outer_flicker": 0.20,
            "fixed_boundary_flicker": 0.20,
        },
        "a7": {
            "formal_eligible_parts": 5,
            "matched_target_coverage": 0.89,
            "pooled_outer_burden": 0.19,
            "pooled_boundary_burden": 0.095,
            "macro_miou": 0.495,
            "micro_iou": 0.696,
            "fixed_outer_flicker": flicker,
            "fixed_boundary_flicker": flicker,
        },
    }
    for key, value in overrides.items():
        section, field = key.split("__", 1)
        report[section][field] = value
    return report


def _candidate_reports(candidate_id, *, flicker):
    return [_report(subject, candidate_id, flicker=flicker) for subject in DONORS]


@pytest.mark.parametrize(
    ("override", "failed_gate"),
    (
        ({"a7__formal_eligible_parts": 4}, "formal_eligible_parts"),
        ({"a7__matched_target_coverage": 0.879}, "matched_target_coverage"),
        ({"a7__pooled_outer_burden": 0.205}, "pooled_outer_burden"),
        ({"a7__pooled_boundary_burden": 0.103}, "pooled_boundary_burden"),
        ({"a7__macro_miou": 0.489}, "macro_miou"),
        ({"a7__micro_iou": 0.694}, "micro_iou"),
    ),
)
def test_each_donor_must_pass_every_hard_gate(override, failed_gate):
    from tools.select_loso_a7_temporal_config import evaluate_donor_report

    result = evaluate_donor_report(_report("377", "candidate", **override))

    assert result["eligible"] is False
    assert result["gates"][failed_gate]["passed"] is False


def test_selection_requires_all_four_donors_and_uses_deterministic_priority():
    from tools.select_loso_a7_temporal_config import select_loso_a7_candidate

    reports = _candidate_reports("slower", flicker=0.20)
    reports += _candidate_reports("faster", flicker=0.10)
    reports[0]["a7"]["matched_target_coverage"] = 0.0

    selected, trace = select_loso_a7_candidate(
        reports,
        held_out_subject="394",
        expected_donor_count=4,
    )

    assert selected["candidate_id"] == "faster"
    assert selected["donor_subjects"] == list(DONORS)
    assert trace["candidates"][0]["candidate_id"] == "faster"
    slower = next(row for row in trace["candidates"] if row["candidate_id"] == "slower")
    assert slower["eligible"] is False
    assert slower["failed_donors"] == ["377"]


def test_tie_breaks_by_spatial_l1_lambda_sum_then_candidate_id():
    from tools.select_loso_a7_temporal_config import select_loso_a7_candidate

    reports = []
    for candidate_id, outer, l1, lambda_target in (
        ("z_spatial", 0.19, 1.0, 0.0),
        ("y_l1", 0.18, 3.0, 0.0),
        ("x_lambda", 0.18, 2.0, 0.25),
        ("b_id", 0.18, 2.0, 0.0),
        ("a_id", 0.18, 2.0, 0.0),
    ):
        candidate_reports = _candidate_reports(candidate_id, flicker=0.1)
        for report in candidate_reports:
            report["a7"]["pooled_outer_burden"] = outer
            report["weight_l1_from_a5"] = l1
            report["parameters"]["lambda_target"] = lambda_target
        reports.extend(candidate_reports)

    selected, trace = select_loso_a7_candidate(reports, held_out_subject="394")

    assert selected["candidate_id"] == "a_id"
    assert [row["candidate_id"] for row in trace["candidates"] if row["eligible"]][:2] == [
        "a_id",
        "b_id",
    ]


def test_held_out_validation_report_is_rejected_even_if_other_donors_exist():
    from tools.select_loso_a7_temporal_config import select_loso_a7_candidate

    reports = _candidate_reports("candidate", flicker=0.1)
    reports.append(_report("394", "candidate", flicker=0.1))

    with pytest.raises(ValueError, match="held-out validation report"):
        select_loso_a7_candidate(reports, held_out_subject="394")


def test_no_eligible_candidate_has_explicit_rejection_code():
    from tools.select_loso_a7_temporal_config import (
        A7SelectionRejected,
        select_loso_a7_candidate,
    )

    reports = _candidate_reports("candidate", flicker=0.1)
    reports[-1]["a7"]["macro_miou"] = 0.0

    with pytest.raises(A7SelectionRejected, match="A7_REJECTED_FOR_HELD_OUT_SUBJECT"):
        select_loso_a7_candidate(reports, held_out_subject="394")


def test_cli_returns_nonzero_and_prints_rejection_code(tmp_path, capsys):
    from tools.select_loso_a7_temporal_config import main

    report_paths = []
    for index, report in enumerate(_candidate_reports("candidate", flicker=0.1)):
        if index == 0:
            report["a7"]["micro_iou"] = 0.0
        path = tmp_path / f"report_{index}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        report_paths.append(path)
    argv = ["--held-out-subject", "394", "--output-dir", str(tmp_path / "out")]
    for path in report_paths:
        argv.extend(("--donor-report", str(path)))

    assert main(argv) == 2
    assert "A7_REJECTED_FOR_HELD_OUT_SUBJECT:394" in capsys.readouterr().err
    assert not (tmp_path / "out" / "selected_config.json").exists()


def test_write_outputs_records_fingerprints_matrix_and_trace(tmp_path):
    from tools.select_loso_a7_temporal_config import (
        select_loso_a7_candidate,
        write_selection_outputs,
    )

    selected, trace = select_loso_a7_candidate(
        _candidate_reports("candidate", flicker=0.1), held_out_subject="394"
    )
    write_selection_outputs(
        tmp_path,
        selected=selected,
        trace=trace,
        held_out_subject="394",
    )

    payload = json.loads((tmp_path / "selected_config.json").read_text(encoding="utf-8"))
    assert payload["held_out_subject"] == "394"
    assert payload["donor_subjects"] == list(DONORS)
    assert payload["candidate_fingerprint"] == "candidate-candidate"
    assert payload["a5_method_freeze_fingerprint"] == "a5-freeze"
    assert payload["a7_contract_fingerprint"] == "a7-contract"
    assert payload["fallback_policy"] == "none"
    assert json.loads((tmp_path / "selection_trace.json").read_text(encoding="utf-8")) == trace
    with (tmp_path / "eligibility_matrix.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert all(row["eligible"] == "True" for row in rows)
