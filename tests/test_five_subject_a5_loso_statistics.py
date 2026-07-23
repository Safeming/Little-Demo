import csv
from pathlib import Path

import pytest


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_a5_candidate_report(
    root: Path,
    *,
    b1_miou: float = 0.50,
    a5_miou: float = 0.49,
    b1_leakage: float = 0.020,
    a5_leakage: float = 0.012,
) -> None:
    _write_csv(
        root / "baseline_summary.csv",
        [
            {"baseline": "B1", "macro_miou": b1_miou, "mean_boundary_f1": 0.40},
            {"baseline": "A5", "macro_miou": a5_miou, "mean_boundary_f1": 0.45},
        ],
    )
    _write_csv(
        root / "leakage_retention_curve.csv",
        [
            {
                "baseline": "B1",
                "retention": retention,
                "actionable_leakage": b1_leakage * retention / 0.5,
                "raw_leakage": b1_leakage * 2.0 * retention / 0.5,
            }
            for retention in (0.5, 0.6)
        ],
    )
    _write_csv(
        root / "matched_retention.csv",
        [
            {
                "baseline": "A5",
                "retention": retention,
                "actionable_leakage": a5_leakage * retention / 0.5,
                "raw_leakage": a5_leakage * 2.0 * retention / 0.5,
            }
            for retention in (0.5, 0.6)
        ],
    )


def test_load_a5_candidate_report_uses_a5_not_historical_b5(tmp_path):
    from tools.select_frozen_a5_loso_config import load_a5_candidate_report

    report = tmp_path / "candidate"
    _write_a5_candidate_report(report)

    candidate = load_a5_candidate_report(
        report,
        donor_subject="377",
        soft_threshold=0.1,
        required_retentions=(0.5, 0.6),
    )

    assert candidate["donor_subject"] == "377"
    assert candidate["a5_macro_miou"] == pytest.approx(0.49)
    assert candidate["a5_mean_boundary_f1"] == pytest.approx(0.45)
    assert [row["retention"] for row in candidate["retention_checks"]] == [0.5, 0.6]
    assert all(row["a5_actionable_leakage"] < row["b1_actionable_leakage"] for row in candidate["retention_checks"])


def test_select_a5_loso_candidate_requires_four_unique_donors():
    from tools.select_frozen_a5_loso_config import select_a5_loso_candidate

    reports = {
        0.1: [
            {
                "donor_subject": subject,
                "b1_macro_miou": 0.50,
                "a5_macro_miou": 0.49,
                "a5_mean_boundary_f1": 0.45,
                "retention_checks": [
                    {"b1_actionable_leakage": 0.02, "a5_actionable_leakage": 0.01}
                ],
            }
            for subject in ("377", "386", "387")
        ]
    }

    with pytest.raises(ValueError, match="exactly 4 unique donor subjects"):
        select_a5_loso_candidate(reports, expected_donor_count=4)


def test_select_a5_loso_candidate_minimizes_leakage_after_gates():
    from tools.select_frozen_a5_loso_config import select_a5_loso_candidate

    def reports(threshold, leakage, gap=0.01):
        return [
            {
                "donor_subject": subject,
                "soft_threshold": threshold,
                "b1_macro_miou": 0.50,
                "a5_macro_miou": 0.50 - gap,
                "a5_mean_boundary_f1": 0.45,
                "retention_checks": [
                    {"b1_actionable_leakage": 0.02, "a5_actionable_leakage": leakage},
                    {"b1_actionable_leakage": 0.024, "a5_actionable_leakage": leakage * 1.2},
                ],
            }
            for subject in ("377", "386", "387", "394")
        ]

    selected = select_a5_loso_candidate(
        {
            0.05: reports(0.05, 0.013),
            0.10: reports(0.10, 0.011),
            0.15: reports(0.15, 0.009, gap=0.03),
        },
        expected_donor_count=4,
        max_miou_gap=0.02,
    )

    assert selected["soft_threshold"] == pytest.approx(0.10)
    assert selected["donor_subjects"] == ["377", "386", "387", "394"]


def test_incomplete_retention_coverage_marks_candidate_ineligible(tmp_path):
    from tools.select_frozen_a5_loso_config import (
        load_a5_candidate_report,
        select_a5_loso_candidate,
    )

    incomplete = tmp_path / "incomplete"
    _write_a5_candidate_report(incomplete)
    rows = list(csv.DictReader((incomplete / "matched_retention.csv").open(newline="")))
    _write_csv(
        incomplete / "matched_retention.csv",
        [row for row in rows if float(row["retention"]) == 0.5],
    )
    incomplete_candidate = load_a5_candidate_report(
        incomplete,
        donor_subject="377",
        soft_threshold=0.5,
        required_retentions=(0.5, 0.6),
    )
    assert incomplete_candidate["coverage_complete"] is False

    def complete(subject, threshold, leakage):
        return {
            "donor_subject": subject,
            "soft_threshold": threshold,
            "coverage_complete": True,
            "b1_macro_miou": 0.50,
            "a5_macro_miou": 0.49,
            "a5_mean_boundary_f1": 0.45,
            "retention_checks": [
                {"b1_actionable_leakage": 0.02, "a5_actionable_leakage": leakage}
            ],
        }

    selected = select_a5_loso_candidate(
        {
            0.2: [complete(subject, 0.2, 0.01) for subject in ("377", "386", "387", "394")],
            0.5: [
                incomplete_candidate,
                *[complete(subject, 0.5, 0.005) for subject in ("386", "387", "394")],
            ],
        },
        expected_donor_count=4,
    )

    assert selected["soft_threshold"] == pytest.approx(0.2)


def test_paired_statistics_use_a5_minus_b1_and_deterministic_bootstrap():
    from tools.summarize_five_subject_a5_loso_statistics import paired_statistics

    first = paired_statistics([0.01, -0.02, 0.03, 0.00, 0.02], repetitions=2000, seed=17)
    second = paired_statistics([0.01, -0.02, 0.03, 0.00, 0.02], repetitions=2000, seed=17)

    assert first == second
    assert first["mean_delta"] == pytest.approx(0.008)
    assert first["sample_std"] == pytest.approx(0.019235384061671343)
    assert first["wins"] == 3
    assert first["ties"] == 1
    assert first["losses"] == 1


def test_formal_main_table_rejects_a6():
    from tools.summarize_five_subject_a5_loso_statistics import validate_main_methods

    with pytest.raises(ValueError, match="A6 is ablation-only"):
        validate_main_methods(["B0", "B1", "A5", "A6"])


def test_main_std_uses_only_subject_rows(tmp_path, monkeypatch):
    import tools.summarize_five_subject_a5_loso_statistics as summary

    monkeypatch.setattr(summary, "SUBJECTS", ("377", "386"))
    for subject, a5_value in (("377", 1.0), ("386", 3.0)):
        rows = []
        for method in sorted(summary.MAIN_METHODS):
            value = a5_value if method == "A5" else 0.5
            rows.append(
                {
                    "baseline": method,
                    "name": method,
                    "macro_miou": value,
                    "mean_boundary_f1": value,
                    "mean_boundary_iou": value,
                    "mean_soft_iou": value,
                    "micro_iou": value,
                }
            )
        _write_csv(tmp_path / f"CoreView_{subject}" / "main" / "baseline_summary.csv", rows)

    rows = summary._aggregate_main(tmp_path)
    by_key = {(row["subject"], row["baseline"]): row for row in rows}

    assert by_key[("MEAN", "A5")]["macro_miou"] == pytest.approx(2.0)
    assert by_key[("STD", "A5")]["macro_miou"] == pytest.approx(2 ** 0.5)


def test_queue_contract_is_five_subject_a5_validation_loso_without_training():
    script = Path("tools/run_five_subject_a5_loso_statistics.sh").read_text(encoding="utf-8")

    for subject in ("377", "386", "387", "393", "394"):
        assert subject in script
    for threshold in ("0.05", "0.1", "0.15", "0.2", "0.25", "0.35", "0.5"):
        assert threshold in script
    assert "--protocol-split validation" in script
    assert "--protocol-split test" in script
    assert "--baselines B1 A5" in script
    assert "--baselines B0 B1 B2 B3 B4 A5" in script
    assert "frozen_a5_main_method_v1.json" in script
    assert "select_frozen_a5_loso_config.py" in script
    assert "summarize_five_subject_a5_loso_statistics.py" in script
    assert 'frozen="$(build_loso_config' not in script
    assert '[[ -s "$frozen" ]]' in script
    assert "train.py" not in script
    assert "semantic-train" not in script
