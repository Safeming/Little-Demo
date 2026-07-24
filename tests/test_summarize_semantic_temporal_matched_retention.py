import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _metric_row(
    *,
    subject="377",
    part="upper",
    frame=0,
    method="voting",
    target_sum=100.0,
    outer_sum=40.0,
    boundary_sum=10.0,
    inside_mass=80.0,
    outside_mass=20.0,
):
    return {
        "subject": subject,
        "part": part,
        "frame": frame,
        "method": method,
        "target_pixel_count": 100,
        "edit_target_delta_sum": target_sum,
        "edit_outer_delta_sum": outer_sum,
        "edit_boundary_outer_delta_sum": boundary_sum,
        "edit_target_delta_mean": target_sum / 100.0,
        "edit_outer_delta_mean": outer_sum / 100.0,
        "edit_boundary_outer_delta_mean": boundary_sum / 100.0,
        "inside_selection_mass": inside_mass,
        "outside_selection_mass": outside_mass,
    }


def test_exact_matching_scales_both_methods_to_the_same_target_effect():
    from tools.summarize_semantic_temporal_matched_retention import match_record_pair

    voting = _metric_row(method="voting", target_sum=100.0, outer_sum=40.0)
    a5 = _metric_row(method="a5", target_sum=50.0, outer_sum=10.0)

    matched = match_record_pair(voting, a5, retention=0.25)

    assert matched["reachable"] is True
    assert matched["max_retention"] == pytest.approx(0.5)
    assert matched["voting"]["strength"] == pytest.approx(0.25)
    assert matched["a5"]["strength"] == pytest.approx(0.5)
    assert matched["voting"]["edit_target_delta_sum"] == pytest.approx(25.0)
    assert matched["a5"]["edit_target_delta_sum"] == pytest.approx(25.0)
    assert matched["voting"]["edit_outer_delta_sum"] == pytest.approx(10.0)
    assert matched["a5"]["edit_outer_delta_sum"] == pytest.approx(5.0)
    assert matched["voting"]["inside_selection_mass"] == pytest.approx(20.0)
    assert matched["a5"]["inside_selection_mass"] == pytest.approx(40.0)


def test_unreachable_pair_is_counted_as_coverage_failure():
    from tools.summarize_semantic_temporal_matched_retention import match_record_pair

    voting = _metric_row(method="voting", target_sum=100.0)
    a5 = _metric_row(method="a5", target_sum=20.0)

    matched = match_record_pair(voting, a5, retention=0.25)

    assert matched["reference_supported"] is True
    assert matched["reachable"] is False
    assert matched["max_retention"] == pytest.approx(0.2)


def test_eligible_parts_require_threshold_in_every_subject():
    from tools.summarize_semantic_temporal_matched_retention import select_eligible_parts

    coverage_rows = [
        {"retention": 0.25, "subject": "377", "part": "upper", "coverage_rate": 0.90, "reference_count": 10},
        {"retention": 0.25, "subject": "386", "part": "upper", "coverage_rate": 0.80, "reference_count": 10},
        {"retention": 0.25, "subject": "377", "part": "shoes", "coverage_rate": 1.00, "reference_count": 10},
        {"retention": 0.25, "subject": "386", "part": "shoes", "coverage_rate": 0.79, "reference_count": 10},
    ]

    selected = select_eligible_parts(
        coverage_rows,
        subjects=("377", "386"),
        parts=("upper", "shoes"),
        retentions=(0.25,),
        threshold=0.80,
    )

    assert selected == {0.25: ["upper"]}


def test_flicker_uses_only_consecutive_supported_frame_pairs():
    from tools.summarize_semantic_temporal_matched_retention import consecutive_supported_flicker

    result = consecutive_supported_flicker([(0, 2.0), (1, 4.0), (3, 10.0)])

    assert result["pair_count"] == 1
    assert result["flicker"] == pytest.approx(2.0 / ((2.0 + 4.0 + 10.0) / 3.0))


def test_script_cli_can_run_directly():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools/summarize_semantic_temporal_matched_retention.py"), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "matched-retention" in result.stdout


def test_end_to_end_writes_all_formal_outputs(tmp_path: Path):
    from tools.summarize_semantic_temporal_matched_retention import summarize

    input_csv = tmp_path / "all_metrics.csv"
    rows = []
    for subject in ("377", "386"):
        for frame in range(3):
            rows.append(_metric_row(subject=subject, part="upper", frame=frame, method="voting"))
            rows.append(
                _metric_row(
                    subject=subject,
                    part="upper",
                    frame=frame,
                    method="a5",
                    target_sum=60.0,
                    outer_sum=12.0,
                    boundary_sum=4.0,
                    inside_mass=60.0,
                    outside_mass=6.0,
                )
            )
            rows.append(_metric_row(subject=subject, part="shoes", frame=frame, method="voting"))
            rows.append(_metric_row(subject=subject, part="shoes", frame=frame, method="a5", target_sum=20.0))
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output_dir = tmp_path / "formal"
    summary = summarize(
        input_csv=input_csv,
        output_dir=output_dir,
        subjects=("377", "386"),
        parts=("upper", "shoes"),
        retentions=(0.25, 0.50),
        coverage_threshold=0.80,
        bootstrap_repetitions=200,
        bootstrap_seed=7,
    )

    expected = {
        "coverage_table.csv",
        "part_table.csv",
        "subject_table.csv",
        "formal_table.csv",
        "paired_statistics.csv",
        "summary.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    assert summary["eligible_parts"] == {"0.25": ["upper"], "0.5": ["upper"]}
    assert summary["source_row_count"] == 24
    on_disk = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["coverage_threshold"] == pytest.approx(0.80)

    with (output_dir / "formal_table.csv").open("r", encoding="utf-8", newline="") as handle:
        formal_rows = list(csv.DictReader(handle))
    assert len(formal_rows) == 4
    a5 = next(row for row in formal_rows if row["method"] == "a5" and float(row["retention"]) == 0.25)
    voting = next(row for row in formal_rows if row["method"] == "voting" and float(row["retention"]) == 0.25)
    assert float(a5["pooled_outer_burden_mean"]) == pytest.approx(0.2)
    assert float(voting["pooled_outer_burden_mean"]) == pytest.approx(0.4)
