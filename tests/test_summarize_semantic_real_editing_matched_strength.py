import math
import subprocess
import sys
from pathlib import Path


def _row(subject, method, strength, target, outer, boundary, *, part="upper", task="recolor", target_pixels=10):
    return {
        "subject": subject,
        "view": "c21_f000180",
        "part": part,
        "task": task,
        "method": method,
        "edit_strength": strength,
        "target_pixel_count": target_pixels,
        "target_delta_sum": target,
        "outer_delta_sum": outer,
        "boundary_outer_delta_sum": boundary,
    }


def _two_subject_rows():
    rows = []
    for subject, scale in (("377", 1.0), ("386", 2.0)):
        rows.extend(
            [
                _row(subject, "voting", 0.5, 5 * scale, 1.0 * scale, 0.5 * scale),
                _row(subject, "voting", 1.0, 10 * scale, 2.0 * scale, 1.0 * scale),
                _row(subject, "raw_hard", 0.5, 6 * scale, 1.8 * scale, 0.9 * scale),
                _row(subject, "raw_hard", 1.0, 12 * scale, 3.6 * scale, 1.8 * scale),
                _row(subject, "a5", 0.5, 4 * scale, 0.4 * scale, 0.2 * scale),
                _row(subject, "a5", 1.0, 8 * scale, 0.8 * scale, 0.4 * scale),
            ]
        )
    return rows


def test_matched_summarizer_cli_can_start_directly():
    result = subprocess.run(
        [sys.executable, "tools/summarize_semantic_real_editing_matched_strength.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_summarize_matched_rows_uses_common_coverage_and_subject_equal_deltas():
    from tools.summarize_semantic_real_editing_matched_strength import summarize_matched_rows

    result = summarize_matched_rows(_two_subject_rows(), retentions=[0.5], bootstrap_repetitions=1000, bootstrap_seed=7)

    coverage = [
        row for row in result["coverage_rows"]
        if row["scope"] == "task" and row["task"] == "recolor" and row["retention"] == 0.5
    ]
    by_method = {row["method_or_comparison"]: row for row in coverage}
    assert by_method["a5"]["covered_count"] == 2
    assert by_method["a5-voting"]["covered_count"] == 2

    paired = [
        row for row in result["paired_rows"]
        if row["comparison"] == "a5-voting" and row["metric"] == "outer_burden"
    ][0]
    assert math.isclose(paired["mean_delta"], -0.05)
    assert paired["wins"] == 2
    assert paired["losses"] == 0


def test_summarize_matched_rows_excludes_zero_reference_and_stays_finite():
    from tools.summarize_semantic_real_editing_matched_strength import summarize_matched_rows

    rows = _two_subject_rows()
    rows.extend(
        [
            _row("393", "voting", 1.0, 0.0, 1.0, 1.0, part="face"),
            _row("393", "a5", 1.0, 3.0, 0.1, 0.1, part="face"),
            _row("393", "raw_hard", 1.0, 4.0, 0.2, 0.2, part="face"),
        ]
    )

    result = summarize_matched_rows(rows, retentions=[0.5], bootstrap_repetitions=100, bootstrap_seed=3)

    assert result["unsupported_reference_count"] == 1
    for group in ("curve_rows", "matched_rows", "subject_rows", "aggregate_rows", "paired_rows"):
        for row in result[group]:
            for value in row.values():
                if isinstance(value, float):
                    assert math.isfinite(value)
