import csv
from pathlib import Path

import pytest


def _row(subject, part, frame, method, target_sum, outer_mean, boundary_mean):
    return {
        "subject": subject,
        "part": part,
        "frame": frame,
        "method": method,
        "target_pixel_count": 100,
        "edit_target_delta_sum": target_sum,
        "edit_outer_delta_sum": outer_mean * 100.0,
        "edit_boundary_outer_delta_sum": boundary_mean * 100.0,
        "edit_target_delta_mean": target_sum / 100.0,
        "edit_outer_delta_mean": outer_mean,
        "edit_boundary_outer_delta_mean": boundary_mean,
        "inside_selection_mass": 80.0,
        "outside_selection_mass": 20.0,
    }


def test_constant_scaling_preserves_normalized_flicker():
    from tools.summarize_semantic_temporal_flicker_diagnostic import mode_flicker

    result = mode_flicker([(0, 1.0, 0.5), (1, 2.0, 1.0), (2, 1.0, 0.5)])

    assert result["fixed_pair_count"] == 2
    assert result["adaptive_pair_count"] == 2
    assert result["adaptive_flicker"] == pytest.approx(result["fixed_flicker"])
    assert result["compensation_penalty"] == pytest.approx(0.0)


def test_varying_adaptive_strength_changes_flicker():
    from tools.summarize_semantic_temporal_flicker_diagnostic import mode_flicker

    result = mode_flicker([(0, 1.0, 1.0), (1, 2.0, 1.0), (2, 1.0, 1.0)])

    assert result["fixed_flicker"] > 0.0
    assert result["adaptive_flicker"] == pytest.approx(0.0)
    assert result["compensation_penalty"] < 0.0


def test_end_to_end_uses_same_reachable_frames_for_both_modes(tmp_path: Path):
    from tools.summarize_semantic_temporal_flicker_diagnostic import summarize

    rows = []
    for subject in ("377", "386"):
        for frame, a5_target in enumerate((50.0, 100.0, 50.0)):
            rows.append(_row(subject, "upper", frame, "voting", 100.0, (1.0, 2.0, 1.0)[frame], 0.5))
            rows.append(_row(subject, "upper", frame, "a5", a5_target, (1.0, 2.0, 1.0)[frame], 0.5))
            rows.append(_row(subject, "shoes", frame, "voting", 100.0, 1.0, 0.5))
            rows.append(_row(subject, "shoes", frame, "a5", 10.0, 1.0, 0.5))
    input_csv = tmp_path / "all_metrics.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output_dir = tmp_path / "diagnostic"
    summary = summarize(
        input_csv=input_csv,
        output_dir=output_dir,
        subjects=("377", "386"),
        parts=("upper", "shoes"),
        retentions=(0.5,),
        coverage_threshold=0.8,
        bootstrap_repetitions=200,
        bootstrap_seed=7,
    )

    assert summary["eligible_parts"] == {"0.5": ["upper"]}
    assert {path.name for path in output_dir.iterdir()} == {
        "sequence_table.csv",
        "subject_table.csv",
        "formal_table.csv",
        "paired_statistics.csv",
        "summary.json",
    }
    with (output_dir / "formal_table.csv").open("r", encoding="utf-8", newline="") as handle:
        formal = list(csv.DictReader(handle))
    voting_fixed = next(row for row in formal if row["method"] == "voting" and row["mode"] == "fixed")
    voting_adaptive = next(row for row in formal if row["method"] == "voting" and row["mode"] == "adaptive")
    assert float(voting_fixed["outer_flicker_mean"]) == pytest.approx(
        float(voting_adaptive["outer_flicker_mean"])
    )
