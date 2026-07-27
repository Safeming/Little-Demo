import csv
import json
from pathlib import Path

import pytest


def _row(subject, part, method, strength, target, outer, boundary, task="recolor", view="c21_f000180"):
    return {
        "subject": subject,
        "view": view,
        "part": part,
        "task": task,
        "method": method,
        "edit_strength": strength,
        "target_pixel_count": 100,
        "target_delta_sum": target,
        "outer_delta_sum": outer,
        "boundary_outer_delta_sum": boundary,
    }


def test_recover_matched_sums_uses_reference_target_denominator():
    from tools.summarize_semantic_real_editing_coverage_constrained import recover_matched_sums

    result = recover_matched_sums(
        reference_target=100.0,
        matched_point={"retention": 0.5, "outer_burden": 0.2, "boundary_burden": 0.05},
    )

    assert result["target_delta_sum"] == pytest.approx(50.0)
    assert result["outer_delta_sum"] == pytest.approx(20.0)
    assert result["boundary_delta_sum"] == pytest.approx(5.0)


def test_part_eligibility_is_per_task_and_requires_every_subject():
    from tools.summarize_semantic_real_editing_coverage_constrained import select_eligible_parts

    coverage = [
        {"task": "recolor", "retention": 0.5, "subject": "377", "part": "upper", "reference_count": 10, "coverage_rate": 0.9},
        {"task": "recolor", "retention": 0.5, "subject": "386", "part": "upper", "reference_count": 10, "coverage_rate": 0.8},
        {"task": "removal", "retention": 0.5, "subject": "377", "part": "upper", "reference_count": 10, "coverage_rate": 0.9},
        {"task": "removal", "retention": 0.5, "subject": "386", "part": "upper", "reference_count": 10, "coverage_rate": 0.79},
    ]

    selected = select_eligible_parts(
        coverage,
        tasks=("recolor", "removal"),
        retentions=(0.5,),
        subjects=("377", "386"),
        parts=("upper",),
        threshold=0.8,
    )

    assert selected == {("recolor", 0.5): ["upper"], ("removal", 0.5): []}


def test_end_to_end_writes_pooled_formal_tables(tmp_path: Path):
    from tools.summarize_semantic_real_editing_coverage_constrained import summarize

    input_root = tmp_path / "input"
    for subject in ("377", "386"):
        rows = []
        for part, a5_max in (("upper", 60.0), ("shoes", 20.0)):
            rows.extend(
                [
                    _row(subject, part, "voting", 0.5, 50.0, 20.0, 5.0),
                    _row(subject, part, "voting", 1.0, 100.0, 40.0, 10.0),
                    _row(subject, part, "a5", 0.5, a5_max / 2.0, a5_max / 10.0, a5_max / 30.0),
                    _row(subject, part, "a5", 1.0, a5_max, a5_max / 5.0, a5_max / 15.0),
                ]
            )
        subject_dir = input_root / f"CoreView_{subject}"
        subject_dir.mkdir(parents=True)
        with (subject_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    output_dir = tmp_path / "formal"
    summary = summarize(
        input_root=input_root,
        output_dir=output_dir,
        subjects=("377", "386"),
        tasks=("recolor",),
        parts=("upper", "shoes"),
        retentions=(0.5,),
        coverage_threshold=0.8,
        bootstrap_repetitions=200,
        bootstrap_seed=7,
    )

    assert summary["eligible_parts"] == {"recolor@0.5": ["upper"]}
    assert {path.name for path in output_dir.iterdir()} == {
        "coverage_table.csv",
        "part_table.csv",
        "subject_table.csv",
        "formal_table.csv",
        "paired_statistics.csv",
        "summary.json",
    }
    with (output_dir / "formal_table.csv").open("r", encoding="utf-8", newline="") as handle:
        formal = list(csv.DictReader(handle))
    a5 = next(row for row in formal if row["method"] == "a5")
    voting = next(row for row in formal if row["method"] == "voting")
    assert float(a5["pooled_outer_burden_mean"]) == pytest.approx(0.2)
    assert float(voting["pooled_outer_burden_mean"]) == pytest.approx(0.4)
    assert json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))["coverage_threshold"] == 0.8
