import csv
from pathlib import Path

import pytest


def _rows(subject="377", method="voting", part="upper", values=(0.5, 0.7, 0.6)):
    rows = []
    for frame, value in enumerate(values):
        rows.append(
            {
                "subject": subject,
                "method": method,
                "part": part,
                "frame": frame,
                "target_pixel_count": 10,
                "screen_soft_iou": value,
                "screen_hard_iou": value,
                "screen_precision": value,
                "screen_recall": value,
                "selection_leakage_ratio": 1.0 - value,
                "edit_target_delta_mean": value,
                "edit_outer_delta_mean": 1.0 - value,
                "edit_boundary_outer_delta_mean": 1.0 - value,
                "edit_outer_to_target_delta_ratio": (1.0 - value) / value,
            }
        )
    return rows


def test_sequence_summary_reports_visibility_and_temporal_statistics():
    from tools.summarize_semantic_temporal_stability import summarize_sequences

    rows = _rows()
    rows.append({**rows[-1], "frame": 3, "target_pixel_count": 0})
    summary = summarize_sequences(rows)

    assert len(summary) == 1
    item = summary[0]
    assert item["frame_count"] == 4
    assert item["supported_frame_count"] == 3
    assert item["target_present_rate"] == pytest.approx(0.75)
    assert item["screen_soft_iou_mean"] == pytest.approx(0.6)
    assert item["screen_soft_iou_adjacent_flicker"] == pytest.approx((0.2 + 0.1) / 2 / 0.6)


def test_subject_equal_aggregate_keeps_subjects_independent():
    from tools.summarize_semantic_temporal_stability import aggregate_sequences, summarize_sequences

    rows = []
    rows += _rows(subject="377", method="a5", values=(0.7, 0.7))
    rows += _rows(subject="386", method="a5", values=(0.9, 0.9))
    rows += _rows(subject="377", method="voting", values=(0.5, 0.5))
    rows += _rows(subject="386", method="voting", values=(0.6, 0.6))
    part_rows, overall_rows = aggregate_sequences(summarize_sequences(rows))

    a5_part = next(row for row in part_rows if row["method"] == "a5")
    a5_overall = next(row for row in overall_rows if row["method"] == "a5")
    assert a5_part["subject_count"] == 2
    assert a5_part["screen_soft_iou_mean"] == pytest.approx(0.8)
    assert a5_overall["screen_soft_iou_mean"] == pytest.approx(0.8)


def test_paired_statistics_compare_a5_with_voting_by_subject():
    from tools.summarize_semantic_temporal_stability import paired_statistics

    overall = [
        {"subject": "377", "method": "voting", "screen_soft_iou_mean": 0.5, "selection_leakage_ratio_mean": 0.3},
        {"subject": "377", "method": "a5", "screen_soft_iou_mean": 0.6, "selection_leakage_ratio_mean": 0.2},
        {"subject": "386", "method": "voting", "screen_soft_iou_mean": 0.4, "selection_leakage_ratio_mean": 0.4},
        {"subject": "386", "method": "a5", "screen_soft_iou_mean": 0.5, "selection_leakage_ratio_mean": 0.25},
    ]
    stats = paired_statistics(
        overall,
        metric_directions={"screen_soft_iou_mean": True, "selection_leakage_ratio_mean": False},
        repetitions=200,
        seed=7,
    )

    assert {row["metric"] for row in stats} == {"screen_soft_iou_mean", "selection_leakage_ratio_mean"}
    assert all(row["subject_count"] == 2 for row in stats)
    assert all(row["wins"] == 2 for row in stats)


def test_read_metrics_rejects_non_finite_values(tmp_path: Path):
    from tools.summarize_semantic_temporal_stability import read_metrics

    path = tmp_path / "metrics.csv"
    rows = _rows(values=(0.5,))
    rows[0]["screen_soft_iou"] = "nan"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="non-finite"):
        read_metrics(path)
