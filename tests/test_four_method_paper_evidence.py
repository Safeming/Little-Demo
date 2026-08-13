import math

import pytest


def test_build_temporal_windows_returns_three_disjoint_21_frame_windows():
    from utils.four_method_paper_evidence import build_temporal_windows

    windows = build_temporal_windows(cameras=(21, 22, 23), anchors=(180, 420, 540), radius=10)

    assert len(windows) == 9
    assert windows[0]["frames"] == list(range(170, 191))
    assert windows[-1]["frames"] == list(range(530, 551))
    assert len({(row["camera"], frame) for row in windows for frame in row["frames"]}) == 189


def test_build_temporal_windows_rejects_negative_and_overlapping_windows():
    from utils.four_method_paper_evidence import build_temporal_windows

    with pytest.raises(ValueError, match="negative frame"):
        build_temporal_windows(cameras=(21,), anchors=(5,), radius=10)
    with pytest.raises(ValueError, match="overlap"):
        build_temporal_windows(cameras=(21,), anchors=(180, 190), radius=10)


def test_resolve_frozen_operating_point_marks_gg_377_as_infeasible_at_target():
    from utils.four_method_paper_evidence import resolve_frozen_operating_point

    rows = [{"baseline": "B4", "retention": "0.4", "edit_strength": "0.5"}]

    point = resolve_frozen_operating_point(rows, method="gaussian_grouping", subject="377")

    assert point["retention"] == pytest.approx(0.4)
    assert point["target_retention_feasible"] is False
    assert point["figure_label"] == "GG\N{DAGGER}"


def test_resolve_frozen_operating_point_requires_unique_60_percent_row():
    from utils.four_method_paper_evidence import resolve_frozen_operating_point

    with pytest.raises(ValueError, match="missing frozen operating point"):
        resolve_frozen_operating_point([], method="saga", subject="377")
    duplicate = [
        {"baseline": "B4", "retention": "0.6", "edit_strength": "0.5"},
        {"baseline": "B4", "retention": "0.60", "edit_strength": "0.6"},
    ]
    with pytest.raises(ValueError, match="not unique"):
        resolve_frozen_operating_point(duplicate, method="sggs", subject="386")


def test_resolve_frozen_operating_point_rejects_invalid_strength():
    from utils.four_method_paper_evidence import resolve_frozen_operating_point

    rows = [{"baseline": "A5", "retention": "0.6", "edit_strength": "0"}]
    with pytest.raises(ValueError, match="edit_strength"):
        resolve_frozen_operating_point(rows, method="a5", subject="394")


def test_aggregate_frame_sums_activations_before_dividing_and_skips_empty_quality():
    from utils.four_method_paper_evidence import aggregate_frame

    rows = [
        {
            "part": "hair",
            "target_activation": 2.0,
            "outer_activation": 1.0,
            "actionable_outer_activation": 0.5,
            "iou": 0.8,
            "boundary_f1": 0.7,
            "target_empty": False,
        },
        {
            "part": "shoes",
            "target_activation": 8.0,
            "outer_activation": 1.0,
            "actionable_outer_activation": 0.5,
            "iou": 0.4,
            "boundary_f1": 0.3,
            "target_empty": False,
        },
        {
            "part": "face",
            "target_activation": 0.0,
            "outer_activation": 0.0,
            "actionable_outer_activation": 0.0,
            "iou": 1.0,
            "boundary_f1": 1.0,
            "target_empty": True,
        },
    ]

    result = aggregate_frame(rows)

    assert result["raw_leakage"] == pytest.approx(0.2)
    assert result["actionable_leakage"] == pytest.approx(0.1)
    assert result["macro_miou"] == pytest.approx(0.6)
    assert result["mean_boundary_f1"] == pytest.approx(0.5)
    assert result["valid_part_count"] == 2


def test_aggregate_frame_rejects_no_valid_target_parts():
    from utils.four_method_paper_evidence import aggregate_frame

    with pytest.raises(ValueError, match="no valid target parts"):
        aggregate_frame(
            [
                {
                    "target_activation": 0.0,
                    "outer_activation": 0.0,
                    "actionable_outer_activation": 0.0,
                    "iou": 1.0,
                    "boundary_f1": 1.0,
                    "target_empty": True,
                }
            ]
        )


def test_exact_block_sign_flip_uses_subject_camera_blocks():
    from utils.four_method_paper_evidence import exact_block_sign_flip

    rows = [
        {"subject": "377", "camera": 21, "difference": -1.0},
        {"subject": "377", "camera": 22, "difference": -1.0},
    ]

    result = exact_block_sign_flip(rows, value_key="difference")

    assert result["block_count"] == 2
    assert result["permutation_count"] == 4
    assert result["observed"] == pytest.approx(-1.0)
    assert result["p_value"] == pytest.approx(0.5)


def test_hierarchical_bootstrap_is_reproducible_and_finite():
    from utils.four_method_paper_evidence import hierarchical_bootstrap_paired

    rows = [
        {
            "subject": subject,
            "camera": camera,
            "frame": frame,
            "difference": float(int(subject) - 390) / 100.0 + camera / 1000.0 + frame / 10000.0,
        }
        for subject in ("377", "386", "394")
        for camera in (21, 22)
        for frame in (180, 420, 540)
    ]

    first = hierarchical_bootstrap_paired(rows, value_key="difference", iterations=500, seed=7)
    second = hierarchical_bootstrap_paired(rows, value_key="difference", iterations=500, seed=7)

    assert first == second
    assert first["iterations"] == 500
    assert first["seed"] == 7
    assert math.isfinite(first["estimate"])
    assert first["ci_low"] <= first["estimate"] <= first["ci_high"]


def test_holm_adjust_preserves_methods_and_is_monotonic():
    from utils.four_method_paper_evidence import holm_adjust

    adjusted = holm_adjust({"saga": 0.01, "gaussian_grouping": 0.04, "sggs": 0.03})

    assert set(adjusted) == {"saga", "gaussian_grouping", "sggs"}
    assert adjusted["saga"] == pytest.approx(0.03)
    assert adjusted["sggs"] == pytest.approx(0.06)
    assert adjusted["gaussian_grouping"] == pytest.approx(0.06)


def test_summarize_temporal_window_reports_level_and_adjacent_changes():
    from utils.four_method_paper_evidence import summarize_temporal_window

    rows = [
        {"frame": frame, "actionable_leakage": value, "macro_miou": 1.0 - value}
        for frame, value in zip((170, 171, 172, 173), (0.0, 1.0, 1.0, 3.0))
    ]

    summary = summarize_temporal_window(
        rows, metric_names=("actionable_leakage", "macro_miou")
    )

    assert summary["frame_count"] == 4
    assert summary["actionable_leakage_mean"] == pytest.approx(1.25)
    assert summary["actionable_leakage_std"] == pytest.approx(1.0897247358851685)
    assert summary["actionable_leakage_mean_abs_delta"] == pytest.approx(1.0)
    assert summary["actionable_leakage_p95_abs_delta"] == pytest.approx(1.9)


def test_summarize_temporal_window_rejects_nonconsecutive_frames():
    from utils.four_method_paper_evidence import summarize_temporal_window

    with pytest.raises(ValueError, match="consecutive"):
        summarize_temporal_window(
            [{"frame": 1, "value": 0.0}, {"frame": 3, "value": 1.0}],
            metric_names=("value",),
        )
