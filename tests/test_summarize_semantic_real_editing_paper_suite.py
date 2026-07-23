import math


def test_add_voting_target_retention_uses_matched_record_reference():
    from tools.summarize_semantic_real_editing_paper_suite import add_voting_target_retention

    rows = [
        {"subject": "377", "view": "v1", "part": "hair", "task": "recolor", "edit_strength": 1.0, "method": "voting", "target_delta_sum": 10.0},
        {"subject": "377", "view": "v1", "part": "hair", "task": "recolor", "edit_strength": 1.0, "method": "a5", "target_delta_sum": 8.0},
        {"subject": "377", "view": "v1", "part": "hair", "task": "recolor", "edit_strength": 1.0, "method": "raw_hard", "target_delta_sum": 12.0},
    ]

    enriched = add_voting_target_retention(rows)

    by_method = {row["method"]: row["target_retention_vs_voting"] for row in enriched}
    assert by_method == {"voting": 1.0, "a5": 0.8, "raw_hard": 1.2}


def test_subject_equal_aggregate_does_not_weight_subject_by_row_count():
    from tools.summarize_semantic_real_editing_paper_suite import subject_equal_aggregate

    rows = [
        {"subject": "377", "method": "a5", "task": "recolor", "outer_delta_mean": 1.0},
        {"subject": "377", "method": "a5", "task": "recolor", "outer_delta_mean": 3.0},
        {"subject": "386", "method": "a5", "task": "recolor", "outer_delta_mean": 10.0},
    ]

    subject_rows, aggregate_rows = subject_equal_aggregate(rows, metrics=["outer_delta_mean"])

    assert {row["subject"]: row["outer_delta_mean"] for row in subject_rows} == {"377": 2.0, "386": 10.0}
    assert aggregate_rows[0]["outer_delta_mean_mean"] == 6.0


def test_paired_bootstrap_is_deterministic_and_reports_wins():
    from tools.summarize_semantic_real_editing_paper_suite import paired_bootstrap

    first = paired_bootstrap([1.0, 2.0, 3.0], seed=20260723, repetitions=1000, lower_is_better=True)
    second = paired_bootstrap([1.0, 2.0, 3.0], seed=20260723, repetitions=1000, lower_is_better=True)

    assert first == second
    assert first["mean_delta"] == 2.0
    assert first["wins"] == 0
    assert first["losses"] == 3
    assert not math.isnan(first["bootstrap_ci95_low"])
