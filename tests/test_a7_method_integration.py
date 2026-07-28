import hashlib

import numpy as np
import pytest


def test_real_edit_weight_resolution_reads_static_a7_without_changing_a5():
    from utils.semantic_real_editing import resolve_method_weights

    labels = np.array([0, 1, 0], dtype=np.int16)
    raw = {"part_label": labels}
    voting = {"part_label": labels}
    a5 = {
        "part_label": labels,
        "soft_edit_weights": np.array(
            [[0.8, 0.1, 0, 0, 0, 0], [0.2, 0.7, 0, 0, 0, 0], [0.4, 0.2, 0, 0, 0, 0]],
            dtype=np.float32,
        ),
    }
    a7 = {
        "part_label": labels,
        "soft_edit_weights": np.array(
            [[0.6, 0.1, 0, 0, 0, 0], [0.1, 0.5, 0, 0, 0, 0], [0.3, 0.2, 0, 0, 0, 0]],
            dtype=np.float32,
        ),
    }

    before = resolve_method_weights(
        raw, voting, a5, method="a5", part="hair", threshold=0.15
    )
    after = resolve_method_weights(
        raw, voting, a5, a7_bank=a7, method="a5", part="hair", threshold=0.15
    )
    a7_values = resolve_method_weights(
        raw, voting, a5, a7_bank=a7, method="a7", part="hair", threshold=0.15
    )

    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(a7_values, np.array([0.6, 0.0, 0.3], dtype=np.float32))


def test_a7_bank_contract_validation_checks_base_and_contract_fingerprints(tmp_path):
    from utils.frozen_semantic_method import validate_a7_bank_against_contract

    a5_path = tmp_path / "a5.npz"
    a5_path.write_bytes(b"frozen-a5")
    a5_sha = hashlib.sha256(b"frozen-a5").hexdigest()
    contract = {
        "_fingerprint": "b" * 64,
        "base_method_freeze_fingerprint": "a" * 64,
    }
    bank = {
        "method_id": np.array("A7"),
        "base_method": np.array("A5"),
        "base_bank_sha256": np.array(a5_sha),
        "base_method_freeze_fingerprint": np.array("a" * 64),
        "a7_contract_fingerprint": np.array("b" * 64),
        "output_bank_fingerprint": np.array("c" * 64),
    }

    provenance = validate_a7_bank_against_contract(
        bank, contract=contract, a5_bank_path=a5_path
    )
    assert provenance["canonical_selection_fixed_across_frames"] is True
    assert provenance["a7_bank_fingerprint"] == "c" * 64

    bank["a7_contract_fingerprint"] = np.array("0" * 64)
    with pytest.raises(ValueError, match="contract fingerprint"):
        validate_a7_bank_against_contract(bank, contract=contract, a5_bank_path=a5_path)


def test_temporal_summary_reports_required_a7_metrics_and_static_fingerprint():
    from tools.render_semantic_temporal_stability import (
        method_weight_fingerprint,
        summarize_temporal_rows,
    )

    weights = np.array([0.2, 0.7, 0.0], dtype=np.float32)
    assert method_weight_fingerprint(weights) == method_weight_fingerprint(weights.copy())
    rows = []
    for frame, outer, boundary, adaptive_outer, adaptive_boundary, area, cx in (
        (0, 0.2, 0.4, 0.1, 0.3, 10.0, 2.0),
        (1, 0.4, 0.2, 0.2, 0.2, 14.0, 4.0),
        (2, 0.3, 0.3, 0.2, 0.1, 12.0, 3.0),
    ):
        rows.append(
            {
                "frame": frame,
                "method": "a7",
                "part": "upper",
                "edit_outer_delta_mean": outer,
                "edit_boundary_outer_delta_mean": boundary,
                "adaptive_edit_outer_delta_mean": adaptive_outer,
                "adaptive_edit_boundary_outer_delta_mean": adaptive_boundary,
                "edit_target_delta_mean": 1.0,
                "selection_mass": area,
                "selection_centroid_x": cx,
                "selection_centroid_y": 5.0,
                "screen_recall": 1.0 if frame != 1 else 0.0,
                "adaptive_strength": 0.5 + 0.1 * frame,
                "canonical_selection_fingerprint": "f" * 64,
            }
        )

    summary = summarize_temporal_rows(rows)
    item = summary["a7"]["upper"]
    for key in (
        "fixed_strength_outer_flicker",
        "fixed_strength_boundary_flicker",
        "adaptive_matched_retention_outer_flicker",
        "adaptive_matched_retention_boundary_flicker",
        "visibility_aware_response_flicker",
        "selection_area_cv",
        "selection_centroid_std",
        "visibility_transition_rate",
        "adaptive_strength_sequence",
    ):
        assert key in item
    assert item["canonical_selection_fingerprint"] == "f" * 64
    assert item["visibility_transition_rate"] == pytest.approx(0.0)


def test_adaptive_strength_is_selected_once_from_common_a5_sequence_reference():
    from tools.render_semantic_temporal_stability import select_global_adaptive_metrics

    rows = [
        {"method": method, "part": "upper", "frame": frame}
        for method in ("a5", "a7")
        for frame in (0, 1)
    ]
    sweeps = {}
    for row in rows:
        method = row["method"]
        frame = row["frame"]
        scale = 1.0 if method == "a5" else 0.8
        sweeps[(method, "upper", frame)] = [
            (0.5, {"target_delta_sum": scale * (frame + 1), "outer_delta_mean": 0.2}),
            (1.0, {"target_delta_sum": 2 * scale * (frame + 1), "outer_delta_mean": 0.4}),
        ]

    select_global_adaptive_metrics(
        rows,
        sweeps,
        target_retention=0.5,
        reference_method="a5",
    )

    assert {row["adaptive_strength"] for row in rows if row["method"] == "a5"} == {0.5}
    assert {row["adaptive_strength"] for row in rows if row["method"] == "a7"} == {0.5}
    assert all(row["adaptive_reference_method"] == "a5" for row in rows)


def test_paper_evaluator_resolves_a7_soft_weights():
    from tools.evaluate_semantic_editing_paper_protocol import (
        METHOD_SPECS,
        resolve_baseline_point_weights,
    )

    raw = {
        "part_label": np.array([0, 1], dtype=np.int16),
        "semantic_probs": np.ones((2, 6), dtype=np.float32) / 6,
        "confidence": np.ones(2, dtype=np.float32),
    }
    a7 = {"soft_edit_weights": np.array([[0.3] * 6, [0.7] * 6], dtype=np.float32)}
    weights, support, metadata = resolve_baseline_point_weights(
        "A7",
        raw_trained_bank=raw,
        evidence_bank=raw,
        voting_bank=raw,
        a7_bank=a7,
        part_index=0,
    )

    assert METHOD_SPECS["A7"]["persistent_asset"] is True
    np.testing.assert_array_equal(weights, np.array([0.3, 0.7], dtype=np.float32))
    assert support is None
    assert metadata["weight_field"] == "a7_soft_edit_weights"


def test_real_edit_aggregate_uses_common_a5_retention_and_coverage_threshold():
    from tools.render_semantic_real_editing_paper_suite import aggregate_real_edit_metrics

    rows = []
    for method, responses in (
        ("a5", {0.5: [2.0, 2.0], 1.0: [4.0, 4.0]}),
        ("a7", {0.5: [2.0, 0.1], 1.0: [4.0, 0.2]}),
    ):
        for strength, values in responses.items():
            for view, target in enumerate(values):
                rows.append(
                    {
                        "view": f"v{view}",
                        "method": method,
                        "task": "recolor",
                        "part": "upper",
                        "edit_strength": strength,
                        "target_delta_sum": target,
                        "outer_delta_sum": target * 0.25,
                        "boundary_outer_delta_sum": target * 0.125,
                    }
                )
    aggregates = aggregate_real_edit_metrics(
        rows, target_retention=0.5, coverage_response_fraction=0.8
    )
    aggregate = next(row for row in aggregates if row["method"] == "a7")

    assert aggregate["pooled_outer_burden"] == pytest.approx(0.25)
    assert aggregate["pooled_boundary_burden"] == pytest.approx(0.125)
    assert aggregate["coverage_rate"] == pytest.approx(0.5)
    assert aggregate["selected_strength"] == pytest.approx(1.0)
    assert aggregate["reference_method"] == "a5"


def test_spatial_guard_aggregation_reports_required_burdens_and_coverage():
    from tools.evaluate_semantic_editing_paper_protocol import (
        aggregate_spatial_guard_rows,
    )

    rows = [
        {
            "baseline": "A7",
            "part": "upper",
            "target_activation": 4.0,
            "outer_activation": 1.0,
            "boundary_activation": 0.5,
            "selection_activation": 5.0,
            "recall": 0.9,
        },
        {
            "baseline": "A7",
            "part": "upper",
            "target_activation": 2.0,
            "outer_activation": 1.0,
            "boundary_activation": 0.25,
            "selection_activation": 3.0,
            "recall": 0.7,
        },
    ]
    aggregate = aggregate_spatial_guard_rows(rows, coverage_recall_threshold=0.8)[0]

    assert aggregate["pooled_outer_burden"] == pytest.approx(2.0 / 6.0)
    assert aggregate["pooled_boundary_burden"] == pytest.approx(0.75 / 6.0)
    assert aggregate["pooled_selection_burden"] == pytest.approx(8.0 / 6.0)
    assert aggregate["coverage_rate"] == pytest.approx(0.5)
