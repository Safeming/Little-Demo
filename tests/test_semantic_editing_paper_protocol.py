import csv
import json

import numpy as np
import pytest


def _trained_bank():
    semantic_probs = np.array(
        [
            [0.7, 0.3, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    return {
        "part_label": np.array([0, 1], dtype=np.int16),
        "editable_label": np.array([1, 1], dtype=np.int16),
        "semantic_probs": semantic_probs,
        "confidence": np.array([0.7, 0.8], dtype=np.float32),
        "semantic_margin": np.array([0.4, 0.6], dtype=np.float32),
        "reliable_mask": np.array([1, 0], dtype=np.uint8),
        "edit_target_weights": semantic_probs * 0.5,
        "edit_support_weights": semantic_probs * 0.1,
    }


def test_baseline_specs_label_parser_as_online_oracle():
    from tools.evaluate_semantic_editing_paper_protocol import BASELINE_SPECS

    assert list(BASELINE_SPECS) == ["B0", "B1", "B2", "B3", "B4", "B5"]
    assert BASELINE_SPECS["B0"]["oracle"] is True
    assert BASELINE_SPECS["B0"]["persistent_asset"] is False
    assert all(BASELINE_SPECS[name]["oracle"] is False for name in ("B1", "B2", "B3", "B4", "B5"))


@pytest.mark.parametrize(
    "baseline,expected",
    [
        ("B1", [1.0, 0.0]),
        ("B2", [0.0, 0.0]),
        ("B3", [0.7, 0.2]),
        ("B5", [0.35, 0.10]),
    ],
)
def test_resolve_baseline_point_weights_uses_expected_bank_field(baseline, expected):
    from tools.evaluate_semantic_editing_paper_protocol import resolve_baseline_point_weights

    trained = _trained_bank()
    voting = {"editable_label": np.array([0, 1], dtype=np.int16)}

    weights, support, metadata = resolve_baseline_point_weights(
        baseline,
        trained_bank=trained,
        voting_bank=voting,
        part_index=0,
    )

    assert np.allclose(weights, expected)
    assert metadata["baseline"] == baseline
    if baseline == "B5":
        assert np.allclose(support, [0.07, 0.02])
    else:
        assert support is None


def test_confidence_margin_baseline_recomputes_reliability_weight():
    from tools.evaluate_semantic_editing_paper_protocol import resolve_baseline_point_weights
    from utils.part_label_bank import compute_soft_edit_weights

    trained = _trained_bank()
    expected = compute_soft_edit_weights(
        semantic_probs=trained["semantic_probs"],
        confidence=trained["confidence"],
        semantic_margin=trained["semantic_margin"],
        reliable_mask=trained["reliable_mask"],
    )[:, 0]

    weights, support, _metadata = resolve_baseline_point_weights(
        "B4",
        trained_bank=trained,
        voting_bank=None,
        part_index=0,
    )

    assert np.allclose(weights, expected)
    assert support is None


def test_voting_baseline_requires_voting_bank():
    from tools.evaluate_semantic_editing_paper_protocol import resolve_baseline_point_weights

    with pytest.raises(ValueError, match="B1 requires a projected multi-view voting bank"):
        resolve_baseline_point_weights(
            "B1",
            trained_bank=_trained_bank(),
            voting_bank=None,
            part_index=0,
        )


def test_parser_oracle_prediction_uses_current_view_part_mask():
    from tools.evaluate_semantic_editing_paper_protocol import resolve_parser_oracle_prediction

    mask = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    prediction = resolve_parser_oracle_prediction({"face": mask}, "face")

    assert np.array_equal(prediction, mask)


def test_write_baseline_reports_writes_required_outputs(tmp_path):
    from tools.evaluate_semantic_editing_paper_protocol import write_baseline_reports

    result = {
        "summary": {
            "protocol_fingerprint": "proto",
            "checkpoint_fingerprint": "ckpt",
            "baseline_count": 1,
        },
        "baseline_summary": [{"baseline": "B2", "macro_miou": 0.5, "oracle": False}],
        "per_part": [{"baseline": "B2", "part": "face", "iou": 0.5}],
        "per_view": [{"baseline": "B2", "view": "v0", "part": "face", "iou": 0.5}],
        "curve": [{"baseline": "B2", "retention": 1.0, "actionable_leakage": 0.2}],
        "matched_retention": [{"baseline": "B2", "retention": 0.8, "actionable_leakage": 0.2}],
    }

    write_baseline_reports(tmp_path, result)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["protocol_fingerprint"] == "proto"
    with (tmp_path / "baseline_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["baseline"] == "B2"
    assert (tmp_path / "per_part_metrics.csv").exists()
    assert (tmp_path / "per_view_metrics.csv").exists()
    assert (tmp_path / "leakage_retention_curve.csv").exists()
    assert (tmp_path / "matched_retention.csv").exists()
    assert (tmp_path / "support_diagnostics.csv").exists()
    assert (tmp_path / "leakage_retention_curve.png").exists()
    assert (tmp_path / "per_part_iou.png").exists()


def test_rasterize_footprint_weight_map_uses_max_weight_in_overlaps():
    from tools.evaluate_semantic_editing_paper_protocol import rasterize_footprint_weight_map

    result = rasterize_footprint_weight_map(
        xy=np.array([[2.0, 2.0], [3.0, 2.0]], dtype=np.float32),
        radii=np.array([1.0, 1.0], dtype=np.float32),
        weights=np.array([0.4, 0.8], dtype=np.float32),
        image_shape=(5, 6),
        threshold=0.2,
        min_radius=1,
        max_radius=1,
    )

    assert result.shape == (5, 6)
    assert result[2, 2] == pytest.approx(0.8)
    assert result[2, 3] == pytest.approx(0.8)
    assert result[0, 0] == pytest.approx(0.0)


def test_b5_support_diagnostics_apply_support_threshold():
    from tools.evaluate_semantic_editing_paper_protocol import _support_diagnostics_for_b5

    trained = _trained_bank()
    trained["edit_support_weights"][:, 1] = np.array([0.8, 0.4], dtype=np.float32)
    face = np.zeros((7, 7), dtype=np.float32)
    face[2:5, 1:3] = 1.0
    hair = np.zeros((7, 7), dtype=np.float32)
    hair[2:5, 4:6] = 1.0
    cache = {
        "view": "c17_f000060",
        "xy": np.array([[2.0, 3.0], [4.0, 3.0]], dtype=np.float32),
        "radii": np.array([1.0, 1.0], dtype=np.float32),
        "projected": np.array([True, True]),
        "part_masks": {"face": face, "hair": hair},
        "valid_mask": np.ones((7, 7), dtype=np.float32),
    }
    protocol = {
        "parts": ["face"],
        "allowed_adjacency": {"face": ["hair"]},
        "validation_grid": {"support_thresholds": [0.2, 0.6]},
    }

    rows = _support_diagnostics_for_b5(
        caches=[cache],
        trained_bank=trained,
        protocol=protocol,
        boundary_radius=2,
    )

    assert [row["support_threshold"] for row in rows] == [0.2, 0.6]
    assert [row["selected_count"] for row in rows] == [2, 1]
    assert all("allowed_support_fraction" in row for row in rows)


def test_cached_footprint_ratios_match_reference_implementation():
    from tools.analyze_projected_soft_edit_leakage import compute_footprint_leakage_for_selection
    from tools.evaluate_semantic_editing_paper_protocol import (
        compute_footprint_ratio_arrays,
        summarize_footprint_selection_from_ratios,
    )

    xy = np.array([[2.0, 3.0], [4.0, 3.0], [-2.0, 1.0]], dtype=np.float32)
    radii = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    selected = np.array([True, True, True])
    weights = np.array([0.8, 0.4, 0.6], dtype=np.float32)
    target = np.zeros((7, 7), dtype=np.float32)
    target[2:5, 1:3] = 1.0
    adjacent = np.zeros((7, 7), dtype=np.float32)
    adjacent[2:5, 4:6] = 1.0
    valid = np.ones((7, 7), dtype=np.float32)

    reference = compute_footprint_leakage_for_selection(
        part="face",
        mode="reference",
        view_name="view",
        xy=xy,
        selected=selected,
        weights=weights,
        radii=radii,
        target_mask=target,
        valid_mask=valid,
        boundary_radius=2,
        allowed_adjacent_masks={"hair": adjacent},
    )
    ratios = compute_footprint_ratio_arrays(
        part="face",
        xy=xy,
        radii=radii,
        target_mask=target,
        valid_mask=valid,
        boundary_radius=2,
        allowed_adjacent_masks={"hair": adjacent},
    )
    cached = summarize_footprint_selection_from_ratios(selected, weights, ratios)

    for key in (
        "target_activation",
        "outer_activation",
        "boundary_activation",
        "allowed_adjacent_activation",
        "actionable_outer_activation",
    ):
        assert cached[key] == pytest.approx(reference[key], abs=1e-6)
    assert cached["selected_count"] == reference["selected_count"]


def test_parse_args_accepts_validation_metric_overrides():
    from tools.evaluate_semantic_editing_paper_protocol import parse_args

    args = parse_args(
        [
            "--protocol", "protocol.json",
            "--protocol-split", "validation",
            "--trained-bank", "trained.npz",
            "--voting-bank", "voting.npz",
            "--checkpoint", "ckpt.pth",
            "--asset-root", "assets",
            "--output-dir", "out",
            "--soft-threshold", "0.05",
            "--boundary-radius", "6",
        ]
    )

    assert args.soft_threshold == pytest.approx(0.05)
    assert args.boundary_radius == 6


def test_test_split_rejects_metric_overrides():
    from tools.evaluate_semantic_editing_paper_protocol import resolve_evaluation_parameters

    with pytest.raises(ValueError, match="test evaluation forbids"):
        resolve_evaluation_parameters(
            protocol_split="test",
            selected_config={"soft_threshold": 0.2, "boundary_radius": 4},
            soft_threshold_override=0.05,
            boundary_radius_override=None,
        )


def test_matched_retention_reference_prefers_fixed_voting_baseline():
    from tools.evaluate_semantic_editing_paper_protocol import resolve_retention_reference

    baseline, activation = resolve_retention_reference(
        {
            "B1": [{"target_activation": 10.0}, {"target_activation": 100.0}],
            "B2": [{"target_activation": 20.0}, {"target_activation": 250.0}],
        }
    )

    assert baseline == "B1"
    assert activation == pytest.approx(100.0)


def test_fixed_soft_curve_sweeps_edit_strength_at_frozen_threshold():
    from tools.evaluate_semantic_editing_paper_protocol import curve_settings_for_baseline

    protocol = {
        "matched_retention_targets": [0.3, 0.5, 0.6, 1.0],
        "validation_grid": {"soft_thresholds": [0.5, 0.2, 0.05]},
    }

    settings = curve_settings_for_baseline(
        "B5",
        protocol=protocol,
        fixed_soft_threshold=0.35,
        soft_strength_sweep=True,
    )

    assert settings == [(0.35, 0.3), (0.35, 0.5), (0.35, 0.6), (0.35, 1.0)]
