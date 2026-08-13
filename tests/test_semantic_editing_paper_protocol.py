import csv
import json
from pathlib import Path

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


def test_ablation_specs_define_incremental_a0_a6_chain():
    from tools.evaluate_semantic_editing_paper_protocol import ABLATION_SPECS

    assert list(ABLATION_SPECS) == ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    assert [ABLATION_SPECS[name]["name"] for name in ABLATION_SPECS] == [
        "raw_trained_hard_label",
        "raw_semantic_probability",
        "raw_probability_confidence",
        "raw_confidence_margin_reliable",
        "multiview_voting_posterior",
        "footprint_evidence_target",
        "target_support_decomposition",
    ]


def test_method_specs_register_a7_without_changing_frozen_a0_a6_chain():
    from tools.evaluate_semantic_editing_paper_protocol import ABLATION_SPECS, METHOD_SPECS

    assert list(ABLATION_SPECS) == ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    assert METHOD_SPECS["A7"] == {
        "name": "temporal_reliable_static_asset",
        "oracle": False,
        "persistent_asset": True,
    }


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


def test_formal_baselines_route_raw_and_evidence_banks_independently():
    from tools.evaluate_semantic_editing_paper_protocol import resolve_baseline_point_weights

    raw = _trained_bank()
    raw["editable_label"] = np.array([0, 1], dtype=np.int16)
    evidence = _trained_bank()
    evidence["editable_label"] = np.array([1, 1], dtype=np.int16)
    evidence["edit_target_weights"] = np.array(
        [[0.11, 0.89], [0.22, 0.78]], dtype=np.float32
    )
    evidence["edit_support_weights"] = np.array(
        [[0.03, 0.07], [0.04, 0.06]], dtype=np.float32
    )
    voting = {
        "editable_label": np.array([1, 1], dtype=np.int16),
        "semantic_probs": np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32),
    }

    b2, support, _ = resolve_baseline_point_weights(
        "B2",
        raw_trained_bank=raw,
        evidence_bank=evidence,
        voting_bank=voting,
        part_index=0,
    )
    assert np.array_equal(b2, [1.0, 0.0])
    assert support is None

    b5, support, _ = resolve_baseline_point_weights(
        "B5",
        raw_trained_bank=raw,
        evidence_bank=evidence,
        voting_bank=voting,
        part_index=0,
    )
    assert np.allclose(b5, [0.11, 0.22])
    assert np.allclose(support, [0.03, 0.04])


@pytest.mark.parametrize(
    "ablation,expected,support_expected",
    [
        ("A0", [1.0, 0.0], None),
        ("A1", [0.7, 0.2], None),
        ("A2", [0.49, 0.16], None),
        ("A4", [0.2, 0.4], None),
        ("A5", [0.11, 0.22], None),
        ("A6", [0.11, 0.22], [0.03, 0.04]),
    ],
)
def test_ablation_weights_use_expected_component_stage(ablation, expected, support_expected):
    from tools.evaluate_semantic_editing_paper_protocol import resolve_baseline_point_weights

    raw = _trained_bank()
    raw["editable_label"] = np.array([0, 1], dtype=np.int16)
    evidence = _trained_bank()
    evidence["edit_target_weights"] = np.array(
        [[0.11, 0.89], [0.22, 0.78]], dtype=np.float32
    )
    evidence["edit_support_weights"] = np.array(
        [[0.03, 0.07], [0.04, 0.06]], dtype=np.float32
    )
    voting = {
        "editable_label": np.array([1, 1], dtype=np.int16),
        "semantic_probs": np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32),
    }

    weights, support, _ = resolve_baseline_point_weights(
        ablation,
        raw_trained_bank=raw,
        evidence_bank=evidence,
        voting_bank=voting,
        part_index=0,
    )

    assert np.allclose(weights, expected)
    if support_expected is None:
        assert support is None
    else:
        assert np.allclose(support, support_expected)


def test_a5_uses_separate_footprint_bank_while_a6_uses_support_aware_bank():
    from tools.evaluate_semantic_editing_paper_protocol import resolve_baseline_point_weights

    raw = _trained_bank()
    voting = {"semantic_probs": raw["semantic_probs"]}
    footprint = _trained_bank()
    footprint["soft_edit_weights"] = np.array(
        [[0.31, 0.69], [0.42, 0.58]], dtype=np.float32
    )
    evidence = _trained_bank()
    evidence["edit_target_weights"] = np.array(
        [[0.11, 0.89], [0.22, 0.78]], dtype=np.float32
    )
    evidence["edit_support_weights"] = np.array(
        [[0.03, 0.07], [0.04, 0.06]], dtype=np.float32
    )

    a5, a5_support, _ = resolve_baseline_point_weights(
        "A5",
        raw_trained_bank=raw,
        footprint_bank=footprint,
        evidence_bank=evidence,
        voting_bank=voting,
        part_index=0,
    )
    a6, a6_support, _ = resolve_baseline_point_weights(
        "A6",
        raw_trained_bank=raw,
        footprint_bank=footprint,
        evidence_bank=evidence,
        voting_bank=voting,
        part_index=0,
    )

    assert np.allclose(a5, [0.31, 0.42])
    assert a5_support is None
    assert np.allclose(a6, [0.11, 0.22])
    assert np.allclose(a6_support, [0.03, 0.04])


def test_ablation_a3_uses_confidence_margin_and_reliability():
    from tools.evaluate_semantic_editing_paper_protocol import resolve_baseline_point_weights
    from utils.part_label_bank import compute_soft_edit_weights

    raw = _trained_bank()
    expected = compute_soft_edit_weights(
        semantic_probs=raw["semantic_probs"],
        confidence=raw["confidence"],
        semantic_margin=raw["semantic_margin"],
        reliable_mask=raw["reliable_mask"],
    )[:, 0]
    weights, support, _ = resolve_baseline_point_weights(
        "A3",
        raw_trained_bank=raw,
        evidence_bank=_trained_bank(),
        voting_bank={"semantic_probs": raw["semantic_probs"]},
        part_index=0,
    )
    assert np.allclose(weights, expected)
    assert support is None


def test_identical_voting_and_raw_hard_labels_are_rejected():
    from tools.evaluate_semantic_editing_paper_protocol import validate_hard_baseline_independence

    labels = np.array([0, 1, 1], dtype=np.int16)
    with pytest.raises(ValueError, match="B1 and B2 hard-label predictions are identical"):
        validate_hard_baseline_independence(
            raw_trained_bank={"editable_label": labels.copy()},
            voting_bank={"editable_label": labels.copy()},
            requested_baselines=("B1", "B2"),
        )


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


def test_b5_fallback_part_uses_raw_semantic_probability():
    from tools.evaluate_semantic_editing_paper_protocol import resolve_baseline_point_weights

    trained = _trained_bank()
    weights, support, metadata = resolve_baseline_point_weights(
        "B5",
        trained_bank=trained,
        voting_bank=None,
        part_index=0,
        part_name="skin",
        b5_fallback_parts={"skin"},
    )

    assert np.allclose(weights, [0.7, 0.2])
    assert support is None
    assert metadata["weight_field"] == "semantic_probs_fallback"
    assert metadata["b5_fallback_applied"] is True


def test_b5_nonfallback_part_keeps_calibrated_target_and_support():
    from tools.evaluate_semantic_editing_paper_protocol import resolve_baseline_point_weights

    trained = _trained_bank()
    weights, support, metadata = resolve_baseline_point_weights(
        "B5",
        trained_bank=trained,
        voting_bank=None,
        part_index=0,
        part_name="face",
        b5_fallback_parts={"skin"},
    )

    assert np.allclose(weights, [0.35, 0.10])
    assert np.allclose(support, [0.07, 0.02])
    assert metadata["weight_field"] == "edit_target_weights"
    assert metadata["b5_fallback_applied"] is False


def test_b5_fallback_part_uses_dedicated_threshold():
    from tools.evaluate_semantic_editing_paper_protocol import resolve_part_threshold

    assert resolve_part_threshold(
        "B5",
        part_name="skin",
        default_threshold=0.1,
        b5_fallback_parts={"skin"},
        b5_fallback_threshold=0.5,
    ) == pytest.approx(0.5)
    assert resolve_part_threshold(
        "B5",
        part_name="face",
        default_threshold=0.1,
        b5_fallback_parts={"skin"},
        b5_fallback_threshold=0.5,
    ) == pytest.approx(0.1)


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
            "--raw-trained-bank", "raw.npz",
            "--trained-bank", "trained.npz",
            "--voting-bank", "voting.npz",
            "--checkpoint", "ckpt.pth",
            "--asset-root", "assets",
            "--output-dir", "out",
            "--soft-threshold", "0.05",
            "--support-threshold", "0.3",
            "--boundary-radius", "6",
        ]
    )

    assert args.soft_threshold == pytest.approx(0.05)
    assert args.support_threshold == pytest.approx(0.3)
    assert args.boundary_radius == 6
    assert str(args.raw_trained_bank) == "raw.npz"


def test_test_split_rejects_metric_overrides():
    from tools.evaluate_semantic_editing_paper_protocol import resolve_evaluation_parameters

    with pytest.raises(ValueError, match="test evaluation forbids"):
        resolve_evaluation_parameters(
            protocol_split="test",
            selected_config={"soft_threshold": 0.2, "boundary_radius": 4},
            soft_threshold_override=0.05,
            boundary_radius_override=None,
        )


def test_test_split_rejects_support_threshold_override():
    from tools.evaluate_semantic_editing_paper_protocol import resolve_support_threshold

    with pytest.raises(ValueError, match="test evaluation forbids"):
        resolve_support_threshold(
            protocol_split="test",
            selected_config={"support_threshold": 0.3},
            support_threshold_override=0.2,
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


def test_matched_retention_reference_accepts_explicit_ablation_reference():
    from tools.evaluate_semantic_editing_paper_protocol import resolve_retention_reference

    baseline, activation = resolve_retention_reference(
        {
            "A0": [{"target_activation": 20.0}, {"target_activation": 80.0}],
            "A4": [{"target_activation": 30.0}, {"target_activation": 120.0}],
        },
        preferred_baseline="A0",
    )

    assert baseline == "A0"
    assert activation == pytest.approx(80.0)


def test_fixed_soft_curve_sweeps_edit_strength_at_frozen_threshold():
    from tools.evaluate_semantic_editing_paper_protocol import curve_settings_for_baseline

    protocol = {
        "matched_retention_targets": [0.3, 0.5, 0.6, 1.0],
        "edit_strength_grid": [0.3, 0.5, 0.6, 1.0],
        "validation_grid": {"soft_thresholds": [0.5, 0.2, 0.05]},
    }

    settings = curve_settings_for_baseline(
        "B5",
        protocol=protocol,
        fixed_soft_threshold=0.35,
        soft_strength_sweep=True,
    )

    assert settings == [(0.35, 0.3), (0.35, 0.5), (0.35, 0.6), (0.35, 1.0)]


def test_fixed_soft_curve_uses_edit_strength_grid_not_reporting_targets():
    from tools.evaluate_semantic_editing_paper_protocol import curve_settings_for_baseline

    protocol = {
        "matched_retention_targets": [0.5, 0.6],
        "edit_strength_grid": [0.05, 0.1, 0.2, 0.4, 1.0],
        "validation_grid": {"soft_thresholds": [0.5, 0.2, 0.05]},
    }

    assert curve_settings_for_baseline(
        "A4",
        protocol=protocol,
        fixed_soft_threshold=0.1,
        soft_strength_sweep=True,
    ) == [(0.1, value) for value in (0.05, 0.1, 0.2, 0.4, 1.0)]


def test_normalized_curve_reports_reference_normalized_leakage_burden():
    from tools.evaluate_semantic_editing_paper_protocol import _normalize_curve_retention

    rows = [
        {
            "baseline": "B5",
            "target_activation": 50.0,
            "outer_activation": 20.0,
            "actionable_activation": 10.0,
            "raw_leakage": 0.4,
            "actionable_leakage": 0.2,
        }
    ]

    row = _normalize_curve_retention(rows, reference_activation=100.0)[0]

    assert row["retention"] == pytest.approx(0.5)
    assert row["raw_leakage"] == pytest.approx(0.2)
    assert row["actionable_leakage"] == pytest.approx(0.1)
    assert row["raw_leakage_ratio"] == pytest.approx(0.4)
    assert row["actionable_leakage_ratio"] == pytest.approx(0.2)


def test_a0_curve_uses_hard_threshold_and_edit_strength():
    from tools.evaluate_semantic_editing_paper_protocol import curve_settings_for_baseline

    protocol = {
        "matched_retention_targets": [0.5, 0.6, 1.0],
        "validation_grid": {"soft_thresholds": [0.5, 0.2, 0.05]},
    }

    assert curve_settings_for_baseline("A0", protocol=protocol) == [
        (0.5, 0.5),
        (0.5, 0.6),
        (0.5, 1.0),
    ]


def test_parser_accepts_frozen_main_method_contract(tmp_path):
    from tools.evaluate_semantic_editing_paper_protocol import parse_args

    args = parse_args(
        [
            "--protocol",
            str(tmp_path / "protocol.json"),
            "--protocol-split",
            "validation",
            "--raw-trained-bank",
            str(tmp_path / "raw.npz"),
            "--trained-bank",
            str(tmp_path / "evidence.npz"),
            "--voting-bank",
            str(tmp_path / "voting.npz"),
            "--footprint-bank",
            str(tmp_path / "footprint.npz"),
            "--method-freeze",
            "configs/semantic/frozen_a5_main_method_v1.json",
            "--checkpoint",
            str(tmp_path / "ckpt.pth"),
            "--asset-root",
            str(tmp_path / "assets"),
            "--output-dir",
            str(tmp_path / "output"),
            "--baselines",
            "A0",
            "A5",
            "A6",
        ]
    )

    assert args.method_freeze == Path(
        "configs/semantic/frozen_a5_main_method_v1.json"
    )


def test_explicit_record_selection_is_exact_and_preserves_requested_order():
    from tools.evaluate_semantic_editing_paper_protocol import select_explicit_records

    records = [
        {"image_name": "c21_f00180", "cam_id": 21, "frame_id": 180},
        {"image_name": "c22_f00180", "cam_id": 22, "frame_id": 180},
        {"image_name": "c23_f00180", "cam_id": 23, "frame_id": 180},
    ]

    selected = select_explicit_records(
        records,
        ["c23_f00180", "c21_f00180"],
    )

    assert [row["image_name"] for row in selected] == ["c23_f00180", "c21_f00180"]
    with pytest.raises(ValueError, match="missing explicit records"):
        select_explicit_records(records, ["c24_f00180"])
    with pytest.raises(ValueError, match="duplicate explicit record"):
        select_explicit_records(records, ["c21_f00180", "c21_f00180"])


def test_fixed_operating_point_rows_scale_activation_and_use_reference_burden():
    from tools.evaluate_semantic_editing_paper_protocol import build_fixed_operating_point_rows

    spatial_rows = [
        {
            "baseline": "B1",
            "view": "c21_f00180",
            "part": "hair",
            "target_activation": 10.0,
            "outer_activation": 4.0,
            "boundary_activation": 2.0,
            "allowed_adjacent_activation": 1.0,
            "actionable_outer_activation": 3.0,
            "selected_count": 10,
            "observed_footprint_count": 9,
        },
        {
            "baseline": "A5",
            "view": "c21_f00180",
            "part": "hair",
            "target_activation": 8.0,
            "outer_activation": 2.0,
            "boundary_activation": 1.5,
            "allowed_adjacent_activation": 1.0,
            "actionable_outer_activation": 1.0,
            "selected_count": 7,
            "observed_footprint_count": 7,
        },
    ]
    quality_rows = [
        {
            "baseline": "A5",
            "view": "c21_f00180",
            "part": "hair",
            "iou": 0.7,
            "boundary_f1": 0.8,
            "target_empty": False,
        }
    ]

    row = build_fixed_operating_point_rows(
        spatial_rows=spatial_rows,
        quality_rows=quality_rows,
        operating_point={
            "method": "Ours",
            "baseline": "A5",
            "reference_baseline": "B1",
            "threshold": 0.2,
            "edit_strength": 0.75,
            "retention": 0.6,
            "target_retention_feasible": True,
        },
    )[0]

    assert row["target_activation"] == pytest.approx(6.0)
    assert row["outer_activation"] == pytest.approx(1.5)
    assert row["actionable_outer_activation"] == pytest.approx(0.75)
    assert row["raw_leakage_ratio"] == pytest.approx(0.25)
    assert row["actionable_leakage_ratio"] == pytest.approx(0.125)
    assert row["raw_leakage"] == pytest.approx(0.15)
    assert row["actionable_leakage"] == pytest.approx(0.075)
    assert row["view_retention"] == pytest.approx(0.6)
    assert row["iou"] == pytest.approx(0.7)
    assert row["boundary_f1"] == pytest.approx(0.8)


def test_parser_accepts_fixed_operating_point_export_options(tmp_path):
    from tools.evaluate_semantic_editing_paper_protocol import parse_args

    args = parse_args(
        [
            "--protocol",
            str(tmp_path / "protocol.json"),
            "--protocol-split",
            "test",
            "--frozen-config",
            str(tmp_path / "frozen.json"),
            "--raw-trained-bank",
            str(tmp_path / "raw.npz"),
            "--trained-bank",
            str(tmp_path / "evidence.npz"),
            "--voting-bank",
            str(tmp_path / "voting.npz"),
            "--checkpoint",
            str(tmp_path / "ckpt.pth"),
            "--asset-root",
            str(tmp_path / "assets"),
            "--output-dir",
            str(tmp_path / "output"),
            "--record-list",
            str(tmp_path / "records.json"),
            "--fixed-operating-point",
            str(tmp_path / "operating_point.json"),
            "--per-view-spatial-output",
            str(tmp_path / "rows.csv"),
        ]
    )

    assert args.record_list == tmp_path / "records.json"
    assert args.fixed_operating_point == tmp_path / "operating_point.json"
    assert args.per_view_spatial_output == tmp_path / "rows.csv"
