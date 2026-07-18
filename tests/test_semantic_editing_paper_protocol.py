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
