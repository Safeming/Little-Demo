import json

import pytest


def _candidate(
    threshold,
    *,
    b1_miou=0.48,
    b5_miou=0.47,
    b1_leak=0.024,
    b5_leak=0.016,
    boundary_f1=0.43,
):
    return {
        "soft_threshold": threshold,
        "support_threshold": 0.1,
        "boundary_radius": 6,
        "b1_macro_miou": b1_miou,
        "b5_macro_miou": b5_miou,
        "b5_mean_boundary_f1": boundary_f1,
        "retention_checks": [
            {
                "retention": 0.5,
                "b1_actionable_leakage": b1_leak,
                "b5_actionable_leakage": b5_leak,
            },
            {
                "retention": 0.6,
                "b1_actionable_leakage": b1_leak,
                "b5_actionable_leakage": b5_leak,
            },
        ],
    }


def test_guarded_selector_rejects_miou_and_leakage_failures():
    from tools.select_guarded_semantic_validation_config import select_guarded_candidate

    rows = [
        _candidate(0.5, b5_miou=0.43),
        _candidate(0.2, b5_leak=0.025),
    ]

    with pytest.raises(ValueError, match="no guarded validation candidate"):
        select_guarded_candidate(rows, max_miou_gap=0.02)


def test_guarded_selector_uses_lowest_leakage_then_miou_and_boundary():
    from tools.select_guarded_semantic_validation_config import select_guarded_candidate

    selected = select_guarded_candidate(
        [
            _candidate(0.2, b5_leak=0.017, b5_miou=0.48, boundary_f1=0.50),
            _candidate(0.1, b5_leak=0.015, b5_miou=0.49, boundary_f1=0.43),
            _candidate(0.05, b5_leak=0.015, b5_miou=0.48, boundary_f1=0.60),
        ],
        max_miou_gap=0.02,
    )

    assert selected["soft_threshold"] == 0.1


def test_derive_fallback_parts_uses_b3_only_for_empty_b5_targets():
    from tools.select_guarded_semantic_validation_config import derive_fallback_parts

    b5_rows = [
        {"part": "skin", "target": "100", "predicted": "0"},
        {"part": "face", "target": "50", "predicted": "20"},
        {"part": "shoes", "target": "30", "predicted": "0"},
    ]
    b3_rows = [
        {"part": "skin", "target": "100", "predicted": "40"},
        {"part": "face", "target": "50", "predicted": "35"},
        {"part": "shoes", "target": "30", "predicted": "0"},
    ]

    assert derive_fallback_parts(b5_rows, b3_rows) == ["skin"]


def test_write_guarded_config_records_selection_and_fallback(tmp_path):
    from tools.select_guarded_semantic_validation_config import write_guarded_config

    output = write_guarded_config(
        tmp_path / "frozen.json",
        candidate=_candidate(0.1),
        fallback_parts=["skin"],
        fallback_threshold=0.5,
        protocol_name="coreview_test",
        protocol_fingerprint="protocol",
        checkpoint_fingerprint="checkpoint",
        bank_fingerprint="bank",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["selected"]["soft_threshold"] == 0.1
    assert payload["selected"]["b5_fallback_parts"] == ["skin"]
    assert payload["selected"]["b5_fallback_threshold"] == 0.5
    assert payload["checkpoint_fingerprint"] == "checkpoint"
