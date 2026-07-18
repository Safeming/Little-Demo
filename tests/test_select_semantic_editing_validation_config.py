import json

import pytest


def test_select_validation_candidate_rejects_all_low_retention():
    from tools.select_semantic_editing_validation_config import select_validation_candidate

    rows = [
        {
            "soft_threshold": 0.2,
            "support_threshold": 0.2,
            "boundary_radius": 2,
            "aggregate_target_retention": 0.59,
            "mean_actionable_footprint_leakage": 0.01,
            "mean_boundary_f1": 0.9,
        }
    ]

    with pytest.raises(ValueError, match="no validation candidate reaches target retention"):
        select_validation_candidate(rows, minimum_retention=0.60)


def test_select_validation_candidate_uses_leakage_boundary_then_radius_order():
    from tools.select_semantic_editing_validation_config import select_validation_candidate

    rows = [
        {
            "soft_threshold": 0.1,
            "support_threshold": 0.2,
            "boundary_radius": 4,
            "aggregate_target_retention": 0.70,
            "mean_actionable_footprint_leakage": 0.04,
            "mean_boundary_f1": 0.80,
        },
        {
            "soft_threshold": 0.2,
            "support_threshold": 0.2,
            "boundary_radius": 4,
            "aggregate_target_retention": 0.65,
            "mean_actionable_footprint_leakage": 0.03,
            "mean_boundary_f1": 0.75,
        },
        {
            "soft_threshold": 0.25,
            "support_threshold": 0.2,
            "boundary_radius": 2,
            "aggregate_target_retention": 0.62,
            "mean_actionable_footprint_leakage": 0.03,
            "mean_boundary_f1": 0.75,
        },
    ]

    selected = select_validation_candidate(rows, minimum_retention=0.60)

    assert selected["soft_threshold"] == 0.25
    assert selected["boundary_radius"] == 2


def test_select_validation_candidate_uses_support_diagnostics_as_final_tiebreak():
    from tools.select_semantic_editing_validation_config import select_validation_candidate

    common = {
        "soft_threshold": 0.2,
        "boundary_radius": 2,
        "aggregate_target_retention": 0.70,
        "mean_actionable_footprint_leakage": 0.03,
        "mean_boundary_f1": 0.80,
    }
    selected = select_validation_candidate(
        [
            {
                **common,
                "support_threshold": 0.1,
                "allowed_support_fraction": 0.4,
                "actionable_support_fraction": 0.2,
            },
            {
                **common,
                "support_threshold": 0.2,
                "allowed_support_fraction": 0.8,
                "actionable_support_fraction": 0.1,
            },
        ]
    )

    assert selected["support_threshold"] == 0.2


def test_write_frozen_validation_config_records_fingerprints(tmp_path):
    from tools.select_semantic_editing_validation_config import write_frozen_validation_config
    from utils.semantic_eval_protocol import protocol_fingerprint

    protocol = {
        "protocol_name": "strict_test",
        "subject": "CoreView_377",
        "semantic_train": {"camera_ids": [1], "frame_ids": [0]},
        "calibration": {"camera_ids": [1], "frame_ids": [0]},
        "validation": {"camera_ids": [17], "frame_ids": [60]},
        "test": {"camera_ids": [21], "frame_ids": [180]},
    }
    candidate = {
        "soft_threshold": 0.2,
        "support_threshold": 0.1,
        "boundary_radius": 2,
        "aggregate_target_retention": 0.65,
        "mean_actionable_footprint_leakage": 0.03,
        "mean_boundary_f1": 0.75,
    }

    path = write_frozen_validation_config(
        tmp_path / "frozen.json",
        candidate,
        protocol=protocol,
        checkpoint_fingerprint="checkpoint",
        bank_fingerprint="bank",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["protocol_fingerprint"] == protocol_fingerprint(protocol)
    assert payload["checkpoint_fingerprint"] == "checkpoint"
    assert payload["bank_fingerprint"] == "bank"
    assert payload["selected"]["boundary_radius"] == 2
