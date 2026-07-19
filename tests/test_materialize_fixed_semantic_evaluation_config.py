import pytest


def _protocol():
    return {
        "protocol_name": "coreview386_strict_paper_v1",
        "subject": "CoreView_386",
        "semantic_train": {"camera_ids": [1], "frame_ids": [0]},
        "calibration": {"camera_ids": [1], "frame_ids": [0]},
        "validation": {"camera_ids": [2], "frame_ids": [1]},
        "test": {"camera_ids": [3], "frame_ids": [2]},
        "parts": ["face"],
        "allowed_adjacency": {"face": []},
        "validation_grid": {
            "soft_thresholds": [0.5],
            "support_thresholds": [0.3],
            "boundary_radii": [6],
        },
        "matched_retention_targets": [0.5, 0.6],
        "minimum_target_retention": 0.6,
        "boundary_metric_tolerance": 2,
    }


def test_materializes_fixed_parameters_with_new_fingerprints():
    from tools.materialize_fixed_semantic_evaluation_config import materialize_fixed_config

    payload = materialize_fixed_config(
        protocol=_protocol(),
        template={
            "protocol_name": "coreview377_strict_paper_v1",
            "selected": {
                "soft_threshold": "0.5",
                "support_threshold": "0.3",
                "boundary_radius": "6",
            },
        },
        checkpoint_fingerprint="ckpt386",
        bank_fingerprint="bank386",
    )

    assert payload["selected"] == {
        "soft_threshold": 0.5,
        "support_threshold": 0.3,
        "boundary_radius": 6,
    }
    assert payload["checkpoint_fingerprint"] == "ckpt386"
    assert payload["bank_fingerprint"] == "bank386"
    assert payload["selection_mode"] == "cross_subject_fixed_from_template"
    assert payload["template_protocol_name"] == "coreview377_strict_paper_v1"


def test_rejects_template_missing_frozen_key():
    from tools.materialize_fixed_semantic_evaluation_config import materialize_fixed_config

    with pytest.raises(ValueError, match="support_threshold"):
        materialize_fixed_config(
            protocol=_protocol(),
            template={"selected": {"soft_threshold": 0.5, "boundary_radius": 6}},
            checkpoint_fingerprint="ckpt386",
            bank_fingerprint="bank386",
        )
