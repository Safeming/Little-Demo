import json
import subprocess
from pathlib import Path

import pytest


A5_FREEZE_PATH = Path("configs/semantic/frozen_a5_main_method_v1.json")
A7_CONTRACT_PATH = Path("configs/semantic/frozen_a7_temporal_reliable_v1.json")
A7_V11_CONTRACT_PATH = Path(
    "configs/semantic/frozen_a7_temporal_reliable_v1_1.json"
)
A7_V2_CONTRACT_PATH = Path(
    "configs/semantic/frozen_a7_renderer_aligned_v2_canary_377.json"
)


EXPECTED_PROTOCOL = {
    "schema_version": 1,
    "freeze_id": "a7_temporal_reliable_v1",
    "base_method": "A5",
    "output_field": "soft_edit_weights",
    "runtime_state": False,
    "retrain_avatar": False,
    "evidence_cameras": ["c01", "c05", "c09", "c13"],
    "evidence_frame_start": 0,
    "evidence_frame_end": 570,
    "evidence_frame_stride": 5,
    "validation_cameras": ["c17", "c18", "c19", "c20"],
    "validation_frame_stride": 5,
    "retrospective_test_cameras": ["c21"],
    "frozen_test_cameras": ["c22", "c23"],
    "parts": ["hair", "face", "upper", "lower", "shoes", "skin"],
    "max_weight_scale_from_posterior": 1.0,
}


def _valid_payload():
    from utils.frozen_semantic_method import load_frozen_semantic_method

    base = load_frozen_semantic_method(A5_FREEZE_PATH)
    payload = dict(EXPECTED_PROTOCOL)
    payload.update(
        {
            "status": "frozen",
            "checkpoint_mutation": False,
            "base_method_freeze_fingerprint": base["_fingerprint"],
        }
    )
    return payload, base


def test_frozen_a7_contract_declares_static_protocol():
    from utils.frozen_semantic_method import load_a7_temporal_contract

    contract = load_a7_temporal_contract(A7_CONTRACT_PATH, A5_FREEZE_PATH)

    for key, expected in EXPECTED_PROTOCOL.items():
        assert contract[key] == expected
    assert contract["checkpoint_mutation"] is False
    assert len(contract["base_method_freeze_fingerprint"]) == 64
    assert len(contract["_fingerprint"]) == 64


def test_a7_contract_rejects_mismatched_base_fingerprint():
    from utils.frozen_semantic_method import validate_a7_temporal_contract

    payload, base = _valid_payload()
    payload["base_method_freeze_fingerprint"] = "0" * 64

    with pytest.raises(ValueError, match="base A5 fingerprint"):
        validate_a7_temporal_contract(payload, base_method=base)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runtime_state", True, "runtime_state"),
        ("retrain_avatar", True, "retrain_avatar"),
        ("checkpoint_mutation", True, "checkpoint"),
    ],
)
def test_a7_contract_rejects_runtime_or_checkpoint_mutation(field, value, message):
    from utils.frozen_semantic_method import validate_a7_temporal_contract

    payload, base = _valid_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        validate_a7_temporal_contract(payload, base_method=base)


@pytest.mark.parametrize("field", ["evidence_cameras", "validation_cameras"])
def test_a7_contract_rejects_c21_for_calibration_or_validation(field):
    from utils.frozen_semantic_method import validate_a7_temporal_contract

    payload, base = _valid_payload()
    payload[field] = [*payload[field], "c21"]

    with pytest.raises(ValueError, match="c21"):
        validate_a7_temporal_contract(payload, base_method=base)


def test_a7_validator_cli_reports_both_fingerprints_and_static_method():
    completed = subprocess.run(
        [
            "/opt/miniconda3/envs/ictrl/bin/python",
            "tools/validate_frozen_semantic_method.py",
            "--config",
            str(A7_CONTRACT_PATH),
            "--base-config",
            str(A5_FREEZE_PATH),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["base_method"] == "A5"
    assert report["runtime_state"] is False
    assert len(report["base_method_freeze_fingerprint"]) == 64
    assert len(report["a7_contract_fingerprint"]) == 64


def test_frozen_a7_v11_contract_freezes_guarded_carrier_policy():
    from utils.frozen_semantic_method import load_a7_temporal_contract

    contract = load_a7_temporal_contract(A7_V11_CONTRACT_PATH, A5_FREEZE_PATH)

    assert contract["freeze_id"] == "a7_temporal_reliable_v1_1"
    assert contract["boundary_dominance_margin"] == pytest.approx(0.2)
    assert contract["minimum_carrier_support_ratio"] == pytest.approx(0.5)
    assert contract["minimum_carrier_existing_weight"] == pytest.approx(0.2)
    assert contract["carrier_ranking"] == "reliability_support_target_posterior"


def test_frozen_a7_v2_contract_freezes_renderer_and_part_policy():
    from utils.frozen_semantic_method import load_a7_temporal_contract

    contract = load_a7_temporal_contract(A7_V2_CONTRACT_PATH, A5_FREEZE_PATH)

    assert contract["freeze_id"] == "a7_renderer_aligned_v2_canary_377"
    assert contract["subject"] == "377"
    assert contract["evidence_mode"] == "renderer_aligned"
    assert contract["renderer_attribution"] == "colors_gradient"
    assert contract["frozen_parts"] == ["face", "upper", "shoes", "skin"]
    assert contract["selection_threshold"] == pytest.approx(0.2)
    assert contract["preserve_a5_selection_topology"] is True
