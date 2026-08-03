import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/semantic/a7c_r1_2b_dense_overlap_set_377_v1.json"


def test_contract_changes_only_the_registered_predictor():
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["status"] == "frozen"
    assert payload["predictor"] == "dense_overlap_set"
    assert payload["score_feature_group"] == "F1"
    assert payload["node_hidden_dimension"] == 32
    assert payload["gate_hidden_dimension"] == 32
    assert payload["spatial_scale"] == 0.03
    assert payload["depth_scale"] == 0.04
    assert payload["teacher_gate_loss_weight"] == 0.0
    assert payload["runtime_state"] is False
    assert payload["paper_test_eligible"] is False
    assert payload["fit_cameras"] == ["c01", "c05", "c09", "c13"]
    assert payload["audit_cameras"] == ["c17", "c18", "c19", "c20"]
    assert payload["forbidden_cameras"] == ["c21", "c22", "c23"]
