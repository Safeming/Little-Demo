import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/semantic/a7c_r1_2a_quotient_compositor_377_v1.json"


def test_contract_freezes_scope_inputs_objective_and_gates():
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert payload["status"] == "frozen"
    assert payload["fit_cameras"] == ["c01", "c05", "c09", "c13"]
    assert payload["audit_cameras"] == ["c17", "c18", "c19", "c20"]
    assert payload["forbidden_cameras"] == ["c21", "c22", "c23"]
    assert payload["score_feature_group"] == "F1"
    assert payload["teacher_gate_loss_weight"] == 0.0
    assert payload["proxy_target_response"] == 0.995
    assert payload["minimum_target_response"] == 0.99
    assert payload["maximum_adjacent_gate_change"] == 0.02
    assert payload["paper_test_eligible"] is False
