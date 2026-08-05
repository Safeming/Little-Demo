import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/semantic/a7c_r1_4vp_r3_crw_contribution_weighted_377_v1.json"


def test_r3_contract_changes_only_contribution_reduction_and_entry_gate():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "a7c_r1_4vp_r3_crw_contribution_weighted_377_v1"
    assert contract["contribution_signals"] == ["target", "outer", "boundary"]
    assert contract["contribution_weight_minimum"] == 0.1
    assert contract["contribution_weight_maximum"] == 10.0
    assert contract["minimum_fit_outer_recovery"] == 0.70
    assert contract["minimum_fit_boundary_recovery"] == 0.70
    assert contract["minimum_fit_positive_fraction"] == 0.90
    assert contract["residual_loss_weight"] == 0.00001
    assert contract["training_epochs"] == 400
    assert contract["maximum_visibility_response_ratio"] == 1.0
    assert contract["r1_1_f1_outer_gain"] == -0.00012761059760764496
    assert contract["r1_1_f1_boundary_gain"] == 0.023481874880317264
