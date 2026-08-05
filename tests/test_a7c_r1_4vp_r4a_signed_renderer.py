import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs/semantic/a7c_r1_4vp_r4a_signed_renderer_trajectory_377_v1.json"
)


def test_r4a_contract_changes_only_the_registered_training_objective():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == (
        "a7c_r1_4vp_r4a_signed_renderer_trajectory_377_v1"
    )
    assert contract["renderer_trajectory_signals"] == ["outer", "boundary"]
    assert contract["renderer_trajectory_huber_delta"] == 0.005
    assert contract["target_response_huber_delta"] == 0.005
    assert contract["renderer_outer_loss_weight"] == 1.0
    assert contract["renderer_boundary_loss_weight"] == 1.0
    assert contract["target_auxiliary_loss_weight"] == 0.1
    assert contract["gate_auxiliary_loss_weight"] == 0.1
    assert contract["initial_scale_minimum"] == 1e-12
    assert contract["training_epochs"] == 400
    assert contract["minimum_fit_outer_recovery"] == 0.70
    assert contract["minimum_fit_boundary_recovery"] == 0.70
