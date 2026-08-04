import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs/semantic/a7c_r1_3p_temporal_joint_projection_377_v1.json"
)


def test_r1_3p_contract_freezes_runtime_and_oracle_boundaries():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["status"] == "frozen"
    assert (
        contract["source_experiment_id"]
        == "a7c_r1_2b_dense_overlap_set_377_v1"
    )
    assert contract["offline_bidirectional"] is True
    assert contract["maximum_projection_gate_jump"] == 0.015
    assert contract["maximum_adjacent_gate_change"] == 0.02
    assert contract["proxy_target_response"] == 0.995
    assert contract["minimum_target_response"] == 0.99
    assert contract["maximum_selection_soft_iou_drop"] == 0.005
    assert contract["solver"] == "highs"
    assert contract["solver_residual_tolerance"] == 1.0e-7
    assert contract["oracle_bisection_tolerance"] == 1.0e-5
    assert contract["fit_cameras"] == ["c01", "c05", "c09", "c13"]
    assert contract["forbidden_cameras"] == [
        "c17",
        "c18",
        "c19",
        "c20",
        "c21",
        "c22",
        "c23",
    ]
    assert contract["retrain_predictor"] is False
    assert contract["paper_test_eligible"] is False
