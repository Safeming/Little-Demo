import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/semantic/a7c_r1_3g_exact_aggregate_oracle_377_v1.json"
)


def test_r1_3g_contract_freezes_constructive_replay_and_isolation():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert (
        contract["experiment_id"]
        == "a7c_r1_3g_exact_aggregate_oracle_377_v1"
    )
    assert contract["status"] == "frozen"
    assert contract["source_r1_3p_records_sha256"] == (
        "7a4d3998408a67cb4754d2bcb799e9e4e2ed8518b23fa3254055c4b88f9d3ce8"
    )
    assert contract["source_r1_3p_summary_sha256"] == (
        "82c1d019002a7ee980d0b13c583bf023ae0df5aac8f5d383e513d2cbff12c5c5"
    )
    assert contract["replay_margin"] == 2.0e-5
    assert contract["oracle_bisection_tolerance"] == 1.0e-5
    assert contract["minimum_outer_gain"] == 0.005
    assert contract["maximum_projection_gate_jump"] == 0.015
    assert contract["solver_residual_tolerance"] == 1.0e-7
    assert contract["fit_cameras"] == ["c01", "c05", "c09", "c13"]
    assert contract["audit_cameras"] == []
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
    assert contract["deployment_eligible"] is False
    assert contract["teacher_eligible"] is False
    assert contract["paper_test_eligible"] is False
