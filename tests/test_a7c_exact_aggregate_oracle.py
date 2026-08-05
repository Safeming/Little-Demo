import json
from pathlib import Path

import pytest


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


def _source_records():
    return [
        {
            "fold": fold,
            "camera_index": camera,
            "boundary_conditioned": {
                "status": "bracketed",
                "feasible_lower": 0.03 + 0.001 * fold + 0.0001 * camera,
                "infeasible_upper": 0.030009
                + 0.001 * fold
                + 0.0001 * camera,
                "interval_width": 9.0e-6,
            },
        }
        for fold in range(6)
        for camera in range(4)
    ]


def test_extract_replay_requests_requires_exact_24_record_grid():
    from utils.a7c_exact_aggregate_oracle import extract_replay_requests

    rows = extract_replay_requests(
        _source_records(),
        replay_margin=2.0e-5,
        maximum_interval_width=1.0e-5,
    )

    assert len(rows) == 24
    assert rows[0]["minimum_outer_gain"] == 0.005
    assert rows[0]["minimum_boundary_gain"] == pytest.approx(0.02998)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "exactly 24"),
        ("duplicate", "duplicate"),
        ("grid", "grid differs"),
        ("status", "bracketed"),
        ("width", "interval width"),
        ("inconsistent_width", "interval width"),
    ],
)
def test_extract_replay_requests_rejects_invalid_sources(case, message):
    from utils.a7c_exact_aggregate_oracle import extract_replay_requests

    records = _source_records()
    if case == "missing":
        records.pop()
    elif case == "duplicate":
        records[-1] = dict(records[0])
    elif case == "grid":
        records[-1]["camera_index"] = 4
    elif case == "status":
        records[0]["boundary_conditioned"]["status"] = "infeasible"
    elif case == "width":
        records[0]["boundary_conditioned"]["interval_width"] = 1.1e-5
    elif case == "inconsistent_width":
        records[0]["boundary_conditioned"]["interval_width"] = 8.0e-6

    with pytest.raises(ValueError, match=message):
        extract_replay_requests(
            records,
            replay_margin=2.0e-5,
            maximum_interval_width=1.0e-5,
        )
