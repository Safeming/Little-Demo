import json
from pathlib import Path
import re

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/semantic/a7c_r1_3g_exact_aggregate_oracle_377_v1.json"
)
RUNNER = ROOT / "tools/run_a7c_r1_3g_exact_aggregate_oracle_377.sh"
AUDITOR = ROOT / "tools/audit_a7c_r1_3g_exact_aggregate_oracle.py"


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


def _solved_segment(maximum_primal_violation=1.0e-9):
    return {
        "gates": np.array([[0.99, 1.0], [0.98, 0.99]]),
        "metrics": {
            "outer_gain": 0.006,
            "boundary_gain": 0.03,
            "minimum_target_response": 0.995,
            "maximum_soft_iou_drop": 0.001,
            "maximum_adjacent_gate_change": 0.01,
        },
        "certificate": {
            "solver": "scipy.optimize.linprog:highs",
            "scipy_version": "1.13.1",
            "status": 0,
            "message": "optimal",
            "iterations": 3,
            "maximum_matrix_violation": 0.0,
            "maximum_primal_violation": maximum_primal_violation,
        },
        "slack": {
            "minimum_topology_slack": 0.08,
            "minimum_bound_slack": 0.0,
            "minimum_proxy_target_slack": 0.0,
        },
        "locations": {
            "maximum_adjacent_gate_change_frame": 5,
            "maximum_adjacent_gate_change_carrier_id": 10,
        },
        "source_fingerprints": {"probe": "a" * 64},
        "sample_order_fingerprint": "sample-order",
        "carrier_order_fingerprint": "carrier-order",
    }


def _replay_request():
    return {
        "fold": 0,
        "camera_index": 0,
        "source_feasible_lower": 0.03002,
        "source_infeasible_upper": 0.030029,
        "source_interval_width": 9.0e-6,
        "minimum_outer_gain": 0.005,
        "minimum_boundary_gain": 0.03,
    }


def test_insert_replay_segment_isolates_nonheld_samples_and_checks_certificate():
    from utils.a7c_exact_aggregate_oracle import insert_replay_segment

    gates = np.full((8, 2), np.nan)
    replay_mask = np.zeros(8, dtype=bool)
    selected = np.array(
        [True, True, False, False, False, False, False, False]
    )

    row = insert_replay_segment(
        replay_gates=gates,
        replay_mask=replay_mask,
        selected=selected,
        solved=_solved_segment(),
        request=_replay_request(),
        frame_index=np.array([0, 5, 0, 5, 0, 5, 0, 5]),
        block_ids=np.zeros(8, dtype=np.int16),
        carrier_ids=np.array([10, 11]),
        residual_tolerance=1.0e-7,
    )

    assert np.isfinite(gates[selected]).all()
    assert np.isnan(gates[~selected]).all()
    assert replay_mask.tolist() == selected.tolist()
    assert row["minimum_topology_slack"] == 0.08
    assert row["maximum_primal_violation"] <= 1.0e-7
    assert row["sample_order_fingerprint"] == "sample-order"
    assert row["source_fingerprints"] == {"probe": "a" * 64}


def test_insert_replay_segment_rejects_overlap_and_bad_residual():
    from utils.a7c_exact_aggregate_oracle import insert_replay_segment

    kwargs = {
        "replay_gates": np.full((2, 2), np.nan),
        "replay_mask": np.zeros(2, dtype=bool),
        "selected": np.ones(2, dtype=bool),
        "solved": _solved_segment(),
        "request": _replay_request(),
        "frame_index": np.array([0, 5]),
        "block_ids": np.zeros(2, dtype=np.int16),
        "carrier_ids": np.array([10, 11]),
        "residual_tolerance": 1.0e-7,
    }
    insert_replay_segment(**kwargs)
    with pytest.raises(ValueError, match="overlap"):
        insert_replay_segment(**kwargs)

    kwargs["replay_gates"] = np.full((2, 2), np.nan)
    kwargs["replay_mask"] = np.zeros(2, dtype=bool)
    kwargs["solved"] = _solved_segment(maximum_primal_violation=2.0e-7)
    with pytest.raises(RuntimeError, match="certificate"):
        insert_replay_segment(**kwargs)


def test_fixed_gain_replay_is_deterministic():
    from utils.a7c_feasibility_oracle import solve_fixed_gain_oracle

    base = np.array([1.0, 2.0, 1.0, 2.0, 1.0])
    point = base[:, None]
    streams = {
        "objective": {
            "outer": {"base": base, "point": point},
            "boundary": {"base": base, "point": point},
        },
        "guard": {
            "target": {
                "base": np.ones(5),
                "point": np.zeros((5, 1)),
            },
            "outer": {"base": base, "point": point},
        },
    }
    kwargs = {
        "runtime_mass": np.zeros((5, 1)),
        "a5_weight": np.array([0.8]),
        "streams": streams,
        "minimum_gate": 0.9,
        "maximum_gate": 1.0,
        "selection_threshold": 0.2,
        "proxy_target_response": 0.995,
        "maximum_gate_jump": 0.015,
        "minimum_target_response": 0.99,
        "maximum_soft_iou_drop": 0.005,
        "minimum_outer_gain": 0.005,
        "minimum_boundary_gain": 0.005,
    }

    first = solve_fixed_gain_oracle(**kwargs)
    second = solve_fixed_gain_oracle(**kwargs)

    np.testing.assert_array_equal(first["gates"], second["gates"])
    assert first["metrics"] == second["metrics"]


def test_write_fold_witness_persists_only_replay_mask(tmp_path):
    from tools.evaluate_a7c_r1_3g_exact_aggregate_oracle import (
        write_fold_witness,
    )

    mask = np.array([True, True, False, False])
    gates = np.array([[0.99], [0.98], [np.nan], [np.nan]])
    request_arrays = {
        "requested_outer_gain": np.array([0.005, 0.005, np.nan, np.nan]),
        "requested_boundary_gain": np.array([0.03, 0.03, np.nan, np.nan]),
        "source_feasible_lower": np.array([0.03002, 0.03002, np.nan, np.nan]),
        "source_infeasible_upper": np.array(
            [0.030029, 0.030029, np.nan, np.nan]
        ),
    }

    write_fold_witness(
        output_dir=tmp_path,
        replay_gates=gates,
        replay_mask=mask,
        request_arrays=request_arrays,
        camera_index=np.array([0, 0, 1, 1]),
        frame_index=np.array([0, 5, 0, 5]),
        block_ids=np.zeros(4, dtype=np.int16),
        carrier_ids=np.array([10]),
        certificates=[{"fold": 0, "camera_index": 0}],
        source_fingerprints={"probe": "a" * 64},
        sample_order_fingerprint="sample-order",
        carrier_order_fingerprint="carrier-order",
    )

    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as saved:
        assert np.array_equal(saved["replay_mask"], mask)
        assert np.isfinite(saved["replay_gates"][mask]).all()
        assert np.isnan(saved["replay_gates"][~mask]).all()
        for key in request_arrays:
            assert np.isfinite(saved[key][mask]).all()
            assert np.isnan(saved[key][~mask]).all()
        for key in (
            "deployment_eligible",
            "teacher_eligible",
            "paper_test_eligible",
        ):
            assert int(saved[key]) == 0
    assert json.loads(
        (tmp_path / "certificates.json").read_text(encoding="utf-8")
    ) == [{"camera_index": 0, "fold": 0}]


def _promotion_contract():
    return {
        "minimum_outer_gain": 0.005,
        "minimum_boundary_gain": 0.005,
        "minimum_positive_block_fraction": 0.9,
        "block_gain_quantile": 0.1,
        "minimum_block_gain_quantile": 0.0,
        "maximum_worst_block_regression": 0.005,
        "minimum_target_response": 0.99,
        "maximum_selection_soft_iou_drop": 0.005,
        "maximum_adjacent_gate_change": 0.02,
        "r1_1_f1_outer_gain": -0.00012761059760764496,
        "r1_1_f1_boundary_gain": 0.023481874880317264,
    }


def test_exact_aggregate_accepts_compensation_below_r1_1_boundary_mean():
    from tools.audit_a7c_r1_2a_quotient_compositor import summarize_records

    records = [
        {
            "outer_gain": 0.006,
            "boundary_gain": 0.02 if index < 3 else 0.03,
            "minimum_target_response": 0.995,
            "maximum_soft_iou_drop": 0.001,
            "maximum_adjacent_gate_change": 0.015,
        }
        for index in range(24)
    ]

    summary = summarize_records(records, _promotion_contract())

    assert (
        min(row["boundary_gain"] for row in records)
        < 0.023481874880317264
    )
    assert summary["boundary_gain"] > 0.023481874880317264
    assert summary["passed"] is True


def test_classify_exact_replay_never_claims_infeasibility():
    from utils.a7c_exact_aggregate_oracle import classify_exact_replay

    assert (
        classify_exact_replay(replay_complete=True, audit_passed=True)
        == "CERTIFIED_FEASIBLE"
    )
    assert (
        classify_exact_replay(replay_complete=True, audit_passed=False)
        == "UNRESOLVED"
    )
    with pytest.raises(ValueError, match="incomplete"):
        classify_exact_replay(replay_complete=False, audit_passed=False)


def test_validate_saved_manifest_rejects_order_or_eligibility_change():
    from utils.a7c_exact_aggregate_oracle import validate_saved_manifest

    expected = {
        "camera_index": np.array([0, 0, 1, 1]),
        "frame_index": np.array([0, 5, 0, 5]),
        "block_ids": np.array([0, 0, 0, 0]),
        "carrier_ids": np.array([10, 11]),
    }
    saved = {
        **expected,
        "deployment_eligible": np.array(0),
        "teacher_eligible": np.array(0),
        "paper_test_eligible": np.array(0),
    }
    validate_saved_manifest(saved, expected)

    broken = dict(saved)
    broken["frame_index"] = np.array([5, 0, 0, 5])
    with pytest.raises(ValueError, match="frame_index"):
        validate_saved_manifest(broken, expected)

    broken = dict(saved)
    broken["teacher_eligible"] = np.array(1)
    with pytest.raises(ValueError, match="teacher_eligible"):
        validate_saved_manifest(broken, expected)


def test_load_fold_witness_revalidates_disk_isolation(tmp_path):
    from tools.audit_a7c_r1_3g_exact_aggregate_oracle import (
        load_fold_witness,
    )
    from tools.evaluate_a7c_r1_3g_exact_aggregate_oracle import (
        _array_fingerprint,
        write_fold_witness,
    )

    expected = {
        "camera_index": np.array([0, 0, 1, 1]),
        "frame_index": np.array([0, 5, 0, 5]),
        "block_ids": np.array([0, 0, 0, 0]),
        "carrier_ids": np.array([10]),
    }
    mask = np.array([True, True, False, False])
    request_arrays = {
        "requested_outer_gain": np.array([0.005, 0.005, np.nan, np.nan]),
        "requested_boundary_gain": np.array([0.03, 0.03, np.nan, np.nan]),
        "source_feasible_lower": np.array([0.03002, 0.03002, np.nan, np.nan]),
        "source_infeasible_upper": np.array(
            [0.030029, 0.030029, np.nan, np.nan]
        ),
    }
    sample_fingerprint = _array_fingerprint(
        expected["camera_index"],
        expected["frame_index"],
        expected["block_ids"],
    )
    carrier_fingerprint = _array_fingerprint(expected["carrier_ids"])
    write_fold_witness(
        output_dir=tmp_path,
        replay_gates=np.array([[0.99], [0.98], [np.nan], [np.nan]]),
        replay_mask=mask,
        request_arrays=request_arrays,
        certificates=[],
        source_fingerprints={"probe": "a" * 64},
        sample_order_fingerprint=sample_fingerprint,
        carrier_order_fingerprint=carrier_fingerprint,
        **expected,
    )

    loaded = load_fold_witness(
        tmp_path / "predictions.npz",
        expected_manifest=expected,
        expected_mask=mask,
        expected_sample_fingerprint=sample_fingerprint,
        expected_carrier_fingerprint=carrier_fingerprint,
    )
    np.testing.assert_array_equal(loaded["replay_gates"][mask], [[0.99], [0.98]])

    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as source:
        tampered = {key: source[key] for key in source.files}
    tampered["replay_gates"] = tampered["replay_gates"].copy()
    tampered["replay_gates"][2] = 0.95
    np.savez_compressed(tmp_path / "predictions.npz", **tampered)
    with pytest.raises(ValueError, match="cross replay mask"):
        load_fold_witness(
            tmp_path / "predictions.npz",
            expected_manifest=expected,
            expected_mask=mask,
            expected_sample_fingerprint=sample_fingerprint,
            expected_carrier_fingerprint=carrier_fingerprint,
        )


def test_r1_3g_runner_is_restart_safe_isolated_and_audit_gated():
    source = RUNNER.read_text(encoding="utf-8")

    assert "/opt/miniconda3/envs/ictrl/bin/python" in source
    assert 'cd "${ROOT}"' in source
    assert "evaluate_a7c_r1_3g_exact_aggregate_oracle.py" in source
    assert "audit_a7c_r1_3g_exact_aggregate_oracle.py" in source
    assert source.index("evaluate_a7c_r1_3g") < source.index(
        "audit_a7c_r1_3g"
    )
    assert "audit_status" in source
    assert "CERTIFIED_FEASIBLE" in source
    assert "UNRESOLVED" in source
    assert "ORACLE_ERROR" in source
    assert "check_sha" in source
    assert re.search(
        r"rg -q .*REPLAY_COMPLETED.*summary\.json", source
    )
    for camera in ("c17", "c18", "c19", "c20", "c21", "c22", "c23"):
        assert re.search(rf"\b{camera}\b", source) is None
    for artifact in (
        "runner.pid",
        "runner.log",
        "started_utc.txt",
        "ended_utc.txt",
    ):
        assert artifact in source
    for marker in (".completed", ".rejected", ".failed"):
        assert marker in source


def test_r1_3g_auditor_reverifies_probe_in_witness_source_fingerprints():
    source = AUDITOR.read_text(encoding="utf-8")

    assert 'contract["source_probe"]' in source
    assert 'contract["source_probe_sha256"]' in source
