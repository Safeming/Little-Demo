import json
import inspect
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs/semantic/a7c_r1_3p_temporal_joint_projection_377_v1.json"
)
PROJECTOR = ROOT / "tools/project_a7c_r1_3p_temporal_joint.py"
AUDITOR = ROOT / "tools/audit_a7c_r1_3p_temporal_joint_projection.py"


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


def test_joint_projection_repairs_jump_and_preserves_guards():
    from utils.a7c_temporal_joint_projection import (
        solve_temporal_joint_projection,
    )

    raw = np.array([[0.90, 1.00], [1.00, 0.90], [0.90, 1.00]])
    result = solve_temporal_joint_projection(
        raw_gates=raw,
        runtime_mass=np.ones_like(raw),
        a5_weight=np.array([0.25, 0.8]),
        minimum_gate=0.9,
        maximum_gate=1.0,
        selection_threshold=0.2,
        proxy_target_response=0.995,
        maximum_gate_jump=0.015,
    )

    gates = result["gates"]
    assert np.max(np.abs(np.diff(gates, axis=0))) <= 0.015 + 1.0e-8
    assert np.min(np.mean(gates, axis=1)) >= 0.995 - 1.0e-8
    assert result["certificate"]["maximum_primal_violation"] <= 1.0e-7


def test_joint_projection_zero_mass_keeps_already_feasible_raw():
    from utils.a7c_temporal_joint_projection import (
        solve_temporal_joint_projection,
    )

    raw = np.array([[0.93], [0.94], [0.95]])
    result = solve_temporal_joint_projection(
        raw_gates=raw,
        runtime_mass=np.zeros_like(raw),
        a5_weight=np.array([0.8]),
        minimum_gate=0.9,
        maximum_gate=1.0,
        selection_threshold=0.2,
        proxy_target_response=0.995,
        maximum_gate_jump=0.015,
    )

    np.testing.assert_allclose(result["gates"], raw, atol=1.0e-9)


def test_runtime_projector_signature_has_no_renderer_inputs():
    from utils.a7c_temporal_joint_projection import (
        solve_temporal_joint_projection,
    )

    names = set(inspect.signature(solve_temporal_joint_projection).parameters)
    assert not names & {
        "evidence",
        "target",
        "outer",
        "boundary",
        "teacher_gates",
    }


@pytest.mark.parametrize(
    ("raw", "mass", "weights", "message"),
    [
        (np.ones(2), np.ones(2), np.ones(2), "raw_gates"),
        (np.ones((2, 2)), np.ones((2, 1)), np.ones(2), "runtime_mass"),
        (np.ones((2, 2)), np.ones((2, 2)), np.ones(1), "a5_weight"),
        (
            np.array([[1.0, np.nan]]),
            np.ones((1, 2)),
            np.ones(2),
            "finite",
        ),
    ],
)
def test_joint_projection_rejects_invalid_arrays(raw, mass, weights, message):
    from utils.a7c_temporal_joint_projection import (
        solve_temporal_joint_projection,
    )

    with pytest.raises(ValueError, match=message):
        solve_temporal_joint_projection(
            raw_gates=raw,
            runtime_mass=mass,
            a5_weight=weights,
            minimum_gate=0.9,
            maximum_gate=1.0,
            selection_threshold=0.2,
            proxy_target_response=0.995,
            maximum_gate_jump=0.015,
        )


def test_joint_projection_rejects_topology_floor_above_one():
    from utils.a7c_temporal_joint_projection import (
        solve_temporal_joint_projection,
    )

    with pytest.raises(ValueError, match="topology floor"):
        solve_temporal_joint_projection(
            raw_gates=np.ones((2, 1)),
            runtime_mass=np.ones((2, 1)),
            a5_weight=np.array([0.1]),
            minimum_gate=0.9,
            maximum_gate=1.0,
            selection_threshold=0.2,
            proxy_target_response=0.995,
            maximum_gate_jump=0.015,
        )


def test_joint_projection_is_deterministic():
    from utils.a7c_temporal_joint_projection import (
        solve_temporal_joint_projection,
    )

    kwargs = {
        "raw_gates": np.array([[0.94, 0.99], [0.97, 0.92]]),
        "runtime_mass": np.array([[0.1, 1.0], [0.2, 1.0]]),
        "a5_weight": np.array([0.4, 0.8]),
        "minimum_gate": 0.9,
        "maximum_gate": 1.0,
        "selection_threshold": 0.2,
        "proxy_target_response": 0.995,
        "maximum_gate_jump": 0.015,
    }
    first = solve_temporal_joint_projection(**kwargs)
    second = solve_temporal_joint_projection(**kwargs)

    np.testing.assert_array_equal(first["gates"], second["gates"])
    assert (
        first["certificate"]["stage_one_objective"]
        == second["certificate"]["stage_one_objective"]
    )
    assert (
        first["certificate"]["stage_two_objective"]
        == second["certificate"]["stage_two_objective"]
    )


def test_projection_cli_has_no_renderer_evidence_access():
    source = PROJECTOR.read_text(encoding="utf-8")

    for forbidden in (
        "_build_streams",
        "evidence",
        "point_outer",
        "point_boundary",
    ):
        assert forbidden not in source
    assert "solve_temporal_joint_projection" in source

    audit_source = AUDITOR.read_text(encoding="utf-8")
    assert "_build_streams" in audit_source
    assert "summarize_records" in audit_source


def test_project_fold_writes_only_four_held_camera_segments():
    from tools.project_a7c_r1_3p_temporal_joint import (
        project_fold_predictions,
    )

    camera_index = np.repeat(np.arange(5), 2)
    frame_index = np.tile(np.array([0, 5]), 5)
    block_ids = np.zeros(10, dtype=np.int16)
    projection_mask = camera_index < 4
    raw = np.full((10, 1), 0.95)
    result = project_fold_predictions(
        raw_gates=raw,
        runtime_mass=np.zeros_like(raw),
        a5_weight=np.array([0.8]),
        projection_mask=projection_mask,
        camera_index=camera_index,
        frame_index=frame_index,
        block_ids=block_ids,
        fit_camera_indices=(0, 1, 2, 3),
        contract={
            "frame_stride": 5,
            "minimum_gate": 0.9,
            "maximum_gate": 1.0,
            "selection_threshold": 0.2,
            "proxy_target_response": 0.995,
            "maximum_projection_gate_jump": 0.015,
            "lexicographic_rho_tolerance": 1.0e-9,
            "solver_primal_tolerance": 1.0e-9,
            "solver_residual_tolerance": 1.0e-7,
        },
    )

    assert np.isfinite(result["projected_gates"][projection_mask]).all()
    assert np.isnan(result["projected_gates"][~projection_mask]).all()
    assert len(result["certificates"]) == 4


def test_projection_summary_includes_every_source_prediction_fingerprint():
    from tools.project_a7c_r1_3p_temporal_joint import summarize_projection

    summary = summarize_projection(
        experiment_id="synthetic",
        fold_summaries=[{"fold": index, "segment_count": 4} for index in range(6)],
        source_fingerprints={"probe": "a" * 64},
        prediction_fingerprints={
            f"fold_{index}_prediction": str(index) * 64 for index in range(6)
        },
    )

    assert summary["segment_count"] == 24
    assert set(summary["prediction_fingerprints"]) == {
        f"fold_{index}_prediction" for index in range(6)
    }
