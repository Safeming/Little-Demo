import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_oracle_contract_freezes_capacity_and_promotion_gates():
    contract = json.loads(
        (ROOT / "configs/semantic/a7c_ray_oracle_capacity_v1.json").read_text()
    )
    assert contract["status"] == "frozen"
    assert contract["subjects"] == ["377", "386", "387", "393", "394"]
    assert contract["minimum_gate"] == 0.9
    assert contract["gate_knot_count"] == 12
    assert contract["minimum_oracle_outer_gain"] == 0.01
    assert contract["paper_test_eligible"] is False


def test_interpolation_basis_is_partition_of_unity_and_hits_endpoints():
    from utils.a7c_oracle_capacity import interpolation_basis

    basis = interpolation_basis(9, 3)

    np.testing.assert_allclose(basis.sum(axis=1), 1.0)
    np.testing.assert_allclose(basis[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(basis[-1], [0.0, 0.0, 1.0])


def test_fractional_gate_reduces_alternating_signal_with_target_guard():
    from utils.a7c_oracle_capacity import solve_fractional_gate

    signal = np.array([1.0, 2.0] * 6)
    target = np.ones_like(signal)
    result = solve_fractional_gate(
        variable_values=signal[:, None],
        fixed_values=np.zeros_like(signal),
        target_variable=target[:, None],
        target_fixed=np.zeros_like(target),
        target_base=target,
        minimum_target_response=0.9,
        minimum_gate=0.9,
        maximum_gate=1.0,
        knot_count=6,
    )

    assert result["success"] is True
    assert result["normalized_flicker_gain"] > 0.0
    assert result["minimum_target_response"] >= 0.9 - 1e-8
    assert np.min(result["gates"]) >= 0.9 - 1e-8
    assert np.max(result["gates"]) <= 1.0 + 1e-8


def test_fractional_gate_can_guard_every_temporal_block():
    from utils.a7c_oracle_capacity import normalized_flicker, solve_fractional_gate

    signal = np.array(
        [1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 1.0, 1.5, 1.0, 1.5]
    )
    result = solve_fractional_gate(
        variable_values=signal[:, None],
        fixed_values=np.zeros_like(signal),
        minimum_gate=0.9,
        maximum_gate=1.0,
        knot_count=6,
        temporal_block_count=3,
        minimum_block_gain=0.0,
    )

    for indices in np.array_split(np.arange(signal.size), 3):
        assert normalized_flicker(result["candidate"][indices]) <= (
            normalized_flicker(signal[indices]) + 1e-8
        )


def test_block_guard_can_use_an_explicit_unmodified_baseline():
    from utils.a7c_oracle_capacity import normalized_flicker, solve_fractional_gate

    variable = np.array([2.0, 1.0] * 6)
    modified_fixed = np.array([0.9, 1.8] * 6)
    original_base = variable + np.array([1.0, 2.0] * 6)
    result = solve_fractional_gate(
        variable_values=variable[:, None],
        fixed_values=modified_fixed,
        block_base_values=original_base,
        minimum_gate=0.9,
        maximum_gate=1.0,
        knot_count=6,
        temporal_block_count=3,
        minimum_block_gain=0.0,
    )

    for indices in np.array_split(np.arange(variable.size), 3):
        assert normalized_flicker(result["candidate"][indices]) <= (
            normalized_flicker(original_base[indices]) + 1e-8
        )


def test_ray_oracle_preserves_target_and_can_outperform_global_gate():
    from utils.a7c_oracle_capacity import evaluate_camera_oracles

    boundary = np.array([1.0, 2.0] * 6)
    outer_nonboundary = np.array([2.0, 1.0] * 6)
    outer = boundary + outer_nonboundary
    target = np.ones_like(outer)
    point = np.stack([outer * 0.4, outer * 0.6], axis=1)
    target_point = np.stack([target * 0.4, target * 0.6], axis=1)

    result = evaluate_camera_oracles(
        target=target,
        outer=outer,
        boundary=boundary,
        point_target=target_point,
        point_outer=point,
        point_boundary=np.stack([boundary * 0.4, boundary * 0.6], axis=1),
        minimum_gate=0.9,
        minimum_target_response=0.99,
        knot_count=6,
    )

    assert result["ray"]["minimum_target_response"] == pytest.approx(1.0)
    assert result["ray"]["boundary_gain"] > result["global"]["boundary_gain"]


def test_teacher_artifact_preserves_sample_order_and_bounds(tmp_path):
    from utils.a7c_oracle_capacity import (
        assemble_teacher_gate_matrix,
        load_teacher_artifact,
        save_teacher_artifact,
    )

    camera_index = np.array([0, 0, 1, 1], dtype=np.int16)
    gates = assemble_teacher_gate_matrix(
        camera_index,
        {
            0: np.array([[0.91, 0.92], [0.93, 0.94]]),
            1: np.array([[0.95, 0.96], [0.97, 0.98]]),
        },
    )
    path = tmp_path / "teacher.npz"
    save_teacher_artifact(
        path,
        gates=gates,
        carrier_ids=np.array([5, 9]),
        camera_index=camera_index,
        frame_index=np.array([0, 5, 0, 5]),
        minimum_gate=0.9,
        maximum_gate=1.0,
        source_fingerprints={"evidence": "abc", "bank": "def"},
    )
    loaded = load_teacher_artifact(path)

    np.testing.assert_allclose(loaded["gates"], gates)
    np.testing.assert_array_equal(loaded["carrier_ids"], [5, 9])
    assert str(loaded["source_evidence_fingerprint"]) == "abc"
    assert int(loaded["paper_test_eligible"]) == 0
    assert str(loaded["output_fingerprint"])
