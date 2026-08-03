import json
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch


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


def test_runtime_target_mass_uses_only_runtime_probe_fields():
    from utils.a7c_quotient_compositor import runtime_target_mass

    mass = runtime_target_mass(
        alpha_transmittance_mass=torch.tensor([[2.0, 1.0]]),
        a5_weight=torch.tensor([0.8, 0.5]),
        semantic_support_mean=torch.tensor([[0.4, 0.0]]),
        alpha_mean=torch.tensor([[0.5, 0.0]]),
    )

    torch.testing.assert_close(mass, torch.tensor([[1.28, 0.0]]))


def test_joint_target_budget_scales_damping_and_preserves_topology():
    from utils.a7c_quotient_compositor import project_joint_target_budget

    projected = project_joint_target_budget(
        raw_gates=torch.tensor([[0.9, 0.9]]),
        runtime_mass=torch.tensor([[1.0, 1.0]]),
        a5_weight=torch.tensor([0.21, 0.8]),
        proxy_target_response=0.995,
        selection_threshold=0.2,
        minimum_gate=0.9,
    )

    assert float(projected.sum()) >= 1.99 - 1e-7
    assert float(projected[0, 0] * 0.21) >= 0.2 - 1e-7
    assert torch.all(projected >= 0.9)
    assert torch.all(projected <= 1.0)


def test_joint_target_budget_zero_mass_is_finite_and_keeps_raw_gate():
    from utils.a7c_quotient_compositor import project_joint_target_budget

    raw = torch.tensor([[0.93, 0.97]])
    projected = project_joint_target_budget(
        raw_gates=raw,
        runtime_mass=torch.zeros_like(raw),
        a5_weight=torch.ones(2),
        proxy_target_response=0.995,
        selection_threshold=0.2,
        minimum_gate=0.9,
    )

    assert torch.isfinite(projected).all()
    torch.testing.assert_close(projected, raw)


def test_projection_schema_has_no_ground_truth_or_contribution_inputs():
    from utils.a7c_quotient_compositor import project_joint_target_budget

    names = set(inspect.signature(project_joint_target_budget).parameters)
    forbidden_fragments = ("mask", "target_contribution", "outer", "boundary")
    assert not any(
        fragment in name
        for name in names
        for fragment in forbidden_fragments
    )


def _synthetic_streams():
    base_outer = torch.tensor([1.0, 2.0, 1.0, 2.0])
    zeros = torch.zeros(4, 1)
    objective = {
        "target": {"base": torch.ones(4), "point": zeros},
        "outer": {"base": base_outer, "point": base_outer[:, None]},
        "boundary": {"base": base_outer, "point": base_outer[:, None]},
    }
    guard = {
        "target": {"base": torch.ones(4), "point": zeros},
        "outer": {"base": base_outer, "point": base_outer[:, None]},
        "boundary": {"base": base_outer, "point": base_outer[:, None]},
    }
    contract = {
        "training_target_response": 0.995,
        "maximum_selection_soft_iou_drop": 0.005,
        "maximum_adjacent_gate_change": 0.02,
        "outer_loss_weight": 1.0,
        "boundary_loss_weight": 1.0,
        "target_hinge_weight": 100.0,
        "soft_iou_hinge_weight": 100.0,
        "gate_jump_hinge_weight": 20.0,
        "damping_regularizer_weight": 0.001,
    }
    return objective, guard, contract


def test_contiguous_training_segments_do_not_cross_gaps_cameras_or_blocks():
    from utils.a7c_quotient_compositor import contiguous_training_segments

    segments = contiguous_training_segments(
        train_mask=np.array([1, 1, 0, 1, 1, 1, 1, 0], bool),
        camera_index=np.array([0, 0, 0, 0, 0, 1, 1, 1]),
        frame_index=np.array([0, 5, 10, 15, 20, 0, 5, 10]),
        frame_stride=5,
        block_ids=np.array([0, 0, 0, 1, 1, 0, 0, 0]),
    )

    assert [value.tolist() for value in segments] == [[0, 1], [3, 4], [5, 6]]


def test_renderer_objective_prefers_gate_reducing_both_flicker_signals():
    from utils.a7c_quotient_compositor import renderer_sequence_objective

    objective, guard, contract = _synthetic_streams()
    good_gates = torch.tensor([[1.0], [0.981], [1.0], [0.981]])
    kwargs = {
        "segments": [np.arange(4)],
        "objective_streams": objective,
        "guard_streams": guard,
        "contract": contract,
    }
    good = renderer_sequence_objective(gates=good_gates, **kwargs)
    identity = renderer_sequence_objective(
        gates=torch.ones_like(good_gates), **kwargs
    )

    assert float(good["loss"]) < float(identity["loss"])
    assert float(good["outer_ratio"]) < 1.0
    assert float(good["boundary_ratio"]) < 1.0


def test_renderer_objective_guard_hinges_and_held_samples_are_isolated():
    from utils.a7c_quotient_compositor import renderer_sequence_objective

    objective, guard, contract = _synthetic_streams()
    guard["target"]["point"] = torch.ones(4, 1)
    guard["outer"]["point"] = torch.zeros(4, 1)
    gates = torch.tensor([[1.0], [1.0], [0.9], [0.9]])
    result = renderer_sequence_objective(
        gates=gates,
        segments=[np.array([0, 1])],
        objective_streams=objective,
        guard_streams=guard,
        contract=contract,
    )

    assert float(result["target_hinge"]) == pytest.approx(0.0)
    assert float(result["soft_iou_hinge"]) == pytest.approx(0.0)
    assert float(result["jump_hinge"]) == pytest.approx(0.0)

    violated = renderer_sequence_objective(
        gates=torch.tensor([[1.0], [0.9], [1.0], [1.0]]),
        segments=[np.array([0, 1])],
        objective_streams=objective,
        guard_streams=guard,
        contract=contract,
    )
    assert float(violated["target_hinge"]) > 0.0
    assert float(violated["soft_iou_hinge"]) > 0.0
    assert float(violated["jump_hinge"]) > 0.0
