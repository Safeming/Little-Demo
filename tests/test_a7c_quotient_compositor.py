import json
import inspect
from pathlib import Path

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
