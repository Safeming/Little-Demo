import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from utils.a7c_r1_4vp_r4a import (
    freeze_initial_scales,
    mean_normalized_trajectory,
    reconstruct_renderer_sequence,
    signed_renderer_trajectory_components,
    signed_renderer_trajectory_loss,
    signed_trajectory_component,
)


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


def _cosine(left, right):
    left = left.flatten()
    right = right.flatten()
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    assert torch.linalg.vector_norm(left) > 0
    assert torch.linalg.vector_norm(right) > 0
    return torch.dot(left, right) / (
        torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    )


def test_reconstruct_renderer_sequence_preserves_signed_point_contributions():
    base = torch.tensor([10.0, 12.0])
    point = torch.tensor([[2.0, -1.0], [4.0, -2.0]])
    gates = torch.tensor([[0.5, 1.0], [0.25, 0.5]], requires_grad=True)

    result = reconstruct_renderer_sequence(base, point, gates, epsilon=1e-12)

    torch.testing.assert_close(result, torch.tensor([9.0, 10.0]))
    result.sum().backward()
    torch.testing.assert_close(gates.grad, point)


def test_mean_normalized_trajectory_uses_its_own_differentiable_mean():
    values = torch.tensor([2.0, 4.0, 6.0], requires_grad=True)

    normalized = mean_normalized_trajectory(values, epsilon=1e-12)

    torch.testing.assert_close(normalized, torch.tensor([0.5, 1.0, 1.5]))
    normalized[0].backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_signed_components_are_zero_for_teacher_trajectory():
    streams = {
        signal: {
            "base": torch.tensor([10.0, 11.0, 9.0]),
            "point": torch.tensor(
                [[2.0, 1.0], [1.0, 3.0], [2.0, 2.0]]
            ),
        }
        for signal in ("target", "outer", "boundary")
    }
    gates = torch.tensor([[0.9, 1.0], [0.95, 0.9], [1.0, 0.95]])

    components = signed_renderer_trajectory_components(
        gates,
        gates,
        streams,
        renderer_delta=0.005,
        target_delta=0.005,
        gate_delta=0.01,
        gate_temporal_weight=0.25,
        epsilon=1e-12,
    )

    for value in components.values():
        assert float(value) == pytest.approx(0.0)


def test_frozen_scales_and_total_loss_follow_registered_coefficients():
    initial = {
        "outer": torch.tensor(2.0),
        "boundary": torch.tensor(4.0),
        "target": torch.tensor(0.5),
        "gate_aux": torch.tensor(0.25),
    }
    scales = freeze_initial_scales(initial, minimum=1e-12)
    components = {
        "outer": torch.tensor(1.0),
        "boundary": torch.tensor(1.0),
        "target": torch.tensor(0.25),
        "gate_aux": torch.tensor(0.125),
    }

    result = signed_renderer_trajectory_loss(
        components,
        scales,
        torch.tensor([[2.0, -2.0]]),
        outer_weight=1.0,
        boundary_weight=1.0,
        target_weight=0.1,
        gate_aux_weight=0.1,
        residual_weight=0.00001,
    )

    assert float(result["loss"]) == pytest.approx(0.85002)


def test_signed_trajectory_gradient_aligns_better_than_absolute_gate_weighting():
    base = torch.tensor(
        [9.5813, 12.1719, 5.2744, 13.3808], dtype=torch.float64
    )
    point = torch.tensor(
        [
            [0.5558, -1.2021, 0.3129],
            [-2.9279, -2.0738, 1.2563],
            [-0.1985, 2.3848, 1.2629],
            [2.6334, 2.1119, -2.2786],
        ],
        dtype=torch.float64,
    )
    candidate = torch.tensor(
        [
            [0.9846, 0.9695, 0.9184],
            [0.9143, 0.9306, 0.9809],
            [0.9388, 0.9541, 0.9501],
            [0.9631, 0.9435, 0.9068],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    teacher = torch.tensor(
        [
            [0.9961, 0.9572, 0.9037],
            [0.9196, 0.9942, 0.9959],
            [0.9512, 0.9362, 0.9438],
            [0.9335, 0.9500, 0.9622],
        ],
        dtype=torch.float64,
    )

    signed = signed_trajectory_component(
        base, point, candidate, teacher, delta=0.005, epsilon=1e-12
    )
    rendered = reconstruct_renderer_sequence(
        base, point, candidate, epsilon=1e-12
    )
    true_flicker = torch.mean(torch.abs(torch.diff(rendered))) / torch.abs(
        rendered.mean()
    )
    absolute_weight = torch.abs(point)
    absolute_weight /= absolute_weight.mean(dim=1, keepdim=True) + 1e-12
    proxy = torch.sum(
        absolute_weight
        * F.huber_loss(candidate, teacher, reduction="none", delta=0.01)
    ) / torch.sum(absolute_weight)
    signed_gradient = torch.autograd.grad(
        signed, candidate, retain_graph=True
    )[0]
    renderer_gradient = torch.autograd.grad(
        true_flicker, candidate, retain_graph=True
    )[0]
    proxy_gradient = torch.autograd.grad(proxy, candidate)[0]

    assert _cosine(signed_gradient, renderer_gradient) > 0.90
    assert _cosine(signed_gradient, renderer_gradient) > (
        _cosine(proxy_gradient, renderer_gradient) + 0.50
    )


def test_renderer_reconstruction_rejects_shape_nonfinite_and_zero_mean():
    with pytest.raises(ValueError, match="align"):
        reconstruct_renderer_sequence(
            torch.ones(2), torch.ones(2, 3), torch.ones(2, 2), epsilon=1e-12
        )
    bad = torch.ones(2, 2)
    bad[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        reconstruct_renderer_sequence(
            torch.ones(2), bad, torch.ones(2, 2), epsilon=1e-12
        )
    with pytest.raises(ValueError, match="mean"):
        reconstruct_renderer_sequence(
            torch.tensor([1.0, -1.0]),
            torch.zeros(2, 2),
            torch.ones(2, 2),
            epsilon=1e-12,
        )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
def test_initial_scales_reject_nonpositive_or_nonfinite_values(value):
    components = {
        name: torch.tensor(1.0)
        for name in ("outer", "boundary", "target", "gate_aux")
    }
    components["outer"] = torch.tensor(value)
    with pytest.raises(ValueError, match="scale"):
        freeze_initial_scales(components, minimum=1e-12)


def test_total_loss_rejects_missing_component_and_wrong_residual_weight():
    scales = {
        name: 1.0 for name in ("outer", "boundary", "target", "gate_aux")
    }
    components = {name: torch.tensor(1.0) for name in scales}
    del components["target"]
    with pytest.raises(ValueError, match="component"):
        signed_renderer_trajectory_loss(
            components,
            scales,
            torch.zeros(2, 2),
            outer_weight=1.0,
            boundary_weight=1.0,
            target_weight=0.1,
            gate_aux_weight=0.1,
            residual_weight=0.00001,
        )
    components["target"] = torch.tensor(1.0)
    with pytest.raises(ValueError, match="residual_weight"):
        signed_renderer_trajectory_loss(
            components,
            scales,
            torch.zeros(2, 2),
            outer_weight=1.0,
            boundary_weight=1.0,
            target_weight=0.1,
            gate_aux_weight=0.1,
            residual_weight=0.001,
        )
