from __future__ import annotations

import torch


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def runtime_target_mass(
    *,
    alpha_transmittance_mass: torch.Tensor,
    a5_weight: torch.Tensor,
    semantic_support_mean: torch.Tensor,
    alpha_mean: torch.Tensor,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    for name, value in (
        ("alpha_transmittance_mass", alpha_transmittance_mass),
        ("a5_weight", a5_weight),
        ("semantic_support_mean", semantic_support_mean),
        ("alpha_mean", alpha_mean),
    ):
        _require_finite(name, value)
    if alpha_transmittance_mass.ndim != 2:
        raise ValueError("alpha_transmittance_mass must have shape [samples, carriers]")
    if semantic_support_mean.shape != alpha_transmittance_mass.shape:
        raise ValueError("semantic_support_mean must match alpha_transmittance_mass")
    if alpha_mean.shape != alpha_transmittance_mass.shape:
        raise ValueError("alpha_mean must match alpha_transmittance_mass")
    if a5_weight.shape != (alpha_transmittance_mass.shape[1],):
        raise ValueError("a5_weight must have shape [carriers]")
    probability = torch.clamp(
        semantic_support_mean / torch.clamp(alpha_mean, min=float(epsilon)),
        min=0.0,
        max=1.0,
    )
    return (
        torch.clamp(alpha_transmittance_mass, min=0.0)
        * torch.clamp(a5_weight, min=0.0).unsqueeze(0)
        * probability
    )


def project_joint_target_budget(
    *,
    raw_gates: torch.Tensor,
    runtime_mass: torch.Tensor,
    a5_weight: torch.Tensor,
    proxy_target_response: float,
    selection_threshold: float,
    minimum_gate: float,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    for name, value in (
        ("raw_gates", raw_gates),
        ("runtime_mass", runtime_mass),
        ("a5_weight", a5_weight),
    ):
        _require_finite(name, value)
    if raw_gates.ndim != 2 or runtime_mass.shape != raw_gates.shape:
        raise ValueError("raw_gates and runtime_mass must have shape [samples, carriers]")
    if a5_weight.shape != (raw_gates.shape[1],):
        raise ValueError("a5_weight must have shape [carriers]")
    if not 0.0 <= float(minimum_gate) <= float(proxy_target_response) <= 1.0:
        raise ValueError("gate and proxy response bounds are invalid")
    if float(selection_threshold) < 0.0:
        raise ValueError("selection_threshold must be nonnegative")
    if torch.any(raw_gates < float(minimum_gate)) or torch.any(raw_gates > 1.0):
        raise ValueError("raw_gates violate frozen bounds")
    damping = 1.0 - raw_gates
    proxy_loss = torch.sum(runtime_mass * damping, dim=1, keepdim=True)
    budget = (1.0 - float(proxy_target_response)) * torch.sum(
        runtime_mass, dim=1, keepdim=True
    )
    scale = torch.clamp(
        budget / torch.clamp(proxy_loss, min=float(epsilon)), max=1.0
    )
    scale = torch.where(
        proxy_loss > float(epsilon), scale, torch.ones_like(scale)
    )
    gates = 1.0 - scale * damping
    topology_floor = torch.clamp(
        float(selection_threshold)
        / torch.clamp(a5_weight, min=float(epsilon)),
        max=1.0,
    )
    return torch.maximum(gates, topology_floor.unsqueeze(0)).clamp(
        min=float(minimum_gate), max=1.0
    )
