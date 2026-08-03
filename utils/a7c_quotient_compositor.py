from __future__ import annotations

import numpy as np
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


def contiguous_training_segments(
    *,
    train_mask,
    camera_index,
    frame_index,
    frame_stride: int,
    block_ids,
) -> list[np.ndarray]:
    mask = np.asarray(train_mask, dtype=bool).reshape(-1)
    cameras = np.asarray(camera_index).reshape(-1)
    frames = np.asarray(frame_index).reshape(-1)
    blocks = np.asarray(block_ids).reshape(-1)
    if not (
        cameras.shape == mask.shape
        and frames.shape == mask.shape
        and blocks.shape == mask.shape
    ):
        raise ValueError("segment arrays must share one sample dimension")
    segments: list[np.ndarray] = []
    current: list[int] = []
    for index in range(mask.size):
        continuous = bool(
            current
            and mask[index]
            and cameras[index] == cameras[current[-1]]
            and blocks[index] == blocks[current[-1]]
            and frames[index] - frames[current[-1]] == int(frame_stride)
        )
        if current and not continuous:
            if len(current) > 1:
                segments.append(np.asarray(current, dtype=np.int64))
            current = []
        if mask[index]:
            current.append(index)
    if len(current) > 1:
        segments.append(np.asarray(current, dtype=np.int64))
    if not segments:
        raise ValueError("training mask contains no adjacent segment")
    return segments


def compose_contribution(
    base: torch.Tensor, point: torch.Tensor, gates: torch.Tensor
) -> torch.Tensor:
    if point.shape != gates.shape or base.shape != (gates.shape[0],):
        raise ValueError("base, point, and gates have incompatible shapes")
    return base - point.sum(dim=1) + (point * gates).sum(dim=1)


def torch_normalized_flicker(
    values: torch.Tensor, epsilon: float = 1.0e-8
) -> torch.Tensor:
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("normalized flicker requires at least two scalar samples")
    adjacent = torch.mean(torch.abs(values[1:] - values[:-1]))
    return adjacent / torch.clamp(
        torch.abs(values.mean()), min=float(epsilon)
    )


def renderer_sequence_objective(
    *,
    gates: torch.Tensor,
    segments,
    objective_streams,
    guard_streams,
    contract,
) -> dict[str, torch.Tensor]:
    if not segments:
        raise ValueError("renderer objective requires non-empty segments")
    candidate = {
        role: {
            signal: compose_contribution(
                stream["base"], stream["point"], gates
            )
            for signal, stream in streams.items()
        }
        for role, streams in (
            ("objective", objective_streams),
            ("guard", guard_streams),
        )
    }
    outer_ratios = []
    boundary_ratios = []
    jump_terms = []
    active_indices = []
    for indices in segments:
        index = torch.as_tensor(
            np.asarray(indices, dtype=np.int64),
            dtype=torch.long,
            device=gates.device,
        )
        if index.numel() < 2:
            raise ValueError("every renderer objective segment needs two samples")
        active_indices.append(index)
        for signal, destination in (
            ("outer", outer_ratios),
            ("boundary", boundary_ratios),
        ):
            base = objective_streams[signal]["base"][index]
            value = candidate["objective"][signal][index]
            destination.append(
                torch_normalized_flicker(value)
                / torch.clamp(torch_normalized_flicker(base), min=1.0e-8)
            )
        jump_terms.append(torch.abs(gates[index][1:] - gates[index][:-1]))
    active = torch.unique(torch.cat(active_indices), sorted=True)
    outer_ratio = torch.stack(outer_ratios).mean()
    boundary_ratio = torch.stack(boundary_ratios).mean()
    base_target = guard_streams["target"]["base"][active]
    candidate_target = candidate["guard"]["target"][active]
    base_outer = guard_streams["outer"]["base"][active]
    candidate_outer = candidate["guard"]["outer"][active]
    target_response = candidate_target / torch.clamp(
        base_target, min=1.0e-8
    )
    base_iou = base_target / torch.clamp(
        base_target + base_outer, min=1.0e-8
    )
    candidate_iou = candidate_target / torch.clamp(
        candidate_target + candidate_outer, min=1.0e-8
    )
    target_hinge = torch.relu(
        float(contract["training_target_response"]) - target_response
    ).mean()
    soft_iou_hinge = torch.relu(
        base_iou
        - candidate_iou
        - float(contract["maximum_selection_soft_iou_drop"])
    ).mean()
    jumps = torch.cat([value.reshape(-1) for value in jump_terms])
    jump_hinge = torch.relu(
        jumps - float(contract["maximum_adjacent_gate_change"])
    ).mean()
    damping = torch.mean(torch.square(1.0 - gates[active]))
    loss = (
        float(contract["outer_loss_weight"]) * outer_ratio
        + float(contract["boundary_loss_weight"]) * boundary_ratio
        + float(contract["target_hinge_weight"]) * target_hinge
        + float(contract["soft_iou_hinge_weight"]) * soft_iou_hinge
        + float(contract["gate_jump_hinge_weight"]) * jump_hinge
        + float(contract["damping_regularizer_weight"]) * damping
    )
    return {
        "loss": loss,
        "outer_ratio": outer_ratio,
        "boundary_ratio": boundary_ratio,
        "target_hinge": target_hinge,
        "soft_iou_hinge": soft_iou_hinge,
        "jump_hinge": jump_hinge,
        "damping_regularizer": damping,
    }
