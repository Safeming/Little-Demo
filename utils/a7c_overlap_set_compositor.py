from __future__ import annotations

import math

import torch
from torch import nn


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def dense_overlap_adjacency(
    *,
    projected_xy: torch.Tensor,
    log_depth: torch.Tensor,
    visibility: torch.Tensor,
    spatial_scale: float,
    depth_scale: float,
    edge_log_weight_minimum: float,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    for name, value in (
        ("projected_xy", projected_xy),
        ("log_depth", log_depth),
        ("visibility", visibility),
    ):
        _require_finite(name, value)
    if projected_xy.ndim != 3 or projected_xy.shape[-1] != 2:
        raise ValueError("projected_xy must have shape [samples, carriers, 2]")
    shape = projected_xy.shape[:2]
    if log_depth.shape != shape or visibility.shape != shape:
        raise ValueError("graph inputs must share [samples, carriers]")
    if torch.any(visibility < 0.0) or torch.any(visibility > 1.0):
        raise ValueError("visibility must be in [0, 1]")
    if not math.isfinite(float(spatial_scale)) or float(spatial_scale) <= 0.0:
        raise ValueError("spatial_scale must be finite and positive")
    if not math.isfinite(float(depth_scale)) or float(depth_scale) <= 0.0:
        raise ValueError("depth_scale must be finite and positive")
    if not math.isfinite(float(edge_log_weight_minimum)):
        raise ValueError("edge_log_weight_minimum must be finite")
    if float(edge_log_weight_minimum) > 0.0:
        raise ValueError("edge_log_weight_minimum must be nonpositive")

    delta_xy = projected_xy[:, :, None, :] - projected_xy[:, None, :, :]
    delta_depth = log_depth[:, :, None] - log_depth[:, None, :]
    log_weight = -0.5 * torch.sum(delta_xy.square(), dim=-1) / float(
        spatial_scale
    ) ** 2
    log_weight = log_weight - 0.5 * delta_depth.square() / float(
        depth_scale
    ) ** 2
    log_weight = torch.clamp(
        log_weight,
        min=float(edge_log_weight_minimum),
        max=0.0,
    )
    visible_pair = visibility[:, :, None] * visibility[:, None, :]
    identity = torch.eye(
        shape[1], dtype=torch.bool, device=projected_xy.device
    )
    weight = (
        torch.exp(log_weight)
        * visible_pair
        * (~identity).unsqueeze(0)
    )
    denominator = weight.sum(dim=-1, keepdim=True)
    return torch.where(
        denominator > float(epsilon),
        weight / denominator.clamp_min(float(epsilon)),
        torch.zeros_like(weight),
    )


class DenseOverlapSetCompositor(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        node_hidden_dimension: int,
        gate_hidden_dimension: int,
        *,
        minimum_gate: float,
        maximum_gate: float = 1.0,
        initial_gate: float = 0.999,
    ) -> None:
        super().__init__()
        if int(input_dimension) <= 0:
            raise ValueError("input_dimension must be positive")
        if int(node_hidden_dimension) <= 0 or int(gate_hidden_dimension) <= 0:
            raise ValueError("hidden dimensions must be positive")
        if not (
            float(minimum_gate)
            < float(initial_gate)
            < float(maximum_gate)
        ):
            raise ValueError("initial_gate must be strictly inside gate bounds")
        self.node_encoder = nn.Sequential(
            nn.Linear(int(input_dimension), int(node_hidden_dimension)),
            nn.SiLU(),
            nn.Linear(int(node_hidden_dimension), int(node_hidden_dimension)),
            nn.SiLU(),
        )
        final = nn.Linear(int(gate_hidden_dimension), 1)
        nn.init.zeros_(final.weight)
        ratio = (float(initial_gate) - float(minimum_gate)) / (
            float(maximum_gate) - float(minimum_gate)
        )
        nn.init.constant_(final.bias, math.log(ratio / (1.0 - ratio)))
        self.gate_head = nn.Sequential(
            nn.Linear(4 * int(node_hidden_dimension), int(gate_hidden_dimension)),
            nn.SiLU(),
            final,
        )
        self.minimum_gate = float(minimum_gate)
        self.maximum_gate = float(maximum_gate)

    def forward(
        self,
        features: torch.Tensor,
        projected_xy: torch.Tensor,
        log_depth: torch.Tensor,
        visibility: torch.Tensor,
        *,
        spatial_scale: float,
        depth_scale: float,
        edge_log_weight_minimum: float,
    ) -> torch.Tensor:
        _require_finite("features", features)
        if features.ndim != 3:
            raise ValueError("features must have shape [samples, carriers, channels]")
        if projected_xy.shape[:2] != features.shape[:2]:
            raise ValueError("features and graph geometry must align")
        adjacency = dense_overlap_adjacency(
            projected_xy=projected_xy,
            log_depth=log_depth,
            visibility=visibility,
            spatial_scale=spatial_scale,
            depth_scale=depth_scale,
            edge_log_weight_minimum=edge_log_weight_minimum,
        )
        node = self.node_encoder(features)
        message = torch.bmm(adjacency, node)
        visible = visibility.unsqueeze(-1)
        global_context = torch.sum(node * visible, dim=1, keepdim=True) / torch.clamp(
            torch.sum(visible, dim=1, keepdim=True), min=1.0
        )
        global_context = global_context.expand(-1, node.shape[1], -1)
        context = torch.cat(
            (node, message, node - message, global_context), dim=-1
        )
        unit = torch.sigmoid(self.gate_head(context)).squeeze(-1)
        return self.minimum_gate + (
            self.maximum_gate - self.minimum_gate
        ) * unit
