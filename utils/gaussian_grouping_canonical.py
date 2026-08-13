from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def estimate_queue_seconds(
    rows: Sequence[Mapping],
    *,
    canary_iterations: int,
    formal_iterations: int,
    subject_count: int,
    buffer_ratio: float = 0.15,
) -> Dict[str, float]:
    """Estimate a formal queue from the canary slope after its first 20 percent."""
    if canary_iterations <= 0 or formal_iterations <= 0 or subject_count <= 0:
        raise ValueError("iteration and subject counts must be positive")
    if buffer_ratio < 0.0:
        raise ValueError("buffer_ratio must be non-negative")
    ordered = sorted(
        (row for row in rows if int(row["iteration"]) >= max(1, int(canary_iterations * 0.2))),
        key=lambda row: int(row["iteration"]),
    )
    if len(ordered) < 2:
        raise ValueError("canary metrics need at least two steady-state rows")
    first, last = ordered[0], ordered[-1]
    iteration_delta = int(last["iteration"]) - int(first["iteration"])
    elapsed_delta = float(last["elapsed_seconds"]) - float(first["elapsed_seconds"])
    if iteration_delta <= 0 or elapsed_delta <= 0.0:
        raise ValueError("canary steady-state slope must be positive")
    seconds_per_iteration = elapsed_delta / iteration_delta
    base_seconds = seconds_per_iteration * int(formal_iterations) * int(subject_count)
    return {
        "steady_seconds_per_iteration": float(seconds_per_iteration),
        "base_seconds": float(base_seconds),
        "buffer_ratio": float(buffer_ratio),
        "estimated_seconds": float(base_seconds * (1.0 + buffer_ratio)),
    }


def _generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def balanced_pixel_indices(
    labels: torch.Tensor,
    *,
    samples_per_class: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample up to a fixed number of pixels from each non-negative class."""
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    flat = labels.reshape(-1)
    generator = _generator(flat.device, seed)
    selected = []
    for class_id in torch.unique(flat[flat >= 0], sorted=True):
        members = torch.nonzero(flat == class_id, as_tuple=False).flatten()
        if members.numel() > samples_per_class:
            order = torch.randperm(members.numel(), device=flat.device, generator=generator)
            members = members[order[:samples_per_class]]
        selected.append(members)
    if not selected:
        return torch.empty((0,), dtype=torch.long, device=flat.device)
    indices = torch.cat(selected)
    order = torch.randperm(indices.numel(), device=flat.device, generator=generator)
    return indices[order]


def grouping_3d_consistency_loss(
    xyz: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    k: int = 5,
    lambda_val: float = 2.0,
    max_points: int = 200000,
    sample_size: int = 1000,
    seed: int | None = None,
) -> torch.Tensor:
    """Gaussian Grouping's KNN probability KL loss on canonical points."""
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if probabilities.ndim != 2 or probabilities.shape[0] != xyz.shape[0]:
        raise ValueError("probabilities must have shape [N, C]")
    if xyz.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError("xyz and probabilities must be non-empty")
    if not 1 <= int(k) <= int(xyz.shape[0]):
        raise ValueError("k must be between 1 and the point count")
    if sample_size <= 0 or max_points <= 0:
        raise ValueError("sample_size and max_points must be positive")

    generator = _generator(xyz.device, seed)
    if xyz.shape[0] > max_points:
        keep = torch.randperm(xyz.shape[0], device=xyz.device, generator=generator)[:max_points]
        xyz = xyz[keep]
        probabilities = probabilities[keep]

    take = min(int(sample_size), int(xyz.shape[0]))
    sampled = torch.randperm(xyz.shape[0], device=xyz.device, generator=generator)[:take]
    sample_xyz = xyz[sampled]
    sample_probabilities = probabilities[sampled]
    distances = torch.cdist(sample_xyz, xyz)
    neighbors = distances.topk(k=min(int(k), int(xyz.shape[0])), largest=False).indices
    neighbor_probabilities = probabilities[neighbors]
    kl = sample_probabilities.unsqueeze(1) * (
        torch.log(sample_probabilities.unsqueeze(1) + 1.0e-10)
        - torch.log(neighbor_probabilities + 1.0e-10)
    )
    normalized = kl.sum(dim=-1).mean() / probabilities.shape[1]
    return float(lambda_val) * normalized


def identity_predictions(
    encodings: torch.Tensor,
    classifier_weight: torch.Tensor,
    classifier_bias: torch.Tensor | None = None,
) -> Dict[str, np.ndarray]:
    """Apply the shared identity classifier and export standard point predictions."""
    if encodings.ndim != 2 or classifier_weight.ndim != 2:
        raise ValueError("encodings and classifier_weight must be matrices")
    if encodings.shape[1] != classifier_weight.shape[1]:
        raise ValueError("identity dimension does not match classifier weight")
    logits = F.linear(encodings.float(), classifier_weight.float(), classifier_bias)
    probabilities = torch.softmax(logits, dim=-1)
    top_values, top_indices = probabilities.topk(k=min(2, probabilities.shape[1]), dim=-1)
    margin = top_values[:, 0] if probabilities.shape[1] == 1 else top_values[:, 0] - top_values[:, 1]
    return {
        "semantic_probs": probabilities.detach().cpu().numpy().astype(np.float32),
        "part_label": top_indices[:, 0].detach().cpu().numpy().astype(np.int16),
        "editable_label": top_indices[:, 0].detach().cpu().numpy().astype(np.int16),
        "confidence": top_values[:, 0].detach().cpu().numpy().astype(np.float32),
        "semantic_margin": margin.detach().cpu().numpy().astype(np.float32),
    }
