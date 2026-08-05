from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F


CONTRIBUTION_SIGNALS = ("target", "outer", "boundary")
REGISTERED_RESIDUAL_LOSS_WEIGHT = 0.00001


def _validated_segments(
    segments: Iterable[np.ndarray], fit_mask: np.ndarray, sample_count: int
) -> list[np.ndarray]:
    normalized = []
    covered = np.zeros(sample_count, dtype=np.int64)
    for segment in segments:
        indices = np.asarray(segment, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("segments must contain nonempty one-dimensional indices")
        if np.any(indices < 0) or np.any(indices >= sample_count):
            raise ValueError("segment index is out of bounds")
        if not np.all(fit_mask[indices]):
            raise ValueError("segments may contain fit rows only")
        covered[indices] += 1
        normalized.append(indices)
    if np.any(covered > 1) or not np.array_equal(covered > 0, fit_mask):
        raise ValueError("segments must cover the fit mask exactly once")
    return normalized


def build_contribution_weights(
    point_contributions: Mapping[str, np.ndarray],
    fit_mask: np.ndarray,
    segments: Iterable[np.ndarray],
    *,
    epsilon: float,
    minimum: float,
    maximum: float,
) -> Dict[str, np.ndarray]:
    if set(point_contributions) != set(CONTRIBUTION_SIGNALS):
        raise ValueError(f"point_contributions must contain {CONTRIBUTION_SIGNALS}")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("weight bounds must be finite")
    if minimum <= 0.0 or maximum < minimum:
        raise ValueError("weight bounds must satisfy 0 < minimum <= maximum")

    arrays = {
        name: np.asarray(point_contributions[name], dtype=np.float64)
        for name in CONTRIBUTION_SIGNALS
    }
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("contribution arrays must have the same shape")
    shape = next(iter(shapes))
    if len(shape) != 2 or shape[0] == 0 or shape[1] == 0:
        raise ValueError("contribution arrays must be nonempty [samples, carriers]")

    mask = np.asarray(fit_mask, dtype=bool)
    if mask.shape != (shape[0],):
        raise ValueError("fit_mask must have one entry per sample")
    if not np.any(mask):
        raise ValueError("fit_mask must select at least one sample")
    for name, array in arrays.items():
        if not np.isfinite(array[mask]).all():
            raise ValueError(f"fit contribution rows for {name} must be finite")
        if np.isfinite(array[~mask]).any():
            raise ValueError("held contribution rows must remain nonfinite")

    fit_segments = _validated_segments(segments, mask, shape[0])
    preliminary = np.full(shape, np.nan, dtype=np.float64)
    normalized_sum = np.zeros((int(mask.sum()), shape[1]), dtype=np.float64)
    for array in arrays.values():
        contribution = np.abs(array[mask])
        frame_mean = contribution.mean(axis=1, keepdims=True)
        normalized_sum += contribution / (frame_mean + float(epsilon))
    preliminary[mask] = normalized_sum

    clipped = np.full(shape, np.nan, dtype=np.float64)
    clipped[mask] = np.clip(normalized_sum, float(minimum), float(maximum))
    gate = np.full(shape, np.nan, dtype=np.float64)
    for indices in fit_segments:
        segment_weight = clipped[indices]
        segment_mean = float(segment_weight.mean())
        if not np.isfinite(segment_mean) or segment_mean <= 0.0:
            raise ValueError("segment contribution weight mean must be positive")
        gate[indices] = segment_weight / segment_mean

    if not np.isfinite(gate[mask]).all() or np.any(gate[mask] <= 0.0):
        raise ValueError("fit contribution weights must be finite and positive")
    return {"gate": gate, "clipped": clipped, "preliminary": preliminary}


def temporal_segment_weights(gate_weight: np.ndarray) -> np.ndarray:
    weight = np.asarray(gate_weight, dtype=np.float64)
    if weight.ndim != 2 or weight.shape[0] < 2 or weight.shape[1] == 0:
        raise ValueError("gate_weight must be [frames >= 2, carriers]")
    if not np.isfinite(weight).all() or np.any(weight <= 0.0):
        raise ValueError("gate_weight must be finite and positive")
    return np.maximum(weight[:-1], weight[1:])


def _require_finite_positive(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()) or not bool((value > 0.0).all()):
        raise ValueError(f"{name} must be finite and positive")


def contribution_weighted_distillation_loss(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    residual: torch.Tensor,
    gate_weight: torch.Tensor,
    temporal_weight: torch.Tensor,
    *,
    gate_delta: float,
    temporal_delta: float,
    temporal_loss_weight: float,
    residual_loss_weight: float,
) -> Dict[str, torch.Tensor]:
    if prediction.ndim != 2 or prediction.shape[0] < 2:
        raise ValueError("prediction must be [frames >= 2, carriers]")
    if teacher.shape != prediction.shape or residual.shape != prediction.shape:
        raise ValueError("teacher and residual must match prediction shape")
    if gate_weight.shape != prediction.shape:
        raise ValueError("gate_weight must match prediction shape")
    if temporal_weight.shape != (prediction.shape[0] - 1, prediction.shape[1]):
        raise ValueError("temporal_weight must match adjacent prediction differences")
    for name, tensor in (
        ("prediction", prediction),
        ("teacher", teacher),
        ("residual", residual),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")
    _require_finite_positive("gate_weight", gate_weight)
    _require_finite_positive("temporal_weight", temporal_weight)
    if abs(float(residual_loss_weight) - REGISTERED_RESIDUAL_LOSS_WEIGHT) > 1e-12:
        raise ValueError(
            "residual_loss_weight must equal the registered R3 value 0.00001"
        )
    if gate_delta <= 0.0 or temporal_delta <= 0.0:
        raise ValueError("Huber deltas must be positive")
    if temporal_loss_weight < 0.0:
        raise ValueError("temporal_loss_weight must be nonnegative")

    gate_element = F.huber_loss(
        prediction, teacher, reduction="none", delta=float(gate_delta)
    )
    temporal_element = F.huber_loss(
        torch.diff(prediction, dim=0),
        torch.diff(teacher, dim=0),
        reduction="none",
        delta=float(temporal_delta),
    )
    gate = torch.sum(gate_weight * gate_element) / torch.sum(gate_weight)
    temporal = torch.sum(temporal_weight * temporal_element) / torch.sum(
        temporal_weight
    )
    latent = torch.mean(torch.abs(residual))
    loss = (
        gate
        + float(temporal_loss_weight) * temporal
        + float(residual_loss_weight) * latent
    )
    return {"loss": loss, "gate": gate, "temporal": temporal, "residual": latent}


def _finite_vector(name: str, value: Iterable[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def evaluate_fit_renderer_entry(
    *,
    learned_outer: Iterable[float],
    teacher_outer: Iterable[float],
    learned_boundary: Iterable[float],
    teacher_boundary: Iterable[float],
    minimum_outer_recovery: float,
    minimum_boundary_recovery: float,
    minimum_positive_fraction: float,
) -> Dict[str, object]:
    vectors = {
        "learned_outer": _finite_vector("learned_outer", learned_outer),
        "teacher_outer": _finite_vector("teacher_outer", teacher_outer),
        "learned_boundary": _finite_vector("learned_boundary", learned_boundary),
        "teacher_boundary": _finite_vector("teacher_boundary", teacher_boundary),
    }
    lengths = {vector.size for vector in vectors.values()}
    if len(lengths) != 1:
        raise ValueError("fit renderer vectors must have the same length")
    for name, value in (
        ("minimum_outer_recovery", minimum_outer_recovery),
        ("minimum_boundary_recovery", minimum_boundary_recovery),
        ("minimum_positive_fraction", minimum_positive_fraction),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if minimum_positive_fraction > 1.0:
        raise ValueError("minimum_positive_fraction must not exceed one")

    teacher_outer_mean = float(vectors["teacher_outer"].mean())
    teacher_boundary_mean = float(vectors["teacher_boundary"].mean())
    if teacher_outer_mean <= 0.0:
        raise ValueError("teacher_outer mean must be positive")
    if teacher_boundary_mean <= 0.0:
        raise ValueError("teacher_boundary mean must be positive")

    learned_outer_mean = float(vectors["learned_outer"].mean())
    learned_boundary_mean = float(vectors["learned_boundary"].mean())
    outer_recovery = learned_outer_mean / teacher_outer_mean
    boundary_recovery = learned_boundary_mean / teacher_boundary_mean
    outer_positive_fraction = float(np.mean(vectors["learned_outer"] > 0.0))
    boundary_positive_fraction = float(
        np.mean(vectors["learned_boundary"] > 0.0)
    )
    checks = {
        "outer_recovery": outer_recovery >= float(minimum_outer_recovery),
        "boundary_recovery": boundary_recovery >= float(minimum_boundary_recovery),
        "outer_positive_fraction": outer_positive_fraction
        >= float(minimum_positive_fraction),
        "boundary_positive_fraction": boundary_positive_fraction
        >= float(minimum_positive_fraction),
    }
    return {
        "passed": bool(all(checks.values())),
        "outer_recovery": outer_recovery,
        "boundary_recovery": boundary_recovery,
        "outer_positive_fraction": outer_positive_fraction,
        "boundary_positive_fraction": boundary_positive_fraction,
        "learned_outer_mean": learned_outer_mean,
        "teacher_outer_mean": teacher_outer_mean,
        "learned_boundary_mean": learned_boundary_mean,
        "teacher_boundary_mean": teacher_boundary_mean,
        "checks": checks,
        "failed_conditions": [name for name, passed in checks.items() if not passed],
    }


def classify_fit_entry_failure(fold_index: int) -> str:
    if not isinstance(fold_index, (int, np.integer)) or fold_index < 0:
        raise ValueError("fold_index must be a nonnegative integer")
    return "FIT_RENDERER_ENTRY_NEGATIVE" if fold_index == 0 else "TRAINING_ERROR"
