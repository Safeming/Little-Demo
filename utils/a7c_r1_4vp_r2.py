from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


REGISTERED_RESIDUAL_LOSS_WEIGHT = 0.00001


def r2_distillation_loss(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    residual: torch.Tensor,
    *,
    gate_delta: float,
    temporal_delta: float,
    temporal_weight: float,
    residual_weight: float,
) -> dict[str, torch.Tensor]:
    if prediction.shape != teacher.shape or residual.shape != prediction.shape:
        raise ValueError("distillation tensors must share shape")
    if prediction.ndim != 2 or prediction.shape[0] < 2:
        raise ValueError("distillation tensors need [frames, carriers]")
    if float(residual_weight) != REGISTERED_RESIDUAL_LOSS_WEIGHT:
        raise ValueError("R2 residual weight differs from preregistration")
    if min(float(gate_delta), float(temporal_delta)) <= 0.0:
        raise ValueError("Huber deltas must be positive")
    if float(temporal_weight) < 0.0:
        raise ValueError("temporal weight must be nonnegative")
    for name, value in (
        ("prediction", prediction),
        ("teacher", teacher),
        ("residual", residual),
    ):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    gate = F.huber_loss(
        prediction, teacher, reduction="mean", delta=float(gate_delta)
    )
    temporal = F.huber_loss(
        torch.diff(prediction, dim=0),
        torch.diff(teacher, dim=0),
        reduction="mean",
        delta=float(temporal_delta),
    )
    latent = torch.mean(torch.abs(residual))
    return {
        "loss": gate
        + float(temporal_weight) * temporal
        + float(residual_weight) * latent,
        "gate": gate,
        "temporal": temporal,
        "residual": latent,
    }


def require_fit_integrity(
    initial_loss: float,
    final_loss: float,
    fit_teacher_mae: float,
    maximum_fit_teacher_mae: float,
) -> None:
    values = np.asarray(
        [initial_loss, final_loss, fit_teacher_mae, maximum_fit_teacher_mae],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise RuntimeError("fit integrity metrics must be finite")
    if float(maximum_fit_teacher_mae) < 0.0:
        raise ValueError("maximum fit teacher MAE must be nonnegative")
    if not float(final_loss) < float(initial_loss):
        raise RuntimeError("fit loss did not improve")
    if float(fit_teacher_mae) > float(maximum_fit_teacher_mae):
        raise RuntimeError("fit teacher MAE exceeds frozen maximum")


def evaluate_topology_guard(base_weight, candidate_weight, threshold: float) -> dict:
    base = np.asarray(base_weight, dtype=np.float64).reshape(1, -1)
    candidate = np.asarray(candidate_weight, dtype=np.float64)
    if candidate.ndim != 2 or candidate.shape[1] != base.shape[1]:
        raise ValueError("candidate topology must have shape [frames, carriers]")
    if not np.isfinite(base).all() or not np.isfinite(candidate).all():
        raise ValueError("topology weights must be finite")
    expected = np.broadcast_to(base >= float(threshold), candidate.shape)
    observed = candidate >= float(threshold)
    mismatch_count = int(np.count_nonzero(observed != expected))
    selected_slack = candidate[expected] - float(threshold)
    return {
        "passed": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "minimum_slack": (
            float(np.min(selected_slack)) if selected_slack.size else float("inf")
        ),
    }


def classify_terminal_status(audit_status: int, verdict: str) -> str:
    mapping = {
        (0, "CANARY_PROMOTED"): "completed",
        (2, "CANARY_NEGATIVE"): "rejected",
    }
    return mapping.get((int(audit_status), str(verdict)), "failed")
