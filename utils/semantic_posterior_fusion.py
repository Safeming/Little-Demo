from __future__ import annotations

import numpy as np


def fuse_semantic_posteriors(
    trained_probs,
    voting_probs,
    *,
    voting_alpha: float,
) -> np.ndarray:
    trained = np.asarray(trained_probs, dtype=np.float32)
    voting = np.asarray(voting_probs, dtype=np.float32)
    if trained.shape != voting.shape:
        raise ValueError("trained and voting semantic probabilities must have the same shape")
    if trained.ndim != 2 or trained.shape[1] == 0:
        raise ValueError("semantic probabilities must have shape [N, C] with at least one class")
    alpha = float(voting_alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("voting alpha must be within [0, 1]")

    trained = np.where(np.isfinite(trained), np.clip(trained, 0.0, None), 0.0)
    voting = np.where(np.isfinite(voting), np.clip(voting, 0.0, None), 0.0)
    fused = (1.0 - alpha) * trained + alpha * voting
    row_sum = fused.sum(axis=1, keepdims=True)
    valid = row_sum.reshape(-1) > 0.0
    normalized = np.empty_like(fused, dtype=np.float32)
    normalized[valid] = fused[valid] / row_sum[valid]
    normalized[~valid] = 1.0 / float(fused.shape[1])
    return normalized.astype(np.float32, copy=False)
