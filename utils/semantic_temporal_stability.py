from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def compute_screen_selection_metrics(
    selection,
    target_mask,
    valid_mask,
    *,
    threshold: float = 0.2,
) -> dict[str, float | int]:
    predicted = np.asarray(selection, dtype=np.float32)
    target = np.asarray(target_mask, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=np.float32)
    if predicted.shape != target.shape or predicted.shape != valid.shape:
        raise ValueError("selection, target_mask, and valid_mask must have matching shape")
    if predicted.ndim != 2:
        raise ValueError("screen-space masks must be two-dimensional")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all() or not np.isfinite(valid).all():
        raise ValueError("screen-space masks must contain finite values")

    valid_region = valid >= 0.5
    target_region = (target >= 0.5) & valid_region
    outer_region = (~target_region) & valid_region
    soft_prediction = np.clip(predicted, 0.0, 1.0) * valid_region.astype(np.float32)
    soft_target = target_region.astype(np.float32)
    inside_mass = float(np.sum(soft_prediction[target_region]))
    outside_mass = float(np.sum(soft_prediction[outer_region]))
    soft_intersection = float(np.minimum(soft_prediction, soft_target).sum())
    soft_union = float(np.maximum(soft_prediction, soft_target).sum())

    hard_prediction = (soft_prediction >= float(threshold)) & valid_region
    hard_intersection = int(np.sum(hard_prediction & target_region))
    hard_union = int(np.sum(hard_prediction | target_region))
    predicted_count = int(np.sum(hard_prediction))
    target_count = int(np.sum(target_region))
    return {
        "target_pixel_count": target_count,
        "valid_pixel_count": int(np.sum(valid_region)),
        "inside_selection_mass": inside_mass,
        "outside_selection_mass": outside_mass,
        "selection_leakage_ratio": outside_mass / max(inside_mass, 1.0e-8),
        "screen_soft_iou": soft_intersection / max(soft_union, 1.0e-8),
        "screen_hard_iou": hard_intersection / max(hard_union, 1),
        "screen_precision": hard_intersection / max(predicted_count, 1),
        "screen_recall": hard_intersection / max(target_count, 1),
    }


def summarize_temporal_signal(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("temporal signal must contain at least one value")
    if not np.isfinite(array).all():
        raise ValueError("temporal signal values must be finite")
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    scale = abs(mean)
    cv = std / scale if scale > 1.0e-8 else 0.0
    adjacent = float(np.mean(np.abs(np.diff(array)))) if array.size > 1 else 0.0
    flicker = adjacent / scale if scale > 1.0e-8 else 0.0
    return {
        "mean": mean,
        "std": std,
        "cv": cv,
        "adjacent_flicker": flicker,
    }
