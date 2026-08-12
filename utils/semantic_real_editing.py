from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from tools.analyze_projected_soft_edit_leakage import make_boundary_band
from utils.part_label_bank import PART_NAMES, compute_soft_edit_weights


REAL_EDIT_METHODS = ("raw_hard", "voting", "a5", "a7", "saga")
REAL_EDIT_TASKS = ("recolor", "removal", "texture")


def _hard_labels(bank: Mapping) -> np.ndarray:
    if "editable_label" in bank:
        return np.asarray(bank["editable_label"], dtype=np.int16).reshape(-1)
    if "part_label" in bank:
        return np.asarray(bank["part_label"], dtype=np.int16).reshape(-1)
    raise ValueError("bank must contain editable_label or part_label")


def resolve_method_weights(
    raw_bank: Mapping,
    voting_bank: Mapping,
    a5_bank: Mapping,
    *,
    a7_bank: Mapping | None = None,
    saga_bank: Mapping | None = None,
    method: str,
    part: str,
    threshold: float,
) -> np.ndarray:
    method_name = str(method).strip().lower()
    if method_name not in REAL_EDIT_METHODS:
        raise ValueError(f"unsupported edit method: {method}")
    if part not in PART_NAMES:
        raise ValueError(f"unsupported semantic part: {part}")
    part_index = PART_NAMES.index(part)
    if method_name in ("raw_hard", "voting"):
        labels = _hard_labels(raw_bank if method_name == "raw_hard" else voting_bank)
        return (labels == part_index).astype(np.float32)

    if method_name == "saga":
        required = ("semantic_probs", "confidence", "semantic_margin")
        if saga_bank is None or any(field not in saga_bank for field in required):
            raise ValueError("SAGA bank must contain semantic_probs, confidence, and semantic_margin")
        probabilities = np.asarray(saga_bank["semantic_probs"], dtype=np.float32)
        point_count = _hard_labels(saga_bank).shape[0]
        if probabilities.shape != (point_count, len(PART_NAMES)):
            raise ValueError(f"SAGA semantic_probs must have shape ({point_count}, {len(PART_NAMES)})")
        weights = compute_soft_edit_weights(
            semantic_probs=probabilities,
            confidence=saga_bank["confidence"],
            semantic_margin=saga_bank["semantic_margin"],
            reliable_mask=saga_bank.get("reliable_mask"),
        )[:, part_index]
        return np.where(weights >= float(threshold), weights, 0.0).astype(np.float32, copy=False)

    soft_bank = a5_bank if method_name == "a5" else a7_bank
    owner = "A5" if method_name == "a5" else "A7"
    if soft_bank is None or "soft_edit_weights" not in soft_bank:
        raise ValueError(f"{owner} bank must contain soft_edit_weights")
    weights = np.asarray(soft_bank["soft_edit_weights"], dtype=np.float32)
    point_count = _hard_labels(soft_bank).shape[0]
    if weights.shape != (point_count, len(PART_NAMES)):
        raise ValueError(f"soft_edit_weights must have shape ({point_count}, {len(PART_NAMES)})")
    values = weights[:, part_index]
    return np.where(values >= float(threshold), values, 0.0).astype(np.float32, copy=False)


def canonical_stripe_colors(
    canonical_xyz,
    *,
    primary_rgb: Sequence[float],
    secondary_rgb: Sequence[float],
    frequency: float = 8.0,
    axis: int = 1,
) -> np.ndarray:
    xyz = np.asarray(canonical_xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("canonical_xyz must have shape [N, 3]")
    axis_index = int(axis)
    if axis_index not in (0, 1, 2):
        raise ValueError("texture axis must be 0, 1, or 2")
    values = xyz[:, axis_index]
    span = float(np.max(values) - np.min(values)) if values.size else 0.0
    normalized = np.zeros_like(values) if span <= 1.0e-8 else (values - float(np.min(values))) / span
    stripe = (np.floor(normalized * max(float(frequency), 1.0)) % 2).astype(bool)
    primary = np.asarray(primary_rgb, dtype=np.float32).reshape(1, 3)
    secondary = np.asarray(secondary_rgb, dtype=np.float32).reshape(1, 3)
    colors = np.where(stripe[:, None], secondary, primary)
    return np.clip(colors, 0.0, 1.0).astype(np.float32, copy=False)


def build_edit_overrides(
    base_colors,
    base_opacities,
    weights,
    *,
    task: str,
    strength: float,
    target_rgb: Sequence[float],
    texture_colors,
) -> dict[str, np.ndarray]:
    colors = np.asarray(base_colors, dtype=np.float32)
    opacities = np.asarray(base_opacities, dtype=np.float32)
    weight_values = np.asarray(weights, dtype=np.float32).reshape(-1)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError("base_colors must have shape [N, 3]")
    if opacities.shape not in ((colors.shape[0],), (colors.shape[0], 1)):
        raise ValueError("base_opacities must have shape [N] or [N, 1]")
    if weight_values.shape[0] != colors.shape[0]:
        raise ValueError("weights must match Gaussian count")
    amount = np.clip(weight_values * float(strength), 0.0, 1.0)[:, None]
    task_name = str(task).strip().lower()
    edited_colors = colors.copy()
    edited_opacities = opacities.reshape(-1, 1).copy()
    if task_name == "recolor":
        target = np.asarray(target_rgb, dtype=np.float32).reshape(1, 3)
        edited_colors = colors * (1.0 - amount) + target * amount
    elif task_name == "removal":
        edited_opacities = edited_opacities * (1.0 - amount)
    elif task_name == "texture":
        pattern = np.asarray(texture_colors, dtype=np.float32)
        if pattern.shape != colors.shape:
            raise ValueError("texture_colors must match base_colors shape")
        edited_colors = colors * (1.0 - amount) + pattern * amount
    else:
        raise ValueError(f"unsupported edit task: {task}")
    return {
        "colors": np.clip(edited_colors, 0.0, 1.0).astype(np.float32, copy=False),
        "opacities": np.clip(edited_opacities, 0.0, 1.0).astype(np.float32, copy=False),
    }


def compute_edit_delta_metrics(
    base_image,
    edited_image,
    target_mask,
    valid_mask,
    *,
    boundary_radius: int = 2,
    mask_threshold: float = 0.5,
) -> dict[str, float | int]:
    base = np.asarray(base_image, dtype=np.float32)
    edited = np.asarray(edited_image, dtype=np.float32)
    target = np.asarray(target_mask, dtype=np.float32) >= float(mask_threshold)
    valid = np.asarray(valid_mask, dtype=np.float32) >= float(mask_threshold)
    if base.shape != edited.shape or base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("base_image and edited_image must have matching shape [H, W, 3]")
    if target.shape != base.shape[:2] or valid.shape != base.shape[:2]:
        raise ValueError("target_mask and valid_mask must match image dimensions")
    target_region = target & valid
    outer_region = (~target) & valid
    boundary_outer = make_boundary_band(target, radius=int(boundary_radius), threshold=0.5) & outer_region
    delta = np.sum(np.abs(edited - base), axis=2)

    def region_values(region: np.ndarray) -> tuple[int, float, float]:
        count = int(np.sum(region))
        total = float(np.sum(delta[region]))
        return count, total, total / max(count, 1)

    target_count, target_sum, target_mean = region_values(target_region)
    outer_count, outer_sum, outer_mean = region_values(outer_region)
    boundary_count, boundary_sum, boundary_mean = region_values(boundary_outer)
    return {
        "target_pixel_count": target_count,
        "outer_pixel_count": outer_count,
        "boundary_outer_pixel_count": boundary_count,
        "target_delta_sum": target_sum,
        "outer_delta_sum": outer_sum,
        "boundary_outer_delta_sum": boundary_sum,
        "target_delta_mean": target_mean,
        "outer_delta_mean": outer_mean,
        "boundary_outer_delta_mean": boundary_mean,
        "outer_to_target_delta_ratio": outer_sum / max(target_sum, 1.0e-8),
    }
