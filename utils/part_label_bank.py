from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


SCHEMA_VERSION = 1
PART_NAMES = ("hair", "face", "upper", "lower", "shoes", "skin")
PART_COLORS_UINT8 = np.array(
    [
        [42, 42, 42],
        [236, 168, 120],
        [55, 126, 184],
        [77, 175, 74],
        [152, 78, 163],
        [255, 204, 153],
    ],
    dtype=np.uint8,
)
UNKNOWN_COLOR_UINT8 = np.array([128, 128, 128], dtype=np.uint8)

A7_TEMPORAL_COUNT_FIELDS = (
    "temporal_visible_count",
    "temporal_consecutive_visible_count",
)
A7_TEMPORAL_FLOAT_FIELDS = (
    "temporal_target_ratio_mean",
    "temporal_target_ratio_std",
    "temporal_target_flicker",
    "temporal_outer_ratio_mean",
    "temporal_outer_ratio_std",
    "temporal_outer_flicker",
    "temporal_boundary_crossing_rate",
    "temporal_visibility_transition_rate",
)
A7_PROVENANCE_FIELDS = (
    "base_method_freeze_fingerprint",
    "a7_contract_fingerprint",
    "evidence_protocol_fingerprint",
    "candidate_config_fingerprint",
)


def _as_array(value):
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _require_int16_range(name: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.size and (value.min() < np.iinfo(np.int16).min or value.max() > np.iinfo(np.int16).max):
        raise ValueError(f"{name} values exceed int16 range")
    return value.astype(np.int16, copy=False)


def _one_hot_label_weights(bank: Mapping[str, np.ndarray], *, point_count: int) -> tuple[np.ndarray, str]:
    source_labels = np.asarray(bank.get("editable_label", bank["part_label"]), dtype=np.int16).reshape(-1)
    if source_labels.shape[0] != int(point_count):
        raise ValueError(f"label point count {source_labels.shape[0]} does not match expected {int(point_count)}")
    weights = np.zeros((int(point_count), len(PART_NAMES)), dtype=np.float32)
    valid = (source_labels >= 0) & (source_labels < len(PART_NAMES))
    if np.any(valid):
        weights[np.nonzero(valid)[0], source_labels[valid].astype(np.int64)] = 1.0
    source = "editable_label_one_hot_fallback" if "editable_label" in bank else "part_label_one_hot_fallback"
    return weights, source


def resolve_soft_edit_weights(
    bank: Mapping[str, np.ndarray],
    *,
    point_count: int,
    weight_source: str = "soft",
) -> tuple[np.ndarray, str]:
    source_key = str(weight_source or "soft").replace("_", "-").lower()
    key_by_source = {
        "soft": "soft_edit_weights",
        "target": "edit_target_weights",
        "support": "edit_support_weights",
    }
    if source_key == "auto-target":
        source_key = "target" if "edit_target_weights" in bank else "soft"
    if source_key == "auto-support":
        if "edit_support_weights" not in bank:
            return np.zeros((int(point_count), len(PART_NAMES)), dtype=np.float32), "zero_support_fallback"
        source_key = "support"
    if source_key not in key_by_source:
        raise ValueError(f"unsupported soft edit weight source: {weight_source}")

    key = key_by_source[source_key]
    if key in bank:
        weights = np.asarray(bank[key], dtype=np.float32)
        if weights.shape != (int(point_count), len(PART_NAMES)):
            raise ValueError(f"{key} must have shape ({int(point_count)}, {len(PART_NAMES)})")
        return weights, key
    if source_key == "soft":
        return _one_hot_label_weights(bank, point_count=point_count)
    raise ValueError(f"{key} is required for weight source {weight_source}")


def finalize_votes(per_part_votes, visible_vote_count, conflict_count) -> dict[str, np.ndarray]:
    votes_i32 = np.asarray(per_part_votes, dtype=np.int32)
    if votes_i32.ndim != 2 or votes_i32.shape[1] != len(PART_NAMES):
        raise ValueError(f"per_part_votes must have shape [N, {len(PART_NAMES)}]")
    visible_i32 = np.asarray(visible_vote_count, dtype=np.int32).reshape(-1)
    conflict_i32 = np.asarray(conflict_count, dtype=np.int32).reshape(-1)
    point_count = votes_i32.shape[0]
    if visible_i32.shape[0] != point_count:
        raise ValueError("visible_vote_count shape mismatch")
    if conflict_i32.shape[0] != point_count:
        raise ValueError("conflict_count shape mismatch")

    vote_count_i32 = votes_i32.sum(axis=1, dtype=np.int32)
    semantic_probs = np.zeros((point_count, len(PART_NAMES)), dtype=np.float32)
    has_any_vote = vote_count_i32 > 0
    if np.any(has_any_vote):
        semantic_probs[has_any_vote] = votes_i32[has_any_vote].astype(np.float32) / vote_count_i32[
            has_any_vote
        ].astype(np.float32)[:, None]
    part_label = np.full((point_count,), -1, dtype=np.int16)
    confidence = np.zeros((point_count,), dtype=np.float32)
    has_vote = vote_count_i32 > 0
    if np.any(has_vote):
        best_label = np.argmax(votes_i32[has_vote], axis=1).astype(np.int16)
        best_votes = np.max(votes_i32[has_vote], axis=1).astype(np.float32)
        part_label[has_vote] = best_label
        confidence[has_vote] = best_votes / vote_count_i32[has_vote].astype(np.float32)

    return {
        "part_label": part_label,
        "confidence": confidence.astype(np.float32, copy=False),
        "vote_count": _require_int16_range("vote_count", vote_count_i32),
        "per_part_votes": _require_int16_range("per_part_votes", votes_i32),
        "visible_vote_count": _require_int16_range("visible_vote_count", visible_i32),
        "conflict_count": _require_int16_range("conflict_count", conflict_i32),
        "semantic_probs": semantic_probs,
    }


def finalize_trained_semantic_probs(semantic_probs, source_names, valid_mask=None) -> dict[str, np.ndarray]:
    probs = np.asarray(semantic_probs, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError("semantic_probs must have shape [N, C]")
    source_names = tuple(str(name) for name in source_names)
    if probs.shape[1] != len(source_names):
        raise ValueError("semantic_probs channel count must match source_names")
    missing = [name for name in PART_NAMES if name not in source_names]
    if missing:
        raise ValueError(f"semantic_probs missing required part names: {missing}")

    remap = [source_names.index(name) for name in PART_NAMES]
    probs = probs[:, remap].astype(np.float32, copy=False)
    row_sum = probs.sum(axis=1, keepdims=True)
    valid = np.isfinite(probs).all(axis=1) & (row_sum.reshape(-1) > 0.0)
    if valid_mask is not None:
        semantic_valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if semantic_valid.shape[0] != probs.shape[0]:
            raise ValueError("valid_mask shape must match semantic_probs point count")
        valid &= semantic_valid
    normalized = np.zeros_like(probs, dtype=np.float32)
    normalized[valid] = probs[valid] / row_sum[valid].clip(min=1.0e-8)

    point_count = normalized.shape[0]
    part_label = np.full((point_count,), -1, dtype=np.int16)
    confidence = np.zeros((point_count,), dtype=np.float32)
    if np.any(valid):
        part_label[valid] = np.argmax(normalized[valid], axis=1).astype(np.int16)
        confidence[valid] = np.max(normalized[valid], axis=1).astype(np.float32)

    return {
        "part_label": part_label,
        "confidence": confidence,
        "vote_count": np.zeros((point_count,), dtype=np.int16),
        "per_part_votes": np.zeros((point_count, len(PART_NAMES)), dtype=np.int16),
        "visible_vote_count": np.zeros((point_count,), dtype=np.int16),
        "conflict_count": np.zeros((point_count,), dtype=np.int16),
        "semantic_probs": normalized,
        "source_type": "trained_semantic_asset_probs",
    }


def compute_semantic_margin(semantic_probs) -> np.ndarray:
    probs = np.asarray(semantic_probs, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError("semantic_probs must have shape [N, C]")
    if probs.shape[1] == 0:
        raise ValueError("semantic_probs must have at least one channel")
    if probs.shape[1] == 1:
        return probs[:, 0].astype(np.float32, copy=False)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    valid = np.isfinite(margin)
    out = np.zeros((probs.shape[0],), dtype=np.float32)
    out[valid] = margin[valid]
    return out


def compute_semantic_reliable_mask(
    *,
    part_label,
    confidence,
    semantic_margin,
    opacity=None,
    min_confidence: float = 0.65,
    min_margin: float = 0.20,
    min_opacity: float = 0.0,
) -> np.ndarray:
    labels = np.asarray(part_label, dtype=np.int16).reshape(-1)
    conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
    margin = np.asarray(semantic_margin, dtype=np.float32).reshape(-1)
    if conf.shape[0] != labels.shape[0]:
        raise ValueError("confidence point count must match part_label")
    if margin.shape[0] != labels.shape[0]:
        raise ValueError("semantic_margin point count must match part_label")

    reliable = labels >= 0
    reliable &= np.isfinite(conf) & (conf >= float(min_confidence))
    reliable &= np.isfinite(margin) & (margin >= float(min_margin))
    if opacity is not None and float(min_opacity) > 0.0:
        opacity_arr = np.asarray(opacity, dtype=np.float32).reshape(-1)
        if opacity_arr.shape[0] != labels.shape[0]:
            raise ValueError("opacity point count must match part_label")
        reliable &= np.isfinite(opacity_arr) & (opacity_arr >= float(min_opacity))
    return reliable.astype(np.uint8, copy=False)


def compute_soft_edit_weights(
    *,
    semantic_probs,
    confidence,
    semantic_margin,
    reliable_mask=None,
    reliable_floor: float = 0.0,
    confidence_power: float = 1.0,
    margin_power: float = 1.0,
) -> np.ndarray:
    probs = np.asarray(semantic_probs, dtype=np.float32)
    if probs.ndim != 2 or probs.shape[1] != len(PART_NAMES):
        raise ValueError(f"semantic_probs must have shape [N, {len(PART_NAMES)}]")
    point_count = probs.shape[0]
    conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
    margin = np.asarray(semantic_margin, dtype=np.float32).reshape(-1)
    if conf.shape[0] != point_count:
        raise ValueError("confidence point count must match semantic_probs")
    if margin.shape[0] != point_count:
        raise ValueError("semantic_margin point count must match semantic_probs")

    conf_factor = np.zeros((point_count,), dtype=np.float32)
    margin_factor = np.zeros((point_count,), dtype=np.float32)
    valid_conf = np.isfinite(conf)
    valid_margin = np.isfinite(margin)
    conf_factor[valid_conf] = np.clip(conf[valid_conf], 0.0, 1.0) ** float(confidence_power)
    margin_factor[valid_margin] = np.clip(margin[valid_margin], 0.0, 1.0) ** float(margin_power)
    reliability_factor = np.ones((point_count,), dtype=np.float32)
    if reliable_mask is not None:
        reliable = np.asarray(reliable_mask, dtype=np.uint8).reshape(-1) > 0
        if reliable.shape[0] != point_count:
            raise ValueError("reliable_mask point count must match semantic_probs")
        reliability_factor[~reliable] = float(reliable_floor)

    weights = probs * conf_factor[:, None] * margin_factor[:, None] * reliability_factor[:, None]
    weights[~np.isfinite(weights)] = 0.0
    return np.clip(weights, 0.0, 1.0).astype(np.float32, copy=False)


def compute_evidence_calibrated_soft_edit_weights(
    *,
    soft_edit_weights,
    footprint_target_ratio,
    footprint_outer_ratio,
    view_support_count,
    conflict_ratio=None,
    center_outer_ratio=None,
    center_valid_count=None,
    parts: tuple[str, ...] | list[str] | None = None,
    min_support_views: int = 5,
    min_center_views: int = 0,
    target_retention_floor: float = 0.60,
    outer_penalty_power: float = 1.0,
    conflict_penalty_power: float = 1.0,
    center_penalty_power: float = 0.0,
    center_target_retention_floor: float = 0.75,
) -> tuple[np.ndarray, dict]:
    weights = np.asarray(soft_edit_weights, dtype=np.float32)
    target = np.asarray(footprint_target_ratio, dtype=np.float32)
    outer = np.asarray(footprint_outer_ratio, dtype=np.float32)
    support = np.asarray(view_support_count)
    if weights.ndim != 2 or weights.shape[1] != len(PART_NAMES):
        raise ValueError(f"soft_edit_weights must have shape [N, {len(PART_NAMES)}]")
    if target.shape != weights.shape:
        raise ValueError("footprint_target_ratio shape must match soft_edit_weights")
    if outer.shape != weights.shape:
        raise ValueError("footprint_outer_ratio shape must match soft_edit_weights")
    if support.shape != weights.shape:
        raise ValueError("view_support_count shape must match soft_edit_weights")
    if conflict_ratio is None:
        conflict = np.zeros_like(weights, dtype=np.float32)
    else:
        conflict = np.asarray(conflict_ratio, dtype=np.float32)
        if conflict.shape != weights.shape:
            raise ValueError("conflict_ratio shape must match soft_edit_weights")
    if center_outer_ratio is None:
        center_outer = np.zeros_like(weights, dtype=np.float32)
    else:
        center_outer = np.asarray(center_outer_ratio, dtype=np.float32)
        if center_outer.shape != weights.shape:
            raise ValueError("center_outer_ratio shape must match soft_edit_weights")
    if center_valid_count is None:
        center_support = np.zeros_like(weights, dtype=np.int16)
    else:
        center_support = np.asarray(center_valid_count)
        if center_support.shape != weights.shape:
            raise ValueError("center_valid_count shape must match soft_edit_weights")

    selected_parts = list(PART_NAMES) if parts is None else [str(part) for part in parts]
    unknown = [part for part in selected_parts if part not in PART_NAMES]
    if unknown:
        raise ValueError(f"unknown part(s): {unknown}")

    updated = weights.copy()
    part_stats = {}
    enough_support = support.astype(np.int32) >= int(min_support_views)
    target_floor = float(target_retention_floor)
    outer_power = float(outer_penalty_power)
    conflict_power = float(conflict_penalty_power)
    center_power = float(center_penalty_power)
    center_target_floor = float(center_target_retention_floor)
    for part in selected_parts:
        idx = PART_NAMES.index(part)
        mask = enough_support[:, idx]
        target_factor = np.clip(target[:, idx], target_floor, 1.0)
        outer_penalty = np.clip(1.0 - outer[:, idx], 0.0, 1.0) ** outer_power
        conflict_penalty = np.clip(1.0 - conflict[:, idx], 0.0, 1.0) ** conflict_power
        raw_penalty = outer_penalty * conflict_penalty
        target_protected = target[:, idx] >= target_floor
        factor = np.where(target_protected, np.maximum(target_factor, raw_penalty), raw_penalty)
        center_mask = np.zeros_like(mask, dtype=bool)
        if center_power > 0.0 and int(min_center_views) > 0:
            center_mask = center_support[:, idx].astype(np.int32) >= int(min_center_views)
            center_penalty = np.clip(1.0 - center_outer[:, idx], 0.0, 1.0) ** center_power
            center_combined = factor * center_penalty
            center_protected = target[:, idx] >= center_target_floor
            factor = np.where(center_mask, center_combined, factor)
            factor = np.where(center_mask & center_protected, np.maximum(factor, center_target_floor), factor)
        before = updated[:, idx].copy()
        updated[mask, idx] = before[mask] * factor[mask]
        changed = np.abs(updated[:, idx] - before) > 1.0e-8
        part_stats[part] = {
            "calibrated_count": int(np.sum(mask)),
            "changed_count": int(np.sum(changed)),
            "center_penalized_count": int(np.sum(mask & center_mask & (factor < 1.0))),
            "old_weight_sum": float(np.sum(before)),
            "new_weight_sum": float(np.sum(updated[:, idx])),
            "removed_weight_sum": float(np.sum(before - updated[:, idx])),
            "min_support_views": int(min_support_views),
            "min_center_views": int(min_center_views),
            "target_retention_floor": target_floor,
            "center_penalty_power": center_power,
            "center_target_retention_floor": center_target_floor,
        }

    summary = {
        "mode": "evidence_calibrated_soft_edit_weights",
        "parts": part_stats,
        "total_removed_weight_sum": float(np.sum(weights - updated)),
    }
    return np.clip(updated, 0.0, 1.0).astype(np.float32, copy=False), summary


def compute_footprint_distilled_soft_edit_weights(
    *,
    soft_edit_weights,
    semantic_probs,
    footprint_target_ratio,
    footprint_outer_ratio,
    view_support_count,
    conflict_ratio=None,
    parts: tuple[str, ...] | list[str] | None = None,
    min_support_views: int = 5,
    target_retention_floor: float = 0.70,
    boundary_target_threshold: float = 0.35,
    boundary_retention_floor: float = 0.45,
    outer_penalty_power: float = 1.0,
    conflict_penalty_power: float = 1.0,
) -> tuple[np.ndarray, dict]:
    weights = np.asarray(soft_edit_weights, dtype=np.float32)
    probs = np.asarray(semantic_probs, dtype=np.float32)
    target = np.asarray(footprint_target_ratio, dtype=np.float32)
    outer = np.asarray(footprint_outer_ratio, dtype=np.float32)
    support = np.asarray(view_support_count)
    if weights.ndim != 2 or weights.shape[1] != len(PART_NAMES):
        raise ValueError(f"soft_edit_weights must have shape [N, {len(PART_NAMES)}]")
    if probs.shape != weights.shape:
        raise ValueError("semantic_probs shape must match soft_edit_weights")
    if target.shape != weights.shape:
        raise ValueError("footprint_target_ratio shape must match soft_edit_weights")
    if outer.shape != weights.shape:
        raise ValueError("footprint_outer_ratio shape must match soft_edit_weights")
    if support.shape != weights.shape:
        raise ValueError("view_support_count shape must match soft_edit_weights")
    if conflict_ratio is None:
        conflict = np.zeros_like(weights, dtype=np.float32)
    else:
        conflict = np.asarray(conflict_ratio, dtype=np.float32)
        if conflict.shape != weights.shape:
            raise ValueError("conflict_ratio shape must match soft_edit_weights")

    selected_parts = list(PART_NAMES) if parts is None else [str(part) for part in parts]
    unknown = [part for part in selected_parts if part not in PART_NAMES]
    if unknown:
        raise ValueError(f"unknown part(s): {unknown}")

    updated = weights.copy()
    part_stats = {}
    enough_support = support.astype(np.int32) >= int(min_support_views)
    target_floor = float(np.clip(target_retention_floor, 0.0, 1.0))
    boundary_threshold = float(np.clip(boundary_target_threshold, 0.0, 1.0))
    boundary_floor = float(np.clip(boundary_retention_floor, 0.0, 1.0))
    outer_power = float(outer_penalty_power)
    conflict_power = float(conflict_penalty_power)
    for part in selected_parts:
        idx = PART_NAMES.index(part)
        mask = enough_support[:, idx]
        semantic_support = np.clip(probs[:, idx], 0.0, 1.0)
        target_support = np.clip(target[:, idx], 0.0, 1.0)
        outer_penalty = np.clip(1.0 - outer[:, idx], 0.0, 1.0) ** outer_power
        conflict_penalty = np.clip(1.0 - conflict[:, idx], 0.0, 1.0) ** conflict_power
        raw_factor = np.clip(outer_penalty * conflict_penalty, 0.0, 1.0)
        target_protected = target_support >= target_floor
        boundary_protected = (
            (target_support >= boundary_threshold)
            & ~target_protected
            & (semantic_support >= boundary_threshold)
        )
        factor = raw_factor.copy()
        factor = np.where(target_protected, np.maximum(factor, target_floor), factor)
        factor = np.where(boundary_protected, np.maximum(factor, boundary_floor), factor)
        before = updated[:, idx].copy()
        updated[mask, idx] = before[mask] * factor[mask]
        changed = np.abs(updated[:, idx] - before) > 1.0e-8
        part_stats[part] = {
            "calibrated_count": int(np.sum(mask)),
            "changed_count": int(np.sum(changed)),
            "target_protected_count": int(np.sum(mask & target_protected)),
            "boundary_protected_count": int(np.sum(mask & boundary_protected)),
            "old_weight_sum": float(np.sum(before)),
            "new_weight_sum": float(np.sum(updated[:, idx])),
            "removed_weight_sum": float(np.sum(before - updated[:, idx])),
            "min_support_views": int(min_support_views),
            "target_retention_floor": target_floor,
            "boundary_target_threshold": boundary_threshold,
            "boundary_retention_floor": boundary_floor,
            "outer_penalty_power": outer_power,
            "conflict_penalty_power": conflict_power,
        }

    summary = {
        "mode": "footprint_distilled_soft_edit_weights",
        "parts": part_stats,
        "total_removed_weight_sum": float(np.sum(weights - updated)),
    }
    return np.clip(updated, 0.0, 1.0).astype(np.float32, copy=False), summary


def compute_footprint_boundary_confidence_weights(
    *,
    soft_edit_weights,
    semantic_probs,
    footprint_target_ratio,
    footprint_outer_ratio,
    adjacent_part_ratio,
    view_support_count,
    conflict_ratio=None,
    boundary_pairs: tuple[tuple[str, str], ...] | list[tuple[str, str]] = (),
    parts: tuple[str, ...] | list[str] | None = None,
    min_support_views: int = 5,
    target_retention_floor: float = 0.70,
    boundary_target_threshold: float = 0.35,
    boundary_semantic_threshold: float = 0.85,
    boundary_adjacent_threshold: float = 0.35,
    boundary_conflict_max: float = 0.30,
    boundary_retention_floor: float = 0.80,
    leak_target_threshold: float = 0.20,
    leak_semantic_threshold: float = 0.50,
    outer_penalty_power: float = 1.0,
    conflict_penalty_power: float = 1.0,
    adjacent_penalty_power: float = 1.0,
) -> tuple[np.ndarray, dict]:
    weights = np.asarray(soft_edit_weights, dtype=np.float32)
    probs = np.asarray(semantic_probs, dtype=np.float32)
    target = np.asarray(footprint_target_ratio, dtype=np.float32)
    outer = np.asarray(footprint_outer_ratio, dtype=np.float32)
    adjacent = np.asarray(adjacent_part_ratio, dtype=np.float32)
    support = np.asarray(view_support_count)
    if weights.ndim != 2 or weights.shape[1] != len(PART_NAMES):
        raise ValueError(f"soft_edit_weights must have shape [N, {len(PART_NAMES)}]")
    if probs.shape != weights.shape:
        raise ValueError("semantic_probs shape must match soft_edit_weights")
    if target.shape != weights.shape:
        raise ValueError("footprint_target_ratio shape must match soft_edit_weights")
    if outer.shape != weights.shape:
        raise ValueError("footprint_outer_ratio shape must match soft_edit_weights")
    if support.shape != weights.shape:
        raise ValueError("view_support_count shape must match soft_edit_weights")
    expected_adjacent_shape = (weights.shape[0], len(PART_NAMES), len(PART_NAMES))
    if adjacent.shape != expected_adjacent_shape:
        raise ValueError(f"adjacent_part_ratio must have shape {expected_adjacent_shape}")
    if conflict_ratio is None:
        conflict = np.zeros_like(weights, dtype=np.float32)
    else:
        conflict = np.asarray(conflict_ratio, dtype=np.float32)
        if conflict.shape != weights.shape:
            raise ValueError("conflict_ratio shape must match soft_edit_weights")

    selected_parts = list(PART_NAMES) if parts is None else [str(part) for part in parts]
    unknown = [part for part in selected_parts if part not in PART_NAMES]
    if unknown:
        raise ValueError(f"unknown part(s): {unknown}")

    parsed_pairs: list[tuple[str, str]] = []
    for pair in boundary_pairs or ():
        if len(pair) != 2:
            raise ValueError(f"boundary pair must contain two part names: {pair}")
        part, adjacent_part = str(pair[0]), str(pair[1])
        if part not in PART_NAMES or adjacent_part not in PART_NAMES:
            raise ValueError(f"unknown boundary pair: {pair}")
        parsed_pairs.append((part, adjacent_part))

    updated = weights.copy()
    part_stats = {}
    min_views = int(min_support_views)
    target_floor = float(np.clip(target_retention_floor, 0.0, 1.0))
    boundary_target_cutoff = float(np.clip(boundary_target_threshold, 0.0, 1.0))
    boundary_semantic_cutoff = float(np.clip(boundary_semantic_threshold, 0.0, 1.0))
    boundary_adjacent_cutoff = float(np.clip(boundary_adjacent_threshold, 0.0, 1.0))
    boundary_conflict_cutoff = float(np.clip(boundary_conflict_max, 0.0, 1.0))
    boundary_floor = float(np.clip(boundary_retention_floor, 0.0, 1.0))
    leak_target_cutoff = float(np.clip(leak_target_threshold, 0.0, 1.0))
    leak_semantic_cutoff = float(np.clip(leak_semantic_threshold, 0.0, 1.0))
    outer_power = float(outer_penalty_power)
    conflict_power = float(conflict_penalty_power)
    adjacent_power = float(adjacent_penalty_power)
    enough_support = support.astype(np.int32) >= min_views

    for part in selected_parts:
        part_idx = PART_NAMES.index(part)
        part_pairs = [(p, adj) for p, adj in parsed_pairs if p == part]
        pair_adjacent = np.zeros((weights.shape[0],), dtype=np.float32)
        for _part, adjacent_part in part_pairs:
            adjacent_idx = PART_NAMES.index(adjacent_part)
            pair_adjacent = np.maximum(pair_adjacent, np.clip(adjacent[:, part_idx, adjacent_idx], 0.0, 1.0))

        mask = enough_support[:, part_idx]
        semantic_support = np.clip(probs[:, part_idx], 0.0, 1.0)
        target_support = np.clip(target[:, part_idx], 0.0, 1.0)
        outer_support = np.clip(outer[:, part_idx], 0.0, 1.0)
        conflict_support = np.clip(conflict[:, part_idx], 0.0, 1.0)
        outer_penalty = np.clip(1.0 - outer_support, 0.0, 1.0) ** outer_power
        conflict_penalty = np.clip(1.0 - conflict_support, 0.0, 1.0) ** conflict_power
        adjacent_penalty = np.clip(1.0 - pair_adjacent, 0.0, 1.0) ** adjacent_power
        raw_factor = np.clip(outer_penalty * conflict_penalty * adjacent_penalty, 0.0, 1.0)

        strong_target = target_support >= target_floor
        boundary_confident = (
            (target_support >= boundary_target_cutoff)
            & (semantic_support >= boundary_semantic_cutoff)
            & (pair_adjacent >= boundary_adjacent_cutoff)
            & (conflict_support <= boundary_conflict_cutoff)
        )
        leak_candidate = (
            (target_support < leak_target_cutoff)
            & (semantic_support < leak_semantic_cutoff)
            & (pair_adjacent >= boundary_adjacent_cutoff)
        )
        factor = raw_factor.copy()
        factor = np.where(strong_target, np.maximum(factor, target_floor), factor)
        factor = np.where(boundary_confident, np.maximum(factor, boundary_floor), factor)
        factor = np.where(leak_candidate, np.minimum(factor, raw_factor), factor)

        before = updated[:, part_idx].copy()
        updated[mask, part_idx] = before[mask] * factor[mask]
        changed = np.abs(updated[:, part_idx] - before) > 1.0e-8
        part_stats[part] = {
            "calibrated_count": int(np.sum(mask)),
            "changed_count": int(np.sum(changed)),
            "boundary_pair_count": int(len(part_pairs)),
            "strong_target_count": int(np.sum(mask & strong_target)),
            "boundary_protected_count": int(np.sum(mask & boundary_confident & ~strong_target)),
            "leak_suppressed_count": int(np.sum(mask & leak_candidate)),
            "old_weight_sum": float(np.sum(before)),
            "new_weight_sum": float(np.sum(updated[:, part_idx])),
            "removed_weight_sum": float(np.sum(before - updated[:, part_idx])),
            "min_support_views": min_views,
            "target_retention_floor": target_floor,
            "boundary_target_threshold": boundary_target_cutoff,
            "boundary_semantic_threshold": boundary_semantic_cutoff,
            "boundary_adjacent_threshold": boundary_adjacent_cutoff,
            "boundary_conflict_max": boundary_conflict_cutoff,
            "boundary_retention_floor": boundary_floor,
            "leak_target_threshold": leak_target_cutoff,
            "leak_semantic_threshold": leak_semantic_cutoff,
        }

    summary = {
        "mode": "footprint_boundary_confidence_weights",
        "boundary_pairs": [f"{part}:{adjacent_part}" for part, adjacent_part in parsed_pairs],
        "parts": part_stats,
        "total_removed_weight_sum": float(np.sum(weights - updated)),
    }
    return np.clip(updated, 0.0, 1.0).astype(np.float32, copy=False), summary


def compute_support_aware_edit_weights(
    *,
    soft_edit_weights,
    footprint_target_ratio,
    footprint_outer_ratio,
    adjacent_part_ratio,
    view_support_count,
    support_pairs,
    parts: tuple[str, ...] | list[str] | None = None,
    min_support_views: int = 5,
    target_retention_floor: float = 0.70,
    support_threshold: float = 0.35,
    outer_penalty_power: float = 1.0,
    support_penalty_power: float = 1.0,
    boundary_role_ratio=None,
    boundary_role_names=(),
    support_roles=(),
    role_threshold: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, dict]:
    weights = np.asarray(soft_edit_weights, dtype=np.float32)
    target = np.asarray(footprint_target_ratio, dtype=np.float32)
    outer = np.asarray(footprint_outer_ratio, dtype=np.float32)
    adjacent = np.asarray(adjacent_part_ratio, dtype=np.float32)
    support = np.asarray(view_support_count)
    if weights.ndim != 2 or weights.shape[1] != len(PART_NAMES):
        raise ValueError(f"soft_edit_weights must have shape [N, {len(PART_NAMES)}]")
    if target.shape != weights.shape:
        raise ValueError("footprint_target_ratio shape must match soft_edit_weights")
    if outer.shape != weights.shape:
        raise ValueError("footprint_outer_ratio shape must match soft_edit_weights")
    if support.shape != weights.shape:
        raise ValueError("view_support_count shape must match soft_edit_weights")
    expected_adjacent_shape = (weights.shape[0], len(PART_NAMES), len(PART_NAMES))
    if adjacent.shape != expected_adjacent_shape:
        raise ValueError(f"adjacent_part_ratio must have shape {expected_adjacent_shape}")
    role_ratio = None
    role_name_to_index: dict[str, int] = {}
    if boundary_role_ratio is not None:
        role_ratio = np.asarray(boundary_role_ratio, dtype=np.float32)
        if role_ratio.ndim != 3 or role_ratio.shape[0] != weights.shape[0] or role_ratio.shape[1] != len(PART_NAMES):
            raise ValueError("boundary_role_ratio must have shape [N, C, R]")
        role_names_list = [str(name) for name in np.asarray(boundary_role_names).tolist()]
        role_name_to_index = {name: idx for idx, name in enumerate(role_names_list)}
        if role_ratio.shape[2] != len(role_name_to_index):
            raise ValueError("boundary_role_names length must match boundary_role_ratio role dimension")

    selected_parts = list(PART_NAMES) if parts is None else [str(part) for part in parts]
    unknown = [part for part in selected_parts if part not in PART_NAMES]
    if unknown:
        raise ValueError(f"unknown part(s): {unknown}")

    parsed_pairs: list[tuple[str, str]] = []
    for pair in support_pairs or ():
        if len(pair) != 2:
            raise ValueError(f"support pair must contain two part names: {pair}")
        part, adjacent_part = str(pair[0]), str(pair[1])
        if part not in PART_NAMES or adjacent_part not in PART_NAMES:
            raise ValueError(f"unknown support pair: {pair}")
        parsed_pairs.append((part, adjacent_part))
    roles_by_pair: dict[tuple[str, str], list[str]] = {}
    for role in support_roles or ():
        if len(role) != 3:
            raise ValueError(f"support role must contain part, adjacent, role name: {role}")
        part, adjacent_part, role_name = str(role[0]), str(role[1]), str(role[2])
        if part not in PART_NAMES or adjacent_part not in PART_NAMES:
            raise ValueError(f"unknown support role pair: {role}")
        if role_name_to_index and role_name not in role_name_to_index:
            raise ValueError(f"unknown boundary role name: {role_name}")
        roles_by_pair.setdefault((part, adjacent_part), []).append(role_name)

    target_weights = weights.copy()
    support_weights = np.zeros_like(weights, dtype=np.float32)
    part_stats = {}
    min_views = int(min_support_views)
    target_floor = float(np.clip(target_retention_floor, 0.0, 1.0))
    support_cutoff = float(np.clip(support_threshold, 0.0, 1.0))
    role_cutoff = float(np.clip(role_threshold, 0.0, 1.0))
    outer_power = float(outer_penalty_power)
    support_power = float(support_penalty_power)
    for part in selected_parts:
        part_idx = PART_NAMES.index(part)
        part_pairs = [(p, adj) for p, adj in parsed_pairs if p == part]
        enough_support = support[:, part_idx].astype(np.int32) >= min_views
        original = weights[:, part_idx]
        factor = np.ones((weights.shape[0],), dtype=np.float32)
        support_mask_total = np.zeros((weights.shape[0],), dtype=bool)
        role_gated_support_total = np.zeros((weights.shape[0],), dtype=bool)
        for _part, adjacent_part in part_pairs:
            adjacent_idx = PART_NAMES.index(adjacent_part)
            adjacent_ratio = np.clip(adjacent[:, part_idx, adjacent_idx], 0.0, 1.0)
            support_candidate = adjacent_ratio >= support_cutoff
            pair_roles = roles_by_pair.get((part, adjacent_part), [])
            role_candidate = np.ones((weights.shape[0],), dtype=bool)
            if pair_roles and role_ratio is not None and role_name_to_index:
                role_candidate = np.zeros((weights.shape[0],), dtype=bool)
                for role_name in pair_roles:
                    role_idx = role_name_to_index[role_name]
                    role_candidate |= np.clip(role_ratio[:, part_idx, role_idx], 0.0, 1.0) >= role_cutoff
            low_target = np.clip(target[:, part_idx], 0.0, 1.0) < target_floor
            pair_support_mask = enough_support & support_candidate & role_candidate & low_target
            support_mask_total |= pair_support_mask
            role_gated_support_total |= pair_support_mask & bool(pair_roles)
            support_values = original * adjacent_ratio
            support_weights[pair_support_mask, part_idx] = np.maximum(
                support_weights[pair_support_mask, part_idx],
                support_values[pair_support_mask],
            )
            pair_factor = (
                np.clip(1.0 - outer[:, part_idx], 0.0, 1.0) ** outer_power
            ) * (
                np.clip(1.0 - adjacent_ratio, 0.0, 1.0) ** support_power
            )
            factor = np.where(pair_support_mask, np.minimum(factor, pair_factor), factor)

        strong_target = enough_support & (np.clip(target[:, part_idx], 0.0, 1.0) >= target_floor)
        factor = np.where(strong_target, np.maximum(factor, target_floor), factor)
        updated = np.minimum(original, original * np.clip(factor, 0.0, 1.0))
        target_weights[:, part_idx] = np.where(enough_support, updated, original)
        changed = np.abs(target_weights[:, part_idx] - original) > 1.0e-8
        part_stats[part] = {
            "calibrated_count": int(np.sum(enough_support)),
            "changed_count": int(np.sum(changed)),
            "support_point_count": int(np.sum(support_mask_total)),
            "role_gated_support_count": int(np.sum(role_gated_support_total)),
            "support_pair_count": int(len(part_pairs)),
            "old_weight_sum": float(np.sum(original)),
            "target_weight_sum": float(np.sum(target_weights[:, part_idx])),
            "support_weight_sum": float(np.sum(support_weights[:, part_idx])),
            "removed_target_weight_sum": float(np.sum(original - target_weights[:, part_idx])),
            "min_support_views": min_views,
            "target_retention_floor": target_floor,
            "support_threshold": support_cutoff,
            "outer_penalty_power": outer_power,
            "support_penalty_power": support_power,
            "role_threshold": role_cutoff,
        }

    summary = {
        "mode": "support_aware_edit_weights",
        "support_pairs": [f"{part}:{adjacent}" for part, adjacent in parsed_pairs],
        "support_roles": [f"{part}:{adjacent}:{role}" for (part, adjacent), roles in roles_by_pair.items() for role in roles],
        "parts": part_stats,
        "total_removed_target_weight_sum": float(np.sum(weights - target_weights)),
        "total_support_weight_sum": float(np.sum(support_weights)),
    }
    return (
        np.clip(target_weights, 0.0, 1.0).astype(np.float32, copy=False),
        np.clip(support_weights, 0.0, 1.0).astype(np.float32, copy=False),
        summary,
    )


def apply_reliable_label_mask(
    bank: Mapping[str, np.ndarray],
    *,
    opacity=None,
    min_confidence: float = 0.65,
    min_margin: float = 0.20,
    min_opacity: float = 0.0,
) -> dict:
    if "semantic_probs" not in bank:
        raise ValueError("reliable label mask requires semantic_probs")
    labels = np.asarray(bank["part_label"], dtype=np.int16)
    confidence = np.asarray(bank["confidence"], dtype=np.float32)
    probs = np.asarray(bank["semantic_probs"], dtype=np.float32)
    if probs.ndim != 2 or probs.shape[1] != len(PART_NAMES):
        raise ValueError(f"semantic_probs must have shape [N, {len(PART_NAMES)}]")
    if labels.shape[0] != probs.shape[0] or confidence.shape[0] != probs.shape[0]:
        raise ValueError("bank arrays must have matching point counts")

    margin = compute_semantic_margin(probs)
    reliable = compute_semantic_reliable_mask(
        part_label=labels,
        confidence=confidence,
        semantic_margin=margin,
        opacity=opacity,
        min_confidence=min_confidence,
        min_margin=min_margin,
        min_opacity=min_opacity,
    )
    editable_label = labels.copy()
    editable_label[reliable == 0] = -1
    bank["semantic_margin"] = margin.astype(np.float32, copy=False)
    bank["reliable_mask"] = reliable.astype(np.uint8, copy=False)
    bank["editable_label"] = editable_label.astype(np.int16, copy=False)

    known = labels >= 0
    low_confidence = known & (confidence < float(min_confidence))
    low_margin = known & (margin < float(min_margin))
    low_opacity_count = 0
    if opacity is not None and float(min_opacity) > 0.0:
        opacity_arr = np.asarray(opacity, dtype=np.float32).reshape(-1)
        if opacity_arr.shape[0] != labels.shape[0]:
            raise ValueError("opacity point count must match part_label")
        low_opacity_count = int(np.sum(known & (opacity_arr < float(min_opacity))))

    return {
        "enabled": True,
        "min_confidence": float(min_confidence),
        "min_margin": float(min_margin),
        "min_opacity": float(min_opacity),
        "reliable_count": int(np.sum(reliable > 0)),
        "unreliable_count": int(labels.shape[0] - int(np.sum(reliable > 0))),
        "low_confidence_count": int(np.sum(low_confidence)),
        "low_margin_count": int(np.sum(low_margin)),
        "low_opacity_count": low_opacity_count,
    }


def apply_neighbor_reliable_fill(
    bank: Mapping[str, np.ndarray],
    *,
    xyz,
    k: int = 12,
    min_reliable_neighbors: int = 5,
    majority_ratio: float = 0.70,
    min_candidate_confidence: float = 0.50,
    confidence=None,
) -> dict:
    labels = np.asarray(bank["part_label"], dtype=np.int16)
    editable = np.asarray(bank["editable_label"], dtype=np.int16)
    reliable = np.asarray(bank["reliable_mask"], dtype=np.uint8).reshape(-1) > 0
    coords = np.asarray(xyz, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    point_count = labels.shape[0]
    if editable.shape[0] != point_count or reliable.shape[0] != point_count or coords.shape[0] != point_count:
        raise ValueError("bank arrays and xyz must have matching point counts")
    if confidence is None:
        conf = np.asarray(bank.get("confidence", np.ones((point_count,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    else:
        conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
    if conf.shape[0] != point_count:
        raise ValueError("confidence point count must match part_label")

    k = max(1, int(k))
    min_reliable_neighbors = max(1, int(min_reliable_neighbors))
    majority_ratio = float(majority_ratio)
    candidates = (editable < 0) & (labels >= 0) & (conf >= float(min_candidate_confidence))
    candidate_idx = np.nonzero(candidates)[0]
    fill_mask = np.zeros((point_count,), dtype=np.uint8)
    rejected_not_enough = 0
    rejected_majority = 0
    rejected_label_mismatch = 0
    filled_by_label = {name: 0 for name in PART_NAMES}

    reliable_idx = np.nonzero(reliable)[0]
    if candidate_idx.size and reliable_idx.size:
        reliable_coords = coords[reliable_idx]
        for idx in candidate_idx:
            diff = reliable_coords - coords[idx]
            dist2 = np.einsum("ij,ij->i", diff, diff)
            take = min(k, reliable_idx.shape[0])
            nearest_local = np.argpartition(dist2, take - 1)[:take]
            nearest_labels = labels[reliable_idx[nearest_local]]
            nearest_labels = nearest_labels[nearest_labels >= 0]
            if nearest_labels.shape[0] < min_reliable_neighbors:
                rejected_not_enough += 1
                continue
            counts = np.bincount(nearest_labels.astype(np.int64), minlength=len(PART_NAMES))
            majority_label = int(np.argmax(counts))
            majority_count = int(counts[majority_label])
            if majority_count / float(nearest_labels.shape[0]) < majority_ratio:
                rejected_majority += 1
                continue
            if majority_label != int(labels[idx]):
                rejected_label_mismatch += 1
                continue
            editable[idx] = labels[idx]
            fill_mask[idx] = 1
            filled_by_label[PART_NAMES[majority_label]] += 1

    bank["editable_label"] = editable.astype(np.int16, copy=False)
    bank["neighbor_fill_mask"] = fill_mask
    stats = {
        "enabled": True,
        "k": int(k),
        "min_reliable_neighbors": int(min_reliable_neighbors),
        "majority_ratio": float(majority_ratio),
        "min_candidate_confidence": float(min_candidate_confidence),
        "candidate_count": int(candidate_idx.size),
        "filled_count": int(fill_mask.sum()),
        "rejected_not_enough_neighbors_count": int(rejected_not_enough),
        "rejected_majority_count": int(rejected_majority),
        "rejected_label_mismatch_count": int(rejected_label_mismatch),
    }
    for name, count in filled_by_label.items():
        stats[f"filled_{name}_count"] = int(count)
    return stats


def apply_face_label_guard(
    bank: Mapping[str, np.ndarray],
    *,
    min_prob: float = 0.70,
    min_margin: float = 0.15,
    max_scale: float | None = None,
    scale_max=None,
    oversized_action: str = "second",
) -> dict:
    labels = np.asarray(bank["part_label"], dtype=np.int16)
    confidence = np.asarray(bank["confidence"], dtype=np.float32)
    if "semantic_probs" not in bank:
        raise ValueError("face label guard requires semantic_probs")
    probs = np.asarray(bank["semantic_probs"], dtype=np.float32)
    if probs.ndim != 2 or probs.shape[1] != len(PART_NAMES):
        raise ValueError(f"semantic_probs must have shape [N, {len(PART_NAMES)}]")
    if labels.shape[0] != probs.shape[0]:
        raise ValueError("part_label and semantic_probs point counts do not match")

    face_idx = PART_NAMES.index("face")
    face_initial = labels == face_idx
    if not np.any(face_initial):
        return {
            "enabled": True,
            "face_initial_count": 0,
            "face_final_count": 0,
            "low_prob_count": 0,
            "low_margin_count": 0,
            "oversized_count": 0,
            "reassigned_to_unknown_count": 0,
        }

    sorted_probs = np.sort(probs, axis=1)
    second_prob = sorted_probs[:, -2] if probs.shape[1] > 1 else np.zeros((probs.shape[0],), dtype=np.float32)
    face_prob = probs[:, face_idx]
    margin = face_prob - second_prob
    reject = face_initial & (face_prob < float(min_prob))
    low_margin = face_initial & (margin < float(min_margin))
    reject |= low_margin

    oversized = np.zeros_like(face_initial, dtype=bool)
    if max_scale is not None and float(max_scale) > 0.0:
        if scale_max is None:
            raise ValueError("scale_max is required when max_scale is set")
        scale_arr = np.asarray(scale_max, dtype=np.float32).reshape(-1)
        if scale_arr.shape[0] != probs.shape[0]:
            raise ValueError("scale_max point count must match semantic_probs")
        oversized = face_initial & (scale_arr > float(max_scale))
        reject |= oversized

    reject_idx = np.nonzero(reject)[0]
    action = str(oversized_action or "second").lower()
    if reject_idx.size:
        labels_mut = labels
        confidence_mut = confidence
        probs_without_face = probs.copy()
        probs_without_face[:, face_idx] = -1.0
        fallback = np.argmax(probs_without_face[reject_idx], axis=1).astype(np.int16)
        fallback_conf = np.max(probs_without_face[reject_idx], axis=1).astype(np.float32)
        if action == "unknown":
            labels_mut[reject_idx] = -1
            confidence_mut[reject_idx] = 0.0
        elif action == "second":
            has_fallback = fallback_conf > 0.0
            labels_mut[reject_idx[has_fallback]] = fallback[has_fallback]
            confidence_mut[reject_idx[has_fallback]] = fallback_conf[has_fallback]
            if np.any(~has_fallback):
                labels_mut[reject_idx[~has_fallback]] = -1
                confidence_mut[reject_idx[~has_fallback]] = 0.0
        else:
            raise ValueError(f"unsupported oversized_action: {oversized_action}")

    stats = {
        "enabled": True,
        "min_prob": float(min_prob),
        "min_margin": float(min_margin),
        "max_scale": None if max_scale is None else float(max_scale),
        "oversized_action": str(oversized_action),
        "face_initial_count": int(face_initial.sum()),
        "face_final_count": int(np.sum(labels == face_idx)),
        "low_prob_count": int(np.sum(face_initial & (face_prob < float(min_prob)))),
        "low_margin_count": int(np.sum(low_margin)),
        "oversized_count": int(np.sum(oversized)),
        "reassigned_to_unknown_count": int(np.sum(reject & (labels < 0))),
    }
    for idx, name in enumerate(PART_NAMES):
        if idx == face_idx:
            continue
        stats[f"reassigned_to_{name}_count"] = int(np.sum(reject & (labels == idx)))
    return stats


def apply_lower_label_guard(
    bank: Mapping[str, np.ndarray],
    *,
    xyz,
    high_y_threshold: float = 0.30,
    max_abs_x: float | None = 0.35,
    max_abs_z: float | None = 0.18,
    target_second_name: str = "upper",
) -> dict:
    labels = np.asarray(bank["part_label"], dtype=np.int16)
    confidence = np.asarray(bank["confidence"], dtype=np.float32)
    if "semantic_probs" not in bank:
        raise ValueError("lower label guard requires semantic_probs")
    probs = np.asarray(bank["semantic_probs"], dtype=np.float32)
    coords = np.asarray(xyz, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if probs.shape[0] != coords.shape[0] or labels.shape[0] != coords.shape[0]:
        raise ValueError("xyz point count must match label bank")
    if target_second_name not in PART_NAMES:
        raise ValueError(f"unknown target_second_name: {target_second_name}")

    lower_idx = PART_NAMES.index("lower")
    target_idx = PART_NAMES.index(target_second_name)
    lower_initial = labels == lower_idx
    probs_without_lower = probs.copy()
    probs_without_lower[:, lower_idx] = -1.0
    fallback = np.argmax(probs_without_lower, axis=1).astype(np.int16)
    fallback_conf = np.max(probs_without_lower, axis=1).astype(np.float32)

    spatial = coords[:, 1] > float(high_y_threshold)
    if max_abs_x is not None and float(max_abs_x) > 0.0:
        spatial &= np.abs(coords[:, 0]) <= float(max_abs_x)
    if max_abs_z is not None and float(max_abs_z) > 0.0:
        spatial &= np.abs(coords[:, 2]) <= float(max_abs_z)
    reject = lower_initial & spatial & (fallback == target_idx) & (fallback_conf > 0.0)
    labels[reject] = target_idx
    confidence[reject] = fallback_conf[reject]

    return {
        "enabled": True,
        "high_y_threshold": float(high_y_threshold),
        "max_abs_x": None if max_abs_x is None else float(max_abs_x),
        "max_abs_z": None if max_abs_z is None else float(max_abs_z),
        "target_second_name": str(target_second_name),
        "lower_initial_count": int(lower_initial.sum()),
        "lower_final_count": int(np.sum(labels == lower_idx)),
        f"reassigned_to_{target_second_name}_count": int(reject.sum()),
    }


def _check_dtype(arrays: Mapping[str, np.ndarray], key: str, dtype) -> None:
    if _as_array(arrays[key]).dtype != np.dtype(dtype):
        raise ValueError(f"{key} dtype must be {np.dtype(dtype)}, got {_as_array(arrays[key]).dtype}")


def _scalar_string(arrays: Mapping[str, np.ndarray], key: str) -> str:
    value = _as_array(arrays[key])
    if value.shape != ():
        raise ValueError(f"{key} must be a scalar string")
    if value.dtype.kind not in ("U", "S"):
        raise ValueError(f"{key} must be a scalar string")
    return str(value)


def _validate_sha256(value: str, *, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 fingerprint")


def part_label_bank_fingerprint(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(str(name) for name in arrays if str(name) != "output_bank_fingerprint"):
        array = _as_array(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        if array.dtype.kind in ("U", "S"):
            digest.update(
                json.dumps(
                    array.tolist(),
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        else:
            digest.update(np.ascontiguousarray(array).tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_a7_bank_arrays(arrays: Mapping[str, np.ndarray], *, point_count: int) -> None:
    required = (
        "method_id",
        "base_method",
        "base_bank_sha256",
        *A7_PROVENANCE_FIELDS,
        *A7_TEMPORAL_COUNT_FIELDS,
        *A7_TEMPORAL_FLOAT_FIELDS,
        "temporal_reliability",
        "soft_edit_weights",
        "output_bank_fingerprint",
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"missing A7 part label bank keys: {missing}")
    if _scalar_string(arrays, "method_id") != "A7":
        raise ValueError("method_id must be A7")
    if _scalar_string(arrays, "base_method") != "A5":
        raise ValueError("A7 base_method must be A5")

    fingerprint_fields = (
        "base_bank_sha256",
        *A7_PROVENANCE_FIELDS,
        "output_bank_fingerprint",
    )
    for field in fingerprint_fields:
        _validate_sha256(_scalar_string(arrays, field), field=field)

    shape = (point_count, len(PART_NAMES))
    for key in A7_TEMPORAL_COUNT_FIELDS:
        _check_dtype(arrays, key, np.int32)
        value = _as_array(arrays[key])
        if value.shape != shape:
            raise ValueError(f"{key} must have shape {shape}")
        if np.any(value < 0):
            raise ValueError(f"{key} must be non-negative")
    for key in (*A7_TEMPORAL_FLOAT_FIELDS, "temporal_reliability"):
        _check_dtype(arrays, key, np.float32)
        value = _as_array(arrays[key])
        if value.shape != shape:
            raise ValueError(f"{key} must have shape {shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{key} must be finite")
        if np.any((value < 0.0) | (value > 1.0)):
            raise ValueError(f"{key} must be in [0, 1]")

    weights = _as_array(arrays["soft_edit_weights"])
    if not np.all(np.isfinite(weights)):
        raise ValueError("soft_edit_weights must be finite")
    if np.any((weights < 0.0) | (weights > 1.0)):
        raise ValueError("soft_edit_weights must be in [0, 1]")

    expected = _scalar_string(arrays, "output_bank_fingerprint")
    actual = part_label_bank_fingerprint(arrays)
    if actual != expected:
        raise ValueError(
            "A7 output_bank_fingerprint does not match bank contents: "
            f"expected {expected}, got {actual}"
        )


def validate_part_label_bank_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    required = (
        "schema_version",
        "point_count",
        "part_names",
        "part_label",
        "confidence",
        "vote_count",
        "per_part_votes",
        "visible_vote_count",
        "conflict_count",
        "source_checkpoint",
        "source_asset_root",
        "source_iteration",
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"missing part label bank keys: {missing}")

    if int(_as_array(arrays["schema_version"])) != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {int(_as_array(arrays['schema_version']))}")
    point_count = int(_as_array(arrays["point_count"]))
    part_names = [str(x) for x in _as_array(arrays["part_names"]).tolist()]
    if part_names != list(PART_NAMES):
        raise ValueError(f"part_names mismatch: {part_names}")

    _check_dtype(arrays, "schema_version", np.int32)
    _check_dtype(arrays, "point_count", np.int64)
    _check_dtype(arrays, "part_label", np.int16)
    _check_dtype(arrays, "confidence", np.float32)
    _check_dtype(arrays, "vote_count", np.int16)
    _check_dtype(arrays, "per_part_votes", np.int16)
    _check_dtype(arrays, "visible_vote_count", np.int16)
    _check_dtype(arrays, "conflict_count", np.int16)
    _check_dtype(arrays, "source_iteration", np.int64)

    expected_1d = ("part_label", "confidence", "vote_count", "visible_vote_count", "conflict_count")
    for key in expected_1d:
        if _as_array(arrays[key]).shape != (point_count,):
            raise ValueError(f"{key} must have shape ({point_count},)")
    if _as_array(arrays["per_part_votes"]).shape != (point_count, len(PART_NAMES)):
        raise ValueError(f"per_part_votes must have shape ({point_count}, {len(PART_NAMES)})")
    if _as_array(arrays["part_names"]).shape != (len(PART_NAMES),):
        raise ValueError(f"part_names must have shape ({len(PART_NAMES)},)")
    if "semantic_probs" in arrays:
        _check_dtype(arrays, "semantic_probs", np.float32)
        if _as_array(arrays["semantic_probs"]).shape != (point_count, len(PART_NAMES)):
            raise ValueError(f"semantic_probs must have shape ({point_count}, {len(PART_NAMES)})")
    if "semantic_margin" in arrays:
        _check_dtype(arrays, "semantic_margin", np.float32)
        if _as_array(arrays["semantic_margin"]).shape != (point_count,):
            raise ValueError(f"semantic_margin must have shape ({point_count},)")
    if "reliable_mask" in arrays:
        _check_dtype(arrays, "reliable_mask", np.uint8)
        if _as_array(arrays["reliable_mask"]).shape != (point_count,):
            raise ValueError(f"reliable_mask must have shape ({point_count},)")
    if "editable_label" in arrays:
        _check_dtype(arrays, "editable_label", np.int16)
        if _as_array(arrays["editable_label"]).shape != (point_count,):
            raise ValueError(f"editable_label must have shape ({point_count},)")
    if "soft_edit_weights" in arrays:
        _check_dtype(arrays, "soft_edit_weights", np.float32)
        if _as_array(arrays["soft_edit_weights"]).shape != (point_count, len(PART_NAMES)):
            raise ValueError(f"soft_edit_weights must have shape ({point_count}, {len(PART_NAMES)})")
    for key in ("edit_target_weights", "edit_support_weights"):
        if key in arrays:
            _check_dtype(arrays, key, np.float32)
            if _as_array(arrays[key]).shape != (point_count, len(PART_NAMES)):
                raise ValueError(f"{key} must have shape ({point_count}, {len(PART_NAMES)})")
    if "neighbor_fill_mask" in arrays:
        _check_dtype(arrays, "neighbor_fill_mask", np.uint8)
        if _as_array(arrays["neighbor_fill_mask"]).shape != (point_count,):
            raise ValueError(f"neighbor_fill_mask must have shape ({point_count},)")
    if "method_id" in arrays or any(
        key in arrays for key in (*A7_TEMPORAL_COUNT_FIELDS, *A7_TEMPORAL_FLOAT_FIELDS)
    ):
        _validate_a7_bank_arrays(arrays, point_count=point_count)


def save_part_label_bank(
    path,
    *,
    part_label,
    confidence,
    vote_count,
    per_part_votes,
    visible_vote_count,
    conflict_count,
    source_checkpoint: str,
    source_asset_root: str,
    source_iteration: int,
    semantic_probs=None,
    semantic_margin=None,
    reliable_mask=None,
    editable_label=None,
    soft_edit_weights=None,
    edit_target_weights=None,
    edit_support_weights=None,
    support_pair_names=None,
    support_aware_summary=None,
    neighbor_fill_mask=None,
    source_type: str = "multiview_2d_mask_votes",
) -> None:
    path = Path(path)
    point_count = int(np.asarray(part_label).reshape(-1).shape[0])
    arrays = {
        "schema_version": np.array(SCHEMA_VERSION, dtype=np.int32),
        "point_count": np.array(point_count, dtype=np.int64),
        "part_names": np.asarray(PART_NAMES, dtype="U16"),
        "part_label": np.asarray(part_label, dtype=np.int16),
        "confidence": np.asarray(confidence, dtype=np.float32),
        "vote_count": np.asarray(vote_count, dtype=np.int16),
        "per_part_votes": np.asarray(per_part_votes, dtype=np.int16),
        "visible_vote_count": np.asarray(visible_vote_count, dtype=np.int16),
        "conflict_count": np.asarray(conflict_count, dtype=np.int16),
        "source_checkpoint": np.array(str(source_checkpoint)),
        "source_asset_root": np.array(str(source_asset_root)),
        "source_iteration": np.array(int(source_iteration), dtype=np.int64),
        "source_type": np.array(str(source_type)),
    }
    if semantic_probs is not None:
        arrays["semantic_probs"] = np.asarray(semantic_probs, dtype=np.float32)
    if semantic_margin is not None:
        arrays["semantic_margin"] = np.asarray(semantic_margin, dtype=np.float32)
    if reliable_mask is not None:
        arrays["reliable_mask"] = np.asarray(reliable_mask, dtype=np.uint8)
    if editable_label is not None:
        arrays["editable_label"] = np.asarray(editable_label, dtype=np.int16)
    if soft_edit_weights is not None:
        arrays["soft_edit_weights"] = np.asarray(soft_edit_weights, dtype=np.float32)
    if edit_target_weights is not None:
        arrays["edit_target_weights"] = np.asarray(edit_target_weights, dtype=np.float32)
    if edit_support_weights is not None:
        arrays["edit_support_weights"] = np.asarray(edit_support_weights, dtype=np.float32)
    if support_pair_names is not None:
        arrays["support_pair_names"] = np.asarray(list(support_pair_names), dtype="U32")
    if support_aware_summary is not None:
        arrays["support_aware_summary"] = np.array(str(support_aware_summary))
    if neighbor_fill_mask is not None:
        arrays["neighbor_fill_mask"] = np.asarray(neighbor_fill_mask, dtype=np.uint8)
    validate_part_label_bank_arrays(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_part_label_bank(path, validate: bool = True) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if validate:
        validate_part_label_bank_arrays(arrays)
    return arrays


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def save_a7_part_label_bank(
    path,
    *,
    base_bank_path,
    temporal_evidence: Mapping[str, np.ndarray],
    temporal_reliability,
    soft_edit_weights,
    provenance: Mapping[str, str],
) -> str:
    output_path = Path(path)
    base_path = Path(base_bank_path)
    if output_path.resolve() == base_path.resolve():
        raise ValueError("A7 output path must not overwrite the base A5 bank")
    base_bank = load_part_label_bank(base_path)
    if str(base_bank.get("method_id", "")) == "A7":
        raise ValueError("A7 base bank must be an A5 bank, not another A7 bank")

    missing_evidence = [
        key
        for key in (*A7_TEMPORAL_COUNT_FIELDS, *A7_TEMPORAL_FLOAT_FIELDS)
        if key not in temporal_evidence
    ]
    if missing_evidence:
        raise ValueError(f"missing A7 temporal evidence fields: {missing_evidence}")
    missing_provenance = [key for key in A7_PROVENANCE_FIELDS if key not in provenance]
    if missing_provenance:
        raise ValueError(f"missing A7 provenance fields: {missing_provenance}")

    arrays = {
        key: np.asarray(value).copy()
        for key, value in base_bank.items()
        if key != "output_bank_fingerprint"
    }
    arrays["method_id"] = np.array("A7")
    arrays["base_method"] = np.array("A5")
    arrays["base_bank_sha256"] = np.array(_file_sha256(base_path))
    for key in A7_PROVENANCE_FIELDS:
        arrays[key] = np.array(str(provenance[key]))
    for key in A7_TEMPORAL_COUNT_FIELDS:
        arrays[key] = np.asarray(temporal_evidence[key], dtype=np.int32)
    for key in A7_TEMPORAL_FLOAT_FIELDS:
        arrays[key] = np.asarray(temporal_evidence[key], dtype=np.float32)
    arrays["temporal_reliability"] = np.asarray(
        temporal_reliability, dtype=np.float32
    )
    arrays["soft_edit_weights"] = np.asarray(soft_edit_weights, dtype=np.float32)
    fingerprint = part_label_bank_fingerprint(arrays)
    arrays["output_bank_fingerprint"] = np.array(fingerprint)
    validate_part_label_bank_arrays(arrays)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    reloaded = load_part_label_bank(output_path)
    reloaded_fingerprint = part_label_bank_fingerprint(reloaded)
    if reloaded_fingerprint != fingerprint:
        raise ValueError(
            "A7 bank fingerprint changed after save/reload: "
            f"expected {fingerprint}, got {reloaded_fingerprint}"
        )
    return fingerprint


def summarize_part_label_bank(bank: Mapping[str, np.ndarray]) -> dict:
    labels = np.asarray(bank["part_label"], dtype=np.int16).reshape(-1)
    confidence = np.asarray(bank["confidence"], dtype=np.float32).reshape(-1)
    vote_count = np.asarray(bank["vote_count"], dtype=np.int16).reshape(-1)
    conflict_count = np.asarray(bank["conflict_count"], dtype=np.int16).reshape(-1)
    summary = {f"{name}_count": int(np.sum(labels == idx)) for idx, name in enumerate(PART_NAMES)}
    summary["unknown_count"] = int(np.sum(labels < 0))
    known_conf = confidence[labels >= 0]
    summary["mean_confidence"] = float(known_conf.mean()) if known_conf.size else 0.0
    unique_votes, vote_bins = np.unique(vote_count.astype(np.int64), return_counts=True)
    summary["vote_count_histogram"] = {str(int(v)): int(c) for v, c in zip(unique_votes, vote_bins)}
    conflicted = conflict_count > 0
    summary["conflict_stats"] = {
        "total_conflicts": int(conflict_count.astype(np.int64).sum()),
        "conflicted_point_count": int(conflicted.sum()),
        "max_conflicts_per_point": int(conflict_count.max()) if conflict_count.size else 0,
        "mean_conflicts_per_point": float(conflict_count.astype(np.float32).mean()) if conflict_count.size else 0.0,
    }
    return summary


def write_summary_json(path, summary: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def write_preview_ply(path, xyz, part_label) -> None:
    path = Path(path)
    xyz = np.asarray(xyz, dtype=np.float32)
    labels = np.asarray(part_label, dtype=np.int16).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if labels.shape[0] != xyz.shape[0]:
        raise ValueError("part_label shape must match xyz point count")
    colors = np.repeat(UNKNOWN_COLOR_UINT8.reshape(1, 3), xyz.shape[0], axis=0)
    known = (labels >= 0) & (labels < len(PART_NAMES))
    colors[known] = PART_COLORS_UINT8[labels[known]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {xyz.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(xyz, colors):
            handle.write(
                f"{float(point[0]):.7g} {float(point[1]):.7g} {float(point[2]):.7g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
