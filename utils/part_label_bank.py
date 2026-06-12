from __future__ import annotations

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


def _as_array(value):
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _require_int16_range(name: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.size and (value.min() < np.iinfo(np.int16).min or value.max() > np.iinfo(np.int16).max):
        raise ValueError(f"{name} values exceed int16 range")
    return value.astype(np.int16, copy=False)


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
    if "neighbor_fill_mask" in arrays:
        _check_dtype(arrays, "neighbor_fill_mask", np.uint8)
        if _as_array(arrays["neighbor_fill_mask"]).shape != (point_count,):
            raise ValueError(f"neighbor_fill_mask must have shape ({point_count},)")


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
