from __future__ import annotations

import hashlib
from typing import Dict, Mapping, Tuple

import torch


BOUNDARY_SUPPORT_BANK_VERSION = 1


def _zeros(point_count: int, device, dtype=torch.float32):
    return torch.zeros((int(point_count),), dtype=dtype, device=device)


def _prepare_score(value, point_count: int, device, dtype=torch.float32):
    if value is None:
        return _zeros(point_count, device, dtype=dtype)
    if not torch.is_tensor(value):
        value = torch.tensor(value, dtype=dtype, device=device)
    value = value.detach().reshape(-1).to(device=device, dtype=dtype)
    if value.shape[0] != int(point_count):
        raise ValueError(f"support score shape mismatch: got {value.shape[0]}, expected {point_count}")
    if dtype.is_floating_point:
        return value.clamp(0.0, 1.0)
    return value


def _hash_key(value: str) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) & ((1 << 63) - 1)


def _bit_index(value: str, modulo: int = 62) -> int:
    return int(_hash_key(value) % int(modulo))


def initialize_boundary_support_bank_state(
    point_count: int,
    device,
    adopted_under=None,
    adopted_over=None,
    source_iteration: int = 0,
) -> Dict[str, torch.Tensor]:
    point_count = int(point_count)
    under = _prepare_score(adopted_under, point_count, device, dtype=torch.float32)
    over = _prepare_score(adopted_over, point_count, device, dtype=torch.float32)
    zeros_f = _zeros(point_count, device, torch.float32)
    zeros_l = _zeros(point_count, device, torch.long)
    zeros_b = torch.zeros((point_count,), dtype=torch.bool, device=device)

    return {
        "boundary_support_bank_version": torch.tensor([BOUNDARY_SUPPORT_BANK_VERSION], dtype=torch.long, device=device),
        "boundary_live_last_under_score": zeros_f.clone(),
        "boundary_live_last_over_score": zeros_f.clone(),
        "boundary_live_last_valid_mask": zeros_b.clone(),
        "boundary_live_last_key_hash": torch.zeros((1,), dtype=torch.long, device=device),
        "boundary_live_last_iteration": torch.zeros((1,), dtype=torch.long, device=device),
        "boundary_candidate_under_score_sum": zeros_f.clone(),
        "boundary_candidate_over_score_sum": zeros_f.clone(),
        "boundary_candidate_under_score_ema": zeros_f.clone(),
        "boundary_candidate_over_score_ema": zeros_f.clone(),
        "boundary_candidate_under_hits": zeros_l.clone(),
        "boundary_candidate_over_hits": zeros_l.clone(),
        "boundary_candidate_valid_hits": zeros_l.clone(),
        "boundary_candidate_view_bits": zeros_l.clone(),
        "boundary_candidate_frame_bits": zeros_l.clone(),
        "boundary_candidate_first_iter": torch.full((point_count,), -1, dtype=torch.long, device=device),
        "boundary_candidate_last_iter": torch.full((point_count,), -1, dtype=torch.long, device=device),
        "boundary_candidate_bad_frame_hits": zeros_l.clone(),
        "boundary_persistent_under_tag": zeros_f.clone(),
        "boundary_persistent_over_tag": zeros_f.clone(),
        "boundary_persistent_under_birth_iter": torch.full((point_count,), -1, dtype=torch.long, device=device),
        "boundary_persistent_over_birth_iter": torch.full((point_count,), -1, dtype=torch.long, device=device),
        "boundary_persistent_under_source": zeros_l.clone(),
        "boundary_persistent_over_source": zeros_l.clone(),
        "boundary_adopted_under_tag": under,
        "boundary_adopted_over_tag": over,
        "boundary_adopted_under_frozen": under > 0,
        "boundary_adopted_over_frozen": over > 0,
        "boundary_adopted_source_iteration": torch.tensor([int(source_iteration)], dtype=torch.long, device=device),
        "boundary_adopted_initialized": torch.tensor([True], dtype=torch.bool, device=device),
    }


def update_boundary_candidate_support_bank(
    state: Dict[str, torch.Tensor],
    under_score,
    over_score,
    valid_mask,
    key: str,
    iteration: int,
    ema: float,
    score_threshold: float,
    bad_frame: bool = False,
) -> None:
    point_count = int(state["boundary_adopted_under_tag"].shape[0])
    device = state["boundary_adopted_under_tag"].device
    under = _prepare_score(under_score, point_count, device)
    over = _prepare_score(over_score, point_count, device)
    valid = _prepare_score(valid_mask, point_count, device, dtype=torch.bool)
    ema = min(max(float(ema), 0.0), 0.999)
    threshold = float(score_threshold)

    state["boundary_live_last_under_score"] = under.clone()
    state["boundary_live_last_over_score"] = over.clone()
    state["boundary_live_last_valid_mask"] = valid.clone()
    state["boundary_live_last_key_hash"] = torch.tensor([_hash_key(key)], dtype=torch.long, device=device)
    state["boundary_live_last_iteration"] = torch.tensor([int(iteration)], dtype=torch.long, device=device)

    under_active = valid & (under >= threshold)
    over_active = valid & (over >= threshold)
    state["boundary_candidate_under_score_sum"][under_active] += under[under_active]
    state["boundary_candidate_over_score_sum"][over_active] += over[over_active]
    state["boundary_candidate_under_hits"][under_active] += 1
    state["boundary_candidate_over_hits"][over_active] += 1
    state["boundary_candidate_valid_hits"][valid] += 1

    state["boundary_candidate_under_score_ema"] = torch.where(
        under_active,
        state["boundary_candidate_under_score_ema"] * ema + under * (1.0 - ema),
        state["boundary_candidate_under_score_ema"],
    )
    state["boundary_candidate_over_score_ema"] = torch.where(
        over_active,
        state["boundary_candidate_over_score_ema"] * ema + over * (1.0 - ema),
        state["boundary_candidate_over_score_ema"],
    )

    active = under_active | over_active
    view_bit = 1 << _bit_index(str(key).split("_f")[0])
    frame_bit = 1 << _bit_index(str(key))
    state["boundary_candidate_view_bits"][active] |= view_bit
    state["boundary_candidate_frame_bits"][active] |= frame_bit
    first = state["boundary_candidate_first_iter"]
    first[active & (first < 0)] = int(iteration)
    state["boundary_candidate_last_iter"][active] = int(iteration)
    if bool(bad_frame):
        state["boundary_candidate_bad_frame_hits"][active] += 1


def _popcount_long(values: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(values, dtype=torch.long)
    work = values.clone().long()
    for _ in range(63):
        out += work & 1
        work = work >> 1
    return out


def _ensure_scalar_long(state: Dict[str, torch.Tensor], key: str, device) -> torch.Tensor:
    value = state.get(key)
    if torch.is_tensor(value) and value.numel() == 1:
        return value
    value = torch.zeros((1,), dtype=torch.long, device=device)
    state[key] = value
    return value


def _cap_limit(point_count: int, ratio) -> int:
    if ratio is None:
        return int(point_count)
    ratio = float(ratio)
    if ratio < 0:
        return int(point_count)
    return max(0, min(int(point_count), int(point_count * ratio)))


def _limit_new_candidates(candidate: torch.Tensor, score: torch.Tensor, slots: int) -> torch.Tensor:
    slots = int(slots)
    if slots <= 0:
        return torch.zeros_like(candidate, dtype=torch.bool)
    count = int(candidate.sum().item())
    if count <= slots:
        return candidate
    indices = torch.nonzero(candidate, as_tuple=False).reshape(-1)
    try:
        ranked = torch.argsort(score[indices], descending=True, stable=True)
    except TypeError:
        ranked = torch.argsort(score[indices], descending=True)
    keep = indices[ranked[:slots]]
    out = torch.zeros_like(candidate, dtype=torch.bool)
    out[keep] = True
    return out


def promote_boundary_candidate_support(
    state: Dict[str, torch.Tensor],
    min_hits: int,
    min_view_bits: int,
    min_frame_bits: int,
    score_threshold: float,
    dominance_margin: float,
    iteration: int,
    under_max_effective_ratio=None,
    over_max_effective_ratio=None,
    under_max_new_only_ratio=None,
    over_max_new_only_ratio=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    min_hits = int(min_hits)
    min_view_bits = int(min_view_bits)
    min_frame_bits = int(min_frame_bits)
    threshold = float(score_threshold)
    margin = max(float(dominance_margin), 1.0)

    under_hits = state["boundary_candidate_under_hits"]
    over_hits = state["boundary_candidate_over_hits"]
    under_score = state["boundary_candidate_under_score_ema"]
    over_score = state["boundary_candidate_over_score_ema"]
    view_count = _popcount_long(state["boundary_candidate_view_bits"])
    frame_count = _popcount_long(state["boundary_candidate_frame_bits"])

    enough_context = (view_count >= min_view_bits) & (frame_count >= min_frame_bits)
    under_candidate = (
        (under_hits >= min_hits)
        & enough_context
        & (under_score >= threshold)
        & ((under_score + 1.0e-6) >= over_score * margin)
    )
    over_candidate = (
        (over_hits >= min_hits)
        & enough_context
        & (over_score >= threshold)
        & ((over_score + 1.0e-6) >= under_score * margin)
    )

    under_candidate = under_candidate & ~state["boundary_adopted_over_frozen"]
    over_candidate = over_candidate & ~state["boundary_adopted_under_frozen"]
    conflict = under_candidate & over_candidate
    under_candidate = under_candidate & ~conflict
    over_candidate = over_candidate & ~conflict

    point_count = int(state["boundary_adopted_under_tag"].shape[0])
    device = state["boundary_adopted_under_tag"].device
    raw_new_under = under_candidate & (state["boundary_persistent_under_tag"] <= 0)
    raw_new_over = over_candidate & (state["boundary_persistent_over_tag"] <= 0)

    adopted_under = state["boundary_adopted_under_tag"].detach().reshape(-1).float() > 0
    adopted_over = state["boundary_adopted_over_tag"].detach().reshape(-1).float() > 0
    persistent_under = state["boundary_persistent_under_tag"].detach().reshape(-1).float() > 0
    persistent_over = state["boundary_persistent_over_tag"].detach().reshape(-1).float() > 0

    current_under_effective = int((adopted_under | persistent_under).sum().item())
    current_over_effective = int((adopted_over | persistent_over).sum().item())
    current_under_new_only = int((~adopted_under & persistent_under).sum().item())
    current_over_new_only = int((~adopted_over & persistent_over).sum().item())

    under_effective_limit = _cap_limit(point_count, under_max_effective_ratio)
    over_effective_limit = _cap_limit(point_count, over_max_effective_ratio)
    under_new_limit = _cap_limit(point_count, under_max_new_only_ratio)
    over_new_limit = _cap_limit(point_count, over_max_new_only_ratio)

    under_slots = min(
        under_effective_limit - current_under_effective,
        under_new_limit - current_under_new_only,
    )
    over_slots = min(
        over_effective_limit - current_over_effective,
        over_new_limit - current_over_new_only,
    )

    new_under = _limit_new_candidates(raw_new_under, under_score, under_slots)
    new_over = _limit_new_candidates(raw_new_over, over_score, over_slots)

    under_blocked = int(raw_new_under.sum().item()) - int(new_under.sum().item())
    over_blocked = int(raw_new_over.sum().item()) - int(new_over.sum().item())
    _ensure_scalar_long(state, "boundary_support_under_last_promote_blocked", device)[0] = max(0, under_blocked)
    _ensure_scalar_long(state, "boundary_support_over_last_promote_blocked", device)[0] = max(0, over_blocked)
    _ensure_scalar_long(state, "boundary_support_under_last_promote_allowed", device)[0] = int(new_under.sum().item())
    _ensure_scalar_long(state, "boundary_support_over_last_promote_allowed", device)[0] = int(new_over.sum().item())
    _ensure_scalar_long(state, "boundary_support_under_cap_blocked_total", device)[0] += max(0, under_blocked)
    _ensure_scalar_long(state, "boundary_support_over_cap_blocked_total", device)[0] += max(0, over_blocked)

    state["boundary_persistent_under_tag"][new_under] = 1.0
    state["boundary_persistent_over_tag"][new_over] = 1.0
    state["boundary_persistent_under_birth_iter"][new_under] = int(iteration)
    state["boundary_persistent_over_birth_iter"][new_over] = int(iteration)
    state["boundary_persistent_under_source"][new_under] = 2
    state["boundary_persistent_over_source"][new_over] = 2
    return new_under, new_over


def materialize_effective_boundary_tags(state: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    under = torch.maximum(state["boundary_adopted_under_tag"], state["boundary_persistent_under_tag"]).float().clamp(0.0, 1.0)
    over = torch.maximum(state["boundary_adopted_over_tag"], state["boundary_persistent_over_tag"]).float().clamp(0.0, 1.0)
    return under, over


def boundary_support_overlap_stats(adopted, effective) -> Dict[str, float]:
    adopted_mask = adopted.detach().reshape(-1).float() > 0
    effective_mask = effective.detach().reshape(-1).float() > 0
    intersection = int((adopted_mask & effective_mask).sum().item())
    union = int((adopted_mask | effective_mask).sum().item())
    adopted_count = int(adopted_mask.sum().item())
    effective_count = int(effective_mask.sum().item())
    return {
        "adopted_count": adopted_count,
        "effective_count": effective_count,
        "intersection": intersection,
        "union": union,
        "jaccard": float(intersection / union) if union > 0 else 1.0,
        "adopted_lost": int((adopted_mask & ~effective_mask).sum().item()),
        "new_only": int((~adopted_mask & effective_mask).sum().item()),
    }


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, abs_value: bool = False) -> float:
    if value is None or not torch.is_tensor(value):
        return 0.0
    flat = value.detach().reshape(value.shape[0], -1).float()
    mask = mask.detach().reshape(-1).bool()
    if not bool(mask.any().item()):
        return 0.0
    selected = flat[mask]
    if abs_value:
        selected = selected.abs()
    return float(selected.mean().item())


def boundary_residual_support_stats(under_tag, over_tag, grow_opacity_residual, shrink_opacity_residual) -> Dict[str, float]:
    under_mask = under_tag.detach().reshape(-1).float() > 0
    over_mask = over_tag.detach().reshape(-1).float() > 0
    return {
        "grow_count": int(under_mask.sum().item()),
        "shrink_count": int(over_mask.sum().item()),
        "grow_mean": _masked_mean(grow_opacity_residual, under_mask, abs_value=False),
        "grow_abs_mean": _masked_mean(grow_opacity_residual, under_mask, abs_value=True),
        "shrink_mean": _masked_mean(shrink_opacity_residual, over_mask, abs_value=False),
        "shrink_abs_mean": _masked_mean(shrink_opacity_residual, over_mask, abs_value=True),
    }


def boundary_bad_frame_attribution_stats(state: Mapping[str, torch.Tensor], effective_under, effective_over) -> Dict[str, int]:
    adopted_under = state["boundary_adopted_under_tag"].detach().reshape(-1).float() > 0
    adopted_over = state["boundary_adopted_over_tag"].detach().reshape(-1).float() > 0
    effective_under = effective_under.detach().reshape(-1).float() > 0
    effective_over = effective_over.detach().reshape(-1).float() > 0
    bad_hits = state["boundary_candidate_bad_frame_hits"].detach().reshape(-1).long()
    under_new = effective_under & ~adopted_under
    over_new = effective_over & ~adopted_over
    under_hits = int(bad_hits[under_new].sum().item()) if bool(under_new.any().item()) else 0
    over_hits = int(bad_hits[over_new].sum().item()) if bool(over_new.any().item()) else 0
    return {
        "under_new_only_bad_hits": under_hits,
        "over_new_only_bad_hits": over_hits,
        "total_new_only_bad_hits": under_hits + over_hits,
    }
