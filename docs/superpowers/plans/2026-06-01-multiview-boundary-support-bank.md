# Multi-View Boundary Support Bank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-view persistent boundary support bank so StageB directional boundary residuals continue from v392 adopted support instead of replacing it with current batch/view tags.

**Architecture:** Keep the current `_boundary_under_tag` / `_boundary_over_tag` tensors as the forward-compatible effective support consumed by `GaussianModel.get_opacity()` and `get_scaling()`, but stop writing current-view tags into them directly. Store adopted support, candidate support bank, persistent promoted support, and diagnostics in `GaussianModel.binding_state`; `train.py` updates the bank from live image scores and materializes effective tags as `adopted OR promoted_candidate`.

**Tech Stack:** Python, PyTorch tensors, existing `GaussianModel.binding_state` checkpoint persistence, pytest-style unit tests, existing StageB shell scripts and raw contour selector tools.

---

## File Structure

- Create `utils/boundary_support_bank.py`
  Pure tensor logic for support-bank state creation, candidate accumulation, promotion, effective tag materialization, overlap/Jaccard/residual diagnostics, and JSON-safe summaries. This file must not import `train.py`, `Scene`, Hydra, or dataset modules.

- Create `tests/test_boundary_support_bank.py`
  CPU-only unit tests for adopted freeze, candidate-only growth, checkpoint-safe tensor state, Jaccard diagnostics, and bad-frame attribution counters.

- Modify `scene/gaussian_model.py`
  Add thin wrappers around `binding_state` for initializing adopted support from loaded checkpoint tags, updating/persisting bank state, applying effective tags, and returning diagnostics. Keep checkpoint tuple length unchanged by storing new support-bank fields inside existing `binding_state`.

- Modify `train.py`
  Route `live_boundary_image_under_score` / `live_boundary_image_over_score` through the support bank when `opt.boundary_support_bank_enable=true`; preserve legacy `_maybe_refresh_boundary_direction_tags` behavior when disabled. Add train-time logging for adopted overlap, Jaccard, candidate growth, residual stats, and bad-frame attribution.

- Create `tools/analyze_377_boundary_support_bank.py`
  Checkpoint-level diagnostic tool that compares adopted/persistent/effective support tags between a baseline checkpoint and candidate checkpoints, emitting TSV and JSON.

- Modify `tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh`
  Add opt-in v397-style support-bank flags and selector integration without changing the old v396 default behavior.

- Create `tools/run_377_explicit_binding_v397_persistent_support_bank.sh`
  New experiment script continuing from v392 with support-bank flags, v392 floors, checkpoint sweep, support diagnostics, and raw contour gate.

---

## Data Model

All persistent support-bank state lives under `scene.gaussians.binding_state` using these keys. Values are tensors unless stated otherwise.

```python
BOUNDARY_SUPPORT_BANK_VERSION = 1

training_live_score = {
    "boundary_support_bank_version": torch.tensor([1], dtype=torch.long),
    "boundary_live_last_under_score": FloatTensor[N],
    "boundary_live_last_over_score": FloatTensor[N],
    "boundary_live_last_valid_mask": BoolTensor[N],
    "boundary_live_last_key_hash": LongTensor[1],
    "boundary_live_last_iteration": LongTensor[1],
}

candidate_support_bank = {
    "boundary_candidate_under_score_sum": FloatTensor[N],
    "boundary_candidate_over_score_sum": FloatTensor[N],
    "boundary_candidate_under_score_ema": FloatTensor[N],
    "boundary_candidate_over_score_ema": FloatTensor[N],
    "boundary_candidate_under_hits": LongTensor[N],
    "boundary_candidate_over_hits": LongTensor[N],
    "boundary_candidate_valid_hits": LongTensor[N],
    "boundary_candidate_view_bits": LongTensor[N],
    "boundary_candidate_frame_bits": LongTensor[N],
    "boundary_candidate_first_iter": LongTensor[N],
    "boundary_candidate_last_iter": LongTensor[N],
    "boundary_candidate_bad_frame_hits": LongTensor[N],
}

persistent_support_tags = {
    "boundary_persistent_under_tag": FloatTensor[N],
    "boundary_persistent_over_tag": FloatTensor[N],
    "boundary_persistent_under_birth_iter": LongTensor[N],
    "boundary_persistent_over_birth_iter": LongTensor[N],
    "boundary_persistent_under_source": LongTensor[N],  # 0 none, 1 adopted, 2 promoted_candidate
    "boundary_persistent_over_source": LongTensor[N],
}

adopted_support = {
    "boundary_adopted_under_tag": FloatTensor[N],
    "boundary_adopted_over_tag": FloatTensor[N],
    "boundary_adopted_under_frozen": BoolTensor[N],
    "boundary_adopted_over_frozen": BoolTensor[N],
    "boundary_adopted_source_iteration": LongTensor[1],
    "boundary_adopted_initialized": BoolTensor[1],
}
```

Effective forward tags are still:

```python
_boundary_under_tag = max(boundary_adopted_under_tag, boundary_persistent_under_tag)
_boundary_over_tag = max(boundary_adopted_over_tag, boundary_persistent_over_tag)
```

Conflict handling remains compatible with `boundary_direction_conflict_mode=freeze`: if a point is active in both directions, the lower-confidence candidate side is suppressed before promotion; adopted points are never removed.

---

## Task 1: Add Pure Support-Bank Module

**Files:**
- Create: `utils/boundary_support_bank.py`
- Test: `tests/test_boundary_support_bank.py`

- [ ] **Step 1: Write failing tests for initialization, adopted freeze, and candidate-only growth**

Create `tests/test_boundary_support_bank.py`:

```python
import torch

from utils.boundary_support_bank import (
    initialize_boundary_support_bank_state,
    update_boundary_candidate_support_bank,
    promote_boundary_candidate_support,
    materialize_effective_boundary_tags,
    boundary_support_overlap_stats,
)


def test_adopted_support_is_materialized_and_frozen():
    under = torch.tensor([1.0, 0.0, 1.0, 0.0])
    over = torch.tensor([0.0, 1.0, 0.0, 0.0])

    state = initialize_boundary_support_bank_state(
        point_count=4,
        device=torch.device("cpu"),
        adopted_under=under,
        adopted_over=over,
        source_iteration=140160,
    )

    effective_under, effective_over = materialize_effective_boundary_tags(state)

    assert torch.equal(effective_under, under)
    assert torch.equal(effective_over, over)
    assert torch.equal(state["boundary_adopted_under_frozen"], under > 0)
    assert torch.equal(state["boundary_adopted_over_frozen"], over > 0)
    assert int(state["boundary_adopted_source_iteration"].item()) == 140160


def test_candidate_promotion_only_adds_new_support_without_erasing_adopted():
    state = initialize_boundary_support_bank_state(
        point_count=5,
        device=torch.device("cpu"),
        adopted_under=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
        adopted_over=torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
        source_iteration=140160,
    )

    update_boundary_candidate_support_bank(
        state,
        under_score=torch.tensor([0.0, 0.92, 0.90, 0.0, 0.0]),
        over_score=torch.tensor([0.0, 0.0, 0.0, 0.91, 0.95]),
        valid_mask=torch.tensor([True, True, True, True, True]),
        key="c01_f000000",
        iteration=140200,
        ema=0.0,
        score_threshold=0.80,
        bad_frame=False,
    )
    promote_boundary_candidate_support(
        state,
        min_hits=1,
        min_view_bits=1,
        min_frame_bits=1,
        score_threshold=0.80,
        dominance_margin=1.10,
        iteration=140240,
    )

    effective_under, effective_over = materialize_effective_boundary_tags(state)

    assert torch.equal(effective_under > 0, torch.tensor([True, True, True, False, False]))
    assert torch.equal(effective_over > 0, torch.tensor([False, False, False, True, True]))
    assert bool(effective_under[0].item()) is True
    assert bool(effective_over[3].item()) is True


def test_overlap_stats_report_low_jaccard_and_lost_adopted_support():
    adopted = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    effective = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0])

    stats = boundary_support_overlap_stats(adopted, effective)

    assert stats["adopted_count"] == 2
    assert stats["effective_count"] == 2
    assert stats["intersection"] == 1
    assert stats["union"] == 3
    assert abs(stats["jaccard"] - (1.0 / 3.0)) < 1e-6
    assert stats["adopted_lost"] == 1
    assert stats["new_only"] == 1
```

- [ ] **Step 2: Run tests and verify they fail because module does not exist**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_boundary_support_bank.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'utils.boundary_support_bank'
```

- [ ] **Step 3: Implement minimal pure support-bank module**

Create `utils/boundary_support_bank.py` with these functions:

```python
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


def promote_boundary_candidate_support(
    state: Dict[str, torch.Tensor],
    min_hits: int,
    min_view_bits: int,
    min_frame_bits: int,
    score_threshold: float,
    dominance_margin: float,
    iteration: int,
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

    new_under = under_candidate & (state["boundary_persistent_under_tag"] <= 0)
    new_over = over_candidate & (state["boundary_persistent_over_tag"] <= 0)
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
```

- [ ] **Step 4: Run unit tests and verify they pass**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_boundary_support_bank.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit**

```bash
git add utils/boundary_support_bank.py tests/test_boundary_support_bank.py
git commit -m "feat: add boundary support bank tensor utilities"
```

---

## Task 2: Add Residual Stats and Bad-Frame Attribution Utilities

**Files:**
- Modify: `utils/boundary_support_bank.py`
- Test: `tests/test_boundary_support_bank.py`

- [ ] **Step 1: Add failing tests for residual stats and bad-frame attribution**

Append to `tests/test_boundary_support_bank.py`:

```python
from utils.boundary_support_bank import (
    boundary_residual_support_stats,
    boundary_bad_frame_attribution_stats,
)


def test_residual_support_stats_are_directional():
    under = torch.tensor([1.0, 0.0, 1.0, 0.0])
    over = torch.tensor([0.0, 1.0, 0.0, 1.0])
    grow = torch.tensor([[0.10], [0.20], [0.30], [0.40]])
    shrink = torch.tensor([[-0.10], [-0.20], [-0.30], [-0.40]])

    stats = boundary_residual_support_stats(under, over, grow, shrink)

    assert stats["grow_count"] == 2
    assert stats["shrink_count"] == 2
    assert abs(stats["grow_mean"] - 0.20) < 1e-6
    assert abs(stats["grow_abs_mean"] - 0.20) < 1e-6
    assert abs(stats["shrink_mean"] - (-0.30)) < 1e-6
    assert abs(stats["shrink_abs_mean"] - 0.30) < 1e-6


def test_bad_frame_attribution_counts_new_only_support():
    state = initialize_boundary_support_bank_state(
        point_count=4,
        device=torch.device("cpu"),
        adopted_under=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        adopted_over=torch.tensor([0.0, 0.0, 1.0, 0.0]),
        source_iteration=140160,
    )
    state["boundary_candidate_bad_frame_hits"] = torch.tensor([0, 3, 0, 5])
    effective_under = torch.tensor([1.0, 1.0, 0.0, 0.0])
    effective_over = torch.tensor([0.0, 0.0, 1.0, 1.0])

    stats = boundary_bad_frame_attribution_stats(state, effective_under, effective_over)

    assert stats["under_new_only_bad_hits"] == 3
    assert stats["over_new_only_bad_hits"] == 5
    assert stats["total_new_only_bad_hits"] == 8
```

- [ ] **Step 2: Run tests and verify new symbols fail**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_boundary_support_bank.py -q
```

Expected:

```text
ImportError: cannot import name 'boundary_residual_support_stats'
```

- [ ] **Step 3: Implement residual and bad-frame stats**

Append to `utils/boundary_support_bank.py`:

```python
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
```

- [ ] **Step 4: Run unit tests**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_boundary_support_bank.py -q
```

Expected:

```text
5 passed
```

- [ ] **Step 5: Commit**

```bash
git add utils/boundary_support_bank.py tests/test_boundary_support_bank.py
git commit -m "feat: add boundary support diagnostics"
```

---

## Task 3: Integrate Bank State With GaussianModel Binding State

**Files:**
- Modify: `scene/gaussian_model.py`
- Test: `tests/test_boundary_support_bank.py`

- [ ] **Step 1: Add failing wrapper tests using a minimal GaussianModel instance**

Append to `tests/test_boundary_support_bank.py`:

```python
from omegaconf import OmegaConf
from scene.gaussian_model import GaussianModel


def _minimal_gaussian_model(point_count=4):
    model = GaussianModel(OmegaConf.create({
        "use_sh": True,
        "sh_degree": 0,
        "feature_dim": 3,
        "directional_boundary_residual_enable": True,
        "directional_boundary_residual_conflict_mode": "freeze",
    }))
    model._xyz = torch.zeros((point_count, 3))
    model._boundary_under_tag = torch.zeros((point_count,), dtype=torch.float32)
    model._boundary_over_tag = torch.zeros((point_count,), dtype=torch.float32)
    return model


def test_gaussian_model_initializes_support_bank_from_current_tags():
    model = _minimal_gaussian_model(4)
    model._boundary_under_tag = torch.tensor([1.0, 0.0, 1.0, 0.0])
    model._boundary_over_tag = torch.tensor([0.0, 1.0, 0.0, 0.0])

    model.initialize_boundary_support_bank_from_current_tags(source_iteration=140160)

    state = model.get_boundary_support_bank_state()
    assert torch.equal(state["boundary_adopted_under_tag"], torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert torch.equal(state["boundary_adopted_over_tag"], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert bool(state["boundary_adopted_initialized"].item()) is True


def test_gaussian_model_applies_effective_support_without_losing_adopted_tags():
    model = _minimal_gaussian_model(4)
    model._boundary_under_tag = torch.tensor([1.0, 0.0, 0.0, 0.0])
    model._boundary_over_tag = torch.tensor([0.0, 0.0, 1.0, 0.0])
    model.initialize_boundary_support_bank_from_current_tags(source_iteration=140160)
    state = model.get_boundary_support_bank_state()
    state["boundary_persistent_under_tag"][1] = 1.0
    state["boundary_persistent_over_tag"][3] = 1.0
    model.set_boundary_support_bank_state(state)

    model.apply_boundary_support_bank_effective_tags(conflict_mode="freeze")

    assert torch.equal(model._boundary_under_tag > 0, torch.tensor([True, True, False, False]))
    assert torch.equal(model._boundary_over_tag > 0, torch.tensor([False, False, True, True]))
```

- [ ] **Step 2: Run wrapper tests and verify missing method failure**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_boundary_support_bank.py::test_gaussian_model_initializes_support_bank_from_current_tags -q
```

Expected:

```text
AttributeError: 'GaussianModel' object has no attribute 'initialize_boundary_support_bank_from_current_tags'
```

- [ ] **Step 3: Add GaussianModel wrapper methods**

In `scene/gaussian_model.py`, add imports near existing utility imports:

```python
from utils.boundary_support_bank import (
    initialize_boundary_support_bank_state,
    materialize_effective_boundary_tags,
    boundary_support_overlap_stats,
    boundary_residual_support_stats,
    boundary_bad_frame_attribution_stats,
)
```

Inside `class GaussianModel`, add methods after `set_binding_state`:

```python
    def has_boundary_support_bank_state(self):
        state = getattr(self, "binding_state", {})
        marker = state.get("boundary_support_bank_version", None) if isinstance(state, dict) else None
        return torch.is_tensor(marker) and marker.numel() == 1

    def get_boundary_support_bank_state(self):
        if not self.has_boundary_support_bank_state():
            return None
        return self.binding_state

    def set_boundary_support_bank_state(self, support_state):
        if support_state is None:
            return
        merged = self.get_binding_state().copy() if self.has_binding_state() else {}
        for key, value in support_state.items():
            merged[key] = value.detach().clone() if torch.is_tensor(value) else value
        self.set_binding_state(merged)

    def initialize_boundary_support_bank_from_current_tags(self, source_iteration=0):
        point_count = int(self.get_xyz.shape[0]) if torch.is_tensor(self._xyz) and self._xyz.ndim >= 2 else 0
        device = self._xyz.device if torch.is_tensor(self._xyz) and self._xyz.numel() > 0 else torch.device("cpu")
        self.ensure_boundary_state_matches_points(verbose=False)
        state = initialize_boundary_support_bank_state(
            point_count=point_count,
            device=device,
            adopted_under=self.get_boundary_direction_tag_state("under"),
            adopted_over=self.get_boundary_direction_tag_state("over"),
            source_iteration=int(source_iteration),
        )
        self.set_boundary_support_bank_state(state)
        return state

    def ensure_boundary_support_bank_initialized(self, source_iteration=0):
        if self.has_boundary_support_bank_state():
            return self.get_boundary_support_bank_state()
        return self.initialize_boundary_support_bank_from_current_tags(source_iteration=source_iteration)

    def apply_boundary_support_bank_effective_tags(self, conflict_mode="freeze"):
        state = self.get_boundary_support_bank_state()
        if state is None:
            return None, None
        under, over = materialize_effective_boundary_tags(state)
        self.set_boundary_direction_tags(under, over, conflict_mode=conflict_mode)
        return self.get_boundary_direction_tag_state("under"), self.get_boundary_direction_tag_state("over")

    def get_boundary_support_bank_diagnostics(self):
        state = self.get_boundary_support_bank_state()
        if state is None:
            return {}
        under, over = materialize_effective_boundary_tags(state)
        out = {}
        for prefix, adopted, effective in (
            ("under", state["boundary_adopted_under_tag"], under),
            ("over", state["boundary_adopted_over_tag"], over),
        ):
            for key, value in boundary_support_overlap_stats(adopted, effective).items():
                out[f"{prefix}_{key}"] = value
        out.update(boundary_residual_support_stats(
            under,
            over,
            self._boundary_grow_opacity_residual,
            self._boundary_shrink_opacity_residual,
        ))
        for key, value in boundary_bad_frame_attribution_stats(state, under, over).items():
            out[key] = value
        return out
```

- [ ] **Step 4: Run wrapper tests**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_boundary_support_bank.py -q
```

Expected:

```text
7 passed
```

- [ ] **Step 5: Verify checkpoint tuple length is unchanged**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python - <<'PY'
import torch
from omegaconf import OmegaConf
from scene.gaussian_model import GaussianModel

model = GaussianModel(OmegaConf.create({
    "use_sh": True,
    "sh_degree": 0,
    "feature_dim": 3,
    "directional_boundary_residual_enable": True,
}))
model._xyz = torch.zeros((3, 3))
model._features_dc = torch.zeros((3, 1, 3))
model._features_rest = torch.zeros((3, 0, 3))
model._scaling = torch.zeros((3, 3))
model._rotation = torch.zeros((3, 4))
model._opacity = torch.zeros((3, 1))
model.ensure_boundary_state_matches_points(verbose=False)
model.initialize_boundary_support_bank_from_current_tags(source_iteration=1)
print(len(model.capture()))
print("boundary_support_bank_version" in model.capture()[22])
PY
```

Expected:

```text
28
True
```

- [ ] **Step 6: Commit**

```bash
git add scene/gaussian_model.py tests/test_boundary_support_bank.py
git commit -m "feat: persist boundary support bank in binding state"
```

---

## Task 4: Replace Direct Direction Tag Refresh With Bank-Gated Refresh

**Files:**
- Modify: `train.py`
- Modify: `utils/boundary_support_bank.py`
- Test: `tests/test_boundary_support_bank.py`

- [ ] **Step 1: Add unit test for promotion thresholds requiring multiple views**

Append to `tests/test_boundary_support_bank.py`:

```python
def test_candidate_requires_multiple_view_bits_before_promotion():
    state = initialize_boundary_support_bank_state(
        point_count=3,
        device=torch.device("cpu"),
        adopted_under=torch.zeros(3),
        adopted_over=torch.zeros(3),
        source_iteration=0,
    )

    update_boundary_candidate_support_bank(
        state,
        under_score=torch.tensor([0.9, 0.0, 0.0]),
        over_score=torch.zeros(3),
        valid_mask=torch.ones(3, dtype=torch.bool),
        key="c01_f000000",
        iteration=1,
        ema=0.0,
        score_threshold=0.8,
    )
    promote_boundary_candidate_support(
        state,
        min_hits=1,
        min_view_bits=2,
        min_frame_bits=1,
        score_threshold=0.8,
        dominance_margin=1.1,
        iteration=2,
    )
    under, _ = materialize_effective_boundary_tags(state)
    assert int((under > 0).sum().item()) == 0

    update_boundary_candidate_support_bank(
        state,
        under_score=torch.tensor([0.95, 0.0, 0.0]),
        over_score=torch.zeros(3),
        valid_mask=torch.ones(3, dtype=torch.bool),
        key="c02_f000000",
        iteration=3,
        ema=0.0,
        score_threshold=0.8,
    )
    promote_boundary_candidate_support(
        state,
        min_hits=2,
        min_view_bits=2,
        min_frame_bits=1,
        score_threshold=0.8,
        dominance_margin=1.1,
        iteration=4,
    )
    under, _ = materialize_effective_boundary_tags(state)
    assert torch.equal(under > 0, torch.tensor([True, False, False]))
```

- [ ] **Step 2: Run unit tests**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_boundary_support_bank.py -q
```

Expected:

```text
8 passed
```

- [ ] **Step 3: Add `train.py` support-bank helper**

In `train.py`, import pure helpers:

```python
from utils.boundary_support_bank import (
    update_boundary_candidate_support_bank,
    promote_boundary_candidate_support,
)
```

Add helper after `_maybe_refresh_boundary_direction_tags`:

```python
def _maybe_update_boundary_support_bank(
    scene,
    data,
    under_score,
    over_score,
    valid_mask,
    config,
    iteration,
    local_iteration=None,
):
    if not bool(config.opt.get("boundary_support_bank_enable", False)):
        return False
    if under_score is None and over_score is None:
        return False
    scene.gaussians.ensure_boundary_support_bank_initialized(source_iteration=iteration)

    key_mode = str(config.opt.get("boundary_support_bank_key", "image")).lower()
    image_name = str(getattr(data, "image_name", ""))
    cam_id = str(getattr(data, "cam_id", "unknown_cam"))
    frame_id = str(getattr(data, "frame_id", "unknown_frame"))
    if key_mode in ("image", "camera_frame", "frame_camera"):
        key = image_name or f"c{cam_id}_f{frame_id}"
    elif key_mode == "frame":
        key = f"f{frame_id}"
    else:
        key = f"c{cam_id}_f{frame_id}"

    state = scene.gaussians.get_boundary_support_bank_state()
    update_boundary_candidate_support_bank(
        state,
        under_score=under_score,
        over_score=over_score,
        valid_mask=valid_mask,
        key=key,
        iteration=int(iteration),
        ema=float(config.opt.get("boundary_support_bank_ema", 0.72)),
        score_threshold=float(config.opt.get("boundary_support_bank_score_threshold", 0.50)),
        bad_frame=bool(getattr(data, "boundary_bad_frame", False)),
    )

    tag_iteration = local_iteration if (
        local_iteration is not None
        and bool(config.opt.get("boundary_tag_schedule_use_local_iteration", False))
    ) else iteration
    init_iter = int(config.opt.get("boundary_support_bank_promote_init_iter", 0))
    interval = int(config.opt.get("boundary_support_bank_promote_interval", 40))
    until_iter = int(config.opt.get("boundary_support_bank_promote_until_iter", init_iter))
    should_promote = tag_iteration >= init_iter and interval > 0 and tag_iteration <= until_iter and (
        (tag_iteration - init_iter) % interval == 0
    )
    if should_promote:
        promote_boundary_candidate_support(
            state,
            min_hits=int(config.opt.get("boundary_support_bank_min_hits", 2)),
            min_view_bits=int(config.opt.get("boundary_support_bank_min_views", 2)),
            min_frame_bits=int(config.opt.get("boundary_support_bank_min_frames", 1)),
            score_threshold=float(config.opt.get("boundary_support_bank_promote_threshold", 0.60)),
            dominance_margin=float(config.opt.get("boundary_support_bank_dominance_margin", 1.15)),
            iteration=int(iteration),
        )
    scene.gaussians.set_boundary_support_bank_state(state)
    scene.gaussians.apply_boundary_support_bank_effective_tags(
        conflict_mode=str(config.opt.get("boundary_direction_conflict_mode", "freeze"))
    )
    return True
```

- [ ] **Step 4: Route live image scores through bank before legacy refresh**

Replace the block at `train.py` around the existing `_maybe_refresh_boundary_direction_tags` call:

```python
        if bool(config.opt.get('boundary_direction_tag_enable', False)):
            _maybe_refresh_boundary_direction_tags(
                scene,
                live_boundary_image_under_score,
                live_boundary_image_over_score,
                render_pkg["deformed_gaussian"],
                config,
                schedule_iteration,
                local_iteration=local_iteration,
            )
```

with:

```python
        boundary_bank_updated = _maybe_update_boundary_support_bank(
            scene,
            data,
            live_boundary_image_under_score,
            live_boundary_image_over_score,
            live_boundary_image_valid,
            config,
            schedule_iteration,
            local_iteration=local_iteration,
        )
        if (
            not boundary_bank_updated
            and bool(config.opt.get('boundary_direction_tag_enable', False))
        ):
            _maybe_refresh_boundary_direction_tags(
                scene,
                live_boundary_image_under_score,
                live_boundary_image_over_score,
                render_pkg["deformed_gaussian"],
                config,
                schedule_iteration,
                local_iteration=local_iteration,
            )
```

- [ ] **Step 5: Add v392-adopted freeze behavior on resume**

After checkpoint load and before training starts, near existing `clear_boundary_tags_on_resume` logic in `train.py`, add:

```python
    if checkpoint and bool(config.opt.get("boundary_support_bank_enable", False)):
        source_iteration = loaded_iter if "loaded_iter" in locals() else 0
        scene.gaussians.ensure_boundary_support_bank_initialized(source_iteration=source_iteration)
        scene.gaussians.apply_boundary_support_bank_effective_tags(
            conflict_mode=str(config.opt.get("boundary_direction_conflict_mode", "freeze"))
        )
        print(
            "[BoundarySupportBank] initialized adopted support from checkpoint tags: "
            f"source_iteration={source_iteration}",
            flush=True,
        )
```

If the local variable name for checkpoint iteration is not `loaded_iter`, use the actual value returned by `scene.load_checkpoint(...)` in that function. The implementation must pass that exact loaded checkpoint iteration into `source_iteration`.

- [ ] **Step 6: Run syntax and unit tests**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m py_compile train.py scene/gaussian_model.py utils/boundary_support_bank.py
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_boundary_support_bank.py -q
```

Expected:

```text
8 passed
```

- [ ] **Step 7: Commit**

```bash
git add train.py utils/boundary_support_bank.py tests/test_boundary_support_bank.py
git commit -m "feat: route boundary direction tags through persistent support bank"
```

---

## Task 5: Add Train-Time Diagnostics

**Files:**
- Modify: `train.py`
- Modify: `scene/gaussian_model.py`
- Test: `tests/test_boundary_support_bank.py`

- [ ] **Step 1: Add diagnostic key test**

Append to `tests/test_boundary_support_bank.py`:

```python
def test_gaussian_model_support_bank_diagnostics_include_overlap_and_residual_keys():
    model = _minimal_gaussian_model(3)
    model._boundary_under_tag = torch.tensor([1.0, 0.0, 0.0])
    model._boundary_over_tag = torch.tensor([0.0, 1.0, 0.0])
    model._boundary_grow_opacity_residual = torch.tensor([[0.1], [0.0], [0.2]])
    model._boundary_shrink_opacity_residual = torch.tensor([[0.0], [-0.3], [-0.4]])
    model.initialize_boundary_support_bank_from_current_tags(source_iteration=140160)
    state = model.get_boundary_support_bank_state()
    state["boundary_persistent_under_tag"][2] = 1.0
    model.set_boundary_support_bank_state(state)
    model.apply_boundary_support_bank_effective_tags(conflict_mode="freeze")

    diag = model.get_boundary_support_bank_diagnostics()

    assert "under_jaccard" in diag
    assert "over_jaccard" in diag
    assert "under_adopted_lost" in diag
    assert "under_new_only" in diag
    assert "grow_abs_mean" in diag
    assert "shrink_abs_mean" in diag
```

- [ ] **Step 2: Run unit tests**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_boundary_support_bank.py -q
```

Expected:

```text
9 passed
```

- [ ] **Step 3: Add wandb/log_loss diagnostics**

In `train.py`, inside the existing logging section near boundary tag logs, add:

```python
            if bool(config.opt.get("boundary_support_bank_enable", False)):
                support_diag = scene.gaussians.get_boundary_support_bank_diagnostics()
                for key, value in support_diag.items():
                    if isinstance(value, (int, float)):
                        log_loss[f"loss/boundary_support_bank_{key}"] = float(value)
                state = scene.gaussians.get_boundary_support_bank_state()
                if state is not None:
                    log_loss["loss/boundary_support_bank_candidate_under_hits_mean"] = (
                        state["boundary_candidate_under_hits"].float().mean().item()
                    )
                    log_loss["loss/boundary_support_bank_candidate_over_hits_mean"] = (
                        state["boundary_candidate_over_hits"].float().mean().item()
                    )
```

- [ ] **Step 4: Add console diagnostics at promotion interval**

Inside `_maybe_update_boundary_support_bank`, after `apply_boundary_support_bank_effective_tags`, add:

```python
    if should_promote and bool(config.opt.get("boundary_support_bank_verbose", True)):
        diag = scene.gaussians.get_boundary_support_bank_diagnostics()
        print(
            "[BoundarySupportBank] "
            f"iter={iteration} "
            f"under_jaccard={diag.get('under_jaccard', 1.0):.4f} "
            f"over_jaccard={diag.get('over_jaccard', 1.0):.4f} "
            f"under_lost={int(diag.get('under_adopted_lost', 0))} "
            f"over_lost={int(diag.get('over_adopted_lost', 0))} "
            f"under_new={int(diag.get('under_new_only', 0))} "
            f"over_new={int(diag.get('over_new_only', 0))}",
            flush=True,
        )
```

- [ ] **Step 5: Run compile and tests**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m py_compile train.py scene/gaussian_model.py utils/boundary_support_bank.py
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_boundary_support_bank.py -q
```

Expected:

```text
9 passed
```

- [ ] **Step 6: Commit**

```bash
git add train.py scene/gaussian_model.py tests/test_boundary_support_bank.py
git commit -m "feat: log boundary support bank diagnostics"
```

---

## Task 6: Add Checkpoint Diagnostic Tool

**Files:**
- Create: `tools/analyze_377_boundary_support_bank.py`

- [ ] **Step 1: Create checkpoint diagnostic script**

Create `tools/analyze_377_boundary_support_bank.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from utils.boundary_support_bank import (
    boundary_residual_support_stats,
    boundary_support_overlap_stats,
)


def _load_model_tuple(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, (tuple, list)) or len(ckpt) < 1:
        raise ValueError(f"Unexpected checkpoint format: {path}")
    model = ckpt[0]
    if not isinstance(model, (tuple, list)):
        raise ValueError(f"Unexpected GaussianModel capture format: {path}")
    return model


def _extract(path: Path):
    model = _load_model_tuple(path)
    if len(model) < 10:
        raise ValueError(f"Checkpoint has no boundary direction tags: {path}")
    binding_state = model[22] if len(model) >= 28 and isinstance(model[22], dict) else {}
    under = model[8].detach().reshape(-1).float()
    over = model[9].detach().reshape(-1).float()
    grow = model[12].detach().reshape(-1, 1).float() if len(model) >= 14 else torch.zeros((under.shape[0], 1))
    shrink = model[13].detach().reshape(-1, 1).float() if len(model) >= 14 else torch.zeros((under.shape[0], 1))
    adopted_under = binding_state.get("boundary_adopted_under_tag", under)
    adopted_over = binding_state.get("boundary_adopted_over_tag", over)
    persistent_under = binding_state.get("boundary_persistent_under_tag", torch.zeros_like(under))
    persistent_over = binding_state.get("boundary_persistent_over_tag", torch.zeros_like(over))
    return {
        "under": under,
        "over": over,
        "grow": grow,
        "shrink": shrink,
        "adopted_under": adopted_under.detach().reshape(-1).float(),
        "adopted_over": adopted_over.detach().reshape(-1).float(),
        "persistent_under": persistent_under.detach().reshape(-1).float(),
        "persistent_over": persistent_over.detach().reshape(-1).float(),
        "has_bank": "boundary_support_bank_version" in binding_state,
    }


def _row(base_name: str, ckpt_name: str, direction: str, base_tensor, cand_tensor, cand):
    stats = boundary_support_overlap_stats(base_tensor, cand_tensor)
    residual = boundary_residual_support_stats(cand["under"], cand["over"], cand["grow"], cand["shrink"])
    return {
        "base": base_name,
        "checkpoint": ckpt_name,
        "direction": direction,
        "has_bank": int(bool(cand["has_bank"])),
        **stats,
        **residual,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, type=Path)
    parser.add_argument("--out-tsv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()

    base = _extract(args.baseline)
    rows = []
    for ckpt in args.checkpoint:
        cand = _extract(ckpt)
        rows.append(_row(str(args.baseline), str(ckpt), "under", base["under"], cand["under"], cand))
        rows.append(_row(str(args.baseline), str(ckpt), "over", base["over"], cand["over"], cand))

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    args.out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "out_tsv": str(args.out_tsv), "out_json": str(args.out_json)}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run script against v392 and v396 checkpoints**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python tools/analyze_377_boundary_support_bank.py \
  --baseline exp/formal/377_v392_legacy_stacked_directional_residual_v392_legacy_stacked_directional_residual_20260531_125104_bjt/ckpt140160.pth \
  --checkpoint exp/formal/377_v396_generalized_boundary_controller_v396_generalized_boundary_controller_20260601_093910_bjt/ckpt140410.pth \
  --checkpoint exp/formal/377_v396_generalized_boundary_controller_v396_generalized_boundary_controller_20260601_093910_bjt/ckpt141660.pth \
  --out-tsv exp/formal/logs/boundary_support_bank_v392_v396.tsv \
  --out-json exp/formal/logs/boundary_support_bank_v392_v396.json
```

Expected output includes:

```text
"rows": 4
```

Expected TSV has `jaccard` values near the previously observed low v396-vs-v392 range before this fix is applied.

- [ ] **Step 3: Commit**

```bash
git add tools/analyze_377_boundary_support_bank.py
git commit -m "feat: add boundary support bank checkpoint diagnostics"
```

---

## Task 7: Add Selector Adopted Baseline Floor

**Files:**
- Modify: `tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh`
- Create: `tools/run_377_explicit_binding_v397_persistent_support_bank.sh`

- [ ] **Step 1: Add support diagnostic call to v396 script without changing default selector**

In `tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh`, after each raw gate result is appended to `SELECTOR_SUMMARY`, add an opt-in block controlled by `SUPPORT_BANK_DIAGNOSTIC_ENABLE`:

```bash
SUPPORT_BANK_DIAGNOSTIC_ENABLE="${SUPPORT_BANK_DIAGNOSTIC_ENABLE:-false}"
SUPPORT_BANK_BASELINE_CKPT="${SUPPORT_BANK_BASELINE_CKPT:-$BASE_CKPT}"
SUPPORT_BANK_SUMMARY="${SUPPORT_BANK_SUMMARY:-$LOG_DIR/support_bank_summary.tsv}"
```

Inside the checkpoint loop after raw-gate parsing:

```bash
  if [ "$SUPPORT_BANK_DIAGNOSTIC_ENABLE" = "true" ]; then
    "$PYTHON_BIN" tools/analyze_377_boundary_support_bank.py \
      --baseline "$SUPPORT_BANK_BASELINE_CKPT" \
      --checkpoint "$ckpt" \
      --out-tsv "$LOG_DIR/support_bank_${ckpt_name}.tsv" \
      --out-json "$LOG_DIR/support_bank_${ckpt_name}.json"
    if [ ! -s "$SUPPORT_BANK_SUMMARY" ]; then
      cat "$LOG_DIR/support_bank_${ckpt_name}.tsv" > "$SUPPORT_BANK_SUMMARY"
    else
      tail -n +2 "$LOG_DIR/support_bank_${ckpt_name}.tsv" >> "$SUPPORT_BANK_SUMMARY"
    fi
  fi
```

- [ ] **Step 2: Create v397 script by copying v396 and enabling support bank**

Create `tools/run_377_explicit_binding_v397_persistent_support_bank.sh` from v396 with these exact semantic changes:

```bash
RUN_ID="${RUN_ID:-v397_persistent_support_bank_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
EXP_DIR="${EXP_DIR:-$ROOT/exp/formal/377_v397_persistent_support_bank_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/formal/logs/377_v397_persistent_support_bank_${RUN_ID}}"
SUPPORT_BANK_DIAGNOSTIC_ENABLE="${SUPPORT_BANK_DIAGNOSTIC_ENABLE:-true}"
SUPPORT_BANK_BASELINE_CKPT="${SUPPORT_BANK_BASELINE_CKPT:-$BASE_CKPT}"
V392_UNDER_JACCARD_FLOOR="${V392_UNDER_JACCARD_FLOOR:-0.95}"
V392_OVER_JACCARD_FLOOR="${V392_OVER_JACCARD_FLOOR:-0.95}"
V392_ADOPTED_LOST_MAX="${V392_ADOPTED_LOST_MAX:-0}"
```

Append these flags to `EXTRA_TRAIN_ARGS_VALUE`:

```bash
++opt.boundary_support_bank_enable=true
++opt.boundary_support_bank_key=image
++opt.boundary_support_bank_ema=0.72
++opt.boundary_support_bank_score_threshold=0.50
++opt.boundary_support_bank_promote_init_iter=80
++opt.boundary_support_bank_promote_interval=40
++opt.boundary_support_bank_promote_until_iter=1200
++opt.boundary_support_bank_min_hits=2
++opt.boundary_support_bank_min_views=2
++opt.boundary_support_bank_min_frames=1
++opt.boundary_support_bank_promote_threshold=0.60
++opt.boundary_support_bank_dominance_margin=$BOUNDARY_DIRECTION_DOMINANCE_MARGIN
++opt.boundary_support_bank_verbose=true
```

Keep these v392 safety floors from v396:

```bash
V392_INNER_FLOOR="${V392_INNER_FLOOR:--5.4333}"
V392_OUTER_FLOOR="${V392_OUTER_FLOOR:--1.6333}"
V392_HARD_FLOOR="${V392_HARD_FLOOR:--0.00023074}"
V392_OPACITY_OUTER_FLOOR="${V392_OPACITY_OUTER_FLOOR:--26.8667}"
```

- [ ] **Step 3: Extend selected-checkpoint Python selector in v397**

In the v397 selector Python block, load `SUPPORT_BANK_SUMMARY` and reject checkpoints unless both directions satisfy:

```python
under_jaccard >= float(V392_UNDER_JACCARD_FLOOR)
over_jaccard >= float(V392_OVER_JACCARD_FLOOR)
under_adopted_lost <= int(V392_ADOPTED_LOST_MAX)
over_adopted_lost <= int(V392_ADOPTED_LOST_MAX)
```

Use this row merge logic:

```python
support_by_ckpt = {}
with support_summary.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        item = support_by_ckpt.setdefault(row["checkpoint"], {})
        direction = row["direction"]
        item[f"{direction}_jaccard"] = float(row["jaccard"])
        item[f"{direction}_adopted_lost"] = int(float(row["adopted_lost"]))
        item[f"{direction}_new_only"] = int(float(row["new_only"]))

for row in rows:
    support = support_by_ckpt.get(row["checkpoint"], {})
    row.update(support)
    row["support_floor_pass"] = (
        row.get("under_jaccard", 0.0) >= under_floor
        and row.get("over_jaccard", 0.0) >= over_floor
        and row.get("under_adopted_lost", 999999) <= lost_max
        and row.get("over_adopted_lost", 999999) <= lost_max
    )
```

The candidate pool order must be:

```python
pool = [
    row for row in strict
    if row["support_floor_pass"]
    and row["outer_delta"] <= outer_floor
    and row["hard_delta"] <= hard_floor
    and row["opacity_outer_delta"] <= opacity_outer_floor
]
```

If `pool` is empty, selected JSON must still be written with `"selected": null` and `"reject_reason": "no_checkpoint_passed_v392_metric_and_adopted_support_floors"`.

- [ ] **Step 4: Run shell syntax checks**

Run:

```bash
bash -n tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh
bash -n tools/run_377_explicit_binding_v397_persistent_support_bank.sh
```

Expected: no output and exit code `0`.

- [ ] **Step 5: Commit**

```bash
git add tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh tools/run_377_explicit_binding_v397_persistent_support_bank.sh
git commit -m "feat: gate boundary candidates by adopted support floor"
```

---

## Task 8: Checkpoint Read/Write Compatibility Verification

**Files:**
- Modify: `tests/test_boundary_support_bank.py`

- [ ] **Step 1: Add in-memory capture/restore compatibility test**

Append to `tests/test_boundary_support_bank.py`:

```python
def test_support_bank_survives_capture_restore_through_binding_state():
    model = _minimal_gaussian_model(3)
    model._features_dc = torch.zeros((3, 1, 3))
    model._features_rest = torch.zeros((3, 0, 3))
    model._scaling = torch.zeros((3, 3))
    model._rotation = torch.zeros((3, 4))
    model._opacity = torch.zeros((3, 1))
    model._boundary_under_tag = torch.tensor([1.0, 0.0, 0.0])
    model._boundary_over_tag = torch.tensor([0.0, 1.0, 0.0])
    model.ensure_boundary_state_matches_points(verbose=False)
    model.initialize_boundary_support_bank_from_current_tags(source_iteration=140160)
    state = model.get_boundary_support_bank_state()
    state["boundary_persistent_under_tag"][2] = 1.0
    model.set_boundary_support_bank_state(state)

    captured = model.capture()
    assert len(captured) == 28

    restored = GaussianModel(model.cfg)
    restored.restore(captured, training_args={}, resume_cfg={})

    restored_state = restored.get_boundary_support_bank_state()
    assert restored_state is not None
    assert torch.equal(restored_state["boundary_adopted_under_tag"], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(restored_state["boundary_persistent_under_tag"], torch.tensor([0.0, 0.0, 1.0]))
```

- [ ] **Step 2: Run test**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_boundary_support_bank.py::test_support_bank_survives_capture_restore_through_binding_state -q
```

Expected:

```text
1 passed
```

- [ ] **Step 3: Verify legacy v392 checkpoint still loads**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python - <<'PY'
import torch
ckpt = torch.load("exp/formal/377_v392_legacy_stacked_directional_residual_v392_legacy_stacked_directional_residual_20260531_125104_bjt/ckpt140160.pth", map_location="cpu")
model = ckpt[0]
print(len(model))
print(model[8].shape, model[9].shape)
print(isinstance(model[22], dict))
PY
```

Expected:

```text
28
torch.Size([...]) torch.Size([...])
True
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_boundary_support_bank.py
git commit -m "test: verify support bank checkpoint compatibility"
```

---

## Task 9: End-to-End Smoke Run and Full v397 Validation

**Files:**
- No code files changed in this task.

- [ ] **Step 1: Run fast CPU-safe checks**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m py_compile train.py scene/gaussian_model.py utils/boundary_support_bank.py tools/analyze_377_boundary_support_bank.py
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_boundary_support_bank.py -q
bash -n tools/run_377_explicit_binding_v397_persistent_support_bank.sh
```

Expected:

```text
10 passed
```

- [ ] **Step 2: Run a short GPU smoke train from v392**

Run:

```bash
GPU=0 \
TRAIN_STEPS=80 \
TEST_INTERVAL=80 \
SAVE_ITERATIONS='[80]' \
CHECKPOINT_ITERATIONS='[80]' \
RUN_ID=v397_support_bank_smoke_$(date +%Y%m%d_%H%M%S) \
tools/run_377_explicit_binding_v397_persistent_support_bank.sh
```

Expected log contains:

```text
[BoundarySupportBank] initialized adopted support from checkpoint tags
[BoundarySupportBank] iter=
```

Expected support diagnostic TSV contains:

```text
direction	has_bank
under	1
over	1
```

- [ ] **Step 3: Inspect smoke support floor**

Run:

```bash
latest_log="$(find exp/formal/logs -maxdepth 1 -type d -name '377_v397_persistent_support_bank_v397_support_bank_smoke_*' | sort | tail -n 1)"
cat "$latest_log/support_bank_summary.tsv"
cat "$latest_log/v396_selected_checkpoint.json" 2>/dev/null || cat "$latest_log/v397_selected_checkpoint.json"
```

Expected:

```text
jaccard values are >= 0.95 for under and over
adopted_lost is 0 for under and over
```

- [ ] **Step 4: Run full v397 validation**

Run:

```bash
GPU=0 \
RUN_ID=v397_persistent_support_bank_$(date +%Y%m%d_%H%M%S) \
tools/run_377_explicit_binding_v397_persistent_support_bank.sh
```

Expected:

```text
SELECTOR_SUMMARY=...
SELECTED_JSON=...
```

The selected JSON must either contain a checkpoint passing both v392 metric floors and adopted support floors, or contain:

```json
{
  "selected": null,
  "reject_reason": "no_checkpoint_passed_v392_metric_and_adopted_support_floors"
}
```

- [ ] **Step 5: Compare against v392 and old v396**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python tools/analyze_377_boundary_support_bank.py \
  --baseline exp/formal/377_v392_legacy_stacked_directional_residual_v392_legacy_stacked_directional_residual_20260531_125104_bjt/ckpt140160.pth \
  --checkpoint "$(python - <<'PY'
import json
from pathlib import Path
paths = sorted(Path('exp/formal/logs').glob('377_v397_persistent_support_bank_*/v397_selected_checkpoint.json'))
payload = json.loads(paths[-1].read_text())
selected = payload.get('selected') or {}
print(selected.get('checkpoint', ''))
PY
)" \
  --out-tsv exp/formal/logs/v397_selected_support_overlap.tsv \
  --out-json exp/formal/logs/v397_selected_support_overlap.json
```

Expected if a checkpoint was selected:

```text
under adopted_lost = 0
over adopted_lost = 0
under jaccard >= 0.95
over jaccard >= 0.95
```

- [ ] **Step 6: Commit verification notes**

Append a concise result block to `docs/计划.md` with the v397 selected checkpoint, raw contour metrics, support Jaccard, adopted_lost, residual stats, and bad-frame attribution counts. Then commit:

```bash
git add docs/计划.md
git commit -m "docs: record v397 persistent support bank validation"
```

---

## Implementation Notes

- `boundary_direction_tag_update_interval` remains available for legacy runs. When `boundary_support_bank_enable=true`, current batch scores update only `candidate_support_bank`; they do not directly overwrite `_boundary_under_tag` / `_boundary_over_tag`.

- From v392 resume, `initialize_boundary_support_bank_from_current_tags()` snapshots the loaded checkpoint tags into `boundary_adopted_under_tag` and `boundary_adopted_over_tag`. These are frozen by construction and are always included in materialized effective tags.

- Candidate support is additive only. Promotion writes to `boundary_persistent_under_tag` / `boundary_persistent_over_tag` only where persistent tag is currently zero. It never clears adopted or existing persistent tags.

- Checkpoint compatibility is preserved by storing support-bank tensors inside `binding_state`, already captured in the 28-entry checkpoint tuple. Older checkpoints without support-bank keys initialize adopted support from their current direction tags at resume time.

- Selector baseline floor must combine metric floors and support floors. A checkpoint with better `inner_delta` but low adopted Jaccard or nonzero adopted loss is rejected because it is replacing v392 support instead of improving it.

- Bad-frame attribution starts as point-level counts from frames marked bad by the training/gate path. If no bad-frame label is available in `data`, the implementation records `False`; checkpoint-level attribution still works through raw gate summaries and support overlap.

---

## Self-Review

- Spec coverage:
  - `training_live_score`, `candidate_support_bank`, `persistent_support_tags`, `adopted_support`: covered in Data Model and Tasks 1-4.
  - v392 adopted freeze: covered in Tasks 3-4 and Implementation Notes.
  - Candidate only adds support: covered in Tasks 1 and 4 tests.
  - Checkpoint read/write compatibility: covered in Tasks 3 and 8.
  - Selector adopted baseline floor: covered in Task 7.
  - Under/over overlap, Jaccard, residual stats, bad-frame attribution: covered in Tasks 2, 5, and 6.
  - Tests and verification commands: covered in every task and Task 9.

- Placeholder scan:
  The plan contains concrete file paths, function names, config names, commands, expected outputs, and code snippets for implementation.

- Type consistency:
  The support bank uses `FloatTensor[N]`, `BoolTensor[N]`, and `LongTensor[N]` consistently. Public method names are defined before later tasks call them.
