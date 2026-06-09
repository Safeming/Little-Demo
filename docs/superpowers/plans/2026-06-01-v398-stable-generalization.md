# v398 Stable Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build v398 stable generalization on top of v397 so support growth is capped, local bad-frame regressions are gated, stable checkpoint windows are reported, and selected checkpoint artifacts are unambiguous.

**Architecture:** Keep v397 support-bank semantics in `utils/boundary_support_bank.py`, `scene/gaussian_model.py`, and `train.py`; add capped promotion while preserving live/candidate accumulation. Extend the existing v396/v397 raw-gate selector wrapper with bad-frame parsing, stable-window selection, artifact writing, and cleaned JSON schema, then expose the full configuration through a v398 wrapper.

**Tech Stack:** Python, PyTorch tensors, pytest, shell wrappers, existing raw contour gate TSV files, existing `GaussianModel.binding_state` checkpoint persistence.

---

## File Structure

- Modify: `utils/boundary_support_bank.py`
  - Add optional cap arguments to `promote_boundary_candidate_support()`.
  - Add deterministic top-k-by-score masking under caps.
  - Add cap diagnostics into support-bank state.

- Modify: `train.py`
  - Pass new cap config values from `config.opt` into `promote_boundary_candidate_support()`.
  - Log cap diagnostics when support bank is enabled.

- Modify: `scene/gaussian_model.py`
  - No data model redesign. Only diagnostics may need to surface new cap counters from `binding_state`.

- Modify: `tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh`
  - Add optional v398 selector flags while keeping v396/v397 compatibility.
  - Parse each raw-gate `worst_frames.tsv`.
  - Add `bad_frame_gate_pass`, stable-window counts, selected artifacts, and cleaned JSON schema.

- Create: `tools/run_377_explicit_binding_v398_stable_generalization.sh`
  - Thin wrapper that enables support bank, support caps, bad-frame gate, stable-window selector, selected artifacts, and v398 naming.

- Modify: `tests/test_boundary_support_bank.py`
  - Add CPU-only tests for cap behavior and diagnostics.

- Modify: `tests/test_v396_raw_gate_paths.py`
  - Add static/script-level tests for new selector schema, worst-frame parsing, and selected artifact generation.

- Create: `tests/test_v398_selector_logic.py`
  - Unit tests for pure selector helpers if selector logic is extracted into a Python module. If implementation stays inline inside the shell, keep these cases as script-text tests in `tests/test_v396_raw_gate_paths.py`.

## Task 1: Add Support Growth Brake Tests

**Files:**
- Modify: `tests/test_boundary_support_bank.py`
- Modify: `utils/boundary_support_bank.py`

- [ ] **Step 1: Write failing test for effective/new-only caps**

Append this test to `tests/test_boundary_support_bank.py`:

```python
def test_promotion_respects_directional_growth_caps_without_dropping_adopted():
    state = initialize_boundary_support_bank_state(
        point_count=10,
        device=torch.device("cpu"),
        adopted_under=torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        adopted_over=torch.tensor([0.0] * 10),
        source_iteration=140160,
    )
    update_boundary_candidate_support_bank(
        state,
        under_score=torch.tensor([0.0, 0.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92]),
        over_score=torch.zeros(10),
        valid_mask=torch.ones(10, dtype=torch.bool),
        key="c01_f000000",
        iteration=140200,
        ema=0.0,
        score_threshold=0.80,
    )

    new_under, new_over = promote_boundary_candidate_support(
        state,
        min_hits=1,
        min_view_bits=1,
        min_frame_bits=1,
        score_threshold=0.80,
        dominance_margin=1.10,
        iteration=140240,
        under_max_effective_ratio=0.40,
        under_max_new_only_ratio=0.20,
        over_max_effective_ratio=1.0,
        over_max_new_only_ratio=1.0,
    )
    effective_under, effective_over = materialize_effective_boundary_tags(state)

    assert int((effective_under > 0).sum().item()) == 4
    assert int(new_under.sum().item()) == 2
    assert int(new_over.sum().item()) == 0
    assert torch.equal(effective_under[:2] > 0, torch.tensor([True, True]))
    assert int(state["boundary_support_under_last_promote_blocked"].item()) == 6
    assert int(state["boundary_support_under_last_promote_allowed"].item()) == 2
```

- [ ] **Step 2: Write failing test that candidate stats keep accumulating after cap**

Append:

```python
def test_candidate_stats_continue_accumulating_after_cap_blocks_promotion():
    state = initialize_boundary_support_bank_state(
        point_count=4,
        device=torch.device("cpu"),
        adopted_under=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        adopted_over=torch.zeros(4),
        source_iteration=140160,
    )
    scores = torch.tensor([0.0, 0.95, 0.94, 0.93])
    for idx, key in enumerate(["c01_f000000", "c02_f000000"]):
        update_boundary_candidate_support_bank(
            state,
            under_score=scores,
            over_score=torch.zeros(4),
            valid_mask=torch.ones(4, dtype=torch.bool),
            key=key,
            iteration=140200 + idx,
            ema=0.0,
            score_threshold=0.80,
        )
        promote_boundary_candidate_support(
            state,
            min_hits=1,
            min_view_bits=1,
            min_frame_bits=1,
            score_threshold=0.80,
            dominance_margin=1.10,
            iteration=140240 + idx,
            under_max_effective_ratio=0.25,
            under_max_new_only_ratio=0.0,
            over_max_effective_ratio=1.0,
            over_max_new_only_ratio=1.0,
        )

    effective_under, _ = materialize_effective_boundary_tags(state)
    assert int((effective_under > 0).sum().item()) == 1
    assert int(state["boundary_candidate_under_hits"][1].item()) == 2
    assert int(state["boundary_support_under_cap_blocked_total"].item()) >= 3
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_boundary_support_bank.py -q
```

Expected:

```text
TypeError: promote_boundary_candidate_support() got an unexpected keyword argument 'under_max_effective_ratio'
```

## Task 2: Implement Minimal Support Cap Logic

**Files:**
- Modify: `utils/boundary_support_bank.py`
- Modify: `train.py`

- [ ] **Step 1: Add cap helper functions**

In `utils/boundary_support_bank.py`, add helpers near `_popcount_long()`:

```python
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
    ranked = torch.argsort(score[indices], descending=True, stable=True)
    keep = indices[ranked[:slots]]
    out = torch.zeros_like(candidate, dtype=torch.bool)
    out[keep] = True
    return out
```

- [ ] **Step 2: Extend `promote_boundary_candidate_support()` signature**

Change the signature to:

```python
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
```

- [ ] **Step 3: Apply caps after candidate masks are computed**

Replace the existing `new_under` / `new_over` block with this logic:

```python
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
```

Keep the existing assignment of persistent tags and birth/source metadata immediately after this block.

- [ ] **Step 4: Pass cap config from training**

In `train.py::_maybe_update_boundary_support_bank`, before calling `promote_boundary_candidate_support()`, read generic and direction-specific caps:

```python
        max_effective = config.opt.get("boundary_support_bank_max_effective_ratio", None)
        max_new_only = config.opt.get("boundary_support_bank_max_new_only_ratio", None)
```

Then add these keyword args to the call:

```python
            under_max_effective_ratio=config.opt.get("boundary_support_bank_under_max_effective_ratio", max_effective),
            over_max_effective_ratio=config.opt.get("boundary_support_bank_over_max_effective_ratio", max_effective),
            under_max_new_only_ratio=config.opt.get("boundary_support_bank_under_max_new_only_ratio", max_new_only),
            over_max_new_only_ratio=config.opt.get("boundary_support_bank_over_max_new_only_ratio", max_new_only),
```

- [ ] **Step 5: Extend verbose support-bank log**

In the existing `[BoundarySupportBank]` print in `train.py`, add:

```python
            f"under_blocked={int(diag.get('under_last_promote_blocked', 0))} "
            f"over_blocked={int(diag.get('over_last_promote_blocked', 0))} "
```

If diagnostics surface raw binding-state keys instead, use:

```python
            f"under_blocked={int(diag.get('support_under_last_promote_blocked', 0))} "
            f"over_blocked={int(diag.get('support_over_last_promote_blocked', 0))} "
```

- [ ] **Step 6: Run support-bank tests**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_boundary_support_bank.py -q
```

Expected:

```text
11 passed
```

The exact pass count may be higher if more tests already exist.

## Task 3: Add Bad-Frame Selector Gate Tests

**Files:**
- Modify: `tests/test_v396_raw_gate_paths.py`
- Modify: `tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh`

- [ ] **Step 1: Add static test for worst-frame path capture**

Append:

```python
def test_selector_records_raw_gate_worst_frames_path():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'gate_worst="$gate_log_dir/worst_frames.tsv"' in text
    assert '"bad_frame_summary"' in text
    assert '"worst_frames_summary"' in text
```

- [ ] **Step 2: Add static test for bad-frame thresholds**

Append:

```python
def test_selector_has_bad_frame_veto_thresholds():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "BAD_FRAME_SELECTOR_ENABLE" in text
    assert "BAD_FRAME_OUTER_VETO" in text
    assert "BAD_FRAME_HARD_VETO" in text
    assert "BAD_FRAME_FG_POSITIVE_MAX" in text
    assert "BAD_FRAME_BOUNDARY_POSITIVE_MAX" in text
    assert "BAD_FRAME_EDGE_POSITIVE_MAX" in text
    assert 'row["bad_frame_gate_pass"]' in text
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_v396_raw_gate_paths.py -q
```

Expected:

```text
FAILED ... assert 'gate_worst="$gate_log_dir/worst_frames.tsv"' in text
```

## Task 4: Implement Bad-Frame Selector Gate

**Files:**
- Modify: `tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh`

- [ ] **Step 1: Add env defaults near selector variables**

Add:

```bash
BAD_FRAME_SELECTOR_ENABLE="${BAD_FRAME_SELECTOR_ENABLE:-false}"
BAD_FRAME_OUTER_VETO="${BAD_FRAME_OUTER_VETO:-5.0}"
BAD_FRAME_HARD_VETO="${BAD_FRAME_HARD_VETO:-0.0}"
BAD_FRAME_HARD_PENALTY="${BAD_FRAME_HARD_PENALTY:-0.00005}"
BAD_FRAME_FG_POSITIVE_MAX="${BAD_FRAME_FG_POSITIVE_MAX:-0}"
BAD_FRAME_BOUNDARY_POSITIVE_MAX="${BAD_FRAME_BOUNDARY_POSITIVE_MAX:-0}"
BAD_FRAME_EDGE_POSITIVE_MAX="${BAD_FRAME_EDGE_POSITIVE_MAX:-0}"
```

Record these in `run_info.txt`.

- [ ] **Step 2: Capture worst-frame file path per checkpoint**

In the checkpoint loop after `gate_summary`, add:

```bash
  gate_worst="$gate_log_dir/worst_frames.tsv"
```

Extend `SELECTOR_SUMMARY` header with:

```text
bad_frame_summary
```

Append `str(gate_worst)` in the row writer after `raw_gate_summary`.

- [ ] **Step 3: Parse worst-frame diagnostics in selector Python**

Pass new argv values into the final Python selector:

```bash
"$BAD_FRAME_SELECTOR_ENABLE" "$BAD_FRAME_OUTER_VETO" "$BAD_FRAME_HARD_VETO" "$BAD_FRAME_HARD_PENALTY" "$BAD_FRAME_FG_POSITIVE_MAX" "$BAD_FRAME_BOUNDARY_POSITIVE_MAX" "$BAD_FRAME_EDGE_POSITIVE_MAX"
```

In the Python block, add:

```python
bad_frame_selector_enable = str(sys.argv[12]).lower() == "true"
bad_frame_outer_veto = float(sys.argv[13])
bad_frame_hard_veto = float(sys.argv[14])
bad_frame_hard_penalty = float(sys.argv[15])
bad_frame_fg_positive_max = int(float(sys.argv[16]))
bad_frame_boundary_positive_max = int(float(sys.argv[17]))
bad_frame_edge_positive_max = int(float(sys.argv[18]))


def _float_value(row, key):
    try:
        return float(row.get(key, 0.0) or 0.0)
    except ValueError:
        return 0.0


def load_bad_frame_stats(path):
    path = Path(path)
    stats = {
        "worst_frames_summary": str(path),
        "bad_frame_max_outer_delta": 0.0,
        "bad_frame_max_hard_delta": 0.0,
        "bad_frame_hard_penalty_count": 0,
        "bad_frame_fg_positive_count": 0,
        "bad_frame_boundary_positive_count": 0,
        "bad_frame_edge_positive_count": 0,
        "bad_frame_reject_reasons": [],
    }
    if not path.exists():
        stats["bad_frame_reject_reasons"].append("missing_worst_frames")
        return stats
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if not str(row.get("variant", "")).startswith("candidate_"):
                continue
            outer = _float_value(row, "outer_delta")
            hard = _float_value(row, "hard_delta")
            fg = _float_value(row, "fg_delta")
            boundary = _float_value(row, "boundary_delta")
            edge = _float_value(row, "edge_delta")
            stats["bad_frame_max_outer_delta"] = max(stats["bad_frame_max_outer_delta"], outer)
            stats["bad_frame_max_hard_delta"] = max(stats["bad_frame_max_hard_delta"], hard)
            stats["bad_frame_hard_penalty_count"] += int(hard > bad_frame_hard_penalty)
            stats["bad_frame_fg_positive_count"] += int(fg > 0.0)
            stats["bad_frame_boundary_positive_count"] += int(boundary > 0.0)
            stats["bad_frame_edge_positive_count"] += int(edge > 0.0)
    if stats["bad_frame_max_outer_delta"] > bad_frame_outer_veto:
        stats["bad_frame_reject_reasons"].append("outer_veto")
    if stats["bad_frame_max_hard_delta"] > bad_frame_hard_veto:
        stats["bad_frame_reject_reasons"].append("hard_veto")
    if stats["bad_frame_fg_positive_count"] > bad_frame_fg_positive_max:
        stats["bad_frame_reject_reasons"].append("fg_positive_count")
    if stats["bad_frame_boundary_positive_count"] > bad_frame_boundary_positive_max:
        stats["bad_frame_reject_reasons"].append("boundary_positive_count")
    if stats["bad_frame_edge_positive_count"] > bad_frame_edge_positive_max:
        stats["bad_frame_reject_reasons"].append("edge_positive_count")
    return stats
```

For each `row`:

```python
    bad_stats = load_bad_frame_stats(row.get("bad_frame_summary", ""))
    row.update(bad_stats)
    row["bad_frame_gate_pass"] = (
        not bad_frame_selector_enable
        or len(bad_stats["bad_frame_reject_reasons"]) == 0
    )
```

- [ ] **Step 4: Require bad-frame pass in v392-safe candidates**

Change `v392_safe` filter to include:

```python
    and row.get("bad_frame_gate_pass", True)
```

- [ ] **Step 5: Run selector script tests**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_v396_raw_gate_paths.py -q
bash -n tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh
```

Expected:

```text
passed
```

and shell syntax check exits `0`.

## Task 5: Add Stable Plateau Selection

**Files:**
- Modify: `tests/test_v396_raw_gate_paths.py`
- Modify: `tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh`

- [ ] **Step 1: Add static test for stable-window fields**

Append:

```python
def test_selector_reports_stable_window_counts():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "STABLE_WINDOW_TARGET" in text
    assert "num_stable_window_pass" in text
    assert "stable_window_pass" in text
    assert "fewer_than_target_stable_checkpoints" in text
```

- [ ] **Step 2: Add env default**

Add:

```bash
STABLE_WINDOW_TARGET="${STABLE_WINDOW_TARGET:-3}"
```

Record it in `run_info.txt`.

- [ ] **Step 3: Compute stable candidates**

In selector Python:

```python
stable = [
    row for row in v392_safe
    if row.get("bad_frame_gate_pass", True)
]
stable_window_target = int(float(sys.argv[19]))
```

Update the argv list to pass `"$STABLE_WINDOW_TARGET"`.

Set:

```python
stable_window_pass = len(stable) >= stable_window_target
if support_selector_enable:
    pool = stable or v392_safe
else:
    pool = stable or v392_safe or strict or [row for row in rows if row["status"] == "strict_pass"] or rows
```

- [ ] **Step 4: Add contiguous-window preference**

Before ranking:

```python
def _iteration(row):
    try:
        return int(float(row.get("iteration", 0)))
    except ValueError:
        return 0


stable_iterations = {_iteration(row) for row in stable}
sorted_stable_iterations = sorted(stable_iterations)
stable_neighbors = set()
for prev_iter, next_iter in zip(sorted_stable_iterations, sorted_stable_iterations[1:]):
    if next_iter > prev_iter:
        stable_neighbors.add(prev_iter)
        stable_neighbors.add(next_iter)

for row in rows:
    row["stable_window_member"] = _iteration(row) in stable_neighbors
```

Add this to the start of `rank_key`:

```python
        0 if row.get("stable_window_member", False) else 1,
```

- [ ] **Step 5: Add payload fields**

Add:

```python
"num_stable_window_pass": len(stable),
"stable_window_target": stable_window_target,
"stable_window_pass": stable_window_pass,
"stability_warning": None if stable_window_pass else "fewer_than_target_stable_checkpoints",
```

- [ ] **Step 6: Run tests**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_v396_raw_gate_paths.py -q
bash -n tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh
```

Expected: pass.

## Task 6: Add Selected Checkpoint Artifacts

**Files:**
- Modify: `tests/test_v396_raw_gate_paths.py`
- Modify: `tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh`

- [ ] **Step 1: Add static test for artifact names**

Append:

```python
def test_selector_writes_selected_checkpoint_artifacts():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "SELECTED_CHECKPOINT_PATH_TXT" in text
    assert "SELECTED_CHECKPOINT_METRICS_JSON" in text
    assert "SELECTED_CKPT_LINK" in text
    assert "selected_checkpoint_path.txt" in text
    assert "selected_checkpoint_metrics.json" in text
    assert "selected_ckpt.pth" in text
    assert "best_ckpt.pth" not in text[text.index("SELECTED_CHECKPOINT_PATH_TXT"):]
```

- [ ] **Step 2: Add env defaults**

Add near `SELECTED_JSON`:

```bash
SELECTED_CHECKPOINT_PATH_TXT="${SELECTED_CHECKPOINT_PATH_TXT:-$LOG_DIR/selected_checkpoint_path.txt}"
SELECTED_CHECKPOINT_METRICS_JSON="${SELECTED_CHECKPOINT_METRICS_JSON:-$LOG_DIR/selected_checkpoint_metrics.json}"
SELECTED_CKPT_LINK="${SELECTED_CKPT_LINK:-$EXP_DIR/selected_ckpt.pth}"
```

- [ ] **Step 3: Pass artifact paths into selector Python**

Append argv:

```bash
"$SELECTED_CHECKPOINT_PATH_TXT" "$SELECTED_CHECKPOINT_METRICS_JSON" "$SELECTED_CKPT_LINK"
```

- [ ] **Step 4: Write artifacts when selection exists**

In selector Python after `selected` is computed:

```python
selected_path_txt = Path(sys.argv[20])
selected_metrics_json = Path(sys.argv[21])
selected_ckpt_link = Path(sys.argv[22])
artifact_mode = None
if selected is not None:
    selected_checkpoint = Path(selected["checkpoint"])
    selected_path_txt.write_text(str(selected_checkpoint) + "\n", encoding="utf-8")
    if selected_ckpt_link.exists() or selected_ckpt_link.is_symlink():
        selected_ckpt_link.unlink()
    try:
        selected_ckpt_link.symlink_to(selected_checkpoint)
        artifact_mode = "symlink"
    except OSError:
        import shutil
        shutil.copy2(selected_checkpoint, selected_ckpt_link)
        artifact_mode = "copy"
```

Write metrics JSON after `payload` is built:

```python
selected_metrics_json.write_text(json.dumps({
    "selected": selected,
    "support_diagnostics": payload.get("support_diagnostics", {}),
    "bad_frame_diagnostics": payload.get("bad_frame_diagnostics", {}),
    "stable_window": {
        "num_stable_window_pass": payload.get("num_stable_window_pass", 0),
        "stable_window_target": payload.get("stable_window_target", 0),
        "stable_window_pass": payload.get("stable_window_pass", False),
    },
    "artifacts": payload.get("artifacts", {}),
}, indent=2), encoding="utf-8")
```

- [ ] **Step 5: Add artifact metadata to payload**

Add:

```python
"artifacts": {
    "selected_checkpoint_path_txt": str(selected_path_txt),
    "selected_checkpoint_metrics_json": str(selected_metrics_json),
    "selected_ckpt": str(selected_ckpt_link),
    "selected_artifact_mode": artifact_mode,
},
```

- [ ] **Step 6: Run tests and syntax check**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_v396_raw_gate_paths.py -q
bash -n tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh
```

Expected: pass.

## Task 7: Clean Selector JSON Schema

**Files:**
- Modify: `tests/test_v396_raw_gate_paths.py`
- Modify: `tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh`

- [ ] **Step 1: Strengthen existing Jaccard diagnostic-only test**

Modify `test_support_floor_keeps_jaccard_diagnostic_only()` to also assert:

```python
    v392_floor_start = text.index('"v392_floor": {')
    v392_floor_end = text.index("}", v392_floor_start)
    v392_floor_block = text[v392_floor_start:v392_floor_end]
    assert "under_jaccard" not in v392_floor_block
    assert "over_jaccard" not in v392_floor_block
    assert '"support_diagnostics"' in text
```

- [ ] **Step 2: Change payload schema**

Replace current flat payload count/floor section with:

```python
selected_support = selected or {}
payload = {
    "selector": "v398_stable_generalization" if bad_frame_selector_enable else "v396_strict_raw_contour_v392_safety_floor_then_inner",
    "summary": str(summary_path),
    "selected": selected,
    "reject_reason": None if selected is not None else "no_checkpoint_passed_v392_metric_support_and_bad_frame_floors",
    "counts": {
        "num_candidates": len(rows),
        "num_strict_edge_safe": len(strict),
        "num_v392_safety_floor": len(v392_safe),
        "num_support_floor": len([row for row in rows if row.get("support_floor_pass")]),
        "num_bad_frame_gate": len([row for row in rows if row.get("bad_frame_gate_pass", True)]),
        "num_stable_window_pass": len(stable),
    },
    "num_candidates": len(rows),
    "num_strict_edge_safe": len(strict),
    "num_v392_safety_floor": len(v392_safe),
    "num_support_floor": len([row for row in rows if row.get("support_floor_pass")]),
    "num_stable_window_pass": len(stable),
    "stable_window_target": stable_window_target,
    "stable_window_pass": stable_window_pass,
    "stability_warning": None if stable_window_pass else "fewer_than_target_stable_checkpoints",
    "v392_floor": {
        "inner_delta": inner_floor,
        "outer_delta": outer_floor,
        "hard_delta": hard_floor,
        "opacity_outer_delta": opacity_outer_floor,
        "adopted_lost_max": lost_max,
    },
    "support_diagnostics": {
        "jaccard_reference": "diagnostic_only",
        "selected_under_jaccard": selected_support.get("under_jaccard"),
        "selected_over_jaccard": selected_support.get("over_jaccard"),
        "selected_under_new_only": selected_support.get("under_new_only"),
        "selected_over_new_only": selected_support.get("over_new_only"),
        "selected_under_adopted_lost": selected_support.get("under_adopted_lost"),
        "selected_over_adopted_lost": selected_support.get("over_adopted_lost"),
    },
    "bad_frame_diagnostics": {
        "selected_bad_frame_gate_pass": selected_support.get("bad_frame_gate_pass"),
        "selected_bad_frame_reject_reasons": selected_support.get("bad_frame_reject_reasons"),
        "selected_bad_frame_max_outer_delta": selected_support.get("bad_frame_max_outer_delta"),
        "selected_bad_frame_max_hard_delta": selected_support.get("bad_frame_max_hard_delta"),
        "selected_bad_frame_fg_positive_count": selected_support.get("bad_frame_fg_positive_count"),
        "selected_bad_frame_boundary_positive_count": selected_support.get("bad_frame_boundary_positive_count"),
        "selected_bad_frame_edge_positive_count": selected_support.get("bad_frame_edge_positive_count"),
    },
}
```

Keep the top-level legacy count fields for one version so existing readers do not break.

- [ ] **Step 3: Run tests**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_v396_raw_gate_paths.py -q
```

Expected: pass.

## Task 8: Add v398 Wrapper

**Files:**
- Create: `tools/run_377_explicit_binding_v398_stable_generalization.sh`
- Modify: `tests/test_v396_raw_gate_paths.py`

- [ ] **Step 1: Add wrapper static test**

Append:

```python
def test_v398_wrapper_enables_stable_generalization_defaults():
    wrapper = ROOT / "tools" / "run_377_explicit_binding_v398_stable_generalization.sh"
    text = wrapper.read_text(encoding="utf-8")

    assert "SUPPORT_BANK_TRAIN_ENABLE" in text
    assert "BAD_FRAME_SELECTOR_ENABLE" in text
    assert "STABLE_WINDOW_TARGET" in text
    assert "boundary_support_bank_under_max_effective_ratio" in text
    assert "boundary_support_bank_over_max_effective_ratio" in text
    assert "run_377_explicit_binding_v396_generalized_boundary_controller.sh" in text
```

- [ ] **Step 2: Run and verify test fails**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_v396_raw_gate_paths.py::test_v398_wrapper_enables_stable_generalization_defaults -q
```

Expected:

```text
FileNotFoundError
```

- [ ] **Step 3: Create wrapper**

Create `tools/run_377_explicit_binding_v398_stable_generalization.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_ID="${RUN_ID:-v398_stable_generalization_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
EXP_DIR="${EXP_DIR:-$ROOT/exp/formal/377_v398_stable_generalization_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/formal/logs/377_v398_stable_generalization_${RUN_ID}}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$EXP_DIR/hydra_runtime}"

SELECTOR_NAME="${SELECTOR_NAME:-v398}"
SELECTOR_SUMMARY="${SELECTOR_SUMMARY:-$LOG_DIR/v398_raw_contour_checkpoint_summary.tsv}"
SELECTED_JSON="${SELECTED_JSON:-$LOG_DIR/v398_selected_checkpoint.json}"

SUPPORT_BANK_TRAIN_ENABLE="${SUPPORT_BANK_TRAIN_ENABLE:-true}"
SUPPORT_BANK_DIAGNOSTIC_ENABLE="${SUPPORT_BANK_DIAGNOSTIC_ENABLE:-true}"
SUPPORT_BANK_SELECTOR_ENABLE="${SUPPORT_BANK_SELECTOR_ENABLE:-true}"
SUPPORT_BANK_SUMMARY="${SUPPORT_BANK_SUMMARY:-$LOG_DIR/support_bank_summary.tsv}"

BAD_FRAME_SELECTOR_ENABLE="${BAD_FRAME_SELECTOR_ENABLE:-true}"
BAD_FRAME_OUTER_VETO="${BAD_FRAME_OUTER_VETO:-5.0}"
BAD_FRAME_HARD_VETO="${BAD_FRAME_HARD_VETO:-0.0}"
BAD_FRAME_HARD_PENALTY="${BAD_FRAME_HARD_PENALTY:-0.00005}"
BAD_FRAME_FG_POSITIVE_MAX="${BAD_FRAME_FG_POSITIVE_MAX:-0}"
BAD_FRAME_BOUNDARY_POSITIVE_MAX="${BAD_FRAME_BOUNDARY_POSITIVE_MAX:-0}"
BAD_FRAME_EDGE_POSITIVE_MAX="${BAD_FRAME_EDGE_POSITIVE_MAX:-0}"

STABLE_WINDOW_TARGET="${STABLE_WINDOW_TARGET:-3}"

V392_ADOPTED_LOST_MAX="${V392_ADOPTED_LOST_MAX:-0}"

EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS ++opt.boundary_support_bank_under_max_effective_ratio=0.30"
EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS ++opt.boundary_support_bank_over_max_effective_ratio=0.45"
EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS ++opt.boundary_support_bank_under_max_new_only_ratio=0.24"
EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS ++opt.boundary_support_bank_over_max_new_only_ratio=0.38"

export RUN_ID EXP_DIR LOG_DIR HYDRA_RUN_DIR
export SELECTOR_NAME SELECTOR_SUMMARY SELECTED_JSON
export SUPPORT_BANK_TRAIN_ENABLE SUPPORT_BANK_DIAGNOSTIC_ENABLE SUPPORT_BANK_SELECTOR_ENABLE SUPPORT_BANK_SUMMARY
export BAD_FRAME_SELECTOR_ENABLE BAD_FRAME_OUTER_VETO BAD_FRAME_HARD_VETO BAD_FRAME_HARD_PENALTY
export BAD_FRAME_FG_POSITIVE_MAX BAD_FRAME_BOUNDARY_POSITIVE_MAX BAD_FRAME_EDGE_POSITIVE_MAX
export STABLE_WINDOW_TARGET V392_ADOPTED_LOST_MAX EXTRA_TRAIN_ARGS

exec "$ROOT/tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh" "$@"
```

- [ ] **Step 4: Make wrapper executable**

Run:

```bash
chmod +x tools/run_377_explicit_binding_v398_stable_generalization.sh
```

- [ ] **Step 5: Run tests and syntax checks**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_v396_raw_gate_paths.py -q
bash -n tools/run_377_explicit_binding_v398_stable_generalization.sh
```

Expected: pass.

## Task 9: Full Local Verification

**Files:**
- All modified files

- [ ] **Step 1: Run unit tests**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_boundary_support_bank.py tests/test_v396_raw_gate_paths.py -q
```

Expected:

```text
passed
```

- [ ] **Step 2: Run Python compile checks**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m py_compile train.py scene/gaussian_model.py utils/boundary_support_bank.py tools/analyze_377_boundary_support_bank.py
```

Expected: no output and exit `0`.

- [ ] **Step 3: Run shell syntax checks**

Run:

```bash
bash -n tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh
bash -n tools/run_377_explicit_binding_v397_persistent_support_bank.sh
bash -n tools/run_377_explicit_binding_v398_stable_generalization.sh
```

Expected: no output and exit `0`.

## Task 10: Short GPU Smoke

**Files:**
- Runtime outputs only

- [ ] **Step 1: Launch short v398 smoke**

Run:

```bash
RUN_ID=v398_debug_smoke_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt') \
GPU=0 \
TRAIN_STEPS=250 \
TEST_INTERVAL=250 \
SAVE_ITERATIONS='[250]' \
CHECKPOINT_ITERATIONS='[250]' \
bash tools/run_377_explicit_binding_v398_stable_generalization.sh
```

Expected:

- script exits `0`;
- `v398_selected_checkpoint.json` exists;
- `selected_checkpoint_path.txt` exists;
- `selected_checkpoint_metrics.json` exists;
- `selected_ckpt.pth` exists as symlink or copy;
- support diagnostics show `under_adopted_lost=0` and `over_adopted_lost=0`.

- [ ] **Step 2: Inspect smoke selector output**

Run:

```bash
cat exp/formal/logs/377_v398_stable_generalization_v398_debug_smoke_*/v398_selected_checkpoint.json
cat exp/formal/logs/377_v398_stable_generalization_v398_debug_smoke_*/support_bank_summary.tsv
```

Expected:

- `v392_floor` has no `under_jaccard` or `over_jaccard`;
- `support_diagnostics` contains selected Jaccard and new-only fields;
- `bad_frame_diagnostics` exists;
- `counts.num_stable_window_pass` exists.

## Task 11: Full v398 Validation Run

**Files:**
- Runtime outputs only

- [ ] **Step 1: Launch full run**

Run:

```bash
RUN_ID=v398_stable_generalization_full_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt') \
GPU=0 \
bash tools/run_377_explicit_binding_v398_stable_generalization.sh
```

Expected:

- train completes;
- raw gate runs for all six checkpoints;
- selector JSON exists;
- selected artifacts exist.

- [ ] **Step 2: Verify selected checkpoint**

Run:

```bash
LOG_DIR=$(ls -td exp/formal/logs/377_v398_stable_generalization_v398_stable_generalization_full_* | head -n 1)
cat "$LOG_DIR/v398_selected_checkpoint.json"
cat "$LOG_DIR/v398_raw_contour_checkpoint_summary.tsv"
cat "$LOG_DIR/support_bank_summary.tsv"
cat "$LOG_DIR/selected_checkpoint_path.txt"
cat "$LOG_DIR/selected_checkpoint_metrics.json"
```

Expected hard requirements:

- selected checkpoint is not `best_ckpt.pth`;
- `fg_delta <= 0`;
- `boundary_delta <= 0`;
- `edge_delta <= 0`;
- `outer_delta <= -1.6333`;
- `hard_delta <= -0.00023074`;
- `opacity_outer_delta <= -26.8667`;
- `under_adopted_lost=0`;
- `over_adopted_lost=0`;
- `bad_frame_gate_pass=true`.

Expected generalization target:

- `num_stable_window_pass >= 3`.

If hard requirements pass but `num_stable_window_pass < 3`, the run is usable as a safe selected checkpoint but not yet accepted as stable-generalized.

## Risk Checklist

- [ ] Caps may block useful support. Inspect `boundary_support_*_cap_blocked_total`, `new_only`, and selected metrics before changing any lr/loss/topk.
- [ ] Bad-frame gate may reject all candidates. If that happens, inspect reject reasons instead of relaxing global strict criteria.
- [ ] `torch.argsort(..., stable=True)` may not be available in older PyTorch. If it fails, replace with deterministic two-step ranking by score plus original index.
- [ ] Shell inline Python argv indexes are easy to break. After each selector argument change, run `bash -n` and the static tests.
- [ ] Do not modify `tools/formal/run_377_v338_raw_contour_gate.sh` unless `worst_frames.tsv` is missing or malformed; it already produces the needed fields.
- [ ] Do not overwrite or reinterpret `best_ckpt.pth`; v398 artifacts must make selected raw-gate checkpoint explicit.
