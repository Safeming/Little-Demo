# v398 Stable Generalization Design

## Goal

Turn v397 from a run where one checkpoint passes into a stable StageB repair path where boundary support growth is bounded, local bad-frame regressions are gated, and downstream users cannot accidentally consume `best_ckpt.pth` when the raw-gate selector chose a different checkpoint.

This design keeps the current v397 semantic fix intact:

- adopted support from v392 is frozen and never removed;
- candidate support only accumulates and promotes into persistent support;
- effective support remains `adopted OR persistent`;
- `fg`, `boundary`, and `edge` must not be sacrificed;
- no first-order changes to lr, loss weights, or top-k schedules.

## Current Evidence

Checked paths:

- `utils/boundary_support_bank.py`
- `train.py::_maybe_update_boundary_support_bank`
- `scene/gaussian_model.py` support-bank wrappers
- `tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh`
- `tools/run_377_explicit_binding_v397_persistent_support_bank.sh`
- `tools/formal/run_377_v338_raw_contour_gate.sh`
- v397 selected artifacts under `exp/formal/logs/377_v397_persistent_support_bank_v397_persistent_support_bank_selectorfix_detached_20260601_135459_bjt/`

v397 solved the original support drift root cause. The selected checkpoint has:

- `under_adopted_lost=0`
- `over_adopted_lost=0`
- `status=strict_pass`
- `fg_delta=-0.00000755`
- `boundary_delta=-0.00001460`
- `edge_delta=-0.013170`
- selected checkpoint: `ckpt141160.pth`

Remaining instability:

- support growth is broad:
  - under `effective_count=12708`, `new_only=11210`
  - over `effective_count=24513`, `new_only=23015`
- only 1 of 6 checkpoints passed strict + v392 floors + support floor;
- local bad frames remain in `worst_frames.tsv`, including `c23_f000000` with `outer_delta=7` and `hard_delta=0.00026392`;
- `best_ckpt.pth` is training-best, not selector-best.

## Considered Approaches

### Approach A: Tighten support promotion thresholds only

Raise `min_hits`, `min_views`, `promote_threshold`, or dominance margin.

Pros:
- small code change;
- keeps current data model.

Cons:
- indirectly controls growth and can under-promote useful support;
- behaves like threshold tuning, close to the "do not first tune topk/loss/lr" constraint;
- does not solve local bad-frame selection or artifact misuse.

### Approach B: Add hard support growth caps plus selector gates

Keep v397 support semantics, but add explicit under/over caps for effective support and new-only support. Continue accumulating candidate statistics after caps are hit, but stop promoting additional points. Add raw-gate worst-frame parsing and stable-window selection.

Pros:
- directly addresses the observed failure modes;
- keeps adopted support safety invariant;
- separates training-time support semantics from selector-time quality gates;
- allows diagnostics to show what would have promoted after caps.

Cons:
- needs new unit tests and selector tests;
- cap defaults must be conservative enough not to block valid support.

### Approach C: Two-stage support bank with rollback

Promote freely during training, then post-process or roll back persistent support based on raw-gate outcomes.

Pros:
- may recover from over-growth after the fact.

Cons:
- breaks "candidate support only增不覆盖" semantics if persistent tags are removed;
- more complex checkpoint semantics;
- raw-gate is too expensive to drive training-time rollback frequently.

## Recommended Design

Use Approach B.

v398 adds four scoped changes in this order:

1. Support Growth Brake
2. Bad-Frame Selector Gate
3. Stable Plateau Selection
4. Artifact/Productization and JSON schema cleanup

## Support Growth Brake

### New Config Keys

Training config keys passed through `EXTRA_TRAIN_ARGS`:

- `opt.boundary_support_bank_under_max_effective_ratio`
- `opt.boundary_support_bank_over_max_effective_ratio`
- `opt.boundary_support_bank_under_max_new_only_ratio`
- `opt.boundary_support_bank_over_max_new_only_ratio`

Generic aliases may also be supported:

- `opt.boundary_support_bank_max_effective_ratio`
- `opt.boundary_support_bank_max_new_only_ratio`

Direction-specific keys override generic keys.

### Default v398 Values

Initial v398 wrapper defaults should be conservative and explicit:

- under max effective ratio: `0.30`
- over max effective ratio: `0.45`
- under max new-only ratio: `0.24`
- over max new-only ratio: `0.38`

Rationale: v397 selected under is about `12708 / 46801 = 0.2715`, over is about `24513 / 46801 = 0.5238`. These defaults allow the selected under scale but brake over before it covers more than half the point set.

### Promotion Semantics

`update_boundary_candidate_support_bank()` continues to update live and candidate state regardless of caps.

`promote_boundary_candidate_support()` receives optional caps:

- `under_max_effective_ratio`
- `over_max_effective_ratio`
- `under_max_new_only_ratio`
- `over_max_new_only_ratio`

For each direction:

1. Compute eligible candidate mask exactly as v397 does.
2. Remove already persistent points.
3. Remove opposite adopted-frozen conflicts exactly as v397 does.
4. Compute remaining promotion slots:
   - effective cap slots = `floor(point_count * max_effective_ratio) - current_effective_count`
   - new-only cap slots = `floor(point_count * max_new_only_ratio) - current_new_only_count`
   - allowed slots = `min(effective cap slots, new-only cap slots)`
5. If slots are exhausted, promote no new persistent tags for that direction.
6. If eligible candidates exceed slots, choose deterministic highest-confidence candidates:
   - under confidence: `boundary_candidate_under_score_ema`
   - over confidence: `boundary_candidate_over_score_ema`
   - tie-breaker: stable tensor order
7. Record diagnostics:
   - promoted count
   - blocked-by-cap count
   - cap limit count
   - current effective/new-only count

Adopted support is not counted as removable. If adopted count already exceeds a cap, adopted support still remains and slots become zero.

### Checkpoint Compatibility

No checkpoint tuple-length change. New diagnostic counters live in `binding_state` as tensors only if needed:

- `boundary_support_under_cap_blocked_total`
- `boundary_support_over_cap_blocked_total`
- `boundary_support_under_last_promote_blocked`
- `boundary_support_over_last_promote_blocked`
- `boundary_support_under_last_promote_allowed`
- `boundary_support_over_last_promote_allowed`

Older checkpoints without these keys load normally. Diagnostics default to zero when keys are absent.

## Bad-Frame Selector Gate

`tools/formal/run_377_v338_raw_contour_gate.sh` already writes:

- `summary.tsv`
- `worst_frames.tsv`

`worst_frames.tsv` contains rows with:

- `variant`
- `image`
- `worsen_score`
- `fg_delta`
- `boundary_delta`
- `edge_delta`
- `inner_delta`
- `outer_delta`
- `hard_delta`
- `opacity_inner_delta`
- `opacity_outer_delta`

The v398 selector parses the `worst_frames.tsv` for each candidate checkpoint after raw gate.

### New Selector Environment Keys

- `BAD_FRAME_SELECTOR_ENABLE=true`
- `BAD_FRAME_OUTER_VETO=5.0`
- `BAD_FRAME_HARD_VETO=0.0`
- `BAD_FRAME_HARD_PENALTY=0.00005`
- `BAD_FRAME_FG_POSITIVE_MAX=0`
- `BAD_FRAME_BOUNDARY_POSITIVE_MAX=0`
- `BAD_FRAME_EDGE_POSITIVE_MAX=0`

### Gate Rules

For candidate rows only:

- veto if any frame has `outer_delta > BAD_FRAME_OUTER_VETO`;
- veto if any frame has `hard_delta > BAD_FRAME_HARD_VETO`;
- record penalty count if any frame has `hard_delta > BAD_FRAME_HARD_PENALTY`;
- veto if count of frames with `fg_delta > 0` exceeds `BAD_FRAME_FG_POSITIVE_MAX`;
- veto if count of frames with `boundary_delta > 0` exceeds `BAD_FRAME_BOUNDARY_POSITIVE_MAX`;
- veto if count of frames with `edge_delta > 0` exceeds `BAD_FRAME_EDGE_POSITIVE_MAX`.

The selector payload records:

- `bad_frame_gate_pass`
- `bad_frame_reject_reasons`
- `bad_frame_max_outer_delta`
- `bad_frame_max_hard_delta`
- `bad_frame_fg_positive_count`
- `bad_frame_boundary_positive_count`
- `bad_frame_edge_positive_count`
- `worst_frames_summary`

## Stable Plateau Selection

v398 defines a stable checkpoint as one satisfying:

- raw gate `status == strict_pass`;
- `fg_delta <= 0`;
- `boundary_delta <= 0`;
- `edge_delta <= 0`;
- v392 floors:
  - `outer_delta <= V392_OUTER_FLOOR`
  - `hard_delta <= V392_HARD_FLOOR`
  - `opacity_outer_delta <= V392_OPACITY_OUTER_FLOOR`
- support floor:
  - `under_adopted_lost <= V392_ADOPTED_LOST_MAX`
  - `over_adopted_lost <= V392_ADOPTED_LOST_MAX`
- bad-frame gate pass.

The selector adds:

- `num_stable_window_pass`
- `stable_window_target=3`
- `stable_window_pass = num_stable_window_pass >= stable_window_target`

Selection order:

1. Prefer stable checkpoints if any exist.
2. Prefer checkpoints that belong to a contiguous stable window of at least two saves.
3. Rank by existing v392 safety miss and inner quality.
4. Use `edge_delta` as a later tie-breaker to avoid hidden edge regressions.

If fewer than three checkpoints pass, the run can still emit a selected checkpoint if one candidate passes all hard gates, but payload must set:

- `stable_window_pass=false`
- `stability_warning="fewer_than_target_stable_checkpoints"`

The validation report must treat this as not fully generalized.

## Artifact/Productization

After selector completes:

- write `selected_checkpoint_path.txt` containing the absolute selected checkpoint path;
- write `selected_checkpoint_metrics.json` with the selected row, support diagnostics, bad-frame diagnostics, and stable-window metadata;
- create or update `selected_ckpt.pth`.

`selected_ckpt.pth` should be a symlink when possible. If symlink creation fails, copy the checkpoint and record:

- `selected_artifact_mode="symlink"` or `"copy"`
- `selected_artifact_path`

The script must never overwrite the training `best_ckpt.pth`.

## JSON Schema Cleanup

Current v397 JSON has `under_jaccard` and `over_jaccard` inside `v392_floor`, even though Jaccard is diagnostic-only.

v398 schema:

```json
{
  "selector": "v398_stable_generalization",
  "selected": {},
  "reject_reason": null,
  "counts": {
    "num_candidates": 6,
    "num_strict_edge_safe": 3,
    "num_v392_safety_floor": 1,
    "num_support_floor": 6,
    "num_bad_frame_gate": 0,
    "num_stable_window_pass": 0
  },
  "floors": {
    "inner_delta": -5.4333,
    "outer_delta": -1.6333,
    "hard_delta": -0.00023074,
    "opacity_outer_delta": -26.8667,
    "adopted_lost_max": 0
  },
  "support_diagnostics": {
    "jaccard_reference": "diagnostic_only",
    "selected_under_jaccard": 0.0,
    "selected_over_jaccard": 0.0,
    "selected_under_new_only": 0,
    "selected_over_new_only": 0
  },
  "bad_frame_diagnostics": {},
  "artifacts": {
    "selected_checkpoint_path_txt": "",
    "selected_checkpoint_metrics_json": "",
    "selected_ckpt": "",
    "selected_artifact_mode": ""
  }
}
```

Backward compatibility:

- keep writing `SELECTED_JSON` at the existing path;
- keep selected row fields available under `selected`;
- consumers that read `payload["selected"]["checkpoint"]` continue to work.

## New Wrapper

Create `tools/run_377_explicit_binding_v398_stable_generalization.sh`.

It should wrap `tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh` the same way v397 does, but enable:

- support bank train/diagnostic/selector;
- support growth caps;
- bad-frame selector;
- stable-window selector;
- selected checkpoint artifacts;
- cleaned JSON schema.

The v396 controller can host generic selector/support implementation so v397 and v398 share code. v398 should be opt-in through env flags; v396 defaults must not change.

## Success Criteria

Unit-level:

- support cap preserves adopted support;
- candidate/live stats still accumulate after cap;
- promotion is deterministic under cap;
- selector rejects bad local frame regressions;
- selector counts stable checkpoints;
- selected artifacts point to selected checkpoint, not `best_ckpt.pth`;
- JSON schema no longer puts Jaccard under floors.

Experiment-level:

- v398 selected checkpoint has `under_adopted_lost=0` and `over_adopted_lost=0`;
- selected checkpoint has no `fg/boundary/edge` regression;
- selected checkpoint meets v392 outer/hard/opacity_outer floors;
- bad-frame gate passes or clearly rejects the run;
- target result: at least 3 of 6 checkpoints pass stable-window criteria.

## Risks

- Caps may be too tight and block useful support; mitigate with diagnostics and conservative defaults.
- Hard-frame gate may reject all candidates if thresholds are too strict; selector should report reject reasons instead of silently falling back.
- Symlink creation may fail on some filesystems; copy fallback is required.
- Moving selector logic while keeping it embedded in shell can become brittle; use small Python helpers inside the shell only as a minimal step, and consider extracting a dedicated selector module if it grows further.
- Existing dirty worktree contains many unrelated changes; implementation must avoid reverting or normalizing unrelated files.
