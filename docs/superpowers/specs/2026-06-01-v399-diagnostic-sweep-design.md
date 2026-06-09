# v399 Diagnostic Sweep Design

## Goal

Design a v398.1/v399 path that tests whether v398's over-support cap is too tight, then upgrades the selector so local bad-frame regressions are ranked with severity instead of making every candidate impossible to select.

This design does not change learning rates, losses, top-k ratios, or the support-bank core semantics. The first step is a diagnostic sweep that only changes over-support caps.

## Current Evidence

Checked code and artifacts:

- `tools/run_377_explicit_binding_v398_stable_generalization.sh`
- `tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh`
- `train.py::_maybe_update_boundary_support_bank`
- `utils/boundary_support_bank.py`
- `scene/gaussian_model.py::get_boundary_support_bank_diagnostics`
- `tests/test_boundary_support_bank.py`
- `tests/test_v396_raw_gate_paths.py`
- `exp/formal/logs/377_v397_persistent_support_bank_v397_persistent_support_bank_selectorfix_detached_20260601_135459_bjt/v397_selected_checkpoint.json`
- `exp/formal/logs/377_v398_stable_generalization_v398_stable_generalization_full_20260601_180530_bjt/v398_selected_checkpoint.json`
- v397/v398 `worst_frames.tsv` for `ckpt141160`

v398 engineering guardrails worked:

- adopted support was preserved: all checked support rows have `adopted_lost=0`;
- support caps were enforced;
- support bank diagnostics and selected artifact schema exist;
- `selected_checkpoint_metrics.json` is generated even when no checkpoint is selected.

v398 selection failed:

- `num_candidates=6`
- `num_strict_edge_safe=3`
- `num_v392_safety_floor=0`
- `num_bad_frame_gate=0`
- `num_stable_window_pass=0`

The strongest evidence is the matched `ckpt141160` comparison:

| Run | hard_delta | v392 hard floor | over new_only | selected |
| --- | ---: | ---: | ---: | --- |
| v397 | `-0.00023707` | `-0.00023074` | `23015` | yes |
| v398 | `-0.00022159` | `-0.00023074` | `17784` | no |

v398 misses the v392 hard floor by about `9.15e-06`. The same checkpoint has `5231` fewer over new-only support points than v397.

Bad-frame evidence for `ckpt141160`:

| Run | candidate rows | max_outer_delta | max_hard_delta | fg positive | boundary positive | edge positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v397 | `29` | `7.0` | `0.00026392` | `12` | `12` | `13` |
| v398 | `28` | `7.0` | `0.00052784` | `15` | `15` | `16` |

The selector already filters `worst_frames.tsv` to `variant.startswith("candidate_")`, so the all-fail bad-frame result is not caused by mixing historical baseline variants into candidate stats.

## Root-Cause Hypotheses

### Hypothesis 1: v398 over cap is too tight

v398 under support is close to v397, but over support is materially smaller:

| Run | under effective | under new_only | over effective | over new_only |
| --- | ---: | ---: | ---: | ---: |
| v397 `ckpt141160` | `12708` | `11210` | `24513` | `23015` |
| v398 `ckpt141160` | `12730` | `11232` | `19282` | `17784` |

The v398 over new-only cap is the binding cap: `0.38 * 46801 = 17784`. The v392 hard-floor miss is small enough that a controlled increase of over support could recover the floor without returning to v397's unconstrained growth.

### Hypothesis 2: bad-frame gate is over-hard for selection

The v398 gate treats any `hard_delta > 0` and any positive `fg/boundary/edge` count above zero as veto reasons. On this dataset, even the v397 selected checkpoint has local positives. This makes the gate useful as a diagnostic, but too strict as a selection precondition.

The gate should keep severe vetoes, then rank non-severe local regressions as penalties.

### Hypothesis 3: stable window cannot be evaluated until floor and veto logic are separated

`num_stable_window_pass=0` is downstream of `num_v392_safety_floor=0` and `num_bad_frame_gate=0`. Stable-window work should report failure reasons, but should not be interpreted as a training plateau failure until the over-cap sweep and bad-frame severity split are tested.

## v399 Design

### Phase 1: Diagnostic Sweep Without Training-Core Changes

Run two or three v399 diagnostic variants. Keep these unchanged:

- under effective ratio: `0.30`
- under new-only ratio: `0.24`
- learning rates
- loss weights
- top-k ratios
- support-bank keying, EMA, promote interval, min hits/views/frames
- adopted support freeze
- candidate support grow-only semantics

Only sweep over caps:

| Variant | over effective | over new-only | Approx effective cap | Approx new-only cap | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| `v399_cap_mid` | `0.48` | `0.44` | `22464` | `20592` | Smallest controlled relaxation |
| `v399_cap_high` | `0.50` | `0.46` | `23400` | `21528` | Middle point for hard-floor recovery |
| `v399_cap_max` | `0.52` | `0.48` | `24336` | `22464` | Near-v397 support, still below v397 over new-only `23015` |

The first implementation should make the sweep easy to run from a wrapper, but not alter `train.py`.

Acceptance per diagnostic run:

- `adopted_lost=0` for under and over at every checkpoint;
- effective and new-only support stay within the configured caps;
- at least one checkpoint has `fg_delta <= 0`, `boundary_delta <= 0`, `edge_delta <= 0`;
- target checkpoint or selected candidate has `hard_delta <= -0.00023074`;
- bad-frame `max_outer_delta` and `max_hard_delta` do not exceed the v398 `ckpt141160` reference unless the run is explicitly labeled rejected:
  - `max_outer_delta <= 7.0`
  - `max_hard_delta <= 0.00052784`

Decision rule:

- If none of the cap variants recover the v392 hard floor, stop and re-investigate training dynamics before changing selector logic.
- If one or more variants recover the hard floor without worse bad-frame severity, choose the smallest over cap that passes and proceed to Phase 2.

### Phase 2: Bad-Frame Hard Veto Plus Penalty

Keep severe local failures as vetoes:

- `bad_frame_outer_hard_veto`: default `8.0`, optional stricter comparison at `10.0`;
- `bad_frame_hard_hard_veto`: default `0.0005`.

Convert these from vetoes to ranking penalties:

- `hard_delta > 0`;
- `hard_delta > BAD_FRAME_HARD_PENALTY`;
- positive `fg_delta` frame count;
- positive `boundary_delta` frame count;
- positive `edge_delta` frame count.

The selector should produce:

- `bad_frame_hard_veto_pass`
- `bad_frame_penalty`
- `bad_frame_penalty_reasons`
- `bad_frame_max_outer_delta`
- `bad_frame_max_hard_delta`
- positive frame counts
- top repeated bad images by `worsen_score`

Selection pool rule:

1. Keep strict + `fg/boundary/edge <= 0`.
2. Keep support floor: `under_adopted_lost <= 0` and `over_adopted_lost <= 0`.
3. Keep v392 floors.
4. Exclude severe bad-frame veto failures.
5. Rank by stable-window membership, then safety miss, then bad-frame penalty, then inner/hard/outer metrics.

### Phase 3: Diagnostics

Add selector diagnostics that explain why each checkpoint failed.

Per checkpoint:

- `v392_floor_pass`
- `v392_floor_miss_reasons`
- `bad_frame_hard_veto_pass`
- `bad_frame_penalty`
- `support_floor_pass`
- `support_cap_saturated`
- under/over `effective_count`, `new_only`, `adopted_lost`, `jaccard`
- under/over cap limits and cap saturation ratios
- under/over cap blocked totals if available

Aggregate JSON:

- `failure_reason_counts.v392_floor_miss`
- `failure_reason_counts.bad_frame_veto`
- `failure_reason_counts.cap_saturation`
- `failure_reason_counts.fg_boundary_edge_regression`
- `stable_window.num_stable_window_pass`
- `stable_window.target`
- `stable_window.failure_reason_counts`

Bad-frame image aggregation:

- group candidate rows by `image`;
- sum and max `worsen_score`;
- max `outer_delta`;
- max `hard_delta`;
- count positive `fg/boundary/edge`;
- emit top images to `bad_frame_image_summary.tsv` and selected metrics JSON.

### Phase 4: Full v399 Selection

Run the chosen cap with the upgraded selector.

Full acceptance:

- `adopted_lost=0`;
- configured caps are not exceeded;
- at least `3/6` checkpoints satisfy:
  - strict pass;
  - `fg/boundary/edge <= 0`;
  - v392 floor;
  - adopted support floor;
  - severe bad-frame veto pass;
- selected checkpoint path and metrics artifacts are generated;
- selected checkpoint satisfies v392 floor and does not regress `fg/boundary/edge`;
- local bad-frame penalty is reported and not hidden.

## Compatibility

- v392 and v397 checkpoints remain readable because no checkpoint tuple schema change is required.
- Existing support-bank state keys remain optional. New diagnostics must default to zero or `null` when absent.
- The v396 controller remains the common implementation point; v399 should be a wrapper plus selector extensions.
- v398 wrapper behavior remains available for direct comparison.

## Non-Goals

- No lr changes.
- No loss-weight changes.
- No top-k changes.
- No single-frame or single-view hard patch.
- No removal of adopted support.
- No persistent support overwrite.
