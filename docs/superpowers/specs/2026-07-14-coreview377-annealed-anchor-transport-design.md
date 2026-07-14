# CoreView377 Annealed Anchor Transport Design

## Goal

Add a from-zero training path that gradually moves stale canonical bindings toward fresh surface anchors while radiance, non-rigid deformation, and visibility continue optimizing. The first experiment must isolate anchor transport on the proven 80k focus neutral baseline: densify, prune, and opacity reset remain disabled.

## Evidence

- The neutral path's tangent-offset median grows from `0.1546` at 8k to `0.1987` at 64k and `0.2023` at 74k, while the current v395 checkpoint is approximately zero at the median.
- A render-only full fresh rebind improves mean symmetric contour distance from `3.645px` to `3.368px`, but worsens full-image LPIPS from `0.01998` to `0.02713`.
- Disabling the learned forward trunk worsens both appearance and contour, so the trunk is compensating for stale binding rather than causing the shell error.
- The legacy topology run never executed its intended path: clone seeds remained zero, split count remained zero, and clone/prune churn disturbed the image function.

The failure is therefore a coupled optimization problem. Anchor correction must be gradual and must happen during training.

## Considered Approaches

### Full periodic rebind

Recompute every anchor at fixed intervals. This directly reduces tangent offset but reproduces the render-only quality collapse and is rejected.

### Persistent old/new dual binding state

Store complete source and target bindings and render both paths during a fixed transition window. This is the most explicit formulation, but it adds many pointwise checkpoint fields and complicates pruning/densification compatibility before the hypothesis is validated.

### Bounded EMA anchor transport

At a fixed interval, rank stale points by tangent-offset magnitude, select a bounded subset, compute fresh bindings through the existing semantic anchor search, and blend continuous binding fields toward the fresh result with a small alpha. Repeat selections gradually move the worst anchors while training adapts the remaining model state.

This is the selected approach because it is checkpoint-compatible, bounded, testable, and reuses the current partial-refresh boundary.

## Runtime Behavior

The rigid deformer receives an opt-in configuration block:

- `anchor_transport_enable`
- `anchor_transport_start_iter` and `anchor_transport_end_iter`
- `anchor_transport_interval`
- `anchor_transport_tangent_threshold`
- `anchor_transport_max_points` and `anchor_transport_max_ratio`
- `anchor_transport_blend_alpha`
- `anchor_transport_log_interval`

On an eligible training iteration:

1. Rebuild derived binding fields from the current canonical points and stored anchor state.
2. Select points whose tangent offset exceeds the threshold.
3. Keep only the highest-offset bounded subset.
4. Mark that subset for the existing partial anchor search.
5. Blend `anchor_xyz`, `anchor_weights`, `anchor_normal`, `semantic_score`, and `semantic_distance` from old to fresh values.
6. Recompute dominant joint, confidence, region label, local offset, normal offset, tangent offset, and layer/region weights.
7. Persist the blended binding state on the canonical Gaussian owner.

Scheduling must occur at most once per global iteration even when the converter is invoked more than once.

## Experiment

Use the exact neutral long-horizon stack that produced `FG_LPIPS=0.1304382`, with:

- 80k `focus_head_hands` initialization
- seed `20260710`
- densify disabled
- opacity reset disabled
- no carrier competition, reallocation, compression, or tether loss
- 8k canary gate before any long continuation

Initial transport schedule:

- start: 500
- end: 8000
- interval: 100
- tangent threshold: 0.08
- max points per event: 512
- max ratio per event: 0.0064
- blend alpha: 0.10

## Success Criteria

At 8k, all of the following are required:

- `FG_LPIPS <= 0.1555340652`
- no single evaluation regresses more than `0.01` from the previous evaluation
- global tangent-offset median and p90 are below the neutral 8k values `0.154572` and `0.332173`
- transport logs show nonzero candidates, selected points, anchor shifts, and reduced selected-point tangent offset
- no densify, prune, or opacity reset event occurs

If the image gate passes and tangent offset improves, continue the same uninterrupted run. If tangent offset improves but LPIPS fails, the next repair is a persistent dual-binding transition rather than stronger EMA or a relaxed gate.

## Failure Handling

- Transport is disabled by default and cannot alter existing recipes.
- Invalid alpha, interval, point caps, or thresholds fail early through configuration tests.
- Empty candidate sets are logged but are not errors.
- Any non-finite blended binding value aborts transport for that event and keeps the previous state.
- The existing image metric gate remains the final protection against an apparently healthy state metric with degraded rendering.
