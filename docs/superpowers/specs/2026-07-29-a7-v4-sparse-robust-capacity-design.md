# A7 V4 Sparse Robust Capacity Design

## Goal

Replace A7 v3's near-uniform weight scaling with a sparse, camera-robust direct
optimizer over the existing renderer contribution sequences, then run one frozen
CoreView377 validation canary before any five-subject expansion.

## Evidence And Scope

V4 reuses the frozen v3 evidence artifact from c01/c05/c09/c13. It does not render
new calibration data and does not read c17-c20 while optimizing. Parts
`face/upper/shoes/skin` remain bitwise frozen. Only A5-selected, supported hair and
lower carriers may be reduced; no weight may exceed A5 and threshold-0.2 topology
must remain exact.

The v3 evidence fingerprint is
`17142db0063bb63b84b8f0a777e9ca9ec21c1d4084608aa791f79e8e595ab965`.

## Considered Approaches

1. **Dense continuous optimization with SLSQP.** Flexible, but likely to reproduce
   v3's dense soft-IoU loss and harder to audit deterministically.
2. **Sparse greedy coordinate optimization against the direct rendered sequence
   objective.** Each accepted move has a measurable camera-level effect, output is
   deterministic, and the changed carrier set stays small.
3. **Pose/view-conditioned A7b.** Potentially stronger, but it changes the static
   bank architecture and is premature while a feasible static sparse solution
   exists.

Approach 2 is selected.

## Optimizer

For each processed part, start from A5. Eligible carriers satisfy A5 weight >= 0.2
and evidence pair support >= 8. At each step evaluate one unused carrier at two
bounded actions: reduce it by 5% or 10% of its A5 weight. Accept the action that
most reduces the robust objective while keeping every optimization camera's target
mean response >= 98% of A5.

The maximum changed count is `floor(0.20 * selected_A5_count)`, yielding at most 15
hair and 25 lower carriers on CoreView377.

For each camera, compute A7/A5 ratios for normalized outer flicker, normalized
boundary flicker, absolute outer adjacent change, and absolute boundary adjacent
change. The score is:

```text
max_camera(max(outer_flicker_ratio, boundary_flicker_ratio))
+ 0.25 * mean_camera(outer_flicker_ratio + boundary_flicker_ratio)
+ 0.05 * mean_camera(outer_adjacent_ratio + boundary_adjacent_ratio)
```

Stop when no feasible action improves the score.

## Leave-One-Camera-Out Capacity Gate

Run four folds. Each fold optimizes hair and lower on three evidence cameras and
evaluates the held-out fourth camera. A fold passes only when:

```text
hair/lower part-equal held-out outer normalized flicker ratio < 1.0
hair/lower part-equal held-out boundary normalized flicker ratio < 1.0
held-out target mean ratio >= 0.98 for each processed part
```

All four folds must pass. The final candidate is then optimized once on all four
evidence cameras. Its per-part, per-camera outer and boundary normalized ratios
must each be below 1.0.

The pre-implementation capacity audit produced:

```text
hair final worst outer ratio = 0.993954
hair final worst boundary ratio = 0.992433
hair changed carriers = 15
lower final worst outer ratio = 0.984063
lower final worst boundary ratio = 0.984074
lower changed carriers = 25
four LOCO part-equal held-out folds = all pass
```

## Formal Validation Gates

Run the single frozen candidate on c17-c20, frames `0:570:5`, followed by the
existing spatial evaluator. Promotion requires all of the following:

```text
hair/lower part-equal fixed outer flicker decreases by at least 0.5%
hair/lower part-equal fixed boundary flicker decreases by at least 0.5%
each of c17/c18/c19/c20 has lower part-equal outer and boundary flicker
coverage remains identical to A5
six-part-equal spatial outer and boundary burden worsen by at most 2%
hair and lower soft IoU absolute drop is at most 0.01 per part
macro mIoU drop <= 0.01
micro IoU drop <= 0.005
frozen columns and selection topology remain exact
```

Failure stops v4. Do not open c21-c23, enter Task12, expand the action set, or
start A7b automatically.

## Outputs

Use independent paths under
`exp/acceptdata/a7_sparse_robust_v4_canary_377/`. Candidate summaries record every
accepted coordinate move, all fold metrics, final evidence metrics, fingerprints,
and formal gate results.

