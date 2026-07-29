# A7 Rendered Objective And Bounded Retention Design

## Goal

Build an A7 v3 CoreView377 canary that directly screens static semantic banks
against part-level renderer-aligned temporal sequences while preventing the large
soft-weight redistribution that caused A7 v2 spatial leakage.

## Context

A7 v2 made three changes that must remain frozen: renderer-aligned target/outer/
boundary evidence, pre-freezing parts with inadequate A5 coverage, and exact A5
selection topology preservation at threshold 0.2. The completed v2 validation
showed that these changes reduced absolute outer response and boundary flicker,
but normalized outer flicker increased because mean outer response fell faster
than adjacent variation. Greedy target-mass restoration also changed hair/lower
weights by L1 89.30 and increased matched-retention spatial outer burden by more
than 6% in aggregate.

## Considered Approaches

1. **Full renderer evaluation for every candidate.** This is closest to the final
   metric but repeats the expensive rasterizer for each bank and turns candidate
   generation into another validation-scale render queue.
2. **Keep the v2 aggregate proxy and only add stronger regularization.** This is
   cheap but retains the verified mismatch between per-Gaussian proxy burden and
   the final part-level normalized temporal signal.
3. **Store renderer contribution sequences and screen bounded candidates with a
   part-level linearized rendered objective.** This reuses one evidence render,
   evaluates the final signal form before validation, and keeps the change scoped.

Approach 3 is selected. It is the smallest route that tests the identified root
cause without opening new validation/test cameras or expanding the old 24-point
lambda grid.

## Evidence Schema

The renderer evidence builder will retain the existing aggregate arrays and add:

```text
renderer_target_contribution_sequence  [S, N, C] float16
renderer_outer_contribution_sequence   [S, N, C] float16
renderer_boundary_contribution_sequence [S, N, C] float16
renderer_sequence_camera_index         [S] int16
renderer_sequence_frame_index          [S] int32
```

`S=456`, `N=2120`, and `C=6` for the registered CoreView377 canary. Samples stay in
camera-major, frame-major order. Adjacent differences reset at camera boundaries.
The sequence is calibration evidence from c01/c05/c09/c13 only; c17-c20 remain
validation and c21-c23 remain unopened.

## Candidate Policies

The v3 contract contains exactly two fixed policies with common reliability
parameters `lambda_outer=1.0`, `lambda_boundary=0.5`, `lambda_target=0.25`, and
`min_pair_support=8`:

```text
bounded_damping_005:
  minimum_weight_ratio_from_a5 = 0.95
  target_mass_floor = 0.95
  restore_target_mass = false

bounded_retention_010:
  minimum_weight_ratio_from_a5 = 0.90
  target_mass_floor = 0.95
  restore_target_mass = true
  maximum_part_weight_l1_from_a5 = 12.0
```

For both policies, output weights may never exceed A5 weights. Selected A5
carriers remain at or above 0.2 and unselected carriers remain below 0.2. Frozen
`face/upper/shoes/skin` columns remain bitwise identical to A5.

The retention policy first applies bounded reliability damping, then restores
target evidence by returning the most reliable supported A5 carriers toward their
original A5 weights. It never creates mass above A5 and never introduces a new
carrier. If the target floor cannot be met inside the L1 bound, the candidate is
invalid rather than silently relaxing the contract.

## Rendered Sequence Objective

For a candidate weight column `w_p`, each evidence sample produces:

```text
target_s   = sum_n w[n,p] * target_contribution[s,n,p]
outer_s    = sum_n w[n,p] * outer_contribution[s,n,p]
boundary_s = sum_n w[n,p] * boundary_contribution[s,n,p]
```

For each camera and processed part, compute mean response, mean adjacent absolute
change, and normalized flicker `adjacent / mean`. Aggregate cameras equally and
then aggregate processed parts equally. The candidate summary records A7/A5
ratios for target response, absolute adjacent change, normalized outer flicker,
and normalized boundary flicker.

Evidence-screen eligibility requires:

```text
target response ratio >= 0.95 for hair and lower
outer mean response ratio <= 1.0
outer adjacent absolute change ratio <= 1.0
boundary adjacent absolute change ratio <= 1.0
selection topology crossings = 0
frozen columns bitwise equal A5
all weights <= A5
part L1 change <= configured bound
```

Normalized flicker ratios are ranking signals, not construction-time hard gates,
because a candidate that strongly reduces absolute leakage can still expose the
same denominator effect observed in v2. Both fixed policies are validated so the
experiment can isolate damping from bounded restoration.

## Validation And Promotion

Run both candidates on c17-c20 with frame range `0:570:5`, followed by the existing
eight-record spatial evaluator. The final promotion gate is unchanged:

```text
fixed outer flicker decreases
fixed boundary flicker decreases
formal eligible parts do not decrease
six-part-equal spatial outer and boundary burden worsen by at most 2%
macro mIoU drop <= 0.01
micro IoU drop <= 0.005
```

Failure stops at CoreView377. Do not enter Task12, open c21-c23, or add another
hyperparameter sweep.

## Testing

Unit tests cover sequence shape/order, camera-boundary adjacency reset, direct
part-level objective arithmetic, bounded weight invariants, target-floor failure,
contract validation, deterministic candidate generation, and runner dry-run.
Existing A7 v1/v1.1/v2 behavior remains unchanged.

