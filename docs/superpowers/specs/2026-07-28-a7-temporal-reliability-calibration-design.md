# A7 Temporal Reliability Calibration Design

## Goal

Develop A7 as a temporally reliable extension of the frozen A5 semantic editing method, improving continuous-pose screen-space stability without sacrificing A5's five-subject spatial leakage, coverage, semantic accuracy, or reconstruction results.

## Method Position

A5 remains the frozen spatial primary baseline:

```text
A5 = voting posterior + footprint evidence calibration
```

A7 is defined as:

```text
A7 = A5 + visibility-aware temporal reliability calibration
     + target-preserving stable-carrier redistribution
```

A7 produces one static canonical `soft_edit_weights` bank. It does not predict per-frame weights, does not use previous rendered frames at runtime, and does not retrain the Avatar backbone. This preserves the persistent semantic asset narrative and adds no state, latency, or video-filtering artifacts during inference.

## Why A7 Is Needed

The current five-subject diagnostic shows that A5 is worse than Voting in both fixed-strength and adaptive matched-retention temporal flicker. Adaptive-minus-fixed differences are near zero, so per-frame strength compensation is not the cause. The likely source is a small set of high-impact canonical Gaussians whose projected footprint, visibility, and boundary role change sharply under pose deformation.

Therefore A7 must calibrate canonical carrier reliability across poses rather than smooth output videos or edit strength.

## Rejected Alternatives

### Pose-Conditioned Per-Frame Network

A pose-conditioned weight residual may reduce flicker, but it adds training, runtime state, subject overfitting risk, and weakens the claim that semantics are a persistent canonical asset. It is reserved as A7b only if static A7 fails validation gates.

### Screen-Space or Strength Smoothing

Output smoothing can create lag and ghosting, and current diagnostics already show that strength compensation does not explain the regression. It is not an acceptable main-conference method contribution.

## Data Protocol

Do not use the already observed c21 temporal test sequence to select A7 design or hyperparameters.

Use the following fixed protocol for each subject:

```text
temporal evidence cameras = c01, c05, c09, c13
temporal evidence frames  = 0:570:5
validation cameras        = c17-c20
validation frames         = 0:570:5
retrospective test        = c21, frames 0:570:1
additional frozen test    = c22-c23, frames 0:570:1 or a predeclared resource-limited stride
parts                     = hair, face, upper, lower, shoes, skin
```

The four evidence cameras are fixed before experiments and span the calibration camera range. Frame stride five keeps evidence construction tractable while covering the full pose trajectory.

Current five subjects are retrospective evidence because their discrete test results have been inspected. A strong main-conference claim requires at least one newly acquired or previously untouched subject whose protocol and A7 configuration are frozen before any test metrics are opened.

## Temporal Evidence

Reuse the existing per-frame projected footprint evidence machinery. For every canonical Gaussian and semantic part, accumulate:

```text
temporal_visible_count
temporal_consecutive_visible_count
temporal_target_ratio_mean
temporal_target_ratio_std
temporal_target_flicker
temporal_outer_ratio_mean
temporal_outer_ratio_std
temporal_outer_flicker
temporal_boundary_crossing_rate
temporal_visibility_transition_rate
temporal_reliability
```

Target and outer statistics operate on footprint-normalized ratios, not raw screen area. Flicker uses only consecutive pairs where the Gaussian is visible in both frames. Visibility transitions are reported separately and are not treated as semantic instability by default.

Boundary crossing is the rate at which a Gaussian changes between target-dominant, allowed-boundary, and outer-dominant footprint states across consecutive visible frames.

## Reliability Factor

For Gaussian `i` and part `p`, define a static reliability factor:

```text
R_ip = support_gate
       * exp(-lambda_outer * temporal_outer_flicker)
       * exp(-lambda_boundary * temporal_boundary_crossing_rate)
       * exp(-lambda_target * temporal_target_flicker)
```

`support_gate` is zero or strongly attenuated when consecutive visible support is below the fixed minimum. All factors are clipped to `[0, 1]`.

The initial damped weight is:

```text
w_damped = w_A5 * R
```

## Target-Preserving Stable-Carrier Redistribution

Pure damping would repeat the existing coverage failure. A7 therefore restores a fixed fraction of A5's validation target response using stable target carriers.

For each part:

1. Estimate the A5 target-response mass using temporal target evidence.
2. Dampen unstable A5 weights with `R`.
3. Compute the deficit relative to `rho * A5_target_mass`.
4. Redistribute the deficit to candidates ranked by voting posterior, mean target ratio, temporal reliability, and support.
5. Cap each candidate by its voting-posterior-derived weight ceiling and `[0, 1]`.
6. Stop when the target floor is reached or no eligible stable carrier remains.

This is deterministic water-filling, not a learned per-frame predictor. The bank records redistributed count, restored target mass, remaining deficit, and cap saturation per part.

## Candidate Hyperparameters

Keep the first formal grid small:

```text
lambda_outer    = 0.25, 0.50, 1.00
lambda_boundary = 0.25, 0.50
lambda_target   = 0.00, 0.25
rho             = 0.90, 0.95
min_pair_support = fixed protocol constant
```

Use evidence-only proxy scores to reject clearly invalid candidates before GPU validation. Render at most the top four candidates per donor subject on c17-c20 validation sequences.

## LOSO Selection

For held-out subject `s`, select one A7 configuration using only c17-c20 validation reports from the other four subjects.

A candidate is eligible only if every donor subject satisfies:

```text
formal eligible parts do not decrease relative to A5;
matched target coverage does not decrease by more than 2 percentage points;
pooled outer burden <= 1.02 * A5;
pooled boundary burden <= 1.02 * A5;
macro mIoU decrease <= 0.01;
micro IoU decrease <= 0.005.
```

Among eligible candidates, minimize subject-equal fixed-strength outer and boundary flicker. Resolve ties by lower spatial burden, then smaller deviation from A5 weights, then lower regularization strength.

No part-specific, frame-specific, camera-specific, or held-out-subject-specific fallback is allowed.

## Evaluation

Compare Voting, A5, and A7 using the same renderer and common support.

### Temporal Metrics

```text
fixed-strength outer flicker
fixed-strength boundary flicker
adaptive matched-retention flicker
visibility-aware consecutive response flicker
warp-aligned edited-mask IoU when correspondence is available
selection area and centroid variation
visibility transition rate
```

### Spatial and Coverage Metrics

```text
pooled outer burden
pooled boundary burden
pooled selection leakage
per-subject/per-part coverage
every-subject 80% formal eligibility
recolor/removal/texture coverage-constrained burden
```

### Semantic and Reconstruction Guards

```text
macro mIoU
micro IoU
Boundary F1 / Boundary IoU
soft IoU
PSNR / SSIM / LPIPS
bank size, construction time, validation time, runtime overhead
```

## Ablations

```text
A5    = frozen spatial method
A7-V  = visibility normalization/support gate only
A7-T  = temporal reliability damping without redistribution
A7-R  = redistribution with target/outer mean evidence but no flicker penalty
A7    = full temporal reliability + redistribution
```

The ablation must show whether improvement comes from excluding visibility transitions, penalizing unstable carriers, or restoring target mass through stable redistribution.

## Promotion Gates

A7 replaces A5 as the paper main method only if validation-frozen five-subject evaluation satisfies all of the following:

```text
fixed outer flicker: statistically significant A7-A5 reduction;
fixed boundary flicker: statistically significant A7-A5 reduction;
at least 4/5 retrospective subjects improve on both fixed metrics;
formal eligible parts do not decrease;
real-edit pooled burden degrades by no more than 2% relative;
macro mIoU degrades by no more than 0.01;
micro IoU degrades by no more than 0.005;
reconstruction changes remain at floating-point/no-practical-effect scale.
```

For the untouched confirmation subject, A7 must reduce both fixed flicker metrics without violating coverage or spatial burden guards.

If temporal improvement requires lower coverage, lower target retention, or test-specific tuning, A7 is rejected and A5 remains the main method with temporal limitations reported honestly.

## Staged Execution

### Stage 0: Metric and Evidence Canary

Use CoreView_377 calibration/validation only. Verify per-Gaussian temporal evidence, visibility handling, deterministic redistribution, and finite outputs on a short frame range.

### Stage 1: One-Subject Validation Canary

Run full evidence stride on 377, generate the small candidate grid, evaluate top candidates on c17-c20, and require spatial/coverage gates before any temporal claim.

### Stage 2: Five-Subject Evidence and LOSO

Build evidence for 377/386/387/393/394, generate candidates, select one LOSO configuration per held-out subject, and freeze all fingerprints.

### Stage 3: Retrospective c21 Evaluation

Evaluate Voting/A5/A7 on existing c21 continuous sequences. This stage diagnoses effectiveness but is not untouched confirmation.

### Stage 4: Frozen c22-c23 and New-Subject Confirmation

Without changing A7, evaluate additional continuous cameras and at least one untouched subject. Only after this stage may A7 be promoted in the main paper claim.

### Stage 5: Full Paper Tables

Regenerate semantic main tables, coverage-constrained real editing, temporal tables, ablations, complexity, videos, and failure cases with A7 included.

## Code Boundaries

Preferred new modules:

```text
utils/temporal_reliability_calibration.py
tools/build_temporal_reliability_evidence.py
tools/calibrate_temporal_reliable_a7_weights.py
tools/select_loso_a7_temporal_config.py
tools/run_a7_temporal_reliability_queue.sh
configs/semantic/frozen_a7_temporal_reliable_v1.json
```

Expected integrations:

```text
utils/part_label_bank.py
tools/render_semantic_temporal_stability.py
tools/evaluate_semantic_editing_paper_protocol.py
tools/render_semantic_real_editing_paper_suite.py
```

All new bank fields require schema validation, source fingerprints, protocol provenance, direct CLI smoke tests, and unit tests.

## A7b Stop/Upgrade Branch

Do not start a pose-conditioned network automatically. Consider A7b only when:

```text
static A7 passes spatial and coverage gates;
static A7 still fails to reduce fixed flicker on c17-c20 validation;
per-Gaussian evidence shows stable carrier capacity is insufficient;
the team accepts retraining and a new untouched-test requirement.
```

A7b requires a separate design and cannot reuse A7 test results for model selection.

## Paper Claim If Successful

The intended contribution becomes:

```text
We extend evidence-calibrated persistent semantic assets with
visibility-aware temporal reliability calibration, rebalancing canonical
edit carriers to reduce pose-induced screen-space flicker while preserving
matched-retention spatial leakage and coverage.
```

This is materially stronger than post-processing because the improved stability is encoded in the reusable canonical Gaussian asset.
