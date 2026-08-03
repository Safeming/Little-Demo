# A7c R1.1 Transmittance And Ray-Context Probe Design

## Status And Scope

This document preregisters the CoreView377 R1.1 feature-observability experiment
before any new transmittance or ray-context result is opened. R1.1 diagnoses
whether runtime-observable renderer context can predict carrier-resolved oracle
actions across held contiguous pose blocks. It is not a paper test, does not
authorize Task12, and cannot use c21-c23.

The source avatar checkpoint, A5 bank, R0 teacher, carrier IDs, camera split,
frame range, temporal blocks, action range, and promotion thresholds remain
frozen from the preceding A7c canary.

## Root Cause Being Tested

The first carrier MLP used 12 independent per-carrier features. On six held
blocks it obtained negative teacher R2 and regressed outer flicker while
improving boundary flicker. Marginal feature OOD was negligible. A stronger
non-parametric regressor improved R2 only modestly. R1.1 tests the specific
hypothesis that the missing information is alpha/transmittance-weighted
visibility and relational ray context, rather than another camera/frame lookup
or a wider version of the same independent-carrier MLP.

## Exact Transmittance-Weighted Carrier Signal

For carrier color `c_i`, the existing rasterizer computes each pixel as:

```text
C_p = sum_i c_i * alpha_i,p * T_i,p + background * T_final,p
```

With independent precomputed carrier colors and scalar loss
`L = sum_p C_p`, autograd gives:

```text
dL/dc_i = sum_p alpha_i,p * T_i,p
```

R1.1 records this gradient as `alpha_transmittance_mass`. This is the true
front-to-back transmittance-weighted carrier contribution accumulated by the
production CUDA backward kernel. It is not `opacity * radius^2`, binary
visibility, or another geometric proxy.

Before formal collection, a finite-difference test on a small render must match
the color gradient within preregistered numerical tolerance. A zero-opacity or
fully invisible carrier must have zero mass. Failure stops R1.1.

## Ray-Context Buffers

The same frozen geometry, opacity, covariance, camera, and rasterizer render the
following buffers with `colors_precomp`:

```text
alpha               = render(1)
depth_numerator      = render(normalized camera depth)
depth2_numerator     = render(normalized camera depth squared)
lower_support        = render(A5 lower soft weight)
semantic_confidence  = render(A5 lower semantic probability)
```

For every selected carrier, its projected center and footprint define a fixed
sampling stencil. R1.1 samples center, footprint mean, and footprint variance
from each buffer. Derived runtime values are:

- expected ray depth and depth variance;
- carrier depth minus expected ray depth;
- local accumulated alpha;
- local A5 lower support and semantic confidence;
- carrier alpha-transmittance mass normalized by accumulated alpha mass;
- continuous projected radius and center coordinates.

All depth values use fit-camera-only normalization. Empty-ray ratios use a
fixed epsilon and an availability bit. No ground-truth mask enters a buffer.

## Static Carrier Descriptors

R1.1 also restores descriptors intended by the original compositor design but
omitted from the first probe:

- body-scale-normalized canonical XYZ;
- log Gaussian scaling and rotation/covariance invariants already stored by the
  frozen checkpoint;
- opacity;
- A5 semantic probabilities, semantic margin, and lower soft weight;
- compact skinning weights and dominant joint when exposed by the frozen
  Gaussian model, with an availability bit.

Gaussian ID is not a feature. Static descriptors must be produced by a shared
schema that can be constructed for every later subject.

## Forbidden Inputs

Feature tensors may not contain camera ID/index, frame ID/index, subject ID,
Gaussian ID, image name, previous-frame state, ground-truth masks,
target/outer/boundary contribution labels, oracle gates, or c21-c23 data.

Camera and frame indices exist only in manifests and split masks. R0 teacher
gates and renderer contribution evidence are supervision/evaluation targets,
never runtime inputs.

## Frozen Feature Ablation

All variants use the same carrier set, samples, splits, normalization rules,
training budget, and evaluator:

```text
F0 = original 12-feature runtime probe
F1 = F0 + static carrier descriptors
F2 = F1 + exact alpha_transmittance_mass
F3 = F2 + sampled alpha/depth/depth2/lower-support/semantic ray context
```

The primary comparison is F3 versus F0. Feature groups cannot be rearranged or
selected after held-block metrics are opened.

## Predictor And Teacher Boundary

R1.1 is a feature-observability experiment, not final method selection. Each
feature set uses the same fixed smooth shared predictor:

```text
input -> 64 -> 32 -> 1, SiLU, gate in [0.9, 1.0]
no embeddings, recurrence, free carrier table, camera table, or frame table
```

The R0 point oracle remains a common teacher so the ablation changes only
features. Teacher R2, MAE, and rank correlation are diagnostics. Promotion is
decided by renderer outer/boundary and hard guards, not by a chosen R2 cutoff.
The teacher remains an oracle upper bound and is not called a generalized
target.

## Split And Leakage Control

```text
fit cameras: c01, c05, c09, c13
unopened audit cameras: c17, c18, c19, c20
forbidden cameras: c21, c22, c23
frames: 0:570:5
temporal folds: six contiguous leave-one-block-out folds per fit camera
```

Feature normalization, optimizer gradients, checkpoint selection, and all
feature decisions use fit-camera training blocks only. Held-block values are
opened once after all four variants finish. Audit-camera metrics remain closed
unless F3 passes the held-block gate.

## Held-Block Promotion Gate

F3 must satisfy every existing canary guard:

```text
mean outer flicker gain >= 0.5%
mean boundary flicker gain >= 0.5%
outer positive-block fraction >= 0.90
boundary positive-block fraction >= 0.90
outer/boundary q10 block gain >= 0
worst outer/boundary block regression <= 0.5%
minimum unweighted target response >= 0.99
maximum unweighted soft-IoU drop <= 0.005
maximum adjacent carrier-gate change <= 0.02
gate range [0.9, 1.0]
A5 topology, coverage, frozen parts, and weight upper bounds unchanged
```

F3 must additionally improve both mean outer and mean boundary gain over F0.
No requirement may be relaxed based on F1/F2/F3 results.

## Stop Rules

- Finite-difference transmittance validation failure stops collection.
- Schema, carrier alignment, camera/frame manifest, or fingerprint mismatch
  stops training.
- F3 held-block rejection stops before c17-c20.
- A failed R1.1 cannot be repaired in place by adding IDs, time features,
  recurrence, extra feature groups, wider networks, threshold changes, or
  opening c21-c23.
- A passing R1.1 only authorizes the already preregistered target-preserving
  compositor canary with held-camera audit; it does not authorize Task12 or a
  paper claim.

## Artifacts And Verification

The formal output stores one common probe manifest, group-specific tensors,
one artifact per fold/feature set, aggregate comparisons, complete source
fingerprints, commands, git commit, and `paper_test_eligible=false`.

Focused tests cover gradient-versus-finite-difference agreement, invisible
mass, buffer ratio limiting cases, footprint sampling, static schema alignment,
forbidden-field rejection, fit-only normalization, carrier/sample order,
contiguous splits, ablation nesting, gate bounds, and promotion aggregation.
