# A7c R1.2-B Dense Overlap-Set Compositor Design

## Status And Scope

This document preregisters the CoreView377 R1.2-B development canary before
implementation or any new held-block result is opened. R1.2-B tests whether the
remaining R1.2-A failure is caused by the independent per-carrier scorer lacking
explicit screen-space carrier relations.

R1.2-B changes the predictor only. It preserves the R1.2-A source artifacts,
runtime target-budget projection, output-space objective, six-fold held-block
protocol, promotion thresholds, and camera leakage boundary. It cannot authorize
Task 12 or a paper claim by itself.

## Frozen Evidence And Root-Cause Hypothesis

R1.2-A preserved target and selection quality but failed temporal promotion:

```text
outer mean gain              +0.480228%   required >= +0.500000%
outer positive blocks        20/24        required >= 90%
outer q10                    -0.172544%   required >= 0
outer worst block            -0.644254%   required >= -0.500000%
boundary mean gain           +0.295324%   required >= +0.500000%
maximum adjacent gate jump    0.078432    required <= 0.02
```

The maximum raw-gate jump is at least as large as the projected-gate jump in 23
of 24 camera-block records. The target-budget projection usually attenuates the
jump and is therefore not the primary instability source. The R1.2-A F1 MLP
scores carriers independently even though outer and boundary contributions are
created by screen-space overlap and depth competition between carriers.

The runtime probe contains all relational inputs needed without label leakage:
projected `camera_x_over_z` and `camera_y_over_z`, `log_depth`, visibility,
radius, and the complete F1 node descriptor. Among visible carriers, the median
nearest-neighbor projected distance is `0.00652`. A projected radius of `0.03`
contains approximately 11 neighbors on average and is frozen before training.

## Alternatives Considered

### Dense continuous overlap set, selected

A dense soft graph is permutation equivariant, differentiable, stateless, and
has no discrete neighborhood switches. With 85 carriers its quadratic cost is
small enough for the canary. It directly tests the missing-relation hypothesis.

### Sparse k-nearest-neighbor graph, rejected

Sparse kNN is cheaper, but neighbor membership changes discontinuously with
pose. That is poorly aligned with the failed adjacent-gate guard and adds an
unnecessary topology hyperparameter.

### Previous-frame recurrent smoothing, rejected

A recurrent model could suppress jumps directly, but it adds runtime state,
sequence-start dependence, and a stronger overfitting path. It would no longer
isolate whether renderer-visible current-frame relations are sufficient.

## Inputs And Leakage Boundary

The predictor reads only the frozen R1.1 F1 per-carrier features. The relation
builder extracts these runtime-safe fields from F1:

```text
visibility
camera_x_over_z
camera_y_over_z
log_depth
log1p_radius (available to the node encoder, not used to set graph labels)
```

It does not read teacher gates, renderer target/outer/boundary contributions,
selection evidence, masks, camera ID, frame ID, subject ID, Gaussian ID, image
name, or previous-frame state. Renderer contribution arrays remain supervision
and audit data only.

Feature normalization is fit independently on each fold's training samples.
Held blocks cannot contribute normalization statistics. Relation geometry uses
the raw dimensionless projected coordinates and log depth with the fixed scales
below, so it does not require held-derived statistics.

## Continuous Overlap Graph

For sample `s` and carriers `i,j`, define projected position `u`, log depth `z`,
and binary renderer visibility `v`. Self edges are excluded. The frozen edge is:

```text
spatial_scale = 0.03
depth_scale = 0.04

log_w_ij = -0.5 * ||u_i - u_j||^2 / spatial_scale^2
           -0.5 * (z_i - z_j)^2 / depth_scale^2

w_ij = v_i * v_j * exp(clamp(log_w_ij, min=-20, max=0))
a_ij = w_ij / max(sum_j w_ij, epsilon)
```

Rows with no visible neighbor produce a zero message. This graph is continuous
in the runtime geometry, symmetric before row normalization, and invariant to a
permutation applied consistently to carrier features.

## Predictor Architecture

The network is deliberately small and shared across all carriers:

```text
F1 node descriptor (30)
  -> Linear(30, 32) -> SiLU -> Linear(32, 32) -> SiLU = h_i

neighbor message m_i = sum_j a_ij h_j
visible global g = sum_j v_j h_j / max(sum_j v_j, 1)

concat(h_i, m_i, h_i - m_i, g) (128)
  -> Linear(128, 32) -> SiLU -> Linear(32, 1) -> sigmoid
  -> raw gate in [0.9, 1.0]
```

The final layer is initialized to emit `0.999`, matching R1.2-A. There is no
carrier embedding or identity table. The implementation must pass a permutation
equivariance test and a no-neighbor finite-output test.

## Projection And Training Objective

R1.2-B applies the unchanged R1.2-A joint target-budget projection after the raw
set prediction. It uses only alpha/transmittance mass, frozen A5 lower weight,
semantic support, and alpha mean. The topology floor, gate range, and proxy target
response remain unchanged.

All optimizer and objective values are copied from the frozen R1.2-A contract:

```text
epochs = 400
learning rate = 0.001
weight decay = 0.0001
outer/boundary weights = 1/1
target hinge weight = 100
soft-IoU hinge weight = 100
adjacent gate-jump hinge weight = 20
damping regularizer weight = 0.001
```

No teacher-gate loss, checkpoint selection, architecture sweep, graph-scale
sweep, seed sweep, or post-result threshold change is permitted. Retaining the
same objective isolates the effect of adding carrier relations.

## Split, Audit, And Promotion

Six leave-one-contiguous-block-out models train only on c01/c05/c09/c13. Each
held block is evaluated separately for the four fit cameras, yielding exactly
24 records. A final model trains on all fit-camera blocks only after the fold
configuration is frozen.

Every R1.2-A guard remains mandatory:

```text
mean outer gain >= 0.5%
mean boundary gain >= 0.5%
outer and boundary positive-block fraction >= 0.90
outer and boundary q10 block gain >= 0
worst outer and boundary block regression <= 0.5%
minimum target response >= 0.99
maximum soft-IoU drop <= 0.005
maximum adjacent projected gate change <= 0.02
gate range [0.9, 1.0]
A5 topology, coverage, frozen parts, and weight upper bounds unchanged
```

R1.2-B must also improve both mean outer and mean boundary gain over the frozen
R1.1-F1 result (`-0.012761%` outer, `+2.348187%` boundary). R1.2-A is reported as
an ablation comparator but does not replace the R1.1 boundary floor.

## Stop Rules

- A fingerprint, manifest, sample order, carrier order, or split mismatch stops
  before training.
- A relation builder that reads renderer labels, IDs, or previous-frame state is
  invalid.
- Any held-block gate failure writes only `.rejected` and stops before c17-c20.
- Failure cannot be repaired in place by changing graph scales, width, loss
  weights, epochs, learning rate, feature group, margins, or thresholds.
- A pass only authorizes a separately frozen c17-c20 audit. It does not authorize
  c21-c23, Task 12, LOSO, or a paper claim.
- If R1.2-B preserves target but still fails outer consistency or gate jump, the
  stateless local-relation family is rejected; the next discussion must compare
  explicit pose-conditioned A7b against a recurrent temporal model rather than
  expanding this canary.

## Artifacts And Verification

The implementation will create a frozen R1.2-B JSON contract, overlap-set model
utility, focused tests, trainer, held-block auditor, and resumable runner. Formal
outputs must contain seven model/prediction/summary triplets, a 24-record audit,
source fingerprints, mutually exclusive terminal markers, and
`paper_test_eligible=false`.

Tests must cover graph masking, row normalization, zero-neighbor behavior,
permutation equivariance, bounded initialization, fit-only normalization,
unchanged target projection, no teacher-gate access, deterministic CPU training,
24-record promotion aggregation, camera isolation, restart behavior, and marker
mutual exclusion.
