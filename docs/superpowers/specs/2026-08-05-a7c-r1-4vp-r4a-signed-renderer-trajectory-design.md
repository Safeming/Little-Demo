# A7c R1.4-VP-R4-A Signed Renderer-Trajectory Distillation Design

## Status And Question

R4-A is a single-variable CoreView377 fit-first canary. It asks whether the
unchanged R3 view-pose recurrent compositor can recover renderer behavior when
the loss directly matches signed, mean-normalized outer and boundary temporal
trajectories instead of reducing per-gate Huber errors with absolute
contribution weights.

R4-A changes only the training objective. It does not change runtime features,
pose representation, overlap graph, model architecture, optimizer, epoch
count, R1.2-B starting gates, teacher, camera/block split, hard projection, or
held promotion thresholds. It does not authorize attention, carrier identity
embeddings, unused cameras, Task 12, LOSO, deployment, or a paper claim.

## Frozen R3 Diagnosis

R3 completed the formal fold-0 400-epoch entry experiment and stopped with
`FIT_RENDERER_ENTRY_NEGATIVE` without opening held metrics:

```text
fit teacher gate MAE                         0.004541  PASS
learned / teacher outer recovery             37.093%  FAIL
learned / teacher boundary recovery          35.342%  FAIL
outer-positive fit segments                    17/20  FAIL
boundary-positive fit segments                 20/20  PASS
```

Fit-only tracing localized the failure before hard projection:

```text
hard-projection mean absolute gate change     0.000161
hard-projection changed-entry fraction          3.07%
R3 action 90%-energy rank                            2
teacher action 90%-energy rank                       14
R3 top-10 suppression overlap                    0.396
missed teacher-suppression entries          3643/6132
weighted-loss / renderer gradient cosine         0.196
uniform-loss / renderer gradient cosine          0.100
```

Absolute contribution weighting is informative: its Spearman correlation with
true fit flicker-gradient magnitude is `0.831`, and it improves R3 fit outer
and boundary gains over R2 by `46.97%` and `43.49%`. It remains only a magnitude
proxy. It discards the signed temporal cancellation, candidate mean
normalization, and cross-carrier coupling used by the actual renderer metric.

A frozen-hidden shared-head fit oracle recovered about `87%` of teacher outer
gain and `58%` of teacher boundary gain. The current representation therefore
contains useful unused modes, but this does not prove that the unchanged model
can pass boundary recovery. R4-A tests the objective before any architecture
change.

## Alternatives

### Signed normalized trajectory distillation, selected

Reconstruct differentiable target, outer, and boundary sequences from the
fit-only renderer point contributions. Match the signed adjacent differences
of mean-normalized outer and boundary trajectories to the teacher. This
matches the data flow that defines normalized flicker while keeping the
teacher and target-preserving projection unchanged.

### Scalar flicker-ratio matching, rejected

Matching only one scalar flicker value per segment is underconstrained. Two
trajectories can have the same normalized flicker and different frame-level or
carrier-level behavior, so this route could pass fit by exploiting the metric.

### Signed loss plus cross-carrier attention, deferred

Attention may be needed if R4-A cannot recover boundary modes, but changing the
loss and architecture together would prevent attribution. It is permitted only
in a separately preregistered R4-B after R4-A reaches a terminal result.

## Immutable Inputs

R4-A starts fresh from the same immutable inputs as R3:

```text
R1.4-VP fit teachers and certificates
R1.1 F3 probe
R1.2-B raw starting gates
R1.3-G held witnesses
CoreView377 pose manifest
A5 bank
renderer evidence
fixed k=4 nearest-neighbor predictions
```

All inputs, reused R2/R3 source files, the R3 fit-entry diagnostic, and this
design must be SHA256-pinned by the R4-A contract. R2 and R3 output roots remain
read-only. The experiment ID is:

```text
a7c_r1_4vp_r4a_signed_renderer_trajectory_377_v1
```

## Differentiable Renderer Reconstruction

For one complete fit camera-block segment, signal
`s in {target, outer, boundary}`, frame `t`, carrier `i`, frozen base signal
`b_s[t]`, point contribution `c_s[t,i]`, and predicted gate `g[t,i]`, reconstruct:

```text
y_s(g)[t] = b_s[t] - sum_i c_s[t,i] + sum_i c_s[t,i] * g[t,i]
```

Teacher sequences use the same equation with the frozen fit teacher gates.
Unlike R3 weights, point contributions retain their sign. Held rows remain
nonfinite and cannot enter renderer reconstruction, normalization, training,
retry logic, or checkpoint selection.

Define the mean-normalized outer or boundary trajectory as:

```text
z_s(g) = y_s(g) / max(abs(mean_t(y_s(g))), 1e-12)
```

The selected renderer components are:

```text
L_outer   = mean Huber(diff(z_outer(g)),
                       diff(z_outer(teacher)); delta=0.005)
L_boundary = mean Huber(diff(z_boundary(g)),
                        diff(z_boundary(teacher)); delta=0.005)
```

This matches signed temporal changes before the absolute-value reduction used
by normalized flicker. Candidate and teacher use their own differentiable
means, preserving the metric denominator. Every reconstructed sequence must be
finite and its absolute mean must be strictly greater than `1e-12`; the `max`
is defense in depth, not permission to train on a zero-mean sequence.

Target preservation is trained as a response trajectory. With
`r_target(g)[t] = y_target(g)[t] / max(abs(b_target[t]), 1e-12)`:

```text
L_target = mean Huber(r_target(g), r_target(teacher); delta=0.005)
```

The auxiliary teacher term remains unweighted and fit-only:

```text
L_gate_aux = mean Huber(g, teacher; delta=0.01)
           + 0.25 * mean Huber(diff(g), diff(teacher); delta=0.005)
```

## Fixed Initial-Scale Normalization

Raw renderer components have different units and magnitudes. For each complete
fit segment, evaluate `L_outer`, `L_boundary`, `L_target`, and `L_gate_aux` once
at the untrained zero-residual model before optimizer construction. Freeze
those four detached values as `q_outer`, `q_boundary`, `q_target`, and
`q_gate_aux` for all 400 epochs.

Every scale must be finite and strictly greater than `1e-12`; otherwise the
fold fails closed. No running normalization or epoch-dependent reweighting is
allowed. The registered segment loss is:

```text
L = L_outer / q_outer
  + L_boundary / q_boundary
  + 0.1 * L_target / q_target
  + 0.1 * L_gate_aux / q_gate_aux
  + 0.00001 * mean(abs(uncompressed_residual))
```

Outer and boundary receive equal primary weight. Target and teacher-gate
distillation are fixed `0.1` auxiliaries. These coefficients, Huber deltas,
and initial-scale rule are frozen and may not be swept or changed after fold 0.

## Fixed Model, Training, And Projection

The R3 model remains byte-for-byte unchanged:

```text
49-field F3 view encoder                 49 -> 16
pose rotation-6D encoder                36 -> 16 -> 16
explicit view-pose interaction          elementwise multiplication
one-hop dense overlap message            unchanged
visibility-weighted global mean context  unchanged
bidirectional GRU                       hidden 16 per direction
shared residual head                    32 -> 1
residual gate map                       base + 0.1*tanh(residual)
parameter count                         9,073
```

Training remains AdamW for 400 epochs with learning rate `0.001`, weight decay
`0.0001`, gradient clip `1.0`, complete camera-block batches in fixed order,
the R3 seed, and final-epoch checkpoint only. Renderer evidence is a training
loss input and is not a runtime model input.

All raw predictions pass through the unchanged R1.3-P temporal hard projection.
The projection may read runtime mass, A5 weights, topology floors, and the
adjacent gate limit only. It may not read renderer contributions.

## Fit Renderer Entry Gate

Fold 0 runs the full formal 400 epochs before any decision. Its projected
prediction is evaluated only on the 20 supervised fit segments and must satisfy:

```text
final signed renderer-trajectory loss < initial loss
fit teacher gate MAE <= 0.007
learned mean outer gain / teacher mean outer gain >= 0.70
learned mean boundary gain / teacher mean boundary gain >= 0.70
learned outer-positive fit-segment fraction >= 0.90
learned boundary-positive fit-segment fraction >= 0.90
held teacher rows and held renderer records accessed = false
```

The fold summary must also report, without adding promotion thresholds:

```text
raw and projected renderer recovery
action cosine and top-10 suppression overlap
action 90% and 95% energy ranks
missed teacher-suppression fraction
false-maximum contribution to gate MAE
initial and final loss components and frozen scales
```

Failure produces `FIT_RENDERER_ENTRY_NEGATIVE`, freezes the fold-0 diagnostic,
and stops without folds 1-5 or held audit. It permits no retry, coefficient
change, checkpoint search, seed search, or architecture change.

If fold 0 passes, folds 1-5 use the identical contract. Any later fit-entry
failure is `TRAINING_ERROR` and does not open held metrics.

## Held Audit And Outcome Routing

Only after all six fit entries pass and all learned artifact hashes are frozen
may the independent auditor execute the unchanged R3/R2 held promotion
protocol. It must preserve the same outer/boundary mean, positive fraction,
q10, worst-record, target, soft-IoU, visibility, adjacent-jump, per-camera,
topology, coverage, frozen-parts, A5 upper-bound, R1.2-B, and k=4 nearest-neighbor
gates.

```text
FIT_RENDERER_ENTRY_NEGATIVE -> signed renderer loss is insufficient on fit;
                               only a separately preregistered R4-B may change
                               cross-carrier architecture.
CANARY_NEGATIVE             -> fit objective succeeds but held generalization
                               or visibility fails; analyze without tuning R4-A.
CANARY_PROMOTED             -> authorizes only a separately preregistered
                               visibility-aware validation design.
```

No outcome automatically opens c17-c23, Task 12, LOSO, deployment, or paper
claims. All eligibility flags remain false.

## Required Tests

TDD must cover:

1. exact signed renderer reconstruction and gradient flow;
2. candidate-mean normalization and zero/nonfinite mean rejection;
3. signed adjacent trajectory matching for outer and boundary;
4. target-response and unweighted teacher auxiliary components;
5. fixed initial scales and rejection of zero/nonfinite scales;
6. held contribution rows remaining nonfinite and inaccessible;
7. loss-gradient alignment improving over R3 on a synthetic cancellation case;
8. unchanged R3 model signature, parameter count, optimizer, projection, split,
   and held thresholds;
9. fit-entry pass and every failure condition;
10. auditor status-2 routing and four mutually exclusive terminal markers;
11. real fold-0 400-epoch entry before folds 1-5;
12. the complete frozen A7c regression suite.
