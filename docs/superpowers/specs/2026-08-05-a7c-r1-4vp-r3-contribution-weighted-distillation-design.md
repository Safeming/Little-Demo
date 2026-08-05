# A7c R1.4-VP-R3 Contribution-Weighted Distillation Design

## Status And Question

This document preregisters a single-variable CoreView377 canary that asks
whether renderer-contribution-weighted teacher distillation can preserve the
carrier allocation that uniform gate Huber loss lost in R1.4-VP-R2.

R3 does not change runtime features, pose, overlap graph, model architecture,
optimizer, epoch count, teacher, camera/block split, hard projection, or held
promotion thresholds. It does not authorize unused cameras, Task 12, LOSO,
deployment, or a paper claim.

## Frozen R2 Diagnosis

R2 completed all six 400-epoch folds and fixed the registered residual-loss
scale error. Its mean fit teacher MAE was `0.003393`, and its mean learned
displacement was `84.10%` of the teacher displacement. Nevertheless, an
independent held replay returned `CANARY_NEGATIVE`:

```text
outer gain                         0.139320%
boundary gain                      0.175195%
outer/boundary positive records    16/24
outer q10 / worst                 -0.402217% / -1.153065%
boundary q10 / worst              -0.118832% / -0.189490%
maximum visibility ratio           1.006518
minimum target response            0.994109
maximum soft-IoU drop               0.001159
maximum adjacent gate change        0.015000
topology mismatches                 0
```

The exact held witness proves renderer-gain capacity: outer `1.573653%`,
boundary `3.446923%`, and `24/24` positive records. It does not prove the
visibility gate because its own visibility ratio is `1.016473`.

The decisive fit-only diagnostic is that uniform gate fitting did not preserve
renderer behavior even on supervised segments:

```text
fit action cosine                         0.928158
fit top-10 suppression overlap            0.535088
learned / teacher outer gain recovery      0.274772
learned / teacher boundary gain recovery   0.258363
learned outer-positive fit segments       78.333%
```

Uniform Huber loss gives a low-contribution carrier the same weight as a
carrier whose small gate error controls the renderer objective. R3 changes
only that weighting.

## Alternatives

### Contribution-weighted distillation, selected

Use fit-only true renderer point contributions to weight gate and temporal
teacher errors. Runtime remains unchanged, and the result cleanly tests the
proven loss/objective mismatch.

### Contribution weighting plus carrier attention, deferred

Global carrier attention could improve held observability, but changing the
loss and architecture together would prevent attribution. It is allowed only
after R3 establishes whether weighted supervision succeeds on fit renderer
behavior.

### Visibility-aware oracle first, deferred

The visibility gate has a separate feasibility gap. Fixing it first would not
address the demonstrated outer/boundary carrier-allocation error. It requires
a separately preregistered canary after R3.

## Immutable Inputs

R3 reuses without regeneration:

```text
R1.4-VP fit teachers (six teacher arrays and certificates)
R1.1 F3 probe
R1.2-B fold predictions
R1.3-G held witnesses
CoreView377 pose manifest
A5 bank
renderer evidence
fixed k=4 nearest-neighbor predictions
```

The contract must SHA256-pin all inputs. The old R1.4-VP and R2 output roots
remain read-only. R3 uses experiment ID:

```text
a7c_r1_4vp_r3_crw_contribution_weighted_377_v1
```

## Contribution Weights

Only fit teacher samples may supply weights. For each complete camera-block
segment and signal `s` in `{target, outer, boundary}`, let `c_s[t,i]` be the
absolute true renderer point contribution of carrier `i` at frame `t`.
Normalize each signal over the carrier dimension:

```text
n_s[t,i] = c_s[t,i] / (mean_i(c_s[t,i]) + 1e-12)
```

The preliminary carrier importance is:

```text
u[t,i] = n_target[t,i] + n_outer[t,i] + n_boundary[t,i]
```

Clip and renormalize over the complete segment:

```text
w_gate = clip(u, 0.1, 10.0)
w_gate = w_gate / mean_{t,i}(w_gate)
w_temporal[t,i] = max(w_gate[t-1,i], w_gate[t,i])
```

All weights must be finite and strictly positive. A signal with zero total
contribution produces zeros before the final clip; it does not cause division
by zero. Held rows must remain nonfinite in the weight artifact and must never
enter normalization, training, retries, checkpoint choice, or the fit renderer
entry gate.

## Fixed Model And Training

R3 starts fresh from the same R1.2-B raw gates. It does not continue R2 model
weights. The frozen model remains:

```text
49-field F3 view encoder                 49 -> 16
pose rotation-6D encoder                36 -> 16 -> 16
explicit view-pose interaction          elementwise multiplication
one-hop dense overlap message            unchanged
visibility-weighted global mean context  unchanged
bidirectional GRU                       hidden 16 per direction
residual gate map                       base + 0.1*tanh(residual)
parameter count                         <= 50,000
```

Training remains AdamW, 400 epochs, learning rate `0.001`, weight decay
`0.0001`, gradient clip `1.0`, complete camera-block batches in fixed order,
and final-epoch checkpoint only. Residual weight remains the repaired R2 value
`0.00001`.

The only changed equation is the reduction used by the teacher loss:

```text
weighted_mean(Huber(gate, teacher; delta=0.01), w_gate)
+ 0.25 * weighted_mean(
    Huber(diff(gate), diff(teacher); delta=0.005), w_temporal)
+ 0.00001 * mean(abs(uncompressed_residual))
```

No renderer contribution is a runtime model input.

## Fit Renderer Entry Gate

Fold 0 runs the full formal 400 epochs first. Before folds 1-5 can start, its
final frozen prediction is evaluated only on the 20 supervised fit segments.
It must satisfy all conditions:

```text
final weighted fit loss < initial weighted fit loss
fit teacher gate MAE <= 0.007
learned mean outer gain / teacher mean outer gain >= 0.70
learned mean boundary gain / teacher mean boundary gain >= 0.70
learned outer-positive fit-segment fraction >= 0.90
learned boundary-positive fit-segment fraction >= 0.90
held teacher rows and held renderer records accessed = false
```

This is an execution-integrity entry gate, not checkpoint or hyperparameter
selection. Failure produces `FIT_RENDERER_ENTRY_NEGATIVE`, freezes the fold-0
diagnostic, and stops without training folds 1-5. It permits no retry, weight
change, architecture change, or held audit.

If fold 0 passes, folds 1-5 use the same frozen contract. Every fold must also
satisfy the same fit renderer integrity conditions before its artifacts are
accepted. A later fold failure is `TRAINING_ERROR`; it does not open held
metrics.

## Hard Projection And Held Audit

All predictions pass through the unchanged R1.3-P hard projection. The
projection may read runtime mass, A5 weights, topology floors, and the adjacent
gate limit only. It may not read true renderer contributions.

After all six folds and artifact hashes are frozen, the independent held audit
uses the unchanged R2 promotion protocol:

```text
mean outer gain >= 0.005
mean boundary gain >= 0.005
positive record fraction >= 0.90 for both
10th-percentile gain >= 0 for both
worst record regression >= -0.005
minimum target response >= 0.99
maximum soft-IoU drop <= 0.005
maximum visibility response ratio <= 1.0
maximum adjacent gate change <= 0.02
all camera means positive for both
strictly exceed R1.2-B and frozen k=4 NN
topology, coverage, frozen parts, and weight bounds pass
```

The R3 contract includes the legacy `r1_1_f1_outer_gain` and
`r1_1_f1_boundary_gain` summary fields so the independent auditor cannot repeat
R2's missing-contract-field execution error.

## Required Tests

TDD must cover:

1. signal normalization, clipping, and mean-one renormalization;
2. a high-contribution carrier receiving a larger gate gradient;
3. temporal weights using the adjacent maximum;
4. zero-contribution signals remaining finite;
5. held contribution rows remaining inaccessible and nonfinite;
6. fit renderer entry pass and each failure condition;
7. unchanged model signature, parameter count, optimizer, projection, and held
   thresholds;
8. auditor baseline-field completeness and status-2 routing;
9. real fold-0 400-epoch entry smoke before folds 1-5 launch;
10. the complete frozen A7c regression suite.

## Outcome Routing

`FIT_RENDERER_ENTRY_NEGATIVE` rejects uniform-architecture contribution
weighting without opening held metrics. The next design may consider explicit
global carrier attention.

`CANARY_NEGATIVE` means weighted supervision passed fit integrity but did not
generalize or failed the independent visibility gate. The next design must use
the failure decomposition rather than tune R3.

`CANARY_PROMOTED` authorizes only a separately preregistered visibility-aware
validation design. It does not automatically open unused cameras or Task 12.

All artifacts remain `deployment_eligible=false`, `teacher_eligible=false`, and
`paper_test_eligible=false`.
