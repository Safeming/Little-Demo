# A7c R1.4-VP-R2 Loss-Scale Repair Design

## Status And Question

This document preregisters a minimal repair canary for the completed
CoreView377 R1.4-VP experiment. R1.4-VP-R2 asks whether correcting one proven
loss-scale error is sufficient for the unchanged view-pose model to distill
the fit-only oracle teacher and meet the existing held-block promotion gates.

R2 does not change the model, runtime features, pose representation, carrier
set, teacher gates, camera split, hard projection, promotion thresholds, or
checkpoint rule. It does not authorize new cameras, Task 12, LOSO,
deployment, or a paper claim.

## Frozen Negative Result And Root Cause

The completed R1.4-VP independent audit returned `CANARY_NEGATIVE`:

```text
learned outer gain                    0.376264%
learned boundary gain                 0.264128%
outer positive held records           15/24
boundary positive held records        21/24
maximum visibility response ratio     1.000330
minimum target response               0.999532
maximum adjacent gate change          0.015000
R1.2-B outer gain                      0.519637%
R1.2-B boundary gain                   0.286637%
```

The learned model beat the fixed `k=4` baseline but did not beat R1.2-B or the
formal `0.5%` gain thresholds. Its mean fit loss increased by `0.1275%` from
initialization to the final checkpoint.

The failure is traced to the registered latent residual penalty:

```text
old loss term = 0.001 * mean(abs(uncompressed_residual))
gate mapping  = base + 0.1 * tanh(uncompressed_residual)
```

Near zero, this produces a gate-space regularization slope of approximately
`0.001 / 0.1 = 0.01`, equal to the maximum slope of the gate Huber loss with
`delta=0.01`. The regularizer can therefore cancel the entire teacher-fitting
gradient.

Real fold evidence confirms anchor locking. The teacher differs from the
R1.2-B base by `0.0081-0.0135` mean absolute gate value, while the trained
model moved from the base by only `4e-6-2.4e-5`, or `0.03%-0.18%` of the
required displacement.

Fit-only diagnostic runs changed one variable at a time for fold 0 at 100
epochs:

```text
residual weight 0.00100    fit teacher MAE 0.01353 (400-epoch frozen run)
residual weight 0.00000    fit teacher MAE 0.00553
residual weight 0.00001    fit teacher MAE 0.00563
residual weight 0.00010    fit teacher MAE 0.00661
```

Segment error had no meaningful optimizer-order correlation, so R2 does not
add gradient surgery, change batching, or enlarge the model. The selected
repair is `residual_loss_weight=0.00001`, which preserves a latent magnitude
penalty while reducing its near-zero gate-space slope to approximately
`0.0001`.

## Alternatives

### Minimal loss-scale repair, selected

Change only the latent residual weight, add fit-only convergence integrity
gates, and rerun all six folds. This directly tests the proven root cause with
the smallest behavioral change.

### Remove residual regularization, rejected

Zero weight gave a slightly lower fold-0 fit MAE but allowed the mean latent
residual to grow to `1.75`, where `tanh` saturation weakens observability and
creates unnecessary extrapolation risk.

### Joint-gradient or PCGrad training, deferred

The current diagnostics do not show update-order forgetting. Adding gradient
surgery would change both optimization and the method claim, preventing a
clean attribution to the loss-scale repair.

## Immutable Inputs

R2 reuses, without regeneration, the completed R1.4-VP teacher root:

```text
exp/acceptdata/a7c_r1_4vp_oracle_distilled_view_pose_377_v1/teachers
```

All six `teacher.npz` files, all six certificate files, the teacher summary,
the R1.1 F3 probe, pose manifest, A5 bank, R1.2-B fold predictions, R1.3-P
projection contract, evidence artifact, and R1.3-G held witnesses must be
SHA256-pinned by the R2 contract before training.

Teacher generation is not rerun. Its 120 certificates remain diagnostic:
111 segments completed stage three normally, 9 used a certified stage-two
feasible fallback, 2 stage-two solves required no-presolve retry, and the
maximum recorded primal residual was `9.532414e-8`, below the frozen `1e-7`
limit. No teacher certificate may be modified by R2.

The old R1.4-VP result directory and its frozen artifacts are read-only. R2
uses a new experiment ID and output directory.

## Fixed Training Behavior

The architecture and training loop remain those preregistered for R1.4-VP:

```text
F3 view encoder                         16 dimensions
pose rotation-6D encoder                36 -> 16 -> 16
explicit view-pose interaction          elementwise multiplication
bidirectional GRU                       hidden 16 per direction
residual gate range                     0.1 * tanh(residual)
parameter budget                        <= 50,000
epochs                                  400
optimizer                               AdamW
learning rate                           0.001
weight decay                            0.0001
gradient clip                           1.0
batch                                   complete camera-block segment
segment order                           camera then block, no shuffle
checkpoint                              final epoch only
```

The R2 loss is:

```text
Huber(predicted gate, teacher gate; delta=0.01)
+ 0.25 * Huber(predicted temporal difference,
               teacher temporal difference; delta=0.005)
+ 0.00001 * mean(abs(uncompressed residual))
```

No held label or held renderer contribution may enter normalization, loss,
training, retry decisions, or checkpoint selection.

## Fit-Only Integrity Gates

Every fold must finish 400 epochs and satisfy both conditions before model
artifacts are frozen:

```text
final aggregate fit loss < initial aggregate fit loss
fit teacher gate MAE <= 0.007
```

These are execution-integrity gates, not model-selection criteria. Failure in
any fold is `TRAINING_ERROR`; it does not trigger a different checkpoint,
epoch count, learning rate, regularization weight, or retry. The thresholds
come only from the preregistered fit-only diagnostics and are fixed before R2
held metrics are opened.

The training summary also records fit temporal-difference MAE, raw/projected
gate ranges, maximum gradient norm, latent residual mean and maximum, and the
ratio of learned displacement to teacher displacement from the R1.2-B base.

## Projection And Independent Audit

Every predicted camera-block segment passes through the unchanged R1.3-P hard
projection. The auditor opens held renderer evidence only after all six model
hashes, predictions, projection certificates, summaries, and a freeze manifest
exist.

The promotion protocol is unchanged:

```text
mean outer gain                         >= 0.005
mean boundary gain                      >= 0.005
positive block fraction                 >= 0.9 for both signals
10th-percentile block gain              >= 0 for both signals
worst block regression                  >= -0.005
minimum target response                 >= 0.99
maximum soft-IoU drop                   <= 0.005
maximum visibility response ratio       <= 1.0
maximum adjacent gate change            <= 0.02
all four camera means                   positive for both signals
learned outer and boundary gains         strictly exceed R1.2-B and k=4 NN
topology, coverage, frozen parts,
and weight upper bounds                 all pass
```

Topology comparison must broadcast the one-row A5 selection mask to the full
candidate frame shape before exact equality. It must also report minimum
continuous topology slack. The completed R1.4-VP arrays have zero topology
mismatches; their earlier `topology_passed=false` was an auditor shape bug.

## Runner Terminal Semantics

The runner treats the independent auditor's status codes explicitly:

```text
0  CANARY_PROMOTED -> .completed
2  CANARY_NEGATIVE -> .rejected
other              -> .failed
```

The audit command must execute inside an `if` statement so Bash `ERR` handling
cannot convert expected status 2 into `TRAINING_ERROR`. The root summary must
preserve the auditor's verdict and must agree with exactly one terminal marker.

## Required Tests

TDD must cover:

1. the registered residual weight and loss-gradient scale;
2. fit convergence gate pass and failure behavior;
3. full-shape topology broadcast with zero mismatch;
4. material topology undercrossing rejection;
5. auditor status 2 mapping to `.rejected`, not `.failed`;
6. held teacher NaN isolation and final-checkpoint behavior;
7. unchanged model signature, parameter budget, projection, and promotion
   thresholds;
8. a real-size fold-0 fit-only smoke run that improves loss and reaches
   teacher MAE `<=0.007` without opening held renderer metrics.

The complete frozen A7c regression suite must remain green before launch.

## Outcome Routing

`CANARY_PROMOTED` only authorizes a separately preregistered validation design.
It does not open unused cameras or Task 12 automatically.

`CANARY_NEGATIVE` closes the minimal loss-scale repair. The next analysis must
compare fit success with held oracle recovery before deciding between an
observability limitation and a new optimization architecture.

`TRAINING_ERROR` permits only a same-contract implementation repair when no
registered value, artifact, model behavior, or threshold changes.
