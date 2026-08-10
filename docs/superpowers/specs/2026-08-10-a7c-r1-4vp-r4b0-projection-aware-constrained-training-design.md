# A7c R1.4-VP-R4-B0 Projection-Aware Constrained Training Design

## Status And Question

R4-B0 is a fit-first CoreView377 experiment. It asks whether the unchanged
9,073-parameter R4-A compositor can recover paper-gate renderer behavior when
training evaluates the exact deployed gates instead of optimizing raw gates
that are projected only after training.

R4-B0 changes one scientific interface: the training objective consumes the
output of the frozen R1.3-P exact temporal projection. The model, F3 features,
pose representation, carrier set, A5 topology, R1.2-B starting gates, teacher,
optimizer, epoch count, split, and held audit remain unchanged. Cross-carrier
attention, carrier-ID embeddings, new cameras, and a larger model are forbidden.

The experiment ID is:

```text
a7c_r1_4vp_r4b0_projection_aware_constrained_377_v1
```

## Frozen Root-Cause Evidence

R4-A stopped at fold 0 with `FIT_RENDERER_ENTRY_NEGATIVE`. Its raw predictions
had enough renderer response, but the deployed exact projection removed most
of that response:

```text
metric                                R4-A raw       R4-A projected
outer recovery                         112.102%           49.799%
boundary recovery                       95.901%           36.628%
fit teacher gate MAE                  0.008636          0.007543
action 90%-energy rank                      23                 7
top-10 suppression overlap               0.227             0.240
missed teacher suppression fraction      0.778             0.694
```

The teacher action has 90%-energy rank 14. R4-A therefore did not fail because
its raw output was low-rank. It selected the wrong high-rank carrier actions:

```text
raw maximum adjacent gate jump                 0.100000
registered maximum adjacent gate jump          0.015000
raw jump-violation fraction                      15.49%
raw proxy-target failing frames                 208/456
raw topology-floor violation fraction                 0
```

Fit-only projection ablations localized the response loss:

```text
projection constraints              outer recovery   boundary recovery
none                                      112.102%             95.901%
proxy target only                          76.210%             67.779%
temporal jump only                         64.202%             42.204%
proxy target + temporal jump               49.799%             36.628%
topology only                             112.102%             95.901%
```

The exact R1.3-G aggregate oracle is `CERTIFIED_FEASIBLE` under the same
carrier set and hard guards, with mean outer/boundary gains of
`0.0157365/0.0344692`. The carrier set and constraint system are therefore not
the root cause. The broken interface is:

```text
R4-A training: raw gates -> renderer objective
R4-A entry:    raw gates -> exact projection -> renderer entry
```

## Considered Alternatives

### Exact-forward projection-aware training, selected

Run the frozen HiGHS projection in every complete fit segment forward pass.
Use its exact projected values for all renderer and action losses, with a
straight-through gradient to the raw model gates. This removes the
train/deploy mismatch without changing model capacity or final constraints.

### Differentiable approximate projection, deferred

An unrolled alternating projection would be faster and differentiable, but it
would add a second projection implementation and a new approximation gap. It
is not justified before the exact-forward canary establishes observability.

### Cross-carrier attention, deferred to R4-B1

Attention may be necessary if the current model cannot represent a useful
solution inside the feasible set. Adding it now would change architecture and
training semantics together, preventing attribution and increasing
CoreView377 overfitting risk.

## Immutable Inputs And Sources

R4-B0 uses the exact R4-A immutable inputs:

```text
R1.4-VP fold teachers and certificates
R1.1 F3 renderer probe
R1.2-B raw starting gates
R1.3-G witnesses
CoreView377 pose manifest
A5 part-label bank
renderer evidence
fixed k=4 nearest-neighbor predictions
```

The contract must SHA256-pin the R4-A contract, policy, trainer, auditor,
fold-0 predictions, fit entry, this design, every inherited R2/R3 source, and
all immutable input artifacts. R4-A outputs remain read-only. No c17-c23 row
may be loaded by the trainer, observability probe, retry logic, or checkpoint
selection.

## Exact-Forward Deployed Gate

For each complete fit camera-block segment, the unchanged model predicts raw
gates `g_raw`. Apply the byte-frozen R1.3-P projection:

```text
g_exact = P_HiGHS(
    g_raw,
    topology_floor(A5),
    proxy_target_response >= 0.995,
    adjacent_gate_jump <= 0.015,
    gate_range [0.9, 1.0]
)
```

The training tensor is an exact-forward straight-through gate:

```text
g_deployed = g_raw + stop_gradient(g_exact - g_raw)
```

The forward value of `g_deployed` must be bitwise equal to `g_exact` after
conversion to the training dtype. Its backward derivative with respect to
`g_raw` is the identity. Renderer, target, gate, action, and temporal losses
must consume `g_deployed`; no loss may consume raw gates except the explicit
projection-consistency component.

The exact projector remains the final deployment and audit projector. The
straight-through rule does not weaken a constraint and does not authorize an
approximate final gate.

## Renderer Reconstruction And Exact Gain

For signal `s in {target, outer, boundary}`, frame `t`, carrier `i`, frozen
base sequence `b_s`, point contribution `c_s`, and deployed gate `g`, use the
same signed reconstruction as R4-A:

```text
y_s(g)[t] = b_s[t] - sum_i c_s[t,i] + sum_i c_s[t,i] * g[t,i]
```

Define the differentiable normalized flicker exactly as the evaluator:

```text
F(v) = mean_t(abs(diff(v))) / max(abs(mean_t(v)), 1e-12)
G_s(g) = 1 - F(y_s(g)) / max(F(b_s), 1e-12)
```

The exact response components are:

```text
L_outer_gain = Huber(G_outer(g_deployed),
                     G_outer(g_teacher); delta=0.005)
L_boundary_gain = Huber(G_boundary(g_deployed),
                        G_boundary(g_teacher); delta=0.005)
```

R4-A signed adjacent trajectory components are retained on deployed gates:

```text
z_s(g) = y_s(g) / max(abs(mean_t(y_s(g))), 1e-12)
L_outer_trajectory = mean Huber(diff(z_outer(g_deployed)),
                                diff(z_outer(g_teacher)); delta=0.005)
L_boundary_trajectory = mean Huber(diff(z_boundary(g_deployed)),
                                   diff(z_boundary(g_teacher)); delta=0.005)
```

The exact gain terms close the amplitude/entry gap. The trajectory terms keep
the scalar gain from being satisfied by a framewise shape mismatch.

## Target, Gate, Action, And Projection Preservation

Target response remains renderer-aligned:

```text
r_target(g)[t] = y_target(g)[t] / max(abs(b_target[t]), 1e-12)
L_target = mean Huber(r_target(g_deployed),
                      r_target(g_teacher); delta=0.005)
```

Gate preservation is evaluated on the deployed gate:

```text
L_gate = mean Huber(g_deployed, g_teacher; delta=0.01)
       + 0.25 * mean Huber(diff(g_deployed),
                           diff(g_teacher); delta=0.005)
```

Define suppression actions relative to the frozen R1.2-B start:

```text
a(g) = base_gate - g
cosine_epsilon(u, v) = dot(u, v) /
    max(norm(u) * norm(v), 1e-12)
L_action = 1 - clamp(cosine_epsilon(a(g_deployed), a(g_teacher)), -1, 1)
```

The teacher action norm must be finite and strictly greater than `1e-12`;
otherwise the segment fails closed. A zero candidate action has cosine zero
and action loss one. This component directly penalizes the wrong-carrier
shortcut observed in R4-A without adding carrier IDs.

Raw-to-deployed consistency is:

```text
L_projection = mean(abs(g_raw - stop_gradient(g_exact))) / 0.0002
```

This term makes the exact projector approach an identity map instead of
allowing the straight-through estimator to hide infeasible raw actions.

## Frozen Component Scaling And Total Loss

At the zero-residual initialization, evaluate the first seven components on all
20 fold-fit segments after exact projection. Freeze one global scale per
component as the median of its 20 detached values:

```text
trajectory_outer, trajectory_boundary,
gain_outer, gain_boundary,
target, gate, action
```

Every median must be finite and greater than `1e-12`. R4-A per-segment scales
are not reused; their observed max/min spread reached `31.17x`, which allowed
small-residual segments to dominate. No running scale, learned uncertainty,
or epoch-dependent reweighting is allowed.

`L_projection` is already dimensionless because it is divided by the registered
fit displacement reserve `0.0002`. It does not use an initial median because an
already feasible zero-residual model can have exactly zero projection
displacement.

Let `N(component)` mean component divided by its frozen global median. The
registered loss is:

```text
L_renderer = mean(
    N(trajectory_outer), N(trajectory_boundary),
    N(gain_outer), N(gain_boundary)
)

L_preservation = mean(
    N(target), N(gate), N(action), L_projection
)

L = L_renderer + L_preservation
  + 0.00001 * mean(abs(uncompressed_residual))
```

No component weight, scale floor, Huber delta, or renderer/action definition
may be changed after fold 0 starts.

## Model, Optimizer, And Split

The R4-A model remains byte-for-byte unchanged:

```text
49-field F3 view encoder                 49 -> 16
pose rotation-6D encoder                36 -> 16 -> 16
one-hop overlap message                  unchanged
visibility-weighted global context       unchanged
bidirectional GRU                        hidden 16 per direction
shared residual head                     32 -> 1
residual map                              base + 0.1*tanh(residual)
parameter count                          9,073
attention                                false
carrier embedding                        false
```

Training remains AdamW, 400 epochs, learning rate `0.001`, weight decay
`0.0001`, gradient clip `1.0`, fixed camera-block order, R4-A seed, and
final-epoch checkpoint only. A run may not search seeds, epochs, component
weights, scale rules, checkpoints, projection settings, or action thresholds.

## Gradient Observability Preflight

Before the formal run, a fit-only observability test must validate the
straight-through interface without updating or saving a model:

1. exact-forward deployed gates satisfy every projection certificate;
2. every registered component and model gradient is finite;
3. aggregate gradient norm is strictly greater than `1e-12`;
4. one AdamW step with learning rate `0.001` on an ephemeral cloned model,
   followed by exact reprojection, decreases aggregate deployed loss; the
   clone and optimizer state are discarded;
5. held teacher and renderer rows remain nonfinite and inaccessible.

Failure is `FEATURE_OBSERVABILITY_NEGATIVE`. It forbids formal 400-epoch
training and does not authorize coefficient changes or attention.

## Fold-0 Fit Entry

Fold 0 completes exactly 400 epochs before entry. All entry metrics use
`g_exact`, the same gate used by the loss forward pass. The fit reserve is
intentionally stricter than the inherited held promotion gate:

```text
final deployed loss < initial deployed loss
projected teacher gate MAE <= 0.0065
outer recovery >= 0.75
boundary recovery >= 0.75
outer-positive segment fraction >= 0.95
boundary-positive segment fraction >= 0.95
action cosine >= 0.90
top-10 suppression overlap >= 0.45
missed teacher-suppression fraction <= 0.55
raw-to-exact mean absolute displacement <= 0.0002
raw-to-exact changed-entry fraction at absolute difference >1e-12 <= 0.05
all exact projection certificates pass
held teacher/renderer rows accessed = false
```

The runner writes `FIT_PROJECTED_ENTRY_NEGATIVE` and stops before folds 1-5 if
any condition fails. It performs no retry and does not open held metrics.

## Outcome Routing And R4-B1 Trigger

If fold 0 passes, folds 1-5 use the identical frozen contract. Only six fit
passes may freeze learned artifacts and open the unchanged R4-A/R3/R2 held
audit. All spatial, topology, coverage, frozen-part, soft-IoU, target,
visibility, temporal, nearest-neighbor, per-camera, and worst-block gates remain
unchanged.

```text
FEATURE_OBSERVABILITY_NEGATIVE
    -> straight-through training is not usable; stop and analyze gradients.

FIT_PROJECTED_ENTRY_NEGATIVE with projection consistency failure
    -> projection-aware optimization failed; do not add attention.

FIT_PROJECTED_ENTRY_NEGATIVE with projection consistency pass but
renderer/action failure
    -> the unchanged model manifold is insufficient inside the feasible set;
       this is the only result that authorizes a separate R4-B1 attention design.

CANARY_NEGATIVE
    -> fit succeeds but held generalization or guards fail; freeze the negative
       result and do not tune R4-B0.

CANARY_PROMOTED
    -> authorizes only the next preregistered validation stage; it is not a
       paper-level temporal reliability claim by itself.
```

No R4-B0 result automatically opens c17-c23, Task 12, LOSO, deployment, or
paper claims. All eligibility flags remain false.

## Required Tests And Artifacts

TDD must cover:

1. exact-forward equality between straight-through and HiGHS gates;
2. identity backward gradient through the straight-through gate;
3. all loss components reading deployed gates rather than raw gates;
4. differentiable normalized flicker and exact evaluator gain parity;
5. signed renderer reconstruction and trajectory parity with R4-A;
6. target, gate, action, and projection component formulas;
7. global median scale freezing and zero/nonfinite rejection;
8. held rows remaining nonfinite and inaccessible;
9. unchanged 9,073-parameter model and optimizer signature;
10. observability positive and every fail-closed condition;
11. fold-0 entry positive and each individual rejection condition;
12. status-2 routing without held audit;
13. six-fit artifact freezing before held audit;
14. complete inherited A7c regression suite.

The fold summary must save raw and exact gates, exact projection certificates,
initial/final raw and normalized loss components, renderer gains, action
diagnostics, projection displacement diagnostics, and held-access flags. Every
source and learned artifact must have a reproducible SHA256 fingerprint.
