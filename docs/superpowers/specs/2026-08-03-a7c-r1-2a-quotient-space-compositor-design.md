# A7c R1.2-A Quotient-Space Compositor Design

## Status And Scope

This document preregisters the CoreView377 R1.2-A canary before implementation
or result inspection. R1.2-A tests whether the R1/R1.1 failure is primarily
caused by direct regression to a non-identifiable per-carrier oracle gate.

R1.2-A is a development canary. It does not authorize Task12, does not produce a
paper result, and cannot read c17-c23 unless the frozen held-block gate permits
the already registered c17-c20 audit stage.

## Frozen Root-Cause Hypothesis

For each camera, the R0 point oracle exposes 85 carrier gates with 12 temporal
knots, or 1020 coefficients. The joint target/outer/boundary contribution map
has rank 342 and nullity 678. Therefore a large family of carrier gate sequences
produces the same renderer-level evidence. R1 and R1.1 selected one member of
that family and trained an independent MLP with per-gate SmoothL1 loss.

R1.2-A tests one change only:

```text
R1/R1.1: runtime features -> independent gates -> oracle-gate regression
R1.2-A:  runtime features -> independent scores -> joint budget projection
          -> renderer-level temporal and guard losses
```

If this change cannot pass the held-block gate, direct teacher imitation is not
the only blocker and the next route must add explicit carrier relations. Network
width, feature groups, camera IDs, frame IDs, recurrent state, and audit data
cannot be added to repair R1.2-A in place.

## Inputs And Leakage Boundary

R1.2-A reuses the frozen CoreView377 artifacts:

- R1.1 probe with 912 samples and 85 carriers;
- v5.3 renderer contribution evidence;
- frozen A5 part-label bank;
- R0 teacher only for carrier/sample alignment and provenance, never as a loss;
- fit cameras c01/c05/c09/c13 and six contiguous time blocks;
- unopened audit cameras c17/c18/c19/c20;
- forbidden cameras c21/c22/c23.

The score network reads exactly the R1.1 F1 feature group. It does not read
oracle gates, target/outer/boundary contribution labels, masks, camera ID,
frame ID, subject ID, Gaussian ID, image name, or previous-frame state.

Training losses may use frozen renderer target/outer/boundary and selection
contributions as supervision. They are not predictor or projection inputs.

## Runtime-Safe Joint Target Budget

The projection must not use ground-truth target masks. It constructs a
per-sample, per-carrier target-mass proxy from fields already present in the
R1.1 runtime probe:

```text
local_lower_probability = clamp(
    semantic_support_mean / max(alpha_mean, epsilon), 0, 1)

runtime_target_mass = alpha_transmittance_mass
                      * a5_lower_weight
                      * local_lower_probability
```

The network emits a raw bounded gate `r_i` in `[0.9, 1.0]`. Define raw damping
`d_i = 1 - r_i`, proxy target loss `L = sum_i m_i d_i`, and the construction
budget `B = (1 - proxy_target_floor) * sum_i m_i`. The joint projection is:

```text
scale = min(1, B / max(L, epsilon))
g_i = 1 - scale * d_i
```

This couples all carriers without adding a relational encoder and guarantees
the runtime proxy target response floor. The topology floor then only restores
gates upward:

```text
g_i = max(g_i, selection_threshold / max(a5_lower_weight_i, epsilon))
g_i = clamp(g_i, 0.9, 1.0)
```

The projection cannot use renderer selection contributions or ground-truth
masks. True target response and soft IoU remain training constraints and formal
audit guards.

## Predictor

The predictor is deliberately unchanged from the R1.1 capacity scale:

```text
F1 input -> 64 -> 32 -> 1
SiLU activations
raw gate in [0.9, 1.0]
no embeddings, recurrence, attention, carrier table, or identity feature
```

Feature normalization is fitted only on the fold's training samples. The same
fixed initialization, optimizer, epoch budget, and loss weights are used for
all six folds and the final fit-camera model. There is no held-block checkpoint
selection or hyperparameter sweep.

## Renderer-Level Training Objective

For every contiguous training segment, projected gates are applied to the
frozen per-carrier contribution sequences. The primary loss is the mean of
candidate-to-A5 normalized-flicker ratios for outer and boundary. Adjacent pairs
cannot cross camera boundaries, held-block gaps, or the six registered temporal
block boundaries.

The fixed objective also contains hinge penalties for:

- true selection target response below the construction margin;
- true selection soft-IoU drop above 0.005;
- adjacent projected gate change above 0.02;
- unnecessary damping magnitude.

Loss weights, construction margins, optimizer settings, and epoch count must be
frozen in the contract before the first held-block result is opened. R0 teacher
R2, gate MAE, and rank correlation are diagnostics only and cannot affect
training, selection, or promotion.

## Split And Evaluation

Six leave-one-contiguous-block-out models are trained on c01/c05/c09/c13. Each
held block is evaluated independently for each fit camera, producing 24 formal
records. The final model is trained on all fit-camera blocks only after the fold
configuration is frozen; it cannot authorize audit-camera evaluation by itself.

R1.2-A must satisfy every existing held-block guard:

```text
mean outer flicker gain >= 0.5%
mean boundary flicker gain >= 0.5%
outer positive-block fraction >= 0.90
boundary positive-block fraction >= 0.90
outer/boundary q10 block gain >= 0
worst outer/boundary block regression <= 0.5%
minimum true unweighted target response >= 0.99
maximum true unweighted soft-IoU drop <= 0.005
maximum adjacent projected gate change <= 0.02
gate range [0.9, 1.0]
A5 selection topology, coverage, frozen parts, and weight upper bounds unchanged
```

It must additionally improve mean outer and boundary gain over the frozen R1.1
F1 held-block result. Passing the teacher-label loss is not a criterion.

## Stop Rules

- Any source fingerprint, sample order, carrier order, schema, or split mismatch
  stops before training.
- A runtime projection that reads GT masks or contribution labels is invalid.
- A failed held-block gate writes `.rejected` and stops before c17-c20.
- Failure cannot be repaired by changing loss weights, epochs, widths, margins,
  feature groups, or thresholds after results are opened.
- A pass only authorizes a separately recorded c17-c20 held-camera audit using
  the frozen final model. It does not authorize c21-c23, Task12, or a paper claim.
- If R1.2-A fails outer while boundary passes, the next design is R1.2-B with a
  permutation-equivariant carrier-set or sparse overlap-graph encoder.

## Artifacts And Verification

The implementation will create one frozen JSON contract, a pure NumPy/Torch
projection and renderer-loss module, focused tests, fold/final training CLI,
held-block audit CLI, and a resumable runner. Formal outputs include complete
source fingerprints, predictions, fold summaries, held-block records, aggregate
promotion decision, commands, commit, and `paper_test_eligible=false`.

Tests must cover target-budget scaling, zero-mass behavior, topology floors,
forbidden projection inputs, contiguous segment construction, renderer-level
loss direction, target/soft-IoU/jump penalties, fit-only normalization, source
alignment, deterministic training, and held-block promotion aggregation.
