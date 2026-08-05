# A7c R1.4-VP Oracle-Distilled View-Pose Interaction Design

## Status And Question

This document preregisters the CoreView377 R1.4-VP canary before any fit-block
oracle teacher is generated, any view-pose model is trained, or any held-block
model metric is opened. It asks whether a small offline bidirectional predictor
can recover enough of the renderer-beneficial R1.3-G gate allocation from
runtime-available view, pose, and overlap context to pass the unchanged temporal
promotion protocol.

R1.4-VP is not the pose-only A7b method preregistered on 2026-07-31. That route
is closed: CoreView377 and all four replication subjects were interaction-
dominant, with pose-only contiguous-block CV R2 below zero. R1.4-VP explicitly
models the renderer-visible view-by-pose interaction. It changes the method
claim from a camera-invariant canonical asset to an offline view-conditioned
compositor.

This is a CoreView377 construction canary. It does not authorize c17-c23,
Task 12, multi-subject LOSO, deployment, or a paper claim.

## Frozen Evidence And Entry Gate

R1.3-G regenerated real gates for all 24 `fold x fit camera held block`
records and returned `CERTIFIED_FEASIBLE`:

```text
outer mean gain                 1.573653%
boundary mean gain              3.446923%
outer/boundary positive         24/24
outer q10                       0.500000%
boundary q10                    2.341608%
minimum target response         0.994766
maximum soft-IoU drop           0.001043
maximum adjacent gate change    0.015000
maximum primal residual         1.818989e-12
```

The R1.3-G result establishes aggregate action-space capacity. It does not
establish runtime observability and its held gate arrays are audit-only. They
must not enter R1.4-VP teacher generation, normalization, loss construction,
checkpoint selection, or hyperparameters.

Frozen source fingerprints include:

```text
R1.3-G contract
  0ed0d588ab4a89abfa50d3213a84dc4e055ecd2d800c1bfc7bc154d3bf927bbb
R1.3-G records
  97b59b5ab0b9f0b0c473748f9beb9af184de9de7acccc13dc1c96794f9340594
R1.3-G audit
  69426279d5bdbc44c5b8f9a353e4175eafa27e6c7b9f22811316586f07db615e
R1.3-G summary
  d84d345be3e15b2a833ea75d03f40fcd0473fe2da567ce0c5ee4b5d58422b830
R1.2-B contract
  e2825c1d59e96ff2ea6124bfa1defafb62c73c04a64519c889e909be9ef2f9b5
runtime probe
  643c541af20f732a9de2c4ac6c20ea804ac27be8ad6dad13b1ead5efb6f8b411
renderer evidence
  8b655f48fad664ba308f51d3291971382d7f9037fc7d69e38fca37907efd77f4
A5 bank
  49ba86b05c4f87eaa8b98ef47822c7083a31fdf050a35bd8cf3a88843f8a45d3
teacher manifest
  698f61e195a78849c72be14b8cf9073f281b94124d804013988e7bf605304aa8
```

The six R1.2-B fold prediction hashes remain those frozen in the R1.3-P
contract. The R1.4-VP contract must pin each path and SHA256 explicitly before
teacher generation.

## Alternatives

### Oracle-distilled residual, selected

Fit-only renderer evidence produces feasible, renderer-beneficial teacher gate
sequences. A small residual predictor starts from R1.2-B and learns only the
missing view-pose interaction. This gives a concrete supervised observability
test while preserving strict fold isolation.

### Direct renderer-loss training, rejected

Direct optimization repeats R1.2-B's direction ambiguity and gives the model
many fit-only ways to reduce a weighted loss without learning the selected
feasible allocation. It also makes overfitting harder to distinguish from
observability.

### View-pose nearest-neighbor bank, baseline only

A fixed `k=4` interpolation bank is useful for testing whether local feature
proximity alone explains the oracle. It is not the primary method because it
does not provide a credible new-view or cross-subject architecture.

## Runtime Boundary

R1.4-VP is allowed one frozen pre-edit renderer pass for each input view. The
pass may expose only quantities already in the frozen R1.1 probe and R1.2-B
overlap graph. It may not expose the ground-truth edit target mask or any true
target, outer, boundary, selection-target, or selection-outer contribution.

Inference is offline and bidirectional within one `camera x temporal block`.
It may read the entire block sequence, but it must not cross a camera or block
boundary. It has no persistent state across calls, no frame-count-dependent
parameters, and no previous-video cache.

The following identifiers remain forbidden model inputs:

```text
camera_id, camera_index, frame_id, frame_index, subject_id,
gaussian_id, image_name, held-block identity
```

View conditioning comes from continuous renderer-visible measurements, not a
camera lookup table. Hair and all non-lower parts remain exact A5. The avatar
checkpoint, geometry, opacity, scaling, binding, appearance, carrier set, A5
weights, and A5 selection mask are immutable.

## Six-Fold Isolation

R1.4-VP reuses the six contiguous temporal blocks and four fit cameras
`c01/c05/c09/c13`. For fold `f`:

```text
fit teacher/training samples = five non-held blocks x four fit cameras
held audit samples           = block f x four fit cameras
```

All six folds must finish teacher generation and training before any model or
nearest-neighbor held renderer metric is opened. Held R1.3-G gates may then be
loaded only by the final auditor for gate-error analysis.

The following are fold-local and must be fit from the five fit blocks only:

```text
F3 normalization
pose normalization
nearest-neighbor distance normalization
all model weights
all teacher gates and capacity endpoints
all fit losses and summaries
```

Held samples must be NaN in teacher artifacts and absent from optimizer masks.
No held quantity may affect fixed epoch count, learning rate, model capacity,
loss weights, checkpoint choice, or retry decisions.

## Fit-Only Teacher Generation

Each fold has 20 teacher segments: five fit blocks times four fit cameras. The
full canary therefore generates 120 fit-only teacher segments.

For each segment, first run a boundary-conditioned capacity search using the
unchanged R1.3-G equations and guards:

```text
outer normalized-flicker gain >= 0.005
runtime proxy target response >= 0.995 per frame
true renderer target response >= 0.99 per frame
true selection soft-IoU drop <= 0.005 per frame
adjacent gate change <= 0.015
topology_floor[i] <= gate[t,i] <= 1.0
bisection tolerance = 1e-5
```

The segment boundary request is its own fit-only
`feasible_lower - 0.00002`. No held endpoint supplies a request or prior.

The final teacher solve preserves the fixed outer/boundary requests and uses a
lexicographic anchor to the frozen R1.2-B raw gate sequence:

1. minimize maximum absolute deviation from R1.2-B raw gates;
2. within `1e-9` of that optimum, minimize total absolute deviation;
3. within `1e-9` of both optima, minimize total absolute adjacent gate change.

Every stage uses HiGHS and maximum recomputed primal residual `<=1e-7`.
Repeated generation must produce bitwise-identical gates, metrics, and
certificates. A non-bracketed capacity search, non-optimal LP, fingerprint
mismatch, or non-deterministic result is `TRAINING_ERROR`, not a negative model
result.

Teacher artifacts are diagnostic and always set:

```text
deployment_eligible = false
teacher_eligible = false
paper_test_eligible = false
```

`teacher_eligible=false` means the artifact cannot become a general reusable
teacher or cross-fold label source; it remains authorized only as the current
fold's fit-only supervision.

## Fixed Input Schema

The carrier input uses all frozen R1.1 F3 fields. No feature-group search is
allowed. F3 comprises the F1 canonical/semantic/view features plus the frozen
alpha/transmittance and ray-context measurements:

```text
alpha_transmittance_mass, log1p_alpha_transmittance_mass,
alpha_center/mean/variance,
ray_depth_center/mean/variance,
ray_depth_var_center/mean/variance,
lower_support_center/mean/variance,
semantic_support_center/mean/variance,
carrier_depth_residual, context_available
```

The overlap graph uses projected XY, log depth, visibility, spatial scale
`0.03`, depth scale `0.04`, and minimum edge log weight `-20`, unchanged from
R1.2-B.

Pose uses `pose_body` joints `[0,1,3,4,6,7]`. Each axis-angle rotation is
converted to the continuous first-two-columns rotation-6D representation,
giving 36 values per frame. Root translation, camera extrinsics, camera ID,
global frame index, hands, and future blocks are forbidden.

## Fixed Model

The frozen R1.2-B raw gates are the base prediction. R1.4-VP trains only the
following residual path:

```text
F3 view encoder       -> 16 dimensions
pose encoder 36 -> 16 -> 16 with SiLU
explicit interaction -> view_embedding * pose_embedding
overlap node/message/global context from the frozen graph construction
one bidirectional GRU per carrier sequence, hidden size 16 per direction
one scalar residual head, zero initialized
```

The candidate raw gate is:

```text
clamp(r1_2b_raw_gate + 0.1 * tanh(residual), 0.9, 1.0)
```

The model has one layer per direction, no dropout, no attention, and no free
carrier embedding. Total trainable parameters must be at most 50,000. Width,
depth, GRU size, feature schema, residual range, and initialization are frozen;
there is no CoreView377 architecture search.

Each camera-block segment is packed independently. Hidden state is zeroed at
every segment boundary. Temporal order is semantic: the runner must sort and
verify it against the frozen sample manifest before packing. Repeated inference
under the same order must be identical within floating-point tolerance `1e-7`.
Sequence reversal is not an invariance requirement for the bidirectional GRU.

## Training

Training is fixed before held audit:

```text
epochs = 400
random seed = 20260805
optimizer = AdamW
learning rate = 0.001
weight decay = 0.0001
gradient clipping norm = 1.0
batch unit = one complete camera-block segment
checkpoint = final epoch only
```

The loss is evaluated only under the fold fit mask:

```text
Huber(predicted gate, teacher gate; delta=0.01)
+ 0.25 * Huber(predicted temporal difference,
               teacher temporal difference; delta=0.005)
+ 0.001 * mean absolute residual
```

There is no held early stopping or best-checkpoint selection. Training records
initial/final loss, component losses, gradient maxima, gate range, fit teacher
MAE, and fit temporal-difference MAE.

After prediction, each camera-block sequence passes through the unchanged
R1.3-P temporal hard projection using only runtime mass, A5 weights, topology,
proxy target, and jump constraints. The hard projection cannot read renderer
target/outer/boundary contributions.

## Nearest-Neighbor Baseline

The fixed baseline uses `k=4`. Its sample key concatenates:

```text
36 normalized pose values
visibility-weighted means of view_dir_x/y/z, log_depth,
alpha_transmittance_mass, semantic_support_mean
```

All normalization comes from the fold fit blocks. Euclidean distance chooses
four fit samples across all fit cameras and blocks; inverse-distance weights
interpolate their carrier-resolved teacher gates. Exact matches use the mean of
all zero-distance neighbors. No camera or frame identifier is included. The
interpolated sequence passes through the same R1.3-P hard projection.

The baseline is diagnostic, CoreView377-local, and ineligible for deployment,
teacher use, or a paper claim.

## Held Audit

Only after all six model and baseline artifacts are frozen does the auditor
load the held block for each fold and fit camera. It verifies source, sample-
order, carrier-order, pose-manifest, normalization-mask, prediction-mask, and
eligibility fingerprints before evaluating gates.

The auditor recomputes from frozen renderer streams:

```text
outer and boundary normalized-flicker gains
target mean response
selection soft-IoU drop
target-response normalized-flicker ratio (visibility response)
maximum adjacent gate change
A5 topology/bounds/frozen-part/coverage guards
```

Visibility response uses the existing constrained-A7 definition: target
contribution divided by target pixel count, summarized as per-camera normalized
flicker, candidate divided by A5 baseline. It must not be approximated from a
new proxy.

Held R1.3-G gates are used only after model metrics are frozen to report gate
MAE, temporal-difference MAE, and renderer gain recovery. These diagnostics do
not alter the verdict.

## Promotion Gates

The learned model must pass every unchanged formal gate:

```text
mean outer gain >= 0.5%
mean boundary gain >= 0.5%
mean outer and boundary gains both strictly improve over R1.1-F1
outer/boundary positive record fraction >= 0.90
outer/boundary record q10 >= 0
worst outer/boundary regression <= 0.5%
minimum target response >= 0.99
maximum visibility response ratio <= 1.0
maximum selection soft-IoU drop <= 0.005
maximum adjacent gate change <= 0.02
A5 topology, coverage, frozen parts, and weight upper bounds pass
```

For each of the four fit cameras, the mean across its six held blocks must have
strictly positive outer and boundary gain.

To justify the learned architecture, its mean outer gain and mean boundary gain
must each be strictly greater than both frozen R1.2-B and the fixed `k=4`
nearest-neighbor baseline. Comparisons use a `1e-9` numerical tolerance. Fit and
held teacher gate MAE, temporal-difference MAE, and renderer gain gap are
reported but have no post-hoc threshold.

## Verdicts And Stop Rules

R1.4-VP has exactly three execution outcomes:

- `CANARY_PROMOTED`: all learned-model formal, per-camera, baseline superiority,
  isolation, and guard checks pass. The root marker is `.completed`.
- `CANARY_NEGATIVE`: execution and audit are correct but any learned-model gate
  fails. The root marker is `.rejected`.
- `TRAINING_ERROR`: source, isolation, teacher LP, determinism, training,
  serialization, or numerical integrity fails. The root marker is `.failed`.

A negative canary cannot be retrained with a changed epoch, seed, feature group,
width, hidden size, residual range, loss weight, or checkpoint. Any such change
requires a new preregistration. An implementation-only error may be repaired
and rerun under the identical contract with the error artifact retained.

No outcome directly enters Task 12. `CANARY_PROMOTED` permits a separate design
for c17-c20 validation and multi-subject LOSO. It does not permit opening
c17-c23 under this runner.

## Artifacts

The frozen output root is:

```text
exp/acceptdata/a7c_r1_4vp_oracle_distilled_view_pose_377_v1/
```

Required outputs are:

```text
teachers/fold_0..5/teacher.npz
teachers/fold_0..5/certificates.json
teachers/summary.json
training/fold_0..5/model.pt
training/fold_0..5/predictions.npz
training/fold_0..5/summary.json
training/summary.json
nearest_neighbor/fold_0..5/predictions.npz
nearest_neighbor/fold_0..5/summary.json
nearest_neighbor/summary.json
audit/held_block_summary.json
summary.json
runner.log, runner.pid, started_utc.txt, ended_utc.txt
exactly one root marker: .completed, .rejected, or .failed
```

All NPZ/PT/JSON artifacts propagate `deployment_eligible=false`,
`teacher_eligible=false`, and `paper_test_eligible=false`.

## Verification Requirements

Focused tests must cover:

- exact fit/held mask disjointness and held NaN teacher isolation;
- exactly 120 unique fit teacher segments and 24 held audit records;
- boundary capacity request construction without held endpoints;
- lexicographic teacher feasibility, residual, and repeat determinism;
- no forbidden identifier in model or nearest-neighbor inputs;
- fit-only F3, pose, and distance normalization;
- fixed parameter count `<=50,000` and zero-initialized residual head;
- manifest-ordered camera-block packing, hidden-state reset, and deterministic
  repeated inference;
- explicit view-pose interaction response when either branch is zeroed;
- hard-projection topology, proxy-target, and jump preservation;
- nearest-neighbor `k=4`, exact-match behavior, and fit-only lookup;
- auditor independence from trainer metrics and held teacher labels;
- exact visibility-response recomputation from unweighted contributions;
- source, mask, order, pose, normalization, and eligibility mismatch rejection;
- mutually exclusive terminal markers and restart behavior;
- no c17-c23 path;
- unchanged R1.2-B, R1.3-P, and R1.3-G regression suites.

## Timing Discipline

No completion time is preregistered before implementation. After the teacher
generator is implemented and verified, the first 4-8 fit segment solves provide
an observed per-segment rate. The runner may then report a Beijing completion
estimate covering all 120 teacher segments, six 400-epoch models, baseline, and
audit. An estimate must be labeled as such; only `ended_utc.txt` is the actual
completion time.
