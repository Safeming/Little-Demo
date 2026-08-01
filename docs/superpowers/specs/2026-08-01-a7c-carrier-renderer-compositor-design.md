# A7c Carrier-Resolved Renderer-Visible Compositor Design

## Status And Scope

This document preregisters the CoreView377 A7c canary before renderer probes or
learned compositor results are opened. The canary uses only development cameras
`c01/c05/c09/c13/c17/c18/c19/c20` and frames `0:570:5`. Cameras `c21-c23` are
forbidden. A passing canary authorizes a separate five-subject LOSO design; it
does not authorize Task12 or a paper claim.

The frozen A5 checkpoint, label bank, lower selection topology, coverage rules,
and frozen parts remain immutable. A7c changes only the rendered edit delta.

## Evidence From R0

The exact five-subject oracle capacity result is:

```text
global target-coupled gate:       0/5 subjects promoted
aggregate two-region ray gate:    0/5 subjects promoted
carrier-resolved point oracle:    4/5 subjects promoted
```

Therefore the canary must retain carrier resolution. A global gate, a two-region
scalar gate, free per-camera parameters, and free per-frame parameters are not
valid alternatives.

## Runtime Architecture

A7c is a deterministic two-pass compositor.

1. The probe pass renders the unedited frozen A5 avatar and exports runtime
   carrier features plus an alpha/transmittance-weighted A5 lower-support map.
2. A shared carrier MLP predicts one attenuation gate per selected lower
   carrier in `[0.9, 1.0]`.
3. The same rasterizer renders the lower-support-weighted gate numerator. Its
   ratio to the lower-support map produces a per-pixel carrier gate.
4. A target-preserving envelope pushes the gate to one as A5 lower support
   approaches one.
5. The compositor applies the resulting gate only to the A5 edit delta:

```text
delta_a5 = edited_a5 - unedited_a5
carrier_gate = render(lower_weight * point_gate) / max(render(lower_weight), eps)
target_gate = 1 - (1 - carrier_gate) * (1 - lower_support)
output = unedited_a5 + target_gate * delta_a5
```

The numerator and denominator use identical geometry, opacity, covariance, and
camera state. Consequently both are accumulated with the rasterizer's real
alpha/transmittance ordering without requiring a CUDA rasterizer change.

The canary may use extra renders for auditable buffers. Runtime fusion and speed
optimization are outside this experiment.

## Feature Contract

The carrier MLP receives only values available at the current render call.

Dynamic per-carrier features:

- binary renderer visibility;
- `log1p` projected radius;
- normalized camera-space center `(x/z, y/z, log(z))`;
- unit view direction in local/world coordinates;
- Gaussian opacity and projected footprint proxy;
- frozen A5 lower weight and selection-mask bit.

Static per-carrier features:

- body-scale-normalized canonical coordinates;
- compact frozen semantic probabilities;
- dominant binding joint and compact skinning weights when already available in
  the frozen asset.

All continuous features are clipped to preregistered finite ranges and
normalized with statistics fit on `c01/c05/c09/c13` only. Missing optional
static fields are represented by zeros plus an availability bit; schema shape
cannot change after the canary starts.

Forbidden model inputs are camera ID, frame index, subject ID, Gaussian ID,
image name, previous-frame state, ground-truth masks, target/outer/boundary
contribution labels, oracle gate values at inference, and any c21-c23 data.
Camera and frame indices may appear only in manifests and split logic.

## Model And Bounds

The first canary has one fixed model:

```text
shared MLP: input -> 32 -> 16 -> 1
activation: SiLU
output: point_gate = 0.9 + 0.1 * sigmoid(logit)
parameter sharing: all carriers and all cameras
runtime state: none
ID embeddings: none
```

The final layer is initialized so the initial point gate is at least `0.999`.
The model cannot alter geometry, opacity, covariance, appearance, semantic
weights, the A5 bank, or selection topology.

## Probe And Teacher Artifacts

The probe artifact stores float16 feature tensors shaped `[sample, carrier,
feature]`, carrier IDs, camera/frame indices for splitting, schema metadata, and
input/output fingerprints. The teacher artifact stores the R0 point-oracle gate
for construction samples and the exact carrier set. Teacher and probe carrier
IDs must match exactly.

Each artifact records the checkpoint SHA-256, A5 bank SHA-256, R0 contract and
summary fingerprints, source camera/frame protocol, git commit, command, and
`paper_test_eligible=false`.

## Canary Splits

Camera roles are fixed:

```text
fit cameras:   c01, c05, c09, c13
audit cameras: c17, c18, c19, c20
frames:        0:570:5
blocks:        six contiguous blocks per camera
```

Feature normalization and optimizer fitting use fit cameras only. Six
leave-one-contiguous-block folds on fit cameras audit temporal extrapolation.
Audit cameras never contribute gradients, normalization statistics, early
stopping, threshold selection, or architecture selection. One fixed training
budget is selected before audit metrics are opened.

This is a development canary, not an untouched-subject or paper-test result.

## Training Objective

Optimization is lexicographic in effect, implemented by rejecting checkpoints
that violate a higher-priority guard:

1. finite outputs, gate bounds, A5 topology equality, and frozen-part equality;
2. target response and soft-IoU guards on fit-camera validation blocks;
3. robust Huber regression to the carrier-resolved R0 teacher on permitted fit
   samples;
4. renderer outer/boundary normalized-flicker reduction;
5. gate Lipschitz penalty under small feature perturbations and adjacent
   construction samples;
6. mean attenuation and model size.

No loss term may use camera/frame ID as a predictor. Ground-truth region masks
and renderer contribution evidence are supervision only and are unavailable to
the runtime model.

## Canary Promotion Gates

Every saved checkpoint must satisfy:

```text
selected lower carrier IDs exactly equal A5 topology
all selected lower point gates in [0.9, 1.0]
hair, face, upper, shoes, and skin exactly equal A5
selection soft-IoU drop <= 0.005
minimum target response >= 0.99
maximum adjacent point-gate change <= 0.02
```

The canary passes only if both the six held-block audit and the four held-camera
audit satisfy:

```text
mean outer flicker gain >= 0.5%
mean boundary flicker gain >= 0.5%
outer positive-block fraction >= 0.90
boundary positive-block fraction >= 0.90
outer/boundary 10th-percentile block gain >= 0
worst outer/boundary block regression <= 0.5%
all spatial, coverage, topology, and frozen-part guards pass
```

Metrics use the existing normalized-flicker definition and subject/camera-equal
aggregation. Frames, carriers, and blocks are not treated as independent
subjects.

## Stop Rules

- A failed probe schema or fingerprint check stops training.
- A held-block failure stops before held-camera audit is opened.
- A held-camera failure stops the route; no threshold, feature, width, depth,
  gate range, or split is changed in place.
- No c21-c23 result may select architecture, features, training budget, or
  checkpoint.
- A passing CoreView377 canary only authorizes a new five-subject LOSO
  preregistration. It does not authorize Task12.

## Verification Requirements

Focused tests must cover probe schema validation, forbidden-field rejection,
train-only normalization, bounded stateless MLP output, carrier-ID equality,
transmittance-weighted gate composition, target-envelope limiting cases,
contiguous split integrity, and promotion aggregation. A deterministic dry run
must finish before the full CoreView377 probe and training queue is launched.
