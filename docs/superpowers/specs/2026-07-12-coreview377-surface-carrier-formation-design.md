# CoreView377 Surface Carrier Formation Design

## Objective

Build a one-command, from-zero CoreView_377 training path that fixes the carrier-formation failure identified in the 64k baseline and anchor-tether experiments. The first candidate must form compact, surface-aligned, high-coverage local carriers during the first 12k iterations, then automatically continue to the proven 64k long-horizon schedule only when image and carrier gates pass.

The target remains better than the current from-zero result (`FG_LPIPS=0.1304382`) and ultimately close to v395 (`FG_LPIPS~=0.1273284`, `FG_PSNR~=22.1021`). v395 checkpoints may be used only for diagnostics and aggregate target ranges, never as training initialization.

## Root Cause Being Addressed

The current `focus_head_hands` initialization oversamples the head while `create_from_pcd` derives scale from nearest-neighbor distance. This creates too many microscopic head Gaussians. All points start with opacity 0.1 and zero learned features. The current 64k recipe disables densification, pruning, and opacity reset, so the redundant support topology never changes and image gradients remain distributed across many weak carriers.

The initialization cache also keys only on sampling mode and point count. Weight or seed changes can silently reuse an old PLY, invalidating initialization experiments.

## Approaches Considered

### A. Recommended: deterministic surface-area initialization plus early competition

Generate deterministic samples with per-region quotas, initialize anisotropic scales from the surface area represented by each sample, and run gradual opacity/coverage competition during 0-12k. This directly fixes the initialization and frozen-topology causes while keeping changes bounded and diagnosable.

### B. Re-enable generic 3DGS densification

The existing split samples children in unconstrained 3D Gaussian space and has already produced noisy or oversized point sets. It does not preserve surface responsibility and is excluded from the first implementation.

### C. Start dense and use hard compression

Prior checkpoint compression caused an immediate LPIPS collapse. Abrupt deletion breaks co-adapted radiance, deformation, visibility, and occlusion ordering, so this approach is excluded.

## Design

### Deterministic cache identity

The cached initialization filename and manifest must include a stable hash of:

- sampling mode and point count;
- head, hand, and shoulder quotas or weights;
- sampler version;
- random seed;
- subject identifier.

The loader must reject a cached PLY when its manifest does not match the requested configuration. Existing legacy PLY files remain readable only for legacy modes, not for the new carrier sampler.

### Surface-area carrier sampler

Add a new `surface_carrier_v1` initialization mode. It samples SMPL faces deterministically and assigns points by surface area with bounded regional quotas rather than multiplying face weights without a cap.

Initial targets:

- total points: 50,000;
- head: 8-12 percent;
- neck/head/collar combined: 12-17 percent;
- shoulder band: 18-24 percent;
- remaining points distributed by surface area.

For each sample, estimate represented surface area from its source face area divided by the number of samples assigned to that face. Initialize an anisotropic local frame with two tangent scales derived from the square root of represented area and a smaller normal scale. Scale must therefore remain stable when a region receives denser sampling.

All initialization metadata needed for diagnostics must be stored alongside the PLY: source face, represented area, region, seed, and configuration hash.

### Early carrier competition

During 0-12k, group points by source surface patch. Competition acts gradually through differentiable opacity and coverage regularization:

- preserve a per-patch projected support budget;
- penalize redundant active carriers within the same patch;
- enforce a minimum tangent coverage floor;
- do not hard-prune during 0-5k;
- from 5k onward, mark persistently weak carriers inactive in capped batches;
- remove at most 1 percent of total points at one gate event;
- do not delete a carrier unless it has stayed weak across multiple evaluation windows.

The first version does not add new residual-driven births. It validates whether correct initial support and gradual competition solve the dominant failure before introducing a second topology-changing mechanism.

### Training stages

1. `0-2k`: radiance and binding warmup; no deletion.
2. `2k-5k`: ramp surface coverage and redundancy competition.
3. `5k-12k`: capped gradual inactivation/pruning with image gates.
4. `12k`: automatic carrier and image gate.
5. `12k-64k`: continue the verified neutral long-horizon schedule with topology frozen.

The pipeline remains one command and requires no manual checkpoint selection.

## Automatic Gates

At 2k, 5k, 8k, and 12k, record:

- FG LPIPS and FG PSNR;
- foreground and head/neck contour error;
- head, neck, and shoulder active carrier counts;
- opacity and tangent/normal scale distributions;
- `opacity * projected_area` support mass;
- `opacity * projected_area * feature_energy` feature support mass;
- surface distance and edge distance.

The 12k candidate may continue only when:

- no gate has an LPIPS regression larger than 0.01 relative to the preceding gate;
- active local support has not collapsed by more than 10 percent in one gate;
- head feature support mass improves over the existing 12k-equivalent baseline;
- head/neck edge distance improves or stays within 1 percent while support consolidates;
- no NaN, CUDA error, empty point set, or metric-evaluation failure occurs.

On gate failure, save the checkpoint and diagnostics, write `PIPELINE_GATE_FAILED`, and stop without launching the remaining 52k iterations.

## Testing

Unit tests must cover cache hash changes, deterministic sampling, quota bounds, scale invariance under denser regional sampling, anisotropic normal scale, and capped gradual pruning. Pipeline tests must verify one-command stage propagation, automatic stop on a failed gate, continuation on a passed gate, and unambiguous progress logging.

A 20-iteration GPU smoke test must verify checkpoint creation and restore. A short gate smoke must verify metrics and state diagnostics are produced before the full run is launched.

## Expected Interpretation

Success at 12k would validate that the main limitation was carrier support formation. Failure without image collapse would still be useful: it would isolate the next missing mechanism as residual-driven surface birth with state inheritance. A result that only changes static state statistics while LPIPS or edge quality degrades is not considered success.
