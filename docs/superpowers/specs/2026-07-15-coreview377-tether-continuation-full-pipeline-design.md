# CoreView_377 Tether Continuation And Full From-Zero Pipeline Design

## Goal

Close the remaining raw-view contour and PSNR gap without sacrificing the current perceptual result, then freeze the successful recipe into one unattended from-zero command and rerun CoreView_377 from input data.

The final production path must not load a v395 or v43 checkpoint, must not train on held-out RGB cameras 21-23, and must not require manual checkpoint selection.

## Evidence And Root Hypothesis

The current 80.5k candidate reaches `FG_LPIPS=0.1261183470`, better than faithful v395 `0.1273224354`, but remains behind v395 with camera geometry disabled on raw foreground PSNR and contour alignment:

- current: `FG_PSNR=21.8957710`, mean symmetric contour distance `3.1744 px`;
- raw v395: `FG_PSNR=21.9559345`, mean symmetric contour distance `2.7790 px`.

The current candidate also has a larger canonical surface distance than v395. Global render-scale expansion was tested from `1.00` through `1.10`; it slightly reduced foreground L1 but monotonically worsened contour distance and halo, so inference footprint scaling is rejected.

Hard compression, abrupt rebind, anchor transport, and late checkpoint surgery are rejected because they either degraded LPIPS or moved state statistics without improving the image function.

The existing surface-coherent anchor tether is the only intervention that improved surface attachment without an image-quality collapse:

- neutral 64k baseline: `FG_LPIPS=0.1304382086`;
- tether 64k: `FG_LPIPS=0.1304872483`, `FG_PSNR=21.7672939`;
- tether head surface-distance p50: `0.0900`, improved from the untethered trajectory while retaining the learned radiance function.

The single hypothesis is therefore: the tether creates a better geometric starting state, but its benefit was never carried through the proven optimizer-preserving continuation, late-clean, and residual-balanced stages.

## Phase 1: Continuation Canary

Use the existing from-zero tether 64k checkpoint only as a diagnostic time-saving start:

```text
exp/zero_train_to_v395/coreview377_surface_coherent_anchor_tether_20260711_bjt/
  run_20260711_222829_bjt/neutral_longhorizon_fromzero/ckpt64000.pth
```

Run an unattended chain:

1. optimizer-preserving continuation from global 64k to 96k;
2. automatic evaluation of saved 72k, 80k, 88k, 96k and continuation-best checkpoints;
3. automatic Pareto selection using same-30 LPIPS/PSNR plus raw contour metrics;
4. 4k late-clean refinement from the selected continuation checkpoint;
5. 3k residual-balanced refinement;
6. same-30, original-57, and raw contour validation with camera geometry disabled.

All avatar geometry, optimizer, and scheduler state handling must match the already verified continuation and late-refine implementations. No v395 state may be copied into the candidate.

## Canary Gates

The canary passes only when one final checkpoint satisfies all conditions:

- same-30 `FG_LPIPS <= 0.1261183471`;
- same-30 `FG_PSNR >= 21.9559345`;
- original-57 `FG_LPIPS <= 0.1288665946`;
- original-57 `FG_PSNR >= 21.7641456`;
- mean symmetric contour distance `<= 2.90 px`;
- mean boundary L1 `<= 0.06720`;
- no held-out camera geometry or camera-affine correction is active;
- no single saved-stage transition causes same-30 LPIPS regression greater than `0.003`.

The `2.90 px` canary threshold is intentionally between current `3.1744 px` and raw v395 `2.7790 px`. The full rerun retains raw v395 `2.7790 px` as the stretch target.

If the canary fails, the launcher writes a rejected selection report and stops. It must not start the expensive full from-zero rerun.

## Phase 2: Frozen Full From-Zero Rerun

When Phase 1 passes, launch a new CoreView_377 run from input data with no checkpoint dependency:

```text
input data
  -> 64k neutral long-horizon with early surface-coherent tether
  -> optimizer-preserving continuation to 96k
  -> automatic multi-metric continuation selection
  -> 4k late-clean refinement
  -> 3k residual-balanced refinement
  -> dual raw/legacy reporting and final automatic selection
```

The full launcher must derive every stage checkpoint from its own run directory. Historical experiment paths are prohibited except for read-only benchmark values in the final report.

## Selection Contract

Checkpoint selection uses two independent profiles:

- `avatar_raw`: camera geometry disabled; this controls checkpoint acceptance and cross-subject claims.
- `legacy_v395`: faithful historical reporting; this never controls candidate selection.

A candidate can replace its baseline only if it improves or preserves both same-30 LPIPS and PSNR within configured tolerances, passes original-57 guards, and improves the contour hard score. An absolute legacy-v395 PSNR gate must never force fallback to a strictly better raw candidate.

## Portability

Subject-specific constants are limited to launcher inputs such as `SUBJECT`, data root, seed, and optional benchmark thresholds. The training options may not contain CoreView_377 frame IDs, held-out camera quality weights, v395 checkpoints, or target-camera learned parameters.

After CoreView_377 reproducibility is confirmed, the same launcher can be invoked for another subject by changing `SUBJECT`; cross-subject quality remains unproven until that second run is completed.

## Verification

Implementation must follow test-first development and include:

- selector tests proving a raw-metric improvement is not rejected by legacy thresholds;
- launcher tests proving the canary stops before Phase 2 on failure;
- launcher tests proving Phase 2 has no historical checkpoint dependency;
- tests proving camera geometry remains disabled in raw validation;
- Bash syntax validation and a short CUDA smoke covering all stage handoffs;
- fresh focused regression tests before the formal launch.

## Runtime Estimate

Based on measured CoreView_377 throughput:

- Phase 1 continuation canary and tail: approximately 7 hours;
- Phase 2 full from-zero chain: approximately 18 hours;
- total if the canary passes: approximately 25 hours, with a 1-2 hour variance from evaluation and I/O.

The formal Beijing completion time is calculated from the actual detached launch timestamp after implementation and smoke verification.
