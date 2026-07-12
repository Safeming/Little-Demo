# CoreView377 Residual Surface Reallocation Design

## Goal

Build a one-command CoreView_377 from-zero pipeline that retains the validated 50k surface-area initialization while replacing per-face opacity competition with multi-view evidence-driven, cross-face carrier reallocation. The pipeline runs a 16k canary, stops automatically when it misses the known 64k-baseline trajectory, and otherwise continues uninterrupted to 64k.

## Root Cause

The prior surface-carrier run matched v395 aggregate point count, region fractions, scale, opacity, and feature energy, but its feature support remained spread across about 3,030 effective SMPL faces versus about 894 for v395. Its per-face support budget cannot move capacity between faces, and its early opacity polarization selects winners before radiance and deformation stabilize.

Generic densification is not suitable because it samples children in unconstrained 3D Gaussian space and initializes new optimizer rows with zero moments. Hard pruning also breaks co-adapted radiance, deformation, visibility, and occlusion ordering.

## Architecture

### Evidence bank

From 2k onward, maintain per-Gaussian exponential moving averages of:

- actual raster visibility and screen radius;
- screen-space position gradient norm;
- opacity gradient magnitude;
- feature gradient norm;
- observation count and persistent-view count.

Gradients are contribution-aware: occluded or irrelevant points receive little opacity/feature gradient even when they are inside the camera frustum.

### Fixed-budget reallocation

Between 4k and 16k, every 500 iterations, replace at most 0.5 percent of the 50k slots:

- donors are persistently low-evidence, low-opacity points;
- parents are persistently high-evidence points from different surface faces;
- no parent face may receive more than a bounded number of children per event;
- total point count remains exactly 50k.

A donor slot is rewritten as a child of a selected parent. The child is placed in the parent's tangent plane, keeps zero additional normal offset, copies feature, scale, rotation, semantic/binding state, and inherits the corresponding Adam moments. Parent opacity mass is split gradually between parent and child so the rendered image function does not jump.

### Surface shell

Use the existing differentiable anchor-tether path globally across all joints. Permit tangent motion but keep normal distance inside a soft shell centered on the SMPL surface. The shell remains active through 64k with low weight instead of disappearing at 12k.

### Absolute gates

At 5k, 8k, 12k, and 16k, compare FG LPIPS with the verified neutral 64k baseline:

- 5k: at most 0.1706;
- 8k: at most 0.1565;
- 12k: at most 0.1487;
- 16k: at most 0.1430.

Missing a threshold writes `RESIDUAL_SURFACE_GATE_FAILED`, saves diagnostics, and stops. Passing 16k automatically continues the same process to 64k.

## Success Criteria

- no abrupt LPIPS regression at a reallocation event;
- exactly 50k points throughout training;
- head points remain 8-12 percent;
- head surface distance remains approximately 0.03-0.055;
- top-face concentration rises while feature-support effective-face count falls;
- 16k FG LPIPS is no worse than 0.1430;
- final FG LPIPS beats 0.1304382, with the long-term target below v395's 0.1273284.

## Validation

Unit tests cover evidence accumulation, donor/parent selection, fixed point count, tangent placement, parameter and optimizer-state inheritance, binding-state remapping, and absolute gates. GPU smoke verifies logging, checkpointing, and a live reallocation event before the formal run is detached.
