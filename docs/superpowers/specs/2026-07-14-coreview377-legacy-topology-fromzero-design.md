# CoreView377 Legacy Topology From-Zero Design

## Goal

Recover the historical topology-formation dynamics inside the current training
stack, then run a single automatic from-input pipeline toward or beyond the v395
rendering target.

## Evidence

The uninterrupted fixed-80k path reaches `FG_LPIPS=0.1293520` at iteration 74000,
but its topology remains unlike v395: 80k small, low-opacity points instead of
about 46.8k mature carriers. Continued training to 96k does not improve the best
metric. Surface tethering, checkpoint compression, reallocation, and delayed
non-rigid activation all regress image quality.

The remaining visual error is concentrated in false-positive silhouettes and
occlusion ordering. The May 2026 implementation used binding-aware newborn clone
seeding and directional split behavior that the current generic densify path no
longer executes. Prior native-densify probes therefore do not test the historical
topology dynamics.

## Alternatives Considered

1. Run the complete May 2026 commit in a separate worktree. This has maximum code
   fidelity but also changes the renderer, evaluator, checkpoint format, and all
   later fixes, so it cannot isolate topology behavior.
2. Replace the current `GaussianModel` with the historical file. This would discard
   optimizer restoration and current pointwise state and is unsafe.
3. Add an isolated compatibility mode to the current model. This retains the
   current renderer and training stack while restoring only the missing topology
   actions. This is the selected approach.

## Compatibility Mode

Add `model.gaussian.densify_mode=legacy_20260508`. The default mode remains
unchanged. In legacy mode:

- Gradient-selected clone candidates are not constrained by the modern per-call
  seed cap; the global point budget remains a safety ceiling.
- Recently born risky or refresh-marked small Gaussians can be cloned again within
  a bounded age window. This recreates historical local lineage growth.
- Split offsets for risky/refresh-marked points are projected into the tangent
  plane of their canonical anchor normal. This avoids creating outward shell
  children while retaining tangent detail capacity.
- Newborn lineage, source-joint, refresh, and risk state continues to use the
  current pointwise-state implementation.
- Pruning uses opacity and screen/scale criteria without a fixed minimum-point
  floor or target-count surgery.

The implementation logs `[LegacyDensify]` events with base candidates, seed
candidates, tangent-projected children, and point counts.

## Training Contract

- Input: CoreView_377 data only; no historical subject checkpoint.
- Initialization: 50k uniform canonical SMPL surface samples.
- Densification: iterations 500-12000, every 100 iterations.
- Opacity reset: every 3000 iterations while densification is active.
- Safety ceiling: 70k points; no forced target count.
- Non-rigid activation: existing 3000-6500 smoothstep path.
- No surface tether, compression, competition, residual reallocation, or late
  appearance stage.
- Total training: 80k in one process with automatic best-checkpoint selection.

## Gates

- 8k: `FG_LPIPS <= 0.1555340652`.
- 12k: `FG_LPIPS <= 0.1476944898`.
- Point count must remain between 20k and 70k and all tensors must remain finite.
- A failed image gate stops the process automatically.
- Passing 12k permits uninterrupted continuation through 80k.

The final target is `FG_LPIPS < 0.1273284`, `FG_PSNR > 22.1021`, with mean
symmetric contour distance near or below 2.7 pixels.

## Interpretation

If legacy mode passes the early image gates while producing a compact natural
topology, the missing topology dynamics hypothesis is supported. If it fails by
8k or 12k, the old topology actions are not sufficient in the current renderer,
and the experiment stops before committing to a long run.
