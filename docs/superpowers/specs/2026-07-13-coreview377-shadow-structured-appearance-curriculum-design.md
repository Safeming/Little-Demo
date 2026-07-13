# CoreView377 Shadow Structured Appearance Curriculum Design

## Goal

Build a one-command CoreView_377 from-zero training path that keeps the verified 80k neutral 64k Gaussian/deformation trajectory while adding the structured and high-frequency appearance capacity present in v395. Historical v395 checkpoints are diagnostic references only and are never loaded as training state.

The first target is to beat the current from-zero best `FG_LPIPS=0.1304382086`. The long-term target remains v395-level `FG_LPIPS~=0.1273284` and `FG_PSNR~=22.1021`.

## Root Cause

The residual surface-reallocation run moved local Gaussian statistics much closer to v395 but failed its 8k image gate. At 7.5k its head opacity and scale were already close to v395, yet `FG_LPIPS=0.1572829`. This confirms that v395-like static carrier statistics are not sufficient.

The current from-zero checkpoints contain about 18,595 texture parameters and no structured-trunk or high-frequency parameters. The v395 diagnostic checkpoint contains about 2,042,606 texture parameters, including about 1,876,025 structured-trunk parameters and 150,607 high-frequency parameters. Late component transplantation has already failed because appearance, Gaussian state, deformation, visibility, and camera-conditioned color are co-adapted.

The missing capability must therefore be trained from zero along the main trajectory, but it cannot be allowed to perturb the rendered image before it has demonstrated useful residual predictions.

## Approaches Considered

### A. Recommended: shadow residual branch with metric-gated handoff

Instantiate the v395-capable structured appearance modules from zero. Train them first as a bounded shadow RGB-residual predictor whose output is evaluated but not composed into the optimization render. Activate the branch only after its composed validation render improves over the base render.

### B. Enable the full v395 appearance stack at iteration zero

This most directly matches v395 capacity, but approximately two million random parameters would compete with unstable early Gaussian, binding, and deformation states. Previous early structured/owner experiments showed collapse around the first few thousand iterations, so this is too risky for the primary run.

### C. Continue carrier-topology experiments

A 50k no-reallocation control would isolate the last experiment scientifically, but the stronger evidence is that closer carrier state still produced worse images. This can remain a diagnostic control and is not the next quality path.

## Architecture

### Stable base path

Reuse the exact 80k neutral-longhorizon64k recipe and seed that produced `FG_LPIPS=0.1304382086`:

- no densification, pruning, opacity reset, compression, or carrier reallocation;
- unchanged Gaussian, deformation, pose-correction, converter, and loss schedules through 64k;
- structured modules exist from initialization but cannot affect the base render before activation.

### Shadow structured residual

The texture model exposes two outputs from the same forward pass:

- `base_rgb`: the existing shallow-MLP color;
- `shadow_residual`: a bounded RGB residual produced by the structured branch.

During shadow warmup, the training render remains `base_rgb`. The structured branch receives a detached residual target derived from ground-truth color minus the base prediction on visible foreground samples. Its loss is masked, robust, and normalized so it cannot change Gaussian, deformation, or base-texture gradients.

At evaluation gates, render both:

- base output with shadow scale `0`;
- composed candidate with the configured probe scale.

The evaluation stores both FG LPIPS and FG PSNR in a machine-readable gate record.

### Metric-gated handoff

Training stages are iteration-based and automatic:

1. `0-8k`: reproduce the base trajectory; train only the shadow shared/structure residual path with zero render contribution.
2. `8k-12k`: continue shadow training and evaluate base versus composed output at 8k, 10k, and 12k.
3. At 12k, activate only when composed FG LPIPS beats base by at least `0.001` and no validation camera regresses by more than `0.002`.
4. `12k-20k`: crossfade structured contribution from `0` to a bounded maximum while retaining the base path.
5. `20k-40k`: jointly optimize the active structured branch with reduced Gaussian learning rates and no topology changes.
6. `40k-64k`: enable the high-frequency residual sub-branch with a separate low-amplitude ramp.

If the 12k shadow candidate fails, write `SHADOW_APPEARANCE_GATE_FAILED`, preserve the best base checkpoint and diagnostics, and stop. If an active-stage absolute LPIPS gate fails, roll the contribution back to the last accepted scale and stop rather than continuing a degraded 64k path.

## Gradient Isolation

The shadow loss must update only explicitly selected structured/high-frequency texture parameters. The base render, Gaussian parameters, deformation parameters, pose correction, base texture MLP, and latent table must receive no shadow-loss gradient. This isolation is enforced by parameter allowlists and a unit test that inspects gradients after a backward pass.

## Gates

- 8k base FG LPIPS must remain at or below `0.1565`.
- 12k composed FG LPIPS must improve over base by at least `0.001`.
- No individual validation camera may regress by more than `0.002` FG LPIPS at handoff.
- 16k active output must remain at or below `0.1430`.
- 32k active output must beat the recorded neutral baseline at the same iteration.
- Final FG LPIPS must beat `0.1304382086`; v395-level success is at or below `0.1273284` with FG PSNR near or above `22.1021`.

Image metrics are primary. Parameter energy, carrier statistics, and structured-branch activation are diagnostics and cannot override a failed image gate.

## Validation

Unit tests cover bounded residual composition, zero-contribution identity, shadow gradient isolation, schedule interpolation, per-camera handoff gating, rollback, and marker creation. Pipeline tests verify the 80k from-zero contract, absence of checkpoint loading and topology changes, 64k horizon, automatic shadow evaluation, and failure/continuation behavior.

A GPU smoke run must prove that both base and composed evaluation paths render, shadow-only gradients are nonzero, base-path gradients remain isolated, and the formal launcher reaches live training before it is detached.
