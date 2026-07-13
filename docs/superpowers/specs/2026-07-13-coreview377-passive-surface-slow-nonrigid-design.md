# CoreView377 Passive Surface Slow Non-Rigid Design

## Goal

Determine whether the passive surface-carrier path loses rendering quality because
the non-rigid deformation reaches full strength too early.

## Evidence And Hypothesis

The passive surface canary matched the neutral trajectory through global iteration
6500, then regressed sharply at 7000. Its non-rigid gate reached 1.0 at iteration
6500. Removing residual reallocation did not remove the regression, so reallocation
is not the primary cause.

The single hypothesis for this experiment is that full non-rigid activation before
the local carrier, radiance, and visibility functions have co-adapted destabilizes
the surface-locked representation.

## Experiment

Resume from the same from-zero global-5000 checkpoint used by the passive surface
canary and restore the Gaussian optimizer, converter optimizer, and converter
scheduler. Keep every passive-canary setting unchanged except the non-rigid
activation schedule:

- `delay`: 5000
- `activation_start`: 5000
- `activation_end`: 10000
- `activation_curve`: `smoothstep`

This produces gates of 0.0 at 5000, 0.028 at 5500, 0.104 at 6000, 0.216 at
6500, 0.352 at 7000, 0.500 at 7500, 0.648 at 8000, and 1.0 at 10000.

The experiment keeps all-joint passive anchor tethering, disables carrier
competition and residual reallocation, forbids densification and opacity reset,
and preserves all learning rates and optimizer state.

## Evaluation And Gates

Evaluate at global iterations 5000, 5500, 6000, 6500, 7000, 7500, 8000,
9000, 10000, and 12000. The existing neutral baselines remain the decision gates:

- At 8000: `FG_LPIPS <= 0.1555340652`.
- At 12000: `FG_LPIPS <= 0.1476944898`.

The 8000 gate stops the run if delaying the handoff does not repair the known
divergence. Passing 8000 permits the run to reach full activation at 10000 and
tests whether the improvement survives through 12000.

## Interpretation

- Passing 8000 and 12000 supports the handoff-mismatch hypothesis and justifies a
  full from-input recipe with the slower schedule.
- Passing 8000 but failing 12000 means delayed activation protects the carrier but
  full activation still destabilizes it; the next variable is non-rigid amplitude
  or learning-rate coupling, not carrier topology.
- Failing 8000 falsifies activation timing as the primary explanation and returns
  attention to the passive tether or deformation parameterization.

This is a diagnostic continuation. It does not use v395 as a training checkpoint.
