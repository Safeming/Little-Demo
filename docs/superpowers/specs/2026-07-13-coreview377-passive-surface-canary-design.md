# CoreView377 Passive Surface Canary Design

## Goal

Determine whether the 50k surface-carrier trajectory can match or beat the verified neutral trajectory when active carrier competition and forced residual reallocation are removed. The diagnostic resumes the existing from-zero residual-surface checkpoint at global iteration 5k, preserves all optimizer and scheduler state, and runs automatically to global iteration 12k unless an image gate rejects it at 8k.

This checkpoint continuation is diagnostic only. A successful recipe will be rebuilt as a one-command input-to-final from-zero pipeline and will not depend on the 5k diagnostic checkpoint.

## Evidence

- The neutral 80k focus initialization allocates 31.1 percent of points to head joint 15 versus 9.0 percent in v395 and keeps local median scale near 0.001 versus about 0.0045 in v395.
- From neutral 8k through 96k, point allocation and local carrier geometry barely change; feature energy is the main quantity that continues to grow.
- The 50k surface-carrier run improves mean contour distance from 3.603 px to 3.404 px and substantially improves local state distance, but early per-face opacity competition causes an image-quality gap.
- The residual-surface run disables competition and beats the neutral LPIPS trajectory through 6.5k. Its evidence p50 and p90 are nearly identical, yet it forcibly reallocates 250 points every 500 iterations and misses the 8k gate.
- Structured and perceptual-band shadow appearance canaries did not produce a meaningful LPIPS gain.

## Root-Cause Hypothesis

Surface initialization is useful, but active topology/support rewriting occurs before radiance, deformation, visibility, and occlusion ordering are stable. The forced reallocation step is therefore the next variable to remove. If the passive continuation remains competitive through 8k and 12k, active reallocation caused the regression. If it still diverges after 6.5k, the next isolated variable is the non-rigid activation handoff.

## Diagnostic Architecture

Start from:

`exp/zero_train_to_v395/coreview377_residual_surface_reallocation_20260713_bjt/run_20260712_211702_bjt/neutral_longhorizon_fromzero/ckpt5000.pth`

Continue for 7k local iterations, producing global schedule iterations 5001 through 12000.

Preserve:

- Gaussian optimizer state;
- converter optimizer and scheduler state;
- Gaussian and converter learning-rate schedules;
- random seed and model configuration;
- the existing low-weight all-joint local anchor shell;
- the original 3k-6.5k smooth non-rigid activation.

Disable:

- surface-carrier competition;
- residual surface evidence updates and reallocation;
- densification, pruning, opacity reset, structured appearance, and any new topology mechanism.

Run a diagnostic validation immediately after loading iteration 5k. Evaluate at global 6k, 6.5k, 7k, 7.5k, 8k, 10k, and 12k. Save global 8k and 12k checkpoints.

## Gates

The 8k image gate rejects the run when foreground LPIPS exceeds `0.1555340652`. The 12k image gate rejects the run when foreground LPIPS exceeds `0.1476944898`.

The report must also retain the matched reference values:

- 6k: `0.1639306396`;
- 6.5k: `0.1629840285`;
- 7k: `0.1571950614`;
- 8k: `0.1550340652`;
- 12k: `0.1471944898`.

Image metrics remain authoritative. Local state statistics and contour diagnostics explain the result but cannot override a failed LPIPS gate.

## Success Criteria

- The loaded 5k metric reproduces `FG_LPIPS=0.1682725102` within normal evaluation precision.
- No optimizer or scheduler silently restarts.
- No reallocation, competition, densification, pruning, or opacity reset event occurs.
- The 8k and 12k gates pass.
- If 12k passes, the experiment identifies passive surface carriers as the recipe to rebuild from zero through 64k.
- If the run fails only after 6.5k, the next experiment changes only the non-rigid activation handoff.

## Validation

Pipeline tests verify the exact checkpoint, 7k local horizon, complete optimization-state restoration, disabled active carrier mechanisms, global-iteration validation schedule, topology safety, and gate thresholds. A GPU smoke verifies checkpoint loading, start-metric reproduction, optimizer restoration, schedule offset, and absence of active carrier logs before the formal detached canary starts.
