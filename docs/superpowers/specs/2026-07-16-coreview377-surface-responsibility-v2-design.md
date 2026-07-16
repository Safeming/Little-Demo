# CoreView377 Surface Responsibility V2 Design

## Goal

Test whether the remaining contour gap comes from the current 80k
`focus_head_hands` initialization assigning too many microscopic, weak Gaussians
to local regions. The experiment is a 12k from-zero canary and must not depend on
any historical subject checkpoint.

## Evidence

- The current 83.5k candidate has 80,000 points and assigns 31.1 percent to the
  head, versus 46,801 total points and 9.0 percent head support in v395.
- Current head median scale is about 0.00103 versus 0.00453 in v395; head mean
  opacity is 0.091 versus 0.250.
- This distribution is already present at 8k, so late refinement cannot repair it.
- Late-clean and residual-balanced improve LPIPS and PSNR while contour remains
  near 3.15 pixels because geometry and deformation are frozen.
- Hard compression damages the co-adapted image function, so the responsibility
  topology must form naturally from initialization.

## Experiment

Use the existing deterministic `surface_carrier_v1` sampler with 80,000 points:

- surface-area sampling with bounded head, shoulder, and hand fractions;
- anisotropic tangent/normal scales from represented surface area;
- no `focus_head_hands` multiplier;
- no competition, residual reallocation, densification, pruning, or opacity reset;
- the proven neutral long-horizon optimizer and standard non-rigid schedule;
- a weak all-joint anchor shell through 12k, with strong normal and very weak
  tangent pressure.

The canary evaluates at 1k, 2k, 3k, 4k, 5k, 6.5k, 8k, 10k, and 12k, and saves
5k, 8k, and 12k checkpoints. It remains subject-agnostic: no held-out camera,
frame, pixel location, or CoreView-specific joint subset enters the option.

## Interpretation

Continue this recipe beyond 12k only when image quality stays within normal
variance of the matched neutral trajectory and contour/state diagnostics improve.
If the carrier state improves but the image trajectory fails, the next mechanism
is delayed contribution-aware responsibility consolidation, not stronger tethering
or post-hoc boundary surgery.

## Success Signals

- 5k, 8k, and 12k foreground LPIPS remain within 0.0015 of the matched neutral
  trajectory.
- Head allocation stays near the configured 10 percent rather than 31 percent.
- Head median surface distance is below 0.06 at 12k.
- Raw symmetric contour and render-to-GT edge distance improve by at least 5
  percent relative to the matched focus-initialized checkpoint.
- No topology-changing event or historical checkpoint load occurs.

