# CoreView377 Occlusion-Aware Consensus Canary Design

## Goal

Test whether the remaining raw contour and foreground PSNR gap is caused by
incorrect per-Gaussian responsibility attribution and diluted per-pose training
exposure. The canary starts from a checkpoint produced entirely by the current
from-zero pipeline and must not load v395 state, held-out camera corrections, or
hand-authored CoreView frame identifiers.

## Evidence

- The previous residual reallocation run reported all 50,000 Gaussians visible
  on every sample and nearly identical evidence percentiles. Its visibility
  signal was `radii > 0`, which means projected into the image rather than
  surviving alpha compositing and occlusion.
- Its evidence ranking also added a visibility-dependent constant floor, so the
  donor/parent ranking was dominated by uniform exposure and opacity instead of
  residual responsibility.
- The existing boundary point score samples a 2D disagreement map at each
  projected Gaussian center. It does not distinguish a front contributor from
  an occluded Gaussian at the same pixel.
- v395 trained on ten temporal frames, while the current sequence contains 570
  training frames. The current 82k path provides about 144 observations per
  frame, versus roughly 13,600 per frame in the historical sparse protocol.
- Appearance-only late refinement improves foreground LPIPS and background
  halo, but the current raw contour remains near 3.14 pixels versus roughly
  2.70 pixels for the historical reference.

## Experiment

### Pose-diverse coreset sampler

Extend the existing residual-balanced multi-view sampler with an optional pose
coreset. Pose vectors come only from training-sequence SMPL metadata. Select a
deterministic set with farthest-point sampling in standardized pose space.

Sampling mixes:

- 65 percent pose-coreset frames;
- 35 percent uniform coverage over all training frames;
- residual weighting within the coreset after a 200-sample warmup;
- four distinct training cameras per selected frame.

The algorithm contains no subject name, frame number, held-out camera ID, or
reference metric.

### Occlusion-aware gradient consensus

Use the rasterizer backward pass as the responsibility oracle. For each of the
four training cameras, compute isolated gradients of the signed boundary loss
with respect to the boundary opacity and scaling residual parameters. Rasterizer
backward already applies alpha compositing and transmittance, so occluded points
receive little or no useful gradient.

For each parameter element:

1. discard gradients below the per-view magnitude quantile;
2. count positive and negative votes independently;
3. accept an update only when at least three of four cameras agree in sign;
4. average accepted gradients and replace ordinary residual gradients before
   the optimizer step.

All legacy boundary support is initialized active for this canary while residual
values remain zero. This makes every Gaussian eligible for attribution without
changing the initial rendering; only consensus-selected residual elements become
nonzero.

### Training regime

Start from the current from-zero 82k canary checkpoint. Train 4,000 sampled views
with four-view accumulation, approximately 1,000 optimizer updates.

- freeze Gaussian xyz, base opacity, base scale, rotation, features, texture,
  pose correction, rigid deformation, and non-rigid deformation;
- train only legacy boundary opacity and scaling residuals;
- disable densification, pruning, opacity reset, camera geometry, camera affine,
  and photometric correction;
- use raw hard masks and existing RGB/boundary losses;
- evaluate same-30 every 500 sampled views;
- select maximum foreground PSNR subject to LPIPS no worse than baseline plus
  0.0005.

## Final Audit And Gates

The launcher evaluates the selected checkpoint on same-30 and original-57 and
renders the same-30 contour audit. The canary is promising only if:

- same-30 foreground PSNR improves by at least 0.03 dB over 22.0143547;
- same-30 foreground LPIPS is at most 0.1266785;
- original-57 foreground LPIPS and PSNR do not regress beyond existing guards;
- mean symmetric contour distance is at most 3.00 pixels;
- mean boundary L1 is at most 0.0680;
- consensus logs show a nonzero but non-global active fraction and at least
  three contributing views per optimizer update.

Failure stops at the canary. Success justifies integrating this stage after the
long-horizon geometry phase and before late-clean, followed by a fresh from-zero
CoreView run and a shortened second-subject validation.

## Non-Goals

- Do not extend the surface-responsibility tether run.
- Do not copy v395 parameters or support assets.
- Do not hard-prune, clone, rebind, or relocate Gaussians.
- Do not use held-out residuals to choose training samples.
- Do not claim cross-subject generalization from CoreView_377 alone.
