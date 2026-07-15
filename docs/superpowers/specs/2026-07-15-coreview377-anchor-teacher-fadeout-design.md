# CoreView377 Anchor Teacher Fade-Out Design

## Goal

Retain the geometry benefit of the monotonic local position teacher before the non-rigid handoff, then stop moving anchors early enough for radiance, opacity, and fully active non-rigid deformation to recover through iteration 8000.

## Evidence

The monotonic teacher improved or matched the neutral image curve through 6500: at 5000 LPIPS was `0.1689627` versus `0.1691349`, and at 6500 it was `0.1628150` versus `0.1629840`. The curve reversed after non-rigid reached full activation at 6500: regression grew to `+0.000724` at 7000 and `+0.001051` at 8000.

The teacher changed only 879 unique Gaussians and preserved global surface distance, yet local opacity still fell. The failure is continued anchor motion while the non-rigid field is fully active, not excessive global geometry change.

## Approaches

### Abrupt stop at 6500

Keep alpha 0.10 through 6500 and disable all later events. This creates a velocity discontinuity exactly at the non-rigid handoff.

### Earlier hard stop at 6000

Provides more recovery time but discards the observed 6000-6500 neutral-or-better interval.

### Linear fade from 6000 to 6500

Keep alpha 0.10 through 6000, then use effective alphas 0.08, 0.06, 0.04, and 0.02 at 6100-6400. Do not schedule an event at or after 6500. This is selected because it preserves the useful interval while making anchor velocity approach zero before full non-rigid activation.

## Runtime

Add optional rigid-deformer configuration:

- `anchor_transport_fade_start_iter: 6000`
- `anchor_transport_fade_end_iter: 6500`

The effective alpha equals the configured base alpha before the fade start, decreases linearly to zero, and is zero at and after the fade end. Transport scheduling skips zero-alpha events. Existing recipes without both fade values retain constant alpha behavior.

The fade-out canary otherwise reuses the monotonic teacher unchanged:

- joints 12-17 only
- same-joint, tangent, normal, and local-offset target gates
- position-only updates
- per-point count cap 8
- fixed 80k topology
- no densify or opacity reset

## Success Criteria

- 5k gate passes.
- No teacher event occurs at or after 6500.
- 8k `FG_LPIPS < 0.1550340652` and `FG_PSNR >= 20.2588196`, or LPIPS improves by at least 0.0005.
- Local tangent improvement is retained without a further local opacity drop relative to the previous monotonic 8k run.
- No traceback, densify, prune, or opacity-reset event occurs.

If the 6500-8000 recovery window removes the LPIPS regression, the same schedule can be continued into the long-horizon recipe. If it does not, anchor motion itself conflicts with the learned image function and the next experiment must use a render-preserving dual-binding path rather than another timing adjustment.
