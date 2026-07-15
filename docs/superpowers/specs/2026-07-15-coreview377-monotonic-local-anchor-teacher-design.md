# CoreView377 Monotonic Local Anchor Teacher Design

## Goal

Repair the failed global anchor-transport canary without disturbing the deformation and visibility state that the neutral from-zero path already learns. The new 8k canary must improve local binding geometry only when a fresh surface target is demonstrably better, while preserving the active skinning, normal, semantic, and discrete face state.

## Evidence From The Failed Canary

The first annealed transport run finished at `FG_LPIPS=0.1552536935`, slightly worse than the same-seed neutral 8k baseline `0.1550340652`. It reduced the global tangent p90 from `0.332189` to `0.323484`, but worsened tangent p50 from `0.154581` to `0.158540`, surface-distance p50 from `0.0822` to `0.0892`, and global opacity mean from `0.1253` to `0.1217`.

Each event reduced the selected tail, but every fresh target was accepted and the implementation blended anchor weights and normals while replacing discrete face metadata immediately. The image model compensated for the resulting deformation/visibility change by lowering opacity. The failure is therefore target validity and ownership scope, not insufficient transport strength.

## Considered Approaches

### Stronger global EMA

Increase alpha or the selected-point cap. This would amplify the exact mechanism that already reduced opacity and worsened the median, so it is rejected.

### Full dual-binding render path

Persist complete old and fresh bindings and blend two deformation paths during rendering. This is expressive but changes a large checkpoint and rendering surface before the fresh target itself has been proven valid.

### Monotonic local position teacher

Restrict candidates to joints 12-17. Use the fresh search only as an `anchor_xyz` teacher. Evaluate that target with the old anchor normal and accept it only when tangent, normal, and total local-offset geometry are all monotonic. Blend only `anchor_xyz`; keep old weights, normal, semantic state, dominant joint, face IDs, vertices, and barycentric coordinates. Track per-point accepted transport count.

This is selected because it isolates stale anchor position while preserving the deformation and visibility fields implicated by the failed run.

## Runtime Behavior

The existing transport schedule remains opt-in and gains:

- `anchor_transport_joint_ids: [12, 13, 14, 15, 16, 17]`
- `anchor_transport_position_only: true`
- `anchor_transport_require_same_joint: true`
- `anchor_transport_tangent_improvement_ratio: 0.02`
- `anchor_transport_normal_regression_tolerance: 0.001`
- `anchor_transport_local_improvement_ratio: 0.01`
- `anchor_transport_max_accept_count: 8`

At an eligible iteration:

1. Rank only configured-joint points above the tangent threshold.
2. Exclude points whose accepted transport count reached the cap.
3. Search a fresh surface target for the bounded subset.
4. Re-evaluate fresh `anchor_xyz` using the old normal and old deformation fields.
5. Accept only same-joint targets whose tangent decreases by at least 2%, normal distance grows by at most 0.001, and total local offset decreases by at least 1%.
6. Blend only accepted `anchor_xyz` by alpha 0.10 and persist an incremented `anchor_transport_count`.
7. Leave rejected points and all non-position binding fields unchanged.

Logs report selected, accepted, rejection reasons, before/after tangent, normal, local offset, and transport-count saturation.

## Experiment

Use the exact same seed-20260710 80k fixed-topology neutral stack and 8k evaluation schedule. Densification, pruning, opacity reset, carrier losses, and tether losses remain disabled. The formal canary stops at 8k.

## Success Criteria

All conditions are required:

- 5k and 8k image gates do not fail.
- 8k `FG_LPIPS < 0.1550340652`, not merely within tolerance.
- 8k `FG_PSNR >= 20.2588196` or LPIPS improves by at least 0.0005.
- Local and global tangent p50 and p90 do not regress from neutral 8k.
- Surface-distance p50 and opacity mean do not regress from neutral 8k.
- Logs contain nonzero accepted targets and no densify/reset events.

If accepted targets are nearly zero, the fresh semantic surface search is not a valid teacher and the next step is a separate canonical carrier surface rather than looser gates. If geometry improves but image quality does not, proceed to a full old/fresh dual-render path with per-view image gating.

## Failure Handling

Disabled behavior remains identical to existing recipes. Empty or fully rejected events are no-ops. Non-finite targets are rejected. Pending partial refreshes take priority. The image gate writes `MONOTONIC_ANCHOR_TEACHER_GATE_FAILED` and terminates the canary on material regression.
