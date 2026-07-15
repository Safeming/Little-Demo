# Subject-Agnostic Residual-Balanced Multi-View Refine Design

## Objective

Close the remaining CoreView_377 foreground PSNR gap to v395 without encoding CoreView_377 frame IDs, camera IDs, validation errors, or hand-authored ROI weights into training. The experiment is a short canary starting from the best checkpoint produced by the from-zero path. A full from-zero run is launched only after the canary reaches the strict v395 comparison gate.

The same mechanism must be usable for other subjects. CoreView_377 absolute metric targets are benchmark gates, not sampler inputs.

## Evidence And Scope

The current from-zero checkpoint at iteration 77500 has:

- same-30 `FG_LPIPS=0.1261972487`, `FG_PSNR=21.8895034790`;
- original-57 `FG_LPIPS=0.1286665946`, `FG_PSNR=21.7941455841`.

It already beats v395 LPIPS on the same 30 samples, but remains `0.212588 dB` behind v395 PSNR. Previous analysis localized much of the remaining error to temporally difficult poses that recur across held-out cameras. The current late-clean stage samples training observations without residual-aware temporal prioritization.

This canary changes only the training sample distribution. It does not change Gaussian topology, canonical binding, deformation, visibility code, renderer behavior, or validation data.

## Alternatives

### Fixed Endpoint Oversampling

Explicitly increase the probability of frames such as 0, 60, 480, and 540. This is fast but rejected because it encodes CoreView_377 validation findings and cannot be expected to transfer to other subjects.

### Camera-Specific Weights And Photometric Correction

Reuse historical CoreView_377 camera weights and corrected photometric targets. This has produced small gains in older chains, but the weights are subject and capture-rig specific. Photometric correction also changes the training target while evaluation uses the raw target. It is retained only as a later ablation, not used in the primary canary.

### Online Residual-Balanced Multi-View Sampling

Estimate temporal difficulty from raw foreground residuals on training cameras, then use a bounded mixture of uniform and difficulty-weighted frame sampling. Draw several cameras from the selected frame without replacement and accumulate their gradients. This is the selected approach because it addresses cross-camera temporal errors without validation leakage or subject-specific identifiers.

## Sampler Design

Add a `residual_balanced_multiview` training sampler alongside the existing random and frame-balanced samplers.

The sampler groups dataset entries by `frame_id`. It maintains one foreground L1 exponential moving average per frame. All frames start with the same score and remain uniformly sampled during a 200-sample warmup.

After warmup, frame probabilities are recomputed every 100 samples:

1. Divide each frame EMA by the median EMA across observed frames.
2. Clamp the relative difficulty to `[0.5, 2.5]`.
3. Normalize the clamped scores into a difficulty distribution.
4. Mix `70%` uniform probability with `30%` difficulty probability.
5. Project the final distribution with the existing bounded-probability routine so every frame remains at or below `2.5 / frame_count` after normalization.

This keeps most coverage uniform and prevents a few frames from dominating. No frame number is given special treatment.

For every selected frame, sample four distinct training cameras when at least four are available. Accumulate their scaled losses and take one optimizer step after the group. Camera selection is uniform for the primary canary. The sampler exposes its selected frame, accumulation state, probability range, and top difficulty ratios in periodic logs.

The training loop reports the raw, uncorrected foreground L1 for each sampled observation back to the sampler after the forward pass. Validation samples never update sampler state.

## Canary Training Regime

Start checkpoint:

`exp/zero_train_to_v395/coreview377_late_clean_refine_20260715_bjt/run_20260715_151938_bjt/late_clean_refine/best_ckpt.pth`

Train for 3000 sample iterations. With four-camera accumulation this produces approximately 750 optimizer updates.

Keep the accepted late-clean regime:

- pose noise `0` and view noise `5`;
- position, rotation, pose correction, rigid deformation, non-rigid deformation, and texture latent learning rates `0`;
- feature LR `2.8e-4`;
- opacity LR `3.5e-5`;
- scaling LR `1.2e-5`;
- texture LR `5.5e-7`;
- gradient clip `0.0025`;
- no densification, pruning, topology change, opacity reset, or photometric target correction.

Evaluate the same 30-sample split every 500 sample iterations. Save checkpoints at the same interval. Candidate selection maximizes same-30 foreground PSNR subject to `FG_LPIPS <= 0.1273224292`. If no new checkpoint satisfies the constraint, keep the 77500 baseline.

## Acceptance Gates

For CoreView_377, the canary passes only when all conditions hold:

1. same-30 `FG_LPIPS <= 0.1273224292`;
2. same-30 `FG_PSNR >= 22.1020915`;
3. original-57 `FG_LPIPS <= 0.1288665946`;
4. original-57 `FG_PSNR >= 21.7641455841`;
5. rank the individual same-30 samples by their 77500-baseline foreground L1, define the lowest-error 70% as the non-hard subset, and require its mean LPIPS regression to be no greater than `0.0005` and its mean PSNR regression to be no greater than `0.03 dB`.

The final gate uses raw ground truth and does not apply photometric correction. Failure preserves the 77500 checkpoint as the selected output.

## Cross-Subject Generalization

The sampler and option must not contain a subject name, frame ID list, camera ID list, joint ID list, or absolute residual threshold. Dataset metadata supplies the available frames and cameras.

For subjects without a v395-style reference, acceptance is relative to that subject's own pre-refine baseline:

- foreground LPIPS must not regress;
- foreground PSNR must improve by at least `0.03 dB`;
- the non-hard sample subset must remain within the same regression limits;
- sampler coverage logs must show every frame remains reachable and no frame exceeds the configured probability cap.

After CoreView_377 passes, run a shortened canary on one additional subject before treating the stage as transferable. A single CoreView_377 success is not sufficient evidence of generalization.

## Implementation Boundaries

The sampler owns grouping, probabilities, EMA state, accumulation grouping, and logs. The training loop only supplies sample indices and reports raw foreground residuals. Metric gating and checkpoint selection remain in the experiment launcher so the sampler has no knowledge of CoreView_377 or v395.

The implementation must preserve existing sampler behavior when the new mode is disabled.

## Tests

Unit tests cover:

- uniform warmup and complete frame reachability;
- EMA updates from training residuals only;
- probability mixture, normalization, and maximum cap;
- four-camera sampling without replacement;
- optimizer-step and gradient scaling behavior;
- deterministic behavior under a fixed seed;
- unchanged existing random and frame-balanced modes.

Pipeline tests verify that the canary contains no hard-coded difficulty frame or camera list, loads only the from-zero 77500 checkpoint, keeps geometry and deformation frozen, disables topology changes and photometric correction, evaluates both splits, applies the strict gates, and falls back to the baseline on failure.

## Operational Flow

1. Run focused unit and pipeline tests.
2. Run a 12-sample smoke test that exercises residual feedback, grouped accumulation, evaluation, and fallback.
3. Launch the 3000-sample CoreView_377 canary detached.
4. If every acceptance gate passes, integrate the sampler stage into the one-command 0-to-final pipeline and launch the approximately 13-to-14-hour full experiment.
5. If the canary fails, do not launch the full experiment. Use its per-frame metrics and sampler coverage logs to determine whether the remaining PSNR gap is temporal sampling, raw photometric mismatch, or representation capacity.
