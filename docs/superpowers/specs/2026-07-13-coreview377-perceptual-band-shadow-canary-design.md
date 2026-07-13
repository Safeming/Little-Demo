# CoreView377 Perceptual-Band Shadow Canary Design

## Goal

Determine whether a structured appearance branch can close the remaining CoreView_377 LPIPS gap when it is trained against perceptual and high-frequency errors rather than foreground L1. The canary resumes the verified from-zero 80k baseline at iteration 8k, freezes the established image function, trains only newly initialized shadow appearance parameters through schedule iteration 12k, and makes an automatic base-versus-candidate decision.

This checkpoint resume is diagnostic only. A successful recipe will later be embedded into the one-command input-to-64k from-zero pipeline and will not require the canary checkpoint.

## Root Cause Addressed

The first shadow run showed two independent failures:

- its base trajectory missed the exact 8k baseline because additional module initialization consumed the PyTorch RNG stream used later by pose noise;
- its candidate improved FG PSNR from `20.2157` to `20.2848` while worsening FG LPIPS from `0.1566980` to `0.1582341`, proving that foreground L1 trained a low-frequency smoothing correction instead of the missing perceptual detail.

The canary isolates the second issue on a fixed, verified base while adding RNG isolation needed by the eventual from-zero integration.

## Architecture

### Fixed base

Load the verified neutral-longhorizon64k `ckpt8000.pth`, whose FG LPIPS is `0.1550340652`. Freeze:

- Gaussian XYZ, feature, opacity, scale, and rotation;
- rigid and non-rigid deformation;
- pose correction;
- base texture MLP and latent table;
- camera corrections.

Instantiate structured trunk and high-frequency modules from random initialization. The main render remains the frozen base render for the entire canary. Candidate colors are rasterized through the existing shadow path and receive isolated gradients only.

### RNG isolation

Capture the PyTorch CPU and CUDA RNG states immediately after the base texture MLP is initialized and restore them after constructing shadow-only modules. Shadow parameters retain their sampled initialization, but subsequent training randomness matches the no-shadow model path.

### Perceptual-band objective

Train the candidate with a weighted sum of:

- low-weight masked foreground L1;
- foreground LPIPS as the primary semantic/perceptual signal;
- masked image-gradient loss;
- masked multiscale high-pass loss;
- low-frequency drift penalty between candidate and detached base render.

LPIPS may run at a configured interval and is scaled to preserve its expected contribution. The low-frequency penalty prevents the branch from earning PSNR by repainting broad smooth regions.

## Schedule

- Schedule iteration `8000`: load and freeze the verified base; initialize shadow parameters.
- `8001-9000`: shadow warmup with L1, gradient, and high-pass losses; LPIPS every fourth step.
- `9000`, `10000`, `11000`: record base/candidate metrics without handoff.
- `12000`: final base/candidate gate and stop.

The canary never activates the candidate in the main render and never changes point topology.

## Success Gate

At 12k, approve only when all conditions hold:

- candidate mean FG LPIPS improves over base by at least `0.001`;
- candidate mean FG PSNR does not regress by more than `0.1 dB`;
- no validation camera regresses by more than `0.002` FG LPIPS;
- no NaN, CUDA error, missing candidate render, or empty shadow parameter set occurs.

On failure, write `PERCEPTUAL_BAND_SHADOW_GATE_FAILED`. On success, write `PERCEPTUAL_BAND_SHADOW_GATE_PASSED`. Both paths save the metrics and final diagnostic checkpoint.

## Validation

Unit tests cover RNG restoration, loss composition, LPIPS interval behavior, low-frequency drift, optimizer allowlists, and gate logic. Pipeline tests verify the 8k from-zero baseline start, frozen base parameters, 4k local horizon, schedule offset to 12k, no topology changes, and automatic marker creation.

A GPU smoke run uses the same baseline checkpoint for a short local horizon and proves partial loading, shadow-only optimizer selection, candidate rasterization, perceptual-band logging, checkpoint creation, and pipeline completion before the formal canary is detached.
