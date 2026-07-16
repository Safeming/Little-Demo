# CoreView377 Occlusion-Aware Consensus Canary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and launch a 4k subject-agnostic pose-coreset, occlusion-aware boundary-gradient consensus canary from the current from-zero checkpoint.

**Architecture:** Extend the existing residual-balanced sampler behind an option flag, add a pure multi-view gradient reducer plus accumulator, and connect it to the existing grouped optimizer step in `train.py`. A separate option and launcher freeze the learned image function, train only boundary residuals, and run final same-30/original-57/contour audits.

**Tech Stack:** Python 3.9, PyTorch autograd, NumPy, Hydra/OmegaConf, pytest, Bash, existing 3DGS renderer.

---

### Task 1: Pose Coreset Sampling

**Files:**
- Modify: `utils/residual_balanced_sampler.py`
- Modify: `train.py`
- Create: `tests/test_pose_coreset_sampler.py`

- [ ] Write failing tests for deterministic farthest-point pose selection,
  coreset/all-frame probability mass, complete frame reachability, residual
  weighting within the coreset, and K=4 camera sampling without replacement.
- [ ] Run `/opt/miniconda3/envs/ictrl/bin/python -m pytest -q tests/test_pose_coreset_sampler.py`
  and confirm failures are caused by missing pose-coreset behavior.
- [ ] Add `_pose_vectors_from_dataset`, deterministic standardized farthest-point
  selection, and optional coreset probability mixing to
  `ResidualBalancedMultiViewSampler`.
- [ ] Keep existing behavior byte-for-byte equivalent when
  `residual_balanced_pose_coreset_enable=false`.
- [ ] Register no new sampler mode; reuse `residual_balanced_multiview` so grouped
  accumulation and residual feedback remain unchanged.
- [ ] Re-run the focused tests and existing residual sampler tests.

### Task 2: Multi-View Gradient Consensus

**Files:**
- Create: `utils/occlusion_aware_consensus.py`
- Create: `tests/test_occlusion_aware_consensus.py`

- [ ] Write failing tests proving two-view agreement is rejected when three are
  required, three-view positive and negative agreement is retained, tiny
  gradients are excluded by the magnitude quantile, and ordinary parameter
  gradients are replaced rather than added.
- [ ] Run the focused test and verify RED.
- [ ] Implement `reduce_multiview_gradients` and
  `BoundaryGradientConsensusAccumulator` with detached tensors and structured
  diagnostics.
- [ ] Re-run the focused test and verify GREEN.

### Task 3: Training Loop Integration

**Files:**
- Modify: `train.py`
- Create: `tests/test_train_occlusion_consensus.py`

- [ ] Write source and unit tests for the disabled-default contract, parameter
  name validation, all-active zero-residual support initialization, isolated
  boundary gradient collection, replacement at grouped optimizer steps, and
  `[BoundaryConsensus]` logs.
- [ ] Run the focused test and verify RED.
- [ ] Initialize the accumulator after checkpoint load and boundary state resize.
- [ ] Collect isolated boundary gradients before the ordinary backward pass.
- [ ] At `accumulation_should_step`, replace only configured residual parameter
  gradients with consensus output; leave every other optimizer group unchanged.
- [ ] Log view count, active fraction, positive/negative agreement, magnitude
  percentiles, and nonfinite count.
- [ ] Re-run focused and adjacent training-loop tests.

### Task 4: Canary Configuration And Launcher

**Files:**
- Create: `configs/option/stageA_fromzero_occlusion_consensus_contour_v1.yaml`
- Create: `tools/run_coreview377_occlusion_consensus_canary.sh`
- Create: `tools/evaluate_occlusion_consensus_canary.py`
- Create: `tests/test_coreview377_occlusion_consensus_pipeline.py`

- [ ] Write failing pipeline tests for the from-zero checkpoint dependency,
  4k/K=4 schedule, pose coreset flags, residual-only learning rates, disabled
  topology/camera/photometric paths, dual evaluation, contour audit, and final
  acceptance report.
- [ ] Run the focused tests and verify RED.
- [ ] Add the option with same-30 PSNR selection and LPIPS constraint.
- [ ] Add the final evaluator with explicit baseline-relative thresholds.
- [ ] Add the launcher using the existing stage runner, diagnostic validation,
  render, and `analyze_render_quality_edges.py` patterns.
- [ ] Re-run focused tests and `bash -n`.

### Task 5: Verification And Launch

**Files:**
- Test: all files above plus adjacent residual sampler, boundary, and pipeline tests.

- [ ] Run focused and adjacent pytest suites.
- [ ] Run a short CUDA smoke with 12 sampled views, K=4 accumulation, one
  evaluation, one checkpoint, and final audit disabled.
- [ ] Verify logs contain nonzero consensus activity, no all-point visibility
  proxy, checkpoint save, and no traceback/runtime error.
- [ ] Launch the formal 4k canary detached on GPU 0.
- [ ] Wait for stable progress and the first consensus log, then calculate the
  Beijing completion estimate from measured sampled-view throughput.
- [ ] Report run directory, PID, progress command, and expected completion time.
