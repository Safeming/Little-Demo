# CoreView377 Residual Surface Reallocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-view gradient evidence and fixed-budget surface carrier reallocation to the 50k from-zero pipeline, then launch an absolute-gated 16k-to-64k experiment.

**Architecture:** `GaussianModel` owns pointwise evidence tensors and an in-place donor-to-child reallocation operation that preserves point count. `train.py` updates evidence after backward, triggers bounded reallocation events, applies a persistent global shell, and stops on absolute baseline gates. A dedicated option and wrapper reuse the verified 64k launcher.

**Tech Stack:** Python 3.9, PyTorch, Hydra, Bash, pytest, CUDA rasterizer.

---

### Task 1: Pointwise Evidence Bank

**Files:**
- Modify: `scene/gaussian_model.py`
- Create: `tests/test_residual_surface_reallocation.py`

- [ ] Write failing tests for evidence initialization, EMA updates from visibility/screen/opacity/feature gradients, observation counts, and normalized ranking scores.
- [ ] Run the focused pytest and verify RED on missing methods.
- [ ] Implement `ensure_surface_evidence_state`, `update_surface_evidence`, and `surface_evidence_score` with finite-value guards and point-count validation.
- [ ] Re-run focused tests and existing densification tests.

### Task 2: Fixed-Budget Surface Reallocation

**Files:**
- Modify: `scene/gaussian_model.py`
- Extend: `tests/test_residual_surface_reallocation.py`

- [ ] Write failing tests proving point count is unchanged, donors and parents come from different faces, children move only in the parent tangent plane, parent parameters and binding rows are inherited, and optimizer moments are copied.
- [ ] Run focused tests and verify RED.
- [ ] Implement bounded donor/parent selection and `reallocate_surface_carriers` using in-place row replacement, deterministic tangent offsets, opacity-mass splitting, binding-state inheritance, metadata updates, and Adam-state inheritance.
- [ ] Re-run focused and Gaussian-model tests.

### Task 3: Training Integration And Absolute Gate

**Files:**
- Modify: `train.py`
- Extend: `tests/test_residual_surface_reallocation.py`

- [ ] Write failing tests for the 4k-16k/500-step schedule and absolute baseline gate behavior.
- [ ] Run focused tests and verify RED.
- [ ] Update evidence after backward and before optimizer step, trigger reallocation at configured intervals, log evidence/concentration/event statistics, and save a failure marker on gate failure.
- [ ] Extend the surface-carrier gate to use an absolute iteration-to-LPIPS map when configured.
- [ ] Re-run focused tests plus anchor-tether and surface-carrier tests.

### Task 4: Recipe And Launcher

**Files:**
- Create: `configs/option/stageA_377_multiview_explicit_hq_fromzero_residual_surface_reallocation_v1.yaml`
- Create: `exp/zero_train_to_v395/coreview377_residual_surface_reallocation_20260713_bjt/launch_residual_surface_reallocation.sh`
- Create: `tests/test_coreview377_residual_surface_reallocation_pipeline.py`

- [ ] Write failing launcher/config tests for 50k surface initialization, no old competition, evidence/reallocation schedule, global persistent shell, absolute gates, 64k horizon, and no pretrained checkpoint.
- [ ] Run pipeline tests and verify RED.
- [ ] Add the option and one-command wrapper with smoke overrides that force an early evidence/reallocation event.
- [ ] Re-run all related launcher tests.

### Task 5: Verification And Launch

**Files:**
- Runtime output: `exp/zero_train_to_v395/coreview377_residual_surface_reallocation_20260713_bjt/`

- [ ] Run Python compilation, shell syntax, focused pytest, and `git diff --check`.
- [ ] Run a GPU smoke that exercises evidence update and at least one reallocation event while retaining exactly 50k points.
- [ ] Confirm GPU availability and detach the formal 64k run with `setsid -f`.
- [ ] Verify the PID, GPU allocation, initialization identity, and first progress update.
- [ ] Estimate completion time from the previous 50k 64k wall time and provide grep-based tracking commands.
