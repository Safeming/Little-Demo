# A7c R1.1 Transmittance Ray-Context Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect exact alpha-transmittance and ray-context features for CoreView377, run frozen F0-F3 held-block observability ablations, and stop before audit cameras unless F3 passes.

**Architecture:** Reuse the production rasterizer. Obtain exact per-carrier `sum(alpha*T)` from gradients with respect to independent precomputed colors, render alpha/depth/depth2/lower-support/semantic buffers with identical Gaussian state, sample fixed footprint context, and compare four nested feature groups under identical six-fold training.

**Tech Stack:** Python, PyTorch, NumPy, existing CUDA rasterizer, JSON/NPZ, pytest, Bash.

---

### Task 1: Freeze Contract And Exact-Mass Primitives

**Files:**
- Create: `configs/semantic/a7c_r1_1_transmittance_ray_context_377_v1.json`
- Create: `utils/a7c_ray_context_probe.py`
- Create: `tests/test_a7c_ray_context_probe.py`

- [ ] Write failing tests for frozen F0-F3 nesting, forbidden fields, gradient mass extraction, zero invisible mass, ray-buffer ratios, and footprint sampling.
- [ ] Run the named tests and confirm missing-module failures.
- [ ] Implement minimal pure/Torch primitives and the frozen contract.
- [ ] Run focused tests and commit only Task 1 files.

### Task 2: Validate Gradient Against Finite Difference

**Files:**
- Create: `tools/validate_a7c_transmittance_gradient.py`
- Modify: `tests/test_a7c_ray_context_probe.py`
- Output: `exp/acceptdata/a7c_r1_1_transmittance_ray_context_377_v1/validation/`

- [ ] Write a failing test for central finite-difference agreement on a differentiable alpha-compositing fixture.
- [ ] Implement analytical extraction and finite-difference reporting.
- [ ] Run a real two-frame CoreView377 smoke validation; require relative error within the contract and zero mass for invisible carriers.
- [ ] Save the validation fingerprint; failure stops all later tasks.

### Task 3: Collect Static, Exact-Mass, And Ray-Context Features

**Files:**
- Create: `tools/build_a7c_r1_1_ray_context_probe.py`
- Modify: `utils/a7c_ray_context_probe.py`
- Modify: `tests/test_a7c_ray_context_probe.py`
- Output: `exp/acceptdata/a7c_r1_1_transmittance_ray_context_377_v1/probe/probe.npz`

- [ ] Write failing schema/alignment tests for static checkpoint descriptors, semantic bank fields, projected centers, buffer sampling, and artifact fingerprints.
- [ ] Implement one shared render state per sample, one color-gradient backward, and the five frozen feature-buffer renders.
- [ ] Run dry-run and two-frame smoke; measure throughput and compute the Beijing completion window.
- [ ] Launch the full 912-sample probe with resumable shard output and final fingerprint verification.

### Task 4: Run Frozen F0-F3 Held-Block Training

**Files:**
- Create: `tools/train_a7c_r1_1_feature_observability.py`
- Create: `tools/audit_a7c_r1_1_feature_observability.py`
- Modify: `tests/test_a7c_ray_context_probe.py`
- Output: `exp/acceptdata/a7c_r1_1_transmittance_ray_context_377_v1/{training,audit}/`

- [ ] Write failing tests for feature nesting, fit-only normalization, shared `64->32->1` bounded predictor, six held-block folds, F0 comparison, and promotion aggregation.
- [ ] Implement fixed-budget training for all four feature groups without IDs, time fields, recurrence, or audit-camera gradients.
- [ ] Train F0-F3, then open held-block metrics once and write immutable per-fold and aggregate artifacts.
- [ ] Create `.held_block_passed` or `.rejected`; never compute c17-c20 metrics on rejection.

### Task 5: Queue, Verification, And Documentation

**Files:**
- Create: `tools/run_a7c_r1_1_transmittance_ray_context_377.sh`
- Modify: `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`
- Modify: `tests/test_a7c_ray_context_probe.py`

- [ ] Test queue dry-run, resume markers, stop rules, and absence of c21-c23.
- [ ] Run both A7c focused suites, Python compilation, Bash syntax validation, and `git diff --check`.
- [ ] Launch or resume the formal queue on GPU 0 and verify active stage/GPU allocation.
- [ ] Record config/code fingerprints, PID, start time, estimated Beijing completion time, output root, and final gate result when available.
