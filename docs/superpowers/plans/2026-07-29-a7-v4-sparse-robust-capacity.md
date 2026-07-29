# A7 V4 Sparse Robust Capacity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a frozen A7 v4 sparse robust CoreView377 validation canary using the existing v3 renderer contribution sequence.

**Architecture:** Add a deterministic sparse coordinate optimizer with camera-aware direct metrics and LOCO reporting, generate one v4 bank through a focused CLI, and reuse the established temporal/spatial validators from an independent runner.

**Tech Stack:** Python, NumPy, pytest, Bash, existing A7 bank and validation utilities.

---

### Task 1: Freeze The V4 Contract

**Files:**
- Create: `configs/semantic/frozen_a7_sparse_robust_v4_canary_377.json`
- Modify: `utils/frozen_semantic_method.py`
- Modify: `tests/test_frozen_a7_temporal_method.py`

- [ ] Add a failing test for the v4 freeze ID, source evidence fingerprint, sparse action set, LOCO gate, processed/frozen parts, and formal thresholds.
- [ ] Run the focused test and confirm the unsupported contract failure.
- [ ] Add the exact config and validation policy without changing older contracts.
- [ ] Re-run the full contract test file and commit.

### Task 2: Implement Sparse Robust Optimization

**Files:**
- Create: `utils/sparse_robust_temporal_optimizer.py`
- Create: `tests/test_sparse_robust_temporal_optimizer.py`

- [ ] Add failing tests for camera reset, target constraints, sparse changed-count limit, deterministic action choice, all-camera direct improvement, and LOCO aggregation.
- [ ] Run the test file and confirm the missing module failure.
- [ ] Implement incremental response updates and deterministic greedy search.
- [ ] Re-run tests and commit.

### Task 3: Generate The Frozen V4 Candidate

**Files:**
- Create: `tools/calibrate_sparse_robust_a7_weights.py`
- Create: `tests/test_calibrate_sparse_robust_a7_weights.py`

- [ ] Add a failing CLI test that consumes sequence evidence, runs four LOCO folds, writes exactly one candidate, and rejects a failed fold.
- [ ] Run the test and confirm the missing tool failure.
- [ ] Implement source-evidence verification, final bank writing, fingerprints, and structured summaries.
- [ ] Re-run the test and commit.

### Task 4: Add The Formal Runner

**Files:**
- Create: `exp/acceptdata/a7_sparse_robust_v4_canary_377/run_377_v4_validation.sh`
- Create: `tests/test_run_377_a7_sparse_robust_v4.py`
- Modify: `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`

- [ ] Add a failing runner test for v3 evidence reuse, one candidate, c17-c20 only, independent paths, and resumable completion.
- [ ] Implement the runner and verify `bash -n`.
- [ ] Run all affected v1-v4 tests and `git diff --check`.
- [ ] Execute the real offline LOCO probe and verify all four folds plus final per-camera evidence gates.
- [ ] Launch the detached c17-c20 temporal/spatial queue, verify PID/GPU/log progress, calculate ETA from the measured v3 single-candidate duration, and append the launch record.

