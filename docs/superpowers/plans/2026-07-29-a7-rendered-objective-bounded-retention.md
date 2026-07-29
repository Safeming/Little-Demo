# A7 Rendered Objective And Bounded Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch an A7 v3 CoreView377 canary that screens two bounded static-bank policies using part-level renderer contribution sequences.

**Architecture:** Extend the v2 evidence artifact with camera-aware per-frame contribution tensors, add a focused bounded calibration module and rendered-sequence scorer, then generate exactly two pre-registered candidate banks. Reuse the existing temporal and spatial validators in a new isolated output root.

**Tech Stack:** Python, NumPy, PyTorch autograd, pytest, Bash, existing Gaussian renderer and A7 validators.

---

### Task 1: Freeze The V3 Contract

**Files:**
- Create: `configs/semantic/frozen_a7_renderer_objective_v3_canary_377.json`
- Modify: `utils/frozen_semantic_method.py`
- Modify: `tests/test_frozen_a7_temporal_method.py`

- [ ] Add a failing test that loads the v3 freeze ID and asserts the exact two policies, frozen parts, thresholds, and no c21 usage.
- [ ] Run `python -m pytest tests/test_frozen_a7_temporal_method.py -q` and confirm the unsupported freeze ID failure.
- [ ] Add the v3 contract and exact validation policy.
- [ ] Re-run the test and commit the contract change.

### Task 2: Export Renderer Contribution Sequences

**Files:**
- Modify: `utils/renderer_aligned_temporal_evidence.py`
- Modify: `tools/build_renderer_aligned_temporal_evidence.py`
- Modify: `tests/test_renderer_aligned_temporal_evidence.py`

- [ ] Add failing tests for float16 sequence export, camera/frame metadata, and camera-boundary reset.
- [ ] Run the focused tests and confirm the missing sequence API failure.
- [ ] Implement a deterministic sequence collector while retaining all v2 aggregate fields.
- [ ] Re-run focused tests and commit.

### Task 3: Implement Bounded Calibration

**Files:**
- Create: `utils/bounded_temporal_calibration.py`
- Create: `tests/test_bounded_temporal_calibration.py`

- [ ] Add failing tests proving weights never exceed A5, topology is unchanged, frozen parts are bitwise identical, L1 is bounded, and an unreachable target floor is invalid.
- [ ] Run the new test file and confirm import/function failures.
- [ ] Implement bounded damping and restore-only-to-A5 retention.
- [ ] Re-run the test file and commit.

### Task 4: Implement The Rendered Sequence Objective

**Files:**
- Create: `utils/renderer_sequence_objective.py`
- Create: `tests/test_renderer_sequence_objective.py`

- [ ] Add failing arithmetic tests for per-camera mean, adjacent absolute change, normalized flicker, camera reset, and A7/A5 ratios.
- [ ] Run the new tests and confirm the missing scorer failure.
- [ ] Implement the scorer with camera-equal and processed-part-equal aggregation.
- [ ] Re-run the tests and commit.

### Task 5: Generate Exactly Two V3 Banks

**Files:**
- Create: `tools/calibrate_renderer_objective_a7_weights.py`
- Create: `tests/test_calibrate_renderer_objective_a7_weights.py`

- [ ] Add a failing CLI test that builds exactly `bounded_damping_005` and `bounded_retention_010`, records rendered objective summaries, and preserves deterministic fingerprints.
- [ ] Run the CLI test and confirm the tool is missing.
- [ ] Implement candidate generation using existing A7 bank provenance helpers.
- [ ] Re-run the CLI test and commit.

### Task 6: Add And Launch The V3 Runner

**Files:**
- Create: `exp/acceptdata/a7_renderer_objective_v3_canary_377/run_377_v3_validation.sh`
- Create: `tests/test_run_377_a7_renderer_objective_v3.py`
- Modify: `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`

- [ ] Add a failing runner contract test for independent paths, two candidates, c17-c20 only, completion marker, and no c21-c23.
- [ ] Implement the resumable runner and verify `bash -n` plus dry-run commands.
- [ ] Run all A7 v3 and affected v2 tests, then run `git diff --check`.
- [ ] Run a one-frame CUDA evidence canary and verify finite nonzero sequence contributions.
- [ ] Launch the formal runner under `nohup`, verify its PID/GPU/log progress, calculate ETA from v2 measured throughput, and append the launch record to the A7 handoff document.

