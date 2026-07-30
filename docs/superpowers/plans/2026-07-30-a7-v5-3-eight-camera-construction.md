# A7 v5.3 Eight-Camera Construction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and freeze eight-camera renderer evidence, construct a v5.3 lower-carrier candidate from it, and launch a CoreView377 construction plus retrospective diagnostic run.

**Architecture:** Add a dedicated evidence contract and a separate candidate contract. Reuse the existing renderer evidence builder and constrained optimizer with eight formal construction cameras, then run c21-c23 only as retrospective diagnostics.

**Tech Stack:** Python 3.9, NumPy, PyTorch renderer, pytest, Bash.

---

### Task 1: Eight-Camera Evidence Contract

**Files:**
- Create: `configs/semantic/frozen_a7_dual_evidence_v5_3_evidence_377.json`
- Modify: `utils/frozen_semantic_method.py`
- Test: `tests/test_frozen_a7_temporal_method.py`
- Test: `tests/test_build_renderer_aligned_temporal_evidence.py`

- [x] Add failing tests for the v5.3 evidence freeze ID and the exact eight-camera sequence.
- [x] Run the tests and confirm the contract is unsupported.
- [x] Add the evidence contract validation and allow the builder to materialize 912 samples.
- [x] Run the tests and confirm they pass.

### Task 2: Candidate Contract And Calibrator

**Files:**
- Create after evidence generation: `configs/semantic/frozen_a7_dual_evidence_v5_3_canary_377.json`
- Modify: `tools/calibrate_constrained_a7_weights.py`
- Modify: `tools/summarize_a7_v5_1_audit.py`
- Test: `tests/test_calibrate_constrained_a7_weights.py`
- Test: `tests/test_summarize_a7_v5_2_development.py`

- [x] Add failing tests for the v5.3 candidate ID and eight-fold capacity identity.
- [x] Generate the formal evidence and record its SHA-256 and evidence contract fingerprint.
- [x] Add the frozen candidate contract and v5.3 candidate mapping.
- [x] Run the focused tests and confirm they pass.

### Task 3: v5.3 Summary And Runner

**Files:**
- Create: `tools/summarize_a7_v5_3_development.py`
- Create: `tests/test_summarize_a7_v5_3_development.py`
- Create: `exp/acceptdata/a7_dual_evidence_v5_3_canary_377/run_377_v5_3_development.sh`
- Create: `tests/test_run_377_a7_dual_evidence_v5_3.py`

- [x] Add failing tests requiring construction evidence identity, c21-c23 retrospective target/visibility gates, spatial gates, and `paper_test_eligible=false`.
- [x] Implement the v5.3 summary and runner with stale-output and freeze-manifest checks.
- [x] Run the focused tests and confirm they pass.

### Task 4: Capacity, Regression, Commit, And Launch

**Files:**
- Modify: `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`

- [x] Run formal eight-camera capacity and require 8/8 folds plus the frozen construction gates.
- [x] Run the complete focused A7 regression, shell syntax, Python compilation, JSON validation, and `git diff --check`.
- [x] Review the scoped diff for test leakage and marker correctness.
- [x] Commit only v5.3 files and launch the runner with `setsid` on the free GPU.
- [x] Confirm evidence/capacity progress before reporting the run status.
