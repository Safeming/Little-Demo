# A7 v5.4 Camera-Time Stability Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a lower-only A7 v5.4 candidate using 48 nested camera-time folds and consensus carrier selection without reading c21-c23.

**Architecture:** Extend the constrained temporal optimizer with fixed temporal blocks, block-risk evaluation, segment-safe support, and consensus stability selection. Add a frozen v5.4 contract, capacity calibrator route, evidence-only development summary, and a background runner that executes capacity plus spatial guards.

**Tech Stack:** Python 3.9, NumPy, pytest, Bash, existing renderer evidence and semantic paper evaluator.

---

### Task 1: Camera-Time Block Metrics

**Files:**
- Modify: `utils/constrained_sparse_temporal_optimizer.py`
- Test: `tests/test_constrained_sparse_temporal_optimizer.py`

- [x] Add failing tests for six balanced blocks per camera, segment-safe adjacent support, and block gain summaries.
- [x] Run the focused tests and confirm the new APIs are missing.
- [x] Implement `assign_temporal_blocks`, segment-aware support, and block robustness metrics.
- [x] Run the focused tests and confirm they pass.

### Task 2: Block-Aware Sparse Ranking

**Files:**
- Modify: `utils/constrained_sparse_temporal_optimizer.py`
- Test: `tests/test_constrained_sparse_temporal_optimizer.py`

- [x] Add a failing test where the mean camera score improves while one block regresses, and require the block-safe move to rank first.
- [x] Run the test and confirm the current full-camera scorer chooses the wrong move or lacks the API.
- [x] Add lexicographic block violation and worst-decile ranking to constrained lower optimization.
- [x] Run the optimizer tests and confirm they pass.

### Task 3: Forty-Eight-Fold Stability Consensus

**Files:**
- Modify: `utils/constrained_sparse_temporal_optimizer.py`
- Test: `tests/test_constrained_sparse_temporal_optimizer.py`

- [x] Add failing tests for 48 `(camera, block)` folds, 36/48 carrier admission, deterministic level consensus, and rejection of full-data-only carriers.
- [x] Run the tests and confirm consensus capacity is unavailable.
- [x] Implement `run_camera_time_stability_capacity` with fold-local support and consensus weights.
- [x] Run the focused tests and confirm they pass.

### Task 4: Frozen v5.4 Contract And Calibrator

**Files:**
- Create: `configs/semantic/frozen_a7_dual_evidence_v5_4_canary_377.json`
- Modify: `utils/frozen_semantic_method.py`
- Modify: `tools/calibrate_constrained_a7_weights.py`
- Modify: `tools/summarize_a7_v5_1_audit.py`
- Test: `tests/test_frozen_a7_temporal_method.py`
- Test: `tests/test_calibrate_constrained_a7_weights.py`

- [x] Add failing contract and CLI tests for the fixed block policy and candidate ID `dual_evidence_camera_time_v5_4`.
- [x] Run the tests and confirm v5.4 is unsupported.
- [x] Add the frozen contract validation and calibrator route.
- [x] Run the focused tests and confirm they pass.

### Task 5: Evidence-Only Summary And Runner

**Files:**
- Create: `tools/summarize_a7_v5_4_development.py`
- Create: `tests/test_summarize_a7_v5_4_development.py`
- Create: `exp/acceptdata/a7_dual_evidence_v5_4_canary_377/run_377_v5_4_development.sh`
- Create: `tests/test_run_377_a7_dual_evidence_v5_4.py`

- [x] Add failing tests requiring 48-fold identity, block gates, spatial gates, `paper_test_eligible=false`, and absence of c21-c23 rendering.
- [x] Run the tests and confirm the files are missing.
- [x] Implement the summary and resumable runner with freeze-manifest and stale-output checks.
- [x] Run the focused tests and confirm they pass.

### Task 6: Capacity, Verification, Documentation, And Launch

**Files:**
- Modify: `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`

- [x] Run formal v5.4 capacity on the frozen eight-camera evidence.
- [x] Reject the route if the consensus candidate fails any frozen development gate.
- [ ] Run the complete A7 regression, Python compilation, JSON validation, shell syntax, and `git diff --check`.
- [ ] Commit only v5.4 files.
- [ ] Launch the runner with `setsid` on the free GPU/CPU and confirm capacity progress.
- [ ] Record the start time, PID, implementation commit, and estimated Beijing completion time.
