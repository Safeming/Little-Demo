# A7 v5.1 Visibility Margin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate the A7 construction visibility margin from the unchanged formal audit gate and launch the frozen c21-c23 audit.

**Architecture:** Extend the existing constrained capacity API with distinct training and audit visibility thresholds. Add a v5.1 frozen contract that reuses the exact v5 evidence, then generate one deterministic candidate and run the existing temporal/spatial audit flow.

**Tech Stack:** Python, NumPy, PyTorch renderer evidence, pytest, Bash.

---

### Task 1: Freeze the v5.1 contract

**Files:**
- Create: `configs/semantic/frozen_a7_dual_evidence_v5_1_canary_377.json`
- Modify: `utils/frozen_semantic_method.py`
- Test: `tests/test_frozen_a7_temporal_method.py`

- [ ] Add a failing contract test for source evidence SHA/fingerprint, construction ratio `0.9995`, and audit ratio `1.0`.
- [ ] Run the focused test and confirm the v5.1 contract is unsupported.
- [ ] Add the frozen contract and validation policy.
- [ ] Run the focused test and confirm it passes.

### Task 2: Separate construction and audit thresholds

**Files:**
- Modify: `utils/constrained_sparse_temporal_optimizer.py`
- Test: `tests/test_constrained_sparse_temporal_optimizer.py`

- [ ] Add a failing test showing training uses `0.9995` while held-out evaluation uses `1.0`.
- [ ] Run the focused test and confirm the API is missing.
- [ ] Pass distinct thresholds through fold optimization, held-out evaluation, and final evaluation.
- [ ] Run optimizer tests and confirm v5 compatibility remains green.

### Task 3: Generate the v5.1 candidate

**Files:**
- Modify: `tools/calibrate_constrained_a7_weights.py`
- Test: `tests/test_calibrate_constrained_a7_weights.py`

- [ ] Add a failing CLI test for reuse of source v5 evidence and candidate id `dual_evidence_constrained_v5_1`.
- [ ] Run the focused test and confirm the v5.1 freeze id is rejected.
- [ ] Validate the frozen evidence SHA/fingerprint and map the two visibility thresholds.
- [ ] Run CLI tests and confirm deterministic output.

### Task 4: Add and launch the audit runner

**Files:**
- Create: `exp/acceptdata/a7_dual_evidence_v5_1_canary_377/run_377_v5_1_audit.sh`
- Test: `tests/test_run_377_a7_dual_evidence_v5_1.py`
- Modify: `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`

- [ ] Add a failing runner test for evidence reuse, one candidate, freeze manifest, c21-c23, and test spatial split.
- [ ] Implement the runner and update the experiment record.
- [ ] Run the complete A7 focused regression, Python compilation, and Bash syntax checks.
- [ ] Commit only v5.1 files, launch with an independent session, and verify GPU/log progress.

