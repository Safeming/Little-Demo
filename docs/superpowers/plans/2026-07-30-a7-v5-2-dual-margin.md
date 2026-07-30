# A7 v5.2 Dual-Margin Development Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add separate construction/audit target-response gates, freeze the v5.2 contract, and launch a CoreView377 validation plus retrospective diagnostic run.

**Architecture:** Extend the existing v5 capacity API symmetrically with the visibility dual gates. Keep candidate generation unchanged except for the stricter construction target/visibility limits. Add a v5.2 development summarizer that distinguishes c17-c20 validation from already-opened c21-c23 retrospective diagnostics.

**Tech Stack:** Python 3.9, NumPy, pytest, Bash, existing semantic renderer/evaluator.

---

### Task 1: Target Dual-Gate Contract

**Files:**
- Create: `configs/semantic/frozen_a7_dual_evidence_v5_2_canary_377.json`
- Modify: `utils/frozen_semantic_method.py`
- Test: `tests/test_frozen_a7_temporal_method.py`

- [ ] Add a failing contract test requiring `minimum_training_target_response_ratio=0.9975` and `minimum_audit_target_response_ratio=0.99`.
- [ ] Run the focused test and confirm the loader rejects the unknown v5.2 freeze ID or missing fields.
- [ ] Add v5.2 contract validation with training target not below audit target.
- [ ] Run the focused test and confirm it passes.

### Task 2: Capacity Target Routing

**Files:**
- Modify: `utils/constrained_sparse_temporal_optimizer.py`
- Modify: `tools/calibrate_constrained_a7_weights.py`
- Test: `tests/test_constrained_sparse_temporal_optimizer.py`
- Test: `tests/test_calibrate_constrained_a7_weights.py`

- [ ] Add failing tests proving construction receives `0.9975`, held-out/final audit receives `0.99`, and an inverted target pair is rejected.
- [ ] Run the focused tests and confirm they fail for missing target-limit routing.
- [ ] Implement `resolve_target_limits`, optional training/audit target arguments, v5.2 candidate ID mapping, and contract field routing.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Development Summary And Runner

**Files:**
- Create: `tools/summarize_a7_v5_2_development.py`
- Create: `tests/test_summarize_a7_v5_2_development.py`
- Create: `exp/acceptdata/a7_dual_evidence_v5_2_canary_377/run_377_v5_2_development.sh`
- Create: `tests/test_run_377_a7_dual_evidence_v5_2.py`
- Modify: `tools/summarize_a7_v5_1_audit.py`

- [ ] Add failing tests for v5.2 candidate identity, target-response aggregation from temporal CSV files, validation reserve gates, retrospective gates, spatial gates, and `paper_test_eligible=false`.
- [ ] Run the focused tests and confirm missing v5.2 summary/runner behavior.
- [ ] Implement the v5.2 summarizer and runner with c17-c20 validation followed by c21-c23 retrospective diagnostics.
- [ ] Run the focused tests and confirm they pass.

### Task 4: Real Capacity, Regression, Commit, And Launch

**Files:**
- Modify: `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`

- [ ] Run real frozen evidence capacity and require 4/4 folds, hair zero changes, lower 25 changes, visibility max at most 0.9990, target min at least 0.9975, and outer/boundary gains above 0.5%.
- [ ] Run the complete focused A7 regression, `bash -n`, Python compilation, JSON validation, and `git diff --check`.
- [ ] Review the scoped diff for protocol leakage, stale-output reuse, and completion-marker correctness.
- [ ] Commit only the v5.2 files and launch the runner with `setsid` on the free GPU.
- [ ] Confirm capacity finishes and c17 starts before reporting the start and estimated Beijing completion time.
