# Coverage-Constrained Real Editing and Flicker Failure Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recompute the three real-editing tasks with pooled burden and per-task cross-subject coverage, diagnose fixed versus adaptive temporal flicker, and publish formal failure cases without retraining or rerendering.

**Architecture:** Add two standalone pure-statistics summarizers and one failure-report builder. Reuse the frozen real-editing curves and temporal full-strength records, write new independent output directories, and run them through one CPU queue that records Beijing start/end/status.

**Tech Stack:** Python standard library, NumPy, pytest, existing CSV/JSON experiment artifacts, Bash queue wrapper.

---

### Task 1: Coverage-constrained pooled real-editing tables

**Files:**
- Create: `tools/summarize_semantic_real_editing_coverage_constrained.py`
- Create: `tests/test_summarize_semantic_real_editing_coverage_constrained.py`

- [ ] **Step 1: Add failing unit tests**

Test exact pooled numerator/denominator recovery, per-task per-retention eligibility requiring every subject coverage `>=0.80`, exclusion of a part with one subject at `0.79`, and end-to-end generation of six finite artifacts.

- [ ] **Step 2: Run the new tests and confirm module-not-found failure**

Run: `/opt/miniconda3/bin/conda run -n ictrl python -m pytest -q tests/test_summarize_semantic_real_editing_coverage_constrained.py`

- [ ] **Step 3: Implement the summarizer**

Read the existing five subject `metrics.csv` files, reuse `build_matched_strength_curves()` and `match_curves_at_retention()`, recover matched target/outer/boundary sums from the Voting reference target, apply per-task cross-subject eligibility, pool within subjects, and bootstrap A5-minus-Voting deltas.

- [ ] **Step 4: Run the tests until green**

Expected: all new tests pass and direct CLI `--help` exits zero.

### Task 2: Fixed-versus-adaptive flicker diagnostic

**Files:**
- Create: `tools/summarize_semantic_temporal_flicker_diagnostic.py`
- Create: `tests/test_summarize_semantic_temporal_flicker_diagnostic.py`

- [ ] **Step 1: Add failing unit tests**

Test that constant scaling leaves normalized flicker unchanged, frame-varying adaptive strength changes A5 flicker, Voting fixed/adaptive agree, only consecutive common frames count, and the diagnostic writes finite output artifacts.

- [ ] **Step 2: Run the tests and confirm module-not-found failure**

Run: `/opt/miniconda3/bin/conda run -n ictrl python -m pytest -q tests/test_summarize_semantic_temporal_flicker_diagnostic.py`

- [ ] **Step 3: Implement the diagnostic**

Reuse the temporal matching and eligibility functions, evaluate full-strength and adaptive values on the exact same reachable frames, summarize each subject-part sequence, average eligible parts within subjects, and bootstrap fixed/adaptive A5-minus-Voting plus adaptive-minus-fixed compensation deltas.

- [ ] **Step 4: Run the tests until green**

Expected: all diagnostic tests pass and Voting fixed/adaptive differences are within floating-point tolerance.

### Task 3: Formal failure report

**Files:**
- Create: `tools/build_semantic_paper_failure_report.py`
- Create: `tests/test_build_semantic_paper_failure_report.py`
- Create: `docs/正式论文失败案例与时序诊断_20260727.md`

- [ ] **Step 1: Add failing report tests**

Use tiny fixture tables and require entries for face/skin/shoes coverage, CoreView_377 lower burden, temporal flicker regression, and referenced qualitative assets.

- [ ] **Step 2: Implement CSV and Markdown generation**

Read the two new formal output directories plus existing temporal part results. Write one machine-readable failure table and one paper-ready Markdown report without selecting or hiding failures.

- [ ] **Step 3: Verify the report tests**

Expected: all required failure categories and artifact paths are present.

### Task 4: CPU queue and formal execution

**Files:**
- Create: `tools/run_coverage_real_edit_and_flicker_failures_queue.sh`

- [ ] **Step 1: Add a queue contract test**

Verify the script references all three tools, uses the `ictrl` environment, writes Beijing timestamps, and stops on the first failure.

- [ ] **Step 2: Run the queue on frozen artifacts**

Expected output directories:

```text
exp/acceptdata/five_subject_real_editing_matched_strength_20260723/aggregate/formal_coverage_constrained/
exp/acceptdata/five_subject_semantic_temporal_stability_20260724/aggregate/formal_flicker_diagnostic/
exp/acceptdata/formal_semantic_failure_cases_20260727/
```

- [ ] **Step 3: Reproduce to temporary directories and compare**

Require byte-identical CSV/JSON/Markdown outputs after normalizing only absolute output paths if present.

### Task 5: Verification and commit

**Files:**
- Verify all files created above.

- [ ] **Step 1: Run focused regression tests**

Run the three new test files plus existing matched-strength and temporal summarizer tests.

- [ ] **Step 2: Check finite values, coverage eligibility, and failure inventory**

Run `rg` for NaN/Inf, inspect formal/paired tables, and verify face/skin/shoes, 377 lower, and flicker entries.

- [ ] **Step 3: Commit only this task's scripts, tests, plan, queue, and failure report**

Commit message: `实验：补充覆盖约束编辑与时序失败分析`.
