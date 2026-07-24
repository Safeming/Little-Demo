# Temporal Pooled Matched-Retention Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Regenerate the five-subject temporal formal tables with pooled burden, a fixed cross-subject coverage rule, and exact matched-retention temporal statistics.

**Architecture:** Add a standalone summarizer that reads the frozen full-strength aggregate CSV, performs exact linear strength matching at retentions 0.25 and 0.50, and writes a separate formal output directory. Keep matching, coverage, temporal-pair, pooling, and bootstrap logic in testable pure functions.

**Tech Stack:** Python standard library, NumPy, pytest, existing experiment CSV/JSON artifacts.

---

### Task 1: Lock the behavior with failing tests

**Files:**
- Create: `tests/test_summarize_semantic_temporal_matched_retention.py`

**Steps:**
1. Add a toy exact-scaling test covering Voting and A5 strength calculation.
2. Add an unreachable-frame and coverage-rate test.
3. Add a cross-subject eligible-part test using the fixed 0.80 threshold.
4. Add a consecutive-pair flicker test proving frame gaps are excluded.
5. Add a small end-to-end CSV test for all six output artifacts.
6. Run the new test file and confirm failure because the summarizer does not exist.

### Task 2: Implement the matched-retention summarizer

**Files:**
- Create: `tools/summarize_semantic_temporal_matched_retention.py`
- Modify: `tests/test_summarize_semantic_temporal_matched_retention.py`

**Steps:**
1. Parse and pair Voting/A5 rows by subject, part, and frame.
2. Apply exact target matching and scale image deltas and selection masses.
3. Compute coverage per subject-part-retention and formal eligible parts.
4. Compute pooled burden and consecutive supported-pair flicker.
5. Pool eligible parts within subjects, then summarize subjects equally.
6. Add deterministic 10,000-resample paired subject bootstrap statistics.
7. Write coverage, part, subject, formal, paired-statistics, and JSON summary outputs.
8. Run the new tests until green.

### Task 3: Regenerate the formal experiment tables

**Files:**
- Create: `exp/acceptdata/five_subject_semantic_temporal_stability_20260724/aggregate/formal_matched_retention/*`

**Steps:**
1. Run the summarizer on the frozen 34,200-row aggregate source.
2. Confirm eligible parts are derived solely from the fixed coverage rule.
3. Confirm every numeric output is finite and all expected subjects are present.
4. Record the source SHA-256 and exact statistical assumptions in `summary.json`.

### Task 4: Verify and report

**Files:**
- Verify: `tools/summarize_semantic_temporal_matched_retention.py`
- Verify: `tests/test_summarize_semantic_temporal_matched_retention.py`
- Verify: `exp/acceptdata/five_subject_semantic_temporal_stability_20260724/aggregate/formal_matched_retention/*`

**Steps:**
1. Run the new tests plus the existing temporal summarizer tests.
2. Inspect formal and paired-statistics tables for A5-vs-Voting direction, effect size, and confidence intervals.
3. Check `git diff --check` on the files added by this task.
4. Commit only the new implementation and test files with a Chinese commit message; experiment artifacts remain reproducible outputs unless already tracked by repository convention.
