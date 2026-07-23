# Five-Subject A5 LOSO Statistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run method-matched five-subject A5 leave-one-subject-out evaluation and generate formal paired statistics without retraining.

**Architecture:** A queue first creates A5 validation reports for every subject/threshold pair, then a dedicated selector freezes one config per held-out subject using the other four donors. Test evaluation reuses existing checkpoints and A5 footprint banks, and a separate summarizer produces paper tables and deterministic paired bootstrap statistics.

**Tech Stack:** Bash, Python standard library, CSV/JSON, pytest, CUDA evaluator.

---

### Task 1: Lock A5 LOSO Selection Behavior

**Files:**
- Create: `tests/test_five_subject_a5_loso_statistics.py`
- Create: `tools/select_frozen_a5_loso_config.py`

- [ ] Add failing tests for A5 report loading, four unique donors, leakage/mIoU gates and deterministic candidate ordering.
- [ ] Run `/opt/miniconda3/envs/ictrl/bin/python -m pytest -q tests/test_five_subject_a5_loso_statistics.py` and confirm the selector import fails.
- [ ] Implement the minimal A5-specific report loader and config writer.
- [ ] Re-run the focused tests and require all selection tests to pass.

### Task 2: Implement Formal Statistics

**Files:**
- Create: `tools/summarize_five_subject_a5_loso_statistics.py`
- Modify: `tests/test_five_subject_a5_loso_statistics.py`

- [ ] Add failing tests for paired delta direction, sample standard deviation, deterministic bootstrap CI and A6 rejection.
- [ ] Implement main/matched-retention aggregation plus subject, part and view paired reports.
- [ ] Use a fixed bootstrap seed and record repetitions/seed in JSON provenance.
- [ ] Re-run the focused tests and require all statistics tests to pass.

### Task 3: Implement Persistent Evaluation Queue

**Files:**
- Create: `tools/run_five_subject_a5_loso_statistics.sh`
- Modify: `tests/test_five_subject_a5_loso_statistics.py`

- [ ] Add a failing source-contract test for five subjects, seven thresholds, four-donor LOSO, A5 method freeze, B0-B4+A5 main table and absence of training commands.
- [ ] Implement validation candidate generation, LOSO selection, frozen test evaluation, status/PID/ETA recording and final statistics invocation.
- [ ] Reuse the existing A5 footprint banks and existing 42k checkpoints.
- [ ] Run `bash -n` and `DRY_RUN=1` and inspect all generated commands.

### Task 4: Verify And Launch

**Files:**
- Generate: `exp/acceptdata/frozen_a5_five_subject_loso_stats_20260723/`

- [ ] Run focused and related semantic protocol tests.
- [ ] Run `git diff --check` for the new implementation files.
- [ ] Launch with `setsid` on GPU 0 and persist stdout/stderr.
- [ ] Verify PID, first validation candidate process, status timestamps and exact Beijing ETA.
- [ ] After completion, validate donor sets, freeze fingerprints, method sets and finite statistics.
