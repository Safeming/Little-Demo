# Unified A5 Paper Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a paper-ready five-subject A0-A6 ablation and isolate the contributions of full Gaussian footprint evidence and the outer leakage penalty under the frozen A5 LOSO protocol.

**Architecture:** A persistent evaluation queue reuses the five frozen A5 LOSO configs and existing 42k checkpoints. It builds two calibration-only footprint banks per subject, evaluates the complete component chain plus two A5 micro variants, and aggregates segmentation, matched-retention leakage, and paired subject statistics.

**Tech Stack:** Bash, Python, CSV/JSON, pytest, CUDA evaluator.

---

### Task 1: Lock Ablation Contracts

**Files:**
- Create: `tests/test_unified_a5_paper_ablation.py`

- [ ] Assert five subjects and the frozen LOSO config root are present.
- [ ] Assert the component stage evaluates `A0 A1 A2 A3 A4 A5 A6`.
- [ ] Assert center-only calibration uses footprint radius 0.
- [ ] Assert no-outer calibration uses outer penalty power 0.
- [ ] Assert all test stages use A0 as the retention reference and no training command exists.
- [ ] Run the focused test and confirm failure because the queue and summarizer do not exist.

### Task 2: Implement Paper Ablation Summarizer

**Files:**
- Create: `tools/summarize_unified_a5_paper_ablation.py`
- Modify: `tests/test_unified_a5_paper_ablation.py`

- [ ] Add failing tests for exact A0-A6 validation and variant labels.
- [ ] Aggregate component segmentation metrics with subject, mean and standard-deviation rows.
- [ ] Aggregate `A4`, `A5-center-only`, `A5-no-outer-penalty`, and full `A5` at retention 0.5/0.6.
- [ ] Produce paired subject bootstrap statistics for each A5 variant against A4.
- [ ] Write `component_table.csv`, `a5_micro_ablation_table.csv`, `matched_retention_table.csv`, `paired_statistics.csv`, and `summary.json`.

### Task 3: Implement Persistent Queue

**Files:**
- Create: `tools/run_unified_a5_paper_ablation.sh`
- Modify: `tests/test_unified_a5_paper_ablation.py`

- [ ] Reuse `exp/acceptdata/frozen_a5_five_subject_loso_stats_20260723/CoreView_<subject>/loso_frozen_config.json`.
- [ ] Build a center-only bank with radius scale/min/max all set to zero and outer penalty 0.2.
- [ ] Build a no-outer bank with the normal footprint and outer penalty zero.
- [ ] Evaluate A0-A6 with the full A5 bank.
- [ ] Evaluate A0/A4/A5 with each micro-ablation bank.
- [ ] Record PID, Beijing timestamps, resumable stage status and ETA.

### Task 4: Verify And Launch

**Files:**
- Generate: `exp/acceptdata/unified_a5_paper_ablation_20260723/`

- [ ] Run focused and related semantic protocol tests.
- [ ] Run `bash -n`, full dry-run and `git diff --check`.
- [ ] Launch persistently on GPU 0.
- [ ] Verify the first alternative bank process and record the Beijing ETA.
- [ ] After completion, verify method sets, LOSO provenance, finite metrics and final aggregate files.
