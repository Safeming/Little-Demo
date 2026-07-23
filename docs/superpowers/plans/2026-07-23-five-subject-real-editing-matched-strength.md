# Five-Subject Real Editing Matched-Strength Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure real-edit leakage at matched target edit magnitude for Raw Hard, Voting, and frozen A5 on five subjects.

**Architecture:** Extend the existing real-edit renderer with a fixed multi-strength, metrics-only mode. Add a separate curve summarizer that excludes unsupported references, interpolates at frozen retention levels, reports common coverage, and produces subject-level paired statistics. A restartable queue runs the five frozen checkpoints.

**Tech Stack:** Python, NumPy, PyTorch, diff-gaussian-rasterization, pytest, Bash.

---

### Task 1: Multi-Strength Renderer Interface

**Files:**
- Modify: `tools/render_semantic_real_editing_paper_suite.py`
- Modify: `tests/test_render_semantic_real_editing_paper_suite.py`

- [ ] Add failing tests for `--edit-strengths`, sorted unique range validation, backward-compatible single strength, and `--metrics-only`.
- [ ] Run the focused tests and confirm failures for the missing interface.
- [ ] Implement `resolve_edit_strengths(args)` and CLI options.
- [ ] Loop the existing method/task/part matrix over resolved strengths without changing weight selection or rasterization.
- [ ] Suppress PNG writes only when `--metrics-only` is active and record the mode in provenance.
- [ ] Run focused and fixed-strength queue regression tests.

### Task 2: Matched-Strength Curve Utilities

**Files:**
- Create: `utils/semantic_matched_strength.py`
- Create: `tests/test_semantic_matched_strength.py`

- [ ] Write failing tests for Voting@1.0 references, zero-reference exclusion, origin insertion, duplicate retention collapse, and interpolation at 0.25/0.50.
- [ ] Run the tests and confirm failure because the module is absent.
- [ ] Implement curve construction and interpolation.
- [ ] Add failing tests for method coverage and pairwise common coverage.
- [ ] Implement coverage helpers and rerun the focused tests.

### Task 3: Five-Subject Matched Summarizer

**Files:**
- Create: `tools/summarize_semantic_real_editing_matched_strength.py`
- Create: `tests/test_summarize_semantic_real_editing_matched_strength.py`

- [ ] Write failing tests for direct CLI startup, subject-equal aggregation, paired A5-baseline deltas, bootstrap intervals, finite output, and per-part coverage.
- [ ] Implement subject metric loading, curve matching, CSV/JSON outputs, and statistics.
- [ ] Run focused tests until green.

### Task 4: Restartable Five-Subject Queue

**Files:**
- Create: `tools/run_five_subject_real_editing_matched_strength.sh`
- Create: `tests/test_run_five_subject_real_editing_matched_strength.py`

- [ ] Write failing source tests for five subjects, five strengths, metrics-only mode, all tasks/methods/parts/views, restart checks, Beijing ETA, and final summarization.
- [ ] Implement the queue and dry-run mode.
- [ ] Run focused tests until green.

### Task 5: Verification And Launch

**Files:**
- Modify: `docs/严格语义论文协议执行记录与下一步投稿缺口_20260718.md`

- [ ] Run the full relevant semantic protocol regression set.
- [ ] Run a one-record CoreView 377 GPU smoke and verify exactly 270 metric rows with no frame requirement.
- [ ] Estimate the five-subject runtime from smoke and launch the detached queue.
- [ ] Record PID, Beijing start time, ETA, output root, and frozen protocol in the strict execution document.
- [ ] Commit only task-related changes with Chinese commit messages.
