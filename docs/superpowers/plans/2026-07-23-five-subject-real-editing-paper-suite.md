# Five-Subject Real Editing Paper Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a frozen five-subject paper suite for recolor, opacity removal, and texture replacement using Raw Hard, Voting, and A5 with a shared renderer.

**Architecture:** Add pure edit/metric helpers, a backward-compatible opacity override in the rasterizer, one subject-level renderer, one aggregate summarizer, and a restartable five-subject queue. Test masks remain evaluation-only and all edit selections come from frozen Gaussian banks and LOSO thresholds.

**Tech Stack:** Python, NumPy, PyTorch, diff-gaussian-rasterization, Pillow/imageio, pytest, Bash.

---

### Task 1: Pure Edit Operations And Metrics

**Files:**
- Create: `utils/semantic_real_editing.py`
- Create: `tests/test_semantic_real_editing.py`

- [ ] Write failing tests for Raw Hard, Voting, and A5 weight routing, including A5 thresholding.
- [ ] Run `pytest -q tests/test_semantic_real_editing.py` and confirm failure because the module is absent.
- [ ] Implement minimal weight routing and validation.
- [ ] Add failing tests for deterministic canonical stripe colors and recolor/removal/texture transforms.
- [ ] Implement the edit transforms and rerun the focused tests.
- [ ] Add failing tests for target, outer, and boundary delta metrics.
- [ ] Implement the metrics and confirm the focused suite passes.

### Task 2: Shared Rasterizer Opacity Override

**Files:**
- Modify: `gaussian_renderer/__init__.py`
- Create: `tests/test_gaussian_rasterizer_overrides.py`

- [ ] Write a failing source-level/API test requiring optional `opacities_precomp` routing with unchanged default behavior.
- [ ] Run `pytest -q tests/test_gaussian_rasterizer_overrides.py` and confirm the expected failure.
- [ ] Add `opacities_precomp=None` to `rasterize_gaussians` and select it only when provided.
- [ ] Run the focused test and existing renderer-related tests.

### Task 3: Subject-Level Real Editing Renderer

**Files:**
- Create: `tools/render_semantic_real_editing_paper_suite.py`
- Create: `tests/test_render_semantic_real_editing_paper_suite.py`

- [ ] Write failing tests for CLI parsing, frozen input validation, declared method/task matrix, and evaluation-only parser provenance.
- [ ] Run the focused tests and confirm they fail because the tool is absent.
- [ ] Implement CLI/config validation and pure manifest helpers.
- [ ] Implement one-scene rendering for all records, parts, tasks, and methods.
- [ ] Export base/edited frames, `metrics.csv`, and `summary.json` with fingerprints and frozen threshold provenance.
- [ ] Run focused tests until green.

### Task 4: Aggregate Statistics And Paper Sheets

**Files:**
- Create: `tools/summarize_semantic_real_editing_paper_suite.py`
- Create: `tests/test_summarize_semantic_real_editing_paper_suite.py`

- [ ] Write failing tests for subject-equal means, paired deltas, deterministic bootstrap intervals, and fixed-subset contact sheets.
- [ ] Run the focused tests and confirm expected failure.
- [ ] Implement CSV aggregation, paired statistics, JSON summary, and paper sheets.
- [ ] Run focused tests until green.

### Task 5: Restartable Five-Subject Queue

**Files:**
- Create: `tools/run_five_subject_real_editing_paper_suite.sh`
- Create: `tests/test_run_five_subject_real_editing_paper_suite.py`

- [ ] Write failing tests for five-subject paths, frozen LOSO config routing, all nine test records, all six parts, three tasks, three methods, status timestamps, restart checks, and final summarization.
- [ ] Run the focused tests and confirm expected failure.
- [ ] Implement the queue and dry-run mode.
- [ ] Run focused tests until green.

### Task 6: Verification And Execution

**Files:**
- Modify: `docs/严格语义论文协议执行记录与下一步投稿缺口_20260718.md`

- [ ] Run all focused tests and the relevant semantic protocol regression suite.
- [ ] Run a single-record GPU smoke for CoreView 377 and verify 54 edited frames plus summaries.
- [ ] Start the five-subject background queue and record PID, Beijing start time, and measured ETA.
- [ ] Append the frozen experiment definition, output directory, PID, and ETA to the strict protocol record.
- [ ] Commit implementation and documentation without staging unrelated dirty files.
