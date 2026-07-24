# Five-Subject Semantic Temporal Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and launch a five-subject continuous-frame evaluation that measures frozen semantic-selection stability and produces synchronized paper videos.

**Architecture:** A small metric module computes screen-space selection and temporal-series statistics. A dedicated renderer loads the existing frozen checkpoints and banks, renders camera 21 frames 0-569, and streams comparison videos. A summarizer performs subject-equal aggregation and paired bootstrap statistics, while a restartable shell queue validates provenance and runs the five subjects sequentially on one GPU.

**Tech Stack:** Python, NumPy, PyTorch, existing Gaussian rasterizer, ImageIO/FFmpeg, pytest, Bash.

---

### Task 1: Screen-Space And Temporal Metrics

**Files:**
- Create: `utils/semantic_temporal_stability.py`
- Create: `tests/test_semantic_temporal_stability.py`

- [ ] Write tests for soft/hard IoU, precision, recall, leakage mass, coefficient of variation, and normalized adjacent-frame flicker.
- [ ] Run `pytest -q tests/test_semantic_temporal_stability.py` and verify the tests fail because the module is absent.
- [ ] Implement finite, zero-safe metric helpers with explicit shape checks.
- [ ] Run `pytest -q tests/test_semantic_temporal_stability.py` and verify all tests pass.

### Task 2: Continuous Frozen-Asset Renderer

**Files:**
- Create: `tools/render_semantic_temporal_stability.py`
- Create: `tests/test_render_semantic_temporal_stability.py`

- [ ] Write tests for the CLI contract, expected row count, compact-mask routing, frozen LOSO validation, and video-part defaults.
- [ ] Run `pytest -q tests/test_render_semantic_temporal_stability.py` and verify failure before implementation.
- [ ] Implement continuous dataset configuration for held-out camera 21 and frames 0-569 while preserving direct Hulk compact masks.
- [ ] Render Voting/A5 screen-space selection masks and fixed recolor edits with the shared rasterizer.
- [ ] Stream `upper`, `hair`, and `shoes` comparison panels to 25 FPS H.264 MP4 files and write per-frame metrics/provenance.
- [ ] Run the renderer tests and verify they pass.

### Task 3: Five-Subject Statistical Summary

**Files:**
- Create: `tools/summarize_semantic_temporal_stability.py`
- Create: `tests/test_summarize_semantic_temporal_stability.py`

- [ ] Write tests for per-sequence mean/std/CV, normalized adjacent-frame flicker, subject-equal aggregation, and paired bootstrap output.
- [ ] Run `pytest -q tests/test_summarize_semantic_temporal_stability.py` and verify failure before implementation.
- [ ] Implement CSV loading, finite-value validation, per-subject/per-part tables, overall tables, and A5-minus-Voting bootstrap statistics.
- [ ] Run the summarizer tests and verify they pass.

### Task 4: Restartable Five-Subject Queue

**Files:**
- Create: `tools/run_five_subject_semantic_temporal_stability.sh`
- Create: `tests/test_run_five_subject_semantic_temporal_stability.py`

- [ ] Write tests that assert all five subject roots, frozen A5/LOSO assets, held-out camera 21, 570 frames, expected 6,840 rows, aggregate invocation, and Beijing timestamps.
- [ ] Run `pytest -q tests/test_run_five_subject_semantic_temporal_stability.py` and verify failure before implementation.
- [ ] Implement input validation, subject reuse checks, per-subject logs, status/PID files, and aggregate execution.
- [ ] Run the queue tests and verify they pass.

### Task 5: Smoke Test, Runtime Estimate, And Launch

**Files:**
- Generate: `exp/acceptdata/five_subject_semantic_temporal_stability_smoke_20260724/`
- Generate: `exp/acceptdata/five_subject_semantic_temporal_stability_20260724/`

- [ ] Run all new tests plus the existing frozen-method and real-editing tests.
- [ ] Run a 10-frame CoreView_377 smoke test and validate metrics, MP4 readability, CUDA memory behavior, and output provenance.
- [ ] Estimate the full runtime from measured per-frame wall time with queue overhead.
- [ ] Launch `tools/run_five_subject_semantic_temporal_stability.sh` detached on GPU 0.
- [ ] Verify the live PID, initial status entry, GPU process, and first subject log before reporting the estimated Beijing completion time.
