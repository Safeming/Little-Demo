# CoreView377 Anchor Teacher Fade-Out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch an 8k from-zero canary whose monotonic local anchor teacher fades to zero from 6000 to 6500 and leaves a 1500-step recovery window.

**Architecture:** Add a pure effective-alpha schedule to `ExplicitBinding`, use it for scheduling, applying, and logging transport events, and expose the schedule through one option and one wrapper launcher. All geometry gates and the neutral fixed-topology pipeline remain unchanged.

**Tech Stack:** PyTorch, Hydra/OmegaConf, pytest, Bash, CoreView377 training pipeline.

---

### Task 1: Effective Alpha Schedule

**Files:**
- Modify: `models/deformer/rigid.py`
- Modify: `tests/test_anchor_transport.py`

- [ ] Add a failing test requiring alpha 0.10 at 6000, 0.08 at 6100, 0.04 at 6300, and 0.0 at 6500.
- [ ] Add a failing scheduling test requiring a zero-alpha event to be skipped.
- [ ] Parse `anchor_transport_fade_start_iter` and `anchor_transport_fade_end_iter` with disabled-compatible defaults.
- [ ] Implement `_anchor_transport_effective_alpha(iteration)` and use it in scheduling, both transport apply paths, and logs.
- [ ] Run `python -m pytest -q tests/test_anchor_transport.py` and require green.

### Task 2: Option And Launcher

**Files:**
- Create: `configs/option/stageA_377_multiview_explicit_hq_fromzero_monotonic_anchor_teacher_fadeout_v1.yaml`
- Create: `tests/test_coreview377_anchor_teacher_fadeout_pipeline.py`
- Create: `exp/zero_train_to_v395/coreview377_anchor_teacher_fadeout_20260715_bjt/launch_anchor_teacher_fadeout.sh`

- [ ] Add failing contract tests for fade start/end, unchanged geometry gates, fixed topology, 8k endpoint, and smoke overrides.
- [ ] Add the option using fade start 6000 and end 6500.
- [ ] Add the wrapper launcher with 20-step smoke fade 10-15 and formal 8k mode.
- [ ] Run contract tests and `bash -n`.

### Task 3: Verification And Launch

**Files:**
- Verify only.

- [ ] Run Python compilation, transport tests, fade pipeline tests, existing monotonic/neutral regressions, and `git diff --check`.
- [ ] Run a GPU smoke and require alpha 0.10 at 10, alpha 0.0/no event at 15, completed evaluation, and checkpoint.
- [ ] Start the formal 8k run in an independent session.
- [ ] Verify process, GPU use, 80k initialization, first progress, and fade configuration propagation.
- [ ] Report the run path and Beijing end-time estimate from the previous 74-minute train plus evaluation/diagnostic overhead.
