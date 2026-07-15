# CoreView377 Monotonic Local Anchor Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch an 8k from-zero canary that transports only geometry-valid local anchor positions while preserving old deformation and visibility binding fields.

**Architecture:** Extend `ExplicitBinding` with pure local candidate and geometry-acceptance helpers, then specialize the existing partial-refresh transport path to update only accepted `anchor_xyz` rows and a pointwise transport counter. Reuse the neutral 80k launcher and dedicated LPIPS gate through one option and one wrapper launcher.

**Tech Stack:** PyTorch, Hydra/OmegaConf, pytest, Bash, existing CoreView377 training pipeline.

---

### Task 1: Local Candidate And Geometry Gate

**Files:**
- Modify: `models/deformer/rigid.py`
- Modify: `tests/test_anchor_transport.py`

- [ ] Add a failing test proving candidate selection intersects tangent threshold, configured joints, and transport-count cap before top-k.
- [ ] Run `python -m pytest -q tests/test_anchor_transport.py` and confirm the failure is the missing local eligibility behavior.
- [ ] Extend `_select_anchor_transport_candidates` with optional `dominant_joint`, `joint_ids`, `transport_count`, and `max_accept_count` arguments.
- [ ] Add a failing test proving geometry acceptance requires same joint, tangent improvement, bounded normal regression, and local-offset improvement.
- [ ] Add `_anchor_transport_geometry_acceptance` returning the accepted mask plus per-reason counts.
- [ ] Re-run the focused tests and require green.

### Task 2: Position-Only Persistent Transport

**Files:**
- Modify: `models/deformer/rigid.py`
- Modify: `tests/test_anchor_transport.py`

- [ ] Add a failing test proving position-only blending changes only `anchor_xyz` and increments accepted rows in `anchor_transport_count`.
- [ ] Add `_apply_position_only_anchor_transport` as a pure helper that preserves every other binding field.
- [ ] Parse the new configuration with disabled-compatible defaults.
- [ ] In `_refresh_binding_subset`, evaluate fresh targets with the old normal, filter through the geometry gate, merge only accepted anchor positions, and clear refresh masks for all selected rows.
- [ ] Add `[MonotonicAnchorTeacher]` logging for selected, accepted, rejection counts, geometry before/after, and saturated counts.
- [ ] Run transport and binding-focused tests.

### Task 3: Canary Option And Launcher

**Files:**
- Create: `configs/option/stageA_377_multiview_explicit_hq_fromzero_monotonic_anchor_teacher_v1.yaml`
- Create: `tests/test_coreview377_monotonic_anchor_teacher_pipeline.py`
- Create: `exp/zero_train_to_v395/coreview377_monotonic_anchor_teacher_20260715_bjt/launch_monotonic_anchor_teacher.sh`
- Modify: `train.py`

- [ ] Add failing contract tests for the joint scope, geometry thresholds, position-only mode, count cap, fixed 80k topology, 8k schedule, and smoke overrides.
- [ ] Add a dedicated image-gate helper/config using neutral 5k and 8k baselines, tolerance 0.0005, and failure marker `MONOTONIC_ANCHOR_TEACHER_GATE_FAILED`.
- [ ] Add the option and wrapper launcher with formal 8k and 20-step smoke modes.
- [ ] Run contract tests and `bash -n`.

### Task 4: Verification And Launch

**Files:**
- Verify only.

- [ ] Run Python compilation, transport tests, pipeline tests, non-rigid/tether regressions, and `git diff --check` on touched files.
- [ ] Run a real GPU smoke with transport at iterations 5, 10, and 15; require a completed checkpoint and nonzero selected-path logs without a traceback.
- [ ] Start the formal 8k run in a detached session.
- [ ] Verify `train.py`, GPU memory, 80k initialization, fixed topology overrides, first progress, and the first scheduled teacher event.
- [ ] Report the run path and an end-time range derived from the previous 1h29m canary plus current startup time.
