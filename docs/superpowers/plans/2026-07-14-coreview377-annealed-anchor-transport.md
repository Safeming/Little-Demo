# CoreView377 Annealed Anchor Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and launch an isolated 8k from-zero canary that gradually transports stale anchors without changing the proven 80k neutral point topology.

**Architecture:** Add pure selection and blending helpers to `ExplicitBinding`, schedule one bounded transport event per eligible iteration, and reuse the existing partial anchor search. Persist the blended state through the canonical Gaussian binding-state checkpoint path and expose the behavior through one opt-in option and launcher.

**Tech Stack:** PyTorch, Hydra/OmegaConf, pytest, Bash, existing CoreView377 training/evaluation pipeline.

---

### Task 1: Transport Selection And Blending

**Files:**
- Modify: `models/deformer/rigid.py`
- Create: `tests/test_anchor_transport.py`

- [ ] Write failing tests proving that candidate selection applies the tangent threshold and top-k cap, and that blending reduces tangent offset while preserving normalized weights and normals.
- [ ] Run `python -m pytest tests/test_anchor_transport.py -q` and confirm RED failures because the helper methods do not exist.
- [ ] Add `_anchor_transport_candidate_mask` and `_blend_anchor_transport_subset` as deterministic helper methods.
- [ ] Re-run the focused tests and confirm GREEN.

### Task 2: Once-Per-Iteration Runtime Scheduling

**Files:**
- Modify: `models/deformer/rigid.py`
- Modify: `tests/test_anchor_transport.py`

- [ ] Write failing tests for disabled behavior, schedule boundaries, once-per-iteration execution, and persisted blended owner state.
- [ ] Run the new tests and confirm RED for missing runtime scheduling.
- [ ] Add configuration parsing, `_maybe_schedule_anchor_transport`, partial-refresh blend integration, and `[AnchorTransport]` diagnostics.
- [ ] Re-run transport tests and relevant explicit-binding tests.

### Task 3: Canary Configuration And Launcher

**Files:**
- Create: `configs/option/stageA_377_multiview_explicit_hq_fromzero_anchor_transport_v1.yaml`
- Create: `tests/test_coreview377_anchor_transport_pipeline.py`
- Create: `exp/zero_train_to_v395/coreview377_anchor_transport_20260714_bjt/launch_anchor_transport.sh`

- [ ] Write contract tests for the 80k focus baseline, disabled densify/reset, transport schedule, 8k gate, smoke overrides, and final marker.
- [ ] Run the contract test and confirm RED because config and launcher are absent.
- [ ] Add the option and one-command launcher with a 20-step smoke mode and an 8k formal canary.
- [ ] Run contract tests and Bash syntax validation.

### Task 4: Verification And Launch

**Files:**
- Verify only; no additional implementation files expected.

- [ ] Run transport tests, existing binding/non-rigid tests, pipeline contract tests, Python compilation, and `git diff --check` on touched files.
- [ ] Run a real GPU smoke and require at least one nonzero `[AnchorTransport]` event plus a completed evaluation/checkpoint marker.
- [ ] Launch the 8k canary in an independent session.
- [ ] Verify PID, GPU allocation, 80k focus initialization, disabled densify, first training progress, and transport logs.
- [ ] Report the run path, gate time estimate, tracking command, and the next decision rule based on LPIPS plus tangent-offset diagnostics.
