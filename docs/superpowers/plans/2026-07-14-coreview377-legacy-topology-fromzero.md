# CoreView377 Legacy Topology From-Zero Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated legacy topology-formation mode and launch an automatically gated 80k CoreView_377 from-zero run.

**Architecture:** Keep the current Gaussian model, renderer, optimizer restoration, and pointwise state. Dispatch only clone candidate selection and split offset generation through legacy helpers when `model.gaussian.densify_mode=legacy_20260508`; a dedicated option and launcher own the from-zero recipe and gates.

**Tech Stack:** Python 3.9, PyTorch, OmegaConf/Hydra, pytest, Bash.

---

### Task 1: Legacy Densify Behavior

**Files:**
- Modify: `scene/gaussian_model.py`
- Modify: `tests/test_gaussian_model_densification.py`

- [ ] **Step 1: Write failing tests**

Add tests proving the default mode does not seed a zero-gradient recent newborn,
legacy mode does seed it, and legacy risky split children remain in the tangent
plane of their anchor normal.

- [ ] **Step 2: Verify RED**

Run `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_gaussian_model_densification.py -q` and confirm failures because the legacy helpers do not exist.

- [ ] **Step 3: Implement minimal legacy dispatch**

Add mode resolution, bounded recent-newborn seed selection, tangent projection,
and concise event logs. Leave the default branches unchanged.

- [ ] **Step 4: Verify GREEN**

Run the focused densification tests and confirm all pass.

### Task 2: From-Zero Pipeline Contract

**Files:**
- Create: `configs/option/stageA_377_multiview_explicit_hq_fromzero_legacy_topology_v1.yaml`
- Create: `tests/test_coreview377_legacy_topology_fromzero_pipeline.py`
- Create: `exp/zero_train_to_v395/coreview377_legacy_topology_fromzero_20260714_bjt/launch_legacy_topology_fromzero.sh`

- [ ] **Step 1: Write the failing contract test**

Require 50k uniform initialization, 500-12k legacy densification, 70k safety
ceiling, natural pruning, 8k/12k image gates, no historical checkpoint, 80k total
training, smoke overrides, and final automatic selection.

- [ ] **Step 2: Verify RED**

Run the new test and confirm failure because the option and launcher are absent.

- [ ] **Step 3: Implement option and launcher**

Layer the option after the verified stable base, configure the neutral non-rigid
handoff and 64k converter horizon, and support a small real-densify smoke.

- [ ] **Step 4: Verify GREEN and syntax**

Run both focused pytest files, Python compilation, and `bash -n`.

### Task 3: Runtime Validation And Formal Launch

**Files:**
- No additional source files.

- [ ] **Step 1: Run a real-densify smoke**

Use 2k initial points, a 2.6k ceiling, and densification at iterations 5 and 10.
Verify `[LegacyDensify]` events, finite losses, pointwise-state alignment, saved
checkpoint, and clean completion.

- [ ] **Step 2: Run regression tests**

Run densification, non-rigid gate, optimizer scheduler, and launcher tests.

- [ ] **Step 3: Launch formal training**

Start the 80k pipeline detached on GPU 0 and verify process, GPU allocation,
Hydra snapshot, 50k initialization, and first progress output.

- [ ] **Step 4: Report conditional Beijing-time ETA**

Report the 8k gate ETA, 12k gate ETA, and full 80k completion ETA. Provide tracking
commands using `grep`, because `rg` is unavailable in the user's shell.
