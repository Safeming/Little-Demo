# CoreView377 Passive Surface Canary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a 5k-to-12k optimizer-preserving diagnostic that disables active carrier competition and reallocation while retaining the passive surface shell.

**Architecture:** A dedicated Hydra overlay disables active surface mechanisms and enables strict checkpoint-state restoration. A standalone launcher reproduces the residual 5k state, evaluates on global iteration numbers, applies 8k/12k LPIPS gates, and writes a final selection report without modifying training internals.

**Tech Stack:** Python 3.9, PyTorch, Hydra/OmegaConf, Bash, pytest, CUDA.

---

### Task 1: Specify The Passive Continuation Contract

**Files:**
- Create: `tests/test_coreview377_passive_surface_canary_pipeline.py`

- [ ] Write a failing test that requires the passive option to restore Gaussian/converter optimizer and scheduler state, disable competition and reallocation, and retain the all-joint passive shell.
- [ ] Write a failing launcher test that requires the exact residual 5k checkpoint, 7k local horizon, global validation milestones, no topology changes, start validation, and final report.
- [ ] Run `/opt/miniconda3/envs/ictrl/bin/python -m pytest -q tests/test_coreview377_passive_surface_canary_pipeline.py` and verify failure because the option and launcher do not exist.

### Task 2: Add The Passive Option And Launcher

**Files:**
- Create: `configs/option/stageA_377_multiview_explicit_hq_fromzero_passive_surface_canary_v1.yaml`
- Create: `exp/zero_train_to_v395/coreview377_passive_surface_canary_20260713_bjt/launch_passive_surface_canary.sh`

- [ ] Add strict optimizer/scheduler restoration and the exact neutral learning-rate schedule used by the 5k checkpoint.
- [ ] Set `lambda_surface_carrier_competition=0`, `residual_surface_reallocation_enable=false`, and preserve the existing passive all-joint anchor-shell schedule.
- [ ] Configure 7k local iterations, global 6k/6.5k/7k/7.5k/8k/10k/12k evaluations, global 8k/12k checkpoint names, and start validation at global 5k.
- [ ] Add 8k and 12k LPIPS gates and a final JSON report that records start, best, final, and gate status.
- [ ] Re-run the pipeline test and verify it passes.

### Task 3: Regression And GPU Smoke

**Files:**
- Test: `tests/test_coreview377_passive_surface_canary_pipeline.py`
- Runtime: `exp/zero_train_to_v395/coreview377_passive_surface_canary_20260713_bjt/smoke_*`

- [ ] Run the passive pipeline test plus optimizer, surface-carrier, tether, reallocation, XYZ scheduler, and converter scheduler tests.
- [ ] Run Python compilation and Bash syntax checks.
- [ ] Run a 12-step GPU smoke with start and end validation.
- [ ] Verify the log contains Gaussian optimizer, converter optimizer, and converter scheduler restoration; global iteration offset 5k; no residual reallocation or surface competition event; and no exception.

### Task 4: Launch And Verify The Formal Canary

**Files:**
- Runtime: `exp/zero_train_to_v395/coreview377_passive_surface_canary_20260713_bjt/run_*`

- [ ] Launch the formal 7k-local-step run detached on GPU 0.
- [ ] Verify the process, GPU allocation, exact 5k start metric, first global training step, optimizer restoration, and absence of fatal errors.
- [ ] Measure live throughput and calculate the Beijing-time completion estimate, including validation overhead.
- [ ] Provide grep-based progress tracking commands that do not require `rg`.
