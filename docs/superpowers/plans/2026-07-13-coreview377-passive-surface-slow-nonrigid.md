# CoreView377 Passive Surface Slow Non-Rigid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch a controlled continuation that changes only the passive surface canary's non-rigid activation from 3k-6.5k to 5k-10k.

**Architecture:** Add a narrow Hydra option layered after the existing passive canary and a dedicated launcher that reuses its checkpoint, optimizer restoration, topology lock, evaluation gates, and reporting. Contract tests prove the schedule and isolation before a smoke run and formal launch.

**Tech Stack:** Python 3.9, pytest, PyYAML, Hydra/OmegaConf, Bash, PyTorch.

---

### Task 1: Configuration And Launcher Contracts

**Files:**
- Create: `tests/test_coreview377_passive_surface_slow_nonrigid_pipeline.py`
- Create: `configs/option/stageA_377_multiview_explicit_hq_fromzero_passive_surface_slow_nonrigid_v1.yaml`
- Create: `exp/zero_train_to_v395/coreview377_passive_surface_slow_nonrigid_20260713_bjt/launch_passive_surface_slow_nonrigid.sh`

- [ ] **Step 1: Write the failing tests**

Require a 5k-to-10k smoothstep schedule, unchanged passive tether and gates,
the same from-zero 5k checkpoint, strict optimizer restoration, no active carrier
changes, exact evaluation points, and smoke-mode gate disabling.

- [ ] **Step 2: Verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_coreview377_passive_surface_slow_nonrigid_pipeline.py -q`

Expected: fail because the option and launcher do not exist.

- [ ] **Step 3: Add the minimal option and launcher**

Layer the new option after the passive canary, override only the non-rigid timing
and experiment marker, and retain the passive canary's training contract.

- [ ] **Step 4: Verify GREEN and shell syntax**

Run the focused pytest file and `bash -n` on the launcher. Expected: all tests
pass and shell syntax exits zero.

### Task 2: Smoke And Formal Launch

**Files:**
- No additional production files.

- [ ] **Step 1: Run a 12-step smoke continuation**

Set `SMOKE=1`, verify all optimizer states restore, global iteration starts at
5000, the non-rigid gate follows the new schedule, and the smoke completes.

- [ ] **Step 2: Run focused regression tests**

Run the new contract test with the passive canary and non-rigid gate tests.
Expected: all selected tests pass.

- [ ] **Step 3: Start the formal 7k-step continuation**

Launch with `setsid`, capture the run directory and PID, and verify the process,
GPU allocation, restored states, starting metric, and first progress output.

- [ ] **Step 4: Calculate ETA and provide tracking commands**

Use the measured passive-canary throughput and the formal start time. Report the
8k decision-gate ETA, full 12k ETA, run path, and commands that do not require
`rg`.
