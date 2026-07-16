# CoreView377 Surface Responsibility V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and launch a 12k from-zero canary using 80k deterministic surface-area carriers without competition or topology changes.

**Architecture:** A small Hydra option owns the weak global shell and disables active carrier mechanisms. A Bash wrapper reuses the proven neutral long-horizon launcher while overriding only initialization, horizon, evaluation, and checkpoint schedules.

**Tech Stack:** Hydra YAML, Bash, pytest, PyTorch/CUDA training stack.

---

### Task 1: Configuration And Launcher Contract

**Files:**
- Create: `tests/test_coreview377_surface_responsibility_v2_pipeline.py`
- Create: `configs/option/stageA_377_multiview_explicit_hq_fromzero_surface_responsibility_v2.yaml`
- Create: `tools/run_coreview377_surface_responsibility_v2.sh`

- [ ] **Step 1: Write the failing test**

Assert that the option disables competition/reallocation, keeps an all-joint weak
normal shell, and that the launcher requests 80k `surface_carrier_v1` points,
12k iterations, standard non-rigid scheduling, no checkpoint load, and saves at
5k/8k/12k.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest -q tests/test_coreview377_surface_responsibility_v2_pipeline.py
```

Expected: failure because the option and launcher do not exist.

- [ ] **Step 3: Add the minimal option and wrapper**

The option must set:

```yaml
opt:
  lambda_surface_carrier_competition: 0.0
  residual_surface_reallocation_enable: false
  lambda_local_anchor_tether: [0.0, 200, 0.001, 1000, 0.004, 12000, 0.004]
  local_anchor_tether_joint_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
  local_anchor_tether_normal_limit: 0.055
  local_anchor_tether_tangent_limit: 0.15
  local_anchor_tether_normal_weight: 1.0
  local_anchor_tether_tangent_weight: 0.02
```

The wrapper must pass 80k deterministic surface-carrier overrides after the base
launcher's focus overrides, keep densification disabled, and expose `RUN`,
`BASE_RUN`, `SMOKE`, and `CUDA_VISIBLE_DEVICES`.

- [ ] **Step 4: Run focused tests and shell syntax verification**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest -q tests/test_coreview377_surface_responsibility_v2_pipeline.py
bash -n tools/run_coreview377_surface_responsibility_v2.sh
```

Expected: all tests pass and shell syntax exits zero.

### Task 2: Runtime Verification And Launch

**Files:**
- Verify: `tools/run_coreview377_surface_responsibility_v2.sh`

- [ ] **Step 1: Run related regression tests**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest -q \
  tests/test_coreview377_surface_responsibility_v2_pipeline.py \
  tests/test_coreview377_surface_carrier_formation_pipeline.py \
  tests/test_surface_carrier_sampling.py \
  tests/test_local_anchor_tether.py
```

- [ ] **Step 2: Run a 20-step CUDA smoke**

Launch with `SMOKE=1` and verify the log contains 80,000 initialized points,
`SurfaceCarrierInit`, local tether activity, checkpoint creation, and no traceback.

- [ ] **Step 3: Launch the formal detached canary**

Create a timestamped run directory under `exp/zero_train_to_v395`, start the
wrapper with `setsid -f`, and verify the pipeline PID, child `train.py`, GPU memory,
and increasing training progress.

- [ ] **Step 4: Report the run path and Beijing-time ETA**

Estimate completion from the observed iteration rate and include a `grep`-based
tracking command because the user's shell does not provide `rg`.

