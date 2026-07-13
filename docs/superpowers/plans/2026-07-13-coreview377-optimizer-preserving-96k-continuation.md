# CoreView377 Optimizer-Preserving 96k Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue the verified neutral64k from-zero trajectory to schedule iteration 96k while preserving all saved optimizer and scheduler state.

**Architecture:** Persist the Gaussian optimizer state loaded by `GaussianModel.restore()` until the post-load `training_setup()` creates matching parameter groups, then apply it under a strict resume contract. Add a dedicated 32k-local-step launcher that restores Gaussian and converter optimization state, uses the 64k checkpoint as schedule offset, disables topology changes, evaluates every 2k, and selects the best continuation metric automatically.

**Tech Stack:** Python 3.9, PyTorch Adam, Hydra/OmegaConf, pytest, Bash, CUDA Gaussian rasterizer.

---

### Task 1: Gaussian Optimizer State Restoration

**Files:**
- Modify: `scene/gaussian_model.py`
- Modify: `tests/test_gaussian_model_densification.py`

- [ ] **Step 1: Write a failing round-trip test**

Create Adam moments on the existing `_densify_ready_model`, capture it, restore it with `restore_gaussian_optimizer_state=true`, call `training_setup`, and assert the restored XYZ `exp_avg`, `exp_avg_sq`, and optimizer group learning rates equal the captured values.

- [ ] **Step 2: Run RED**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_gaussian_model_densification.py::test_gaussian_optimizer_state_survives_checkpoint_restore -q
```

Expected: FAIL because `GaussianModel.restore()` currently discards `opt_dict` and the second `training_setup()` creates an empty Adam state.

- [ ] **Step 3: Implement pending strict restore**

Store `opt_dict` as a pending optimizer state when requested. After `training_setup()` creates the optimizer, validate parameter-group count, load the pending state, clear it, and print `Resume safety: restored Gaussian optimizer state.`. When `require_gaussian_optimizer_state_restore=true`, raise on missing state, mismatched groups, or load failure.

- [ ] **Step 4: Add and verify strict failure test**

Capture a model without Adam state, request strict restore, call `training_setup`, and assert a `RuntimeError` mentioning Gaussian optimizer restoration.

- [ ] **Step 5: Run GREEN**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_gaussian_model_densification.py -q
```

Expected: PASS.

### Task 2: Continuation Configuration and Launcher

**Files:**
- Create: `configs/option/stageA_377_multiview_explicit_hq_fromzero_optimizer_preserving_continuation_v1.yaml`
- Create: `exp/zero_train_to_v395/coreview377_optimizer_preserving_96k_20260713_bjt/launch_optimizer_preserving_96k.sh`
- Create: `tests/test_coreview377_optimizer_preserving_96k_pipeline.py`

- [ ] **Step 1: Write failing pipeline contract tests**

Assert a 32k local horizon, exact neutral64k `ckpt64000.pth`, checkpoint iteration offset, all three restore flags, strict Gaussian restore, no densification/pruning/opacity reset, and no v395 checkpoint dependency.

- [ ] **Step 2: Run RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_coreview377_optimizer_preserving_96k_pipeline.py -q
```

Expected: FAIL because the option and launcher do not exist.

- [ ] **Step 3: Add option and launcher**

Use the exact neutral option stack and start checkpoint. Run local iterations `32000`, evaluate local steps corresponding to schedule 66k-96k, save at 72k/80k/88k/96k, restore optimization state, and write a final JSON containing baseline and continuation-best metrics.

- [ ] **Step 4: Run GREEN and shell syntax**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_coreview377_optimizer_preserving_96k_pipeline.py -q
bash -n exp/zero_train_to_v395/coreview377_optimizer_preserving_96k_20260713_bjt/launch_optimizer_preserving_96k.sh
```

Expected: PASS.

### Task 3: Verification, GPU Smoke, and Formal Launch

**Files:**
- Runtime output: `exp/zero_train_to_v395/coreview377_optimizer_preserving_96k_20260713_bjt/`

- [ ] **Step 1: Run focused regression and compile checks**

Run Gaussian, converter scheduler, XYZ scheduler, and new pipeline tests plus Python and shell compilation. Expected: zero failures.

- [ ] **Step 2: Run a real-checkpoint GPU smoke**

Use a short local horizon from `ckpt64000.pth`. Verify strict Gaussian optimizer restoration, converter optimizer and scheduler restoration, schedule-offset logs, a nonzero training step, evaluation, and checkpoint creation.

- [ ] **Step 3: Launch the formal 32k continuation detached**

Use `setsid -f`, verify launcher and training PIDs, checkpoint load, all restore logs, schedule progress above 64k, GPU allocation, and no traceback.

- [ ] **Step 4: Report tracking commands and BJT ETA**

Estimate completion from measured formal throughput and evaluation overhead. Provide `grep`-based commands because `rg` is unavailable in the user's shell.
