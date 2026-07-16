# CoreView_377 Tether Continuation Full Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch an automatic tether64 continuation canary that starts a complete CoreView_377 from-zero rerun only after strict raw LPIPS, PSNR, and contour gates pass.

**Architecture:** Add a focused Python selector for multi-profile raw metrics and a Bash orchestrator that reuses the verified tether, optimizer-preserving, late-clean, and residual-balanced training options. The canary begins from an existing from-zero tether64 checkpoint to test the missing continuation hypothesis; a passing canary triggers a second chain whose stage1 starts from input data and whose later checkpoints all remain inside the new run directory.

**Tech Stack:** Python 3.9, PyTorch/Hydra, Bash, pytest, CUDA renderer.

---

### Task 1: Raw multi-metric selector

**Files:**
- Create: `tools/select_tether_quality_candidate.py`
- Create: `tests/test_tether_quality_selector.py`

- [ ] **Step 1: Write failing selector tests**

```python
from tools.select_tether_quality_candidate import evaluate_final_candidate, select_continuation


def test_continuation_selection_prefers_raw_pareto_candidate():
    candidates = [
        {"label": "lpips_only", "lpips_fg": 0.1290, "psnr_fg": 21.80,
         "edge_px": 3.20, "boundary_l1": 0.069},
        {"label": "balanced", "lpips_fg": 0.1294, "psnr_fg": 21.90,
         "edge_px": 3.00, "boundary_l1": 0.0675},
    ]
    assert select_continuation(candidates)["label"] == "balanced"


def test_final_gate_ignores_legacy_psnr_for_raw_selection():
    result = evaluate_final_candidate(
        same30={"lpips_fg": 0.1260, "psnr_fg": 21.97},
        original57={"lpips_fg": 0.1287, "psnr_fg": 21.80},
        contour={"edge_px": 2.88, "boundary_l1": 0.0670},
    )
    assert result["accepted"] is True
```

- [ ] **Step 2: Run RED**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_tether_quality_selector.py -q
```

Expected: collection fails because `tools.select_tether_quality_candidate` does not exist.

- [ ] **Step 3: Implement minimal selector**

Implement `select_continuation(candidates)` by filtering candidates within `0.001` of the minimum LPIPS, then sorting by `(edge_px, boundary_l1, -psnr_fg, lpips_fg)`. Implement `evaluate_final_candidate` with the exact design thresholds and return per-gate booleans plus `accepted`.

- [ ] **Step 4: Run GREEN**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_tether_quality_selector.py -q
```

Expected: all selector tests pass.

### Task 2: Automatic canary/full-chain launcher contract

**Files:**
- Create: `tools/run_coreview377_tether_quality_full_pipeline.sh`
- Create: `tests/test_coreview377_tether_quality_full_pipeline.py`

- [ ] **Step 1: Write failing launcher contract tests**

Assert that the launcher contains:

```python
assert "CANARY_TETHER_CKPT" in source
assert "ckpt64000.pth" in source
assert "run_continuation_tail" in source
assert "select_tether_quality_candidate.py" in source
assert "camera_geometry_enable=false" in source
assert "CANARY_REJECTED" in source
assert "FULL_FROMZERO_START" in source
assert "launch_surface_coherent_anchor_tether.sh" in source
assert "v395" not in checkpoint_assignment.lower()
```

Also assert that full-stage checkpoint arguments are derived from `$FULL_RUN`, not historical run directories.

- [ ] **Step 2: Run RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_coreview377_tether_quality_full_pipeline.py -q
```

Expected: FAIL because the launcher is absent.

- [ ] **Step 3: Implement launcher**

The launcher must:

```text
canary tether64 checkpoint
  -> 32k optimizer-preserving continuation
  -> render/evaluate saved continuation candidates
  -> automatic raw Pareto selection
  -> 4k late-clean
  -> 3k residual-balanced
  -> strict final gate
  -> if pass: start from-zero tether64 and repeat the same tail
```

Use subject/data/seed environment variables, write `pipeline_state.json`, `FINAL_BEST_CKPT.txt`, `CANARY_RESULT.json`, and `FULL_RESULT.json`, and emit `PIPELINE_DONE_BJT`. All raw validation commands explicitly set `opt.camera_geometry_enable=false` and `opt.camera_affine_enable=false`.

- [ ] **Step 4: Run GREEN and shell syntax**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_coreview377_tether_quality_full_pipeline.py -q
bash -n tools/run_coreview377_tether_quality_full_pipeline.sh
```

Expected: tests pass and Bash exits 0.

### Task 3: Focused regressions and CUDA smoke

**Files:**
- Modify only if a failing regression exposes a launcher/selector defect.

- [ ] **Step 1: Run focused regressions**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest -q \
  tests/test_tether_quality_selector.py \
  tests/test_coreview377_tether_quality_full_pipeline.py \
  tests/test_coreview377_surface_coherent_anchor_tether_pipeline.py \
  tests/test_coreview377_optimizer_preserving_96k_pipeline.py \
  tests/test_coreview377_late_clean_refine_pipeline.py \
  tests/test_coreview377_residual_balanced_refine_pipeline.py \
  tests/test_residual_balanced_refine_selector.py
```

Expected: zero failures.

- [ ] **Step 2: Run short CUDA smoke**

Run the new launcher with `SMOKE=1`, continuation/tail steps reduced to 2-4 iterations, and `STOP_AFTER_CANARY=1`. Verify checkpoint loading, optimizer restoration, diagnostic export, selector output, rejection/pass handling, and clean process exit.

- [ ] **Step 3: Inspect smoke artifacts**

Verify:

```text
logs/pipeline.log
CANARY_RESULT.json
canary/continuation/ckpt*.pth
canary/final_selection.json
```

No traceback, missing checkpoint, or camera-geometry activation is allowed.

### Task 4: Formal detached launch and ETA

**Files:**
- Runtime output: `exp/zero_train_to_v395/coreview377_tether_quality_full_pipeline_20260716_bjt/`

- [ ] **Step 1: Launch detached**

```bash
RUN_TAG=coreview377_tether_quality_full_pipeline_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S')_bjt
setsid -f env RUN_TAG="$RUN_TAG" CUDA_VISIBLE_DEVICES=0 \
  bash tools/run_coreview377_tether_quality_full_pipeline.sh \
  > "exp/zero_train_to_v395/$RUN_TAG/nohup.log" 2>&1
```

- [ ] **Step 2: Verify live progress**

Confirm the launcher PID, training PID, GPU allocation, successful tether64 checkpoint load, optimizer/scheduler restoration, and increasing training iteration.

- [ ] **Step 3: Calculate Beijing ETA**

Use the actual start timestamp and measured smoke/formal throughput. Report the canary ETA and conditional full-pipeline ETA separately, including a 1-2 hour evaluation/I/O range.

- [ ] **Step 4: Provide tracking command**

Use `grep`, not `rg`, because the user's base shell lacks ripgrep:

```bash
tail -F "$RUN/logs/pipeline.log" | stdbuf -oL tr '\r' '\n' | \
  grep --line-buffered -E 'Training progress:|Evaluating best_eval|Saving Best|CANARY_|FULL_|PIPELINE_DONE|Traceback|RuntimeError'
```
