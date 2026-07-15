# Subject-Agnostic Residual-Balanced Refine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a subject-agnostic residual-balanced four-view refinement canary that reaches the CoreView_377 v395 dual-metric gate without validation leakage.

**Architecture:** Put the adaptive sampler in a focused utility module and connect it to the existing sampler interface in `train.py`. Add generic metric constraints and optional per-sample validation export, then use a standalone selector to apply CoreView_377 benchmark gates and baseline fallback. Keep all CoreView_377 paths and absolute targets in the experiment launcher, outside reusable training logic.

**Tech Stack:** Python, PyTorch, NumPy, Hydra YAML, Bash, pytest.

---

### Task 1: Residual-Balanced Multi-View Sampler

**Files:**
- Create: `utils/residual_balanced_sampler.py`
- Create: `tests/test_residual_balanced_sampler.py`

- [ ] **Step 1: Write failing sampler tests**

Create a fake dataset whose `data` contains `frame_idx`, `cam_name`, and four observations per frame. Test uniform warmup, EMA updates, bounded probabilities, four distinct cameras, accumulation scaling, and deterministic draws.

```python
sampler = ResidualBalancedMultiViewSampler(dataset, opt)
assert np.allclose(sampler.frame_probabilities, np.full(3, 1 / 3))
sampler.update_residual(index_for_frame_2, 0.4)
sampler.sample_count = 200
sampler.recompute_probabilities()
assert sampler.probability_for_frame(2) > sampler.probability_for_frame(0)
assert sampler.frame_probabilities.max() <= pytest.approx(2.5 / 3)
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest -q tests/test_residual_balanced_sampler.py`

Expected: import failure because `utils.residual_balanced_sampler` does not exist.

- [ ] **Step 3: Implement the sampler**

Implement `ResidualBalancedMultiViewSampler(dataset, opt)` with the existing `next_index`, `gradient_accumulation_scale`, `should_optimizer_step`, and `maybe_log` interface plus:

```python
def update_residual(self, data_index: int, residual: float) -> None: ...
def recompute_probabilities(self) -> None: ...
def probability_for_frame(self, frame_id: int) -> float: ...
```

Use `residual_balanced_warmup_samples=200`, `residual_balanced_ema=0.9`, `residual_balanced_uniform_mix=0.7`, `residual_balanced_relative_min=0.5`, `residual_balanced_relative_max=2.5`, `residual_balanced_frame_max_multiplier=2.5`, `train_sample_accumulation_steps=4`, and uniform camera selection without replacement.

- [ ] **Step 4: Run tests and confirm GREEN**

Run the focused sampler test and confirm all cases pass.

### Task 2: Training Loop Integration And Metric Constraints

**Files:**
- Modify: `train.py`
- Create: `tests/test_train_residual_balanced_sampling.py`
- Create: `tests/test_train_best_metric_constraints.py`

- [ ] **Step 1: Write failing integration tests**

Test `_build_train_sampler` returns the new sampler for `train_sample_mode=residual_balanced_multiview`. Test `_metrics_satisfy_constraints` accepts and rejects min/max metric bounds.

```python
assert _metrics_satisfy_constraints(
    {"psnr_fg": 22.2, "lpips_fg": 0.127},
    {"lpips_fg": {"max": 0.1273224292}},
)
```

- [ ] **Step 2: Run tests and confirm RED**

Expected: the new mode is unsupported and `_metrics_satisfy_constraints` is missing.

- [ ] **Step 3: Connect the sampler and residual feedback**

Import the sampler, add the new `_build_train_sampler` branch, and after raw `gt_image` and `fg_mask` are available report:

```python
update_residual = getattr(train_sampler, "update_residual", None)
if callable(update_residual):
    raw_fg_l1 = _masked_l1_loss(image.detach(), gt_image.detach(), fg_mask)
    update_residual(data_idx, float(raw_fg_l1.item()))
```

Add `_metrics_satisfy_constraints(metrics, constraints)` and require it alongside `is_better` before saving a best checkpoint. Existing behavior remains unchanged when constraints are empty.

- [ ] **Step 4: Run focused and existing sampler-related tests**

Run the two new tests plus existing training helper tests and confirm they pass.

### Task 3: Per-Sample Validation Export And Gate Selector

**Files:**
- Modify: `train.py`
- Create: `tools/select_residual_balanced_refine.py`
- Create: `tests/test_residual_balanced_refine_selector.py`

- [ ] **Step 1: Write failing selector tests**

Build baseline and candidate JSON fixtures with matching camera/frame identities. Verify the selector ranks baseline samples by foreground L1, evaluates the lowest-error 70%, accepts a passing candidate, and falls back on PSNR or easy-subset regression.

- [ ] **Step 2: Run tests and confirm RED**

Expected: selector module does not exist.

- [ ] **Step 3: Export validation sample rows**

When `opt.validation_per_sample_metrics_path` is set, collect for each validation observation:

```python
{
    "index": data_idx,
    "frame_id": int(data.frame_id),
    "camera_id": int(data.cam_id),
    "l1_fg": float(metrics_fg["l1_fg"]),
    "psnr_fg": float(metrics_fg["psnr_fg"]),
    "lpips_fg": float(metrics_fg["lpips_fg"]),
}
```

Write one JSON payload containing `iteration` and rows grouped by validation split. Do not add row lists to `best_test_metrics.json`.

- [ ] **Step 4: Implement the selector**

The selector accepts baseline/candidate aggregate and per-sample paths, candidate and baseline checkpoints, output directory, strict 377 thresholds, and easy-subset tolerances. It writes `final_selection.json`, `FINAL_BEST_CKPT.txt`, and `final_best_ckpt.pth`.

- [ ] **Step 5: Run selector tests and confirm GREEN**

Run the focused selector tests and verify both acceptance and fallback cases.

### Task 4: Canary Option And Launcher

**Files:**
- Create: `configs/option/stageA_fromzero_residual_balanced_late_refine_v1.yaml`
- Create: `exp/zero_train_to_v395/coreview377_residual_balanced_refine_20260715_bjt/launch_residual_balanced_refine.sh`
- Create: `tests/test_coreview377_residual_balanced_refine_pipeline.py`

- [ ] **Step 1: Write the failing pipeline test**

Assert the generic option contains no subject, frame, camera, or joint list; uses the residual-balanced sampler and four-view accumulation; preserves the late-clean frozen learning rates; disables photometric correction and topology changes. Assert the launcher starts only from the from-zero 77500 checkpoint, runs baseline and candidate per-sample diagnostics, applies both split gates, and contains no explicit difficult-frame training list.

- [ ] **Step 2: Run the pipeline test and confirm RED**

Expected: option and launcher files are missing.

- [ ] **Step 3: Implement option and launcher**

Set 3000 sample iterations with evaluations and saves every 500. Configure `best_metric=psnr_fg`, `best_metric_mode=max`, and `best_metric_constraints.lpips_fg.max=0.1273224292`. Run diagnostic-only baseline same-30 export, candidate training, candidate same-30 export, candidate original-57 export, then invoke the selector.

- [ ] **Step 4: Run pipeline test and Bash syntax check**

Run pytest and `bash -n` and confirm both pass.

### Task 5: Verification And Launch

**Files:**
- Update: `docs/当前会话交接_CoreView377从零训练与v395复现_20260617.md`

- [ ] **Step 1: Run the focused test suite**

Run all new tests plus the existing late-clean pipeline test. Confirm zero failures.

- [ ] **Step 2: Run a complete 12-sample smoke**

Use `SMOKE=1`. Confirm the log contains residual sampler initialization, residual updates/probability logs, grouped optimizer behavior, diagnostic exports, selector output, and a pipeline completion marker.

- [ ] **Step 3: Launch the 3000-sample canary detached**

Use `setsid`, record the run directory and PID, and verify the process survives the launching shell and advances beyond initial validation.

- [ ] **Step 4: Record handoff details**

Append the design, run directory, tracking command, gates, and expected completion time to the main handoff document without altering earlier conclusions.

