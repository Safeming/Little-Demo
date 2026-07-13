# CoreView377 Perceptual-Band Shadow Canary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a randomly initialized structured shadow branch from the verified CoreView_377 8k baseline to schedule iteration 12k using perceptual and frequency-selective losses, with the base image function frozen.

**Architecture:** Preserve the post-base-MLP RNG stream while constructing optional shadow modules. Extend the existing isolated shadow-gradient path with L1, LPIPS, gradient, high-pass, and low-frequency-drift components. Launch a 4k local resume whose optimizer allowlist contains only structured/high-frequency texture parameters and whose final dual-render gate writes an explicit pass/fail marker.

**Tech Stack:** Python 3.9, PyTorch, LPIPS, Hydra/OmegaConf, CUDA Gaussian rasterizer, pytest, Bash.

---

### Task 1: RNG-Isolated Shadow Initialization

**Files:**
- Modify: `models/texture/texture.py`
- Modify: `tests/test_shadow_structured_appearance.py`

- [ ] **Step 1: Write a failing RNG restoration test**

```python
def test_restore_torch_rng_state_replays_the_same_draw():
    torch.manual_seed(77)
    state = capture_torch_rng_state()
    expected = torch.rand(4)
    torch.rand(100)
    restore_torch_rng_state(state)
    assert torch.equal(torch.rand(4), expected)
```

- [ ] **Step 2: Run RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_shadow_structured_appearance.py::test_restore_torch_rng_state_replays_the_same_draw -q`

Expected: import failure because the helpers do not exist.

- [ ] **Step 3: Implement capture/restore and use it around shadow construction**

```python
def capture_torch_rng_state():
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() and torch.cuda.is_initialized() else None,
    }


def restore_torch_rng_state(state):
    torch.set_rng_state(state["cpu"])
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
```

Capture immediately after `self.mlp` construction when `shadow_appearance.preserve_rng_state=true`; restore at the end of `ColorMLP.__init__` after structured/detail modules have retained their initialized weights.

- [ ] **Step 4: Run GREEN and regressions**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_shadow_structured_appearance.py -q`

Expected: PASS.

### Task 2: Perceptual-Band Shadow Loss

**Files:**
- Modify: `train.py`
- Modify: `tests/test_shadow_structured_appearance.py`

- [ ] **Step 1: Write failing pure loss-schedule tests**

```python
def test_shadow_lpips_interval_scales_sampled_steps():
    assert shadow_interval_weight(8, 4, 0.25) == 1.0
    assert shadow_interval_weight(9, 4, 0.25) == 0.0


def test_low_frequency_drift_is_zero_for_identical_images():
    image = torch.rand(3, 16, 16)
    mask = torch.ones(1, 16, 16)
    assert shadow_low_frequency_drift(image, image, mask, kernel=5).item() == 0.0
```

- [ ] **Step 2: Run RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_shadow_structured_appearance.py -q`

Expected: FAIL for missing helpers.

- [ ] **Step 3: Implement helpers and composed loss**

Use:

```python
shadow_loss = (
    lambda_l1 * shadow_l1
    + lambda_gradient * _masked_gradient_l1_loss(shadow_render, gt_image, fg_mask)
    + lambda_highpass * _masked_multiscale_highpass_loss(
        shadow_render, gt_image, fg_mask, scales=[1, 2, 4],
        blur_kernel=5, scale_decay=0.6, gradient_mix=0.25,
    )
    + lambda_low_frequency * shadow_low_frequency_drift(
        shadow_render, image.detach(), fg_mask, kernel=9,
    )
    + interval_weight * shadow_lpips
)
```

Compute LPIPS on masked foreground images every configured interval, multiply sampled LPIPS weight by the interval, retain isolated `autograd.grad` collection, and log every component in `[PerceptualBandShadowTrain]`.

- [ ] **Step 4: Extend final gate with PSNR guard and markers**

At the handoff iteration, require candidate LPIPS gain, per-camera LPIPS guard, and `candidate_psnr >= base_psnr - 0.1`. Write `PERCEPTUAL_BAND_SHADOW_GATE_PASSED` or `PERCEPTUAL_BAND_SHADOW_GATE_FAILED`, then end the diagnostic pipeline intentionally after checkpointing.

- [ ] **Step 5: Run GREEN and compilation**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_shadow_structured_appearance.py -q
/opt/miniconda3/envs/ictrl/bin/python -m py_compile train.py models/texture/texture.py
```

Expected: PASS.

### Task 3: Frozen 8k→12k Canary Pipeline

**Files:**
- Create: `configs/option/stageA_377_multiview_explicit_hq_fromzero_perceptual_band_shadow_canary_v1.yaml`
- Create: `exp/zero_train_to_v395/coreview377_perceptual_band_shadow_canary_20260713_bjt/launch_perceptual_band_shadow_canary.sh`
- Create: `tests/test_coreview377_perceptual_band_shadow_canary_pipeline.py`

- [ ] **Step 1: Write failing pipeline contract tests**

Assert:

```python
assert opt["iterations"] == 4000
assert opt["position_lr_init"] == 0.0
assert opt["feature_lr"] == 0.0
assert opt["texture_trainable_name_patterns"] == [
    "structured_trunk_*", "detail_high_freq_*"
]
assert "ckpt8000.pth" in launcher
assert "use_checkpoint_iteration_as_offset=true" in launcher
assert "densify_until_iter=0" in launcher
assert "v395" not in checkpoint_start_line
```

- [ ] **Step 2: Run RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_coreview377_perceptual_band_shadow_canary_pipeline.py -q`

Expected: missing option and launcher failures.

- [ ] **Step 3: Add option and launcher**

The option freezes all non-shadow learning rates, uses the structured option stack from the first shadow experiment, sets local `iterations=4000`, evaluation at schedule iterations 9000/10000/11000/12000, handoff at 12000, and configures the perceptual-band weights. The launcher supplies the verified baseline `ckpt8000.pth`, enables checkpoint-iteration offset, disables all topology changes, and writes a final result record.

- [ ] **Step 4: Run pipeline tests and shell syntax**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_coreview377_perceptual_band_shadow_canary_pipeline.py \
  tests/test_shadow_structured_appearance.py -q
bash -n exp/zero_train_to_v395/coreview377_perceptual_band_shadow_canary_20260713_bjt/launch_perceptual_band_shadow_canary.sh
```

Expected: PASS.

### Task 4: Verification, Smoke, and Formal Launch

**Files:**
- Runtime output: `exp/zero_train_to_v395/coreview377_perceptual_band_shadow_canary_20260713_bjt/`

- [ ] **Step 1: Run focused regression and compile suite**

Run the new tests plus existing converter, densification, and local-anchor regressions. Expected: zero failures.

- [ ] **Step 2: Run GPU smoke**

Use a short local horizon with the real 8k checkpoint. Verify partial load, shadow-only optimizer allowlist, nonzero L1/high-pass/gradient/LPIPS logs, dual evaluation, checkpoint creation, and no CUDA error.

- [ ] **Step 3: Launch formal canary with `setsid -f`**

Verify PID, GPU allocation, `Loaded checkpoint ... ckpt8000.pth (iteration 8000)`, live schedule progress, and output log path.

- [ ] **Step 4: Report grep-based tracking and BJT ETA**

Estimate completion using measured smoke throughput. Mention that the canary ends at schedule iteration 12k regardless of pass/fail.
