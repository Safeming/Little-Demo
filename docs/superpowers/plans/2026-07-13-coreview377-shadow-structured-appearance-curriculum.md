# CoreView377 Shadow Structured Appearance Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a from-zero, metric-gated shadow structured-appearance branch to the proven CoreView_377 80k/64k baseline and launch it with one command.

**Architecture:** `ColorMLP` computes base and structured candidate colors in one forward pass. During warmup the renderer uses base colors for the main image and separately rasterizes candidate colors; `train.py` obtains gradients only for structured parameters with `torch.autograd.grad`. At 12k, a base-versus-candidate validation gate either approves a persisted handoff flag and begins a bounded crossfade or writes a failure marker and stops.

**Tech Stack:** Python 3.9, PyTorch, CUDA Gaussian rasterizer, Hydra/OmegaConf, pytest, Bash.

---

### Task 1: Texture Shadow State and Composition

**Files:**
- Modify: `models/texture/texture.py`
- Test: `tests/test_shadow_structured_appearance.py`

- [ ] **Step 1: Write failing texture tests**

Add tests for the pure composition and state contract:

```python
def test_shadow_zero_scale_is_exact_base():
    base = torch.tensor([[0.2, 0.4, 0.6]])
    candidate = torch.tensor([[0.8, 0.1, 0.3]])
    assert torch.equal(compose_shadow_colors(base, candidate, 0.0), base)


def test_shadow_handoff_is_persistent_and_bounded():
    texture = _shadow_texture(active_scale=[0.0, 12, 0.0, 20, 0.35])
    assert texture.shadow_active_scale(20) == 0.0
    texture.approve_shadow_handoff()
    assert texture.shadow_active_scale(20) == pytest.approx(0.35)
    assert "shadow_handoff_approved" in texture.state_dict()
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_shadow_structured_appearance.py -q`

Expected: FAIL because shadow composition/state APIs do not exist.

- [ ] **Step 3: Implement minimal shadow APIs**

Add a pure helper and `ColorMLP` state:

```python
def compose_shadow_colors(base, candidate, scale):
    scale = max(0.0, min(1.0, float(scale)))
    return base + (candidate - base) * scale


self.shadow_appearance_cfg = cfg.get("shadow_appearance", None)
self.shadow_appearance_enable = bool(
    self.shadow_appearance_cfg.get("enable", False)
) if self.shadow_appearance_cfg is not None else False
self.shadow_active_scale_cfg = self.shadow_appearance_cfg.get("active_scale", 0.0)
self.shadow_train_until = int(self.shadow_appearance_cfg.get("train_until", 12000))
self.register_buffer("shadow_handoff_approved", torch.tensor(False), persistent=True)
self.last_shadow_base_color = None
self.last_shadow_candidate_color = None
self.shadow_render_mode = "active"
```

Expose `set_shadow_render_mode`, `approve_shadow_handoff`, `shadow_active_scale`, `shadow_parameters`, and `should_render_shadow`. In `forward`, preserve `base_logits`, compute the existing structured/detail candidate, activate both, store both tensors, and return base/candidate/active composition according to runtime mode.

- [ ] **Step 4: Run focused tests**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_shadow_structured_appearance.py -q`

Expected: texture tests PASS.

- [ ] **Step 5: Commit**

```bash
git add models/texture/texture.py tests/test_shadow_structured_appearance.py
git commit -m "feat: add shadow appearance texture state"
```

### Task 2: Candidate Rasterization and Isolated Shadow Gradients

**Files:**
- Modify: `gaussian_renderer/__init__.py`
- Modify: `train.py`
- Test: `tests/test_shadow_structured_appearance.py`

- [ ] **Step 1: Add failing renderer and gradient-isolation tests**

```python
def test_shadow_gradient_merge_only_updates_allowlisted_parameters():
    base = nn.Parameter(torch.tensor(1.0))
    shadow = nn.Parameter(torch.tensor(1.0))
    shadow_loss = (base + shadow) ** 2
    grads = isolated_parameter_grads(shadow_loss, [shadow])
    assert base.grad is None
    assert grads[0] is not None


def test_renderer_source_exposes_shadow_render():
    source = (ROOT / "gaussian_renderer/__init__.py").read_text()
    assert 'raster_pkg["shadow_render"]' in source
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_shadow_structured_appearance.py -q`

Expected: FAIL for missing isolated-gradient and renderer behavior.

- [ ] **Step 3: Add candidate rasterization**

After the normal rasterization in `render`, reuse the same deformed Gaussians and rasterize `texture.last_shadow_candidate_color` when `compute_loss` and `texture.should_render_shadow(iteration)` are true:

```python
shadow_pkg = rasterize_gaussians(
    data, pc, pipe, bg_color,
    colors_precomp=texture.last_shadow_candidate_color,
    scaling_modifier=scaling_modifier,
    return_opacity=False,
)
raster_pkg["shadow_render"] = shadow_pkg["render"]
```

- [ ] **Step 4: Add isolated gradient collection and merge**

Implement:

```python
def isolated_parameter_grads(loss, parameters, retain_graph=True):
    return torch.autograd.grad(
        loss, tuple(parameters), retain_graph=retain_graph, allow_unused=True
    )


def merge_parameter_grads(parameters, grads, scale=1.0):
    for parameter, grad in zip(parameters, grads):
        if grad is None:
            continue
        contribution = grad.detach() * float(scale)
        parameter.grad = contribution if parameter.grad is None else parameter.grad + contribution
```

Before the normal backward, compute a foreground-normalized L1 shadow loss from `shadow_render` and collect gradients only for `texture.shadow_parameters()`. After normal backward, merge those gradients and log `[ShadowAppearanceTrain]` with loss, candidate residual magnitude, and parameter count.

- [ ] **Step 5: Run focused and regression tests**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_shadow_structured_appearance.py \
  tests/test_gaussian_converter_lr_schedule.py \
  tests/test_local_anchor_tether.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add gaussian_renderer/__init__.py train.py tests/test_shadow_structured_appearance.py
git commit -m "feat: train shadow appearance with isolated gradients"
```

### Task 3: Dual Validation and Automatic Handoff Gate

**Files:**
- Modify: `train.py`
- Test: `tests/test_shadow_structured_appearance.py`

- [ ] **Step 1: Add failing gate tests**

```python
def test_handoff_requires_mean_gain_and_per_camera_guard():
    event = shadow_handoff_gate(
        iteration=12000,
        base_lpips=[0.14, 0.15, 0.16],
        candidate_lpips=[0.138, 0.148, 0.159],
        minimum_gain=0.001,
        max_camera_regression=0.002,
    )
    assert event["approved"] is True


def test_handoff_rejects_single_camera_regression():
    event = shadow_handoff_gate(
        iteration=12000,
        base_lpips=[0.14, 0.15, 0.16],
        candidate_lpips=[0.135, 0.153, 0.155],
        minimum_gain=0.001,
        max_camera_regression=0.002,
    )
    assert event["approved"] is False
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_shadow_structured_appearance.py -q`

Expected: FAIL for missing gate.

- [ ] **Step 3: Implement dual validation**

Add `shadow_validation` that temporarily selects `base` and `candidate` texture modes, renders the `best_eval` cameras, records aggregate and per-camera FG LPIPS/PSNR, restores `active` mode, and logs `[ShadowAppearanceEval]`.

- [ ] **Step 4: Implement gate and marker**

At configured evaluation iterations `[8000, 10000, 12000]`, write `shadow_appearance_gate.json`. At 12k, approve the persisted handoff flag only if mean gain is at least `0.001` and maximum per-camera regression is at most `0.002`. Otherwise write `SHADOW_APPEARANCE_GATE_FAILED` and raise `RuntimeError` after preserving the current best base checkpoint.

Add active absolute gates at 16k (`0.1430`) and 32k (`0.1362977028`), with tolerance `0.0015`.

- [ ] **Step 5: Run tests**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_shadow_structured_appearance.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add train.py tests/test_shadow_structured_appearance.py
git commit -m "feat: gate shadow appearance handoff"
```

### Task 4: Experiment Option and One-Command Launcher

**Files:**
- Create: `configs/option/stageA_377_multiview_explicit_hq_fromzero_shadow_structured_appearance_v1.yaml`
- Create: `exp/zero_train_to_v395/coreview377_shadow_structured_appearance_20260713_bjt/launch_shadow_structured_appearance.sh`
- Create: `tests/test_coreview377_shadow_structured_appearance_pipeline.py`

- [ ] **Step 1: Write failing pipeline contract tests**

Assert that the option enables structured trunk/output head/shadow mode, uses the approved schedules and gates, and that the launcher is 80k, 64k, seed `20260710`, no densification, no checkpoint loading, and includes smoke overrides that force a shadow render and gate.

- [ ] **Step 2: Run pipeline test and confirm RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_coreview377_shadow_structured_appearance_pipeline.py -q`

Expected: FAIL because option and launcher are absent.

- [ ] **Step 3: Add the option**

Compose the existing structured capability options in the launcher:

```text
stageA_377_multiview_explicit_hq_v1160_v115b_trunkmain_base_v1
stageA_377_multiview_explicit_hq_v1170_v116b_trunkmlp_base_v1
stageA_377_multiview_explicit_hq_v1220_v121a_trunkcolor_full_tinyrepair_v1
stageA_377_multiview_explicit_hq_fromzero_shadow_structured_appearance_v1
```

The new option sets `model.texture.shadow_appearance.enable=true`, active scale `[0.0,12000,0.0,20000,0.20,40000,0.28,64000,0.32]`, shadow training through 12k, high-frequency contribution zero through 40k, and all metric gates from the design.

- [ ] **Step 4: Add launcher**

Wrap the verified neutral-longhorizon launcher, override only the option stack, keep `dataset.init_point_count=80000`, and set `TESTS`/`SAVES` at all shadow and absolute gates. `SMOKE=1` uses 20 iterations, shadow render from iteration 1, and a permissive handoff gate at iteration 10.

- [ ] **Step 5: Run pipeline and shell tests**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_coreview377_shadow_structured_appearance_pipeline.py \
  tests/test_shadow_structured_appearance.py -q
bash -n exp/zero_train_to_v395/coreview377_shadow_structured_appearance_20260713_bjt/launch_shadow_structured_appearance.sh
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/option/stageA_377_multiview_explicit_hq_fromzero_shadow_structured_appearance_v1.yaml \
  exp/zero_train_to_v395/coreview377_shadow_structured_appearance_20260713_bjt/launch_shadow_structured_appearance.sh \
  tests/test_coreview377_shadow_structured_appearance_pipeline.py
git commit -m "feat: add shadow appearance training pipeline"
```

### Task 5: Verification, GPU Smoke, and Formal Launch

**Files:**
- Runtime output: `exp/zero_train_to_v395/coreview377_shadow_structured_appearance_20260713_bjt/`

- [ ] **Step 1: Run focused regression suite and compilation**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_shadow_structured_appearance.py \
  tests/test_coreview377_shadow_structured_appearance_pipeline.py \
  tests/test_gaussian_converter_lr_schedule.py \
  tests/test_gaussian_model_densification.py -q
/opt/miniconda3/envs/ictrl/bin/python -m py_compile \
  models/texture/texture.py gaussian_renderer/__init__.py train.py
```

Expected: all tests PASS and compilation exits zero.

- [ ] **Step 2: Run 20-iteration GPU smoke**

Run the launcher with `SMOKE=1`. Verify logs contain `Number of points at initialisation : 80000`, `[ShadowAppearanceTrain]`, `[ShadowAppearanceEval]`, an approved smoke handoff, checkpoint creation, and `PIPELINE_DONE`.

- [ ] **Step 3: Launch formal 64k training**

Create a BJT-tagged run directory, start with `setsid -f`, redirect to `logs/pipeline.log`, and verify the PID, GPU allocation, 80k initialization, and live progress.

- [ ] **Step 4: Report tracking and ETA**

Give grep-based commands that do not require `rg`. Estimate completion from the measured smoke throughput and the previous 80k/64k baseline, explicitly noting that the 12k gate may stop the run early.
