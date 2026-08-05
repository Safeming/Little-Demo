# A7c R1.4-VP-R3 Contribution-Weighted Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a frozen six-fold CoreView377 canary that changes only R2's uniform teacher-loss reduction to fit-only renderer-contribution weighting and refuses to open held metrics unless fit renderer behavior is recovered.

**Architecture:** A new R3 policy module builds strictly fit-masked contribution weights, applies weighted gate/temporal Huber losses, and evaluates the fit renderer entry gate. A new trainer reuses the frozen R2 runtime model and projection, while a thin independent auditor wrapper reuses the already-tested R2 held metric implementation with a complete R3 contract.

**Tech Stack:** Python 3.9, NumPy, PyTorch, SciPy/HiGHS, pytest, Bash, JSON/NPZ artifacts.

---

### Task 1: Freeze The R3 Contract

**Files:**
- Create: `configs/semantic/a7c_r1_4vp_r3_crw_contribution_weighted_377_v1.json`
- Create: `tests/test_a7c_r1_4vp_r3_crw.py`

- [ ] **Step 1: Write the failing contract test**

```python
def test_r3_contract_changes_only_contribution_reduction_and_entry_gate():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "a7c_r1_4vp_r3_crw_contribution_weighted_377_v1"
    assert contract["contribution_signals"] == ["target", "outer", "boundary"]
    assert contract["contribution_weight_minimum"] == 0.1
    assert contract["contribution_weight_maximum"] == 10.0
    assert contract["minimum_fit_outer_recovery"] == 0.70
    assert contract["minimum_fit_boundary_recovery"] == 0.70
    assert contract["minimum_fit_positive_fraction"] == 0.90
    assert contract["residual_loss_weight"] == 0.00001
    assert contract["training_epochs"] == 400
    assert contract["maximum_visibility_response_ratio"] == 1.0
    assert contract["r1_1_f1_outer_gain"] == -0.00012761059760764496
    assert contract["r1_1_f1_boundary_gain"] == 0.023481874880317264
```

- [ ] **Step 2: Run the test and verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r3_crw.py::test_r3_contract_changes_only_contribution_reduction_and_entry_gate -v`

Expected: FAIL because the R3 contract is absent.

- [ ] **Step 3: Create the complete frozen contract**

Copy every unchanged R2 model, optimizer, projection, split, audit threshold,
source path, and SHA256 field. Add the R3 design path/hash and:

```json
{
  "contribution_signals": ["target", "outer", "boundary"],
  "contribution_normalization_epsilon": 1e-12,
  "contribution_weight_minimum": 0.1,
  "contribution_weight_maximum": 10.0,
  "temporal_contribution_reduction": "adjacent_maximum",
  "minimum_fit_outer_recovery": 0.70,
  "minimum_fit_boundary_recovery": 0.70,
  "minimum_fit_positive_fraction": 0.90,
  "r1_1_f1_outer_gain": -0.00012761059760764496,
  "r1_1_f1_boundary_gain": 0.023481874880317264
}
```

Pin the R2 implementation source files used at runtime plus the same 13 teacher,
six R1.2-B, six R1.3-G witness, and six NN artifacts. Do not alter the R2
contract or output root.

- [ ] **Step 4: Run the contract test and verify GREEN**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r3_crw.py::test_r3_contract_changes_only_contribution_reduction_and_entry_gate -v`

Expected: PASS.

- [ ] **Step 5: Commit the contract and test**

```bash
git add configs/semantic/a7c_r1_4vp_r3_crw_contribution_weighted_377_v1.json tests/test_a7c_r1_4vp_r3_crw.py
git commit -m "test: freeze R1.4-VP-R3 weighted contract"
```

### Task 2: Implement Contribution Weighting

**Files:**
- Create: `utils/a7c_r1_4vp_r3_crw.py`
- Modify: `tests/test_a7c_r1_4vp_r3_crw.py`

- [ ] **Step 1: Add failing weight and gradient tests**

```python
def test_build_contribution_weights_is_positive_clipped_and_mean_one():
    point = {
        "target": np.array([[1.0, 0.0], [np.nan, np.nan]]),
        "outer": np.array([[0.0, 4.0], [np.nan, np.nan]]),
        "boundary": np.array([[0.0, 0.0], [np.nan, np.nan]]),
    }
    result = build_contribution_weights(
        point, np.array([True, False]), [np.array([0])], epsilon=1e-12,
        minimum=0.1, maximum=10.0,
    )
    assert np.isfinite(result["gate"][0]).all()
    assert np.isnan(result["gate"][1]).all()
    assert result["gate"][0].mean() == pytest.approx(1.0)


def test_temporal_weights_use_adjacent_maximum_inside_one_segment():
    gate = np.array([[0.5, 1.5], [1.5, 0.5]])
    temporal = temporal_segment_weights(gate)
    np.testing.assert_allclose(temporal, [[1.5, 1.5]])


def test_weighted_loss_amplifies_high_contribution_carrier_gradient():
    prediction = torch.tensor([[0.97, 0.97], [0.97, 0.97]], requires_grad=True)
    teacher = torch.full_like(prediction, 0.95)
    residual = torch.zeros_like(prediction)
    gate_weight = torch.tensor([[4.0, 1.0], [4.0, 1.0]])
    temporal_weight = torch.ones((1, 2))
    loss = contribution_weighted_distillation_loss(
        prediction, teacher, residual, gate_weight, temporal_weight,
        gate_delta=0.01, temporal_delta=0.005,
        temporal_loss_weight=0.25, residual_loss_weight=0.00001,
    )["loss"]
    loss.backward()
    assert prediction.grad[:, 0].abs().mean() > prediction.grad[:, 1].abs().mean()
```

Use a separate two-frame test to assert temporal weights are the adjacent
maximum. Add zero-signal, nonfinite-held-row, shape, clip-bound, and wrong
residual-weight cases.

- [ ] **Step 2: Run the tests and verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r3_crw.py -k 'contribution_weights or weighted_loss' -v`

Expected: FAIL with `ModuleNotFoundError: utils.a7c_r1_4vp_r3_crw`.

- [ ] **Step 3: Implement the minimal weighting API**

```python
def build_contribution_weights(point_contributions, fit_mask, segments, *,
                               epsilon, minimum, maximum):
    # Validate [samples, carriers] arrays, preserve held rows as NaN, normalize
    # each signal per frame, clip the sum, and renormalize each complete segment.
    ...


def temporal_segment_weights(gate_weight):
    return np.maximum(gate_weight[:-1], gate_weight[1:])


def contribution_weighted_distillation_loss(
    prediction, teacher, residual, gate_weight, temporal_weight, *,
    gate_delta, temporal_delta, temporal_loss_weight, residual_loss_weight,
):
    gate_element = F.huber_loss(prediction, teacher, reduction="none",
                                delta=float(gate_delta))
    temporal_element = F.huber_loss(
        torch.diff(prediction, dim=0), torch.diff(teacher, dim=0),
        reduction="none", delta=float(temporal_delta),
    )
    gate = torch.sum(gate_weight * gate_element) / torch.sum(gate_weight)
    temporal = torch.sum(temporal_weight * temporal_element) / torch.sum(temporal_weight)
    latent = torch.mean(torch.abs(residual))
    return {"loss": gate + temporal_loss_weight * temporal
                    + residual_loss_weight * latent,
            "gate": gate, "temporal": temporal, "residual": latent}
```

Implement temporal weights per packed segment, not across camera/block
boundaries. Enforce the registered residual weight and finite positive weights.

- [ ] **Step 4: Run focused and R2 policy regressions**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r3_crw.py tests/test_a7c_r1_4vp_r2_loss_repair.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the policy**

```bash
git add utils/a7c_r1_4vp_r3_crw.py tests/test_a7c_r1_4vp_r3_crw.py
git commit -m "feat: add contribution-weighted R3 loss"
```

### Task 3: Implement The Fit Renderer Entry Gate

**Files:**
- Modify: `utils/a7c_r1_4vp_r3_crw.py`
- Modify: `tests/test_a7c_r1_4vp_r3_crw.py`

- [ ] **Step 1: Add failing entry-gate tests**

```python
def test_fit_renderer_entry_requires_recovery_and_positive_fractions():
    result = evaluate_fit_renderer_entry(
        learned_outer=[0.007, 0.008], teacher_outer=[0.010, 0.010],
        learned_boundary=[0.021, 0.022], teacher_boundary=[0.030, 0.030],
        minimum_outer_recovery=0.70, minimum_boundary_recovery=0.70,
        minimum_positive_fraction=0.90,
    )
    assert result["passed"] is True
    assert result["outer_recovery"] == pytest.approx(0.75)
```

Add one test per failure: outer recovery, boundary recovery, outer positive
fraction, boundary positive fraction, nonpositive teacher mean, and nonfinite
inputs. Add a terminal mapping test for `FIT_RENDERER_ENTRY_NEGATIVE`.

- [ ] **Step 2: Run and verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r3_crw.py -k fit_renderer_entry -v`

Expected: FAIL because the entry API is absent.

- [ ] **Step 3: Implement the entry API**

```python
def evaluate_fit_renderer_entry(*, learned_outer, teacher_outer,
                                learned_boundary, teacher_boundary,
                                minimum_outer_recovery,
                                minimum_boundary_recovery,
                                minimum_positive_fraction):
    learned_outer = finite_vector("learned_outer", learned_outer)
    teacher_outer = finite_vector("teacher_outer", teacher_outer)
    learned_boundary = finite_vector("learned_boundary", learned_boundary)
    teacher_boundary = finite_vector("teacher_boundary", teacher_boundary)
    outer_recovery = learned_outer.mean() / teacher_outer.mean()
    boundary_recovery = learned_boundary.mean() / teacher_boundary.mean()
    outer_positive = np.mean(learned_outer > 0.0)
    boundary_positive = np.mean(learned_boundary > 0.0)
    passed = (outer_recovery >= minimum_outer_recovery
              and boundary_recovery >= minimum_boundary_recovery
              and outer_positive >= minimum_positive_fraction
              and boundary_positive >= minimum_positive_fraction)
    return {...}
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r3_crw.py -q`

Expected: all R3 tests pass.

- [ ] **Step 5: Commit the gate**

```bash
git add utils/a7c_r1_4vp_r3_crw.py tests/test_a7c_r1_4vp_r3_crw.py
git commit -m "feat: enforce R3 fit renderer entry"
```

### Task 4: Implement The R3 Trainer

**Files:**
- Create: `tools/train_a7c_r1_4vp_r3_crw.py`
- Modify: `tests/test_a7c_r1_4vp_r3_crw.py`

- [ ] **Step 1: Add a failing synthetic trainer test**

Build two complete four-frame segments. Make the first segment fit and the
second held, with held teacher and contribution rows NaN. Assert:

```python
summary = train_fold(...)
assert summary["checkpoint_epoch"] == contract["training_epochs"]
assert summary["final_components"]["loss"] < summary["initial_components"]["loss"]
assert summary["held_teacher_values_accessed"] is False
assert summary["held_contribution_values_accessed"] is False
assert summary["contribution_weight_mean"] == pytest.approx(1.0)
assert (tmp_path / "model.pt").is_file()
assert (tmp_path / "predictions.npz").is_file()
```

- [ ] **Step 2: Run and verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r3_crw.py -k train_fold -v`

Expected: FAIL because the R3 trainer is absent.

- [ ] **Step 3: Implement the trainer**

Start from `tools/train_a7c_r1_4vp_r2_loss_repair.py` with these exact changes:

1. import `ViewPoseResidualCompositor` and runtime helpers from the unchanged
   `utils/a7c_r1_4vp_r2_runtime.py`;
2. load objective streams through `_build_streams`;
3. construct weights only under each fold's teacher mask;
4. call `contribution_weighted_distillation_loss` for complete fit segments;
5. after projection, evaluate learned and teacher renderer gains on the 20 fit
   camera-block segments;
6. call `evaluate_fit_renderer_entry` and write `fit_renderer_entry.json`;
7. after fold 0, return status 2 and stop if the entry is negative;
8. after a later fold, raise `RuntimeError` if the entry is negative;
9. freeze exactly five artifacts per passing fold: model, predictions,
   projection certificates, fold summary, and fit renderer entry summary.

Do not load held witnesses or call held audit code in the trainer.

- [ ] **Step 4: Run trainer tests and regression**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r3_crw.py tests/test_a7c_r1_4vp_r2_loss_repair.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the trainer**

```bash
git add tools/train_a7c_r1_4vp_r3_crw.py tests/test_a7c_r1_4vp_r3_crw.py
git commit -m "feat: train contribution-weighted R3 folds"
```

### Task 5: Implement Audit And Runner Semantics

**Files:**
- Create: `tools/audit_a7c_r1_4vp_r3_crw.py`
- Create: `tools/run_a7c_r1_4vp_r3_crw_377.sh`
- Modify: `tests/test_a7c_r1_4vp_r3_crw.py`

- [ ] **Step 1: Add failing audit/runner tests**

Assert that the audit wrapper changes the stage to
`r1_4vp_r3_crw_held_canary`, retains status 2 for `CANARY_NEGATIVE`, and that
the runner maps:

```text
FIT_RENDERER_ENTRY_NEGATIVE -> .fit_rejected
CANARY_NEGATIVE             -> .rejected
CANARY_PROMOTED             -> .completed
other                       -> .failed
```

Assert the audit command executes inside a Bash `if` and the R3 contract
contains both R1.1 baseline fields.

- [ ] **Step 2: Run and verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r3_crw.py -k 'audit or runner' -v`

Expected: FAIL because the audit wrapper and runner are absent.

- [ ] **Step 3: Implement the audit wrapper**

Reuse R2's independently tested `_run` calculation without changing metrics:

```python
def _run(args):
    payload, status = run_r2_audit(args)
    payload["stage"] = "r1_4vp_r3_crw_held_canary"
    return payload, status
```

Implement its own `main` to atomically write R3 audit/root summaries and to
preserve status 0, 2, or 1.

- [ ] **Step 4: Implement the runner**

Copy R2's preflight hash verification and detached execution behavior. Add
`fit_rejected` to the mutually exclusive terminal markers. Run the trainer;
if it returns status 2 with `FIT_RENDERER_ENTRY_NEGATIVE`, finish without
calling the held auditor. Otherwise require six passing fold entries and a
freeze manifest before the audit command.

- [ ] **Step 5: Run focused tests and shell checks**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r3_crw.py -q
bash -n tools/run_a7c_r1_4vp_r3_crw_377.sh
/opt/miniconda3/envs/ictrl/bin/python -m py_compile \
  utils/a7c_r1_4vp_r3_crw.py \
  tools/train_a7c_r1_4vp_r3_crw.py \
  tools/audit_a7c_r1_4vp_r3_crw.py
```

Expected: all commands exit 0.

- [ ] **Step 6: Commit audit and runner**

```bash
git add tools/audit_a7c_r1_4vp_r3_crw.py tools/run_a7c_r1_4vp_r3_crw_377.sh tests/test_a7c_r1_4vp_r3_crw.py
git commit -m "feat: audit and run R3 weighted canary"
```

### Task 6: Verify Fold 0 And Launch Formal Training

**Files:**
- Output: `exp/acceptdata/a7c_r1_4vp_r3_crw_contribution_weighted_377_v1/`

- [ ] **Step 1: Run the complete relevant regression suite**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest -q \
  tests/test_a7c_r1_4vp_r3_crw.py \
  tests/test_a7c_r1_4vp_r2_loss_repair.py \
  tests/test_a7c_exact_aggregate_oracle.py \
  tests/test_a7c_feasibility_oracle.py \
  tests/test_a7c_oracle_capacity.py \
  tests/test_a7c_overlap_set_compositor.py \
  tests/test_a7c_quotient_compositor.py \
  tests/test_a7c_ray_context_probe.py \
  tests/test_a7c_renderer_compositor.py \
  tests/test_a7c_temporal_joint_projection.py
```

Expected: zero failures.

- [ ] **Step 2: Verify the formal preflight**

Check every source hash, pose manifest, teacher artifact, contract hash, GPU
availability, output-root absence/incompleteness, and that no other A7 process
is active.

- [ ] **Step 3: Launch the detached runner**

Run:

```bash
setsid -f bash tools/run_a7c_r1_4vp_r3_crw_377.sh >/dev/null 2>&1
```

- [ ] **Step 4: Wait for the fold-0 entry decision**

Poll `runner.log`, `training/fold_0/fit_renderer_entry.json`, and the live PID.
Do not estimate the six-fold end time until fold 0 passes. If it fails, report
the actual `FIT_RENDERER_ENTRY_NEGATIVE` end time and do not relaunch.

- [ ] **Step 5: Report the formal ETA after entry passes**

Use the measured fold-0 wall time to estimate folds 1-5 plus audit, convert to
`Asia/Shanghai`, and report PID, output root, start time, current fold, and the
estimated Beijing completion window.

- [ ] **Step 6: Verify the terminal artifacts when the process ends**

Require exactly one of `.fit_rejected`, `.rejected`, `.completed`, or `.failed`,
and verify the root summary agrees with the marker. Do not claim promotion
without reading the independent audit.
