# A7c R1.4-VP-R4-A Signed Renderer-Trajectory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a frozen CoreView377 canary that replaces R3's absolute-contribution-weighted gate loss with fit-only signed renderer-trajectory distillation and stops before held audit unless fold 0 recovers at least 70% of teacher outer and boundary behavior.

**Architecture:** A pure R4-A policy module reconstructs differentiable target/outer/boundary renderer sequences, freezes initial component scales, computes the registered loss, and reports action diagnostics. A new trainer reuses the byte-frozen R2 runtime model, R3 fit entry, and R1.3-P projection; a thin auditor wrapper opens unchanged held metrics only after all six fit entries pass.

**Tech Stack:** Python 3.9, NumPy, PyTorch, SciPy/HiGHS, pytest, Bash, JSON/NPZ artifacts.

---

### Task 1: Freeze The R4-A Contract

**Files:**
- Create: `configs/semantic/a7c_r1_4vp_r4a_signed_renderer_trajectory_377_v1.json`
- Create: `tests/test_a7c_r1_4vp_r4a_signed_renderer.py`

- [ ] **Step 1: Write the failing contract test**

```python
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/semantic/a7c_r1_4vp_r4a_signed_renderer_trajectory_377_v1.json"


def test_r4a_contract_changes_only_the_registered_training_objective():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "a7c_r1_4vp_r4a_signed_renderer_trajectory_377_v1"
    assert contract["renderer_trajectory_signals"] == ["outer", "boundary"]
    assert contract["renderer_trajectory_huber_delta"] == 0.005
    assert contract["target_response_huber_delta"] == 0.005
    assert contract["renderer_outer_loss_weight"] == 1.0
    assert contract["renderer_boundary_loss_weight"] == 1.0
    assert contract["target_auxiliary_loss_weight"] == 0.1
    assert contract["gate_auxiliary_loss_weight"] == 0.1
    assert contract["initial_scale_minimum"] == 1e-12
    assert contract["training_epochs"] == 400
    assert contract["minimum_fit_outer_recovery"] == 0.70
    assert contract["minimum_fit_boundary_recovery"] == 0.70
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_a7c_r1_4vp_r4a_signed_renderer.py::test_r4a_contract_changes_only_the_registered_training_objective -v
```

Expected: FAIL because the R4-A contract is absent.

- [ ] **Step 3: Create the complete frozen contract**

Copy every unchanged model, optimizer, projection, split, entry, audit, source,
teacher, nearest-neighbor, and witness field from the R3 contract. Replace the
R3 contribution-reduction fields with:

```json
{
  "experiment_id": "a7c_r1_4vp_r4a_signed_renderer_trajectory_377_v1",
  "renderer_trajectory_signals": ["outer", "boundary"],
  "renderer_reconstruction_epsilon": 1e-12,
  "renderer_trajectory_huber_delta": 0.005,
  "target_response_huber_delta": 0.005,
  "renderer_outer_loss_weight": 1.0,
  "renderer_boundary_loss_weight": 1.0,
  "target_auxiliary_loss_weight": 0.1,
  "gate_auxiliary_loss_weight": 0.1,
  "gate_auxiliary_temporal_weight": 0.25,
  "initial_scale_minimum": 1e-12,
  "source_r3_policy": "utils/a7c_r1_4vp_r3_crw.py",
  "source_r3_policy_sha256": "4c6988d52f5a7eb8a7bdb3c30334de651e0d7b7d44c1cf934d8f6414f7fac899",
  "source_r3_trainer": "tools/train_a7c_r1_4vp_r3_crw.py",
  "source_r3_trainer_sha256": "ec211a00481e38e613dc7e46076e0626c9144ec1a9c52066e755801c67985ebf",
  "source_r3_auditor": "tools/audit_a7c_r1_4vp_r3_crw.py",
  "source_r3_auditor_sha256": "2dc819e34060b0a8ea7b651f2516626ab21ead9210f71f7c03c42935b39c6e82",
  "source_r3_contract": "configs/semantic/a7c_r1_4vp_r3_crw_contribution_weighted_377_v1.json",
  "source_r3_contract_sha256": "a81238d5daa8d0233cd36909f912c129aca0ec5bb72d457ce385633770322a19",
  "source_r3_fit_entry": "exp/acceptdata/a7c_r1_4vp_r3_crw_contribution_weighted_377_v1/training/fold_0/fit_renderer_entry.json",
  "source_r3_fit_entry_sha256": "b0a2a4f25cca1e24c371b8404dcbd9989911db7a7fb888f8d722837b49090420"
}
```

Pin the R4-A design SHA256
`69ced93c196f8972455e7c6c8f9a744f3a36d4ba59968e6b6ac10a2609acc764`.
Keep `attention=false`, `carrier_embedding=false`, parameter count, seed,
residual weight, fit entry, and held thresholds unchanged.

- [ ] **Step 4: Run the contract test and verify GREEN**

Run Step 2 again. Expected: PASS.

- [ ] **Step 5: Commit the contract and test**

```bash
git add configs/semantic/a7c_r1_4vp_r4a_signed_renderer_trajectory_377_v1.json \
  tests/test_a7c_r1_4vp_r4a_signed_renderer.py
git commit -m "test: freeze R1.4-VP-R4-A contract"
```

### Task 2: Implement Signed Renderer Reconstruction And Loss

**Files:**
- Create: `utils/a7c_r1_4vp_r4a.py`
- Modify: `tests/test_a7c_r1_4vp_r4a_signed_renderer.py`

- [ ] **Step 1: Add failing reconstruction and loss tests**

```python
import numpy as np
import pytest
import torch

from utils.a7c_r1_4vp_r4a import (
    freeze_initial_scales,
    mean_normalized_trajectory,
    reconstruct_renderer_sequence,
    signed_renderer_trajectory_components,
    signed_renderer_trajectory_loss,
)


def test_reconstruct_renderer_sequence_preserves_signed_point_contributions():
    base = torch.tensor([10.0, 12.0])
    point = torch.tensor([[2.0, -1.0], [4.0, -2.0]])
    gates = torch.tensor([[0.5, 1.0], [0.25, 0.5]], requires_grad=True)
    result = reconstruct_renderer_sequence(base, point, gates, epsilon=1e-12)
    torch.testing.assert_close(result, torch.tensor([9.0, 10.0]))
    result.sum().backward()
    torch.testing.assert_close(gates.grad, point)


def test_mean_normalized_trajectory_uses_its_own_differentiable_mean():
    values = torch.tensor([2.0, 4.0, 6.0], requires_grad=True)
    normalized = mean_normalized_trajectory(values, epsilon=1e-12)
    torch.testing.assert_close(normalized, torch.tensor([0.5, 1.0, 1.5]))
    normalized[0].backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_signed_components_are_zero_for_teacher_trajectory():
    streams = {
        signal: {
            "base": torch.tensor([10.0, 11.0, 9.0]),
            "point": torch.tensor([[2.0, 1.0], [1.0, 3.0], [2.0, 2.0]]),
        }
        for signal in ("target", "outer", "boundary")
    }
    gates = torch.tensor([[0.9, 1.0], [0.95, 0.9], [1.0, 0.95]])
    components = signed_renderer_trajectory_components(
        gates, gates, streams, renderer_delta=0.005,
        target_delta=0.005, gate_delta=0.01,
        gate_temporal_weight=0.25, epsilon=1e-12,
    )
    for value in components.values():
        assert float(value) == pytest.approx(0.0)


def test_frozen_scales_and_total_loss_follow_registered_coefficients():
    initial = {
        "outer": torch.tensor(2.0), "boundary": torch.tensor(4.0),
        "target": torch.tensor(0.5), "gate_aux": torch.tensor(0.25),
    }
    scales = freeze_initial_scales(initial, minimum=1e-12)
    components = {
        "outer": torch.tensor(1.0), "boundary": torch.tensor(1.0),
        "target": torch.tensor(0.25), "gate_aux": torch.tensor(0.125),
    }
    result = signed_renderer_trajectory_loss(
        components, scales, torch.tensor([[2.0, -2.0]]),
        outer_weight=1.0, boundary_weight=1.0,
        target_weight=0.1, gate_aux_weight=0.1,
        residual_weight=0.00001,
    )
    assert float(result["loss"]) == pytest.approx(0.85002)
```

Add the frozen synthetic cancellation regression below. It compares gradients
in gate space, so it tests the objective rather than optimizer behavior:

```python
def test_signed_trajectory_gradient_aligns_better_than_absolute_gate_weighting():
    base = torch.tensor([9.5813, 12.1719, 5.2744, 13.3808], dtype=torch.float64)
    point = torch.tensor([
        [0.5558, -1.2021, 0.3129], [-2.9279, -2.0738, 1.2563],
        [-0.1985, 2.3848, 1.2629], [2.6334, 2.1119, -2.2786],
    ], dtype=torch.float64)
    candidate = torch.tensor([
        [0.9846, 0.9695, 0.9184], [0.9143, 0.9306, 0.9809],
        [0.9388, 0.9541, 0.9501], [0.9631, 0.9435, 0.9068],
    ], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([
        [0.9961, 0.9572, 0.9037], [0.9196, 0.9942, 0.9959],
        [0.9512, 0.9362, 0.9438], [0.9335, 0.9500, 0.9622],
    ], dtype=torch.float64)
    signed = signed_trajectory_component(base, point, candidate, teacher,
                                         delta=0.005, epsilon=1e-12)
    rendered = reconstruct_renderer_sequence(base, point, candidate, epsilon=1e-12)
    true_flicker = torch.mean(torch.abs(torch.diff(rendered))) / torch.abs(rendered.mean())
    absolute_weight = torch.abs(point)
    absolute_weight /= absolute_weight.mean(dim=1, keepdim=True) + 1e-12
    proxy = torch.sum(absolute_weight * F.huber_loss(
        candidate, teacher, reduction="none", delta=0.01,
    )) / torch.sum(absolute_weight)
    signed_gradient = torch.autograd.grad(signed, candidate, retain_graph=True)[0]
    renderer_gradient = torch.autograd.grad(true_flicker, candidate, retain_graph=True)[0]
    proxy_gradient = torch.autograd.grad(proxy, candidate)[0]
    assert cosine(signed_gradient, renderer_gradient) > 0.90
    assert cosine(signed_gradient, renderer_gradient) \
        > cosine(proxy_gradient, renderer_gradient) + 0.50
```

Expose `signed_trajectory_component` as the single-signal primitive used by
`signed_renderer_trajectory_components`; define the test-local `cosine` as the
flattened dot product divided by both finite nonzero norms.

Use explicit fail-closed tests:

```python
def test_renderer_reconstruction_rejects_shape_nonfinite_and_zero_mean():
    with pytest.raises(ValueError, match="align"):
        reconstruct_renderer_sequence(
            torch.ones(2), torch.ones(2, 3), torch.ones(2, 2), epsilon=1e-12,
        )
    bad = torch.ones(2, 2)
    bad[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        reconstruct_renderer_sequence(
            torch.ones(2), bad, torch.ones(2, 2), epsilon=1e-12,
        )
    with pytest.raises(ValueError, match="mean"):
        reconstruct_renderer_sequence(
            torch.tensor([1.0, -1.0]), torch.zeros(2, 2),
            torch.ones(2, 2), epsilon=1e-12,
        )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
def test_initial_scales_reject_nonpositive_or_nonfinite_values(value):
    components = {name: torch.tensor(1.0) for name in
                  ("outer", "boundary", "target", "gate_aux")}
    components["outer"] = torch.tensor(value)
    with pytest.raises(ValueError, match="scale"):
        freeze_initial_scales(components, minimum=1e-12)


def test_total_loss_rejects_missing_component_and_wrong_residual_weight():
    scales = {name: 1.0 for name in ("outer", "boundary", "target", "gate_aux")}
    components = {name: torch.tensor(1.0) for name in scales}
    del components["target"]
    with pytest.raises(ValueError, match="component"):
        signed_renderer_trajectory_loss(
            components, scales, torch.zeros(2, 2), outer_weight=1.0,
            boundary_weight=1.0, target_weight=0.1, gate_aux_weight=0.1,
            residual_weight=0.00001,
        )
    components["target"] = torch.tensor(1.0)
    with pytest.raises(ValueError, match="residual_weight"):
        signed_renderer_trajectory_loss(
            components, scales, torch.zeros(2, 2), outer_weight=1.0,
            boundary_weight=1.0, target_weight=0.1, gate_aux_weight=0.1,
            residual_weight=0.001,
        )
```

- [ ] **Step 2: Run and verify RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_a7c_r1_4vp_r4a_signed_renderer.py \
  -k 'reconstruct or normalized_trajectory or signed_components or frozen_scales' -v
```

Expected: collection FAIL with `ModuleNotFoundError: utils.a7c_r1_4vp_r4a`.

- [ ] **Step 3: Implement the minimal pure loss API**

```python
def reconstruct_renderer_sequence(base, point, gates, *, epsilon):
    require_finite_aligned(base, point, gates)
    result = base - point.sum(dim=1) + (point * gates).sum(dim=1)
    if not torch.isfinite(result).all() or torch.abs(result.mean()) <= epsilon:
        raise ValueError("renderer sequence mean must be finite and nonzero")
    return result


def mean_normalized_trajectory(values, *, epsilon):
    if values.ndim != 1 or values.numel() < 2 or not torch.isfinite(values).all():
        raise ValueError("trajectory must be a finite vector with at least two frames")
    mean = values.mean()
    if torch.abs(mean) <= epsilon:
        raise ValueError("trajectory mean must be nonzero")
    return values / torch.clamp(torch.abs(mean), min=epsilon)
```

`signed_renderer_trajectory_components` reconstructs candidate and teacher
signals, matches signed adjacent mean-normalized outer/boundary differences,
matches target responses against frozen base target, and computes the
unweighted gate plus `0.25` temporal Huber auxiliary.

`freeze_initial_scales` returns detached Python floats for exactly
`outer/boundary/target/gate_aux`. `signed_renderer_trajectory_loss` validates
the four fixed component weights and residual weight and returns total plus
normalized component values.

- [ ] **Step 4: Run focused tests and R3 regressions**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest -q \
  tests/test_a7c_r1_4vp_r4a_signed_renderer.py \
  tests/test_a7c_r1_4vp_r3_crw.py
```

- [ ] **Step 5: Commit the loss module**

```bash
git add utils/a7c_r1_4vp_r4a.py tests/test_a7c_r1_4vp_r4a_signed_renderer.py
git commit -m "feat: add signed renderer-trajectory loss"
```

### Task 3: Add Carrier-Action Diagnostics

**Files:**
- Modify: `utils/a7c_r1_4vp_r4a.py`
- Modify: `tests/test_a7c_r1_4vp_r4a_signed_renderer.py`

- [ ] **Step 1: Add failing diagnostic tests**

```python
from utils.a7c_r1_4vp_r4a import summarize_action_recovery


def test_action_diagnostics_report_rank_overlap_and_false_maximum():
    base = np.ones((4, 3))
    teacher = np.array([
        [0.9, 1.0, 1.0], [0.9, 0.95, 1.0],
        [0.95, 0.9, 1.0], [1.0, 0.9, 0.95],
    ])
    learned = teacher.copy()
    learned[0, 0] = 1.0
    result = summarize_action_recovery(
        learned, teacher, base, top_k=2, suppression_tolerance=0.001,
    )
    assert result["missed_teacher_suppression_count"] == 1
    assert result["missed_teacher_suppression_fraction"] == pytest.approx(0.2)
    assert 0.0 <= result["top_k_suppression_overlap"] <= 1.0
    assert result["action_rank_90"] >= 1
    assert result["action_rank_95"] >= result["action_rank_90"]
```

Add exact edge tests:

```python
def test_action_diagnostics_reject_undefined_or_invalid_inputs():
    base = np.ones((3, 2))
    with pytest.raises(ValueError, match="teacher action"):
        summarize_action_recovery(base, base, base, top_k=1,
                                  suppression_tolerance=0.001)
    with pytest.raises(ValueError, match="top_k"):
        summarize_action_recovery(base - 0.1, base - 0.2, base, top_k=3,
                                  suppression_tolerance=0.001)
    bad = base.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        summarize_action_recovery(bad, base - 0.1, base, top_k=1,
                                  suppression_tolerance=0.001)


def test_action_diagnostics_define_zero_mae_share_as_zero():
    base = np.ones((3, 2))
    teacher = base - np.array([[0.1, 0.0], [0.1, 0.0], [0.1, 0.0]])
    result = summarize_action_recovery(
        teacher, teacher, base, top_k=1, suppression_tolerance=0.001,
    )
    assert result["false_maximum_mae_share"] == 0.0
```

- [ ] **Step 2: Run and verify RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_a7c_r1_4vp_r4a_signed_renderer.py -k action_diagnostics -v
```

Expected: FAIL because `summarize_action_recovery` is absent.

- [ ] **Step 3: Implement deterministic diagnostics**

Compute action cosine from `(gate - base)`, per-frame top-k overlap from
`base - gate`, SVD energy ranks from the learned action matrix, and missed
suppression from `teacher < base - tolerance` combined with
`learned >= base - tolerance`. Return JSON-native numbers only. Diagnostics
must not affect loss, checkpoint selection, entry classification, or retries.

- [ ] **Step 4: Run and commit**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_a7c_r1_4vp_r4a_signed_renderer.py -q
git add utils/a7c_r1_4vp_r4a.py tests/test_a7c_r1_4vp_r4a_signed_renderer.py
git commit -m "feat: report R4-A carrier-action diagnostics"
```

### Task 4: Implement The R4-A Trainer

**Files:**
- Create: `tools/train_a7c_r1_4vp_r4a_signed_renderer.py`
- Modify: `tests/test_a7c_r1_4vp_r4a_signed_renderer.py`

- [ ] **Step 1: Add a failing synthetic trainer test**

Use this exact two-segment fixture. Only the first segment is fit; held teacher
and renderer rows remain NaN. Monkeypatch `_project_segments` to identity and
use four CPU epochs:

```python
samples, carriers = 8, 2
fit_mask = np.array([True] * 4 + [False] * 4)
teacher = np.full((samples, carriers), np.nan, np.float32)
teacher[:4] = np.array([[0.93, 0.95], [0.94, 0.92],
                        [0.92, 0.94], [0.95, 0.93]], np.float32)
streams = {}
for signal, level in (("target", 20.0), ("outer", 8.0), ("boundary", 5.0)):
    stream_base = np.full(samples, np.nan, np.float32)
    stream_point = np.full((samples, carriers), np.nan, np.float32)
    stream_base[:4] = level + np.array([0.0, 1.0, -0.5, 0.5])
    stream_point[:4] = np.array([[1.0, 0.5], [0.5, 1.0],
                                 [1.0, 0.25], [0.25, 1.0]])
    streams[signal] = {"base": stream_base, "point": stream_point}
features = np.linspace(-1.0, 1.0, samples * carriers * 3).reshape(samples, carriers, 3)
pose = np.linspace(-1.0, 1.0, samples * 36).reshape(samples, 36)
adjacency = np.repeat(np.eye(carriers)[None], samples, axis=0)
visibility = np.ones((samples, carriers), np.float32)
base = np.full((samples, carriers), 0.97, np.float32)
camera = np.zeros(samples, np.int64)
frames = np.tile(np.arange(4), 2)
blocks = np.repeat([0, 1], 4)
contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
contract.update({"training_epochs": 4, "frame_stride": 1,
                 "view_embedding_dimension": 4,
                 "pose_embedding_dimension": 4,
                 "gru_hidden_dimension": 4,
                 "maximum_fit_teacher_mae": 0.1})
```

Call the trainer and assert:

```python
summary = train_fold(
    fold=0, features=features, pose=pose, adjacency=adjacency,
    visibility=visibility, base_gates=base, teacher_gates=teacher,
    renderer_streams=streams, teacher_mask=fit_mask,
    prediction_mask=np.ones(8, bool), camera_index=camera,
    frame_index=frames, block_ids=blocks, runtime_mass=mass,
    a5_weight=a5, contract=contract, output_dir=tmp_path, device="cpu",
)
assert summary["checkpoint_epoch"] == 4
assert summary["final_components"]["loss"] < summary["initial_components"]["loss"]
assert summary["held_teacher_values_accessed"] is False
assert summary["held_renderer_values_accessed"] is False
assert len(summary["segment_initial_scales"]) == 1
assert set(summary["segment_initial_scales"][0]["scales"]) \
    == {"outer", "boundary", "target", "gate_aux"}
assert (tmp_path / "model.pt").is_file()
assert (tmp_path / "predictions.npz").is_file()
```

Add a model-freeze regression in the same RED batch:

```python
def test_r4a_keeps_the_r3_model_signature_and_budget():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    model = ViewPoseResidualCompositor(
        view_dimension=49, view_embedding_dimension=16,
        pose_dimension=36, pose_embedding_dimension=16,
        gru_hidden_dimension=16, residual_gate_scale=0.1,
        minimum_gate=0.9, maximum_gate=1.0,
    )
    assert sum(value.numel() for value in model.parameters()) == 9073
    assert contract["attention"] is False
    assert contract["carrier_embedding"] is False
    assert contract["maximum_projection_gate_jump"] == 0.015
```

- [ ] **Step 2: Run and verify RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_a7c_r1_4vp_r4a_signed_renderer.py -k train_fold -v
```

Expected: collection FAIL because the trainer is absent.

- [ ] **Step 3: Implement `train_fold`**

Reuse pinned R3 helpers and the R2 runtime. Validate fit streams as finite and
held rows as nonfinite. Instantiate the unchanged model and require exactly
`9073` parameters. Before optimizer construction, evaluate the zero-residual
model once per fit segment and freeze four detached scales. Train complete
segments with the registered R4-A components and total loss for 400 epochs.

Write R3-compatible model, predictions, projection certificates, summary, and
fit entry artifacts. Store `segment_initial_scales` as a fixed-order list of
`{"camera_index", "block_id", "scales"}` records. Add raw/projected action diagnostics and
initial/final raw and normalized components. Never load held witnesses or
nearest-neighbor predictions in the trainer.

- [ ] **Step 4: Implement formal orchestration**

Parse R3's arguments including `--evidence`. Verify every R4-A, R3, R2, input,
pose, teacher, and R1.2-B hash. Build objective streams once, but replace every
row outside each fold's teacher mask with NaN before `train_fold`.

Use the unchanged R3 fit entry calculation. Combine renderer conditions with
final-loss improvement and MAE. Fold-0 failure writes
`FIT_RENDERER_ENTRY_NEGATIVE`, returns 2, and stops. A later fit failure raises
`RuntimeError`. Six passes freeze exactly 30 artifacts.

- [ ] **Step 5: Run trainer regressions**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest -q \
  tests/test_a7c_r1_4vp_r4a_signed_renderer.py \
  tests/test_a7c_r1_4vp_r3_crw.py \
  tests/test_a7c_r1_4vp_r2_loss_repair.py
```

- [ ] **Step 6: Commit the trainer**

```bash
git add tools/train_a7c_r1_4vp_r4a_signed_renderer.py \
  tests/test_a7c_r1_4vp_r4a_signed_renderer.py
git commit -m "feat: train R4-A signed renderer folds"
```

### Task 5: Implement Audit And Fail-Closed Runner

**Files:**
- Create: `tools/audit_a7c_r1_4vp_r4a_signed_renderer.py`
- Create: `tools/run_a7c_r1_4vp_r4a_signed_renderer_377.sh`
- Modify: `tests/test_a7c_r1_4vp_r4a_signed_renderer.py`

- [ ] **Step 1: Add failing wrapper and runner tests**

```python
def test_r4a_audit_wrapper_preserves_negative_status(monkeypatch):
    monkeypatch.setattr(
        "tools.audit_a7c_r1_4vp_r4a_signed_renderer.r3_audit._run",
        lambda args: ({"stage": "r1_4vp_r3_crw_held_canary",
                       "verdict": "CANARY_NEGATIVE"}, 2),
    )
    payload, status = run_r4a_audit(SimpleNamespace())
    assert payload["stage"] == "r1_4vp_r4a_signed_renderer_held_canary"
    assert status == 2


def test_r4a_runner_maps_fit_rejection_without_opening_audit():
    source = RUNNER.read_text(encoding="utf-8")
    assert "FIT_RENDERER_ENTRY_NEGATIVE) mark_terminal fit_rejected" in source
    assert "CANARY_NEGATIVE) mark_terminal rejected" in source
    assert "CANARY_PROMOTED) mark_terminal completed" in source
    assert 'if "${PYTHON}" "${ROOT}/tools/audit_a7c_r1_4vp_r4a_signed_renderer.py"' in source
    assert 'for marker in completed rejected fit_rejected failed' in source
```

- [ ] **Step 2: Run and verify RED**

Run the R4-A test file with `-k 'audit or runner'`. Expected: FAIL because both
files are absent.

- [ ] **Step 3: Implement the audit wrapper**

Import `tools.audit_a7c_r1_4vp_r3_crw` as `r3_audit`. Delegate `_run(args)` to
the pinned calculation and change only `stage` to
`r1_4vp_r4a_signed_renderer_held_canary`. Write audit/root summaries atomically
and preserve statuses 0, 2, and 1.

- [ ] **Step 4: Implement the runner**

Copy R3 detached semantics. Change paths, contract hash, trainer, and auditor.
Preflight-verify contract, design, R2/R3 sources, R3 diagnostic, all teachers,
R1.2-B/NN/witness artifacts, pose manifest, and CUDA. Trainer status 2 is
terminal only when verdict is exactly `FIT_RENDERER_ENTRY_NEGATIVE`; write
`.fit_rejected` and do not audit. Require six fit entries and freeze manifest
before held audit.

- [ ] **Step 5: Run focused checks**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_a7c_r1_4vp_r4a_signed_renderer.py -q
bash -n tools/run_a7c_r1_4vp_r4a_signed_renderer_377.sh
/opt/miniconda3/envs/ictrl/bin/python -m py_compile \
  utils/a7c_r1_4vp_r4a.py \
  tools/train_a7c_r1_4vp_r4a_signed_renderer.py \
  tools/audit_a7c_r1_4vp_r4a_signed_renderer.py
```

- [ ] **Step 6: Commit audit and runner**

```bash
git add tools/audit_a7c_r1_4vp_r4a_signed_renderer.py \
  tools/run_a7c_r1_4vp_r4a_signed_renderer_377.sh \
  tests/test_a7c_r1_4vp_r4a_signed_renderer.py
git commit -m "feat: audit and run R4-A signed renderer canary"
```

### Task 6: Verify And Launch Formal Fold 0

**Files:**
- Output: `exp/acceptdata/a7c_r1_4vp_r4a_signed_renderer_trajectory_377_v1/`
- Modify: `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`

- [ ] **Step 1: Run the complete regression suite**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest -q \
  tests/test_a7c_r1_4vp_r4a_signed_renderer.py \
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

- [ ] **Step 2: Verify formal preflight**

Require every pinned SHA256, pose manifest, CUDA, idle GPU, no active A7 process,
and an absent R4-A output root. Never delete or reuse a partial formal root.

- [ ] **Step 3: Launch exactly once**

```bash
setsid -f bash tools/run_a7c_r1_4vp_r4a_signed_renderer_377.sh \
  >/dev/null 2>&1
```

- [ ] **Step 4: Wait for fold-0 entry**

Poll PID, `runner.log`, and `training/fold_0/fit_renderer_entry.json`. On
failure require only `.fit_rejected`, verify no audit directory, record actual
Beijing end time, and do not relaunch. On pass use measured fold-0 duration to
report the six-fold ETA while the same runner continues.

- [ ] **Step 5: Verify and document terminal data**

Require one terminal marker and root-summary agreement. Recompute reported
metrics and hashes from disk. Add an R4-A result section to the A7 handoff
document without staging unrelated pre-existing edits.

- [ ] **Step 6: Run final verification**

Repeat Step 1 plus `bash -n`, `py_compile`, `git diff --check`, process-exit
checks for a terminal run, and held-isolation checks. Never claim promotion
without an independent held audit reporting `CANARY_PROMOTED`.
