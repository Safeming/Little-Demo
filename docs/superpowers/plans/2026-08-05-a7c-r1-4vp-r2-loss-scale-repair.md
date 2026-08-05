# A7c R1.4-VP-R2 Loss-Scale Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a new six-fold R1.4-VP-R2 canary that changes only the latent residual loss weight, enforces fit-only convergence, and correctly classifies negative audit results.

**Architecture:** New R2 files consume the already-frozen R1.4-VP teachers and the existing runtime model utilities; deleted R1.4 scripts and tests remain deleted. A small R2 policy module owns convergence, topology, and terminal semantics so trainer, auditor, and runner behavior can be tested independently.

**Tech Stack:** Python 3.9, NumPy, PyTorch, SciPy/HiGHS hard projection, pytest, Bash, JSON/NPZ artifacts.

---

### Task 1: Freeze The R2 Contract

**Files:**
- Create: `configs/semantic/a7c_r1_4vp_r2_loss_scale_repair_377_v1.json`
- Create: `tests/test_a7c_r1_4vp_r2_loss_repair.py`

- [ ] **Step 1: Write the failing contract test**

```python
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/semantic/a7c_r1_4vp_r2_loss_scale_repair_377_v1.json"


def test_r2_contract_changes_only_registered_loss_and_integrity_behavior():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "a7c_r1_4vp_r2_loss_scale_repair_377_v1"
    assert contract["status"] == "frozen"
    assert contract["residual_loss_weight"] == 0.00001
    assert contract["training_epochs"] == 400
    assert contract["maximum_fit_teacher_mae"] == 0.007
    assert contract["require_final_fit_loss_improvement"] is True
    assert contract["fit_cameras"] == ["c01", "c05", "c09", "c13"]
    assert contract["temporal_block_count"] == 6
    assert contract["view_feature_group"] == "F3"
    assert contract["maximum_parameter_count"] == 50_000
    assert contract["minimum_outer_gain"] == 0.005
    assert contract["minimum_boundary_gain"] == 0.005
    assert contract["maximum_visibility_response_ratio"] == 1.0
    assert contract["deployment_eligible"] is False
    assert contract["teacher_eligible"] is False
    assert contract["paper_test_eligible"] is False
```

- [ ] **Step 2: Run the test and verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r2_loss_repair.py::test_r2_contract_changes_only_registered_loss_and_integrity_behavior -v`

Expected: FAIL because the R2 contract does not exist.

- [ ] **Step 3: Create the frozen contract**

Create a JSON contract containing the exact values from the approved R2 design. Pin:

```json
{
  "schema_version": 1,
  "experiment_id": "a7c_r1_4vp_r2_loss_scale_repair_377_v1",
  "status": "frozen",
  "subject": "377",
  "fit_cameras": ["c01", "c05", "c09", "c13"],
  "temporal_block_count": 6,
  "view_feature_group": "F3",
  "residual_loss_weight": 0.00001,
  "training_epochs": 400,
  "maximum_fit_teacher_mae": 0.007,
  "require_final_fit_loss_improvement": true,
  "minimum_outer_gain": 0.005,
  "minimum_boundary_gain": 0.005,
  "maximum_visibility_response_ratio": 1.0,
  "deployment_eligible": false,
  "teacher_eligible": false,
  "paper_test_eligible": false
}
```

Also copy every unchanged model, optimizer, projection, audit, source path, and source SHA256 value required by `build_runtime_inputs`, `train_fold`, and the independent auditor. Pin the R2 design SHA256 plus these 13 frozen teacher artifacts:

```text
teachers/summary.json
teachers/fold_0..5/teacher.npz
teachers/fold_0..5/certificates.json
```

Do not reference or recreate the deleted old R1.4 contract path.

- [ ] **Step 4: Run the contract test and verify GREEN**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r2_loss_repair.py::test_r2_contract_changes_only_registered_loss_and_integrity_behavior -v`

Expected: PASS.

- [ ] **Step 5: Commit the contract**

```bash
git add configs/semantic/a7c_r1_4vp_r2_loss_scale_repair_377_v1.json tests/test_a7c_r1_4vp_r2_loss_repair.py
git commit -m "test: freeze R1.4-VP-R2 contract"
```

### Task 2: Implement Loss And Integrity Policy

**Files:**
- Create: `utils/a7c_r1_4vp_r2.py`
- Modify: `tests/test_a7c_r1_4vp_r2_loss_repair.py`

- [ ] **Step 1: Add failing policy tests**

```python
import numpy as np
import pytest
import torch


def test_registered_loss_can_move_off_anchor_without_canceling_huber_gradient():
    from utils.a7c_r1_4vp_r2 import r2_distillation_loss

    residual = torch.zeros((2, 1), requires_grad=True)
    base = torch.full((2, 1), 0.97)
    teacher = torch.full((2, 1), 0.95)
    prediction = base + 0.1 * torch.tanh(residual)
    components = r2_distillation_loss(
        prediction, teacher, residual,
        gate_delta=0.01, temporal_delta=0.005,
        temporal_weight=0.25, residual_weight=0.00001,
    )
    components["loss"].backward()
    assert residual.grad.mean().item() > 0.0
    assert abs(residual.grad.mean().item()) >= 0.00049


def test_fit_integrity_requires_loss_improvement_and_mae_limit():
    from utils.a7c_r1_4vp_r2 import require_fit_integrity

    require_fit_integrity(
        initial_loss=1.0e-4, final_loss=5.0e-5,
        fit_teacher_mae=0.006, maximum_fit_teacher_mae=0.007,
    )
    with pytest.raises(RuntimeError, match="fit loss did not improve"):
        require_fit_integrity(1.0e-4, 1.0e-4, 0.006, 0.007)
    with pytest.raises(RuntimeError, match="fit teacher MAE"):
        require_fit_integrity(1.0e-4, 5.0e-5, 0.008, 0.007)


def test_topology_guard_broadcasts_base_mask_and_rejects_real_undercrossing():
    from utils.a7c_r1_4vp_r2 import evaluate_topology_guard

    base = np.array([0.3, 0.4])
    candidate = np.array([[0.29, 0.39], [0.28, 0.38]])
    passed = evaluate_topology_guard(base, candidate, threshold=0.2)
    assert passed["passed"] is True
    assert passed["mismatch_count"] == 0
    broken = candidate.copy()
    broken[0, 0] = 0.19
    failed = evaluate_topology_guard(base, broken, threshold=0.2)
    assert failed["passed"] is False
    assert failed["mismatch_count"] == 1


def test_terminal_classification_preserves_negative_audit_status():
    from utils.a7c_r1_4vp_r2 import classify_terminal_status

    assert classify_terminal_status(0, "CANARY_PROMOTED") == "completed"
    assert classify_terminal_status(2, "CANARY_NEGATIVE") == "rejected"
    assert classify_terminal_status(1, "TRAINING_ERROR") == "failed"
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r2_loss_repair.py -k 'registered_loss or fit_integrity or topology_guard or terminal_classification' -v`

Expected: FAIL with `ModuleNotFoundError: utils.a7c_r1_4vp_r2`.

- [ ] **Step 3: Implement the R2 policy module**

Implement:

```python
def r2_distillation_loss(prediction, teacher, residual, *, gate_delta,
                         temporal_delta, temporal_weight, residual_weight):
    if float(residual_weight) != 0.00001:
        raise ValueError("R2 residual weight differs from preregistration")
    gate = F.huber_loss(prediction, teacher, reduction="mean", delta=gate_delta)
    temporal = F.huber_loss(
        torch.diff(prediction, dim=0), torch.diff(teacher, dim=0),
        reduction="mean", delta=temporal_delta,
    )
    latent = torch.mean(torch.abs(residual))
    return {"loss": gate + temporal_weight * temporal + residual_weight * latent,
            "gate": gate, "temporal": temporal, "residual": latent}


def require_fit_integrity(initial_loss, final_loss, fit_teacher_mae,
                          maximum_fit_teacher_mae):
    if not float(final_loss) < float(initial_loss):
        raise RuntimeError("fit loss did not improve")
    if float(fit_teacher_mae) > float(maximum_fit_teacher_mae):
        raise RuntimeError("fit teacher MAE exceeds frozen maximum")


def evaluate_topology_guard(base_weight, candidate_weight, threshold):
    base = np.asarray(base_weight, dtype=np.float64).reshape(1, -1)
    candidate = np.asarray(candidate_weight, dtype=np.float64)
    expected = np.broadcast_to(base >= threshold, candidate.shape)
    observed = candidate >= threshold
    return {"passed": bool(np.array_equal(observed, expected)),
            "mismatch_count": int(np.count_nonzero(observed != expected)),
            "minimum_slack": float(np.min(candidate - threshold))}


def classify_terminal_status(audit_status, verdict):
    mapping = {(0, "CANARY_PROMOTED"): "completed",
               (2, "CANARY_NEGATIVE"): "rejected"}
    return mapping.get((int(audit_status), str(verdict)), "failed")
```

Add finite/shape/value validation matching existing A7c utilities.

- [ ] **Step 4: Run policy tests and verify GREEN**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r2_loss_repair.py -k 'registered_loss or fit_integrity or topology_guard or terminal_classification' -v`

Expected: PASS.

- [ ] **Step 5: Commit the policy module**

```bash
git add utils/a7c_r1_4vp_r2.py tests/test_a7c_r1_4vp_r2_loss_repair.py
git commit -m "feat: add R1.4-VP-R2 repair policy"
```

### Task 3: Implement The R2 Trainer

**Files:**
- Create: `tools/train_a7c_r1_4vp_r2_loss_repair.py`
- Modify: `tests/test_a7c_r1_4vp_r2_loss_repair.py`

- [ ] **Step 1: Add a failing fold-training test**

Add a synthetic `train_fold` test using eight samples, two complete blocks,
finite teacher values only in the first block, `training_epochs=20`, and an
easy teacher offset of `-0.01`. Assert:

```python
assert summary["checkpoint_epoch"] == 20
assert summary["final_components"]["loss"] < summary["initial_components"]["loss"]
assert summary["fit_teacher_mae"] <= 0.007
assert summary["held_teacher_values_accessed"] is False
assert summary["residual_loss_weight"] == 0.00001
assert (tmp_path / "model.pt").is_file()
assert (tmp_path / "predictions.npz").is_file()
```

- [ ] **Step 2: Run the trainer test and verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r2_loss_repair.py -k train_fold -v`

Expected: FAIL because the R2 trainer does not exist.

- [ ] **Step 3: Implement the trainer**

Create a self-contained R2 trainer under the new filename. Reuse only stable
utilities from `utils/a7c_view_pose_compositor.py`, R1.3-P projection, existing
probe loaders, and `utils/a7c_r1_4vp_r2.py`. Do not import a deleted R1.4 tool.

The public `train_fold` must:

```text
validate fit teacher finite / held teacher NaN isolation
fit F3 and pose normalization from teacher_mask only
pack complete camera-block segments
train final epoch only with R2 loss weight 0.00001
infer only the four-camera prediction mask
hard-project each camera-block independently
record latent residual mean/max and base-displacement recovery
call require_fit_integrity before writing model.pt
write all eligibility flags false
```

The CLI verifies all source and teacher hashes, processes folds 0-5, writes
`training/fold_N`, and creates `models_frozen.json` only after every fold
passes fit integrity. It reuses the frozen R1.4 nearest-neighbor predictions by
hash rather than recomputing or tuning them.

- [ ] **Step 4: Run trainer and projection tests**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r2_loss_repair.py tests/test_a7c_temporal_joint_projection.py tests/test_a7c_overlap_set_compositor.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the trainer**

```bash
git add tools/train_a7c_r1_4vp_r2_loss_repair.py tests/test_a7c_r1_4vp_r2_loss_repair.py
git commit -m "feat: train R1.4-VP-R2 loss repair"
```

### Task 4: Implement The Corrected Independent Auditor

**Files:**
- Create: `tools/audit_a7c_r1_4vp_r2_loss_repair.py`
- Modify: `tests/test_a7c_r1_4vp_r2_loss_repair.py`

- [ ] **Step 1: Add failing auditor tests**

Add tests that pass 24 formal learned records and 24 NN records to
`classify_canary`, then break one promotion gate and expect
`CANARY_NEGATIVE`. Add an artifact test requiring six R2 model hashes and
unchanged frozen NN hashes. Add a topology regression using the real R1.4
projected array shape and assert zero mismatch.

- [ ] **Step 2: Run auditor tests and verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r2_loss_repair.py -k auditor -v`

Expected: FAIL because the R2 auditor does not exist.

- [ ] **Step 3: Implement independent audit**

Create a new auditor that does not import the trainer. Before evidence metrics,
verify contract, teacher, model, NN, prediction, normalization, mask, and
projection hashes. Recompute all 24 records and unchanged formal thresholds.
Use `evaluate_topology_guard` on full-shape candidate weights. Write atomically:

```text
audit/held_block_summary.json
summary.json
```

Return 0 for `CANARY_PROMOTED`, 2 for `CANARY_NEGATIVE`, and 1 for integrity or
numerical errors.

- [ ] **Step 4: Run auditor and frozen regressions**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r2_loss_repair.py tests/test_a7c_exact_aggregate_oracle.py tests/test_a7c_temporal_joint_projection.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the auditor**

```bash
git add tools/audit_a7c_r1_4vp_r2_loss_repair.py tests/test_a7c_r1_4vp_r2_loss_repair.py
git commit -m "feat: audit R1.4-VP-R2 held canary"
```

### Task 5: Implement The Restart-Safe R2 Runner

**Files:**
- Create: `tools/run_a7c_r1_4vp_r2_loss_repair_377.sh`
- Modify: `tests/test_a7c_r1_4vp_r2_loss_repair.py`

- [ ] **Step 1: Add a failing runner protocol test**

Assert the runner contains the R2 trainer, `models_frozen.json`, R2 auditor,
explicit `completed/rejected/failed` terminal mapping, timestamps, PID, and
restart checks. Assert the audit is invoked inside an `if` statement and the
source contains none of the forbidden camera names.

- [ ] **Step 2: Run the runner test and verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r2_loss_repair.py -k runner -v`

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement the runner**

The runner order is:

```text
verify contract/design/source/teacher/pose hashes
write started_utc.txt once
train or validate six folds and fit-integrity summaries
require models_frozen.json
if auditor succeeds: audit_status=0
else: audit_status=$? without firing ERR trap
read verdict and map through classify_terminal_status
write ended_utc.txt and exactly one terminal marker
```

Use `/opt/miniconda3/envs/ictrl/bin/python`, `set -euo pipefail`, atomic error
summaries, and no old R1.4 script dependency.

- [ ] **Step 4: Verify shell and runner tests**

Run: `bash -n tools/run_a7c_r1_4vp_r2_loss_repair_377.sh && /opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r2_loss_repair.py -k runner -v`

Expected: PASS.

- [ ] **Step 5: Commit the runner**

```bash
git add tools/run_a7c_r1_4vp_r2_loss_repair_377.sh tests/test_a7c_r1_4vp_r2_loss_repair.py
git commit -m "feat: orchestrate R1.4-VP-R2 canary"
```

### Task 6: Verify, Smoke-Test, And Launch

**Files:**
- Modify after terminal result: `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`

- [ ] **Step 1: Run focused R2 tests**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_r1_4vp_r2_loss_repair.py -v`

Expected: PASS.

- [ ] **Step 2: Run frozen A7c regressions**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_exact_aggregate_oracle.py tests/test_a7c_feasibility_oracle.py tests/test_a7c_temporal_joint_projection.py tests/test_a7c_overlap_set_compositor.py tests/test_a7c_quotient_compositor.py tests/test_a7c_ray_context_probe.py -v`

Expected: PASS.

- [ ] **Step 3: Run the real-size fit-only smoke test**

Run fold 0 for 100 epochs under the R2 loss in a temporary output. Require:

```text
final fit loss < initial fit loss
fit teacher MAE <= 0.007
held teacher finite count = 0
no held renderer metric opened
```

- [ ] **Step 4: Verify GPU and launch detached runner**

```bash
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
OUT=/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/a7c_r1_4vp_r2_loss_scale_repair_377_v1
mkdir -p "$OUT"
setsid -f bash tools/run_a7c_r1_4vp_r2_loss_repair_377.sh "$OUT" \
  > "$OUT/launch.log" 2>&1 < /dev/null
```

Verify a live `runner.pid`, growing `runner.log`, no terminal marker, and GPU
training activity.

- [ ] **Step 5: Estimate Beijing completion from observed fold work**

Use the measured smoke/full-fold duration plus remaining folds and audit
allowance. Report one estimated Beijing end time and a bounded uncertainty
window. Do not report an actual end until `ended_utc.txt` exists.

- [ ] **Step 6: Record the terminal result later without staging user edits**

When exactly one terminal marker exists, append a dated R2 result block to the
A7 handoff using a narrowly constructed patch. Include corrected terminal
status, fit integrity, 24-record metrics, topology mismatch/slack, comparison
margins, and next-route decision. Stage only that added block.
