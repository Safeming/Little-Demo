# A7c R1.2-B Dense Overlap-Set Compositor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and held-block audit a stateless CoreView377 compositor whose carrier gates use a continuous permutation-equivariant screen-space overlap set.

**Architecture:** Reuse the frozen R1.2-A inputs, joint target-budget projection, renderer sequence objective, and auditor. Replace only the independent F1 MLP with a shared node encoder and dense continuous overlap aggregation, then use the unchanged six-fold protocol and promotion gates.

**Tech Stack:** Python, NumPy, PyTorch, pytest, existing A7c utilities, Bash runner.

---

### Task 1: Freeze The R1.2-B Contract

**Files:**
- Create: `configs/semantic/a7c_r1_2b_dense_overlap_set_377_v1.json`
- Create: `tests/test_a7c_overlap_set_compositor.py`

- [ ] **Step 1: Write the failing contract test**

```python
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/semantic/a7c_r1_2b_dense_overlap_set_377_v1.json"


def test_contract_changes_only_the_registered_predictor():
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["status"] == "frozen"
    assert payload["predictor"] == "dense_overlap_set"
    assert payload["score_feature_group"] == "F1"
    assert payload["node_hidden_dimension"] == 32
    assert payload["gate_hidden_dimension"] == 32
    assert payload["spatial_scale"] == 0.03
    assert payload["depth_scale"] == 0.04
    assert payload["teacher_gate_loss_weight"] == 0.0
    assert payload["runtime_state"] is False
    assert payload["paper_test_eligible"] is False
    assert payload["fit_cameras"] == ["c01", "c05", "c09", "c13"]
    assert payload["audit_cameras"] == ["c17", "c18", "c19", "c20"]
    assert payload["forbidden_cameras"] == ["c21", "c22", "c23"]
```

- [ ] **Step 2: Run the test and verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_overlap_set_compositor.py::test_contract_changes_only_the_registered_predictor -q`

Expected: FAIL because the contract does not exist.

- [ ] **Step 3: Add the frozen contract**

Start with the complete JSON object from
`a7c_r1_2a_quotient_compositor_377_v1.json`. Replace only `experiment_id` and
add the predictor fields below; retain every other key and value byte-for-value.
Record the two shown design/parent fingerprints:

```json
{
  "schema_version": 1,
  "experiment_id": "a7c_r1_2b_dense_overlap_set_377_v1",
  "status": "frozen",
  "predictor": "dense_overlap_set",
  "score_feature_group": "F1",
  "node_hidden_dimension": 32,
  "gate_hidden_dimension": 32,
  "spatial_scale": 0.03,
  "depth_scale": 0.04,
  "edge_log_weight_minimum": -20.0,
  "source_r1_2b_design": "docs/superpowers/specs/2026-08-03-a7c-r1-2b-dense-overlap-set-compositor-design.md",
  "source_r1_2b_design_sha256": "fd099423bcb9bee1ae4b10608468744b511ab00d07866a5d8f5551e310302550",
  "source_r1_2a_contract": "configs/semantic/a7c_r1_2a_quotient_compositor_377_v1.json",
  "source_r1_2a_contract_sha256": "8f59a97ddbf730516ba0ba26871487210e2bddcb3cf8958619bca3a465c25ff4",
  "initial_minimum_gate": 0.999,
  "teacher_gate_loss_weight": 0.0,
  "runtime_state": false,
  "paper_test_eligible": false
}
```

Do not add any tunable graph alternative.

- [ ] **Step 4: Verify GREEN and fingerprints**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_overlap_set_compositor.py::test_contract_changes_only_the_registered_predictor -q`

Run: `sha256sum docs/superpowers/specs/2026-08-03-a7c-r1-2b-dense-overlap-set-compositor-design.md configs/semantic/a7c_r1_2a_quotient_compositor_377_v1.json`

Expected: PASS and fingerprints equal the contract.

- [ ] **Step 5: Commit**

```bash
git add configs/semantic/a7c_r1_2b_dense_overlap_set_377_v1.json tests/test_a7c_overlap_set_compositor.py
git commit -m "方法：冻结A7c R1.2-B软重叠集合契约"
```

### Task 2: Implement The Continuous Overlap Set

**Files:**
- Create: `utils/a7c_overlap_set_compositor.py`
- Modify: `tests/test_a7c_overlap_set_compositor.py`

- [ ] **Step 1: Write failing graph tests**

```python
def test_overlap_adjacency_masks_self_and_invisible_nodes():
    adjacency = dense_overlap_adjacency(
        projected_xy=torch.tensor([[[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]]]),
        log_depth=torch.zeros(1, 3),
        visibility=torch.tensor([[1.0, 1.0, 0.0]]),
        spatial_scale=0.03,
        depth_scale=0.04,
        edge_log_weight_minimum=-20.0,
    )
    torch.testing.assert_close(torch.diagonal(adjacency, dim1=1, dim2=2), torch.zeros(1, 3))
    torch.testing.assert_close(adjacency[:, :, 2], torch.zeros(1, 3))
    torch.testing.assert_close(adjacency[0, 0].sum(), torch.tensor(1.0))
    torch.testing.assert_close(adjacency[0, 2], torch.zeros(3))


def test_overlap_set_is_permutation_equivariant():
    model = DenseOverlapSetCompositor(30, 32, 32, minimum_gate=0.9, initial_gate=0.999)
    original = model(features, projected_xy, log_depth, visibility, 0.03, 0.04, -20.0)
    permuted = model(features[:, permutation], projected_xy[:, permutation], log_depth[:, permutation], visibility[:, permutation], 0.03, 0.04, -20.0)
    torch.testing.assert_close(permuted, original[:, permutation])
```

Add separate tests requiring finite bounded output for an all-invisible set and
initial output equal to `0.999` within `1e-6`.

- [ ] **Step 2: Run tests and verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_overlap_set_compositor.py -k 'adjacency or permutation or invisible or initial' -q`

Expected: FAIL because the overlap-set module does not exist.

- [ ] **Step 3: Implement the graph and model**

Implement these public APIs:

```python
def dense_overlap_adjacency(*, projected_xy, log_depth, visibility,
                            spatial_scale, depth_scale,
                            edge_log_weight_minimum, epsilon=1e-8):
    delta_xy = projected_xy[:, :, None, :] - projected_xy[:, None, :, :]
    delta_depth = log_depth[:, :, None] - log_depth[:, None, :]
    log_weight = -0.5 * torch.sum(delta_xy.square(), dim=-1) / spatial_scale**2
    log_weight -= 0.5 * delta_depth.square() / depth_scale**2
    log_weight = torch.clamp(log_weight, min=edge_log_weight_minimum, max=0.0)
    weight = torch.exp(log_weight)
    visible_pair = visibility[:, :, None] * visibility[:, None, :]
    identity = torch.eye(weight.shape[1], dtype=torch.bool, device=weight.device)
    weight = weight * visible_pair * (~identity).unsqueeze(0)
    denominator = weight.sum(dim=-1, keepdim=True)
    return torch.where(denominator > epsilon, weight / denominator.clamp_min(epsilon), torch.zeros_like(weight))


class DenseOverlapSetCompositor(torch.nn.Module):
    # node encoder 30->32->32; gate input 128->32->1
    # concatenate h, neighbor message, h-message, visible global context
    # zero-initialize final weight and bias for initial_gate
```

Validate finite inputs, positive scales, matching `[samples, carriers]` shapes,
and valid gate bounds. Do not accept labels or IDs in either public signature.

- [ ] **Step 4: Run focused and existing projection tests**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_overlap_set_compositor.py tests/test_a7c_quotient_compositor.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/a7c_overlap_set_compositor.py tests/test_a7c_overlap_set_compositor.py
git commit -m "方法：实现A7c连续软重叠集合编码器"
```

### Task 3: Implement Fold Training With The New Predictor

**Files:**
- Create: `tools/train_a7c_r1_2b_overlap_set_compositor.py`
- Modify: `tests/test_a7c_overlap_set_compositor.py`

- [ ] **Step 1: Write failing trainer tests**

```python
def test_r1_2b_trainer_has_no_teacher_gate_objective():
    source = TRAINER.read_text(encoding="utf-8")
    assert 'teacher["gates"]' not in source
    assert "teacher_gate_loss" not in source
    assert "DenseOverlapSetCompositor" in source


def test_r1_2b_cpu_training_is_deterministic_and_bounded(tmp_path):
    kwargs = dict(
        train_mask=train_mask,
        features=features,
        projected_xy=projected_xy,
        log_depth=log_depth,
        visibility=visibility,
        runtime_mass=runtime_mass,
        a5_weight=a5_weight,
        objective_streams=objective_streams,
        guard_streams=guard_streams,
        camera_index=camera_index,
        frame_index=frame_index,
        block_ids=block_ids,
        contract=contract,
        output_dir=tmp_path,
        device="cpu",
    )
    first = train_one(name="first", **kwargs)
    second = train_one(name="second", **kwargs)
    assert first["final_loss"] <= first["initial_loss"]
    with np.load(tmp_path / "first/predictions.npz") as a, np.load(tmp_path / "second/predictions.npz") as b:
        np.testing.assert_allclose(a["projected_gates"], b["projected_gates"], atol=0, rtol=0)
        assert a["projected_gates"].min() >= 0.9
        assert a["projected_gates"].max() <= 1.0
```

The synthetic arrays contain two carriers and four adjacent samples and exercise
the real model, projection, and renderer objective on CPU.

- [ ] **Step 2: Run tests and verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_overlap_set_compositor.py -k 'trainer or training' -q`

Expected: FAIL because the trainer does not exist.

- [ ] **Step 3: Implement the trainer**

Reuse these frozen helpers without altering them:

```python
from tools.train_a7c_r1_2a_quotient_compositor import (
    _build_streams, _load_probe, _load_teacher_manifest, _torch_streams,
    sample_block_ids, verify_source_file,
)
from utils.a7c_quotient_compositor import (
    contiguous_training_segments, project_joint_target_budget,
    renderer_sequence_objective, runtime_target_mass,
)
```

Implement `train_one` with fold-only normalization and
`DenseOverlapSetCompositor`. Extract normalized F1 features for the node encoder,
but pass the raw F1 projected coordinates, log depth, and visibility to the graph.
Train six held-block folds and one final model, writing the same artifact schema
as R1.2-A plus predictor and graph-scale provenance.

- [ ] **Step 4: Verify trainer tests and CLI**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_overlap_set_compositor.py -q`

Run: `/opt/miniconda3/envs/anim/bin/python tools/train_a7c_r1_2b_overlap_set_compositor.py --help`

Expected: PASS and CLI exit 0.

- [ ] **Step 5: Commit**

```bash
git add tools/train_a7c_r1_2b_overlap_set_compositor.py tests/test_a7c_overlap_set_compositor.py
git commit -m "实验：实现A7c R1.2-B软集合训练"
```

### Task 4: Add The Resumable Held-Block Runner

**Files:**
- Create: `tools/run_a7c_r1_2b_overlap_set_377.sh`
- Modify: `tests/test_a7c_overlap_set_compositor.py`

- [ ] **Step 1: Write the failing runner test**

```python
def test_r1_2b_runner_is_restart_safe_and_camera_isolated():
    source = RUNNER.read_text(encoding="utf-8")
    for camera in ("c17", "c18", "c19", "c20", "c21", "c22", "c23"):
        assert re.search(rf"\b{camera}\b", source) is None
    assert "training/final/model.pt" in source
    assert "train_a7c_r1_2b_overlap_set_compositor.py" in source
    assert "audit_a7c_r1_2a_quotient_compositor.py" in source
    assert source.index("trap - ERR") < source.index("audit_a7c_r1_2a_quotient_compositor.py")
```

- [ ] **Step 2: Run test and verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_overlap_set_compositor.py -k runner -q`

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement the runner**

Copy the verified R1.2-A orchestration pattern and change only the contract,
trainer, and output root. Verify all source hashes before training; skip training
only when all 21 fold/final artifacts and `training_summary.json` exist. Reuse the
same auditor with the R1.2-B contract. Disarm the ERR trap before accepting audit
exit 2. Write exactly one root terminal marker.

- [ ] **Step 4: Run complete focused verification**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_overlap_set_compositor.py tests/test_a7c_quotient_compositor.py tests/test_a7c_renderer_compositor.py tests/test_a7c_ray_context_probe.py tests/test_a7c_oracle_capacity.py -q`

Run: `bash -n tools/run_a7c_r1_2b_overlap_set_377.sh`

Run: `/opt/miniconda3/envs/ictrl/bin/python -m py_compile utils/a7c_overlap_set_compositor.py tools/train_a7c_r1_2b_overlap_set_compositor.py`

Expected: all commands exit 0.

- [ ] **Step 5: Commit**

```bash
git add tools/run_a7c_r1_2b_overlap_set_377.sh tests/test_a7c_overlap_set_compositor.py
git commit -m "实验：门控A7c R1.2-B held-block canary"
```

### Task 5: Launch, Monitor, Verify, And Record

**Files:**
- Modify: `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`
- Output: `exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/`

- [ ] **Step 1: Verify formal preconditions**

Run the complete focused suite, Python compilation, shell syntax, `git diff
--check`, source SHA-256 checks, and `nvidia-smi`. Stop before launch on any
mismatch.

- [ ] **Step 2: Launch in an independent session**

```bash
mkdir -p exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1
setsid -f bash -c 'echo $$ > exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/runner.pid; exec bash tools/run_a7c_r1_2b_overlap_set_377.sh exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1' > exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/runner.log 2>&1 < /dev/null
```

- [ ] **Step 3: Measure and report the completion estimate**

Confirm the runner has `PPID=1` and its own session ID. Wait for fold 0 to write
all three artifacts, calculate elapsed wall time from `started_utc.txt` and file
mtime, multiply by seven, add a 15% audit margin, and report the resulting
Beijing completion time.

- [ ] **Step 4: Monitor to an explicit terminal state**

Poll until exactly one of `.done`, `.rejected`, or `.failed` exists. Process exit
without a marker is not completion. On reject, do not open c17-c23.

- [ ] **Step 5: Independently verify and document the result**

Recompute the 24-record means, positive fractions, q10, worst blocks, target
minimum, soft-IoU maximum, and gate-jump maximum from `records`. Verify seven
artifact triplets, source/model/audit hashes, marker mutual exclusion, and absence
of c17-c23 outputs. Append timing, metrics, decision, hashes, and the next allowed
route to the existing A7 handoff document without overwriting or committing its
unrelated user changes.
