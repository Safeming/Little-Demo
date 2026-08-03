# A7c R1.2-A Quotient-Space Compositor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and held-block audit a CoreView377 compositor that replaces non-identifiable oracle-gate regression with renderer-output losses and a runtime-safe joint target-budget projection.

**Architecture:** Reuse the frozen R1.1 probe, A5 bank, R0 manifest, and v5.3 renderer contribution evidence. A small F1 MLP emits independent raw gates; a deterministic projection couples all carrier damping through a runtime semantic target-mass budget, and training optimizes the final outer/boundary temporal metrics plus frozen guards.

**Tech Stack:** Python, NumPy, PyTorch, pytest, existing A7c artifact/fingerprint helpers, Bash runner.

---

### Task 1: Freeze The R1.2-A Contract

**Files:**
- Create: `configs/semantic/a7c_r1_2a_quotient_compositor_377_v1.json`
- Create: `tests/test_a7c_quotient_compositor.py`

- [ ] **Step 1: Write the failing contract test**

```python
ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/semantic/a7c_r1_2a_quotient_compositor_377_v1.json"

def test_contract_freezes_scope_inputs_objective_and_gates():
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["status"] == "frozen"
    assert payload["fit_cameras"] == ["c01", "c05", "c09", "c13"]
    assert payload["audit_cameras"] == ["c17", "c18", "c19", "c20"]
    assert payload["forbidden_cameras"] == ["c21", "c22", "c23"]
    assert payload["score_feature_group"] == "F1"
    assert payload["teacher_gate_loss_weight"] == 0.0
    assert payload["proxy_target_response"] == 0.995
    assert payload["minimum_target_response"] == 0.99
    assert payload["maximum_adjacent_gate_change"] == 0.02
    assert payload["paper_test_eligible"] is False
```

- [ ] **Step 2: Run the test and verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_quotient_compositor.py::test_contract_freezes_scope_inputs_objective_and_gates -q`

Expected: FAIL because the contract does not exist.

- [ ] **Step 3: Add the frozen JSON contract**

Use these exact method fields:

```json
{
  "schema_version": 1,
  "experiment_id": "a7c_r1_2a_quotient_compositor_377_v1",
  "status": "frozen",
  "subject": "377",
  "fit_cameras": ["c01", "c05", "c09", "c13"],
  "audit_cameras": ["c17", "c18", "c19", "c20"],
  "forbidden_cameras": ["c21", "c22", "c23"],
  "frame_start": 0,
  "frame_end": 570,
  "frame_stride": 5,
  "temporal_block_count": 6,
  "part": "lower",
  "score_feature_group": "F1",
  "hidden_dimensions": [64, 32],
  "minimum_gate": 0.9,
  "maximum_gate": 1.0,
  "initial_minimum_gate": 0.999,
  "selection_threshold": 0.2,
  "proxy_target_response": 0.995,
  "training_target_response": 0.995,
  "training_epochs": 400,
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "outer_loss_weight": 1.0,
  "boundary_loss_weight": 1.0,
  "target_hinge_weight": 100.0,
  "soft_iou_hinge_weight": 100.0,
  "gate_jump_hinge_weight": 20.0,
  "damping_regularizer_weight": 0.001,
  "teacher_gate_loss_weight": 0.0,
  "minimum_outer_gain": 0.005,
  "minimum_boundary_gain": 0.005,
  "minimum_positive_block_fraction": 0.9,
  "block_gain_quantile": 0.1,
  "minimum_block_gain_quantile": 0.0,
  "maximum_worst_block_regression": 0.005,
  "minimum_target_response": 0.99,
  "maximum_selection_soft_iou_drop": 0.005,
  "maximum_adjacent_gate_change": 0.02,
  "random_seed": 20260803,
  "source_r1_1_contract_sha256": "1abb5955042958950c4d197a39f907feef142f210ddf8d1e4b0d1b05f48d7f02",
  "source_probe_sha256": "643c541af20f732a9de2c4ac6c20ea804ac27be8ad6dad13b1ead5efb6f8b411",
  "source_teacher_sha256": "698f61e195a78849c72be14b8cf9073f281b94124d804013988e7bf605304aa8",
  "source_evidence_sha256": "8b655f48fad664ba308f51d3291971382d7f9037fc7d69e38fca37907efd77f4",
  "source_a5_bank_sha256": "49ba86b05c4f87eaa8b98ef47822c7083a31fdf050a35bd8cf3a88843f8a45d3",
  "paper_test_eligible": false
}
```

Also freeze the current R1.1 contract/probe, R0 teacher, v5.3 evidence, and A5 bank fingerprints in the same JSON before formal training.

- [ ] **Step 4: Run the contract test and focused baseline**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_quotient_compositor.py::test_contract_freezes_scope_inputs_objective_and_gates tests/test_a7c_ray_context_probe.py tests/test_a7c_renderer_compositor.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/semantic/a7c_r1_2a_quotient_compositor_377_v1.json tests/test_a7c_quotient_compositor.py
git commit -m "方法：冻结A7c R1.2-A输出空间契约"
```

### Task 2: Implement Runtime-Safe Joint Projection

**Files:**
- Create: `utils/a7c_quotient_compositor.py`
- Modify: `tests/test_a7c_quotient_compositor.py`

- [ ] **Step 1: Write failing tests for proxy mass and joint budget**

```python
def test_runtime_target_mass_uses_only_runtime_probe_fields():
    mass = runtime_target_mass(
        alpha_transmittance_mass=torch.tensor([[2.0, 1.0]]),
        a5_weight=torch.tensor([0.8, 0.5]),
        semantic_support_mean=torch.tensor([[0.4, 0.0]]),
        alpha_mean=torch.tensor([[0.5, 0.0]]),
    )
    torch.testing.assert_close(mass, torch.tensor([[1.28, 0.0]]))

def test_joint_target_budget_scales_all_damping_and_preserves_topology():
    projected = project_joint_target_budget(
        raw_gates=torch.tensor([[0.9, 0.9]]),
        runtime_mass=torch.tensor([[1.0, 1.0]]),
        a5_weight=torch.tensor([0.21, 0.8]),
        proxy_target_response=0.995,
        selection_threshold=0.2,
        minimum_gate=0.9,
    )
    assert float((projected * torch.tensor([[1.0, 1.0]])).sum()) >= 1.99
    assert float(projected[0, 0] * 0.21) >= 0.2 - 1e-7
    assert torch.all(projected >= 0.9) and torch.all(projected <= 1.0)
```

Add a zero-runtime-mass test requiring finite unchanged raw gates and a schema
test rejecting any projection keyword containing `mask`, `target_contribution`,
`outer`, or `boundary`.

- [ ] **Step 2: Run the tests and verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_quotient_compositor.py -k 'runtime_target_mass or joint_target_budget or projection_schema' -q`

Expected: FAIL because the module and functions do not exist.

- [ ] **Step 3: Implement the projection API**

```python
def runtime_target_mass(*, alpha_transmittance_mass, a5_weight,
                        semantic_support_mean, alpha_mean, epsilon=1e-8):
    probability = torch.clamp(
        semantic_support_mean / torch.clamp(alpha_mean, min=epsilon), 0.0, 1.0
    )
    return torch.clamp(alpha_transmittance_mass, min=0.0) * a5_weight * probability

def project_joint_target_budget(*, raw_gates, runtime_mass, a5_weight,
                                proxy_target_response, selection_threshold,
                                minimum_gate, epsilon=1e-8):
    damping = 1.0 - raw_gates
    loss = torch.sum(runtime_mass * damping, dim=1, keepdim=True)
    budget = (1.0 - proxy_target_response) * torch.sum(runtime_mass, dim=1, keepdim=True)
    scale = torch.clamp(budget / torch.clamp(loss, min=epsilon), max=1.0)
    scale = torch.where(loss > epsilon, scale, torch.ones_like(scale))
    gates = 1.0 - scale * damping
    topology_floor = torch.clamp(selection_threshold / torch.clamp(a5_weight, min=epsilon), max=1.0)
    return torch.maximum(gates, topology_floor.unsqueeze(0)).clamp(minimum_gate, 1.0)
```

Keep the public signature closed and validate finite tensors and matching shapes.

- [ ] **Step 4: Run RED tests and all A7c utility tests**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_quotient_compositor.py tests/test_a7c_ray_context_probe.py tests/test_a7c_renderer_compositor.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/a7c_quotient_compositor.py tests/test_a7c_quotient_compositor.py
git commit -m "方法：实现A7c联合目标预算投影"
```

### Task 3: Implement Renderer-Level Sequence Losses

**Files:**
- Modify: `utils/a7c_quotient_compositor.py`
- Modify: `tests/test_a7c_quotient_compositor.py`

- [ ] **Step 1: Write failing segment and objective tests**

```python
def test_contiguous_training_segments_do_not_cross_held_gaps_or_cameras():
    segments = contiguous_training_segments(
        train_mask=np.array([1, 1, 0, 1, 1, 1, 1, 0], bool),
        camera_index=np.array([0, 0, 0, 0, 0, 1, 1, 1]),
        frame_index=np.array([0, 5, 10, 15, 20, 0, 5, 10]),
        frame_stride=5,
        block_ids=np.array([0, 0, 0, 1, 1, 0, 0, 0]),
    )
    assert [x.tolist() for x in segments] == [[0, 1], [3, 4], [5, 6]]

def test_renderer_objective_prefers_gate_that_reduces_outer_and_boundary_flicker():
    good = renderer_sequence_objective(gates=good_gates, **synthetic_streams)
    identity = renderer_sequence_objective(gates=torch.ones_like(good_gates), **synthetic_streams)
    assert good["loss"] < identity["loss"]
    assert good["outer_ratio"] < 1.0
    assert good["boundary_ratio"] < 1.0
```

Add tests that target, soft-IoU, and jump violations each produce a positive
hinge while an exactly compliant synthetic sequence produces zero hinge.

- [ ] **Step 2: Run the tests and verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_quotient_compositor.py -k 'contiguous_training_segments or renderer_objective' -q`

Expected: FAIL because the sequence APIs are missing.

- [ ] **Step 3: Implement differentiable contribution composition and losses**

Implement these exact public functions:

```python
def contiguous_training_segments(*, train_mask, camera_index, frame_index,
                                 frame_stride, block_ids):
    mask = np.asarray(train_mask, dtype=bool)
    cameras = np.asarray(camera_index)
    frames = np.asarray(frame_index)
    blocks = np.asarray(block_ids)
    segments, current = [], []
    for index in range(mask.size):
        continuous = bool(
            current
            and mask[index]
            and cameras[index] == cameras[current[-1]]
            and blocks[index] == blocks[current[-1]]
            and frames[index] - frames[current[-1]] == frame_stride
        )
        if current and not continuous:
            if len(current) > 1:
                segments.append(np.asarray(current, dtype=np.int64))
            current = []
        if mask[index]:
            current.append(index)
    if len(current) > 1:
        segments.append(np.asarray(current, dtype=np.int64))
    return segments

def compose_contribution(base, point, gates):
    return base - point.sum(dim=1) + (point * gates).sum(dim=1)

def torch_normalized_flicker(values, epsilon=1e-8):
    return torch.mean(torch.abs(values[1:] - values[:-1])) / torch.clamp(torch.abs(values.mean()), min=epsilon)

def renderer_sequence_objective(*, gates, segments, objective_streams,
                                guard_streams, contract):
    candidate = {
        role: {
            signal: compose_contribution(stream["base"], stream["point"], gates)
            for signal, stream in streams.items()
        }
        for role, streams in (
            ("objective", objective_streams), ("guard", guard_streams)
        )
    }
    outer_ratios, boundary_ratios, jump_terms = [], [], []
    for indices in segments:
        index = torch.as_tensor(indices, device=gates.device)
        for signal, destination in (
            ("outer", outer_ratios), ("boundary", boundary_ratios)
        ):
            base = objective_streams[signal]["base"][index]
            value = candidate["objective"][signal][index]
            destination.append(
                torch_normalized_flicker(value)
                / torch.clamp(torch_normalized_flicker(base), min=1e-8)
            )
        jump_terms.append(torch.abs(gates[index][1:] - gates[index][:-1]))
    outer_ratio = torch.stack(outer_ratios).mean()
    boundary_ratio = torch.stack(boundary_ratios).mean()
    base_target = guard_streams["target"]["base"]
    target_response = candidate["guard"]["target"] / torch.clamp(base_target, min=1e-8)
    base_iou = base_target / torch.clamp(
        base_target + guard_streams["outer"]["base"], min=1e-8
    )
    candidate_iou = candidate["guard"]["target"] / torch.clamp(
        candidate["guard"]["target"] + candidate["guard"]["outer"], min=1e-8
    )
    target_hinge = torch.relu(
        contract["training_target_response"] - target_response
    ).mean()
    soft_iou_hinge = torch.relu(
        base_iou - candidate_iou - contract["maximum_selection_soft_iou_drop"]
    ).mean()
    jumps = torch.cat([value.reshape(-1) for value in jump_terms])
    jump_hinge = torch.relu(
        jumps - contract["maximum_adjacent_gate_change"]
    ).mean()
    damping = torch.mean(torch.square(1.0 - gates))
    loss = (
        contract["outer_loss_weight"] * outer_ratio
        + contract["boundary_loss_weight"] * boundary_ratio
        + contract["target_hinge_weight"] * target_hinge
        + contract["soft_iou_hinge_weight"] * soft_iou_hinge
        + contract["gate_jump_hinge_weight"] * jump_hinge
        + contract["damping_regularizer_weight"] * damping
    )
    return {
        "loss": loss,
        "outer_ratio": outer_ratio,
        "boundary_ratio": boundary_ratio,
        "target_hinge": target_hinge,
        "soft_iou_hinge": soft_iou_hinge,
        "jump_hinge": jump_hinge,
        "damping_regularizer": damping,
    }
```

For each segment, compute outer and boundary candidate/base flicker ratios. Use
the true selection streams only inside target-response and soft-IoU loss terms.
Compute gate-jump hinge only for adjacent pairs inside registered segments.
Return the total loss and detached component scalars for artifact summaries.

- [ ] **Step 4: Run all quotient and A7c tests**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_quotient_compositor.py tests/test_a7c_renderer_compositor.py tests/test_a7c_ray_context_probe.py tests/test_a7c_oracle_capacity.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/a7c_quotient_compositor.py tests/test_a7c_quotient_compositor.py
git commit -m "方法：实现A7c渲染输出空间时序目标"
```

### Task 4: Implement Fold Training Without Teacher-Gate Loss

**Files:**
- Create: `tools/train_a7c_r1_2a_quotient_compositor.py`
- Modify: `tests/test_a7c_quotient_compositor.py`

- [ ] **Step 1: Write failing trainer contract tests**

Test parser requirements, source fingerprint rejection, carrier/sample alignment,
F1-only score input, and a CPU synthetic run. The synthetic run must assert:

```python
assert summary["teacher_gate_loss_weight"] == 0.0
assert summary["training_sample_count"] > 0
assert summary["final_loss"] <= summary["initial_loss"]
assert predictions["raw_gates"].shape == predictions["projected_gates"].shape
assert np.max(predictions["projected_gates"]) <= 1.0
assert np.min(predictions["projected_gates"]) >= 0.9
```

- [ ] **Step 2: Run trainer tests and verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_quotient_compositor.py -k trainer -q`

Expected: FAIL because the trainer does not exist.

- [ ] **Step 3: Implement trainer**

The CLI is:

```text
--contract PATH --probe PATH --evidence PATH --a5-bank PATH
--teacher PATH --output-dir PATH --device cuda
```

Load and fingerprint all artifacts, use the teacher artifact only for
`carrier_ids/camera_index/frame_index`, select F1 features, derive runtime target
mass from probe fields, construct objective/guard streams exactly as the existing
auditor does, and train six folds plus one final model. Save for each model:

```text
model.pt
predictions.npz     # raw_gates, projected_gates, train_mask
summary.json        # loss components, gate range, source fingerprints
```

Set the random seed before model construction. Do not select epochs or models
using held-block metrics.

- [ ] **Step 4: Run trainer tests and focused suite**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_quotient_compositor.py tests/test_a7c_renderer_compositor.py tests/test_a7c_ray_context_probe.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/train_a7c_r1_2a_quotient_compositor.py tests/test_a7c_quotient_compositor.py
git commit -m "实验：实现A7c R1.2-A输出空间训练"
```

### Task 5: Implement Held-Block Audit And Resumable Runner

**Files:**
- Create: `tools/audit_a7c_r1_2a_quotient_compositor.py`
- Create: `tools/run_a7c_r1_2a_quotient_377.sh`
- Modify: `tests/test_a7c_quotient_compositor.py`

- [ ] **Step 1: Write failing audit and runner tests**

Use synthetic 24-record summaries to assert every promotion guard, R1.1-F1
comparison, `.held_block_passed`/`.rejected` marker behavior, and exit code 2 on
rejection. Assert the runner contains no c17-c23 command before the held-block
pass marker and validates all source fingerprints before training.

- [ ] **Step 2: Run audit tests and verify RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_quotient_compositor.py -k 'audit or runner' -q`

Expected: FAIL because auditor and runner are missing.

- [ ] **Step 3: Implement auditor and runner**

The auditor must load `projected_gates`, compute the same 24 camera-block records
as R1.1, and write:

```text
audit/held_block_summary.json
audit/.held_block_passed  # only on pass
audit/.rejected           # only on reject
```

The runner must be restart-safe: skip training only when all seven model
artifacts and their summaries exist, then run the audit. Rejection is an
expected terminal state and writes root `.rejected`; unexpected command failure
writes root `.failed`. It must not open c17-c20 in this first formal execution.

- [ ] **Step 4: Run full focused suite and shell syntax check**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_quotient_compositor.py tests/test_a7c_renderer_compositor.py tests/test_a7c_ray_context_probe.py tests/test_a7c_oracle_capacity.py -q`

Run: `bash -n tools/run_a7c_r1_2a_quotient_377.sh`

Expected: all tests PASS and shell syntax exits 0.

- [ ] **Step 5: Commit**

```bash
git add tools/audit_a7c_r1_2a_quotient_compositor.py tools/run_a7c_r1_2a_quotient_377.sh tests/test_a7c_quotient_compositor.py
git commit -m "实验：门控A7c R1.2-A held-block canary"
```

### Task 6: Launch Formal Training And Record The Result

**Files:**
- Modify: `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`
- Output: `exp/acceptdata/a7c_r1_2a_quotient_compositor_377_v1/`

- [ ] **Step 1: Verify the frozen contract and repository state**

Run SHA-256 checks for the contract, probe, evidence, A5 bank, and teacher. Run
the full focused suite and `python -m py_compile` for all new Python files.

- [ ] **Step 2: Start the detached formal runner**

```bash
nohup bash tools/run_a7c_r1_2a_quotient_377.sh \
  exp/acceptdata/a7c_r1_2a_quotient_compositor_377_v1 \
  > exp/acceptdata/a7c_r1_2a_quotient_compositor_377_v1/runner.log 2>&1 &
```

Record PID, start UTC/BJT, GPU, observed per-fold time, and a BJT completion
estimate based on the first completed fold rather than a guessed throughput.

- [ ] **Step 3: Monitor to terminal state**

Poll process state, runner log, seven model artifacts, audit summary, and exactly
one of `.done`, `.rejected`, or `.failed`. Do not report success from process
exit alone.

- [ ] **Step 4: Verify formal artifacts and metrics**

Run the focused suite again, hash contract/model/audit outputs, confirm no
c17-c23 result exists, and independently recompute the held-block aggregation.

- [ ] **Step 5: Append the formal record without overwriting user edits**

Add start/end BJT, actual runtime, contract/model/audit hashes, 24-record
aggregate metrics, pass/fail reasons, and next allowed route to the A7 handoff
document. Do not commit this shared document when unrelated user modifications
remain in it.
