# A7c R1.4-VP Oracle-Distilled View-Pose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, audit, and launch the frozen CoreView377 six-fold R1.4-VP canary that distills fit-only renderer-oracle gates into a runtime-visible offline view-pose residual compositor and compares it with a fixed `k=4` nearest-neighbor baseline.

**Architecture:** A frozen contract pins all upstream evidence, R1.2-B fold predictions, pose files, and thresholds. A dedicated lexicographic teacher solver generates only the five fit blocks of each fold; a runtime-only data module constructs F3, rotation-6D pose, and overlap context; a small bidirectional residual model and a fixed nearest-neighbor baseline produce held-block gates and pass them through the unchanged R1.3-P hard projection. A separate auditor opens renderer contributions only after all six learned and baseline artifacts are frozen, recomputes the formal metrics, and owns the three terminal verdicts.

**Tech Stack:** Python 3.10, NumPy, SciPy 1.13 HiGHS LP, PyTorch, pytest, Bash.

---

## File Map

- Create `configs/semantic/a7c_r1_4vp_oracle_distilled_view_pose_377_v1.json`: frozen experiment, source, model, training, baseline, audit, and eligibility contract.
- Create `utils/a7c_oracle_distillation.py`: fit-only capacity search and three-stage lexicographic anchored teacher solver.
- Create `utils/a7c_view_pose_compositor.py`: pose loading/rotation-6D conversion, fit-only normalization, segment packing, view-pose model, loss, and nearest-neighbor interpolation.
- Create `tools/build_a7c_r1_4vp_fit_teachers.py`: validate sources and generate 120 fit-only teacher segments with held values left NaN.
- Create `tools/train_a7c_r1_4vp_view_pose.py`: train six fixed models, generate model and `k=4` baseline predictions, hard-project all segments, and freeze artifacts before audit.
- Create `tools/audit_a7c_r1_4vp_view_pose.py`: independently validate artifacts and recompute 24 held renderer records, visibility response, baselines, per-camera gates, and verdict.
- Create `tools/run_a7c_r1_4vp_view_pose_377.sh`: restart-safe, detached-compatible orchestration with timestamps and exactly one terminal marker.
- Create `tests/test_a7c_oracle_distillation.py`: anchored LP, capacity, isolation, determinism, and certificate tests.
- Create `tests/test_a7c_view_pose_compositor.py`: runtime schema, model, training, nearest-neighbor, audit, and runner tests.
- Modify `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`: append only the real launch/result block after execution, preserving unrelated user changes.

### Task 1: Freeze The R1.4-VP Contract

**Files:**
- Create: `configs/semantic/a7c_r1_4vp_oracle_distilled_view_pose_377_v1.json`
- Create: `tests/test_a7c_view_pose_compositor.py`

- [ ] **Step 1: Write the failing contract test**

```python
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/semantic/a7c_r1_4vp_oracle_distilled_view_pose_377_v1.json"


def test_r1_4vp_contract_freezes_model_isolation_and_promotion():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "a7c_r1_4vp_oracle_distilled_view_pose_377_v1"
    assert contract["status"] == "frozen"
    assert contract["fit_cameras"] == ["c01", "c05", "c09", "c13"]
    assert contract["audit_cameras"] == []
    assert contract["forbidden_cameras"] == [
        "c17", "c18", "c19", "c20", "c21", "c22", "c23"
    ]
    assert contract["temporal_block_count"] == 6
    assert contract["fit_teacher_segment_count"] == 120
    assert contract["held_audit_record_count"] == 24
    assert contract["pose_body_joint_indices"] == [0, 1, 3, 4, 6, 7]
    assert contract["pose_dimension"] == 36
    assert contract["view_feature_group"] == "F3"
    assert contract["view_embedding_dimension"] == 16
    assert contract["pose_embedding_dimension"] == 16
    assert contract["gru_hidden_dimension"] == 16
    assert contract["maximum_parameter_count"] == 50000
    assert contract["residual_gate_scale"] == 0.1
    assert contract["training_epochs"] == 400
    assert contract["random_seed"] == 20260805
    assert contract["nearest_neighbor_k"] == 4
    assert contract["minimum_outer_gain"] == 0.005
    assert contract["minimum_boundary_gain"] == 0.005
    assert contract["maximum_visibility_response_ratio"] == 1.0
    assert contract["maximum_selection_soft_iou_drop"] == 0.005
    assert contract["maximum_projection_gate_jump"] == 0.015
    assert contract["maximum_adjacent_gate_change"] == 0.02
    assert contract["processed_parts"] == ["lower"]
    assert contract["frozen_parts"] == ["hair", "face", "upper", "shoes", "skin"]
    assert contract["min_pair_support"] == 8
    assert contract["minimum_evidence_support_coverage"] == 0.8
    assert contract["deployment_eligible"] is False
    assert contract["teacher_eligible"] is False
    assert contract["paper_test_eligible"] is False
```

- [ ] **Step 2: Run the test and verify the missing contract fails**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_view_pose_compositor.py::test_r1_4vp_contract_freezes_model_isolation_and_promotion -v`

Expected: `FAIL` with `FileNotFoundError` for the R1.4-VP contract.

- [ ] **Step 3: Create the frozen contract**

Create the JSON with these exact registered values and with the six R1.2-B prediction paths/hashes listed below in fold order:

```json
{
  "schema_version": 1,
  "experiment_id": "a7c_r1_4vp_oracle_distilled_view_pose_377_v1",
  "status": "frozen",
  "subject": "377",
  "fit_cameras": ["c01", "c05", "c09", "c13"],
  "audit_cameras": [],
  "forbidden_cameras": ["c17", "c18", "c19", "c20", "c21", "c22", "c23"],
  "frame_start": 0,
  "frame_end": 570,
  "frame_stride": 5,
  "temporal_block_count": 6,
  "fit_teacher_segment_count": 120,
  "held_audit_record_count": 24,
  "part": "lower",
  "processed_parts": ["lower"],
  "frozen_parts": ["hair", "face", "upper", "shoes", "skin"],
  "min_pair_support": 8,
  "minimum_evidence_support_coverage": 0.8,
  "maximum_weight_above_a5": 0.0,
  "offline_bidirectional": true,
  "view_feature_group": "F3",
  "pose_body_joint_indices": [0, 1, 3, 4, 6, 7],
  "pose_dimension": 36,
  "view_embedding_dimension": 16,
  "pose_embedding_dimension": 16,
  "gru_hidden_dimension": 16,
  "maximum_parameter_count": 50000,
  "residual_gate_scale": 0.1,
  "spatial_scale": 0.03,
  "depth_scale": 0.04,
  "edge_log_weight_minimum": -20.0,
  "minimum_gate": 0.9,
  "maximum_gate": 1.0,
  "selection_threshold": 0.2,
  "proxy_target_response": 0.995,
  "maximum_projection_gate_jump": 0.015,
  "teacher_minimum_outer_gain": 0.005,
  "teacher_boundary_margin": 0.00002,
  "oracle_bisection_tolerance": 0.00001,
  "lexicographic_tolerance": 1e-9,
  "solver": "highs",
  "solver_primal_tolerance": 1e-9,
  "solver_residual_tolerance": 1e-7,
  "training_epochs": 400,
  "random_seed": 20260805,
  "optimizer": "AdamW",
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "gradient_clip_norm": 1.0,
  "gate_huber_delta": 0.01,
  "temporal_huber_delta": 0.005,
  "temporal_loss_weight": 0.25,
  "residual_loss_weight": 0.001,
  "checkpoint_selection": "final_epoch_only",
  "nearest_neighbor_k": 4,
  "minimum_outer_gain": 0.005,
  "minimum_boundary_gain": 0.005,
  "minimum_positive_block_fraction": 0.9,
  "block_gain_quantile": 0.1,
  "minimum_block_gain_quantile": 0.0,
  "maximum_worst_block_regression": 0.005,
  "minimum_target_response": 0.99,
  "maximum_visibility_response_ratio": 1.0,
  "maximum_selection_soft_iou_drop": 0.005,
  "maximum_adjacent_gate_change": 0.02,
  "r1_1_f1_outer_gain": -0.00012761059760764496,
  "r1_1_f1_boundary_gain": 0.023481874880317264,
  "r1_2b_outer_gain": 0.005196372744170267,
  "r1_2b_boundary_gain": 0.002866365549963367,
  "comparison_tolerance": 1e-9,
  "source_design": "docs/superpowers/specs/2026-08-05-a7c-r1-4vp-oracle-distilled-view-pose-design.md",
  "source_design_sha256": "596ad6ada3c7676c502c3e6fc67b6c4c16e852bc107b837fa0b0d866f1fd6543",
  "source_r1_3g_contract": "configs/semantic/a7c_r1_3g_exact_aggregate_oracle_377_v1.json",
  "source_r1_3g_contract_sha256": "0ed0d588ab4a89abfa50d3213a84dc4e055ecd2d800c1bfc7bc154d3bf927bbb",
  "source_r1_3g_records": "exp/acceptdata/a7c_r1_3g_exact_aggregate_oracle_377_v1/records.json",
  "source_r1_3g_records_sha256": "97b59b5ab0b9f0b0c473748f9beb9af184de9de7acccc13dc1c96794f9340594",
  "source_r1_3g_audit": "exp/acceptdata/a7c_r1_3g_exact_aggregate_oracle_377_v1/audit/held_block_summary.json",
  "source_r1_3g_audit_sha256": "69426279d5bdbc44c5b8f9a353e4175eafa27e6c7b9f22811316586f07db615e",
  "source_r1_3g_summary": "exp/acceptdata/a7c_r1_3g_exact_aggregate_oracle_377_v1/summary.json",
  "source_r1_3g_summary_sha256": "d84d345be3e15b2a833ea75d03f40fcd0473fe2da567ce0c5ee4b5d58422b830",
  "source_r1_3p_contract": "configs/semantic/a7c_r1_3p_temporal_joint_projection_377_v1.json",
  "source_r1_3p_contract_sha256": "a62d99f65d1358d2b985db3c5dec5221396a7fb1c8cbf287abc8943788f4c61c",
  "source_r1_2b_contract": "configs/semantic/a7c_r1_2b_dense_overlap_set_377_v1.json",
  "source_r1_2b_contract_sha256": "e2825c1d59e96ff2ea6124bfa1defafb62c73c04a64519c889e909be9ef2f9b5",
  "source_r1_2b_training_summary": "exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/training/training_summary.json",
  "source_r1_2b_training_summary_sha256": "8a604cd5df7407b8b559adcea11304c93de2d9272c0bb8f1f60ca8b4f8efc46d",
  "source_r1_2b_audit": "exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/audit/held_block_summary.json",
  "source_r1_2b_audit_sha256": "b0aa6d3420ecb4a2e23874efdbafad59e90e6f8e2c0d485aca60b8c7130e6f1d",
  "source_r1_2b_predictions": [
    "exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/training/fold_0/predictions.npz",
    "exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/training/fold_1/predictions.npz",
    "exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/training/fold_2/predictions.npz",
    "exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/training/fold_3/predictions.npz",
    "exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/training/fold_4/predictions.npz",
    "exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/training/fold_5/predictions.npz"
  ],
  "source_r1_2b_prediction_sha256": [
    "5e53226483194c26ec46e7da08602ee0b72818076d72e3bcafb6190328516dee",
    "95a56b28b50f538f0e7128954700233904fbb5eaf8762fa242e699284e3f4300",
    "87e5db5244e1d6bff8342cd2e84b33aaeb4901e0669e4e041c0adc90c36a26e8",
    "160e54f8e1b0e45d6fef2bcc800817278cd0dc3293d82db701436f4fc8fa7758",
    "d348a458812105e356855909df06893edb404e484944ffbd21a0eeba69a7a25d",
    "8c1ef0e5c5fe9b4001e2829c7da536811b6de4cef1761ef878b8adc141ff357c"
  ],
  "source_r1_1_contract": "configs/semantic/a7c_r1_1_transmittance_ray_context_377_v1.json",
  "source_r1_1_contract_sha256": "1abb5955042958950c4d197a39f907feef142f210ddf8d1e4b0d1b05f48d7f02",
  "source_probe": "exp/acceptdata/a7c_r1_1_transmittance_ray_context_377_v1/probe/probe.npz",
  "source_probe_sha256": "643c541af20f732a9de2c4ac6c20ea804ac27be8ad6dad13b1ead5efb6f8b411",
  "source_teacher": "exp/acceptdata/a7c_carrier_compositor_canary_377_v1/teacher/teacher.npz",
  "source_teacher_sha256": "698f61e195a78849c72be14b8cf9073f281b94124d804013988e7bf605304aa8",
  "source_evidence": "exp/acceptdata/a7_dual_evidence_v5_3_canary_377/evidence/377/evidence.npz",
  "source_evidence_sha256": "8b655f48fad664ba308f51d3291971382d7f9037fc7d69e38fca37907efd77f4",
  "source_a5_bank": "exp/acceptdata/frozen_a5_five_subject_main_20260723/CoreView_377/banks/footprint_evidence_target/part_label_bank.npz",
  "source_a5_bank_sha256": "49ba86b05c4f87eaa8b98ef47822c7083a31fdf050a35bd8cf3a88843f8a45d3",
  "source_pose_model_dir": "data/ZJUMoCap/CoreView_377/models",
  "source_pose_manifest_sha256": "5d138f7f06ffaccb6b9a59d538028f0f298f0c538ea6845a49e9e6c2eda6f116",
  "preserve_a5_selection_topology": true,
  "retrain_avatar": false,
  "deployment_eligible": false,
  "teacher_eligible": false,
  "paper_test_eligible": false
}
```

The pose manifest hash is recomputed as SHA256 of the ordered `sha256sum`
output for frames `0..565` at stride 5.

- [ ] **Step 4: Run the contract test**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_view_pose_compositor.py::test_r1_4vp_contract_freezes_model_isolation_and_promotion -v`

Expected: `1 passed`.

- [ ] **Step 5: Commit the contract and test**

```bash
git add configs/semantic/a7c_r1_4vp_oracle_distilled_view_pose_377_v1.json tests/test_a7c_view_pose_compositor.py
git commit -m "test: freeze R1.4-VP canary contract"
```

### Task 2: Build Runtime-Only Pose, Features, Segments, And Model

**Files:**
- Create: `utils/a7c_view_pose_compositor.py`
- Modify: `tests/test_a7c_view_pose_compositor.py`

- [ ] **Step 1: Add failing runtime-schema tests**

```python
import inspect
import numpy as np
import torch


def test_axis_angle_to_rotation_6d_is_continuous_and_36d():
    from utils.a7c_view_pose_compositor import axis_angle_pose_to_rotation_6d

    pose = np.zeros((3, 6, 3), dtype=np.float64)
    pose[1, 0, 0] = 1e-5
    output = axis_angle_pose_to_rotation_6d(pose)
    assert output.shape == (3, 36)
    assert np.isfinite(output).all()
    assert np.linalg.norm(output[1] - output[0]) < 1e-4


def test_fit_normalization_never_reads_held_samples():
    from utils.a7c_view_pose_compositor import fit_normalization

    values = np.array([[0.0], [2.0], [1000.0]])
    stats = fit_normalization(values, np.array([True, True, False]))
    np.testing.assert_allclose(stats["mean"], [1.0])
    np.testing.assert_allclose(stats["scale"], [1.0])


def test_model_signature_forbids_renderer_labels_and_ids():
    from utils.a7c_view_pose_compositor import ViewPoseResidualCompositor

    names = set(inspect.signature(ViewPoseResidualCompositor.forward).parameters)
    assert not names & {
        "camera_id", "camera_index", "frame_id", "frame_index",
        "subject_id", "gaussian_id", "image_name", "held_block_identity",
        "target", "outer", "boundary", "teacher_gates", "evidence"
    }


def test_segment_packing_sorts_manifest_and_never_crosses_boundary():
    from utils.a7c_view_pose_compositor import pack_camera_block_segments

    camera = np.array([1, 0, 0, 1])
    block = np.array([0, 0, 0, 0])
    frame = np.array([5, 5, 0, 0])
    segments = pack_camera_block_segments(camera, block, frame, frame_stride=5)
    assert [row.tolist() for row in segments] == [[2, 1], [3, 0]]
```

- [ ] **Step 2: Add failing model tests**

```python
def test_view_pose_model_is_bounded_small_interactive_and_deterministic():
    from utils.a7c_view_pose_compositor import ViewPoseResidualCompositor

    torch.manual_seed(1)
    model = ViewPoseResidualCompositor(
        view_dimension=30,
        view_embedding_dimension=16,
        pose_dimension=36,
        pose_embedding_dimension=16,
        gru_hidden_dimension=16,
        residual_gate_scale=0.1,
        minimum_gate=0.9,
        maximum_gate=1.0,
    )
    assert sum(value.numel() for value in model.parameters()) <= 50000
    assert torch.count_nonzero(model.residual_head.weight) == 0
    view = torch.randn(8, 3, 30)
    pose = torch.randn(8, 36)
    base = torch.full((8, 3), 0.97)
    adjacency = torch.eye(3).expand(8, 3, 3)
    visibility = torch.ones(8, 3)
    first = model(view, pose, adjacency, visibility, base)
    second = model(view, pose, adjacency, visibility, base)
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first, base, atol=1e-7, rtol=0.0)
    assert torch.all(first >= 0.9) and torch.all(first <= 1.0)


def test_interaction_changes_when_either_runtime_branch_changes():
    from utils.a7c_view_pose_compositor import ViewPoseResidualCompositor

    model = ViewPoseResidualCompositor(4, 16, 36, 16, 16, 0.1, 0.9, 1.0)
    with torch.no_grad():
        model.residual_head.weight.fill_(0.05)
    view = torch.ones(6, 2, 4)
    pose = torch.ones(6, 36)
    adjacency = torch.zeros(6, 2, 2)
    visibility = torch.ones(6, 2)
    base = torch.full((6, 2), 0.97)
    both = model(view, pose, adjacency, visibility, base)
    zero_view = model(torch.zeros_like(view), pose, adjacency, visibility, base)
    zero_pose = model(view, torch.zeros_like(pose), adjacency, visibility, base)
    assert not torch.allclose(both, zero_view)
    assert not torch.allclose(both, zero_pose)
```

- [ ] **Step 3: Run the focused tests and verify import failure**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_view_pose_compositor.py -k 'axis_angle or normalization or model_signature or segment_packing or view_pose_model or interaction' -v`

Expected: `FAIL` with `ModuleNotFoundError: utils.a7c_view_pose_compositor`.

- [ ] **Step 4: Implement the runtime-only module**

Implement these public interfaces and validation rules exactly. The core
rotation and normalization operations are:

```python
def axis_angle_pose_to_rotation_6d(pose) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("pose must have shape [frames, joints, 3]")
    if not np.isfinite(values).all():
        raise ValueError("pose must be finite")
    matrices = Rotation.from_rotvec(values.reshape(-1, 3)).as_matrix()
    matrices = matrices.reshape(values.shape[:-1] + (3, 3))
    columns = np.swapaxes(matrices[..., :, :2], -1, -2)
    return columns.reshape(values.shape[0], -1)


def fit_normalization(values, fit_mask) -> dict[str, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    mask = np.asarray(fit_mask, dtype=bool).reshape(-1)
    if array.shape[0] != mask.size or not np.any(mask):
        raise ValueError("normalization requires aligned nonempty fit rows")
    mean = array[mask].mean(axis=0)
    scale = array[mask].std(axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    return {"mean": mean, "scale": scale, "fit_mask": mask.copy()}


def apply_normalization(values, stats) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - stats["mean"]) / stats["scale"]
```

Add `pose_manifest_sha256(model_dir, frame_ids, repo_root)`,
`load_pose_rotation_6d(model_dir, frame_ids, joint_indices)`,
`pack_camera_block_segments(camera_index, block_ids, frame_index,
frame_stride=5)`, and `build_runtime_inputs` with the exact keyword inputs in
the signature test. The manifest digest hashes lines formatted as
`<file_sha256><two spaces><repo-relative path><newline>` in increasing frame
order. Pose loading requires one `models/%06d.npz` per frame and reads only
`pose_body` joints `[0,1,3,4,6,7]`.

Construct graph context as `node`, adjacency-weighted `message`,
`node-message`, and visibility-weighted global context; project it to 16 view
dimensions. Encode pose with `Linear(36,16)`, `SiLU`, `Linear(16,16)`, `SiLU`;
multiply the view and pose embeddings elementwise; concatenate view, pose,
interaction, and graph context per carrier. Transpose to
`[carriers, frames, channels]` and run one shared bidirectional
`GRU(input_size,16,batch_first=True)` so each carrier is an independent
sequence. Zero hidden state at every call. The forward signature is
`forward(view, pose, adjacency, visibility, base_gates)`. Compute
`clamp(base + 0.1 * tanh(residual_head(gru_output)), 0.9, 1.0)` and
zero-initialize the scalar head.

The module must import no renderer contribution evaluator. `build_runtime_inputs`
uses camera/frame indices only to validate, order, and join the frozen manifest;
it may emit only F3, pose, projected geometry, adjacency, visibility, runtime
mass, and A5/base-gate tensors. Camera/frame indices and all forbidden IDs are
metadata and cannot appear in the model input schema. Validate finite values,
exact manifests, carrier alignment, contiguous stride, and forbidden feature
names.

- [ ] **Step 5: Run the runtime/model tests**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_view_pose_compositor.py -k 'axis_angle or normalization or model_signature or segment_packing or view_pose_model or interaction' -v`

Expected: all selected tests pass.

- [ ] **Step 6: Commit the runtime module**

```bash
git add utils/a7c_view_pose_compositor.py tests/test_a7c_view_pose_compositor.py
git commit -m "feat: add R1.4-VP runtime view-pose model"
```

### Task 3: Implement The Fit-Only Lexicographic Teacher Oracle

**Files:**
- Create: `utils/a7c_oracle_distillation.py`
- Create: `tests/test_a7c_oracle_distillation.py`

- [ ] **Step 1: Write failing tests for capacity and lexicographic anchoring**

```python
import numpy as np


def _teacher_problem():
    frames, carriers = 4, 2
    zeros = np.zeros((frames, carriers), dtype=np.float64)
    base = np.array([1.0, 2.0, 1.0, 2.0])
    point = np.stack([base, np.zeros_like(base)], axis=1)
    streams = {
        "objective": {
            "outer": {"base": base, "point": point},
            "boundary": {"base": base, "point": point},
        },
        "guard": {
            "target": {"base": np.ones(frames), "point": zeros},
            "outer": {"base": np.ones(frames), "point": zeros},
        },
    }
    return dict(
        runtime_mass=np.zeros((frames, carriers)),
        a5_weight=np.array([0.8, 0.8]),
        streams=streams,
        base_gates=np.array([[0.97, 0.98]] * frames),
    )


def test_teacher_capacity_and_anchor_are_feasible_and_deterministic():
    from utils.a7c_oracle_distillation import solve_fit_teacher

    kwargs = _teacher_problem()
    common = dict(
        minimum_gate=0.9, maximum_gate=1.0, selection_threshold=0.2,
        proxy_target_response=0.995, maximum_gate_jump=0.015,
        minimum_target_response=0.99, maximum_soft_iou_drop=0.005,
        minimum_outer_gain=0.005, boundary_margin=2e-5,
        bisection_tolerance=1e-5, lexicographic_tolerance=1e-9,
        primal_tolerance=1e-9, residual_tolerance=1e-7,
    )
    first = solve_fit_teacher(**kwargs, **common)
    second = solve_fit_teacher(**kwargs, **common)
    np.testing.assert_array_equal(first["gates"], second["gates"])
    assert first["capacity"]["interval_width"] <= 1e-5
    assert first["request"]["boundary_gain"] == (
        first["capacity"]["feasible_lower"] - 2e-5
    )
    assert first["certificate"]["maximum_primal_violation"] <= 1e-7
    assert first["certificate"]["stage_one_maximum_deviation"] >= 0.0
    assert first["certificate"]["stage_two_total_deviation"] >= 0.0
    assert first["certificate"]["stage_three_total_gate_change"] >= 0.0
```

- [ ] **Step 2: Write failing tests for fit/held isolation**

```python
def test_insert_teacher_segment_keeps_every_nonfit_value_nan():
    from utils.a7c_oracle_distillation import insert_teacher_segment

    gates = np.full((12, 2), np.nan)
    teacher_mask = np.zeros(12, dtype=bool)
    selected = np.array([True, True, False, False] * 3)
    solved = {"gates": np.full((6, 2), 0.97), "certificate": {
        "maximum_primal_violation": 1e-9
    }}
    insert_teacher_segment(gates, teacher_mask, selected, solved, 1e-7)
    assert np.isfinite(gates[selected]).all()
    assert np.isnan(gates[~selected]).all()
    assert teacher_mask.tolist() == selected.tolist()
```

- [ ] **Step 3: Run the tests and verify import failure**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_oracle_distillation.py -v`

Expected: `FAIL` with `ModuleNotFoundError: utils.a7c_oracle_distillation`.

- [ ] **Step 4: Implement capacity search and the three LP stages**

Implement three public functions. `solve_lexicographic_fixed_gain_oracle`
takes keyword-only runtime mass, A5 weight, renderer streams, anchor gates,
gate/topology/proxy/jump/true-target/soft-IoU/gain thresholds, and the three
solver tolerances, and returns `gates`, `metrics`, and `certificate`.
`solve_fit_teacher` adds `boundary_margin` and `bisection_tolerance`, performs
the capacity search, and returns `capacity`, `request`, `gates`, `metrics`, and
`certificate`. `insert_teacher_segment` accepts the full gate/mask arrays, one
selected segment, one solved result, and the residual tolerance; it mutates only
selected NaN rows and rejects overlap.

The result schemas are fixed as:

```python
result = {
    "capacity": {
        "feasible_lower": float(feasible_lower),
        "infeasible_upper": float(infeasible_upper),
        "interval_width": float(infeasible_upper - feasible_lower),
        "iterations": int(iterations),
    },
    "request": {
        "outer_gain": float(minimum_outer_gain),
        "boundary_gain": float(feasible_lower - boundary_margin),
    },
    "gates": gates,
    "metrics": evaluate_oracle_gates(gates, streams),
    "certificate": {
        "solver": "scipy.optimize.linprog:highs",
        "stage_one_maximum_deviation": float(rho_star),
        "stage_two_total_deviation": float(total_deviation_star),
        "stage_three_total_gate_change": float(total_gate_change),
        "maximum_primal_violation": float(maximum_violation),
    },
}
```

Build the same gate bounds and linear renderer/target/soft-IoU/jump constraints as `solve_fixed_gain_oracle`, in a new module so the frozen R1.3-G function does not change. The first anchored solve minimizes scalar `rho` subject to `abs(gate-anchor)<=rho`. The second constrains `rho<=rho_star+1e-9` and minimizes the sum of per-gate absolute deviations. The third constrains both preceding optima within `1e-9` and minimizes the sum of per-carrier adjacent absolute gate differences. Recompute direct renderer metrics with `evaluate_oracle_gates`; reject any optimum with non-finite gates or maximum residual above `1e-7`.

For capacity, call `bisect_feasible_gain` with outer fixed at `0.005` and boundary as the bisection variable, then request `feasible_lower-0.00002`. A non-bracketed endpoint or failed final solve raises `RuntimeError` and is a training error.

- [ ] **Step 5: Run oracle tests plus frozen regressions**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_oracle_distillation.py tests/test_a7c_exact_aggregate_oracle.py tests/test_a7c_feasibility_oracle.py -v`

Expected: all tests pass and R1.3-G behavior remains unchanged.

- [ ] **Step 6: Commit the teacher solver**

```bash
git add utils/a7c_oracle_distillation.py tests/test_a7c_oracle_distillation.py
git commit -m "feat: add fit-only oracle distillation solver"
```

### Task 4: Generate And Freeze 120 Fit-Only Teacher Segments

**Files:**
- Create: `tools/build_a7c_r1_4vp_fit_teachers.py`
- Modify: `tests/test_a7c_oracle_distillation.py`

- [ ] **Step 1: Add failing artifact-isolation tests**

```python
def test_teacher_artifact_has_exact_fold_local_masks_and_eligibility(tmp_path):
    from tools.build_a7c_r1_4vp_fit_teachers import write_fold_teacher

    camera = np.repeat(np.arange(4), 6)
    frame = np.tile(np.arange(6) * 5, 4)
    block = np.tile(np.arange(6), 4)
    fit_mask = block != 2
    gates = np.full((24, 2), np.nan)
    gates[fit_mask] = 0.97
    write_fold_teacher(
        output_dir=tmp_path, fold=2, gates=gates, teacher_mask=fit_mask,
        camera_index=camera, frame_index=frame, block_ids=block,
        carrier_ids=np.array([3, 7]), certificates=[{}] * 20,
        source_fingerprints={"probe": "abc"},
    )
    with np.load(tmp_path / "fold_2/teacher.npz", allow_pickle=False) as saved:
        assert np.array_equal(saved["teacher_mask"], fit_mask)
        assert np.isnan(saved["teacher_gates"][~fit_mask]).all()
        assert int(saved["deployment_eligible"]) == 0
        assert int(saved["teacher_eligible"]) == 0
        assert int(saved["paper_test_eligible"]) == 0
```

- [ ] **Step 2: Run the test and verify the missing CLI fails**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_oracle_distillation.py::test_teacher_artifact_has_exact_fold_local_masks_and_eligibility -v`

Expected: `FAIL` with `ModuleNotFoundError` for the teacher CLI.

- [ ] **Step 3: Implement the teacher generator**

The CLI accepts `--contract`, `--probe`, `--evidence`, `--a5-bank`, `--teacher`, `--r1-2b-training-dir`, and `--output-dir`. Reuse source loaders from R1.3-G, but generate each fold's 20 fit segments from `split["fit_mask"] & ~held_block_mask`. For each camera and non-held block, load the matching R1.2-B raw gates as the anchor and call `solve_fit_teacher`. Save held rows as NaN and attach camera/frame/block/carrier manifests, source fingerprints, sample/carrier fingerprints, capacity endpoints, three LP optima, direct metrics, and eligibility false.

Before returning success, enforce:

```python
assert total_segment_count == 120
assert all(fold["segment_count"] == 20 for fold in fold_summaries)
assert all(fold["held_finite_count"] == 0 for fold in fold_summaries)
assert maximum_primal_violation <= contract["solver_residual_tolerance"]
```

Write `teachers/summary.json` atomically. Repeated execution into a clean temporary directory must produce identical teacher gate array fingerprints and certificate metrics.

- [ ] **Step 4: Run the teacher unit and CLI smoke tests**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_oracle_distillation.py -v`

Expected: all teacher tests pass.

- [ ] **Step 5: Commit the teacher generator**

```bash
git add tools/build_a7c_r1_4vp_fit_teachers.py tests/test_a7c_oracle_distillation.py
git commit -m "feat: generate R1.4-VP fit-only teachers"
```

### Task 5: Train Six Models And Build The Fixed Nearest-Neighbor Baseline

**Files:**
- Create: `tools/train_a7c_r1_4vp_view_pose.py`
- Modify: `utils/a7c_view_pose_compositor.py`
- Modify: `tests/test_a7c_view_pose_compositor.py`

- [ ] **Step 1: Add failing loss and nearest-neighbor tests**

```python
def test_registered_distillation_loss_matches_three_terms():
    from utils.a7c_view_pose_compositor import distillation_loss

    prediction = torch.tensor([[0.96], [0.98], [0.97]])
    teacher = torch.tensor([[0.95], [0.99], [0.97]])
    residual = prediction - 0.97
    result = distillation_loss(
        prediction, teacher, residual,
        gate_delta=0.01, temporal_delta=0.005,
        temporal_weight=0.25, residual_weight=0.001,
    )
    assert set(result) == {"loss", "gate", "temporal", "residual"}
    torch.testing.assert_close(
        result["loss"], result["gate"] + 0.25 * result["temporal"]
        + 0.001 * result["residual"]
    )


def test_k4_baseline_uses_fit_rows_only_and_averages_exact_matches():
    from utils.a7c_view_pose_compositor import nearest_neighbor_predict

    fit_keys = np.array([[0.0], [0.0], [1.0], [2.0], [3.0]])
    fit_gates = np.array([[0.9], [1.0], [0.8], [0.7], [0.6]])
    query = np.array([[0.0], [1.5]])
    result = nearest_neighbor_predict(fit_keys, fit_gates, query, k=4)
    np.testing.assert_allclose(result[0], [0.95])
    assert result.shape == (2, 1)


def test_nearest_neighbor_key_is_pose_plus_six_registered_view_means():
    from utils.a7c_view_pose_compositor import build_nearest_neighbor_keys

    pose = np.zeros((2, 36))
    features = np.zeros((2, 3, 7))
    names = [
        "visibility", "view_dir_x", "view_dir_y", "view_dir_z",
        "log_depth", "alpha_transmittance_mass", "semantic_support_mean",
    ]
    features[:, :, 0] = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    keys = build_nearest_neighbor_keys(features, names, pose)
    assert keys.shape == (2, 42)
```

- [ ] **Step 2: Add failing fold-training isolation test**

```python
def test_training_uses_final_epoch_and_masks_held_labels(tmp_path):
    from tools.train_a7c_r1_4vp_view_pose import train_fold

    samples, carriers, channels = 8, 2, 4
    teacher_mask = np.array([True] * 4 + [False] * 4)
    teacher_gates = np.full((samples, carriers), np.nan, dtype=np.float32)
    teacher_gates[teacher_mask] = 0.96
    summary = train_fold(
        fold=0,
        features=np.zeros((samples, carriers, channels), np.float32),
        pose=np.zeros((samples, 36), np.float32),
        adjacency=np.zeros((samples, carriers, carriers), np.float32),
        visibility=np.ones((samples, carriers), np.float32),
        base_gates=np.full((samples, carriers), 0.97, np.float32),
        teacher_gates=teacher_gates,
        teacher_mask=teacher_mask,
        camera_index=np.zeros(samples, np.int16),
        frame_index=np.arange(samples, dtype=np.int32) * 5,
        block_ids=np.array([0] * 4 + [1] * 4, np.int16),
        runtime_mass=np.zeros((samples, carriers), np.float32),
        a5_weight=np.full(carriers, 0.8, np.float32),
        contract={
            "view_embedding_dimension": 16, "pose_embedding_dimension": 16,
            "gru_hidden_dimension": 16, "residual_gate_scale": 0.1,
            "minimum_gate": 0.9, "maximum_gate": 1.0,
            "selection_threshold": 0.2, "proxy_target_response": 0.995,
            "maximum_projection_gate_jump": 0.015,
            "lexicographic_tolerance": 1e-9,
            "solver_primal_tolerance": 1e-9,
            "solver_residual_tolerance": 1e-7,
            "training_epochs": 3, "random_seed": 7,
            "learning_rate": 0.001, "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0, "gate_huber_delta": 0.01,
            "temporal_huber_delta": 0.005, "temporal_loss_weight": 0.25,
            "residual_loss_weight": 0.001, "maximum_parameter_count": 50000,
            "frame_stride": 5,
        },
        output_dir=tmp_path,
        device="cpu",
    )
    assert summary["epochs"] == 3
    assert summary["checkpoint_epoch"] == 3
    assert summary["held_teacher_values_accessed"] is False
    assert summary["parameter_count"] <= 50000
    assert summary["maximum_gradient_norm_before_clip"] >= 0.0
```

- [ ] **Step 3: Run tests and verify missing functions fail**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_view_pose_compositor.py -k 'distillation_loss or k4_baseline or training_uses' -v`

Expected: `FAIL` because loss, nearest-neighbor, and trainer interfaces do not exist.

- [ ] **Step 4: Implement loss, baseline, and fold training**

Implement `distillation_loss` with
`torch.nn.functional.huber_loss(input, target, reduction="mean", delta=delta)`,
temporal differences within each packed segment only, and mean absolute
residual. Implement `nearest_neighbor_predict` with fit-only normalization,
Euclidean distance, exact-zero mean, otherwise inverse-distance weighting over
exactly four nearest fit samples. Implement `build_nearest_neighbor_keys` as
the 36 pose values followed by visibility-weighted means of exactly
`view_dir_x`, `view_dir_y`, `view_dir_z`, `log_depth`,
`alpha_transmittance_mass`, and `semantic_support_mean`, in that order.

In `train_fold`:

1. load only the fold's fit teacher mask and verify held values are NaN;
2. fit F3, pose, and nearest-neighbor statistics only on `fit_mask`;
3. use one full camera-block segment per optimizer batch for 400 epochs;
4. use AdamW, `lr=0.001`, `weight_decay=0.0001`, global gradient clip 1.0, seed 20260805;
5. choose the final epoch only;
6. infer learned and nearest-neighbor raw gates for the full four-camera sample manifest;
7. call `solve_temporal_joint_projection` separately for every camera-block segment using runtime mass only;
8. save learned/NN raw and projected gates, masks, normalization statistics, projection certificates, and all three eligibility flags false.

Process segments in stable `(camera_index, block_id)` order without shuffling.
The model has no dropout, attention, or carrier embedding. Each fold summary
records initial/final total and component losses, maximum gradient norm before
clipping, raw/projected gate ranges, fit teacher MAE, fit temporal-difference
MAE, parameter count, final checkpoint epoch, and false held-label access.

The CLI processes folds 0-5, writes `training/fold_N/{model.pt,predictions.npz,summary.json}`, `training/summary.json`, `nearest_neighbor/fold_N/{predictions.npz,summary.json}`, and `nearest_neighbor/summary.json`. It writes `models_frozen.json` only after every required artifact exists and records their SHA256 values. It must not import `_build_streams`, `evaluate_contribution_predictions`, or any renderer metric function.

- [ ] **Step 5: Run model/baseline tests and projection regressions**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_view_pose_compositor.py tests/test_a7c_temporal_joint_projection.py tests/test_a7c_overlap_set_compositor.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit training and baseline code**

```bash
git add tools/train_a7c_r1_4vp_view_pose.py utils/a7c_view_pose_compositor.py tests/test_a7c_view_pose_compositor.py
git commit -m "feat: train R1.4-VP and k4 baseline"
```

### Task 6: Implement The Independent Held Auditor

**Files:**
- Create: `tools/audit_a7c_r1_4vp_view_pose.py`
- Modify: `tests/test_a7c_view_pose_compositor.py`

- [ ] **Step 1: Add failing formal-summary tests**

```python
def _passing_records():
    return [dict(
        fold=fold, camera_index=camera,
        outer_gain=0.006, boundary_gain=0.025,
        minimum_target_response=0.995,
        maximum_soft_iou_drop=0.001,
        visibility_response_ratio=0.999,
        maximum_adjacent_gate_change=0.015,
        topology_passed=True, coverage_passed=True,
        frozen_parts_passed=True, weight_upper_bound_passed=True,
    ) for fold in range(6) for camera in range(4)]


def promotion_contract():
    return {
        "minimum_outer_gain": 0.005,
        "minimum_boundary_gain": 0.005,
        "minimum_positive_block_fraction": 0.9,
        "block_gain_quantile": 0.1,
        "minimum_block_gain_quantile": 0.0,
        "maximum_worst_block_regression": 0.005,
        "minimum_target_response": 0.99,
        "maximum_visibility_response_ratio": 1.0,
        "maximum_selection_soft_iou_drop": 0.005,
        "maximum_adjacent_gate_change": 0.02,
        "r1_1_f1_outer_gain": -0.00012761059760764496,
        "r1_1_f1_boundary_gain": 0.023481874880317264,
        "r1_2b_outer_gain": 0.005196372744170267,
        "r1_2b_boundary_gain": 0.002866365549963367,
        "comparison_tolerance": 1e-9,
    }


def test_auditor_requires_formal_per_camera_and_baseline_superiority():
    from tools.audit_a7c_r1_4vp_view_pose import classify_canary

    contract = promotion_contract()
    learned = _passing_records()
    nn = [dict(row, outer_gain=0.0055, boundary_gain=0.024) for row in learned]
    assert classify_canary(learned, nn, contract) == "CANARY_PROMOTED"
    broken = [dict(row) for row in learned]
    for row in broken:
        if row["camera_index"] == 3:
            row["boundary_gain"] = -0.001
    assert classify_canary(broken, nn, contract) == "CANARY_NEGATIVE"


def test_visibility_response_uses_target_contribution_over_pixel_count():
    from tools.audit_a7c_r1_4vp_view_pose import visibility_response_ratio

    pixels = np.array([10.0, 20.0, 10.0])
    base = np.array([5.0, 8.0, 6.0])
    candidate = np.array([5.0, 8.0, 6.0])
    assert visibility_response_ratio(base, candidate, pixels) == 1.0
```

- [ ] **Step 2: Add failing artifact-integrity test**

```python
def test_auditor_rejects_missing_freeze_manifest_or_label_leakage(tmp_path):
    from tools.audit_a7c_r1_4vp_view_pose import verify_frozen_artifacts

    with pytest.raises(ValueError, match="models_frozen"):
        verify_frozen_artifacts(tmp_path, expected={})
```

- [ ] **Step 3: Run tests and verify missing auditor fails**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_view_pose_compositor.py -k 'auditor or visibility_response' -v`

Expected: `FAIL` with `ModuleNotFoundError` for the R1.4-VP auditor.

- [ ] **Step 4: Implement independent audit and verdict**

The auditor accepts only the contract, evidence, A5 bank, original teacher manifest, R1.3-G witness directory, frozen model/baseline root, and output directory. It must not import the trainer. Before opening evidence metrics, verify all six model and NN hashes from `models_frozen.json`, source/sample/carrier/pose/normalization/prediction masks, NaN isolation, projection certificates, and eligibility flags.

For the exact held mask of every fold and each fit camera, independently recompute:

```python
outer_gain
boundary_gain
minimum_target_response
maximum_soft_iou_drop
visibility_response_ratio
maximum_adjacent_gate_change
topology_passed
coverage_passed
frozen_parts_passed
weight_upper_bound_passed
```

Compute visibility response from unweighted target contribution divided by `renderer_sequence_target_pixel_count`, then normalized flicker candidate divided by normalized flicker A5. Load held R1.3-G gates only after learned/NN renderer records are frozen; report gate MAE, temporal-difference MAE, and gain recovery without using them in the verdict.

For spatial guards, materialize the candidate from a repeated full A5 weight
array and multiply only `lower` at the frozen `carrier_ids`. Require all
hair/face/upper/shoes/skin values and all non-carrier lower values to be
bitwise equal to A5, every candidate weight to be at most A5, and the lower
`weight >= 0.2` selection mask to equal A5 at every frame. Recompute evidence
support coverage with `temporal_consecutive_visible_count >= 8` and require the
same A5-supported carrier set and coverage ratio at least `0.8`.

`classify_canary` returns `CANARY_PROMOTED` only when every formal aggregate threshold passes, all four cameras have positive mean outer and boundary gain, visibility response is at most 1.0, and learned mean outer and boundary are each greater than both R1.2-B and NN by more than the `1e-9` comparison tolerance. Correct execution with any failed gate returns `CANARY_NEGATIVE`; integrity or numerical errors raise and become `TRAINING_ERROR` in the runner.

Write `audit/held_block_summary.json` and the root `summary.json` atomically, including all 24 learned and 24 NN records, per-camera summaries, comparison margins, oracle diagnostics, source/artifact hashes, and false eligibility flags. Return status 0 for promoted, 2 for negative, and 1 for error.

- [ ] **Step 5: Run audit and frozen regression tests**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_view_pose_compositor.py tests/test_a7c_exact_aggregate_oracle.py tests/test_a7c_temporal_joint_projection.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit the auditor**

```bash
git add tools/audit_a7c_r1_4vp_view_pose.py tests/test_a7c_view_pose_compositor.py
git commit -m "feat: audit R1.4-VP held canary"
```

### Task 7: Add The Restart-Safe Experiment Runner

**Files:**
- Create: `tools/run_a7c_r1_4vp_view_pose_377.sh`
- Modify: `tests/test_a7c_view_pose_compositor.py`

- [ ] **Step 1: Add a failing runner protocol test**

```python
import re


def test_r1_4vp_runner_is_restart_safe_audit_gated_and_camera_isolated():
    runner = ROOT / "tools/run_a7c_r1_4vp_view_pose_377.sh"
    source = runner.read_text(encoding="utf-8")
    for camera in ("c17", "c18", "c19", "c20", "c21", "c22", "c23"):
        assert re.search(rf"\\b{camera}\\b", source) is None
    assert "build_a7c_r1_4vp_fit_teachers.py" in source
    assert "train_a7c_r1_4vp_view_pose.py" in source
    assert "models_frozen.json" in source
    audit = source.index("audit_a7c_r1_4vp_view_pose.py")
    assert source.index("models_frozen.json") < audit
    assert 'mark_terminal completed' in source
    assert 'mark_terminal rejected' in source
    assert 'mark_terminal failed' in source
    assert "started_utc.txt" in source and "ended_utc.txt" in source
```

- [ ] **Step 2: Run the test and verify the missing runner fails**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_view_pose_compositor.py::test_r1_4vp_runner_is_restart_safe_audit_gated_and_camera_isolated -v`

Expected: `FAIL` with `FileNotFoundError` for the runner.

- [ ] **Step 3: Implement the runner**

Use the R1.3-G runner's atomic error summary, SHA checks, restart detection, and terminal-marker functions. The runner executes in this order:

```text
1. verify contract/design/R1.3-G/R1.3-P/R1.2-B/probe/evidence/A5/teacher/pose hashes
2. write started_utc.txt once
3. generate or validate teachers/summary.json and six teacher folds
4. train or validate six learned folds and six nearest-neighbor folds
5. require models_frozen.json with every artifact hash
6. run the independent held auditor once
7. write ended_utc.txt
8. CANARY_PROMOTED -> .completed
   CANARY_NEGATIVE -> .rejected
   TRAINING_ERROR -> .failed
```

`required_outputs_complete` requires exactly one terminal marker, root summary, teacher/training/NN summaries, six fold artifact sets, audit summary, PID, log, and timestamps. A `.failed` state is not silently treated as success. Use `/opt/miniconda3/envs/ictrl/bin/python`, `set -euo pipefail`, atomic summary replacement, and an `ERR` trap. Do not mention forbidden camera names in executable command arguments; source verification is driven by the contract.

- [ ] **Step 4: Run runner tests and shell syntax check**

Run: `bash -n tools/run_a7c_r1_4vp_view_pose_377.sh && /opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_view_pose_compositor.py -k runner -v`

Expected: shell syntax succeeds and runner tests pass.

- [ ] **Step 5: Commit the runner**

```bash
git add tools/run_a7c_r1_4vp_view_pose_377.sh tests/test_a7c_view_pose_compositor.py
git commit -m "feat: orchestrate R1.4-VP canary"
```

### Task 8: Verify, Launch, Estimate, And Record The Real Experiment

**Files:**
- Modify: `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`

- [ ] **Step 1: Run focused R1.4-VP tests**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_oracle_distillation.py tests/test_a7c_view_pose_compositor.py -v`

Expected: all focused tests pass.

- [ ] **Step 2: Run frozen A7c regression tests**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_exact_aggregate_oracle.py tests/test_a7c_feasibility_oracle.py tests/test_a7c_temporal_joint_projection.py tests/test_a7c_overlap_set_compositor.py tests/test_a7c_quotient_compositor.py tests/test_a7c_ray_context_probe.py -v`

Expected: all frozen A7c regressions pass.

- [ ] **Step 3: Verify source hashes and GPU availability**

Run: `sha256sum configs/semantic/a7c_r1_4vp_oracle_distilled_view_pose_377_v1.json docs/superpowers/specs/2026-08-05-a7c-r1-4vp-oracle-distilled-view-pose-design.md && nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader`

Expected: contract/design hashes match the runner and at least one GPU has enough memory for the under-50k-parameter model and full segment tensors.

- [ ] **Step 4: Start the detached runner**

Run:

```bash
OUT=/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/a7c_r1_4vp_oracle_distilled_view_pose_377_v1
setsid bash tools/run_a7c_r1_4vp_view_pose_377.sh "$OUT" \
  > "$OUT.launch.log" 2>&1 < /dev/null &
echo $!
```

Expected: a live PID, `started_utc.txt`, `runner.pid`, and growing `runner.log`.

- [ ] **Step 5: Estimate Beijing completion only after observed work**

After 4-8 teacher segments complete, compute the observed teacher seconds per segment from log timestamps. Add the measured first-fold training duration and fixed audit allowance, then report:

```text
estimated_remaining_seconds = observed_teacher_seconds_per_segment * remaining_teacher_segments
                            + observed_model_seconds_per_fold * remaining_model_folds
                            + baseline_and_audit_allowance
estimated_end_bjt = current_utc + estimated_remaining_seconds + 8 hours
```

Label this as an estimate. Do not report an actual completion time until `ended_utc.txt` exists.

- [ ] **Step 6: Inspect the terminal result**

Run: `ps -p "$(cat "$OUT/runner.pid")" -o pid,etime,cmd || true; ls -la "$OUT"; /opt/miniconda3/envs/ictrl/bin/python -m json.tool "$OUT/summary.json"`

Expected: exactly one of `.completed`, `.rejected`, `.failed`; `summary.json` explains every promotion gate.

No canary outcome opens `c17-c23` or authorizes `Task 12`; either action needs
a separately preregistered validation design.

- [ ] **Step 7: Append the real result without staging user changes**

Add a dated R1.4-VP block near the top of the A7 handoff containing commit/config hashes, start/end UTC and Beijing times, terminal verdict, 24-record learned and NN metrics, per-camera signs, visibility response, target/soft-IoU/jump/topology guards, oracle recovery, fit/held errors, and the registered next-route decision. Preserve all unrelated document edits by constructing and committing only the exact added patch.

- [ ] **Step 8: Verify and commit only the result block**

Run: `git diff --check -- docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`

Expected: no whitespace errors. Stage only the newly added block, verify the cached diff, then commit:

```bash
git diff --cached --check
git commit -m "docs: record R1.4-VP canary result"
```
