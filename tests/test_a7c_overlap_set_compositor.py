import json
from pathlib import Path

import numpy as np
import torch

from utils.a7c_overlap_set_compositor import (
    DenseOverlapSetCompositor,
    dense_overlap_adjacency,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/semantic/a7c_r1_2b_dense_overlap_set_377_v1.json"
TRAINER = ROOT / "tools/train_a7c_r1_2b_overlap_set_compositor.py"


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


def test_overlap_adjacency_masks_self_and_invisible_nodes():
    adjacency = dense_overlap_adjacency(
        projected_xy=torch.tensor(
            [[[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]]]
        ),
        log_depth=torch.zeros(1, 3),
        visibility=torch.tensor([[1.0, 1.0, 0.0]]),
        spatial_scale=0.03,
        depth_scale=0.04,
        edge_log_weight_minimum=-20.0,
    )
    torch.testing.assert_close(
        torch.diagonal(adjacency, dim1=1, dim2=2), torch.zeros(1, 3)
    )
    torch.testing.assert_close(adjacency[:, :, 2], torch.zeros(1, 3))
    torch.testing.assert_close(adjacency[0, 0].sum(), torch.tensor(1.0))
    torch.testing.assert_close(adjacency[0, 2], torch.zeros(3))


def test_overlap_set_is_permutation_equivariant():
    torch.manual_seed(4)
    model = DenseOverlapSetCompositor(
        30,
        32,
        32,
        minimum_gate=0.9,
        initial_gate=0.999,
    )
    with torch.no_grad():
        model.gate_head[-1].weight.fill_(0.01)
    features = torch.randn(2, 5, 30)
    projected_xy = torch.randn(2, 5, 2) * 0.01
    log_depth = torch.randn(2, 5) * 0.02
    visibility = torch.tensor(
        [[1.0, 1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0, 1.0]]
    )
    permutation = torch.tensor([3, 0, 4, 1, 2])
    original = model(
        features,
        projected_xy,
        log_depth,
        visibility,
        spatial_scale=0.03,
        depth_scale=0.04,
        edge_log_weight_minimum=-20.0,
    )
    permuted = model(
        features[:, permutation],
        projected_xy[:, permutation],
        log_depth[:, permutation],
        visibility[:, permutation],
        spatial_scale=0.03,
        depth_scale=0.04,
        edge_log_weight_minimum=-20.0,
    )
    torch.testing.assert_close(permuted, original[:, permutation])


def test_overlap_set_all_invisible_output_is_finite_and_initialized():
    model = DenseOverlapSetCompositor(
        30,
        32,
        32,
        minimum_gate=0.9,
        initial_gate=0.999,
    )
    output = model(
        torch.zeros(2, 4, 30),
        torch.zeros(2, 4, 2),
        torch.zeros(2, 4),
        torch.zeros(2, 4),
        spatial_scale=0.03,
        depth_scale=0.04,
        edge_log_weight_minimum=-20.0,
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output, torch.full((2, 4), 0.999), atol=1.0e-6, rtol=0.0
    )
    assert torch.all(output >= 0.9) and torch.all(output <= 1.0)


def test_r1_2b_trainer_has_no_teacher_gate_objective():
    source = TRAINER.read_text(encoding="utf-8")
    assert 'teacher["gates"]' not in source
    assert "DenseOverlapSetCompositor" in source
    assert '"teacher_gate_values_accessed": False' in source


def test_r1_2b_cpu_training_is_deterministic_and_bounded(tmp_path):
    from tools.train_a7c_r1_2b_overlap_set_compositor import train_one

    samples, carriers = 12, 2
    peak = (np.arange(samples) % 2).astype(np.float32)
    features = np.zeros((samples, carriers, 30), dtype=np.float32)
    features[:, :, 0] = peak[:, None]
    features[:, :, 1] = np.array([0.0, 1.0], dtype=np.float32)
    projected_xy = np.zeros((samples, carriers, 2), dtype=np.float32)
    projected_xy[:, 1, 0] = 0.01
    log_depth = np.zeros((samples, carriers), dtype=np.float32)
    visibility = np.ones((samples, carriers), dtype=np.float32)
    outer = (1.0 + peak).astype(np.float32)
    point_outer = np.stack([outer, np.zeros_like(outer)], axis=1)
    zeros = np.zeros((samples, carriers), dtype=np.float32)
    streams = {
        "target": {"base": np.ones(samples, np.float32), "point": zeros},
        "outer": {"base": outer, "point": point_outer},
        "boundary": {"base": outer, "point": point_outer},
    }
    contract = {
        "node_hidden_dimension": 8,
        "gate_hidden_dimension": 8,
        "spatial_scale": 0.03,
        "depth_scale": 0.04,
        "edge_log_weight_minimum": -20.0,
        "minimum_gate": 0.9,
        "maximum_gate": 1.0,
        "initial_minimum_gate": 0.999,
        "proxy_target_response": 0.995,
        "selection_threshold": 0.2,
        "training_target_response": 0.995,
        "training_epochs": 40,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "outer_loss_weight": 1.0,
        "boundary_loss_weight": 1.0,
        "target_hinge_weight": 100.0,
        "soft_iou_hinge_weight": 100.0,
        "gate_jump_hinge_weight": 0.0,
        "damping_regularizer_weight": 0.0,
        "maximum_selection_soft_iou_drop": 0.005,
        "maximum_adjacent_gate_change": 0.02,
        "teacher_gate_loss_weight": 0.0,
        "random_seed": 7,
        "frame_stride": 5,
        "paper_test_eligible": False,
    }
    kwargs = {
        "train_mask": np.ones(samples, bool),
        "features": features,
        "projected_xy": projected_xy,
        "log_depth": log_depth,
        "visibility": visibility,
        "runtime_mass": np.tile(
            np.array([[0.0, 1.0]], np.float32), (samples, 1)
        ),
        "a5_weight": np.array([0.8, 0.8], np.float32),
        "objective_streams": streams,
        "guard_streams": streams,
        "camera_index": np.zeros(samples, np.int16),
        "frame_index": np.arange(samples, dtype=np.int32) * 5,
        "block_ids": np.zeros(samples, np.int16),
        "contract": contract,
        "output_dir": tmp_path,
        "device": "cpu",
    }
    first = train_one(name="first", **kwargs)
    second = train_one(name="second", **kwargs)
    assert first["final_loss"] <= first["initial_loss"]
    with np.load(tmp_path / "first/predictions.npz") as a, np.load(
        tmp_path / "second/predictions.npz"
    ) as b:
        np.testing.assert_allclose(
            a["projected_gates"], b["projected_gates"], atol=0, rtol=0
        )
        assert a["projected_gates"].min() >= 0.9
        assert a["projected_gates"].max() <= 1.0
