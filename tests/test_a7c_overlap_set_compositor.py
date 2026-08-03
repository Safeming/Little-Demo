import json
from pathlib import Path

import torch

from utils.a7c_overlap_set_compositor import (
    DenseOverlapSetCompositor,
    dense_overlap_adjacency,
)


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
