import json
from pathlib import Path

import numpy as np
import pytest
import torch

from utils.a7c_r1_4vp_r3_crw import (
    build_contribution_weights,
    contribution_weighted_distillation_loss,
    temporal_segment_weights,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/semantic/a7c_r1_4vp_r3_crw_contribution_weighted_377_v1.json"


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


def test_build_contribution_weights_is_positive_clipped_and_mean_one():
    point = {
        "target": np.array([[1.0, 0.0], [np.nan, np.nan]]),
        "outer": np.array([[0.0, 4.0], [np.nan, np.nan]]),
        "boundary": np.array([[0.0, 0.0], [np.nan, np.nan]]),
    }
    result = build_contribution_weights(
        point,
        np.array([True, False]),
        [np.array([0])],
        epsilon=1e-12,
        minimum=0.1,
        maximum=10.0,
    )

    assert np.isfinite(result["gate"][0]).all()
    assert (result["gate"][0] > 0.0).all()
    assert np.isnan(result["gate"][1]).all()
    assert result["gate"][0].mean() == pytest.approx(1.0)
    assert result["clipped"][0].min() >= 0.1
    assert result["clipped"][0].max() <= 10.0


def test_temporal_weights_use_adjacent_maximum_inside_one_segment():
    gate = np.array([[0.5, 1.5], [1.5, 0.5]])

    temporal = temporal_segment_weights(gate)

    np.testing.assert_allclose(temporal, [[1.5, 1.5]])


def test_weighted_loss_amplifies_high_contribution_carrier_gradient():
    prediction = torch.tensor(
        [[0.97, 0.97], [0.97, 0.97]], requires_grad=True
    )
    teacher = torch.full_like(prediction, 0.95)
    residual = torch.zeros_like(prediction)
    gate_weight = torch.tensor([[4.0, 1.0], [4.0, 1.0]])
    temporal_weight = torch.ones((1, 2))

    loss = contribution_weighted_distillation_loss(
        prediction,
        teacher,
        residual,
        gate_weight,
        temporal_weight,
        gate_delta=0.01,
        temporal_delta=0.005,
        temporal_loss_weight=0.25,
        residual_loss_weight=0.00001,
    )["loss"]
    loss.backward()

    assert prediction.grad[:, 0].abs().mean() > prediction.grad[:, 1].abs().mean()


def test_zero_contribution_signals_remain_finite_and_mean_one():
    point = {
        name: np.zeros((2, 3), dtype=np.float64)
        for name in ("target", "outer", "boundary")
    }

    result = build_contribution_weights(
        point,
        np.ones(2, dtype=bool),
        [np.array([0, 1])],
        epsilon=1e-12,
        minimum=0.1,
        maximum=10.0,
    )

    np.testing.assert_allclose(result["gate"], np.ones((2, 3)))


@pytest.mark.parametrize(
    "point,fit_mask,segments,match",
    [
        (
            {name: np.ones((2, 2)) for name in ("target", "outer", "boundary")},
            np.array([True, False]),
            [np.array([0])],
            "held contribution rows",
        ),
        (
            {
                "target": np.ones((2, 2)),
                "outer": np.ones((2, 3)),
                "boundary": np.ones((2, 2)),
            },
            np.ones(2, dtype=bool),
            [np.array([0, 1])],
            "same shape",
        ),
        (
            {
                name: np.ones((2, 2))
                for name in ("target", "outer", "boundary")
            },
            np.ones(2, dtype=bool),
            [np.array([0])],
            "cover the fit mask",
        ),
    ],
)
def test_contribution_weights_fail_closed(point, fit_mask, segments, match):
    with pytest.raises(ValueError, match=match):
        build_contribution_weights(
            point,
            fit_mask,
            segments,
            epsilon=1e-12,
            minimum=0.1,
            maximum=10.0,
        )


def test_weighted_loss_rejects_nonfinite_weights_and_wrong_residual_scale():
    prediction = torch.full((2, 2), 0.97)
    teacher = torch.full_like(prediction, 0.95)
    residual = torch.zeros_like(prediction)
    gate_weight = torch.ones_like(prediction)
    temporal_weight = torch.ones((1, 2))

    with pytest.raises(ValueError, match="residual_loss_weight"):
        contribution_weighted_distillation_loss(
            prediction,
            teacher,
            residual,
            gate_weight,
            temporal_weight,
            gate_delta=0.01,
            temporal_delta=0.005,
            temporal_loss_weight=0.25,
            residual_loss_weight=0.001,
        )

    gate_weight[0, 0] = torch.nan
    with pytest.raises(ValueError, match="gate_weight"):
        contribution_weighted_distillation_loss(
            prediction,
            teacher,
            residual,
            gate_weight,
            temporal_weight,
            gate_delta=0.01,
            temporal_delta=0.005,
            temporal_loss_weight=0.25,
            residual_loss_weight=0.00001,
        )
