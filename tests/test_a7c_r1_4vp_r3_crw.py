import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.train_a7c_r1_4vp_r3_crw import train_fold
from utils.a7c_r1_4vp_r3_crw import (
    build_contribution_weights,
    classify_fit_entry_failure,
    contribution_weighted_distillation_loss,
    evaluate_fit_renderer_entry,
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


def test_fit_renderer_entry_requires_recovery_and_positive_fractions():
    result = evaluate_fit_renderer_entry(
        learned_outer=[0.007, 0.008],
        teacher_outer=[0.010, 0.010],
        learned_boundary=[0.021, 0.022],
        teacher_boundary=[0.030, 0.030],
        minimum_outer_recovery=0.70,
        minimum_boundary_recovery=0.70,
        minimum_positive_fraction=0.90,
    )

    assert result["passed"] is True
    assert result["outer_recovery"] == pytest.approx(0.75)
    assert result["boundary_recovery"] == pytest.approx(0.0215 / 0.03)
    assert result["outer_positive_fraction"] == pytest.approx(1.0)
    assert result["boundary_positive_fraction"] == pytest.approx(1.0)
    assert result["failed_conditions"] == []


@pytest.mark.parametrize(
    "override,failed_condition",
    [
        ({"learned_outer": [0.006, 0.006]}, "outer_recovery"),
        ({"learned_boundary": [0.020, 0.020]}, "boundary_recovery"),
        ({"learned_outer": [0.010, -0.001]}, "outer_positive_fraction"),
        ({"learned_boundary": [0.030, -0.001]}, "boundary_positive_fraction"),
    ],
)
def test_fit_renderer_entry_reports_each_negative_condition(
    override, failed_condition
):
    arguments = {
        "learned_outer": [0.008, 0.008],
        "teacher_outer": [0.010, 0.010],
        "learned_boundary": [0.024, 0.024],
        "teacher_boundary": [0.030, 0.030],
        "minimum_outer_recovery": 0.70,
        "minimum_boundary_recovery": 0.70,
        "minimum_positive_fraction": 0.90,
    }
    arguments.update(override)

    result = evaluate_fit_renderer_entry(**arguments)

    assert result["passed"] is False
    assert failed_condition in result["failed_conditions"]


@pytest.mark.parametrize(
    "override,match",
    [
        ({"teacher_outer": [0.0, 0.0]}, "teacher_outer mean"),
        ({"teacher_boundary": [-0.1, -0.1]}, "teacher_boundary mean"),
        ({"learned_outer": [0.1, np.nan]}, "learned_outer"),
        ({"learned_boundary": [0.1]}, "same length"),
    ],
)
def test_fit_renderer_entry_rejects_invalid_inputs(override, match):
    arguments = {
        "learned_outer": [0.008, 0.008],
        "teacher_outer": [0.010, 0.010],
        "learned_boundary": [0.024, 0.024],
        "teacher_boundary": [0.030, 0.030],
        "minimum_outer_recovery": 0.70,
        "minimum_boundary_recovery": 0.70,
        "minimum_positive_fraction": 0.90,
    }
    arguments.update(override)

    with pytest.raises(ValueError, match=match):
        evaluate_fit_renderer_entry(**arguments)


def test_fit_renderer_entry_failure_has_fold_specific_terminal_status():
    assert classify_fit_entry_failure(0) == "FIT_RENDERER_ENTRY_NEGATIVE"
    assert classify_fit_entry_failure(1) == "TRAINING_ERROR"
    with pytest.raises(ValueError, match="fold_index"):
        classify_fit_entry_failure(-1)


def test_train_fold_uses_only_fit_teacher_and_contribution_rows(tmp_path, monkeypatch):
    samples, carriers = 8, 2
    fit_mask = np.array([True] * 4 + [False] * 4)
    teacher = np.full((samples, carriers), np.nan, dtype=np.float32)
    teacher[fit_mask] = 0.93
    contribution_weight = np.full((samples, carriers), np.nan, dtype=np.float64)
    contribution_weight[fit_mask] = np.array(
        [[1.5, 0.5], [1.25, 0.75], [0.75, 1.25], [0.5, 1.5]]
    )
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract.update(
        {
            "training_epochs": 4,
            "view_embedding_dimension": 4,
            "pose_embedding_dimension": 4,
            "gru_hidden_dimension": 4,
            "maximum_fit_teacher_mae": 0.1,
            "frame_stride": 1,
        }
    )

    def fake_projection(raw, prediction_mask, *args, **kwargs):
        projected = np.full_like(raw, np.nan, dtype=np.float64)
        projected[prediction_mask] = raw[prediction_mask]
        return projected, [{"maximum_primal_violation": 0.0}]

    monkeypatch.setattr(
        "tools.train_a7c_r1_4vp_r3_crw._project_segments", fake_projection
    )
    summary = train_fold(
        fold=0,
        features=np.linspace(-1.0, 1.0, samples * carriers * 3).reshape(
            samples, carriers, 3
        ),
        pose=np.linspace(-1.0, 1.0, samples * 36).reshape(samples, 36),
        adjacency=np.repeat(np.eye(carriers)[None, :, :], samples, axis=0),
        visibility=np.ones((samples, carriers), dtype=np.float32),
        base_gates=np.full((samples, carriers), 0.95, dtype=np.float32),
        teacher_gates=teacher,
        contribution_weight=contribution_weight,
        teacher_mask=fit_mask,
        prediction_mask=np.ones(samples, dtype=bool),
        camera_index=np.zeros(samples, dtype=np.int64),
        frame_index=np.tile(np.arange(4), 2),
        block_ids=np.repeat([0, 1], 4),
        runtime_mass=np.ones((samples, carriers), dtype=np.float32),
        a5_weight=np.ones(carriers, dtype=np.float32),
        contract=contract,
        output_dir=tmp_path,
        device="cpu",
    )

    assert summary["checkpoint_epoch"] == contract["training_epochs"]
    assert summary["final_components"]["loss"] < summary["initial_components"]["loss"]
    assert summary["held_teacher_values_accessed"] is False
    assert summary["held_contribution_values_accessed"] is False
    assert summary["contribution_weight_mean"] == pytest.approx(1.0)
    assert (tmp_path / "model.pt").is_file()
    assert (tmp_path / "predictions.npz").is_file()
