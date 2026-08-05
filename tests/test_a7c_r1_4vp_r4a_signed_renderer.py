import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tools.audit_a7c_r1_4vp_r4a_signed_renderer import _run as run_r4a_audit
from tools.train_a7c_r1_4vp_r4a_signed_renderer import train_fold
from utils.a7c_r1_4vp_r4a import (
    freeze_initial_scales,
    mean_normalized_trajectory,
    reconstruct_renderer_sequence,
    summarize_action_recovery,
    signed_renderer_trajectory_components,
    signed_renderer_trajectory_loss,
    signed_trajectory_component,
)
from utils.a7c_r1_4vp_r2_runtime import ViewPoseResidualCompositor


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs/semantic/a7c_r1_4vp_r4a_signed_renderer_trajectory_377_v1.json"
)
RUNNER = ROOT / "tools/run_a7c_r1_4vp_r4a_signed_renderer_377.sh"


def test_r4a_contract_changes_only_the_registered_training_objective():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == (
        "a7c_r1_4vp_r4a_signed_renderer_trajectory_377_v1"
    )
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


def _cosine(left, right):
    left = left.flatten()
    right = right.flatten()
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    assert torch.linalg.vector_norm(left) > 0
    assert torch.linalg.vector_norm(right) > 0
    return torch.dot(left, right) / (
        torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
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
            "point": torch.tensor(
                [[2.0, 1.0], [1.0, 3.0], [2.0, 2.0]]
            ),
        }
        for signal in ("target", "outer", "boundary")
    }
    gates = torch.tensor([[0.9, 1.0], [0.95, 0.9], [1.0, 0.95]])

    components = signed_renderer_trajectory_components(
        gates,
        gates,
        streams,
        renderer_delta=0.005,
        target_delta=0.005,
        gate_delta=0.01,
        gate_temporal_weight=0.25,
        epsilon=1e-12,
    )

    for value in components.values():
        assert float(value) == pytest.approx(0.0)


def test_frozen_scales_and_total_loss_follow_registered_coefficients():
    initial = {
        "outer": torch.tensor(2.0),
        "boundary": torch.tensor(4.0),
        "target": torch.tensor(0.5),
        "gate_aux": torch.tensor(0.25),
    }
    scales = freeze_initial_scales(initial, minimum=1e-12)
    components = {
        "outer": torch.tensor(1.0),
        "boundary": torch.tensor(1.0),
        "target": torch.tensor(0.25),
        "gate_aux": torch.tensor(0.125),
    }

    result = signed_renderer_trajectory_loss(
        components,
        scales,
        torch.tensor([[2.0, -2.0]]),
        outer_weight=1.0,
        boundary_weight=1.0,
        target_weight=0.1,
        gate_aux_weight=0.1,
        residual_weight=0.00001,
    )

    assert float(result["loss"]) == pytest.approx(0.85002)


def test_signed_trajectory_gradient_aligns_better_than_absolute_gate_weighting():
    base = torch.tensor(
        [9.5813, 12.1719, 5.2744, 13.3808], dtype=torch.float64
    )
    point = torch.tensor(
        [
            [0.5558, -1.2021, 0.3129],
            [-2.9279, -2.0738, 1.2563],
            [-0.1985, 2.3848, 1.2629],
            [2.6334, 2.1119, -2.2786],
        ],
        dtype=torch.float64,
    )
    candidate = torch.tensor(
        [
            [0.9846, 0.9695, 0.9184],
            [0.9143, 0.9306, 0.9809],
            [0.9388, 0.9541, 0.9501],
            [0.9631, 0.9435, 0.9068],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    teacher = torch.tensor(
        [
            [0.9961, 0.9572, 0.9037],
            [0.9196, 0.9942, 0.9959],
            [0.9512, 0.9362, 0.9438],
            [0.9335, 0.9500, 0.9622],
        ],
        dtype=torch.float64,
    )

    signed = signed_trajectory_component(
        base, point, candidate, teacher, delta=0.005, epsilon=1e-12
    )
    rendered = reconstruct_renderer_sequence(
        base, point, candidate, epsilon=1e-12
    )
    true_flicker = torch.mean(torch.abs(torch.diff(rendered))) / torch.abs(
        rendered.mean()
    )
    absolute_weight = torch.abs(point)
    absolute_weight /= absolute_weight.mean(dim=1, keepdim=True) + 1e-12
    proxy = torch.sum(
        absolute_weight
        * F.huber_loss(candidate, teacher, reduction="none", delta=0.01)
    ) / torch.sum(absolute_weight)
    signed_gradient = torch.autograd.grad(
        signed, candidate, retain_graph=True
    )[0]
    renderer_gradient = torch.autograd.grad(
        true_flicker, candidate, retain_graph=True
    )[0]
    proxy_gradient = torch.autograd.grad(proxy, candidate)[0]

    assert _cosine(signed_gradient, renderer_gradient) > 0.90
    assert _cosine(signed_gradient, renderer_gradient) > (
        _cosine(proxy_gradient, renderer_gradient) + 0.50
    )


def test_renderer_reconstruction_rejects_shape_nonfinite_and_zero_mean():
    with pytest.raises(ValueError, match="align"):
        reconstruct_renderer_sequence(
            torch.ones(2), torch.ones(2, 3), torch.ones(2, 2), epsilon=1e-12
        )
    bad = torch.ones(2, 2)
    bad[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        reconstruct_renderer_sequence(
            torch.ones(2), bad, torch.ones(2, 2), epsilon=1e-12
        )
    with pytest.raises(ValueError, match="mean"):
        reconstruct_renderer_sequence(
            torch.tensor([1.0, -1.0]),
            torch.zeros(2, 2),
            torch.ones(2, 2),
            epsilon=1e-12,
        )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
def test_initial_scales_reject_nonpositive_or_nonfinite_values(value):
    components = {
        name: torch.tensor(1.0)
        for name in ("outer", "boundary", "target", "gate_aux")
    }
    components["outer"] = torch.tensor(value)
    with pytest.raises(ValueError, match="scale"):
        freeze_initial_scales(components, minimum=1e-12)


def test_total_loss_rejects_missing_component_and_wrong_residual_weight():
    scales = {
        name: 1.0 for name in ("outer", "boundary", "target", "gate_aux")
    }
    components = {name: torch.tensor(1.0) for name in scales}
    del components["target"]
    with pytest.raises(ValueError, match="component"):
        signed_renderer_trajectory_loss(
            components,
            scales,
            torch.zeros(2, 2),
            outer_weight=1.0,
            boundary_weight=1.0,
            target_weight=0.1,
            gate_aux_weight=0.1,
            residual_weight=0.00001,
        )
    components["target"] = torch.tensor(1.0)
    with pytest.raises(ValueError, match="residual_weight"):
        signed_renderer_trajectory_loss(
            components,
            scales,
            torch.zeros(2, 2),
            outer_weight=1.0,
            boundary_weight=1.0,
            target_weight=0.1,
            gate_aux_weight=0.1,
            residual_weight=0.001,
        )


def test_action_diagnostics_report_rank_overlap_and_false_maximum():
    base = np.ones((4, 3))
    teacher = np.array(
        [
            [0.9, 1.0, 1.0],
            [0.9, 0.95, 1.0],
            [0.95, 0.9, 1.0],
            [1.0, 0.9, 0.95],
        ]
    )
    learned = teacher.copy()
    learned[0, 0] = 1.0

    result = summarize_action_recovery(
        learned, teacher, base, top_k=2, suppression_tolerance=0.001
    )

    assert result["missed_teacher_suppression_count"] == 1
    assert result["missed_teacher_suppression_fraction"] == pytest.approx(1.0 / 7.0)
    assert 0.0 <= result["top_k_suppression_overlap"] <= 1.0
    assert result["action_rank_90"] >= 1
    assert result["action_rank_95"] >= result["action_rank_90"]


def test_action_diagnostics_reject_undefined_or_invalid_inputs():
    base = np.ones((3, 2))
    with pytest.raises(ValueError, match="teacher action"):
        summarize_action_recovery(
            base, base, base, top_k=1, suppression_tolerance=0.001
        )
    with pytest.raises(ValueError, match="top_k"):
        summarize_action_recovery(
            base - 0.1,
            base - 0.2,
            base,
            top_k=3,
            suppression_tolerance=0.001,
        )
    bad = base.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        summarize_action_recovery(
            bad, base - 0.1, base, top_k=1, suppression_tolerance=0.001
        )


def test_action_diagnostics_define_zero_mae_share_as_zero():
    base = np.ones((3, 2))
    teacher = base - np.array([[0.1, 0.0], [0.1, 0.0], [0.1, 0.0]])

    result = summarize_action_recovery(
        teacher, teacher, base, top_k=1, suppression_tolerance=0.001
    )

    assert result["false_maximum_mae_share"] == 0.0


def test_train_fold_uses_only_fit_teacher_and_renderer_rows(tmp_path, monkeypatch):
    samples, carriers = 8, 2
    fit_mask = np.array([True] * 4 + [False] * 4)
    teacher = np.full((samples, carriers), np.nan, np.float32)
    teacher[:4] = np.array(
        [[0.93, 0.95], [0.94, 0.92], [0.92, 0.94], [0.95, 0.93]],
        np.float32,
    )
    streams = {}
    for signal, level in (("target", 20.0), ("outer", 8.0), ("boundary", 5.0)):
        stream_base = np.full(samples, np.nan, np.float32)
        stream_point = np.full((samples, carriers), np.nan, np.float32)
        stream_base[:4] = level + np.array([0.0, 1.0, -0.5, 0.5])
        stream_point[:4] = np.array(
            [[1.0, 0.5], [0.5, 1.0], [1.0, 0.25], [0.25, 1.0]]
        )
        streams[signal] = {"base": stream_base, "point": stream_point}
    features = np.linspace(-1.0, 1.0, samples * carriers * 3).reshape(
        samples, carriers, 3
    )
    pose = np.linspace(-1.0, 1.0, samples * 36).reshape(samples, 36)
    adjacency = np.repeat(np.eye(carriers)[None], samples, axis=0)
    visibility = np.ones((samples, carriers), np.float32)
    base = np.full((samples, carriers), 0.97, np.float32)
    camera = np.zeros(samples, np.int64)
    frames = np.tile(np.arange(4), 2)
    blocks = np.repeat([0, 1], 4)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract.update(
        {
            "training_epochs": 4,
            "frame_stride": 1,
            "view_embedding_dimension": 4,
            "pose_embedding_dimension": 4,
            "gru_hidden_dimension": 4,
            "maximum_fit_teacher_mae": 0.1,
        }
    )

    def fake_projection(raw, prediction_mask, *args, **kwargs):
        projected = np.full_like(raw, np.nan, dtype=np.float64)
        projected[prediction_mask] = raw[prediction_mask]
        return projected, [{"maximum_primal_violation": 0.0}]

    monkeypatch.setattr(
        "tools.train_a7c_r1_4vp_r4a_signed_renderer._project_segments",
        fake_projection,
    )
    summary = train_fold(
        fold=0,
        features=features,
        pose=pose,
        adjacency=adjacency,
        visibility=visibility,
        base_gates=base,
        teacher_gates=teacher,
        renderer_streams=streams,
        teacher_mask=fit_mask,
        prediction_mask=np.ones(8, bool),
        camera_index=camera,
        frame_index=frames,
        block_ids=blocks,
        runtime_mass=np.ones((samples, carriers), dtype=np.float32),
        a5_weight=np.ones(carriers, dtype=np.float32),
        contract=contract,
        output_dir=tmp_path,
        device="cpu",
    )

    assert summary["checkpoint_epoch"] == 4
    assert summary["final_components"]["loss"] < summary["initial_components"]["loss"]
    assert summary["held_teacher_values_accessed"] is False
    assert summary["held_renderer_values_accessed"] is False
    assert len(summary["segment_initial_scales"]) == 1
    assert set(summary["segment_initial_scales"][0]["scales"]) == {
        "outer",
        "boundary",
        "target",
        "gate_aux",
    }
    assert (tmp_path / "model.pt").is_file()
    assert (tmp_path / "predictions.npz").is_file()


def test_r4a_keeps_the_r3_model_signature_and_budget():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    model = ViewPoseResidualCompositor(
        view_dimension=49,
        view_embedding_dimension=16,
        pose_dimension=36,
        pose_embedding_dimension=16,
        gru_hidden_dimension=16,
        residual_gate_scale=0.1,
        minimum_gate=0.9,
        maximum_gate=1.0,
    )

    assert sum(value.numel() for value in model.parameters()) == 9073
    assert contract["attention"] is False
    assert contract["carrier_embedding"] is False
    assert contract["maximum_projection_gate_jump"] == 0.015


def test_r4a_audit_wrapper_preserves_negative_status(monkeypatch):
    monkeypatch.setattr(
        "tools.audit_a7c_r1_4vp_r4a_signed_renderer.r3_audit._run",
        lambda args: (
            {
                "stage": "r1_4vp_r3_crw_held_canary",
                "verdict": "CANARY_NEGATIVE",
            },
            2,
        ),
    )

    payload, status = run_r4a_audit(SimpleNamespace())

    assert payload["stage"] == "r1_4vp_r4a_signed_renderer_held_canary"
    assert status == 2


def test_r4a_runner_maps_fit_rejection_without_opening_audit():
    source = RUNNER.read_text(encoding="utf-8")
    assert "FIT_RENDERER_ENTRY_NEGATIVE) mark_terminal fit_rejected" in source
    assert "CANARY_NEGATIVE) mark_terminal rejected" in source
    assert "CANARY_PROMOTED) mark_terminal completed" in source
    assert (
        'if "${PYTHON}" "${ROOT}/tools/audit_a7c_r1_4vp_r4a_signed_renderer.py"'
        in source
    )
    assert "for marker in completed rejected fit_rejected failed" in source
