import json
from pathlib import Path

import numpy as np
import pytest
import torch

from utils.a7c_renderer_compositor import normalized_flicker as numpy_flicker
from utils.a7c_temporal_joint_projection import solve_temporal_joint_projection


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs/semantic/a7c_r1_4vp_r4b0_projection_aware_constrained_377_v1.json"
)
RUNNER = ROOT / "tools/run_a7c_r1_4vp_r4b0_projection_aware_377.sh"


def test_r4b0_contract_freezes_projection_aware_training_only():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["experiment_id"] == (
        "a7c_r1_4vp_r4b0_projection_aware_constrained_377_v1"
    )
    assert contract["status"] == "frozen"
    assert contract["projection_training_mode"] == (
        "exact_highs_straight_through"
    )
    assert contract["straight_through_forward"] == (
        "raw_plus_stop_gradient_exact_minus_raw"
    )
    assert contract["solver"] == "highs"
    assert contract["batch_unit"] == "complete_camera_block_segment"
    assert contract["scale_scope"] == (
        "global_median_over_fit_segments_at_initialization"
    )
    assert contract["scale_component_names"] == [
        "trajectory_outer",
        "trajectory_boundary",
        "gain_outer",
        "gain_boundary",
        "target",
        "gate",
        "action",
    ]
    assert contract["projection_consistency_scale"] == 0.0002
    assert contract["gain_huber_delta"] == 0.005
    assert contract["renderer_trajectory_huber_delta"] == 0.005
    assert contract["target_response_huber_delta"] == 0.005
    assert contract["gate_huber_delta"] == 0.01
    assert contract["temporal_huber_delta"] == 0.005
    assert contract["temporal_loss_weight"] == 0.25
    assert contract["action_cosine_epsilon"] == 1e-12

    assert contract["training_epochs"] == 400
    assert contract["random_seed"] == 20260805
    assert contract["optimizer"] == "AdamW"
    assert contract["learning_rate"] == 0.001
    assert contract["weight_decay"] == 0.0001
    assert contract["gradient_clip_norm"] == 1.0
    assert contract["expected_parameter_count"] == 9073
    assert contract["attention"] is False
    assert contract["carrier_embedding"] is False

    assert contract["minimum_observability_gradient_norm"] == 1e-12
    assert contract["observability_step_count"] == 1
    assert contract["maximum_fit_projected_teacher_mae"] == 0.0065
    assert contract["minimum_fit_outer_recovery"] == 0.75
    assert contract["minimum_fit_boundary_recovery"] == 0.75
    assert contract["minimum_fit_positive_segment_fraction"] == 0.95
    assert contract["minimum_fit_action_cosine"] == 0.90
    assert contract["minimum_fit_top_k_overlap"] == 0.45
    assert contract["maximum_fit_missed_suppression_fraction"] == 0.55
    assert contract["maximum_fit_raw_to_exact_mae"] == 0.0002
    assert contract["maximum_fit_projection_changed_fraction"] == 0.05
    assert contract["projection_changed_threshold"] == 1e-12

    assert contract["fit_cameras"] == ["c01", "c05", "c09", "c13"]
    assert contract["audit_cameras"] == []
    assert contract["forbidden_cameras"] == [
        "c17", "c18", "c19", "c20", "c21", "c22", "c23"
    ]
    assert contract["open_held_after_fit_fold_count"] == 6
    assert contract["observability_negative_status"] == (
        "FEATURE_OBSERVABILITY_NEGATIVE"
    )
    assert contract["fit_negative_status"] == "FIT_PROJECTED_ENTRY_NEGATIVE"
    assert contract["deployment_eligible"] is False
    assert contract["teacher_eligible"] is False
    assert contract["paper_test_eligible"] is False

    assert contract["source_design_sha256"] == (
        "3e71d92496e5d21d3ec2235c683857e260e162ba6e0923c64eb8c96aa907f704"
    )
    assert contract["source_r4a_contract_sha256"] == (
        "d397642d86013eae446c5af1484cfb3f4d537dc79f417c3657fd1e9fc9ddd9e7"
    )
    assert contract["source_r4a_policy_sha256"] == (
        "74fa9528b7683364b1b4dd5a767be16f26c66f80d5d534ec3103bcf63b35bf7f"
    )
    assert contract["source_r4a_trainer_sha256"] == (
        "f627c5147dd912344307568fc74e3b8f403877b8bcd838c2b354065d99082ff3"
    )
    assert contract["source_r4a_auditor_sha256"] == (
        "d8ef2bc7a93bcee8abb3b2bfdc4e808e9af4fb7c2954a5c1848801f21d64d69b"
    )
    assert contract["source_r4a_runner_sha256"] == (
        "ab4d06f4401d50f8c1546fabd9cc033b338516cf141d77cc3bc4734e5292a9ff"
    )


def _projection_contract():
    return {
        "minimum_gate": 0.9,
        "maximum_gate": 1.0,
        "selection_threshold": 0.2,
        "proxy_target_response": 0.995,
        "maximum_projection_gate_jump": 0.015,
        "lexicographic_tolerance": 1e-9,
        "solver_primal_tolerance": 1e-9,
        "solver_residual_tolerance": 1e-7,
    }


def _streams(dtype=torch.float64):
    return {
        "target": {
            "base": torch.tensor([10.0, 10.5, 9.5], dtype=dtype),
            "point": torch.tensor(
                [[1.0, -0.5], [0.5, -1.0], [1.5, -0.25]], dtype=dtype
            ),
        },
        "outer": {
            "base": torch.tensor([8.0, 10.0, 7.0], dtype=dtype),
            "point": torch.tensor(
                [[1.0, -2.0], [2.0, 1.0], [-1.0, 2.0]], dtype=dtype
            ),
        },
        "boundary": {
            "base": torch.tensor([7.0, 8.0, 6.0], dtype=dtype),
            "point": torch.tensor(
                [[0.5, -1.0], [1.5, 0.5], [-0.5, 1.0]], dtype=dtype
            ),
        },
    }


def test_exact_projection_straight_through_has_exact_forward_and_identity_backward():
    from utils.a7c_r1_4vp_r4b0 import exact_projected_straight_through

    raw = torch.tensor(
        [[0.91, 0.99], [1.0, 0.90], [0.92, 0.98]],
        dtype=torch.float64,
        requires_grad=True,
    )
    mass = np.ones((3, 2), dtype=np.float64)
    weight = np.ones(2, dtype=np.float64)
    direct = solve_temporal_joint_projection(
        raw_gates=raw.detach().numpy(),
        runtime_mass=mass,
        a5_weight=weight,
        minimum_gate=0.9,
        maximum_gate=1.0,
        selection_threshold=0.2,
        proxy_target_response=0.995,
        maximum_gate_jump=0.015,
        rho_tolerance=1e-9,
        primal_tolerance=1e-9,
        residual_tolerance=1e-7,
    )

    deployed, certificate = exact_projected_straight_through(
        raw, mass, weight, _projection_contract()
    )

    torch.testing.assert_close(
        deployed, torch.as_tensor(direct["gates"], dtype=torch.float64)
    )
    assert certificate == direct["certificate"]
    deployed.sum().backward()
    torch.testing.assert_close(raw.grad, torch.ones_like(raw))


def test_differentiable_flicker_and_gain_match_exact_numpy_evaluator():
    from utils.a7c_r1_4vp_r4b0 import normalized_flicker, renderer_gain

    values = torch.tensor([2.0, 3.0, 2.5, 4.0], dtype=torch.float64)
    edited = torch.tensor([2.0, 2.7, 2.6, 3.1], dtype=torch.float64)

    assert float(normalized_flicker(values, epsilon=1e-12)) == pytest.approx(
        numpy_flicker(values.numpy()), abs=1e-15
    )
    expected = 1.0 - numpy_flicker(edited.numpy()) / max(
        numpy_flicker(values.numpy()), 1e-12
    )
    assert float(renderer_gain(values, edited, epsilon=1e-12)) == pytest.approx(
        expected, abs=1e-15
    )


def test_projection_aware_components_use_deployed_gate_except_projection():
    from utils.a7c_r1_4vp_r4b0 import projection_aware_components

    teacher = torch.tensor(
        [[0.99, 0.98], [0.98, 0.99], [0.99, 0.97]], dtype=torch.float64
    )
    deployed = torch.tensor(
        [[0.98, 0.99], [0.99, 0.98], [0.98, 0.98]], dtype=torch.float64
    )
    base = torch.ones_like(teacher)
    raw_a = deployed.clone()
    raw_b = torch.full_like(deployed, 0.9)

    first = projection_aware_components(
        deployed, raw_a, teacher, base, _streams(),
        trajectory_delta=0.005,
        gain_delta=0.005,
        target_delta=0.005,
        gate_delta=0.01,
        temporal_delta=0.005,
        temporal_weight=0.25,
        projection_scale=0.0002,
        epsilon=1e-12,
    )
    second = projection_aware_components(
        deployed, raw_b, teacher, base, _streams(),
        trajectory_delta=0.005,
        gain_delta=0.005,
        target_delta=0.005,
        gate_delta=0.01,
        temporal_delta=0.005,
        temporal_weight=0.25,
        projection_scale=0.0002,
        epsilon=1e-12,
    )

    assert set(first) == {
        "trajectory_outer", "trajectory_boundary", "gain_outer",
        "gain_boundary", "target", "gate", "action", "projection"
    }
    for name in set(first) - {"projection"}:
        torch.testing.assert_close(first[name], second[name])
    assert float(second["projection"]) > float(first["projection"])


def test_action_component_is_one_for_zero_candidate_action():
    from utils.a7c_r1_4vp_r4b0 import projection_aware_components

    base = torch.ones((3, 2), dtype=torch.float64)
    teacher = base - torch.tensor(
        [[0.01, 0.02], [0.02, 0.01], [0.01, 0.03]], dtype=torch.float64
    )
    components = projection_aware_components(
        base, base, teacher, base, _streams(),
        trajectory_delta=0.005,
        gain_delta=0.005,
        target_delta=0.005,
        gate_delta=0.01,
        temporal_delta=0.005,
        temporal_weight=0.25,
        projection_scale=0.0002,
        epsilon=1e-12,
    )
    assert float(components["action"]) == pytest.approx(1.0)


def test_global_median_scales_and_grouped_total_follow_registered_formula():
    from utils.a7c_r1_4vp_r4b0 import (
        freeze_global_median_scales,
        projection_aware_loss,
    )

    names = [
        "trajectory_outer", "trajectory_boundary", "gain_outer",
        "gain_boundary", "target", "gate", "action"
    ]
    rows = [
        {name: torch.tensor(float(index + offset)) for name in names}
        for index, offset in ((1, 0), (2, 1), (3, 2))
    ]
    scales = freeze_global_median_scales(rows, minimum=1e-12)
    assert scales == {name: 3.0 for name in names}
    components = {name: torch.tensor(3.0) for name in names}
    components["projection"] = torch.tensor(2.0)
    result = projection_aware_loss(
        components, scales, torch.tensor([[2.0, -2.0]]),
        residual_weight=1e-5,
    )
    assert float(result["renderer_loss"]) == pytest.approx(1.0)
    assert float(result["preservation_loss"]) == pytest.approx(1.25)
    assert float(result["loss"]) == pytest.approx(2.25002)


@pytest.mark.parametrize("bad", [0.0, float("nan")])
def test_global_median_scales_reject_zero_or_nonfinite_values(bad):
    from utils.a7c_r1_4vp_r4b0 import freeze_global_median_scales

    names = [
        "trajectory_outer", "trajectory_boundary", "gain_outer",
        "gain_boundary", "target", "gate", "action"
    ]
    rows = [{name: torch.tensor(1.0) for name in names} for _ in range(3)]
    rows[1]["gain_outer"] = torch.tensor(bad)
    with pytest.raises(ValueError, match="scale"):
        freeze_global_median_scales(rows, minimum=1e-12)


def test_projection_diagnostics_and_fit_entry_are_fail_closed():
    from utils.a7c_r1_4vp_r4b0 import (
        evaluate_fit_projected_entry,
        projection_diagnostics,
    )

    diagnostics = projection_diagnostics(
        np.ones((2, 2)), np.array([[1.0, 0.999], [1.0, 1.0]]),
        changed_threshold=1e-12,
    )
    assert diagnostics["raw_to_exact_mean_absolute_displacement"] == pytest.approx(
        0.00025
    )
    assert diagnostics["raw_to_exact_changed_fraction"] == pytest.approx(0.25)

    summary = {
        "fit_loss_improved": True,
        "fit_projected_teacher_mae": 0.006,
        "fit_outer_recovery": 0.80,
        "fit_boundary_recovery": 0.80,
        "fit_outer_positive_segment_fraction": 1.0,
        "fit_boundary_positive_segment_fraction": 1.0,
        "projected_action_diagnostics": {
            "action_cosine": 0.95,
            "top_k_suppression_overlap": 0.50,
            "missed_teacher_suppression_fraction": 0.50,
        },
        "raw_to_exact_mean_absolute_displacement": 0.0001,
        "raw_to_exact_changed_fraction": 0.04,
        "projection_certificates_passed": True,
        "held_teacher_values_accessed": False,
        "held_renderer_values_accessed": False,
    }
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    positive = evaluate_fit_projected_entry(summary, contract)
    assert positive["passed"] is True
    assert positive["failure_reasons"] == []

    for key in (
        "fit_loss_improved",
        "projection_certificates_passed",
    ):
        rejected_summary = dict(summary)
        rejected_summary[key] = False
        rejected = evaluate_fit_projected_entry(rejected_summary, contract)
        assert rejected["passed"] is False
        assert key in rejected["failure_reasons"]
    rejected_summary = dict(summary)
    rejected_summary["held_teacher_values_accessed"] = True
    rejected = evaluate_fit_projected_entry(rejected_summary, contract)
    assert rejected["passed"] is False
    assert "held_teacher_values_accessed" in rejected["failure_reasons"]


class _ScalarModel(torch.nn.Module):
    def __init__(self, value=0.2):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value)))


def _observability_rows(*, expose_held=False):
    teacher = np.array([[0.98], [0.99], [np.nan], [np.nan]])
    renderer = {
        signal: {
            "base": np.array([1.0, 1.0, np.nan, np.nan]),
            "point": np.array([[0.1], [0.1], [np.nan], [np.nan]]),
        }
        for signal in ("target", "outer", "boundary")
    }
    if expose_held:
        teacher[-1] = 1.0
    return teacher, renderer, np.array([True, True, False, False])


def _observable_loss(model, *, certificate=True, nonfinite_component=False):
    loss = (model.value - 0.5).square()
    component = loss
    if nonfinite_component:
        component = loss * torch.tensor(float("nan"))
    return {
        "loss": loss,
        "components": {"registered": component},
        "projection_certificates_passed": certificate,
    }


def test_observability_preflight_is_positive_and_does_not_mutate_source_model():
    from utils.a7c_r1_4vp_r4b0 import run_gradient_observability_preflight

    model = _ScalarModel()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    teacher, renderer, fit_mask = _observability_rows()

    result = run_gradient_observability_preflight(
        model,
        lambda clone: _observable_loss(clone),
        teacher_values=teacher,
        renderer_streams=renderer,
        fit_mask=fit_mask,
        learning_rate=0.001,
        weight_decay=0.0001,
        minimum_gradient_norm=1e-12,
        step_count=1,
    )

    assert result["verdict"] == "FEATURE_OBSERVABILITY_POSITIVE"
    assert result["passed"] is True
    assert result["final_loss"] < result["initial_loss"]
    assert result["gradient_norm"] > 1e-12
    assert result["failure_reasons"] == []
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "case,model_factory,closure_factory,expose_held,reason",
    [
        (
            "certificate",
            lambda: _ScalarModel(),
            lambda: (lambda model: _observable_loss(model, certificate=False)),
            False,
            "projection_certificates_passed",
        ),
        (
            "component",
            lambda: _ScalarModel(),
            lambda: (
                lambda model: _observable_loss(model, nonfinite_component=True)
            ),
            False,
            "components_finite",
        ),
        (
            "gradient",
            lambda: _ScalarModel(0.0),
            lambda: (
                lambda model: {
                    "loss": torch.sqrt(model.value),
                    "components": {"registered": model.value.square()},
                    "projection_certificates_passed": True,
                }
            ),
            False,
            "gradients_finite",
        ),
        (
            "zero_gradient",
            lambda: _ScalarModel(0.2),
            lambda: (
                lambda model: {
                    "loss": model.value * 0.0 + 1.0,
                    "components": {"registered": model.value * 0.0 + 1.0},
                    "projection_certificates_passed": True,
                }
            ),
            False,
            "gradient_observable",
        ),
        (
            "no_decrease",
            lambda: _ScalarModel(0.0001),
            lambda: (
                lambda model: {
                    "loss": model.value.square(),
                    "components": {"registered": model.value.square()},
                    "projection_certificates_passed": True,
                }
            ),
            False,
            "ephemeral_step_decreased_loss",
        ),
        (
            "held",
            lambda: _ScalarModel(),
            lambda: (lambda model: _observable_loss(model)),
            True,
            "held_rows_inaccessible",
        ),
    ],
)
def test_observability_preflight_fails_closed(
    case, model_factory, closure_factory, expose_held, reason
):
    del case
    from utils.a7c_r1_4vp_r4b0 import run_gradient_observability_preflight

    teacher, renderer, fit_mask = _observability_rows(expose_held=expose_held)
    result = run_gradient_observability_preflight(
        model_factory(),
        closure_factory(),
        teacher_values=teacher,
        renderer_streams=renderer,
        fit_mask=fit_mask,
        learning_rate=0.001,
        weight_decay=0.0001,
        minimum_gradient_norm=1e-12,
        step_count=1,
    )

    assert result["verdict"] == "FEATURE_OBSERVABILITY_NEGATIVE"
    assert result["passed"] is False
    assert reason in result["failure_reasons"]


def _tiny_fold_inputs(tmp_path):
    samples, carriers = 8, 2
    fit_mask = np.array([True] * 4 + [False] * 4)
    teacher = np.full((samples, carriers), np.nan, np.float32)
    teacher[:4] = np.array(
        [[0.996, 0.999], [0.998, 0.996], [0.997, 0.999], [0.999, 0.996]],
        np.float32,
    )
    streams = {}
    for signal, level, multiplier in (
        ("target", 20.0, 0.8),
        ("outer", 8.0, 1.0),
        ("boundary", 5.0, 1.2),
    ):
        stream_base = np.full(samples, np.nan, np.float32)
        stream_point = np.full((samples, carriers), np.nan, np.float32)
        stream_base[:4] = level + np.array([0.0, 1.0, -0.6, 0.7])
        stream_point[:4] = multiplier * np.array(
            [[1.0, -0.4], [0.3, 1.1], [-0.8, 0.4], [0.6, -1.0]]
        )
        streams[signal] = {"base": stream_base, "point": stream_point}
    features = np.linspace(-1.0, 1.0, samples * carriers * 3).reshape(
        samples, carriers, 3
    )
    pose = np.linspace(-1.0, 1.0, samples * 36).reshape(samples, 36)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract.update(
        {
            "training_epochs": 2,
            "frame_stride": 1,
            "view_embedding_dimension": 4,
            "pose_embedding_dimension": 4,
            "gru_hidden_dimension": 4,
        }
    )
    return {
        "fold": 0,
        "features": features,
        "pose": pose,
        "adjacency": np.repeat(np.eye(carriers)[None], samples, axis=0),
        "visibility": np.ones((samples, carriers), np.float32),
        "base_gates": np.ones((samples, carriers), np.float32),
        "teacher_gates": teacher,
        "renderer_streams": streams,
        "teacher_mask": fit_mask,
        "prediction_mask": np.ones(samples, bool),
        "camera_index": np.zeros(samples, np.int64),
        "frame_index": np.tile(np.arange(4), 2),
        "block_ids": np.repeat([0, 1], 4),
        "runtime_mass": np.ones((samples, carriers), np.float32),
        "a5_weight": np.ones(carriers, np.float32),
        "contract": contract,
        "output_dir": tmp_path,
        "device": "cpu",
    }


def test_r4b0_trainer_uses_global_scales_and_saves_exact_deployed_gates(
    tmp_path, monkeypatch
):
    from tools.train_a7c_r1_4vp_r4b0_projection_aware import train_fold

    observed = {"calls": 0}

    def positive_preflight(*args, **kwargs):
        del args, kwargs
        observed["calls"] += 1
        return {
            "verdict": "FEATURE_OBSERVABILITY_POSITIVE",
            "passed": True,
            "initial_loss": 2.0,
            "final_loss": 1.9,
            "gradient_norm": 1.0,
            "checks": {},
            "failure_reasons": [],
            "held_teacher_values_accessed": False,
            "held_renderer_values_accessed": False,
        }

    monkeypatch.setattr(
        "tools.train_a7c_r1_4vp_r4b0_projection_aware."
        "run_gradient_observability_preflight",
        positive_preflight,
    )
    summary = train_fold(**_tiny_fold_inputs(tmp_path))

    assert observed["calls"] == 1
    assert summary["checkpoint_epoch"] == 2
    assert summary["observability"]["passed"] is True
    assert set(summary["global_initial_scales"]) == {
        "trajectory_outer", "trajectory_boundary", "gain_outer",
        "gain_boundary", "target", "gate", "action"
    }
    assert "segment_initial_scales" not in summary
    assert summary["held_teacher_values_accessed"] is False
    assert summary["held_renderer_values_accessed"] is False
    assert summary["optimizer_signature"] == {
        "name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001
    }
    assert (tmp_path / "model.pt").is_file()
    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as predictions:
        assert "raw_gates" in predictions.files
        assert "exact_gates" in predictions.files
        assert "projected_gates" in predictions.files
        np.testing.assert_array_equal(
            predictions["exact_gates"], predictions["projected_gates"]
        )
    certificates = json.loads(
        (tmp_path / "projection_certificates.json").read_text(encoding="utf-8")
    )
    assert certificates


def test_r4b0_trainer_stops_before_epoch_one_when_observability_is_negative(
    tmp_path, monkeypatch
):
    from tools.train_a7c_r1_4vp_r4b0_projection_aware import train_fold

    monkeypatch.setattr(
        "tools.train_a7c_r1_4vp_r4b0_projection_aware."
        "run_gradient_observability_preflight",
        lambda *args, **kwargs: {
            "verdict": "FEATURE_OBSERVABILITY_NEGATIVE",
            "passed": False,
            "failure_reasons": ["gradient_observable"],
        },
    )
    summary = train_fold(**_tiny_fold_inputs(tmp_path))

    assert summary["execution_status"] == "FEATURE_OBSERVABILITY_NEGATIVE"
    assert not (tmp_path / "model.pt").exists()
    assert not (tmp_path / "predictions.npz").exists()
    assert (tmp_path / "observability.json").is_file()


def test_r4b0_model_signature_remains_exactly_9073_parameters():
    from utils.a7c_r1_4vp_r2_runtime import ViewPoseResidualCompositor

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
