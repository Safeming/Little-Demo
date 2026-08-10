import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

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
    assert contract["fit_camera_indices"] == [0, 1, 2, 3]
    assert contract["fit_only_expected_sample_count"] == 456
    assert contract["source_expected_sample_count"] == 912
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
        "d1f69c1fe6ddbb2c201ce29efd3998b61a406d02d80f64dc20b488525565023c"
    )
    assert contract["source_r4b0_policy_sha256"] == (
        "6c876a5930a2d1a4d14f634f55a00cd9c5e1b369628f4f2a96989f28b16028e7"
    )
    assert contract["source_r4b0_trainer_sha256"] == (
        "e4b7c4bdb6459f7d297ab483c3272a6270cbe5cd3e2e5531d917506604d1ffbe"
    )
    assert contract["source_r4b0_auditor_sha256"] == (
        "6dd39410ed9780607849a1380711d60ef9b84bcbf15b44b9dc740052cc707ff8"
    )
    assert contract["source_r4b0_stager_sha256"] == (
        "3b75c9889e7ebcf3dc12e0d0e9411602faf8c93de1cc3f9d30983b36d2d6ac21"
    )
    assert contract["source_fit_only_manifest_sha256"] == (
        "f8688473ea442b1f5d2a93e85bca8d91e67cfff2395acad7df7af6b30f7c87f6"
    )
    fit_manifest = json.loads(
        (
            ROOT
            / "exp/acceptdata/a7c_r1_4vp_r4b0_fit_only_inputs_377_v1/manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["source_fit_only_artifact_sha256"] == fit_manifest[
        "artifact_sha256"
    ]
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


def test_global_median_scales_reject_nonfinite_segment_values():
    from utils.a7c_r1_4vp_r4b0 import freeze_global_median_scales

    names = [
        "trajectory_outer", "trajectory_boundary", "gain_outer",
        "gain_boundary", "target", "gate", "action"
    ]
    rows = [{name: torch.tensor(1.0) for name in names} for _ in range(3)]
    rows[1]["gain_outer"] = torch.tensor(float("nan"))
    with pytest.raises(ValueError, match="scale"):
        freeze_global_median_scales(rows, minimum=1e-12)


def test_global_median_scales_reject_nonpositive_global_median():
    from utils.a7c_r1_4vp_r4b0 import freeze_global_median_scales

    names = [
        "trajectory_outer", "trajectory_boundary", "gain_outer",
        "gain_boundary", "target", "gate", "action"
    ]
    rows = [{name: torch.tensor(0.0) for name in names} for _ in range(3)]
    with pytest.raises(ValueError, match="median scale"):
        freeze_global_median_scales(rows, minimum=1e-12)


def test_global_median_scales_allow_finite_zero_segments_when_median_is_positive():
    from utils.a7c_r1_4vp_r4b0 import freeze_global_median_scales

    names = [
        "trajectory_outer", "trajectory_boundary", "gain_outer",
        "gain_boundary", "target", "gate", "action"
    ]
    rows = [{name: torch.tensor(value) for name in names} for value in (0.0, 2.0, 4.0)]

    scales = freeze_global_median_scales(rows, minimum=1e-12)

    assert scales == {name: 2.0 for name in names}


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
        "source_row_indices": np.arange(samples, dtype=np.int64),
        "source_sample_count": samples + 4,
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
        assert predictions["raw_gates"].shape == (12, 2)
        assert np.isnan(predictions["raw_gates"][8:]).all()
        assert np.isnan(predictions["exact_gates"][8:]).all()
        np.testing.assert_array_equal(
            predictions["prediction_mask"],
            np.array([True] * 8 + [False] * 4),
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


def test_r4b0_audit_wrapper_preserves_inherited_status(monkeypatch):
    from tools.audit_a7c_r1_4vp_r4b0_projection_aware import _run

    monkeypatch.setattr(
        "tools.audit_a7c_r1_4vp_r4b0_projection_aware.r4a_audit._run",
        lambda args: (
            {"stage": "r1_4vp_r4a_signed_renderer_held_canary",
             "verdict": "CANARY_NEGATIVE"},
            2,
        ),
    )
    payload, status = _run(SimpleNamespace())
    assert payload["stage"] == "r1_4vp_r4b0_projection_aware_held_canary"
    assert payload["verdict"] == "CANARY_NEGATIVE"
    assert status == 2


def _write_r4b0_frozen_manifest(root):
    names = (
        "model.pt", "predictions.npz", "projection_certificates.json",
        "observability.json", "summary.json", "fit_projected_entry.json",
    )
    artifacts = {}
    for fold in range(6):
        for name in names:
            relative = Path("training") / f"fold_{fold}" / name
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{fold}:{name}".encode("ascii"))
            artifacts[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "models_frozen.json").write_text(
        json.dumps({"artifacts": artifacts}), encoding="utf-8"
    )
    return artifacts


def test_r4b0_frozen_artifact_verifier_accepts_exact_36_file_schema(tmp_path):
    from tools.audit_a7c_r1_4vp_r4b0_projection_aware import (
        verify_frozen_artifacts,
    )

    expected = _write_r4b0_frozen_manifest(tmp_path)
    assert verify_frozen_artifacts(tmp_path) == expected


@pytest.mark.parametrize("mutation", ["missing", "changed", "extra"])
def test_r4b0_frozen_artifact_verifier_rejects_schema_or_hash_changes(
    tmp_path, mutation
):
    from tools.audit_a7c_r1_4vp_r4b0_projection_aware import (
        verify_frozen_artifacts,
    )

    artifacts = _write_r4b0_frozen_manifest(tmp_path)
    first = next(iter(artifacts))
    if mutation == "missing":
        del artifacts[first]
    elif mutation == "changed":
        (tmp_path / first).write_bytes(b"changed")
    else:
        artifacts["training/extra.json"] = hashlib.sha256(b"extra").hexdigest()
    if mutation != "changed":
        (tmp_path / "models_frozen.json").write_text(
            json.dumps({"artifacts": artifacts}), encoding="utf-8"
        )

    with pytest.raises(ValueError, match="artifact|36"):
        verify_frozen_artifacts(tmp_path)


def test_r4b0_runner_routes_fit_failures_without_opening_held_audit():
    source = RUNNER.read_text(encoding="utf-8")

    assert "FEATURE_OBSERVABILITY_NEGATIVE) mark_terminal observability_rejected" in source
    assert "FIT_PROJECTED_ENTRY_NEGATIVE) mark_terminal fit_rejected" in source
    assert "CANARY_NEGATIVE) mark_terminal rejected" in source
    assert "CANARY_PROMOTED) mark_terminal completed" in source
    assert "for marker in completed rejected observability_rejected fit_rejected failed" in source
    assert "fit_projected_entry.json" in source
    assert "observability.json" in source
    assert "source_fingerprints.json" in source
    assert "--fit-input-manifest" in source
    assert "a7c_r1_4vp_r4b0_fit_only_inputs_377_v1" in source
    assert "R4-B0 must freeze exactly 36 training artifacts" in (
        ROOT / "tools/train_a7c_r1_4vp_r4b0_projection_aware.py"
    ).read_text(encoding="utf-8")
    trainer_offset = source.rindex("train_a7c_r1_4vp_r4b0_projection_aware.py")
    audit_offset = source.rindex("audit_a7c_r1_4vp_r4b0_projection_aware.py")
    frozen_guard_offset = source.index('[[ -f "${OUT}/models_frozen.json" ]]')
    assert trainer_offset < frozen_guard_offset < audit_offset


def test_r4b0_runner_uses_frozen_contract_and_mutually_exclusive_markers():
    source = RUNNER.read_text(encoding="utf-8")
    assert (
        "configs/semantic/a7c_r1_4vp_r4b0_projection_aware_constrained_377_v1.json"
        in source
    )
    assert (
        "a7c_r1_4vp_r4b0_projection_aware_constrained_377_v1" in source
    )
    assert (
        'rm -f "${OUT}/.completed" "${OUT}/.rejected" '
        '"${OUT}/.observability_rejected" "${OUT}/.fit_rejected" '
        '"${OUT}/.failed"'
        in source
    )


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _make_tiny_full_input_tree(root):
    from utils.a7c_oracle_capacity import _artifact_fingerprint

    cameras = np.arange(8, dtype=np.int16)
    frames = np.zeros(8, dtype=np.int32)
    carriers = np.array([2, 5], dtype=np.int64)
    probe = {
        "schema_version": np.array(1, np.int64),
        "features": np.arange(16, dtype=np.float32).reshape(8, 2, 1),
        "feature_names": np.array(["visibility"]),
        "carrier_ids": carriers,
        "camera_index": cameras,
        "frame_index": frames,
        "source_teacher_fingerprint": np.array("parent"),
        "paper_test_eligible": np.array(0, np.uint8),
    }
    probe["output_fingerprint"] = np.array(_artifact_fingerprint(probe))
    probe_path = root / "probe.npz"
    np.savez_compressed(probe_path, **probe)
    teacher_path = root / "teacher.npz"
    np.savez_compressed(
        teacher_path,
        carrier_ids=carriers,
        camera_index=cameras,
        frame_index=frames,
        output_fingerprint=np.array("parent"),
    )
    evidence_path = root / "evidence.npz"
    evidence = {
        "renderer_sequence_camera_index": cameras,
        "renderer_sequence_frame_index": frames,
    }
    for signal_index, signal in enumerate(("target", "outer", "boundary")):
        evidence[f"renderer_{signal}_contribution_sequence"] = (
            np.arange(8 * 6, dtype=np.float32).reshape(8, 3, 2) + signal_index
        )
    np.savez_compressed(evidence_path, **evidence)
    base_root = root / "base"
    teachers_root = root / "teachers"
    base_hashes = []
    teacher_artifacts = {}
    for fold in range(6):
        base_path = base_root / f"fold_{fold}/predictions.npz"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            base_path, raw_gates=np.full((8, 2), 0.99 - fold * 0.001)
        )
        base_hashes.append(_sha256(base_path))
        fold_path = teachers_root / f"fold_{fold}/teacher.npz"
        fold_path.parent.mkdir(parents=True, exist_ok=True)
        gates = np.full((8, 2), np.nan)
        gates[:4] = 0.995
        np.savez_compressed(
            fold_path,
            teacher_gates=gates,
            teacher_mask=np.isfinite(gates).all(axis=1),
            camera_index=cameras,
            frame_index=frames,
            block_ids=np.zeros(8, np.int16),
            carrier_ids=carriers,
        )
        teacher_artifacts[f"fold_{fold}/teacher.npz"] = _sha256(fold_path)
    contract = {
        "fit_camera_indices": [0, 1, 2, 3],
        "fit_only_expected_sample_count": 4,
        "source_expected_sample_count": 8,
        "source_probe_sha256": _sha256(probe_path),
        "source_teacher_sha256": _sha256(teacher_path),
        "source_evidence_sha256": _sha256(evidence_path),
        "source_r1_2b_prediction_sha256": base_hashes,
        "source_teacher_artifacts": teacher_artifacts,
    }
    return contract, probe_path, teacher_path, evidence_path, base_root, teachers_root


def test_fit_only_staging_is_deterministic_and_contains_no_forbidden_camera(tmp_path):
    from tools.stage_a7c_r1_4vp_r4b0_fit_inputs import stage_fit_only_inputs

    source_root = tmp_path / "source"
    source_root.mkdir()
    values = _make_tiny_full_input_tree(source_root)
    contract, probe, teacher, evidence, base, teachers = values
    first = tmp_path / "first"
    second = tmp_path / "second"
    arguments = dict(
        contract=contract,
        probe_path=probe,
        teacher_path=teacher,
        evidence_path=evidence,
        r1_2b_training_dir=base,
        teachers_dir=teachers,
    )

    manifest_a = stage_fit_only_inputs(output_dir=first, **arguments)
    manifest_b = stage_fit_only_inputs(output_dir=second, **arguments)

    assert manifest_a["artifact_sha256"] == manifest_b["artifact_sha256"]
    assert manifest_a["source_row_indices"] == [0, 1, 2, 3]
    assert manifest_a["fit_camera_indices"] == [0, 1, 2, 3]
    with np.load(first / "probe/probe.npz", allow_pickle=False) as staged:
        assert staged["features"].shape[0] == 4
        assert set(map(int, np.unique(staged["camera_index"]))) == {0, 1, 2, 3}
    with np.load(first / "evidence/evidence.npz", allow_pickle=False) as staged:
        assert set(map(int, np.unique(staged["renderer_sequence_camera_index"]))) == {
            0, 1, 2, 3
        }
        assert all(
            staged[key].shape[0] == 4
            for key in staged.files
            if key.startswith("renderer_") and key.endswith("_sequence")
        )


def test_trainer_rejects_a_forbidden_camera_in_fit_only_manifest():
    from tools.train_a7c_r1_4vp_r4b0_projection_aware import (
        validate_fit_only_training_manifest,
    )

    with pytest.raises(ValueError, match="fit-only camera"):
        validate_fit_only_training_manifest(
            np.array([0, 1, 2, 4]),
            fit_camera_indices=[0, 1, 2, 3],
            expected_sample_count=4,
        )


def test_fit_predictions_expand_to_source_order_without_held_values():
    from tools.train_a7c_r1_4vp_r4b0_projection_aware import (
        expand_fit_predictions,
    )

    values = np.array([[0.91, 0.92], [0.93, 0.94]])
    expanded, mask = expand_fit_predictions(
        values, source_row_indices=np.array([0, 2]), source_sample_count=4
    )

    np.testing.assert_array_equal(expanded[[0, 2]], values)
    assert np.isnan(expanded[[1, 3]]).all()
    np.testing.assert_array_equal(mask, np.array([True, False, True, False]))


def test_trainer_verifies_fit_only_manifest_and_every_staged_artifact(tmp_path):
    from tools.stage_a7c_r1_4vp_r4b0_fit_inputs import stage_fit_only_inputs
    from tools.train_a7c_r1_4vp_r4b0_projection_aware import verify_fit_only_bundle

    source_root = tmp_path / "source"
    source_root.mkdir()
    contract, probe, teacher, evidence, base, teachers = _make_tiny_full_input_tree(
        source_root
    )
    staged = tmp_path / "staged"
    manifest = stage_fit_only_inputs(
        contract=contract,
        probe_path=probe,
        teacher_path=teacher,
        evidence_path=evidence,
        r1_2b_training_dir=base,
        teachers_dir=teachers,
        output_dir=staged,
    )
    contract.update({
        "source_fit_only_manifest_sha256": _sha256(staged / "manifest.json"),
        "source_fit_only_artifact_sha256": manifest["artifact_sha256"],
    })

    verified = verify_fit_only_bundle(staged / "manifest.json", contract)
    assert verified["source_row_indices"] == [0, 1, 2, 3]

    first = staged / next(iter(manifest["artifact_sha256"]))
    first.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="fit-only artifact"):
        verify_fit_only_bundle(staged / "manifest.json", contract)


def test_trainer_derives_every_consumed_input_from_fit_manifest(tmp_path):
    from tools.train_a7c_r1_4vp_r4b0_projection_aware import (
        fit_only_bundle_paths,
        parse_args,
    )

    manifest = tmp_path / "fit/manifest.json"
    paths = fit_only_bundle_paths(manifest)
    assert paths == {
        "probe": tmp_path / "fit/probe/probe.npz",
        "teacher": tmp_path / "fit/teacher/teacher.npz",
        "evidence": tmp_path / "fit/evidence/evidence.npz",
        "r1_2b_training_dir": tmp_path / "fit/training",
        "teachers_dir": tmp_path / "fit/teachers",
    }
    args = parse_args([
        "--contract", str(tmp_path / "contract.json"),
        "--fit-input-manifest", str(manifest),
        "--a5-bank", str(tmp_path / "bank.npz"),
        "--pose-model-dir", str(tmp_path / "poses"),
        "--output-dir", str(tmp_path / "output"),
        "--device", "cpu",
    ])
    assert args.fit_input_manifest == manifest


def test_runner_preflights_before_creating_output_and_never_marks_live_run_failed():
    source = RUNNER.read_text(encoding="utf-8")
    mkdir_offset = source.index('mkdir -p "${OUT}"')
    prefix = source[:mkdir_offset]

    assert "git diff --quiet HEAD" in prefix
    assert "expected_contract =" in prefix
    assert "pose_manifest_sha256" in prefix
    assert "torch.cuda.is_available" in prefix
    assert source.count("\nmark_terminal failed\n") == 0
