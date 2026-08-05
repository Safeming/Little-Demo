import json
import inspect
import re
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs/semantic/a7c_r1_4vp_oracle_distilled_view_pose_377_v1.json"
)


def test_r1_4vp_contract_freezes_model_isolation_and_promotion():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert (
        contract["experiment_id"]
        == "a7c_r1_4vp_oracle_distilled_view_pose_377_v1"
    )
    assert contract["status"] == "frozen"
    assert contract["fit_cameras"] == ["c01", "c05", "c09", "c13"]
    assert contract["audit_cameras"] == []
    assert contract["forbidden_cameras"] == [
        "c17",
        "c18",
        "c19",
        "c20",
        "c21",
        "c22",
        "c23",
    ]
    assert contract["temporal_block_count"] == 6
    assert contract["fit_teacher_segment_count"] == 120
    assert contract["held_audit_record_count"] == 24
    assert contract["processed_parts"] == ["lower"]
    assert contract["frozen_parts"] == [
        "hair",
        "face",
        "upper",
        "shoes",
        "skin",
    ]
    assert contract["min_pair_support"] == 8
    assert contract["minimum_evidence_support_coverage"] == 0.8
    assert contract["pose_body_joint_indices"] == [0, 1, 3, 4, 6, 7]
    assert contract["pose_dimension"] == 36
    assert contract["view_feature_group"] == "F3"
    assert contract["view_embedding_dimension"] == 16
    assert contract["pose_embedding_dimension"] == 16
    assert contract["gru_hidden_dimension"] == 16
    assert contract["maximum_parameter_count"] == 50_000
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
    assert contract["r1_2b_outer_gain"] == 0.005196372744170267
    assert contract["r1_2b_boundary_gain"] == 0.002866365549963367
    assert contract["source_pose_manifest_sha256"] == (
        "5d138f7f06ffaccb6b9a59d538028f0f298f0c538ea6845a49e9e6c2eda6f116"
    )
    assert contract["deployment_eligible"] is False
    assert contract["teacher_eligible"] is False
    assert contract["paper_test_eligible"] is False


def test_pose_manifest_and_rotation_6d_are_frozen_and_continuous(tmp_path):
    from utils.a7c_view_pose_compositor import (
        axis_angle_pose_to_rotation_6d,
        load_pose_rotation_6d,
        pose_manifest_sha256,
    )

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    pose = np.zeros((2, 6, 3), dtype=np.float64)
    pose[1, 0, 0] = 1.0e-5
    for frame, values in zip((0, 5), pose):
        np.savez(model_dir / f"{frame:06d}.npz", pose_body=values)

    output = axis_angle_pose_to_rotation_6d(pose)
    loaded = load_pose_rotation_6d(model_dir, (0, 5), tuple(range(6)))
    assert output.shape == (2, 36)
    assert np.isfinite(output).all()
    assert np.linalg.norm(output[1] - output[0]) < 1.0e-4
    np.testing.assert_array_equal(loaded, output)

    fingerprint = pose_manifest_sha256(model_dir, (0, 5), tmp_path)
    assert len(fingerprint) == 64
    assert fingerprint == pose_manifest_sha256(model_dir, (0, 5), tmp_path)


def test_fit_normalization_never_reads_held_samples():
    from utils.a7c_view_pose_compositor import (
        apply_normalization,
        fit_normalization,
    )

    values = np.array([[0.0], [2.0], [1000.0]])
    mask = np.array([True, True, False])
    stats = fit_normalization(values, mask)
    np.testing.assert_allclose(stats["mean"], [1.0])
    np.testing.assert_allclose(stats["scale"], [1.0])
    np.testing.assert_array_equal(stats["fit_mask"], mask)
    np.testing.assert_allclose(
        apply_normalization(values, stats)[:2], [[-1.0], [1.0]]
    )


def test_runtime_inputs_emit_only_registered_continuous_tensors():
    from utils.a7c_view_pose_compositor import build_runtime_inputs

    names = [
        "visibility",
        "camera_x_over_z",
        "camera_y_over_z",
        "log_depth",
        "alpha_transmittance_mass",
        "semantic_support_mean",
        "alpha_mean",
    ]
    features = np.zeros((2, 2, len(names)), dtype=np.float32)
    features[:, :, names.index("visibility")] = 1.0
    features[:, :, names.index("alpha_transmittance_mass")] = 0.5
    features[:, :, names.index("semantic_support_mean")] = 1.0
    features[:, :, names.index("alpha_mean")] = 1.0
    camera_index = np.array([0, 0], dtype=np.int16)
    frame_index = np.array([0, 5], dtype=np.int32)
    carrier_ids = np.array([3, 7], dtype=np.int64)
    probe = {
        "features": features,
        "carrier_ids": carrier_ids,
        "camera_index": camera_index,
        "frame_index": frame_index,
    }
    runtime = build_runtime_inputs(
        probe=probe,
        feature_names=names,
        feature_group=names,
        pose_by_frame={0: np.zeros(36), 5: np.ones(36)},
        camera_index=camera_index,
        frame_index=frame_index,
        carrier_ids=carrier_ids,
        a5_weight=np.array([0.8, 0.9]),
        spatial_scale=0.03,
        depth_scale=0.04,
        edge_log_weight_minimum=-20.0,
    )
    assert set(runtime) == {
        "features",
        "pose",
        "projected_xy",
        "log_depth",
        "visibility",
        "adjacency",
        "runtime_mass",
        "feature_names",
    }
    assert runtime["features"].shape == (2, 2, len(names))
    assert runtime["pose"].shape == (2, 36)
    assert runtime["adjacency"].shape == (2, 2, 2)
    assert runtime["runtime_mass"].shape == (2, 2)


def test_model_signature_forbids_renderer_labels_and_ids():
    from utils.a7c_view_pose_compositor import ViewPoseResidualCompositor

    names = set(inspect.signature(ViewPoseResidualCompositor.forward).parameters)
    assert not names & {
        "camera_id",
        "camera_index",
        "frame_id",
        "frame_index",
        "subject_id",
        "gaussian_id",
        "image_name",
        "held_block_identity",
        "target",
        "outer",
        "boundary",
        "teacher_gates",
        "evidence",
    }


def test_segment_packing_sorts_manifest_and_never_crosses_boundary():
    from utils.a7c_view_pose_compositor import pack_camera_block_segments

    camera = np.array([1, 0, 0, 1])
    block = np.array([0, 0, 0, 0])
    frame = np.array([5, 5, 0, 0])
    segments = pack_camera_block_segments(
        camera, block, frame, frame_stride=5
    )
    assert [row.tolist() for row in segments] == [[2, 1], [3, 0]]


def _view_pose_model(view_dimension=4):
    from utils.a7c_view_pose_compositor import ViewPoseResidualCompositor

    return ViewPoseResidualCompositor(
        view_dimension=view_dimension,
        view_embedding_dimension=16,
        pose_dimension=36,
        pose_embedding_dimension=16,
        gru_hidden_dimension=16,
        residual_gate_scale=0.1,
        minimum_gate=0.9,
        maximum_gate=1.0,
    )


def test_view_pose_model_is_bounded_small_and_deterministic():
    torch.manual_seed(1)
    model = _view_pose_model()
    assert sum(value.numel() for value in model.parameters()) <= 50_000
    assert torch.count_nonzero(model.residual_head.weight) == 0
    view = torch.randn(8, 3, 4)
    pose = torch.randn(8, 36)
    base = torch.full((8, 3), 0.97)
    adjacency = torch.eye(3).expand(8, 3, 3)
    visibility = torch.ones(8, 3)
    first = model(view, pose, adjacency, visibility, base)
    second = model(view, pose, adjacency, visibility, base)
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first, base, atol=1.0e-7, rtol=0.0)
    assert torch.all(first >= 0.9) and torch.all(first <= 1.0)


def test_view_pose_interaction_responds_to_either_runtime_branch():
    torch.manual_seed(2)
    model = _view_pose_model()
    with torch.no_grad():
        model.residual_head.weight.fill_(0.05)
    view = torch.ones(6, 2, 4)
    pose = torch.ones(6, 36)
    adjacency = torch.zeros(6, 2, 2)
    visibility = torch.ones(6, 2)
    base = torch.full((6, 2), 0.97)
    both = model(view, pose, adjacency, visibility, base)
    zero_view = model(
        torch.zeros_like(view), pose, adjacency, visibility, base
    )
    zero_pose = model(
        view, torch.zeros_like(pose), adjacency, visibility, base
    )
    assert not torch.allclose(both, zero_view)
    assert not torch.allclose(both, zero_pose)


def test_registered_distillation_loss_matches_three_terms():
    from utils.a7c_view_pose_compositor import distillation_loss

    prediction = torch.tensor([[0.96], [0.98], [0.97]])
    teacher = torch.tensor([[0.95], [0.99], [0.97]])
    residual = torch.tensor([[0.1], [-0.1], [0.0]])
    result = distillation_loss(
        prediction,
        teacher,
        residual,
        gate_delta=0.01,
        temporal_delta=0.005,
        temporal_weight=0.25,
        residual_weight=0.001,
    )
    assert set(result) == {"loss", "gate", "temporal", "residual"}
    torch.testing.assert_close(
        result["loss"],
        result["gate"]
        + 0.25 * result["temporal"]
        + 0.001 * result["residual"],
    )


def test_nearest_neighbor_key_is_pose_plus_six_registered_view_means():
    from utils.a7c_view_pose_compositor import build_nearest_neighbor_keys

    pose = np.zeros((2, 36))
    features = np.zeros((2, 3, 7))
    names = [
        "visibility",
        "view_dir_x",
        "view_dir_y",
        "view_dir_z",
        "log_depth",
        "alpha_transmittance_mass",
        "semantic_support_mean",
    ]
    features[:, :, 0] = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    features[0, :, 1] = np.array([1.0, 100.0, 3.0])
    keys = build_nearest_neighbor_keys(features, names, pose)
    assert keys.shape == (2, 42)
    assert keys[0, 36] == 2.0


def test_k4_baseline_uses_fit_rows_only_and_averages_exact_matches():
    from utils.a7c_view_pose_compositor import nearest_neighbor_predict

    fit_keys = np.array([[0.0], [0.0], [1.0], [2.0], [3.0]])
    fit_gates = np.array([[0.9], [1.0], [0.8], [0.7], [0.6]])
    query = np.array([[0.0], [1.5]])
    result = nearest_neighbor_predict(fit_keys, fit_gates, query, k=4)
    np.testing.assert_allclose(result[0], [0.95])
    assert result.shape == (2, 1)


def test_model_exposes_zero_initialized_uncompressed_residual():
    model = _view_pose_model()
    view = torch.zeros(3, 2, 4)
    pose = torch.zeros(3, 36)
    adjacency = torch.zeros(3, 2, 2)
    visibility = torch.ones(3, 2)
    base = torch.full((3, 2), 0.97)
    gates, residual = model.predict_with_residual(
        view, pose, adjacency, visibility, base
    )
    torch.testing.assert_close(gates, base)
    torch.testing.assert_close(residual, torch.zeros_like(residual))


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
        prediction_mask=np.ones(samples, dtype=bool),
        camera_index=np.zeros(samples, np.int16),
        frame_index=np.arange(samples, dtype=np.int32) * 5,
        block_ids=np.array([0] * 4 + [1] * 4, np.int16),
        runtime_mass=np.zeros((samples, carriers), np.float32),
        a5_weight=np.full(carriers, 0.8, np.float32),
        contract={
            "view_embedding_dimension": 16,
            "pose_embedding_dimension": 16,
            "gru_hidden_dimension": 16,
            "residual_gate_scale": 0.1,
            "minimum_gate": 0.9,
            "maximum_gate": 1.0,
            "selection_threshold": 0.2,
            "proxy_target_response": 0.995,
            "maximum_projection_gate_jump": 0.015,
            "lexicographic_tolerance": 1.0e-9,
            "solver_primal_tolerance": 1.0e-9,
            "solver_residual_tolerance": 1.0e-7,
            "training_epochs": 3,
            "random_seed": 7,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "gate_huber_delta": 0.01,
            "temporal_huber_delta": 0.005,
            "temporal_loss_weight": 0.25,
            "residual_loss_weight": 0.001,
            "maximum_parameter_count": 50_000,
            "frame_stride": 5,
        },
        output_dir=tmp_path,
        device="cpu",
    )
    assert summary["epochs"] == 3
    assert summary["checkpoint_epoch"] == 3
    assert summary["held_teacher_values_accessed"] is False
    assert summary["parameter_count"] <= 50_000
    assert summary["maximum_gradient_norm_before_clip"] >= 0.0
    assert (tmp_path / "model.pt").is_file()
    assert (tmp_path / "predictions.npz").is_file()


def _passing_records():
    return [dict(
        fold=fold, camera_index=camera, outer_gain=0.006,
        boundary_gain=0.025, minimum_target_response=0.995,
        maximum_soft_iou_drop=0.001, visibility_response_ratio=0.999,
        maximum_adjacent_gate_change=0.015, topology_passed=True,
        coverage_passed=True, frozen_parts_passed=True,
        weight_upper_bound_passed=True,
    ) for fold in range(6) for camera in range(4)]


def _promotion_contract():
    return {
        "minimum_outer_gain": 0.005, "minimum_boundary_gain": 0.005,
        "minimum_positive_block_fraction": 0.9, "block_gain_quantile": 0.1,
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

    learned = _passing_records()
    nn = [dict(row, outer_gain=0.0055, boundary_gain=0.024) for row in learned]
    assert classify_canary(learned, nn, _promotion_contract()) == "CANARY_PROMOTED"
    broken = [dict(row) for row in learned]
    for row in broken:
        if row["camera_index"] == 3:
            row["boundary_gain"] = -0.001
    assert classify_canary(broken, nn, _promotion_contract()) == "CANARY_NEGATIVE"


def test_visibility_response_uses_target_contribution_over_pixel_count():
    from tools.audit_a7c_r1_4vp_view_pose import visibility_response_ratio

    pixels = np.array([10.0, 20.0, 10.0])
    base = np.array([5.0, 8.0, 6.0])
    candidate = np.array([5.0, 8.0, 6.0])
    assert visibility_response_ratio(base, candidate, pixels) == 1.0


def test_auditor_rejects_missing_freeze_manifest_or_label_leakage(tmp_path):
    from tools.audit_a7c_r1_4vp_view_pose import verify_frozen_artifacts

    with pytest.raises(ValueError, match="models_frozen"):
        verify_frozen_artifacts(tmp_path, expected={})


def test_r1_4vp_runner_is_restart_safe_audit_gated_and_camera_isolated():
    runner = ROOT / "tools/run_a7c_r1_4vp_view_pose_377.sh"
    source = runner.read_text(encoding="utf-8")
    for camera in ("c17", "c18", "c19", "c20", "c21", "c22", "c23"):
        assert re.search(rf"\b{camera}\b", source) is None
    assert "build_a7c_r1_4vp_fit_teachers.py" in source
    assert "train_a7c_r1_4vp_view_pose.py" in source
    assert "models_frozen.json" in source
    audit = source.index("audit_a7c_r1_4vp_view_pose.py")
    assert source.index("models_frozen.json") < audit
    assert "mark_terminal completed" in source
    assert "mark_terminal rejected" in source
    assert "mark_terminal failed" in source
    assert "started_utc.txt" in source and "ended_utc.txt" in source
