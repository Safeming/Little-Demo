import json
import inspect
from pathlib import Path

import numpy as np
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
