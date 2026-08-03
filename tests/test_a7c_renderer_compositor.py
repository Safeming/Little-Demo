import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/semantic/a7c_carrier_compositor_canary_377_v1.json"


def test_contract_freezes_canary_scope_and_model():
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert payload["status"] == "frozen"
    assert payload["subject"] == "377"
    assert payload["fit_cameras"] == ["c01", "c05", "c09", "c13"]
    assert payload["audit_cameras"] == ["c17", "c18", "c19", "c20"]
    assert payload["forbidden_cameras"] == ["c21", "c22", "c23"]
    assert payload["frame_start"] == 0
    assert payload["frame_end"] == 570
    assert payload["frame_stride"] == 5
    assert payload["temporal_block_count"] == 6
    assert payload["minimum_gate"] == 0.9
    assert payload["maximum_gate"] == 1.0
    assert payload["selection_threshold"] == 0.2
    assert payload["hidden_dimensions"] == [32, 16]
    assert payload["paper_test_eligible"] is False


def test_feature_schema_rejects_identity_and_time_fields():
    from utils.a7c_renderer_compositor import validate_feature_schema

    valid = validate_feature_schema(["visibility", "log1p_radius", "opacity"])
    assert valid == ("visibility", "log1p_radius", "opacity")

    for forbidden in ("camera_id", "frame_index", "subject_id", "gaussian_id"):
        with pytest.raises(ValueError, match="forbidden"):
            validate_feature_schema(["visibility", forbidden])


def test_target_preserving_gate_limits():
    from utils.a7c_renderer_compositor import target_preserving_gate

    point_gate = np.array([0.9, 0.95, 1.0])
    np.testing.assert_allclose(target_preserving_gate(point_gate, 0.0), point_gate)
    np.testing.assert_allclose(target_preserving_gate(point_gate, 1.0), 1.0)
    middle = target_preserving_gate(point_gate, 0.5)
    assert np.all(middle >= point_gate)
    assert np.all(middle <= 1.0)


def test_contiguous_block_ids_do_not_interleave_time():
    from utils.a7c_renderer_compositor import contiguous_block_ids

    frames = np.arange(0, 60, 5)
    blocks = contiguous_block_ids(frames, 3)

    assert blocks.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    with pytest.raises(ValueError, match="sorted"):
        contiguous_block_ids(frames[::-1], 3)


def test_runtime_probe_features_are_finite_and_schema_ordered():
    from utils.a7c_renderer_compositor import extract_runtime_probe_features

    features = extract_runtime_probe_features(
        means3d=np.array([[1.0, 2.0, 4.0], [0.0, 1.0, 2.0]]),
        world_view_transform=np.eye(4),
        camera_center=np.zeros(3),
        visibility=np.array([True, False]),
        radii=np.array([2.0, 0.0]),
        opacity=np.array([0.5, 0.25]),
        a5_lower_weight=np.array([0.8, 0.7]),
        selected_lower=np.array([1.0, 1.0]),
    )

    assert features.shape == (2, 12)
    assert np.all(np.isfinite(features))
    assert features[0, 0] == 1.0
    assert features[1, 0] == 0.0
    assert features[0, 1] == pytest.approx(np.log1p(2.0))


def test_probe_normalization_uses_only_selected_fit_samples():
    from utils.a7c_renderer_compositor import fit_feature_normalization

    features = np.array(
        [
            [[0.0, 1.0], [0.0, 3.0]],
            [[0.0, 5.0], [0.0, 7.0]],
            [[100.0, 100.0], [100.0, 100.0]],
        ]
    )
    stats = fit_feature_normalization(features, sample_mask=np.array([1, 1, 0], bool))

    np.testing.assert_allclose(stats["mean"], [0.0, 4.0])
    assert stats["scale"][0] == 1.0
    assert stats["mean"][0] != pytest.approx(100.0)


def test_bounded_carrier_mlp_is_stateless_bounded_and_near_identity():
    import torch
    from utils.a7c_renderer_compositor import BoundedCarrierMLP

    torch.manual_seed(1)
    model = BoundedCarrierMLP(12, [32, 16], minimum_gate=0.9, initial_gate=0.999)
    values = torch.randn(9, 12)
    first = model(values)
    second = torch.cat([model(values[4:]), model(values[:4])])

    assert float(first.min()) >= 0.9
    assert float(first.max()) <= 1.0
    assert float(first.min()) >= 0.999 - 1e-6
    torch.testing.assert_close(second, torch.cat([first[4:], first[:4]]))
    assert not any("embedding" in name.lower() for name, _ in model.named_parameters())


def test_canary_splits_hold_out_contiguous_blocks_and_audit_cameras():
    from utils.a7c_renderer_compositor import build_canary_splits

    camera = np.repeat(np.arange(8), 12)
    frame = np.tile(np.arange(0, 60, 5), 8)
    split = build_canary_splits(
        camera_index=camera,
        frame_index=frame,
        fit_camera_indices=(0, 1, 2, 3),
        audit_camera_indices=(4, 5, 6, 7),
        block_count=3,
    )

    assert np.all(camera[split["audit_mask"]] >= 4)
    assert not np.any(split["fit_mask"] & split["audit_mask"])
    assert len(split["held_block_masks"]) == 3
    for mask in split["held_block_masks"]:
        assert np.all(camera[mask] < 4)
        for cam in range(4):
            selected_frames = frame[mask & (camera == cam)]
            assert np.all(np.diff(selected_frames) == 5)


def test_contribution_prediction_metrics_preserve_target_and_reduce_flicker():
    from utils.a7c_renderer_compositor import evaluate_contribution_predictions

    outer = np.array([1.0, 2.0, 1.0, 2.0])
    result = evaluate_contribution_predictions(
        target=np.ones(4),
        outer=outer,
        boundary=outer,
        point_target=np.zeros((4, 1)),
        point_outer=outer[:, None],
        point_boundary=outer[:, None],
        gates=np.array([[1.0], [0.9], [1.0], [0.9]]),
    )

    assert result["outer_gain"] > 0.0
    assert result["boundary_gain"] > 0.0
    assert result["minimum_target_response"] == pytest.approx(1.0)
    assert result["maximum_soft_iou_drop"] <= 0.0
