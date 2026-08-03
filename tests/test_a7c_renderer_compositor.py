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
