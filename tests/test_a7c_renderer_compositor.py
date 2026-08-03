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
