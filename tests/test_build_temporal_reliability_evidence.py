from argparse import Namespace

import numpy as np
import pytest


def _contract():
    return {
        "evidence_cameras": ["c01", "c05", "c09", "c13"],
        "evidence_frame_start": 0,
        "evidence_frame_end": 570,
        "evidence_frame_stride": 5,
        "parts": ["hair", "face", "upper", "lower", "shoes", "skin"],
    }


def _args(**overrides):
    values = {
        "cameras": "c01,c05,c09,c13",
        "frame_start": 0,
        "frame_end": 570,
        "frame_stride": 5,
        "parts": "hair,face,upper,lower,shoes,skin",
        "allow_canary_protocol": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_parser_supports_required_a7_evidence_cli_contract(tmp_path):
    from tools.build_temporal_reliability_evidence import parse_args

    args = parse_args(
        [
            "--config",
            "config.yaml",
            "--checkpoint",
            "checkpoint.pth",
            "--a5-bank",
            "a5.npz",
            "--method-freeze",
            "a5.json",
            "--a7-contract",
            "a7.json",
            "--cameras",
            "c01,c05",
            "--frame-start",
            "0",
            "--frame-end",
            "20",
            "--frame-stride",
            "5",
            "--parts",
            "hair,face",
            "--output",
            str(tmp_path / "evidence.npz"),
            "--resume",
            "--dry-run",
            "--allow-canary-protocol",
        ]
    )

    assert args.resume is True
    assert args.dry_run is True
    assert args.allow_canary_protocol is True
    assert args.cameras == "c01,c05"
    assert args.frame_stride == 5


def test_formal_protocol_accepts_only_frozen_evidence_split():
    from tools.build_temporal_reliability_evidence import validate_requested_protocol

    protocol = validate_requested_protocol(_args(), _contract())

    assert protocol["formal_protocol"] is True
    assert protocol["cameras"] == ["c01", "c05", "c09", "c13"]
    assert protocol["frames"] == list(range(0, 570, 5))
    assert protocol["sample_count"] == 4 * len(range(0, 570, 5))


def test_nonformal_protocol_requires_explicit_canary_flag():
    from tools.build_temporal_reliability_evidence import validate_requested_protocol

    args = _args(cameras="c01,c05", frame_end=20)
    with pytest.raises(ValueError, match="allow-canary-protocol"):
        validate_requested_protocol(args, _contract())

    args.allow_canary_protocol = True
    protocol = validate_requested_protocol(args, _contract())
    assert protocol["formal_protocol"] is False
    assert protocol["sample_count"] == 2 * len(range(0, 20, 5))


def test_evidence_protocol_rejects_c21_even_for_canary():
    from tools.build_temporal_reliability_evidence import validate_requested_protocol

    with pytest.raises(ValueError, match="c21"):
        validate_requested_protocol(
            _args(cameras="c01,c21", allow_canary_protocol=True), _contract()
        )


def test_boundary_state_encoding_marks_interior_boundary_outer_and_invisible():
    from tools.build_temporal_reliability_evidence import encode_boundary_state

    state = encode_boundary_state(
        visible=np.array([True, True, True, False]),
        target_ratio=np.array([1.0, 0.6, 0.0, 0.0], dtype=np.float32),
        outer_ratio=np.array([0.0, 0.4, 1.0, 0.0], dtype=np.float32),
    )

    np.testing.assert_array_equal(state, np.array([1, 2, 3, 0], dtype=np.int8))


def test_combine_camera_evidence_uses_support_weighted_statistics():
    from tools.build_temporal_reliability_evidence import combine_camera_evidence

    first = {
        "temporal_visible_count": np.array([[2]], dtype=np.int32),
        "temporal_consecutive_visible_count": np.array([[1]], dtype=np.int32),
        "temporal_target_ratio_mean": np.array([[0.25]], dtype=np.float32),
        "temporal_target_ratio_std": np.array([[0.25]], dtype=np.float32),
        "temporal_target_flicker": np.array([[0.5]], dtype=np.float32),
        "temporal_outer_ratio_mean": np.array([[0.75]], dtype=np.float32),
        "temporal_outer_ratio_std": np.array([[0.25]], dtype=np.float32),
        "temporal_outer_flicker": np.array([[0.5]], dtype=np.float32),
        "temporal_boundary_crossing_rate": np.array([[1.0]], dtype=np.float32),
        "temporal_visibility_transition_rate": np.array([[0.5]], dtype=np.float32),
    }
    second = {
        **first,
        "temporal_visible_count": np.array([[1]], dtype=np.int32),
        "temporal_consecutive_visible_count": np.array([[0]], dtype=np.int32),
        "temporal_target_ratio_mean": np.array([[1.0]], dtype=np.float32),
        "temporal_target_ratio_std": np.array([[0.0]], dtype=np.float32),
        "temporal_target_flicker": np.array([[0.0]], dtype=np.float32),
        "temporal_outer_ratio_mean": np.array([[0.0]], dtype=np.float32),
        "temporal_outer_ratio_std": np.array([[0.0]], dtype=np.float32),
        "temporal_outer_flicker": np.array([[0.0]], dtype=np.float32),
        "temporal_boundary_crossing_rate": np.array([[0.0]], dtype=np.float32),
        "temporal_visibility_transition_rate": np.array([[1.0]], dtype=np.float32),
    }

    combined = combine_camera_evidence(
        [(first, 3), (second, 2)]
    )

    assert combined["temporal_visible_count"][0, 0] == 3
    assert combined["temporal_consecutive_visible_count"][0, 0] == 1
    assert combined["temporal_target_ratio_mean"][0, 0] == pytest.approx(0.5)
    assert combined["temporal_target_ratio_std"][0, 0] == pytest.approx(
        np.std([0.0, 0.5, 1.0], ddof=0)
    )
    assert combined["temporal_target_flicker"][0, 0] == pytest.approx(0.5)
    assert combined["temporal_visibility_transition_rate"][0, 0] == pytest.approx(
        (0.5 * 2 + 1.0 * 1) / 3
    )


def test_resume_manifest_rejects_input_or_protocol_mismatch():
    from tools.build_temporal_reliability_evidence import validate_resume_manifest

    expected = {
        "checkpoint_sha256": "a" * 64,
        "a5_bank_sha256": "b" * 64,
        "protocol_fingerprint": "c" * 64,
    }
    validate_resume_manifest(dict(expected), expected)
    for key in expected:
        mismatched = dict(expected)
        mismatched[key] = "0" * 64
        with pytest.raises(ValueError, match=key):
            validate_resume_manifest(mismatched, expected)
