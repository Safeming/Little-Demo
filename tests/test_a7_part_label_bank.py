import numpy as np
import pytest

from utils.part_label_bank import PART_NAMES


TEMPORAL_FLOAT_FIELDS = (
    "temporal_target_ratio_mean",
    "temporal_target_ratio_std",
    "temporal_target_flicker",
    "temporal_outer_ratio_mean",
    "temporal_outer_ratio_std",
    "temporal_outer_flicker",
    "temporal_boundary_crossing_rate",
    "temporal_visibility_transition_rate",
)
TEMPORAL_COUNT_FIELDS = (
    "temporal_visible_count",
    "temporal_consecutive_visible_count",
)


def _save_base_bank(path, point_count=3):
    from utils.part_label_bank import save_part_label_bank

    save_part_label_bank(
        path,
        part_label=np.array([0, 1, 2], dtype=np.int16)[:point_count],
        confidence=np.ones(point_count, dtype=np.float32),
        vote_count=np.ones(point_count, dtype=np.int16),
        per_part_votes=np.zeros((point_count, len(PART_NAMES)), dtype=np.int16),
        visible_vote_count=np.ones(point_count, dtype=np.int16),
        conflict_count=np.zeros(point_count, dtype=np.int16),
        source_checkpoint="checkpoint.pth",
        source_asset_root="assets",
        source_iteration=42000,
        semantic_probs=np.full(
            (point_count, len(PART_NAMES)), 1.0 / len(PART_NAMES), dtype=np.float32
        ),
        soft_edit_weights=np.full(
            (point_count, len(PART_NAMES)), 0.5, dtype=np.float32
        ),
    )


def _temporal_evidence(point_count=3):
    shape = (point_count, len(PART_NAMES))
    evidence = {
        key: np.full(shape, 4, dtype=np.int32) for key in TEMPORAL_COUNT_FIELDS
    }
    evidence.update(
        {key: np.full(shape, 0.25, dtype=np.float32) for key in TEMPORAL_FLOAT_FIELDS}
    )
    return evidence


def _provenance():
    return {
        "base_method_freeze_fingerprint": "a" * 64,
        "a7_contract_fingerprint": "b" * 64,
        "evidence_protocol_fingerprint": "c" * 64,
        "candidate_config_fingerprint": "d" * 64,
    }


def _save_a7(tmp_path):
    from utils.part_label_bank import save_a7_part_label_bank

    base_path = tmp_path / "a5.npz"
    output_path = tmp_path / "a7.npz"
    _save_base_bank(base_path)
    evidence = _temporal_evidence()
    reliability = np.full((3, len(PART_NAMES)), 0.75, dtype=np.float32)
    weights = np.full((3, len(PART_NAMES)), 0.4, dtype=np.float32)
    fingerprint = save_a7_part_label_bank(
        output_path,
        base_bank_path=base_path,
        temporal_evidence=evidence,
        temporal_reliability=reliability,
        soft_edit_weights=weights,
        provenance=_provenance(),
    )
    return base_path, output_path, evidence, reliability, weights, fingerprint


def test_a7_bank_roundtrips_temporal_arrays_static_weights_and_provenance(tmp_path):
    from utils.part_label_bank import (
        load_part_label_bank,
        part_label_bank_fingerprint,
        resolve_soft_edit_weights,
    )

    _, output_path, evidence, reliability, weights, fingerprint = _save_a7(tmp_path)
    loaded = load_part_label_bank(output_path)

    assert str(loaded["method_id"]) == "A7"
    assert str(loaded["base_method"]) == "A5"
    for key, expected in evidence.items():
        np.testing.assert_array_equal(loaded[key], expected)
    np.testing.assert_array_equal(loaded["temporal_reliability"], reliability)
    np.testing.assert_array_equal(loaded["soft_edit_weights"], weights)
    for key, expected in _provenance().items():
        assert str(loaded[key]) == expected
    assert len(str(loaded["base_bank_sha256"])) == 64
    assert str(loaded["output_bank_fingerprint"]) == fingerprint
    assert part_label_bank_fingerprint(loaded) == fingerprint

    resolved, source = resolve_soft_edit_weights(loaded, point_count=3)
    np.testing.assert_array_equal(resolved, weights)
    assert source == "soft_edit_weights"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda arrays: arrays.__setitem__(
                "temporal_visible_count", np.zeros((2, len(PART_NAMES)), dtype=np.int32)
            ),
            "shape",
        ),
        (
            lambda arrays: arrays["temporal_visible_count"].__setitem__((0, 0), -1),
            "non-negative",
        ),
        (
            lambda arrays: arrays["temporal_target_ratio_mean"].__setitem__(
                (0, 0), np.nan
            ),
            "finite",
        ),
        (
            lambda arrays: arrays["temporal_reliability"].__setitem__((0, 0), 1.1),
            r"\[0, 1\]",
        ),
        (
            lambda arrays: arrays.pop("base_method_freeze_fingerprint"),
            "base_method_freeze_fingerprint",
        ),
    ],
)
def test_a7_schema_rejects_invalid_arrays(tmp_path, mutation, message):
    from utils.part_label_bank import load_part_label_bank, validate_part_label_bank_arrays

    _, output_path, _, _, _, _ = _save_a7(tmp_path)
    arrays = load_part_label_bank(output_path)
    arrays = {key: value.copy() for key, value in arrays.items()}
    mutation(arrays)

    with pytest.raises(ValueError, match=message):
        validate_part_label_bank_arrays(arrays)


def test_a7_save_rejects_overwriting_base_bank(tmp_path):
    from utils.part_label_bank import save_a7_part_label_bank

    base_path = tmp_path / "a5.npz"
    _save_base_bank(base_path)

    with pytest.raises(ValueError, match="must not overwrite"):
        save_a7_part_label_bank(
            base_path,
            base_bank_path=base_path,
            temporal_evidence=_temporal_evidence(),
            temporal_reliability=np.ones((3, len(PART_NAMES)), dtype=np.float32),
            soft_edit_weights=np.ones((3, len(PART_NAMES)), dtype=np.float32),
            provenance=_provenance(),
        )


def test_a7_load_rejects_tampered_output_fingerprint(tmp_path):
    from utils.part_label_bank import load_part_label_bank

    _, output_path, _, _, _, _ = _save_a7(tmp_path)
    arrays = load_part_label_bank(output_path)
    arrays["soft_edit_weights"] = arrays["soft_edit_weights"].copy()
    arrays["soft_edit_weights"][0, 0] += np.float32(0.1)
    np.savez_compressed(output_path, **arrays)

    with pytest.raises(ValueError, match="output_bank_fingerprint"):
        load_part_label_bank(output_path)
