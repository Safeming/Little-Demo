import json
from pathlib import Path

import numpy as np

from utils.part_label_bank import PART_NAMES


def _save_bank(path: Path, weights):
    from utils.part_label_bank import save_part_label_bank

    values = np.asarray(weights, dtype=np.float32)
    point_count = values.shape[0]
    save_part_label_bank(
        path,
        part_label=np.arange(point_count, dtype=np.int16) % len(PART_NAMES),
        confidence=np.ones(point_count, dtype=np.float32),
        vote_count=np.ones(point_count, dtype=np.int16),
        per_part_votes=np.zeros((point_count, len(PART_NAMES)), dtype=np.int16),
        visible_vote_count=np.ones(point_count, dtype=np.int16),
        conflict_count=np.zeros(point_count, dtype=np.int16),
        source_checkpoint="checkpoint.pth",
        source_asset_root="assets",
        source_iteration=42000,
        semantic_probs=np.full_like(values, 0.8),
        soft_edit_weights=values,
    )


def _save_dual_evidence(
    path: Path,
    contract_fingerprint: str,
    point_count: int,
    *,
    camera_count: int = 2,
    samples_per_camera: int = 10,
):
    from tools.build_temporal_reliability_evidence import _payload_fingerprint

    channels = len(PART_NAMES)
    samples = int(camera_count) * int(samples_per_camera)
    shape = (point_count, channels)
    sequence_shape = (samples, point_count, channels)
    target = np.full(sequence_shape, 10.0, dtype=np.float16)
    outer = np.ones(sequence_shape, dtype=np.float16)
    boundary = np.ones(sequence_shape, dtype=np.float16)
    unstable = np.tile(np.array([0.0, 2.0], dtype=np.float16), samples // 2)
    lower_index = PART_NAMES.index("lower")
    target[:, 0, lower_index] = unstable + 0.1
    outer[:, 0, lower_index] = unstable
    boundary[:, 0, lower_index] = unstable
    selection_target = np.full(sequence_shape, 0.01, dtype=np.float16)
    selection_outer = np.full(sequence_shape, 0.01, dtype=np.float16)
    arrays = {
        "schema_version": np.array(4, dtype=np.int32),
        "point_count": np.array(point_count, dtype=np.int64),
        "part_names": np.asarray(PART_NAMES, dtype="U16"),
        "formal_protocol": np.array(0, dtype=np.uint8),
        "cameras": np.asarray(
            [f"c{1 + 4 * index:02d}" for index in range(camera_count)], dtype="U3"
        ),
        "frame_start": np.array(0, dtype=np.int64),
        "frame_end": np.array(10, dtype=np.int64),
        "frame_stride": np.array(5, dtype=np.int64),
        "evidence_mode": np.array("renderer_aligned_dual_sequence_constrained"),
        "renderer_attribution": np.array("colors_gradient_dual"),
        "protocol_fingerprint": np.array("e" * 64),
        "a7_contract_fingerprint": np.array(contract_fingerprint),
        "temporal_visible_count": np.full(shape, 12, dtype=np.int32),
        "temporal_consecutive_visible_count": np.full(shape, 10, dtype=np.int32),
        "temporal_target_ratio_mean": np.full(shape, 0.8, dtype=np.float32),
        "temporal_target_ratio_std": np.full(shape, 0.1, dtype=np.float32),
        "temporal_target_flicker": np.full(shape, 0.1, dtype=np.float32),
        "temporal_outer_ratio_mean": np.full(shape, 0.2, dtype=np.float32),
        "temporal_outer_ratio_std": np.full(shape, 0.1, dtype=np.float32),
        "temporal_outer_flicker": np.full(shape, 0.1, dtype=np.float32),
        "temporal_boundary_crossing_rate": np.full(shape, 0.1, dtype=np.float32),
        "temporal_visibility_transition_rate": np.full(shape, 0.1, dtype=np.float32),
        "renderer_target_contribution_sequence": target,
        "renderer_outer_contribution_sequence": outer,
        "renderer_boundary_contribution_sequence": boundary,
        "renderer_selection_target_contribution_sequence": selection_target,
        "renderer_selection_outer_contribution_sequence": selection_outer,
        "renderer_selection_boundary_contribution_sequence": selection_outer,
        "renderer_sequence_target_pixel_count": np.full(
            (samples, channels), 100.0, dtype=np.float32
        ),
        "renderer_sequence_camera_index": np.repeat(
            np.arange(camera_count, dtype=np.int16), samples_per_camera
        ),
        "renderer_sequence_frame_index": np.tile(
            np.arange(0, samples_per_camera * 5, 5, dtype=np.int32), camera_count
        ),
    }
    arrays["output_fingerprint"] = np.array(_payload_fingerprint(arrays))
    np.savez_compressed(path, **arrays)


def test_constrained_v5_cli_restores_hair_and_keeps_valid_lower_seed(tmp_path):
    from tools.calibrate_constrained_a7_weights import main
    from utils.frozen_semantic_method import load_a7_temporal_contract
    from utils.part_label_bank import load_part_label_bank

    freeze = Path("configs/semantic/frozen_a5_main_method_v1.json")
    contract_path = Path(
        "configs/semantic/frozen_a7_dual_evidence_v5_canary_377.json"
    )
    contract = load_a7_temporal_contract(contract_path, freeze)
    weights = np.full((5, len(PART_NAMES)), 0.6, dtype=np.float32)
    v4_weights = weights.copy()
    v4_weights[0, PART_NAMES.index("hair")] *= 0.9
    v4_weights[0, PART_NAMES.index("lower")] *= 0.9
    a5 = tmp_path / "a5.npz"
    v4 = tmp_path / "v4.npz"
    evidence = tmp_path / "evidence.npz"
    output = tmp_path / "candidates"
    _save_bank(a5, weights)
    _save_bank(v4, v4_weights)
    _save_dual_evidence(evidence, contract["_fingerprint"], len(weights))

    assert main(
        [
            "--a5-bank", str(a5),
            "--v4-bank", str(v4),
            "--evidence", str(evidence),
            "--method-freeze", str(freeze),
            "--a7-contract", str(contract_path),
            "--output-dir", str(output),
            "--allow-canary-inputs",
        ]
    ) == 0

    index = json.loads((output / "candidate_index.json").read_text())
    assert index["validation_shortlist"] == ["dual_evidence_constrained_v5"]
    bank = load_part_label_bank(
        output / "dual_evidence_constrained_v5" / "part_label_bank.npz"
    )
    np.testing.assert_array_equal(
        bank["soft_edit_weights"][:, PART_NAMES.index("hair")],
        weights[:, PART_NAMES.index("hair")],
    )
    np.testing.assert_array_equal(
        bank["soft_edit_weights"][:, PART_NAMES.index("lower")],
        v4_weights[:, PART_NAMES.index("lower")],
    )
    summary = index["candidates"][0]
    assert summary["selection_crossing_count"] == 0
    assert summary["maximum_weight_above_a5"] == 0.0
    assert summary["capacity_summary"]["all_folds_passed"] is True


def test_constrained_v5_loader_rejects_non_dual_attribution(tmp_path):
    from tools.calibrate_constrained_a7_weights import _load_evidence
    from utils.frozen_semantic_method import load_a7_temporal_contract

    freeze = Path("configs/semantic/frozen_a5_main_method_v1.json")
    contract = load_a7_temporal_contract(
        "configs/semantic/frozen_a7_dual_evidence_v5_canary_377.json", freeze
    )
    evidence = tmp_path / "evidence.npz"
    _save_dual_evidence(evidence, contract["_fingerprint"], 5)
    with np.load(evidence, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files if key != "output_fingerprint"}
    arrays["renderer_attribution"] = np.array("colors_gradient")
    from tools.build_temporal_reliability_evidence import _payload_fingerprint

    arrays["output_fingerprint"] = np.array(_payload_fingerprint(arrays))
    np.savez_compressed(evidence, **arrays)

    import pytest

    with pytest.raises(ValueError, match="renderer_attribution"):
        _load_evidence(evidence, contract=contract, allow_canary=True)


def test_constrained_v5_1_cli_reuses_v5_evidence_with_separate_visibility_gates(
    tmp_path,
):
    from tools.calibrate_constrained_a7_weights import main
    from utils.frozen_semantic_method import load_a7_temporal_contract

    freeze = Path("configs/semantic/frozen_a5_main_method_v1.json")
    v5_contract = load_a7_temporal_contract(
        "configs/semantic/frozen_a7_dual_evidence_v5_canary_377.json", freeze
    )
    v5_1_contract_path = Path(
        "configs/semantic/frozen_a7_dual_evidence_v5_1_canary_377.json"
    )
    weights = np.full((5, len(PART_NAMES)), 0.6, dtype=np.float32)
    v4_weights = weights.copy()
    v4_weights[0, PART_NAMES.index("lower")] *= 0.9
    a5 = tmp_path / "a5.npz"
    v4 = tmp_path / "v4.npz"
    evidence = tmp_path / "evidence.npz"
    output = tmp_path / "candidates"
    _save_bank(a5, weights)
    _save_bank(v4, v4_weights)
    _save_dual_evidence(evidence, v5_contract["_fingerprint"], len(weights))

    assert main(
        [
            "--a5-bank", str(a5),
            "--v4-bank", str(v4),
            "--evidence", str(evidence),
            "--method-freeze", str(freeze),
            "--a7-contract", str(v5_1_contract_path),
            "--output-dir", str(output),
            "--allow-canary-inputs",
        ]
    ) == 0

    index = json.loads((output / "candidate_index.json").read_text())
    assert index["validation_shortlist"] == ["dual_evidence_constrained_v5_1"]
    capacity = index["candidates"][0]["capacity_summary"]
    assert capacity["maximum_training_visibility_response_ratio"] == 0.9995
    assert capacity["maximum_audit_visibility_response_ratio"] == 1.0


def test_constrained_v5_2_cli_routes_separate_target_and_visibility_gates(tmp_path):
    from tools.calibrate_constrained_a7_weights import main
    from utils.frozen_semantic_method import load_a7_temporal_contract

    freeze = Path("configs/semantic/frozen_a5_main_method_v1.json")
    v5_contract = load_a7_temporal_contract(
        "configs/semantic/frozen_a7_dual_evidence_v5_canary_377.json", freeze
    )
    contract = Path(
        "configs/semantic/frozen_a7_dual_evidence_v5_2_canary_377.json"
    )
    weights = np.full((5, len(PART_NAMES)), 0.6, dtype=np.float32)
    v4_weights = weights.copy()
    v4_weights[0, PART_NAMES.index("lower")] *= 0.9
    a5 = tmp_path / "a5.npz"
    v4 = tmp_path / "v4.npz"
    evidence = tmp_path / "evidence.npz"
    output = tmp_path / "candidates"
    _save_bank(a5, weights)
    _save_bank(v4, v4_weights)
    _save_dual_evidence(evidence, v5_contract["_fingerprint"], len(weights))

    assert main(
        [
            "--a5-bank", str(a5),
            "--v4-bank", str(v4),
            "--evidence", str(evidence),
            "--method-freeze", str(freeze),
            "--a7-contract", str(contract),
            "--output-dir", str(output),
            "--allow-canary-inputs",
        ]
    ) == 0

    index = json.loads((output / "candidate_index.json").read_text())
    candidate = index["candidates"][0]
    capacity = candidate["capacity_summary"]
    assert candidate["candidate_id"] == "dual_evidence_constrained_v5_2"
    assert capacity["minimum_training_target_response_ratio"] == 0.9975
    assert capacity["minimum_audit_target_response_ratio"] == 0.99
    assert capacity["maximum_training_visibility_response_ratio"] == 0.999
    assert capacity["maximum_audit_visibility_response_ratio"] == 1.0


def test_constrained_v5_3_cli_accepts_eight_camera_source_contract(tmp_path):
    from tools.calibrate_constrained_a7_weights import main
    from utils.frozen_semantic_method import load_a7_temporal_contract

    freeze = Path("configs/semantic/frozen_a5_main_method_v1.json")
    evidence_contract = load_a7_temporal_contract(
        "configs/semantic/frozen_a7_dual_evidence_v5_3_evidence_377.json", freeze
    )
    contract = Path(
        "configs/semantic/frozen_a7_dual_evidence_v5_3_canary_377.json"
    )
    weights = np.full((5, len(PART_NAMES)), 0.6, dtype=np.float32)
    v4_weights = weights.copy()
    v4_weights[0, PART_NAMES.index("lower")] *= 0.9
    a5 = tmp_path / "a5.npz"
    v4 = tmp_path / "v4.npz"
    evidence = tmp_path / "evidence.npz"
    output = tmp_path / "candidates"
    _save_bank(a5, weights)
    _save_bank(v4, v4_weights)
    _save_dual_evidence(evidence, evidence_contract["_fingerprint"], len(weights))

    assert main(
        [
            "--a5-bank", str(a5),
            "--v4-bank", str(v4),
            "--evidence", str(evidence),
            "--method-freeze", str(freeze),
            "--a7-contract", str(contract),
            "--output-dir", str(output),
            "--allow-canary-inputs",
        ]
    ) == 0

    index = json.loads((output / "candidate_index.json").read_text())
    candidate = index["candidates"][0]
    assert candidate["candidate_id"] == "dual_evidence_constrained_v5_3"
    assert candidate["capacity_summary"]["minimum_training_target_response_ratio"] == 0.995
    assert candidate["capacity_summary"]["maximum_training_visibility_response_ratio"] == 0.998
    assert candidate["capacity_summary"]["minimum_held_out_temporal_gain"] == 0.0


def test_constrained_v5_4_cli_builds_camera_time_stability_capacity(tmp_path):
    from tools.calibrate_constrained_a7_weights import main
    from utils.frozen_semantic_method import load_a7_temporal_contract

    freeze = Path("configs/semantic/frozen_a5_main_method_v1.json")
    evidence_contract = load_a7_temporal_contract(
        "configs/semantic/frozen_a7_dual_evidence_v5_3_evidence_377.json", freeze
    )
    contract = Path(
        "configs/semantic/frozen_a7_dual_evidence_v5_4_canary_377.json"
    )
    weights = np.full((5, len(PART_NAMES)), 0.6, dtype=np.float32)
    v4_weights = weights.copy()
    v4_weights[0, PART_NAMES.index("lower")] *= 0.9
    a5 = tmp_path / "a5.npz"
    v4 = tmp_path / "v4.npz"
    evidence = tmp_path / "evidence.npz"
    output = tmp_path / "candidates"
    _save_bank(a5, weights)
    _save_bank(v4, v4_weights)
    _save_dual_evidence(
        evidence,
        evidence_contract["_fingerprint"],
        len(weights),
        camera_count=8,
        samples_per_camera=12,
    )

    assert main(
        [
            "--a5-bank", str(a5),
            "--v4-bank", str(v4),
            "--evidence", str(evidence),
            "--method-freeze", str(freeze),
            "--a7-contract", str(contract),
            "--output-dir", str(output),
            "--allow-canary-inputs",
        ]
    ) == 0

    index = json.loads((output / "candidate_index.json").read_text())
    candidate = index["candidates"][0]
    assert candidate["candidate_id"] == "dual_evidence_camera_time_v5_4"
    assert candidate["capacity_summary"]["fold_count"] == 48
    assert candidate["capacity_summary"]["consensus"]["minimum_fold_count"] == 36
