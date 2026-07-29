import json
from pathlib import Path

import numpy as np

from utils.part_label_bank import PART_NAMES


def _save_base_bank(path: Path, point_count: int = 5):
    from utils.part_label_bank import save_part_label_bank

    posterior = np.full((point_count, len(PART_NAMES)), 0.8, dtype=np.float32)
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
        semantic_probs=posterior,
        soft_edit_weights=np.full_like(posterior, 0.6),
    )


def _save_evidence(path: Path, source_contract_fingerprint: str, point_count: int = 5):
    from tools.build_temporal_reliability_evidence import _payload_fingerprint

    channels = len(PART_NAMES)
    shape = (point_count, channels)
    sequence_shape = (4, point_count, channels)
    target = np.full(sequence_shape, 10.0, dtype=np.float16)
    outer = np.ones(sequence_shape, dtype=np.float16)
    boundary = np.ones(sequence_shape, dtype=np.float16)
    unstable = np.array([0.0, 2.0, 0.0, 2.0], dtype=np.float16)
    for part_index in (0, 3):
        target[:, 0, part_index] = 0.1
        outer[:, 0, part_index] = unstable
        boundary[:, 0, part_index] = unstable
    arrays = {
        "schema_version": np.array(3, dtype=np.int32),
        "point_count": np.array(point_count, dtype=np.int64),
        "part_names": np.asarray(PART_NAMES, dtype="U16"),
        "formal_protocol": np.array(0, dtype=np.uint8),
        "protocol_fingerprint": np.array("e" * 64),
        "a7_contract_fingerprint": np.array(source_contract_fingerprint),
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
        "renderer_sequence_camera_index": np.array([0, 0, 1, 1], dtype=np.int16),
        "renderer_sequence_frame_index": np.array([0, 5, 0, 5], dtype=np.int32),
    }
    arrays["output_fingerprint"] = np.array(_payload_fingerprint(arrays))
    np.savez_compressed(path, **arrays)


def test_sparse_robust_cli_writes_one_loco_valid_candidate(tmp_path):
    from tools.calibrate_sparse_robust_a7_weights import main
    from utils.frozen_semantic_method import load_a7_temporal_contract

    freeze = Path("configs/semantic/frozen_a5_main_method_v1.json")
    v3 = load_a7_temporal_contract(
        "configs/semantic/frozen_a7_renderer_objective_v3_canary_377.json", freeze
    )
    base = tmp_path / "a5.npz"
    evidence = tmp_path / "evidence.npz"
    output = tmp_path / "candidate"
    _save_base_bank(base)
    _save_evidence(evidence, v3["_fingerprint"])

    argv = [
        "--a5-bank", str(base),
        "--evidence", str(evidence),
        "--method-freeze", str(freeze),
        "--a7-contract", "configs/semantic/frozen_a7_sparse_robust_v4_canary_377.json",
        "--output-dir", str(output),
        "--allow-canary-evidence",
    ]
    assert main(argv) == 0
    first = json.loads((output / "candidate_index.json").read_text())
    assert main(argv) == 0
    second = json.loads((output / "candidate_index.json").read_text())

    assert first == second
    assert first["candidate_count"] == 1
    assert first["valid_candidate_count"] == 1
    assert first["validation_shortlist"] == ["sparse_robust_loco_v4"]
    row = first["candidates"][0]
    assert row["capacity_summary"]["all_folds_passed"] is True
    assert len(row["capacity_summary"]["folds"]) == 2
    assert (output / "sparse_robust_loco_v4" / "part_label_bank.npz").is_file()
    assert row["selection_crossing_count"] == 0
    assert row["maximum_weight_above_a5"] == 0.0
