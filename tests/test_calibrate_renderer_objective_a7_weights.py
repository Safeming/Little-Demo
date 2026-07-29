import json
from pathlib import Path

import numpy as np

from utils.part_label_bank import PART_NAMES


def _save_base_bank(path: Path, point_count: int = 4):
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


def _save_sequence_evidence(path: Path, contract: dict, point_count: int = 4):
    from tools.build_temporal_reliability_evidence import _payload_fingerprint

    shape = (point_count, len(PART_NAMES))
    sequence_shape = (4, *shape)
    arrays = {
        "schema_version": np.array(3, dtype=np.int32),
        "point_count": np.array(point_count, dtype=np.int64),
        "part_names": np.asarray(PART_NAMES, dtype="U16"),
        "formal_protocol": np.array(1, dtype=np.uint8),
        "protocol_fingerprint": np.array("e" * 64),
        "a7_contract_fingerprint": np.array(contract["_fingerprint"]),
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
        "renderer_target_contribution_mean_raw": np.full(shape, 0.8, dtype=np.float32),
        "renderer_outer_contribution_mean_raw": np.full(shape, 0.2, dtype=np.float32),
        "renderer_boundary_contribution_mean_raw": np.full(shape, 0.1, dtype=np.float32),
        "renderer_target_contribution_flicker": np.full(shape, 0.1, dtype=np.float32),
        "renderer_outer_contribution_flicker": np.full(shape, 0.1, dtype=np.float32),
        "renderer_boundary_contribution_flicker": np.full(shape, 0.1, dtype=np.float32),
        "renderer_target_contribution_sequence": np.ones(sequence_shape, dtype=np.float16),
        "renderer_outer_contribution_sequence": np.ones(sequence_shape, dtype=np.float16),
        "renderer_boundary_contribution_sequence": np.ones(sequence_shape, dtype=np.float16),
        "renderer_sequence_camera_index": np.array([0, 0, 1, 1], dtype=np.int16),
        "renderer_sequence_frame_index": np.array([0, 5, 0, 5], dtype=np.int32),
    }
    arrays["output_fingerprint"] = np.array(_payload_fingerprint(arrays))
    np.savez_compressed(path, **arrays)


def test_renderer_objective_cli_generates_exact_two_deterministic_candidates(tmp_path):
    from tools.calibrate_renderer_objective_a7_weights import main
    from utils.frozen_semantic_method import load_a7_temporal_contract

    contract_path = Path(
        "configs/semantic/frozen_a7_renderer_objective_v3_canary_377.json"
    )
    freeze_path = Path("configs/semantic/frozen_a5_main_method_v1.json")
    contract = load_a7_temporal_contract(contract_path, freeze_path)
    base = tmp_path / "a5.npz"
    evidence = tmp_path / "evidence.npz"
    output = tmp_path / "candidates"
    _save_base_bank(base)
    _save_sequence_evidence(evidence, contract)
    argv = [
        "--a5-bank", str(base),
        "--evidence", str(evidence),
        "--method-freeze", str(freeze_path),
        "--a7-contract", str(contract_path),
        "--output-dir", str(output),
    ]

    assert main(argv) == 0
    first = json.loads((output / "candidate_index.json").read_text())
    assert main(argv) == 0
    second = json.loads((output / "candidate_index.json").read_text())

    assert first == second
    assert first["candidate_count"] == 2
    assert first["validation_shortlist"] == [
        "bounded_damping_005",
        "bounded_retention_010",
    ]
    assert [row["candidate_id"] for row in first["candidates"]] == first[
        "validation_shortlist"
    ]
    for row in first["candidates"]:
        assert (output / row["candidate_id"] / "part_label_bank.npz").is_file()
        assert row["calibration_summary"]["maximum_weight_above_a5"] == 0.0
        assert "aggregate_ratios" in row["renderer_sequence_objective"]
