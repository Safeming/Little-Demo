import json
from pathlib import Path

import numpy as np

from utils.part_label_bank import PART_NAMES


def test_candidate_grid_is_exact_frozen_24_combinations():
    from tools.calibrate_temporal_reliable_a7_weights import candidate_parameter_grid

    grid = candidate_parameter_grid()

    assert len(grid) == 24
    assert {row["lambda_outer"] for row in grid} == {0.25, 0.5, 1.0}
    assert {row["lambda_boundary"] for row in grid} == {0.25, 0.5}
    assert {row["lambda_target"] for row in grid} == {0.0, 0.25}
    assert {row["rho"] for row in grid} == {0.9, 0.95}


def test_candidate_id_depends_only_on_normalized_parameter_json():
    from tools.calibrate_temporal_reliable_a7_weights import (
        candidate_config_fingerprint,
        candidate_id,
    )

    first = {
        "lambda_outer": 0.5,
        "lambda_boundary": 0.25,
        "lambda_target": 0.0,
        "rho": 0.9,
        "min_pair_support": 8,
    }
    second = dict(reversed(list(first.items())))

    assert candidate_id(first) == candidate_id(second)
    assert candidate_id(first) == candidate_config_fingerprint(first)[:12]
    assert len(candidate_config_fingerprint(first)) == 64


def test_a7_contract_freezes_posterior_ceiling_scale():
    from utils.frozen_semantic_method import load_a7_temporal_contract

    contract = load_a7_temporal_contract(
        "configs/semantic/frozen_a7_temporal_reliable_v1.json",
        "configs/semantic/frozen_a5_main_method_v1.json",
    )

    assert contract["max_weight_scale_from_posterior"] == 1.0


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


def _save_evidence(path: Path, point_count: int = 4):
    from tools.build_temporal_reliability_evidence import _payload_fingerprint

    shape = (point_count, len(PART_NAMES))
    arrays = {
        "schema_version": np.array(1, dtype=np.int32),
        "point_count": np.array(point_count, dtype=np.int64),
        "part_names": np.asarray(PART_NAMES, dtype="U16"),
        "formal_protocol": np.array(0, dtype=np.uint8),
        "protocol_fingerprint": np.array("e" * 64),
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
    }
    arrays["output_fingerprint"] = np.array(_payload_fingerprint(arrays))
    np.savez_compressed(path, **arrays)


def test_candidate_cli_generates_all_banks_and_shortlists_at_most_four(tmp_path):
    from tools.calibrate_temporal_reliable_a7_weights import main
    from utils.part_label_bank import load_part_label_bank

    base_path = tmp_path / "a5.npz"
    evidence_path = tmp_path / "evidence.npz"
    output_dir = tmp_path / "candidates"
    _save_base_bank(base_path)
    _save_evidence(evidence_path)

    argv = [
        "--a5-bank",
        str(base_path),
        "--evidence",
        str(evidence_path),
        "--method-freeze",
        "configs/semantic/frozen_a5_main_method_v1.json",
        "--a7-contract",
        "configs/semantic/frozen_a7_temporal_reliable_v1.json",
        "--output-dir",
        str(output_dir),
        "--allow-canary-evidence",
    ]
    assert main(argv) == 0
    first_index = json.loads((output_dir / "candidate_index.json").read_text())
    assert main(argv) == 0
    second_index = json.loads((output_dir / "candidate_index.json").read_text())

    assert first_index == second_index
    assert len(first_index["candidates"]) == 24
    assert len({row["candidate_id"] for row in first_index["candidates"]}) == 24
    assert len(first_index["validation_shortlist"]) <= 4
    for row in first_index["candidates"]:
        bank_path = output_dir / row["candidate_id"] / "part_label_bank.npz"
        summary_path = output_dir / row["candidate_id"] / "candidate_summary.json"
        assert bank_path.is_file()
        assert summary_path.is_file()
        bank = load_part_label_bank(bank_path)
        assert str(bank["candidate_config_fingerprint"]) == row[
            "candidate_config_fingerprint"
        ]


def test_proxy_marks_low_support_candidate_invalid():
    from tools.calibrate_temporal_reliable_a7_weights import evaluate_candidate

    shape = (3, len(PART_NAMES))
    result = evaluate_candidate(
        a5_weights=np.full(shape, 0.5, dtype=np.float32),
        semantic_probs=np.full(shape, 0.8, dtype=np.float32),
        evidence={
            "temporal_consecutive_visible_count": np.zeros(shape, dtype=np.int32),
            "temporal_target_ratio_mean": np.full(shape, 0.8, dtype=np.float32),
            "temporal_outer_ratio_mean": np.full(shape, 0.2, dtype=np.float32),
            "temporal_outer_flicker": np.full(shape, 0.1, dtype=np.float32),
            "temporal_boundary_crossing_rate": np.full(shape, 0.1, dtype=np.float32),
            "temporal_target_flicker": np.full(shape, 0.1, dtype=np.float32),
        },
        parameters={
            "lambda_outer": 0.25,
            "lambda_boundary": 0.25,
            "lambda_target": 0.0,
            "rho": 0.9,
            "min_pair_support": 8,
        },
        max_weight_scale_from_posterior=1.0,
        minimum_evidence_support_coverage=0.8,
    )

    assert result["valid"] is False
    assert "evidence_support_coverage" in result["invalid_reasons"]
