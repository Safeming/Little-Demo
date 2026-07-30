import csv
import json
from pathlib import Path

import pytest


PARTS = ["hair", "face", "upper", "lower", "shoes", "skin"]


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _capacity_summary():
    return {
        "camera_ids": [0, 1, 2, 3],
        "all_folds_passed": True,
        "folds": [
            {
                "held_out_camera": camera,
                "construction": {"passed": True},
                "held_out": {"passed": True},
                "passed": True,
            }
            for camera in range(4)
        ],
        "final": {
            "construction_evaluation": {"passed": True},
            "evaluation": {"passed": True},
        },
    }


def _write_temporal(
    root: Path,
    *,
    camera: int,
    contract_fingerprint: str,
    bank_fingerprint: str,
    lower_visibility_ratio: float,
    lower_target_ratio: float,
    lower_temporal_ratio: float,
):
    methods = {"a5": {}, "a7": {}}
    for part in ("hair", "lower"):
        methods["a5"][part] = {
            "fixed_strength_outer_flicker": 1.0,
            "fixed_strength_boundary_flicker": 1.0,
            "visibility_aware_response_flicker": 1.0,
        }
        methods["a7"][part] = {
            "fixed_strength_outer_flicker": (
                lower_temporal_ratio if part == "lower" else 1.0
            ),
            "fixed_strength_boundary_flicker": (
                lower_temporal_ratio if part == "lower" else 1.0
            ),
            "visibility_aware_response_flicker": (
                lower_visibility_ratio if part == "lower" else 1.0
            ),
        }
    output = root / f"c{camera}"
    _write_json(
        output / "summary.json",
        {
            "subject": "377",
            "camera": camera,
            "frame_start": 0,
            "frame_end": 570,
            "frame_step": 5,
            "frame_count": 114,
            "parts": PARTS,
            "methods": ["a5", "a7"],
            "metric_row_count": 1368,
            "held_out_camera": camera in (21, 22, 23),
            "a7_bank_fingerprint": bank_fingerprint,
            "a7_contract_fingerprint": contract_fingerprint,
            "canonical_selection_fixed_across_frames": True,
            "common_support_across_methods": True,
            "temporal_metrics": methods,
        },
    )
    fields = ["frame", "part", "method", "edit_target_delta_mean"]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in range(0, 570, 5):
            for part in ("hair", "lower"):
                for method in ("a5", "a7"):
                    ratio = (
                        lower_target_ratio
                        if part == "lower" and method == "a7"
                        else 1.0
                    )
                    writer.writerow(
                        {
                            "frame": frame,
                            "part": part,
                            "method": method,
                            "edit_target_delta_mean": ratio,
                        }
                    )


def _write_spatial(root: Path, contract_fingerprint: str, bank_fingerprint: str):
    guards = []
    for baseline, scale in (("A5", 1.0), ("A7", 1.001)):
        for part in ("face", "hair", "upper", "lower", "shoes", "skin"):
            guards.append(
                {
                    "baseline": baseline,
                    "part": part,
                    "coverage_rate": 1.0,
                    "pooled_outer_burden": scale,
                    "pooled_boundary_burden": scale,
                }
            )
    _write_json(
        root / "summary.json",
        {
            "a7_bank_fingerprint": bank_fingerprint,
            "a7_contract_fingerprint": contract_fingerprint,
            "protocol_split": "test",
            "canonical_selection_fixed_across_frames": True,
            "common_support_across_methods": True,
            "spatial_guard_metrics": guards,
        },
    )
    fields = ["baseline", "part", "soft_iou", "iou", "intersection", "union"]
    with (root / "per_part_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for baseline, delta in (("A5", 0.0), ("A7", -0.001)):
            for part in ("face", "hair", "upper", "lower", "shoes", "skin"):
                writer.writerow(
                    {
                        "baseline": baseline,
                        "part": part,
                        "soft_iou": 0.7 + delta,
                        "iou": 0.8 + delta,
                        "intersection": 80,
                        "union": 100,
                    }
                )


def test_v5_2_development_summary_separates_validation_and_retrospective(tmp_path):
    from tools.summarize_a7_v5_2_development import main
    from utils.frozen_semantic_method import load_a7_temporal_contract

    contract_path = Path(
        "configs/semantic/frozen_a7_dual_evidence_v5_2_canary_377.json"
    )
    method_freeze = Path("configs/semantic/frozen_a5_main_method_v1.json")
    contract = load_a7_temporal_contract(contract_path, method_freeze)
    bank_fingerprint = "b" * 64
    candidate_index = tmp_path / "candidate_index.json"
    _write_json(
        candidate_index,
        {
            "a7_contract_fingerprint": contract["_fingerprint"],
            "validation_shortlist": ["dual_evidence_constrained_v5_2"],
            "candidates": [
                {
                    "candidate_id": "dual_evidence_constrained_v5_2",
                    "output_bank_fingerprint": bank_fingerprint,
                    "valid": True,
                    "capacity_summary": _capacity_summary(),
                }
            ],
        },
    )
    temporal_root = tmp_path / "temporal"
    for camera in (17, 18, 19, 20):
        _write_temporal(
            temporal_root / "validation",
            camera=camera,
            contract_fingerprint=contract["_fingerprint"],
            bank_fingerprint=bank_fingerprint,
            lower_visibility_ratio=0.999,
            lower_target_ratio=0.996,
            lower_temporal_ratio=0.98,
        )
    for camera in (21, 22, 23):
        _write_temporal(
            temporal_root / "retrospective",
            camera=camera,
            contract_fingerprint=contract["_fingerprint"],
            bank_fingerprint=bank_fingerprint,
            lower_visibility_ratio=0.9999,
            lower_target_ratio=0.995,
            lower_temporal_ratio=0.99,
        )
    spatial = tmp_path / "spatial"
    _write_spatial(spatial, contract["_fingerprint"], bank_fingerprint)
    output = tmp_path / "development_summary.json"

    assert main(
        [
            "--candidate-index", str(candidate_index),
            "--a7-contract", str(contract_path),
            "--method-freeze", str(method_freeze),
            "--temporal-root", str(temporal_root),
            "--spatial-root", str(spatial),
            "--output", str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text())
    assert payload["development_passed"] is True
    assert payload["paper_test_eligible"] is False
    assert payload["validation"]["minimum_target_response_ratio"] == pytest.approx(
        0.996
    )
    assert payload["retrospective"]["minimum_target_response_ratio"] == pytest.approx(
        0.995
    )

    _write_temporal(
        temporal_root / "retrospective",
        camera=22,
        contract_fingerprint=contract["_fingerprint"],
        bank_fingerprint=bank_fingerprint,
        lower_visibility_ratio=1.0001,
        lower_target_ratio=0.995,
        lower_temporal_ratio=0.99,
    )
    assert main(
        [
            "--candidate-index", str(candidate_index),
            "--a7-contract", str(contract_path),
            "--method-freeze", str(method_freeze),
            "--temporal-root", str(temporal_root),
            "--spatial-root", str(spatial),
            "--output", str(output),
        ]
    ) == 2
    assert "retrospective_visibility_response" in json.loads(
        output.read_text()
    )["invalid_reasons"]
