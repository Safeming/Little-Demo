import csv
import json
from pathlib import Path


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _temporal_summary(
    contract_fingerprint: str,
    bank_fingerprint: str,
    *,
    camera: int,
    bad=False,
):
    methods = {"a5": {}, "a7": {}}
    for part in ("hair", "lower"):
        methods["a5"][part] = {
            "fixed_strength_outer_flicker": 1.0,
            "fixed_strength_boundary_flicker": 1.0,
            "visibility_aware_response_flicker": 1.0,
        }
        methods["a7"][part] = {
            "fixed_strength_outer_flicker": 0.99 if part == "lower" else 1.0,
            "fixed_strength_boundary_flicker": 0.99 if part == "lower" else 1.0,
            "visibility_aware_response_flicker": (
                1.01 if bad and part == "lower" else (0.99 if part == "lower" else 1.0)
            ),
        }
    return {
        "subject": "377",
        "camera": camera,
        "frame_start": 0,
        "frame_end": 570,
        "frame_step": 5,
        "frame_count": 114,
        "parts": ["hair", "face", "upper", "lower", "shoes", "skin"],
        "validation_frame_start": 0,
        "validation_frame_end": 570,
        "validation_frame_stride": 5,
        "methods": ["a5", "a7"],
        "metric_row_count": 1368,
        "held_out_camera": camera == 21,
        "a7_bank_fingerprint": bank_fingerprint,
        "a7_contract_fingerprint": contract_fingerprint,
        "canonical_selection_fixed_across_frames": True,
        "common_support_across_methods": True,
        "temporal_metrics": methods,
    }


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
    fieldnames = [
        "baseline", "part", "soft_iou", "iou", "intersection", "union"
    ]
    with (root / "per_part_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def test_v5_1_audit_summary_requires_temporal_and_spatial_gates(tmp_path):
    from tools.summarize_a7_v5_1_audit import main
    from utils.frozen_semantic_method import load_a7_temporal_contract

    contract_path = Path(
        "configs/semantic/frozen_a7_dual_evidence_v5_1_canary_377.json"
    )
    method_freeze = Path("configs/semantic/frozen_a5_main_method_v1.json")
    contract = load_a7_temporal_contract(contract_path, method_freeze)
    bank_fingerprint = "b" * 64
    candidate_index = tmp_path / "candidate_index.json"
    _write_json(
        candidate_index,
        {
            "a7_contract_fingerprint": contract["_fingerprint"],
            "validation_shortlist": ["dual_evidence_constrained_v5_1"],
            "candidates": [
                {
                    "candidate_id": "dual_evidence_constrained_v5_1",
                    "output_bank_fingerprint": bank_fingerprint,
                    "valid": True,
                    "capacity_summary": {
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
                    },
                }
            ],
        },
    )
    temporal_root = tmp_path / "temporal"
    for camera in (21, 22, 23):
        _write_json(
            temporal_root / f"c{camera}" / "summary.json",
            _temporal_summary(
                contract["_fingerprint"], bank_fingerprint, camera=camera
            ),
        )
    spatial_root = tmp_path / "spatial"
    _write_spatial(spatial_root, contract["_fingerprint"], bank_fingerprint)
    output = tmp_path / "promotion_summary.json"

    assert main(
        [
            "--candidate-index", str(candidate_index),
            "--a7-contract", str(contract_path),
            "--method-freeze", str(method_freeze),
            "--temporal-root", str(temporal_root),
            "--spatial-root", str(spatial_root),
            "--output", str(output),
        ]
    ) == 0
    assert json.loads(output.read_text())["passed"] is True

    _write_json(
        temporal_root / "c21" / "summary.json",
        _temporal_summary(
            contract["_fingerprint"], bank_fingerprint, camera=21, bad=True
        ),
    )
    assert main(
        [
            "--candidate-index", str(candidate_index),
            "--a7-contract", str(contract_path),
            "--method-freeze", str(method_freeze),
            "--temporal-root", str(temporal_root),
            "--spatial-root", str(spatial_root),
            "--output", str(output),
        ]
    ) == 2
    assert "visibility_response" in json.loads(output.read_text())["invalid_reasons"]


def test_v5_1_audit_rejects_wrong_temporal_protocol(tmp_path):
    from tools.summarize_a7_v5_1_audit import _summarize_temporal

    contract = {
        "subject": "377",
        "audit_cameras": ["c21"],
        "retrospective_test_cameras": ["c21"],
        "parts": ["hair", "face", "upper", "lower", "shoes", "skin"],
        "validation_frame_start": 0,
        "validation_frame_end": 570,
        "validation_frame_stride": 5,
        "_fingerprint": "c" * 64,
        "minimum_active_temporal_gain": 0.005,
        "maximum_audit_visibility_response_ratio": 1.0,
    }
    root = tmp_path / "temporal"
    _write_json(
        root / "c21" / "summary.json",
        _temporal_summary(contract["_fingerprint"], "b" * 64, camera=17),
    )

    _summary, reasons = _summarize_temporal(root, contract, "b" * 64)

    assert "temporal_protocol:c21" in reasons


def test_v5_1_candidate_requires_construction_gate(tmp_path):
    from tools.summarize_a7_v5_1_audit import load_validated_candidate

    contract = {"_fingerprint": "c" * 64, "evidence_cameras": ["c01", "c05"]}
    index = tmp_path / "candidate_index.json"
    _write_json(
        index,
        {
            "a7_contract_fingerprint": contract["_fingerprint"],
            "validation_shortlist": ["dual_evidence_constrained_v5_1"],
            "candidates": [
                {
                    "candidate_id": "dual_evidence_constrained_v5_1",
                    "output_bank_fingerprint": "b" * 64,
                    "valid": True,
                    "capacity_summary": {
                        "camera_ids": [0, 1],
                        "all_folds_passed": True,
                        "folds": [
                            {
                                "held_out_camera": 0,
                                "construction": {"passed": False},
                                "held_out": {"passed": True},
                                "passed": True,
                            },
                            {
                                "held_out_camera": 1,
                                "construction": {"passed": True},
                                "held_out": {"passed": True},
                                "passed": True,
                            },
                        ],
                        "final": {
                            "construction_evaluation": {"passed": True},
                            "evaluation": {"passed": True},
                        },
                    },
                }
            ],
        },
    )

    try:
        load_validated_candidate(index, contract)
    except ValueError as error:
        assert "construction" in str(error)
    else:
        raise AssertionError("pre-fix candidate must be rejected")
