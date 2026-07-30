#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_temporal_reliability_evidence import (
    _file_sha256,
    _payload_fingerprint,
)
from utils.constrained_sparse_temporal_optimizer import run_constrained_v5_capacity
from utils.frozen_semantic_method import load_a7_temporal_contract
from utils.part_label_bank import PART_NAMES, load_part_label_bank, save_a7_part_label_bank


CANDIDATE_ID = "dual_evidence_constrained_v5"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the frozen A7 v5 dual-evidence constrained candidate."
    )
    parser.add_argument("--a5-bank", required=True, type=Path)
    parser.add_argument("--v4-bank", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--method-freeze", required=True, type=Path)
    parser.add_argument("--a7-contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-canary-inputs", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _load_evidence(path: Path, *, contract: dict, allow_canary: bool) -> tuple[dict, str]:
    evidence_sha256 = _file_sha256(path)
    with np.load(path, allow_pickle=False) as data:
        evidence = {key: data[key] for key in data.files}
    if str(evidence.get("output_fingerprint", "")) != _payload_fingerprint(evidence):
        raise ValueError("evidence output_fingerprint mismatch")
    if not bool(int(np.asarray(evidence.get("formal_protocol", 0)))) and not allow_canary:
        raise ValueError("non-formal evidence requires --allow-canary-inputs")
    if str(evidence.get("a7_contract_fingerprint", "")) != contract["_fingerprint"]:
        raise ValueError("evidence A7 contract fingerprint mismatch")
    required = {
        "point_count",
        "protocol_fingerprint",
        "temporal_consecutive_visible_count",
        "renderer_target_contribution_sequence",
        "renderer_outer_contribution_sequence",
        "renderer_boundary_contribution_sequence",
        "renderer_selection_target_contribution_sequence",
        "renderer_selection_outer_contribution_sequence",
        "renderer_selection_boundary_contribution_sequence",
        "renderer_sequence_target_pixel_count",
        "renderer_sequence_camera_index",
        "renderer_sequence_frame_index",
    }
    missing = sorted(required.difference(evidence))
    if missing:
        raise ValueError(f"missing A7 v5 evidence fields: {missing}")
    if int(np.asarray(evidence["schema_version"])) != 4:
        raise ValueError("A7 v5 evidence schema_version must be 4")
    if str(evidence.get("evidence_mode", "")) != contract["evidence_mode"]:
        raise ValueError("A7 v5 evidence_mode mismatch")
    if str(evidence.get("renderer_attribution", "")) != contract[
        "renderer_attribution"
    ]:
        raise ValueError("A7 v5 renderer_attribution mismatch")
    if tuple(str(value) for value in np.asarray(evidence["part_names"]).reshape(-1)) != tuple(
        PART_NAMES
    ):
        raise ValueError("A7 v5 evidence part_names mismatch")

    point_count = int(np.asarray(evidence["point_count"]))
    camera_index = np.asarray(evidence["renderer_sequence_camera_index"]).reshape(-1)
    frame_index = np.asarray(evidence["renderer_sequence_frame_index"]).reshape(-1)
    sample_count = camera_index.size
    expected_shape = (sample_count, point_count, len(PART_NAMES))
    sequence_fields = (
        "renderer_target_contribution_sequence",
        "renderer_outer_contribution_sequence",
        "renderer_boundary_contribution_sequence",
        "renderer_selection_target_contribution_sequence",
        "renderer_selection_outer_contribution_sequence",
        "renderer_selection_boundary_contribution_sequence",
    )
    if any(np.asarray(evidence[field]).shape != expected_shape for field in sequence_fields):
        raise ValueError("A7 v5 contribution sequence shape mismatch")
    if np.asarray(evidence["renderer_sequence_target_pixel_count"]).shape != (
        sample_count,
        len(PART_NAMES),
    ):
        raise ValueError("A7 v5 target pixel count shape mismatch")
    if np.asarray(evidence["temporal_consecutive_visible_count"]).shape != (
        point_count,
        len(PART_NAMES),
    ):
        raise ValueError("A7 v5 support shape mismatch")
    if frame_index.shape != (sample_count,) or np.any(np.diff(camera_index) < 0):
        raise ValueError("A7 v5 renderer sequence ordering mismatch")
    for camera in np.unique(camera_index):
        frames = frame_index[camera_index == camera]
        if frames.size > 1 and np.any(np.diff(frames) <= 0):
            raise ValueError("A7 v5 frames must increase within each camera")
    if not allow_canary:
        expected_frames = list(
            range(
                int(contract["evidence_frame_start"]),
                int(contract["evidence_frame_end"]),
                int(contract["evidence_frame_stride"]),
            )
        )
        expected_cameras = np.repeat(
            np.arange(len(contract["evidence_cameras"]), dtype=np.int16),
            len(expected_frames),
        )
        expected_frame_index = np.tile(
            np.asarray(expected_frames, dtype=np.int32), len(contract["evidence_cameras"])
        )
        if not np.array_equal(camera_index, expected_cameras) or not np.array_equal(
            frame_index, expected_frame_index
        ):
            raise ValueError("formal A7 v5 evidence camera/frame sequence mismatch")
        if tuple(str(value) for value in np.asarray(evidence["cameras"]).reshape(-1)) != tuple(
            contract["evidence_cameras"]
        ):
            raise ValueError("formal A7 v5 evidence cameras mismatch")
    return evidence, evidence_sha256


def _candidate_fingerprint(contract: dict, capacity: dict) -> str:
    encoded = json.dumps(
        {
            "contract": contract["_fingerprint"],
            "candidate_id": CANDIDATE_ID,
            "optimization": capacity["final"]["optimization"],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contract = load_a7_temporal_contract(args.a7_contract, args.method_freeze)
    if contract["freeze_id"] != "a7_dual_evidence_v5_canary_377":
        raise ValueError("constrained calibration requires the A7 v5 contract")
    if (
        not args.allow_canary_inputs
        and _file_sha256(args.v4_bank) != contract["source_v4_bank_sha256"]
    ):
        raise ValueError("source v4 bank SHA-256 mismatch")

    base_bank = load_part_label_bank(args.a5_bank)
    v4_bank = load_part_label_bank(args.v4_bank)
    evidence, evidence_sha256 = _load_evidence(
        args.evidence, contract=contract, allow_canary=bool(args.allow_canary_inputs)
    )
    a5 = np.asarray(base_bank["soft_edit_weights"], dtype=np.float32)
    v4 = np.asarray(v4_bank["soft_edit_weights"], dtype=np.float32)
    if v4.shape != a5.shape:
        raise ValueError("v4 bank weights must match A5")
    if int(np.asarray(evidence["point_count"])) != a5.shape[0]:
        raise ValueError("evidence point_count does not match A5 bank")

    capacity = run_constrained_v5_capacity(
        a5_weights=a5,
        v4_weights=v4,
        sequences={
            "target": evidence["renderer_target_contribution_sequence"],
            "outer": evidence["renderer_outer_contribution_sequence"],
            "boundary": evidence["renderer_boundary_contribution_sequence"],
            "selection_target": evidence[
                "renderer_selection_target_contribution_sequence"
            ],
            "selection_outer": evidence[
                "renderer_selection_outer_contribution_sequence"
            ],
            "selection_boundary": evidence[
                "renderer_selection_boundary_contribution_sequence"
            ],
        },
        target_pixel_count=evidence["renderer_sequence_target_pixel_count"],
        camera_index=evidence["renderer_sequence_camera_index"],
        consecutive_visible_count=evidence["temporal_consecutive_visible_count"],
        hair_index=PART_NAMES.index("hair"),
        lower_index=PART_NAMES.index("lower"),
        selection_threshold=float(contract["selection_threshold"]),
        min_pair_support=int(contract["min_pair_support"]),
        reduction_fractions=tuple(contract["coordinate_reduction_fractions"]),
        maximum_changed_fraction=float(contract["maximum_changed_fraction"]),
        maximum_hair_changed_count=int(contract["maximum_hair_changed_count"]),
        minimum_camera_target_ratio=float(contract["minimum_camera_target_ratio"]),
        maximum_camera_soft_iou_drop=float(
            contract["maximum_evidence_soft_iou_drop"]
        ),
        maximum_camera_visibility_response_ratio=float(
            contract["maximum_visibility_response_ratio"]
        ),
        objective_mean_weight=float(contract["objective_mean_weight"]),
        objective_absolute_adjacent_weight=float(
            contract["objective_absolute_adjacent_weight"]
        ),
        minimum_active_temporal_gain=float(contract["minimum_active_temporal_gain"]),
        source_v4_minimum_camera_target_ratio=float(
            contract["source_v4_minimum_camera_target_ratio"]
        ),
    )
    weights = np.asarray(capacity.pop("weights"), dtype=np.float32)
    selected_a5 = a5 >= float(contract["selection_threshold"])
    selected_v5 = weights >= float(contract["selection_threshold"])
    crossing_count = int(np.count_nonzero(selected_a5 != selected_v5))
    maximum_above = max(
        0.0, float(np.max(weights.astype(np.float64) - a5.astype(np.float64)))
    )
    frozen_indices = [PART_NAMES.index(part) for part in contract["frozen_parts"]]
    frozen_exact = bool(np.array_equal(weights[:, frozen_indices], a5[:, frozen_indices]))
    valid = bool(
        capacity["all_folds_passed"]
        and capacity["final"]["evaluation"]["passed"]
        and crossing_count == 0
        and maximum_above <= 1.0e-7
        and frozen_exact
    )
    invalid_reasons = []
    if not capacity["all_folds_passed"]:
        invalid_reasons.append("loco_fold_failure")
    if not capacity["final"]["evaluation"]["passed"]:
        invalid_reasons.append("final_evidence_gate_failure")
    if crossing_count:
        invalid_reasons.append("selection_topology_crossing")
    if maximum_above > 1.0e-7:
        invalid_reasons.append("weight_above_a5")
    if not frozen_exact:
        invalid_reasons.append("frozen_part_changed")

    candidate_fingerprint = _candidate_fingerprint(contract, capacity)
    output_dir = args.output_dir.resolve()
    candidate_dir = output_dir / CANDIDATE_ID
    bank_path = candidate_dir / "part_label_bank.npz"
    reliability = (
        np.asarray(evidence["temporal_consecutive_visible_count"])
        >= int(contract["min_pair_support"])
    ).astype(np.float32)
    bank_fingerprint = save_a7_part_label_bank(
        bank_path,
        base_bank_path=args.a5_bank,
        temporal_evidence=evidence,
        temporal_reliability=reliability,
        soft_edit_weights=weights,
        provenance={
            "base_method_freeze_fingerprint": contract[
                "base_method_freeze_fingerprint"
            ],
            "a7_contract_fingerprint": contract["_fingerprint"],
            "evidence_protocol_fingerprint": str(evidence["protocol_fingerprint"]),
            "candidate_config_fingerprint": candidate_fingerprint,
        },
    )
    summary = {
        "candidate_id": CANDIDATE_ID,
        "candidate_config_fingerprint": candidate_fingerprint,
        "valid": valid,
        "invalid_reasons": invalid_reasons,
        "bank": str(Path(CANDIDATE_ID) / "part_label_bank.npz"),
        "output_bank_fingerprint": bank_fingerprint,
        "selection_crossing_count": crossing_count,
        "maximum_weight_above_a5": maximum_above,
        "frozen_parts_exact": frozen_exact,
        "source_evidence_sha256": evidence_sha256,
        "source_v4_bank_sha256": _file_sha256(args.v4_bank),
        "capacity_summary": capacity,
    }
    _write_json(candidate_dir / "candidate_summary.json", summary)
    index = {
        "schema_version": 1,
        "base_a5_bank": str(args.a5_bank.resolve()),
        "source_v4_bank": str(args.v4_bank.resolve()),
        "evidence": str(args.evidence.resolve()),
        "a7_contract_fingerprint": contract["_fingerprint"],
        "candidate_count": 1,
        "valid_candidate_count": int(valid),
        "validation_shortlist": [CANDIDATE_ID] if valid else [],
        "candidates": [summary],
    }
    _write_json(output_dir / "candidate_index.json", index)
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
