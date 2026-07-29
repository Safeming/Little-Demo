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
from utils.frozen_semantic_method import load_a7_temporal_contract
from utils.part_label_bank import PART_NAMES, load_part_label_bank, save_a7_part_label_bank
from utils.sparse_robust_temporal_optimizer import run_loco_sparse_capacity


CANDIDATE_ID = "sparse_robust_loco_v4"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the frozen A7 v4 sparse robust candidate."
    )
    parser.add_argument("--a5-bank", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--method-freeze", required=True, type=Path)
    parser.add_argument("--a7-contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-canary-evidence", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _load_evidence(path: Path, *, contract: dict, allow_canary: bool) -> tuple[dict, str]:
    evidence_sha256 = _file_sha256(path)
    if not allow_canary and evidence_sha256 != contract["source_evidence_sha256"]:
        raise ValueError("source evidence SHA-256 does not match the v4 contract")
    with np.load(path, allow_pickle=False) as data:
        evidence = {key: data[key] for key in data.files}
    if str(evidence.get("output_fingerprint", "")) != _payload_fingerprint(evidence):
        raise ValueError("evidence output_fingerprint mismatch")
    if not bool(int(np.asarray(evidence.get("formal_protocol", 0)))) and not allow_canary:
        raise ValueError("non-formal evidence requires --allow-canary-evidence")
    if str(evidence.get("a7_contract_fingerprint", "")) != contract[
        "source_evidence_contract_fingerprint"
    ]:
        raise ValueError("source evidence contract fingerprint mismatch")
    required = {
        "point_count",
        "protocol_fingerprint",
        "temporal_visible_count",
        "temporal_consecutive_visible_count",
        "temporal_target_ratio_mean",
        "temporal_target_ratio_std",
        "temporal_target_flicker",
        "temporal_outer_ratio_mean",
        "temporal_outer_ratio_std",
        "temporal_outer_flicker",
        "temporal_boundary_crossing_rate",
        "temporal_visibility_transition_rate",
        "renderer_target_contribution_sequence",
        "renderer_outer_contribution_sequence",
        "renderer_boundary_contribution_sequence",
        "renderer_sequence_camera_index",
    }
    missing = sorted(required.difference(evidence))
    if missing:
        raise ValueError(f"missing sparse robust evidence fields: {missing}")
    return evidence, evidence_sha256


def _candidate_fingerprint(contract: dict, capacity: dict) -> str:
    moves = {
        part: data["accepted_moves"]
        for part, data in capacity["final"]["per_part"].items()
    }
    encoded = json.dumps(
        {
            "contract": contract["_fingerprint"],
            "candidate_id": CANDIDATE_ID,
            "moves": moves,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _capacity_valid(capacity: dict, *, minimum_target_ratio: float) -> tuple[bool, list[str]]:
    reasons = []
    if not capacity["all_folds_passed"]:
        reasons.append("loco_fold_failure")
    for part, data in capacity["final"]["per_part"].items():
        ratios = data["final_ratios"]
        if min(ratios["target_mean_response"]) < float(minimum_target_ratio):
            reasons.append(f"target_ratio:{part}")
        if max(ratios["outer_normalized_flicker"]) >= 1.0:
            reasons.append(f"outer_flicker:{part}")
        if max(ratios["boundary_normalized_flicker"]) >= 1.0:
            reasons.append(f"boundary_flicker:{part}")
    return not reasons, sorted(reasons)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contract = load_a7_temporal_contract(args.a7_contract, args.method_freeze)
    if contract["freeze_id"] != "a7_sparse_robust_v4_canary_377":
        raise ValueError("sparse robust calibration requires the A7 v4 contract")
    base_bank = load_part_label_bank(args.a5_bank)
    evidence, evidence_sha256 = _load_evidence(
        args.evidence,
        contract=contract,
        allow_canary=bool(args.allow_canary_evidence),
    )
    a5 = np.asarray(base_bank["soft_edit_weights"], dtype=np.float32)
    if int(np.asarray(evidence["point_count"])) != a5.shape[0]:
        raise ValueError("evidence point_count does not match A5 bank")
    processed = tuple(PART_NAMES.index(part) for part in contract["processed_parts"])
    capacity = run_loco_sparse_capacity(
        a5_weights=a5,
        sequences={
            "target": evidence["renderer_target_contribution_sequence"],
            "outer": evidence["renderer_outer_contribution_sequence"],
            "boundary": evidence["renderer_boundary_contribution_sequence"],
        },
        camera_index=evidence["renderer_sequence_camera_index"],
        consecutive_visible_count=evidence["temporal_consecutive_visible_count"],
        processed_part_indices=processed,
        selection_threshold=float(contract["selection_threshold"]),
        min_pair_support=int(contract["min_pair_support"]),
        reduction_fractions=tuple(contract["coordinate_reduction_fractions"]),
        maximum_changed_fraction=float(contract["maximum_changed_fraction"]),
        minimum_camera_target_ratio=float(contract["minimum_camera_target_ratio"]),
        objective_mean_weight=float(contract["objective_mean_weight"]),
        objective_absolute_adjacent_weight=float(
            contract["objective_absolute_adjacent_weight"]
        ),
    )
    weights = np.asarray(capacity.pop("weights"), dtype=np.float32)
    selected_a5 = a5 >= float(contract["selection_threshold"])
    selected_a7 = weights >= float(contract["selection_threshold"])
    crossing_count = int(np.count_nonzero(selected_a5 != selected_a7))
    maximum_above = max(0.0, float(np.max(weights.astype(np.float64) - a5.astype(np.float64))))
    valid, invalid_reasons = _capacity_valid(
        capacity, minimum_target_ratio=float(contract["minimum_camera_target_ratio"])
    )
    if crossing_count:
        invalid_reasons.append("selection_topology_crossing")
    if maximum_above > 1.0e-7:
        invalid_reasons.append("weight_above_a5")
    valid = valid and not invalid_reasons
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
        "valid": bool(valid),
        "invalid_reasons": sorted(set(invalid_reasons)),
        "bank": str(Path(CANDIDATE_ID) / "part_label_bank.npz"),
        "output_bank_fingerprint": bank_fingerprint,
        "selection_crossing_count": crossing_count,
        "maximum_weight_above_a5": maximum_above,
        "source_evidence_sha256": evidence_sha256,
        "capacity_summary": capacity,
    }
    _write_json(candidate_dir / "candidate_summary.json", summary)
    index = {
        "schema_version": 1,
        "base_a5_bank": str(args.a5_bank.resolve()),
        "evidence": str(args.evidence.resolve()),
        "source_evidence_sha256": evidence_sha256,
        "base_method_freeze_fingerprint": contract[
            "base_method_freeze_fingerprint"
        ],
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
