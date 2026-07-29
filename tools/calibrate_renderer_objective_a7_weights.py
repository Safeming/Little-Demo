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

from tools.build_temporal_reliability_evidence import _payload_fingerprint
from utils.bounded_temporal_calibration import calibrate_bounded_a7_weights
from utils.frozen_semantic_method import load_a7_temporal_contract
from utils.part_label_bank import (
    PART_NAMES,
    load_part_label_bank,
    save_a7_part_label_bank,
)
from utils.renderer_sequence_objective import summarize_renderer_sequence_objective
from utils.temporal_reliability_calibration import compute_temporal_reliability


SEQUENCE_FIELDS = (
    "renderer_target_contribution_sequence",
    "renderer_outer_contribution_sequence",
    "renderer_boundary_contribution_sequence",
    "renderer_sequence_camera_index",
    "renderer_sequence_frame_index",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate two bounded A7 banks and score renderer sequences."
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
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _candidate_fingerprint(contract: dict, policy: dict) -> str:
    payload = {
        "policy": policy,
        "lambda_outer": float(contract["lambda_outer"]),
        "lambda_boundary": float(contract["lambda_boundary"]),
        "lambda_target": float(contract["lambda_target"]),
        "min_pair_support": int(contract["min_pair_support"]),
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_evidence(path: Path, *, contract: dict, allow_canary: bool) -> dict:
    with np.load(path, allow_pickle=False) as data:
        evidence = {key: data[key] for key in data.files}
    stored = str(evidence.get("output_fingerprint", ""))
    if not stored or stored != _payload_fingerprint(evidence):
        raise ValueError("evidence output_fingerprint mismatch")
    if not bool(int(np.asarray(evidence.get("formal_protocol", 0)))) and not allow_canary:
        raise ValueError("non-formal evidence requires --allow-canary-evidence")
    if str(evidence.get("a7_contract_fingerprint", "")) != contract["_fingerprint"]:
        raise ValueError("evidence A7 contract fingerprint mismatch")
    required = {
        "point_count",
        "protocol_fingerprint",
        "temporal_consecutive_visible_count",
        "renderer_target_contribution_mean_raw",
        "renderer_target_contribution_flicker",
        "renderer_outer_contribution_flicker",
        "renderer_boundary_contribution_flicker",
        *SEQUENCE_FIELDS,
    }
    missing = sorted(required.difference(evidence))
    if missing:
        raise ValueError(f"missing renderer sequence evidence fields: {missing}")
    return evidence


def _evidence_screen(objective: dict, processed_parts: list[int]) -> dict:
    failures = []
    checks = {}
    for part_index in processed_parts:
        ratios = objective["per_part_ratios"][str(part_index)]
        part_checks = {
            "target_response_ge_0_95": ratios["target_mean_response"] >= 0.95,
            "outer_mean_le_1": ratios["outer_mean_response"] <= 1.0,
            "outer_adjacent_le_1": ratios["outer_adjacent_absolute_change"] <= 1.0,
            "boundary_adjacent_le_1": ratios[
                "boundary_adjacent_absolute_change"
            ] <= 1.0,
        }
        checks[str(part_index)] = part_checks
        failures.extend(
            f"{name}:{part_index}" for name, passed in part_checks.items() if not passed
        )
    return {"passed": not failures, "failures": failures, "checks": checks}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contract = load_a7_temporal_contract(args.a7_contract, args.method_freeze)
    if contract["freeze_id"] != "a7_renderer_objective_v3_canary_377":
        raise ValueError("renderer objective calibration requires the A7 v3 contract")
    base_bank = load_part_label_bank(args.a5_bank)
    evidence = _load_evidence(
        args.evidence, contract=contract, allow_canary=bool(args.allow_canary_evidence)
    )
    a5 = np.asarray(base_bank["soft_edit_weights"], dtype=np.float32)
    if int(np.asarray(evidence["point_count"])) != a5.shape[0]:
        raise ValueError("evidence point_count does not match A5 bank")

    reliability, reliability_summary = compute_temporal_reliability(
        consecutive_visible_count=evidence["temporal_consecutive_visible_count"],
        temporal_outer_flicker=evidence["renderer_outer_contribution_flicker"],
        temporal_boundary_crossing_rate=evidence[
            "renderer_boundary_contribution_flicker"
        ],
        temporal_target_flicker=evidence["renderer_target_contribution_flicker"],
        lambda_outer=float(contract["lambda_outer"]),
        lambda_boundary=float(contract["lambda_boundary"]),
        lambda_target=float(contract["lambda_target"]),
        min_pair_support=int(contract["min_pair_support"]),
    )
    frozen_parts = [PART_NAMES.index(part) for part in contract["frozen_parts"]]
    processed_parts = [index for index in range(len(PART_NAMES)) if index not in frozen_parts]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for policy in contract["candidate_policies"]:
        identifier = str(policy["name"])
        fingerprint = _candidate_fingerprint(contract, policy)
        weights, calibration = calibrate_bounded_a7_weights(
            a5_weights=a5,
            target_contribution_mean=evidence[
                "renderer_target_contribution_mean_raw"
            ],
            temporal_reliability=reliability,
            consecutive_visible_count=evidence[
                "temporal_consecutive_visible_count"
            ],
            rho=float(policy["rho"]),
            min_pair_support=int(contract["min_pair_support"]),
            minimum_weight_ratio_from_a5=float(
                policy["minimum_weight_ratio_from_a5"]
            ),
            restore_target_mass=bool(policy["restore_target_mass"]),
            maximum_part_weight_l1_from_a5=float(
                policy["maximum_part_weight_l1_from_a5"]
            ),
            frozen_part_indices=tuple(frozen_parts),
            selection_threshold=float(contract["selection_threshold"]),
            preserve_selection_topology=bool(
                contract["preserve_a5_selection_topology"]
            ),
        )
        objective = summarize_renderer_sequence_objective(
            a5_weights=a5,
            candidate_weights=weights,
            target_sequence=evidence["renderer_target_contribution_sequence"],
            outer_sequence=evidence["renderer_outer_contribution_sequence"],
            boundary_sequence=evidence["renderer_boundary_contribution_sequence"],
            camera_index=evidence["renderer_sequence_camera_index"],
            frame_index=evidence["renderer_sequence_frame_index"],
            processed_part_indices=processed_parts,
        )
        support = np.asarray(evidence["temporal_consecutive_visible_count"]) >= int(
            contract["min_pair_support"]
        )
        coverage = {}
        invalid_reasons = list(calibration["invalid_reasons"])
        for part_index in processed_parts:
            denominator = float(np.sum(a5[:, part_index], dtype=np.float64))
            numerator = float(
                np.sum(a5[:, part_index] * support[:, part_index], dtype=np.float64)
            )
            ratio = numerator / denominator if denominator > 0.0 else 1.0
            coverage[str(part_index)] = ratio
            if ratio < float(contract["minimum_evidence_support_coverage"]):
                invalid_reasons.append(f"evidence_support_coverage:{part_index}")
        bank_dir = output_dir / identifier
        bank_path = bank_dir / "part_label_bank.npz"
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
                "evidence_protocol_fingerprint": str(
                    evidence["protocol_fingerprint"]
                ),
                "candidate_config_fingerprint": fingerprint,
            },
        )
        summary = {
            "candidate_id": identifier,
            "candidate_config_fingerprint": fingerprint,
            "policy": policy,
            "valid": not invalid_reasons,
            "invalid_reasons": sorted(set(invalid_reasons)),
            "bank": str(Path(identifier) / "part_label_bank.npz"),
            "output_bank_fingerprint": bank_fingerprint,
            "calibration_summary": calibration,
            "reliability_summary": reliability_summary,
            "evidence_support_coverage": coverage,
            "renderer_sequence_objective": objective,
            "evidence_screen": _evidence_screen(objective, processed_parts),
        }
        _write_json(bank_dir / "candidate_summary.json", summary)
        rows.append(summary)

    shortlist = [row["candidate_id"] for row in rows if row["valid"]]
    index = {
        "schema_version": 1,
        "base_a5_bank": str(args.a5_bank.resolve()),
        "evidence": str(args.evidence.resolve()),
        "base_method_freeze_fingerprint": contract[
            "base_method_freeze_fingerprint"
        ],
        "a7_contract_fingerprint": contract["_fingerprint"],
        "evidence_protocol_fingerprint": str(evidence["protocol_fingerprint"]),
        "candidate_count": len(rows),
        "valid_candidate_count": len(shortlist),
        "validation_shortlist": shortlist,
        "candidates": rows,
    }
    _write_json(output_dir / "candidate_index.json", index)
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
