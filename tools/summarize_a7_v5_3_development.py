#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.summarize_a7_v5_1_audit import (
    ACTIVE_PARTS,
    _summarize_spatial,
    _write_json,
    load_validated_candidate,
)
from tools.summarize_a7_v5_2_development import _summarize_temporal_group
from utils.frozen_semantic_method import load_a7_temporal_contract


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarize the A7 v5.3 development run.")
    parser.add_argument("--candidate-index", required=True, type=Path)
    parser.add_argument("--a7-contract", required=True, type=Path)
    parser.add_argument("--method-freeze", required=True, type=Path)
    parser.add_argument("--temporal-root", required=True, type=Path)
    parser.add_argument("--spatial-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    contract = load_a7_temporal_contract(args.a7_contract, args.method_freeze)
    candidate, bank_fingerprint = load_validated_candidate(
        args.candidate_index, contract
    )
    retrospective, retrospective_reasons = _summarize_temporal_group(
        args.temporal_root / "retrospective",
        contract,
        bank_fingerprint,
        cameras=contract["retrospective_test_cameras"],
        maximum_visibility_ratio=contract["maximum_audit_visibility_response_ratio"],
        minimum_target_ratio=contract["minimum_audit_target_response_ratio"],
        constrained_parts=ACTIVE_PARTS,
    )
    spatial, spatial_reasons = _summarize_spatial(
        args.spatial_root, contract, bank_fingerprint
    )
    reasons = sorted(
        {f"retrospective_{reason}" for reason in retrospective_reasons}
        | {f"spatial_{reason}" for reason in spatial_reasons}
    )
    capacity = candidate["capacity_summary"]
    construction = {
        "camera_ids": capacity["camera_ids"],
        "all_folds_passed": bool(capacity["all_folds_passed"]),
        "final_construction_passed": bool(
            capacity.get("final", {})
            .get("construction_evaluation", {})
            .get("passed", False)
        ),
        "final_audit_passed": bool(
            capacity.get("final", {}).get("evaluation", {}).get("passed", False)
        ),
    }
    payload = {
        "schema_version": 1,
        "candidate_id": candidate["candidate_id"],
        "a7_contract_fingerprint": contract["_fingerprint"],
        "a7_bank_fingerprint": bank_fingerprint,
        "paper_test_eligible": False,
        "development_passed": not reasons,
        "invalid_reasons": reasons,
        "construction": construction,
        "retrospective": retrospective,
        "spatial": spatial,
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not reasons else 2


if __name__ == "__main__":
    raise SystemExit(main())
