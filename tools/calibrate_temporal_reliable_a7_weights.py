#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_temporal_reliability_evidence import _payload_fingerprint
from utils.frozen_semantic_method import load_a7_temporal_contract
from utils.part_label_bank import (
    PART_NAMES,
    load_part_label_bank,
    save_a7_part_label_bank,
)
from utils.temporal_reliability_calibration import (
    calibrate_a7_soft_edit_weights,
    compute_temporal_reliability,
)


def candidate_parameter_grid() -> list[dict[str, float]]:
    return [
        {
            "lambda_outer": float(lambda_outer),
            "lambda_boundary": float(lambda_boundary),
            "lambda_target": float(lambda_target),
            "rho": float(rho),
        }
        for lambda_outer, lambda_boundary, lambda_target, rho in itertools.product(
            (0.25, 0.50, 1.00),
            (0.25, 0.50),
            (0.00, 0.25),
            (0.90, 0.95),
        )
    ]


def _normalized_candidate_json(parameters: dict) -> str:
    normalized = {
        "lambda_boundary": float(parameters["lambda_boundary"]),
        "lambda_outer": float(parameters["lambda_outer"]),
        "lambda_target": float(parameters["lambda_target"]),
        "min_pair_support": int(parameters["min_pair_support"]),
        "rho": float(parameters["rho"]),
    }
    return json.dumps(
        normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


def candidate_config_fingerprint(parameters: dict) -> str:
    return hashlib.sha256(_normalized_candidate_json(parameters).encode("utf-8")).hexdigest()


def candidate_id(parameters: dict) -> str:
    return candidate_config_fingerprint(parameters)[:12]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and proxy-rank frozen A7 temporal-reliable candidate banks."
    )
    parser.add_argument("--a5-bank", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--method-freeze", required=True, type=Path)
    parser.add_argument("--a7-contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-validation-candidates", type=int, default=4)
    parser.add_argument("--allow-canary-evidence", action="store_true")
    return parser.parse_args(argv)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def evaluate_candidate(
    *,
    a5_weights: np.ndarray,
    semantic_probs: np.ndarray,
    evidence: dict[str, np.ndarray],
    parameters: dict,
    max_weight_scale_from_posterior: float,
    minimum_evidence_support_coverage: float,
    minimum_carrier_support_ratio: float = 0.0,
    minimum_carrier_existing_weight: float = 0.0,
    carrier_ranking: str = "posterior_target_reliability_support",
    evidence_mode: str = "footprint",
    frozen_part_indices: tuple[int, ...] | list[int] = (),
    selection_threshold: float = 0.0,
    preserve_selection_topology: bool = False,
) -> dict:
    mode = str(evidence_mode)
    if mode == "renderer_aligned":
        target_key = "renderer_target_contribution_weight"
        outer_key = "renderer_outer_contribution_weight"
        target_flicker_key = "renderer_target_contribution_flicker"
        outer_flicker_key = "renderer_outer_contribution_flicker"
        boundary_key = "renderer_boundary_contribution_flicker"
    elif mode == "footprint":
        target_key = "temporal_target_ratio_mean"
        outer_key = "temporal_outer_ratio_mean"
        target_flicker_key = "temporal_target_flicker"
        outer_flicker_key = "temporal_outer_flicker"
        boundary_key = "temporal_boundary_crossing_rate"
    else:
        raise ValueError(f"unsupported evidence_mode: {mode}")
    reliability, reliability_summary = compute_temporal_reliability(
        consecutive_visible_count=evidence["temporal_consecutive_visible_count"],
        temporal_outer_flicker=evidence[outer_flicker_key],
        temporal_boundary_crossing_rate=evidence[boundary_key],
        temporal_target_flicker=evidence[target_flicker_key],
        lambda_outer=float(parameters["lambda_outer"]),
        lambda_boundary=float(parameters["lambda_boundary"]),
        lambda_target=float(parameters["lambda_target"]),
        min_pair_support=int(parameters["min_pair_support"]),
    )
    weights, calibration_summary = calibrate_a7_soft_edit_weights(
        a5_weights=a5_weights,
        semantic_probs=semantic_probs,
        temporal_target_ratio_mean=evidence[target_key],
        temporal_outer_ratio_mean=evidence[outer_key],
        temporal_reliability=reliability,
        consecutive_visible_count=evidence["temporal_consecutive_visible_count"],
        rho=float(parameters["rho"]),
        min_pair_support=int(parameters["min_pair_support"]),
        max_weight_scale_from_posterior=float(max_weight_scale_from_posterior),
        minimum_carrier_support_ratio=float(minimum_carrier_support_ratio),
        minimum_carrier_existing_weight=float(minimum_carrier_existing_weight),
        carrier_ranking=str(carrier_ranking),
        frozen_part_indices=frozen_part_indices,
        selection_threshold=float(selection_threshold),
        preserve_selection_topology=bool(preserve_selection_topology),
    )

    support = evidence["temporal_consecutive_visible_count"] >= int(
        parameters["min_pair_support"]
    )
    target = np.asarray(evidence[target_key], dtype=np.float64)
    outer = np.asarray(evidence[outer_key], dtype=np.float64)
    boundary = np.asarray(evidence[boundary_key], dtype=np.float64)
    outer_flicker = np.asarray(evidence[outer_flicker_key], dtype=np.float64)
    a5 = np.asarray(a5_weights, dtype=np.float64)
    calibrated = np.asarray(weights, dtype=np.float64)

    per_part = []
    invalid_reasons = []
    for part_index, part_name in enumerate(PART_NAMES):
        a5_weight_sum = float(np.sum(a5[:, part_index], dtype=np.float64))
        supported_weight = float(
            np.sum(a5[:, part_index] * support[:, part_index], dtype=np.float64)
        )
        coverage = _safe_ratio(supported_weight, a5_weight_sum)
        calibration = calibration_summary["per_part"][part_index]
        if int(calibration.get("selection_crossing_count", 0)) > 0:
            invalid_reasons.append("selection_topology_crossing")
            invalid_reasons.append(f"selection_topology_crossing:{part_name}")
        deficit_limit = 0.02 * float(calibration["a5_target_mass"])
        deficit_excess = float(calibration["remaining_deficit"]) > deficit_limit + 1e-8
        if a5_weight_sum > 0.0 and coverage < float(
            minimum_evidence_support_coverage
        ):
            invalid_reasons.append("evidence_support_coverage")
            invalid_reasons.append(f"evidence_support_coverage:{part_name}")
        if deficit_excess:
            invalid_reasons.append("target_deficit")
            invalid_reasons.append(f"target_deficit:{part_name}")

        a5_target_mass = float(
            np.sum(a5[:, part_index] * target[:, part_index], dtype=np.float64)
        )
        a7_target_mass = float(
            np.sum(calibrated[:, part_index] * target[:, part_index], dtype=np.float64)
        )
        a5_outer_mass = float(
            np.sum(a5[:, part_index] * outer[:, part_index], dtype=np.float64)
        )
        a7_outer_mass = float(
            np.sum(calibrated[:, part_index] * outer[:, part_index], dtype=np.float64)
        )
        weight_sum = float(np.sum(calibrated[:, part_index], dtype=np.float64))
        per_part.append(
            {
                "part": part_name,
                "target_mass_ratio": _safe_ratio(a7_target_mass, a5_target_mass),
                "outer_mass_ratio": _safe_ratio(a7_outer_mass, a5_outer_mass),
                "boundary_crossing_weighted_mass": float(
                    np.sum(calibrated[:, part_index] * boundary[:, part_index])
                ),
                "low_support_weight": float(
                    np.sum(calibrated[:, part_index] * ~support[:, part_index])
                ),
                "evidence_support_coverage": coverage,
                "remaining_deficit": float(calibration["remaining_deficit"]),
                "cap_saturated_count": int(calibration["cap_saturated_count"]),
                "weight_l1_from_a5": float(calibration["weight_l1_from_a5"]),
                "proxy_temporal_burden": _safe_ratio(
                    float(
                        np.sum(
                            calibrated[:, part_index]
                            * (outer_flicker[:, part_index] + boundary[:, part_index])
                        )
                    ),
                    weight_sum,
                ),
            }
        )

    total_weight = float(np.sum(calibrated, dtype=np.float64))
    a5_outer_total = float(np.sum(a5 * outer, dtype=np.float64))
    a7_outer_total = float(np.sum(calibrated * outer, dtype=np.float64))
    proxy = {
        "proxy_temporal_burden": _safe_ratio(
            float(np.sum(calibrated * (outer_flicker + boundary), dtype=np.float64)),
            total_weight,
        ),
        "proxy_outer_mass_ratio": _safe_ratio(a7_outer_total, a5_outer_total),
        "weight_l1_from_a5": float(calibration_summary["weight_l1_from_a5"]),
        "remaining_deficit": float(
            sum(row["remaining_deficit"] for row in calibration_summary["per_part"])
        ),
        "cap_saturated_count": int(
            sum(row["cap_saturated_count"] for row in calibration_summary["per_part"])
        ),
        "low_support_weight": float(np.sum(calibrated * ~support, dtype=np.float64)),
    }
    return {
        "weights": weights,
        "temporal_reliability": reliability,
        "valid": not invalid_reasons,
        "invalid_reasons": sorted(set(invalid_reasons)),
        "proxy": proxy,
        "per_part": per_part,
        "reliability_summary": reliability_summary,
        "calibration_summary": calibration_summary,
        "evidence_mode": mode,
    }


def _load_evidence(path: Path, *, contract: dict, allow_canary: bool) -> dict:
    with np.load(path, allow_pickle=False) as data:
        evidence = {key: data[key] for key in data.files}
    stored_fingerprint = str(evidence.get("output_fingerprint", ""))
    if not stored_fingerprint or stored_fingerprint != _payload_fingerprint(evidence):
        raise ValueError("evidence output_fingerprint mismatch")
    formal = bool(int(np.asarray(evidence.get("formal_protocol", 0))))
    if not formal and not allow_canary:
        raise ValueError("non-formal evidence requires --allow-canary-evidence")
    evidence_contract = str(evidence.get("a7_contract_fingerprint", ""))
    if evidence_contract and evidence_contract != str(contract["_fingerprint"]):
        raise ValueError("evidence A7 contract fingerprint mismatch")
    required = (
        "temporal_consecutive_visible_count",
        "temporal_target_ratio_mean",
        "temporal_outer_ratio_mean",
        "temporal_outer_flicker",
        "temporal_boundary_crossing_rate",
        "temporal_target_flicker",
        "protocol_fingerprint",
    )
    missing = [key for key in required if key not in evidence]
    if missing:
        raise ValueError(f"missing temporal evidence fields: {missing}")
    return evidence


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _jsonable_result(result: dict) -> dict:
    return {
        key: value
        for key, value in result.items()
        if key not in ("weights", "temporal_reliability")
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.max_validation_candidates) < 0:
        raise ValueError("max-validation-candidates must be non-negative")
    contract = load_a7_temporal_contract(args.a7_contract, args.method_freeze)
    base_bank = load_part_label_bank(args.a5_bank)
    evidence = _load_evidence(
        args.evidence,
        contract=contract,
        allow_canary=bool(args.allow_canary_evidence),
    )
    a5_weights = np.asarray(base_bank["soft_edit_weights"], dtype=np.float32)
    semantic_probs = np.asarray(base_bank["semantic_probs"], dtype=np.float32)
    if a5_weights.shape != semantic_probs.shape:
        raise ValueError("A5 weights and semantic_probs shapes must match")
    if int(np.asarray(evidence["point_count"])) != a5_weights.shape[0]:
        raise ValueError("evidence point_count does not match A5 bank")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for grid_parameters in candidate_parameter_grid():
        parameters = {
            **grid_parameters,
            "min_pair_support": int(contract["min_pair_support"]),
        }
        fingerprint = candidate_config_fingerprint(parameters)
        identifier = fingerprint[:12]
        result = evaluate_candidate(
            a5_weights=a5_weights,
            semantic_probs=semantic_probs,
            evidence=evidence,
            parameters=parameters,
            max_weight_scale_from_posterior=float(
                contract["max_weight_scale_from_posterior"]
            ),
            minimum_evidence_support_coverage=float(
                contract["minimum_evidence_support_coverage"]
            ),
            minimum_carrier_support_ratio=float(
                contract.get("minimum_carrier_support_ratio", 0.0)
            ),
            minimum_carrier_existing_weight=float(
                contract.get("minimum_carrier_existing_weight", 0.0)
            ),
            carrier_ranking=str(
                contract.get(
                    "carrier_ranking", "posterior_target_reliability_support"
                )
            ),
            evidence_mode=str(contract.get("evidence_mode", "footprint")),
            frozen_part_indices=tuple(
                PART_NAMES.index(part)
                for part in contract.get("frozen_parts", [])
            ),
            selection_threshold=float(contract.get("selection_threshold", 0.0)),
            preserve_selection_topology=bool(
                contract.get("preserve_a5_selection_topology", False)
            ),
        )
        candidate_dir = output_dir / identifier
        bank_path = candidate_dir / "part_label_bank.npz"
        bank_fingerprint = save_a7_part_label_bank(
            bank_path,
            base_bank_path=args.a5_bank,
            temporal_evidence=evidence,
            temporal_reliability=result["temporal_reliability"],
            soft_edit_weights=result["weights"],
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
            "parameters": parameters,
            "bank": str(Path(identifier) / "part_label_bank.npz"),
            "output_bank_fingerprint": bank_fingerprint,
            **_jsonable_result(result),
        }
        _write_json(candidate_dir / "candidate_summary.json", summary)
        rows.append(summary)

    valid_rows = [row for row in rows if row["valid"]]
    valid_rows.sort(
        key=lambda row: (
            row["proxy"]["proxy_temporal_burden"],
            row["proxy"]["proxy_outer_mass_ratio"],
            row["proxy"]["weight_l1_from_a5"],
            row["candidate_id"],
        )
    )
    shortlist = [
        row["candidate_id"]
        for row in valid_rows[: int(args.max_validation_candidates)]
    ]
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
        "valid_candidate_count": len(valid_rows),
        "validation_shortlist": shortlist,
        "candidates": rows,
    }
    _write_json(output_dir / "candidate_index.json", index)
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
