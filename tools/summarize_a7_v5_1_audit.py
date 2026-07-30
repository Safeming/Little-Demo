#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.frozen_semantic_method import load_a7_temporal_contract


ACTIVE_PARTS = ("hair", "lower")
ALL_PARTS = ("face", "hair", "upper", "lower", "shoes", "skin")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarize the frozen A7 v5.1 audit.")
    parser.add_argument("--candidate-index", required=True, type=Path)
    parser.add_argument("--a7-contract", required=True, type=Path)
    parser.add_argument("--method-freeze", required=True, type=Path)
    parser.add_argument("--temporal-root", required=True, type=Path)
    parser.add_argument("--spatial-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) > 1.0e-12:
        return float(numerator / denominator)
    return 1.0 if abs(numerator) <= 1.0e-12 else float("inf")


def load_validated_candidate(index_path: Path, contract: dict) -> tuple[dict, str]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("a7_contract_fingerprint") != contract["_fingerprint"]:
        raise ValueError("candidate index contract fingerprint mismatch")
    shortlist = index.get("validation_shortlist", [])
    if shortlist != ["dual_evidence_constrained_v5_1"]:
        raise ValueError("candidate index must contain the frozen v5.1 candidate")
    candidates = [row for row in index.get("candidates", []) if row.get("candidate_id") == shortlist[0]]
    if len(candidates) != 1 or not candidates[0].get("valid"):
        raise ValueError("frozen v5.1 candidate is not valid")
    candidate = candidates[0]
    capacity = candidate.get("capacity_summary", {})
    expected_cameras = list(range(len(contract.get("evidence_cameras", []))))
    folds = capacity.get("folds", [])
    fold_cameras = sorted(int(fold.get("held_out_camera", -1)) for fold in folds)
    if capacity.get("camera_ids") != expected_cameras or fold_cameras != expected_cameras:
        raise ValueError("candidate capacity does not contain the frozen LOCO folds")
    if not capacity.get("all_folds_passed"):
        raise ValueError("candidate capacity LOCO did not pass")
    if any(
        not fold.get("passed")
        or not fold.get("construction", {}).get("passed")
        or not fold.get("held_out", {}).get("passed")
        for fold in folds
    ):
        raise ValueError("candidate construction or held-out audit did not pass")
    final = capacity.get("final", {})
    if not final.get("construction_evaluation", {}).get("passed"):
        raise ValueError("candidate final construction gate did not pass")
    if not final.get("evaluation", {}).get("passed"):
        raise ValueError("candidate final audit gate did not pass")
    return candidate, str(candidate["output_bank_fingerprint"])


def _summarize_temporal(root: Path, contract: dict, bank_fingerprint: str) -> tuple[dict, list[str]]:
    reasons = []
    camera_rows = []
    total = {method: {metric: 0.0 for metric in ("outer", "boundary")} for method in ("a5", "a7")}
    visibility_ratios = []
    for camera_name in contract["audit_cameras"]:
        camera = int(str(camera_name).removeprefix("c"))
        summary = json.loads((root / f"c{camera}" / "summary.json").read_text(encoding="utf-8"))
        frame_start = int(contract["validation_frame_start"])
        frame_end = int(contract["validation_frame_end"])
        frame_step = int(contract["validation_frame_stride"])
        frame_count = len(range(frame_start, frame_end, frame_step))
        expected_rows = frame_count * len(contract["parts"]) * 2
        retrospective = {
            int(str(value).removeprefix("c"))
            for value in contract["retrospective_test_cameras"]
        }
        protocol_matches = (
            str(summary.get("subject")) == str(contract["subject"])
            and int(summary.get("camera", -1)) == camera
            and int(summary.get("frame_start", -1)) == frame_start
            and int(summary.get("frame_end", -1)) == frame_end
            and int(summary.get("frame_step", -1)) == frame_step
            and int(summary.get("frame_count", -1)) == frame_count
            and list(summary.get("parts", [])) == list(contract["parts"])
            and list(summary.get("methods", [])) == ["a5", "a7"]
            and int(summary.get("metric_row_count", -1)) == expected_rows
            and bool(summary.get("held_out_camera")) == (camera in retrospective)
        )
        if not protocol_matches:
            reasons.append(f"temporal_protocol:c{camera}")
        if summary.get("a7_contract_fingerprint") != contract["_fingerprint"]:
            reasons.append(f"temporal_contract:c{camera}")
        if summary.get("a7_bank_fingerprint") != bank_fingerprint:
            reasons.append(f"temporal_bank:c{camera}")
        if not summary.get("canonical_selection_fixed_across_frames") or not summary.get(
            "common_support_across_methods"
        ):
            reasons.append(f"temporal_topology:c{camera}")
        metrics = summary["temporal_metrics"]
        camera_values = {method: {"outer": 0.0, "boundary": 0.0} for method in ("a5", "a7")}
        for part in ACTIVE_PARTS:
            for method in ("a5", "a7"):
                item = metrics[method][part]
                camera_values[method]["outer"] += float(item["fixed_strength_outer_flicker"])
                camera_values[method]["boundary"] += float(item["fixed_strength_boundary_flicker"])
            visibility_ratios.append(
                {
                    "camera": camera,
                    "part": part,
                    "ratio": _ratio(
                        float(metrics["a7"][part]["visibility_aware_response_flicker"]),
                        float(metrics["a5"][part]["visibility_aware_response_flicker"]),
                    ),
                }
            )
        for method in ("a5", "a7"):
            for metric in ("outer", "boundary"):
                total[method][metric] += camera_values[method][metric]
        camera_outer = _ratio(camera_values["a7"]["outer"], camera_values["a5"]["outer"])
        camera_boundary = _ratio(
            camera_values["a7"]["boundary"], camera_values["a5"]["boundary"]
        )
        if camera_outer >= 1.0 or camera_boundary >= 1.0:
            reasons.append(f"temporal_direction:c{camera}")
        camera_rows.append(
            {"camera": camera, "outer_ratio": camera_outer, "boundary_ratio": camera_boundary}
        )
    outer_ratio = _ratio(total["a7"]["outer"], total["a5"]["outer"])
    boundary_ratio = _ratio(total["a7"]["boundary"], total["a5"]["boundary"])
    if outer_ratio > 1.0 - float(contract["minimum_active_temporal_gain"]) + 1.0e-7:
        reasons.append("fixed_outer_gain")
    if boundary_ratio > 1.0 - float(contract["minimum_active_temporal_gain"]) + 1.0e-7:
        reasons.append("fixed_boundary_gain")
    if any(
        row["ratio"] > float(contract["maximum_audit_visibility_response_ratio"]) + 1.0e-7
        for row in visibility_ratios
    ):
        reasons.append("visibility_response")
    return {
        "outer_ratio": outer_ratio,
        "boundary_ratio": boundary_ratio,
        "per_camera": camera_rows,
        "visibility_ratios": visibility_ratios,
    }, reasons


def _read_per_part(path: Path) -> dict[tuple[str, str], dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["baseline"], row["part"]): row for row in rows}


def _summarize_spatial(root: Path, contract: dict, bank_fingerprint: str) -> tuple[dict, list[str]]:
    reasons = []
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("a7_contract_fingerprint") != contract["_fingerprint"]:
        reasons.append("spatial_contract")
    if summary.get("a7_bank_fingerprint") != bank_fingerprint:
        reasons.append("spatial_bank")
    if summary.get("protocol_split") != "test":
        reasons.append("spatial_split")
    if not summary.get("canonical_selection_fixed_across_frames") or not summary.get(
        "common_support_across_methods"
    ):
        reasons.append("spatial_topology")

    guards = {
        (row["baseline"], row["part"]): row for row in summary["spatial_guard_metrics"]
    }
    burden_ratios = {}
    for metric in ("pooled_outer_burden", "pooled_boundary_burden"):
        a5_mean = float(np.mean([float(guards[("A5", part)][metric]) for part in ALL_PARTS]))
        a7_mean = float(np.mean([float(guards[("A7", part)][metric]) for part in ALL_PARTS]))
        burden_ratios[metric] = _ratio(a7_mean, a5_mean)
        if burden_ratios[metric] > 1.0 + float(contract["maximum_spatial_burden_worsening"]):
            reasons.append(metric)
    if any(
        abs(float(guards[("A7", part)]["coverage_rate"]) - float(guards[("A5", part)]["coverage_rate"]))
        > 1.0e-12
        for part in ALL_PARTS
    ):
        reasons.append("coverage")

    per_part = _read_per_part(root / "per_part_metrics.csv")
    soft_drops = {
        part: float(per_part[("A5", part)]["soft_iou"])
        - float(per_part[("A7", part)]["soft_iou"])
        for part in ACTIVE_PARTS
    }
    if any(value > float(contract["maximum_part_soft_iou_drop"]) + 1.0e-7 for value in soft_drops.values()):
        reasons.append("part_soft_iou")
    a5_macro = float(np.mean([float(per_part[("A5", part)]["iou"]) for part in ALL_PARTS]))
    a7_macro = float(np.mean([float(per_part[("A7", part)]["iou"]) for part in ALL_PARTS]))
    macro_drop = a5_macro - a7_macro
    if macro_drop > float(contract["maximum_macro_miou_drop"]) + 1.0e-7:
        reasons.append("macro_miou")
    a5_micro = _ratio(
        sum(float(per_part[("A5", part)]["intersection"]) for part in ALL_PARTS),
        sum(float(per_part[("A5", part)]["union"]) for part in ALL_PARTS),
    )
    a7_micro = _ratio(
        sum(float(per_part[("A7", part)]["intersection"]) for part in ALL_PARTS),
        sum(float(per_part[("A7", part)]["union"]) for part in ALL_PARTS),
    )
    micro_drop = a5_micro - a7_micro
    if micro_drop > float(contract["maximum_micro_iou_drop"]) + 1.0e-7:
        reasons.append("micro_iou")
    return {
        "burden_ratios": burden_ratios,
        "soft_iou_drops": soft_drops,
        "macro_miou_drop": macro_drop,
        "micro_iou_drop": micro_drop,
    }, reasons


def main(argv=None) -> int:
    args = parse_args(argv)
    contract = load_a7_temporal_contract(args.a7_contract, args.method_freeze)
    candidate, bank_fingerprint = load_validated_candidate(
        args.candidate_index, contract
    )
    temporal, temporal_reasons = _summarize_temporal(
        args.temporal_root, contract, bank_fingerprint
    )
    spatial, spatial_reasons = _summarize_spatial(
        args.spatial_root, contract, bank_fingerprint
    )
    reasons = sorted(set(temporal_reasons + spatial_reasons))
    payload = {
        "schema_version": 1,
        "candidate_id": candidate["candidate_id"],
        "a7_contract_fingerprint": contract["_fingerprint"],
        "a7_bank_fingerprint": bank_fingerprint,
        "passed": not reasons,
        "invalid_reasons": reasons,
        "temporal": temporal,
        "spatial": spatial,
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not reasons else 2


if __name__ == "__main__":
    raise SystemExit(main())
