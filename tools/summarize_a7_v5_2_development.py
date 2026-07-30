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

from tools.summarize_a7_v5_1_audit import (
    ACTIVE_PARTS,
    _ratio,
    _summarize_spatial,
    _write_json,
    load_validated_candidate,
)
from utils.frozen_semantic_method import load_a7_temporal_contract


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarize the A7 v5.2 development run.")
    parser.add_argument("--candidate-index", required=True, type=Path)
    parser.add_argument("--a7-contract", required=True, type=Path)
    parser.add_argument("--method-freeze", required=True, type=Path)
    parser.add_argument("--temporal-root", required=True, type=Path)
    parser.add_argument("--spatial-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _target_response_ratios(metrics_path: Path, *, frame_count: int) -> list[dict]:
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    outputs = []
    for part in ACTIVE_PARTS:
        by_method = {}
        for method in ("a5", "a7"):
            values = [
                float(row["edit_target_delta_mean"])
                for row in rows
                if row["part"] == part and row["method"] == method
            ]
            if len(values) != frame_count:
                raise ValueError(
                    f"temporal target rows for {method}/{part} must equal frame_count"
                )
            by_method[method] = float(np.mean(values))
        outputs.append(
            {"part": part, "ratio": _ratio(by_method["a7"], by_method["a5"])}
        )
    return outputs


def _summarize_temporal_group(
    root: Path,
    contract: dict,
    bank_fingerprint: str,
    *,
    cameras,
    maximum_visibility_ratio: float,
    minimum_target_ratio: float,
    constrained_parts,
) -> tuple[dict, list[str]]:
    reasons = []
    camera_rows = []
    total = {method: {metric: 0.0 for metric in ("outer", "boundary")} for method in ("a5", "a7")}
    visibility_rows = []
    target_rows = []
    frame_start = int(contract["validation_frame_start"])
    frame_end = int(contract["validation_frame_end"])
    frame_step = int(contract["validation_frame_stride"])
    frame_count = len(range(frame_start, frame_end, frame_step))
    expected_rows = frame_count * len(contract["parts"]) * 2
    for camera_name in cameras:
        camera = int(str(camera_name).removeprefix("c"))
        camera_root = root / f"c{camera}"
        summary = json.loads((camera_root / "summary.json").read_text(encoding="utf-8"))
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
        )
        if not protocol_matches:
            reasons.append(f"protocol:c{camera}")
        if summary.get("a7_contract_fingerprint") != contract["_fingerprint"]:
            reasons.append(f"contract:c{camera}")
        if summary.get("a7_bank_fingerprint") != bank_fingerprint:
            reasons.append(f"bank:c{camera}")
        if not summary.get("canonical_selection_fixed_across_frames") or not summary.get(
            "common_support_across_methods"
        ):
            reasons.append(f"topology:c{camera}")
        metrics = summary["temporal_metrics"]
        camera_values = {method: {"outer": 0.0, "boundary": 0.0} for method in ("a5", "a7")}
        for part in ACTIVE_PARTS:
            for method in ("a5", "a7"):
                item = metrics[method][part]
                camera_values[method]["outer"] += float(item["fixed_strength_outer_flicker"])
                camera_values[method]["boundary"] += float(item["fixed_strength_boundary_flicker"])
            visibility_rows.append(
                {
                    "camera": camera,
                    "part": part,
                    "ratio": _ratio(
                        float(metrics["a7"][part]["visibility_aware_response_flicker"]),
                        float(metrics["a5"][part]["visibility_aware_response_flicker"]),
                    ),
                }
            )
        for row in _target_response_ratios(camera_root / "metrics.csv", frame_count=frame_count):
            target_rows.append({"camera": camera, **row})
        for method in ("a5", "a7"):
            for metric in ("outer", "boundary"):
                total[method][metric] += camera_values[method][metric]
        outer_ratio = _ratio(camera_values["a7"]["outer"], camera_values["a5"]["outer"])
        boundary_ratio = _ratio(
            camera_values["a7"]["boundary"], camera_values["a5"]["boundary"]
        )
        if outer_ratio >= 1.0 or boundary_ratio >= 1.0:
            reasons.append(f"direction:c{camera}")
        camera_rows.append(
            {"camera": camera, "outer_ratio": outer_ratio, "boundary_ratio": boundary_ratio}
        )
    outer_ratio = _ratio(total["a7"]["outer"], total["a5"]["outer"])
    boundary_ratio = _ratio(total["a7"]["boundary"], total["a5"]["boundary"])
    if outer_ratio > 1.0 - float(contract["minimum_active_temporal_gain"]) + 1.0e-7:
        reasons.append("fixed_outer_gain")
    if boundary_ratio > 1.0 - float(contract["minimum_active_temporal_gain"]) + 1.0e-7:
        reasons.append("fixed_boundary_gain")
    constrained = set(constrained_parts)
    if any(
        row["part"] in constrained
        and row["ratio"] > float(maximum_visibility_ratio) + 1.0e-7
        for row in visibility_rows
    ):
        reasons.append("visibility_response")
    if any(
        row["part"] in constrained
        and row["ratio"] < float(minimum_target_ratio) - 1.0e-7
        for row in target_rows
    ):
        reasons.append("target_response")
    return {
        "outer_ratio": outer_ratio,
        "boundary_ratio": boundary_ratio,
        "per_camera": camera_rows,
        "visibility_ratios": visibility_rows,
        "target_response_ratios": target_rows,
        "minimum_target_response_ratio": min(
            row["ratio"] for row in target_rows if row["part"] in constrained
        ),
    }, sorted(set(reasons))


def main(argv=None) -> int:
    args = parse_args(argv)
    contract = load_a7_temporal_contract(args.a7_contract, args.method_freeze)
    candidate, bank_fingerprint = load_validated_candidate(
        args.candidate_index, contract
    )
    validation, validation_reasons = _summarize_temporal_group(
        args.temporal_root / "validation",
        contract,
        bank_fingerprint,
        cameras=contract["validation_cameras"],
        maximum_visibility_ratio=contract[
            "maximum_validation_visibility_response_ratio"
        ],
        minimum_target_ratio=contract["minimum_validation_target_response_ratio"],
        constrained_parts=("lower",),
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
        {f"validation_{reason}" for reason in validation_reasons}
        | {f"retrospective_{reason}" for reason in retrospective_reasons}
        | {f"spatial_{reason}" for reason in spatial_reasons}
    )
    payload = {
        "schema_version": 1,
        "candidate_id": candidate["candidate_id"],
        "a7_contract_fingerprint": contract["_fingerprint"],
        "a7_bank_fingerprint": bank_fingerprint,
        "paper_test_eligible": False,
        "development_passed": not reasons,
        "invalid_reasons": reasons,
        "validation": validation,
        "retrospective": retrospective,
        "spatial": spatial,
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not reasons else 2


if __name__ == "__main__":
    raise SystemExit(main())
