#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.a7c_oracle_capacity import load_teacher_artifact
from utils.a7c_renderer_compositor import (
    build_canary_splits,
    evaluate_contribution_predictions,
)
from utils.part_label_bank import PART_NAMES, load_part_label_bank


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Audit the A7c carrier compositor canary.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--a5-bank", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _summary(records, contract):
    output = {
        "record_count": len(records),
        "outer_gain": float(np.mean([row["outer_gain"] for row in records])),
        "boundary_gain": float(np.mean([row["boundary_gain"] for row in records])),
        "minimum_target_response": float(min(row["minimum_target_response"] for row in records)),
        "maximum_soft_iou_drop": float(max(row["maximum_soft_iou_drop"] for row in records)),
        "maximum_adjacent_gate_change": float(max(row["maximum_adjacent_gate_change"] for row in records)),
        "paper_test_eligible": False,
    }
    for signal in ("outer", "boundary"):
        gains = np.asarray([row[f"{signal}_gain"] for row in records])
        output[f"{signal}_positive_block_fraction"] = float(np.mean(gains > 0.0))
        output[f"{signal}_block_gain_quantile"] = float(np.quantile(gains, contract["block_gain_quantile"]))
        output[f"{signal}_worst_block_gain"] = float(np.min(gains))
    output["passed"] = bool(
        output["outer_gain"] >= contract["minimum_outer_gain"]
        and output["boundary_gain"] >= contract["minimum_boundary_gain"]
        and output["minimum_target_response"] >= contract["minimum_target_response"] - 1e-7
        and output["maximum_soft_iou_drop"] <= contract["maximum_selection_soft_iou_drop"] + 1e-7
        and output["maximum_adjacent_gate_change"] <= contract["maximum_adjacent_gate_change"] + 1e-7
        and all(
            output[f"{signal}_positive_block_fraction"] >= contract["minimum_positive_block_fraction"]
            and output[f"{signal}_block_gain_quantile"] >= contract["minimum_block_gain_quantile"] - 1e-9
            and output[f"{signal}_worst_block_gain"] >= -contract["maximum_worst_block_regression"] - 1e-9
            for signal in ("outer", "boundary")
        )
    )
    return output


def main(argv=None):
    args = parse_args(argv)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    teacher = load_teacher_artifact(args.teacher)
    with np.load(args.evidence, allow_pickle=False) as source:
        evidence = {key: source[key] for key in source.files}
    bank = load_part_label_bank(args.a5_bank)
    weights = np.asarray(bank["soft_edit_weights"], dtype=np.float64)
    part = PART_NAMES.index(contract["part"])
    part_weights = weights[:, part]
    carrier_ids = np.asarray(teacher["carrier_ids"], dtype=np.int64)
    camera_index = np.asarray(teacher["camera_index"])
    frame_index = np.asarray(teacher["frame_index"])
    split = build_canary_splits(
        camera_index=camera_index, frame_index=frame_index,
        fit_camera_indices=(0, 1, 2, 3), audit_camera_indices=(4, 5, 6, 7),
        block_count=contract["temporal_block_count"],
    )

    streams = {}
    for prefix, source_prefix in (("objective", "renderer"), ("guard", "renderer_selection")):
        streams[prefix] = {}
        for signal in ("target", "outer", "boundary"):
            values = np.asarray(evidence[f"{source_prefix}_{signal}_contribution_sequence"], dtype=np.float64)[:, :, part]
            streams[prefix][signal] = {
                "total": values @ part_weights,
                "point": values[:, carrier_ids] * part_weights[carrier_ids][None, :],
            }

    records = []
    for fold, held in enumerate(split["held_block_masks"]):
        with np.load(args.training_dir / f"fold_{fold}" / "predictions.npz") as source:
            predictions = np.asarray(source["gates"], dtype=np.float64)
        for camera in range(4):
            selected = held & (camera_index == camera)
            objective = evaluate_contribution_predictions(
                **{signal: streams["objective"][signal]["total"][selected] for signal in ("target", "outer", "boundary")},
                **{f"point_{signal}": streams["objective"][signal]["point"][selected] for signal in ("target", "outer", "boundary")},
                gates=predictions[selected],
            )
            guard = evaluate_contribution_predictions(
                **{signal: streams["guard"][signal]["total"][selected] for signal in ("target", "outer", "boundary")},
                **{f"point_{signal}": streams["guard"][signal]["point"][selected] for signal in ("target", "outer", "boundary")},
                gates=predictions[selected],
            )
            records.append(
                {
                    "fold": fold,
                    "camera_index": camera,
                    "outer_gain": objective["outer_gain"],
                    "boundary_gain": objective["boundary_gain"],
                    "minimum_target_response": guard["minimum_target_response"],
                    "maximum_soft_iou_drop": guard["maximum_soft_iou_drop"],
                    "maximum_adjacent_gate_change": float(np.max(np.abs(np.diff(predictions[selected], axis=0)))) if np.count_nonzero(selected) > 1 else 0.0,
                }
            )
    summary = _summary(records, contract)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"stage": "held_block", "summary": summary, "records": records}
    (args.output_dir / "held_block_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    marker = args.output_dir / (".held_block_passed" if summary["passed"] else ".rejected")
    marker.touch()
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
