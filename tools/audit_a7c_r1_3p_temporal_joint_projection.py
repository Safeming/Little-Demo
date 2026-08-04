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

from tools.audit_a7c_r1_2a_quotient_compositor import (
    summarize_records,
    write_audit,
)
from tools.train_a7c_r1_2a_quotient_compositor import (
    _build_streams,
    _load_teacher_manifest,
    verify_source_file,
)
from utils.a7c_renderer_compositor import (
    build_canary_splits,
    evaluate_contribution_predictions,
)
from utils.part_label_bank import PART_NAMES, load_part_label_bank


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit R1.3-P held-block temporal projections."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--a5-bank", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--projection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    source_fingerprints = {
        "evidence": verify_source_file(
            args.evidence, contract["source_evidence_sha256"], "evidence"
        ),
        "A5 bank": verify_source_file(
            args.a5_bank, contract["source_a5_bank_sha256"], "A5 bank"
        ),
        "teacher": verify_source_file(
            args.teacher, contract["source_teacher_sha256"], "teacher"
        ),
    }
    teacher = _load_teacher_manifest(args.teacher)
    with np.load(args.evidence, allow_pickle=False) as source:
        evidence = {key: source[key] for key in source.files}
    camera_index = np.asarray(teacher["camera_index"])
    frame_index = np.asarray(teacher["frame_index"])
    if not np.array_equal(
        evidence["renderer_sequence_camera_index"], camera_index
    ) or not np.array_equal(
        evidence["renderer_sequence_frame_index"], frame_index
    ):
        raise ValueError("evidence and teacher sample manifest differ")
    bank = load_part_label_bank(args.a5_bank)
    part_index = PART_NAMES.index(str(contract["part"]))
    weights = np.asarray(bank["soft_edit_weights"], dtype=np.float64)[
        :, part_index
    ]
    carrier_ids = np.asarray(teacher["carrier_ids"], dtype=np.int64)
    streams = _build_streams(evidence, weights, carrier_ids, part_index)
    split = build_canary_splits(
        camera_index=camera_index,
        frame_index=frame_index,
        fit_camera_indices=(0, 1, 2, 3),
        audit_camera_indices=(4, 5, 6, 7),
        block_count=int(contract["temporal_block_count"]),
    )

    records = []
    certificate_maxima = {
        "maximum_displacement": 0.0,
        "maximum_primal_violation": 0.0,
    }
    for fold, held in enumerate(split["held_block_masks"]):
        fold_root = args.projection_dir / f"fold_{fold}"
        with np.load(fold_root / "predictions.npz", allow_pickle=False) as source:
            gates = np.asarray(source["projected_gates"], dtype=np.float64)
            projection_mask = np.asarray(source["projection_mask"], dtype=bool)
            if not np.array_equal(source["camera_index"], camera_index):
                raise ValueError("projection camera manifest differs")
            if not np.array_equal(source["frame_index"], frame_index):
                raise ValueError("projection frame manifest differs")
            if not np.array_equal(source["carrier_ids"], carrier_ids):
                raise ValueError("projection carrier manifest differs")
        expected_mask = np.asarray(held & split["fit_mask"], dtype=bool)
        if not np.array_equal(projection_mask, expected_mask):
            raise ValueError("projection mask differs from frozen held split")
        if np.any(~np.isfinite(gates[expected_mask])) or np.any(
            np.isfinite(gates[~expected_mask])
        ):
            raise ValueError("projection finite values cross held split")
        certificates = json.loads(
            (fold_root / "segment_certificates.json").read_text(encoding="utf-8")
        )
        if len(certificates) != 4:
            raise ValueError("every fold needs four segment certificates")
        certificate_maxima["maximum_displacement"] = max(
            certificate_maxima["maximum_displacement"],
            max(float(row["maximum_displacement"]) for row in certificates),
        )
        certificate_maxima["maximum_primal_violation"] = max(
            certificate_maxima["maximum_primal_violation"],
            max(float(row["maximum_primal_violation"]) for row in certificates),
        )
        for camera in range(4):
            selected = expected_mask & (camera_index == camera)
            outputs = {}
            for role in ("objective", "guard"):
                kwargs = {}
                for signal in ("target", "outer", "boundary"):
                    kwargs[signal] = streams[role][signal]["base"][selected]
                    kwargs[f"point_{signal}"] = streams[role][signal]["point"][
                        selected
                    ]
                outputs[role] = evaluate_contribution_predictions(
                    **kwargs, gates=gates[selected]
                )
            records.append(
                {
                    "fold": fold,
                    "camera_index": camera,
                    "outer_gain": outputs["objective"]["outer_gain"],
                    "boundary_gain": outputs["objective"]["boundary_gain"],
                    "minimum_target_response": outputs["guard"][
                        "minimum_target_response"
                    ],
                    "maximum_soft_iou_drop": outputs["guard"][
                        "maximum_soft_iou_drop"
                    ],
                    "maximum_adjacent_gate_change": float(
                        np.max(np.abs(np.diff(gates[selected], axis=0)))
                    ),
                }
            )
    summary = summarize_records(records, contract)
    payload = {
        "stage": "r1_3p_held_block",
        "summary": summary,
        "records": records,
        "certificate_maxima": certificate_maxima,
        "source_fingerprints": source_fingerprints,
        "audit_camera_metrics_opened": False,
        "paper_test_eligible": False,
    }
    write_audit(args.output_dir, payload)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
