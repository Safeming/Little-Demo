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

from tools.audit_a7c_r1_2a_quotient_compositor import summarize_records
from tools.evaluate_a7c_r1_3g_exact_aggregate_oracle import (
    _array_fingerprint,
    _write_json,
)
from tools.train_a7c_r1_2a_quotient_compositor import (
    _build_streams,
    _load_teacher_manifest,
    sample_block_ids,
    verify_source_file,
)
from utils.a7c_exact_aggregate_oracle import validate_saved_manifest
from utils.a7c_renderer_compositor import (
    build_canary_splits,
    evaluate_contribution_predictions,
)
from utils.part_label_bank import PART_NAMES, load_part_label_bank


REQUEST_ARRAY_NAMES = (
    "requested_outer_gain",
    "requested_boundary_gain",
    "source_feasible_lower",
    "source_infeasible_upper",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fold_witness(
    path: Path,
    *,
    expected_manifest: dict,
    expected_mask,
    expected_sample_fingerprint: str,
    expected_carrier_fingerprint: str,
) -> dict:
    with np.load(path, allow_pickle=False) as source:
        saved = {key: source[key] for key in source.files}
    validate_saved_manifest(saved, expected_manifest)
    mask = np.asarray(saved["replay_mask"], dtype=bool).reshape(-1)
    expected = np.asarray(expected_mask, dtype=bool).reshape(-1)
    if not np.array_equal(mask, expected):
        raise ValueError("replay mask differs from frozen held split")
    gates = np.asarray(saved["replay_gates"], dtype=np.float64)
    carrier_count = np.asarray(expected_manifest["carrier_ids"]).size
    if gates.shape != (mask.size, carrier_count):
        raise ValueError("replay gate shape differs from frozen manifest")
    if np.any(~np.isfinite(gates[mask])) or np.any(np.isfinite(gates[~mask])):
        raise ValueError("replay gates cross replay mask")
    for name in REQUEST_ARRAY_NAMES:
        values = np.asarray(saved[name], dtype=np.float64).reshape(-1)
        if values.shape != mask.shape:
            raise ValueError(f"{name} shape differs from replay mask")
        if np.any(~np.isfinite(values[mask])) or np.any(
            np.isfinite(values[~mask])
        ):
            raise ValueError(f"{name} values cross replay mask")
    if str(saved["sample_order_fingerprint"].item()) != str(
        expected_sample_fingerprint
    ):
        raise ValueError("saved sample order fingerprint differs")
    if str(saved["carrier_order_fingerprint"].item()) != str(
        expected_carrier_fingerprint
    ):
        raise ValueError("saved carrier order fingerprint differs")
    json.loads(str(saved["source_fingerprints_json"].item()))
    return saved


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit saved R1.3-G gate witnesses with frozen renderer streams."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--a5-bank", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--witness-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _evaluate_record(streams, gates, selected, fold: int, camera: int) -> dict:
    outputs = {}
    for role in ("objective", "guard"):
        kwargs = {}
        for signal in ("target", "outer", "boundary"):
            kwargs[signal] = streams[role][signal]["base"][selected]
            kwargs[f"point_{signal}"] = streams[role][signal]["point"][selected]
        outputs[role] = evaluate_contribution_predictions(
            **kwargs, gates=gates[selected]
        )
    return {
        "fold": int(fold),
        "camera_index": int(camera),
        "outer_gain": float(outputs["objective"]["outer_gain"]),
        "boundary_gain": float(outputs["objective"]["boundary_gain"]),
        "minimum_target_response": float(
            outputs["guard"]["minimum_target_response"]
        ),
        "maximum_soft_iou_drop": float(
            outputs["guard"]["maximum_soft_iou_drop"]
        ),
        "maximum_adjacent_gate_change": float(
            np.max(np.abs(np.diff(gates[selected], axis=0)))
        ),
    }


def _run(args) -> tuple[dict, int]:
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("preserve_a5_selection_topology") is not True:
        raise ValueError("A5 selection topology guard is not frozen")
    source_specs = (
        (
            REPO_ROOT / contract["source_r1_3p_contract"],
            contract["source_r1_3p_contract_sha256"],
            "R1.3-P contract",
        ),
        (
            REPO_ROOT / contract["source_r1_3p_records"],
            contract["source_r1_3p_records_sha256"],
            "R1.3-P records",
        ),
        (
            REPO_ROOT / contract["source_r1_3p_summary"],
            contract["source_r1_3p_summary_sha256"],
            "R1.3-P summary",
        ),
        (
            REPO_ROOT / contract["source_r1_3g_design"],
            contract["source_r1_3g_design_sha256"],
            "R1.3-G design",
        ),
        (args.evidence, contract["source_evidence_sha256"], "evidence"),
        (args.a5_bank, contract["source_a5_bank_sha256"], "A5 bank"),
        (args.teacher, contract["source_teacher_sha256"], "teacher"),
    )
    source_fingerprints = {
        name: verify_source_file(path, expected, name)
        for path, expected, name in source_specs
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
    all_weights = np.asarray(bank["soft_edit_weights"], dtype=np.float64)[
        :, part_index
    ]
    carrier_ids = np.asarray(teacher["carrier_ids"], dtype=np.int64)
    streams = _build_streams(evidence, all_weights, carrier_ids, part_index)
    block_ids = sample_block_ids(
        camera_index, frame_index, int(contract["temporal_block_count"])
    )
    split = build_canary_splits(
        camera_index=camera_index,
        frame_index=frame_index,
        fit_camera_indices=(0, 1, 2, 3),
        audit_camera_indices=(4, 5, 6, 7),
        block_count=int(contract["temporal_block_count"]),
    )
    expected_manifest = {
        "camera_index": camera_index,
        "frame_index": frame_index,
        "block_ids": block_ids,
        "carrier_ids": carrier_ids,
    }
    sample_fingerprint = _array_fingerprint(
        camera_index, frame_index, block_ids
    )
    carrier_fingerprint = _array_fingerprint(carrier_ids)
    topology_floor = np.maximum(
        float(contract["minimum_gate"]),
        float(contract["selection_threshold"])
        / np.maximum(all_weights[carrier_ids], 1.0e-8),
    )

    records = []
    witness_fingerprints = {}
    certificate_maximum = 0.0
    for fold, held in enumerate(split["held_block_masks"]):
        expected_mask = np.asarray(held & split["fit_mask"], dtype=bool)
        prediction_path = args.witness_dir / f"fold_{fold}/predictions.npz"
        witness_fingerprints[f"fold_{fold}_prediction"] = _sha256(
            prediction_path
        )
        saved = load_fold_witness(
            prediction_path,
            expected_manifest=expected_manifest,
            expected_mask=expected_mask,
            expected_sample_fingerprint=sample_fingerprint,
            expected_carrier_fingerprint=carrier_fingerprint,
        )
        gates = np.asarray(saved["replay_gates"], dtype=np.float64)
        if np.min(gates[expected_mask] - topology_floor[None, :]) < -1.0e-7:
            raise ValueError("saved gates violate frozen A5 topology floor")
        if np.max(gates[expected_mask]) > float(contract["maximum_gate"]) + 1e-7:
            raise ValueError("saved gates exceed frozen upper bound")
        certificates = json.loads(
            (args.witness_dir / f"fold_{fold}/certificates.json").read_text(
                encoding="utf-8"
            )
        )
        if len(certificates) != 4:
            raise ValueError("every fold requires four replay certificates")
        certificate_by_camera = {
            int(row["camera_index"]): row for row in certificates
        }
        if set(certificate_by_camera) != {0, 1, 2, 3}:
            raise ValueError("replay certificate cameras differ")
        for camera in range(4):
            selected = expected_mask & (camera_index == camera)
            certificate = certificate_by_camera[camera]
            if int(certificate["fold"]) != fold:
                raise ValueError("replay certificate fold differs")
            if (
                float(certificate["maximum_primal_violation"])
                > float(contract["solver_residual_tolerance"])
            ):
                raise ValueError("replay certificate residual exceeds tolerance")
            for key, array_name in (
                ("minimum_outer_gain", "requested_outer_gain"),
                ("minimum_boundary_gain", "requested_boundary_gain"),
                ("source_feasible_lower", "source_feasible_lower"),
                ("source_infeasible_upper", "source_infeasible_upper"),
            ):
                values = np.asarray(saved[array_name], dtype=np.float64)[selected]
                if not np.all(values == float(certificate[key])):
                    raise ValueError(f"saved {array_name} differs from certificate")
            segment_fingerprint = _array_fingerprint(
                camera_index[selected], frame_index[selected], block_ids[selected]
            )
            if certificate["sample_order_fingerprint"] != segment_fingerprint:
                raise ValueError("certificate sample order fingerprint differs")
            if certificate["carrier_order_fingerprint"] != carrier_fingerprint:
                raise ValueError("certificate carrier order fingerprint differs")
            if certificate["source_fingerprints"] != source_fingerprints:
                raise ValueError("certificate source fingerprints differ")
            certificate_maximum = max(
                certificate_maximum,
                float(certificate["maximum_primal_violation"]),
            )
            records.append(
                _evaluate_record(streams, gates, selected, fold, camera)
            )
    if len(records) != 24 or len(
        {(row["fold"], row["camera_index"]) for row in records}
    ) != 24:
        raise ValueError("exact aggregate audit requires 24 unique records")
    summary = summarize_records(records, contract)
    payload = {
        "stage": "r1_3g_exact_aggregate",
        "summary": summary,
        "records": records,
        "certificate_maximum_primal_violation": certificate_maximum,
        "source_fingerprints": source_fingerprints,
        "witness_fingerprints": witness_fingerprints,
        "guards": {
            "a5_bank_frozen": True,
            "selection_topology_preserved": True,
            "coverage_preserved": True,
            "frozen_parts_preserved": True,
            "weight_upper_bounds_preserved": True,
        },
        "audit_camera_metrics_opened": False,
        "deployment_eligible": False,
        "teacher_eligible": False,
        "paper_test_eligible": False,
    }
    return payload, 0 if summary["passed"] else 2


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload, status = _run(args)
    except Exception as error:
        payload = {
            "stage": "r1_3g_exact_aggregate",
            "execution_status": "ORACLE_ERROR",
            "error_type": type(error).__name__,
            "error": str(error),
            "deployment_eligible": False,
            "teacher_eligible": False,
            "paper_test_eligible": False,
        }
        _write_json(args.output_dir / "held_block_summary.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    _write_json(args.output_dir / "held_block_summary.json", payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
