#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train_a7c_r1_2a_quotient_compositor import (
    _build_streams,
    _load_probe,
    _load_teacher_manifest,
    sample_block_ids,
    verify_source_file,
)
from utils.a7c_exact_aggregate_oracle import (
    extract_replay_requests,
    insert_replay_segment,
)
from utils.a7c_feasibility_oracle import solve_fixed_gain_oracle
from utils.a7c_quotient_compositor import runtime_target_mass
from utils.a7c_renderer_compositor import build_canary_splits
from utils.part_label_bank import PART_NAMES, load_part_label_bank


def _array_fingerprint(*arrays) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def write_fold_witness(
    *,
    output_dir: Path,
    replay_gates,
    replay_mask,
    request_arrays,
    camera_index,
    frame_index,
    block_ids,
    carrier_ids,
    certificates,
    source_fingerprints,
    sample_order_fingerprint: str,
    carrier_order_fingerprint: str,
) -> None:
    output = Path(output_dir)
    gates = np.asarray(replay_gates, dtype=np.float64)
    mask = np.asarray(replay_mask, dtype=bool).reshape(-1)
    if gates.ndim != 2 or mask.shape != (gates.shape[0],):
        raise ValueError("fold replay gates and mask differ")
    if np.any(~np.isfinite(gates[mask])) or np.any(np.isfinite(gates[~mask])):
        raise ValueError("fold replay gates cross replay mask")
    arrays = {}
    for name, values in request_arrays.items():
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.shape != mask.shape:
            raise ValueError(f"{name} shape differs from replay mask")
        if np.any(~np.isfinite(array[mask])) or np.any(np.isfinite(array[~mask])):
            raise ValueError(f"{name} values cross replay mask")
        arrays[name] = array
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "predictions.npz",
        replay_gates=gates,
        replay_mask=mask,
        camera_index=np.asarray(camera_index),
        frame_index=np.asarray(frame_index),
        block_ids=np.asarray(block_ids),
        carrier_ids=np.asarray(carrier_ids),
        **arrays,
        source_fingerprints_json=np.array(
            json.dumps(dict(source_fingerprints), sort_keys=True)
        ),
        sample_order_fingerprint=np.array(str(sample_order_fingerprint)),
        carrier_order_fingerprint=np.array(str(carrier_order_fingerprint)),
        deployment_eligible=np.array(0, dtype=np.uint8),
        teacher_eligible=np.array(0, dtype=np.uint8),
        paper_test_eligible=np.array(0, dtype=np.uint8),
    )
    _write_json(output / "certificates.json", list(certificates))


def _record_streams(streams, selected) -> dict:
    return {
        "objective": {
            signal: {
                key: np.asarray(value)[selected]
                for key, value in streams["objective"][signal].items()
            }
            for signal in ("outer", "boundary")
        },
        "guard": {
            signal: {
                key: np.asarray(value)[selected]
                for key, value in streams["guard"][signal].items()
            }
            for signal in ("target", "outer")
        },
    }


def _replay_diagnostics(
    *, gates, runtime_mass, a5_weight, frame_index, carrier_ids, contract
) -> dict:
    values = np.asarray(gates, dtype=np.float64)
    mass = np.asarray(runtime_mass, dtype=np.float64)
    weights = np.asarray(a5_weight, dtype=np.float64).reshape(-1)
    topology_floor = np.maximum(
        float(contract["minimum_gate"]),
        float(contract["selection_threshold"])
        / np.maximum(weights, 1.0e-8),
    )
    required_proxy = float(contract["proxy_target_response"]) * mass.sum(axis=1)
    achieved_proxy = np.sum(mass * values, axis=1)
    jumps = np.abs(np.diff(values, axis=0))
    jump_flat = int(np.argmax(jumps))
    jump_frame, jump_carrier = np.unravel_index(jump_flat, jumps.shape)
    frames = np.asarray(frame_index).reshape(-1)
    carriers = np.asarray(carrier_ids).reshape(-1)
    return {
        "slack": {
            "minimum_topology_slack": float(
                np.min(values - topology_floor[None, :])
            ),
            "minimum_bound_slack": float(
                min(
                    np.min(values - float(contract["minimum_gate"])),
                    np.min(float(contract["maximum_gate"]) - values),
                )
            ),
            "minimum_proxy_target_slack": float(
                np.min(achieved_proxy - required_proxy)
            ),
        },
        "locations": {
            "maximum_adjacent_gate_change_previous_frame": int(
                frames[jump_frame]
            ),
            "maximum_adjacent_gate_change_frame": int(frames[jump_frame + 1]),
            "maximum_adjacent_gate_change_carrier_id": int(
                carriers[jump_carrier]
            ),
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate R1.3-G exact aggregate oracle gate witnesses."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--a5-bank", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _run(args) -> dict:
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    expected_records = REPO_ROOT / contract["source_r1_3p_records"]
    if args.source_records.resolve() != expected_records.resolve():
        raise ValueError("source records path differs from frozen contract")
    source_specs = (
        (
            REPO_ROOT / contract["source_r1_3p_contract"],
            contract["source_r1_3p_contract_sha256"],
            "R1.3-P contract",
        ),
        (
            args.source_records,
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
        (args.probe, contract["source_probe_sha256"], "probe"),
        (args.evidence, contract["source_evidence_sha256"], "evidence"),
        (args.a5_bank, contract["source_a5_bank_sha256"], "A5 bank"),
        (args.teacher, contract["source_teacher_sha256"], "teacher"),
    )
    source_fingerprints = {
        name: verify_source_file(path, expected, name)
        for path, expected, name in source_specs
    }
    source_summary = json.loads(
        (REPO_ROOT / contract["source_r1_3p_summary"]).read_text(
            encoding="utf-8"
        )
    )
    if source_summary.get("execution_status") != "COMPLETED" or source_summary.get(
        "verdict"
    ) != "UNRESOLVED":
        raise ValueError("R1.3-P source summary is not the frozen unresolved result")
    source_payload = json.loads(args.source_records.read_text(encoding="utf-8"))
    requests = extract_replay_requests(
        source_payload["records"],
        replay_margin=float(contract["replay_margin"]),
        maximum_interval_width=float(contract["oracle_bisection_tolerance"]),
    )
    request_by_key = {
        (row["fold"], row["camera_index"]): row for row in requests
    }

    probe = _load_probe(args.probe)
    teacher = _load_teacher_manifest(args.teacher)
    for key in ("carrier_ids", "camera_index", "frame_index"):
        if not np.array_equal(probe[key], teacher[key]):
            raise ValueError(f"probe and teacher {key} differ")
    with np.load(args.evidence, allow_pickle=False) as source:
        evidence = {key: source[key] for key in source.files}
    camera_index = np.asarray(teacher["camera_index"])
    frame_index = np.asarray(teacher["frame_index"])
    if not np.array_equal(
        evidence["renderer_sequence_camera_index"], camera_index
    ) or not np.array_equal(
        evidence["renderer_sequence_frame_index"], frame_index
    ):
        raise ValueError("source sample manifests differ")

    bank = load_part_label_bank(args.a5_bank)
    part_index = PART_NAMES.index(str(contract["part"]))
    all_weights = np.asarray(bank["soft_edit_weights"], dtype=np.float64)[
        :, part_index
    ]
    carrier_ids = np.asarray(teacher["carrier_ids"], dtype=np.int64)
    a5_weight = all_weights[carrier_ids]
    streams = _build_streams(evidence, all_weights, carrier_ids, part_index)
    features = np.asarray(probe["features"], dtype=np.float32)
    feature_names = list(map(str, probe["feature_names"]))
    fields = {
        name: features[:, :, feature_names.index(name)]
        for name in (
            "alpha_transmittance_mass",
            "semantic_support_mean",
            "alpha_mean",
        )
    }
    mass = runtime_target_mass(
        alpha_transmittance_mass=torch.from_numpy(
            fields["alpha_transmittance_mass"]
        ),
        a5_weight=torch.from_numpy(a5_weight.astype(np.float32)),
        semantic_support_mean=torch.from_numpy(fields["semantic_support_mean"]),
        alpha_mean=torch.from_numpy(fields["alpha_mean"]),
    ).numpy()
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
    sample_order_fingerprint = _array_fingerprint(
        camera_index, frame_index, block_ids
    )
    carrier_order_fingerprint = _array_fingerprint(carrier_ids)

    witness_root = args.output_dir / "witness"
    records = []
    fold_summaries = []
    for fold, held in enumerate(split["held_block_masks"]):
        expected_mask = np.asarray(held & split["fit_mask"], dtype=bool)
        replay_gates = np.full(mass.shape, np.nan, dtype=np.float64)
        replay_mask = np.zeros(camera_index.size, dtype=bool)
        request_arrays = {
            name: np.full(camera_index.size, np.nan, dtype=np.float64)
            for name in (
                "requested_outer_gain",
                "requested_boundary_gain",
                "source_feasible_lower",
                "source_infeasible_upper",
            )
        }
        certificates = []
        for camera in range(4):
            selected = np.asarray(
                expected_mask & (camera_index == camera), dtype=bool
            )
            indices = np.flatnonzero(selected)
            selected_blocks = np.unique(block_ids[selected])
            if (
                indices.size < 2
                or selected_blocks.size != 1
                or not np.all(
                    np.diff(frame_index[indices]) == int(contract["frame_stride"])
                )
            ):
                raise ValueError("replay held segment is not contiguous")
            request = request_by_key[(fold, camera)]
            solved = solve_fixed_gain_oracle(
                runtime_mass=mass[selected],
                a5_weight=a5_weight,
                streams=_record_streams(streams, selected),
                minimum_gate=float(contract["minimum_gate"]),
                maximum_gate=float(contract["maximum_gate"]),
                selection_threshold=float(contract["selection_threshold"]),
                proxy_target_response=float(contract["proxy_target_response"]),
                maximum_gate_jump=float(
                    contract["maximum_projection_gate_jump"]
                ),
                minimum_target_response=float(contract["minimum_target_response"]),
                maximum_soft_iou_drop=float(
                    contract["maximum_selection_soft_iou_drop"]
                ),
                minimum_outer_gain=float(request["minimum_outer_gain"]),
                minimum_boundary_gain=float(request["minimum_boundary_gain"]),
                primal_tolerance=float(contract["solver_primal_tolerance"]),
                residual_tolerance=float(contract["solver_residual_tolerance"]),
            )
            solved.update(
                _replay_diagnostics(
                    gates=solved["gates"],
                    runtime_mass=mass[selected],
                    a5_weight=a5_weight,
                    frame_index=frame_index[selected],
                    carrier_ids=carrier_ids,
                    contract=contract,
                )
            )
            solved["source_fingerprints"] = source_fingerprints
            solved["sample_order_fingerprint"] = _array_fingerprint(
                camera_index[selected], frame_index[selected], block_ids[selected]
            )
            solved["carrier_order_fingerprint"] = carrier_order_fingerprint
            record = insert_replay_segment(
                replay_gates=replay_gates,
                replay_mask=replay_mask,
                selected=selected,
                solved=solved,
                request=request,
                frame_index=frame_index,
                block_ids=block_ids,
                carrier_ids=carrier_ids,
                residual_tolerance=float(contract["solver_residual_tolerance"]),
            )
            records.append(record)
            certificates.append(record)
            for name, value in (
                ("requested_outer_gain", request["minimum_outer_gain"]),
                ("requested_boundary_gain", request["minimum_boundary_gain"]),
                ("source_feasible_lower", request["source_feasible_lower"]),
                ("source_infeasible_upper", request["source_infeasible_upper"]),
            ):
                request_arrays[name][selected] = float(value)
            print(
                json.dumps(
                    {
                        "replay_record_complete": len(records),
                        "fold": fold,
                        "camera_index": camera,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if not np.array_equal(replay_mask, expected_mask):
            raise RuntimeError("fold replay mask differs from frozen held split")
        write_fold_witness(
            output_dir=witness_root / f"fold_{fold}",
            replay_gates=replay_gates,
            replay_mask=replay_mask,
            request_arrays=request_arrays,
            camera_index=camera_index,
            frame_index=frame_index,
            block_ids=block_ids,
            carrier_ids=carrier_ids,
            certificates=certificates,
            source_fingerprints=source_fingerprints,
            sample_order_fingerprint=sample_order_fingerprint,
            carrier_order_fingerprint=carrier_order_fingerprint,
        )
        fold_summaries.append(
            {
                "fold": fold,
                "record_count": len(certificates),
                "sample_count": int(np.count_nonzero(replay_mask)),
                "maximum_primal_violation": max(
                    row["maximum_primal_violation"] for row in certificates
                ),
                "paper_test_eligible": False,
            }
        )
    if len(records) != 24:
        raise RuntimeError("exact replay requires exactly 24 records")
    witness_summary = {
        "experiment_id": contract["experiment_id"],
        "folds": fold_summaries,
        "record_count": len(records),
        "source_fingerprints": source_fingerprints,
        "sample_order_fingerprint": sample_order_fingerprint,
        "carrier_order_fingerprint": carrier_order_fingerprint,
        "deployment_eligible": False,
        "teacher_eligible": False,
        "paper_test_eligible": False,
    }
    summary = {
        "experiment_id": contract["experiment_id"],
        "execution_status": "REPLAY_COMPLETED",
        "verdict": "UNRESOLVED",
        "record_count": len(records),
        "aggregate_audit_opened": False,
        "source_fingerprints": source_fingerprints,
        "deployment_eligible": False,
        "teacher_eligible": False,
        "paper_test_eligible": False,
    }
    _write_json(witness_root / "summary.json", witness_summary)
    _write_json(
        args.output_dir / "records.json",
        {
            "records": records,
            "deployment_eligible": False,
            "teacher_eligible": False,
            "paper_test_eligible": False,
        },
    )
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary = _run(args)
    except Exception as error:
        summary = {
            "execution_status": "ORACLE_ERROR",
            "error_type": type(error).__name__,
            "error": str(error),
            "deployment_eligible": False,
            "teacher_eligible": False,
            "paper_test_eligible": False,
        }
        _write_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
