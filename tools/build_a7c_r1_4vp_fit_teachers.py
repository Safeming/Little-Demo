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
from utils.a7c_oracle_distillation import (
    insert_teacher_segment,
    solve_fit_teacher,
)
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


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_fold_teacher(
    *,
    output_dir: Path,
    fold: int,
    gates,
    teacher_mask,
    camera_index,
    frame_index,
    block_ids,
    carrier_ids,
    certificates,
    source_fingerprints,
) -> None:
    output = Path(output_dir) / f"fold_{int(fold)}"
    values = np.asarray(gates, dtype=np.float64)
    mask = np.asarray(teacher_mask, dtype=bool).reshape(-1)
    cameras = np.asarray(camera_index).reshape(-1)
    frames = np.asarray(frame_index).reshape(-1)
    blocks = np.asarray(block_ids).reshape(-1)
    carriers = np.asarray(carrier_ids, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or mask.shape != (values.shape[0],):
        raise ValueError("teacher gates and mask differ")
    if not (
        cameras.shape == frames.shape == blocks.shape == mask.shape
    ) or carriers.shape != (values.shape[1],):
        raise ValueError("teacher manifests differ from gates")
    if np.any(~np.isfinite(values[mask])) or np.any(np.isfinite(values[~mask])):
        raise ValueError("teacher finite values cross fit mask")
    rows = list(certificates)
    if len(rows) != 20:
        raise ValueError("every fold requires exactly 20 teacher certificates")
    if {(int(row["camera_index"]), int(row["block_id"])) for row in rows} != {
        (camera, block)
        for camera in range(4)
        for block in range(6)
        if block != int(fold)
    }:
        raise ValueError("teacher certificates differ from fold fit grid")
    output.mkdir(parents=True, exist_ok=True)
    fingerprint_names = np.asarray(sorted(map(str, source_fingerprints)))
    fingerprint_values = np.asarray(
        [str(source_fingerprints[name]) for name in fingerprint_names]
    )
    np.savez_compressed(
        output / "teacher.npz",
        teacher_gates=values,
        teacher_mask=mask,
        camera_index=cameras,
        frame_index=frames,
        block_ids=blocks,
        carrier_ids=carriers,
        source_fingerprint_names=fingerprint_names,
        source_fingerprint_values=fingerprint_values,
        sample_order_fingerprint=np.asarray(
            _array_fingerprint(cameras, frames, blocks)
        ),
        carrier_order_fingerprint=np.asarray(_array_fingerprint(carriers)),
        gate_fingerprint=np.asarray(_array_fingerprint(values[mask])),
        deployment_eligible=np.asarray(0, dtype=np.uint8),
        teacher_eligible=np.asarray(0, dtype=np.uint8),
        paper_test_eligible=np.asarray(0, dtype=np.uint8),
    )
    _write_json(
        output / "certificates.json",
        {
            "fold": int(fold),
            "segment_count": len(rows),
            "certificates": _jsonable(rows),
            "source_fingerprints": dict(source_fingerprints),
            "deployment_eligible": False,
            "teacher_eligible": False,
            "paper_test_eligible": False,
        },
    )


def _record_streams(streams, selected) -> dict:
    return {
        role: {
            signal: {
                key: np.asarray(values)[selected]
                for key, values in stream.items()
            }
            for signal, stream in role_streams.items()
        }
        for role, role_streams in streams.items()
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate frozen R1.4-VP fit-only oracle teachers."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--a5-bank", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--r1-2b-training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _verify_sources(args, contract) -> dict[str, str]:
    expected_paths = {
        "probe": REPO_ROOT / contract["source_probe"],
        "evidence": REPO_ROOT / contract["source_evidence"],
        "A5 bank": REPO_ROOT / contract["source_a5_bank"],
        "teacher": REPO_ROOT / contract["source_teacher"],
    }
    actual_paths = {
        "probe": args.probe,
        "evidence": args.evidence,
        "A5 bank": args.a5_bank,
        "teacher": args.teacher,
    }
    for name in expected_paths:
        if actual_paths[name].resolve() != expected_paths[name].resolve():
            raise ValueError(f"{name} path differs from frozen contract")
    source_specs = [
        (args.probe, contract["source_probe_sha256"], "probe"),
        (args.evidence, contract["source_evidence_sha256"], "evidence"),
        (args.a5_bank, contract["source_a5_bank_sha256"], "A5 bank"),
        (args.teacher, contract["source_teacher_sha256"], "teacher"),
    ]
    for prefix, name in (
        ("source_design", "R1.4-VP design"),
        ("source_r1_3g_contract", "R1.3-G contract"),
        ("source_r1_3g_records", "R1.3-G records"),
        ("source_r1_3g_audit", "R1.3-G audit"),
        ("source_r1_3g_summary", "R1.3-G summary"),
        ("source_r1_3p_contract", "R1.3-P contract"),
        ("source_r1_2b_contract", "R1.2-B contract"),
        ("source_r1_2b_training_summary", "R1.2-B training"),
        ("source_r1_2b_audit", "R1.2-B audit"),
        ("source_r1_1_contract", "R1.1 contract"),
    ):
        source_specs.append(
            (
                REPO_ROOT / contract[prefix],
                contract[f"{prefix}_sha256"],
                name,
            )
        )
    fingerprints = {
        name: verify_source_file(path, expected, name)
        for path, expected, name in source_specs
    }
    expected_training = (
        REPO_ROOT / contract["source_r1_2b_predictions"][0]
    ).parents[1]
    if args.r1_2b_training_dir.resolve() != expected_training.resolve():
        raise ValueError("R1.2-B training directory differs from frozen contract")
    for fold, (path_text, expected) in enumerate(
        zip(
            contract["source_r1_2b_predictions"],
            contract["source_r1_2b_prediction_sha256"],
        )
    ):
        path = REPO_ROOT / path_text
        fingerprints[f"R1.2-B fold {fold}"] = verify_source_file(
            path, expected, f"R1.2-B fold {fold}"
        )
    entry = json.loads(
        (REPO_ROOT / contract["source_r1_3g_summary"]).read_text(
            encoding="utf-8"
        )
    )
    if entry.get("verdict") != "CERTIFIED_FEASIBLE":
        raise ValueError("R1.3-G entry gate is not certified feasible")
    return fingerprints


def _run(args) -> dict:
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen":
        raise ValueError("R1.4-VP contract must be frozen")
    source_fingerprints = _verify_sources(args, contract)
    probe = _load_probe(args.probe)
    teacher = _load_teacher_manifest(args.teacher)
    for key in ("carrier_ids", "camera_index", "frame_index"):
        if not np.array_equal(probe[key], teacher[key]):
            raise ValueError(f"probe and teacher {key} differ")
    if str(probe["source_teacher_fingerprint"]) != str(
        teacher["output_fingerprint"]
    ):
        raise ValueError("probe source teacher fingerprint differs")
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
    fold_summaries = []
    total_segments = 0
    for fold, held in enumerate(split["held_block_masks"]):
        prediction_path = REPO_ROOT / contract["source_r1_2b_predictions"][fold]
        with np.load(prediction_path, allow_pickle=False) as source:
            anchor = np.asarray(source["raw_gates"], dtype=np.float64)
        if anchor.shape != mass.shape or not np.isfinite(anchor).all():
            raise ValueError("R1.2-B anchor gates differ from runtime manifest")
        expected_mask = np.asarray(split["fit_mask"] & ~held, dtype=bool)
        gates = np.full(mass.shape, np.nan, dtype=np.float64)
        teacher_mask = np.zeros(camera_index.size, dtype=bool)
        certificates = []
        for camera in range(4):
            for block in range(int(contract["temporal_block_count"])):
                if block == fold:
                    continue
                selected = np.asarray(
                    expected_mask
                    & (camera_index == camera)
                    & (block_ids == block),
                    dtype=bool,
                )
                indices = np.flatnonzero(selected)
                if indices.size < 2 or not np.all(
                    np.diff(frame_index[indices]) == int(contract["frame_stride"])
                ):
                    raise ValueError("fit teacher segment is not contiguous")
                solved = solve_fit_teacher(
                    runtime_mass=mass[selected],
                    a5_weight=a5_weight,
                    streams=_record_streams(streams, selected),
                    anchor_gates=anchor[selected],
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
                    minimum_outer_gain=float(
                        contract["teacher_minimum_outer_gain"]
                    ),
                    boundary_margin=float(contract["teacher_boundary_margin"]),
                    bisection_tolerance=float(
                        contract["oracle_bisection_tolerance"]
                    ),
                    lexicographic_tolerance=float(
                        contract["lexicographic_tolerance"]
                    ),
                    primal_tolerance=float(contract["solver_primal_tolerance"]),
                    residual_tolerance=float(
                        contract["solver_residual_tolerance"]
                    ),
                )
                insert_teacher_segment(
                    gates,
                    teacher_mask,
                    selected,
                    solved,
                    residual_tolerance=float(
                        contract["solver_residual_tolerance"]
                    ),
                )
                row = {
                    "fold": fold,
                    "camera_index": camera,
                    "block_id": block,
                    "sample_count": int(indices.size),
                    "frame_start": int(frame_index[indices[0]]),
                    "frame_end": int(frame_index[indices[-1]]),
                    "capacity": solved["capacity"],
                    "request": solved["request"],
                    "metrics": solved["metrics"],
                    **solved["certificate"],
                    "sample_order_fingerprint": _array_fingerprint(
                        camera_index[selected],
                        frame_index[selected],
                        block_ids[selected],
                    ),
                    "carrier_order_fingerprint": _array_fingerprint(carrier_ids),
                }
                certificates.append(row)
                total_segments += 1
                print(
                    json.dumps(
                        {
                            "teacher_segment_complete": total_segments,
                            "fold": fold,
                            "camera_index": camera,
                            "block_id": block,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if not np.array_equal(teacher_mask, expected_mask):
            raise RuntimeError("fold teacher mask differs from frozen fit split")
        write_fold_teacher(
            output_dir=args.output_dir,
            fold=fold,
            gates=gates,
            teacher_mask=teacher_mask,
            camera_index=camera_index,
            frame_index=frame_index,
            block_ids=block_ids,
            carrier_ids=carrier_ids,
            certificates=certificates,
            source_fingerprints=source_fingerprints,
        )
        fold_summaries.append(
            {
                "fold": fold,
                "segment_count": len(certificates),
                "fit_sample_count": int(np.count_nonzero(teacher_mask)),
                "held_finite_count": int(np.count_nonzero(np.isfinite(gates[~teacher_mask]))),
                "maximum_primal_violation": max(
                    float(row["maximum_primal_violation"])
                    for row in certificates
                ),
                "gate_fingerprint": _array_fingerprint(gates[teacher_mask]),
                "deployment_eligible": False,
                "teacher_eligible": False,
                "paper_test_eligible": False,
            }
        )
    if total_segments != int(contract["fit_teacher_segment_count"]):
        raise RuntimeError("fit teacher segment count differs from contract")
    if any(row["segment_count"] != 20 for row in fold_summaries):
        raise RuntimeError("every fold must contain exactly 20 teacher segments")
    if any(row["held_finite_count"] != 0 for row in fold_summaries):
        raise RuntimeError("held teacher values became finite")
    summary = {
        "experiment_id": contract["experiment_id"],
        "execution_status": "TEACHERS_COMPLETED",
        "fit_teacher_segment_count": total_segments,
        "folds": fold_summaries,
        "source_fingerprints": source_fingerprints,
        "sample_order_fingerprint": _array_fingerprint(
            camera_index, frame_index, block_ids
        ),
        "carrier_order_fingerprint": _array_fingerprint(carrier_ids),
        "deployment_eligible": False,
        "teacher_eligible": False,
        "paper_test_eligible": False,
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary = _run(args)
    except Exception as error:
        summary = {
            "execution_status": "TRAINING_ERROR",
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
