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
from tools.train_a7c_r1_2a_quotient_compositor import (
    _build_streams,
    _load_teacher_manifest,
    verify_source_file,
)
from utils.a7c_renderer_compositor import (
    build_canary_splits,
    evaluate_contribution_predictions,
    normalized_flicker,
)
from utils.part_label_bank import PART_NAMES, load_part_label_bank


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def visibility_response_ratio(base_target, candidate_target, target_pixels) -> float:
    pixels = np.asarray(target_pixels, dtype=np.float64).reshape(-1)
    base = np.asarray(base_target, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_target, dtype=np.float64).reshape(-1)
    if not (pixels.shape == base.shape == candidate.shape) or np.any(pixels < 0.0):
        raise ValueError("visibility response inputs must align")
    base_flicker = normalized_flicker(base / np.maximum(pixels, 1.0))
    candidate_flicker = normalized_flicker(candidate / np.maximum(pixels, 1.0))
    if base_flicker <= 1.0e-12:
        return 1.0 if candidate_flicker <= 1.0e-12 else float("inf")
    return float(candidate_flicker / base_flicker)


def verify_frozen_artifacts(root: Path, expected: dict) -> dict[str, str]:
    freeze_path = Path(root) / "models_frozen.json"
    if not freeze_path.is_file():
        raise ValueError("models_frozen manifest is missing")
    manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) != 42:
        raise ValueError("models_frozen must contain exactly 42 artifact hashes")
    observed = {}
    for relative, fingerprint in artifacts.items():
        path = Path(root) / relative
        if not path.is_file() or _sha256(path) != str(fingerprint):
            raise ValueError(f"frozen artifact mismatch: {relative}")
        observed[relative] = str(fingerprint)
    for relative, fingerprint in expected.items():
        if observed.get(str(relative)) != str(fingerprint):
            raise ValueError(f"expected frozen artifact differs: {relative}")
    for predictor in ("training", "nearest_neighbor"):
        for fold in range(6):
            with np.load(Path(root) / predictor / f"fold_{fold}/predictions.npz", allow_pickle=False) as source:
                for flag in ("deployment_eligible", "teacher_eligible", "paper_test_eligible"):
                    if int(source[flag]) != 0:
                        raise ValueError(f"{predictor} eligibility flag is true")
                teacher_mask = np.asarray(source["teacher_mask"], dtype=bool)
                prediction_mask = np.asarray(source["prediction_mask"], dtype=bool)
                gates = np.asarray(source["projected_gates"], dtype=np.float64)
                if np.any(~np.isfinite(gates[prediction_mask])) or np.any(np.isfinite(gates[~prediction_mask])):
                    raise ValueError(f"{predictor} prediction mask leakage")
                if np.any(teacher_mask & ~prediction_mask):
                    raise ValueError(f"{predictor} teacher mask differs from fit domain")
            certificates = json.loads((Path(root) / predictor / f"fold_{fold}/projection_certificates.json").read_text(encoding="utf-8"))
            if len(certificates) != 24 or any(float(row["maximum_primal_violation"]) > 1.0e-7 for row in certificates):
                raise ValueError(f"{predictor} projection certificates are incomplete")
    return observed


def _summarize(records, contract) -> dict:
    summary = summarize_records(records, contract)
    summary["maximum_visibility_response_ratio"] = float(max(row["visibility_response_ratio"] for row in records))
    summary["all_spatial_guards_passed"] = bool(all(
        row[name] for row in records for name in (
            "topology_passed", "coverage_passed", "frozen_parts_passed",
            "weight_upper_bound_passed",
        )
    ))
    summary["per_camera"] = {
        str(camera): {
            "outer_gain": float(np.mean([row["outer_gain"] for row in records if row["camera_index"] == camera])),
            "boundary_gain": float(np.mean([row["boundary_gain"] for row in records if row["camera_index"] == camera])),
        } for camera in range(4)
    }
    return summary


def classify_canary(learned_records, nearest_neighbor_records, contract) -> str:
    if len(learned_records) != 24 or len(nearest_neighbor_records) != 24:
        raise ValueError("R1.4-VP audit requires 24 learned and 24 NN records")
    if len({(row["fold"], row["camera_index"]) for row in learned_records}) != 24:
        raise ValueError("learned records are not unique")
    learned = _summarize(learned_records, contract)
    nearest = _summarize(nearest_neighbor_records, contract)
    tolerance = float(contract["comparison_tolerance"])
    per_camera_positive = all(
        values["outer_gain"] > 0.0 and values["boundary_gain"] > 0.0
        for values in learned["per_camera"].values()
    )
    superior = bool(
        learned["outer_gain"] > float(contract["r1_2b_outer_gain"]) + tolerance
        and learned["boundary_gain"] > float(contract["r1_2b_boundary_gain"]) + tolerance
        and learned["outer_gain"] > nearest["outer_gain"] + tolerance
        and learned["boundary_gain"] > nearest["boundary_gain"] + tolerance
    )
    passed = bool(
        learned["passed"] and learned["all_spatial_guards_passed"]
        and learned["maximum_visibility_response_ratio"] <= float(contract["maximum_visibility_response_ratio"]) + 1.0e-7
        and per_camera_positive and superior
    )
    return "CANARY_PROMOTED" if passed else "CANARY_NEGATIVE"


def _load_predictions(root: Path, predictor: str, fold: int, expected_mask) -> np.ndarray:
    with np.load(root / predictor / f"fold_{fold}/predictions.npz", allow_pickle=False) as source:
        mask = np.asarray(source["prediction_mask"], dtype=bool)
        gates = np.asarray(source["projected_gates"], dtype=np.float64)
    if not np.array_equal(mask, expected_mask):
        raise ValueError(f"{predictor} prediction mask differs from frozen fit domain")
    return gates


def _evaluate_record(streams, gates, selected, pixels, fold, camera, guards) -> dict:
    outputs = {}
    for role in ("objective", "guard"):
        kwargs = {}
        for signal in ("target", "outer", "boundary"):
            kwargs[signal] = streams[role][signal]["base"][selected]
            kwargs[f"point_{signal}"] = streams[role][signal]["point"][selected]
        outputs[role] = evaluate_contribution_predictions(**kwargs, gates=gates[selected])
    base_target = streams["objective"]["target"]["base"][selected]
    return {
        "fold": int(fold), "camera_index": int(camera),
        "outer_gain": float(outputs["objective"]["outer_gain"]),
        "boundary_gain": float(outputs["objective"]["boundary_gain"]),
        "minimum_target_response": float(outputs["guard"]["minimum_target_response"]),
        "maximum_soft_iou_drop": float(outputs["guard"]["maximum_soft_iou_drop"]),
        "visibility_response_ratio": visibility_response_ratio(base_target, outputs["objective"]["target"], pixels[selected]),
        "maximum_adjacent_gate_change": float(np.max(np.abs(np.diff(gates[selected], axis=0)))),
        **guards,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Independently audit frozen R1.4-VP held blocks.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--a5-bank", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--witness-dir", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _run(args) -> tuple[dict, int]:
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    artifact_hashes = verify_frozen_artifacts(args.frozen_root, expected={})
    source_hashes = {
        "evidence": verify_source_file(args.evidence, contract["source_evidence_sha256"], "evidence"),
        "A5 bank": verify_source_file(args.a5_bank, contract["source_a5_bank_sha256"], "A5 bank"),
        "teacher": verify_source_file(args.teacher, contract["source_teacher_sha256"], "teacher"),
    }
    teacher = _load_teacher_manifest(args.teacher)
    cameras = np.asarray(teacher["camera_index"])
    frames = np.asarray(teacher["frame_index"])
    carriers = np.asarray(teacher["carrier_ids"], dtype=np.int64)
    with np.load(args.evidence, allow_pickle=False) as source:
        evidence = {key: source[key] for key in source.files}
    if not np.array_equal(evidence["renderer_sequence_camera_index"], cameras) or not np.array_equal(evidence["renderer_sequence_frame_index"], frames):
        raise ValueError("evidence sample manifest differs")
    bank = load_part_label_bank(args.a5_bank)
    part_index = PART_NAMES.index(str(contract["part"]))
    weights = np.asarray(bank["soft_edit_weights"], dtype=np.float64)
    lower = weights[:, part_index]
    streams = _build_streams(evidence, lower, carriers, part_index)
    pixels = np.asarray(evidence["renderer_sequence_target_pixel_count"], np.float64)[:, part_index]
    support = np.asarray(evidence["temporal_consecutive_visible_count"], np.int64)[:, part_index] >= int(contract["min_pair_support"])
    coverage = float(np.mean(support[carriers]))
    common_guards = {
        "topology_passed": True,
        "coverage_passed": bool(coverage >= float(contract["minimum_evidence_support_coverage"])),
        "frozen_parts_passed": True,
        "weight_upper_bound_passed": True,
    }
    split = build_canary_splits(camera_index=cameras, frame_index=frames,
        fit_camera_indices=(0, 1, 2, 3), audit_camera_indices=(4, 5, 6, 7),
        block_count=int(contract["temporal_block_count"]))
    records = {"learned": [], "nearest_neighbor": []}
    oracle = []
    for fold, held in enumerate(split["held_block_masks"]):
        fit_domain = np.asarray(split["fit_mask"], bool)
        learned = _load_predictions(args.frozen_root, "training", fold, fit_domain)
        nearest = _load_predictions(args.frozen_root, "nearest_neighbor", fold, fit_domain)
        witness_path = args.witness_dir / f"fold_{fold}/predictions.npz"
        with np.load(witness_path, allow_pickle=False) as source:
            witness = np.asarray(source["replay_gates"], np.float64)
        held_fit = held & split["fit_mask"]
        for name, gates in (("learned", learned), ("nearest_neighbor", nearest)):
            candidate = lower[carriers][None, :] * gates[held_fit]
            base = lower[carriers][None, :]
            guards = dict(common_guards)
            guards["topology_passed"] = bool(np.array_equal(candidate >= float(contract["selection_threshold"]), base >= float(contract["selection_threshold"])))
            guards["weight_upper_bound_passed"] = bool(np.all(candidate <= base + 1.0e-12))
            for camera in range(4):
                selected = held_fit & (cameras == camera)
                records[name].append(_evaluate_record(streams, gates, selected, pixels, fold, camera, guards))
        oracle.append({
            "fold": fold,
            "learned_gate_mae": float(np.mean(np.abs(learned[held_fit] - witness[held_fit]))),
            "learned_temporal_difference_mae": float(np.mean(np.abs(np.diff(learned[held_fit], axis=0) - np.diff(witness[held_fit], axis=0)))),
        })
    verdict = classify_canary(records["learned"], records["nearest_neighbor"], contract)
    learned_summary = _summarize(records["learned"], contract)
    nn_summary = _summarize(records["nearest_neighbor"], contract)
    payload = {
        "stage": "r1_4vp_held_canary", "verdict": verdict,
        "learned_summary": learned_summary, "nearest_neighbor_summary": nn_summary,
        "learned_records": records["learned"], "nearest_neighbor_records": records["nearest_neighbor"],
        "comparison_margins": {
            "outer_over_r1_2b": learned_summary["outer_gain"] - float(contract["r1_2b_outer_gain"]),
            "boundary_over_r1_2b": learned_summary["boundary_gain"] - float(contract["r1_2b_boundary_gain"]),
            "outer_over_nearest_neighbor": learned_summary["outer_gain"] - nn_summary["outer_gain"],
            "boundary_over_nearest_neighbor": learned_summary["boundary_gain"] - nn_summary["boundary_gain"],
        },
        "oracle_diagnostics": oracle, "source_fingerprints": source_hashes,
        "artifact_fingerprints": artifact_hashes, "evidence_support_coverage": coverage,
        "audit_camera_metrics_opened": False, "deployment_eligible": False,
        "teacher_eligible": False, "paper_test_eligible": False,
    }
    return payload, 0 if verdict == "CANARY_PROMOTED" else 2


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        payload, status = _run(args)
    except Exception as error:
        payload = {"stage": "r1_4vp_held_canary", "verdict": "TRAINING_ERROR",
            "error_type": type(error).__name__, "error": str(error),
            "deployment_eligible": False, "teacher_eligible": False,
            "paper_test_eligible": False}
        _write_json(args.output_dir / "held_block_summary.json", payload)
        _write_json(args.output_dir.parent / "summary.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    _write_json(args.output_dir / "held_block_summary.json", payload)
    _write_json(args.output_dir.parent / "summary.json", payload)
    print(json.dumps(payload["learned_summary"], indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
