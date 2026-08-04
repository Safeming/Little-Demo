#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train_a7c_r1_2a_quotient_compositor import (
    _load_probe,
    _load_teacher_manifest,
    sample_block_ids,
    verify_source_file,
)
from utils.a7c_quotient_compositor import runtime_target_mass
from utils.a7c_renderer_compositor import build_canary_splits
from utils.a7c_temporal_joint_projection import (
    solve_temporal_joint_projection,
)
from utils.part_label_bank import PART_NAMES, load_part_label_bank


def project_fold_predictions(
    *,
    raw_gates,
    runtime_mass,
    a5_weight,
    projection_mask,
    camera_index,
    frame_index,
    block_ids,
    fit_camera_indices,
    contract,
) -> dict:
    raw = np.asarray(raw_gates, dtype=np.float64)
    mass = np.asarray(runtime_mass, dtype=np.float64)
    mask = np.asarray(projection_mask, dtype=bool).reshape(-1)
    cameras = np.asarray(camera_index).reshape(-1)
    frames = np.asarray(frame_index).reshape(-1)
    blocks = np.asarray(block_ids).reshape(-1)
    if raw.ndim != 2 or mass.shape != raw.shape:
        raise ValueError("raw gates and runtime mass must share [samples, carriers]")
    if not (
        mask.shape == cameras.shape == frames.shape == blocks.shape == (raw.shape[0],)
    ):
        raise ValueError("projection manifest arrays must share sample count")

    projected = np.full(raw.shape, np.nan, dtype=np.float64)
    certificates = []
    for camera in tuple(map(int, fit_camera_indices)):
        selected = mask & (cameras == camera)
        indices = np.flatnonzero(selected)
        if indices.size < 2:
            raise ValueError(f"camera {camera} held segment is incomplete")
        selected_blocks = np.unique(blocks[indices])
        if selected_blocks.size != 1:
            raise ValueError("projection segment crosses temporal blocks")
        if not np.all(np.diff(frames[indices]) == int(contract["frame_stride"])):
            raise ValueError("projection segment has a frame gap")
        solved = solve_temporal_joint_projection(
            raw_gates=raw[indices],
            runtime_mass=mass[indices],
            a5_weight=a5_weight,
            minimum_gate=float(contract["minimum_gate"]),
            maximum_gate=float(contract["maximum_gate"]),
            selection_threshold=float(contract["selection_threshold"]),
            proxy_target_response=float(contract["proxy_target_response"]),
            maximum_gate_jump=float(contract["maximum_projection_gate_jump"]),
            rho_tolerance=float(contract["lexicographic_rho_tolerance"]),
            primal_tolerance=float(contract["solver_primal_tolerance"]),
            residual_tolerance=float(contract["solver_residual_tolerance"]),
        )
        projected[indices] = solved["gates"]
        certificate = dict(solved["certificate"])
        certificate.update(
            {
                "camera_index": camera,
                "block_index": int(selected_blocks[0]),
                "first_frame": int(frames[indices[0]]),
                "last_frame": int(frames[indices[-1]]),
                "sample_count": int(indices.size),
            }
        )
        certificates.append(certificate)
    if np.any(~np.isfinite(projected[mask])) or np.any(
        np.isfinite(projected[~mask])
    ):
        raise RuntimeError("projected gate mask is not isolated")
    return {"projected_gates": projected, "certificates": certificates}


def summarize_projection(
    *,
    experiment_id: str,
    fold_summaries,
    source_fingerprints,
    prediction_fingerprints,
) -> dict:
    folds = list(fold_summaries)
    predictions = dict(prediction_fingerprints)
    if len(folds) != 6 or len(predictions) != 6:
        raise ValueError("projection summary requires six folds and fingerprints")
    return {
        "experiment_id": str(experiment_id),
        "folds": folds,
        "segment_count": int(sum(row["segment_count"] for row in folds)),
        "source_fingerprints": dict(source_fingerprints),
        "prediction_fingerprints": predictions,
        "renderer_values_accessed": False,
        "paper_test_eligible": False,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Project frozen R1.2-B raw gates with temporal hard constraints."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--a5-bank", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--source-training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    source_specs = [
        (
            REPO_ROOT / contract["source_r1_3p_design"],
            contract["source_r1_3p_design_sha256"],
            "R1.3-P design",
        ),
        (
            REPO_ROOT / contract["source_r1_2b_contract"],
            contract["source_r1_2b_contract_sha256"],
            "R1.2-B contract",
        ),
        (
            REPO_ROOT / contract["source_r1_2b_training_summary"],
            contract["source_r1_2b_training_summary_sha256"],
            "R1.2-B training summary",
        ),
        (args.probe, contract["source_probe_sha256"], "probe"),
        (args.a5_bank, contract["source_a5_bank_sha256"], "A5 bank"),
        (args.teacher, contract["source_teacher_sha256"], "teacher"),
    ]
    source_fingerprints = {
        name: verify_source_file(path, expected, name)
        for path, expected, name in source_specs
    }
    probe = _load_probe(args.probe)
    teacher = _load_teacher_manifest(args.teacher)
    for key in ("carrier_ids", "camera_index", "frame_index"):
        if not np.array_equal(probe[key], teacher[key]):
            raise ValueError(f"probe and teacher {key} differ")
    bank = load_part_label_bank(args.a5_bank)
    part_index = PART_NAMES.index(str(contract["part"]))
    all_weights = np.asarray(bank["soft_edit_weights"], dtype=np.float64)[
        :, part_index
    ]
    carrier_ids = np.asarray(teacher["carrier_ids"], dtype=np.int64)
    a5_weight = all_weights[carrier_ids]
    features = np.asarray(probe["features"], dtype=np.float32)
    feature_names = list(map(str, probe["feature_names"]))
    field = {
        name: features[:, :, feature_names.index(name)]
        for name in (
            "alpha_transmittance_mass",
            "semantic_support_mean",
            "alpha_mean",
        )
    }
    mass = runtime_target_mass(
        alpha_transmittance_mass=torch.from_numpy(
            field["alpha_transmittance_mass"]
        ),
        a5_weight=torch.from_numpy(a5_weight.astype(np.float32)),
        semantic_support_mean=torch.from_numpy(field["semantic_support_mean"]),
        alpha_mean=torch.from_numpy(field["alpha_mean"]),
    ).numpy()
    camera_index = np.asarray(teacher["camera_index"])
    frame_index = np.asarray(teacher["frame_index"])
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_summaries = []
    prediction_fingerprints = {}
    prediction_paths = list(contract["source_r1_2b_predictions"])
    prediction_hashes = list(contract["source_r1_2b_prediction_sha256"])
    if len(prediction_paths) != 6 or len(prediction_hashes) != 6:
        raise ValueError("contract needs six source predictions")
    for fold, held in enumerate(split["held_block_masks"]):
        prediction_path = args.source_training_dir / f"fold_{fold}/predictions.npz"
        expected_path = REPO_ROOT / prediction_paths[fold]
        if prediction_path.resolve() != expected_path.resolve():
            raise ValueError("source training directory differs from contract")
        source_fingerprint = verify_source_file(
            prediction_path, prediction_hashes[fold], f"fold {fold} prediction"
        )
        prediction_fingerprints[f"fold_{fold}_prediction"] = source_fingerprint
        with np.load(prediction_path, allow_pickle=False) as source:
            raw_gates = np.asarray(source["raw_gates"], dtype=np.float64)
            train_mask = np.asarray(source["train_mask"], dtype=bool)
        if raw_gates.shape != mass.shape:
            raise ValueError("source prediction shape differs from runtime mass")
        projection_mask = np.asarray(held & split["fit_mask"], dtype=bool)
        projected = project_fold_predictions(
            raw_gates=raw_gates,
            runtime_mass=mass,
            a5_weight=a5_weight,
            projection_mask=projection_mask,
            camera_index=camera_index,
            frame_index=frame_index,
            block_ids=block_ids,
            fit_camera_indices=(0, 1, 2, 3),
            contract=contract,
        )
        fold_root = args.output_dir / f"fold_{fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        masked_raw = np.full(raw_gates.shape, np.nan, dtype=np.float64)
        masked_raw[projection_mask] = raw_gates[projection_mask]
        np.savez_compressed(
            fold_root / "predictions.npz",
            raw_gates=masked_raw,
            projected_gates=projected["projected_gates"],
            projection_mask=projection_mask,
            train_mask=train_mask,
            camera_index=camera_index,
            frame_index=frame_index,
            block_ids=block_ids,
            carrier_ids=carrier_ids,
            source_prediction_sha256=np.array(source_fingerprint),
            paper_test_eligible=np.array(0, dtype=np.uint8),
        )
        (fold_root / "segment_certificates.json").write_text(
            json.dumps(projected["certificates"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        fold_summaries.append(
            {
                "fold": fold,
                "segment_count": len(projected["certificates"]),
                "sample_count": int(np.count_nonzero(projection_mask)),
                "maximum_displacement": max(
                    row["maximum_displacement"]
                    for row in projected["certificates"]
                ),
                "maximum_adjacent_gate_change": max(
                    row["maximum_adjacent_gate_change"]
                    for row in projected["certificates"]
                ),
                "maximum_primal_violation": max(
                    row["maximum_primal_violation"]
                    for row in projected["certificates"]
                ),
            }
        )
    summary = summarize_projection(
        experiment_id=contract["experiment_id"],
        fold_summaries=fold_summaries,
        source_fingerprints=source_fingerprints,
        prediction_fingerprints=prediction_fingerprints,
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
