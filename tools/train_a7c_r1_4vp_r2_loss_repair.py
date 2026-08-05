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
    _load_probe,
    _load_teacher_manifest,
    sample_block_ids,
    verify_source_file,
)
from utils.a7c_r1_4vp_r2 import r2_distillation_loss, require_fit_integrity
from utils.a7c_r1_4vp_r2_runtime import (
    ViewPoseResidualCompositor,
    apply_normalization,
    build_runtime_inputs,
    fit_normalization,
    load_pose_rotation_6d,
    pack_camera_block_segments,
    pose_manifest_sha256,
)
from utils.a7c_renderer_compositor import build_canary_splits
from utils.a7c_temporal_joint_projection import solve_temporal_joint_projection
from utils.part_label_bank import PART_NAMES, load_part_label_bank


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _project_segments(
    raw, prediction_mask, runtime_mass, a5_weight, camera_index, frame_index,
    block_ids, contract,
) -> tuple[np.ndarray, list[dict]]:
    raw = np.asarray(raw, dtype=np.float64)
    mask = np.asarray(prediction_mask, dtype=bool).reshape(-1)
    projected = np.full(raw.shape, np.nan, dtype=np.float64)
    certificates = []
    for indices in pack_camera_block_segments(
        np.asarray(camera_index)[mask], np.asarray(block_ids)[mask],
        np.asarray(frame_index)[mask], frame_stride=int(contract["frame_stride"]),
    ):
        full = np.flatnonzero(mask)[indices]
        solved = solve_temporal_joint_projection(
            raw_gates=raw[full], runtime_mass=np.asarray(runtime_mass)[full],
            a5_weight=a5_weight, minimum_gate=float(contract["minimum_gate"]),
            maximum_gate=float(contract["maximum_gate"]),
            selection_threshold=float(contract["selection_threshold"]),
            proxy_target_response=float(contract["proxy_target_response"]),
            maximum_gate_jump=float(contract["maximum_projection_gate_jump"]),
            rho_tolerance=float(contract["lexicographic_tolerance"]),
            primal_tolerance=float(contract["solver_primal_tolerance"]),
            residual_tolerance=float(contract["solver_residual_tolerance"]),
        )
        projected[full] = solved["gates"]
        certificates.append({
            "camera_index": int(np.asarray(camera_index)[full[0]]),
            "block_id": int(np.asarray(block_ids)[full[0]]),
            "sample_count": int(full.size),
            **solved["certificate"],
        })
    return projected, certificates


def train_fold(
    *, fold, features, pose, adjacency, visibility, base_gates, teacher_gates,
    teacher_mask, prediction_mask, camera_index, frame_index, block_ids,
    runtime_mass, a5_weight, contract, output_dir, device,
) -> dict:
    output = Path(output_dir)
    values = np.asarray(features, dtype=np.float32)
    poses = np.asarray(pose, dtype=np.float32)
    base = np.asarray(base_gates, dtype=np.float32)
    teachers = np.asarray(teacher_gates, dtype=np.float32)
    fit_mask = np.asarray(teacher_mask, dtype=bool).reshape(-1)
    predict_mask = np.asarray(prediction_mask, dtype=bool).reshape(-1)
    samples, carriers, channels = values.shape
    if poses.shape != (samples, 36) or base.shape != (samples, carriers):
        raise ValueError("training tensors do not align")
    if teachers.shape != base.shape:
        raise ValueError("teacher tensor differs from base gates")
    if fit_mask.shape != (samples,) or predict_mask.shape != fit_mask.shape:
        raise ValueError("training masks do not align")
    if not np.any(fit_mask) or np.any(~np.isfinite(teachers[fit_mask])):
        raise ValueError("fit teacher values must be finite")
    if np.any(np.isfinite(teachers[~fit_mask])):
        raise ValueError("held teacher values must remain NaN")

    feature_stats = fit_normalization(values, fit_mask)
    pose_stats = fit_normalization(poses, fit_mask)
    normalized_features = apply_normalization(values, feature_stats).astype(np.float32)
    normalized_pose = apply_normalization(poses, pose_stats).astype(np.float32)
    segments = pack_camera_block_segments(
        camera_index, block_ids, frame_index,
        frame_stride=int(contract["frame_stride"]),
    )
    training_segments = [segment for segment in segments if np.all(fit_mask[segment])]
    if not training_segments or sum(row.size for row in training_segments) != int(fit_mask.sum()):
        raise ValueError("fit mask must contain complete camera-block segments")

    torch.manual_seed(int(contract["random_seed"]) + int(fold))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(contract["random_seed"]) + int(fold))
    model = ViewPoseResidualCompositor(
        view_dimension=channels,
        view_embedding_dimension=int(contract["view_embedding_dimension"]),
        pose_dimension=36,
        pose_embedding_dimension=int(contract["pose_embedding_dimension"]),
        gru_hidden_dimension=int(contract["gru_hidden_dimension"]),
        residual_gate_scale=float(contract["residual_gate_scale"]),
        minimum_gate=float(contract["minimum_gate"]),
        maximum_gate=float(contract["maximum_gate"]),
    ).to(device)
    parameter_count = sum(value.numel() for value in model.parameters())
    if parameter_count > int(contract["maximum_parameter_count"]):
        raise ValueError("model exceeds frozen parameter budget")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(contract["learning_rate"]),
        weight_decay=float(contract["weight_decay"]),
    )
    tensors = {
        "features": torch.as_tensor(normalized_features, device=device),
        "pose": torch.as_tensor(normalized_pose, device=device),
        "adjacency": torch.as_tensor(np.asarray(adjacency, np.float32), device=device),
        "visibility": torch.as_tensor(np.asarray(visibility, np.float32), device=device),
        "base": torch.as_tensor(base, device=device),
        "teacher": torch.as_tensor(np.nan_to_num(teachers, nan=0.0), device=device),
    }

    def segment_components(indices):
        gates, residual = model.predict_with_residual(
            tensors["features"][indices], tensors["pose"][indices],
            tensors["adjacency"][indices], tensors["visibility"][indices],
            tensors["base"][indices],
        )
        return r2_distillation_loss(
            gates, tensors["teacher"][indices], residual,
            gate_delta=float(contract["gate_huber_delta"]),
            temporal_delta=float(contract["temporal_huber_delta"]),
            temporal_weight=float(contract["temporal_loss_weight"]),
            residual_weight=float(contract["residual_loss_weight"]),
        )

    def aggregate_components():
        rows = [segment_components(segment) for segment in training_segments]
        return {
            key: torch.stack([row[key] for row in rows]).mean()
            for key in rows[0]
        }

    with torch.no_grad():
        initial = {
            key: float(value.detach().cpu())
            for key, value in aggregate_components().items()
        }
    maximum_gradient_norm = 0.0
    epochs = int(contract["training_epochs"])
    for epoch in range(epochs):
        for segment in training_segments:
            optimizer.zero_grad(set_to_none=True)
            components = segment_components(segment)
            components["loss"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(contract["gradient_clip_norm"])
            )
            maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
            optimizer.step()
        if epoch + 1 in {1, epochs}:
            print(json.dumps({"fold": int(fold), "epoch": epoch + 1,
                              "epochs": epochs}), flush=True)

    with torch.no_grad():
        final = {
            key: float(value.detach().cpu())
            for key, value in aggregate_components().items()
        }
        raw = np.full((samples, carriers), np.nan, dtype=np.float64)
        residual_values = np.full_like(raw, np.nan)
        for segment in segments:
            if not np.all(predict_mask[segment]):
                continue
            predicted, residual = model.predict_with_residual(
                tensors["features"][segment], tensors["pose"][segment],
                tensors["adjacency"][segment], tensors["visibility"][segment],
                tensors["base"][segment],
            )
            raw[segment] = predicted.cpu().numpy()
            residual_values[segment] = residual.cpu().numpy()
    fit_mae = float(np.mean(np.abs(raw[fit_mask] - teachers[fit_mask])))
    require_fit_integrity(
        initial["loss"], final["loss"], fit_mae,
        float(contract["maximum_fit_teacher_mae"]),
    )
    projected, certificates = _project_segments(
        raw, predict_mask, runtime_mass, a5_weight, camera_index, frame_index,
        block_ids, contract,
    )
    temporal_errors = [
        np.abs(np.diff(raw[segment], axis=0) - np.diff(teachers[segment], axis=0))
        for segment in training_segments
    ]
    temporal_mae = float(np.mean(np.concatenate(
        [row.reshape(-1) for row in temporal_errors]
    )))
    teacher_displacement = float(np.mean(np.abs(teachers[fit_mask] - base[fit_mask])))
    learned_displacement = float(np.mean(np.abs(raw[fit_mask] - base[fit_mask])))
    recovery_ratio = (
        learned_displacement / teacher_displacement
        if teacher_displacement > 1.0e-12 else 1.0
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(), "feature_mean": feature_stats["mean"],
        "feature_scale": feature_stats["scale"], "pose_mean": pose_stats["mean"],
        "pose_scale": pose_stats["scale"], "contract": dict(contract),
        "checkpoint_epoch": epochs, "deployment_eligible": False,
        "teacher_eligible": False, "paper_test_eligible": False,
    }
    torch.save(checkpoint, output / "model.pt")
    np.savez_compressed(
        output / "predictions.npz", raw_gates=raw, projected_gates=projected,
        raw_residual=residual_values, prediction_mask=predict_mask,
        teacher_mask=fit_mask, camera_index=np.asarray(camera_index),
        frame_index=np.asarray(frame_index), block_ids=np.asarray(block_ids),
        feature_mean=feature_stats["mean"], feature_scale=feature_stats["scale"],
        pose_mean=pose_stats["mean"], pose_scale=pose_stats["scale"],
        deployment_eligible=np.asarray(0, np.uint8),
        teacher_eligible=np.asarray(0, np.uint8),
        paper_test_eligible=np.asarray(0, np.uint8),
    )
    _write_json(output / "projection_certificates.json", _jsonable(certificates))
    finite_residual = residual_values[predict_mask]
    summary = {
        "fold": int(fold), "epochs": epochs, "checkpoint_epoch": epochs,
        "parameter_count": int(parameter_count),
        "training_segment_count": len(training_segments),
        "training_sample_count": int(fit_mask.sum()),
        "prediction_sample_count": int(predict_mask.sum()),
        "initial_components": initial, "final_components": final,
        "maximum_gradient_norm_before_clip": maximum_gradient_norm,
        "raw_minimum_gate": float(np.nanmin(raw)),
        "raw_maximum_gate": float(np.nanmax(raw)),
        "projected_minimum_gate": float(np.nanmin(projected)),
        "projected_maximum_gate": float(np.nanmax(projected)),
        "fit_teacher_mae": fit_mae,
        "fit_temporal_difference_mae": temporal_mae,
        "latent_residual_mean": float(np.mean(np.abs(finite_residual))),
        "latent_residual_maximum": float(np.max(np.abs(finite_residual))),
        "base_to_teacher_mean_displacement": teacher_displacement,
        "base_to_learned_mean_displacement": learned_displacement,
        "teacher_displacement_recovery_ratio": float(recovery_ratio),
        "residual_loss_weight": float(contract["residual_loss_weight"]),
        "held_teacher_values_accessed": False, "deployment_eligible": False,
        "teacher_eligible": False, "paper_test_eligible": False,
    }
    _write_json(output / "summary.json", summary)
    return summary


def _verify_frozen_inputs(contract: dict, teachers_dir: Path) -> None:
    for relative, expected in contract["source_teacher_artifacts"].items():
        verify_source_file(teachers_dir / relative, expected, f"teacher {relative}")
    for relative, expected in zip(
        contract["source_r1_2b_predictions"],
        contract["source_r1_2b_prediction_sha256"],
    ):
        verify_source_file(REPO_ROOT / relative, expected, f"R1.2-B {relative}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train frozen R1.4-VP-R2 folds.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--a5-bank", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--teachers-dir", type=Path, required=True)
    parser.add_argument("--r1-2b-training-dir", type=Path, required=True)
    parser.add_argument("--pose-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if float(contract["residual_loss_weight"]) != 0.00001:
        raise ValueError("R2 contract residual loss weight differs")
    for path, expected, name in (
        (args.probe, contract["source_probe_sha256"], "probe"),
        (args.a5_bank, contract["source_a5_bank_sha256"], "A5 bank"),
        (args.teacher, contract["source_teacher_sha256"], "teacher"),
    ):
        verify_source_file(path, expected, name)
    _verify_frozen_inputs(contract, args.teachers_dir)
    probe = _load_probe(args.probe)
    manifest = _load_teacher_manifest(args.teacher)
    for key in ("carrier_ids", "camera_index", "frame_index"):
        if not np.array_equal(probe[key], manifest[key]):
            raise ValueError(f"probe and teacher {key} differ")
    cameras = np.asarray(manifest["camera_index"])
    frames = np.asarray(manifest["frame_index"])
    carriers = np.asarray(manifest["carrier_ids"], dtype=np.int64)
    unique_frames = np.unique(frames)
    pose_values = load_pose_rotation_6d(
        args.pose_model_dir, unique_frames, contract["pose_body_joint_indices"]
    )
    if pose_manifest_sha256(args.pose_model_dir, unique_frames, REPO_ROOT) != contract["source_pose_manifest_sha256"]:
        raise ValueError("pose manifest fingerprint mismatch")
    pose_by_frame = dict(zip(map(int, unique_frames), pose_values))
    r1_contract = json.loads(
        (REPO_ROOT / contract["source_r1_1_contract"]).read_text(encoding="utf-8")
    )
    bank = load_part_label_bank(args.a5_bank)
    part_index = PART_NAMES.index(str(contract["part"]))
    a5_weight = np.asarray(bank["soft_edit_weights"], np.float32)[carriers, part_index]
    runtime = build_runtime_inputs(
        probe=probe, feature_names=list(map(str, probe["feature_names"])),
        feature_group=r1_contract["feature_groups"][contract["view_feature_group"]],
        pose_by_frame=pose_by_frame, camera_index=cameras, frame_index=frames,
        carrier_ids=carriers, a5_weight=a5_weight,
        spatial_scale=float(contract["spatial_scale"]),
        depth_scale=float(contract["depth_scale"]),
        edge_log_weight_minimum=float(contract["edge_log_weight_minimum"]),
    )
    blocks = sample_block_ids(cameras, frames, int(contract["temporal_block_count"]))
    split = build_canary_splits(
        camera_index=cameras, frame_index=frames,
        fit_camera_indices=(0, 1, 2, 3), audit_camera_indices=(4, 5, 6, 7),
        block_count=int(contract["temporal_block_count"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    learned, freeze_paths = [], []
    for fold in range(6):
        teacher_path = args.teachers_dir / f"fold_{fold}/teacher.npz"
        with np.load(teacher_path, allow_pickle=False) as source:
            teacher_gates = np.asarray(source["teacher_gates"], np.float32)
            teacher_mask = np.asarray(source["teacher_mask"], bool)
        base_path = args.r1_2b_training_dir / f"fold_{fold}/predictions.npz"
        with np.load(base_path, allow_pickle=False) as source:
            base = np.asarray(source["raw_gates"], np.float32)
        prediction_mask = np.asarray(split["fit_mask"], bool)
        fold_root = args.output_dir / "training" / f"fold_{fold}"
        learned.append(train_fold(
            fold=fold, features=runtime["features"], pose=runtime["pose"],
            adjacency=runtime["adjacency"], visibility=runtime["visibility"],
            base_gates=base, teacher_gates=teacher_gates,
            teacher_mask=teacher_mask, prediction_mask=prediction_mask,
            camera_index=cameras, frame_index=frames, block_ids=blocks,
            runtime_mass=runtime["runtime_mass"], a5_weight=a5_weight,
            contract=contract, output_dir=fold_root, device=args.device,
        ))
        freeze_paths.extend([
            fold_root / "model.pt", fold_root / "predictions.npz",
            fold_root / "projection_certificates.json", fold_root / "summary.json",
        ])
    _write_json(args.output_dir / "training/summary.json", {
        "folds": learned, "execution_status": "TRAINING_COMPLETED",
        "paper_test_eligible": False,
    })
    frozen = {
        str(path.relative_to(args.output_dir)): _sha256(path)
        for path in freeze_paths
    }
    _write_json(args.output_dir / "models_frozen.json", {
        "artifacts": frozen,
        "source_nearest_neighbor_prediction_sha256":
            contract["source_nearest_neighbor_prediction_sha256"],
        "paper_test_eligible": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
