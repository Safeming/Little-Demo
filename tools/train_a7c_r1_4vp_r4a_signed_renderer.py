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
    _build_streams,
    _load_probe,
    _load_teacher_manifest,
    sample_block_ids,
    verify_source_file,
)
from tools.train_a7c_r1_4vp_r2_loss_repair import (
    _jsonable,
    _project_segments,
    _sha256,
    _verify_frozen_inputs,
    _write_json,
)
from tools.train_a7c_r1_4vp_r3_crw import _fit_entry
from utils.a7c_r1_4vp_r2_runtime import (
    ViewPoseResidualCompositor,
    apply_normalization,
    build_runtime_inputs,
    fit_normalization,
    load_pose_rotation_6d,
    pack_camera_block_segments,
    pose_manifest_sha256,
)
from utils.a7c_r1_4vp_r3_crw import classify_fit_entry_failure
from utils.a7c_r1_4vp_r4a import (
    freeze_initial_scales,
    signed_renderer_trajectory_components,
    signed_renderer_trajectory_loss,
    summarize_action_recovery,
)
from utils.a7c_renderer_compositor import build_canary_splits
from utils.part_label_bank import PART_NAMES, load_part_label_bank


def _validate_renderer_streams(streams, fit_mask, samples, carriers):
    if not isinstance(streams, dict) or set(streams) != {
        "target",
        "outer",
        "boundary",
    }:
        raise ValueError("renderer streams must contain target, outer, and boundary")
    validated = {}
    for signal in ("target", "outer", "boundary"):
        stream = streams[signal]
        if not isinstance(stream, dict) or set(stream) != {"base", "point"}:
            raise ValueError(f"{signal} renderer stream must contain base and point")
        base = np.asarray(stream["base"], dtype=np.float32)
        point = np.asarray(stream["point"], dtype=np.float32)
        if base.shape != (samples,) or point.shape != (samples, carriers):
            raise ValueError(f"{signal} renderer stream does not align")
        if not np.isfinite(base[fit_mask]).all() or not np.isfinite(point[fit_mask]).all():
            raise ValueError(f"fit {signal} renderer values must be finite")
        if np.isfinite(base[~fit_mask]).any() or np.isfinite(point[~fit_mask]).any():
            raise ValueError(f"held {signal} renderer values must remain NaN")
        validated[signal] = {"base": base, "point": point}
    return validated


def train_fold(
    *,
    fold,
    features,
    pose,
    adjacency,
    visibility,
    base_gates,
    teacher_gates,
    renderer_streams,
    teacher_mask,
    prediction_mask,
    camera_index,
    frame_index,
    block_ids,
    runtime_mass,
    a5_weight,
    contract,
    output_dir,
    device,
) -> dict:
    output = Path(output_dir)
    values = np.asarray(features, dtype=np.float32)
    poses = np.asarray(pose, dtype=np.float32)
    base = np.asarray(base_gates, dtype=np.float32)
    teachers = np.asarray(teacher_gates, dtype=np.float32)
    fit_mask = np.asarray(teacher_mask, dtype=bool).reshape(-1)
    predict_mask = np.asarray(prediction_mask, dtype=bool).reshape(-1)
    if values.ndim != 3:
        raise ValueError("features must have shape [samples, carriers, channels]")
    samples, carriers, channels = values.shape
    if poses.shape != (samples, 36) or base.shape != (samples, carriers):
        raise ValueError("training tensors do not align")
    if teachers.shape != base.shape:
        raise ValueError("teacher tensors must match base gates")
    if fit_mask.shape != (samples,) or predict_mask.shape != fit_mask.shape:
        raise ValueError("training masks do not align")
    if not np.any(fit_mask) or not np.isfinite(teachers[fit_mask]).all():
        raise ValueError("fit teacher values must be finite")
    if np.isfinite(teachers[~fit_mask]).any():
        raise ValueError("held teacher values must remain NaN")
    streams = _validate_renderer_streams(
        renderer_streams, fit_mask, samples, carriers
    )

    feature_stats = fit_normalization(values, fit_mask)
    pose_stats = fit_normalization(poses, fit_mask)
    normalized_features = apply_normalization(values, feature_stats).astype(np.float32)
    normalized_pose = apply_normalization(poses, pose_stats).astype(np.float32)
    segments = pack_camera_block_segments(
        camera_index,
        block_ids,
        frame_index,
        frame_stride=int(contract["frame_stride"]),
    )
    training_segments = [segment for segment in segments if np.all(fit_mask[segment])]
    if not training_segments or sum(row.size for row in training_segments) != int(
        fit_mask.sum()
    ):
        raise ValueError("fit mask must contain complete camera-block segments")

    seed = int(contract["random_seed"]) + int(fold)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
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
    if channels == 49 and all(
        int(contract[key]) == 16
        for key in (
            "view_embedding_dimension",
            "pose_embedding_dimension",
            "gru_hidden_dimension",
        )
    ) and parameter_count != 9073:
        raise ValueError("R4-A model signature differs from the frozen R3 model")

    tensors = {
        "features": torch.as_tensor(normalized_features, device=device),
        "pose": torch.as_tensor(normalized_pose, device=device),
        "adjacency": torch.as_tensor(np.asarray(adjacency, np.float32), device=device),
        "visibility": torch.as_tensor(np.asarray(visibility, np.float32), device=device),
        "base": torch.as_tensor(base, device=device),
        "teacher": torch.as_tensor(np.nan_to_num(teachers, nan=0.0), device=device),
        "streams": {
            signal: {
                key: torch.as_tensor(np.nan_to_num(value, nan=0.0), device=device)
                for key, value in stream.items()
            }
            for signal, stream in streams.items()
        },
    }

    def raw_segment_components(indices, gates):
        segment_streams = {
            signal: {
                key: value[indices] for key, value in stream.items()
            }
            for signal, stream in tensors["streams"].items()
        }
        return signed_renderer_trajectory_components(
            gates,
            tensors["teacher"][indices],
            segment_streams,
            renderer_delta=float(contract["renderer_trajectory_huber_delta"]),
            target_delta=float(contract["target_response_huber_delta"]),
            gate_delta=float(contract["gate_huber_delta"]),
            gate_temporal_weight=float(contract["gate_auxiliary_temporal_weight"]),
            epsilon=float(contract["renderer_reconstruction_epsilon"]),
        )

    initial_scales = []
    with torch.no_grad():
        for segment in training_segments:
            gates, _ = model.predict_with_residual(
                tensors["features"][segment],
                tensors["pose"][segment],
                tensors["adjacency"][segment],
                tensors["visibility"][segment],
                tensors["base"][segment],
            )
            scales = freeze_initial_scales(
                raw_segment_components(segment, gates),
                minimum=float(contract["initial_scale_minimum"]),
            )
            initial_scales.append(scales)

    def segment_components(segment_index, indices):
        gates, residual = model.predict_with_residual(
            tensors["features"][indices],
            tensors["pose"][indices],
            tensors["adjacency"][indices],
            tensors["visibility"][indices],
            tensors["base"][indices],
        )
        raw = raw_segment_components(indices, gates)
        total = signed_renderer_trajectory_loss(
            raw,
            initial_scales[segment_index],
            residual,
            outer_weight=float(contract["renderer_outer_loss_weight"]),
            boundary_weight=float(contract["renderer_boundary_loss_weight"]),
            target_weight=float(contract["target_auxiliary_loss_weight"]),
            gate_aux_weight=float(contract["gate_auxiliary_loss_weight"]),
            residual_weight=float(contract["residual_loss_weight"]),
        )
        return {"loss": total["loss"], **raw, **{
            key: value for key, value in total.items() if key != "loss"
        }}

    def aggregate_components():
        rows = [
            segment_components(index, segment)
            for index, segment in enumerate(training_segments)
        ]
        return {key: torch.stack([row[key] for row in rows]).mean() for key in rows[0]}

    with torch.no_grad():
        initial = {
            key: float(value.detach().cpu())
            for key, value in aggregate_components().items()
        }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(contract["learning_rate"]),
        weight_decay=float(contract["weight_decay"]),
    )
    maximum_gradient_norm = 0.0
    epochs = int(contract["training_epochs"])
    for epoch in range(epochs):
        for segment_index, segment in enumerate(training_segments):
            optimizer.zero_grad(set_to_none=True)
            components = segment_components(segment_index, segment)
            components["loss"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(contract["gradient_clip_norm"])
            )
            maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
            optimizer.step()
        if epoch + 1 in {1, epochs}:
            print(
                json.dumps({"fold": int(fold), "epoch": epoch + 1, "epochs": epochs}),
                flush=True,
            )

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
                tensors["features"][segment],
                tensors["pose"][segment],
                tensors["adjacency"][segment],
                tensors["visibility"][segment],
                tensors["base"][segment],
            )
            raw[segment] = predicted.cpu().numpy()
            residual_values[segment] = residual.cpu().numpy()

    fit_mae = float(np.mean(np.abs(raw[fit_mask] - teachers[fit_mask])))
    projected, certificates = _project_segments(
        raw,
        predict_mask,
        runtime_mass,
        a5_weight,
        camera_index,
        frame_index,
        block_ids,
        contract,
    )
    temporal_errors = [
        np.abs(np.diff(raw[segment], axis=0) - np.diff(teachers[segment], axis=0))
        for segment in training_segments
    ]
    temporal_mae = float(
        np.mean(np.concatenate([row.reshape(-1) for row in temporal_errors]))
    )
    teacher_displacement = float(np.mean(np.abs(teachers[fit_mask] - base[fit_mask])))
    learned_displacement = float(np.mean(np.abs(raw[fit_mask] - base[fit_mask])))
    recovery_ratio = (
        learned_displacement / teacher_displacement
        if teacher_displacement > 1.0e-12
        else 1.0
    )
    top_k = min(10, carriers)
    raw_diagnostics = summarize_action_recovery(
        raw[fit_mask], teachers[fit_mask], base[fit_mask],
        top_k=top_k, suppression_tolerance=1.0e-3,
    )
    projected_diagnostics = summarize_action_recovery(
        projected[fit_mask], teachers[fit_mask], base[fit_mask],
        top_k=top_k, suppression_tolerance=1.0e-3,
    )

    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "feature_mean": feature_stats["mean"],
        "feature_scale": feature_stats["scale"],
        "pose_mean": pose_stats["mean"],
        "pose_scale": pose_stats["scale"],
        "contract": dict(contract),
        "checkpoint_epoch": epochs,
        "deployment_eligible": False,
        "teacher_eligible": False,
        "paper_test_eligible": False,
    }
    torch.save(checkpoint, output / "model.pt")
    np.savez_compressed(
        output / "predictions.npz",
        raw_gates=raw,
        projected_gates=projected,
        raw_residual=residual_values,
        prediction_mask=predict_mask,
        teacher_mask=fit_mask,
        camera_index=np.asarray(camera_index),
        frame_index=np.asarray(frame_index),
        block_ids=np.asarray(block_ids),
        feature_mean=feature_stats["mean"],
        feature_scale=feature_stats["scale"],
        pose_mean=pose_stats["mean"],
        pose_scale=pose_stats["scale"],
        deployment_eligible=np.asarray(0, np.uint8),
        teacher_eligible=np.asarray(0, np.uint8),
        paper_test_eligible=np.asarray(0, np.uint8),
    )
    _write_json(output / "projection_certificates.json", _jsonable(certificates))
    finite_residual = residual_values[predict_mask]
    scale_records = [
        {
            "camera_index": int(np.asarray(camera_index)[segment[0]]),
            "block_id": int(np.asarray(block_ids)[segment[0]]),
            "scales": scales,
        }
        for segment, scales in zip(training_segments, initial_scales)
    ]
    summary = {
        "fold": int(fold),
        "epochs": epochs,
        "checkpoint_epoch": epochs,
        "parameter_count": int(parameter_count),
        "training_segment_count": len(training_segments),
        "training_sample_count": int(fit_mask.sum()),
        "prediction_sample_count": int(predict_mask.sum()),
        "initial_components": initial,
        "final_components": final,
        "segment_initial_scales": scale_records,
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
        "raw_action_diagnostics": raw_diagnostics,
        "projected_action_diagnostics": projected_diagnostics,
        "residual_loss_weight": float(contract["residual_loss_weight"]),
        "fit_loss_improved": bool(final["loss"] < initial["loss"]),
        "fit_teacher_mae_passed": bool(
            fit_mae <= float(contract["maximum_fit_teacher_mae"])
        ),
        "held_teacher_values_accessed": False,
        "held_renderer_values_accessed": False,
        "deployment_eligible": False,
        "teacher_eligible": False,
        "paper_test_eligible": False,
    }
    _write_json(output / "summary.json", summary)
    return summary


def _verify_r4a_sources(contract):
    source_keys = (
        ("source_design", "R4-A design"),
        ("source_r3_policy", "R3 policy"),
        ("source_r3_trainer", "R3 trainer"),
        ("source_r3_auditor", "R3 auditor"),
        ("source_r3_contract", "R3 contract"),
        ("source_r3_fit_entry", "R3 fit entry"),
        ("source_r2_policy", "R2 policy"),
        ("source_r2_runtime", "R2 runtime"),
        ("source_r2_auditor", "R2 auditor"),
        ("source_r2_trainer", "R2 trainer"),
        ("source_r1_1_contract", "R1.1 contract"),
    )
    for key, label in source_keys:
        verify_source_file(REPO_ROOT / contract[key], contract[f"{key}_sha256"], label)
    for key, hashes, label in (
        ("source_nearest_neighbor_dir", "source_nearest_neighbor_prediction_sha256", "NN"),
        (None, "source_r1_3g_witness_prediction_sha256", "R1.3-G witness"),
    ):
        if key is not None:
            paths = [
                str(Path(contract[key]) / f"fold_{fold}/predictions.npz")
                for fold in range(6)
            ]
        else:
            paths = contract["source_r1_3g_witness_predictions"]
        for path, expected in zip(paths, contract[hashes]):
            verify_source_file(REPO_ROOT / path, expected, f"{label} {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train frozen R1.4-VP-R4-A folds.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
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
        raise ValueError("R4-A contract residual loss weight differs")
    _verify_r4a_sources(contract)
    for path, expected, name in (
        (args.probe, contract["source_probe_sha256"], "probe"),
        (args.evidence, contract["source_evidence_sha256"], "evidence"),
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
    if pose_manifest_sha256(args.pose_model_dir, unique_frames, REPO_ROOT) != contract[
        "source_pose_manifest_sha256"
    ]:
        raise ValueError("pose manifest fingerprint mismatch")
    pose_by_frame = dict(zip(map(int, unique_frames), pose_values))
    r1_contract = json.loads(
        (REPO_ROOT / contract["source_r1_1_contract"]).read_text(encoding="utf-8")
    )
    bank = load_part_label_bank(args.a5_bank)
    part_index = PART_NAMES.index(str(contract["part"]))
    all_weight = np.asarray(bank["soft_edit_weights"], np.float64)[:, part_index]
    a5_weight = all_weight[carriers].astype(np.float32)
    runtime = build_runtime_inputs(
        probe=probe,
        feature_names=list(map(str, probe["feature_names"])),
        feature_group=r1_contract["feature_groups"][contract["view_feature_group"]],
        pose_by_frame=pose_by_frame,
        camera_index=cameras,
        frame_index=frames,
        carrier_ids=carriers,
        a5_weight=a5_weight,
        spatial_scale=float(contract["spatial_scale"]),
        depth_scale=float(contract["depth_scale"]),
        edge_log_weight_minimum=float(contract["edge_log_weight_minimum"]),
    )
    with np.load(args.evidence, allow_pickle=False) as source:
        evidence = {key: source[key] for key in source.files}
    if not np.array_equal(evidence["renderer_sequence_camera_index"], cameras):
        raise ValueError("evidence camera manifest differs")
    if not np.array_equal(evidence["renderer_sequence_frame_index"], frames):
        raise ValueError("evidence frame manifest differs")
    objective_streams = _build_streams(evidence, all_weight, carriers, part_index)[
        "objective"
    ]
    blocks = sample_block_ids(cameras, frames, int(contract["temporal_block_count"]))
    segments = pack_camera_block_segments(
        cameras, blocks, frames, frame_stride=int(contract["frame_stride"])
    )
    split = build_canary_splits(
        camera_index=cameras,
        frame_index=frames,
        fit_camera_indices=(0, 1, 2, 3),
        audit_camera_indices=(4, 5, 6, 7),
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
        fit_segments = [segment for segment in segments if np.all(teacher_mask[segment])]
        masked_streams = {
            signal: {
                "base": np.where(teacher_mask, stream["base"], np.nan),
                "point": np.where(teacher_mask[:, None], stream["point"], np.nan),
            }
            for signal, stream in objective_streams.items()
        }
        prediction_mask = np.asarray(split["fit_mask"], bool)
        fold_root = args.output_dir / "training" / f"fold_{fold}"
        summary = train_fold(
            fold=fold,
            features=runtime["features"],
            pose=runtime["pose"],
            adjacency=runtime["adjacency"],
            visibility=runtime["visibility"],
            base_gates=base,
            teacher_gates=teacher_gates,
            renderer_streams=masked_streams,
            teacher_mask=teacher_mask,
            prediction_mask=prediction_mask,
            camera_index=cameras,
            frame_index=frames,
            block_ids=blocks,
            runtime_mass=runtime["runtime_mass"],
            a5_weight=a5_weight,
            contract=contract,
            output_dir=fold_root,
            device=args.device,
        )
        with np.load(fold_root / "predictions.npz", allow_pickle=False) as source:
            projected = np.asarray(source["projected_gates"], np.float64)
        entry = _fit_entry(
            learned_gates=projected,
            teacher_gates=teacher_gates,
            fit_segments=fit_segments,
            streams=objective_streams,
            summary=summary,
            contract=contract,
        )
        entry["fold"] = fold
        _write_json(fold_root / "fit_renderer_entry.json", _jsonable(entry))
        summary["fit_renderer_entry_passed"] = bool(entry["passed"])
        _write_json(fold_root / "summary.json", summary)
        learned.append(summary)
        if not entry["passed"]:
            verdict = classify_fit_entry_failure(fold)
            _write_json(
                args.output_dir / "training/summary.json",
                {"folds": learned, "execution_status": verdict,
                 "failed_fold": fold, "paper_test_eligible": False},
            )
            _write_json(
                args.output_dir / "summary.json",
                {"verdict": verdict, "failed_fold": fold,
                 "fit_renderer_entry": entry, "paper_test_eligible": False},
            )
            if fold == 0:
                return 2
            raise RuntimeError(f"R4-A fit renderer entry failed for fold {fold}")
        freeze_paths.extend(
            [
                fold_root / "model.pt",
                fold_root / "predictions.npz",
                fold_root / "projection_certificates.json",
                fold_root / "summary.json",
                fold_root / "fit_renderer_entry.json",
            ]
        )
    _write_json(
        args.output_dir / "training/summary.json",
        {"folds": learned, "execution_status": "TRAINING_COMPLETED",
         "paper_test_eligible": False},
    )
    frozen = {str(path.relative_to(args.output_dir)): _sha256(path) for path in freeze_paths}
    if len(frozen) != 30:
        raise RuntimeError("R4-A must freeze exactly 30 training artifacts")
    _write_json(
        args.output_dir / "models_frozen.json",
        {"artifacts": frozen,
         "source_nearest_neighbor_prediction_sha256": contract[
             "source_nearest_neighbor_prediction_sha256"
         ], "paper_test_eligible": False},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
