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
from tools.train_a7c_r1_4vp_r3_crw import _evaluate_segment_gains
from utils.a7c_r1_4vp_r2_runtime import (
    ViewPoseResidualCompositor,
    apply_normalization,
    build_runtime_inputs,
    fit_normalization,
    load_pose_rotation_6d,
    pack_camera_block_segments,
    pose_manifest_sha256,
)
from utils.a7c_r1_4vp_r4a import summarize_action_recovery
from utils.a7c_r1_4vp_r4b0 import (
    evaluate_fit_projected_entry,
    exact_projected_straight_through,
    freeze_global_median_scales,
    projection_aware_components,
    projection_aware_loss,
    projection_diagnostics,
    run_gradient_observability_preflight,
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
    mass = np.asarray(runtime_mass, dtype=np.float64)
    weights = np.asarray(a5_weight, dtype=np.float64).reshape(-1)
    if values.ndim != 3:
        raise ValueError("features must have shape [samples, carriers, channels]")
    samples, carriers, channels = values.shape
    if poses.shape != (samples, 36) or base.shape != (samples, carriers):
        raise ValueError("training tensors do not align")
    if teachers.shape != base.shape or mass.shape != base.shape:
        raise ValueError("teacher and runtime-mass tensors must match base gates")
    if weights.shape != (carriers,):
        raise ValueError("A5 weights must match the carrier dimension")
    if fit_mask.shape != (samples,) or predict_mask.shape != fit_mask.shape:
        raise ValueError("training masks do not align")
    if not np.any(fit_mask) or not np.isfinite(teachers[fit_mask]).all():
        raise ValueError("fit teacher values must be finite")
    if np.isfinite(teachers[~fit_mask]).any():
        raise ValueError("held teacher values must remain NaN")
    streams = _validate_renderer_streams(renderer_streams, fit_mask, samples, carriers)

    feature_stats = fit_normalization(values, fit_mask)
    pose_stats = fit_normalization(poses, fit_mask)
    normalized_features = apply_normalization(values, feature_stats).astype(np.float32)
    normalized_pose = apply_normalization(poses, pose_stats).astype(np.float32)
    segments = pack_camera_block_segments(
        camera_index, block_ids, frame_index,
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
    formal_signature = channels == 49 and all(
        int(contract[key]) == 16
        for key in (
            "view_embedding_dimension", "pose_embedding_dimension",
            "gru_hidden_dimension",
        )
    )
    if formal_signature and parameter_count != int(contract["expected_parameter_count"]):
        raise ValueError("R4-B0 model signature differs from the frozen R4-A model")

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

    def project_model_segment(model_value, indices):
        raw_gates, residual = model_value.predict_with_residual(
            tensors["features"][indices], tensors["pose"][indices],
            tensors["adjacency"][indices], tensors["visibility"][indices],
            tensors["base"][indices],
        )
        deployed, certificate = exact_projected_straight_through(
            raw_gates, mass[indices], weights, contract
        )
        segment_streams = {
            signal: {key: value[indices] for key, value in stream.items()}
            for signal, stream in tensors["streams"].items()
        }
        components = projection_aware_components(
            deployed, raw_gates, tensors["teacher"][indices],
            tensors["base"][indices], segment_streams,
            trajectory_delta=float(contract["renderer_trajectory_huber_delta"]),
            gain_delta=float(contract["gain_huber_delta"]),
            target_delta=float(contract["target_response_huber_delta"]),
            gate_delta=float(contract["gate_huber_delta"]),
            temporal_delta=float(contract["temporal_huber_delta"]),
            temporal_weight=float(contract["temporal_loss_weight"]),
            projection_scale=float(contract["projection_consistency_scale"]),
            epsilon=float(contract["renderer_reconstruction_epsilon"]),
        )
        return raw_gates, deployed, residual, components, certificate

    scale_names = tuple(contract["scale_component_names"])
    with torch.no_grad():
        scale_rows = []
        for segment in training_segments:
            _, _, _, components, _ = project_model_segment(model, segment)
            scale_rows.append({name: components[name] for name in scale_names})
        initial_scales = freeze_global_median_scales(
            scale_rows, minimum=float(contract["initial_scale_minimum"])
        )

    def segment_loss(model_value, indices):
        raw_gates, deployed, residual, components, certificate = (
            project_model_segment(model_value, indices)
        )
        total = projection_aware_loss(
            components, initial_scales, residual,
            residual_weight=float(contract["residual_loss_weight"]),
        )
        row = {**components, **total}
        return row, raw_gates, deployed, certificate

    def aggregate_components(model_value):
        rows, certificates = [], []
        for segment in training_segments:
            row, _, _, certificate = segment_loss(model_value, segment)
            rows.append(row)
            certificates.append(certificate)
        aggregate = {
            key: torch.stack([row[key] for row in rows]).mean()
            for key in rows[0]
        }
        return aggregate, certificates

    def certificates_pass(certificates):
        tolerance = float(contract["solver_residual_tolerance"])
        return bool(certificates) and all(
            int(row["stage_one_status"]) == 0
            and int(row["stage_two_status"]) == 0
            and float(row["maximum_primal_violation"]) <= tolerance
            for row in certificates
        )

    with torch.no_grad():
        initial_tensors, _ = aggregate_components(model)
        initial = {
            key: float(value.detach().cpu())
            for key, value in initial_tensors.items()
        }

    def observability_closure(model_value):
        aggregate, certificates = aggregate_components(model_value)
        return {
            "loss": aggregate["loss"],
            "components": aggregate,
            "projection_certificates_passed": certificates_pass(certificates),
        }

    observability = run_gradient_observability_preflight(
        model, observability_closure,
        teacher_values=teachers,
        renderer_streams=streams,
        fit_mask=fit_mask,
        learning_rate=float(contract["learning_rate"]),
        weight_decay=float(contract["weight_decay"]),
        minimum_gradient_norm=float(contract["minimum_observability_gradient_norm"]),
        step_count=int(contract["observability_step_count"]),
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "observability.json", _jsonable(observability))
    if not observability["passed"]:
        summary = {
            "fold": int(fold),
            "execution_status": str(contract["observability_negative_status"]),
            "observability": observability,
            "held_teacher_values_accessed": False,
            "held_renderer_values_accessed": False,
            "deployment_eligible": False,
            "teacher_eligible": False,
            "paper_test_eligible": False,
        }
        _write_json(output / "summary.json", _jsonable(summary))
        return summary

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(contract["learning_rate"]),
        weight_decay=float(contract["weight_decay"]),
    )
    maximum_gradient_norm = 0.0
    epochs = int(contract["training_epochs"])
    for epoch in range(epochs):
        for segment in training_segments:
            optimizer.zero_grad(set_to_none=True)
            components, _, _, _ = segment_loss(model, segment)
            components["loss"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(contract["gradient_clip_norm"])
            )
            maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
            optimizer.step()
        if epoch + 1 in {1, epochs}:
            print(json.dumps(
                {"fold": int(fold), "epoch": epoch + 1, "epochs": epochs}
            ), flush=True)

    with torch.no_grad():
        final_tensors, _ = aggregate_components(model)
        final = {
            key: float(value.detach().cpu())
            for key, value in final_tensors.items()
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

    exact, certificates = _project_segments(
        raw, predict_mask, mass, weights, camera_index, frame_index,
        block_ids, contract,
    )
    projection_passed = certificates_pass(certificates)
    fit_mae = float(np.mean(np.abs(exact[fit_mask] - teachers[fit_mask])))
    temporal_errors = [
        np.abs(np.diff(exact[segment], axis=0) - np.diff(teachers[segment], axis=0))
        for segment in training_segments
    ]
    temporal_mae = float(np.mean(np.concatenate(
        [row.reshape(-1) for row in temporal_errors]
    )))
    learned_gains = _evaluate_segment_gains(streams, exact, training_segments)
    teacher_gains = _evaluate_segment_gains(streams, teachers, training_segments)
    outer_recovery = float(np.mean(learned_gains["outer"]) / np.mean(
        teacher_gains["outer"]
    ))
    boundary_recovery = float(np.mean(learned_gains["boundary"]) / np.mean(
        teacher_gains["boundary"]
    ))
    raw_action = summarize_action_recovery(
        raw[fit_mask], teachers[fit_mask], base[fit_mask],
        top_k=min(10, carriers), suppression_tolerance=1.0e-3,
    )
    exact_action = summarize_action_recovery(
        exact[fit_mask], teachers[fit_mask], base[fit_mask],
        top_k=min(10, carriers), suppression_tolerance=1.0e-3,
    )
    projection_stats = projection_diagnostics(
        raw[fit_mask], exact[fit_mask],
        changed_threshold=float(contract["projection_changed_threshold"]),
    )
    teacher_displacement = float(np.mean(np.abs(teachers[fit_mask] - base[fit_mask])))
    learned_displacement = float(np.mean(np.abs(exact[fit_mask] - base[fit_mask])))
    finite_residual = residual_values[predict_mask]

    checkpoint = {
        "state_dict": model.state_dict(),
        "feature_mean": feature_stats["mean"],
        "feature_scale": feature_stats["scale"],
        "pose_mean": pose_stats["mean"],
        "pose_scale": pose_stats["scale"],
        "global_initial_scales": initial_scales,
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
        exact_gates=exact,
        projected_gates=exact,
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
    summary = {
        "fold": int(fold),
        "epochs": epochs,
        "checkpoint_epoch": epochs,
        "parameter_count": int(parameter_count),
        "optimizer_signature": {
            "name": str(contract["optimizer"]),
            "learning_rate": float(contract["learning_rate"]),
            "weight_decay": float(contract["weight_decay"]),
        },
        "training_segment_count": len(training_segments),
        "training_sample_count": int(fit_mask.sum()),
        "prediction_sample_count": int(predict_mask.sum()),
        "global_initial_scales": initial_scales,
        "initial_components": initial,
        "final_components": final,
        "observability": observability,
        "maximum_gradient_norm_before_clip": maximum_gradient_norm,
        "raw_minimum_gate": float(np.nanmin(raw)),
        "raw_maximum_gate": float(np.nanmax(raw)),
        "exact_minimum_gate": float(np.nanmin(exact)),
        "exact_maximum_gate": float(np.nanmax(exact)),
        "fit_projected_teacher_mae": fit_mae,
        "fit_temporal_difference_mae": temporal_mae,
        "fit_outer_recovery": outer_recovery,
        "fit_boundary_recovery": boundary_recovery,
        "fit_outer_positive_segment_fraction": float(
            np.mean(np.asarray(learned_gains["outer"]) > 0.0)
        ),
        "fit_boundary_positive_segment_fraction": float(
            np.mean(np.asarray(learned_gains["boundary"]) > 0.0)
        ),
        "learned_outer_gains": learned_gains["outer"],
        "teacher_outer_gains": teacher_gains["outer"],
        "learned_boundary_gains": learned_gains["boundary"],
        "teacher_boundary_gains": teacher_gains["boundary"],
        "latent_residual_mean": float(np.mean(np.abs(finite_residual))),
        "latent_residual_maximum": float(np.max(np.abs(finite_residual))),
        "base_to_teacher_mean_displacement": teacher_displacement,
        "base_to_learned_mean_displacement": learned_displacement,
        "teacher_displacement_recovery_ratio": (
            learned_displacement / teacher_displacement
            if teacher_displacement > 1.0e-12 else 1.0
        ),
        "raw_action_diagnostics": raw_action,
        "projected_action_diagnostics": exact_action,
        **projection_stats,
        "projection_certificates_passed": projection_passed,
        "residual_loss_weight": float(contract["residual_loss_weight"]),
        "fit_loss_improved": bool(final["loss"] < initial["loss"]),
        "held_teacher_values_accessed": False,
        "held_renderer_values_accessed": False,
        "deployment_eligible": False,
        "teacher_eligible": False,
        "paper_test_eligible": False,
    }
    entry = evaluate_fit_projected_entry(summary, contract)
    summary["fit_projected_entry"] = entry
    summary["fit_projected_entry_passed"] = bool(entry["passed"])
    _write_json(output / "fit_projected_entry.json", _jsonable(entry))
    _write_json(output / "summary.json", _jsonable(summary))
    return summary


def _verify_r4a_sources(contract):
    source_keys = (
        ("source_design", "R4-B0 design"),
        ("source_r4a_contract", "R4-A contract"),
        ("source_r4a_policy", "R4-A policy"),
        ("source_r4a_trainer", "R4-A trainer"),
        ("source_r4a_auditor", "R4-A auditor"),
        ("source_r4a_runner", "R4-A runner"),
        ("source_r4a_fit_entry", "R4-A fit entry"),
        ("source_r4a_fold0_predictions", "R4-A fold-0 predictions"),
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
    parser = argparse.ArgumentParser(
        description="Train frozen R1.4-VP-R4-B0 projection-aware folds."
    )
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
        raise ValueError("R4-B0 contract residual loss weight differs")
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
        learned.append(summary)
        if summary.get("execution_status") == contract["observability_negative_status"]:
            payload = {
                "folds": learned,
                "execution_status": str(contract["observability_negative_status"]),
                "failed_fold": fold,
                "paper_test_eligible": False,
            }
            _write_json(args.output_dir / "training/summary.json", _jsonable(payload))
            _write_json(args.output_dir / "summary.json", _jsonable(payload))
            return 2 if fold == 0 else 1
        entry = dict(summary["fit_projected_entry"])
        entry["fold"] = fold
        _write_json(fold_root / "fit_projected_entry.json", _jsonable(entry))
        summary["fit_projected_entry"] = entry
        _write_json(fold_root / "summary.json", _jsonable(summary))
        if not entry["passed"]:
            verdict = str(contract["fit_negative_status"])
            _write_json(
                args.output_dir / "training/summary.json",
                {"folds": learned, "execution_status": verdict,
                 "failed_fold": fold, "paper_test_eligible": False},
            )
            _write_json(
                args.output_dir / "summary.json",
                {"verdict": verdict, "failed_fold": fold,
                 "fit_projected_entry": entry, "paper_test_eligible": False},
            )
            if fold == 0:
                return 2
            raise RuntimeError(f"R4-B0 fit projected entry failed for fold {fold}")
        freeze_paths.extend(
            [
                fold_root / "model.pt",
                fold_root / "predictions.npz",
                fold_root / "projection_certificates.json",
                fold_root / "observability.json",
                fold_root / "summary.json",
                fold_root / "fit_projected_entry.json",
            ]
        )
    _write_json(
        args.output_dir / "training/summary.json",
        {"folds": learned, "execution_status": "TRAINING_COMPLETED",
         "paper_test_eligible": False},
    )
    frozen = {str(path.relative_to(args.output_dir)): _sha256(path) for path in freeze_paths}
    if len(frozen) != 36:
        raise RuntimeError("R4-B0 must freeze exactly 36 training artifacts")
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
