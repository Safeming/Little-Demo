#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.calibrate_evidence_soft_edit_weights import build_footprint_evidence_record
from utils.frozen_semantic_method import (
    frozen_method_fingerprint,
    load_a7_temporal_contract,
)
from utils.part_label_bank import PART_NAMES, load_part_label_bank
from utils.temporal_reliability_calibration import (
    accumulate_temporal_footprint_frame,
    finalize_temporal_footprint_evidence,
)


BOUNDARY_STATE_ENCODING = {
    "0": "not visible",
    "1": "target-dominant",
    "2": "allowed-boundary",
    "3": "outer-dominant",
}
STATE_ARRAY_KEYS = (
    "visible_count",
    "target_mean",
    "target_m2",
    "outer_mean",
    "outer_m2",
    "consecutive_visible_count",
    "target_flicker_sum",
    "outer_flicker_sum",
    "boundary_crossing_count",
    "visibility_transition_count",
    "visibility_pair_count",
    "previous_visible",
    "previous_target_ratio",
    "previous_outer_ratio",
    "previous_boundary_state",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build static A7 temporal footprint reliability evidence."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--a5-bank", required=True, type=Path)
    parser.add_argument("--method-freeze", required=True, type=Path)
    parser.add_argument("--a7-contract", required=True, type=Path)
    parser.add_argument("--cameras", default="c01,c05,c09,c13")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=570)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--parts", default=",".join(PART_NAMES))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-canary-protocol", action="store_true")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--subject", default="")
    parser.add_argument(
        "--explicit-binding-render-preset",
        default="v338_temporal_selector_grow_only_guard",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--footprint-radius-scale", type=float, default=1.0)
    parser.add_argument("--min-footprint-radius", type=int, default=1)
    parser.add_argument("--max-footprint-radius", type=int, default=12)
    return parser.parse_args(argv)


def _parse_csv(value: str, *, field: str) -> list[str]:
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{field} must be a non-empty list without duplicates")
    return values


def validate_requested_protocol(args, contract: dict) -> dict:
    cameras = _parse_csv(args.cameras, field="cameras")
    parts = _parse_csv(args.parts, field="parts")
    invalid_cameras = [camera for camera in cameras if not camera.startswith("c")]
    if invalid_cameras:
        raise ValueError(f"invalid camera names: {invalid_cameras}")
    if "c21" in cameras:
        raise ValueError("c21 must not be used for A7 temporal evidence")
    unknown_parts = [part for part in parts if part not in PART_NAMES]
    if unknown_parts:
        raise ValueError(f"unknown parts: {unknown_parts}")
    if int(args.frame_stride) <= 0 or int(args.frame_end) <= int(args.frame_start):
        raise ValueError("frame range must be non-empty with a positive stride")
    frames = list(
        range(int(args.frame_start), int(args.frame_end), int(args.frame_stride))
    )
    formal = (
        cameras == [str(value) for value in contract["evidence_cameras"]]
        and int(args.frame_start) == int(contract["evidence_frame_start"])
        and int(args.frame_end) == int(contract["evidence_frame_end"])
        and int(args.frame_stride) == int(contract["evidence_frame_stride"])
        and parts == [str(value) for value in contract["parts"]]
    )
    if not formal and not bool(args.allow_canary_protocol):
        raise ValueError(
            "non-formal evidence protocol requires --allow-canary-protocol"
        )
    payload = {
        "cameras": cameras,
        "frames": frames,
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frame_stride": int(args.frame_stride),
        "parts": parts,
        "formal_protocol": bool(formal),
        "sample_count": len(cameras) * len(frames),
    }
    payload["protocol_fingerprint"] = frozen_method_fingerprint(payload)
    return payload


def encode_boundary_state(
    *,
    visible: np.ndarray,
    target_ratio: np.ndarray,
    outer_ratio: np.ndarray,
) -> np.ndarray:
    visible_array = np.asarray(visible, dtype=np.bool_)
    target = np.asarray(target_ratio, dtype=np.float32)
    outer = np.asarray(outer_ratio, dtype=np.float32)
    if visible_array.shape != target.shape or target.shape != outer.shape:
        raise ValueError("boundary state inputs must have matching shapes")
    state = np.zeros(visible_array.shape, dtype=np.int8)
    mixed = visible_array & (target > 0.0) & (outer > 0.0)
    state[mixed] = 2
    state[visible_array & ~mixed & (target >= outer)] = 1
    state[visible_array & ~mixed & (outer > target)] = 3
    return state


def _weighted_mean(
    values: list[np.ndarray], weights: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    total = np.sum(np.stack(weights, axis=0), axis=0, dtype=np.float64)
    numerator = np.sum(
        np.stack(
            [value.astype(np.float64) * weight for value, weight in zip(values, weights)],
            axis=0,
        ),
        axis=0,
        dtype=np.float64,
    )
    mean = np.zeros(total.shape, dtype=np.float64)
    np.divide(numerator, total, out=mean, where=total > 0)
    return mean, total


def combine_camera_evidence(
    camera_evidence: list[tuple[dict[str, np.ndarray], int]],
) -> dict[str, np.ndarray]:
    if not camera_evidence:
        raise ValueError("camera_evidence must be non-empty")
    shape = camera_evidence[0][0]["temporal_visible_count"].shape
    for evidence, frame_count in camera_evidence:
        if evidence["temporal_visible_count"].shape != shape:
            raise ValueError("camera evidence shapes must match")
        if int(frame_count) <= 0:
            raise ValueError("camera frame_count must be positive")

    visible_weights = [
        evidence["temporal_visible_count"].astype(np.float64)
        for evidence, _ in camera_evidence
    ]
    pair_weights = [
        evidence["temporal_consecutive_visible_count"].astype(np.float64)
        for evidence, _ in camera_evidence
    ]
    target_mean, visible_total = _weighted_mean(
        [evidence["temporal_target_ratio_mean"] for evidence, _ in camera_evidence],
        visible_weights,
    )
    outer_mean, _ = _weighted_mean(
        [evidence["temporal_outer_ratio_mean"] for evidence, _ in camera_evidence],
        visible_weights,
    )

    def combined_std(mean_key: str, std_key: str, combined_mean: np.ndarray) -> np.ndarray:
        second_moment = np.zeros(shape, dtype=np.float64)
        for (evidence, _), weight in zip(camera_evidence, visible_weights):
            mean = evidence[mean_key].astype(np.float64)
            std = evidence[std_key].astype(np.float64)
            second_moment += weight * (std * std + mean * mean)
        np.divide(
            second_moment,
            visible_total,
            out=second_moment,
            where=visible_total > 0,
        )
        variance = np.maximum(second_moment - combined_mean * combined_mean, 0.0)
        return np.sqrt(variance).astype(np.float32)

    def pair_weighted(key: str) -> np.ndarray:
        value, _ = _weighted_mean(
            [evidence[key] for evidence, _ in camera_evidence], pair_weights
        )
        return value.astype(np.float32)

    transition_weights = [
        np.full(shape, max(0, int(frame_count) - 1), dtype=np.float64)
        for _, frame_count in camera_evidence
    ]
    transition_rate, _ = _weighted_mean(
        [
            evidence["temporal_visibility_transition_rate"]
            for evidence, _ in camera_evidence
        ],
        transition_weights,
    )
    return {
        "temporal_visible_count": np.sum(
            np.stack(visible_weights, axis=0), axis=0, dtype=np.float64
        ).astype(np.int32),
        "temporal_consecutive_visible_count": np.sum(
            np.stack(pair_weights, axis=0), axis=0, dtype=np.float64
        ).astype(np.int32),
        "temporal_target_ratio_mean": target_mean.astype(np.float32),
        "temporal_target_ratio_std": combined_std(
            "temporal_target_ratio_mean",
            "temporal_target_ratio_std",
            target_mean,
        ),
        "temporal_target_flicker": pair_weighted("temporal_target_flicker"),
        "temporal_outer_ratio_mean": outer_mean.astype(np.float32),
        "temporal_outer_ratio_std": combined_std(
            "temporal_outer_ratio_mean",
            "temporal_outer_ratio_std",
            outer_mean,
        ),
        "temporal_outer_flicker": pair_weighted("temporal_outer_flicker"),
        "temporal_boundary_crossing_rate": pair_weighted(
            "temporal_boundary_crossing_rate"
        ),
        "temporal_visibility_transition_rate": transition_rate.astype(np.float32),
    }


def validate_resume_manifest(actual: dict, expected: dict) -> None:
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            raise ValueError(
                f"resume manifest {key} mismatch: "
                f"expected {expected_value}, got {actual.get(key)}"
            )


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _payload_fingerprint(arrays: dict[str, np.ndarray]) -> str:
    excluded = {"output_fingerprint", "generated_at_utc", "command"}
    digest = hashlib.sha256()
    for key in sorted(name for name in arrays if name not in excluded):
        array = np.asarray(arrays[key])
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        if array.dtype.kind in ("U", "S"):
            digest.update(
                json.dumps(array.tolist(), ensure_ascii=True, sort_keys=True).encode(
                    "utf-8"
                )
            )
        else:
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _checkpoint_paths(output: Path) -> tuple[Path, Path]:
    return Path(str(output) + ".ledger.json"), Path(str(output) + ".partial.npz")


def _save_resume_checkpoint(
    output: Path,
    *,
    states: dict[str, dict],
    completed: set[tuple[str, int]],
    manifest: dict,
) -> None:
    ledger_path, partial_path = _checkpoint_paths(output)
    ledger = {
        **manifest,
        "completed": [[camera, frame] for camera, frame in sorted(completed)],
        "complete": False,
    }
    arrays = {
        "resume_manifest_json": np.array(json.dumps(manifest, sort_keys=True)),
        "completed_json": np.array(json.dumps(ledger["completed"])),
        "camera_metadata_json": np.array(
            json.dumps(
                {
                    camera: {
                        "shape": list(state["shape"]),
                        "last_frame_index": int(state["last_frame_index"]),
                    }
                    for camera, state in states.items()
                },
                sort_keys=True,
            )
        ),
    }
    for camera, state in states.items():
        for key in STATE_ARRAY_KEYS:
            value = state.get(key)
            if value is not None:
                arrays[f"{camera}__{key}"] = np.asarray(value)
    temporary = Path(str(partial_path) + ".tmp.npz")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(temporary, **arrays)
    temporary.replace(partial_path)
    _atomic_write_json(ledger_path, ledger)


def _load_resume_checkpoint(
    output: Path, expected_manifest: dict
) -> tuple[dict[str, dict], set[tuple[str, int]]]:
    ledger_path, partial_path = _checkpoint_paths(output)
    if not ledger_path.exists() or not partial_path.exists():
        raise FileNotFoundError("--resume requires both ledger and partial state files")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    validate_resume_manifest(ledger, expected_manifest)
    with np.load(partial_path, allow_pickle=False) as data:
        saved_manifest = json.loads(str(data["resume_manifest_json"]))
        validate_resume_manifest(saved_manifest, expected_manifest)
        completed = {
            (str(camera), int(frame))
            for camera, frame in json.loads(str(data["completed_json"]))
        }
        metadata = json.loads(str(data["camera_metadata_json"]))
        states = {}
        for camera, camera_meta in metadata.items():
            state = {
                "shape": tuple(int(value) for value in camera_meta["shape"]),
                "last_frame_index": int(camera_meta["last_frame_index"]),
            }
            for key in STATE_ARRAY_KEYS:
                array_key = f"{camera}__{key}"
                state[key] = data[array_key] if array_key in data.files else None
            states[camera] = state
    return states, completed


def _list_override(values) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _build_config(args, protocol: dict):
    from omegaconf import OmegaConf
    from utils.adopted_geometry import apply_explicit_binding_render_preset

    camera_ids = [int(camera[1:]) for camera in protocol["cameras"]]
    config = OmegaConf.load(args.config.resolve())
    overrides = [
        "mode=test",
        f"load_ckpt={args.checkpoint.resolve()}",
        "dataset.preload=false",
        "dataset.test_mode=view",
        f"dataset.test_views.view={_list_override(camera_ids)}",
        "dataset.test_frames.view="
        f"{_list_override([protocol['frame_start'], protocol['frame_end'], protocol['frame_stride']])}",
        "dataset.parsing_prior.enable=true",
        "dataset.parsing_prior.roi_enable=false",
        "dataset.parsing_prior.use_direct_parser_labels=true",
        f"exp_dir={args.output.resolve().parent}",
        "wandb_disable=true",
        f"explicit_binding_render_preset={args.explicit_binding_render_preset}",
    ]
    if args.dataset_root:
        overrides.append(f"dataset.root_dir={args.dataset_root}")
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    OmegaConf.set_struct(config, False)
    if args.subject:
        subject = str(args.subject)
        config.dataset.subject = (
            subject if subject.startswith("CoreView_") else f"CoreView_{subject}"
        )
    config.suffix = "test-view"
    apply_explicit_binding_render_preset(config, repo_root=REPO_ROOT)
    return config


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _view_masks(view) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    from tools.render_semantic_temporal_stability import extract_compact_masks

    compact = getattr(view, "parsing_compact_masks", None)
    names = getattr(view, "parsing_compact_class_names", None)
    if compact is None or names is None:
        raise ValueError(f"{getattr(view, 'image_name', 'view')} lacks compact parser masks")
    foreground = _as_numpy(view.original_mask).astype(np.float32, copy=False)
    if foreground.ndim == 3:
        foreground = foreground[0]
    parsing_valid = getattr(view, "parsing_valid_mask", None)
    valid = (
        _as_numpy(parsing_valid).astype(np.float32, copy=False)
        if parsing_valid is not None
        else foreground
    )
    if valid.ndim == 3:
        valid = valid[0]
    part_masks = extract_compact_masks(compact, names, valid)
    return part_masks, foreground, valid


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _save_final_evidence(
    output: Path,
    *,
    evidence: dict[str, np.ndarray],
    point_count: int,
    protocol: dict,
    manifest: dict,
) -> str:
    arrays = {key: np.asarray(value) for key, value in evidence.items()}
    arrays.update(
        {
            "schema_version": np.array(1, dtype=np.int32),
            "point_count": np.array(point_count, dtype=np.int64),
            "part_names": np.asarray(PART_NAMES, dtype="U16"),
            "cameras": np.asarray(protocol["cameras"], dtype="U3"),
            "parts": np.asarray(protocol["parts"], dtype="U16"),
            "frame_start": np.array(protocol["frame_start"], dtype=np.int64),
            "frame_end": np.array(protocol["frame_end"], dtype=np.int64),
            "frame_stride": np.array(protocol["frame_stride"], dtype=np.int64),
            "formal_protocol": np.array(protocol["formal_protocol"], dtype=np.uint8),
            "boundary_state_encoding": np.array(
                json.dumps(BOUNDARY_STATE_ENCODING, sort_keys=True)
            ),
            "git_commit": np.array(_git_commit()),
            "generated_at_utc": np.array(datetime.now(timezone.utc).isoformat()),
            "command": np.array(" ".join(sys.argv)),
        }
    )
    for key, value in manifest.items():
        arrays[key] = np.array(value)
    fingerprint = _payload_fingerprint(arrays)
    arrays["output_fingerprint"] = np.array(fingerprint)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)
    with np.load(output, allow_pickle=False) as loaded:
        reloaded = {key: loaded[key] for key in loaded.files}
    if str(reloaded["output_fingerprint"]) != _payload_fingerprint(reloaded):
        raise ValueError("evidence output_fingerprint changed after save/reload")
    return fingerprint


def run_evidence_build(args, protocol: dict, contract: dict, manifest: dict) -> dict:
    import torch
    from gaussian_renderer import rasterize_gaussians
    from scene import GaussianModel, Scene
    from tools.semantic_viewer.build_part_label_bank import (
        _find_dataset_index,
        _project_points,
    )

    bank = load_part_label_bank(args.a5_bank)
    point_count = int(np.asarray(bank["point_count"]))
    weights = np.asarray(bank["soft_edit_weights"], dtype=np.float32)
    if weights.shape != (point_count, len(PART_NAMES)):
        raise ValueError("A5 soft_edit_weights shape does not match point_count/parts")

    ledger_path, partial_path = _checkpoint_paths(args.output)
    if args.resume:
        states, completed = _load_resume_checkpoint(args.output, manifest)
    else:
        states, completed = {}, set()
        ledger_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)

    config = _build_config(args, protocol)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    expected_pairs = [
        (camera, frame)
        for camera in protocol["cameras"]
        for frame in protocol["frames"]
    ]
    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, str(args.output.resolve().parent))
        scene.eval()
        iteration = int(scene.load_checkpoint(str(args.checkpoint.resolve())))
        if int(scene.gaussians.get_xyz.shape[0]) != point_count:
            raise ValueError("checkpoint point count does not match A5 bank")
        if len(scene.test_dataset) != len(expected_pairs):
            raise ValueError(
                f"evidence dataset has {len(scene.test_dataset)} samples, "
                f"expected {len(expected_pairs)}"
            )

        for sample_index, (camera, frame) in enumerate(expected_pairs):
            if (camera, frame) in completed:
                continue
            image_name = f"{camera}_f{frame:06d}"
            dataset_index = _find_dataset_index(scene.test_dataset, image_name)
            if dataset_index is None:
                raise RuntimeError(f"{image_name} not present in evidence dataset")
            view = scene.test_dataset[dataset_index]
            part_masks, foreground, valid = _view_masks(view)
            deformed, _, colors_precomp = scene.convert_gaussians(
                view, iteration, compute_loss=False
            )
            if int(deformed.get_xyz.shape[0]) != point_count:
                raise ValueError("deformed Gaussian point count does not match A5 bank")
            pkg = rasterize_gaussians(
                view,
                deformed,
                config.pipeline,
                background,
                colors_precomp=colors_precomp,
                return_opacity=False,
            )
            xy, proj_valid, _ = _project_points(deformed.get_xyz, view)
            visible = np.zeros((point_count, len(PART_NAMES)), dtype=np.bool_)
            target = np.zeros((point_count, len(PART_NAMES)), dtype=np.float32)
            outer = np.zeros((point_count, len(PART_NAMES)), dtype=np.float32)
            boundary = np.zeros((point_count, len(PART_NAMES)), dtype=np.int8)
            for part in protocol["parts"]:
                part_index = PART_NAMES.index(part)
                record = build_footprint_evidence_record(
                    xy=xy,
                    proj_valid=proj_valid,
                    visibility_filter=pkg["visibility_filter"],
                    radii=pkg["radii"],
                    image_size=(int(view.image_width), int(view.image_height)),
                    part_masks=part_masks,
                    foreground_mask=foreground,
                    valid_mask=valid,
                    part_name=part,
                    candidate_mask=None,
                    mask_threshold=float(args.mask_threshold),
                    footprint_radius_scale=float(args.footprint_radius_scale),
                    min_footprint_radius=int(args.min_footprint_radius),
                    max_footprint_radius=int(args.max_footprint_radius),
                )
                visible[:, part_index] = record["observed"]
                target[:, part_index] = record["target_ratio"]
                outer[:, part_index] = record["outer_ratio"]
                boundary[:, part_index] = encode_boundary_state(
                    visible=record["observed"],
                    target_ratio=record["target_ratio"],
                    outer_ratio=record["outer_ratio"],
                )
            state = states.setdefault(camera, {})
            accumulate_temporal_footprint_frame(
                state,
                frame_index=frame,
                visible=visible,
                target_ratio=target,
                outer_ratio=outer,
                boundary_state=boundary,
            )
            completed.add((camera, frame))
            _save_resume_checkpoint(
                args.output,
                states=states,
                completed=completed,
                manifest=manifest,
            )
            print(
                f"[A7 evidence] {sample_index + 1}/{len(expected_pairs)} {image_name}",
                flush=True,
            )
            del pkg, deformed, colors_precomp
            if (sample_index + 1) % 10 == 0:
                torch.cuda.empty_cache()

    camera_results = [
        (finalize_temporal_footprint_evidence(states[camera]), len(protocol["frames"]))
        for camera in protocol["cameras"]
    ]
    evidence = combine_camera_evidence(camera_results)
    for key, value in evidence.items():
        if value.shape != (point_count, len(PART_NAMES)):
            raise ValueError(f"{key} has invalid shape {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{key} contains NaN or Inf")
    fingerprint = _save_final_evidence(
        args.output,
        evidence=evidence,
        point_count=point_count,
        protocol=protocol,
        manifest=manifest,
    )
    ledger_path, partial_path = _checkpoint_paths(args.output)
    ledger = {
        **manifest,
        "completed": [[camera, frame] for camera, frame in sorted(completed)],
        "complete": True,
        "output_fingerprint": fingerprint,
    }
    _atomic_write_json(ledger_path, ledger)
    partial_path.unlink(missing_ok=True)
    return {
        "output": str(args.output.resolve()),
        "output_fingerprint": fingerprint,
        "point_count": point_count,
        "sample_count": protocol["sample_count"],
        "formal_protocol": protocol["formal_protocol"],
        "visible_entry_count": int(
            np.count_nonzero(evidence["temporal_visible_count"])
        ),
        "pair_entry_count": int(
            np.count_nonzero(evidence["temporal_consecutive_visible_count"])
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contract = load_a7_temporal_contract(args.a7_contract, args.method_freeze)
    protocol = validate_requested_protocol(args, contract)
    for path in (
        args.config,
        args.checkpoint,
        args.a5_bank,
        args.method_freeze,
        args.a7_contract,
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    manifest = {
        "config_sha256": _file_sha256(args.config),
        "checkpoint_sha256": _file_sha256(args.checkpoint),
        "a5_bank_sha256": _file_sha256(args.a5_bank),
        "base_method_freeze_fingerprint": contract[
            "base_method_freeze_fingerprint"
        ],
        "a7_contract_fingerprint": contract["_fingerprint"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
    }
    if args.dry_run:
        report = {
            "dry_run": True,
            "formal_protocol": protocol["formal_protocol"],
            "cameras": protocol["cameras"],
            "frame_start": protocol["frame_start"],
            "frame_end": protocol["frame_end"],
            "frame_stride": protocol["frame_stride"],
            "sample_count": protocol["sample_count"],
            "parts": protocol["parts"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "output": str(args.output.resolve()),
            **manifest,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    summary = run_evidence_build(args, protocol, contract, manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
