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

from tools.build_renderer_aligned_temporal_evidence import _build_config, _file_sha256
from utils.a7c_oracle_capacity import _artifact_fingerprint, load_teacher_artifact
from utils.a7c_renderer_compositor import extract_runtime_probe_features, validate_feature_schema
from utils.part_label_bank import PART_NAMES, load_part_label_bank


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build A7c renderer runtime probes.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--a5-bank", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--subject", default="377")
    parser.add_argument("--explicit-binding-render-preset", default="none")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _save_probe(path: Path, arrays: dict[str, np.ndarray]) -> str:
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    fingerprint = _artifact_fingerprint(payload)
    payload["output_fingerprint"] = np.array(fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)
    with np.load(path, allow_pickle=False) as source:
        reloaded = {key: source[key] for key in source.files}
    if _artifact_fingerprint(reloaded) != str(reloaded["output_fingerprint"]):
        raise ValueError("probe fingerprint mismatch after reload")
    return fingerprint


def run(args, contract):
    import torch

    from gaussian_renderer import rasterize_gaussians
    from scene import GaussianModel, Scene
    from tools.semantic_viewer.build_part_label_bank import _find_dataset_index

    teacher = load_teacher_artifact(args.teacher)
    bank = load_part_label_bank(args.a5_bank)
    carrier_ids = np.asarray(teacher["carrier_ids"], dtype=np.int64)
    lower = np.asarray(bank["soft_edit_weights"], dtype=np.float32)[:, PART_NAMES.index("lower")]
    point_count = lower.size
    if np.any(carrier_ids < 0) or np.any(carrier_ids >= point_count):
        raise ValueError("teacher carrier IDs exceed A5 bank")
    cameras = list(contract["fit_cameras"]) + list(contract["audit_cameras"])
    frames = list(range(contract["frame_start"], contract["frame_end"], contract["frame_stride"]))
    expected = [(camera, frame) for camera in cameras for frame in frames]
    if args.max_samples is not None:
        expected = expected[: int(args.max_samples)]
    config = _build_config(
        args,
        {
            "cameras": cameras,
            "frame_start": contract["frame_start"],
            "frame_end": contract["frame_end"],
            "frame_stride": contract["frame_stride"],
            "frames": frames,
            "parts": ["lower"],
            "formal_protocol": True,
        },
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, str(args.output.resolve().parent))
        scene.eval()
        iteration = int(scene.load_checkpoint(str(args.checkpoint.resolve())))
    if int(scene.gaussians.get_xyz.shape[0]) != point_count:
        raise ValueError("checkpoint point count does not match A5 bank")
    rows = []
    camera_rows = []
    frame_rows = []
    for sample, (camera, frame) in enumerate(expected):
        image_name = f"{camera}_f{frame:06d}"
        dataset_index = _find_dataset_index(scene.test_dataset, image_name)
        if dataset_index is None:
            raise RuntimeError(f"missing probe sample {image_name}")
        view = scene.test_dataset[dataset_index]
        with torch.no_grad():
            deformed, _, colors = scene.convert_gaussians(view, iteration, compute_loss=False)
            pkg = rasterize_gaussians(
                view, deformed, config.pipeline, background,
                colors_precomp=colors, return_opacity=False,
            )
        means = deformed.get_xyz[:point_count].detach().float().cpu().numpy()
        opacity = deformed.get_opacity[:point_count].detach().float().cpu().numpy().reshape(-1)
        feature = extract_runtime_probe_features(
            means3d=means,
            world_view_transform=view.world_view_transform.detach().float().cpu().numpy(),
            camera_center=view.camera_center.detach().float().cpu().numpy(),
            visibility=pkg["visibility_filter"][:point_count].detach().cpu().numpy(),
            radii=pkg["radii"][:point_count].detach().float().cpu().numpy(),
            opacity=opacity,
            a5_lower_weight=lower,
            selected_lower=(lower >= float(contract["selection_threshold"])),
        )
        rows.append(feature[carrier_ids].astype(np.float16))
        camera_rows.append(cameras.index(camera))
        frame_rows.append(frame)
        print(f"[A7c probe] {sample + 1}/{len(expected)} {image_name}", flush=True)
        del deformed, colors, pkg
    camera_index = np.asarray(camera_rows, dtype=np.int16)
    frame_index = np.asarray(frame_rows, dtype=np.int32)
    if args.max_samples is None:
        if not np.array_equal(camera_index, teacher["camera_index"]):
            raise ValueError("probe camera order differs from teacher")
        if not np.array_equal(frame_index, teacher["frame_index"]):
            raise ValueError("probe frame order differs from teacher")
    return _save_probe(
        args.output,
        {
            "schema_version": np.array(1, dtype=np.int32),
            "features": np.stack(rows),
            "feature_names": np.asarray(validate_feature_schema(contract["feature_names"]), dtype="U32"),
            "carrier_ids": carrier_ids,
            "camera_index": camera_index,
            "frame_index": frame_index,
            "source_checkpoint_sha256": np.array(_file_sha256(args.checkpoint)),
            "source_a5_bank_sha256": np.array(_file_sha256(args.a5_bank)),
            "source_teacher_fingerprint": np.array(str(teacher["output_fingerprint"])),
            "paper_test_eligible": np.array(0, dtype=np.uint8),
        },
    )


def main(argv=None):
    args = parse_args(argv)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    for path in (args.config, args.checkpoint, args.a5_bank, args.teacher):
        if not path.is_file():
            raise FileNotFoundError(path)
    validate_feature_schema(contract["feature_names"])
    samples = len(contract["fit_cameras"] + contract["audit_cameras"]) * len(
        range(contract["frame_start"], contract["frame_end"], contract["frame_stride"])
    )
    if args.dry_run:
        print(json.dumps({"dry_run": True, "samples": samples, "carriers": int(load_teacher_artifact(args.teacher)["carrier_ids"].size), "features": len(contract["feature_names"])}))
        return 0
    fingerprint = run(args, contract)
    print(json.dumps({"output": str(args.output), "output_fingerprint": fingerprint}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
