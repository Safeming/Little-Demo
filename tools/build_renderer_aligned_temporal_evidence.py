#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_projected_soft_edit_leakage import make_boundary_band
from tools.build_temporal_reliability_evidence import (
    _build_config,
    _file_sha256,
    _git_commit,
    _payload_fingerprint,
    _view_masks,
)
from tools.make_semantic_edit_render_preview import EDIT_COLORS
from utils.frozen_semantic_method import load_a7_temporal_contract
from utils.part_label_bank import PART_NAMES, load_part_label_bank
from utils.renderer_aligned_temporal_evidence import (
    accumulate_renderer_contribution_frame,
    append_renderer_contribution_sequence,
    extract_renderer_region_contributions,
    finalize_renderer_contribution_evidence,
    finalize_renderer_contribution_sequence,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build renderer-aligned A7 temporal contribution evidence."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--a5-bank", required=True, type=Path)
    parser.add_argument("--method-freeze", required=True, type=Path)
    parser.add_argument("--a7-contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--subject", default="377")
    parser.add_argument("--explicit-binding-render-preset", default="none")
    return parser.parse_args(argv)


def _reset_camera_adjacency(state: dict) -> None:
    state["last_frame_index"] = None
    state["previous_visible"] = None
    state["previous_boundary_state"] = None
    for signal in ("target", "outer", "boundary"):
        state[f"previous_{signal}"] = None


def _save_evidence(
    path: Path,
    *,
    evidence: dict[str, np.ndarray],
    contract: dict,
    checkpoint: Path,
    a5_bank: Path,
    sample_count: int,
) -> str:
    arrays = {key: np.asarray(value) for key, value in evidence.items()}
    arrays.update(
        {
            "schema_version": np.array(
                4 if "renderer_selection_target_contribution_sequence" in arrays else 3,
                dtype=np.int32,
            ),
            "point_count": np.array(evidence["temporal_visible_count"].shape[0], dtype=np.int64),
            "part_names": np.asarray(PART_NAMES, dtype="U16"),
            "cameras": np.asarray(contract["evidence_cameras"], dtype="U3"),
            "frame_start": np.array(contract["evidence_frame_start"], dtype=np.int64),
            "frame_end": np.array(contract["evidence_frame_end"], dtype=np.int64),
            "frame_stride": np.array(contract["evidence_frame_stride"], dtype=np.int64),
            "formal_protocol": np.array(1, dtype=np.uint8),
            "sample_count": np.array(sample_count, dtype=np.int64),
            "evidence_mode": np.array(contract["evidence_mode"]),
            "renderer_attribution": np.array(
                contract.get("renderer_attribution", "colors_gradient")
            ),
            "a7_contract_fingerprint": np.array(contract["_fingerprint"]),
            "base_method_freeze_fingerprint": np.array(
                contract["base_method_freeze_fingerprint"]
            ),
            "checkpoint_sha256": np.array(_file_sha256(checkpoint)),
            "a5_bank_sha256": np.array(_file_sha256(a5_bank)),
            "git_commit": np.array(_git_commit()),
            "generated_at_utc": np.array(datetime.now(timezone.utc).isoformat()),
            "command": np.array(" ".join(sys.argv)),
        }
    )
    arrays["protocol_fingerprint"] = np.array(
        _payload_fingerprint(
            {
                key: arrays[key]
                for key in (
                    "cameras",
                    "frame_start",
                    "frame_end",
                    "frame_stride",
                    "part_names",
                    "a7_contract_fingerprint",
                )
            }
        )
    )
    fingerprint = _payload_fingerprint(arrays)
    arrays["output_fingerprint"] = np.array(fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    with np.load(path, allow_pickle=False) as loaded:
        reloaded = {key: loaded[key] for key in loaded.files}
    if str(reloaded["output_fingerprint"]) != _payload_fingerprint(reloaded):
        raise ValueError("renderer evidence fingerprint changed after reload")
    return fingerprint


def run(args: argparse.Namespace, contract: dict) -> dict:
    import torch

    from gaussian_renderer import rasterize_gaussians
    from scene import GaussianModel, Scene
    from tools.semantic_viewer.build_part_label_bank import _find_dataset_index

    bank = load_part_label_bank(args.a5_bank)
    point_count = int(np.asarray(bank["point_count"]))
    config = _build_config(
        args,
        {
            "cameras": list(contract["evidence_cameras"]),
            "frame_start": int(contract["evidence_frame_start"]),
            "frame_end": int(contract["evidence_frame_end"]),
            "frame_stride": int(contract["evidence_frame_stride"]),
            "frames": list(
                range(
                    int(contract["evidence_frame_start"]),
                    int(contract["evidence_frame_end"]),
                    int(contract["evidence_frame_stride"]),
                )
            ),
            "parts": list(contract["parts"]),
            "formal_protocol": True,
        },
    )
    frames = list(
        range(
            int(contract["evidence_frame_start"]),
            int(contract["evidence_frame_end"]),
            int(contract["evidence_frame_stride"]),
        )
    )
    expected = [
        (camera, frame) for camera in contract["evidence_cameras"] for frame in frames
    ]
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    state: dict = {}
    sequence_state: dict = {}
    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, str(args.output.resolve().parent))
        scene.eval()
        iteration = int(scene.load_checkpoint(str(args.checkpoint.resolve())))
    if int(scene.gaussians.get_xyz.shape[0]) != point_count:
        raise ValueError("checkpoint point count does not match A5 bank")
    if len(scene.test_dataset) != len(expected):
        raise ValueError(f"renderer evidence dataset has {len(scene.test_dataset)} samples, expected {len(expected)}")

    previous_camera = None
    for sample_index, (camera, frame) in enumerate(expected):
        if previous_camera is not None and camera != previous_camera:
            _reset_camera_adjacency(state)
        previous_camera = camera
        image_name = f"{camera}_f{frame:06d}"
        dataset_index = _find_dataset_index(scene.test_dataset, image_name)
        if dataset_index is None:
            raise RuntimeError(f"{image_name} not present in renderer evidence dataset")
        view = scene.test_dataset[dataset_index]
        part_masks, _foreground, valid = _view_masks(view)
        with torch.no_grad():
            deformed, _, base_colors = scene.convert_gaussians(
                view, iteration, compute_loss=False
            )
        attribution_colors = torch.ones(
            (point_count, 3),
            device="cuda",
            dtype=base_colors.dtype,
            requires_grad=True,
        )
        render_pkg = rasterize_gaussians(
            view,
            deformed,
            config.pipeline,
            background,
            colors_precomp=attribution_colors,
            return_opacity=False,
        )
        target_values = np.zeros((point_count, len(PART_NAMES)), dtype=np.float32)
        outer_values = np.zeros_like(target_values)
        boundary_values = np.zeros_like(target_values)
        selection_target_values = np.zeros_like(target_values)
        selection_outer_values = np.zeros_like(target_values)
        selection_boundary_values = np.zeros_like(target_values)
        target_pixel_counts = np.zeros((len(PART_NAMES),), dtype=np.float32)
        valid_mask = np.asarray(valid, dtype=np.float32) >= 0.5
        for part_offset, part in enumerate(contract["parts"]):
            part_index = PART_NAMES.index(part)
            target_mask = np.asarray(part_masks[part], dtype=np.float32) >= 0.5
            target_pixel_counts[part_index] = float(np.count_nonzero(target_mask & valid_mask))
            boundary_mask = make_boundary_band(
                target_mask.astype(np.float32),
                radius=int(contract["renderer_boundary_radius"]),
                threshold=0.5,
            ) & (~target_mask) & valid_mask
            target_rgb = torch.tensor(
                np.asarray(EDIT_COLORS[part], dtype=np.float32) / 255.0,
                device=base_colors.device,
                dtype=base_colors.dtype,
            )
            sensitivity = torch.mean(torch.abs(target_rgb[None, :] - base_colors.detach()), dim=1)
            contributions = extract_renderer_region_contributions(
                rendered=render_pkg["render"],
                attribution_colors=attribution_colors,
                target_mask=torch.from_numpy(target_mask.astype(np.float32)).to("cuda"),
                valid_mask=torch.from_numpy(valid_mask.astype(np.float32)).to("cuda"),
                boundary_mask=torch.from_numpy(boundary_mask.astype(np.float32)).to("cuda"),
                edit_sensitivity=sensitivity,
                retain_graph=part_offset + 1 < len(contract["parts"]),
            )
            target_values[:, part_index] = contributions["target"].detach().cpu().numpy()
            outer_values[:, part_index] = contributions["outer"].detach().cpu().numpy()
            boundary_values[:, part_index] = contributions["boundary"].detach().cpu().numpy()
            selection_target_values[:, part_index] = contributions[
                "selection_target"
            ].detach().cpu().numpy()
            selection_outer_values[:, part_index] = contributions[
                "selection_outer"
            ].detach().cpu().numpy()
            selection_boundary_values[:, part_index] = contributions[
                "selection_boundary"
            ].detach().cpu().numpy()
        accumulate_renderer_contribution_frame(
            state,
            frame_index=frame,
            target_contribution=target_values,
            outer_contribution=outer_values,
            boundary_contribution=boundary_values,
            visibility_epsilon=float(contract["renderer_contribution_epsilon"]),
        )
        append_renderer_contribution_sequence(
            sequence_state,
            camera_index=list(contract["evidence_cameras"]).index(camera),
            frame_index=frame,
            target_contribution=target_values,
            outer_contribution=outer_values,
            boundary_contribution=boundary_values,
            selection_target_contribution=selection_target_values,
            selection_outer_contribution=selection_outer_values,
            selection_boundary_contribution=selection_boundary_values,
            target_pixel_count=target_pixel_counts,
        )
        print(f"[A7 renderer evidence] {sample_index + 1}/{len(expected)} {image_name}", flush=True)
        del render_pkg, attribution_colors, deformed, base_colors
        if (sample_index + 1) % 10 == 0:
            torch.cuda.empty_cache()

    evidence = {
        **finalize_renderer_contribution_evidence(state),
        **finalize_renderer_contribution_sequence(sequence_state),
    }
    fingerprint = _save_evidence(
        args.output,
        evidence=evidence,
        contract=contract,
        checkpoint=args.checkpoint,
        a5_bank=args.a5_bank,
        sample_count=len(expected),
    )
    return {
        "output": str(args.output.resolve()),
        "sample_count": len(expected),
        "point_count": point_count,
        "output_fingerprint": fingerprint,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contract = load_a7_temporal_contract(args.a7_contract, args.method_freeze)
    if str(args.subject) != str(contract["subject"]):
        raise ValueError("subject does not match renderer-aligned A7 contract")
    if args.dry_run:
        frames = len(
            range(
                int(contract["evidence_frame_start"]),
                int(contract["evidence_frame_end"]),
                int(contract["evidence_frame_stride"]),
            )
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "samples": len(contract["evidence_cameras"]) * frames,
                    "backward_calls_per_sample": len(contract["parts"]),
                    "output": str(args.output.resolve()),
                    "frozen_parts": contract["frozen_parts"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(json.dumps(run(args, contract), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
