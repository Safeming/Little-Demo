#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.make_semantic_edit_render_preview import EDIT_COLORS, _tensor_to_rgb_uint8
from tools.render_semantic_real_editing_paper_suite import (
    _validate_bank_points,
    load_frozen_run_config,
    normalize_dataset_subject,
)
from utils.frozen_semantic_method import (
    load_a7_temporal_contract,
    validate_a7_bank_against_contract,
)
from utils.part_label_bank import PART_NAMES, load_part_label_bank
from utils.semantic_real_editing import build_edit_overrides, compute_edit_delta_metrics, resolve_method_weights
from utils.semantic_temporal_stability import compute_screen_selection_metrics


TEMPORAL_METHODS = ("voting", "a5", "a7")
DEFAULT_TEMPORAL_METHODS = ("voting", "a5")
DEFAULT_VIDEO_PARTS = ("upper", "hair", "shoes")


class FFmpegVideoWriter:
    def __init__(self, path: Path, *, fps: int):
        self.path = Path(path)
        self.fps = int(fps)
        self.process = None
        self.frame_shape = None

    def _start(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frame_shape = frame.shape

    def append_data(self, frame: np.ndarray) -> None:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("video frames must have shape [H, W, 3]")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if self.process is None:
            self._start(array)
        if array.shape != self.frame_shape:
            raise ValueError(f"video frame shape changed from {self.frame_shape} to {array.shape}")
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        self.process.stdin.write(np.ascontiguousarray(array).tobytes())

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
        return_code = self.process.wait()
        self.process = None
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path}: {stderr.strip()}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a continuous-frame frozen semantic temporal-stability evaluation."
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--voting-bank", required=True, type=Path)
    parser.add_argument("--a5-bank", required=True, type=Path)
    parser.add_argument("--a7-bank", type=Path, default=None)
    parser.add_argument("--a7-contract", type=Path, default=None)
    parser.add_argument("--loso-config", required=True, type=Path)
    parser.add_argument("--method-freeze", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--camera", type=int, default=21)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=570)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--parts", nargs="+", choices=list(PART_NAMES), default=list(PART_NAMES))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(TEMPORAL_METHODS),
        default=list(DEFAULT_TEMPORAL_METHODS),
    )
    parser.add_argument("--video-parts", nargs="+", choices=list(PART_NAMES), default=list(DEFAULT_VIDEO_PARTS))
    parser.add_argument("--video-fps", type=int, default=25)
    parser.add_argument("--screen-threshold", type=float, default=0.2)
    parser.add_argument("--edit-strength", type=float, default=1.0)
    parser.add_argument("--adaptive-target-retention", type=float, default=0.5)
    parser.add_argument(
        "--adaptive-strength-grid",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 0.75, 1.00],
    )
    parser.add_argument("--boundary-radius", type=int, default=None)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--explicit-binding-render-preset", default="none")
    parser.add_argument("--no-videos", action="store_true")
    return parser.parse_args(argv)


def expected_metric_row_count(
    frame_start: int,
    frame_end: int,
    frame_step: int,
    *,
    part_count: int,
    method_count: int,
) -> int:
    if int(frame_step) <= 0 or int(frame_end) <= int(frame_start):
        raise ValueError("frame range must be non-empty with a positive step")
    return len(range(int(frame_start), int(frame_end), int(frame_step))) * int(part_count) * int(method_count)


def method_weight_fingerprint(weights: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(weights, dtype=np.float32).reshape(-1))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _normalized_adjacent_flicker(values: list[float], support=None) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size <= 1:
        return 0.0
    pair_mask = np.ones(array.size - 1, dtype=bool)
    if support is not None:
        support_array = np.asarray(support, dtype=bool)
        pair_mask = support_array[1:] & support_array[:-1]
    if not np.any(pair_mask):
        return 0.0
    differences = np.abs(np.diff(array))[pair_mask]
    supported_values = array if support is None else array[np.asarray(support, dtype=bool)]
    scale = abs(float(np.mean(supported_values))) if supported_values.size else 0.0
    return float(np.mean(differences) / scale) if scale > 1.0e-8 else 0.0


def summarize_temporal_rows(rows: list[dict]) -> dict:
    grouped = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), str(row["part"])), []).append(row)
    summary = {}
    for (method, part), members in sorted(grouped.items()):
        members.sort(key=lambda row: int(row["frame"]))
        fingerprints = {
            str(row["canonical_selection_fingerprint"]) for row in members
        }
        if len(fingerprints) != 1:
            raise ValueError(f"{method}/{part} changed canonical selection across frames")
        visible = [float(row["selection_mass"]) > 1.0e-8 for row in members]
        areas = np.asarray([float(row["selection_mass"]) for row in members])
        centroid_x = np.asarray(
            [float(row["selection_centroid_x"]) for row in members]
        )
        centroid_y = np.asarray(
            [float(row["selection_centroid_y"]) for row in members]
        )
        area_mean = float(np.mean(areas)) if areas.size else 0.0
        transitions = (
            float(np.mean(np.asarray(visible[1:]) != np.asarray(visible[:-1])))
            if len(visible) > 1
            else 0.0
        )
        summary.setdefault(method, {})[part] = {
            "fixed_strength_outer_flicker": _normalized_adjacent_flicker(
                [float(row["edit_outer_delta_mean"]) for row in members]
            ),
            "fixed_strength_boundary_flicker": _normalized_adjacent_flicker(
                [float(row["edit_boundary_outer_delta_mean"]) for row in members]
            ),
            "adaptive_matched_retention_outer_flicker": _normalized_adjacent_flicker(
                [float(row["adaptive_edit_outer_delta_mean"]) for row in members]
            ),
            "adaptive_matched_retention_boundary_flicker": _normalized_adjacent_flicker(
                [
                    float(row["adaptive_edit_boundary_outer_delta_mean"])
                    for row in members
                ]
            ),
            "visibility_aware_response_flicker": _normalized_adjacent_flicker(
                [float(row["edit_target_delta_mean"]) for row in members], visible
            ),
            "selection_area_cv": (
                float(np.std(areas, ddof=0) / area_mean)
                if areas.size and abs(area_mean) > 1.0e-8
                else 0.0
            ),
            "selection_centroid_std": float(
                np.sqrt(
                    np.std(centroid_x, ddof=0) ** 2
                    + np.std(centroid_y, ddof=0) ** 2
                )
            ),
            "visibility_transition_rate": transitions,
            "adaptive_strength_sequence": [
                float(row["adaptive_strength"]) for row in members
            ],
            "canonical_selection_fingerprint": next(iter(fingerprints)),
        }
    return summary


def select_global_adaptive_metrics(
    rows: list[dict],
    sweeps: dict[tuple[str, str, int], list[tuple[float, dict]]],
    *,
    target_retention: float,
    reference_method: str = "a5",
    reference_strength: float = 1.0,
) -> None:
    parts = sorted({str(row["part"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows})
    row_lookup = {
        (str(row["method"]), str(row["part"]), int(row["frame"])): row
        for row in rows
    }
    for part in parts:
        if reference_method not in methods:
            raise ValueError(f"adaptive reference method is unavailable: {reference_method}")
        reference_keys = sorted(
            key
            for key in sweeps
            if key[0] == reference_method and key[1] == part
        )
        if not reference_keys:
            raise ValueError(f"missing adaptive sweeps for {reference_method}/{part}")

        def metrics_at(key, strength):
            candidates = sweeps[key]
            return min(
                candidates,
                key=lambda item: (abs(float(item[0]) - float(strength)), float(item[0])),
            )[1]

        reference_target = sum(
            float(metrics_at(key, reference_strength)["target_delta_sum"])
            for key in reference_keys
        )
        desired_target = float(target_retention) * reference_target
        for method in methods:
            method_keys = sorted(
                key for key in sweeps if key[0] == method and key[1] == part
            )
            if not method_keys:
                raise ValueError(f"missing adaptive sweeps for {method}/{part}")
            strengths = sorted(
                {float(strength) for strength, _ in sweeps[method_keys[0]]}
            )
            pooled = []
            for strength in strengths:
                response = sum(
                    float(metrics_at(key, strength)["target_delta_sum"])
                    for key in method_keys
                )
                pooled.append((strength, response))
            selected_strength, selected_response = min(
                pooled,
                key=lambda item: (
                    abs(float(item[1]) - desired_target),
                    float(item[0]),
                ),
            )
            for key in method_keys:
                row = row_lookup[key]
                selected_metrics = metrics_at(key, selected_strength)
                row.update(
                    {
                        "adaptive_target_retention": float(target_retention),
                        "adaptive_strength": float(selected_strength),
                        "adaptive_reference_method": str(reference_method),
                        "adaptive_reference_target_response": float(reference_target),
                        "adaptive_desired_target_response": float(desired_target),
                        "adaptive_pooled_target_response": float(selected_response),
                        **{
                            f"adaptive_edit_{name}": value
                            for name, value in selected_metrics.items()
                        },
                    }
                )


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def extract_compact_masks(compact_masks, class_names, valid_mask) -> dict[str, np.ndarray]:
    compact = _as_numpy(compact_masks).astype(np.float32, copy=False)
    valid = _as_numpy(valid_mask).astype(np.float32, copy=False)
    if valid.ndim == 3:
        valid = valid[0]
    names = tuple(str(name) for name in class_names)
    if compact.ndim != 3 or compact.shape[0] != len(names) or compact.shape[1:] != valid.shape:
        raise ValueError("compact parser masks, names, and valid mask have incompatible shapes")
    lookup = {name: compact[index] for index, name in enumerate(names)}
    missing = [part for part in PART_NAMES if part not in lookup]
    if missing:
        raise ValueError(f"missing compact parser parts: {missing}")
    valid_binary = (valid >= 0.5).astype(np.float32)
    return {part: np.clip(lookup[part], 0.0, 1.0) * valid_binary for part in PART_NAMES}


def _labeled_square(image: np.ndarray, label: str) -> np.ndarray:
    square = cv2.resize(np.asarray(image, dtype=np.uint8), (384, 384), interpolation=cv2.INTER_AREA)
    square = square.copy()
    cv2.rectangle(square, (0, 0), (384, 34), (0, 0, 0), thickness=-1)
    cv2.putText(square, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return square


def compose_video_panel(
    base: np.ndarray,
    voting_selection: np.ndarray,
    a5_selection: np.ndarray,
    voting_edit: np.ndarray,
    a5_edit: np.ndarray,
) -> np.ndarray:
    panels = (
        _labeled_square(base, "RGB"),
        _labeled_square(voting_selection, "Voting mask"),
        _labeled_square(a5_selection, "A5 mask"),
        _labeled_square(voting_edit, "Voting edit"),
        _labeled_square(a5_edit, "A5 edit"),
    )
    return np.concatenate(panels, axis=1)


def compose_method_video_panel(
    base: np.ndarray,
    frame_data: dict[str, tuple[np.ndarray, np.ndarray]],
    methods: list[str],
) -> np.ndarray:
    panels = [_labeled_square(base, "RGB")]
    for method in methods:
        selection, edit = frame_data[method]
        label = method.upper()
        panels.append(_labeled_square(selection, f"{label} mask"))
        panels.append(_labeled_square(edit, f"{label} edit"))
    return np.concatenate(panels, axis=1)


def _selection_overlay(base: np.ndarray, selection: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    alpha = np.clip(np.asarray(selection, dtype=np.float32), 0.0, 1.0)[..., None] * 0.72
    tint = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(np.asarray(base, dtype=np.float32) * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _list_override(values) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _build_config(args):
    from omegaconf import OmegaConf
    from utils.adopted_geometry import apply_explicit_binding_render_preset

    config = OmegaConf.load(args.config.resolve())
    overrides = [
        "mode=test",
        f"load_ckpt={args.checkpoint.resolve()}",
        "dataset.preload=false",
        "dataset.test_mode=view",
        f"dataset.test_views.view={_list_override([int(args.camera)])}",
        f"dataset.test_frames.view={_list_override([int(args.frame_start), int(args.frame_end), int(args.frame_step)])}",
        "dataset.parsing_prior.enable=true",
        "dataset.parsing_prior.roi_enable=false",
        "dataset.parsing_prior.use_direct_parser_labels=true",
        f"exp_dir={args.output_dir.resolve()}",
        "wandb_disable=true",
        f"explicit_binding_render_preset={args.explicit_binding_render_preset}",
    ]
    if args.dataset_root:
        overrides.append(f"dataset.root_dir={args.dataset_root}")
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    OmegaConf.set_struct(config, False)
    normalize_dataset_subject(config, str(args.subject))
    config.suffix = "test-view"
    apply_explicit_binding_render_preset(config, repo_root=REPO_ROOT)
    return config


def _write_metrics(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("temporal renderer produced no metric rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_temporal_evaluation(args: argparse.Namespace) -> dict:
    import torch
    from gaussian_renderer import rasterize_gaussians
    from scene import GaussianModel, Scene

    if not 0.0 < float(args.edit_strength) <= 1.0:
        raise ValueError("edit strength must be in (0, 1]")
    if not 0.0 < float(args.screen_threshold) <= 1.0:
        raise ValueError("screen threshold must be in (0, 1]")
    if not 0.0 < float(args.adaptive_target_retention) <= 1.0:
        raise ValueError("adaptive target retention must be in (0, 1]")
    adaptive_strengths = [float(value) for value in args.adaptive_strength_grid]
    if (
        not adaptive_strengths
        or adaptive_strengths != sorted(adaptive_strengths)
        or len(adaptive_strengths) != len(set(adaptive_strengths))
        or any(value <= 0.0 or value > 1.0 for value in adaptive_strengths)
    ):
        raise ValueError("adaptive strength grid must be unique, increasing, and in (0, 1]")
    if "a7" in args.methods and (args.a7_bank is None or args.a7_contract is None):
        raise ValueError("A7 temporal evaluation requires --a7-bank and --a7-contract")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    if not args.no_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    run_config = load_frozen_run_config(
        subject=str(args.subject),
        loso_config=args.loso_config,
        method_freeze=args.method_freeze,
    )
    voting_bank = load_part_label_bank(args.voting_bank)
    a5_bank = load_part_label_bank(args.a5_bank)
    a7_bank = load_part_label_bank(args.a7_bank) if args.a7_bank is not None else None
    a7_provenance = {}
    if a7_bank is not None:
        a7_contract = load_a7_temporal_contract(args.a7_contract, args.method_freeze)
        a7_provenance = validate_a7_bank_against_contract(
            a7_bank,
            contract=a7_contract,
            a5_bank_path=args.a5_bank,
        )
    point_count = _validate_bank_points(
        voting_bank, voting_bank, a5_bank, a7_bank=a7_bank
    )
    config = _build_config(args)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    metric_rows: list[dict] = []
    adaptive_sweeps: dict[tuple[str, str, int], list[tuple[float, dict]]] = {}
    video_paths = {part: video_dir / f"CoreView_{args.subject}_c{int(args.camera):02d}_{part}.mp4" for part in args.video_parts}
    writers = {}
    started = time.time()

    try:
        if not args.no_videos:
            writers = {part: FFmpegVideoWriter(path, fps=int(args.video_fps)) for part, path in video_paths.items()}
        with torch.no_grad():
            gaussians = GaussianModel(config.model.gaussian)
            scene = Scene(config, gaussians, str(output_dir))
            scene.eval()
            iteration = int(scene.load_checkpoint(str(args.checkpoint.resolve())))
            if int(scene.gaussians.get_xyz.shape[0]) != point_count:
                raise ValueError("checkpoint point count does not match semantic banks")
            dataset = scene.test_dataset
            expected_frames = len(range(int(args.frame_start), int(args.frame_end), int(args.frame_step)))
            if len(dataset) != expected_frames:
                raise ValueError(f"continuous dataset has {len(dataset)} frames, expected {expected_frames}")
            boundary_radius = int(args.boundary_radius or run_config["boundary_radius"])
            weights_by_method_part = {
                (method, part): resolve_method_weights(
                    voting_bank,
                    voting_bank,
                    a5_bank,
                    a7_bank=a7_bank,
                    method=method,
                    part=part,
                    threshold=float(run_config["soft_threshold"]),
                )
                for method in args.methods
                for part in args.parts
            }
            weight_fingerprints = {
                key: method_weight_fingerprint(weights)
                for key, weights in weights_by_method_part.items()
            }
            selection_colors = {
                key: torch.from_numpy(np.repeat(weights[:, None], 3, axis=1)).to(device="cuda", dtype=torch.float32)
                for key, weights in weights_by_method_part.items()
            }

            for frame_index in range(len(dataset)):
                view = dataset[frame_index]
                compact = getattr(view, "parsing_compact_masks", None)
                compact_names = getattr(view, "parsing_compact_class_names", None)
                if compact is None or compact_names is None:
                    raise ValueError(f"frame {frame_index} is missing compact parser masks")
                valid_mask = _as_numpy(view.original_mask)
                part_masks = extract_compact_masks(compact, compact_names, valid_mask)
                valid_2d = valid_mask[0] if valid_mask.ndim == 3 else valid_mask

                deformed, _, colors_precomp = scene.convert_gaussians(view, iteration, compute_loss=False)
                if int(deformed.get_xyz.shape[0]) != point_count:
                    raise ValueError("deformed Gaussian point count does not match semantic banks")
                base_pkg = rasterize_gaussians(
                    view,
                    deformed,
                    config.pipeline,
                    background,
                    colors_precomp=colors_precomp,
                    return_opacity=False,
                )
                base_rgb = _tensor_to_rgb_uint8(base_pkg["render"].clamp(0.0, 1.0))
                base_float = base_rgb.astype(np.float32) / 255.0
                base_colors = colors_precomp.detach().float().cpu().numpy()
                base_opacities = deformed.get_opacity.detach().float().cpu().numpy()
                frame_video = {part: {} for part in args.video_parts}

                for part in args.parts:
                    for method in args.methods:
                        key = (method, part)
                        selection_pkg = rasterize_gaussians(
                            view,
                            deformed,
                            config.pipeline,
                            background,
                            colors_precomp=selection_colors[key].to(dtype=colors_precomp.dtype),
                            return_opacity=False,
                        )
                        selection = selection_pkg["render"].mean(dim=0).clamp(0.0, 1.0).detach().cpu().numpy()
                        selection_metrics = compute_screen_selection_metrics(
                            selection,
                            part_masks[part],
                            valid_2d,
                            threshold=float(args.screen_threshold),
                        )
                        target_rgb = np.asarray(EDIT_COLORS[part], dtype=np.float32) / 255.0
                        overrides = build_edit_overrides(
                            base_colors,
                            base_opacities,
                            weights_by_method_part[key],
                            task="recolor",
                            strength=float(args.edit_strength),
                            target_rgb=target_rgb,
                            texture_colors=base_colors,
                        )
                        edited_pkg = rasterize_gaussians(
                            view,
                            deformed,
                            config.pipeline,
                            background,
                            colors_precomp=torch.from_numpy(overrides["colors"]).to(
                                device=colors_precomp.device, dtype=colors_precomp.dtype
                            ),
                            opacities_precomp=torch.from_numpy(overrides["opacities"]).to(
                                device=deformed.get_opacity.device, dtype=deformed.get_opacity.dtype
                            ),
                            return_opacity=False,
                        )
                        edited_rgb = _tensor_to_rgb_uint8(edited_pkg["render"].clamp(0.0, 1.0))
                        edit_metrics = compute_edit_delta_metrics(
                            base_float,
                            edited_rgb.astype(np.float32) / 255.0,
                            part_masks[part],
                            valid_2d,
                            boundary_radius=boundary_radius,
                        )
                        adaptive_candidates = []
                        for adaptive_strength in adaptive_strengths:
                            if abs(adaptive_strength - float(args.edit_strength)) <= 1e-12:
                                adaptive_metrics = edit_metrics
                            else:
                                adaptive_overrides = build_edit_overrides(
                                    base_colors,
                                    base_opacities,
                                    weights_by_method_part[key],
                                    task="recolor",
                                    strength=adaptive_strength,
                                    target_rgb=target_rgb,
                                    texture_colors=base_colors,
                                )
                                adaptive_pkg = rasterize_gaussians(
                                    view,
                                    deformed,
                                    config.pipeline,
                                    background,
                                    colors_precomp=torch.from_numpy(
                                        adaptive_overrides["colors"]
                                    ).to(
                                        device=colors_precomp.device,
                                        dtype=colors_precomp.dtype,
                                    ),
                                    opacities_precomp=torch.from_numpy(
                                        adaptive_overrides["opacities"]
                                    ).to(
                                        device=deformed.get_opacity.device,
                                        dtype=deformed.get_opacity.dtype,
                                    ),
                                    return_opacity=False,
                                )
                                adaptive_rgb = _tensor_to_rgb_uint8(
                                    adaptive_pkg["render"].clamp(0.0, 1.0)
                                )
                                adaptive_metrics = compute_edit_delta_metrics(
                                    base_float,
                                    adaptive_rgb.astype(np.float32) / 255.0,
                                    part_masks[part],
                                    valid_2d,
                                    boundary_radius=boundary_radius,
                                )
                                del adaptive_pkg
                            adaptive_candidates.append(
                                (adaptive_strength, adaptive_metrics)
                            )
                        actual_frame = int(getattr(view, "frame_id", frame_index))
                        adaptive_sweeps[(method, part, actual_frame)] = adaptive_candidates
                        adaptive_strength, adaptive_metrics = adaptive_candidates[0]
                        metric_rows.append(
                            {
                                "subject": str(args.subject),
                                "camera": int(args.camera),
                                "frame": actual_frame,
                                "view": str(getattr(view, "image_name", f"c{int(args.camera):02d}_f{frame_index:06d}")),
                                "part": part,
                                "method": method,
                                "soft_threshold": float(run_config["soft_threshold"]),
                                "screen_threshold": float(args.screen_threshold),
                                "selected_gaussian_count": int(np.sum(weights_by_method_part[key] > 0.0)),
                                "edit_weight_sum": float(np.sum(weights_by_method_part[key])),
                                "canonical_selection_fingerprint": weight_fingerprints[key],
                                "adaptive_strength": float(adaptive_strength),
                                **selection_metrics,
                                **{f"edit_{name}": value for name, value in edit_metrics.items()},
                                **{
                                    f"adaptive_edit_{name}": value
                                    for name, value in adaptive_metrics.items()
                                },
                            }
                        )
                        if part in frame_video:
                            overlay_color = {
                                "voting": (65, 210, 255),
                                "a5": (255, 90, 120),
                                "a7": (100, 220, 120),
                            }[method]
                            frame_video[part][method] = (
                                _selection_overlay(base_rgb, selection, overlay_color),
                                edited_rgb,
                            )
                        del selection_pkg, edited_pkg

                for part, writer in writers.items():
                    if set(frame_video[part]) != set(args.methods):
                        raise ValueError(f"video part {part} is missing method renders")
                    writer.append_data(
                        compose_method_video_panel(
                            base_rgb, frame_video[part], list(args.methods)
                        )
                    )
                del base_pkg, deformed, colors_precomp
                if (frame_index + 1) % 10 == 0 or frame_index + 1 == len(dataset):
                    print(f"[{args.subject}] temporal frame {frame_index + 1}/{len(dataset)}", flush=True)
                if (frame_index + 1) % 50 == 0:
                    torch.cuda.empty_cache()
    finally:
        for writer in writers.values():
            writer.close()

    select_global_adaptive_metrics(
        metric_rows,
        adaptive_sweeps,
        target_retention=float(args.adaptive_target_retention),
        reference_method="a5" if "a5" in args.methods else str(args.methods[0]),
        reference_strength=float(args.edit_strength),
    )
    metrics_path = output_dir / "metrics.csv"
    _write_metrics(metrics_path, metric_rows)
    expected_rows = expected_metric_row_count(
        args.frame_start,
        args.frame_end,
        args.frame_step,
        part_count=len(args.parts),
        method_count=len(args.methods),
    )
    if len(metric_rows) != expected_rows:
        raise ValueError(f"temporal metric row count {len(metric_rows)} does not match expected {expected_rows}")
    if not args.no_videos:
        missing_videos = [str(path) for path in video_paths.values() if not path.exists() or path.stat().st_size == 0]
        if missing_videos:
            raise ValueError(f"missing or empty temporal videos: {missing_videos}")
    summary = {
        "subject": str(args.subject),
        "camera": int(args.camera),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frame_step": int(args.frame_step),
        "frame_count": len(range(int(args.frame_start), int(args.frame_end), int(args.frame_step))),
        "parts": list(args.parts),
        "methods": list(args.methods),
        "metric_row_count": len(metric_rows),
        "screen_threshold": float(args.screen_threshold),
        "edit_strength": float(args.edit_strength),
        "adaptive_target_retention": float(args.adaptive_target_retention),
        "adaptive_strength_grid": adaptive_strengths,
        "boundary_radius": int(args.boundary_radius or run_config["boundary_radius"]),
        "soft_threshold": float(run_config["soft_threshold"]),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _file_sha256(args.checkpoint),
        "voting_bank": str(args.voting_bank.resolve()),
        "voting_bank_sha256": _file_sha256(args.voting_bank),
        "a5_bank": str(args.a5_bank.resolve()),
        "a5_bank_sha256": _file_sha256(args.a5_bank),
        "a7_bank": str(args.a7_bank.resolve()) if args.a7_bank is not None else "",
        "a7_contract": (
            str(args.a7_contract.resolve()) if args.a7_contract is not None else ""
        ),
        "loso_config": str(args.loso_config.resolve()),
        "method_freeze": str(args.method_freeze.resolve()),
        "method_freeze_id": run_config["method_freeze_id"],
        "method_freeze_fingerprint": run_config["method_freeze_fingerprint"],
        "metrics_csv": str(metrics_path),
        "videos": [str(video_paths[part]) for part in args.video_parts] if not args.no_videos else [],
        "video_fps": int(args.video_fps),
        "elapsed_seconds": float(time.time() - started),
        "uses_test_parser_for_edit_selection": False,
        "uses_test_masks_for_metrics": True,
        "uses_test_masks_for_adaptive_metric_matching": True,
        "shared_rasterizer_across_methods": True,
        "canonical_selection_fixed_across_frames": True,
        "common_support_across_methods": True,
        "temporal_metrics": summarize_temporal_rows(metric_rows),
        **a7_provenance,
        "held_out_camera": int(args.camera) == 21,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    summary = run_temporal_evaluation(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
