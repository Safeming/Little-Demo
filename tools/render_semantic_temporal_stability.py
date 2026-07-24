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
from utils.part_label_bank import PART_NAMES, load_part_label_bank
from utils.semantic_real_editing import build_edit_overrides, compute_edit_delta_metrics, resolve_method_weights
from utils.semantic_temporal_stability import compute_screen_selection_metrics


TEMPORAL_METHODS = ("voting", "a5")
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
    parser.add_argument("--methods", nargs="+", choices=list(TEMPORAL_METHODS), default=list(TEMPORAL_METHODS))
    parser.add_argument("--video-parts", nargs="+", choices=list(PART_NAMES), default=list(DEFAULT_VIDEO_PARTS))
    parser.add_argument("--video-fps", type=int, default=25)
    parser.add_argument("--screen-threshold", type=float, default=0.2)
    parser.add_argument("--edit-strength", type=float, default=1.0)
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
    point_count = _validate_bank_points(voting_bank, voting_bank, a5_bank)
    config = _build_config(args)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    metric_rows: list[dict] = []
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
                    method=method,
                    part=part,
                    threshold=float(run_config["soft_threshold"]),
                )
                for method in args.methods
                for part in args.parts
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
                        metric_rows.append(
                            {
                                "subject": str(args.subject),
                                "camera": int(args.camera),
                                "frame": int(getattr(view, "frame_id", frame_index)),
                                "view": str(getattr(view, "image_name", f"c{int(args.camera):02d}_f{frame_index:06d}")),
                                "part": part,
                                "method": method,
                                "soft_threshold": float(run_config["soft_threshold"]),
                                "screen_threshold": float(args.screen_threshold),
                                "selected_gaussian_count": int(np.sum(weights_by_method_part[key] > 0.0)),
                                "edit_weight_sum": float(np.sum(weights_by_method_part[key])),
                                **selection_metrics,
                                **{f"edit_{name}": value for name, value in edit_metrics.items()},
                            }
                        )
                        if part in frame_video:
                            overlay_color = (65, 210, 255) if method == "voting" else (255, 90, 120)
                            frame_video[part][method] = (
                                _selection_overlay(base_rgb, selection, overlay_color),
                                edited_rgb,
                            )
                        del selection_pkg, edited_pkg

                for part, writer in writers.items():
                    if set(frame_video[part]) != set(args.methods):
                        raise ValueError(f"video part {part} is missing method renders")
                    voting_overlay, voting_edit = frame_video[part]["voting"]
                    a5_overlay, a5_edit = frame_video[part]["a5"]
                    writer.append_data(compose_video_panel(base_rgb, voting_overlay, a5_overlay, voting_edit, a5_edit))
                del base_pkg, deformed, colors_precomp
                if (frame_index + 1) % 10 == 0 or frame_index + 1 == len(dataset):
                    print(f"[{args.subject}] temporal frame {frame_index + 1}/{len(dataset)}", flush=True)
                if (frame_index + 1) % 50 == 0:
                    torch.cuda.empty_cache()
    finally:
        for writer in writers.values():
            writer.close()

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
        "boundary_radius": int(args.boundary_radius or run_config["boundary_radius"]),
        "soft_threshold": float(run_config["soft_threshold"]),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _file_sha256(args.checkpoint),
        "voting_bank": str(args.voting_bank.resolve()),
        "voting_bank_sha256": _file_sha256(args.voting_bank),
        "a5_bank": str(args.a5_bank.resolve()),
        "a5_bank_sha256": _file_sha256(args.a5_bank),
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
        "shared_rasterizer_across_methods": True,
        "canonical_selection_fixed_across_frames": True,
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
