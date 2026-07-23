#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.make_semantic_edit_render_preview import EDIT_COLORS, _select_named_records, _tensor_to_rgb_uint8
from utils.frozen_semantic_method import load_frozen_semantic_method
from utils.part_label_bank import PART_NAMES, load_part_label_bank
from utils.semantic_real_editing import (
    REAL_EDIT_METHODS,
    REAL_EDIT_TASKS,
    build_edit_overrides,
    canonical_stripe_colors,
    compute_edit_delta_metrics,
    resolve_method_weights,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the formal multi-method real semantic editing paper suite.")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--raw-bank", required=True, type=Path)
    parser.add_argument("--voting-bank", required=True, type=Path)
    parser.add_argument("--a5-bank", required=True, type=Path)
    parser.add_argument("--loso-config", required=True, type=Path)
    parser.add_argument("--method-freeze", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--views", nargs="*", default=None)
    parser.add_argument("--max-views", type=int, default=9)
    parser.add_argument("--parts", nargs="+", choices=list(PART_NAMES), default=list(PART_NAMES))
    parser.add_argument("--methods", nargs="+", choices=list(REAL_EDIT_METHODS), default=list(REAL_EDIT_METHODS))
    parser.add_argument("--tasks", nargs="+", choices=list(REAL_EDIT_TASKS), default=list(REAL_EDIT_TASKS))
    parser.add_argument("--edit-strength", type=float, default=1.0)
    parser.add_argument("--texture-frequency", type=float, default=8.0)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--explicit-binding-render-preset", default="none")
    return parser.parse_args(argv)


def build_experiment_matrix(*, methods, tasks, parts) -> list[dict[str, str]]:
    return [
        {"method": str(method), "task": str(task), "part": str(part)}
        for method, task, part in product(methods, tasks, parts)
    ]


def formal_provenance() -> dict[str, bool]:
    return {
        "uses_test_parser_for_edit_selection": False,
        "uses_test_masks_for_metrics": True,
        "shared_rasterizer_across_methods": True,
        "shared_edit_parameters_across_methods": True,
    }


def normalize_dataset_subject(config, subject: str) -> None:
    value = str(subject)
    config.dataset.subject = value if value.startswith("CoreView_") else f"CoreView_{value}"


def load_frozen_run_config(*, subject: str, loso_config: Path, method_freeze: Path) -> dict:
    frozen = load_frozen_semantic_method(method_freeze)
    loso = json.loads(Path(loso_config).read_text(encoding="utf-8"))
    if str(loso.get("held_out_subject", "")) != str(subject):
        raise ValueError("LOSO held-out subject does not match requested subject")
    expected = str(loso.get("method_freeze_fingerprint", ""))
    if expected and expected != str(frozen["_fingerprint"]):
        raise ValueError("LOSO method-freeze fingerprint does not match frozen method")
    selected = loso.get("selected") or {}
    return {
        "soft_threshold": float(selected["soft_threshold"]),
        "boundary_radius": int(selected.get("boundary_radius", 6)),
        "method_freeze_id": str(frozen["freeze_id"]),
        "method_freeze_fingerprint": str(frozen["_fingerprint"]),
        "loso": loso,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_bank_points(raw_bank: dict, voting_bank: dict, a5_bank: dict) -> int:
    counts = []
    for owner, bank in (("raw", raw_bank), ("voting", voting_bank), ("a5", a5_bank)):
        labels = np.asarray(bank.get("editable_label", bank.get("part_label", []))).reshape(-1)
        if labels.size == 0:
            raise ValueError(f"{owner} bank is missing part labels")
        counts.append(int(labels.shape[0]))
    if len(set(counts)) != 1:
        raise ValueError(f"bank point counts do not match: {counts}")
    weights = np.asarray(a5_bank.get("soft_edit_weights", []))
    if weights.shape != (counts[0], len(PART_NAMES)):
        raise ValueError(f"A5 soft_edit_weights must have shape ({counts[0]}, {len(PART_NAMES)})")
    return counts[0]


def _write_metrics(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("real editing suite produced no metric rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_suite(args: argparse.Namespace) -> dict:
    import torch
    from gaussian_renderer import rasterize_gaussians
    from scene import GaussianModel, Scene
    from tools.semantic_viewer.build_part_label_bank import (
        _find_dataset_index,
        _load_config,
        _load_record_masks,
        _load_view_records,
    )

    run_config = load_frozen_run_config(
        subject=str(args.subject),
        loso_config=args.loso_config,
        method_freeze=args.method_freeze,
    )
    raw_bank = load_part_label_bank(args.raw_bank)
    voting_bank = load_part_label_bank(args.voting_bank)
    a5_bank = load_part_label_bank(args.a5_bank)
    bank_point_count = _validate_bank_points(raw_bank, voting_bank, a5_bank)

    asset_root = args.asset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    records = _select_named_records(_load_view_records(asset_root), args.views, int(args.max_views))
    config_path = args.config.resolve() if args.config else asset_root.parent.parent / ".hydra" / "config.yaml"
    config = _load_config(config_path, checkpoint, asset_root, records, args)
    normalize_dataset_subject(config, str(args.subject))
    bg_color = [1, 1, 1] if bool(config.dataset.white_background) else [0, 0, 0]
    matrix = build_experiment_matrix(methods=args.methods, tasks=args.tasks, parts=args.parts)
    metric_rows: list[dict] = []

    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, str(asset_root.parent))
        scene.eval()
        iteration = int(scene.load_checkpoint(str(checkpoint)))
        if int(scene.gaussians.get_xyz.shape[0]) != bank_point_count:
            raise ValueError("checkpoint point count does not match semantic banks")
        canonical_xyz = scene.gaussians.get_xyz.detach().float().cpu().numpy()
        texture_by_part = {
            part: canonical_stripe_colors(
                canonical_xyz,
                primary_rgb=np.asarray(EDIT_COLORS[part], dtype=np.float32) / 255.0,
                secondary_rgb=(0.04, 0.04, 0.04),
                frequency=float(args.texture_frequency),
            )
            for part in args.parts
        }
        weights_by_method_part = {
            (method, part): resolve_method_weights(
                raw_bank,
                voting_bank,
                a5_bank,
                method=method,
                part=part,
                threshold=float(run_config["soft_threshold"]),
            )
            for method in args.methods
            for part in args.parts
        }
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        for record in records:
            view_name = str(record["image_name"])
            dataset_index = _find_dataset_index(scene.test_dataset, view_name)
            if dataset_index is None:
                raise RuntimeError(f"image {view_name} not present in dataset")
            view = scene.test_dataset[dataset_index]
            deformed, _, colors_precomp = scene.convert_gaussians(view, iteration, compute_loss=False)
            if int(deformed.get_xyz.shape[0]) != bank_point_count:
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
            imageio.imwrite(frames_dir / f"{view_name}_rgb.png", base_rgb)
            base_float = np.asarray(base_rgb, dtype=np.float32) / 255.0
            base_colors = colors_precomp.detach().float().cpu().numpy()
            base_opacities = deformed.get_opacity.detach().float().cpu().numpy()
            part_masks, foreground_mask, valid_mask = _load_record_masks(asset_root, record)
            valid_mask = np.minimum(np.asarray(foreground_mask, dtype=np.float32), np.asarray(valid_mask, dtype=np.float32))

            for item in matrix:
                method = item["method"]
                task = item["task"]
                part = item["part"]
                weights = weights_by_method_part[(method, part)]
                target_rgb = np.asarray(EDIT_COLORS[part], dtype=np.float32) / 255.0
                overrides = build_edit_overrides(
                    base_colors,
                    base_opacities,
                    weights,
                    task=task,
                    strength=float(args.edit_strength),
                    target_rgb=target_rgb,
                    texture_colors=texture_by_part[part],
                )
                colors_override = torch.from_numpy(overrides["colors"]).to(
                    device=colors_precomp.device,
                    dtype=colors_precomp.dtype,
                )
                opacity_override = torch.from_numpy(overrides["opacities"]).to(
                    device=deformed.get_opacity.device,
                    dtype=deformed.get_opacity.dtype,
                )
                edited_pkg = rasterize_gaussians(
                    view,
                    deformed,
                    config.pipeline,
                    background,
                    colors_precomp=colors_override,
                    opacities_precomp=opacity_override,
                    return_opacity=False,
                )
                edited_rgb = _tensor_to_rgb_uint8(edited_pkg["render"].clamp(0.0, 1.0))
                frame_path = frames_dir / task / method / f"{view_name}_{part}.png"
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(frame_path, edited_rgb)
                metrics = compute_edit_delta_metrics(
                    base_float,
                    np.asarray(edited_rgb, dtype=np.float32) / 255.0,
                    part_masks[part],
                    valid_mask,
                    boundary_radius=int(run_config["boundary_radius"]),
                )
                metric_rows.append(
                    {
                        "subject": str(args.subject),
                        "view": view_name,
                        "part": part,
                        "task": task,
                        "method": method,
                        "edit_strength": float(args.edit_strength),
                        "soft_threshold": float(run_config["soft_threshold"]),
                        "selected_gaussian_count": int(np.sum(weights > 0.0)),
                        "edit_weight_sum": float(np.sum(weights)),
                        **metrics,
                        "frame": str(frame_path),
                    }
                )
                del edited_pkg
            del base_pkg, deformed, colors_precomp
            torch.cuda.empty_cache()

    metrics_path = output_dir / "metrics.csv"
    _write_metrics(metrics_path, metric_rows)
    summary = {
        "subject": str(args.subject),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "asset_root": str(asset_root),
        "raw_bank": str(args.raw_bank.resolve()),
        "raw_bank_sha256": _file_sha256(args.raw_bank),
        "voting_bank": str(args.voting_bank.resolve()),
        "voting_bank_sha256": _file_sha256(args.voting_bank),
        "a5_bank": str(args.a5_bank.resolve()),
        "a5_bank_sha256": _file_sha256(args.a5_bank),
        "loso_config": str(args.loso_config.resolve()),
        "method_freeze": str(args.method_freeze.resolve()),
        "method_freeze_id": run_config["method_freeze_id"],
        "method_freeze_fingerprint": run_config["method_freeze_fingerprint"],
        "soft_threshold": float(run_config["soft_threshold"]),
        "boundary_radius": int(run_config["boundary_radius"]),
        "edit_strength": float(args.edit_strength),
        "texture_frequency": float(args.texture_frequency),
        "methods": list(args.methods),
        "tasks": list(args.tasks),
        "parts": list(args.parts),
        "views": [str(record["image_name"]) for record in records],
        "metric_row_count": len(metric_rows),
        "metrics_csv": str(metrics_path),
        **formal_provenance(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    summary = run_suite(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
