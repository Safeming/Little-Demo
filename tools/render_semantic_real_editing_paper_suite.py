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
from utils.frozen_semantic_method import (
    load_a7_temporal_contract,
    load_frozen_semantic_method,
    validate_a7_bank_against_contract,
)
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
    parser.add_argument("--a7-bank", type=Path, default=None)
    parser.add_argument("--a7-contract", type=Path, default=None)
    parser.add_argument("--loso-config", required=True, type=Path)
    parser.add_argument("--method-freeze", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--views", nargs="*", default=None)
    parser.add_argument("--max-views", type=int, default=9)
    parser.add_argument("--parts", nargs="+", choices=list(PART_NAMES), default=list(PART_NAMES))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(REAL_EDIT_METHODS),
        default=["raw_hard", "voting", "a5"],
    )
    parser.add_argument("--tasks", nargs="+", choices=list(REAL_EDIT_TASKS), default=list(REAL_EDIT_TASKS))
    parser.add_argument("--edit-strength", type=float, default=1.0)
    parser.add_argument("--edit-strengths", nargs="+", type=float, default=None)
    parser.add_argument("--coverage-target-retention", type=float, default=0.5)
    parser.add_argument("--coverage-response-fraction", type=float, default=0.8)
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--texture-frequency", type=float, default=8.0)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--explicit-binding-render-preset", default="none")
    return parser.parse_args(argv)


def resolve_edit_strengths(args) -> list[float]:
    values = getattr(args, "edit_strengths", None)
    if values is None:
        values = [getattr(args, "edit_strength", 1.0)]
    strengths = [float(value) for value in values]
    if not strengths or any(value <= 0.0 or value > 1.0 for value in strengths):
        raise ValueError("edit strength values must be in (0, 1]")
    if strengths != sorted(strengths) or len(set(strengths)) != len(strengths):
        raise ValueError("edit strength grid must be strictly increasing and unique")
    return strengths


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


def _validate_bank_points(
    raw_bank: dict,
    voting_bank: dict,
    a5_bank: dict,
    *,
    a7_bank: dict | None = None,
) -> int:
    counts = []
    banks = [("raw", raw_bank), ("voting", voting_bank), ("a5", a5_bank)]
    if a7_bank is not None:
        banks.append(("a7", a7_bank))
    for owner, bank in banks:
        labels = np.asarray(bank.get("editable_label", bank.get("part_label", []))).reshape(-1)
        if labels.size == 0:
            raise ValueError(f"{owner} bank is missing part labels")
        counts.append(int(labels.shape[0]))
    if len(set(counts)) != 1:
        raise ValueError(f"bank point counts do not match: {counts}")
    weights = np.asarray(a5_bank.get("soft_edit_weights", []))
    if weights.shape != (counts[0], len(PART_NAMES)):
        raise ValueError(f"A5 soft_edit_weights must have shape ({counts[0]}, {len(PART_NAMES)})")
    if a7_bank is not None:
        a7_weights = np.asarray(a7_bank.get("soft_edit_weights", []))
        if a7_weights.shape != (counts[0], len(PART_NAMES)):
            raise ValueError(
                f"A7 soft_edit_weights must have shape ({counts[0]}, {len(PART_NAMES)})"
            )
    return counts[0]


def aggregate_real_edit_metrics(
    rows: list[dict],
    *,
    target_retention: float = 0.5,
    coverage_response_fraction: float = 0.8,
    reference_method: str = "a5",
) -> list[dict]:
    grouped = {}
    for row in rows:
        key = (str(row["task"]), str(row["part"]))
        grouped.setdefault(key, []).append(row)
    outputs = []
    for (task, part), members in sorted(grouped.items()):
        methods = sorted({str(row["method"]) for row in members})
        effective_reference = (
            reference_method if reference_method in methods else methods[0]
        )
        reference_rows = [
            row for row in members if str(row["method"]) == effective_reference
        ]
        reference_strength = max(float(row["edit_strength"]) for row in reference_rows)
        reference_at_max = [
            row
            for row in reference_rows
            if float(row["edit_strength"]) == reference_strength
        ]
        reference_by_view = {
            str(row["view"]): float(row["target_delta_sum"])
            for row in reference_at_max
        }
        reference_target = float(sum(reference_by_view.values()))
        desired_target = float(target_retention) * reference_target
        for method in methods:
            method_rows = [row for row in members if str(row["method"]) == method]
            strengths = sorted({float(row["edit_strength"]) for row in method_rows})
            pooled = []
            for strength in strengths:
                strength_rows = [
                    row
                    for row in method_rows
                    if float(row["edit_strength"]) == strength
                ]
                pooled.append(
                    (
                        strength,
                        float(
                            sum(float(row["target_delta_sum"]) for row in strength_rows)
                        ),
                        strength_rows,
                    )
                )
            selected_strength, target, selected_rows = min(
                pooled,
                key=lambda item: (
                    abs(float(item[1]) - desired_target),
                    float(item[0]),
                ),
            )
            outer = float(
                sum(float(row["outer_delta_sum"]) for row in selected_rows)
            )
            boundary = float(
                sum(
                    float(row["boundary_outer_delta_sum"]) for row in selected_rows
                )
            )
            coverage = []
            for row in selected_rows:
                reference_response = reference_by_view.get(str(row["view"]), 0.0)
                coverage.append(
                    reference_response > 0.0
                    and float(row["target_delta_sum"])
                    >= float(coverage_response_fraction) * reference_response
                )
            outputs.append(
                {
                    "method": method,
                    "task": task,
                    "part": part,
                    "selected_strength": float(selected_strength),
                    "target_retention": float(target_retention),
                    "reference_method": effective_reference,
                    "reference_strength": reference_strength,
                    "reference_target_response": reference_target,
                    "desired_target_response": desired_target,
                    "selected_target_response": target,
                    "reachable": max(item[1] for item in pooled) >= desired_target,
                    "view_count": len(selected_rows),
                    "pooled_outer_burden": outer / max(target, 1.0e-8),
                    "pooled_boundary_burden": boundary / max(target, 1.0e-8),
                    "coverage_response_fraction": float(
                        coverage_response_fraction
                    ),
                    "coverage_rate": float(np.mean(coverage)) if coverage else 0.0,
                }
            )
    return outputs


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

    if not 0.0 < float(args.coverage_target_retention) <= 1.0:
        raise ValueError("coverage target retention must be in (0, 1]")
    if not 0.0 < float(args.coverage_response_fraction) <= 1.0:
        raise ValueError("coverage response fraction must be in (0, 1]")
    run_config = load_frozen_run_config(
        subject=str(args.subject),
        loso_config=args.loso_config,
        method_freeze=args.method_freeze,
    )
    raw_bank = load_part_label_bank(args.raw_bank)
    voting_bank = load_part_label_bank(args.voting_bank)
    a5_bank = load_part_label_bank(args.a5_bank)
    if "a7" in args.methods and (args.a7_bank is None or args.a7_contract is None):
        raise ValueError("A7 real editing requires --a7-bank and --a7-contract")
    a7_bank = load_part_label_bank(args.a7_bank) if args.a7_bank is not None else None
    a7_provenance = {}
    if a7_bank is not None:
        a7_contract = load_a7_temporal_contract(args.a7_contract, args.method_freeze)
        a7_provenance = validate_a7_bank_against_contract(
            a7_bank,
            contract=a7_contract,
            a5_bank_path=args.a5_bank,
        )
    bank_point_count = _validate_bank_points(
        raw_bank, voting_bank, a5_bank, a7_bank=a7_bank
    )

    asset_root = args.asset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    frames_dir = output_dir / "frames"
    metrics_only = bool(getattr(args, "metrics_only", False))
    if not metrics_only:
        frames_dir.mkdir(parents=True, exist_ok=True)
    records = _select_named_records(_load_view_records(asset_root), args.views, int(args.max_views))
    config_path = args.config.resolve() if args.config else asset_root.parent.parent / ".hydra" / "config.yaml"
    config = _load_config(config_path, checkpoint, asset_root, records, args)
    normalize_dataset_subject(config, str(args.subject))
    bg_color = [1, 1, 1] if bool(config.dataset.white_background) else [0, 0, 0]
    matrix = build_experiment_matrix(methods=args.methods, tasks=args.tasks, parts=args.parts)
    edit_strengths = resolve_edit_strengths(args)
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
                a7_bank=a7_bank,
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
            if not metrics_only:
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
                for edit_strength in edit_strengths:
                    overrides = build_edit_overrides(
                        base_colors,
                        base_opacities,
                        weights,
                        task=task,
                        strength=float(edit_strength),
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
                    frame_path = ""
                    if not metrics_only:
                        strength_token = str(edit_strength).replace(".", "p")
                        suffix = f"_s{strength_token}" if len(edit_strengths) > 1 else ""
                        frame_output = frames_dir / task / method / f"{view_name}_{part}{suffix}.png"
                        frame_output.parent.mkdir(parents=True, exist_ok=True)
                        imageio.imwrite(frame_output, edited_rgb)
                        frame_path = str(frame_output)
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
                            "edit_strength": float(edit_strength),
                            "soft_threshold": float(run_config["soft_threshold"]),
                            "selected_gaussian_count": int(np.sum(weights > 0.0)),
                            "edit_weight_sum": float(np.sum(weights)),
                            **metrics,
                            "frame": frame_path,
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
        "a7_bank": str(args.a7_bank.resolve()) if args.a7_bank is not None else "",
        "a7_contract": (
            str(args.a7_contract.resolve()) if args.a7_contract is not None else ""
        ),
        "loso_config": str(args.loso_config.resolve()),
        "method_freeze": str(args.method_freeze.resolve()),
        "method_freeze_id": run_config["method_freeze_id"],
        "method_freeze_fingerprint": run_config["method_freeze_fingerprint"],
        "soft_threshold": float(run_config["soft_threshold"]),
        "boundary_radius": int(run_config["boundary_radius"]),
        "edit_strength": float(edit_strengths[0]) if len(edit_strengths) == 1 else None,
        "edit_strengths": edit_strengths,
        "metrics_only": metrics_only,
        "texture_frequency": float(args.texture_frequency),
        "methods": list(args.methods),
        "tasks": list(args.tasks),
        "parts": list(args.parts),
        "views": [str(record["image_name"]) for record in records],
        "metric_row_count": len(metric_rows),
        "pooled_metrics": aggregate_real_edit_metrics(
            metric_rows,
            target_retention=float(args.coverage_target_retention),
            coverage_response_fraction=float(args.coverage_response_fraction),
        ),
        "coverage_target_retention": float(args.coverage_target_retention),
        "coverage_response_fraction": float(args.coverage_response_fraction),
        "canonical_selection_fixed_across_frames": True,
        "common_support_across_methods": True,
        **a7_provenance,
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
