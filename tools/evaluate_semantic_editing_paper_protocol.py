#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from tools.analyze_projected_soft_edit_leakage import compute_footprint_leakage_for_selection
from utils.part_label_bank import (
    PART_NAMES,
    compute_semantic_margin,
    compute_soft_edit_weights,
    load_part_label_bank,
)
from utils.semantic_eval_protocol import (
    file_fingerprint,
    load_protocol,
    protocol_fingerprint,
    record_fingerprint,
    select_protocol_records,
    validate_frozen_config,
    write_protocol_provenance,
)
from utils.semantic_paper_metrics import (
    aggregate_part_metrics,
    binary_segmentation_metrics,
    boundary_metrics,
    interpolate_curve_at_retention,
    shared_retention_targets,
    soft_iou,
)


BASELINE_SPECS = OrderedDict(
    (
        ("B0", {"name": "parser_oracle", "oracle": True, "persistent_asset": False}),
        ("B1", {"name": "projected_multiview_voting", "oracle": False, "persistent_asset": True}),
        ("B2", {"name": "hard_trained_label", "oracle": False, "persistent_asset": True}),
        ("B3", {"name": "raw_semantic_probability", "oracle": False, "persistent_asset": True}),
        ("B4", {"name": "confidence_margin", "oracle": False, "persistent_asset": True}),
        ("B5", {"name": "evidence_target_support", "oracle": False, "persistent_asset": True}),
    )
)


def _one_hot_labels(labels, *, part_index: int) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int16).reshape(-1)
    return (values == int(part_index)).astype(np.float32)


def _required_matrix(bank: dict, field: str, *, part_index: int) -> np.ndarray:
    if field not in bank:
        raise ValueError(f"baseline requires bank field: {field}")
    matrix = np.asarray(bank[field], dtype=np.float32)
    if matrix.ndim != 2 or not (0 <= int(part_index) < matrix.shape[1]):
        raise ValueError(f"{field} must be a 2D part-weight matrix")
    return matrix[:, int(part_index)].astype(np.float32, copy=False)


def resolve_baseline_point_weights(
    baseline: str,
    *,
    trained_bank: dict,
    voting_bank: dict | None,
    part_index: int,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    baseline = str(baseline)
    if baseline not in BASELINE_SPECS:
        raise ValueError(f"unknown baseline: {baseline}")
    if baseline == "B0":
        raise ValueError("B0 parser oracle is a screen-space baseline without Gaussian point weights")
    if baseline == "B1":
        if voting_bank is None:
            raise ValueError("B1 requires a projected multi-view voting bank")
        labels = voting_bank.get("editable_label", voting_bank.get("part_label"))
        if labels is None:
            raise ValueError("B1 voting bank requires editable_label or part_label")
        weights = _one_hot_labels(labels, part_index=part_index)
        support = None
        weight_field = "voting_editable_label"
    elif baseline == "B2":
        labels = trained_bank.get("editable_label", trained_bank.get("part_label"))
        if labels is None:
            raise ValueError("B2 trained bank requires editable_label or part_label")
        weights = _one_hot_labels(labels, part_index=part_index)
        support = None
        weight_field = "editable_label"
    elif baseline == "B3":
        weights = _required_matrix(trained_bank, "semantic_probs", part_index=part_index)
        support = None
        weight_field = "semantic_probs"
    elif baseline == "B4":
        if "semantic_probs" not in trained_bank or "confidence" not in trained_bank:
            raise ValueError("B4 requires semantic_probs and confidence")
        probs = np.asarray(trained_bank["semantic_probs"], dtype=np.float32)
        margin = np.asarray(
            trained_bank.get("semantic_margin", compute_semantic_margin(probs)),
            dtype=np.float32,
        )
        reliable = trained_bank.get("reliable_mask", np.ones((probs.shape[0],), dtype=np.uint8))
        matrix = compute_soft_edit_weights(
            semantic_probs=probs,
            confidence=trained_bank["confidence"],
            semantic_margin=margin,
            reliable_mask=reliable,
        )
        weights = matrix[:, int(part_index)].astype(np.float32, copy=False)
        support = None
        weight_field = "confidence_margin_recomputed"
    else:
        weights = _required_matrix(trained_bank, "edit_target_weights", part_index=part_index)
        support = _required_matrix(trained_bank, "edit_support_weights", part_index=part_index)
        weight_field = "edit_target_weights"
    metadata = {
        "baseline": baseline,
        **BASELINE_SPECS[baseline],
        "weight_field": weight_field,
    }
    return weights, support, metadata


def resolve_parser_oracle_prediction(part_masks: dict[str, np.ndarray], part: str) -> np.ndarray:
    if part not in part_masks:
        raise ValueError(f"parser oracle is missing part mask: {part}")
    return np.asarray(part_masks[part], dtype=np.float32)


def rasterize_footprint_weight_map(
    *,
    xy,
    radii,
    weights,
    image_shape: tuple[int, int],
    threshold: float,
    min_radius: int = 1,
    max_radius: int = 12,
) -> np.ndarray:
    points = np.asarray(xy, dtype=np.float32)
    radii_arr = np.asarray(radii, dtype=np.float32).reshape(-1)
    weights_arr = np.asarray(weights, dtype=np.float32).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("xy must have shape [N, 2]")
    if points.shape[0] != radii_arr.shape[0] or points.shape[0] != weights_arr.shape[0]:
        raise ValueError("xy, radii, and weights must have matching point counts")
    height, width = int(image_shape[0]), int(image_shape[1])
    output = np.zeros((height, width), dtype=np.float32)
    active = np.nonzero(
        np.isfinite(weights_arr)
        & (weights_arr >= float(threshold))
        & np.isfinite(radii_arr)
        & (radii_arr > 0.0)
    )[0]
    min_radius = max(0, int(min_radius))
    max_radius = max(min_radius, int(max_radius))
    for point_index in active:
        x = int(round(float(points[point_index, 0])))
        y = int(round(float(points[point_index, 1])))
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        radius = int(np.ceil(float(radii_arr[point_index])))
        radius = max(min_radius, min(max_radius, radius))
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = ((yy - y) ** 2 + (xx - x) ** 2) <= radius * radius
        patch = output[y0:y1, x0:x1]
        patch[disk] = np.maximum(patch[disk], float(weights_arr[point_index]))
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_baseline_reports(output_dir: Path | str, result: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    reports = (
        ("baseline_summary.csv", "baseline_summary"),
        ("per_part_metrics.csv", "per_part"),
        ("per_view_metrics.csv", "per_view"),
        ("leakage_retention_curve.csv", "curve"),
        ("matched_retention.csv", "matched_retention"),
        ("support_diagnostics.csv", "support_diagnostics"),
        ("boundary_radius_sensitivity.csv", "boundary_radius_sensitivity"),
        ("validation_candidates.csv", "validation_candidates"),
    )
    for filename, key in reports:
        _write_csv(output_dir / filename, list(result.get(key, [])))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if float(denominator) <= 0.0 else float(numerator) / float(denominator)


def _aggregate_metric_rows(per_view: list[dict]) -> tuple[list[dict], list[dict]]:
    per_part = []
    baselines = sorted({row["baseline"] for row in per_view})
    for baseline in baselines:
        baseline_rows = [row for row in per_view if row["baseline"] == baseline]
        for part in sorted({row["part"] for row in baseline_rows}):
            rows = [row for row in baseline_rows if row["part"] == part]
            intersection = sum(int(row["intersection"]) for row in rows)
            union = sum(int(row["union"]) for row in rows)
            predicted = sum(int(row["predicted"]) for row in rows)
            target = sum(int(row["target"]) for row in rows)
            per_part.append(
                {
                    "baseline": baseline,
                    "part": part,
                    "oracle": bool(rows[0]["oracle"]),
                    "view_count": len(rows),
                    "intersection": intersection,
                    "union": union,
                    "predicted": predicted,
                    "target": target,
                    "target_empty": target == 0,
                    "iou": _safe_ratio(intersection, union),
                    "precision": _safe_ratio(intersection, predicted),
                    "recall": _safe_ratio(intersection, target),
                    "soft_iou": float(np.mean([float(row["soft_iou"]) for row in rows])),
                    "boundary_precision": float(np.mean([float(row["boundary_precision"]) for row in rows])),
                    "boundary_recall": float(np.mean([float(row["boundary_recall"]) for row in rows])),
                    "boundary_f1": float(np.mean([float(row["boundary_f1"]) for row in rows])),
                    "boundary_iou": float(np.mean([float(row["boundary_iou"]) for row in rows])),
                }
            )
    summaries = []
    for baseline in baselines:
        rows = [row for row in per_part if row["baseline"] == baseline]
        aggregate = aggregate_part_metrics(rows)
        summaries.append(
            {
                "baseline": baseline,
                **BASELINE_SPECS[baseline],
                **aggregate,
                "mean_boundary_f1": float(np.mean([float(row["boundary_f1"]) for row in rows])) if rows else 0.0,
                "mean_boundary_iou": float(np.mean([float(row["boundary_iou"]) for row in rows])) if rows else 0.0,
                "mean_soft_iou": float(np.mean([float(row["soft_iou"]) for row in rows])) if rows else 0.0,
            }
        )
    return per_part, summaries


def _allowed_adjacent_masks(protocol: dict, part: str, part_masks: dict) -> dict[str, np.ndarray]:
    names = protocol.get("allowed_adjacency", {}).get(part, [])
    return {name: part_masks[name] for name in names if name in part_masks}


def _curve_for_baseline(
    baseline: str,
    *,
    caches: list[dict],
    trained_bank: dict,
    voting_bank: dict,
    protocol: dict,
    boundary_radius: int,
    hard_target_activation: float | None = None,
) -> list[dict]:
    if baseline in ("B1", "B2"):
        settings = [(0.5, float(strength)) for strength in protocol.get("matched_retention_targets", [0.3, 0.5, 0.7, 1.0])]
    else:
        settings = [(float(threshold), 1.0) for threshold in protocol["validation_grid"]["soft_thresholds"]]
    rows = []
    for threshold, strength in settings:
        target_activation = 0.0
        outer_activation = 0.0
        actionable_activation = 0.0
        selected_count = 0
        for cache in caches:
            for part in protocol["parts"]:
                part_index = PART_NAMES.index(part)
                weights, _support, _metadata = resolve_baseline_point_weights(
                    baseline,
                    trained_bank=trained_bank,
                    voting_bank=voting_bank,
                    part_index=part_index,
                )
                point_weights = weights * float(strength)
                selected = (weights >= float(threshold)) & cache["projected"]
                row = compute_footprint_leakage_for_selection(
                    part=part,
                    mode=baseline,
                    view_name=cache["view"],
                    xy=cache["xy"],
                    selected=selected,
                    weights=point_weights,
                    radii=cache["radii"],
                    target_mask=cache["part_masks"][part],
                    valid_mask=cache["valid_mask"],
                    boundary_radius=int(boundary_radius),
                    allowed_adjacent_masks=_allowed_adjacent_masks(protocol, part, cache["part_masks"]),
                )
                target_activation += float(row["target_activation"])
                outer_activation += float(row["outer_activation"])
                actionable_activation += float(row["actionable_outer_activation"])
                selected_count += int(row["selected_count"])
        rows.append(
            {
                "baseline": baseline,
                "threshold": threshold,
                "edit_strength": strength,
                "boundary_radius": int(boundary_radius),
                "target_activation": target_activation,
                "raw_leakage": _safe_ratio(outer_activation, target_activation),
                "actionable_leakage": _safe_ratio(actionable_activation, target_activation),
                "selected_count": selected_count,
                "retention": (
                    _safe_ratio(target_activation, hard_target_activation)
                    if hard_target_activation is not None
                    else 1.0
                ),
            }
        )
    return sorted(rows, key=lambda row: (float(row["retention"]), float(row["threshold"])))


def _support_diagnostics_for_b5(
    *,
    caches: list[dict],
    trained_bank: dict,
    protocol: dict,
    boundary_radius: int,
) -> list[dict]:
    rows = []
    for support_threshold in protocol["validation_grid"]["support_thresholds"]:
        target_activation = 0.0
        outer_activation = 0.0
        allowed_activation = 0.0
        actionable_activation = 0.0
        selected_count = 0
        for cache in caches:
            for part in protocol["parts"]:
                _target, support, _metadata = resolve_baseline_point_weights(
                    "B5",
                    trained_bank=trained_bank,
                    voting_bank=None,
                    part_index=PART_NAMES.index(part),
                )
                selected = (support >= float(support_threshold)) & cache["projected"]
                row = compute_footprint_leakage_for_selection(
                    part=part,
                    mode="B5_support",
                    view_name=cache["view"],
                    xy=cache["xy"],
                    selected=selected,
                    weights=support,
                    radii=cache["radii"],
                    target_mask=cache["part_masks"][part],
                    valid_mask=cache["valid_mask"],
                    boundary_radius=int(boundary_radius),
                    allowed_adjacent_masks=_allowed_adjacent_masks(protocol, part, cache["part_masks"]),
                )
                target_activation += float(row["target_activation"])
                outer_activation += float(row["outer_activation"])
                allowed_activation += float(row["allowed_adjacent_activation"])
                actionable_activation += float(row["actionable_outer_activation"])
                selected_count += int(row["selected_count"])
        rows.append(
            {
                "baseline": "B5",
                "support_threshold": float(support_threshold),
                "boundary_radius": int(boundary_radius),
                "selected_count": selected_count,
                "support_target_activation": target_activation,
                "support_outer_activation": outer_activation,
                "allowed_support_activation": allowed_activation,
                "actionable_support_activation": actionable_activation,
                "allowed_support_fraction": _safe_ratio(allowed_activation, outer_activation),
                "actionable_support_fraction": _safe_ratio(actionable_activation, outer_activation),
            }
        )
    return rows


def _mean_boundary_f1_for_threshold(
    *,
    baseline: str,
    threshold: float,
    caches: list[dict],
    trained_bank: dict,
    voting_bank: dict,
    protocol: dict,
) -> float:
    values = []
    for cache in caches:
        for part in protocol["parts"]:
            weights, _support, _metadata = resolve_baseline_point_weights(
                baseline,
                trained_bank=trained_bank,
                voting_bank=voting_bank,
                part_index=PART_NAMES.index(part),
            )
            point_weights = np.where(cache["projected"], weights, 0.0)
            prediction = rasterize_footprint_weight_map(
                xy=cache["xy"],
                radii=cache["radii"],
                weights=point_weights,
                image_shape=cache["valid_mask"].shape,
                threshold=float(threshold),
            )
            values.append(
                float(
                    boundary_metrics(
                        prediction > 0.0,
                        cache["part_masks"][part] >= 0.5,
                        tolerance=int(protocol.get("boundary_metric_tolerance", 2)),
                        valid_mask=cache["valid_mask"] >= 0.5,
                    )["boundary_f1"]
                )
            )
    return float(np.mean(values)) if values else 0.0


def evaluate_scene(args: argparse.Namespace) -> dict:
    import torch
    from gaussian_renderer import render
    from scene import GaussianModel, Scene
    from tools.semantic_viewer.build_part_label_bank import (
        _find_dataset_index,
        _load_config,
        _load_record_masks,
        _load_view_records,
        _project_points,
    )

    protocol = load_protocol(args.protocol)
    if args.protocol_split not in ("validation", "test"):
        raise ValueError("paper evaluator only accepts validation or test protocol splits")
    asset_root = args.asset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    records = select_protocol_records(_load_view_records(asset_root), protocol, args.protocol_split)
    checkpoint_fp = file_fingerprint(checkpoint)
    bank_fp = file_fingerprint(args.trained_bank)
    frozen = None
    if args.protocol_split == "test":
        if args.frozen_config is None:
            raise ValueError("test evaluation requires --frozen-config")
        frozen = json.loads(args.frozen_config.read_text(encoding="utf-8"))
        validate_frozen_config(frozen, protocol=protocol, checkpoint_fingerprint=checkpoint_fp)
        if str(frozen.get("bank_fingerprint", "")) != bank_fp:
            raise ValueError("bank fingerprint mismatch in frozen validation config")
    selected_config = (frozen or {}).get("selected", {})
    fixed_threshold = float(selected_config.get("soft_threshold", 0.20))
    fixed_support_threshold = float(selected_config.get("support_threshold", 0.20))
    boundary_radius = int(selected_config.get("boundary_radius", 2))
    trained_bank = load_part_label_bank(args.trained_bank)
    voting_bank = load_part_label_bank(args.voting_bank)
    config_path = args.config.resolve() if args.config else asset_root.parent.parent / ".hydra" / "config.yaml"
    config = _load_config(config_path, checkpoint, asset_root, records, args)
    caches = []
    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, str(asset_root.parent))
        scene.eval()
        loaded_iteration = int(scene.load_checkpoint(str(checkpoint)))
        point_count = int(scene.gaussians.get_xyz.shape[0])
        for bank_name, bank in (("trained", trained_bank), ("voting", voting_bank)):
            labels = np.asarray(bank.get("part_label", []))
            if labels.shape[0] != point_count:
                raise ValueError(f"{bank_name} bank point count {labels.shape[0]} does not match scene {point_count}")
        background = torch.zeros(3, dtype=torch.float32, device="cuda")
        for record in records:
            dataset_index = _find_dataset_index(scene.test_dataset, record["image_name"])
            if dataset_index is None:
                raise RuntimeError(f"image {record['image_name']} not present in dataset")
            view = scene.test_dataset[dataset_index]
            deformed, _, _colors = scene.convert_gaussians(view, loaded_iteration, compute_loss=False)
            xy, projected, _depth = _project_points(deformed.get_xyz, view)
            render_pkg = render(
                view,
                loaded_iteration,
                scene,
                config.pipeline,
                background,
                compute_loss=False,
                return_opacity=False,
            )
            part_masks, foreground, valid = _load_record_masks(asset_root, record)
            combined_valid = np.minimum(np.asarray(foreground, dtype=np.float32), np.asarray(valid, dtype=np.float32))
            caches.append(
                {
                    "view": str(record["image_name"]),
                    "xy": xy.detach().float().cpu().numpy(),
                    "projected": projected.detach().bool().cpu().numpy(),
                    "radii": render_pkg["radii"].detach().float().cpu().numpy().reshape(-1)[:point_count],
                    "part_masks": part_masks,
                    "valid_mask": combined_valid,
                }
            )

    per_view = []
    for cache in caches:
        for part in protocol["parts"]:
            target = np.asarray(cache["part_masks"][part], dtype=np.float32)
            valid = np.asarray(cache["valid_mask"], dtype=np.float32)
            for baseline in args.baselines:
                metadata = BASELINE_SPECS[baseline]
                if baseline == "B0":
                    prediction = resolve_parser_oracle_prediction(cache["part_masks"], part)
                else:
                    weights, _support, metadata = resolve_baseline_point_weights(
                        baseline,
                        trained_bank=trained_bank,
                        voting_bank=voting_bank,
                        part_index=PART_NAMES.index(part),
                    )
                    point_weights = np.where(cache["projected"], weights, 0.0)
                    threshold = 0.5 if baseline in ("B1", "B2") else fixed_threshold
                    prediction = rasterize_footprint_weight_map(
                        xy=cache["xy"],
                        radii=cache["radii"],
                        weights=point_weights,
                        image_shape=valid.shape,
                        threshold=threshold,
                    )
                binary = prediction >= 0.5 if baseline == "B0" else prediction > 0.0
                binary_stats = binary_segmentation_metrics(binary, target >= 0.5, valid_mask=valid >= 0.5)
                boundary_stats = boundary_metrics(
                    binary,
                    target >= 0.5,
                    tolerance=int(protocol.get("boundary_metric_tolerance", 2)),
                    valid_mask=valid >= 0.5,
                )
                per_view.append(
                    {
                        "baseline": baseline,
                        "view": cache["view"],
                        "part": part,
                        "oracle": bool(metadata["oracle"]),
                        **binary_stats,
                        "soft_iou": soft_iou(prediction, target, valid_mask=valid >= 0.5),
                        **boundary_stats,
                    }
                )
    per_part, baseline_summary = _aggregate_metric_rows(per_view)

    hard_curve = _curve_for_baseline(
        "B2",
        caches=caches,
        trained_bank=trained_bank,
        voting_bank=voting_bank,
        protocol=protocol,
        boundary_radius=boundary_radius,
    )
    hard_target = max(float(row["target_activation"]) for row in hard_curve)
    curves = {"B2": _curve_for_baseline(
        "B2", caches=caches, trained_bank=trained_bank, voting_bank=voting_bank,
        protocol=protocol, boundary_radius=boundary_radius, hard_target_activation=hard_target,
    )}
    for baseline in ("B1", "B3", "B4", "B5"):
        if baseline in args.baselines:
            curves[baseline] = _curve_for_baseline(
                baseline,
                caches=caches,
                trained_bank=trained_bank,
                voting_bank=voting_bank,
                protocol=protocol,
                boundary_radius=boundary_radius,
                hard_target_activation=hard_target,
            )
    curve_rows = [row for rows in curves.values() for row in rows]
    matched = []
    for baseline, rows in curves.items():
        if baseline == "B2":
            continue
        targets = shared_retention_targets(
            {"B2": curves["B2"], baseline: rows},
            protocol.get("matched_retention_targets", []),
        )
        for target_retention in targets:
            row = interpolate_curve_at_retention(rows, target_retention)
            row.update({"baseline": baseline, "reference_baseline": "B2"})
            matched.append(row)

    validation_candidates = []
    radius_rows = []
    support_diagnostics = _support_diagnostics_for_b5(
        caches=caches,
        trained_bank=trained_bank,
        protocol=protocol,
        boundary_radius=boundary_radius,
    )
    if bool(args.validation_sweep):
        for radius in protocol["validation_grid"]["boundary_radii"]:
            b5_curve = _curve_for_baseline(
                "B5",
                caches=caches,
                trained_bank=trained_bank,
                voting_bank=voting_bank,
                protocol=protocol,
                boundary_radius=int(radius),
                hard_target_activation=hard_target,
            )
            support_rows = _support_diagnostics_for_b5(
                caches=caches,
                trained_bank=trained_bank,
                protocol=protocol,
                boundary_radius=int(radius),
            )
            support_by_threshold = {
                float(row["support_threshold"]): row for row in support_rows
            }
            for row in b5_curve:
                boundary_f1 = _mean_boundary_f1_for_threshold(
                    baseline="B5",
                    threshold=float(row["threshold"]),
                    caches=caches,
                    trained_bank=trained_bank,
                    voting_bank=voting_bank,
                    protocol=protocol,
                )
                radius_rows.append({**row, "mean_boundary_f1": boundary_f1})
                for support_threshold in protocol["validation_grid"]["support_thresholds"]:
                    support_row = support_by_threshold[float(support_threshold)]
                    validation_candidates.append(
                        {
                            "soft_threshold": float(row["threshold"]),
                            "support_threshold": float(support_threshold),
                            "boundary_radius": int(radius),
                            "aggregate_target_retention": float(row["retention"]),
                            "mean_actionable_footprint_leakage": float(row["actionable_leakage"]),
                            "mean_raw_footprint_leakage": float(row["raw_leakage"]),
                            "mean_boundary_f1": boundary_f1,
                            "allowed_support_fraction": float(support_row["allowed_support_fraction"]),
                            "actionable_support_fraction": float(support_row["actionable_support_fraction"]),
                            "allowed_support_activation": float(support_row["allowed_support_activation"]),
                            "actionable_support_activation": float(support_row["actionable_support_activation"]),
                        }
                    )
    summary = {
        "protocol_name": protocol["protocol_name"],
        "protocol_split": args.protocol_split,
        "protocol_fingerprint": protocol_fingerprint(protocol),
        "record_fingerprint": record_fingerprint(records),
        "checkpoint_fingerprint": checkpoint_fp,
        "bank_fingerprint": bank_fp,
        "baseline_count": len(args.baselines),
        "processed_views": len(records),
        "fixed_soft_threshold": fixed_threshold,
        "fixed_support_threshold": fixed_support_threshold,
        "fixed_boundary_radius": boundary_radius,
        "uses_test_parser_for_calibration": False,
        "parser_oracle_baseline": "B0",
    }
    write_protocol_provenance(
        args.output_dir,
        protocol,
        records,
        split_name=args.protocol_split,
        source_asset_root=asset_root,
        frozen_config=frozen,
    )
    return {
        "summary": summary,
        "baseline_summary": baseline_summary,
        "per_part": per_part,
        "per_view": per_view,
        "curve": curve_rows,
        "matched_retention": matched,
        "support_diagnostics": support_diagnostics,
        "boundary_radius_sensitivity": radius_rows,
        "validation_candidates": validation_candidates,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate B0-B5 strict semantic editing baselines.")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-split", required=True, choices=("validation", "test"))
    parser.add_argument("--frozen-config", type=Path, default=None)
    parser.add_argument("--validation-sweep", action="store_true")
    parser.add_argument("--trained-bank", required=True, type=Path)
    parser.add_argument("--voting-bank", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--subject", default="")
    parser.add_argument("--explicit-binding-render-preset", default="v338_temporal_selector_grow_only_guard")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baselines", nargs="+", default=list(BASELINE_SPECS), choices=list(BASELINE_SPECS))
    parser.add_argument("--parts", nargs="+", default=list(PART_NAMES), choices=list(PART_NAMES))
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    result = evaluate_scene(args)
    write_baseline_reports(args.output_dir, result)
    print(args.output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
