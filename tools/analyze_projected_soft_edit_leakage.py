#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.part_label_bank import PART_NAMES, load_part_label_bank
from tools.calibrate_evidence_soft_edit_weights import build_soft_boundary_target_mask


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if float(denominator) <= 0.0 else float(numerator) / float(denominator)


def resolve_soft_edit_weights(bank: dict, *, point_count: int) -> tuple[np.ndarray, str]:
    if "soft_edit_weights" in bank:
        weights = np.asarray(bank["soft_edit_weights"], dtype=np.float32)
        if weights.shape != (int(point_count), len(PART_NAMES)):
            raise ValueError(f"soft_edit_weights must have shape ({int(point_count)}, {len(PART_NAMES)})")
        return weights, "soft_edit_weights"

    source_labels = np.asarray(bank.get("editable_label", bank["part_label"]), dtype=np.int16).reshape(-1)
    if source_labels.shape[0] != int(point_count):
        raise ValueError(f"label point count {source_labels.shape[0]} does not match expected {int(point_count)}")
    weights = np.zeros((int(point_count), len(PART_NAMES)), dtype=np.float32)
    valid = (source_labels >= 0) & (source_labels < len(PART_NAMES))
    if np.any(valid):
        weights[np.nonzero(valid)[0], source_labels[valid].astype(np.int64)] = 1.0
    source = "editable_label_one_hot_fallback" if "editable_label" in bank else "part_label_one_hot_fallback"
    return weights, source


def make_boundary_band(mask, radius: int = 2, threshold: float = 0.5) -> np.ndarray:
    inside = np.asarray(mask, dtype=np.float32) >= float(threshold)
    radius = max(0, int(radius))
    if radius == 0:
        return np.zeros_like(inside, dtype=bool)
    padded = np.pad(inside, radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(inside, dtype=bool)
    eroded = np.ones_like(inside, dtype=bool)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            if abs(dy - radius) + abs(dx - radius) > radius:
                continue
            window = padded[dy : dy + inside.shape[0], dx : dx + inside.shape[1]]
            dilated |= window
            eroded &= window
    return (dilated ^ eroded).astype(bool, copy=False)


def upper_torso_x_bounds(mask: np.ndarray, *, mask_threshold: float, width: int) -> tuple[int, int]:
    mask = np.asarray(mask, dtype=np.float32)
    width = max(1, int(width))
    if mask.ndim != 2:
        return 0, width - 1
    active = mask >= float(mask_threshold)
    ys, xs = np.nonzero(active)
    if xs.size == 0:
        return 0, width - 1
    y_min = int(np.min(ys))
    y_max = int(np.max(ys))
    span = max(1, y_max - y_min + 1)
    mid_y_min = y_min + int(round(span * 0.30))
    mid_y_max = y_min + int(round(span * 0.80))
    yy = np.arange(mask.shape[0], dtype=np.int64)[:, None]
    mid = active & (yy >= mid_y_min) & (yy <= mid_y_max)
    _mid_ys, mid_xs = np.nonzero(mid)
    usable_xs = mid_xs if mid_xs.size else xs
    return int(np.min(usable_xs)), int(np.max(usable_xs))


def region_support_mask(
    *,
    part: str,
    target_mask,
    allowed_adjacent_masks: dict[str, np.ndarray] | None,
    mask_threshold: float,
    region_support_mode: str,
) -> np.ndarray:
    target = np.asarray(target_mask, dtype=np.float32)
    support = np.zeros_like(target, dtype=bool)
    modes = [
        item.strip().lower()
        for item in str(region_support_mode or "none").replace("+", ",").split(",")
        if item.strip()
    ]
    if not modes or modes == ["none"]:
        return support
    for mode in modes:
        if mode == "none":
            continue
        if mode == "upper_torso_skin":
            if str(part) != "upper" or not allowed_adjacent_masks or "skin" not in allowed_adjacent_masks:
                continue
            skin = np.asarray(allowed_adjacent_masks["skin"], dtype=np.float32) >= float(mask_threshold)
            if skin.shape != target.shape:
                raise ValueError("skin support mask must match target_mask shape")
            height, width = target.shape
            x_min, x_max = upper_torso_x_bounds(
                target,
                mask_threshold=float(mask_threshold),
                width=width,
            )
            _yy, xx = np.indices((height, width), dtype=np.int64)
            support |= skin & (xx >= x_min) & (xx <= x_max)
            continue
        if mode == "hair_face":
            if not allowed_adjacent_masks:
                continue
            neighbor = "face" if str(part) == "hair" else "hair" if str(part) == "face" else ""
            if not neighbor or neighbor not in allowed_adjacent_masks:
                continue
            neighbor_mask = (
                np.asarray(allowed_adjacent_masks[neighbor], dtype=np.float32)
                >= float(mask_threshold)
            )
            if neighbor_mask.shape != target.shape:
                raise ValueError(f"{neighbor} support mask must match target_mask shape")
            support |= neighbor_mask & make_boundary_band(
                target,
                radius=1,
                threshold=float(mask_threshold),
            )
            continue
        raise ValueError(f"unsupported region_support_mode: {mode}")
    return support


def compute_projected_leakage_for_selection(
    *,
    part: str,
    mode: str,
    view_name: str,
    px,
    py,
    selected,
    weights,
    target_mask,
    valid_mask,
    boundary_radius: int = 2,
    mask_threshold: float = 0.5,
) -> dict:
    px = np.asarray(px, dtype=np.int64).reshape(-1)
    py = np.asarray(py, dtype=np.int64).reshape(-1)
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    target = np.asarray(target_mask, dtype=np.float32) >= float(mask_threshold)
    valid = np.asarray(valid_mask, dtype=np.float32) >= float(mask_threshold)
    if not (px.shape == py.shape == selected.shape == weights.shape):
        raise ValueError("px, py, selected, and weights must have matching shapes")
    if target.shape != valid.shape:
        raise ValueError("target_mask and valid_mask must have matching shapes")

    height, width = target.shape
    in_image = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    active = selected & in_image
    active_idx = np.nonzero(active)[0]
    sample_target = np.zeros((px.shape[0],), dtype=bool)
    sample_valid = np.zeros((px.shape[0],), dtype=bool)
    sample_boundary = np.zeros((px.shape[0],), dtype=bool)
    if active_idx.size:
        band = make_boundary_band(target, radius=boundary_radius, threshold=0.5)
        sample_target[active_idx] = target[py[active_idx], px[active_idx]]
        sample_valid[active_idx] = valid[py[active_idx], px[active_idx]]
        sample_boundary[active_idx] = band[py[active_idx], px[active_idx]]

    target_active = active & sample_valid & sample_target
    outer_active = active & sample_valid & ~sample_target
    boundary_active = active & sample_valid & sample_boundary
    invalid_active = active & ~sample_valid
    target_activation = float(weights[target_active].sum())
    outer_activation = float(weights[outer_active].sum())
    boundary_activation = float(weights[boundary_active].sum())
    return {
        "part": str(part),
        "mode": str(mode),
        "view": str(view_name),
        "selected_count": int(np.sum(selected)),
        "projected_selected_count": int(np.sum(active)),
        "invalid_projection_count": int(np.sum(selected & ~in_image)),
        "invalid_mask_count": int(np.sum(invalid_active)),
        "target_pixel_count": int(np.sum(target & valid)),
        "target_activation": target_activation,
        "outer_activation": outer_activation,
        "boundary_activation": boundary_activation,
        "leakage_ratio": _safe_ratio(outer_activation, target_activation),
        "boundary_leakage_ratio": _safe_ratio(boundary_activation, target_activation),
        "target_coverage": _safe_ratio(float(np.sum(target_active)), float(np.sum(target & valid))),
    }


def compute_footprint_leakage_for_selection(
    *,
    part: str,
    mode: str,
    view_name: str,
    xy,
    selected,
    weights,
    radii,
    target_mask,
    valid_mask,
    mask_threshold: float = 0.5,
    footprint_radius_scale: float = 1.0,
    min_footprint_radius: int = 1,
    max_footprint_radius: int = 12,
    boundary_radius: int = 2,
    use_soft_target: bool = False,
    allowed_adjacent_masks: dict[str, np.ndarray] | None = None,
    region_support_mode: str = "none",
) -> dict:
    xy_np = np.asarray(xy, dtype=np.float32)
    if xy_np.ndim != 2 or xy_np.shape[1] != 2:
        raise ValueError("xy must have shape [N, 2]")
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    radii = np.asarray(radii, dtype=np.float32).reshape(-1)
    if not (selected.shape[0] == weights.shape[0] == radii.shape[0] == xy_np.shape[0]):
        raise ValueError("xy, selected, weights, and radii must have matching point counts")
    target_values = np.asarray(target_mask, dtype=np.float32)
    if use_soft_target:
        target = target_values > 0.0
    else:
        target = target_values >= float(mask_threshold)
    valid = np.asarray(valid_mask, dtype=np.float32) >= float(mask_threshold)
    if target.shape != valid.shape:
        raise ValueError("target_mask and valid_mask must have matching shapes")

    height, width = target.shape
    px = np.rint(xy_np[:, 0]).astype(np.int64)
    py = np.rint(xy_np[:, 1]).astype(np.int64)
    in_image = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    active = selected & in_image & (radii > 0.0)
    min_radius = max(0, int(min_footprint_radius))
    max_radius = max(min_radius, int(max_footprint_radius))
    target_activation = 0.0
    outer_activation = 0.0
    boundary_activation = 0.0
    allowed_adjacent_activation = 0.0
    actionable_outer_activation = 0.0
    observed_count = 0
    boundary = make_boundary_band(
        target,
        radius=int(boundary_radius),
        threshold=float(mask_threshold),
    )
    adjacent_masks = []
    for adjacent_mask in (allowed_adjacent_masks or {}).values():
        adjacent = np.asarray(adjacent_mask, dtype=np.float32) >= float(mask_threshold)
        if adjacent.shape != valid.shape:
            raise ValueError("allowed adjacent masks must match target_mask shape")
        adjacent_masks.append(adjacent)
    region_support = region_support_mask(
        part=part,
        target_mask=target_mask,
        allowed_adjacent_masks=allowed_adjacent_masks,
        mask_threshold=float(mask_threshold),
        region_support_mode=str(region_support_mode),
    )
    for point_idx in np.nonzero(active)[0]:
        x = int(px[point_idx])
        y = int(py[point_idx])
        radius = int(np.ceil(float(radii[point_idx]) * float(footprint_radius_scale)))
        radius = max(min_radius, min(max_radius, radius))
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = ((yy - y) * (yy - y) + (xx - x) * (xx - x)) <= (radius * radius)
        support = disk & valid[y0:y1, x0:x1]
        support_count = int(np.sum(support))
        if support_count <= 0:
            continue
        observed_count += 1
        if use_soft_target:
            local_target_values = np.clip(target_values[y0:y1, x0:x1], 0.0, 1.0)
            target_ratio = float(np.sum(local_target_values[support])) / float(support_count)
        else:
            target_ratio = float(np.sum(support & target[y0:y1, x0:x1])) / float(support_count)
        boundary_ratio = float(np.sum(support & boundary[y0:y1, x0:x1])) / float(support_count)
        adjacent_ratio = 0.0
        for adjacent in adjacent_masks:
            adjacent_ratio = max(
                adjacent_ratio,
                float(np.sum(support & adjacent[y0:y1, x0:x1])) / float(support_count),
            )
        region_support_ratio = float(
            np.sum(support & region_support[y0:y1, x0:x1])
        ) / float(support_count)
        outer_ratio = max(0.0, 1.0 - target_ratio)
        boundary_allowed_ratio = (
            min(outer_ratio, adjacent_ratio) if boundary_ratio > 0.0 else 0.0
        )
        allowed_ratio = min(
            outer_ratio,
            max(boundary_allowed_ratio, region_support_ratio),
        )
        actionable_ratio = max(0.0, outer_ratio - allowed_ratio)
        weight = float(weights[point_idx])
        target_activation += weight * target_ratio
        outer_activation += weight * outer_ratio
        boundary_activation += weight * boundary_ratio
        allowed_adjacent_activation += weight * allowed_ratio
        actionable_outer_activation += weight * actionable_ratio
    return {
        "part": str(part),
        "mode": str(mode),
        "view": str(view_name),
        "selected_count": int(np.sum(selected)),
        "observed_footprint_count": int(observed_count),
        "target_activation": float(target_activation),
        "outer_activation": float(outer_activation),
        "boundary_activation": float(boundary_activation),
        "allowed_adjacent_activation": float(allowed_adjacent_activation),
        "actionable_outer_activation": float(actionable_outer_activation),
        "leakage_ratio": _safe_ratio(outer_activation, target_activation),
        "boundary_leakage_ratio": _safe_ratio(boundary_activation, target_activation),
        "allowed_adjacent_leakage_ratio": _safe_ratio(
            allowed_adjacent_activation,
            target_activation,
        ),
        "actionable_leakage_ratio": _safe_ratio(
            actionable_outer_activation,
            target_activation,
        ),
    }


def _sum(rows, key):
    return float(sum(float(row.get(key, 0.0)) for row in rows))


def summarize_rows(rows: list[dict], *, soft_threshold: float) -> dict:
    rows = list(rows)
    parts = sorted({row["part"] for row in rows})
    per_part = []
    for part in parts:
        part_rows = [row for row in rows if row["part"] == part]
        out = {"part": part}
        for mode in ("hard", "soft", "hard_footprint", "soft_footprint"):
            mode_rows = [row for row in part_rows if row["mode"] == mode]
            if not mode_rows and mode.endswith("_footprint"):
                continue
            target = _sum(mode_rows, "target_activation")
            outer = _sum(mode_rows, "outer_activation")
            boundary = _sum(mode_rows, "boundary_activation")
            out[f"{mode}_selected_count"] = int(sum(int(row.get("selected_count", 0)) for row in mode_rows))
            out[f"{mode}_target_activation"] = target
            out[f"{mode}_outer_activation"] = outer
            out[f"{mode}_boundary_activation"] = boundary
            out[f"{mode}_leakage_ratio"] = _safe_ratio(outer, target)
            out[f"{mode}_boundary_leakage_ratio"] = _safe_ratio(boundary, target)
        out["leakage_delta_soft_minus_hard"] = out["soft_leakage_ratio"] - out["hard_leakage_ratio"]
        out["boundary_leakage_delta_soft_minus_hard"] = (
            out["soft_boundary_leakage_ratio"] - out["hard_boundary_leakage_ratio"]
        )
        per_part.append(out)
    hard = np.array([row["hard_leakage_ratio"] for row in per_part], dtype=np.float32)
    soft = np.array([row["soft_leakage_ratio"] for row in per_part], dtype=np.float32)
    hard_boundary = np.array([row["hard_boundary_leakage_ratio"] for row in per_part], dtype=np.float32)
    soft_boundary = np.array([row["soft_boundary_leakage_ratio"] for row in per_part], dtype=np.float32)
    summary = {
        "part_count": int(len(per_part)),
        "view_row_count": int(len(rows)),
        "soft_threshold": float(soft_threshold),
        "mean_hard_leakage_ratio": float(hard.mean()) if hard.size else 0.0,
        "mean_soft_leakage_ratio": float(soft.mean()) if soft.size else 0.0,
        "mean_leakage_delta_soft_minus_hard": float((soft - hard).mean()) if hard.size else 0.0,
        "mean_hard_boundary_leakage_ratio": float(hard_boundary.mean()) if hard_boundary.size else 0.0,
        "mean_soft_boundary_leakage_ratio": float(soft_boundary.mean()) if soft_boundary.size else 0.0,
        "mean_boundary_leakage_delta_soft_minus_hard": (
            float((soft_boundary - hard_boundary).mean()) if hard_boundary.size else 0.0
        ),
    }
    if per_part and all("soft_footprint_leakage_ratio" in row for row in per_part):
        hard_fp = np.array([row["hard_footprint_leakage_ratio"] for row in per_part], dtype=np.float32)
        soft_fp = np.array([row["soft_footprint_leakage_ratio"] for row in per_part], dtype=np.float32)
        summary.update(
            {
                "mean_hard_footprint_leakage_ratio": float(hard_fp.mean()) if hard_fp.size else 0.0,
                "mean_soft_footprint_leakage_ratio": float(soft_fp.mean()) if soft_fp.size else 0.0,
                "mean_footprint_leakage_delta_soft_minus_hard": (
                    float((soft_fp - hard_fp).mean()) if hard_fp.size else 0.0
                ),
            }
        )
    return {"summary": summary, "per_part": per_part, "per_view": rows}


def _write_csv(path: Path, rows: list[dict], preferred: list[str]) -> None:
    keys = {key for row in rows for key in row.keys()}
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys - set(fieldnames)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(output_dir: Path | str, result: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(result["summary"], indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(
        output_dir / "per_part.csv",
        list(result["per_part"]),
        [
            "part",
            "soft_leakage_ratio",
            "hard_leakage_ratio",
            "leakage_delta_soft_minus_hard",
            "soft_boundary_leakage_ratio",
            "hard_boundary_leakage_ratio",
            "boundary_leakage_delta_soft_minus_hard",
        ],
    )
    _write_csv(
        output_dir / "per_view.csv",
        list(result["per_view"]),
        ["part", "mode", "view", "leakage_ratio", "boundary_leakage_ratio"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure hard-vs-soft semantic edit leakage in projected 2D mask space.")
    parser.add_argument("--part-label-bank", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--parts", nargs="+", default=list(PART_NAMES), choices=list(PART_NAMES))
    parser.add_argument("--soft-threshold", type=float, default=0.25)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--soft-boundary-radius", type=int, default=0)
    parser.add_argument("--soft-boundary-min-value", type=float, default=0.25)
    parser.add_argument("--depth-margin", type=float, default=0.02)
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--subject", default="")
    parser.add_argument("--explicit-binding-render-preset", default="v338_temporal_selector_grow_only_guard")
    return parser.parse_args()


def analyze_scene(args: argparse.Namespace) -> dict:
    import torch
    from scene import GaussianModel, Scene
    from tools.semantic_viewer.build_part_label_bank import (
        _find_dataset_index,
        _load_config,
        _load_record_masks,
        _load_view_records,
        _project_points,
        _select_records,
    )

    bank = load_part_label_bank(args.part_label_bank)
    labels = np.asarray(bank["part_label"], dtype=np.int16).reshape(-1)
    editable = np.asarray(bank.get("editable_label", labels), dtype=np.int16).reshape(-1)
    weights, weight_source = resolve_soft_edit_weights(bank, point_count=labels.shape[0])

    asset_root = args.asset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    config_path = args.config.resolve() if args.config else asset_root.parent.parent / ".hydra" / "config.yaml"
    records = _select_records(_load_view_records(asset_root), args.max_views)
    config = _load_config(config_path, checkpoint, asset_root, records, args)
    rows = []
    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, str(asset_root.parent))
        scene.eval()
        loaded_iteration = int(scene.load_checkpoint(str(checkpoint)))
        point_count = int(scene.gaussians.get_xyz.shape[0])
        if labels.shape[0] != point_count:
            raise ValueError(f"label bank point count {labels.shape[0]} does not match scene {point_count}")
        if weights.shape != (point_count, len(PART_NAMES)):
            raise ValueError(f"resolved soft weights must have shape ({point_count}, {len(PART_NAMES)})")
        for record in records:
            dataset_index = _find_dataset_index(scene.test_dataset, record["image_name"])
            if dataset_index is None:
                raise RuntimeError(f"image {record['image_name']} not present in dataset")
            view = scene.test_dataset[dataset_index]
            deformed_gaussian, _, colors_precomp = scene.convert_gaussians(view, loaded_iteration, compute_loss=False)
            xy, proj_valid, depth = _project_points(deformed_gaussian.get_xyz, view)
            xy_np = xy.detach().float().cpu().numpy()
            px = np.rint(xy_np[:, 0]).astype(np.int64)
            py = np.rint(xy_np[:, 1]).astype(np.int64)
            proj = proj_valid.detach().bool().cpu().numpy()
            radii = np.ones((point_count,), dtype=np.float32)
            render_pkg = None
            try:
                from gaussian_renderer import render

                background = torch.zeros(3, dtype=torch.float32, device="cuda")
                render_pkg = render(
                    view,
                    loaded_iteration,
                    scene,
                    config.pipeline,
                    background,
                    compute_loss=False,
                    return_opacity=False,
                )
                radii = render_pkg["radii"].detach().float().cpu().numpy().reshape(-1)[:point_count]
            except Exception:
                radii = np.ones((point_count,), dtype=np.float32)
            part_masks, foreground_mask, valid_mask = _load_record_masks(asset_root, record)
            combined_valid = np.minimum(np.asarray(foreground_mask, dtype=np.float32), np.asarray(valid_mask, dtype=np.float32))
            for part in args.parts:
                target = PART_NAMES.index(part)
                hard_selected = (editable == target) & proj
                soft_values = weights[:, target]
                soft_selected = (soft_values >= float(args.soft_threshold)) & proj
                footprint_target_mask = build_soft_boundary_target_mask(
                    part_masks[part],
                    radius=int(args.soft_boundary_radius),
                    threshold=float(args.mask_threshold),
                    min_boundary_value=float(args.soft_boundary_min_value),
                )
                rows.append(
                    compute_projected_leakage_for_selection(
                        part=part,
                        mode="hard",
                        view_name=str(record["image_name"]),
                        px=px,
                        py=py,
                        selected=hard_selected,
                        weights=np.ones_like(soft_values, dtype=np.float32),
                        target_mask=part_masks[part],
                        valid_mask=combined_valid,
                        boundary_radius=int(args.boundary_radius),
                        mask_threshold=float(args.mask_threshold),
                    )
                )
                rows.append(
                    compute_footprint_leakage_for_selection(
                        part=part,
                        mode="hard_footprint",
                        view_name=str(record["image_name"]),
                        xy=xy_np,
                        selected=hard_selected,
                        weights=np.ones_like(soft_values, dtype=np.float32),
                        radii=radii,
                        target_mask=footprint_target_mask,
                        valid_mask=combined_valid,
                        mask_threshold=float(args.mask_threshold),
                        use_soft_target=int(args.soft_boundary_radius) > 0,
                    )
                )
                rows.append(
                    compute_projected_leakage_for_selection(
                        part=part,
                        mode="soft",
                        view_name=str(record["image_name"]),
                        px=px,
                        py=py,
                        selected=soft_selected,
                        weights=soft_values,
                        target_mask=part_masks[part],
                        valid_mask=combined_valid,
                        boundary_radius=int(args.boundary_radius),
                        mask_threshold=float(args.mask_threshold),
                    )
                )
                rows.append(
                    compute_footprint_leakage_for_selection(
                        part=part,
                        mode="soft_footprint",
                        view_name=str(record["image_name"]),
                        xy=xy_np,
                        selected=soft_selected,
                        weights=soft_values,
                        radii=radii,
                        target_mask=footprint_target_mask,
                        valid_mask=combined_valid,
                        mask_threshold=float(args.mask_threshold),
                        use_soft_target=int(args.soft_boundary_radius) > 0,
                    )
                )
            del colors_precomp
            del deformed_gaussian
            if render_pkg is not None:
                del render_pkg
            torch.cuda.empty_cache()
    result = summarize_rows(rows, soft_threshold=float(args.soft_threshold))
    result["summary"]["part_label_bank"] = str(args.part_label_bank)
    result["summary"]["checkpoint"] = str(checkpoint)
    result["summary"]["asset_root"] = str(asset_root)
    result["summary"]["processed_views"] = int(len(records))
    result["summary"]["boundary_radius"] = int(args.boundary_radius)
    result["summary"]["soft_boundary_radius"] = int(args.soft_boundary_radius)
    result["summary"]["soft_boundary_min_value"] = float(args.soft_boundary_min_value)
    result["summary"]["soft_edit_weight_source"] = str(weight_source)
    return result


def main() -> int:
    args = parse_args()
    result = analyze_scene(args)
    write_reports(args.output_dir, result)
    print(f"wrote {args.output_dir / 'summary.json'}")
    print(f"wrote {args.output_dir / 'per_part.csv'}")
    print(f"wrote {args.output_dir / 'per_view.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
