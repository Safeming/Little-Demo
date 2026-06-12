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

from utils.part_label_bank import PART_NAMES, load_part_label_bank, save_part_label_bank


def _copy_array(value):
    return np.asarray(value).copy()


def _scalar_str(bank: dict, key: str, default: str = "") -> str:
    if key not in bank:
        return default
    return str(np.asarray(bank[key]).item())


def _scalar_int(bank: dict, key: str, default: int = 0) -> int:
    if key not in bank:
        return int(default)
    return int(np.asarray(bank[key]).item())


def _require_soft_weights(bank: dict) -> np.ndarray:
    if "soft_edit_weights" not in bank:
        raise ValueError("part label bank does not contain soft_edit_weights")
    weights = np.asarray(bank["soft_edit_weights"], dtype=np.float32)
    if weights.ndim != 2 or weights.shape[1] != len(PART_NAMES):
        raise ValueError(f"soft_edit_weights must have shape [N, {len(PART_NAMES)}]")
    return weights


def _validate_part_names(bank: dict) -> None:
    if "part_names" not in bank:
        return
    part_names = [str(x) for x in np.asarray(bank["part_names"]).tolist()]
    if part_names != list(PART_NAMES):
        raise ValueError(f"part_names mismatch: {part_names}")


def _validate_part(part_name: str) -> int:
    if part_name not in PART_NAMES:
        raise ValueError(f"unknown part: {part_name}")
    return PART_NAMES.index(part_name)


def _as_bool_vector(record: dict, key: str, point_count: int) -> np.ndarray:
    if key not in record:
        raise ValueError(f"leakage record missing key: {key}")
    value = np.asarray(record[key], dtype=bool).reshape(-1)
    if value.shape[0] != int(point_count):
        raise ValueError(f"{key} shape mismatch: got {value.shape[0]}, expected {int(point_count)}")
    return value


def compute_point_leakage_stats(records, soft_edit_weights, *, part_index: int) -> dict[str, np.ndarray]:
    weights = np.asarray(soft_edit_weights, dtype=np.float32)
    if weights.ndim != 2 or weights.shape[1] <= int(part_index):
        raise ValueError("soft_edit_weights shape does not contain requested part_index")
    point_count = int(weights.shape[0])
    part_weight = weights[:, int(part_index)].astype(np.float32, copy=False)

    observed_count = np.zeros((point_count,), dtype=np.int32)
    target_count = np.zeros((point_count,), dtype=np.int32)
    outer_count = np.zeros((point_count,), dtype=np.int32)
    boundary_count = np.zeros((point_count,), dtype=np.int32)
    target_weight_sum = np.zeros((point_count,), dtype=np.float32)
    outer_weight_sum = np.zeros((point_count,), dtype=np.float32)
    boundary_weight_sum = np.zeros((point_count,), dtype=np.float32)

    for record in records:
        observed = _as_bool_vector(record, "observed", point_count)
        target = _as_bool_vector(record, "target", point_count) & observed
        outer = _as_bool_vector(record, "outer", point_count) & observed
        boundary = _as_bool_vector(record, "boundary", point_count) & observed
        observed_count += observed.astype(np.int32)
        target_count += target.astype(np.int32)
        outer_count += outer.astype(np.int32)
        boundary_count += boundary.astype(np.int32)
        target_weight_sum += np.where(target, part_weight, 0.0).astype(np.float32)
        outer_weight_sum += np.where(outer, part_weight, 0.0).astype(np.float32)
        boundary_weight_sum += np.where(boundary, part_weight, 0.0).astype(np.float32)

    denom = np.maximum(observed_count.astype(np.float32), 1.0)
    target_ratio = target_count.astype(np.float32) / denom
    outer_ratio = outer_count.astype(np.float32) / denom
    boundary_ratio = boundary_count.astype(np.float32) / denom
    return {
        "observed_view_count": observed_count,
        "target_hit_count": target_count,
        "outer_hit_count": outer_count,
        "boundary_hit_count": boundary_count,
        "target_weight_sum": target_weight_sum,
        "outer_weight_sum": outer_weight_sum,
        "boundary_weight_sum": boundary_weight_sum,
        "target_hit_ratio": target_ratio.astype(np.float32, copy=False),
        "outer_hit_ratio": outer_ratio.astype(np.float32, copy=False),
        "boundary_hit_ratio": boundary_ratio.astype(np.float32, copy=False),
        "stable_leak_score": (outer_ratio - target_ratio).astype(np.float32, copy=False),
    }


def classify_suppression_mask(
    stats: dict[str, np.ndarray],
    part_weights,
    *,
    soft_threshold: float,
    min_observed_views: int,
    min_outer_views: int,
    max_target_hit_ratio: float,
    min_outer_hit_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(part_weights, dtype=np.float32).reshape(-1)
    observed = np.asarray(stats["observed_view_count"], dtype=np.int32).reshape(-1)
    outer_count = np.asarray(stats["outer_hit_count"], dtype=np.int32).reshape(-1)
    boundary_count = np.asarray(stats["boundary_hit_count"], dtype=np.int32).reshape(-1)
    target_ratio = np.asarray(stats["target_hit_ratio"], dtype=np.float32).reshape(-1)
    outer_ratio = np.asarray(stats["outer_hit_ratio"], dtype=np.float32).reshape(-1)
    boundary_ratio = np.asarray(
        stats.get("boundary_hit_ratio", boundary_count / np.maximum(observed, 1).astype(np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    if not (weights.shape == observed.shape == outer_count.shape == target_ratio.shape == outer_ratio.shape):
        raise ValueError("stats and part_weights point counts must match")

    enough_views = observed >= int(min_observed_views)
    low_target = target_ratio <= float(max_target_hit_ratio)
    severe = (
        (weights >= float(soft_threshold))
        & enough_views
        & (outer_count >= int(min_outer_views))
        & (outer_ratio >= float(min_outer_hit_ratio))
        & low_target
    )
    boundary = (
        ~severe
        & enough_views
        & low_target
        & (boundary_count >= int(min_outer_views))
        & (boundary_ratio >= float(min_outer_hit_ratio))
    )
    return severe.astype(bool, copy=False), boundary.astype(bool, copy=False)


def apply_soft_weight_suppression(
    soft_edit_weights,
    *,
    part_name: str,
    severe_mask,
    boundary_mask,
    suppress_factor: float,
    boundary_cap: float,
) -> tuple[np.ndarray, dict]:
    part_index = _validate_part(part_name)
    weights = np.asarray(soft_edit_weights, dtype=np.float32)
    if weights.ndim != 2 or weights.shape[1] != len(PART_NAMES):
        raise ValueError(f"soft_edit_weights must have shape [N, {len(PART_NAMES)}]")
    severe = np.asarray(severe_mask, dtype=bool).reshape(-1)
    boundary = np.asarray(boundary_mask, dtype=bool).reshape(-1) & ~severe
    if severe.shape[0] != weights.shape[0] or boundary.shape[0] != weights.shape[0]:
        raise ValueError("suppression masks must match soft_edit_weights point count")

    updated = weights.copy()
    before = updated[:, part_index].copy()
    updated[severe, part_index] = before[severe] * float(suppress_factor)
    updated[boundary, part_index] = np.minimum(updated[boundary, part_index], float(boundary_cap))
    after = updated[:, part_index]
    changed = np.abs(after - before) > 1.0e-8
    summary = {
        "part": str(part_name),
        "part_index": int(part_index),
        "severe_suppressed_count": int(np.sum(severe)),
        "boundary_capped_count": int(np.sum(boundary & changed)),
        "changed_count": int(np.sum(changed)),
        "old_weight_sum": float(np.sum(before)),
        "new_weight_sum": float(np.sum(after)),
        "removed_weight_sum": float(np.sum(before - after)),
        "suppress_factor": float(suppress_factor),
        "boundary_cap": float(boundary_cap),
    }
    return updated.astype(np.float32, copy=False), summary


def save_suppressed_bank(path: Path, bank: dict, soft_edit_weights, *, source_type: str = "leak_suppressed") -> None:
    _validate_part_names(bank)
    base_weights = _require_soft_weights(bank)
    updated = np.asarray(soft_edit_weights, dtype=np.float32)
    if updated.shape != base_weights.shape:
        raise ValueError(
            "updated soft_edit_weights shape mismatch: "
            f"{updated.shape} != {base_weights.shape}"
        )
    save_part_label_bank(
        path,
        part_label=bank["part_label"],
        confidence=bank["confidence"],
        vote_count=bank["vote_count"],
        per_part_votes=bank["per_part_votes"],
        visible_vote_count=bank["visible_vote_count"],
        conflict_count=bank["conflict_count"],
        source_checkpoint=_scalar_str(bank, "source_checkpoint"),
        source_asset_root=_scalar_str(bank, "source_asset_root"),
        source_iteration=_scalar_int(bank, "source_iteration"),
        semantic_probs=bank.get("semantic_probs"),
        semantic_margin=bank.get("semantic_margin"),
        reliable_mask=bank.get("reliable_mask"),
        editable_label=bank.get("editable_label"),
        soft_edit_weights=updated,
        neighbor_fill_mask=bank.get("neighbor_fill_mask"),
        source_type=str(source_type),
    )


def _sample_binary_mask(mask: np.ndarray, px: np.ndarray, py: np.ndarray, valid: np.ndarray, threshold: float) -> np.ndarray:
    out = np.zeros_like(valid, dtype=bool)
    if not np.any(valid):
        return out
    out[valid] = np.asarray(mask, dtype=np.float32)[py[valid], px[valid]] >= float(threshold)
    return out


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if int(radius) <= 0:
        return np.asarray(mask, dtype=bool)
    try:
        import cv2

        kernel = np.ones((int(radius) * 2 + 1, int(radius) * 2 + 1), dtype=np.uint8)
        return cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel, iterations=1).astype(bool)
    except Exception:
        out = np.asarray(mask, dtype=bool)
        for _ in range(int(radius)):
            padded = np.pad(out, 1, mode="constant", constant_values=False)
            out = (
                padded[1:-1, 1:-1]
                | padded[:-2, 1:-1]
                | padded[2:, 1:-1]
                | padded[1:-1, :-2]
                | padded[1:-1, 2:]
                | padded[:-2, :-2]
                | padded[:-2, 2:]
                | padded[2:, :-2]
                | padded[2:, 2:]
            )
        return out


def build_view_leakage_record(
    *,
    xy,
    proj_valid,
    visibility_filter,
    radii,
    image_size: tuple[int, int],
    part_masks: dict[str, np.ndarray],
    foreground_mask: np.ndarray,
    valid_mask: np.ndarray,
    part_name: str,
    mask_threshold: float,
    boundary_radius: int,
) -> dict[str, np.ndarray]:
    part_index = _validate_part(part_name)
    if hasattr(xy, "detach"):
        xy_np = xy.detach().float().cpu().numpy()
    else:
        xy_np = np.asarray(xy, dtype=np.float32)
    point_count = int(xy_np.shape[0])
    if hasattr(proj_valid, "detach"):
        proj_np = proj_valid.detach().bool().cpu().numpy().reshape(-1)
    else:
        proj_np = np.asarray(proj_valid, dtype=bool).reshape(-1)
    if hasattr(visibility_filter, "detach"):
        vis_np = visibility_filter.detach().bool().cpu().numpy().reshape(-1)
    else:
        vis_np = np.asarray(visibility_filter, dtype=bool).reshape(-1)
    if hasattr(radii, "detach"):
        radii_np = radii.detach().float().cpu().numpy().reshape(-1)
    else:
        radii_np = np.asarray(radii, dtype=np.float32).reshape(-1)
    if proj_np.shape[0] != point_count:
        raise ValueError("projection array must match xy point count")
    if vis_np.shape[0] < point_count or radii_np.shape[0] < point_count:
        raise ValueError("visibility and radii arrays must cover xy point count")
    if vis_np.shape[0] > point_count:
        vis_np = vis_np[:point_count]
    if radii_np.shape[0] > point_count:
        radii_np = radii_np[:point_count]

    width, height = int(image_size[0]), int(image_size[1])
    px = np.rint(xy_np[:, 0]).astype(np.int64)
    py = np.rint(xy_np[:, 1]).astype(np.int64)
    in_image = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    rendered = proj_np & vis_np & (radii_np > 0.0) & in_image
    fg_hit = _sample_binary_mask(foreground_mask, px, py, rendered, mask_threshold)
    valid_hit = _sample_binary_mask(valid_mask, px, py, rendered, mask_threshold)
    observed = rendered & fg_hit & valid_hit

    target_mask = np.asarray(part_masks[PART_NAMES[part_index]], dtype=np.float32) >= float(mask_threshold)
    if target_mask.shape != (height, width):
        raise ValueError(f"{part_name} mask shape {target_mask.shape} does not match image {(height, width)}")
    target = _sample_binary_mask(target_mask.astype(np.float32), px, py, observed, 0.5)
    outer = observed & ~target
    boundary_mask = _dilate_mask(target_mask, int(boundary_radius)) & ~target_mask
    boundary = _sample_binary_mask(boundary_mask.astype(np.float32), px, py, observed, 0.5)
    return {
        "observed": observed,
        "target": target,
        "outer": outer,
        "boundary": boundary,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_per_point_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _per_point_rows(part_name: str, stats: dict[str, np.ndarray], weights: np.ndarray, severe: np.ndarray, boundary: np.ndarray) -> list[dict]:
    rows = []
    part_index = _validate_part(part_name)
    part_weights = weights[:, part_index]
    changed = severe | boundary
    for point_idx in np.nonzero(changed)[0].tolist():
        rows.append(
            {
                "part": part_name,
                "point_idx": int(point_idx),
                "old_weight": float(part_weights[point_idx]),
                "severe_leak": int(bool(severe[point_idx])),
                "boundary_dominated": int(bool(boundary[point_idx])),
                "observed_view_count": int(stats["observed_view_count"][point_idx]),
                "target_hit_count": int(stats["target_hit_count"][point_idx]),
                "outer_hit_count": int(stats["outer_hit_count"][point_idx]),
                "boundary_hit_count": int(stats["boundary_hit_count"][point_idx]),
                "target_hit_ratio": float(stats["target_hit_ratio"][point_idx]),
                "outer_hit_ratio": float(stats["outer_hit_ratio"][point_idx]),
                "boundary_hit_ratio": float(stats["boundary_hit_ratio"][point_idx]),
                "stable_leak_score": float(stats["stable_leak_score"][point_idx]),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suppress stable multi-view projected soft-edit leakage in selected part channels.")
    parser.add_argument("--part-label-bank", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parts", nargs="+", required=True, choices=list(PART_NAMES))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--subject", default="")
    parser.add_argument("--explicit-binding-render-preset", default="v338_temporal_selector_grow_only_guard")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--soft-threshold", type=float, default=0.20)
    parser.add_argument("--min-observed-views", type=int, default=5)
    parser.add_argument("--min-outer-views", type=int, default=3)
    parser.add_argument("--max-target-hit-ratio", type=float, default=0.35)
    parser.add_argument("--min-outer-hit-ratio", type=float, default=0.55)
    parser.add_argument("--suppress-factor", type=float, default=0.25)
    parser.add_argument("--boundary-cap", type=float, default=0.30)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--per-point-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    import torch
    from gaussian_renderer import render
    from scene import GaussianModel, Scene
    from tools.semantic_viewer.build_part_label_bank import (
        _find_dataset_index,
        _load_config,
        _load_record_masks,
        _load_view_records,
        _project_points,
        _select_records,
    )

    args = parse_args()
    bank = load_part_label_bank(args.part_label_bank)
    _validate_part_names(bank)
    weights = _require_soft_weights(bank)
    updated_weights = weights.copy()
    asset_root = args.asset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    records = _select_records(_load_view_records(asset_root), args.max_views)
    config_path = args.config.resolve() if args.config else asset_root.parent.parent / ".hydra" / "config.yaml"
    config = _load_config(config_path, checkpoint, asset_root, records, args)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    part_records = {part: [] for part in args.parts}
    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, str(asset_root.parent))
        scene.eval()
        loaded_iteration = int(scene.load_checkpoint(str(checkpoint)))
        point_count = int(scene.gaussians.get_xyz.shape[0])
        if point_count != int(weights.shape[0]):
            raise ValueError(f"checkpoint point count {point_count} does not match bank point count {weights.shape[0]}")
        for record in records:
            dataset_index = _find_dataset_index(scene.test_dataset, record["image_name"])
            if dataset_index is None:
                raise RuntimeError(f"image {record['image_name']} not present in dataset")
            view = scene.test_dataset[dataset_index]
            pkg = render(view, loaded_iteration, scene, config.pipeline, background, compute_loss=False, return_opacity=False)
            deformed = pkg["deformed_gaussian"]
            xy, proj_valid, _depth = _project_points(deformed.get_xyz, view)
            part_masks, foreground_mask, valid_mask = _load_record_masks(asset_root, record)
            image_size = (int(view.image_width), int(view.image_height))
            for part in args.parts:
                part_records[part].append(
                    build_view_leakage_record(
                        xy=xy,
                        proj_valid=proj_valid,
                        visibility_filter=pkg["visibility_filter"],
                        radii=pkg["radii"],
                        image_size=image_size,
                        part_masks=part_masks,
                        foreground_mask=foreground_mask,
                        valid_mask=valid_mask,
                        part_name=part,
                        mask_threshold=float(args.mask_threshold),
                        boundary_radius=int(args.boundary_radius),
                    )
                )
            del pkg, deformed
            torch.cuda.empty_cache()

    summary = {
        "mode": "projected_soft_leak_suppression",
        "part_label_bank": str(args.part_label_bank),
        "checkpoint": str(checkpoint),
        "asset_root": str(asset_root),
        "output": str(args.output),
        "parts": list(args.parts),
        "processed_views": len(records),
        "point_count": int(weights.shape[0]),
        "parameters": {
            "soft_threshold": float(args.soft_threshold),
            "min_observed_views": int(args.min_observed_views),
            "min_outer_views": int(args.min_outer_views),
            "max_target_hit_ratio": float(args.max_target_hit_ratio),
            "min_outer_hit_ratio": float(args.min_outer_hit_ratio),
            "suppress_factor": float(args.suppress_factor),
            "boundary_cap": float(args.boundary_cap),
            "mask_threshold": float(args.mask_threshold),
            "boundary_radius": int(args.boundary_radius),
        },
        "part_summaries": {},
    }
    per_point_rows = []
    for part in args.parts:
        part_index = _validate_part(part)
        stats = compute_point_leakage_stats(part_records[part], updated_weights, part_index=part_index)
        severe, boundary = classify_suppression_mask(
            stats,
            updated_weights[:, part_index],
            soft_threshold=float(args.soft_threshold),
            min_observed_views=int(args.min_observed_views),
            min_outer_views=int(args.min_outer_views),
            max_target_hit_ratio=float(args.max_target_hit_ratio),
            min_outer_hit_ratio=float(args.min_outer_hit_ratio),
        )
        per_point_rows.extend(_per_point_rows(part, stats, updated_weights, severe, boundary))
        updated_weights, part_summary = apply_soft_weight_suppression(
            updated_weights,
            part_name=part,
            severe_mask=severe,
            boundary_mask=boundary,
            suppress_factor=float(args.suppress_factor),
            boundary_cap=float(args.boundary_cap),
        )
        summary["part_summaries"][part] = part_summary

    source_type = "leak_suppressed_" + "_".join(str(part) for part in args.parts)
    save_suppressed_bank(args.output, bank, updated_weights, source_type=source_type)
    summary["total_changed_count"] = int(np.sum(np.any(np.abs(updated_weights - weights) > 1.0e-8, axis=1)))
    summary["total_removed_weight_sum"] = float(np.sum(weights - updated_weights))
    if args.summary_json is not None:
        _write_json(args.summary_json, summary)
    if args.per_point_csv is not None:
        _write_per_point_csv(args.per_point_csv, per_point_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
