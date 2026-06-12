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

from utils.part_label_bank import (
    PART_NAMES,
    compute_evidence_calibrated_soft_edit_weights,
    load_part_label_bank,
    save_part_label_bank,
)


def _scalar_str(bank: dict, key: str, default: str = "") -> str:
    if key not in bank:
        return default
    return str(np.asarray(bank[key]).item())


def _scalar_int(bank: dict, key: str, default: int = 0) -> int:
    if key not in bank:
        return int(default)
    return int(np.asarray(bank[key]).item())


def _validate_part(part_name: str) -> int:
    if part_name not in PART_NAMES:
        raise ValueError(f"unknown part: {part_name}")
    return PART_NAMES.index(part_name)


def _as_bool_vector(value, point_count: int, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        out = value.detach().bool().cpu().numpy()
    else:
        out = np.asarray(value, dtype=bool)
    out = out.reshape(-1)
    if out.shape[0] < int(point_count):
        raise ValueError(f"{name} has {out.shape[0]} entries, expected at least {point_count}")
    return out[:point_count]


def _as_float_vector(value, point_count: int, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        out = value.detach().float().cpu().numpy()
    else:
        out = np.asarray(value, dtype=np.float32)
    out = out.reshape(-1).astype(np.float32, copy=False)
    if out.shape[0] < int(point_count):
        raise ValueError(f"{name} has {out.shape[0]} entries, expected at least {point_count}")
    return out[:point_count]


def build_part_candidate_mask(
    *,
    soft_edit_weights: np.ndarray,
    editable_label: np.ndarray,
    part_name: str,
    soft_min_weight: float = 0.05,
) -> np.ndarray:
    part_index = _validate_part(part_name)
    weights = np.asarray(soft_edit_weights, dtype=np.float32)
    if weights.ndim != 2 or weights.shape[1] < len(PART_NAMES):
        raise ValueError("soft_edit_weights must have shape [N, C]")
    editable = np.asarray(editable_label, dtype=np.int16).reshape(-1)
    if editable.shape[0] != weights.shape[0]:
        raise ValueError("editable_label point count mismatch")
    return (weights[:, part_index] >= float(soft_min_weight)) | (editable == part_index)


def build_soft_boundary_target_mask(
    mask: np.ndarray,
    *,
    radius: int = 0,
    threshold: float = 0.5,
    min_boundary_value: float = 0.25,
) -> np.ndarray:
    inside = np.asarray(mask, dtype=np.float32) >= float(threshold)
    soft = inside.astype(np.float32)
    radius = max(0, int(radius))
    if radius <= 0 or not np.any(inside):
        return soft
    min_value = float(np.clip(min_boundary_value, 0.0, 1.0))
    height, width = inside.shape
    padded = np.pad(inside, radius, mode="constant", constant_values=False)
    for distance in range(1, radius + 1):
        ring = np.zeros_like(inside, dtype=bool)
        size = 2 * distance + 1
        center = distance
        for dy in range(size):
            for dx in range(size):
                if abs(dy - center) + abs(dx - center) != distance:
                    continue
                y0 = radius + dy - center
                x0 = radius + dx - center
                ring |= padded[y0 : y0 + height, x0 : x0 + width]
        ring &= ~inside
        value = 1.0 - (float(distance) / float(radius)) * (1.0 - min_value)
        soft[ring] = np.maximum(soft[ring], np.float32(value))
    return np.clip(soft, 0.0, 1.0).astype(np.float32, copy=False)


def build_footprint_evidence_record(
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
    candidate_mask=None,
    mask_threshold: float = 0.5,
    soft_boundary_radius: int = 0,
    soft_boundary_min_value: float = 0.25,
    footprint_radius_scale: float = 1.0,
    min_footprint_radius: int = 1,
    max_footprint_radius: int = 12,
) -> dict[str, np.ndarray]:
    part_index = _validate_part(part_name)
    if hasattr(xy, "detach"):
        xy_np = xy.detach().float().cpu().numpy()
    else:
        xy_np = np.asarray(xy, dtype=np.float32)
    if xy_np.ndim != 2 or xy_np.shape[1] != 2:
        raise ValueError("xy must have shape [N, 2]")
    point_count = int(xy_np.shape[0])
    proj = _as_bool_vector(proj_valid, point_count, "proj_valid")
    visible = _as_bool_vector(visibility_filter, point_count, "visibility_filter")
    radii_np = _as_float_vector(radii, point_count, "radii")
    if candidate_mask is None:
        candidate = np.ones((point_count,), dtype=bool)
    else:
        candidate = _as_bool_vector(candidate_mask, point_count, "candidate_mask")

    width, height = int(image_size[0]), int(image_size[1])
    px = np.rint(xy_np[:, 0]).astype(np.int64)
    py = np.rint(xy_np[:, 1]).astype(np.int64)
    in_image = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    rendered = candidate & proj & visible & (radii_np > 0.0) & in_image
    fg = np.asarray(foreground_mask, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=np.float32)
    if fg.shape != (height, width):
        raise ValueError(f"foreground mask shape {fg.shape} does not match {(height, width)}")
    if valid.shape != (height, width):
        raise ValueError(f"valid mask shape {valid.shape} does not match {(height, width)}")
    masks = []
    target_soft = None
    for name in PART_NAMES:
        mask = np.asarray(part_masks[name], dtype=np.float32)
        if mask.shape != (height, width):
            raise ValueError(f"{name} mask shape {mask.shape} does not match {(height, width)}")
        masks.append(mask >= float(mask_threshold))
        if name == part_name:
            target_soft = build_soft_boundary_target_mask(
                mask,
                radius=int(soft_boundary_radius),
                threshold=float(mask_threshold),
                min_boundary_value=float(soft_boundary_min_value),
            )
    if target_soft is None:
        raise ValueError(f"missing target mask for part {part_name}")

    observed = np.zeros((point_count,), dtype=bool)
    target_ratio = np.zeros((point_count,), dtype=np.float32)
    outer_ratio = np.zeros((point_count,), dtype=np.float32)
    conflict_ratio = np.zeros((point_count,), dtype=np.float32)
    min_radius = max(0, int(min_footprint_radius))
    max_radius = max(min_radius, int(max_footprint_radius))
    for point_idx in np.nonzero(rendered)[0]:
        x = int(px[point_idx])
        y = int(py[point_idx])
        radius = int(np.ceil(float(radii_np[point_idx]) * float(footprint_radius_scale)))
        radius = max(min_radius, min(max_radius, radius))
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = ((yy - y) * (yy - y) + (xx - x) * (xx - x)) <= (radius * radius)
        support = (
            disk
            & (fg[y0:y1, x0:x1] >= float(mask_threshold))
            & (valid[y0:y1, x0:x1] >= float(mask_threshold))
        )
        support_count = int(np.sum(support))
        if support_count <= 0:
            continue
        target_values = target_soft[y0:y1, x0:x1]
        any_part_hits = np.zeros_like(support, dtype=bool)
        multi_hits = np.zeros_like(support, dtype=np.int16)
        for mask in masks:
            hit = mask[y0:y1, x0:x1] & support
            any_part_hits |= hit
            multi_hits += hit.astype(np.int16)
        observed[point_idx] = True
        target_value = float(np.sum(target_values[support])) / float(support_count)
        target_value = float(np.clip(target_value, 0.0, 1.0))
        target_ratio[point_idx] = target_value
        outer_ratio[point_idx] = max(0.0, 1.0 - target_value)
        conflict_ratio[point_idx] = float(np.sum((multi_hits > 1) & support)) / float(support_count)

    return {
        "observed": observed,
        "target_ratio": target_ratio,
        "outer_ratio": outer_ratio,
        "conflict_ratio": conflict_ratio,
    }


def build_center_consistency_evidence_record(
    *,
    xy,
    proj_valid,
    image_size: tuple[int, int],
    part_masks: dict[str, np.ndarray],
    foreground_mask: np.ndarray,
    valid_mask: np.ndarray,
    part_name: str,
    candidate_mask=None,
    mask_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    part_index = _validate_part(part_name)
    if hasattr(xy, "detach"):
        xy_np = xy.detach().float().cpu().numpy()
    else:
        xy_np = np.asarray(xy, dtype=np.float32)
    if xy_np.ndim != 2 or xy_np.shape[1] != 2:
        raise ValueError("xy must have shape [N, 2]")
    point_count = int(xy_np.shape[0])
    proj = _as_bool_vector(proj_valid, point_count, "proj_valid")
    if candidate_mask is None:
        candidate = np.ones((point_count,), dtype=bool)
    else:
        candidate = _as_bool_vector(candidate_mask, point_count, "candidate_mask")

    width, height = int(image_size[0]), int(image_size[1])
    px = np.rint(xy_np[:, 0]).astype(np.int64)
    py = np.rint(xy_np[:, 1]).astype(np.int64)
    in_image = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    active = candidate & proj & in_image
    fg = np.asarray(foreground_mask, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=np.float32)
    if fg.shape != (height, width):
        raise ValueError(f"foreground mask shape {fg.shape} does not match {(height, width)}")
    if valid.shape != (height, width):
        raise ValueError(f"valid mask shape {valid.shape} does not match {(height, width)}")
    target = np.asarray(part_masks[PART_NAMES[part_index]], dtype=np.float32)
    if target.shape != (height, width):
        raise ValueError(f"{part_name} mask shape {target.shape} does not match {(height, width)}")

    valid_center = np.zeros((point_count,), dtype=bool)
    target_center = np.zeros((point_count,), dtype=bool)
    outer_center = np.zeros((point_count,), dtype=bool)
    idx = np.nonzero(active)[0]
    if idx.size:
        valid_hit = (fg[py[idx], px[idx]] >= float(mask_threshold)) & (
            valid[py[idx], px[idx]] >= float(mask_threshold)
        )
        target_hit = target[py[idx], px[idx]] >= float(mask_threshold)
        valid_center[idx] = valid_hit
        target_center[idx] = valid_hit & target_hit
        outer_center[idx] = valid_hit & ~target_hit
    return {
        "valid_center": valid_center,
        "target_center": target_center,
        "outer_center": outer_center,
    }


def accumulate_footprint_evidence(records, *, point_count: int) -> dict[str, np.ndarray]:
    target_sum = np.zeros((int(point_count),), dtype=np.float32)
    outer_sum = np.zeros((int(point_count),), dtype=np.float32)
    conflict_sum = np.zeros((int(point_count),), dtype=np.float32)
    support = np.zeros((int(point_count),), dtype=np.int16)
    for record in records:
        observed = np.asarray(record["observed"], dtype=bool).reshape(-1)
        if observed.shape[0] != int(point_count):
            raise ValueError("record observed point count mismatch")
        target = np.asarray(record["target_ratio"], dtype=np.float32).reshape(-1)
        outer = np.asarray(record["outer_ratio"], dtype=np.float32).reshape(-1)
        conflict = np.asarray(record["conflict_ratio"], dtype=np.float32).reshape(-1)
        if target.shape[0] != int(point_count) or outer.shape[0] != int(point_count) or conflict.shape[0] != int(point_count):
            raise ValueError("record ratio point count mismatch")
        target_sum += np.where(observed, target, 0.0).astype(np.float32)
        outer_sum += np.where(observed, outer, 0.0).astype(np.float32)
        conflict_sum += np.where(observed, conflict, 0.0).astype(np.float32)
        support += observed.astype(np.int16)
    denom = np.maximum(support.astype(np.float32), 1.0)
    return {
        "view_support_count": support,
        "footprint_target_ratio": (target_sum / denom).astype(np.float32, copy=False),
        "footprint_outer_ratio": (outer_sum / denom).astype(np.float32, copy=False),
        "conflict_ratio": (conflict_sum / denom).astype(np.float32, copy=False),
    }


def accumulate_center_consistency_evidence(records, *, point_count: int) -> dict[str, np.ndarray]:
    valid_count = np.zeros((int(point_count),), dtype=np.int16)
    target_count = np.zeros((int(point_count),), dtype=np.int16)
    outer_count = np.zeros((int(point_count),), dtype=np.int16)
    for record in records:
        valid = np.asarray(record["valid_center"], dtype=bool).reshape(-1)
        target = np.asarray(record["target_center"], dtype=bool).reshape(-1)
        outer = np.asarray(record["outer_center"], dtype=bool).reshape(-1)
        if valid.shape[0] != int(point_count) or target.shape[0] != int(point_count) or outer.shape[0] != int(point_count):
            raise ValueError("center evidence point count mismatch")
        valid_count += valid.astype(np.int16)
        target_count += target.astype(np.int16)
        outer_count += outer.astype(np.int16)
    denom = np.maximum(valid_count.astype(np.float32), 1.0)
    return {
        "center_valid_count": valid_count,
        "center_target_hit_count": target_count,
        "center_outer_hit_count": outer_count,
        "center_outer_ratio": (outer_count.astype(np.float32) / denom).astype(np.float32, copy=False),
    }


def _empty_evidence(point_count: int) -> dict[str, np.ndarray]:
    shape = (int(point_count), len(PART_NAMES))
    return {
        "footprint_target_ratio": np.zeros(shape, dtype=np.float32),
        "footprint_outer_ratio": np.zeros(shape, dtype=np.float32),
        "view_support_count": np.zeros(shape, dtype=np.int16),
        "conflict_ratio": np.zeros(shape, dtype=np.float32),
        "center_valid_count": np.zeros(shape, dtype=np.int16),
        "center_target_hit_count": np.zeros(shape, dtype=np.int16),
        "center_outer_hit_count": np.zeros(shape, dtype=np.int16),
        "center_outer_ratio": np.zeros(shape, dtype=np.float32),
    }


def save_footprint_evidence_npz(path: Path, evidence: dict[str, np.ndarray], *, part_names) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "footprint_target_ratio": np.asarray(evidence["footprint_target_ratio"], dtype=np.float32),
        "footprint_outer_ratio": np.asarray(evidence["footprint_outer_ratio"], dtype=np.float32),
        "view_support_count": np.asarray(evidence["view_support_count"], dtype=np.int16),
        "conflict_ratio": np.asarray(evidence["conflict_ratio"], dtype=np.float32),
        "center_valid_count": np.asarray(evidence.get("center_valid_count", np.zeros_like(evidence["view_support_count"])), dtype=np.int16),
        "center_target_hit_count": np.asarray(evidence.get("center_target_hit_count", np.zeros_like(evidence["view_support_count"])), dtype=np.int16),
        "center_outer_hit_count": np.asarray(evidence.get("center_outer_hit_count", np.zeros_like(evidence["view_support_count"])), dtype=np.int16),
        "center_outer_ratio": np.asarray(evidence.get("center_outer_ratio", np.zeros_like(evidence["footprint_target_ratio"])), dtype=np.float32),
        "part_names": np.asarray(list(part_names), dtype=str),
        "bank_part_names": np.asarray(PART_NAMES, dtype=str),
    }
    np.savez_compressed(path, **payload)


def apply_evidence_calibration_to_bank(
    bank: dict,
    evidence: dict[str, np.ndarray],
    *,
    output: Path,
    parts: tuple[str, ...] | list[str],
    min_support_views: int = 5,
    min_center_views: int = 0,
    target_retention_floor: float = 0.60,
    outer_penalty_power: float = 1.0,
    conflict_penalty_power: float = 1.0,
    center_penalty_power: float = 0.0,
    center_target_retention_floor: float = 0.75,
) -> dict:
    if "soft_edit_weights" not in bank:
        raise ValueError("bank must contain soft_edit_weights")
    weights = np.asarray(bank["soft_edit_weights"], dtype=np.float32)
    updated, summary = compute_evidence_calibrated_soft_edit_weights(
        soft_edit_weights=weights,
        footprint_target_ratio=evidence["footprint_target_ratio"],
        footprint_outer_ratio=evidence["footprint_outer_ratio"],
        view_support_count=evidence["view_support_count"],
        conflict_ratio=evidence.get("conflict_ratio"),
        center_outer_ratio=evidence.get("center_outer_ratio"),
        center_valid_count=evidence.get("center_valid_count"),
        parts=parts,
        min_support_views=int(min_support_views),
        min_center_views=int(min_center_views),
        target_retention_floor=float(target_retention_floor),
        outer_penalty_power=float(outer_penalty_power),
        conflict_penalty_power=float(conflict_penalty_power),
        center_penalty_power=float(center_penalty_power),
        center_target_retention_floor=float(center_target_retention_floor),
    )
    save_part_label_bank(
        output,
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
        source_type="evidence_calibrated_" + "_".join(str(part) for part in parts),
    )
    return summary


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate soft edit weights with multi-view footprint evidence.")
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
    parser.add_argument("--soft-boundary-radius", type=int, default=0)
    parser.add_argument("--soft-boundary-min-value", type=float, default=0.25)
    parser.add_argument("--footprint-radius-scale", type=float, default=1.0)
    parser.add_argument("--min-footprint-radius", type=int, default=1)
    parser.add_argument("--max-footprint-radius", type=int, default=12)
    parser.add_argument("--min-support-views", type=int, default=5)
    parser.add_argument("--min-center-views", type=int, default=0)
    parser.add_argument("--target-retention-floor", type=float, default=0.60)
    parser.add_argument("--outer-penalty-power", type=float, default=1.0)
    parser.add_argument("--conflict-penalty-power", type=float, default=1.0)
    parser.add_argument("--center-penalty-power", type=float, default=0.0)
    parser.add_argument("--center-target-retention-floor", type=float, default=0.75)
    parser.add_argument("--candidate-soft-min-weight", type=float, default=0.05)
    parser.add_argument("--evidence-output", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
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
    weights = np.asarray(bank["soft_edit_weights"], dtype=np.float32)
    point_count = int(weights.shape[0])
    editable = np.asarray(bank.get("editable_label", bank["part_label"]), dtype=np.int16).reshape(-1)
    evidence = _empty_evidence(point_count)
    asset_root = args.asset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    records = _select_records(_load_view_records(asset_root), args.max_views)
    config_path = args.config.resolve() if args.config else asset_root.parent.parent / ".hydra" / "config.yaml"
    config = _load_config(config_path, checkpoint, asset_root, records, args)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    part_records = {part: [] for part in args.parts}
    center_records = {part: [] for part in args.parts}
    candidate_masks = {
        part: build_part_candidate_mask(
            soft_edit_weights=weights,
            editable_label=editable,
            part_name=part,
            soft_min_weight=float(args.candidate_soft_min_weight),
        )
        for part in args.parts
    }
    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, str(asset_root.parent))
        scene.eval()
        loaded_iteration = int(scene.load_checkpoint(str(checkpoint)))
        scene_point_count = int(scene.gaussians.get_xyz.shape[0])
        if scene_point_count != point_count:
            raise ValueError(f"checkpoint point count {scene_point_count} does not match bank {point_count}")
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
                    build_footprint_evidence_record(
                        xy=xy,
                        proj_valid=proj_valid,
                        visibility_filter=pkg["visibility_filter"],
                        radii=pkg["radii"],
                        image_size=image_size,
                        part_masks=part_masks,
                        foreground_mask=foreground_mask,
                        valid_mask=valid_mask,
                        part_name=part,
                        candidate_mask=candidate_masks[part],
                        mask_threshold=float(args.mask_threshold),
                        soft_boundary_radius=int(args.soft_boundary_radius),
                        soft_boundary_min_value=float(args.soft_boundary_min_value),
                        footprint_radius_scale=float(args.footprint_radius_scale),
                        min_footprint_radius=int(args.min_footprint_radius),
                        max_footprint_radius=int(args.max_footprint_radius),
                    )
                )
                center_records[part].append(
                    build_center_consistency_evidence_record(
                        xy=xy,
                        proj_valid=proj_valid,
                        image_size=image_size,
                        part_masks=part_masks,
                        foreground_mask=foreground_mask,
                        valid_mask=valid_mask,
                        part_name=part,
                        candidate_mask=candidate_masks[part],
                        mask_threshold=float(args.mask_threshold),
                    )
                )
            del pkg, deformed
            torch.cuda.empty_cache()

    for part in args.parts:
        part_index = _validate_part(part)
        part_stats = accumulate_footprint_evidence(part_records[part], point_count=point_count)
        evidence["footprint_target_ratio"][:, part_index] = part_stats["footprint_target_ratio"]
        evidence["footprint_outer_ratio"][:, part_index] = part_stats["footprint_outer_ratio"]
        evidence["view_support_count"][:, part_index] = part_stats["view_support_count"]
        evidence["conflict_ratio"][:, part_index] = part_stats["conflict_ratio"]
        center_stats = accumulate_center_consistency_evidence(center_records[part], point_count=point_count)
        evidence["center_valid_count"][:, part_index] = center_stats["center_valid_count"]
        evidence["center_target_hit_count"][:, part_index] = center_stats["center_target_hit_count"]
        evidence["center_outer_hit_count"][:, part_index] = center_stats["center_outer_hit_count"]
        evidence["center_outer_ratio"][:, part_index] = center_stats["center_outer_ratio"]

    if args.evidence_output is not None:
        save_footprint_evidence_npz(args.evidence_output, evidence, part_names=tuple(args.parts))

    summary = apply_evidence_calibration_to_bank(
        bank,
        evidence,
        output=args.output,
        parts=tuple(args.parts),
        min_support_views=int(args.min_support_views),
        min_center_views=int(args.min_center_views),
        target_retention_floor=float(args.target_retention_floor),
        outer_penalty_power=float(args.outer_penalty_power),
        conflict_penalty_power=float(args.conflict_penalty_power),
        center_penalty_power=float(args.center_penalty_power),
        center_target_retention_floor=float(args.center_target_retention_floor),
    )
    summary.update(
        {
            "part_label_bank": str(args.part_label_bank),
            "output": str(args.output),
            "checkpoint": str(checkpoint),
            "asset_root": str(asset_root),
            "processed_views": int(len(records)),
            "evidence_output": str(args.evidence_output) if args.evidence_output is not None else "",
            "candidate_counts": {
                part: int(np.sum(candidate_masks[part]))
                for part in args.parts
            },
            "parameters": {
                "mask_threshold": float(args.mask_threshold),
                "soft_boundary_radius": int(args.soft_boundary_radius),
                "soft_boundary_min_value": float(args.soft_boundary_min_value),
                "footprint_radius_scale": float(args.footprint_radius_scale),
                "min_footprint_radius": int(args.min_footprint_radius),
                "max_footprint_radius": int(args.max_footprint_radius),
                "min_support_views": int(args.min_support_views),
                "min_center_views": int(args.min_center_views),
                "target_retention_floor": float(args.target_retention_floor),
                "outer_penalty_power": float(args.outer_penalty_power),
                "conflict_penalty_power": float(args.conflict_penalty_power),
                "center_penalty_power": float(args.center_penalty_power),
                "center_target_retention_floor": float(args.center_target_retention_floor),
                "candidate_soft_min_weight": float(args.candidate_soft_min_weight),
            },
        }
    )
    if args.summary_json is not None:
        _write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
