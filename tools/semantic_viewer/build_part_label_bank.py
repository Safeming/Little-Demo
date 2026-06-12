#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.graphics_utils import geom_transform_points
from utils.part_label_bank import (
    PART_NAMES,
    apply_face_label_guard,
    apply_lower_label_guard,
    apply_neighbor_reliable_fill,
    apply_reliable_label_mask,
    compute_semantic_margin,
    compute_soft_edit_weights,
    finalize_votes,
    finalize_trained_semantic_probs,
    save_part_label_bank,
    summarize_part_label_bank,
    write_preview_ply,
    write_summary_json,
)


IMAGE_RE = re.compile(r"c(?P<cam>\d+)_f(?P<frame>\d+)$")


def _list_to_omegaconf_repr(values) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _range_spec_from_values(values) -> str:
    values = sorted({int(v) for v in values})
    if not values:
        raise ValueError("empty frame set")
    if len(values) == 1:
        return _list_to_omegaconf_repr([values[0], values[0] + 1, 1])
    frame_set = set(values)
    span = max(values) - min(values)
    step = 1
    for candidate in range(1, span + 1):
        generated = set(range(min(values), max(values) + 1, candidate))
        if frame_set.issubset(generated):
            step = candidate
            break
    return _list_to_omegaconf_repr([min(values), max(values) + step, step])


def _view_ids(image_name: str) -> tuple[int, int]:
    match = IMAGE_RE.match(str(image_name))
    if match is None:
        raise ValueError(f"invalid image_name: {image_name}")
    return int(match.group("cam")), int(match.group("frame"))


def _load_config(config_path: Path, checkpoint: Path, asset_root: Path, records, args):
    from utils.adopted_geometry import apply_explicit_binding_render_preset

    views = sorted({int(record.get("cam_id", _view_ids(record["image_name"])[0])) for record in records})
    frames = sorted({int(record.get("frame_id", _view_ids(record["image_name"])[1])) for record in records})
    config = OmegaConf.load(config_path)
    overrides = [
        "mode=test",
        f"load_ckpt={checkpoint}",
        "dataset.preload=false",
        f"dataset.test_views.view={_list_to_omegaconf_repr(views)}",
        f"dataset.test_frames.view={_range_spec_from_values(frames)}",
        "dataset.parsing_prior.enable=false",
        "dataset.parsing_prior.roi_enable=false",
        f"exp_dir={asset_root.parent}",
        "wandb_disable=true",
    ]
    if args.dataset_root:
        overrides.append(f"dataset.root_dir={args.dataset_root}")
    if args.subject:
        overrides.append(f"dataset.subject={args.subject}")
    if args.explicit_binding_render_preset:
        overrides.append(f"explicit_binding_render_preset={args.explicit_binding_render_preset}")
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    OmegaConf.set_struct(config, False)
    apply_explicit_binding_render_preset(config, repo_root=REPO_ROOT)
    return config


def _load_view_records(asset_root: Path) -> list[dict]:
    path = asset_root / "view_records.json"
    if not path.exists():
        raise FileNotFoundError(f"missing view_records.json: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("view_records.json must be a list")
    out = []
    for record in records:
        if not isinstance(record, dict) or not record.get("image_name"):
            continue
        out.append(record)
    if not out:
        raise ValueError("view_records.json contains no usable records")
    return out


def _select_records(records: list[dict], max_views: int | None) -> list[dict]:
    selected = list(records)
    if max_views is not None and int(max_views) > 0:
        selected = selected[: int(max_views)]
    if not selected:
        raise ValueError("no selected view records")
    return selected


def _find_dataset_index(dataset, image_name: str) -> int | None:
    target_cam, target_frame = _view_ids(image_name)
    for index, row in enumerate(getattr(dataset, "data", [])):
        cam_name = row.get("cam_name", "")
        frame_idx = int(row.get("frame_idx", -1))
        candidate = f"c{int(cam_name):02d}_f{frame_idx if frame_idx >= 0 else -frame_idx - 1:06d}"
        if candidate == image_name:
            return index
    for index in range(len(dataset)):
        view = dataset[index]
        candidate = str(getattr(view, "image_name", ""))
        if candidate == image_name:
            return index
        try:
            cam, frame = _view_ids(candidate)
        except ValueError:
            continue
        if cam == target_cam and frame == target_frame:
            return index
    return None


def _project_points(points: torch.Tensor, view) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ndc = geom_transform_points(points.detach(), view.full_proj_transform.detach().to(device=points.device, dtype=points.dtype))
    camera_xyz = geom_transform_points(
        points.detach(),
        view.world_view_transform.detach().to(device=points.device, dtype=points.dtype),
    )
    width = int(view.image_width)
    height = int(view.image_height)
    px = (ndc[:, 0] + 1.0) * 0.5 * float(max(width - 1, 1))
    # The exported semantic asset masks use the same y-axis convention as NDC here.
    py = (ndc[:, 1] + 1.0) * 0.5 * float(max(height - 1, 1))
    valid = torch.isfinite(ndc).all(dim=-1)
    valid &= ndc[:, 2] > 0.0
    valid &= px >= 0.0
    valid &= px <= float(max(width - 1, 0))
    valid &= py >= 0.0
    valid &= py <= float(max(height - 1, 0))
    return torch.stack((px, py), dim=-1), valid, camera_xyz[:, 2]


def _read_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing mask: {path}")
    arr = imageio.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size and arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _load_record_masks(asset_root: Path, record: dict) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    compact = record.get("compact_head_mask_files", {}) or {}
    part_masks = {}
    for name in PART_NAMES:
        rel = compact.get(name)
        if not rel:
            rel = f"compact_head_masks/{name}/render_{record['image_name']}.png"
        part_masks[name] = _read_mask(asset_root / rel)
    coarse = record.get("coarse_mask_files", {}) or {}
    first_shape = next(iter(part_masks.values())).shape
    fg = _read_mask(asset_root / coarse["foreground"]) if coarse.get("foreground") else np.ones(first_shape, dtype=np.float32)
    valid = _read_mask(asset_root / coarse["valid"]) if coarse.get("valid") else np.ones(first_shape, dtype=np.float32)
    return part_masks, fg, valid


def build_output_manifest(
    *,
    checkpoint,
    config,
    asset_root,
    output,
    summary_json,
    preview_ply,
    point_count: int,
    source_iteration: int,
    processed_views: int,
    depth_margin: float,
    min_part_hit_ratio: float,
    source_type: str = "trained_semantic_asset_probs",
    soft_edit_weight_field: str | None = None,
) -> dict:
    manifest = {
        "schema_version": 1,
        "source_checkpoint": str(checkpoint),
        "source_config": str(config),
        "source_asset_root": str(asset_root),
        "part_label_bank": str(output),
        "summary_json": str(summary_json),
        "preview_ply": str(preview_ply),
        "point_count": int(point_count),
        "source_iteration": int(source_iteration),
        "processed_views": int(processed_views),
        "part_names": list(PART_NAMES),
        "depth_margin": float(depth_margin),
        "min_part_hit_ratio": float(min_part_hit_ratio),
        "source_type": str(source_type),
    }
    if soft_edit_weight_field:
        manifest["soft_edit_weight_field"] = str(soft_edit_weight_field)
        manifest["soft_edit_part_names"] = list(PART_NAMES)
    return manifest


def _as_numpy_bool(value, point_count: int) -> np.ndarray:
    if torch.is_tensor(value):
        out = value.detach().bool().cpu().numpy()
    else:
        out = np.asarray(value, dtype=bool)
    out = out.reshape(-1)
    if out.shape[0] != point_count:
        raise ValueError(f"visibility shape mismatch: got {out.shape[0]}, expected {point_count}")
    return out


def _as_numpy_float(value, point_count: int) -> np.ndarray:
    if torch.is_tensor(value):
        out = value.detach().float().cpu().numpy()
    else:
        out = np.asarray(value, dtype=np.float32)
    out = out.reshape(-1)
    if out.shape[0] != point_count:
        raise ValueError(f"radii shape mismatch: got {out.shape[0]}, expected {point_count}")
    return out


def _front_surface_mask(px: np.ndarray, py: np.ndarray, depth: np.ndarray, candidate: np.ndarray, width: int, height: int, margin: float) -> np.ndarray:
    active = np.nonzero(candidate)[0]
    out = np.zeros((candidate.shape[0],), dtype=bool)
    if active.size == 0:
        return out
    depth_active = depth[active].astype(np.float32, copy=False)
    finite = np.isfinite(depth_active)
    if not np.any(finite):
        return out
    active = active[finite]
    depth_active = depth_active[finite]
    linear = py[active].astype(np.int64) * int(width) + px[active].astype(np.int64)
    min_depth = np.full((int(width) * int(height),), np.inf, dtype=np.float32)
    np.minimum.at(min_depth, linear, depth_active)
    out[active] = depth_active <= (min_depth[linear] + float(margin))
    return out


def accumulate_projected_votes(
    *,
    xy,
    proj_valid,
    visibility_filter,
    radii,
    depth=None,
    depth_margin: float | None = None,
    image_size: tuple[int, int],
    part_masks: dict[str, np.ndarray],
    foreground_mask: np.ndarray,
    valid_mask: np.ndarray,
    per_part_votes: np.ndarray,
    visible_vote_count: np.ndarray,
    conflict_count: np.ndarray,
    mask_threshold: float = 0.5,
    min_part_hit_ratio: float = 0.0,
    view_name: str = "",
    footprint_mode: str = "center",
    footprint_radius_scale: float = 1.0,
    min_footprint_radius: int = 1,
    max_footprint_radius: int = 12,
    min_footprint_hit_ratio: float = 0.50,
) -> dict[str, int]:
    if torch.is_tensor(xy):
        xy_np = xy.detach().float().cpu().numpy()
    else:
        xy_np = np.asarray(xy, dtype=np.float32)
    point_count = xy_np.shape[0]
    proj_np = _as_numpy_bool(proj_valid, point_count)
    vis_np = _as_numpy_bool(visibility_filter, point_count)
    radii_np = _as_numpy_float(radii, point_count)
    width, height = int(image_size[0]), int(image_size[1])

    px = np.rint(xy_np[:, 0]).astype(np.int64)
    py = np.rint(xy_np[:, 1]).astype(np.int64)
    in_image = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    visible_projected = proj_np & vis_np & (radii_np > 0.0) & in_image
    depth_visible_count = int(visible_projected.sum())
    if depth is not None and depth_margin is not None and float(depth_margin) >= 0.0:
        depth_np = _as_numpy_float(depth, point_count)
        front_surface = _front_surface_mask(px, py, depth_np, visible_projected, width, height, float(depth_margin))
        visible_projected = visible_projected & front_surface
        depth_visible_count = int(visible_projected.sum())
    if not np.any(visible_projected):
        if float(min_part_hit_ratio) > 0.0:
            raise RuntimeError(f"{view_name or 'view'} part hit ratio 0.0000 below minimum {float(min_part_hit_ratio):.4f}")
        return {"visible_projected_count": 0, "depth_visible_count": 0, "valid_mask_count": 0, "part_vote_count": 0, "conflict_count": 0}

    idx = np.nonzero(visible_projected)[0]
    x_sel = px[idx]
    y_sel = py[idx]
    fg = np.asarray(foreground_mask, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=np.float32)
    if fg.shape != (height, width):
        raise ValueError(f"foreground mask shape {fg.shape} does not match image {(height, width)}")
    if valid.shape != (height, width):
        raise ValueError(f"valid mask shape {valid.shape} does not match image {(height, width)}")
    fg_valid = (fg[y_sel, x_sel] >= float(mask_threshold)) & (valid[y_sel, x_sel] >= float(mask_threshold))
    valid_idx = idx[fg_valid]
    if valid_idx.size == 0:
        return {
            "visible_projected_count": int(idx.size),
            "depth_visible_count": depth_visible_count,
            "valid_mask_count": 0,
            "part_vote_count": 0,
            "conflict_count": 0,
        }
    visible_vote_count[valid_idx] += 1

    values = []
    for name in PART_NAMES:
        mask = np.asarray(part_masks[name], dtype=np.float32)
        if mask.shape != (height, width):
            raise ValueError(f"{name} mask shape {mask.shape} does not match image {(height, width)}")
        values.append(mask)

    mode = str(footprint_mode or "center").strip().lower()
    if mode not in ("center", "footprint"):
        raise ValueError(f"unsupported footprint_mode: {footprint_mode}")
    if mode == "center":
        part_values = np.stack([mask[py[valid_idx], px[valid_idx]] for mask in values], axis=1)
        hits = part_values >= float(mask_threshold)
        footprint_vote_count = 0
        mean_winning_footprint_hit_ratio = 0.0
    else:
        part_values = np.zeros((valid_idx.size, len(PART_NAMES)), dtype=np.float32)
        support_valid = np.zeros((valid_idx.size,), dtype=bool)
        min_radius = max(0, int(min_footprint_radius))
        max_radius = max(min_radius, int(max_footprint_radius))
        for local_row, point_idx in enumerate(valid_idx):
            x = int(px[point_idx])
            y = int(py[point_idx])
            point_radius = int(np.ceil(float(radii_np[point_idx]) * float(footprint_radius_scale)))
            point_radius = max(min_radius, min(max_radius, point_radius))
            y0 = max(0, y - point_radius)
            y1 = min(height, y + point_radius + 1)
            x0 = max(0, x - point_radius)
            x1 = min(width, x + point_radius + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            disk = ((yy - y) * (yy - y) + (xx - x) * (xx - x)) <= (point_radius * point_radius)
            support = (
                disk
                & (fg[y0:y1, x0:x1] >= float(mask_threshold))
                & (valid[y0:y1, x0:x1] >= float(mask_threshold))
            )
            support_count = int(np.sum(support))
            if support_count <= 0:
                continue
            support_valid[local_row] = True
            for part_idx, mask in enumerate(values):
                part_values[local_row, part_idx] = float(
                    np.sum((mask[y0:y1, x0:x1] >= float(mask_threshold)) & support)
                ) / float(support_count)
        hits = (part_values >= float(min_footprint_hit_ratio)) & support_valid[:, None]
        footprint_vote_count = int(np.sum(support_valid))
        winning = part_values.max(axis=1) if part_values.size else np.zeros((0,), dtype=np.float32)
        accepted = hits.sum(axis=1) > 0
        mean_winning_footprint_hit_ratio = float(winning[accepted].mean()) if np.any(accepted) else 0.0
    hit_count = hits.sum(axis=1)
    has_hit = hit_count > 0
    if not np.any(has_hit):
        if float(min_part_hit_ratio) > 0.0:
            raise RuntimeError(
                f"{view_name or 'view'} part hit ratio 0.0000 below minimum {float(min_part_hit_ratio):.4f} "
                f"(valid_mask_count={int(valid_idx.size)})"
            )
        return {
            "visible_projected_count": int(idx.size),
            "depth_visible_count": depth_visible_count,
            "valid_mask_count": int(valid_idx.size),
            "part_vote_count": 0,
            "conflict_count": 0,
            "footprint_vote_count": int(footprint_vote_count),
            "mean_winning_footprint_hit_ratio": float(mean_winning_footprint_hit_ratio),
        }
    conflicted = hit_count > 1
    vote_idx = valid_idx[has_hit]
    winning_part = np.argmax(np.where(hits[has_hit], part_values[has_hit], -1.0), axis=1)
    per_part_votes[vote_idx, winning_part] += 1
    conflict_count[valid_idx[conflicted]] += 1
    part_hit_ratio = float(vote_idx.size) / float(max(int(valid_idx.size), 1))
    if part_hit_ratio < float(min_part_hit_ratio):
        raise RuntimeError(
            f"{view_name or 'view'} part hit ratio {part_hit_ratio:.4f} below minimum {float(min_part_hit_ratio):.4f} "
            f"(part_vote_count={int(vote_idx.size)} valid_mask_count={int(valid_idx.size)})"
        )
    return {
        "visible_projected_count": int(idx.size),
        "depth_visible_count": depth_visible_count,
        "valid_mask_count": int(valid_idx.size),
        "part_vote_count": int(vote_idx.size),
        "conflict_count": int(conflicted.sum()),
        "part_hit_ratio": part_hit_ratio,
        "footprint_vote_count": int(footprint_vote_count),
        "mean_winning_footprint_hit_ratio": float(mean_winning_footprint_hit_ratio),
    }


def _semantic_probs_from_deformed_gaussian(deformed_gaussian) -> tuple[np.ndarray, tuple[str, ...]]:
    probs = getattr(deformed_gaussian, "binding_compact_semantic_probs_asset", None)
    if probs is None:
        probs = getattr(deformed_gaussian, "binding_compact_semantic_probs_asset_raw", None)
    if probs is None:
        probs = getattr(deformed_gaussian, "binding_compact_semantic_probs", None)
    if probs is None:
        raise RuntimeError("deformed Gaussian has no binding compact semantic probabilities")
    if torch.is_tensor(probs):
        probs_np = probs.detach().float().cpu().numpy()
    else:
        probs_np = np.asarray(probs, dtype=np.float32)
    if probs_np.ndim != 2:
        raise RuntimeError(f"binding compact semantic probabilities must have shape [N, C], got {probs_np.shape}")
    names = tuple(str(name) for name in getattr(deformed_gaussian, "binding_compact_semantic_names", ()))
    if not names:
        names = ("hair", "face", "skin", "upper", "lower", "shoes")
    if probs_np.shape[1] != len(names):
        raise RuntimeError(
            "binding compact semantic channel count does not match names: "
            f"probs={probs_np.shape[1]} names={len(names)}"
        )
    return probs_np.astype(np.float32, copy=False), names


def _gaussian_opacity_numpy(gaussians, point_count: int) -> np.ndarray:
    opacity = getattr(gaussians, "get_opacity", None)
    if opacity is None:
        raw = getattr(gaussians, "_opacity", None)
        if raw is None:
            raise RuntimeError("GaussianModel has no opacity tensor")
        opacity = torch.sigmoid(raw) if torch.is_tensor(raw) else 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=np.float32)))
    if torch.is_tensor(opacity):
        opacity_np = opacity.detach().float().cpu().numpy()
    else:
        opacity_np = np.asarray(opacity, dtype=np.float32)
    opacity_np = opacity_np.reshape(-1).astype(np.float32, copy=False)
    if opacity_np.shape[0] != int(point_count):
        raise RuntimeError(f"opacity point count mismatch: got {opacity_np.shape[0]}, expected {int(point_count)}")
    return opacity_np


def _gaussian_scale_max_numpy(gaussians, point_count: int) -> np.ndarray:
    scaling = getattr(gaussians, "get_scaling", None)
    if scaling is None:
        raw = getattr(gaussians, "_scaling", None)
        if raw is None:
            raise RuntimeError("GaussianModel has no scaling tensor")
        scaling = torch.exp(raw) if torch.is_tensor(raw) else np.exp(np.asarray(raw, dtype=np.float32))
    if torch.is_tensor(scaling):
        scaling_np = scaling.detach().float().cpu().numpy()
    else:
        scaling_np = np.asarray(scaling, dtype=np.float32)
    scaling_np = scaling_np.reshape(int(point_count), -1).astype(np.float32, copy=False)
    if scaling_np.shape[0] != int(point_count):
        raise RuntimeError(f"scaling point count mismatch: got {scaling_np.shape[0]}, expected {int(point_count)}")
    return scaling_np.max(axis=1).astype(np.float32, copy=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Build per-Gaussian 3D part label bank from trained compact semantic probabilities.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument(
        "--label-bank-source",
        choices=("trained-semantic", "projected-2d-voting"),
        default="trained-semantic",
        help="Source used to assign per-Gaussian part labels.",
    )
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--preview-ply", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--subject", default="")
    parser.add_argument("--explicit-binding-render-preset", default="v338_temporal_selector_grow_only_guard")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--depth-margin", type=float, default=0.02)
    parser.add_argument("--min-part-hit-ratio", type=float, default=0.0)
    parser.add_argument("--vote-footprint-mode", choices=("center", "footprint"), default="center")
    parser.add_argument("--vote-use-render-radii", action="store_true")
    parser.add_argument("--vote-footprint-radius-scale", type=float, default=1.0)
    parser.add_argument("--vote-min-footprint-radius", type=int, default=1)
    parser.add_argument("--vote-max-footprint-radius", type=int, default=12)
    parser.add_argument("--vote-min-footprint-hit-ratio", type=float, default=0.50)
    parser.add_argument(
        "--min-opacity",
        type=float,
        default=0.0,
        help="Mark Gaussians below this activated opacity as unknown in the exported semantic label bank.",
    )
    parser.add_argument("--face-guard-enable", action="store_true", help="Apply stricter guards before exporting face labels.")
    parser.add_argument("--face-min-prob", type=float, default=0.70)
    parser.add_argument("--face-min-margin", type=float, default=0.15)
    parser.add_argument("--face-max-scale", type=float, default=0.12)
    parser.add_argument(
        "--face-oversized-action",
        choices=("second", "unknown"),
        default="second",
        help="How to relabel rejected face points.",
    )
    parser.add_argument("--lower-guard-enable", action="store_true", help="Relabel high upper-body lower points to upper.")
    parser.add_argument("--lower-high-y-threshold", type=float, default=0.30)
    parser.add_argument("--lower-guard-max-abs-x", type=float, default=0.35)
    parser.add_argument("--lower-guard-max-abs-z", type=float, default=0.18)
    parser.add_argument(
        "--reliability-enable",
        action="store_true",
        help="Export semantic_margin, reliable_mask, and editable_label without changing the primary part_label.",
    )
    parser.add_argument("--reliability-min-confidence", type=float, default=0.65)
    parser.add_argument("--reliability-min-margin", type=float, default=0.20)
    parser.add_argument("--reliability-min-opacity", type=float, default=0.005)
    parser.add_argument(
        "--preview-use-editable-label",
        action="store_true",
        help="Write the preview PLY with editable_label when reliability export is enabled.",
    )
    parser.add_argument(
        "--neighbor-fill-enable",
        action="store_true",
        help="Fill editable_label unknowns when reliable 3D neighbors agree with the original part_label.",
    )
    parser.add_argument("--neighbor-fill-k", type=int, default=12)
    parser.add_argument("--neighbor-fill-min-reliable-neighbors", type=int, default=5)
    parser.add_argument("--neighbor-fill-majority-ratio", type=float, default=0.70)
    parser.add_argument("--neighbor-fill-min-candidate-confidence", type=float, default=0.50)
    parser.add_argument(
        "--export-soft-edit-weights",
        action="store_true",
        help="Export soft_edit_weights for reliability-aware semantic edit selection.",
    )
    parser.add_argument("--soft-edit-reliable-floor", type=float, default=0.0)
    parser.add_argument("--soft-edit-margin-power", type=float, default=1.0)
    parser.add_argument("--soft-edit-confidence-power", type=float, default=1.0)
    return parser.parse_args()


def collect_trained_semantic_bank(scene, records, iteration: int, point_count: int) -> tuple[dict, list[dict]]:
    semantic_prob_sum = np.zeros((point_count, len(PART_NAMES)), dtype=np.float64)
    semantic_prob_count = np.zeros((point_count,), dtype=np.int32)
    semantic_source_names = None
    view_stats = []
    for record in records:
        dataset_index = _find_dataset_index(scene.test_dataset, record["image_name"])
        if dataset_index is None:
            raise RuntimeError(f"image {record['image_name']} not present in dataset")
        view = scene.test_dataset[dataset_index]
        deformed_gaussian, _, colors_precomp = scene.convert_gaussians(view, iteration, compute_loss=False)
        if not torch.is_tensor(getattr(deformed_gaussian, "get_xyz", None)):
            raise RuntimeError("scene.convert_gaussians did not return GaussianModel with get_xyz tensor")
        if int(deformed_gaussian.get_xyz.shape[0]) != point_count:
            raise RuntimeError(
                f"deformed point count changed: got {int(deformed_gaussian.get_xyz.shape[0])}, expected {point_count}"
            )
        probs_np, names = _semantic_probs_from_deformed_gaussian(deformed_gaussian)
        if probs_np.shape[0] != point_count:
            raise RuntimeError(
                f"semantic probability point count mismatch: got {probs_np.shape[0]}, expected {point_count}"
            )
        if semantic_source_names is None:
            semantic_source_names = names
        elif tuple(semantic_source_names) != tuple(names):
            raise RuntimeError(f"semantic class names changed across views: {semantic_source_names} vs {names}")
        remap = [names.index(name) for name in PART_NAMES]
        semantic_prob_sum += probs_np[:, remap].astype(np.float64, copy=False)
        semantic_prob_count += 1
        view_stats.append(
            {
                "image_name": str(record["image_name"]),
                "semantic_prob_count": int(probs_np.shape[0]),
                "semantic_class_names": list(names),
            }
        )
        del colors_precomp
        del deformed_gaussian
        torch.cuda.empty_cache()

    if semantic_source_names is None:
        raise RuntimeError("no semantic probabilities were collected")
    semantic_probs_bank_order = semantic_prob_sum / np.maximum(semantic_prob_count.reshape(-1, 1), 1)
    return finalize_trained_semantic_probs(semantic_probs_bank_order, PART_NAMES), view_stats


def collect_projected_2d_voting_bank(scene, asset_root: Path, records, iteration: int, point_count: int, args) -> tuple[dict, list[dict]]:
    render_scene = None
    background = None
    if bool(getattr(args, "vote_use_render_radii", False)):
        from gaussian_renderer import render as render_scene

        bg_color = [1, 1, 1] if bool(scene.cfg.dataset.white_background) else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    per_part_votes = np.zeros((point_count, len(PART_NAMES)), dtype=np.int32)
    visible_vote_count = np.zeros((point_count,), dtype=np.int32)
    conflict_count = np.zeros((point_count,), dtype=np.int32)
    view_stats = []
    for record in records:
        dataset_index = _find_dataset_index(scene.test_dataset, record["image_name"])
        if dataset_index is None:
            raise RuntimeError(f"image {record['image_name']} not present in dataset")
        view = scene.test_dataset[dataset_index]
        render_pkg = None
        if render_scene is not None:
            render_pkg = render_scene(
                view,
                iteration,
                scene,
                scene.cfg.pipeline,
                background,
                compute_loss=False,
                return_opacity=False,
            )
            deformed_gaussian = render_pkg["deformed_gaussian"]
            colors_precomp = None
        else:
            deformed_gaussian, _, colors_precomp = scene.convert_gaussians(view, iteration, compute_loss=False)
        if not torch.is_tensor(getattr(deformed_gaussian, "get_xyz", None)):
            raise RuntimeError("scene.convert_gaussians did not return GaussianModel with get_xyz tensor")
        if int(deformed_gaussian.get_xyz.shape[0]) != point_count:
            raise RuntimeError(
                f"deformed point count changed: got {int(deformed_gaussian.get_xyz.shape[0])}, expected {point_count}"
            )
        xy, proj_valid, depth = _project_points(deformed_gaussian.get_xyz, view)
        part_masks, foreground_mask, valid_mask = _load_record_masks(asset_root, record)
        if render_pkg is not None:
            visibility_filter = render_pkg["visibility_filter"][:point_count].detach().bool().to(deformed_gaussian.get_xyz.device)
            radii = render_pkg["radii"][:point_count].detach().float().to(deformed_gaussian.get_xyz.device)
        else:
            visibility_filter = torch.ones((point_count,), dtype=torch.bool, device=deformed_gaussian.get_xyz.device)
            radii = torch.ones((point_count,), dtype=torch.float32, device=deformed_gaussian.get_xyz.device)
        stats = accumulate_projected_votes(
            xy=xy,
            depth=depth,
            depth_margin=float(args.depth_margin),
            proj_valid=proj_valid,
            visibility_filter=visibility_filter,
            radii=radii,
            image_size=(int(view.image_width), int(view.image_height)),
            part_masks=part_masks,
            foreground_mask=foreground_mask,
            valid_mask=valid_mask,
            per_part_votes=per_part_votes,
            visible_vote_count=visible_vote_count,
            conflict_count=conflict_count,
            mask_threshold=float(args.mask_threshold),
            min_part_hit_ratio=float(args.min_part_hit_ratio),
            view_name=str(record["image_name"]),
            footprint_mode=str(getattr(args, "vote_footprint_mode", "center")),
            footprint_radius_scale=float(getattr(args, "vote_footprint_radius_scale", 1.0)),
            min_footprint_radius=int(getattr(args, "vote_min_footprint_radius", 1)),
            max_footprint_radius=int(getattr(args, "vote_max_footprint_radius", 12)),
            min_footprint_hit_ratio=float(getattr(args, "vote_min_footprint_hit_ratio", 0.50)),
        )
        stats["image_name"] = str(record["image_name"])
        view_stats.append(stats)
        del colors_precomp
        del deformed_gaussian
        if render_pkg is not None:
            del render_pkg
        torch.cuda.empty_cache()
    finalized = finalize_votes(per_part_votes, visible_vote_count, conflict_count)
    finalized["source_type"] = "multiview_2d_mask_votes"
    return finalized, view_stats


def main() -> int:
    from scene import GaussianModel, Scene

    args = parse_args()
    asset_root = args.asset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    config_path = args.config.resolve() if args.config else asset_root.parent.parent / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config; pass --config explicitly: {config_path}")
    records = _select_records(_load_view_records(asset_root), args.max_views)
    config = _load_config(config_path, checkpoint, asset_root, records, args)
    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, str(asset_root.parent))
        scene.eval()
        loaded_iteration = int(scene.load_checkpoint(str(checkpoint)))
        iteration = int(args.iteration) if int(args.iteration) >= 0 else loaded_iteration
        point_count = int(scene.gaussians.get_xyz.shape[0])
        semantic_valid_mask = None
        opacity_stats = None
        if float(args.min_opacity) > 0.0:
            gaussian_opacity = _gaussian_opacity_numpy(scene.gaussians, point_count)
            semantic_valid_mask = gaussian_opacity >= float(args.min_opacity)
            opacity_stats = {
                "min_opacity": float(args.min_opacity),
                "valid_point_count": int(semantic_valid_mask.sum()),
                "filtered_point_count": int(point_count - int(semantic_valid_mask.sum())),
            }
        if args.label_bank_source == "trained-semantic":
            finalized, view_stats = collect_trained_semantic_bank(scene, records, iteration, point_count)
            if semantic_valid_mask is not None:
                finalized = finalize_trained_semantic_probs(finalized["semantic_probs"], PART_NAMES, valid_mask=semantic_valid_mask)
        elif args.label_bank_source == "projected-2d-voting":
            finalized, view_stats = collect_projected_2d_voting_bank(scene, asset_root, records, iteration, point_count, args)
        else:
            raise RuntimeError(f"unsupported label bank source: {args.label_bank_source}")
        face_guard_stats = None
        if bool(args.face_guard_enable):
            scale_max = _gaussian_scale_max_numpy(scene.gaussians, point_count)
            face_guard_stats = apply_face_label_guard(
                finalized,
                min_prob=float(args.face_min_prob),
                min_margin=float(args.face_min_margin),
                max_scale=float(args.face_max_scale),
                scale_max=scale_max,
                oversized_action=str(args.face_oversized_action),
            )
            finalized["source_type"] = str(finalized.get("source_type", "trained_semantic_asset_probs")) + "_face_guard"
        lower_guard_stats = None
        if bool(args.lower_guard_enable):
            lower_guard_stats = apply_lower_label_guard(
                finalized,
                xyz=scene.gaussians.get_xyz.detach().cpu().numpy().astype(np.float32),
                high_y_threshold=float(args.lower_high_y_threshold),
                max_abs_x=float(args.lower_guard_max_abs_x),
                max_abs_z=float(args.lower_guard_max_abs_z),
                target_second_name="upper",
            )
            finalized["source_type"] = str(finalized.get("source_type", "trained_semantic_asset_probs")) + "_lower_guard"
        reliability_stats = None
        if bool(args.reliability_enable):
            gaussian_opacity = _gaussian_opacity_numpy(scene.gaussians, point_count)
            reliability_stats = apply_reliable_label_mask(
                finalized,
                opacity=gaussian_opacity,
                min_confidence=float(args.reliability_min_confidence),
                min_margin=float(args.reliability_min_margin),
                min_opacity=float(args.reliability_min_opacity),
            )
            finalized["source_type"] = str(finalized.get("source_type", "trained_semantic_asset_probs")) + "_reliability_export"
        neighbor_fill_stats = None
        if bool(args.neighbor_fill_enable):
            if "editable_label" not in finalized or "reliable_mask" not in finalized:
                raise RuntimeError("--neighbor-fill-enable requires --reliability-enable")
            neighbor_fill_stats = apply_neighbor_reliable_fill(
                finalized,
                xyz=scene.gaussians.get_xyz.detach().cpu().numpy().astype(np.float32),
                k=int(args.neighbor_fill_k),
                min_reliable_neighbors=int(args.neighbor_fill_min_reliable_neighbors),
                majority_ratio=float(args.neighbor_fill_majority_ratio),
                min_candidate_confidence=float(args.neighbor_fill_min_candidate_confidence),
            )
            finalized["source_type"] = str(finalized.get("source_type", "trained_semantic_asset_probs")) + "_neighbor_fill"
        soft_edit_stats = None
        if bool(args.export_soft_edit_weights):
            if "semantic_margin" not in finalized:
                finalized["semantic_margin"] = compute_semantic_margin(finalized["semantic_probs"])
            finalized["soft_edit_weights"] = compute_soft_edit_weights(
                semantic_probs=finalized["semantic_probs"],
                confidence=finalized["confidence"],
                semantic_margin=finalized["semantic_margin"],
                reliable_mask=finalized.get("reliable_mask"),
                reliable_floor=float(args.soft_edit_reliable_floor),
                confidence_power=float(args.soft_edit_confidence_power),
                margin_power=float(args.soft_edit_margin_power),
            )
            soft_edit_stats = {
                "enabled": True,
                "weight_field": "soft_edit_weights",
                "reliable_floor": float(args.soft_edit_reliable_floor),
                "confidence_power": float(args.soft_edit_confidence_power),
                "margin_power": float(args.soft_edit_margin_power),
                "mean_weight": float(np.mean(finalized["soft_edit_weights"])),
                "max_weight": float(np.max(finalized["soft_edit_weights"])) if point_count else 0.0,
            }
            finalized["source_type"] = str(finalized.get("source_type", "trained_semantic_asset_probs")) + "_soft_edit"
        save_part_label_bank(
            output,
            **finalized,
            source_checkpoint=str(checkpoint),
            source_asset_root=str(asset_root),
            source_iteration=iteration,
        )
        summary = summarize_part_label_bank(finalized)
        summary["processed_views"] = len(records)
        summary["view_stats"] = view_stats
        if opacity_stats is not None:
            summary["opacity_filter"] = opacity_stats
        if face_guard_stats is not None:
            summary["face_label_guard"] = face_guard_stats
        if lower_guard_stats is not None:
            summary["lower_label_guard"] = lower_guard_stats
        if reliability_stats is not None:
            summary["semantic_reliability"] = reliability_stats
        if neighbor_fill_stats is not None:
            summary["semantic_neighbor_fill"] = neighbor_fill_stats
            summary["editable_unknown_count"] = int(np.sum(np.asarray(finalized["editable_label"], dtype=np.int16) < 0))
            summary["editable_known_count"] = int(np.sum(np.asarray(finalized["editable_label"], dtype=np.int16) >= 0))
        if soft_edit_stats is not None:
            summary["soft_edit_weights"] = soft_edit_stats
        summary_path = args.summary_json.resolve() if args.summary_json else output.with_name("part_label_bank_summary.json")
        write_summary_json(summary_path, summary)
        preview_path = args.preview_ply.resolve() if args.preview_ply else output.with_name("part_label_bank_preview.ply")
        preview_labels = finalized["part_label"]
        if bool(args.preview_use_editable_label) and "editable_label" in finalized:
            preview_labels = finalized["editable_label"]
        write_preview_ply(preview_path, scene.gaussians.get_xyz.detach().cpu().numpy().astype(np.float32), preview_labels)
        manifest_path = args.manifest_json.resolve() if args.manifest_json else output.with_name("part_label_bank_manifest.json")
        manifest = build_output_manifest(
            checkpoint=str(checkpoint),
            config=str(config_path),
            asset_root=str(asset_root),
            output=str(output),
            summary_json=str(summary_path),
            preview_ply=str(preview_path),
            point_count=point_count,
            source_iteration=iteration,
            processed_views=len(records),
            depth_margin=float(args.depth_margin),
            min_part_hit_ratio=float(args.min_part_hit_ratio),
            soft_edit_weight_field="soft_edit_weights" if soft_edit_stats is not None else None,
        )
        if opacity_stats is not None:
            manifest["min_opacity"] = float(args.min_opacity)
            manifest["opacity_filter"] = opacity_stats
        if face_guard_stats is not None:
            manifest["face_label_guard"] = face_guard_stats
        if lower_guard_stats is not None:
            manifest["lower_label_guard"] = lower_guard_stats
        if reliability_stats is not None:
            manifest["semantic_reliability"] = reliability_stats
            manifest["preview_label_field"] = "editable_label" if bool(args.preview_use_editable_label) else "part_label"
            manifest["source_type"] = str(finalized.get("source_type", manifest.get("source_type", "trained_semantic_asset_probs")))
        if soft_edit_stats is not None:
            manifest["soft_edit_weights"] = soft_edit_stats
            manifest["source_type"] = str(finalized.get("source_type", manifest.get("source_type", "trained_semantic_asset_probs")))
        if neighbor_fill_stats is not None:
            manifest["semantic_neighbor_fill"] = neighbor_fill_stats
            manifest["editable_unknown_count"] = int(np.sum(np.asarray(finalized["editable_label"], dtype=np.int16) < 0))
            manifest["editable_known_count"] = int(np.sum(np.asarray(finalized["editable_label"], dtype=np.int16) >= 0))
            manifest["source_type"] = str(finalized.get("source_type", manifest.get("source_type", "trained_semantic_asset_probs")))
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output} views={len(records)} iteration={iteration}")
    print(f"wrote {summary_path}")
    print(f"wrote {preview_path}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
