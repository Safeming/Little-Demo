#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.graphics_utils import geom_transform_points


REGION_NAMES = ("body", "soft", "cloth")
LAYER_NAMES = ("rigid", "soft", "free")


def _parse_csv_ints(value: str) -> list[int]:
    out = []
    for part in str(value).replace("[", "").replace("]", "").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _range_or_values(spec: str) -> list[int]:
    values = _parse_csv_ints(spec)
    if len(values) == 3:
        return list(range(values[0], values[1], values[2]))
    return values


def _triple_or_values(spec: str) -> list[int]:
    return _parse_csv_ints(spec)


def _dilate(mask: torch.Tensor, width: int) -> torch.Tensor:
    width = max(0, int(width))
    if width <= 0:
        return mask
    kernel = width * 2 + 1
    return F.max_pool2d(mask.unsqueeze(0), kernel, stride=1, padding=width)[0]


def _erode(mask: torch.Tensor, width: int) -> torch.Tensor:
    width = max(0, int(width))
    if width <= 0:
        return mask
    return 1.0 - _dilate(1.0 - mask, width)


def _binary_close(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    closed = F.max_pool2d(mask.unsqueeze(0), kernel_size, stride=1, padding=pad)[0]
    closed = 1.0 - F.max_pool2d((1.0 - closed).unsqueeze(0), kernel_size, stride=1, padding=pad)[0]
    return closed.clamp(0.0, 1.0)


def _boundary_band(mask: torch.Tensor, width: int) -> torch.Tensor:
    binary = (mask > 0.5).float()
    return (_dilate(binary, width) - _erode(binary, width)).clamp(0.0, 1.0)


def _mask_from_view(view) -> torch.Tensor:
    mask = getattr(view, "hard_mask", None)
    if not torch.is_tensor(mask):
        mask = getattr(view, "original_mask", None)
    if not torch.is_tensor(mask):
        raise RuntimeError(f"view {getattr(view, 'image_name', '<unknown>')} has no mask")
    mask = mask.detach().float().cuda()
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask[:1]
    else:
        mask = mask.reshape(1, *mask.shape[-2:])
    return (mask > 0.5).float()


def _render_support_mask(render_rgb: torch.Tensor, threshold: float, close_kernel: int) -> torch.Tensor:
    luma = render_rgb[0] * 0.299 + render_rgb[1] * 0.587 + render_rgb[2] * 0.114
    chroma = render_rgb.max(dim=0).values - render_rgb.min(dim=0).values
    support = ((luma > threshold) | (chroma > threshold * 0.75)).float().unsqueeze(0)
    return _binary_close(support, close_kernel)


def _project_points(points: torch.Tensor, view) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ndc = geom_transform_points(points.detach(), view.full_proj_transform)
    width = int(view.image_width)
    height = int(view.image_height)
    px = (ndc[:, 0] + 1.0) * 0.5 * float(max(width - 1, 1))
    py = (1.0 - (ndc[:, 1] + 1.0) * 0.5) * float(max(height - 1, 1))
    valid = torch.isfinite(ndc).all(dim=-1)
    valid &= ndc[:, 2] > 0.0
    valid &= px >= 0.0
    valid &= px <= float(max(width - 1, 0))
    valid &= py >= 0.0
    valid &= py <= float(max(height - 1, 0))
    return torch.stack((px, py), dim=-1), valid, ndc


def _distance_to_mask(mask_np: np.ndarray) -> np.ndarray:
    if bool(mask_np.any()):
        return cv2.distanceTransform((~mask_np).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    return np.full(mask_np.shape, np.inf, dtype=np.float32)


def _sample_distance(dist_np: np.ndarray, xy: torch.Tensor) -> torch.Tensor:
    h, w = dist_np.shape
    x = xy[:, 0].round().long().clamp(0, w - 1).detach().cpu().numpy()
    y = xy[:, 1].round().long().clamp(0, h - 1).detach().cpu().numpy()
    return torch.from_numpy(dist_np[y, x]).to(device=xy.device, dtype=torch.float32)


def _point_residual_score(
    residual_mask: torch.Tensor,
    xy: torch.Tensor,
    valid: torch.Tensor,
    radii: torch.Tensor,
    opacity: torch.Tensor,
    *,
    radius_scale: float,
    min_radius: float,
    max_radius: float,
    opacity_power: float,
    radius_power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    point_count = int(xy.shape[0])
    score = torch.zeros((point_count,), dtype=torch.float32, device=xy.device)
    distance = torch.full((point_count,), float("inf"), dtype=torch.float32, device=xy.device)
    mask_np = residual_mask.detach().reshape(residual_mask.shape[-2], residual_mask.shape[-1]).cpu().numpy() > 0.5
    if not bool(mask_np.any()):
        return score, distance

    dist_np = _distance_to_mask(mask_np)
    sampled_dist = _sample_distance(dist_np, xy)
    support_radius = (radii.detach().float().to(xy.device) * float(radius_scale)).clamp(
        min=float(min_radius),
        max=float(max_radius),
    )
    valid_mask = valid.detach().bool().to(xy.device) & torch.isfinite(sampled_dist) & (support_radius > 0.0)
    overlap = (1.0 - sampled_dist / support_radius.clamp_min(1.0e-6)).clamp(0.0, 1.0)
    opacity_weight = opacity.detach().reshape(-1).float().to(xy.device).clamp(0.0, 1.0).pow(float(opacity_power))
    radius_weight = (support_radius / max(float(max_radius), 1.0e-6)).clamp(0.0, 1.0).pow(float(radius_power))
    score = overlap * opacity_weight * radius_weight
    score = torch.where(valid_mask, score, torch.zeros_like(score))
    distance = torch.where(valid_mask, sampled_dist, distance)
    return score, distance


def _component_records(
    residual_mask: torch.Tensor,
    direction: str,
    frame_record: dict,
    xy: torch.Tensor,
    valid: torch.Tensor,
    radii: torch.Tensor,
    opacity: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> list[dict]:
    mask_np = residual_mask.detach().reshape(residual_mask.shape[-2], residual_mask.shape[-1]).cpu().numpy() > 0.5
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np.astype(np.uint8), 8)
    records: list[dict] = []
    if n_labels <= 1:
        return records
    component_ids = list(range(1, n_labels))
    component_ids.sort(key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]), reverse=True)
    component_ids = component_ids[: int(args.max_components_per_frame)]

    radius_scale = args.outer_radius_scale if direction == "outer" else args.inner_radius_scale
    for comp_id in component_ids:
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area < int(args.min_component_area):
            continue
        comp_mask_np = labels == comp_id
        comp_mask = torch.from_numpy(comp_mask_np.astype(np.float32)).to(device=xy.device).unsqueeze(0)
        comp_score, comp_dist = _point_residual_score(
            comp_mask,
            xy,
            valid,
            radii,
            opacity,
            radius_scale=radius_scale,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            opacity_power=args.opacity_power,
            radius_power=args.radius_power,
        )
        active = comp_score > 0.0
        top_values, top_indices = torch.topk(comp_score, k=min(int(args.component_top_points), int(comp_score.numel())))
        keep = top_values > 0.0
        top_indices = top_indices[keep]
        top_values = top_values[keep]
        if bool(valid.any().item()):
            min_center_dist = float(comp_dist[valid].min().item())
        else:
            min_center_dist = float("inf")
        records.append({
            "cam": frame_record["cam"],
            "frame": frame_record["frame"],
            "image_name": frame_record["image_name"],
            "direction": direction,
            "component_id": int(comp_id),
            "area": area,
            "bbox_x": int(stats[comp_id, cv2.CC_STAT_LEFT]),
            "bbox_y": int(stats[comp_id, cv2.CC_STAT_TOP]),
            "bbox_w": int(stats[comp_id, cv2.CC_STAT_WIDTH]),
            "bbox_h": int(stats[comp_id, cv2.CC_STAT_HEIGHT]),
            "centroid_x": float(centroids[comp_id][0]),
            "centroid_y": float(centroids[comp_id][1]),
            "near_point_count": int(active.sum().item()),
            "near_score_sum": float(comp_score.sum().item()),
            "min_center_dist": min_center_dist,
            "top_point_ids": ";".join(str(int(x)) for x in top_indices.detach().cpu().tolist()),
            "top_point_scores": ";".join(f"{float(x):.6f}" for x in top_values.detach().cpu().tolist()),
        })
    return records


def _tensor_attr(pc, name: str, point_count: int, width: int | None = None):
    value = getattr(pc, name, None)
    if not torch.is_tensor(value) or value.shape[0] != point_count:
        if width is None:
            return torch.zeros((point_count,), dtype=torch.float32, device=pc.get_xyz.device)
        return torch.zeros((point_count, width), dtype=torch.float32, device=pc.get_xyz.device)
    return value.detach()


def _safe_float(value) -> float:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().item())
    try:
        return float(value)
    except Exception:
        return 0.0


def _update_attr_sums(attr_sums: dict[str, torch.Tensor], pc, point_count: int) -> None:
    specs = {
        "layer": ("binding_weights", 3),
        "region": ("binding_region_probs", 3),
        "boundary_score": ("binding_boundary_score", None),
        "surface_distance": ("binding_surface_distance", None),
        "thin_score": ("binding_thin_score", None),
    }
    for out_name, (attr_name, width) in specs.items():
        value = _tensor_attr(pc, attr_name, point_count, width)
        attr_sums[out_name] += value.float()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_overlay(render_rgb: torch.Tensor, inner: torch.Tensor, outer: torch.Tensor, out_path: Path) -> None:
    image = render_rgb.detach().float().cpu().clamp(0.0, 1.0)
    overlay = image.clone()
    inner_mask = inner.detach().cpu().reshape(inner.shape[-2], inner.shape[-1]) > 0.5
    outer_mask = outer.detach().cpu().reshape(outer.shape[-2], outer.shape[-1]) > 0.5
    overlay[:, inner_mask] = torch.tensor([0.10, 0.95, 0.25]).reshape(3, 1)
    overlay[:, outer_mask] = torch.tensor([1.0, 0.08, 0.10]).reshape(3, 1)
    panel = torch.cat([image, overlay], dim=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(panel, str(out_path))


def _build_config(args: argparse.Namespace):
    config = OmegaConf.load(args.config_path)
    OmegaConf.set_struct(config, False)
    config.mode = "test"
    config.exp_dir = str(args.out_dir / "scene")
    config.load_ckpt = str(args.load_ckpt)
    config.dataset.root_dir = str(args.dataset_root)
    config.dataset.preload = False
    config.dataset.test_views.view = _parse_csv_ints(args.eval_views)
    config.dataset.test_frames.view = _triple_or_values(args.eval_frames)
    config.dataset.parsing_prior.enable = False
    config.dataset.parsing_prior.roi_enable = False
    config.wandb_disable = True
    config.export_interpretability = False
    config.export_semantic_editable_assets = False
    config.render_export_refine = False
    config.render_scaling_modifier = float(args.render_scaling_modifier)
    config.model.deformer.rigid.rotation_orthogonalize_enable = bool(args.rotation_orthogonalize)
    config.pipeline.compute_cov3D_python = bool(args.compute_cov3d_python)
    config.opt.camera_geometry_enable = bool(args.camera_geometry)
    config.opt.camera_geometry_lr = 0.0
    if "resume" not in config:
        config.resume = {}
    config.resume.allow_partial_converter_load = False
    config.resume.restore_gaussian_optimizer_state = False
    config.resume.restore_converter_optimizer_state = False
    config.resume.restore_converter_scheduler_state = False
    config.resume.disable_densify_on_resume = True
    config.resume.disable_opacity_reset_on_resume = True
    return config


def _summarize_group(rows: list[dict], key: str) -> list[dict]:
    grouped: dict[str, dict] = {}
    for row in rows:
        value = str(row.get(key, ""))
        item = grouped.setdefault(value, {"group": value, "count": 0, "over_sum": 0.0, "under_sum": 0.0})
        item["count"] += 1
        item["over_sum"] += float(row.get("over_score_sum", 0.0))
        item["under_sum"] += float(row.get("under_score_sum", 0.0))
    return sorted(grouped.values(), key=lambda x: max(x["over_sum"], x["under_sum"]), reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="v274 per-Gaussian contributor audit for CoreView_377 boundary residuals.")
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--load-ckpt", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--eval-views", default="21,22,23")
    parser.add_argument("--eval-frames", default="0,570,60")
    parser.add_argument("--top-frames", type=int, default=12)
    parser.add_argument("--render-support-threshold", type=float, default=0.025)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--band-width", type=int, default=7)
    parser.add_argument("--search-band-width", type=int, default=24)
    parser.add_argument("--residual-dilate", type=int, default=1)
    parser.add_argument("--outer-radius-scale", type=float, default=1.20)
    parser.add_argument("--inner-radius-scale", type=float, default=1.75)
    parser.add_argument("--min-radius", type=float, default=1.5)
    parser.add_argument("--max-radius", type=float, default=18.0)
    parser.add_argument("--opacity-power", type=float, default=0.50)
    parser.add_argument("--radius-power", type=float, default=0.35)
    parser.add_argument("--min-component-area", type=int, default=18)
    parser.add_argument("--max-components-per-frame", type=int, default=8)
    parser.add_argument("--component-top-points", type=int, default=8)
    parser.add_argument("--top-points", type=int, default=160)
    parser.add_argument("--candidate-points", type=int, default=96)
    parser.add_argument("--render-scaling-modifier", type=float, default=1.0)
    parser.add_argument("--rotation-orthogonalize", action="store_true")
    parser.add_argument("--compute-cov3d-python", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-geometry", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.out_dir / "overlays"
    render_dir = args.out_dir / "renders"
    overlay_dir.mkdir(exist_ok=True)
    render_dir.mkdir(exist_ok=True)

    config = _build_config(args)
    gaussians = GaussianModel(config.model.gaussian)
    scene = Scene(config, gaussians, str(args.out_dir / "scene"))
    scene.eval()
    loaded_iteration = scene.load_checkpoint(str(args.load_ckpt))

    bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    frame_records = []
    with torch.no_grad():
        for view_idx, view in enumerate(scene.test_dataset):
            render_pkg = render(
                view,
                loaded_iteration,
                scene,
                config.pipeline,
                background,
                scaling_modifier=float(args.render_scaling_modifier),
                compute_loss=False,
                return_opacity=False,
            )
            render_rgb = render_pkg["render"].detach().clamp(0.0, 1.0)
            gt_mask = _mask_from_view(view)
            support = _render_support_mask(render_rgb, args.render_support_threshold, args.close_kernel)
            near_gt = _dilate(gt_mask, args.search_band_width)
            inner = (gt_mask * (1.0 - support)).clamp(0.0, 1.0)
            outer = (support * (1.0 - gt_mask) * near_gt).clamp(0.0, 1.0)
            band = _boundary_band(gt_mask, args.band_width)
            outer_band = (outer * _boundary_band(gt_mask, max(args.band_width * 2, 3))).clamp(0.0, 1.0)
            inner_band = (inner * band).clamp(0.0, 1.0)
            fg_pixels = max(float(gt_mask.sum().item()), 1.0)
            record = {
                "view_idx": view_idx,
                "image_name": str(view.image_name),
                "cam": str(view.cam_id),
                "frame": int(view.frame_id),
                "fg_pixels": int(gt_mask.sum().item()),
                "render_support_pixels": int(support.sum().item()),
                "inner_missing_pixels": int(inner.sum().item()),
                "outer_leak_pixels": int(outer.sum().item()),
                "inner_band_pixels": int(inner_band.sum().item()),
                "outer_band_pixels": int(outer_band.sum().item()),
                "hard_residual_score": float((inner_band.sum() + outer_band.sum()).item() / fg_pixels),
            }
            frame_records.append(record)
            del render_pkg

    frame_records.sort(key=lambda item: item["hard_residual_score"], reverse=True)
    audit_records = frame_records[: int(args.top_frames)]

    with torch.no_grad():
        first_view = scene.test_dataset[audit_records[0]["view_idx"]]
        first_pkg = render(
            first_view,
            loaded_iteration,
            scene,
            config.pipeline,
            background,
            scaling_modifier=float(args.render_scaling_modifier),
            compute_loss=False,
            return_opacity=False,
        )
    point_count = int(first_pkg["deformed_gaussian"].get_xyz.shape[0])
    device = first_pkg["deformed_gaussian"].get_xyz.device
    over_sum = torch.zeros((point_count,), dtype=torch.float32, device=device)
    under_sum = torch.zeros((point_count,), dtype=torch.float32, device=device)
    over_max = torch.zeros((point_count,), dtype=torch.float32, device=device)
    under_max = torch.zeros((point_count,), dtype=torch.float32, device=device)
    over_frames = torch.zeros((point_count,), dtype=torch.float32, device=device)
    under_frames = torch.zeros((point_count,), dtype=torch.float32, device=device)
    visible_frames = torch.zeros((point_count,), dtype=torch.float32, device=device)
    radius_sum = torch.zeros((point_count,), dtype=torch.float32, device=device)
    radius_max = torch.zeros((point_count,), dtype=torch.float32, device=device)
    attr_sums = {
        "layer": torch.zeros((point_count, 3), dtype=torch.float32, device=device),
        "region": torch.zeros((point_count, 3), dtype=torch.float32, device=device),
        "boundary_score": torch.zeros((point_count,), dtype=torch.float32, device=device),
        "surface_distance": torch.zeros((point_count,), dtype=torch.float32, device=device),
        "thin_score": torch.zeros((point_count,), dtype=torch.float32, device=device),
    }
    attr_count = torch.zeros((point_count,), dtype=torch.float32, device=device)
    frame_top_rows: list[dict] = []
    component_rows: list[dict] = []

    with torch.no_grad():
        for record in audit_records:
            view = scene.test_dataset[record["view_idx"]]
            render_pkg = render(
                view,
                loaded_iteration,
                scene,
                config.pipeline,
                background,
                scaling_modifier=float(args.render_scaling_modifier),
                compute_loss=False,
                return_opacity=False,
            )
            render_rgb = render_pkg["render"].detach().clamp(0.0, 1.0)
            gt_mask = _mask_from_view(view)
            support = _render_support_mask(render_rgb, args.render_support_threshold, args.close_kernel)
            near_gt = _dilate(gt_mask, args.search_band_width)
            inner = (gt_mask * (1.0 - support)).clamp(0.0, 1.0)
            outer = (support * (1.0 - gt_mask) * near_gt).clamp(0.0, 1.0)
            band = _boundary_band(gt_mask, args.band_width)
            inner_band = (inner * band).clamp(0.0, 1.0)
            outer_band = (outer * _boundary_band(gt_mask, max(args.band_width * 2, 3))).clamp(0.0, 1.0)
            pc = render_pkg["deformed_gaussian"]
            xy, valid, _ = _project_points(pc.get_xyz, view)
            radii = render_pkg["radii"].detach().float().to(device)
            opacity = pc.get_opacity.detach().reshape(-1).float().to(device)
            visible = valid & (radii > 0.0)
            visible_frames += visible.float()
            radius_sum += torch.where(visible, radii, torch.zeros_like(radii))
            radius_max = torch.maximum(radius_max, torch.where(visible, radii, torch.zeros_like(radii)))
            _update_attr_sums(attr_sums, pc, point_count)
            attr_count += 1.0

            inner_score_mask = _dilate(inner_band, args.residual_dilate)
            outer_score_mask = _dilate(outer_band, args.residual_dilate)
            frame_over, frame_over_dist = _point_residual_score(
                outer_score_mask,
                xy,
                visible,
                radii,
                opacity,
                radius_scale=args.outer_radius_scale,
                min_radius=args.min_radius,
                max_radius=args.max_radius,
                opacity_power=args.opacity_power,
                radius_power=args.radius_power,
            )
            frame_under, frame_under_dist = _point_residual_score(
                inner_score_mask,
                xy,
                visible,
                radii,
                opacity,
                radius_scale=args.inner_radius_scale,
                min_radius=args.min_radius,
                max_radius=args.max_radius,
                opacity_power=args.opacity_power,
                radius_power=args.radius_power,
            )

            over_sum += frame_over
            under_sum += frame_under
            over_max = torch.maximum(over_max, frame_over)
            under_max = torch.maximum(under_max, frame_under)
            over_frames += (frame_over > 0.0).float()
            under_frames += (frame_under > 0.0).float()

            for direction, scores, distances in (
                ("outer", frame_over, frame_over_dist),
                ("inner", frame_under, frame_under_dist),
            ):
                top_values, top_indices = torch.topk(scores, k=min(int(args.component_top_points), point_count))
                keep = top_values > 0.0
                for rank, (pid, value) in enumerate(zip(top_indices[keep].detach().cpu().tolist(), top_values[keep].detach().cpu().tolist()), start=1):
                    frame_top_rows.append({
                        "cam": record["cam"],
                        "frame": record["frame"],
                        "image_name": record["image_name"],
                        "direction": direction,
                        "rank": rank,
                        "point_idx": int(pid),
                        "score": float(value),
                        "distance_px": float(distances[int(pid)].item()) if torch.isfinite(distances[int(pid)]) else float("inf"),
                        "radius_px": float(radii[int(pid)].item()),
                        "opacity": float(opacity[int(pid)].item()),
                    })

            component_rows.extend(_component_records(
                outer_score_mask,
                "outer",
                record,
                xy,
                visible,
                radii,
                opacity,
                args=args,
            ))
            component_rows.extend(_component_records(
                inner_score_mask,
                "inner",
                record,
                xy,
                visible,
                radii,
                opacity,
                args=args,
            ))

            torchvision.utils.save_image(render_rgb, str(render_dir / f"render_{record['image_name']}.png"))
            _make_overlay(render_rgb, inner_band, outer_band, overlay_dir / f"residual_{record['image_name']}.png")
            del render_pkg

    attr_count_safe = attr_count.clamp_min(1.0)
    layer_mean = attr_sums["layer"] / attr_count_safe.unsqueeze(-1)
    region_mean = attr_sums["region"] / attr_count_safe.unsqueeze(-1)
    boundary_mean = attr_sums["boundary_score"] / attr_count_safe
    surface_mean = attr_sums["surface_distance"] / attr_count_safe
    thin_mean = attr_sums["thin_score"] / attr_count_safe
    radius_mean = radius_sum / visible_frames.clamp_min(1.0)
    opacity_ref = first_pkg["deformed_gaussian"].get_opacity.detach().reshape(-1).float().to(device)
    scale_ref = first_pkg["deformed_gaussian"].get_scaling.detach().float().to(device)
    xyz_ref = first_pkg["deformed_gaussian"].get_xyz.detach().float().to(device)
    canonical_ref = getattr(first_pkg["deformed_gaussian"], "canonical_xyz", None)
    if not torch.is_tensor(canonical_ref) or canonical_ref.shape[0] != point_count:
        canonical_ref = scene.gaussians.get_xyz.detach()
    canonical_ref = canonical_ref.detach().float().to(device)
    layer_id = torch.argmax(layer_mean, dim=-1)
    region_id = torch.argmax(region_mean, dim=-1)
    dominant_joint = _tensor_attr(first_pkg["deformed_gaussian"], "binding_dominant_joint", point_count).reshape(-1).long()

    point_rows: list[dict] = []
    active_mask = (over_sum > 0.0) | (under_sum > 0.0)
    active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
    for pid in active_indices.detach().cpu().tolist():
        pid = int(pid)
        over = float(over_sum[pid].item())
        under = float(under_sum[pid].item())
        row = {
            "point_idx": pid,
            "over_score_sum": over,
            "under_score_sum": under,
            "signed_score": over - under,
            "over_score_max": float(over_max[pid].item()),
            "under_score_max": float(under_max[pid].item()),
            "over_frame_hits": int(over_frames[pid].item()),
            "under_frame_hits": int(under_frames[pid].item()),
            "visible_frame_hits": int(visible_frames[pid].item()),
            "radius_px_mean": float(radius_mean[pid].item()),
            "radius_px_max": float(radius_max[pid].item()),
            "opacity": float(opacity_ref[pid].item()),
            "scale_min": float(scale_ref[pid].min().item()),
            "scale_mean": float(scale_ref[pid].mean().item()),
            "layer_id": int(layer_id[pid].item()),
            "layer_name": LAYER_NAMES[int(layer_id[pid].item())] if int(layer_id[pid].item()) < len(LAYER_NAMES) else "unknown",
            "layer_rigid": float(layer_mean[pid, 0].item()),
            "layer_soft": float(layer_mean[pid, 1].item()),
            "layer_free": float(layer_mean[pid, 2].item()),
            "region_id": int(region_id[pid].item()),
            "region_name": REGION_NAMES[int(region_id[pid].item())] if int(region_id[pid].item()) < len(REGION_NAMES) else "unknown",
            "region_body": float(region_mean[pid, 0].item()),
            "region_soft": float(region_mean[pid, 1].item()),
            "region_cloth": float(region_mean[pid, 2].item()),
            "dominant_joint": int(dominant_joint[pid].item()),
            "boundary_score": float(boundary_mean[pid].item()),
            "surface_distance": float(surface_mean[pid].item()),
            "thin_score": float(thin_mean[pid].item()),
            "canonical_x": float(canonical_ref[pid, 0].item()),
            "canonical_y": float(canonical_ref[pid, 1].item()),
            "canonical_z": float(canonical_ref[pid, 2].item()),
            "deformed_x": float(xyz_ref[pid, 0].item()),
            "deformed_y": float(xyz_ref[pid, 1].item()),
            "deformed_z": float(xyz_ref[pid, 2].item()),
        }
        point_rows.append(row)

    point_rows.sort(key=lambda row: max(float(row["over_score_sum"]), float(row["under_score_sum"])), reverse=True)
    over_rows = sorted(point_rows, key=lambda row: float(row["over_score_sum"]), reverse=True)[: int(args.top_points)]
    under_rows = sorted(point_rows, key=lambda row: float(row["under_score_sum"]), reverse=True)[: int(args.top_points)]

    shrink_ids = [
        int(row["point_idx"])
        for row in over_rows
        if float(row["over_score_sum"]) > 0.0
        and float(row["over_score_sum"]) >= max(float(row["under_score_sum"]) * 1.25, 1.0e-6)
    ][: int(args.candidate_points)]
    grow_ids = [
        int(row["point_idx"])
        for row in under_rows
        if float(row["under_score_sum"]) > 0.0
        and float(row["under_score_sum"]) >= max(float(row["over_score_sum"]) * 1.10, 1.0e-6)
    ][: int(args.candidate_points)]

    sample_metrics_path = args.out_dir / "sample_metrics.csv"
    _write_csv(sample_metrics_path, sorted(frame_records, key=lambda item: (str(item["cam"]), int(item["frame"]))))
    _write_csv(args.out_dir / "audited_frame_top_points.csv", frame_top_rows)
    _write_csv(args.out_dir / "component_contributors.csv", component_rows)
    _write_csv(args.out_dir / "point_contributors_all.csv", point_rows)
    _write_csv(args.out_dir / "point_contributors_top_over.csv", over_rows)
    _write_csv(args.out_dir / "point_contributors_top_under.csv", under_rows)

    summary = {
        "load_ckpt": str(args.load_ckpt),
        "loaded_iteration": int(loaded_iteration),
        "n_samples": len(frame_records),
        "top_frames": audit_records,
        "mean_inner_missing_pixels": float(np.mean([r["inner_missing_pixels"] for r in frame_records])) if frame_records else 0.0,
        "mean_outer_leak_pixels": float(np.mean([r["outer_leak_pixels"] for r in frame_records])) if frame_records else 0.0,
        "mean_hard_residual_score": float(np.mean([r["hard_residual_score"] for r in frame_records])) if frame_records else 0.0,
        "active_contributor_count": int(len(point_rows)),
        "top_over": over_rows[:20],
        "top_under": under_rows[:20],
        "group_by_layer_over": _summarize_group(over_rows, "layer_name"),
        "group_by_region_over": _summarize_group(over_rows, "region_name"),
        "group_by_layer_under": _summarize_group(under_rows, "layer_name"),
        "group_by_region_under": _summarize_group(under_rows, "region_name"),
        "candidate_point_sets": {
            "shrink_point_ids": shrink_ids,
            "grow_point_ids": grow_ids,
        },
    }
    (args.out_dir / "contributor_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "v275_candidate_point_sets.json").write_text(
        json.dumps(summary["candidate_point_sets"], indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
