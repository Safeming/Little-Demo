#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
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


def _parse_ints(value: str) -> list[int]:
    out = []
    for part in str(value or "").replace("[", "").replace("]", "").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _triple_or_values(value: str) -> list[int]:
    return _parse_ints(value)


def _dilate(mask: torch.Tensor, width: int) -> torch.Tensor:
    width = max(0, int(width))
    if width <= 0:
        return mask
    kernel = 2 * width + 1
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


def _project_points(points: torch.Tensor, view) -> tuple[torch.Tensor, torch.Tensor]:
    ndc = geom_transform_points(points.detach(), view.full_proj_transform.to(device=points.device, dtype=points.dtype))
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
    return torch.stack((px, py), dim=-1), valid


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
) -> torch.Tensor:
    point_count = int(xy.shape[0])
    score = torch.zeros((point_count,), dtype=torch.float32, device=xy.device)
    mask_np = residual_mask.detach().reshape(residual_mask.shape[-2], residual_mask.shape[-1]).cpu().numpy() > 0.5
    if not bool(mask_np.any()):
        return score
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
    return torch.where(valid_mask, overlap * opacity_weight * radius_weight, score)


def _tensor_attr(pc, name: str, point_count: int, width: int | None = None) -> torch.Tensor:
    value = getattr(pc, name, None)
    if torch.is_tensor(value) and value.shape[0] == point_count:
        return value.detach()
    if width is None:
        return torch.zeros((point_count,), dtype=torch.float32, device=pc.get_xyz.device)
    return torch.zeros((point_count, width), dtype=torch.float32, device=pc.get_xyz.device)


def _screen_delta(pc, view, attr_name: str, reference_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = getattr(pc, attr_name, None)
    point_count = int(pc.get_xyz.shape[0])
    if not torch.is_tensor(target) or target.shape[0] != point_count:
        return torch.zeros((point_count,), dtype=torch.float32, device=pc.get_xyz.device), torch.zeros((point_count, 2), dtype=torch.float32, device=pc.get_xyz.device)
    xy, valid = _project_points(target.detach(), view)
    delta = xy - reference_xy
    dist = torch.linalg.norm(delta, dim=-1)
    dist = torch.where(valid, dist, torch.full_like(dist, float("nan")))
    return dist, delta


def _build_config(args: argparse.Namespace):
    config = OmegaConf.load(args.config_path)
    OmegaConf.set_struct(config, False)
    config.mode = "test"
    config.exp_dir = str(args.out_dir / "scene")
    config.load_ckpt = str(args.load_ckpt)
    config.dataset.root_dir = str(args.dataset_root)
    config.dataset.preload = False
    config.dataset.train_views = _parse_ints(args.train_views)
    config.dataset.train_frames = _triple_or_values(args.train_frames)
    config.dataset.test_views.view = _parse_ints(args.eval_views)
    config.dataset.test_frames.view = _triple_or_values(args.eval_frames)
    config.dataset.parsing_prior.enable = False
    config.dataset.parsing_prior.roi_enable = False
    config.wandb_disable = True
    config.export_interpretability = False
    config.export_semantic_editable_assets = False
    config.render_export_refine = False
    config.pipeline.compute_cov3D_python = True
    config.opt.camera_geometry_enable = True
    config.opt.camera_geometry_lr = 0.0
    if "resume" not in config:
        config.resume = {}
    config.resume.allow_partial_converter_load = True
    config.resume.restore_gaussian_optimizer_state = False
    config.resume.restore_converter_optimizer_state = False
    config.resume.restore_converter_scheduler_state = False
    return config


def _apply_adopted_preset(config, component_csv: Path, point_csv: Path, args: argparse.Namespace) -> None:
    config.pipeline.compute_cov3D_python = True
    config.pipeline.covariance_mode = "default"
    config.pipeline.covariance_signed_dynamic_enable = True
    config.pipeline.covariance_signed_dynamic_component_csv = str(component_csv)
    config.pipeline.covariance_signed_dynamic_point_csv = str(point_csv)
    config.pipeline.covariance_signed_dynamic_component_signature_enable = False
    config.pipeline.covariance_signed_dynamic_over_layer_ids = "soft,free"
    config.pipeline.covariance_signed_dynamic_over_region_ids = "cloth"
    config.pipeline.covariance_signed_dynamic_over_joint_ids = "6,9,12,13,14,15"
    config.pipeline.covariance_signed_dynamic_under_layer_ids = "soft,rigid,free"
    config.pipeline.covariance_signed_dynamic_under_region_ids = "cloth,body,soft"
    config.pipeline.covariance_signed_dynamic_under_joint_ids = "0,1,2,4,7,8,10"
    config.pipeline.covariance_signed_dynamic_boundary_min = 0.0
    config.pipeline.covariance_signed_dynamic_component_pad_px = 10
    config.pipeline.covariance_signed_dynamic_component_ellipse_scale = 1.25
    config.pipeline.covariance_signed_dynamic_component_max_over = 16
    config.pipeline.covariance_signed_dynamic_component_max_under = 16
    config.pipeline.covariance_signed_dynamic_component_min_area = 20
    config.pipeline.covariance_signed_dynamic_component_required = True
    config.pipeline.covariance_signed_dynamic_component_top_ids_enable = False
    config.pipeline.covariance_signed_dynamic_component_top_ids_only = False
    config.pipeline.covariance_signed_dynamic_max_over_points = 96
    config.pipeline.covariance_signed_dynamic_max_under_points = 96
    config.pipeline.covariance_signed_screen_actuator_enable = True
    config.pipeline.covariance_signed_screen_normal_shrink_factor = 0.940
    config.pipeline.covariance_signed_screen_normal_grow_factor = 1.025
    config.pipeline.covariance_signed_screen_tangent_factor = 1.000
    config.pipeline.covariance_signed_center_offset_enable = True
    config.pipeline.covariance_signed_center_offset_outer_px = float(args.outer_px)
    config.pipeline.covariance_signed_center_offset_inner_px = 0.0
    config.pipeline.covariance_signed_center_offset_outer_direction = "view_center"
    config.pipeline.covariance_signed_center_offset_inner_direction = "component_center"
    config.pipeline.covariance_signed_center_offset_score_weight_power = 1.0
    config.pipeline.covariance_signed_center_offset_score_weight_min = 0.15
    config.pipeline.covariance_signed_center_offset_score_weight_quantile = 0.90
    config.pipeline.covariance_signed_center_offset_jacobian_eps = 0.001
    config.pipeline.covariance_signed_center_offset_jacobian_damping = 0.00001
    config.pipeline.covariance_signed_center_offset_max_world_step = 0.0020
    config.pipeline.boundary_cov_residual_enable = False
    config.pipeline.binding_covariance_guard_enable = False
    config.model.deformer.rigid.rotation_orthogonalize_enable = False
    config.model.deformer.rigid.geometry_fidelity_gate_enable = True
    config.model.deformer.rigid.geometry_fidelity_target = "free_lbs"
    config.model.deformer.rigid.geometry_fidelity_center_strength = float(args.center_strength)
    config.model.deformer.rigid.geometry_fidelity_rotation_strength = 0.0
    config.model.deformer.rigid.geometry_fidelity_boundary_min = 0.12
    config.model.deformer.rigid.geometry_fidelity_layer_ids = "soft,free"
    config.model.deformer.rigid.geometry_fidelity_region_ids = "cloth,soft"
    config.model.deformer.rigid.geometry_fidelity_joint_ids = ""
    config.model.deformer.rigid.geometry_fidelity_thin_min = ""
    config.model.deformer.rigid.geometry_fidelity_surface_min = ""
    config.model.deformer.rigid.geometry_fidelity_surface_max = ""
    config.model.deformer.rigid.geometry_fidelity_non_rigid_min = 0.0
    config.model.deformer.rigid.geometry_fidelity_power = 1.2
    config.model.deformer.rigid.geometry_fidelity_max_points = 1024
    config.model.deformer.rigid.geometry_fidelity_component_enable = True
    config.model.deformer.rigid.geometry_fidelity_component_csv = str(component_csv)
    config.model.deformer.rigid.geometry_fidelity_component_direction = "inner"
    config.model.deformer.rigid.geometry_fidelity_component_pad_px = 2
    config.model.deformer.rigid.geometry_fidelity_component_ellipse_scale = 1.05
    config.model.deformer.rigid.geometry_fidelity_component_max = 12
    config.model.deformer.rigid.geometry_fidelity_component_min_area = 40
    config.model.deformer.rigid.geometry_fidelity_component_required = True
    config.model.deformer.rigid.geometry_fidelity_component_improvement_enable = True
    config.model.deformer.rigid.geometry_fidelity_component_improvement_margin_px = 0.0


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float:
    mask = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0.0)
    if not bool(mask.any().item()):
        return 0.0
    return float((values[mask] * weights[mask]).sum().item() / weights[mask].sum().clamp_min(1.0e-6).item())


def _weighted_quantile(values: torch.Tensor, weights: torch.Tensor, q: float) -> float:
    mask = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0.0)
    if not bool(mask.any().item()):
        return 0.0
    v = values[mask].detach().float().cpu()
    w = weights[mask].detach().float().cpu()
    order = torch.argsort(v)
    v = v[order]
    w = w[order]
    cdf = torch.cumsum(w, dim=0) / w.sum().clamp_min(1.0e-6)
    idx = int(torch.searchsorted(cdf, torch.tensor(float(q))).clamp(max=v.numel() - 1).item())
    return float(v[idx].item())


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    mask = torch.isfinite(a) & torch.isfinite(b)
    if int(mask.sum().item()) < 3:
        return 0.0
    x = a[mask].detach().float()
    y = b[mask].detach().float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.square().sum() * y.square().sum()).clamp_min(1.0e-12))
    return float((x * y).sum().item() / denom.item())


def _summarize_direction(rows: list[dict], direction: str, score_key: str, metrics: list[str]) -> dict:
    if not rows:
        return {}
    weights = torch.tensor([float(row[score_key]) for row in rows], dtype=torch.float32)
    summary = {"count": len(rows), "score_sum": float(weights.sum().item())}
    for metric in metrics:
        values = torch.tensor([float(row[metric]) for row in rows], dtype=torch.float32)
        summary[f"{direction}_{metric}_weighted_mean"] = _weighted_mean(values, weights)
        summary[f"{direction}_{metric}_weighted_p90"] = _weighted_quantile(values, weights, 0.90)
        summary[f"{direction}_{metric}_corr"] = _corr(values, weights)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="v312 audit: correlate StageB boundary residual contributors with explicit-binding geometry mismatch.")
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--load-ckpt", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--component-csv", required=True, type=Path)
    parser.add_argument("--point-csv", required=True, type=Path)
    parser.add_argument("--variant", choices=["baseline", "adopted"], default="baseline")
    parser.add_argument("--train-views", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--train-frames", default="0,570,60")
    parser.add_argument("--eval-views", default="21,22,23")
    parser.add_argument("--eval-frames", default="0,570,60")
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
    parser.add_argument("--top-points", type=int, default=256)
    parser.add_argument("--save-renders", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--center-strength", type=float, default=0.45)
    parser.add_argument("--outer-px", type=float, default=0.35)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_renders:
        (args.out_dir / "renders").mkdir(exist_ok=True)

    config = _build_config(args)
    if args.variant == "adopted":
        _apply_adopted_preset(config, args.component_csv, args.point_csv, args)
    gaussians = GaussianModel(config.model.gaussian)
    scene = Scene(config, gaussians, str(args.out_dir / "scene"))
    scene.eval()
    loaded_iteration = scene.load_checkpoint(str(args.load_ckpt))
    bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    rows: list[dict] = []
    frame_rows: list[dict] = []
    with torch.no_grad():
        for view_idx, view in enumerate(scene.test_dataset):
            render_pkg = render(
                view,
                loaded_iteration,
                scene,
                config.pipeline,
                background,
                scaling_modifier=1.0,
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
            point_count = int(pc.get_xyz.shape[0])
            xy, valid = _project_points(pc.get_xyz, view)
            radii = render_pkg["radii"].detach().float().to(xy.device)
            opacity = pc.get_opacity.detach().reshape(-1).float().to(xy.device)
            visible = valid & (radii > 0.0)
            inner_score = _point_residual_score(
                _dilate(inner_band, args.residual_dilate),
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
            outer_score = _point_residual_score(
                _dilate(outer_band, args.residual_dilate),
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
            xfree_dist, xfree_delta = _screen_delta(pc, view, "binding_x_free", xy)
            xbar_base_dist, xbar_base_delta = _screen_delta(pc, view, "binding_x_bar_base", xy)
            pre_geom_dist, pre_geom_delta = _screen_delta(pc, view, "binding_x_bar_pre_geometry_fidelity", xy)
            xrigid_dist, _ = _screen_delta(pc, view, "binding_x_rigid", xy)
            xsoft_dist, _ = _screen_delta(pc, view, "binding_x_soft", xy)
            geometry_target_dist, geometry_target_delta = _screen_delta(pc, view, "binding_geometry_target_xyz", xy)
            free_delta_norm = torch.linalg.norm(xfree_delta, dim=-1)
            free_delta_x = xfree_delta[:, 0]
            free_delta_y = xfree_delta[:, 1]
            target_delta_x = geometry_target_delta[:, 0]
            target_delta_y = geometry_target_delta[:, 1]
            layer_ids = _tensor_attr(pc, "binding_layer_ids", point_count).reshape(-1).long()
            region_ids = _tensor_attr(pc, "binding_region_ids", point_count).reshape(-1).long()
            dominant_joint = _tensor_attr(pc, "binding_dominant_joint", point_count).reshape(-1).long()
            boundary_score = _tensor_attr(pc, "binding_boundary_score", point_count).reshape(-1).float()
            surface_distance = _tensor_attr(pc, "binding_surface_distance", point_count).reshape(-1).float()
            thin_score = _tensor_attr(pc, "binding_thin_score", point_count).reshape(-1).float()
            fidelity_weight = _tensor_attr(pc, "binding_geometry_fidelity_weight", point_count).reshape(-1).float()
            center_blend = _tensor_attr(pc, "binding_geometry_fidelity_center_blend", point_count).reshape(-1).float()
            non_rigid_delta = _tensor_attr(pc, "binding_non_rigid_delta", point_count).reshape(-1).float()

            frame_rows.append({
                "image_name": str(view.image_name),
                "cam": int(view.cam_id),
                "frame": int(view.frame_id),
                "inner_missing_pixels": int(inner.sum().item()),
                "outer_leak_pixels": int(outer.sum().item()),
                "inner_band_pixels": int(inner_band.sum().item()),
                "outer_band_pixels": int(outer_band.sum().item()),
                "hard_residual_score": float((inner_band.sum() + outer_band.sum()).item() / max(float(gt_mask.sum().item()), 1.0)),
                "visible_points": int(visible.sum().item()),
                "outer_score_sum": float(outer_score.sum().item()),
                "inner_score_sum": float(inner_score.sum().item()),
                "visible_xfree_delta_mean": _weighted_mean(xfree_dist, visible.float()),
                "outer_xfree_delta_mean": _weighted_mean(xfree_dist, outer_score),
                "inner_xfree_delta_mean": _weighted_mean(xfree_dist, inner_score),
                "outer_fidelity_weight_mean": _weighted_mean(fidelity_weight, outer_score),
                "inner_fidelity_weight_mean": _weighted_mean(fidelity_weight, inner_score),
            })

            combined_score = torch.maximum(inner_score, outer_score)
            top_values, top_indices = torch.topk(combined_score, k=min(int(args.top_points), point_count))
            keep = top_values > 0.0
            for rank, pid in enumerate(top_indices[keep].detach().cpu().tolist(), start=1):
                pid = int(pid)
                rows.append({
                    "image_name": str(view.image_name),
                    "cam": int(view.cam_id),
                    "frame": int(view.frame_id),
                    "rank": rank,
                    "point_idx": pid,
                    "outer_score": float(outer_score[pid].item()),
                    "inner_score": float(inner_score[pid].item()),
                    "combined_score": float(combined_score[pid].item()),
                    "radii": float(radii[pid].item()),
                    "opacity": float(opacity[pid].item()),
                    "screen_x": float(xy[pid, 0].item()),
                    "screen_y": float(xy[pid, 1].item()),
                    "xfree_delta_px": float(xfree_dist[pid].item()) if torch.isfinite(xfree_dist[pid]) else 0.0,
                    "xfree_delta_x": float(free_delta_x[pid].item()),
                    "xfree_delta_y": float(free_delta_y[pid].item()),
                    "xfree_delta_norm_raw": float(free_delta_norm[pid].item()),
                    "xbar_base_delta_px": float(xbar_base_dist[pid].item()) if torch.isfinite(xbar_base_dist[pid]) else 0.0,
                    "pre_geometry_delta_px": float(pre_geom_dist[pid].item()) if torch.isfinite(pre_geom_dist[pid]) else 0.0,
                    "xrigid_delta_px": float(xrigid_dist[pid].item()) if torch.isfinite(xrigid_dist[pid]) else 0.0,
                    "xsoft_delta_px": float(xsoft_dist[pid].item()) if torch.isfinite(xsoft_dist[pid]) else 0.0,
                    "geometry_target_delta_px": float(geometry_target_dist[pid].item()) if torch.isfinite(geometry_target_dist[pid]) else 0.0,
                    "geometry_target_delta_x": float(target_delta_x[pid].item()),
                    "geometry_target_delta_y": float(target_delta_y[pid].item()),
                    "layer_id": int(layer_ids[pid].item()),
                    "region_id": int(region_ids[pid].item()),
                    "dominant_joint": int(dominant_joint[pid].item()),
                    "boundary_score": float(boundary_score[pid].item()),
                    "surface_distance": float(surface_distance[pid].item()),
                    "thin_score": float(thin_score[pid].item()),
                    "fidelity_weight": float(fidelity_weight[pid].item()),
                    "center_blend": float(center_blend[pid].item()),
                    "non_rigid_delta": float(non_rigid_delta[pid].item()),
                })
            if args.save_renders:
                torchvision.utils.save_image(render_rgb, str(args.out_dir / "renders" / f"render_{view.image_name}.png"))
            del render_pkg

    metrics = [
        "xfree_delta_px",
        "xbar_base_delta_px",
        "pre_geometry_delta_px",
        "xrigid_delta_px",
        "xsoft_delta_px",
        "geometry_target_delta_px",
        "boundary_score",
        "surface_distance",
        "thin_score",
        "fidelity_weight",
        "center_blend",
        "non_rigid_delta",
    ]
    outer_rows = [row for row in rows if float(row["outer_score"]) > 0.0]
    inner_rows = [row for row in rows if float(row["inner_score"]) > 0.0]
    summary = {
        "variant": args.variant,
        "load_ckpt": str(args.load_ckpt),
        "loaded_iteration": int(loaded_iteration),
        "n_frames": len(frame_rows),
        "n_point_rows": len(rows),
        "frame_mean_inner_missing_pixels": float(np.mean([r["inner_missing_pixels"] for r in frame_rows])) if frame_rows else 0.0,
        "frame_mean_outer_leak_pixels": float(np.mean([r["outer_leak_pixels"] for r in frame_rows])) if frame_rows else 0.0,
        "frame_mean_hard_residual_score": float(np.mean([r["hard_residual_score"] for r in frame_rows])) if frame_rows else 0.0,
        "outer_summary": _summarize_direction(outer_rows, "outer", "outer_score", metrics),
        "inner_summary": _summarize_direction(inner_rows, "inner", "inner_score", metrics),
        "top_outer": sorted(outer_rows, key=lambda row: float(row["outer_score"]), reverse=True)[:20],
        "top_inner": sorted(inner_rows, key=lambda row: float(row["inner_score"]), reverse=True)[:20],
    }
    (args.out_dir / "geometry_mismatch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out_dir / "frame_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        if frame_rows:
            writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0].keys()))
            writer.writeheader()
            writer.writerows(frame_rows)
    with (args.out_dir / "point_geometry_mismatch.csv").open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
