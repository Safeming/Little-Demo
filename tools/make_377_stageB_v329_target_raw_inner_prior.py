#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.graphics_utils import geom_transform_points


def _parse_ints(value: str) -> list[int]:
    out: list[int] = []
    for token in str(value or "").replace("[", "").replace("]", "").split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return out


def _render_support_mask(render_rgb: torch.Tensor, threshold: float, close_kernel: int) -> torch.Tensor:
    luma = render_rgb[0] * 0.299 + render_rgb[1] * 0.587 + render_rgb[2] * 0.114
    chroma = render_rgb.max(dim=0).values - render_rgb.min(dim=0).values
    support = ((luma > float(threshold)) | (chroma > float(threshold) * 0.75)).float().unsqueeze(0)
    close_kernel = int(close_kernel)
    if close_kernel > 1:
        if close_kernel % 2 == 0:
            close_kernel += 1
        pad = close_kernel // 2
        support = F.max_pool2d(support.unsqueeze(0), close_kernel, stride=1, padding=pad)[0]
        support = 1.0 - F.max_pool2d((1.0 - support).unsqueeze(0), close_kernel, stride=1, padding=pad)[0]
    return support.clamp(0.0, 1.0)


def _view_mask(view) -> torch.Tensor:
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


def _dilate(mask: torch.Tensor, width: int) -> torch.Tensor:
    width = max(0, int(width))
    if width <= 0:
        return mask
    kernel = 2 * width + 1
    return F.max_pool2d(mask.unsqueeze(0), kernel, stride=1, padding=width)[0]


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


def _sample_binary(mask: torch.Tensor, xy: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    sampled = torch.zeros((xy.shape[0],), dtype=torch.bool, device=xy.device)
    if xy.numel() == 0 or not bool(valid.any().item()):
        return sampled
    height, width = mask.shape[-2:]
    x = xy[:, 0].round().long().clamp(0, width - 1)
    y = xy[:, 1].round().long().clamp(0, height - 1)
    sampled[valid] = mask.reshape(height, width)[y[valid], x[valid]].bool()
    return sampled


def _attr(pc, name: str, default: float = 0.0) -> torch.Tensor:
    value = getattr(pc, name, None)
    point_count = int(pc.get_xyz.shape[0])
    if torch.is_tensor(value) and value.shape[0] == point_count:
        return value.detach().reshape(point_count).float().to(device=pc.get_xyz.device)
    return torch.full((point_count,), float(default), dtype=torch.float32, device=pc.get_xyz.device)


def _build_config(args: argparse.Namespace):
    config = OmegaConf.load(args.config_path)
    OmegaConf.set_struct(config, False)
    config.mode = "test"
    config.exp_dir = str(args.out_dir / "prior_scene")
    config.load_ckpt = str(args.load_ckpt)
    config.dataset.root_dir = str(args.dataset_root)
    config.dataset.preload = False
    config.dataset.train_views = _parse_ints(args.train_views)
    config.dataset.train_frames = _parse_ints(args.train_frames)
    config.dataset.test_views.view = _parse_ints(args.target_views)
    config.dataset.test_frames.view = _parse_ints(args.target_frames)
    config.dataset.parsing_prior.enable = False
    config.dataset.parsing_prior.roi_enable = False
    config.wandb_disable = True
    config.render_export_refine = False
    if "resume" not in config:
        config.resume = {}
    config.resume.allow_partial_converter_load = True
    config.resume.restore_gaussian_optimizer_state = False
    config.resume.restore_converter_optimizer_state = False
    config.resume.restore_converter_scheduler_state = False
    config.pipeline.compute_cov3D_python = True
    config.pipeline.covariance_mode = "default"
    config.pipeline.covariance_signed_dynamic_enable = False
    config.pipeline.covariance_signed_point_screen_actuator_enable = False
    config.pipeline.covariance_signed_center_offset_enable = False
    config.model.deformer.rigid.rotation_orthogonalize_enable = False
    config.model.deformer.rigid.geometry_fidelity_gate_enable = False
    return config


def _components(binary: np.ndarray, min_area: int, max_components: int) -> list[dict]:
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    rows = []
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        rows.append({
            "label": idx,
            "area": area,
            "cx": float(centroids[idx][0]),
            "cy": float(centroids[idx][1]),
            "mask": labels == idx,
        })
    rows.sort(key=lambda item: item["area"], reverse=True)
    if int(max_components) > 0:
        rows = rows[: int(max_components)]
    return rows


def _image_name(view) -> str:
    name = getattr(view, "image_name", "")
    if isinstance(name, (list, tuple)):
        name = name[0] if name else ""
    return str(name)


def _select_inner_points(view, render_pkg: dict, args: argparse.Namespace) -> tuple[list[int], list[float], dict]:
    pc = render_pkg["deformed_gaussian"]
    render_rgb = render_pkg["render"].detach().float().clamp(0.0, 1.0)
    gt_mask = _view_mask(view)
    support = _render_support_mask(render_rgb, args.render_support_threshold, args.close_kernel)
    inner = (gt_mask > 0.5) & (support <= 0.5)
    inner_np = inner[0].detach().cpu().numpy().astype(bool)
    comps = _components(inner_np, args.min_component_area, args.max_components)
    xy, proj_valid = _project_points(pc.get_xyz, view)
    radii = render_pkg["radii"].detach().float().to(device=pc.get_xyz.device).reshape(-1)
    visible = proj_valid & render_pkg["visibility_filter"].detach().bool().to(device=pc.get_xyz.device) & (radii > float(args.min_radius_px))
    boundary = _attr(pc, "binding_boundary_score", 0.0)
    thin = _attr(pc, "binding_thin_score", 0.0)
    surface = _attr(pc, "binding_surface_distance", 0.0)
    opacity = pc.get_opacity.detach().reshape(-1).float()
    if float(args.min_boundary) >= 0.0:
        visible &= boundary >= float(args.min_boundary)
    if float(args.min_thin) >= 0.0:
        visible &= thin >= float(args.min_thin)
    if args.surface_max is not None:
        visible &= surface <= float(args.surface_max)
    if args.surface_min is not None:
        visible &= surface >= float(args.surface_min)

    point_scores: dict[int, float] = {}
    per_component = []
    for comp in comps:
        comp_mask = torch.from_numpy(comp["mask"].astype(np.float32)).to(device=pc.get_xyz.device).reshape(1, *inner.shape[-2:])
        comp_pad = _dilate(comp_mask, int(args.point_pad_px)) > 0.5
        comp_core = comp_mask > 0.5
        inside_pad = _sample_binary(comp_pad, xy, visible)
        if not bool(inside_pad.any().item()):
            per_component.append({"area": comp["area"], "selected": 0})
            continue
        inside_core = _sample_binary(comp_core, xy, visible)
        dx = xy[:, 0] - float(comp["cx"])
        dy = xy[:, 1] - float(comp["cy"])
        dist = torch.sqrt(dx * dx + dy * dy)
        spatial = torch.exp(-dist / max(float(args.spatial_decay_px), 1.0))
        radius_norm = radii / radii[inside_pad].detach().quantile(0.90).clamp_min(1.0)
        score = (
            spatial
            + 1.25 * inside_core.float()
            + 0.35 * boundary.clamp(0.0, 1.0)
            + 0.20 * thin.clamp(0.0, 1.0)
            + 0.15 * opacity.clamp(0.0, 1.0)
            + 0.10 * radius_norm.clamp(0.0, 2.0)
            + 0.002 * min(float(comp["area"]), 512.0)
        )
        score = torch.where(inside_pad, score, torch.full_like(score, -float("inf")))
        k = min(int(args.points_per_component), int(inside_pad.sum().item()))
        values, ids = torch.topk(score, k=max(k, 0))
        selected = 0
        for value, point_id in zip(values.detach().cpu().tolist(), ids.detach().cpu().tolist()):
            if not math.isfinite(float(value)):
                continue
            point_scores[int(point_id)] = max(float(value), point_scores.get(int(point_id), -float("inf")))
            selected += 1
        per_component.append({"area": comp["area"], "selected": selected})

    ranked = sorted(point_scores.items(), key=lambda pair: pair[1], reverse=True)[: max(int(args.max_grow), 0)]
    grow_ids = [int(pid) for pid, _ in ranked]
    grow_scores = [float(score) for _, score in ranked]
    stats = {
        "inner_pixels": int(inner_np.sum()),
        "component_count": len(comps),
        "grow_count": len(grow_ids),
        "per_component": per_component,
    }
    return grow_ids, grow_scores, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v329 target-raw self-localized inner grow point prior.")
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--load-ckpt", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--train-views", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--target-views", default="21,22,23")
    parser.add_argument("--train-frames", default="0,570,60")
    parser.add_argument("--target-frames", default="0,570,60")
    parser.add_argument("--render-support-threshold", type=float, default=0.025)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--min-component-area", type=int, default=18)
    parser.add_argument("--max-components", type=int, default=8)
    parser.add_argument("--point-pad-px", type=int, default=10)
    parser.add_argument("--points-per-component", type=int, default=12)
    parser.add_argument("--max-grow", type=int, default=96)
    parser.add_argument("--spatial-decay-px", type=float, default=18.0)
    parser.add_argument("--min-radius-px", type=float, default=0.0)
    parser.add_argument("--min-boundary", type=float, default=0.08)
    parser.add_argument("--min-thin", type=float, default=-1.0)
    parser.add_argument("--surface-min", type=float, default=None)
    parser.add_argument("--surface-max", type=float, default=0.10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = _build_config(args)
    gaussians = GaussianModel(config.model.gaussian)
    scene = Scene(config, gaussians, str(args.out_dir / "prior_scene"))
    scene.eval()
    loaded_iteration = scene.load_checkpoint(str(args.load_ckpt))
    bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    by_image: dict[str, dict] = {}
    target_records = []
    with torch.no_grad():
        for idx in range(len(scene.test_dataset)):
            view = scene.test_dataset[idx]
            render_pkg = render(view, loaded_iteration, scene, config.pipeline, background, compute_loss=False)
            image = _image_name(view)
            grow_ids, grow_scores, stats = _select_inner_points(view, render_pkg, args)
            by_image[image] = {
                "grow_point_ids": grow_ids,
                "grow_scores": grow_scores,
                "shrink_point_ids": [],
                "source": "target_raw_inner_missing",
                **stats,
            }
            target_records.append({"image_name": image, **stats})
            del render_pkg

    payload = {
        "selection": {
            "render_support_threshold": float(args.render_support_threshold),
            "close_kernel": int(args.close_kernel),
            "min_component_area": int(args.min_component_area),
            "max_components": int(args.max_components),
            "point_pad_px": int(args.point_pad_px),
            "points_per_component": int(args.points_per_component),
            "max_grow": int(args.max_grow),
            "spatial_decay_px": float(args.spatial_decay_px),
            "min_radius_px": float(args.min_radius_px),
            "min_boundary": float(args.min_boundary),
            "min_thin": float(args.min_thin),
            "surface_min": args.surface_min,
            "surface_max": args.surface_max,
        },
        "by_image": by_image,
        "target_records": target_records,
        "shrink_point_ids": [],
        "grow_point_ids": [],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_json": str(args.out_json),
        "images": len(target_records),
        "mean_inner_pixels": sum(r["inner_pixels"] for r in target_records) / max(len(target_records), 1),
        "mean_grow": sum(r["grow_count"] for r in target_records) / max(len(target_records), 1),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
