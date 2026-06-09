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

import train as train_mod
from gaussian_renderer import render, rasterize_gaussians
from scene import GaussianModel, Scene
from utils.graphics_utils import geom_transform_points


def _parse_csv_ints(value: str) -> list[int]:
    out = []
    for part in str(value).replace("[", "").replace("]", "").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _range_values(spec: str) -> list[int]:
    vals = _parse_csv_ints(spec)
    if len(vals) == 3:
        return list(range(vals[0], vals[1], vals[2]))
    return vals


def _set_dataset_subset(config, split: str, views: list[int], frames: list[int]) -> None:
    if split == "train":
        config.dataset.train_views = views
        config.dataset.train_frames = frames
    elif split == "val":
        config.dataset.val_views = views
        config.dataset.val_frames = frames
    elif split == "test":
        config.dataset.test_views.view = views
        config.dataset.test_frames.view = frames
    else:
        raise ValueError(f"unsupported split: {split}")


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


def _boundary_band(mask: torch.Tensor, width: int) -> torch.Tensor:
    binary = (mask > 0.5).float()
    return (_dilate(binary, width) - _erode(binary, width)).clamp(0.0, 1.0)


def _render_support_mask(render_rgb: torch.Tensor, threshold: float, close_kernel: int) -> torch.Tensor:
    luma = render_rgb[0] * 0.299 + render_rgb[1] * 0.587 + render_rgb[2] * 0.114
    chroma = render_rgb.max(dim=0).values - render_rgb.min(dim=0).values
    support = ((luma > threshold) | (chroma > threshold * 0.75)).float().unsqueeze(0)
    close_kernel = int(close_kernel)
    if close_kernel > 1:
        if close_kernel % 2 == 0:
            close_kernel += 1
        pad = close_kernel // 2
        support = F.max_pool2d(support.unsqueeze(0), close_kernel, stride=1, padding=pad)[0]
        support = 1.0 - F.max_pool2d((1.0 - support).unsqueeze(0), close_kernel, stride=1, padding=pad)[0]
    return support.clamp(0.0, 1.0)


def _project_points(points: torch.Tensor, view) -> tuple[torch.Tensor, torch.Tensor]:
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
    return torch.stack((px, py), dim=-1), valid


def _sample_binary(mask: torch.Tensor, xy: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    sampled = torch.zeros((xy.shape[0],), dtype=torch.float32, device=xy.device)
    if xy.numel() == 0 or not bool(valid.any().item()):
        return sampled
    height, width = mask.shape[-2:]
    x = xy[:, 0].round().long().clamp(0, width - 1)
    y = xy[:, 1].round().long().clamp(0, height - 1)
    sampled[valid] = mask.reshape(height, width)[y[valid], x[valid]].float()
    return sampled


def _distance_to_mask(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    if xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if mask.any():
        dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        x = np.clip(np.rint(xy[:, 0]).astype(np.int32), 0, dist.shape[1] - 1)
        y = np.clip(np.rint(xy[:, 1]).astype(np.int32), 0, dist.shape[0] - 1)
        return dist[y, x].astype(np.float32)
    return np.full((xy.shape[0],), np.inf, dtype=np.float32)


def _tensor_to_public(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def _merge_specs_for_validation(gaussians, specs: list[dict], iteration: int):
    merged = gaussians._merge_boundary_component_support_specs(specs)
    if not isinstance(merged, dict):
        return None
    if bool(gaussians.cfg.get("boundary_component_support_candidate_consensus_enable", False)):
        merged = gaussians._merge_boundary_component_support_specs_by_candidate_consensus(
            merged,
            iteration=iteration,
        )
    elif bool(gaussians.cfg.get("boundary_component_support_parent_consensus_enable", False)):
        merged = gaussians._merge_boundary_component_support_specs_by_parent_consensus(
            merged,
            iteration=iteration,
        )
    if not isinstance(merged, dict):
        return None
    return {key: _tensor_to_public(value) for key, value in merged.items()}


def _make_candidate_overlay(record_paths: list[dict], out_path: Path, topk: int) -> None:
    if not record_paths:
        return
    chosen = sorted(record_paths, key=lambda item: item["inner_missing_pixels"], reverse=True)[:topk]
    panels = []
    for item in chosen:
        render_img = torchvision.io.read_image(item["render_path"]).float() / 255.0
        support_img = torchvision.io.read_image(item["candidate_map_path"]).float() / 255.0
        inner = torch.from_numpy(np.load(item["inner_mask_path"])).bool()
        outer = torch.from_numpy(np.load(item["outer_mask_path"])).bool()
        overlay = render_img.clone()
        overlay[:, inner] = torch.tensor([0.15, 0.95, 0.25]).reshape(3, 1)
        overlay[:, outer] = torch.tensor([1.0, 0.1, 0.12]).reshape(3, 1)
        overlay[:, support_img.max(dim=0).values > 0.01] = torch.tensor([1.0, 0.9, 0.05]).reshape(3, 1)
        panels.append(torch.cat([render_img, support_img, overlay], dim=2))
    grid = torch.cat(panels, dim=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(grid, str(out_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="v261 offline support candidate validator for CoreView_377.")
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--load-ckpt", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--parser-root", default="")
    parser.add_argument("--compact-mapping", default="")
    parser.add_argument("--candidate-views", default="1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--candidate-frames", default="0,570,60")
    parser.add_argument("--eval-views", default="21,22,23")
    parser.add_argument("--eval-frames", default="0,570,60")
    parser.add_argument("--iteration", type=int, default=135711)
    parser.add_argument("--render-support-threshold", type=float, default=0.025)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--band-width", type=int, default=7)
    parser.add_argument("--search-band-width", type=int, default=24)
    parser.add_argument("--near-radius", type=int, default=4)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--max-candidate-points", type=int, default=72)
    parser.add_argument("--min-accepted-candidates", type=int, default=2)
    parser.add_argument("--min-inner-outer-ratio", type=float, default=1.35)
    parser.add_argument("--min-inner-hit-pixels", type=float, default=12.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "candidate_maps"
    mask_dir = args.out_dir / "masks"
    image_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    config = OmegaConf.load(args.config_path)
    OmegaConf.set_struct(config, False)
    config.mode = "test"
    config.exp_dir = str(args.out_dir / "scene")
    config.load_ckpt = str(args.load_ckpt)
    config.dataset.root_dir = str(args.dataset_root)
    config.dataset.preload = False
    if args.parser_root:
        config.dataset.parsing_prior.parser_root = str(args.parser_root)
    if args.compact_mapping:
        config.dataset.parsing_prior.compact_mapping_file = str(args.compact_mapping)
    if "resume" not in config:
        config.resume = {}
    config.resume.allow_partial_converter_load = False
    config.resume.restore_gaussian_optimizer_state = False
    config.resume.restore_converter_optimizer_state = False
    config.resume.restore_converter_scheduler_state = False
    config.resume.disable_densify_on_resume = True
    config.resume.disable_opacity_reset_on_resume = True

    opt = config.opt
    opt.boundary_image_error_score_enable = True
    opt.boundary_image_error_score_signed_enable = True
    opt.boundary_image_error_score_min = float(opt.get("boundary_image_error_score_min", 0.008))
    opt.boundary_image_error_pred_threshold = float(opt.get("boundary_image_error_pred_threshold", 0.32))
    opt.boundary_image_error_target_threshold = float(opt.get("boundary_image_error_target_threshold", 0.50))
    opt.boundary_image_error_score_band_width = int(opt.get("boundary_image_error_score_band_width", 8))
    opt.boundary_image_error_score_focus_dilate = int(opt.get("boundary_image_error_score_focus_dilate", 4))
    opt.boundary_component_support_enable = True
    opt.boundary_component_support_interval = 1
    opt.boundary_component_support_verbose = False
    opt.boundary_component_support_residual_threshold = float(opt.get("boundary_component_support_residual_threshold", 0.50))
    opt.boundary_component_support_min_area = int(opt.get("boundary_component_support_min_area", 18))
    opt.boundary_component_support_max_components = int(opt.get("boundary_component_support_max_components", 12))
    opt.boundary_component_support_points_per_component = int(opt.get("boundary_component_support_points_per_component", 2))
    opt.boundary_component_support_max_points_per_view = int(opt.get("boundary_component_support_max_points_per_view", 24))
    opt.boundary_component_support_target_project_enable = True
    opt.boundary_component_support_target_project_offset_min = float(opt.get("boundary_component_support_target_project_offset_min", 1.0e-6))
    opt.boundary_component_support_target_project_offset_max = float(opt.get("boundary_component_support_target_project_offset_max", 0.080))
    model_gaussian = config.model.gaussian
    model_gaussian.boundary_component_support_enable = True
    model_gaussian.boundary_component_support_use_target_offsets = True
    model_gaussian.boundary_component_support_target_offset_max = float(model_gaussian.get("boundary_component_support_target_offset_max", 0.080))
    model_gaussian.boundary_component_support_candidate_consensus_enable = True
    model_gaussian.boundary_component_support_candidate_consensus_strict = True
    model_gaussian.boundary_component_support_candidate_consensus_min_votes = int(model_gaussian.get("boundary_component_support_candidate_consensus_min_votes", 2))
    model_gaussian.boundary_component_support_candidate_consensus_min_unique_views = int(model_gaussian.get("boundary_component_support_candidate_consensus_min_unique_views", 2))
    model_gaussian.boundary_component_support_candidate_consensus_cluster_radius = float(model_gaussian.get("boundary_component_support_candidate_consensus_cluster_radius", 0.018))
    model_gaussian.boundary_component_support_candidate_consensus_max_xyz_std = float(model_gaussian.get("boundary_component_support_candidate_consensus_max_xyz_std", 0.020))
    model_gaussian.boundary_component_support_candidate_consensus_target_under_min = float(model_gaussian.get("boundary_component_support_candidate_consensus_target_under_min", 0.55))
    model_gaussian.boundary_component_support_candidate_consensus_target_over_max = float(model_gaussian.get("boundary_component_support_candidate_consensus_target_over_max", 0.35))
    model_gaussian.boundary_component_support_candidate_consensus_anchor_over_max = float(model_gaussian.get("boundary_component_support_candidate_consensus_anchor_over_max", 0.30))
    model_gaussian.boundary_component_support_candidate_consensus_max_points = int(args.max_candidate_points)
    model_gaussian.boundary_component_support_max_points = int(args.max_candidate_points)

    background = torch.tensor(
        [1, 1, 1] if bool(config.dataset.white_background) else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    candidate_views = _parse_csv_ints(args.candidate_views)
    candidate_frames = _range_values(args.candidate_frames)
    eval_views = _parse_csv_ints(args.eval_views)
    eval_frames = _range_values(args.eval_frames)

    _set_dataset_subset(config, "test", candidate_views, [candidate_frames[0], candidate_frames[-1] + 1, max(1, candidate_frames[1] - candidate_frames[0]) if len(candidate_frames) > 1 else 1])
    gaussians = GaussianModel(config.model.gaussian)
    scene = Scene(config, gaussians, config.exp_dir)
    scene.eval()
    loaded_iteration = scene.load_checkpoint(str(args.load_ckpt))

    candidate_specs = []
    candidate_records = []
    for view in scene.test_dataset:
        render_pkg = render(
            view,
            int(loaded_iteration),
            scene,
            config.pipeline,
            background,
            compute_loss=False,
            return_opacity=True,
        )
        payload, valid = train_mod._build_boundary_image_error_point_score(
            render_pkg["deformed_gaussian"],
            view,
            render_pkg.get("opacity_render"),
            _mask_from_view(view),
            render_pkg["visibility_filter"],
            render_pkg["radii"],
            config,
            iteration=int(args.iteration),
        )
        specs = payload.get("component_support") if isinstance(payload, dict) else None
        if isinstance(specs, dict):
            candidate_specs.append(specs)
            candidate_records.append({
                "image_name": view.image_name,
                "spec_count": int(specs["parent_idx"].reshape(-1).numel()),
            })

    merged_specs = _merge_specs_for_validation(scene.gaussians, candidate_specs, int(args.iteration))
    if not isinstance(merged_specs, dict):
        summary = {
            "status": "blocked",
            "reason": "no_consensus_candidates",
            "candidate_birth_samples": candidate_records,
            "accepted_candidate_count": 0,
        }
        (args.out_dir / "candidate_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    _set_dataset_subset(config, "test", eval_views, [eval_frames[0], eval_frames[-1] + 1, max(1, eval_frames[1] - eval_frames[0]) if len(eval_frames) > 1 else 1])
    eval_gaussians = GaussianModel(config.model.gaussian)
    eval_scene = Scene(config, eval_gaussians, config.exp_dir)
    eval_scene.eval()
    eval_scene.load_checkpoint(str(args.load_ckpt))
    base_count = int(eval_scene.gaussians.get_xyz.shape[0])
    support_count = eval_scene.gaussians._append_boundary_component_support_points(
        merged_specs.get("parent_idx"),
        merged_specs.get("directions"),
        scores=merged_specs.get("scores"),
        component_ids=merged_specs.get("component_ids"),
        source_pixels=merged_specs.get("source_pixels"),
        target_pixels=merged_specs.get("target_pixels"),
        screen_gaps=merged_specs.get("screen_gaps"),
        parent_radii=merged_specs.get("parent_radii"),
        anchor_over_scores=merged_specs.get("anchor_over_scores"),
        target_under_scores=merged_specs.get("target_under_scores"),
        target_over_scores=merged_specs.get("target_over_scores"),
        target_net_gains=merged_specs.get("target_net_gains"),
        target_offset_vectors=merged_specs.get("target_offset_vectors"),
        target_offset_valid=merged_specs.get("target_offset_valid"),
        view_ids=merged_specs.get("view_ids"),
        frame_ids=merged_specs.get("frame_ids"),
        parent_consensus_votes=merged_specs.get("parent_consensus_votes"),
        parent_consensus_unique_views=merged_specs.get("parent_consensus_unique_views"),
        parent_consensus_unique_frames=merged_specs.get("parent_consensus_unique_frames"),
        parent_consensus_offset_std=merged_specs.get("parent_consensus_offset_std"),
        iteration=int(args.iteration),
    )
    accepted_count = int(support_count)
    new_indices = torch.arange(base_count, base_count + accepted_count, dtype=torch.long, device="cuda")

    records = []
    path_records = []
    per_candidate = defaultdict(lambda: {"active": 0, "inner": 0.0, "outer": 0.0, "inner_band": 0.0})
    for view in eval_scene.test_dataset:
        render_pkg = render(
            view,
            int(loaded_iteration),
            eval_scene,
            config.pipeline,
            background,
            compute_loss=False,
            return_opacity=True,
        )
        rendering = render_pkg["render"].detach().clamp(0.0, 1.0)
        deformed = render_pkg["deformed_gaussian"]
        gt_mask = _mask_from_view(view)
        support = _render_support_mask(rendering, args.render_support_threshold, args.close_kernel)
        near_gt = _dilate(gt_mask, args.search_band_width)
        inner_missing = (gt_mask > 0.5) & (support <= 0.5)
        outer_leak = (support > 0.5) & (gt_mask <= 0.5) & (near_gt > 0.5)
        inner_band = inner_missing & (_boundary_band(gt_mask, args.band_width) > 0.5)

        if accepted_count > 0:
            xy, proj_valid = _project_points(deformed.get_xyz[new_indices], view)
            visible = render_pkg["visibility_filter"].detach().bool()[new_indices]
            radii = render_pkg["radii"].detach()[new_indices]
            active = proj_valid & visible & (radii > 0)
        else:
            xy = torch.zeros((0, 2), dtype=torch.float32, device="cuda")
            proj_valid = torch.zeros((0,), dtype=torch.bool, device="cuda")
            active = proj_valid
        inner_sample = _sample_binary(inner_missing.float()[0], xy, active)
        outer_sample = _sample_binary(outer_leak.float()[0], xy, active)
        inner_band_sample = _sample_binary(inner_band.float()[0], xy, active)
        xy_np = xy.detach().cpu().numpy() if xy.numel() else np.zeros((0, 2), dtype=np.float32)
        inner_np = inner_missing[0].detach().cpu().numpy().astype(bool)
        outer_np = outer_leak[0].detach().cpu().numpy().astype(bool)
        inner_dist = _distance_to_mask(inner_np, xy_np)
        outer_dist = _distance_to_mask(outer_np, xy_np)

        for idx in range(accepted_count):
            if bool(active[idx].item()):
                per_candidate[idx]["active"] += 1
                per_candidate[idx]["inner"] += float(inner_sample[idx].item())
                per_candidate[idx]["outer"] += float(outer_sample[idx].item())
                per_candidate[idx]["inner_band"] += float(inner_band_sample[idx].item())

        candidate_map = torch.zeros_like(rendering)
        if accepted_count > 0:
            colors = torch.zeros((deformed.get_xyz.shape[0], 3), dtype=torch.float32, device="cuda")
            colors[new_indices] = torch.tensor([1.0, 0.82, 0.04], device="cuda")
            support_pkg = rasterize_gaussians(view, deformed, config.pipeline, background, colors_precomp=colors, return_opacity=False)
            candidate_map = support_pkg["render"].detach().clamp(0.0, 1.0)

        render_name = f"render_{view.image_name}.png"
        render_path = args.out_dir / "renders" / render_name
        candidate_path = image_dir / render_name
        render_path.parent.mkdir(exist_ok=True)
        torchvision.utils.save_image(rendering, str(render_path))
        torchvision.utils.save_image(candidate_map.clamp(0.0, 1.0), str(candidate_path))
        inner_path = mask_dir / f"{view.image_name}_inner.npy"
        outer_path = mask_dir / f"{view.image_name}_outer.npy"
        np.save(inner_path, inner_np)
        np.save(outer_path, outer_np)

        active_np = active.detach().cpu().numpy().astype(bool)
        record = {
            "image_name": view.image_name,
            "candidate_count": accepted_count,
            "projected_valid": int(proj_valid.sum().item()),
            "active_count": int(active.sum().item()),
            "inner_missing_pixels": int(inner_missing.sum().item()),
            "outer_leak_pixels": int(outer_leak.sum().item()),
            "inner_band_pixels": int(inner_band.sum().item()),
            "projected_candidates_on_inner_points": int((inner_sample > 0.5).sum().item()),
            "projected_candidates_on_inner_band_points": int((inner_band_sample > 0.5).sum().item()),
            "projected_candidates_on_outer_points": int((outer_sample > 0.5).sum().item()),
            "active_min_dist_to_inner": float(np.min(inner_dist[active_np])) if active_np.any() else math.inf,
            "active_mean_dist_to_inner": float(np.mean(inner_dist[active_np])) if active_np.any() else math.inf,
            "active_min_dist_to_outer": float(np.min(outer_dist[active_np])) if active_np.any() else math.inf,
            "active_mean_dist_to_outer": float(np.mean(outer_dist[active_np])) if active_np.any() else math.inf,
        }
        records.append(record)
        path_records.append({
            **record,
            "render_path": str(render_path),
            "candidate_map_path": str(candidate_path),
            "inner_mask_path": str(inner_path),
            "outer_mask_path": str(outer_path),
        })

    per_candidate_rows = []
    for idx in range(accepted_count):
        stats = per_candidate[idx]
        active_count = max(int(stats["active"]), 1)
        per_candidate_rows.append({
            "candidate_index": idx,
            "active_views": int(stats["active"]),
            "inner_hits": float(stats["inner"]),
            "inner_band_hits": float(stats["inner_band"]),
            "outer_hits": float(stats["outer"]),
            "inner_hit_rate": float(stats["inner"]) / active_count,
            "outer_hit_rate": float(stats["outer"]) / active_count,
        })
    valid_candidate_rows = [
        row for row in per_candidate_rows
        if row["inner_hits"] > 0 and row["inner_hits"] >= row["outer_hits"] * float(args.min_inner_outer_ratio)
    ]

    mean_inner_points = float(np.mean([r["projected_candidates_on_inner_points"] for r in records])) if records else 0.0
    mean_outer_points = float(np.mean([r["projected_candidates_on_outer_points"] for r in records])) if records else 0.0
    inner_outer_ratio = mean_inner_points / max(mean_outer_points, 1.0e-6)
    status = "ok"
    reasons = []
    if accepted_count < int(args.min_accepted_candidates):
        reasons.append("too_few_consensus_candidates")
    if len(valid_candidate_rows) < int(args.min_accepted_candidates):
        reasons.append("too_few_inner_dominant_candidates")
    if mean_inner_points < float(args.min_inner_hit_pixels):
        reasons.append("mean_inner_hits_too_low")
    if inner_outer_ratio < float(args.min_inner_outer_ratio):
        reasons.append("inner_outer_ratio_too_low")
    if reasons:
        status = "blocked"

    summary = {
        "status": status,
        "reasons": reasons,
        "config_path": str(args.config_path),
        "load_ckpt": str(args.load_ckpt),
        "loaded_iteration": int(loaded_iteration),
        "candidate_views": candidate_views,
        "candidate_frames": candidate_frames,
        "eval_views": eval_views,
        "eval_frames": eval_frames,
        "birth_spec_samples": candidate_records,
        "birth_spec_sample_count": len(candidate_records),
        "accepted_candidate_count": accepted_count,
        "inner_dominant_candidate_count": len(valid_candidate_rows),
        "mean_projected_candidates_on_inner_points": mean_inner_points,
        "mean_projected_candidates_on_outer_points": mean_outer_points,
        "projected_inner_outer_ratio": inner_outer_ratio,
        "mean_inner_missing_pixels": float(np.mean([r["inner_missing_pixels"] for r in records])) if records else 0.0,
        "mean_outer_leak_pixels": float(np.mean([r["outer_leak_pixels"] for r in records])) if records else 0.0,
        "records": records,
        "per_candidate": per_candidate_rows,
    }
    (args.out_dir / "candidate_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out_dir / "candidate_validation_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        if records:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    with (args.out_dir / "candidate_validation_per_candidate.csv").open("w", newline="", encoding="utf-8") as handle:
        if per_candidate_rows:
            writer = csv.DictWriter(handle, fieldnames=list(per_candidate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_candidate_rows)
    _make_candidate_overlay(path_records, args.out_dir / "top_candidate_projection_overlay.png", args.topk)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
