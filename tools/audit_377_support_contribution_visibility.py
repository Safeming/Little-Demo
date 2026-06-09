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

from gaussian_renderer import render, rasterize_gaussians
from scene import GaussianModel, Scene
from utils.graphics_utils import geom_transform_points


def _stat(values: torch.Tensor) -> dict:
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return {"n": 0}
    qs = torch.quantile(
        values,
        torch.tensor([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0], device=values.device),
    )
    return {
        "n": int(values.numel()),
        "mean": float(values.mean().item()),
        "min": float(qs[0].item()),
        "p05": float(qs[1].item()),
        "p25": float(qs[2].item()),
        "median": float(qs[3].item()),
        "p75": float(qs[4].item()),
        "p95": float(qs[5].item()),
        "max": float(qs[6].item()),
    }


def _project_points(points: torch.Tensor, view) -> tuple[torch.Tensor, torch.Tensor]:
    projected = geom_transform_points(points.detach(), view.full_proj_transform)
    px = (projected[:, 0] + 1.0) * 0.5 * float(max(int(view.image_width) - 1, 1))
    py = (1.0 - (projected[:, 1] + 1.0) * 0.5) * float(max(int(view.image_height) - 1, 1))
    valid = torch.isfinite(projected).all(dim=-1)
    valid &= projected[:, 2] > 0.0
    valid &= px >= 0.0
    valid &= px <= float(max(int(view.image_width) - 1, 0))
    valid &= py >= 0.0
    valid &= py <= float(max(int(view.image_height) - 1, 0))
    return torch.stack((px, py), dim=-1), valid


def _kernel_size(width: int) -> int:
    width = max(0, int(width))
    return width * 2 + 1


def _dilate(mask: torch.Tensor, width: int) -> torch.Tensor:
    kernel = _kernel_size(width)
    if kernel <= 1:
        return mask
    return F.max_pool2d(mask.unsqueeze(0), kernel, stride=1, padding=kernel // 2)[0]


def _erode(mask: torch.Tensor, width: int) -> torch.Tensor:
    kernel = _kernel_size(width)
    if kernel <= 1:
        return mask
    return 1.0 - _dilate(1.0 - mask, width)


def _render_support_mask(render_rgb: torch.Tensor, threshold: float, close_kernel: int) -> torch.Tensor:
    luma = render_rgb[0] * 0.299 + render_rgb[1] * 0.587 + render_rgb[2] * 0.114
    chroma = render_rgb.max(dim=0).values - render_rgb.min(dim=0).values
    support = ((luma > threshold) | (chroma > threshold * 0.75)).float().unsqueeze(0)
    close_kernel = int(close_kernel)
    if close_kernel > 1:
        pad = close_kernel // 2
        support = F.max_pool2d(support.unsqueeze(0), close_kernel, stride=1, padding=pad)[0]
        support = 1.0 - F.max_pool2d((1.0 - support).unsqueeze(0), close_kernel, stride=1, padding=pad)[0]
    return support.clamp(0.0, 1.0)


def _boundary_band(mask: torch.Tensor, width: int) -> torch.Tensor:
    binary = (mask > 0.5).float()
    return (_dilate(binary, width) - _erode(binary, width)).clamp(0.0, 1.0)


def _distance_to_mask(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    if mask.any():
        inv = (~mask).astype(np.uint8)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        x = np.clip(np.rint(xy[:, 0]).astype(np.int32), 0, dist.shape[1] - 1)
        y = np.clip(np.rint(xy[:, 1]).astype(np.int32), 0, dist.shape[0] - 1)
        return dist[y, x].astype(np.float32)
    return np.full((xy.shape[0],), np.inf, dtype=np.float32)


def _sample_mask(mask: torch.Tensor, xy: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    sampled = torch.zeros((xy.shape[0],), dtype=torch.float32, device=xy.device)
    if xy.numel() == 0:
        return sampled
    height, width = mask.shape[-2:]
    x = xy[:, 0].round().long().clamp(0, width - 1)
    y = xy[:, 1].round().long().clamp(0, height - 1)
    sampled[valid] = mask.reshape(height, width)[y[valid], x[valid]].float()
    return sampled


def _make_montage(records: list[dict], out_path: Path, topk: int = 12) -> None:
    if not records:
        return
    chosen = sorted(records, key=lambda item: item["inner_missing_pixels"], reverse=True)[:topk]
    tiles = []
    for record in chosen:
        render_img = torchvision.io.read_image(record["render_path"]).float() / 255.0
        new_img = torchvision.io.read_image(record["new_support_path"]).float() / 255.0
        overlay = render_img.clone()
        inner = torch.from_numpy(np.load(record["inner_mask_path"])).bool()
        outer = torch.from_numpy(np.load(record["outer_mask_path"])).bool()
        new_mask = (new_img.max(dim=0).values > 0.02)
        overlay[:, inner] = torch.tensor([0.15, 0.95, 0.25]).reshape(3, 1)
        overlay[:, outer] = torch.tensor([1.0, 0.1, 0.12]).reshape(3, 1)
        overlay[:, new_mask] = torch.tensor([1.0, 0.9, 0.05]).reshape(3, 1)
        panel = torch.cat([render_img, new_img, overlay], dim=2)
        tiles.append(panel)
    grid = torch.cat(tiles, dim=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(grid, str(out_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit newly spawned boundary support visibility and residual contribution.")
    parser.add_argument("--render-exp", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--birth-min", type=int, default=145000)
    parser.add_argument("--render-support-threshold", type=float, default=0.025)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--band-width", type=int, default=7)
    parser.add_argument("--search-band-width", type=int, default=24)
    parser.add_argument("--near-radius", type=int, default=4)
    parser.add_argument("--topk", type=int, default=12)
    args = parser.parse_args()

    config_path = args.render_exp / ".hydra" / "config.yaml"
    config = OmegaConf.load(config_path)
    config.mode = "test"
    config.dataset.preload = False

    out_dir = args.out_dir or (args.render_exp / "diagnostics" / "test-view_new_support_visibility")
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "masks"
    image_dir = out_dir / "new_support_maps"
    mask_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    gaussians = GaussianModel(config.model.gaussian)
    scene = Scene(config, gaussians, config.exp_dir)
    scene.eval()
    loaded_iteration = scene.load_checkpoint(config.load_ckpt)
    render_iteration = int(loaded_iteration)

    background = torch.tensor([1, 1, 1] if config.dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    binding_state = scene.gaussians.get_binding_state()
    role = binding_state.get("boundary_support_role", None)
    birth = binding_state.get("boundary_support_birth_iter", None)
    if not torch.is_tensor(role) or not torch.is_tensor(birth):
        raise RuntimeError("checkpoint has no boundary_support_role/birth_iter state")
    role = role.to(device="cuda").reshape(-1)
    birth = birth.to(device="cuda").reshape(-1)
    new_support_mask = (role > 0) & (birth >= int(args.birth_min))
    all_support_mask = role > 0
    new_indices = torch.nonzero(new_support_mask, as_tuple=False).flatten()
    all_support_indices = torch.nonzero(all_support_mask, as_tuple=False).flatten()

    opacity_all = scene.gaussians.get_opacity.detach().reshape(-1)
    scale_all = scene.gaussians.get_scaling.detach().amax(dim=-1)

    records = []
    for idx in range(len(scene.test_dataset)):
        view = scene.test_dataset[idx]
        render_pkg = render(
            view,
            render_iteration,
            scene,
            config.pipeline,
            background,
            compute_loss=False,
            return_opacity=True,
        )
        deformed = render_pkg["deformed_gaussian"]
        rendering = render_pkg["render"].detach().clamp(0.0, 1.0)
        opacity_render = render_pkg["opacity_render"].detach().clamp(0.0, 1.0)
        radii = render_pkg["radii"].detach()
        visible = render_pkg["visibility_filter"].detach().bool()

        gt_mask = (view.hard_mask if hasattr(view, "hard_mask") else view.original_mask).detach().float().cuda()
        if gt_mask.dim() == 2:
            gt_mask = gt_mask.unsqueeze(0)
        gt_mask = (gt_mask[:1] > 0.5).float()
        render_support = _render_support_mask(rendering, args.render_support_threshold, args.close_kernel)
        near_gt = _dilate(gt_mask, args.search_band_width)
        inner_missing = (gt_mask > 0.5) & (render_support <= 0.5)
        outer_leak = (render_support > 0.5) & (gt_mask <= 0.5) & (near_gt > 0.5)
        inner_band = inner_missing & (_boundary_band(gt_mask, args.band_width) > 0.5)

        colors = torch.zeros((deformed.get_xyz.shape[0], 3), dtype=torch.float32, device="cuda")
        colors[new_support_mask] = torch.tensor([1.0, 0.82, 0.04], device="cuda")
        support_pkg = rasterize_gaussians(view, deformed, config.pipeline, background, colors_precomp=colors, return_opacity=False)
        new_support_image = support_pkg["render"].detach().clamp(0.0, 1.0)
        new_support_value = new_support_image.max(dim=0).values
        new_support_pixels = new_support_value > 0.01
        new_support_near = _dilate(new_support_pixels.float().unsqueeze(0), args.near_radius)[0] > 0.5

        xy, proj_valid = _project_points(deformed.get_xyz[new_indices], view)
        new_visible = visible[new_indices] if new_indices.numel() > 0 else torch.zeros((0,), dtype=torch.bool, device="cuda")
        new_radii = radii[new_indices] if new_indices.numel() > 0 else torch.zeros((0,), device="cuda")
        new_opacity = deformed.get_opacity.detach().reshape(-1)[new_indices]
        new_scale = deformed.get_scaling.detach().amax(dim=-1)[new_indices]
        active = proj_valid & new_visible & (new_radii > 0)

        xy_np = xy.detach().cpu().numpy() if xy.numel() > 0 else np.zeros((0, 2), dtype=np.float32)
        inner_np = inner_missing[0].detach().cpu().numpy().astype(bool)
        outer_np = outer_leak[0].detach().cpu().numpy().astype(bool)
        inner_dist = _distance_to_mask(inner_np, xy_np)
        outer_dist = _distance_to_mask(outer_np, xy_np)

        on_inner = _sample_mask(inner_missing.float()[0], xy, active) if xy.numel() > 0 else torch.zeros((0,), device="cuda")
        on_inner_band = _sample_mask(inner_band.float()[0], xy, active) if xy.numel() > 0 else torch.zeros((0,), device="cuda")
        on_outer = _sample_mask(outer_leak.float()[0], xy, active) if xy.numel() > 0 else torch.zeros((0,), device="cuda")

        render_name = f"render_{view.image_name}.png"
        new_support_path = image_dir / render_name
        render_path = args.render_exp / "test-view" / "renders" / render_name
        torchvision.utils.save_image(new_support_image, str(new_support_path))
        inner_mask_path = mask_dir / f"{view.image_name}_inner.npy"
        outer_mask_path = mask_dir / f"{view.image_name}_outer.npy"
        np.save(inner_mask_path, inner_np)
        np.save(outer_mask_path, outer_np)

        active_np = active.detach().cpu().numpy().astype(bool)
        record = {
            "image_name": view.image_name,
            "loaded_iteration": render_iteration,
            "point_count": int(deformed.get_xyz.shape[0]),
            "all_support_count": int(all_support_indices.numel()),
            "new_support_count": int(new_indices.numel()),
            "new_projected_valid": int(proj_valid.sum().item()),
            "new_visible_count": int(new_visible.sum().item()),
            "new_active_count": int(active.sum().item()),
            "new_visible_ratio": float(new_visible.float().mean().item()) if new_visible.numel() else 0.0,
            "new_active_ratio": float(active.float().mean().item()) if active.numel() else 0.0,
            "new_radii_mean": float(new_radii.float().mean().item()) if new_radii.numel() else 0.0,
            "new_radii_positive_mean": float(new_radii[new_radii > 0].float().mean().item()) if bool((new_radii > 0).any().item()) else 0.0,
            "new_opacity_mean": float(new_opacity.float().mean().item()) if new_opacity.numel() else 0.0,
            "new_scale_mean": float(new_scale.float().mean().item()) if new_scale.numel() else 0.0,
            "new_support_opacity_pixels": int(new_support_pixels.sum().item()),
            "new_support_near_pixels": int(new_support_near.sum().item()),
            "inner_missing_pixels": int(inner_missing.sum().item()),
            "outer_leak_pixels": int(outer_leak.sum().item()),
            "inner_band_pixels": int(inner_band.sum().item()),
            "new_support_on_inner_pixels": int((new_support_near & inner_missing[0]).sum().item()),
            "new_support_on_inner_band_pixels": int((new_support_near & inner_band[0]).sum().item()),
            "new_support_on_outer_pixels": int((new_support_near & outer_leak[0]).sum().item()),
            "projected_new_on_inner_points": int((on_inner > 0.5).sum().item()),
            "projected_new_on_inner_band_points": int((on_inner_band > 0.5).sum().item()),
            "projected_new_on_outer_points": int((on_outer > 0.5).sum().item()),
            "active_new_min_dist_to_inner": float(np.min(inner_dist[active_np])) if active_np.any() else float("inf"),
            "active_new_mean_dist_to_inner": float(np.mean(inner_dist[active_np])) if active_np.any() else float("inf"),
            "active_new_min_dist_to_outer": float(np.min(outer_dist[active_np])) if active_np.any() else float("inf"),
            "active_new_mean_dist_to_outer": float(np.mean(outer_dist[active_np])) if active_np.any() else float("inf"),
            "new_support_path": str(new_support_path),
            "render_path": str(render_path),
            "inner_mask_path": str(inner_mask_path),
            "outer_mask_path": str(outer_mask_path),
        }
        records.append(record)

    public_records = [
        {key: value for key, value in record.items() if not key.endswith("_path")}
        for record in records
    ]
    summary = {
        "render_exp": str(args.render_exp),
        "load_ckpt": str(config.load_ckpt),
        "birth_min": int(args.birth_min),
        "loaded_iteration": render_iteration,
        "new_support_count": int(new_indices.numel()),
        "all_support_count": int(all_support_indices.numel()),
        "checkpoint_new_support_opacity": _stat(opacity_all[new_indices]),
        "checkpoint_new_support_scale": _stat(scale_all[new_indices]),
        "checkpoint_all_support_opacity": _stat(opacity_all[all_support_indices]),
        "checkpoint_all_support_scale": _stat(scale_all[all_support_indices]),
        "mean_new_visible_ratio": float(np.mean([r["new_visible_ratio"] for r in records])) if records else 0.0,
        "mean_new_active_ratio": float(np.mean([r["new_active_ratio"] for r in records])) if records else 0.0,
        "mean_new_radii": float(np.mean([r["new_radii_mean"] for r in records])) if records else 0.0,
        "mean_new_radii_positive": float(np.mean([r["new_radii_positive_mean"] for r in records])) if records else 0.0,
        "mean_new_support_opacity_pixels": float(np.mean([r["new_support_opacity_pixels"] for r in records])) if records else 0.0,
        "mean_new_support_on_inner_pixels": float(np.mean([r["new_support_on_inner_pixels"] for r in records])) if records else 0.0,
        "mean_new_support_on_inner_band_pixels": float(np.mean([r["new_support_on_inner_band_pixels"] for r in records])) if records else 0.0,
        "mean_new_support_on_outer_pixels": float(np.mean([r["new_support_on_outer_pixels"] for r in records])) if records else 0.0,
        "mean_inner_missing_pixels": float(np.mean([r["inner_missing_pixels"] for r in records])) if records else 0.0,
        "mean_outer_leak_pixels": float(np.mean([r["outer_leak_pixels"] for r in records])) if records else 0.0,
        "records": public_records,
    }
    (out_dir / "new_support_visibility_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "new_support_visibility_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        if public_records:
            writer = csv.DictWriter(handle, fieldnames=list(public_records[0].keys()))
            writer.writeheader()
            writer.writerows(public_records)
    _make_montage(records, out_dir / "top_new_support_contribution.png", topk=args.topk)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
