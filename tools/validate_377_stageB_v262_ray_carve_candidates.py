#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
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


IMAGE_RE = re.compile(r"c(?P<cam>\d+)_f(?P<frame>\d+)")


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


def _range_spec_from_values(values: list[int]) -> list[int]:
    values = sorted(set(int(v) for v in values))
    if len(values) <= 1:
        return [values[0], values[0] + 1, 1] if values else [0, 1, 1]
    step = values[1] - values[0]
    if step > 0 and all(values[i + 1] - values[i] == step for i in range(len(values) - 1)):
        return [values[0], values[-1] + step, step]
    return values


def _view_ids(name: str) -> tuple[int, int]:
    match = IMAGE_RE.search(str(name))
    if not match:
        return -1, -1
    return int(match.group("cam")), int(match.group("frame"))


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


def _sample_mask(mask: torch.Tensor, xy: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    sampled = torch.zeros((xy.shape[0],), dtype=torch.float32, device=xy.device)
    if xy.numel() == 0 or not bool(valid.any().item()):
        return sampled
    height, width = mask.shape[-2:]
    x = xy[:, 0].round().long().clamp(0, width - 1)
    y = xy[:, 1].round().long().clamp(0, height - 1)
    sampled[valid] = mask.reshape(height, width)[y[valid], x[valid]].float()
    return sampled


def _unproject_pixels(view, xy: torch.Tensor, ndc_z: float) -> torch.Tensor:
    device = xy.device
    dtype = torch.float32
    width = int(view.image_width)
    height = int(view.image_height)
    ndc_x = (xy[:, 0] / float(max(width - 1, 1))) * 2.0 - 1.0
    ndc_y = 1.0 - (xy[:, 1] / float(max(height - 1, 1))) * 2.0
    z = torch.full_like(ndc_x, float(ndc_z))
    clip = torch.stack((ndc_x, ndc_y, z, torch.ones_like(z)), dim=-1).to(device=device, dtype=dtype)
    full_proj = view.full_proj_transform.detach().to(device=device, dtype=dtype)
    try:
        inv = torch.inverse(full_proj)
    except RuntimeError:
        inv = torch.linalg.pinv(full_proj)
    world_h = torch.matmul(clip, inv)
    return world_h[:, :3] / world_h[:, 3:4].clamp(min=1.0e-8)


def _component_sample_pixels(mask: torch.Tensor, score: torch.Tensor, max_components: int, points_per_component: int, min_area: int) -> list[tuple[float, float]]:
    mask_np = (mask.detach().cpu().numpy().astype(np.uint8) > 0).astype(np.uint8)
    score_np = score.detach().cpu().numpy().astype(np.float32)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, 8)
    components = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        label_mask = labels == label
        value = float(score_np[label_mask].max()) + 0.002 * min(area, 512)
        components.append((value, area, label))
    components.sort(reverse=True)
    if max_components > 0:
        components = components[:max_components]
    samples = []
    for _, _, label in components:
        coords_yx = np.column_stack(np.where(labels == label))
        if coords_yx.shape[0] == 0:
            continue
        coords_xy = coords_yx[:, [1, 0]].astype(np.float32)
        center = np.asarray(centroids[label], dtype=np.float32)
        centered = coords_xy - coords_xy.mean(axis=0, keepdims=True)
        if coords_xy.shape[0] >= 3:
            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                axis = vh[0].astype(np.float32)
            except np.linalg.LinAlgError:
                axis = np.array([1.0, 0.0], dtype=np.float32)
        else:
            axis = np.array([1.0, 0.0], dtype=np.float32)
        proj = centered @ axis
        picks = [center]
        if points_per_component > 1:
            for q in np.linspace(0.20, 0.80, int(points_per_component) - 1):
                target = np.quantile(proj, float(q))
                picks.append(coords_xy[int(np.argmin(np.abs(proj - target)))])
        for xy in picks[: int(points_per_component)]:
            samples.append((float(xy[0]), float(xy[1])))
    return samples


def _distance_to_inner_outer(record: dict, xy: torch.Tensor, valid: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    xy_np = xy.detach().cpu().numpy() if xy.numel() else np.zeros((0, 2), dtype=np.float32)
    valid_np = valid.detach().cpu().numpy().astype(bool) if valid.numel() else np.zeros((0,), dtype=bool)
    inner_np = record["inner"][0].detach().cpu().numpy().astype(bool)
    outer_np = record["outer"][0].detach().cpu().numpy().astype(bool)
    out_inner = np.full((xy_np.shape[0],), np.inf, dtype=np.float32)
    out_outer = np.full((xy_np.shape[0],), np.inf, dtype=np.float32)
    if inner_np.any():
        dist = cv2.distanceTransform((~inner_np).astype(np.uint8), cv2.DIST_L2, 3)
        x = np.clip(np.rint(xy_np[:, 0]).astype(np.int32), 0, dist.shape[1] - 1)
        y = np.clip(np.rint(xy_np[:, 1]).astype(np.int32), 0, dist.shape[0] - 1)
        out_inner[valid_np] = dist[y[valid_np], x[valid_np]]
    if outer_np.any():
        dist = cv2.distanceTransform((~outer_np).astype(np.uint8), cv2.DIST_L2, 3)
        x = np.clip(np.rint(xy_np[:, 0]).astype(np.int32), 0, dist.shape[1] - 1)
        y = np.clip(np.rint(xy_np[:, 1]).astype(np.int32), 0, dist.shape[0] - 1)
        out_outer[valid_np] = dist[y[valid_np], x[valid_np]]
    return out_inner, out_outer


def _make_overlay(frame_records: dict[int, list[dict]], accepted: list[dict], out_path: Path, topk: int) -> None:
    if not accepted:
        return
    tiles = []
    for cand in accepted[:topk]:
        frame = int(cand["frame"])
        views = frame_records.get(frame, [])[:3]
        xyz = torch.tensor(cand["xyz"], dtype=torch.float32, device="cuda").reshape(1, 3)
        panels = []
        for rec in views:
            img = rec["render"].detach().cpu().clamp(0, 1)
            overlay = img.clone()
            overlay[:, rec["inner"][0].detach().cpu().bool()] = torch.tensor([0.15, 0.95, 0.25]).reshape(3, 1)
            overlay[:, rec["outer"][0].detach().cpu().bool()] = torch.tensor([1.0, 0.1, 0.12]).reshape(3, 1)
            xy, valid = _project_points(xyz, rec["view"])
            if bool(valid[0].item()):
                x = int(round(float(xy[0, 0].item())))
                y = int(round(float(xy[0, 1].item())))
                h, w = overlay.shape[-2:]
                rr = 5
                overlay[:, max(0, y - rr):min(h, y + rr + 1), max(0, x - rr):min(w, x + rr + 1)] = torch.tensor([1.0, 0.9, 0.05]).reshape(3, 1, 1)
            panels.append(torch.cat([img, overlay], dim=2))
        if panels:
            tiles.append(torch.cat(panels, dim=2))
    if not tiles:
        return
    grid = torch.cat(tiles, dim=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(grid, str(out_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="v262 offline multiview ray-carve support candidate validator.")
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--load-ckpt", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--parser-root", default="")
    parser.add_argument("--compact-mapping", default="")
    parser.add_argument("--candidate-views", default="1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--eval-views", default="21,22,23")
    parser.add_argument("--frames", default="0,570,60")
    parser.add_argument("--render-support-threshold", type=float, default=0.025)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--band-width", type=int, default=7)
    parser.add_argument("--search-band-width", type=int, default=24)
    parser.add_argument("--max-components-per-view", type=int, default=6)
    parser.add_argument("--points-per-component", type=int, default=2)
    parser.add_argument("--min-component-area", type=int, default=18)
    parser.add_argument("--depth-samples", type=int, default=9)
    parser.add_argument("--depth-margin", type=float, default=0.060)
    parser.add_argument("--depth-search-radius", type=float, default=32.0)
    parser.add_argument("--min-inner-views", type=int, default=2)
    parser.add_argument("--max-outer-views", type=int, default=0)
    parser.add_argument("--min-heldout-inner-views", type=int, default=1)
    parser.add_argument("--max-heldout-outer-views", type=int, default=0)
    parser.add_argument("--max-candidates-per-frame", type=int, default=16)
    parser.add_argument("--topk", type=int, default=12)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    candidate_views = _parse_csv_ints(args.candidate_views)
    eval_views = _parse_csv_ints(args.eval_views)
    all_views = sorted(set(candidate_views + eval_views))
    frames = _range_values(args.frames)

    config = OmegaConf.load(args.config_path)
    OmegaConf.set_struct(config, False)
    config.mode = "test"
    config.exp_dir = str(args.out_dir / "scene")
    config.load_ckpt = str(args.load_ckpt)
    config.dataset.root_dir = str(args.dataset_root)
    config.dataset.preload = False
    config.dataset.test_views.view = all_views
    config.dataset.test_frames.view = _range_spec_from_values(frames)
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

    background = torch.tensor(
        [1, 1, 1] if bool(config.dataset.white_background) else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    gaussians = GaussianModel(config.model.gaussian)
    scene = Scene(config, gaussians, config.exp_dir)
    scene.eval()
    loaded_iteration = int(scene.load_checkpoint(str(args.load_ckpt)))

    frame_records: dict[int, list[dict]] = defaultdict(list)
    for idx in range(len(scene.test_dataset)):
        view = scene.test_dataset[idx]
        cam, frame = _view_ids(view.image_name)
        if frame not in frames or cam not in all_views:
            continue
        pkg = render(view, loaded_iteration, scene, config.pipeline, background, compute_loss=False, return_opacity=True)
        rendering = pkg["render"].detach().clamp(0.0, 1.0)
        gt_mask = _mask_from_view(view)
        support = _render_support_mask(rendering, args.render_support_threshold, args.close_kernel)
        near_gt = _dilate(gt_mask, args.search_band_width)
        inner = (gt_mask > 0.5) & (support <= 0.5)
        outer = (support > 0.5) & (gt_mask <= 0.5) & (near_gt > 0.5)
        inner_band = inner & (_boundary_band(gt_mask, args.band_width) > 0.5)
        deformed = pkg["deformed_gaussian"]
        xy, valid = _project_points(deformed.get_xyz, view)
        visible = valid & pkg["visibility_filter"].detach().bool() & (pkg["radii"].detach() > 0)
        center = view.camera_center.detach().float().cuda().reshape(1, 3)
        dirs_to_points = deformed.get_xyz.detach().float() - center
        point_depths = torch.norm(dirs_to_points, dim=-1)
        frame_records[frame].append({
            "view": view,
            "cam": cam,
            "frame": frame,
            "render": rendering,
            "inner": inner.float(),
            "outer": outer.float(),
            "inner_band": inner_band.float(),
            "xy": xy.detach(),
            "visible": visible.detach(),
            "point_depths": point_depths.detach(),
            "deformed_xyz": deformed.get_xyz.detach().float(),
            "inner_pixels": int(inner.sum().item()),
            "outer_pixels": int(outer.sum().item()),
        })

    accepted = []
    all_candidates = []
    for frame in sorted(frame_records):
        records = frame_records[frame]
        by_cam = {r["cam"]: r for r in records}
        frame_candidates = []
        for cam in candidate_views:
            rec = by_cam.get(cam)
            if rec is None:
                continue
            sample_pixels = _component_sample_pixels(
                rec["inner"][0] > 0.5,
                rec["inner"][0].float(),
                args.max_components_per_view,
                args.points_per_component,
                args.min_component_area,
            )
            if not sample_pixels:
                continue
            sample_xy = torch.tensor(sample_pixels, dtype=torch.float32, device="cuda")
            p_near = _unproject_pixels(rec["view"], sample_xy, 0.05)
            p_far = _unproject_pixels(rec["view"], sample_xy, 0.95)
            ray_dir = F.normalize(p_far - p_near, dim=-1)
            center = rec["view"].camera_center.detach().float().cuda().reshape(1, 3)
            visible_xy = rec["xy"][rec["visible"]]
            visible_depths = rec["point_depths"][rec["visible"]]
            if visible_depths.numel() == 0:
                continue
            global_depth = torch.median(visible_depths)
            for pix_idx, target_xy in enumerate(sample_xy):
                dxy = visible_xy - target_xy.reshape(1, 2)
                dist = torch.norm(dxy, dim=-1)
                near = dist <= float(args.depth_search_radius)
                if bool(near.any().item()):
                    depth_center = torch.median(visible_depths[near])
                else:
                    depth_center = global_depth
                offsets = torch.linspace(
                    -float(args.depth_margin),
                    float(args.depth_margin),
                    max(1, int(args.depth_samples)),
                    device="cuda",
                )
                depths = (depth_center + offsets).clamp_min(0.01)
                points = center + ray_dir[pix_idx:pix_idx + 1] * depths.reshape(-1, 1)
                for depth_idx, point in enumerate(points):
                    candidate = {
                        "frame": int(frame),
                        "birth_cam": int(cam),
                        "birth_x": float(target_xy[0].item()),
                        "birth_y": float(target_xy[1].item()),
                        "depth": float(depths[depth_idx].item()),
                        "xyz": [float(v) for v in point.detach().cpu().tolist()],
                    }
                    frame_candidates.append(candidate)

        if not frame_candidates:
            continue
        points = torch.tensor([c["xyz"] for c in frame_candidates], dtype=torch.float32, device="cuda")
        total_inner = torch.zeros((points.shape[0],), dtype=torch.float32, device="cuda")
        total_outer = torch.zeros_like(total_inner)
        total_inner_band = torch.zeros_like(total_inner)
        heldout_inner = torch.zeros_like(total_inner)
        heldout_outer = torch.zeros_like(total_inner)
        valid_views = torch.zeros_like(total_inner)
        min_inner_dist = np.full((points.shape[0],), np.inf, dtype=np.float32)
        min_outer_dist = np.full((points.shape[0],), np.inf, dtype=np.float32)
        for rec in records:
            xy, valid = _project_points(points, rec["view"])
            valid_views += valid.float()
            inner = _sample_mask(rec["inner"][0].float(), xy, valid)
            outer = _sample_mask(rec["outer"][0].float(), xy, valid)
            inner_band = _sample_mask(rec["inner_band"][0].float(), xy, valid)
            total_inner += inner
            total_outer += outer
            total_inner_band += inner_band
            if rec["cam"] in eval_views:
                heldout_inner += inner
                heldout_outer += outer
            inner_dist, outer_dist = _distance_to_inner_outer(rec, xy, valid)
            min_inner_dist = np.minimum(min_inner_dist, inner_dist)
            min_outer_dist = np.minimum(min_outer_dist, outer_dist)
        score = total_inner + 0.5 * total_inner_band + heldout_inner - 2.0 * total_outer - 3.0 * heldout_outer
        order = torch.argsort(score, descending=True).detach().cpu().tolist()
        kept_for_frame = 0
        for idx in order:
            c = frame_candidates[idx]
            row = {
                **c,
                "valid_views": int(valid_views[idx].item()),
                "inner_views": float(total_inner[idx].item()),
                "inner_band_views": float(total_inner_band[idx].item()),
                "outer_views": float(total_outer[idx].item()),
                "heldout_inner_views": float(heldout_inner[idx].item()),
                "heldout_outer_views": float(heldout_outer[idx].item()),
                "min_dist_to_inner": float(min_inner_dist[idx]),
                "min_dist_to_outer": float(min_outer_dist[idx]),
                "score": float(score[idx].item()),
            }
            all_candidates.append(row)
            is_ok = (
                row["inner_views"] >= int(args.min_inner_views)
                and row["outer_views"] <= int(args.max_outer_views)
                and row["heldout_inner_views"] >= int(args.min_heldout_inner_views)
                and row["heldout_outer_views"] <= int(args.max_heldout_outer_views)
            )
            if is_ok and kept_for_frame < int(args.max_candidates_per_frame):
                accepted.append(row)
                kept_for_frame += 1

    accepted.sort(key=lambda r: (r["heldout_inner_views"], r["inner_views"], -r["outer_views"], r["score"]), reverse=True)
    all_candidates.sort(key=lambda r: r["score"], reverse=True)
    status = "ok" if accepted else "blocked"
    reasons = [] if accepted else ["no_inner_dominant_ray_carve_candidates"]
    summary = {
        "status": status,
        "reasons": reasons,
        "config_path": str(args.config_path),
        "load_ckpt": str(args.load_ckpt),
        "loaded_iteration": loaded_iteration,
        "candidate_views": candidate_views,
        "eval_views": eval_views,
        "frames": frames,
        "rendered_view_count": sum(len(v) for v in frame_records.values()),
        "raw_candidate_count": len(all_candidates),
        "accepted_candidate_count": len(accepted),
        "mean_inner_views": float(np.mean([r["inner_views"] for r in accepted])) if accepted else 0.0,
        "mean_outer_views": float(np.mean([r["outer_views"] for r in accepted])) if accepted else 0.0,
        "mean_heldout_inner_views": float(np.mean([r["heldout_inner_views"] for r in accepted])) if accepted else 0.0,
        "mean_heldout_outer_views": float(np.mean([r["heldout_outer_views"] for r in accepted])) if accepted else 0.0,
        "top_accepted": accepted[: args.topk],
        "top_candidates": all_candidates[: args.topk],
    }
    (args.out_dir / "ray_carve_candidate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out_dir / "ray_carve_accepted_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        if accepted:
            writer = csv.DictWriter(handle, fieldnames=list(accepted[0].keys()))
            writer.writeheader()
            writer.writerows(accepted)
    with (args.out_dir / "ray_carve_top_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        top = all_candidates[: max(1000, args.topk)]
        if top:
            writer = csv.DictWriter(handle, fieldnames=list(top[0].keys()))
            writer.writeheader()
            writer.writerows(top)
    _make_overlay(frame_records, accepted, args.out_dir / "top_ray_carve_candidates_overlay.png", args.topk)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
