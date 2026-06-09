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
import torchvision
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaussian_renderer import render
from scene import GaussianModel, Scene
from tools.audit_377_stageB_v274_contributors import (
    _boundary_band,
    _build_config,
    _dilate,
    _mask_from_view,
    _point_residual_score,
    _project_points,
    _render_support_mask,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _component_masks(mask: torch.Tensor, min_area: int, max_components: int) -> list[dict]:
    mask_np = mask.detach().reshape(mask.shape[-2], mask.shape[-1]).cpu().numpy() > 0.5
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np.astype(np.uint8), 8)
    records = []
    for comp_id in range(1, n_labels):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        records.append({
            "component_id": int(comp_id),
            "area": area,
            "bbox_x": int(stats[comp_id, cv2.CC_STAT_LEFT]),
            "bbox_y": int(stats[comp_id, cv2.CC_STAT_TOP]),
            "bbox_w": int(stats[comp_id, cv2.CC_STAT_WIDTH]),
            "bbox_h": int(stats[comp_id, cv2.CC_STAT_HEIGHT]),
            "centroid_x": float(centroids[comp_id][0]),
            "centroid_y": float(centroids[comp_id][1]),
            "mask_np": labels == comp_id,
        })
    records.sort(key=lambda row: int(row["area"]), reverse=True)
    return records[: int(max_components)]


def _nearest_point(source_xy: np.ndarray, target_mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(target_mask)
    if xs.size == 0:
        return None
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    delta = coords - source_xy.reshape(1, 2).astype(np.float32)
    idx = int(np.argmin(np.sum(delta * delta, axis=1)))
    return coords[idx]


def _screen_xy(points: torch.Tensor, view) -> tuple[torch.Tensor, torch.Tensor]:
    xy, valid, _ = _project_points(points, view)
    return xy, valid


def _screen_jacobian_world(points: torch.Tensor, view, eps: float) -> torch.Tensor:
    point_count = int(points.shape[0])
    device = points.device
    dtype = points.dtype
    basis = torch.eye(3, dtype=dtype, device=device).view(1, 3, 3).expand(point_count, 3, 3)
    base_xy, _ = _screen_xy(points, view)
    shifted = points[:, None, :] + basis * float(eps)
    shifted_xy, _ = _screen_xy(shifted.reshape(-1, 3), view)
    shifted_xy = shifted_xy.reshape(point_count, 3, 2)
    return ((shifted_xy - base_xy[:, None, :]) / float(eps)).permute(0, 2, 1).contiguous()


def _world_delta_from_screen_shift(points: torch.Tensor, view, shift_px: torch.Tensor, eps: float, damping: float) -> torch.Tensor:
    if points.numel() == 0:
        return torch.zeros_like(points)
    jac = _screen_jacobian_world(points, view, eps=eps).float()
    jj_t = torch.bmm(jac, jac.transpose(1, 2))
    eye = torch.eye(2, device=points.device, dtype=torch.float32).unsqueeze(0)
    inv = torch.linalg.inv(jj_t + float(damping) * eye)
    rhs = shift_px.float().unsqueeze(-1)
    world = torch.bmm(jac.transpose(1, 2), torch.bmm(inv, rhs)).squeeze(-1)
    return torch.nan_to_num(world.to(dtype=points.dtype), nan=0.0, posinf=0.0, neginf=0.0)


def _cap_norm(delta: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0.0 or delta.numel() == 0:
        return delta
    norm = torch.norm(delta, dim=-1, keepdim=True).clamp_min(1.0e-12)
    return delta * torch.clamp(float(max_norm) / norm, max=1.0)


def _canonical_delta_from_world(pc, point_ids: torch.Tensor, world_delta: torch.Tensor) -> torch.Tensor:
    transform = getattr(pc, "fwd_transform", None)
    if torch.is_tensor(transform) and transform.shape[0] > int(point_ids.max().item()):
        rot = transform[point_ids, :3, :3].detach().float()
        return torch.bmm(rot.transpose(1, 2), world_delta.float().unsqueeze(-1)).squeeze(-1).to(world_delta.dtype)
    return world_delta


def _direction_vector_outer(comp: dict, gt_mask: torch.Tensor, args: argparse.Namespace) -> tuple[np.ndarray, float]:
    mask_np = gt_mask.detach().reshape(gt_mask.shape[-2], gt_mask.shape[-1]).cpu().numpy() > 0.5
    band_np = (_boundary_band(gt_mask, args.band_width).detach().reshape(mask_np.shape).cpu().numpy() > 0.5)
    target = _nearest_point(
        np.array([comp["centroid_x"], comp["centroid_y"]], dtype=np.float32),
        band_np if bool(band_np.any()) else mask_np,
    )
    if target is None:
        return np.zeros((2,), dtype=np.float32), 0.0
    source = np.array([comp["centroid_x"], comp["centroid_y"]], dtype=np.float32)
    direction = target - source
    length = float(np.linalg.norm(direction))
    if length < 1.0e-6:
        return np.zeros((2,), dtype=np.float32), 0.0
    return direction / length, length


def _direction_vector_inner(comp: dict, point_xy: torch.Tensor) -> torch.Tensor:
    centroid = point_xy.new_tensor([comp["centroid_x"], comp["centroid_y"]]).view(1, 2)
    direction = centroid - point_xy
    length = torch.norm(direction, dim=-1, keepdim=True).clamp_min(1.0e-6)
    return direction / length


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v276 component-level signed directional actuator plan.")
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
    parser.add_argument("--component-shift-px", type=float, default=1.25)
    parser.add_argument("--component-weight-area-power", type=float, default=0.5)
    parser.add_argument("--jacobian-eps", type=float, default=1.0e-3)
    parser.add_argument("--jacobian-damping", type=float, default=1.0e-5)
    parser.add_argument("--max-component-world-step", type=float, default=0.003)
    parser.add_argument("--max-point-canonical-step", type=float, default=0.006)
    parser.add_argument("--min-point-weight", type=float, default=1.0)
    parser.add_argument("--min-direction-consistency", type=float, default=0.25)
    parser.add_argument("--max-plan-points", type=int, default=384)
    parser.add_argument("--render-scaling-modifier", type=float, default=1.0)
    parser.add_argument("--rotation-orthogonalize", action="store_true")
    parser.add_argument("--compute-cov3d-python", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-geometry", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
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
    background = torch.tensor(
        [1, 1, 1] if config.dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    frame_records: list[dict] = []
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
            inner_band = (inner * band).clamp(0.0, 1.0)
            outer_band = (outer * _boundary_band(gt_mask, max(args.band_width * 2, 3))).clamp(0.0, 1.0)
            fg_pixels = max(float(gt_mask.sum().item()), 1.0)
            frame_records.append({
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
            })
            del render_pkg

    frame_records.sort(key=lambda item: item["hard_residual_score"], reverse=True)
    audit_records = frame_records[: int(args.top_frames)]

    accum: dict[int, dict] = {}
    component_rows: list[dict] = []
    point_component_rows: list[dict] = []

    with torch.no_grad():
        for frame_rank, record in enumerate(audit_records, start=1):
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
            point_count = int(pc.get_xyz.shape[0])
            device = pc.get_xyz.device
            xy, valid, _ = _project_points(pc.get_xyz, view)
            radii = render_pkg["radii"].detach().float().to(device)
            opacity = pc.get_opacity.detach().reshape(-1).float().to(device)
            visible = valid & (radii > 0.0)

            for direction, residual_mask, radius_scale in (
                ("outer", _dilate(outer_band, args.residual_dilate), args.outer_radius_scale),
                ("inner", _dilate(inner_band, args.residual_dilate), args.inner_radius_scale),
            ):
                comps = _component_masks(residual_mask, args.min_component_area, args.max_components_per_frame)
                for comp_rank, comp in enumerate(comps, start=1):
                    comp_mask = torch.from_numpy(comp["mask_np"].astype(np.float32)).to(device=device).unsqueeze(0)
                    comp_score, _ = _point_residual_score(
                        comp_mask,
                        xy,
                        visible,
                        radii,
                        opacity,
                        radius_scale=radius_scale,
                        min_radius=args.min_radius,
                        max_radius=args.max_radius,
                        opacity_power=args.opacity_power,
                        radius_power=args.radius_power,
                    )
                    top_values, top_indices = torch.topk(comp_score, k=min(int(args.component_top_points), point_count))
                    keep = top_values > 0.0
                    top_values = top_values[keep]
                    top_indices = top_indices[keep].long()
                    if top_indices.numel() == 0:
                        continue

                    if direction == "outer":
                        unit_np, raw_len = _direction_vector_outer(comp, gt_mask, args)
                        if raw_len <= 1.0e-6:
                            continue
                        shift = top_values.new_tensor(unit_np).view(1, 2).expand(top_indices.numel(), 2)
                    else:
                        shift = _direction_vector_inner(comp, xy[top_indices])
                        raw_len = float(torch.norm(
                            xy[top_indices].mean(dim=0) - xy.new_tensor([comp["centroid_x"], comp["centroid_y"]])
                        ).item())
                    shift = shift * float(args.component_shift_px)

                    world_delta = _world_delta_from_screen_shift(
                        pc.get_xyz[top_indices],
                        view,
                        shift,
                        eps=float(args.jacobian_eps),
                        damping=float(args.jacobian_damping),
                    )
                    world_delta = _cap_norm(world_delta, float(args.max_component_world_step))
                    canonical_delta = _canonical_delta_from_world(pc, top_indices, world_delta)
                    canonical_delta = _cap_norm(canonical_delta, float(args.max_component_world_step))

                    area_weight = float(comp["area"]) ** float(args.component_weight_area_power)
                    for local_rank, (pid, score_value) in enumerate(
                        zip(top_indices.detach().cpu().tolist(), top_values.detach().cpu().tolist()),
                        start=1,
                    ):
                        pid = int(pid)
                        weight = float(score_value) * area_weight
                        delta = canonical_delta[local_rank - 1].detach().float()
                        item = accum.setdefault(pid, {
                            "point_idx": pid,
                            "delta_sum": torch.zeros((3,), dtype=torch.float32, device=device),
                            "delta_abs_sum": 0.0,
                            "weight_sum": 0.0,
                            "outer_weight": 0.0,
                            "inner_weight": 0.0,
                            "component_hits": 0,
                            "outer_hits": 0,
                            "inner_hits": 0,
                        })
                        item["delta_sum"] = item["delta_sum"] + delta * weight
                        item["delta_abs_sum"] += float(torch.norm(delta).item()) * weight
                        item["weight_sum"] += weight
                        item["component_hits"] += 1
                        if direction == "outer":
                            item["outer_weight"] += weight
                            item["outer_hits"] += 1
                        else:
                            item["inner_weight"] += weight
                            item["inner_hits"] += 1

                        point_component_rows.append({
                            "cam": record["cam"],
                            "frame": record["frame"],
                            "image_name": record["image_name"],
                            "frame_rank": frame_rank,
                            "direction": direction,
                            "component_rank": comp_rank,
                            "component_id": int(comp["component_id"]),
                            "component_area": int(comp["area"]),
                            "component_centroid_x": float(comp["centroid_x"]),
                            "component_centroid_y": float(comp["centroid_y"]),
                            "point_rank": local_rank,
                            "point_idx": pid,
                            "point_score": float(score_value),
                            "point_x": float(xy[pid, 0].item()),
                            "point_y": float(xy[pid, 1].item()),
                            "shift_px_x": float(shift[local_rank - 1, 0].item()),
                            "shift_px_y": float(shift[local_rank - 1, 1].item()),
                            "canonical_delta_x": float(delta[0].item()),
                            "canonical_delta_y": float(delta[1].item()),
                            "canonical_delta_z": float(delta[2].item()),
                            "weight": weight,
                        })

                    component_rows.append({
                        "cam": record["cam"],
                        "frame": record["frame"],
                        "image_name": record["image_name"],
                        "frame_rank": frame_rank,
                        "direction": direction,
                        "component_rank": comp_rank,
                        "component_id": int(comp["component_id"]),
                        "area": int(comp["area"]),
                        "bbox_x": int(comp["bbox_x"]),
                        "bbox_y": int(comp["bbox_y"]),
                        "bbox_w": int(comp["bbox_w"]),
                        "bbox_h": int(comp["bbox_h"]),
                        "centroid_x": float(comp["centroid_x"]),
                        "centroid_y": float(comp["centroid_y"]),
                        "raw_direction_len_px": float(raw_len),
                        "top_point_count": int(top_indices.numel()),
                        "score_sum": float(top_values.sum().item()),
                        "top_point_ids": ";".join(str(int(x)) for x in top_indices.detach().cpu().tolist()),
                        "top_point_scores": ";".join(f"{float(x):.6f}" for x in top_values.detach().cpu().tolist()),
                    })

            torchvision.utils.save_image(render_rgb, str(render_dir / f"render_{record['image_name']}.png"))
            _make_overlay(render_rgb, inner_band, outer_band, overlay_dir / f"residual_{record['image_name']}.png")
            del render_pkg

    point_rows = []
    for item in accum.values():
        weight = max(float(item["weight_sum"]), 1.0e-8)
        delta = item["delta_sum"] / weight
        delta = _cap_norm(delta.view(1, 3), float(args.max_point_canonical_step)).view(3)
        delta_norm = float(torch.norm(delta).item())
        consistency = 0.0
        if float(item["delta_abs_sum"]) > 1.0e-8:
            consistency = float(torch.norm(item["delta_sum"]).item()) / float(item["delta_abs_sum"])
        outer_weight = float(item["outer_weight"])
        inner_weight = float(item["inner_weight"])
        dominant_direction = "outer" if outer_weight >= inner_weight else "inner"
        conflict_ratio = min(outer_weight, inner_weight) / max(max(outer_weight, inner_weight), 1.0e-8)
        if weight < float(args.min_point_weight) or consistency < float(args.min_direction_consistency) or delta_norm <= 1.0e-8:
            continue
        point_rows.append({
            "point_idx": int(item["point_idx"]),
            "delta_x": float(delta[0].item()),
            "delta_y": float(delta[1].item()),
            "delta_z": float(delta[2].item()),
            "delta_norm": delta_norm,
            "weight_sum": weight,
            "outer_weight": outer_weight,
            "inner_weight": inner_weight,
            "dominant_direction": dominant_direction,
            "conflict_ratio": conflict_ratio,
            "direction_consistency": consistency,
            "component_hits": int(item["component_hits"]),
            "outer_hits": int(item["outer_hits"]),
            "inner_hits": int(item["inner_hits"]),
        })
    point_rows.sort(key=lambda row: float(row["weight_sum"]) * float(row["direction_consistency"]), reverse=True)
    point_rows = point_rows[: int(args.max_plan_points)]

    _write_csv(args.out_dir / "sample_metrics.csv", sorted(frame_records, key=lambda item: (str(item["cam"]), int(item["frame"]))))
    _write_csv(args.out_dir / "component_direction_records.csv", component_rows)
    _write_csv(args.out_dir / "point_component_direction_records.csv", point_component_rows)
    _write_csv(args.out_dir / "point_direction_plan.csv", point_rows)

    by_direction = defaultdict(int)
    for row in point_rows:
        by_direction[str(row["dominant_direction"])] += 1
    plan = {
        "load_ckpt": str(args.load_ckpt),
        "loaded_iteration": int(loaded_iteration),
        "n_samples": len(frame_records),
        "top_frames": audit_records,
        "mean_inner_missing_pixels": float(np.mean([r["inner_missing_pixels"] for r in frame_records])) if frame_records else 0.0,
        "mean_outer_leak_pixels": float(np.mean([r["outer_leak_pixels"] for r in frame_records])) if frame_records else 0.0,
        "mean_hard_residual_score": float(np.mean([r["hard_residual_score"] for r in frame_records])) if frame_records else 0.0,
        "component_count": len(component_rows),
        "point_count": len(point_rows),
        "point_count_by_dominant_direction": dict(by_direction),
        "settings": {
            "component_shift_px": float(args.component_shift_px),
            "max_component_world_step": float(args.max_component_world_step),
            "max_point_canonical_step": float(args.max_point_canonical_step),
            "min_point_weight": float(args.min_point_weight),
            "min_direction_consistency": float(args.min_direction_consistency),
        },
        "points": point_rows,
    }
    (args.out_dir / "component_direction_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
