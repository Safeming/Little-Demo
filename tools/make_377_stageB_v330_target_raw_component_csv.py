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


def _build_config(args):
    config = OmegaConf.load(args.config_path)
    OmegaConf.set_struct(config, False)
    config.mode = "test"
    config.exp_dir = str(args.out_dir / "component_scene")
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


def _view_mask(view):
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
    return mask > 0.5


def _render_support(render_rgb, threshold, close_kernel):
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
    return support > 0.5


def _dilate_np(mask: np.ndarray, width: int) -> np.ndarray:
    width = max(0, int(width))
    if width <= 0:
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _project_points(points, view):
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


def _top_points(component_mask, xy, visible, radii, max_points, pad_px):
    padded = _dilate_np(component_mask, int(pad_px))
    if xy.numel() == 0:
        return [], []
    height, width = component_mask.shape
    x = xy[:, 0].round().long().clamp(0, width - 1)
    y = xy[:, 1].round().long().clamp(0, height - 1)
    comp_t = torch.from_numpy(component_mask.astype(np.bool_)).to(device=xy.device)
    pad_t = torch.from_numpy(padded.astype(np.bool_)).to(device=xy.device)
    inside = visible & pad_t[y, x]
    if not bool(inside.any().item()):
        return [], []
    core = comp_t[y, x]
    yy, xx = np.where(component_mask)
    cx = float(xx.mean()) if xx.size else 0.0
    cy = float(yy.mean()) if yy.size else 0.0
    dist = torch.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
    spatial = torch.exp(-dist / 18.0)
    score = spatial + core.float() * 1.5 + (radii / radii[inside].quantile(0.90).clamp_min(1.0)).clamp(0.0, 2.0) * 0.15
    score = torch.where(inside, score, torch.full_like(score, -float("inf")))
    k = min(int(max_points), int(inside.sum().item()))
    values, ids = torch.topk(score, k=max(k, 0))
    point_ids = []
    point_scores = []
    for value, idx in zip(values.detach().cpu().tolist(), ids.detach().cpu().tolist()):
        if np.isfinite(float(value)):
            point_ids.append(int(idx))
            point_scores.append(float(value))
    return point_ids, point_scores


def _rows_for_direction(mask_np, direction, cam, frame, image_name, xy, visible, radii, args):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np.astype(np.uint8), 8)
    comps = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(args.min_component_area):
            continue
        comps.append((area, label))
    comps.sort(reverse=True)
    comps = comps[: max(int(args.max_components_per_direction), 0)]
    rows = []
    for component_id, (area, label) in enumerate(comps, start=1):
        comp = labels == label
        ids, scores = _top_points(comp, xy, visible, radii, args.top_points, args.point_pad_px)
        if not ids and bool(args.require_top_points):
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        rows.append({
            "cam": int(cam),
            "frame": int(frame),
            "image_name": image_name,
            "direction": direction,
            "component_id": component_id,
            "area": int(area),
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "centroid_x": float(centroids[label][0]),
            "centroid_y": float(centroids[label][1]),
            "near_point_count": int(len(ids)),
            "near_score_sum": float(sum(scores)),
            "min_center_dist": 0.0,
            "top_point_ids": ";".join(str(x) for x in ids),
            "top_point_scores": ";".join(f"{x:.6f}" for x in scores),
        })
    return rows


def _image_ids(view):
    image_name = str(getattr(view, "image_name", ""))
    def _safe_attr(*names):
        for name in names:
            try:
                return int(getattr(view, name))
            except Exception:
                continue
        return -1

    cam = _safe_attr("cam_id", "uid")
    frame = _safe_attr("frame_id", "frame_idx")
    if cam < 0 and image_name.startswith("c") and "_f" in image_name:
        cam = int(image_name.split("_f", 1)[0][1:])
    if frame < 0 and "_f" in image_name:
        frame = int(image_name.rsplit("_f", 1)[1])
    return cam, frame, image_name


def main():
    parser = argparse.ArgumentParser(description="Build v330 target-raw residual component CSV.")
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--load-ckpt", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--train-views", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--target-views", default="21,22,23")
    parser.add_argument("--train-frames", default="0,570,60")
    parser.add_argument("--target-frames", default="0,570,60")
    parser.add_argument("--render-support-threshold", type=float, default=0.025)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--search-band-width", type=int, default=24)
    parser.add_argument("--min-component-area", type=int, default=20)
    parser.add_argument("--max-components-per-direction", type=int, default=16)
    parser.add_argument("--top-points", type=int, default=8)
    parser.add_argument("--point-pad-px", type=int, default=10)
    parser.add_argument("--require-top-points", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = _build_config(args)
    gaussians = GaussianModel(config.model.gaussian)
    scene = Scene(config, gaussians, str(args.out_dir / "component_scene"))
    scene.eval()
    loaded_iteration = scene.load_checkpoint(str(args.load_ckpt))
    bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    rows = []
    with torch.no_grad():
        for idx in range(len(scene.test_dataset)):
            view = scene.test_dataset[idx]
            pkg = render(view, loaded_iteration, scene, config.pipeline, background, compute_loss=False)
            pc = pkg["deformed_gaussian"]
            xy, proj_valid = _project_points(pc.get_xyz, view)
            radii = pkg["radii"].detach().float().to(device=pc.get_xyz.device).reshape(-1)
            visible = proj_valid & pkg["visibility_filter"].detach().bool().to(device=pc.get_xyz.device) & (radii > 0.0)
            gt = _view_mask(view)
            support = _render_support(pkg["render"].detach().float().clamp(0.0, 1.0), args.render_support_threshold, args.close_kernel)
            gt_np = gt[0].detach().cpu().numpy().astype(bool)
            support_np = support[0].detach().cpu().numpy().astype(bool)
            near_gt = _dilate_np(gt_np, int(args.search_band_width))
            inner = gt_np & (~support_np)
            outer = support_np & (~gt_np) & near_gt
            cam, frame, image_name = _image_ids(view)
            rows.extend(_rows_for_direction(inner, "inner", cam, frame, image_name, xy, visible, radii, args))
            rows.extend(_rows_for_direction(outer, "outer", cam, frame, image_name, xy, visible, radii, args))
            del pkg

    fieldnames = [
        "cam", "frame", "image_name", "direction", "component_id", "area",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "centroid_x", "centroid_y",
        "near_point_count", "near_score_sum", "min_center_dist", "top_point_ids", "top_point_scores",
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "out_csv": str(args.out_csv),
        "rows": len(rows),
        "inner_rows": sum(1 for r in rows if r["direction"] == "inner"),
        "outer_rows": sum(1 for r in rows if r["direction"] == "outer"),
    }
    (args.out_dir / "component_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
