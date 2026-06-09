#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
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
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.graphics_utils import geom_transform_points


def _parse_csv_ints(value: str) -> list[int]:
    out = []
    for part in str(value).replace("[", "").replace("]", "").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _range_spec_from_values(values: list[int]) -> list[int]:
    values = sorted(set(int(v) for v in values))
    if len(values) <= 1:
        return [values[0], values[0] + 1, 1] if values else [0, 1, 1]
    step = values[1] - values[0]
    if step > 0 and all(values[i + 1] - values[i] == step for i in range(len(values) - 1)):
        return [values[0], values[-1] + step, step]
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1) if values[i + 1] > values[i]]
    gcd_step = diffs[0]
    for diff in diffs[1:]:
        gcd_step = math.gcd(gcd_step, diff)
    return [values[0], values[-1] + max(1, int(gcd_step)), max(1, int(gcd_step))]


def _read_candidates(path: Path, max_candidates: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["frame"] = int(float(row["frame"]))
            row["birth_cam"] = int(float(row["birth_cam"]))
            row["birth_x"] = float(row["birth_x"])
            row["birth_y"] = float(row["birth_y"])
            row["depth"] = float(row.get("depth", 0.0))
            row["score"] = float(row.get("score", 0.0))
            row["xyz"] = [float(v) for v in ast.literal_eval(row["xyz"])]
            for key in ("inner_views", "inner_band_views", "outer_views", "heldout_inner_views", "heldout_outer_views"):
                if key in row:
                    row[key] = float(row[key])
            rows.append(row)
    rows.sort(key=lambda item: (item.get("heldout_inner_views", 0.0), item.get("inner_views", 0.0), -item.get("outer_views", 0.0), item.get("score", 0.0)), reverse=True)
    if max_candidates > 0:
        rows = rows[:max_candidates]
    return rows


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
    ndc = geom_transform_points(points.detach(), view.full_proj_transform.detach().to(points.device))
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


def _disk_counts(mask: np.ndarray, center_xy: tuple[float, float], radius: float) -> int:
    if radius <= 0.0:
        x = int(round(center_xy[0]))
        y = int(round(center_xy[1]))
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            return int(mask[y, x])
        return 0
    x0 = max(0, int(math.floor(center_xy[0] - radius)))
    x1 = min(mask.shape[1] - 1, int(math.ceil(center_xy[0] + radius)))
    y0 = max(0, int(math.floor(center_xy[1] - radius)))
    y1 = min(mask.shape[0] - 1, int(math.ceil(center_xy[1] + radius)))
    if x1 < x0 or y1 < y0:
        return 0
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    disk = (xx - float(center_xy[0])) ** 2 + (yy - float(center_xy[1])) ** 2 <= float(radius) ** 2
    return int((mask[y0 : y1 + 1, x0 : x1 + 1] & disk).sum())


def _estimate_radius_px(point: torch.Tensor, scale: float, view, radius_multiplier: float, min_radius: float, max_radius: float) -> float:
    scale = max(float(scale), 1.0e-6)
    point = point.reshape(1, 3)
    axes = torch.eye(3, dtype=point.dtype, device=point.device) * scale
    probes = torch.cat([point, point + axes, point - axes], dim=0)
    xy, valid = _project_points(probes, view)
    if not bool(valid[0].item()):
        return 0.0
    deltas = torch.norm(xy[1:] - xy[:1], dim=-1)
    deltas = deltas[valid[1:]]
    if deltas.numel() == 0:
        radius = min_radius
    else:
        radius = float(deltas.max().item()) * float(radius_multiplier)
    return float(np.clip(radius, float(min_radius), float(max_radius)))


def _view_ids(name: str) -> tuple[int, int]:
    text = str(name)
    cam = int(text.split("_f")[0].replace("c", ""))
    frame = int(text.split("_f")[-1])
    return cam, frame


def main() -> int:
    parser = argparse.ArgumentParser(description="v263 footprint-aware support candidate validator.")
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--load-ckpt", required=True, type=Path)
    parser.add_argument("--candidates-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--parser-root", default="")
    parser.add_argument("--compact-mapping", default="")
    parser.add_argument("--eval-views", default="21,22,23")
    parser.add_argument("--max-input-candidates", type=int, default=0)
    parser.add_argument("--max-output-candidates", type=int, default=64)
    parser.add_argument("--parent-screen-radius", type=float, default=42.0)
    parser.add_argument("--child-scale-factor", type=float, default=0.55)
    parser.add_argument("--radius-multiplier", type=float, default=1.60)
    parser.add_argument("--min-radius-px", type=float, default=2.0)
    parser.add_argument("--max-radius-px", type=float, default=18.0)
    parser.add_argument("--render-support-threshold", type=float, default=0.025)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--band-width", type=int, default=7)
    parser.add_argument("--search-band-width", type=int, default=24)
    parser.add_argument("--min-footprint-inner-pixels", type=float, default=2.0)
    parser.add_argument("--max-footprint-outer-inner-ratio", type=float, default=0.50)
    parser.add_argument("--max-footprint-outer-pixels", type=float, default=6.0)
    parser.add_argument("--max-footprint-mean-radius-px", type=float, default=0.0)
    parser.add_argument("--max-child-scale", type=float, default=0.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = _read_candidates(args.candidates_csv, args.max_input_candidates)
    if not candidates:
        raise RuntimeError(f"no candidates in {args.candidates_csv}")

    eval_views = _parse_csv_ints(args.eval_views)
    birth_views = sorted({int(c["birth_cam"]) for c in candidates})
    frames = sorted({int(c["frame"]) for c in candidates})
    all_views = sorted(set(eval_views + birth_views))

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

    gaussians = GaussianModel(config.model.gaussian)
    scene = Scene(config, gaussians, config.exp_dir)
    scene.eval()
    loaded_iteration = int(scene.load_checkpoint(str(args.load_ckpt)))
    background = torch.tensor([1, 1, 1] if bool(config.dataset.white_background) else [0, 0, 0], dtype=torch.float32, device="cuda")

    base_count = int(scene.gaussians.get_xyz.shape[0])
    base_canonical = scene.gaussians.get_xyz[:base_count].detach().float().cpu()
    base_scaling = scene.gaussians.get_scaling[:base_count].detach().float().cpu()

    records = {}
    for idx in range(len(scene.test_dataset)):
        view = scene.test_dataset[idx]
        cam, frame = _view_ids(view.image_name)
        if cam not in all_views or frame not in frames:
            continue
        with torch.no_grad():
            pkg = render(view, loaded_iteration, scene, config.pipeline, background, compute_loss=False, return_opacity=False)
        rendering = pkg["render"].detach().clamp(0.0, 1.0)
        gt_mask = _mask_from_view(view)
        support = _render_support_mask(rendering, args.render_support_threshold, args.close_kernel)
        near_gt = _dilate(gt_mask, args.search_band_width)
        inner = ((gt_mask > 0.5) & (support <= 0.5))[0].detach().cpu().numpy().astype(bool)
        outer = ((support > 0.5) & (gt_mask <= 0.5) & (near_gt > 0.5))[0].detach().cpu().numpy().astype(bool)
        inner_band = (((gt_mask > 0.5) & (support <= 0.5)) & (_boundary_band(gt_mask, args.band_width) > 0.5))[0].detach().cpu().numpy().astype(bool)
        deformed = pkg["deformed_gaussian"]
        xy, proj_valid = _project_points(deformed.get_xyz[:base_count], view)
        visible = (proj_valid & pkg["visibility_filter"][:base_count].detach().bool() & (pkg["radii"][:base_count].detach() > 0)).detach().cpu()
        fwd_transform = getattr(deformed, "fwd_transform", None)
        records[(cam, frame)] = {
            "view": view,
            "deformed_xyz": deformed.get_xyz[:base_count].detach().float().cpu(),
            "fwd_transform": fwd_transform[:base_count].detach().float().cpu() if torch.is_tensor(fwd_transform) else None,
            "xy": xy.detach().float().cpu(),
            "visible": visible,
            "inner": inner,
            "outer": outer,
            "inner_band": inner_band,
        }
        del pkg, deformed, rendering, gt_mask, support
        torch.cuda.empty_cache()

    by_key = defaultdict(list)
    for idx, cand in enumerate(candidates):
        by_key[(int(cand["birth_cam"]), int(cand["frame"]))].append((idx, cand))

    enriched = []
    for key, key_candidates in by_key.items():
        rec = records.get(key)
        if rec is None:
            continue
        visible_idx = torch.nonzero(rec["visible"], as_tuple=False).reshape(-1)
        visible_xy = rec["xy"][visible_idx] if visible_idx.numel() else torch.empty((0, 2))
        for cand_idx, cand in key_candidates:
            target_xy = torch.tensor([cand["birth_x"], cand["birth_y"]], dtype=torch.float32)
            candidate_deformed = torch.tensor(cand["xyz"], dtype=torch.float32)
            if visible_idx.numel() > 0:
                screen_dist = torch.norm(visible_xy - target_xy.reshape(1, 2), dim=-1)
                within = screen_dist <= float(args.parent_screen_radius)
                if bool(within.any().item()):
                    local_idx = visible_idx[within]
                    local_deformed = rec["deformed_xyz"][local_idx]
                    local_screen = screen_dist[within]
                    local_3d = torch.norm(local_deformed - candidate_deformed.reshape(1, 3), dim=-1)
                    pick_local = int(torch.argmin(local_screen / max(args.parent_screen_radius, 1.0) + 8.0 * local_3d).item())
                    parent_idx = int(local_idx[pick_local].item())
                else:
                    parent_idx = int(visible_idx[int(torch.argmin(screen_dist).item())].item())
            else:
                dist3d = torch.norm(rec["deformed_xyz"] - candidate_deformed.reshape(1, 3), dim=-1)
                parent_idx = int(torch.argmin(dist3d).item())

            parent_deformed = rec["deformed_xyz"][parent_idx]
            parent_canonical = base_canonical[parent_idx]
            fwd = rec["fwd_transform"]
            if torch.is_tensor(fwd):
                rot = fwd[parent_idx, :3, :3]
                delta_deformed = candidate_deformed - parent_deformed
                try:
                    delta_canonical = torch.linalg.solve(rot, delta_deformed.reshape(3, 1)).reshape(3)
                except RuntimeError:
                    delta_canonical = torch.matmul(torch.linalg.pinv(rot), delta_deformed.reshape(3, 1)).reshape(3)
            else:
                delta_canonical = candidate_deformed - parent_deformed
            canonical_xyz = parent_canonical + delta_canonical
            child_scale = float(base_scaling[parent_idx].amax().item()) * float(args.child_scale_factor)

            footprint_inner = 0
            footprint_outer = 0
            footprint_inner_band = 0
            footprint_valid_views = 0
            footprint_inner_views = 0
            footprint_outer_views = 0
            radii = []
            for eval_cam in eval_views:
                eval_rec = records.get((eval_cam, int(cand["frame"])))
                if eval_rec is None:
                    continue
                parent_eval = eval_rec["deformed_xyz"][parent_idx]
                eval_fwd = eval_rec["fwd_transform"]
                if torch.is_tensor(eval_fwd):
                    child_eval = parent_eval + torch.matmul(eval_fwd[parent_idx, :3, :3], delta_canonical.reshape(3, 1)).reshape(3)
                else:
                    child_eval = parent_eval + (canonical_xyz - parent_canonical)
                point_cuda = child_eval.cuda().reshape(1, 3)
                xy, valid = _project_points(point_cuda, eval_rec["view"])
                if not bool(valid[0].item()):
                    continue
                radius = _estimate_radius_px(
                    point_cuda[0],
                    child_scale,
                    eval_rec["view"],
                    args.radius_multiplier,
                    args.min_radius_px,
                    args.max_radius_px,
                )
                center = (float(xy[0, 0].item()), float(xy[0, 1].item()))
                inner_count = _disk_counts(eval_rec["inner"], center, radius)
                outer_count = _disk_counts(eval_rec["outer"], center, radius)
                inner_band_count = _disk_counts(eval_rec["inner_band"], center, radius)
                footprint_inner += inner_count
                footprint_outer += outer_count
                footprint_inner_band += inner_band_count
                footprint_valid_views += 1
                footprint_inner_views += 1 if inner_count >= float(args.min_footprint_inner_pixels) else 0
                footprint_outer_views += 1 if outer_count > 0 else 0
                radii.append(radius)

            ratio = float(footprint_outer) / max(float(footprint_inner), 1.0)
            footprint_score = (
                float(footprint_inner)
                + 0.5 * float(footprint_inner_band)
                + 4.0 * float(footprint_inner_views)
                - 2.0 * float(footprint_outer)
                - 6.0 * float(footprint_outer_views)
            )
            row = {
                **cand,
                "source_candidate_index": cand_idx,
                "parent_idx": parent_idx,
                "canonical_xyz": [float(v) for v in canonical_xyz.tolist()],
                "child_scale": child_scale,
                "footprint_valid_views": footprint_valid_views,
                "footprint_inner_pixels": footprint_inner,
                "footprint_inner_band_pixels": footprint_inner_band,
                "footprint_outer_pixels": footprint_outer,
                "footprint_inner_views": footprint_inner_views,
                "footprint_outer_views": footprint_outer_views,
                "footprint_outer_inner_ratio": ratio,
                "footprint_mean_radius_px": float(np.mean(radii)) if radii else 0.0,
                "footprint_score": footprint_score,
            }
            enriched.append(row)

    enriched.sort(key=lambda r: (r["footprint_score"], r["footprint_inner_pixels"], -r["footprint_outer_pixels"]), reverse=True)
    accepted = []
    for row in enriched:
        if row["footprint_valid_views"] <= 0:
            continue
        if row["footprint_inner_pixels"] < float(args.min_footprint_inner_pixels):
            continue
        if row["footprint_outer_pixels"] > float(args.max_footprint_outer_pixels):
            continue
        if row["footprint_outer_inner_ratio"] > float(args.max_footprint_outer_inner_ratio):
            continue
        if float(args.max_footprint_mean_radius_px) > 0.0 and row["footprint_mean_radius_px"] > float(args.max_footprint_mean_radius_px):
            continue
        if float(args.max_child_scale) > 0.0 and row["child_scale"] > float(args.max_child_scale):
            continue
        accepted.append(row)
        if int(args.max_output_candidates) > 0 and len(accepted) >= int(args.max_output_candidates):
            break

    status = "ok" if accepted else "blocked"
    summary = {
        "status": status,
        "reasons": [] if accepted else ["no_footprint_inner_dominant_candidates"],
        "config_path": str(args.config_path),
        "load_ckpt": str(args.load_ckpt),
        "loaded_iteration": loaded_iteration,
        "input_candidate_count": len(candidates),
        "scored_candidate_count": len(enriched),
        "accepted_candidate_count": len(accepted),
        "eval_views": eval_views,
        "mean_footprint_inner_pixels": float(np.mean([r["footprint_inner_pixels"] for r in accepted])) if accepted else 0.0,
        "mean_footprint_outer_pixels": float(np.mean([r["footprint_outer_pixels"] for r in accepted])) if accepted else 0.0,
        "mean_footprint_outer_inner_ratio": float(np.mean([r["footprint_outer_inner_ratio"] for r in accepted])) if accepted else 0.0,
        "top_accepted": accepted[:12],
        "top_scored": enriched[:12],
    }
    (args.out_dir / "footprint_candidate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out_dir / "footprint_accepted_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        if accepted:
            writer = csv.DictWriter(handle, fieldnames=list(accepted[0].keys()))
            writer.writeheader()
            writer.writerows(accepted)
    with (args.out_dir / "footprint_scored_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        if enriched:
            writer = csv.DictWriter(handle, fieldnames=list(enriched[0].keys()))
            writer.writeheader()
            writer.writerows(enriched)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
