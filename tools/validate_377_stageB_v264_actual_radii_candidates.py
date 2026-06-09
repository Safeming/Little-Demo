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
                if key in row and row[key] != "":
                    row[key] = float(row[key])
            rows.append(row)
    rows.sort(key=lambda r: (r.get("heldout_inner_views", 0.0), r.get("inner_views", 0.0), -r.get("outer_views", 0.0), r.get("score", 0.0)), reverse=True)
    if max_candidates > 0:
        rows = rows[:max_candidates]
    return rows


def _view_ids(name: str) -> tuple[int, int]:
    text = str(name)
    return int(text.split("_f")[0].replace("c", "")), int(text.split("_f")[-1])


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


def _disk_mask(shape: tuple[int, int], center_xy: tuple[float, float], radius: float) -> np.ndarray:
    h, w = shape
    x0 = max(0, int(math.floor(center_xy[0] - radius)))
    x1 = min(w - 1, int(math.ceil(center_xy[0] + radius)))
    y0 = max(0, int(math.floor(center_xy[1] - radius)))
    y1 = min(h - 1, int(math.ceil(center_xy[1] + radius)))
    out = np.zeros((h, w), dtype=bool)
    if x1 < x0 or y1 < y0:
        return out
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    out[y0 : y1 + 1, x0 : x1 + 1] = (xx - float(center_xy[0])) ** 2 + (yy - float(center_xy[1])) ** 2 <= float(radius) ** 2
    return out


def _components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, dict[int, int]]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = {}
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            areas[label] = area
        else:
            labels[labels == label] = 0
    return labels, areas


def main() -> int:
    parser = argparse.ArgumentParser(description="v264 actual-radii support validator using real append/deformer/rasterizer radii.")
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--load-ckpt", required=True, type=Path)
    parser.add_argument("--candidates-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--parser-root", default="")
    parser.add_argument("--compact-mapping", default="")
    parser.add_argument("--eval-views", default="21,22,23")
    parser.add_argument("--max-input-candidates", type=int, default=117)
    parser.add_argument("--max-output-candidates", type=int, default=48)
    parser.add_argument("--parent-screen-radius", type=float, default=42.0)
    parser.add_argument("--child-opacity-factor", type=float, default=0.80)
    parser.add_argument("--child-opacity-floor", type=float, default=0.040)
    parser.add_argument("--child-opacity-ceiling", type=float, default=0.32)
    parser.add_argument("--child-scale-factor", type=float, default=0.55)
    parser.add_argument("--radii-scale", type=float, default=1.0)
    parser.add_argument("--min-radius-px", type=float, default=1.0)
    parser.add_argument("--max-radius-px", type=float, default=12.0)
    parser.add_argument("--render-support-threshold", type=float, default=0.025)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--band-width", type=int, default=7)
    parser.add_argument("--search-band-width", type=int, default=24)
    parser.add_argument("--min-component-area", type=int, default=18)
    parser.add_argument("--min-actual-inner-pixels", type=float, default=4.0)
    parser.add_argument("--max-actual-outer-pixels", type=float, default=3.0)
    parser.add_argument("--max-actual-outer-inner-ratio", type=float, default=0.25)
    parser.add_argument("--max-per-frame", type=int, default=8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = _read_candidates(args.candidates_csv, args.max_input_candidates)
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

    view_lookup = {}
    base_records = {}
    for idx in range(len(scene.test_dataset)):
        view = scene.test_dataset[idx]
        cam, frame = _view_ids(view.image_name)
        if cam not in all_views or frame not in frames:
            continue
        view_lookup[(cam, frame)] = view
        with torch.no_grad():
            pkg = render(view, loaded_iteration, scene, config.pipeline, background, compute_loss=False, return_opacity=False)
        gt = _mask_from_view(view)
        support = _render_support_mask(pkg["render"].detach().clamp(0.0, 1.0), args.render_support_threshold, args.close_kernel)
        near_gt = _dilate(gt, args.search_band_width)
        inner = ((gt > 0.5) & (support <= 0.5))[0].detach().cpu().numpy().astype(bool)
        outer = ((support > 0.5) & (gt <= 0.5) & (near_gt > 0.5))[0].detach().cpu().numpy().astype(bool)
        inner_band = (((gt > 0.5) & (support <= 0.5)) & (_boundary_band(gt, args.band_width) > 0.5))[0].detach().cpu().numpy().astype(bool)
        labels, areas = _components(inner_band, args.min_component_area)
        base_records[(cam, frame)] = {"inner": inner, "outer": outer, "inner_band": inner_band, "labels": labels, "areas": areas}
        del pkg
        torch.cuda.empty_cache()

    base_count = int(scene.gaussians.get_xyz.shape[0])
    candidates_by_key = defaultdict(list)
    for cand_idx, cand in enumerate(candidates):
        candidates_by_key[(int(cand["birth_cam"]), int(cand["frame"]))].append((cand_idx, cand))

    parent_indices = []
    canonical_xyz = []
    append_rows = []
    for key, key_candidates in candidates_by_key.items():
        view = view_lookup.get(key)
        if view is None:
            continue
        with torch.no_grad():
            pkg = render(view, loaded_iteration, scene, config.pipeline, background, compute_loss=False, return_opacity=False)
        deformed = pkg["deformed_gaussian"]
        xy, proj_valid = _project_points(deformed.get_xyz[:base_count], view)
        visible = proj_valid & pkg["visibility_filter"][:base_count].detach().bool() & (pkg["radii"][:base_count].detach() > 0)
        visible_idx = torch.nonzero(visible, as_tuple=False).reshape(-1)
        visible_xy = xy[visible].detach()
        fwd_transform = getattr(deformed, "fwd_transform", None)
        for cand_idx, cand in key_candidates:
            target_xy = torch.tensor([cand["birth_x"], cand["birth_y"]], dtype=torch.float32, device="cuda")
            candidate_deformed = torch.tensor(cand["xyz"], dtype=torch.float32, device="cuda")
            if visible_idx.numel() > 0:
                screen_dist = torch.norm(visible_xy - target_xy.reshape(1, 2), dim=-1)
                within = screen_dist <= float(args.parent_screen_radius)
                if bool(within.any().item()):
                    local_idx = visible_idx[within]
                    local_deformed = deformed.get_xyz[local_idx].detach().float()
                    local_screen = screen_dist[within]
                    local_3d = torch.norm(local_deformed - candidate_deformed.reshape(1, 3), dim=-1)
                    pick_local = int(torch.argmin(local_screen / max(args.parent_screen_radius, 1.0) + 8.0 * local_3d).item())
                    parent_idx = int(local_idx[pick_local].item())
                else:
                    parent_idx = int(visible_idx[int(torch.argmin(screen_dist).item())].item())
            else:
                dist3d = torch.norm(deformed.get_xyz[:base_count].detach().float() - candidate_deformed.reshape(1, 3), dim=-1)
                parent_idx = int(torch.argmin(dist3d).item())
            parent_deformed = deformed.get_xyz[parent_idx].detach().float()
            parent_canonical = scene.gaussians.get_xyz[parent_idx].detach().float()
            if torch.is_tensor(fwd_transform) and fwd_transform.ndim == 3 and fwd_transform.shape[0] > parent_idx:
                rot = fwd_transform[parent_idx, :3, :3].detach().float()
                delta_deformed = candidate_deformed - parent_deformed
                try:
                    delta_canonical = torch.linalg.solve(rot, delta_deformed.reshape(3, 1)).reshape(3)
                except RuntimeError:
                    delta_canonical = torch.matmul(torch.linalg.pinv(rot), delta_deformed.reshape(3, 1)).reshape(3)
                new_xyz = parent_canonical + delta_canonical
            else:
                new_xyz = parent_canonical + (candidate_deformed - parent_deformed)
            parent_indices.append(parent_idx)
            canonical_xyz.append(new_xyz.detach())
            append_rows.append((cand_idx, cand, parent_idx))
        del pkg, deformed
        torch.cuda.empty_cache()

    if not parent_indices:
        raise RuntimeError("no candidates could be assigned parents")

    parent_idx = torch.tensor(parent_indices, dtype=torch.long, device="cuda")
    new_xyz = torch.stack(canonical_xyz, dim=0).to(device="cuda", dtype=scene.gaussians._xyz.dtype)
    new_features_dc = scene.gaussians._features_dc[parent_idx].detach().clone()
    new_features_rest = scene.gaussians._features_rest[parent_idx].detach().clone()
    child_opacity = (scene.gaussians.get_opacity[parent_idx].detach() * float(args.child_opacity_factor)).clamp(
        min=max(float(args.child_opacity_floor), 1.0e-4),
        max=min(float(args.child_opacity_ceiling), 1.0 - 1.0e-4),
    )
    new_opacity = scene.gaussians.inverse_opacity_activation(child_opacity)
    child_scaling = (scene.gaussians.get_scaling[parent_idx].detach() * float(args.child_scale_factor)).clamp_min(1.0e-6)
    new_scaling = scene.gaussians.scaling_inverse_activation(child_scaling)
    new_rotation = scene.gaussians._rotation[parent_idx].detach().clone()
    new_boundary_opacity_residual = scene.gaussians._boundary_opacity_residual[parent_idx].detach().clone()
    new_boundary_scaling_residual = scene.gaussians._boundary_scaling_residual[parent_idx].detach().clone()
    new_binding_state = None
    if scene.gaussians.has_binding_state():
        new_binding_state = {}
        for key, value in scene.gaussians.binding_state.items():
            if torch.is_tensor(value) and value.shape[0] == base_count:
                new_binding_state[key] = value[parent_idx].detach().clone()
        new_binding_state = scene.gaussians._clear_newborn_binding_flags(new_binding_state)
        new_binding_state = scene.gaussians._annotate_densified_binding_lineage(new_binding_state, parent_idx, iteration=135711)
        new_binding_state = scene.gaussians._update_binding_offsets(new_binding_state, new_xyz - scene.gaussians.get_xyz[parent_idx].detach())
        count = int(parent_idx.shape[0])
        new_binding_state["boundary_support_role"] = torch.ones((count,), dtype=torch.long, device="cuda")
        new_binding_state["boundary_support_anchor_index"] = parent_idx.clone()
        new_binding_state["boundary_support_birth_iter"] = torch.full((count,), 135711, dtype=torch.long, device="cuda")
        new_binding_state["boundary_support_confidence"] = torch.ones((count,), dtype=torch.float32, device="cuda")
        new_binding_state["boundary_support_view_id"] = torch.tensor([int(c["birth_cam"]) for _, c, _ in append_rows], dtype=torch.long, device="cuda")
        new_binding_state["boundary_support_frame_id"] = torch.tensor([int(c["frame"]) for _, c, _ in append_rows], dtype=torch.long, device="cuda")

    scene.gaussians.densification_postfix(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacity,
        new_scaling,
        new_rotation,
        new_binding_state=new_binding_state,
        new_boundary_tags=torch.ones((new_xyz.shape[0],), dtype=torch.float32, device="cuda"),
        new_boundary_opacity_residual=new_boundary_opacity_residual,
        new_boundary_scaling_residual=new_boundary_scaling_residual,
        new_live_boundary_score=torch.ones((new_xyz.shape[0],), dtype=torch.float32, device="cuda"),
    )

    scores = []
    for appended_local, (source_idx, cand, parent_i) in enumerate(append_rows):
        point_idx = base_count + appended_local
        row = {
            **cand,
            "source_candidate_index": source_idx,
            "parent_idx": parent_i,
            "actual_valid_views": 0,
            "actual_visible_views": 0,
            "actual_inner_pixels": 0,
            "actual_inner_band_pixels": 0,
            "actual_outer_pixels": 0,
            "actual_inner_views": 0,
            "actual_outer_views": 0,
            "actual_mean_radius_px": 0.0,
            "covered_components": [],
        }
        radii_seen = []
        component_hits = []
        for cam in eval_views:
            view = view_lookup.get((cam, int(cand["frame"])))
            rec = base_records.get((cam, int(cand["frame"])))
            if view is None or rec is None:
                continue
            with torch.no_grad():
                pkg = render(view, loaded_iteration, scene, config.pipeline, background, compute_loss=False, return_opacity=False)
            deformed = pkg["deformed_gaussian"]
            xy, valid = _project_points(deformed.get_xyz[point_idx : point_idx + 1], view)
            visible = bool(pkg["visibility_filter"][point_idx].detach().bool().item()) and float(pkg["radii"][point_idx].detach().item()) > 0.0
            if bool(valid[0].item()):
                row["actual_valid_views"] += 1
            if bool(valid[0].item()) and visible:
                row["actual_visible_views"] += 1
                radius = float(pkg["radii"][point_idx].detach().item()) * float(args.radii_scale)
                radius = float(np.clip(radius, float(args.min_radius_px), float(args.max_radius_px)))
                radii_seen.append(radius)
                disk = _disk_mask(rec["inner"].shape, (float(xy[0, 0].item()), float(xy[0, 1].item())), radius)
                inner = int((disk & rec["inner"]).sum())
                inner_band = int((disk & rec["inner_band"]).sum())
                outer = int((disk & rec["outer"]).sum())
                row["actual_inner_pixels"] += inner
                row["actual_inner_band_pixels"] += inner_band
                row["actual_outer_pixels"] += outer
                row["actual_inner_views"] += 1 if inner >= float(args.min_actual_inner_pixels) else 0
                row["actual_outer_views"] += 1 if outer > 0 else 0
                labels = rec["labels"][disk & rec["inner_band"]]
                for label in sorted(set(int(v) for v in labels.tolist() if int(v) > 0)):
                    component_hits.append(f"{cam}:{int(cand['frame'])}:{label}")
            del pkg
            torch.cuda.empty_cache()
        row["actual_mean_radius_px"] = float(np.mean(radii_seen)) if radii_seen else 0.0
        row["actual_outer_inner_ratio"] = float(row["actual_outer_pixels"]) / max(float(row["actual_inner_pixels"]), 1.0)
        row["covered_components"] = sorted(set(component_hits))
        row["footprint_score"] = (
            float(row["actual_inner_pixels"])
            + 0.5 * float(row["actual_inner_band_pixels"])
            + 5.0 * float(row["actual_inner_views"])
            - 2.5 * float(row["actual_outer_pixels"])
            - 8.0 * float(row["actual_outer_views"])
        )
        row["footprint_inner_pixels"] = row["actual_inner_pixels"]
        row["footprint_outer_pixels"] = row["actual_outer_pixels"]
        scores.append(row)

    eligible = [
        r for r in scores
        if r["actual_inner_pixels"] >= float(args.min_actual_inner_pixels)
        and r["actual_outer_pixels"] <= float(args.max_actual_outer_pixels)
        and r["actual_outer_inner_ratio"] <= float(args.max_actual_outer_inner_ratio)
        and r["actual_visible_views"] > 0
    ]
    selected = []
    covered = set()
    per_frame = defaultdict(int)
    remaining = list(eligible)
    while remaining and (int(args.max_output_candidates) <= 0 or len(selected) < int(args.max_output_candidates)):
        best = None
        best_gain = None
        for row in remaining:
            if per_frame[int(row["frame"])] >= int(args.max_per_frame):
                continue
            new_components = [c for c in row["covered_components"] if c not in covered]
            gain = (
                12.0 * len(new_components)
                + float(row["actual_inner_pixels"])
                + 0.5 * float(row["actual_inner_band_pixels"])
                - 3.0 * float(row["actual_outer_pixels"])
            )
            if best is None or gain > best_gain:
                best = row
                best_gain = gain
        if best is None or best_gain is None or best_gain <= 0.0:
            break
        selected.append(best)
        covered.update(best["covered_components"])
        per_frame[int(best["frame"])] += 1
        remaining = [r for r in remaining if r is not best]

    selected.sort(key=lambda r: (r["footprint_score"], r["actual_inner_pixels"], -r["actual_outer_pixels"]), reverse=True)
    scores.sort(key=lambda r: (r["footprint_score"], r["actual_inner_pixels"], -r["actual_outer_pixels"]), reverse=True)
    summary = {
        "status": "ok" if selected else "blocked",
        "reasons": [] if selected else ["no_actual_radii_inner_dominant_candidates"],
        "loaded_iteration": loaded_iteration,
        "input_candidate_count": len(candidates),
        "scored_candidate_count": len(scores),
        "eligible_candidate_count": len(eligible),
        "accepted_candidate_count": len(selected),
        "covered_component_count": len(covered),
        "mean_actual_inner_pixels": float(np.mean([r["actual_inner_pixels"] for r in selected])) if selected else 0.0,
        "mean_actual_outer_pixels": float(np.mean([r["actual_outer_pixels"] for r in selected])) if selected else 0.0,
        "mean_actual_outer_inner_ratio": float(np.mean([r["actual_outer_inner_ratio"] for r in selected])) if selected else 0.0,
        "top_accepted": selected[:12],
        "top_scored": scores[:12],
    }
    (args.out_dir / "actual_radii_candidate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out_dir / "actual_radii_accepted_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        if selected:
            public = [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items()} for row in selected]
            writer = csv.DictWriter(handle, fieldnames=list(public[0].keys()))
            writer.writeheader()
            writer.writerows(public)
    with (args.out_dir / "actual_radii_scored_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        if scores:
            public = [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items()} for row in scores]
            writer = csv.DictWriter(handle, fieldnames=list(public[0].keys()))
            writer.writeheader()
            writer.writerows(public)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
