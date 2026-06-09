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
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.graphics_utils import geom_transform_points

RENDER_RE = re.compile(r"render_c(?P<cam>\d+)_f(?P<frame>\d+)\.png$")


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


def _view_ids(name: str) -> tuple[int, int]:
    text = str(name)
    return int(text.split("_f")[0].replace("c", "")), int(text.split("_f")[-1])


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _render_support(rgb: np.ndarray, threshold: float, close_kernel: int) -> np.ndarray:
    luma = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    support = (luma > threshold) | (chroma > threshold * 0.75)
    if close_kernel > 1:
        k = int(close_kernel)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        support = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return support


def _mask_from_view(view) -> np.ndarray:
    mask = getattr(view, "hard_mask", None)
    if not torch.is_tensor(mask):
        mask = getattr(view, "original_mask", None)
    if not torch.is_tensor(mask):
        raise RuntimeError(f"view {getattr(view, 'image_name', '<unknown>')} has no mask")
    mask = mask.detach().float()
    if mask.dim() == 3:
        mask = mask[:1]
    return (mask.squeeze().detach().cpu().numpy() > 0.5)


def _morph(mask: np.ndarray, width: int, mode: str) -> np.ndarray:
    width = max(1, int(width))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    fn = cv2.dilate if mode == "dilate" else cv2.erode
    return fn(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _boundary_band(mask: np.ndarray, width: int) -> np.ndarray:
    return _morph(mask, width, "dilate") & (~_morph(mask, width, "erode"))


def _components(mask: np.ndarray, min_area: int) -> list[dict]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    rows = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        rows.append({
            "label": label,
            "area": area,
            "cx": float(centroids[label][0]),
            "cy": float(centroids[label][1]),
        })
    rows.sort(key=lambda item: item["area"], reverse=True)
    return rows


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


def _ray_from_pixel(view, x: float, y: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = float(max(int(view.image_width) - 1, 1))
    height = float(max(int(view.image_height) - 1, 1))
    ndc_x = float(x) / width * 2.0 - 1.0
    ndc_y = (1.0 - float(y) / height) * 2.0 - 1.0
    inv_full = torch.inverse(view.full_proj_transform.detach().to(device=device, dtype=torch.float32))

    def unproject(z: float) -> torch.Tensor:
        p = torch.tensor([ndc_x, ndc_y, float(z), 1.0], device=device, dtype=torch.float32)
        w = torch.matmul(p, inv_full)
        return w[:3] / w[3].clamp_min(1.0e-8)

    near = unproject(0.10)
    far = unproject(0.90)
    cam = torch.as_tensor(view.camera_center, device=device, dtype=torch.float32)
    direction = F.normalize((far - near).reshape(1, 3), dim=-1).reshape(3)
    if torch.dot(direction, far - cam) < 0:
        direction = -direction
    return cam, direction


def _disk_counts(mask: np.ndarray, x: float, y: float, radius: float) -> int:
    h, w = mask.shape
    x0 = max(0, int(math.floor(x - radius)))
    x1 = min(w - 1, int(math.ceil(x + radius)))
    y0 = max(0, int(math.floor(y - radius)))
    y1 = min(h - 1, int(math.ceil(y + radius)))
    if x1 < x0 or y1 < y0:
        return 0
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    disk = (xx - float(x)) ** 2 + (yy - float(y)) ** 2 <= float(radius) ** 2
    return int((disk & mask[y0 : y1 + 1, x0 : x1 + 1]).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate visual-hull/ray-consistency 3D boundary support candidates for v269.")
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--load-ckpt", required=True, type=Path)
    parser.add_argument("--base-render-exp", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-summary", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--parser-root", default="")
    parser.add_argument("--compact-mapping", default="")
    parser.add_argument("--eval-views", default="21,22,23")
    parser.add_argument("--frames", default="0,60,120,180,240,300,360,420,480,540")
    parser.add_argument("--render-support-threshold", type=float, default=0.025)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--band-width", type=int, default=7)
    parser.add_argument("--search-band-width", type=int, default=24)
    parser.add_argument("--min-component-area", type=int, default=14)
    parser.add_argument("--max-components-per-view", type=int, default=5)
    parser.add_argument("--depth-samples", type=int, default=96)
    parser.add_argument("--depth-margin", type=float, default=0.08)
    parser.add_argument("--disk-radius", type=float, default=3.5)
    parser.add_argument("--max-output-candidates", type=int, default=48)
    parser.add_argument("--min-inner-pixels", type=float, default=8.0)
    parser.add_argument("--max-outer-pixels", type=float, default=3.0)
    parser.add_argument("--min-mask-views", type=int, default=2)
    parser.add_argument("--max-per-frame", type=int, default=8)
    args = parser.parse_args()

    eval_views = _parse_csv_ints(args.eval_views)
    frames = _parse_csv_ints(args.frames)

    config = OmegaConf.load(args.config_path)
    OmegaConf.set_struct(config, False)
    config.mode = "test"
    config.exp_dir = str(args.out_csv.parent / "scene_for_visual_hull")
    config.load_ckpt = str(args.load_ckpt)
    config.dataset.root_dir = str(args.dataset_root)
    config.dataset.preload = False
    config.dataset.test_views.view = eval_views
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

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    gaussians = GaussianModel(config.model.gaussian)
    scene = Scene(config, gaussians, config.exp_dir)
    scene.eval()
    loaded_iteration = int(scene.load_checkpoint(str(args.load_ckpt)))
    background = torch.tensor([1, 1, 1] if bool(config.dataset.white_background) else [0, 0, 0], dtype=torch.float32, device="cuda")

    view_lookup = {}
    records = {}
    render_dir = args.base_render_exp / "test-view" / "renders"
    if not render_dir.exists():
        raise FileNotFoundError(render_dir)
    for idx in range(len(scene.test_dataset)):
        view = scene.test_dataset[idx]
        cam, frame = _view_ids(view.image_name)
        if cam not in eval_views or frame not in frames:
            continue
        render_path = render_dir / f"render_c{cam:02d}_f{frame:06d}.png"
        if not render_path.exists():
            continue
        mask = _mask_from_view(view)
        support = _render_support(_read_rgb(render_path), args.render_support_threshold, args.close_kernel)
        near_gt = _morph(mask, args.search_band_width, "dilate")
        inner = mask & (~support)
        inner_band = inner & _boundary_band(mask, args.band_width)
        outer = support & (~mask) & near_gt
        records[(cam, frame)] = {
            "mask": mask,
            "support": support,
            "inner": inner,
            "inner_band": inner_band,
            "outer": outer,
            "components": _components(inner_band, args.min_component_area)[: int(args.max_components_per_view)],
        }
        view_lookup[(cam, frame)] = view

    frame_depth_ranges = {}
    for frame in frames:
        first_view = next((view_lookup.get((cam, frame)) for cam in eval_views if view_lookup.get((cam, frame)) is not None), None)
        if first_view is None:
            continue
        with torch.no_grad():
            pkg = render(first_view, loaded_iteration, scene, config.pipeline, background, compute_loss=False, return_opacity=False)
        xyz = pkg["deformed_gaussian"].get_xyz.detach().float()
        cam_center = torch.as_tensor(first_view.camera_center, device=xyz.device, dtype=torch.float32)
        dist = torch.norm(xyz - cam_center.reshape(1, 3), dim=-1)
        frame_depth_ranges[frame] = (
            float(torch.quantile(dist, 0.01).item()) - float(args.depth_margin),
            float(torch.quantile(dist, 0.99).item()) + float(args.depth_margin),
        )
        del pkg
        torch.cuda.empty_cache()

    scored = []
    for (birth_cam, frame), rec in records.items():
        birth_view = view_lookup.get((birth_cam, frame))
        if birth_view is None or frame not in frame_depth_ranges:
            continue
        near_depth, far_depth = frame_depth_ranges[frame]
        for comp in rec["components"]:
            cam_center, direction = _ray_from_pixel(birth_view, comp["cx"], comp["cy"], torch.device("cuda"))
            depths = torch.linspace(float(near_depth), float(far_depth), int(args.depth_samples), device="cuda")
            points = cam_center.reshape(1, 3) + depths.reshape(-1, 1) * direction.reshape(1, 3)
            for sample_idx in range(points.shape[0]):
                point = points[sample_idx : sample_idx + 1]
                row = {
                    "frame": int(frame),
                    "birth_cam": int(birth_cam),
                    "birth_x": float(comp["cx"]),
                    "birth_y": float(comp["cy"]),
                    "depth": float(depths[sample_idx].item()),
                    "xyz": [float(v) for v in point[0].detach().cpu().tolist()],
                    "source_component": f"{birth_cam}:{frame}:{comp['label']}",
                    "component_area": int(comp["area"]),
                    "valid_views": 0,
                    "mask_views": 0,
                    "inner_views": 0,
                    "outer_views": 0,
                    "inner_pixels": 0,
                    "inner_band_pixels": 0,
                    "outer_pixels": 0,
                    "score": 0.0,
                }
                for cam in eval_views:
                    view = view_lookup.get((cam, frame))
                    peer = records.get((cam, frame))
                    if view is None or peer is None:
                        continue
                    xy, valid = _project_points(point, view)
                    if not bool(valid[0].item()):
                        continue
                    row["valid_views"] += 1
                    x = float(xy[0, 0].item())
                    y = float(xy[0, 1].item())
                    xi = int(round(x))
                    yi = int(round(y))
                    h, w = peer["mask"].shape
                    if xi < 0 or xi >= w or yi < 0 or yi >= h:
                        continue
                    if bool(peer["mask"][yi, xi]):
                        row["mask_views"] += 1
                    inner_px = _disk_counts(peer["inner"], x, y, args.disk_radius)
                    inner_band_px = _disk_counts(peer["inner_band"], x, y, args.disk_radius)
                    outer_px = _disk_counts(peer["outer"], x, y, args.disk_radius)
                    row["inner_pixels"] += inner_px
                    row["inner_band_pixels"] += inner_band_px
                    row["outer_pixels"] += outer_px
                    row["inner_views"] += 1 if inner_px >= 2 else 0
                    row["outer_views"] += 1 if outer_px > 0 else 0
                row["score"] = (
                    float(row["inner_pixels"])
                    + 0.5 * float(row["inner_band_pixels"])
                    + 8.0 * float(row["inner_views"])
                    + 2.0 * float(row["mask_views"])
                    - 4.0 * float(row["outer_pixels"])
                    - 12.0 * float(row["outer_views"])
                )
                scored.append(row)

    eligible = [
        r for r in scored
        if r["mask_views"] >= int(args.min_mask_views)
        and r["inner_pixels"] >= float(args.min_inner_pixels)
        and r["outer_pixels"] <= float(args.max_outer_pixels)
        and r["score"] > 0
    ]
    eligible.sort(key=lambda r: (r["score"], r["inner_pixels"], -r["outer_pixels"]), reverse=True)

    selected = []
    used_components = set()
    per_frame = defaultdict(int)
    for row in eligible:
        if len(selected) >= int(args.max_output_candidates):
            break
        if per_frame[int(row["frame"])] >= int(args.max_per_frame):
            continue
        key = row["source_component"]
        if key in used_components:
            continue
        selected.append(row)
        used_components.add(key)
        per_frame[int(row["frame"])] += 1

    fieldnames = [
        "frame", "birth_cam", "birth_x", "birth_y", "depth", "xyz", "valid_views", "inner_views",
        "inner_band_views", "outer_views", "heldout_inner_views", "heldout_outer_views",
        "min_dist_to_inner", "min_dist_to_outer", "score", "source_candidate_index", "parent_idx",
        "actual_valid_views", "actual_visible_views", "actual_inner_pixels", "actual_inner_band_pixels",
        "actual_outer_pixels", "actual_inner_views", "actual_outer_views", "actual_mean_radius_px",
        "covered_components", "actual_outer_inner_ratio", "footprint_score", "footprint_inner_pixels",
        "footprint_outer_pixels", "source_component", "component_area", "mask_views",
    ]
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for idx, row in enumerate(selected):
            public = dict(row)
            public.update({
                "xyz": json.dumps(public["xyz"]),
                "inner_band_views": public["inner_views"],
                "heldout_inner_views": public["inner_views"],
                "heldout_outer_views": public["outer_views"],
                "min_dist_to_inner": 0.0,
                "min_dist_to_outer": 0.0,
                "source_candidate_index": idx,
                "parent_idx": -1,
                "actual_valid_views": public["valid_views"],
                "actual_visible_views": public["valid_views"],
                "actual_inner_pixels": public["inner_pixels"],
                "actual_inner_band_pixels": public["inner_band_pixels"],
                "actual_outer_pixels": public["outer_pixels"],
                "actual_inner_views": public["inner_views"],
                "actual_outer_views": public["outer_views"],
                "actual_mean_radius_px": float(args.disk_radius),
                "covered_components": json.dumps([public["source_component"]]),
                "actual_outer_inner_ratio": float(public["outer_pixels"]) / max(float(public["inner_pixels"]), 1.0),
                "footprint_score": public["score"],
                "footprint_inner_pixels": public["inner_pixels"],
                "footprint_outer_pixels": public["outer_pixels"],
            })
            writer.writerow(public)

    summary = {
        "status": "ok" if selected else "blocked",
        "reasons": [] if selected else ["no_visual_hull_inner_dominant_candidates"],
        "base_render_exp": str(args.base_render_exp),
        "loaded_iteration": loaded_iteration,
        "scored_candidate_count": len(scored),
        "eligible_candidate_count": len(eligible),
        "accepted_candidate_count": len(selected),
        "mean_inner_pixels": float(np.mean([r["inner_pixels"] for r in selected])) if selected else 0.0,
        "mean_outer_pixels": float(np.mean([r["outer_pixels"] for r in selected])) if selected else 0.0,
        "top_accepted": selected[:12],
    }
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
