#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.graphics_utils import geom_transform_points


FIELDNAMES = [
    "cam",
    "frame",
    "image_name",
    "direction",
    "component_id",
    "area",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "centroid_x",
    "centroid_y",
    "near_point_count",
    "near_score_sum",
    "min_center_dist",
    "top_point_ids",
    "top_point_scores",
]


def _parse_ints(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value or "").replace("[", "").replace("]", "").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _range_or_values(spec: str) -> list[int]:
    values = _parse_ints(spec)
    if len(values) == 3:
        return list(range(values[0], values[1], values[2]))
    return values


def _triple_or_values(spec: str) -> list[int]:
    return _parse_ints(spec)


def _parse_point_ids(value: str) -> list[int]:
    ids: list[int] = []
    for part in str(value or "").replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            continue
    return ids


def _parse_point_scores(value: str, count: int) -> list[float]:
    scores: list[float] = []
    for part in str(value or "").replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            scores.append(float(part))
        except ValueError:
            scores.append(1.0)
    if len(scores) < count:
        scores.extend([1.0] * (count - len(scores)))
    return scores[:count]


def _row_cam(row: dict) -> int | None:
    try:
        return int(float(row.get("cam", "")))
    except Exception:
        image_name = str(row.get("image_name", ""))
        if image_name.startswith("c") and "_f" in image_name:
            try:
                return int(image_name.split("_f", 1)[0][1:])
            except Exception:
                return None
    return None


def _row_frame(row: dict) -> int | None:
    try:
        return int(float(row.get("frame", "")))
    except Exception:
        image_name = str(row.get("image_name", ""))
        if "_f" in image_name:
            try:
                return int(image_name.rsplit("_f", 1)[1])
            except Exception:
                return None
    return None


def _load_source_components(path: Path, source_views: set[int], target_frames: set[int], min_area: float) -> dict[tuple[int, str], list[dict]]:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cam = _row_cam(row)
            frame = _row_frame(row)
            direction = str(row.get("direction", "")).strip().lower()
            if cam is None or frame is None or direction not in ("outer", "inner"):
                continue
            if source_views and cam not in source_views:
                continue
            if target_frames and frame not in target_frames:
                continue
            try:
                area = float(row.get("area", 0.0) or 0.0)
                near = float(row.get("near_score_sum", 0.0) or 0.0)
            except ValueError:
                continue
            point_ids = _parse_point_ids(row.get("top_point_ids", ""))
            if area < float(min_area) or not point_ids:
                continue
            point_scores = _parse_point_scores(row.get("top_point_scores", ""), len(point_ids))
            item = dict(row)
            item["_cam"] = cam
            item["_frame"] = frame
            item["_direction"] = direction
            item["_area"] = area
            item["_near"] = near
            item["_point_ids"] = point_ids
            item["_point_scores"] = point_scores
            item["_rank_score"] = math.log1p(max(near, 0.0)) * math.sqrt(max(area, 1.0))
            grouped[(frame, direction)].append(item)
    for key in list(grouped.keys()):
        grouped[key].sort(key=lambda row: row["_rank_score"], reverse=True)
    return grouped


def _select_rows(rows: list[dict], max_components: int, jaccard_threshold: float) -> list[dict]:
    selected: list[dict] = []
    selected_sets: list[set[int]] = []
    for row in rows:
        ids = set(int(x) for x in row["_point_ids"])
        if not ids:
            continue
        duplicate = False
        for prev in selected_sets:
            inter = len(ids & prev)
            union = len(ids | prev)
            if union > 0 and inter / union >= float(jaccard_threshold):
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(row)
        selected_sets.append(ids)
        if len(selected) >= int(max_components):
            break
    return selected


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


def _build_config(args: argparse.Namespace):
    config = OmegaConf.load(args.config_path)
    OmegaConf.set_struct(config, False)
    config.mode = "test"
    config.exp_dir = str(args.out_dir / "predictor_scene")
    config.load_ckpt = str(args.load_ckpt)
    config.dataset.root_dir = str(args.dataset_root)
    config.dataset.preload = False
    config.dataset.train_views = _parse_ints(args.train_views)
    config.dataset.train_frames = _triple_or_values(args.train_frames)
    config.dataset.test_views.view = _parse_ints(args.target_views)
    config.dataset.test_frames.view = _triple_or_values(args.target_frames)
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
    config.resume.disable_densify_on_resume = True
    config.resume.disable_opacity_reset_on_resume = True
    return config


def _make_prediction_row(
    *,
    source_row: dict,
    target_cam: int,
    target_frame: int,
    image_name: str,
    component_id: int,
    xy: torch.Tensor,
    valid: torch.Tensor,
    radii: torch.Tensor,
    point_count: int,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> dict | None:
    ids = [idx for idx in source_row["_point_ids"] if 0 <= int(idx) < int(point_count)]
    if not ids:
        return None
    idx_tensor = torch.tensor(ids, dtype=torch.long, device=xy.device)
    visible = valid[idx_tensor].detach().bool()
    if int(visible.sum().item()) < int(args.min_visible_points):
        return None
    idx_tensor = idx_tensor[visible]
    ids = [int(x) for x in idx_tensor.detach().cpu().tolist()]
    if not ids:
        return None
    scores = _parse_point_scores(source_row.get("top_point_scores", ""), len(source_row["_point_ids"]))
    score_by_id = {int(pid): float(score) for pid, score in zip(source_row["_point_ids"], scores)}
    weights = torch.tensor([max(score_by_id.get(int(pid), 1.0), 1.0e-6) for pid in ids], dtype=xy.dtype, device=xy.device)
    pts = xy[idx_tensor].detach()
    rad = radii[idx_tensor].detach().float().clamp(min=float(args.min_radius), max=float(args.max_radius))
    radius_scale = float(args.outer_radius_scale if source_row["_direction"] == "outer" else args.inner_radius_scale)
    pad = rad * radius_scale + float(args.extra_pad_px)
    center = (pts * weights.reshape(-1, 1)).sum(dim=0) / weights.sum().clamp_min(1.0e-6)
    x0 = torch.min(pts[:, 0] - pad).item()
    x1 = torch.max(pts[:, 0] + pad).item()
    y0 = torch.min(pts[:, 1] - pad).item()
    y1 = torch.max(pts[:, 1] + pad).item()
    min_size = float(args.min_bbox_size)
    if x1 - x0 < min_size:
        mid = 0.5 * (x0 + x1)
        x0 = mid - 0.5 * min_size
        x1 = mid + 0.5 * min_size
    if y1 - y0 < min_size:
        mid = 0.5 * (y0 + y1)
        y0 = mid - 0.5 * min_size
        y1 = mid + 0.5 * min_size
    x0 = max(0.0, min(float(width - 1), x0))
    y0 = max(0.0, min(float(height - 1), y0))
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    bbox_w = max(1, int(round(x1 - x0 + 1.0)))
    bbox_h = max(1, int(round(y1 - y0 + 1.0)))
    area = int(max(float(args.min_component_area), float(bbox_w * bbox_h)))
    visible_ratio = len(ids) / max(1, len(source_row["_point_ids"]))
    near_score_sum = float(source_row["_near"]) * float(visible_ratio)
    if near_score_sum <= 0.0:
        near_score_sum = float(len(ids))
    return {
        "cam": int(target_cam),
        "frame": int(target_frame),
        "image_name": image_name,
        "direction": source_row["_direction"],
        "component_id": int(component_id),
        "area": int(area),
        "bbox_x": int(round(x0)),
        "bbox_y": int(round(y0)),
        "bbox_w": int(bbox_w),
        "bbox_h": int(bbox_h),
        "centroid_x": float(center[0].item()),
        "centroid_y": float(center[1].item()),
        "near_point_count": int(len(ids)),
        "near_score_sum": float(near_score_sum),
        "min_center_dist": 0.0,
        "top_point_ids": ";".join(str(int(x)) for x in ids),
        "top_point_scores": ";".join(f"{float(score_by_id.get(int(x), 1.0)):.6f}" for x in ids),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build v311 predicted residual-component CSV by learning source-view "
            "top point clusters and projecting them into held-out target views."
        )
    )
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--load-ckpt", required=True, type=Path)
    parser.add_argument("--source-component-csv", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--source-views", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--train-views", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--train-frames", default="0,570,60")
    parser.add_argument("--target-views", default="21,22,23")
    parser.add_argument("--target-frames", default="0,570,60")
    parser.add_argument("--max-components-per-direction", type=int, default=16)
    parser.add_argument("--source-min-area", type=float, default=40.0)
    parser.add_argument("--min-component-area", type=float, default=40.0)
    parser.add_argument("--jaccard-threshold", type=float, default=0.65)
    parser.add_argument("--min-visible-points", type=int, default=3)
    parser.add_argument("--outer-radius-scale", type=float, default=1.25)
    parser.add_argument("--inner-radius-scale", type=float, default=1.75)
    parser.add_argument("--extra-pad-px", type=float, default=6.0)
    parser.add_argument("--min-radius", type=float, default=1.5)
    parser.add_argument("--max-radius", type=float, default=24.0)
    parser.add_argument("--min-bbox-size", type=float, default=18.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_views = set(_parse_ints(args.source_views))
    target_frames = set(_range_or_values(args.target_frames))
    source_groups = _load_source_components(
        args.source_component_csv,
        source_views=source_views,
        target_frames=target_frames,
        min_area=float(args.source_min_area),
    )
    selected_groups = {
        key: _select_rows(rows, args.max_components_per_direction, args.jaccard_threshold)
        for key, rows in source_groups.items()
    }

    config = _build_config(args)
    gaussians = GaussianModel(config.model.gaussian)
    scene = Scene(config, gaussians, str(args.out_dir / "scene"))
    scene.eval()
    loaded_iteration = scene.load_checkpoint(str(args.load_ckpt))
    bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    rows: list[dict] = []
    per_image_counts: dict[str, dict[str, int]] = {}
    with torch.no_grad():
        for view in scene.test_dataset:
            image_name = str(view.image_name)
            target_cam = int(view.cam_id)
            target_frame = int(view.frame_id)
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
            pc = render_pkg["deformed_gaussian"]
            xy, valid = _project_points(pc.get_xyz, view)
            radii = render_pkg["radii"].detach().float().to(xy.device)
            point_count = int(pc.get_xyz.shape[0])
            width = int(view.image_width)
            height = int(view.image_height)
            image_counts = {"outer": 0, "inner": 0}
            for direction in ("outer", "inner"):
                selected = selected_groups.get((target_frame, direction), [])
                component_id = 1
                for source_row in selected:
                    predicted = _make_prediction_row(
                        source_row=source_row,
                        target_cam=target_cam,
                        target_frame=target_frame,
                        image_name=image_name,
                        component_id=component_id,
                        xy=xy,
                        valid=valid,
                        radii=radii,
                        point_count=point_count,
                        width=width,
                        height=height,
                        args=args,
                    )
                    if predicted is None:
                        continue
                    rows.append(predicted)
                    component_id += 1
                    image_counts[direction] += 1
            per_image_counts[image_name] = image_counts
            del render_pkg

    rows.sort(key=lambda row: (int(row["cam"]), int(row["frame"]), str(row["direction"]), int(row["component_id"])))
    _write_csv(args.output_csv, rows)
    summary = {
        "load_ckpt": str(args.load_ckpt),
        "loaded_iteration": int(loaded_iteration),
        "source_component_csv": str(args.source_component_csv),
        "output_csv": str(args.output_csv),
        "source_views": sorted(source_views),
        "target_views": _parse_ints(args.target_views),
        "target_frames": sorted(target_frames),
        "source_group_count": len(source_groups),
        "selected_group_count": len(selected_groups),
        "predicted_component_count": len(rows),
        "per_image_counts": per_image_counts,
        "args": _json_safe(vars(args)),
    }
    (args.out_dir / "prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
