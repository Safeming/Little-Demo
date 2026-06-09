#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene import GaussianModel, Scene


def _parse_ints(value: str) -> list[int]:
    out: list[int] = []
    for token in str(value or "").replace("[", "").replace("]", "").split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return out


def _range_or_values(value: str) -> list[int]:
    values = _parse_ints(value)
    if len(values) == 3:
        start, end, step = values
        return list(range(start, end, max(step, 1)))
    return values


def _parse_point_ids(value: str) -> list[int]:
    out: list[int] = []
    for token in str(value or "").replace(",", ";").split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out


def _parse_scores(value: str, count: int) -> list[float]:
    out: list[float] = []
    for token in str(value or "").replace(",", ";").split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            out.append(1.0)
    if len(out) < count:
        out.extend([1.0] * (count - len(out)))
    return out[:count]


def _safe_int(row: dict[str, str], key: str, default: int = -1) -> int:
    try:
        return int(float(row.get(key, default) or default))
    except Exception:
        return default


def _safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def _row_frame(row: dict[str, str]) -> int:
    frame = _safe_int(row, "frame", -1)
    if frame >= 0:
        return frame
    image_name = str(row.get("image_name", ""))
    if "_f" in image_name:
        try:
            return int(image_name.rsplit("_f", 1)[1])
        except Exception:
            return -1
    return -1


def _row_cam(row: dict[str, str]) -> int:
    cam = _safe_int(row, "cam", -1)
    if cam >= 0:
        return cam
    image_name = str(row.get("image_name", ""))
    if image_name.startswith("c") and "_f" in image_name:
        try:
            return int(image_name.split("_f", 1)[0][1:])
        except Exception:
            return -1
    return -1


def _load_point_stats(path: Path | None) -> dict[int, dict[str, float]]:
    if path is None or not str(path) or not path.exists():
        return {}
    stats: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            point_idx = _safe_int(row, "point_idx", -1)
            if point_idx < 0:
                continue
            stats[point_idx] = {
                "boundary_score": _safe_float(row, "boundary_score", 0.0),
                "surface_distance": _safe_float(row, "surface_distance", 0.0),
                "thin_score": _safe_float(row, "thin_score", 0.0),
                "visible_frame_hits": _safe_float(row, "visible_frame_hits", 0.0),
                "layer_id": float(_safe_int(row, "layer_id", -1)),
                "region_id": float(_safe_int(row, "region_id", -1)),
                "dominant_joint": float(_safe_int(row, "dominant_joint", -1)),
            }
    return stats


def _parse_optional_ids(value: str) -> set[int]:
    text = str(value or "").strip()
    if not text or text.lower() in ("all", "*", "none", "null"):
        return set()
    return set(_parse_ints(text))


def _passes_point_filters(point_id: int, stats: dict[int, dict[str, float]], args: argparse.Namespace, prefix: str) -> bool:
    if not stats:
        return True
    item = stats.get(int(point_id))
    if item is None:
        return False
    min_boundary = float(getattr(args, f"{prefix}_min_boundary", -1.0))
    min_thin = float(getattr(args, f"{prefix}_min_thin", -1.0))
    min_visible = float(getattr(args, f"{prefix}_min_visible_hits", -1.0))
    surface_min = getattr(args, f"{prefix}_surface_min", None)
    surface_max = getattr(args, f"{prefix}_surface_max", None)
    if min_boundary >= 0.0 and float(item.get("boundary_score", 0.0)) < min_boundary:
        return False
    if min_thin >= 0.0 and float(item.get("thin_score", 0.0)) < min_thin:
        return False
    if min_visible >= 0.0 and float(item.get("visible_frame_hits", 0.0)) < min_visible:
        return False
    if surface_min is not None and float(item.get("surface_distance", 0.0)) < float(surface_min):
        return False
    if surface_max is not None and float(item.get("surface_distance", 0.0)) > float(surface_max):
        return False
    for field, arg_name in (
        ("layer_id", f"{prefix}_allowed_layers"),
        ("region_id", f"{prefix}_allowed_regions"),
        ("dominant_joint", f"{prefix}_allowed_joints"),
    ):
        allowed = _parse_optional_ids(str(getattr(args, arg_name, "") or ""))
        if allowed and int(item.get(field, -1)) not in allowed:
            return False
    return True


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
    return config


def _camera_vector(view) -> list[float]:
    center = getattr(view, "camera_center", None)
    if not torch.is_tensor(center):
        return [0.0, 0.0, 1.0]
    vec = center.detach().float().cpu()
    norm = float(torch.linalg.norm(vec).item())
    if norm <= 1.0e-8:
        return [0.0, 0.0, 1.0]
    vec = vec / norm
    return [float(vec[0]), float(vec[1]), float(vec[2])]


def _dot(a: list[float], b: list[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def _image_name(view) -> str:
    try:
        name = getattr(view, "image_name")
    except Exception:
        name = ""
    if isinstance(name, (list, tuple)):
        name = name[0] if name else ""
    if name:
        return str(name)
    cam = _view_int(view, "cam_id", "uid", default=-1)
    frame = _view_int(view, "frame_id", "frame_idx", default=-1)
    return f"c{cam:02d}_f{frame:06d}"


def _view_int(view, *names: str, default: int = -1) -> int:
    for name in names:
        try:
            value = getattr(view, name)
        except Exception:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return int(default)


def _load_train_votes(args: argparse.Namespace) -> dict[tuple[int, int, str], dict[int, float]]:
    source_views = set(_parse_ints(args.source_views))
    train_frames = set(_range_or_values(args.train_frames))
    direction_mode = str(args.direction_mode).lower()
    allowed_directions = {"outer", "inner"} if direction_mode == "both" else {direction_mode}
    votes: dict[tuple[int, int, str], dict[int, float]] = defaultdict(lambda: defaultdict(float))
    with args.component_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cam = _row_cam(row)
            frame = _row_frame(row)
            direction = str(row.get("direction", "")).strip().lower()
            if cam not in source_views or frame not in train_frames or direction not in allowed_directions:
                continue
            area = _safe_float(row, "area", 0.0)
            if area < float(args.min_area):
                continue
            ids = _parse_point_ids(row.get("top_point_ids", ""))
            if not ids:
                continue
            scores = _parse_scores(row.get("top_point_scores", ""), len(ids))
            near = max(_safe_float(row, "near_score_sum", 0.0), 1.0)
            component_weight = math.log1p(max(area, 1.0)) * math.log1p(near)
            sign = 1.0 if direction == "outer" else -1.0
            key = (frame, cam, str(row.get("image_name", "")))
            for point_id, score in zip(ids[: int(args.top_points_per_component)], scores[: int(args.top_points_per_component)]):
                if int(point_id) < 0:
                    continue
                votes[key][int(point_id)] += sign * component_weight * max(float(score), 1.0e-4)
    return votes


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v325 view-conditioned signed point prior.")
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--load-ckpt", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--component-csv", required=True, type=Path)
    parser.add_argument("--point-csv", default=None, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--source-views", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--train-views", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--target-views", default="21,22,23")
    parser.add_argument("--train-frames", default="0,570,60")
    parser.add_argument("--target-frames", default="0,570,60")
    parser.add_argument("--nearest-views", type=int, default=4)
    parser.add_argument("--direction-mode", choices=("both", "outer", "inner"), default="both")
    parser.add_argument("--view-power", type=float, default=3.0)
    parser.add_argument("--min-area", type=float, default=20.0)
    parser.add_argument("--top-points-per-component", type=int, default=8)
    parser.add_argument("--max-shrink", type=int, default=96)
    parser.add_argument("--max-grow", type=int, default=96)
    parser.add_argument("--min-abs-score", type=float, default=0.0)
    parser.add_argument("--shrink-min-boundary", type=float, default=-1.0)
    parser.add_argument("--shrink-min-thin", type=float, default=-1.0)
    parser.add_argument("--shrink-min-visible-hits", type=float, default=-1.0)
    parser.add_argument("--shrink-surface-min", type=float, default=None)
    parser.add_argument("--shrink-surface-max", type=float, default=None)
    parser.add_argument("--shrink-allowed-layers", default="")
    parser.add_argument("--shrink-allowed-regions", default="")
    parser.add_argument("--shrink-allowed-joints", default="")
    parser.add_argument("--grow-min-boundary", type=float, default=-1.0)
    parser.add_argument("--grow-min-thin", type=float, default=-1.0)
    parser.add_argument("--grow-min-visible-hits", type=float, default=-1.0)
    parser.add_argument("--grow-surface-min", type=float, default=None)
    parser.add_argument("--grow-surface-max", type=float, default=None)
    parser.add_argument("--grow-allowed-layers", default="")
    parser.add_argument("--grow-allowed-regions", default="")
    parser.add_argument("--grow-allowed-joints", default="")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    point_stats = _load_point_stats(args.point_csv)
    config = _build_config(args)
    gaussians = GaussianModel(config.model.gaussian)
    scene = Scene(config, gaussians, str(args.out_dir / "prior_scene"))
    scene.load_checkpoint(str(args.load_ckpt))

    train_votes = _load_train_votes(args)
    train_by_frame: dict[int, list[dict]] = defaultdict(list)
    train_views = [scene.train_dataset[idx] for idx in range(len(scene.train_dataset))]
    test_views = [scene.test_dataset[idx] for idx in range(len(scene.test_dataset))]
    for view in train_views:
        image = _image_name(view)
        frame = _view_int(view, "frame_id", "frame_idx", default=-1)
        cam = _view_int(view, "cam_id", "uid", default=-1)
        key = (frame, cam, image)
        if key not in train_votes:
            continue
        train_by_frame[frame].append({
            "cam": cam,
            "image_name": image,
            "view_vec": _camera_vector(view),
            "votes": dict(train_votes[key]),
        })

    by_image: dict[str, dict] = {}
    target_records = []
    for view in test_views:
        image = _image_name(view)
        frame = _view_int(view, "frame_id", "frame_idx", default=-1)
        target_vec = _camera_vector(view)
        candidates = train_by_frame.get(frame, [])
        ranked = []
        for item in candidates:
            sim = max(_dot(target_vec, item["view_vec"]), 0.0)
            weight = max(sim, 1.0e-6) ** float(args.view_power)
            ranked.append((weight, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        ranked = ranked[: max(int(args.nearest_views), 1)]
        aggregate: dict[int, float] = defaultdict(float)
        for weight, item in ranked:
            for point_id, vote in item["votes"].items():
                aggregate[int(point_id)] += float(weight) * float(vote)
        shrink = [
            (pid, score)
            for pid, score in aggregate.items()
            if score > float(args.min_abs_score) and _passes_point_filters(pid, point_stats, args, "shrink")
        ]
        grow = [
            (pid, -score)
            for pid, score in aggregate.items()
            if score < -float(args.min_abs_score) and _passes_point_filters(pid, point_stats, args, "grow")
        ]
        shrink.sort(key=lambda pair: pair[1], reverse=True)
        grow.sort(key=lambda pair: pair[1], reverse=True)
        shrink_ids = [int(pid) for pid, _ in shrink[: max(int(args.max_shrink), 0)]]
        grow_ids = [int(pid) for pid, _ in grow[: max(int(args.max_grow), 0)]]
        by_image[image] = {
            "frame": frame,
            "target_view": _view_int(view, "cam_id", "uid", default=-1),
            "nearest_train_views": [
                {
                    "cam": int(item["cam"]),
                    "image_name": str(item["image_name"]),
                    "weight": float(weight),
                }
                for weight, item in ranked
            ],
            "shrink_point_ids": shrink_ids,
            "grow_point_ids": grow_ids,
            "shrink_scores": [float(score) for _, score in shrink[: max(int(args.max_shrink), 0)]],
            "grow_scores": [float(score) for _, score in grow[: max(int(args.max_grow), 0)]],
        }
        target_records.append({
            "image_name": image,
            "frame": frame,
            "shrink_count": len(shrink_ids),
            "grow_count": len(grow_ids),
            "nearest": [int(item["cam"]) for _, item in ranked],
        })

    payload = {
        "source_component_csv": str(args.component_csv),
        "selection": {
            "nearest_views": int(args.nearest_views),
            "direction_mode": str(args.direction_mode),
            "view_power": float(args.view_power),
            "min_area": float(args.min_area),
            "top_points_per_component": int(args.top_points_per_component),
            "max_shrink": int(args.max_shrink),
            "max_grow": int(args.max_grow),
            "min_abs_score": float(args.min_abs_score),
            "point_csv": str(args.point_csv or ""),
            "shrink_filters": {
                "min_boundary": float(args.shrink_min_boundary),
                "min_thin": float(args.shrink_min_thin),
                "min_visible_hits": float(args.shrink_min_visible_hits),
                "surface_min": args.shrink_surface_min,
                "surface_max": args.shrink_surface_max,
                "allowed_layers": str(args.shrink_allowed_layers or ""),
                "allowed_regions": str(args.shrink_allowed_regions or ""),
                "allowed_joints": str(args.shrink_allowed_joints or ""),
            },
            "grow_filters": {
                "min_boundary": float(args.grow_min_boundary),
                "min_thin": float(args.grow_min_thin),
                "min_visible_hits": float(args.grow_min_visible_hits),
                "surface_min": args.grow_surface_min,
                "surface_max": args.grow_surface_max,
                "allowed_layers": str(args.grow_allowed_layers or ""),
                "allowed_regions": str(args.grow_allowed_regions or ""),
                "allowed_joints": str(args.grow_allowed_joints or ""),
            },
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
        "images": len(by_image),
        "mean_shrink": sum(r["shrink_count"] for r in target_records) / max(len(target_records), 1),
        "mean_grow": sum(r["grow_count"] for r in target_records) / max(len(target_records), 1),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
