#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.make_377_stageB_v347_component_3d_asset import (  # noqa: E402
    METRICS,
    _canonical_support,
    _load_component_rows,
    _load_point_stats,
    _load_samples,
    _metric_delta,
    _owner,
)
from tools.make_377_stageB_v361_residual_shaped_micro_child_asset import (  # noqa: E402
    _camera_after_dataset_adjust,
    _cov6,
    _load_checkpoint_xyz,
    _load_json,
    _load_renderer_space_cache,
    _project,
    _rigid_transform_from_correspondences,
    _unproject,
    _unit,
    _world_covariance_from_screen,
)


IMAGE_RE = re.compile(r"c(?P<cam>\d+)_f(?P<frame>\d+)$")


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")


def _csv_source_key(rec: dict[str, object]) -> str:
    return f"{rec['image_name']}:{rec['direction']}:row{int(rec['row_index'])}"


def _parse_float_list(text: str) -> list[float]:
    return [float(token) for token in str(text).replace(";", ",").split(",") if token.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(float(token)) for token in str(text).replace(";", ",").split(",") if token.strip()]


def _image_size(
    dataset_root: Path,
    subject: str,
    image_name: str,
    fallback: tuple[int, int] = (512, 512),
    render_exp: Path | None = None,
) -> tuple[int, int]:
    if render_exp is not None:
        for rel in (
            Path("test-view") / "renders" / f"render_{image_name}.png",
            Path("test-view") / "opacity" / f"opacity_{image_name}.png",
        ):
            image_path = render_exp / rel
            if image_path.exists():
                with Image.open(image_path) as image:
                    return int(image.size[0]), int(image.size[1])
    match = IMAGE_RE.match(image_name)
    if match is None:
        return fallback
    cam = str(int(match.group("cam")))
    frame = int(match.group("frame"))
    mask_path = dataset_root / subject / cam / f"{frame:06d}.png"
    if not mask_path.exists():
        return fallback
    with Image.open(mask_path) as image:
        return int(image.size[0]), int(image.size[1])


def _read_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0


def _morph(mask: np.ndarray, width: int, mode: str) -> np.ndarray:
    width = max(1, int(width))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    op = cv2.dilate if mode == "dilate" else cv2.erode
    return op(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _render_support(rgb: np.ndarray, threshold: float, close_kernel: int) -> np.ndarray:
    luma = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    support = (luma > float(threshold)) | (chroma > float(threshold) * 0.75)
    if int(close_kernel) > 1:
        k = int(close_kernel)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        support = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return support


def _component_gate_mask(rec: dict[str, object], shape: tuple[int, int], pad_px: float = 6.0) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=bool)
    cx = float(rec.get("centroid_x", 0.0) or 0.0)
    cy = float(rec.get("centroid_y", 0.0) or 0.0)
    half_w = max(0.5 * float(rec.get("bbox_w", 1.0) or 1.0) + float(pad_px), 1.0)
    half_h = max(0.5 * float(rec.get("bbox_h", 1.0) or 1.0) + float(pad_px), 1.0)
    x0 = max(0, int(math.floor(cx - half_w)))
    x1 = min(width - 1, int(math.ceil(cx + half_w)))
    y0 = max(0, int(math.floor(cy - half_h)))
    y1 = min(height - 1, int(math.ceil(cy + half_h)))
    if x1 < x0 or y1 < y0:
        return mask
    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    local = ((xx - cx) / half_w) ** 2 + ((yy - cy) / half_h) ** 2 <= 1.0
    mask[y0 : y1 + 1, x0 : x1 + 1] = local
    return mask


def _inner_residual_mask_for_component(
    *,
    render_exp: Path,
    dataset_root: Path,
    subject: str,
    image_name: str,
    rec: dict[str, object],
    render_support_threshold: float,
    close_kernel: int,
    search_band_width: int,
    min_pixels: int,
) -> tuple[np.ndarray | None, dict[str, float]]:
    match = IMAGE_RE.match(image_name)
    if match is None:
        return None, {}
    cam = str(int(match.group("cam")))
    frame = int(match.group("frame"))
    render_path = render_exp / "test-view" / "renders" / f"render_{image_name}.png"
    mask_path = dataset_root / subject / cam / f"{frame:06d}.png"
    if not render_path.exists() or not mask_path.exists():
        return None, {}
    rgb = _read_rgb(render_path)
    size = (rgb.shape[1], rgb.shape[0])
    gt = _read_mask(mask_path, size)
    support = _render_support(rgb, render_support_threshold, close_kernel)
    inner = gt & (~support)
    # Keep the same neighborhood convention as the raw diagnostic; this avoids
    # fitting micro children to far interior holes unrelated to the boundary.
    near_boundary = _morph(gt, search_band_width, "dilate") & (~_morph(gt, max(1, search_band_width // 3), "erode"))
    inner = inner & near_boundary
    gate = _component_gate_mask(rec, inner.shape, pad_px=8.0)
    if not bool(inner.any()):
        return None, {}
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(inner.astype(np.uint8), 8)
    best = None
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < int(min_pixels):
            continue
        comp = labels == idx
        overlap = int((comp & gate).sum())
        cx, cy = centroids[idx]
        dist = math.sqrt((float(cx) - float(rec.get("centroid_x", 0.0) or 0.0)) ** 2 + (float(cy) - float(rec.get("centroid_y", 0.0) or 0.0)) ** 2)
        score = overlap * 12.0 + min(area, 400) - 2.0 * dist
        if best is None or score > best[0]:
            best = (score, comp, area, overlap, float(cx), float(cy), float(dist))
    if best is None:
        return None, {}
    _score, comp, area, overlap, cx, cy, dist = best
    return comp, {
        "residual_mask_pixels": int(area),
        "residual_mask_overlap_pixels": int(overlap),
        "residual_mask_cx": float(cx),
        "residual_mask_cy": float(cy),
        "residual_mask_component_dist_px": float(dist),
    }


def _frame_score(current: dict[str, float], delta: dict[str, float]) -> float:
    return (
        max(float(current.get("inner", 0.0)), 0.0)
        + 0.22 * max(float(current.get("outer", 0.0)), 0.0)
        + 0.35 * max(float(current.get("opacity_inner", 0.0)), 0.0)
        + 0.08 * max(float(current.get("opacity_outer", 0.0)), 0.0)
        + 1000.0 * max(float(current.get("hard", 0.0)), 0.0)
        + 25.0 * max(float(current.get("edge", 0.0)), 0.0)
        + 1.5 * max(float(delta.get("inner", 0.0)), 0.0)
        + 1.0 * max(float(delta.get("opacity_inner", 0.0)), 0.0)
    )


def _component_score(rec: dict[str, object]) -> float:
    return float(rec.get("area", 0.0) or 0.0) * math.log1p(float(rec.get("near_score_sum", 0.0) or 0.0))


def _micro_specs_from_component(rec: dict[str, object], count: int) -> list[dict[str, object]]:
    count = max(1, int(count))
    cx = float(rec.get("centroid_x", 0.0) or 0.0)
    cy = float(rec.get("centroid_y", 0.0) or 0.0)
    w = max(float(rec.get("bbox_w", 1.0) or 1.0), 1.0)
    h = max(float(rec.get("bbox_h", 1.0) or 1.0), 1.0)
    if w >= h:
        major = np.array([1.0, 0.0], dtype=np.float64)
        minor = np.array([0.0, 1.0], dtype=np.float64)
        major_span = w
        minor_span = h
    else:
        major = np.array([0.0, 1.0], dtype=np.float64)
        minor = np.array([1.0, 0.0], dtype=np.float64)
        major_span = h
        minor_span = w
    if count == 1:
        offsets = [0.0]
    else:
        offsets = np.linspace(-0.42 * major_span, 0.42 * major_span, count).tolist()
    specs = []
    for index, offset in enumerate(offsets):
        center = np.array([cx, cy], dtype=np.float64) + major * float(offset)
        specs.append({
            "screen_x": float(center[0]),
            "screen_y": float(center[1]),
            "major": major.copy(),
            "minor": minor.copy(),
            "major_sigma_px": max(1.0, 0.32 * major_span / max(count, 1) + 0.35),
            "minor_sigma_px": max(0.9, 0.34 * minor_span + 0.35),
            "component_major_span_px": float(major_span),
            "component_minor_span_px": float(minor_span),
            "micro_index": int(index),
        })
    return specs


def _micro_specs_from_mask(mask: np.ndarray, count: int) -> list[dict[str, object]]:
    ys, xs = np.where(mask)
    if xs.size <= 0:
        return []
    count = max(1, int(count))
    coords = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    center = coords.mean(axis=0)
    centered = coords - center.reshape(1, 2)
    if coords.shape[0] >= 3:
        cov = np.cov(centered.T)
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1]
        vectors = vectors[:, order]
        major = _unit(vectors[:, 0], np.array([1.0, 0.0], dtype=np.float64))
    else:
        major = np.array([1.0, 0.0], dtype=np.float64)
    minor = np.array([-major[1], major[0]], dtype=np.float64)
    major_coord = centered @ major
    minor_coord = centered @ minor
    bins = np.linspace(float(major_coord.min()), float(major_coord.max()), count + 1)
    specs: list[dict[str, object]] = []
    for index in range(count):
        if index == count - 1:
            selected = (major_coord >= bins[index]) & (major_coord <= bins[index + 1])
        else:
            selected = (major_coord >= bins[index]) & (major_coord < bins[index + 1])
        if not bool(selected.any()):
            target = 0.5 * (bins[index] + bins[index + 1])
            nearest = int(np.argmin(np.abs(major_coord - target)))
            selected = np.zeros_like(major_coord, dtype=bool)
            selected[nearest] = True
        local = coords[selected]
        local_major = major_coord[selected]
        local_minor = minor_coord[selected]
        local_center = local.mean(axis=0)
        if local_major.size <= 1:
            major_span = 1.0
        else:
            lo, hi = np.percentile(local_major, [8.0, 92.0])
            major_span = max(float(hi - lo), 1.0)
        if local_minor.size <= 1:
            minor_span = 1.0
        else:
            lo, hi = np.percentile(local_minor, [8.0, 92.0])
            minor_span = max(float(hi - lo), 1.0)
        specs.append({
            "screen_x": float(local_center[0]),
            "screen_y": float(local_center[1]),
            "major": major.copy(),
            "minor": minor.copy(),
            "major_sigma_px": max(0.75, 0.34 * major_span + 0.25),
            "minor_sigma_px": max(0.55, 0.34 * minor_span + 0.20),
            "component_major_span_px": float(major_span),
            "component_minor_span_px": float(minor_span),
            "micro_index": int(index),
        })
    return specs


def _owner_tuple(owner: dict[str, object]) -> tuple[int, int, int]:
    def _to_int(value: object) -> int:
        try:
            return int(float(value))
        except Exception:
            return -999

    return (
        _to_int(owner.get("owner_layer_id", "")),
        _to_int(owner.get("owner_region_id", "")),
        _to_int(owner.get("owner_joint", owner.get("owner_joint_id", ""))),
    )


def _outer_candidates_for_inner(
    inner: dict[str, object],
    outer_rows: list[dict[str, object]],
    point_stats: dict[int, dict[str, object]],
    *,
    max_outer: int,
    require_owner_match: bool,
    min_owner_consistency: float,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    if not outer_rows or max_outer <= 0:
        return []
    inner_owner = _owner(list(inner.get("top_point_ids", [])), point_stats)
    inner_owner_tuple = _owner_tuple(inner_owner)
    ix = float(inner.get("centroid_x", 0.0) or 0.0)
    iy = float(inner.get("centroid_y", 0.0) or 0.0)
    ranked = []
    for outer in outer_rows:
        owner = _owner(list(outer.get("top_point_ids", [])), point_stats)
        if float(owner.get("owner_consistency", 0.0) or 0.0) < float(min_owner_consistency):
            continue
        owner_tuple = _owner_tuple(owner)
        owner_match = owner_tuple == inner_owner_tuple
        if require_owner_match and not owner_match:
            continue
        ox = float(outer.get("centroid_x", 0.0) or 0.0)
        oy = float(outer.get("centroid_y", 0.0) or 0.0)
        dist = math.sqrt((ox - ix) ** 2 + (oy - iy) ** 2)
        score = (
            -dist
            + (240.0 if owner_match else 0.0)
            + 0.12 * float(outer.get("area", 0.0) or 0.0)
            + 0.02 * float(outer.get("near_score_sum", 0.0) or 0.0)
        )
        ranked.append((score, outer, owner, dist, owner_match))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [(outer, owner) for _, outer, owner, _, _ in ranked[:max_outer]]


def _support_from_renderer_cache(
    top_ids: list[int],
    image_name: str,
    renderer_space_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    checkpoint_posed_xyz: np.ndarray,
    checkpoint_canonical_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    posed_xyz, canonical_xyz = renderer_space_cache.get(image_name, (checkpoint_posed_xyz, checkpoint_canonical_xyz))
    ids = [idx for idx in top_ids if 0 <= idx < posed_xyz.shape[0] and 0 <= idx < canonical_xyz.shape[0]]
    if not ids:
        return None
    idx = np.asarray(ids, dtype=np.int64)
    return posed_xyz, canonical_xyz, posed_xyz[idx], canonical_xyz[idx]


def _make_outer_action(
    *,
    outer: dict[str, object],
    outer_owner: dict[str, object],
    pair_id: str,
    action_index: int,
    point_stats: dict[int, dict[str, object]],
    max_top_ids: int,
    radius_floor: float,
    radius_pad: float,
    radius_scale: float,
    outer_radius_scale: float,
    outer_score_scale: float,
) -> dict[str, object] | None:
    support = _canonical_support(
        list(outer.get("top_point_ids", [])),
        point_stats,
        radius_floor=float(radius_floor),
        radius_pad=float(radius_pad),
        radius_scale=float(radius_scale),
        max_top_ids=int(max_top_ids),
        cluster_enable=True,
        cluster_min_points=4,
        cluster_owner_gate=True,
        cluster_radius_max=0.18,
    )
    if not support:
        return None
    radius = max(float(support.get("canonical_radius", 0.0) or 0.0) * float(outer_radius_scale), float(radius_floor))
    source_key = _csv_source_key(outer)
    action = {
        "component_key": f"v368_outer:{_safe_key(source_key)}:a{action_index}",
        "source_component_key": source_key,
        "source_row_index": int(outer["row_index"]),
        "source_component_id": int(outer["component_id"]),
        "source_image_name": str(outer["image_name"]),
        "direction": "outer",
        "scope": "global",
        "asset_scope": "global",
        "image_name": "",
        "pair_id": pair_id,
        "pair_role": "outer_protect_shrink",
        "mode": "paired_local_3d_intersect",
        "component_match_mode": "semantic_local_3d",
        "semantic_override": True,
        "targeted_only": True,
        "owner_gate": True,
        "activation_required": True,
        "activation_direction": "outer",
        "activation_pad_px": 4.0,
        "activation_ellipse_scale": 1.15,
        "activation_min_area": 20.0,
        "activation_owner_gate": True,
        "activation_owner_primary_only": True,
        "anchor_mode": "semantic_local_frame",
        "anchor_local_frame": True,
        "rotate_covariance_with_anchor": True,
        "anchor_owner_gate": True,
        "anchor_explicit_ids_required": True,
        "anchor_knn": 24,
        "anchor_min_points": 3,
        "top_ids_enable": False,
        "top_ids_only": False,
        "local_3d_fallback_top_ids": True,
        "score_scale": float(outer_score_scale),
        "canonical_radius": float(radius),
        "canonical_radius_outer": float(radius),
        "canonical_radius_scale": 1.0,
        "reason": "v368_residual_grouped_outer_semantic_local_3d_protect",
        "v368_source_bbox_x": float(outer.get("bbox_x", 0.0) or 0.0),
        "v368_source_bbox_y": float(outer.get("bbox_y", 0.0) or 0.0),
        "v368_source_bbox_w": float(outer.get("bbox_w", 0.0) or 0.0),
        "v368_source_bbox_h": float(outer.get("bbox_h", 0.0) or 0.0),
        "v368_source_centroid_x": float(outer.get("centroid_x", 0.0) or 0.0),
        "v368_source_centroid_y": float(outer.get("centroid_y", 0.0) or 0.0),
        **support,
        **outer_owner,
    }
    action["anchor_point_ids"] = list(action.get("top_point_ids", action.get("source_top_point_ids", [])))
    return action


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build v368 subject-general self-protected residual grouped actuator asset."
    )
    parser.add_argument("--baseline-render-exp", required=True, type=Path)
    parser.add_argument("--current-render-exp", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--renderer-space-cache", default="")
    parser.add_argument("--component-csv", default="assets/adopted_geometry/377/v320_selected_components.csv", type=Path)
    parser.add_argument("--point-csv", default="assets/adopted_geometry/377/v304_point_contributors_all.csv", type=Path)
    parser.add_argument("--dataset-root", default="data/ZJUMoCap", type=Path)
    parser.add_argument("--subject", default="CoreView_377")
    parser.add_argument("--source-width", default=1024, type=int)
    parser.add_argument("--source-height", default=1024, type=int)
    parser.add_argument("--top-frames", default=30, type=int)
    parser.add_argument("--inner-per-frame", default=2, type=int)
    parser.add_argument("--outer-per-inner", default=1, type=int)
    parser.add_argument("--max-groups", default=160, type=int)
    parser.add_argument("--micro-counts", default="3,5")
    parser.add_argument("--radius-scales", default="0.50,0.70")
    parser.add_argument("--minor-scales", default="0.50,0.75")
    parser.add_argument("--depth-scales", default="1.0")
    parser.add_argument("--covariance-scales", default="1.0")
    parser.add_argument("--depth-sigma-px", default=1.4, type=float)
    parser.add_argument("--child-opacity", default=0.045, type=float)
    parser.add_argument("--child-opacity-mode", default="constant", choices=("constant", "sqrt", "divide"))
    parser.add_argument("--max-top-ids", default=8, type=int)
    parser.add_argument("--min-owner-consistency", default=0.50, type=float)
    parser.add_argument("--require-owner-match", action="store_true")
    parser.add_argument("--radius-floor", default=0.010, type=float)
    parser.add_argument("--radius-pad", default=0.006, type=float)
    parser.add_argument("--row-radius-scale", default=1.15, type=float)
    parser.add_argument("--outer-radius-scale", default=0.72, type=float)
    parser.add_argument("--outer-score-scale", default=0.80, type=float)
    parser.add_argument("--activation-pad-px", default=4.0, type=float)
    parser.add_argument("--activation-ellipse-scale", default=1.15, type=float)
    parser.add_argument("--anchor-radius-scale", default=0.0, type=float)
    parser.add_argument("--residual-mask-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--residual-render-support-threshold", default=0.025, type=float)
    parser.add_argument("--residual-close-kernel", default=5, type=int)
    parser.add_argument("--residual-search-band-width", default=24, type=int)
    parser.add_argument("--residual-min-mask-pixels", default=4, type=int)
    parser.add_argument("--self-protect-inner-radius-fraction", default=0.80, type=float)
    parser.add_argument("--self-protect-shrink-factor", default=0.75, type=float)
    parser.add_argument("--self-protect-opacity-factor", default=0.50, type=float)
    parser.add_argument("--self-protect-drop-on-outer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-candidates-tsv", required=True, type=Path)
    args = parser.parse_args()

    baseline = _load_samples(args.baseline_render_exp)
    current = _load_samples(args.current_render_exp)
    components = _load_component_rows(args.component_csv)
    point_stats = _load_point_stats(args.point_csv)
    checkpoint_posed_xyz, checkpoint_canonical_xyz = _load_checkpoint_xyz(args.checkpoint)
    renderer_space_cache_path = Path(args.renderer_space_cache) if str(args.renderer_space_cache).strip() else None
    renderer_space_cache = _load_renderer_space_cache(renderer_space_cache_path)
    cams = _load_json(args.dataset_root / args.subject / "cam_params.json")

    micro_counts = _parse_int_list(args.micro_counts)
    radius_scales = _parse_float_list(args.radius_scales)
    minor_scales = _parse_float_list(args.minor_scales)
    depth_scales = _parse_float_list(args.depth_scales)
    covariance_scales = _parse_float_list(args.covariance_scales)

    frames = []
    for image_name, cur in current.items():
        base = baseline.get(image_name)
        rows = components.get(image_name, {})
        if not base or not rows.get("inner") or not rows.get("outer"):
            continue
        delta = {metric: _metric_delta(cur, base, metric) for metric in METRICS}
        if max(cur.get("inner", 0.0), cur.get("opacity_inner", 0.0), cur.get("hard", 0.0) * 1000.0) <= 0.0:
            continue
        frames.append({
            "image_name": image_name,
            "score": _frame_score(cur, delta),
            **{metric: float(cur.get(metric, 0.0)) for metric in METRICS},
            **{f"{metric}_delta_base": float(delta[metric]) for metric in METRICS},
        })
    frames.sort(key=lambda row: (float(row["score"]), str(row["image_name"])), reverse=True)
    if int(args.top_frames) > 0:
        frames = frames[: int(args.top_frames)]

    children: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    groups: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    owner_counts: Counter[tuple[int, int, int]] = Counter()

    for frame in frames:
        if int(args.max_groups) > 0 and len(groups) >= int(args.max_groups):
            break
        image_name = str(frame["image_name"])
        match = IMAGE_RE.match(image_name)
        if match is None:
            continue
        cam_name = str(int(match.group("cam")))
        if cam_name not in cams:
            continue
        width, height = _image_size(
            args.dataset_root,
            args.subject,
            image_name,
            render_exp=args.current_render_exp,
        )
        k, r_dataset, t = _camera_after_dataset_adjust(
            cams[cam_name],
            width,
            height,
            source_width=int(args.source_width),
            source_height=int(args.source_height),
        )
        rows = components.get(image_name, {})
        inner_rows = sorted(rows.get("inner", []), key=_component_score, reverse=True)[: max(int(args.inner_per_frame), 0)]
        outer_rows = list(rows.get("outer", []))

        for inner in inner_rows:
            if int(args.max_groups) > 0 and len(groups) >= int(args.max_groups):
                break
            inner_owner = _owner(list(inner.get("top_point_ids", [])), point_stats)
            if float(inner_owner.get("owner_consistency", 0.0) or 0.0) < float(args.min_owner_consistency):
                continue
            support_cache = _support_from_renderer_cache(
                list(inner.get("top_point_ids", [])),
                image_name,
                renderer_space_cache,
                checkpoint_posed_xyz,
                checkpoint_canonical_xyz,
            )
            if support_cache is None:
                continue
            posed_xyz, canonical_xyz, posed_support, canonical_support = support_cache
            posed_anchor = posed_support.mean(axis=0)
            canonical_anchor = canonical_support.mean(axis=0)
            local_rot = _rigid_transform_from_correspondences(canonical_support, posed_support)
            anchor_u, anchor_v, anchor_depth = _project(posed_anchor, k, r_dataset, t)
            if not np.isfinite(anchor_depth) or anchor_depth <= 0.0:
                continue

            outer_selected = _outer_candidates_for_inner(
                inner,
                outer_rows,
                point_stats,
                max_outer=int(args.outer_per_inner),
                require_owner_match=bool(args.require_owner_match),
                min_owner_consistency=float(args.min_owner_consistency),
            )
            if not outer_selected:
                continue
            residual_mask = None
            residual_stats: dict[str, float] = {}
            if bool(args.residual_mask_enable):
                residual_mask, residual_stats = _inner_residual_mask_for_component(
                    render_exp=args.current_render_exp,
                    dataset_root=args.dataset_root,
                    subject=args.subject,
                    image_name=image_name,
                    rec=inner,
                    render_support_threshold=float(args.residual_render_support_threshold),
                    close_kernel=int(args.residual_close_kernel),
                    search_band_width=int(args.residual_search_band_width),
                    min_pixels=int(args.residual_min_mask_pixels),
                )

            source_key = _csv_source_key(inner)
            for count in micro_counts:
                specs = _micro_specs_from_mask(residual_mask, count) if residual_mask is not None else []
                if not specs:
                    specs = _micro_specs_from_component(inner, count)
                if not specs:
                    continue
                for depth_scale in depth_scales:
                    depth = max(1.0e-5, anchor_depth * float(depth_scale))
                    for radius_scale in radius_scales:
                        for minor_scale in minor_scales:
                            for covariance_scale in covariance_scales:
                                if int(args.max_groups) > 0 and len(groups) >= int(args.max_groups):
                                    break
                                group_index = len(groups)
                                mode = (
                                    f"n{count}_r{radius_scale:g}_m{minor_scale:g}_"
                                    f"d{depth_scale:g}_cg{covariance_scale:g}"
                                ).replace(".", "p")
                                pair_id = f"v368:{source_key}:{mode}:g{group_index}"
                                group_children = []
                                for spec in specs:
                                    target_u = float(spec["screen_x"])
                                    target_v = float(spec["screen_y"])
                                    desired_posed_center = _unproject(target_u, target_v, depth, k, r_dataset, t)
                                    anchored_center = canonical_anchor + local_rot.T @ (desired_posed_center - posed_anchor)
                                    sigma_scale = max(float(covariance_scale), 1.0e-6)
                                    posed_cov = _world_covariance_from_screen(
                                        k=k,
                                        r_dataset=r_dataset,
                                        depth=depth,
                                        major=np.asarray(spec["major"], dtype=np.float64),
                                        minor=np.asarray(spec["minor"], dtype=np.float64),
                                        major_sigma_px=float(spec["major_sigma_px"]) * float(radius_scale) * sigma_scale,
                                        minor_sigma_px=(
                                            float(spec["minor_sigma_px"])
                                            * float(radius_scale)
                                            * float(minor_scale)
                                            * sigma_scale
                                        ),
                                        depth_sigma_px=float(args.depth_sigma_px) * float(radius_scale) * sigma_scale,
                                    )
                                    cov = local_rot.T @ posed_cov @ local_rot
                                    cov = 0.5 * (cov + cov.T)
                                    cov += np.eye(3, dtype=np.float64) * 1.0e-10
                                    eig = np.linalg.eigvalsh(cov)
                                    cov_radius = float(math.sqrt(max(float(eig.max()), 1.0e-12)))
                                    if args.child_opacity_mode == "sqrt":
                                        child_opacity = float(args.child_opacity) / math.sqrt(max(len(specs), 1))
                                    elif args.child_opacity_mode == "divide":
                                        child_opacity = float(args.child_opacity) / max(len(specs), 1)
                                    else:
                                        child_opacity = float(args.child_opacity)
                                    child = {
                                        "component_key": (
                                            f"v368_inner:{_safe_key(source_key)}:{mode}:"
                                            f"g{group_index}:m{int(spec['micro_index'])}"
                                        ),
                                        "source_component_key": source_key,
                                        "source_row_index": int(inner["row_index"]),
                                        "source_component_id": int(inner["component_id"]),
                                        "source_image_name": image_name,
                                        "direction": "inner",
                                        "scope": "global",
                                        "asset_scope": "global",
                                        "image_name": "",
                                        "pair_id": pair_id,
                                        "action_group_key": pair_id,
                                        "pair_role": "inner_residual_supplement",
                                        "pair_required": True,
                                        "split_child_enable": True,
                                        "child_role": "inner_supplement",
                                        "child_color_source": "top_ids_mean",
                                        "child_pose_mode": "semantic_local_frame",
                                        "anchor_mode": "semantic_local_frame",
                                        "anchor_local_frame": True,
                                        "rotate_covariance_with_anchor": True,
                                        "anchor_point_ids": [int(idx) for idx in inner.get("top_point_ids", [])[: int(args.max_top_ids)]],
                                        "top_point_ids": [int(idx) for idx in inner.get("top_point_ids", [])[: int(args.max_top_ids)]],
                                        "source_top_point_ids": [int(idx) for idx in inner.get("top_point_ids", [])],
                                        "anchor_explicit_ids_required": True,
                                        "anchor_owner_gate": True,
                                        "anchor_knn": 24,
                                        "anchor_min_points": 3,
                                        "owner_gate": True,
                                        "activation_required": True,
                                        "activation_direction": "inner",
                                        "activation_pad_px": float(args.activation_pad_px),
                                        "activation_ellipse_scale": float(args.activation_ellipse_scale),
                                        "activation_min_area": 20.0,
                                        "activation_owner_gate": True,
                                        "activation_owner_primary_only": True,
                                        "activation_screen_x": float(target_u),
                                        "activation_screen_y": float(target_v),
                                        "target_screen_x": float(target_u),
                                        "target_screen_y": float(target_v),
                                        "child_self_protect_enable": True,
                                        "child_self_protect_mode": "paired_outer_screen_ellipse",
                                        "child_self_protect_inner_min_overlap": 0.01,
                                        "child_self_protect_outer_max_overlap": 0.0,
                                        "child_self_protect_outer_margin_px": 4.0,
                                        "child_self_protect_inner_radius_fraction": float(args.self_protect_inner_radius_fraction),
                                        "child_self_protect_shrink_factor": float(args.self_protect_shrink_factor),
                                        "child_self_protect_opacity_factor": float(args.self_protect_opacity_factor),
                                        "child_self_protect_drop_on_outer": bool(args.self_protect_drop_on_outer),
                                        "child_opacity": max(0.0, min(float(child_opacity), 1.0)),
                                        "child_radius_scale": 1.0,
                                        "canonical_center": [float(x) for x in anchored_center.tolist()],
                                        "canonical_radius": float(max(cov_radius, 1.0e-5)),
                                        "canonical_covariance": [[float(cov[r, c]) for c in range(3)] for r in range(3)],
                                        "canonical_covariance_6": _cov6(cov),
                                        "reason": "v368_self_protected_residual_grouped_inner_micro_child",
                                        "v368_mode": mode,
                                        "v368_micro_count": int(len(specs)),
                                        "v368_micro_index": int(spec["micro_index"]),
                                        "v368_screen_x": float(target_u),
                                        "v368_screen_y": float(target_v),
                                        "v368_anchor_screen_x": float(anchor_u),
                                        "v368_anchor_screen_y": float(anchor_v),
                                        "v368_anchor_depth": float(anchor_depth),
                                        "v368_depth": float(depth),
                                        "v368_renderer_space_cache": bool(image_name in renderer_space_cache),
                                        "v368_source_bbox_x": float(inner.get("bbox_x", 0.0) or 0.0),
                                        "v368_source_bbox_y": float(inner.get("bbox_y", 0.0) or 0.0),
                                        "v368_source_bbox_w": float(inner.get("bbox_w", 0.0) or 0.0),
                                        "v368_source_bbox_h": float(inner.get("bbox_h", 0.0) or 0.0),
                                        "v368_source_centroid_x": float(inner.get("centroid_x", 0.0) or 0.0),
                                        "v368_source_centroid_y": float(inner.get("centroid_y", 0.0) or 0.0),
                                        "v368_residual_mask_enable": bool(residual_mask is not None),
                                        **{f"v368_{key}": value for key, value in residual_stats.items()},
                                        **inner_owner,
                                    }
                                    if float(args.anchor_radius_scale) > 0.0:
                                        child["anchor_radius"] = float(max(cov_radius, 1.0e-5) * float(args.anchor_radius_scale))
                                    group_children.append(child)

                                group_actions = []
                                for outer, outer_owner in outer_selected:
                                    action = _make_outer_action(
                                        outer=outer,
                                        outer_owner=outer_owner,
                                        pair_id=pair_id,
                                        action_index=len(actions) + len(group_actions),
                                        point_stats=point_stats,
                                        max_top_ids=int(args.max_top_ids),
                                        radius_floor=float(args.radius_floor),
                                        radius_pad=float(args.radius_pad),
                                        radius_scale=float(args.row_radius_scale),
                                        outer_radius_scale=float(args.outer_radius_scale),
                                        outer_score_scale=float(args.outer_score_scale),
                                    )
                                    if action is not None:
                                        group_actions.append(action)
                                if not group_children or not group_actions:
                                    continue
                                group = {
                                    "pair_id": pair_id,
                                    "source_component_key": source_key,
                                    "source_image_name": image_name,
                                    "image_name": image_name,
                                    "mode": mode,
                                    "micro_count": len(group_children),
                                    "outer_action_count": len(group_actions),
                                    "child_component_keys": [str(item["component_key"]) for item in group_children],
                                    "outer_action_keys": [str(item["component_key"]) for item in group_actions],
                                    "source_row_index": int(inner["row_index"]),
                                    "source_component_id": int(inner["component_id"]),
                                    "source_bbox_x": float(inner.get("bbox_x", 0.0) or 0.0),
                                    "source_bbox_y": float(inner.get("bbox_y", 0.0) or 0.0),
                                    "source_bbox_w": float(inner.get("bbox_w", 0.0) or 0.0),
                                    "source_bbox_h": float(inner.get("bbox_h", 0.0) or 0.0),
                                    "source_centroid_x": float(inner.get("centroid_x", 0.0) or 0.0),
                                    "source_centroid_y": float(inner.get("centroid_y", 0.0) or 0.0),
                                    "residual_mask_enable": bool(residual_mask is not None),
                                    **residual_stats,
                                    "frame_score": float(frame["score"]),
                                    **{metric: float(frame.get(metric, 0.0) or 0.0) for metric in METRICS},
                                    **{f"{metric}_delta_base": float(frame.get(f"{metric}_delta_base", 0.0) or 0.0) for metric in METRICS},
                                }
                                children.extend(group_children)
                                actions.extend(group_actions)
                                groups.append(group)
                                owner_counts[_owner_tuple(inner_owner)] += 1
                                audit_rows.append({
                                    **group,
                                    "children": ";".join(group["child_component_keys"]),
                                    "outer_actions": ";".join(group["outer_action_keys"]),
                                    "owner_layer_id": inner_owner.get("owner_layer_id", ""),
                                    "owner_region_id": inner_owner.get("owner_region_id", ""),
                                    "owner_joint": inner_owner.get("owner_joint", ""),
                                    "renderer_space_cache": bool(image_name in renderer_space_cache),
                                })

    payload = {
        "version": "v368_self_protected_residual_grouped_actuator_asset",
        "policy": (
            "Subject-general residual discovery: current raw diagnostics select residual component rows; "
            "inner residual components become global semantic-local-frame micro-children, and paired "
            "outer components become semantic local-3D protect/shrink actions. Split children also carry "
            "a self-protection contract so paired outer activation can shrink/suppress child footprint "
            "before rasterization. All pairs still require action-level raw contour validation before training."
        ),
        "source": {
            "baseline_render_exp": str(args.baseline_render_exp),
            "current_render_exp": str(args.current_render_exp),
            "checkpoint": str(args.checkpoint),
            "renderer_space_cache": str(renderer_space_cache_path or ""),
            "component_csv": str(args.component_csv),
            "point_csv": str(args.point_csv),
            "dataset_root": str(args.dataset_root),
            "subject": str(args.subject),
        },
        "thresholds": {
            "top_frames": int(args.top_frames),
            "inner_per_frame": int(args.inner_per_frame),
            "outer_per_inner": int(args.outer_per_inner),
            "max_groups": int(args.max_groups),
            "micro_counts": micro_counts,
            "radius_scales": radius_scales,
            "minor_scales": minor_scales,
            "depth_scales": depth_scales,
            "covariance_scales": covariance_scales,
            "child_opacity": float(args.child_opacity),
            "child_opacity_mode": str(args.child_opacity_mode),
            "outer_radius_scale": float(args.outer_radius_scale),
            "outer_score_scale": float(args.outer_score_scale),
            "component_match_mode": "semantic_local_3d",
            "residual_mask_enable": bool(args.residual_mask_enable),
            "residual_render_support_threshold": float(args.residual_render_support_threshold),
            "residual_close_kernel": int(args.residual_close_kernel),
            "residual_search_band_width": int(args.residual_search_band_width),
            "residual_min_mask_pixels": int(args.residual_min_mask_pixels),
            "child_self_protect_enable": True,
            "child_self_protect_mode": "paired_outer_screen_ellipse",
            "child_self_protect_outer_max_overlap": 0.0,
            "child_self_protect_inner_radius_fraction": float(args.self_protect_inner_radius_fraction),
            "child_self_protect_shrink_factor": float(args.self_protect_shrink_factor),
            "child_self_protect_opacity_factor": float(args.self_protect_opacity_factor),
            "child_self_protect_drop_on_outer": bool(args.self_protect_drop_on_outer),
        },
        "frame_count": len(frames),
        "group_count": len(groups),
        "child_count": len(children),
        "action_count": len(actions),
        "owner_group_counts": {str(key): int(value) for key, value in owner_counts.items()},
        "action_groups": groups,
        "children": children,
        "actions": actions,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_candidates_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames: list[str] = []
    for row in audit_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with args.out_candidates_tsv.open("w", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(audit_rows)
        else:
            handle.write("pair_id\n")
    print(
        f"wrote {args.out_json} groups={len(groups)} children={len(children)} "
        f"outer_actions={len(actions)} renderer_space_images={len(renderer_space_cache)}"
    )
    print(f"wrote {args.out_candidates_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
