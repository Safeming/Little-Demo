#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


IMAGE_RE = re.compile(r"c(?P<cam>\d+)_f(?P<frame>\d+)$")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(value))


def _load_children(path: Path) -> dict[str, dict[str, object]]:
    data = _load_json(path)
    rows = data.get("children") or data.get("actions") or []
    return {str(row["component_key"]): row for row in rows if isinstance(row, dict) and row.get("component_key")}


def _read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def _mask_stats(mask: np.ndarray) -> dict[str, float] | None:
    ys, xs = np.where(mask)
    if xs.size <= 0:
        return None
    return {
        "pixels": int(xs.size),
        "cx": float(xs.mean()),
        "cy": float(ys.mean()),
        "x0": int(xs.min()),
        "y0": int(ys.min()),
        "x1": int(xs.max()),
        "y1": int(ys.max()),
        "w": int(xs.max() - xs.min() + 1),
        "h": int(ys.max() - ys.min() + 1),
    }


def _camera_after_dataset_adjust(
    cam: dict,
    width: int,
    height: int,
    source_width: int = 1024,
    source_height: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = np.array(cam["K"], dtype=np.float64).copy()
    r = np.array(cam["R"], dtype=np.float64).copy()
    t = np.array(cam["T"], dtype=np.float64).copy()
    m = np.eye(3, dtype=np.float64)
    # Match dataset/zjumocap.py exactly: the camera-center correction is applied
    # in the uncropped source resolution, then K is scaled to the render size.
    m[0, 2] = (k[0, 2] - float(source_width) / 2.0) / k[0, 0]
    m[1, 2] = (k[1, 2] - float(source_height) / 2.0) / k[1, 1]
    k[0, 2] = float(source_width) / 2.0
    k[1, 2] = float(source_height) / 2.0
    r = m @ r
    t = m @ t
    k[0, :] *= float(width) / float(source_width)
    k[1, :] *= float(height) / float(source_height)
    return k, r.T, t[:, 0]


def _project(point: np.ndarray, k: np.ndarray, r_dataset: np.ndarray, t: np.ndarray) -> tuple[float, float, float]:
    # Dataset Camera stores R transposed before Camera/getWorld2View2. World-to-camera uses R.T.
    x_cam = r_dataset.T @ point.reshape(3) + t.reshape(3)
    z = float(x_cam[2])
    if abs(z) < 1.0e-8:
        z = 1.0e-8
    u = float(k[0, 0] * x_cam[0] / z + k[0, 2])
    v = float(k[1, 1] * x_cam[1] / z + k[1, 2])
    return u, v, z


def _unproject(u: float, v: float, depth: float, k: np.ndarray, r_dataset: np.ndarray, t: np.ndarray) -> np.ndarray:
    z = max(float(depth), 1.0e-8)
    x_cam = np.array([
        (float(u) - float(k[0, 2])) * z / float(k[0, 0]),
        (float(v) - float(k[1, 2])) * z / float(k[1, 1]),
        z,
    ], dtype=np.float64)
    # The ZJU camera-center adjustment premultiplies the world-to-camera matrix by M,
    # so the linear part is not guaranteed orthonormal.  Use the true inverse rather
    # than R.T; otherwise residual-anchored centers land tens of pixels off target.
    return np.linalg.solve(r_dataset.T, x_cam - t.reshape(3))


def _screen_delta_to_world(dx: float, dy: float, depth: float, k: np.ndarray, r_dataset: np.ndarray) -> np.ndarray:
    delta_cam = np.array([dx * depth / k[0, 0], dy * depth / k[1, 1], 0.0], dtype=np.float64)
    return np.linalg.solve(r_dataset.T, delta_cam)


def _parse_ids(value: object) -> list[int]:
    if isinstance(value, (list, tuple)):
        tokens = value
    else:
        tokens = re.split(r"[,\s;]+", str(value or "").strip())
    out: list[int] = []
    for token in tokens:
        if token == "":
            continue
        try:
            out.append(int(float(token)))
        except Exception:
            pass
    return out


def _load_checkpoint_xyz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ckpt = torch.load(path, map_location="cpu")
    model = ckpt[0] if isinstance(ckpt, (list, tuple)) else ckpt
    if not isinstance(model, (list, tuple)) or len(model) <= 12:
        raise ValueError(f"unexpected checkpoint model payload: {path}")
    posed = model[1].detach().cpu().numpy().astype(np.float64)
    canonical = model[12].detach().cpu().numpy().astype(np.float64)
    if posed.ndim != 2 or posed.shape[1] != 3 or canonical.shape != posed.shape:
        raise ValueError(f"unexpected xyz shapes posed={posed.shape} canonical={canonical.shape}")
    return posed, canonical


def _load_renderer_space_cache(path: Path | None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if path is None or str(path).strip() == "":
        return {}
    payload = np.load(path, allow_pickle=False)
    image_names = [str(value) for value in payload["image_names"].tolist()]
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, image_name in enumerate(image_names):
        posed = payload[f"posed_xyz_{index}"].astype(np.float64)
        canonical = payload[f"canonical_xyz_{index}"].astype(np.float64)
        if posed.ndim != 2 or posed.shape[1] != 3 or canonical.shape != posed.shape:
            raise ValueError(
                f"unexpected renderer-space cache shapes for {image_name}: "
                f"posed={posed.shape} canonical={canonical.shape}"
            )
        cache[image_name] = (posed, canonical)
    return cache


def _rigid_transform_from_correspondences(canonical_pts: np.ndarray, posed_pts: np.ndarray) -> np.ndarray:
    if canonical_pts.shape[0] < 3 or posed_pts.shape[0] < 3:
        return np.eye(3, dtype=np.float64)
    canonical_mean = canonical_pts.mean(axis=0)
    posed_mean = posed_pts.mean(axis=0)
    src = canonical_pts - canonical_mean.reshape(1, 3)
    dst = posed_pts - posed_mean.reshape(1, 3)
    try:
        h = src.T @ dst
        u, _, vh = np.linalg.svd(h)
        rot = vh.T @ u.T
        if np.linalg.det(rot) < 0.0:
            vh = vh.copy()
            vh[-1, :] *= -1.0
            rot = vh.T @ u.T
    except Exception:
        rot = np.eye(3, dtype=np.float64)
    return rot.astype(np.float64)


def _unit(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < 1.0e-12:
        return fallback.astype(np.float64)
    return (vec / norm).astype(np.float64)


def _pca_axes(mask: np.ndarray) -> dict[str, object] | None:
    ys, xs = np.where(mask)
    if xs.size <= 0:
        return None
    coords = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    center = coords.mean(axis=0)
    centered = coords - center.reshape(1, 2)
    if coords.shape[0] >= 3:
        cov = np.cov(centered.T)
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1]
        values = values[order]
        vectors = vectors[:, order]
        major = _unit(vectors[:, 0], np.array([1.0, 0.0], dtype=np.float64))
    else:
        values = np.array([1.0, 1.0], dtype=np.float64)
        major = np.array([1.0, 0.0], dtype=np.float64)
    minor = np.array([-major[1], major[0]], dtype=np.float64)
    major_coord = centered @ major
    minor_coord = centered @ minor
    return {
        "coords": coords,
        "center": center,
        "major": major,
        "minor": minor,
        "major_coord": major_coord,
        "minor_coord": minor_coord,
        "eigenvalues": values,
    }


def _robust_span(values: np.ndarray) -> float:
    if values.size <= 1:
        return 1.0
    lo, hi = np.percentile(values, [8.0, 92.0])
    return max(float(hi - lo), 1.0)


def _micro_specs(mask: np.ndarray, count: int) -> list[dict[str, float]]:
    pca = _pca_axes(mask)
    if pca is None:
        return []
    count = max(1, int(count))
    coords = pca["coords"]
    major = pca["major"]
    minor = pca["minor"]
    major_coord = pca["major_coord"]
    minor_coord = pca["minor_coord"]
    bins = np.linspace(float(major_coord.min()), float(major_coord.max()), count + 1)
    specs: list[dict[str, float]] = []
    for idx in range(count):
        if idx == count - 1:
            selected = (major_coord >= bins[idx]) & (major_coord <= bins[idx + 1])
        else:
            selected = (major_coord >= bins[idx]) & (major_coord < bins[idx + 1])
        if not bool(selected.any()):
            target_t = 0.5 * (bins[idx] + bins[idx + 1])
            nearest = int(np.argmin(np.abs(major_coord - target_t)))
            selected = np.zeros_like(major_coord, dtype=bool)
            selected[nearest] = True
        local_coords = coords[selected]
        local_major = major_coord[selected]
        local_minor = minor_coord[selected]
        center = local_coords.mean(axis=0)
        major_span = _robust_span(local_major)
        minor_span = _robust_span(local_minor)
        specs.append({
            "screen_x": float(center[0]),
            "screen_y": float(center[1]),
            "major_sigma_px": max(1.0, 0.38 * major_span + 0.35),
            "minor_sigma_px": max(0.9, 0.42 * minor_span + 0.35),
            "slice_pixels": int(local_coords.shape[0]),
            "slice_major_span_px": float(major_span),
            "slice_minor_span_px": float(minor_span),
            "pca_major_x": float(major[0]),
            "pca_major_y": float(major[1]),
            "pca_minor_x": float(minor[0]),
            "pca_minor_y": float(minor[1]),
        })
    return specs


def _world_covariance_from_screen(
    *,
    k: np.ndarray,
    r_dataset: np.ndarray,
    depth: float,
    major: np.ndarray,
    minor: np.ndarray,
    major_sigma_px: float,
    minor_sigma_px: float,
    depth_sigma_px: float,
) -> np.ndarray:
    major_step = _screen_delta_to_world(float(major[0]), float(major[1]), depth, k, r_dataset)
    minor_step = _screen_delta_to_world(float(minor[0]), float(minor[1]), depth, k, r_dataset)
    pixel_to_world = 0.5 * (np.linalg.norm(major_step) + np.linalg.norm(minor_step))
    if not np.isfinite(pixel_to_world) or pixel_to_world <= 0.0:
        pixel_to_world = abs(float(depth)) / max(float(k[0, 0]), float(k[1, 1]), 1.0)
    e_major = _unit(major_step, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    e_minor = _unit(minor_step, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    e_depth = _unit(r_dataset @ np.array([0.0, 0.0, 1.0], dtype=np.float64), np.array([0.0, 0.0, 1.0]))
    basis = np.stack([e_major, e_minor, e_depth], axis=1)
    sigmas = np.array([
        max(float(major_sigma_px) * pixel_to_world, 1.0e-5),
        max(float(minor_sigma_px) * pixel_to_world, 1.0e-5),
        max(float(depth_sigma_px) * pixel_to_world, 1.0e-5),
    ], dtype=np.float64)
    cov = basis @ np.diag(sigmas * sigmas) @ basis.T
    cov = 0.5 * (cov + cov.T)
    cov += np.eye(3, dtype=np.float64) * 1.0e-10
    return cov


def _cov6(cov: np.ndarray) -> list[float]:
    return [
        float(cov[0, 0]),
        float(cov[0, 1]),
        float(cov[0, 2]),
        float(cov[1, 1]),
        float(cov[1, 2]),
        float(cov[2, 2]),
    ]


def _candidate_mode(
    count: int,
    offset_scale: float,
    radius_scale: float,
    minor_scale: float,
    depth_scale: float,
    covariance_scale: float,
) -> str:
    mode = (
        f"n{count}_o{offset_scale:g}_r{radius_scale:g}_m{minor_scale:g}_d{depth_scale:g}"
        f"_cg{covariance_scale:g}"
    )
    return mode.replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v361 residual-shaped micro-child split-child group candidates.")
    parser.add_argument("--seed-asset-json", required=True, type=Path)
    parser.add_argument("--v359-log-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--renderer-space-cache", default="")
    parser.add_argument("--dataset-root", default="data/ZJUMoCap", type=Path)
    parser.add_argument("--subject", default="CoreView_377")
    parser.add_argument("--source-width", default=1024, type=int)
    parser.add_argument("--source-height", default=1024, type=int)
    parser.add_argument("--action-keys", default="auto_no_overlap")
    parser.add_argument("--micro-counts", default="5,7")
    parser.add_argument("--offset-scales", default="0.90,1.0")
    parser.add_argument("--radius-scales", default="0.55,0.75")
    parser.add_argument("--minor-scales", default="0.60,0.85")
    parser.add_argument("--depth-scales", default="1.0")
    parser.add_argument("--depth-sigma-px", default=1.5, type=float)
    parser.add_argument("--covariance-scales", default="1.0")
    parser.add_argument("--radius-floor", default=0.0, type=float)
    parser.add_argument("--anchor-radius-scale", default=0.0, type=float)
    parser.add_argument("--max-groups-per-action", default=16, type=int)
    parser.add_argument("--child-opacity", default=0.14, type=float)
    parser.add_argument("--opacity-mode", default="constant", choices=("constant", "sqrt", "divide"))
    parser.add_argument("--pose-mode", default="top_ids_translation", choices=("top_ids_translation", "canonical_local_frame"))
    parser.add_argument("--asset-scope", default="image", choices=("image", "global"))
    parser.add_argument("--activation-required", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--activation-pad-px", default=4.0, type=float)
    parser.add_argument("--activation-ellipse-scale", default=1.15, type=float)
    parser.add_argument("--activation-owner-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anchor-owner-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anchor-explicit-ids-required", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-candidates-tsv", required=True, type=Path)
    args = parser.parse_args()

    seed = _load_children(args.seed_asset_json)
    checkpoint_posed_xyz, checkpoint_canonical_xyz = _load_checkpoint_xyz(args.checkpoint)
    renderer_space_cache_path = Path(args.renderer_space_cache) if str(args.renderer_space_cache).strip() else None
    renderer_space_cache = _load_renderer_space_cache(renderer_space_cache_path)
    cams = _load_json(args.dataset_root / args.subject / "cam_params.json")
    micro_counts = [int(float(x)) for x in args.micro_counts.split(",") if x.strip()]
    offset_scales = [float(x) for x in args.offset_scales.split(",") if x.strip()]
    radius_scales = [float(x) for x in args.radius_scales.split(",") if x.strip()]
    minor_scales = [float(x) for x in args.minor_scales.split(",") if x.strip()]
    depth_scales = [float(x) for x in args.depth_scales.split(",") if x.strip()]
    covariance_scales = [float(x) for x in args.covariance_scales.split(",") if x.strip()]

    if args.action_keys in ("auto_no_overlap", "auto_all"):
        action_dirs = sorted((args.v359_log_dir / "action_validation").glob("*"))
        safe_to_key = {_safe_key(key): key for key in seed}
        action_keys: list[str] = []
        for action_dir in action_dirs:
            if not action_dir.is_dir():
                continue
            key = safe_to_key.get(action_dir.name)
            if key is None:
                continue
            image = key.split(":", 1)[0]
            mask_root = action_dir / "child_footprint_oracle_plus_v345" / "test-view" / "oracle_masks"
            mask_path = mask_root / f"mask_{image}.png"
            target_path = mask_root / f"target_{image}.png"
            footprint_path = mask_root / f"actual_footprint_{image}.png"
            if not (mask_path.exists() and target_path.exists() and footprint_path.exists()):
                continue
            mask = _read_mask(mask_path)
            target = _read_mask(target_path)
            footprint = _read_mask(footprint_path)
            if int(target.sum()) <= 0 or int(footprint.sum()) <= 0:
                continue
            if args.action_keys == "auto_all" or int(mask.sum()) == 0:
                action_keys.append(key)
    else:
        action_keys = [x.strip() for x in args.action_keys.split(",") if x.strip()]

    children: list[dict[str, object]] = []
    groups: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for action_key in action_keys:
        child = seed.get(action_key)
        if child is None:
            continue
        image = str(child.get("image_name", "") or action_key.split(":", 1)[0])
        match = IMAGE_RE.match(image)
        if match is None:
            continue
        cam_name = str(int(match.group("cam")))
        mask_root = args.v359_log_dir / "action_validation" / _safe_key(action_key) / "child_footprint_oracle_plus_v345" / "test-view" / "oracle_masks"
        target = _read_mask(mask_root / f"target_{image}.png")
        footprint = _read_mask(mask_root / f"actual_footprint_{image}.png")
        target_stats = _mask_stats(target)
        footprint_stats = _mask_stats(footprint)
        pca = _pca_axes(target)
        if target_stats is None or footprint_stats is None or pca is None:
            continue

        height, width = target.shape
        k, r_dataset, t = _camera_after_dataset_adjust(
            cams[cam_name],
            width,
            height,
            source_width=int(args.source_width),
            source_height=int(args.source_height),
        )
        posed_xyz, canonical_xyz = renderer_space_cache.get(image, (checkpoint_posed_xyz, checkpoint_canonical_xyz))
        top_ids = [idx for idx in _parse_ids(child.get("top_point_ids") or child.get("source_top_point_ids")) if 0 <= idx < posed_xyz.shape[0]]
        if not top_ids:
            continue
        idx = np.asarray(top_ids, dtype=np.int64)
        posed_anchor = posed_xyz[idx].mean(axis=0)
        canonical_anchor = canonical_xyz[idx].mean(axis=0)
        local_rot = _rigid_transform_from_correspondences(canonical_xyz[idx], posed_xyz[idx])
        canonical_center = np.asarray(child.get("canonical_center", [0.0, 0.0, 0.0]), dtype=np.float64)
        anchor_u, anchor_v, anchor_depth = _project(posed_anchor, k, r_dataset, t)
        if args.pose_mode == "canonical_local_frame":
            base_posed_center = posed_anchor
        else:
            base_posed_center = canonical_center + (posed_anchor - canonical_anchor)
        base_u, base_v, base_depth = _project(base_posed_center, k, r_dataset, t)

        local_groups: list[dict[str, object]] = []
        for count in micro_counts:
            specs = _micro_specs(target, count)
            if not specs:
                continue
            for depth_scale in depth_scales:
                depth = max(1.0e-5, base_depth * depth_scale)
                for offset_scale in offset_scales:
                    for radius_scale in radius_scales:
                        for minor_scale in minor_scales:
                            for covariance_scale in covariance_scales:
                                mode = _candidate_mode(
                                    count,
                                    offset_scale,
                                    radius_scale,
                                    minor_scale,
                                    depth_scale,
                                    covariance_scale,
                                )
                                group_id = f"{action_key}:v361:{mode}:g{len(local_groups)}"
                                group_children = []
                                for micro_index, spec in enumerate(specs):
                                    target_u = float(footprint_stats["cx"]) + float(spec["screen_x"] - footprint_stats["cx"]) * offset_scale
                                    target_v = float(footprint_stats["cy"]) + float(spec["screen_y"] - footprint_stats["cy"]) * offset_scale
                                    if args.pose_mode == "canonical_local_frame":
                                        desired_posed_center = _unproject(target_u, target_v, depth, k, r_dataset, t)
                                        screen_delta_world = desired_posed_center - base_posed_center
                                    else:
                                        dx = float(spec["screen_x"] - footprint_stats["cx"]) * offset_scale
                                        dy = float(spec["screen_y"] - footprint_stats["cy"]) * offset_scale
                                        screen_delta_world = _screen_delta_to_world(dx, dy, depth, k, r_dataset)
                                        desired_posed_center = base_posed_center + screen_delta_world
                                    sigma_scale = max(float(covariance_scale), 1.0e-6)
                                    posed_cov = _world_covariance_from_screen(
                                        k=k,
                                        r_dataset=r_dataset,
                                        depth=depth,
                                        major=np.array([spec["pca_major_x"], spec["pca_major_y"]], dtype=np.float64),
                                        minor=np.array([spec["pca_minor_x"], spec["pca_minor_y"]], dtype=np.float64),
                                        major_sigma_px=float(spec["major_sigma_px"]) * radius_scale * sigma_scale,
                                        minor_sigma_px=float(spec["minor_sigma_px"]) * radius_scale * minor_scale * sigma_scale,
                                        depth_sigma_px=float(args.depth_sigma_px) * radius_scale * sigma_scale,
                                    )
                                    if args.pose_mode == "canonical_local_frame":
                                        anchored_center = canonical_anchor + local_rot.T @ (desired_posed_center - posed_anchor)
                                        cov = local_rot.T @ posed_cov @ local_rot
                                        child_pose_mode = "semantic_local_frame"
                                    else:
                                        anchored_center = canonical_center + screen_delta_world
                                        cov = posed_cov
                                        child_pose_mode = "v361_residual_shape_top_ids_translation"
                                    cov = 0.5 * (cov + cov.T)
                                    cov += np.eye(3, dtype=np.float64) * 1.0e-10
                                    if args.opacity_mode == "sqrt":
                                        child_opacity = float(args.child_opacity) / math.sqrt(max(len(specs), 1))
                                    elif args.opacity_mode == "divide":
                                        child_opacity = float(args.child_opacity) / max(len(specs), 1)
                                    else:
                                        child_opacity = float(args.child_opacity)
                                    cov_radius = float(math.sqrt(max(np.linalg.eigvalsh(cov).max(), 1.0e-12)))
                                    canonical_radius = max(cov_radius, float(args.radius_floor))
                                    item = dict(child)
                                    item["component_key"] = f"{action_key}:v361:{mode}:g{len(local_groups)}:m{micro_index}"
                                    item["source_component_key"] = action_key
                                    item["pair_id"] = group_id
                                    item["action_group_key"] = group_id
                                    if args.asset_scope == "global":
                                        item["scope"] = "global"
                                        item["asset_scope"] = "global"
                                        item["image_name"] = ""
                                    item["split_child_enable"] = True
                                    item["anchor_mode"] = child_pose_mode
                                    item["child_pose_mode"] = child_pose_mode
                                    item["anchor_local_frame"] = args.pose_mode == "canonical_local_frame"
                                    item["rotate_covariance_with_anchor"] = args.pose_mode == "canonical_local_frame"
                                    item["anchor_point_ids"] = [int(x) for x in top_ids]
                                    item["anchor_explicit_ids_required"] = bool(args.anchor_explicit_ids_required)
                                    item["anchor_owner_gate"] = bool(args.anchor_owner_gate)
                                    if float(args.anchor_radius_scale) > 0.0:
                                        item["anchor_radius"] = float(canonical_radius * float(args.anchor_radius_scale))
                                    item["activation_required"] = bool(args.activation_required)
                                    item["activation_direction"] = "inner"
                                    item["activation_pad_px"] = float(args.activation_pad_px)
                                    item["activation_ellipse_scale"] = float(args.activation_ellipse_scale)
                                    item["activation_min_area"] = 1.0
                                    item["activation_owner_gate"] = bool(args.activation_owner_gate)
                                    item["activation_owner_primary_only"] = True
                                    item["child_opacity"] = max(0.0, min(child_opacity, 1.0))
                                    item["child_radius_scale"] = 1.0
                                    item["canonical_center"] = [float(x) for x in anchored_center.tolist()]
                                    item["canonical_radius"] = canonical_radius
                                    item["canonical_covariance"] = [[float(cov[r, c]) for c in range(3)] for r in range(3)]
                                    item["canonical_covariance_6"] = _cov6(cov)
                                    item["reason"] = "v361_residual_shaped_micro_child_search"
                                    item["v361_source_component_key"] = action_key
                                    item["v361_group_id"] = group_id
                                    item["v361_micro_index"] = int(micro_index)
                                    item["v361_micro_count"] = int(len(specs))
                                    item["v361_target_cx"] = float(target_stats["cx"])
                                    item["v361_target_cy"] = float(target_stats["cy"])
                                    item["v361_target_pixels"] = int(target_stats["pixels"])
                                    item["v361_footprint_cx"] = float(footprint_stats["cx"])
                                    item["v361_footprint_cy"] = float(footprint_stats["cy"])
                                    item["v361_micro_screen_x"] = float(spec["screen_x"])
                                    item["v361_micro_screen_y"] = float(spec["screen_y"])
                                    item["v361_offset_scale"] = float(offset_scale)
                                    item["v361_radius_scale"] = float(radius_scale)
                                    item["v361_minor_scale"] = float(minor_scale)
                                    item["v361_depth_scale"] = float(depth_scale)
                                    item["v361_covariance_scale"] = float(covariance_scale)
                                    item["v361_cov_radius"] = float(cov_radius)
                                    item["v361_radius_floor"] = float(args.radius_floor)
                                    item["v361_major_sigma_px"] = float(spec["major_sigma_px"]) * radius_scale * sigma_scale
                                    item["v361_minor_sigma_px"] = float(spec["minor_sigma_px"]) * radius_scale * minor_scale * sigma_scale
                                    item["v361_slice_pixels"] = int(spec["slice_pixels"])
                                    item["v361_base_projected_u"] = float(base_u)
                                    item["v361_base_projected_v"] = float(base_v)
                                    item["v361_base_depth"] = float(base_depth)
                                    item["v361_anchor_projected_u"] = float(anchor_u)
                                    item["v361_anchor_projected_v"] = float(anchor_v)
                                    item["v361_anchor_depth"] = float(anchor_depth)
                                    item["v361_target_projected_u"] = float(target_u)
                                    item["v361_target_projected_v"] = float(target_v)
                                    item["v361_renderer_space_cache"] = bool(image in renderer_space_cache)
                                    item["v361_pose_mode"] = str(args.pose_mode)
                                    item["v361_asset_scope"] = str(args.asset_scope)
                                    group_children.append(item)
                                local_groups.append({
                                    "pair_id": group_id,
                                    "source_component_key": action_key,
                                    "image_name": image,
                                    "mode": mode,
                                    "micro_count": len(group_children),
                                    "child_component_keys": [str(item["component_key"]) for item in group_children],
                                    "children": group_children,
                                    "target_pixels": int(target_stats["pixels"]),
                                    "footprint_pixels": int(footprint_stats["pixels"]),
                                    "target_cx": float(target_stats["cx"]),
                                    "target_cy": float(target_stats["cy"]),
                                    "footprint_cx": float(footprint_stats["cx"]),
                                    "footprint_cy": float(footprint_stats["cy"]),
                                    "base_projected_u": float(base_u),
                                    "base_projected_v": float(base_v),
                                    "base_depth": float(base_depth),
                                })
        local_groups = local_groups[: max(1, int(args.max_groups_per_action))]
        for group in local_groups:
            groups.append({key: value for key, value in group.items() if key != "children"})
            children.extend(group["children"])
            audit_rows.append({
                "pair_id": group["pair_id"],
                "source_component_key": group["source_component_key"],
                "image_name": group["image_name"],
                "mode": group["mode"],
                "micro_count": group["micro_count"],
                "target_pixels": group["target_pixels"],
                "footprint_pixels": group["footprint_pixels"],
                "target_cx": group["target_cx"],
                "target_cy": group["target_cy"],
                "footprint_cx": group["footprint_cx"],
                "footprint_cy": group["footprint_cy"],
                "base_projected_u": group["base_projected_u"],
                "base_projected_v": group["base_projected_v"],
                "base_depth": group["base_depth"],
                "child_component_keys": ";".join(group["child_component_keys"]),
            })

    payload = {
        "version": "v361_residual_shaped_micro_child_search_asset",
        "policy": (
            "Generate action-level split-child groups whose micro-child centers and covariances follow "
            "the target inner residual mask PCA, using v359 actual footprint centroids as the screen calibration."
        ),
        "source": {
            "seed_asset_json": str(args.seed_asset_json),
            "v359_log_dir": str(args.v359_log_dir),
            "checkpoint": str(args.checkpoint),
            "renderer_space_cache": str(renderer_space_cache_path or ""),
            "action_keys": action_keys,
        },
        "thresholds": {
            "micro_counts": micro_counts,
            "offset_scales": offset_scales,
            "radius_scales": radius_scales,
            "minor_scales": minor_scales,
            "depth_scales": depth_scales,
            "depth_sigma_px": float(args.depth_sigma_px),
            "covariance_scales": covariance_scales,
            "radius_floor": float(args.radius_floor),
            "anchor_radius_scale": float(args.anchor_radius_scale),
            "child_opacity": float(args.child_opacity),
            "opacity_mode": str(args.opacity_mode),
            "pose_mode": str(args.pose_mode),
            "asset_scope": str(args.asset_scope),
            "activation_required": bool(args.activation_required),
            "activation_pad_px": float(args.activation_pad_px),
            "activation_ellipse_scale": float(args.activation_ellipse_scale),
            "activation_owner_gate": bool(args.activation_owner_gate),
            "anchor_owner_gate": bool(args.anchor_owner_gate),
            "anchor_explicit_ids_required": bool(args.anchor_explicit_ids_required),
            "source_width": int(args.source_width),
            "source_height": int(args.source_height),
            "renderer_space_image_count": len(renderer_space_cache),
            "max_groups_per_action": int(args.max_groups_per_action),
        },
        "group_count": len(groups),
        "child_count": len(children),
        "action_groups": groups,
        "children": children,
        "actions": children,
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
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(audit_rows)
        else:
            handle.write("pair_id\n")
    print(f"wrote {args.out_json} groups={len(groups)} children={len(children)} actions={len(action_keys)}")
    print(f"wrote {args.out_candidates_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
