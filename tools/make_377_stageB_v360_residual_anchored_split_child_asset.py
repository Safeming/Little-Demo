#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


IMAGE_RE = re.compile(r"c(?P<cam>\d+)_f(?P<frame>\d+)$")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    # Dataset Camera stores R as transposed before Camera/getWorld2View2. World-to-camera uses R.T.
    x_cam = r_dataset.T @ point.reshape(3) + t.reshape(3)
    z = float(x_cam[2])
    if abs(z) < 1.0e-8:
        z = 1.0e-8
    u = float(k[0, 0] * x_cam[0] / z + k[0, 2])
    v = float(k[1, 1] * x_cam[1] / z + k[1, 2])
    return u, v, z


def _screen_delta_to_world(dx: float, dy: float, depth: float, k: np.ndarray, r_dataset: np.ndarray) -> np.ndarray:
    delta_cam = np.array([dx * depth / k[0, 0], dy * depth / k[1, 1], 0.0], dtype=np.float64)
    return np.linalg.solve(r_dataset.T, delta_cam)


def _parse_ids(value: object) -> list[int]:
    if isinstance(value, (list, tuple)):
        tokens = value
    else:
        tokens = re.split(r"[,\s]+", str(value or "").strip())
    out = []
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


def _candidate_key(base_key: str, index: int, mode: str) -> str:
    return f"{base_key}:v360:{mode}:k{index}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build residual-anchored split-child candidate asset for v360.")
    parser.add_argument("--seed-asset-json", required=True, type=Path)
    parser.add_argument("--v359-log-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", default="data/ZJUMoCap", type=Path)
    parser.add_argument("--subject", default="CoreView_377")
    parser.add_argument("--action-keys", default="auto_no_overlap")
    parser.add_argument("--offset-scales", default="0.65,0.85,1.0,1.15")
    parser.add_argument("--radius-scales", default="0.70,0.90,1.10")
    parser.add_argument("--depth-scales", default="0.98,1.0,1.02")
    parser.add_argument("--max-candidates-per-action", default=18, type=int)
    parser.add_argument("--child-opacity", default=0.18, type=float)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-candidates-tsv", required=True, type=Path)
    args = parser.parse_args()

    seed = _load_children(args.seed_asset_json)
    posed_xyz, canonical_xyz = _load_checkpoint_xyz(args.checkpoint)
    cams = _load_json(args.dataset_root / args.subject / "cam_params.json")
    offset_scales = [float(x) for x in args.offset_scales.split(",") if x.strip()]
    radius_scales = [float(x) for x in args.radius_scales.split(",") if x.strip()]
    depth_scales = [float(x) for x in args.depth_scales.split(",") if x.strip()]

    if args.action_keys == "auto_no_overlap":
        action_dirs = sorted((args.v359_log_dir / "action_validation").glob("*"))
        action_keys = []
        safe_to_key = {re.sub(r"[^A-Za-z0-9_]", "_", key): key for key in seed}
        for action_dir in action_dirs:
            if not action_dir.is_dir():
                continue
            key = safe_to_key.get(action_dir.name)
            if key is None:
                continue
            image = key.split(":", 1)[0]
            mask_path = action_dir / "child_footprint_oracle_plus_v345" / "test-view" / "oracle_masks" / f"mask_{image}.png"
            target_path = action_dir / "child_footprint_oracle_plus_v345" / "test-view" / "oracle_masks" / f"target_{image}.png"
            footprint_path = action_dir / "child_footprint_oracle_plus_v345" / "test-view" / "oracle_masks" / f"actual_footprint_{image}.png"
            if not (mask_path.exists() and target_path.exists() and footprint_path.exists()):
                continue
            if int(_read_mask(mask_path).sum()) == 0 and int(_read_mask(target_path).sum()) > 0 and int(_read_mask(footprint_path).sum()) > 0:
                action_keys.append(key)
    else:
        action_keys = [x.strip() for x in args.action_keys.split(",") if x.strip()]

    candidates: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for action_key in action_keys:
        child = seed.get(action_key)
        if child is None:
            continue
        image = str(child.get("image_name", "") or action_key.split(":", 1)[0])
        match = IMAGE_RE.match(image)
        if match is None:
            continue
        cam_name = str(int(match.group("cam")))
        mask_root = args.v359_log_dir / "action_validation" / re.sub(r"[^A-Za-z0-9_]", "_", action_key) / "child_footprint_oracle_plus_v345" / "test-view" / "oracle_masks"
        target = _read_mask(mask_root / f"target_{image}.png")
        footprint = _read_mask(mask_root / f"actual_footprint_{image}.png")
        target_stats = _mask_stats(target)
        footprint_stats = _mask_stats(footprint)
        if target_stats is None or footprint_stats is None:
            continue

        height, width = target.shape
        k, r_dataset, t = _camera_after_dataset_adjust(cams[cam_name], width, height)
        top_ids = [idx for idx in _parse_ids(child.get("top_point_ids") or child.get("source_top_point_ids")) if 0 <= idx < posed_xyz.shape[0]]
        if not top_ids:
            continue
        idx = np.asarray(top_ids, dtype=np.int64)
        posed_anchor = posed_xyz[idx].mean(axis=0)
        canonical_anchor = canonical_xyz[idx].mean(axis=0)
        canonical_center = np.asarray(child.get("canonical_center", [0.0, 0.0, 0.0]), dtype=np.float64)
        base_posed_center = canonical_center + (posed_anchor - canonical_anchor)
        base_u, base_v, base_depth = _project(base_posed_center, k, r_dataset, t)
        dx = float(target_stats["cx"] - footprint_stats["cx"])
        dy = float(target_stats["cy"] - footprint_stats["cy"])
        target_radius_px = max(1.0, 0.5 * max(float(target_stats["w"]), float(target_stats["h"])))

        local = []
        for depth_scale in depth_scales:
            depth = max(1.0e-5, base_depth * depth_scale)
            for offset_scale in offset_scales:
                world_delta = _screen_delta_to_world(dx * offset_scale, dy * offset_scale, depth, k, r_dataset)
                anchored_center = canonical_center + world_delta
                _, _, anchored_depth = _project(anchored_center + (posed_anchor - canonical_anchor), k, r_dataset, t)
                base_radius = max(float(child.get("canonical_radius", 0.01) or 0.01), 1.0e-5)
                pixel_to_world = abs(anchored_depth) / max(float(k[0, 0]), float(k[1, 1]))
                residual_radius = max(base_radius * 0.5, target_radius_px * pixel_to_world)
                for radius_scale in radius_scales:
                    item = dict(child)
                    mode = f"d{depth_scale:g}_o{offset_scale:g}_r{radius_scale:g}".replace(".", "p")
                    item["component_key"] = _candidate_key(action_key, len(local), mode)
                    item["source_component_key"] = action_key
                    item["pair_id"] = action_key
                    item["split_child_enable"] = True
                    item["child_pose_mode"] = "v360_residual_anchor_top_ids_translation"
                    item["child_opacity"] = float(args.child_opacity)
                    item["canonical_center"] = [float(x) for x in anchored_center.tolist()]
                    item["canonical_radius"] = float(residual_radius * radius_scale)
                    item["child_radius_scale"] = 1.0
                    item["reason"] = "v360_residual_anchored_split_child_search"
                    item["v360_source_component_key"] = action_key
                    item["v360_target_cx"] = float(target_stats["cx"])
                    item["v360_target_cy"] = float(target_stats["cy"])
                    item["v360_footprint_cx"] = float(footprint_stats["cx"])
                    item["v360_footprint_cy"] = float(footprint_stats["cy"])
                    item["v360_dx_px"] = float(dx)
                    item["v360_dy_px"] = float(dy)
                    item["v360_offset_scale"] = float(offset_scale)
                    item["v360_depth_scale"] = float(depth_scale)
                    item["v360_radius_scale"] = float(radius_scale)
                    item["v360_base_projected_u"] = float(base_u)
                    item["v360_base_projected_v"] = float(base_v)
                    item["v360_base_depth"] = float(base_depth)
                    local.append(item)
                    audit.append({
                        "component_key": item["component_key"],
                        "source_component_key": action_key,
                        "image_name": image,
                        "target_pixels": int(target_stats["pixels"]),
                        "footprint_pixels": int(footprint_stats["pixels"]),
                        "dx_px": float(dx),
                        "dy_px": float(dy),
                        "offset_scale": float(offset_scale),
                        "depth_scale": float(depth_scale),
                        "radius_scale": float(radius_scale),
                        "canonical_radius": item["canonical_radius"],
                    })
        candidates.extend(local[: max(1, int(args.max_candidates_per_action))])

    payload = {
        "version": "v360_residual_anchored_split_child_search_asset",
        "policy": "Generate split-child candidates by shifting child canonical centers so actual v359 child footprint targets control inner residual.",
        "source": {
            "seed_asset_json": str(args.seed_asset_json),
            "v359_log_dir": str(args.v359_log_dir),
            "checkpoint": str(args.checkpoint),
            "action_keys": action_keys,
        },
        "child_count": len(candidates),
        "children": candidates,
        "actions": candidates,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_candidates_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = []
    for row in audit:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with args.out_candidates_tsv.open("w", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(audit)
        else:
            handle.write("component_key\n")
    print(f"wrote {args.out_json} candidates={len(candidates)} actions={len(action_keys)}")
    print(f"wrote {args.out_candidates_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
