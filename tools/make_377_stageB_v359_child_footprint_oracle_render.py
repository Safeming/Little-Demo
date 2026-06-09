#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_RE = re.compile(r"c(?P<cam>\d+)_f(?P<frame>\d+)$")


def _load_children(path: Path) -> dict[str, dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("children") or data.get("actions") or []
    out: dict[str, dict[str, object]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("component_key"):
            out[str(row["component_key"])] = row
    return out


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _read_gray(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.NEAREST)
    return np.asarray(image, dtype=np.float32) / 255.0


def _write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(path)


def _write_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(path)


def _morph(mask: np.ndarray, width: int, mode: str) -> np.ndarray:
    width = max(1, int(width))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    op = cv2.dilate if mode == "dilate" else cv2.erode
    return op(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


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


def _component_prior_mask(shape: tuple[int, int], child: dict[str, object], *, pad_px: float) -> np.ndarray:
    height, width = shape
    x = float(child.get("bbox_x", 0.0) or 0.0)
    y = float(child.get("bbox_y", 0.0) or 0.0)
    w = max(float(child.get("bbox_w", 0.0) or 0.0), 1.0)
    h = max(float(child.get("bbox_h", 0.0) or 0.0), 1.0)
    x0 = max(0, int(math.floor(x - pad_px)))
    y0 = max(0, int(math.floor(y - pad_px)))
    x1 = min(width, int(math.ceil(x + w + pad_px + 1)))
    y1 = min(height, int(math.ceil(y + h + pad_px + 1)))
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _select_target_component(inner_missing: np.ndarray, prior: np.ndarray, min_overlap: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inner_missing.astype(np.uint8), 8)
    selected = np.zeros_like(inner_missing, dtype=bool)
    for idx in range(1, n):
        comp = labels == idx
        overlap = int((comp & prior).sum())
        if overlap >= int(min_overlap):
            selected |= comp
    return selected


def _copy_hydra(src: Path, dst: Path) -> None:
    hydra_src = src / ".hydra"
    hydra_dst = dst / ".hydra"
    if hydra_src.exists() and not hydra_dst.exists():
        shutil.copytree(hydra_src, hydra_dst)


def _parse_image_name(image_name: str) -> tuple[str, str]:
    match = IMAGE_RE.match(image_name)
    if match is None:
        raise ValueError(f"unexpected image_name: {image_name}")
    return match.group("cam"), f"{int(match.group('frame')):06d}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compose a child contribution footprint oracle render from control and split-child renders."
    )
    parser.add_argument("--control-exp", required=True, type=Path)
    parser.add_argument("--child-exp", required=True, type=Path)
    parser.add_argument("--asset-json", required=True, type=Path)
    parser.add_argument("--component-key", required=True)
    parser.add_argument("--out-exp", required=True, type=Path)
    parser.add_argument("--dataset-root", default="data/ZJUMoCap", type=Path)
    parser.add_argument("--subject", default="CoreView_377")
    parser.add_argument("--split-dir", default="test-view")
    parser.add_argument("--render-support-threshold", default=0.025, type=float)
    parser.add_argument("--close-kernel", default=5, type=int)
    parser.add_argument("--rgb-delta-threshold", default=0.0035, type=float)
    parser.add_argument("--opacity-delta-threshold", default=0.0035, type=float)
    parser.add_argument("--component-pad-px", default=6.0, type=float)
    parser.add_argument("--target-dilate-px", default=0, type=int)
    parser.add_argument("--target-mode", default="component_overlap", choices=("component_overlap", "inner_missing"))
    parser.add_argument("--min-component-overlap", default=1, type=int)
    parser.add_argument("--write-mask", action="store_true")
    args = parser.parse_args()

    children = _load_children(args.asset_json)
    child = children.get(str(args.component_key))
    if child is None:
        raise KeyError(f"component key not found in asset: {args.component_key}")
    image_name = str(child.get("image_name", "") or "")
    if not image_name:
        image_name = str(args.component_key).split(":", 1)[0]
    cam, frame_name = _parse_image_name(image_name)

    control_render_dir = args.control_exp / args.split_dir / "renders"
    child_render_dir = args.child_exp / args.split_dir / "renders"
    control_opacity_dir = args.control_exp / args.split_dir / "opacity"
    child_opacity_dir = args.child_exp / args.split_dir / "opacity"
    out_render_dir = args.out_exp / args.split_dir / "renders"
    out_opacity_dir = args.out_exp / args.split_dir / "opacity"
    _copy_hydra(args.control_exp, args.out_exp)

    render_name = f"render_{image_name}.png"
    opacity_name = f"opacity_{image_name}.png"
    control_rgb = _read_rgb(control_render_dir / render_name)
    child_rgb = _read_rgb(child_render_dir / render_name)
    control_opacity = _read_gray(control_opacity_dir / opacity_name)
    child_opacity = _read_gray(child_opacity_dir / opacity_name)
    size = (control_rgb.shape[1], control_rgb.shape[0])
    gt_mask = _read_gray(args.dataset_root / args.subject / cam / f"{frame_name}.png", size=size) > 0.0

    control_support = _render_support(control_rgb, args.render_support_threshold, args.close_kernel)
    inner_missing = gt_mask & (~control_support)
    if str(args.target_mode) == "component_overlap":
        prior = _component_prior_mask(inner_missing.shape, child, pad_px=float(args.component_pad_px))
        target = _select_target_component(inner_missing, prior, int(args.min_component_overlap))
    else:
        prior = np.zeros_like(inner_missing, dtype=bool)
        target = inner_missing
    if int(args.target_dilate_px) > 0:
        target = _morph(target, int(args.target_dilate_px), "dilate") & gt_mask

    rgb_delta = np.max(np.abs(child_rgb - control_rgb), axis=2)
    opacity_delta = child_opacity - control_opacity
    actual_footprint = (rgb_delta > float(args.rgb_delta_threshold)) | (
        opacity_delta > float(args.opacity_delta_threshold)
    )
    mask = actual_footprint & target

    mask_f = mask.astype(np.float32)
    composed_rgb = control_rgb * (1.0 - mask_f[..., None]) + child_rgb * mask_f[..., None]
    composed_opacity = control_opacity * (1.0 - mask_f) + child_opacity * mask_f
    _write_rgb(out_render_dir / render_name, composed_rgb)
    _write_gray(out_opacity_dir / opacity_name, composed_opacity)
    if bool(args.write_mask):
        mask_root = args.out_exp / args.split_dir / "oracle_masks"
        _write_gray(mask_root / f"actual_footprint_{image_name}.png", actual_footprint.astype(np.float32))
        _write_gray(mask_root / f"target_{image_name}.png", target.astype(np.float32))
        _write_gray(mask_root / f"mask_{image_name}.png", mask_f)

    meta = {
        "control_exp": str(args.control_exp),
        "child_exp": str(args.child_exp),
        "asset_json": str(args.asset_json),
        "component_key": str(args.component_key),
        "image_name": image_name,
        "target_mode": str(args.target_mode),
        "render_support_threshold": float(args.render_support_threshold),
        "close_kernel": int(args.close_kernel),
        "rgb_delta_threshold": float(args.rgb_delta_threshold),
        "opacity_delta_threshold": float(args.opacity_delta_threshold),
        "component_pad_px": float(args.component_pad_px),
        "target_dilate_px": int(args.target_dilate_px),
        "inner_missing_pixels": int(inner_missing.sum()),
        "target_pixels": int(target.sum()),
        "actual_footprint_pixels": int(actual_footprint.sum()),
        "mask_pixels": int(mask.sum()),
        "out_exp": str(args.out_exp),
    }
    args.out_exp.mkdir(parents=True, exist_ok=True)
    (args.out_exp / "v359_child_footprint_oracle_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
