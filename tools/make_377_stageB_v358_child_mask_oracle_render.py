#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


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


def _read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(path)


def _write_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(path)


def _ellipse_mask(shape: tuple[int, int], child: dict[str, object], *, pad_px: float, scale: float) -> np.ndarray:
    height, width = shape
    x = float(child.get("bbox_x", 0.0) or 0.0)
    y = float(child.get("bbox_y", 0.0) or 0.0)
    w = max(float(child.get("bbox_w", 0.0) or 0.0), 1.0)
    h = max(float(child.get("bbox_h", 0.0) or 0.0), 1.0)
    cx = float(child.get("centroid_x", x + 0.5 * w) or (x + 0.5 * w))
    cy = float(child.get("centroid_y", y + 0.5 * h) or (y + 0.5 * h))
    rx = max(1.0, 0.5 * w * float(scale) + float(pad_px))
    ry = max(1.0, 0.5 * h * float(scale) + float(pad_px))
    yy, xx = np.ogrid[:height, :width]
    mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    return mask


def _bbox_mask(shape: tuple[int, int], child: dict[str, object], *, pad_px: float, scale: float) -> np.ndarray:
    height, width = shape
    x = float(child.get("bbox_x", 0.0) or 0.0)
    y = float(child.get("bbox_y", 0.0) or 0.0)
    w = max(float(child.get("bbox_w", 0.0) or 0.0), 1.0)
    h = max(float(child.get("bbox_h", 0.0) or 0.0), 1.0)
    cx = float(child.get("centroid_x", x + 0.5 * w) or (x + 0.5 * w))
    cy = float(child.get("centroid_y", y + 0.5 * h) or (y + 0.5 * h))
    half_w = 0.5 * w * float(scale) + float(pad_px)
    half_h = 0.5 * h * float(scale) + float(pad_px)
    x0 = max(0, int(math.floor(cx - half_w)))
    x1 = min(width, int(math.ceil(cx + half_w + 1)))
    y0 = max(0, int(math.floor(cy - half_h)))
    y1 = min(height, int(math.ceil(cy + half_h + 1)))
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _component_mask(shape: tuple[int, int], child: dict[str, object], *, mode: str, pad_px: float, scale: float, dilate_px: int) -> np.ndarray:
    if mode == "bbox":
        mask = _bbox_mask(shape, child, pad_px=pad_px, scale=scale)
    else:
        mask = _ellipse_mask(shape, child, pad_px=pad_px, scale=scale)
    if int(dilate_px) > 0:
        k = int(dilate_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask


def _copy_hydra(src: Path, dst: Path) -> None:
    hydra_src = src / ".hydra"
    hydra_dst = dst / ".hydra"
    if hydra_src.exists() and not hydra_dst.exists():
        shutil.copytree(hydra_src, hydra_dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose a child-side screen mask oracle render from control and split-child renders.")
    parser.add_argument("--control-exp", required=True, type=Path)
    parser.add_argument("--child-exp", required=True, type=Path)
    parser.add_argument("--asset-json", required=True, type=Path)
    parser.add_argument("--component-key", required=True)
    parser.add_argument("--out-exp", required=True, type=Path)
    parser.add_argument("--split-dir", default="test-view")
    parser.add_argument("--mask-mode", default="ellipse", choices=("ellipse", "bbox"))
    parser.add_argument("--mask-pad-px", default=0.0, type=float)
    parser.add_argument("--mask-scale", default=1.0, type=float)
    parser.add_argument("--mask-dilate-px", default=0, type=int)
    parser.add_argument("--write-mask", action="store_true")
    args = parser.parse_args()

    children = _load_children(args.asset_json)
    child = children.get(str(args.component_key))
    if child is None:
        raise KeyError(f"component key not found in asset: {args.component_key}")
    image_name = str(child.get("image_name", "") or "")
    if not image_name:
        raise ValueError(f"asset child has no image_name: {args.component_key}")

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
    mask = _component_mask(
        control_rgb.shape[:2],
        child,
        mode=str(args.mask_mode),
        pad_px=float(args.mask_pad_px),
        scale=float(args.mask_scale),
        dilate_px=int(args.mask_dilate_px),
    )
    mask_f = mask.astype(np.float32)
    composed_rgb = control_rgb * (1.0 - mask_f[..., None]) + child_rgb * mask_f[..., None]
    composed_opacity = control_opacity * (1.0 - mask_f) + child_opacity * mask_f
    _write_rgb(out_render_dir / render_name, composed_rgb)
    _write_gray(out_opacity_dir / opacity_name, composed_opacity)
    if bool(args.write_mask):
        _write_gray(args.out_exp / args.split_dir / "oracle_masks" / f"mask_{image_name}.png", mask_f)
    meta = {
        "control_exp": str(args.control_exp),
        "child_exp": str(args.child_exp),
        "asset_json": str(args.asset_json),
        "component_key": str(args.component_key),
        "image_name": image_name,
        "mask_mode": str(args.mask_mode),
        "mask_pad_px": float(args.mask_pad_px),
        "mask_scale": float(args.mask_scale),
        "mask_dilate_px": int(args.mask_dilate_px),
        "mask_pixels": int(mask.sum()),
        "out_exp": str(args.out_exp),
    }
    args.out_exp.mkdir(parents=True, exist_ok=True)
    (args.out_exp / "v358_child_mask_oracle_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
