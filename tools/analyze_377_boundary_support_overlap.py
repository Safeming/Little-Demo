#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


RENDER_RE = re.compile(r"render_c(?P<cam>\d+)_f(?P<frame>\d+)\.png$")


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


def _boundary_band(mask: np.ndarray, width: int) -> np.ndarray:
    return _morph(mask, width, "dilate") & (~_morph(mask, width, "erode"))


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


def _support_mask(rgb: np.ndarray, threshold: float) -> np.ndarray:
    # boundary_support maps are orange/yellow on black; use brightness and chroma together.
    luma = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    return (luma > threshold) & (chroma > threshold * 0.35)


def _safe_ratio(num: int | float, den: int | float) -> float:
    den = float(den)
    return float(num) / den if den > 0.0 else 0.0


def _overlay(base: Image.Image, masks: list[tuple[np.ndarray, tuple[int, int, int]]]) -> Image.Image:
    arr = np.asarray(base).copy()
    for mask, color in masks:
        arr[mask] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(arr)


def analyze_sample(render_path: Path, support_path: Path, dataset_root: Path, subject: str, args: argparse.Namespace) -> dict:
    match = RENDER_RE.match(render_path.name)
    if match is None:
        raise ValueError(f"Unexpected render filename: {render_path.name}")
    cam = match.group("cam")
    frame = int(match.group("frame"))
    frame_name = f"{frame:06d}"
    mask_path = dataset_root / subject / cam / f"{frame_name}.png"
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    render_rgb = _read_rgb(render_path)
    size = (render_rgb.shape[1], render_rgb.shape[0])
    support_rgb = _read_rgb(support_path, size=size)
    gt_mask = _read_mask(mask_path, size)

    render_binary = _render_support(render_rgb, args.render_support_threshold, args.close_kernel)
    support_binary = _support_mask(support_rgb, args.support_threshold)
    if args.support_dilate > 0:
        support_near = _morph(support_binary, args.support_dilate, "dilate")
    else:
        support_near = support_binary

    near_gt = _morph(gt_mask, args.search_band_width, "dilate")
    inner_missing = gt_mask & (~render_binary)
    outer_leak = render_binary & (~gt_mask) & near_gt
    boundary = _boundary_band(gt_mask, args.band_width)
    inner_band = inner_missing & boundary
    outer_band = outer_leak & _boundary_band(gt_mask, max(args.band_width * 2, 3))

    support_pixels = int(support_binary.sum())
    support_near_pixels = int(support_near.sum())
    inner_pixels = int(inner_missing.sum())
    outer_pixels = int(outer_leak.sum())
    inner_band_pixels = int(inner_band.sum())
    outer_band_pixels = int(outer_band.sum())
    support_on_inner = int((support_near & inner_missing).sum())
    support_on_inner_band = int((support_near & inner_band).sum())
    support_on_outer = int((support_near & outer_leak).sum())
    support_on_gt = int((support_binary & gt_mask).sum())
    support_on_render = int((support_binary & render_binary).sum())

    return {
        "render": str(render_path),
        "support": str(support_path),
        "cam": cam,
        "frame": frame,
        "support_pixels": support_pixels,
        "support_near_pixels": support_near_pixels,
        "inner_missing_pixels": inner_pixels,
        "outer_leak_pixels": outer_pixels,
        "inner_band_pixels": inner_band_pixels,
        "outer_band_pixels": outer_band_pixels,
        "support_on_inner_pixels": support_on_inner,
        "support_on_inner_band_pixels": support_on_inner_band,
        "support_on_outer_pixels": support_on_outer,
        "support_on_gt_pixels": support_on_gt,
        "support_on_render_pixels": support_on_render,
        "inner_support_coverage": _safe_ratio(support_on_inner, inner_pixels),
        "inner_band_support_coverage": _safe_ratio(support_on_inner_band, inner_band_pixels),
        "outer_support_coverage": _safe_ratio(support_on_outer, outer_pixels),
        "support_precision_gt": _safe_ratio(support_on_gt, support_pixels),
        "support_precision_render": _safe_ratio(support_on_render, support_pixels),
        "_render_rgb": render_rgb,
        "_support_mask": support_binary,
        "_support_near": support_near,
        "_inner_missing": inner_missing,
        "_outer_leak": outer_leak,
    }


def write_montage(records: list[dict], out_path: Path, topk: int, width: int) -> None:
    chosen = records[:topk]
    if not chosen:
        return
    tiles = []
    for record in chosen:
        base = Image.fromarray((record["_render_rgb"] * 255.0).clip(0, 255).astype(np.uint8))
        scale = width / base.size[0]
        height = int(round(base.size[1] * scale))
        base = base.resize((width, height), Image.BILINEAR)
        masks = []
        for key, color in (
            ("_inner_missing", (40, 220, 80)),
            ("_outer_leak", (255, 60, 70)),
            ("_support_near", (255, 220, 40)),
        ):
            mask = Image.fromarray(record[key].astype(np.uint8) * 255).resize((width, height), Image.NEAREST)
            masks.append((np.asarray(mask) > 0, color))
        overlay = _overlay(base, masks)
        label_h = 34
        tile = Image.new("RGB", (width, height + label_h), (16, 16, 16))
        tile.paste(overlay, (0, label_h))
        draw = ImageDraw.Draw(tile)
        draw.text(
            (6, 8),
            (
                f"c{record['cam']} f{record['frame']:06d} "
                f"inner_cov={record['inner_support_coverage']:.3f} "
                f"band_cov={record['inner_band_support_coverage']:.3f} "
                f"support={record['support_pixels']}"
            ),
            fill=(238, 238, 238),
        )
        tiles.append(tile)
    sheet = Image.new("RGB", (max(t.size[0] for t in tiles), sum(t.size[1] for t in tiles)), (12, 12, 12))
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y))
        y += tile.size[1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure overlap between boundary_support binding maps and boundary residual masks.")
    parser.add_argument("--render-exp", required=True, type=Path)
    parser.add_argument("--dataset-root", default="data/ZJUMoCap", type=Path)
    parser.add_argument("--subject", default="CoreView_377")
    parser.add_argument("--split-dir", default="test-view")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--render-support-threshold", type=float, default=0.025)
    parser.add_argument("--support-threshold", type=float, default=0.035)
    parser.add_argument("--support-dilate", type=int, default=4)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--band-width", type=int, default=7)
    parser.add_argument("--search-band-width", type=int, default=24)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--panel-width", type=int, default=260)
    args = parser.parse_args()

    render_dir = args.render_exp / args.split_dir / "renders"
    support_dir = args.render_exp / args.split_dir / "binding_maps" / "boundary_support"
    if not render_dir.exists():
        raise FileNotFoundError(render_dir)
    if not support_dir.exists():
        raise FileNotFoundError(support_dir)
    out_dir = args.out_dir or (args.render_exp / "diagnostics" / f"{args.split_dir}_boundary_support_overlap")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for render_path in sorted(render_dir.glob("render_c*_f*.png")):
        support_path = support_dir / render_path.name
        if not support_path.exists():
            continue
        records.append(analyze_sample(render_path, support_path, args.dataset_root, args.subject, args))
    records.sort(key=lambda item: item["inner_missing_pixels"], reverse=True)
    public = [{k: v for k, v in item.items() if not k.startswith("_")} for item in records]

    summary = {
        "render_exp": str(args.render_exp),
        "split_dir": args.split_dir,
        "n_samples": len(records),
        "mean_support_pixels": float(np.mean([r["support_pixels"] for r in records])) if records else 0.0,
        "mean_inner_support_coverage": float(np.mean([r["inner_support_coverage"] for r in records])) if records else 0.0,
        "mean_inner_band_support_coverage": float(np.mean([r["inner_band_support_coverage"] for r in records])) if records else 0.0,
        "mean_outer_support_coverage": float(np.mean([r["outer_support_coverage"] for r in records])) if records else 0.0,
        "mean_support_precision_gt": float(np.mean([r["support_precision_gt"] for r in records])) if records else 0.0,
        "mean_support_precision_render": float(np.mean([r["support_precision_render"] for r in records])) if records else 0.0,
        "top_samples": public[: args.topk],
    }
    (out_dir / "boundary_support_overlap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "boundary_support_overlap_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        if public:
            writer = csv.DictWriter(handle, fieldnames=list(public[0].keys()))
            writer.writeheader()
            writer.writerows(public)
    write_montage(records, out_dir / "top_boundary_support_overlap.png", args.topk, args.panel_width)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
