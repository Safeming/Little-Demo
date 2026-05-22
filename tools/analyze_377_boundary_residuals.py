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


def _component_count(mask: np.ndarray, min_area: int) -> int:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    count = 0
    for idx in range(1, n):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area:
            count += 1
    return count


def analyze_sample(render_path: Path, dataset_root: Path, subject: str, args: argparse.Namespace) -> dict:
    match = RENDER_RE.match(render_path.name)
    if match is None:
        raise ValueError(f"Unexpected render filename: {render_path.name}")
    cam = match.group("cam")
    frame = int(match.group("frame"))
    frame_name = f"{frame:06d}"
    gt_path = dataset_root / subject / cam / f"{frame_name}.jpg"
    mask_path = dataset_root / subject / cam / f"{frame_name}.png"
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    render_rgb = _read_rgb(render_path)
    size = (render_rgb.shape[1], render_rgb.shape[0])
    mask = _read_mask(mask_path, size)
    render_support = _render_support(render_rgb, args.render_support_threshold, args.close_kernel)

    near_gt = _morph(mask, args.search_band_width, "dilate")
    inner_missing = mask & (~render_support)
    outer_leak = render_support & (~mask) & near_gt
    band = _boundary_band(mask, args.band_width)
    inner_band = inner_missing & band
    outer_band = outer_leak & _boundary_band(mask, max(args.band_width * 2, 3))

    fg_pixels = max(1, int(mask.sum()))
    return {
        "render": str(render_path),
        "cam": cam,
        "frame": frame,
        "fg_pixels": int(mask.sum()),
        "render_support_pixels": int(render_support.sum()),
        "inner_missing_pixels": int(inner_missing.sum()),
        "outer_leak_pixels": int(outer_leak.sum()),
        "inner_band_pixels": int(inner_band.sum()),
        "outer_band_pixels": int(outer_band.sum()),
        "inner_missing_ratio": float(inner_missing.sum() / fg_pixels),
        "outer_leak_ratio": float(outer_leak.sum() / fg_pixels),
        "inner_components": _component_count(inner_missing, args.min_component_area),
        "outer_components": _component_count(outer_leak, args.min_component_area),
        "hard_residual_score": float((inner_band.sum() + outer_band.sum()) / fg_pixels),
        "_gt_path": str(gt_path),
        "_mask_path": str(mask_path),
        "_render_support": render_support,
        "_inner_missing": inner_missing,
        "_outer_leak": outer_leak,
    }


def _overlay(base: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    image = np.asarray(base).copy()
    image[mask] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(image)


def _tile(record: dict, width: int) -> Image.Image:
    render = Image.open(record["render"]).convert("RGB")
    scale = width / render.size[0]
    height = int(round(render.size[1] * scale))
    render_small = render.resize((width, height), Image.BILINEAR)
    gt = Image.open(record["_gt_path"]).convert("RGB").resize((width, height), Image.BILINEAR)
    mask = Image.open(record["_mask_path"]).convert("L").resize((width, height), Image.NEAREST)
    size = (width, height)
    support = Image.fromarray(record["_render_support"].astype(np.uint8) * 255).resize(size, Image.NEAREST)
    inner = Image.fromarray(record["_inner_missing"].astype(np.uint8) * 255).resize(size, Image.NEAREST)
    outer = Image.fromarray(record["_outer_leak"].astype(np.uint8) * 255).resize(size, Image.NEAREST)

    mask_np = np.asarray(mask) > 0
    support_np = np.asarray(support) > 0
    inner_np = np.asarray(inner) > 0
    outer_np = np.asarray(outer) > 0

    gt_overlay = _overlay(gt, mask_np, (80, 180, 255))
    support_overlay = _overlay(render_small, support_np, (255, 210, 60))
    inner_overlay = _overlay(render_small, inner_np, (50, 220, 80))
    outer_overlay = _overlay(render_small, outer_np, (255, 50, 60))

    label_h = 30
    cols = [gt_overlay, render_small, support_overlay, inner_overlay, outer_overlay]
    names = ["GT mask", "Render", "Render support", "Inner missing", "Outer leak"]
    tile = Image.new("RGB", (width * len(cols), height + label_h), (18, 18, 18))
    draw = ImageDraw.Draw(tile)
    for idx, (name, image) in enumerate(zip(names, cols)):
        x = idx * width
        tile.paste(image, (x, label_h))
        draw.text((x + 6, 7), name, fill=(238, 238, 238))
    draw.text(
        (8, height + label_h - 18),
        (
            f"c{record['cam']} f{record['frame']:06d} "
            f"inner={record['inner_missing_pixels']} outer={record['outer_leak_pixels']} "
            f"score={record['hard_residual_score']:.4f}"
        ),
        fill=(255, 230, 130),
    )
    return tile


def write_montage(records: list[dict], out_path: Path, topk: int, width: int) -> None:
    chosen = records[:topk]
    if not chosen:
        return
    tiles = [_tile(record, width) for record in chosen]
    sheet = Image.new("RGB", (max(t.size[0] for t in tiles), sum(t.size[1] for t in tiles)), (12, 12, 12))
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y))
        y += tile.size[1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine inner-missing and outer-leak boundary residuals for CoreView_377 renders.")
    parser.add_argument("--render-exp", required=True, type=Path)
    parser.add_argument("--dataset-root", default="data/ZJUMoCap", type=Path)
    parser.add_argument("--subject", default="CoreView_377")
    parser.add_argument("--split-dir", default="test-view")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--render-support-threshold", type=float, default=0.025)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--band-width", type=int, default=7)
    parser.add_argument("--search-band-width", type=int, default=24)
    parser.add_argument("--min-component-area", type=int, default=18)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--panel-width", type=int, default=220)
    args = parser.parse_args()

    render_dir = args.render_exp / args.split_dir / "renders"
    if not render_dir.exists():
        raise FileNotFoundError(render_dir)
    out_dir = args.out_dir or (args.render_exp / "diagnostics" / "boundary_residuals")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [
        analyze_sample(path, args.dataset_root, args.subject, args)
        for path in sorted(render_dir.glob("render_c*_f*.png"))
    ]
    records.sort(key=lambda item: item["hard_residual_score"], reverse=True)
    public = [{k: v for k, v in item.items() if not k.startswith("_")} for item in records]

    summary = {
        "render_exp": str(args.render_exp),
        "n_samples": len(records),
        "mean_inner_missing_pixels": float(np.mean([r["inner_missing_pixels"] for r in records])) if records else 0.0,
        "mean_outer_leak_pixels": float(np.mean([r["outer_leak_pixels"] for r in records])) if records else 0.0,
        "mean_inner_missing_ratio": float(np.mean([r["inner_missing_ratio"] for r in records])) if records else 0.0,
        "mean_outer_leak_ratio": float(np.mean([r["outer_leak_ratio"] for r in records])) if records else 0.0,
        "mean_hard_residual_score": float(np.mean([r["hard_residual_score"] for r in records])) if records else 0.0,
        "top_samples": public[: args.topk],
    }
    (out_dir / "boundary_residual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "boundary_residual_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        if public:
            writer = csv.DictWriter(handle, fieldnames=list(public[0].keys()))
            writer.writeheader()
            writer.writerows(public)
    write_montage(records, out_dir / "top_boundary_residuals.png", args.topk, args.panel_width)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
