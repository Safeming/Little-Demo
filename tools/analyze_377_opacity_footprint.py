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


def _read_gray(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
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


def _opacity_support(opacity: np.ndarray, threshold: float, close_kernel: int) -> np.ndarray:
    support = opacity > float(threshold)
    if close_kernel > 1:
        k = int(close_kernel)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        support = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return support


def _component_count(mask: np.ndarray, min_area: int) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return int(sum(int(stats[idx, cv2.CC_STAT_AREA]) >= min_area for idx in range(1, n)))


def _opacity_path_for(render_path: Path, opacity_dir: Path) -> Path:
    return opacity_dir / render_path.name.replace("render_", "opacity_", 1)


def analyze_sample(render_path: Path, opacity_dir: Path, dataset_root: Path, subject: str, args: argparse.Namespace) -> dict:
    match = RENDER_RE.match(render_path.name)
    if match is None:
        raise ValueError(f"Unexpected render filename: {render_path.name}")
    cam = match.group("cam")
    frame = int(match.group("frame"))
    frame_name = f"{frame:06d}"
    mask_path = dataset_root / subject / cam / f"{frame_name}.png"
    gt_path = dataset_root / subject / cam / f"{frame_name}.jpg"
    opacity_path = _opacity_path_for(render_path, opacity_dir)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    if not opacity_path.exists():
        raise FileNotFoundError(opacity_path)

    render_rgb = _read_rgb(render_path)
    size = (render_rgb.shape[1], render_rgb.shape[0])
    opacity = _read_gray(opacity_path, size)
    gt_mask = _read_mask(mask_path, size)
    near_gt = _morph(gt_mask, args.search_band_width, "dilate")
    band = _boundary_band(gt_mask, args.band_width)
    wide_band = _boundary_band(gt_mask, max(args.band_width * 2, 3))

    rgb_support = _render_support(render_rgb, args.render_support_threshold, args.rgb_close_kernel)
    rgb_inner = gt_mask & (~rgb_support)
    rgb_outer = rgb_support & (~gt_mask) & near_gt
    rgb_inner_band = rgb_inner & band
    rgb_outer_band = rgb_outer & wide_band

    primary = {}
    threshold_records = []
    for threshold in args.opacity_thresholds:
        op_support = _opacity_support(opacity, threshold, args.opacity_close_kernel)
        op_inner = gt_mask & (~op_support)
        op_outer = op_support & (~gt_mask) & near_gt
        opacity_on_rgb_inner = op_support & rgb_inner
        both_missing = rgb_inner & op_inner
        rgb_outer_with_opacity = rgb_outer & op_support
        record = {
            "threshold": float(threshold),
            "opacity_support_pixels": int(op_support.sum()),
            "opacity_inner_missing_pixels": int(op_inner.sum()),
            "opacity_outer_leak_pixels": int(op_outer.sum()),
            "opacity_inner_band_pixels": int((op_inner & band).sum()),
            "opacity_outer_band_pixels": int((op_outer & wide_band).sum()),
            "opacity_inner_components": _component_count(op_inner, args.min_component_area),
            "opacity_outer_components": _component_count(op_outer, args.min_component_area),
            "opacity_on_rgb_inner_pixels": int(opacity_on_rgb_inner.sum()),
            "opacity_on_rgb_inner_ratio": float(opacity_on_rgb_inner.sum() / max(1, int(rgb_inner.sum()))),
            "both_rgb_opacity_inner_missing_pixels": int(both_missing.sum()),
            "both_rgb_opacity_inner_missing_ratio": float(both_missing.sum() / max(1, int(rgb_inner.sum()))),
            "rgb_outer_with_opacity_pixels": int(rgb_outer_with_opacity.sum()),
            "rgb_outer_with_opacity_ratio": float(rgb_outer_with_opacity.sum() / max(1, int(rgb_outer.sum()))),
        }
        threshold_records.append(record)
        if abs(float(threshold) - float(args.primary_opacity_threshold)) < 1.0e-9:
            primary = record

    if not primary:
        primary = threshold_records[0]

    fg_pixels = max(1, int(gt_mask.sum()))
    return {
        "render": str(render_path),
        "opacity": str(opacity_path),
        "gt": str(gt_path),
        "mask": str(mask_path),
        "cam": cam,
        "frame": frame,
        "fg_pixels": int(gt_mask.sum()),
        "rgb_support_pixels": int(rgb_support.sum()),
        "rgb_inner_missing_pixels": int(rgb_inner.sum()),
        "rgb_outer_leak_pixels": int(rgb_outer.sum()),
        "rgb_inner_band_pixels": int(rgb_inner_band.sum()),
        "rgb_outer_band_pixels": int(rgb_outer_band.sum()),
        "rgb_inner_missing_ratio": float(rgb_inner.sum() / fg_pixels),
        "rgb_outer_leak_ratio": float(rgb_outer.sum() / fg_pixels),
        "rgb_inner_components": _component_count(rgb_inner, args.min_component_area),
        "rgb_outer_components": _component_count(rgb_outer, args.min_component_area),
        "mean_opacity_in_gt": float(opacity[gt_mask].mean()) if gt_mask.any() else 0.0,
        "mean_opacity_near_outer": float(opacity[(~gt_mask) & near_gt].mean()) if ((~gt_mask) & near_gt).any() else 0.0,
        "primary_opacity_threshold": float(primary["threshold"]),
        **{f"primary_{key}": value for key, value in primary.items() if key != "threshold"},
        "threshold_records": threshold_records,
        "_render_rgb": render_rgb,
        "_opacity_array": opacity,
        "_rgb_support": rgb_support,
        "_rgb_inner": rgb_inner,
        "_rgb_outer": rgb_outer,
        "_primary_opacity_support": _opacity_support(opacity, primary["threshold"], args.opacity_close_kernel),
        "_primary_opacity_inner": gt_mask & (~_opacity_support(opacity, primary["threshold"], args.opacity_close_kernel)),
        "_primary_opacity_outer": _opacity_support(opacity, primary["threshold"], args.opacity_close_kernel) & (~gt_mask) & near_gt,
    }


def _overlay(base: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    image = np.asarray(base).copy()
    image[mask] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(image)


def _heatmap(gray: np.ndarray) -> Image.Image:
    mapped = cv2.applyColorMap(np.clip(gray * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    mapped = cv2.cvtColor(mapped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(mapped)


def _tile(record: dict, width: int) -> Image.Image:
    render = Image.open(record["render"]).convert("RGB")
    scale = width / render.size[0]
    height = int(round(render.size[1] * scale))
    size = (width, height)
    render_small = render.resize(size, Image.BILINEAR)
    gt = Image.open(record["gt"]).convert("RGB").resize(size, Image.BILINEAR)
    opacity = _heatmap(record["_opacity_array"]).resize(size, Image.BILINEAR)
    rgb_inner = Image.fromarray(record["_rgb_inner"].astype(np.uint8) * 255).resize(size, Image.NEAREST)
    op_inner = Image.fromarray(record["_primary_opacity_inner"].astype(np.uint8) * 255).resize(size, Image.NEAREST)
    op_outer = Image.fromarray(record["_primary_opacity_outer"].astype(np.uint8) * 255).resize(size, Image.NEAREST)

    rgb_inner_overlay = _overlay(render_small, np.asarray(rgb_inner) > 0, (40, 220, 80))
    op_inner_overlay = _overlay(render_small, np.asarray(op_inner) > 0, (80, 180, 255))
    op_outer_overlay = _overlay(render_small, np.asarray(op_outer) > 0, (255, 60, 70))

    label_h = 30
    cols = [gt, render_small, opacity, rgb_inner_overlay, op_inner_overlay, op_outer_overlay]
    names = ["GT", "Render", "Opacity", "RGB inner", "Opacity inner", "Opacity outer"]
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
            f"rgb_in={record['rgb_inner_missing_pixels']} op_in={record['primary_opacity_inner_missing_pixels']} "
            f"op_on_rgb={record['primary_opacity_on_rgb_inner_ratio']:.2f} "
            f"rgb_out={record['rgb_outer_leak_pixels']} op_out={record['primary_opacity_outer_leak_pixels']}"
        ),
        fill=(255, 230, 130),
    )
    return tile


def write_montage(records: list[dict], out_path: Path, topk: int, width: int) -> None:
    chosen = records[:topk]
    if not chosen:
        return
    tiles = [_tile(record, width) for record in chosen]
    sheet = Image.new("RGB", (max(tile.size[0] for tile in tiles), sum(tile.size[1] for tile in tiles)), (12, 12, 12))
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y))
        y += tile.size[1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _mean(records: list[dict], key: str) -> float:
    return float(np.mean([record[key] for record in records])) if records else 0.0


def _diagnose(summary: dict) -> str:
    ratio = summary["mean_primary_opacity_on_rgb_inner_ratio"]
    op_inner = summary["mean_primary_opacity_inner_missing_pixels"]
    rgb_inner = max(1.0, summary["mean_rgb_inner_missing_pixels"])
    if ratio < 0.35 and op_inner / rgb_inner > 0.65:
        return "opacity_or_footprint_gap"
    if ratio > 0.65 and op_inner / rgb_inner < 0.45:
        return "rgb_or_color_support_gap"
    return "mixed_opacity_and_rgb_gap"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare RGB support and opacity footprint support for CoreView_377 renders.")
    parser.add_argument("--render-exp", required=True, type=Path)
    parser.add_argument("--dataset-root", default=Path("data/ZJUMoCap"), type=Path)
    parser.add_argument("--subject", default="CoreView_377")
    parser.add_argument("--split-dir", default="test-view")
    parser.add_argument("--opacity-dir-name", default="opacity")
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--render-support-threshold", default=0.025, type=float)
    parser.add_argument("--primary-opacity-threshold", default=0.06, type=float)
    parser.add_argument("--opacity-thresholds", default="0.02,0.04,0.06,0.08,0.10")
    parser.add_argument("--rgb-close-kernel", default=5, type=int)
    parser.add_argument("--opacity-close-kernel", default=3, type=int)
    parser.add_argument("--band-width", default=7, type=int)
    parser.add_argument("--search-band-width", default=24, type=int)
    parser.add_argument("--min-component-area", default=18, type=int)
    parser.add_argument("--topk", default=16, type=int)
    parser.add_argument("--panel-width", default=220, type=int)
    args = parser.parse_args()

    args.opacity_thresholds = [
        float(item.strip())
        for item in str(args.opacity_thresholds).split(",")
        if item.strip()
    ]
    if args.primary_opacity_threshold not in args.opacity_thresholds:
        args.opacity_thresholds.append(args.primary_opacity_threshold)
        args.opacity_thresholds = sorted(set(args.opacity_thresholds))

    render_dir = args.render_exp / args.split_dir / "renders"
    opacity_dir = args.render_exp / args.split_dir / args.opacity_dir_name
    if not render_dir.exists():
        raise FileNotFoundError(render_dir)
    if not opacity_dir.exists():
        raise FileNotFoundError(opacity_dir)
    out_dir = args.out_dir or (args.render_exp / "diagnostics" / "opacity_footprint")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [
        analyze_sample(path, opacity_dir, args.dataset_root, args.subject, args)
        for path in sorted(render_dir.glob("render_c*_f*.png"))
    ]
    records.sort(
        key=lambda item: (
            item["primary_opacity_inner_missing_pixels"] + item["rgb_inner_missing_pixels"],
            item["primary_opacity_outer_leak_pixels"] + item["rgb_outer_leak_pixels"],
        ),
        reverse=True,
    )
    public = [{k: v for k, v in item.items() if not k.startswith("_") and k != "threshold_records"} for item in records]
    threshold_summary = {}
    for threshold in args.opacity_thresholds:
        label = f"t{threshold:.3f}".replace(".", "p")
        vals = [next(row for row in record["threshold_records"] if abs(row["threshold"] - threshold) < 1.0e-9) for record in records]
        threshold_summary[label] = {
            "threshold": float(threshold),
            "mean_opacity_inner_missing_pixels": float(np.mean([row["opacity_inner_missing_pixels"] for row in vals])) if vals else 0.0,
            "mean_opacity_outer_leak_pixels": float(np.mean([row["opacity_outer_leak_pixels"] for row in vals])) if vals else 0.0,
            "mean_opacity_on_rgb_inner_ratio": float(np.mean([row["opacity_on_rgb_inner_ratio"] for row in vals])) if vals else 0.0,
            "mean_rgb_outer_with_opacity_ratio": float(np.mean([row["rgb_outer_with_opacity_ratio"] for row in vals])) if vals else 0.0,
        }

    summary = {
        "render_exp": str(args.render_exp),
        "n_samples": len(records),
        "primary_opacity_threshold": float(args.primary_opacity_threshold),
        "mean_rgb_inner_missing_pixels": _mean(records, "rgb_inner_missing_pixels"),
        "mean_rgb_outer_leak_pixels": _mean(records, "rgb_outer_leak_pixels"),
        "mean_primary_opacity_inner_missing_pixels": _mean(records, "primary_opacity_inner_missing_pixels"),
        "mean_primary_opacity_outer_leak_pixels": _mean(records, "primary_opacity_outer_leak_pixels"),
        "mean_primary_opacity_on_rgb_inner_pixels": _mean(records, "primary_opacity_on_rgb_inner_pixels"),
        "mean_primary_opacity_on_rgb_inner_ratio": _mean(records, "primary_opacity_on_rgb_inner_ratio"),
        "mean_primary_both_rgb_opacity_inner_missing_pixels": _mean(records, "primary_both_rgb_opacity_inner_missing_pixels"),
        "mean_primary_both_rgb_opacity_inner_missing_ratio": _mean(records, "primary_both_rgb_opacity_inner_missing_ratio"),
        "mean_primary_rgb_outer_with_opacity_pixels": _mean(records, "primary_rgb_outer_with_opacity_pixels"),
        "mean_primary_rgb_outer_with_opacity_ratio": _mean(records, "primary_rgb_outer_with_opacity_ratio"),
        "mean_opacity_in_gt": _mean(records, "mean_opacity_in_gt"),
        "mean_opacity_near_outer": _mean(records, "mean_opacity_near_outer"),
        "diagnosis": "",
        "threshold_summary": threshold_summary,
        "top_samples": public[: args.topk],
    }
    summary["diagnosis"] = _diagnose(summary)

    (out_dir / "opacity_footprint_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "opacity_footprint_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        if public:
            writer = csv.DictWriter(handle, fieldnames=list(public[0].keys()))
            writer.writeheader()
            writer.writerows(public)
    write_montage(records, out_dir / "top_opacity_footprint.png", args.topk, args.panel_width)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
