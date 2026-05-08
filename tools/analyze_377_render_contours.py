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
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    return np.asarray(mask, dtype=np.uint8) > 0


def _boundary_band(mask: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    u8 = mask.astype(np.uint8)
    dilated = cv2.dilate(u8, kernel, iterations=1).astype(bool)
    eroded = cv2.erode(u8, kernel, iterations=1).astype(bool)
    return dilated & (~eroded)


def _edges(gray: np.ndarray, support: np.ndarray) -> np.ndarray:
    u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    edges = cv2.Canny(u8, 80, 160).astype(bool)
    return edges & support


def _edge_distance(src_edges: np.ndarray, dst_edges: np.ndarray) -> float:
    if not bool(src_edges.any()):
        return 0.0
    if not bool(dst_edges.any()):
        return 99.0
    dist = cv2.distanceTransform((~dst_edges).astype(np.uint8), cv2.DIST_L2, 3)
    return float(dist[src_edges].mean())


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.mean())


def analyze_sample(render_path: Path, dataset_root: Path, subject: str, band_width: int) -> dict:
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
    gt_rgb = _read_rgb(gt_path, size=size)
    mask = _read_mask(mask_path, size=size)
    band = _boundary_band(mask, band_width)
    support = _boundary_band(mask, max(band_width * 2, 3)) | mask

    abs_diff = np.abs(render_rgb - gt_rgb).mean(axis=2)
    fg_l1 = _safe_mean(abs_diff[mask])
    boundary_l1 = _safe_mean(abs_diff[band])
    interior = mask & (~band)
    interior_l1 = _safe_mean(abs_diff[interior])

    render_gray = _luma(render_rgb)
    gt_gray = _luma(gt_rgb)
    gt_edges = _edges(gt_gray, support)
    render_edges = _edges(render_gray, support)
    gt_to_render = _edge_distance(gt_edges, render_edges)
    render_to_gt = _edge_distance(render_edges, gt_edges)
    edge_symmetric = 0.5 * (gt_to_render + render_to_gt)

    boundary_edge_l1 = _safe_mean(abs_diff[band & (gt_edges | render_edges)])
    hard_score = boundary_l1 + 0.015 * edge_symmetric

    return {
        "render": str(render_path),
        "cam": cam,
        "frame": frame,
        "fg_pixels": int(mask.sum()),
        "boundary_pixels": int(band.sum()),
        "fg_l1": fg_l1,
        "boundary_l1": boundary_l1,
        "interior_l1": interior_l1,
        "boundary_minus_interior_l1": boundary_l1 - interior_l1,
        "boundary_edge_l1": boundary_edge_l1,
        "gt_to_render_edge_dist_px": gt_to_render,
        "render_to_gt_edge_dist_px": render_to_gt,
        "edge_symmetric_dist_px": edge_symmetric,
        "hard_score": hard_score,
        "_gt_path": str(gt_path),
        "_mask_path": str(mask_path),
    }


def _make_tile(record: dict, width: int = 320) -> Image.Image:
    render_path = Path(record["render"])
    gt_path = Path(record["_gt_path"])
    mask_path = Path(record["_mask_path"])

    render = Image.open(render_path).convert("RGB")
    scale = width / render.size[0]
    height = int(round(render.size[1] * scale))
    render = render.resize((width, height), Image.BILINEAR)
    gt = Image.open(gt_path).convert("RGB").resize((width, height), Image.BILINEAR)
    mask = Image.open(mask_path).convert("L").resize((width, height), Image.NEAREST)

    render_np = np.asarray(render, dtype=np.float32) / 255.0
    gt_np = np.asarray(gt, dtype=np.float32) / 255.0
    diff = np.abs(render_np - gt_np).mean(axis=2)
    diff_img = Image.fromarray(np.clip(diff * 5.0 * 255.0, 0, 255).astype(np.uint8)).convert("RGB")

    band = _boundary_band(np.asarray(mask) > 0, 5)
    overlay = np.asarray(render).copy()
    overlay[band] = (255, 48, 48)
    overlay_img = Image.fromarray(overlay)

    label_h = 26
    tile = Image.new("RGB", (width * 4, height + label_h), (20, 20, 20))
    for i, (name, image) in enumerate((("GT", gt), ("Render", render), ("Diff x5", diff_img), ("Band", overlay_img))):
        tile.paste(image, (i * width, label_h))
        draw = ImageDraw.Draw(tile)
        draw.text((i * width + 6, 6), name, fill=(240, 240, 240))
    draw = ImageDraw.Draw(tile)
    draw.text(
        (6, height + label_h - 18),
        f"c{record['cam']} f{record['frame']:06d} boundary_l1={record['boundary_l1']:.4f} edge={record['edge_symmetric_dist_px']:.2f}",
        fill=(255, 220, 120),
    )
    return tile


def write_montage(records: list[dict], out_path: Path, topk: int) -> None:
    chosen = records[:topk]
    if not chosen:
        return
    tiles = [_make_tile(record) for record in chosen]
    width = max(tile.size[0] for tile in tiles)
    height = sum(tile.size[1] for tile in tiles)
    sheet = Image.new("RGB", (width, height), (12, 12, 12))
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y))
        y += tile.size[1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank CoreView_377 renders by foreground/boundary/edge error.")
    parser.add_argument("--render-exp", required=True, type=Path)
    parser.add_argument("--dataset-root", default="data/ZJUMoCap", type=Path)
    parser.add_argument("--subject", default="CoreView_377")
    parser.add_argument("--split-dir", default="test-view")
    parser.add_argument("--band-width", type=int, default=7)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    render_dir = args.render_exp / args.split_dir / "renders"
    if not render_dir.exists():
        raise FileNotFoundError(render_dir)
    out_dir = args.out_dir or (args.render_exp / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for render_path in sorted(render_dir.glob("render_c*_f*.png")):
        records.append(analyze_sample(render_path, args.dataset_root, args.subject, args.band_width))
    records.sort(key=lambda item: item["hard_score"], reverse=True)

    public_records = [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in records
    ]
    summary = {
        "render_exp": str(args.render_exp),
        "n_samples": len(records),
        "mean_fg_l1": float(np.mean([r["fg_l1"] for r in records])) if records else 0.0,
        "mean_boundary_l1": float(np.mean([r["boundary_l1"] for r in records])) if records else 0.0,
        "mean_interior_l1": float(np.mean([r["interior_l1"] for r in records])) if records else 0.0,
        "mean_edge_symmetric_dist_px": float(np.mean([r["edge_symmetric_dist_px"] for r in records])) if records else 0.0,
        "top_hard_samples": public_records[: args.topk],
    }

    (out_dir / "contour_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "contour_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(public_records[0].keys()) if public_records else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(public_records)
    write_montage(records, out_dir / "top_hard_contours.png", args.topk)

    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
