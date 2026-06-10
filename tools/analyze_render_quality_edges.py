#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


RENDER_RE = re.compile(r"render_c(?P<cam>\d+)_f(?P<frame>\d+)\.png$")


def _as_float_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected RGB image with shape HxWx3, got {arr.shape}")
    arr = arr.astype(np.float32)
    if arr.max(initial=0.0) > 1.5:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def _as_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def _read_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BILINEAR)
    return _as_float_rgb(np.asarray(image))


def _read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.NEAREST)
    return _as_mask(np.asarray(image))


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _morph_kernel(width: int) -> np.ndarray:
    width = max(1, int(width))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))


def _boundary_band(mask: np.ndarray, width: int) -> np.ndarray:
    kernel = _morph_kernel(width)
    u8 = mask.astype(np.uint8)
    dilated = cv2.dilate(u8, kernel, iterations=1).astype(bool)
    eroded = cv2.erode(u8, kernel, iterations=1).astype(bool)
    return dilated & (~eroded)


def _outside_halo_ring(mask: np.ndarray, width: int) -> np.ndarray:
    kernel = _morph_kernel(width)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return dilated & (~mask)


def _edges(gray: np.ndarray, support: np.ndarray) -> np.ndarray:
    u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    return cv2.Canny(u8, 80, 160).astype(bool) & support


def _edge_distance(src_edges: np.ndarray, dst_edges: np.ndarray) -> float:
    if not bool(src_edges.any()):
        return 0.0
    if not bool(dst_edges.any()):
        return 99.0
    dist = cv2.distanceTransform((~dst_edges).astype(np.uint8), cv2.DIST_L2, 3)
    return float(dist[src_edges].mean())


def analyze_one(
    render_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    mask_u8: np.ndarray,
    band_width: int,
) -> dict:
    render = _as_float_rgb(render_rgb)
    gt = _as_float_rgb(gt_rgb)
    mask = _as_mask(mask_u8)
    if render.shape != gt.shape:
        raise ValueError(f"render and gt shapes differ: {render.shape} vs {gt.shape}")
    if mask.shape != render.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match image shape {render.shape[:2]}")

    band = _boundary_band(mask, band_width)
    interior = mask & (~band)
    halo = _outside_halo_ring(mask, band_width)
    support = _boundary_band(mask, max(int(band_width) * 2, 3)) | mask

    abs_diff = np.abs(render - gt).mean(axis=2)
    render_luma = _luma(render)
    gt_luma = _luma(gt)

    gt_edges = _edges(gt_luma, support)
    render_edges = _edges(render_luma, support)
    gt_to_render = _edge_distance(gt_edges, render_edges)
    render_to_gt = _edge_distance(render_edges, gt_edges)
    edge_symmetric = 0.5 * (gt_to_render + render_to_gt)

    foreground_l1 = _safe_mean(abs_diff[mask])
    boundary_l1 = _safe_mean(abs_diff[band])
    interior_l1 = _safe_mean(abs_diff[interior])
    halo_luma = _safe_mean(render_luma[halo])
    hard_score = boundary_l1 + 0.015 * edge_symmetric + 0.25 * halo_luma

    return {
        "foreground_pixels": int(mask.sum()),
        "boundary_pixels": int(band.sum()),
        "halo_pixels": int(halo.sum()),
        "foreground_l1": foreground_l1,
        "boundary_l1": boundary_l1,
        "interior_l1": interior_l1,
        "boundary_minus_interior_l1": boundary_l1 - interior_l1,
        "gt_mean_luma_fg": _safe_mean(gt_luma[mask]),
        "render_mean_luma_fg": _safe_mean(render_luma[mask]),
        "render_minus_gt_luma_fg": _safe_mean(render_luma[mask]) - _safe_mean(gt_luma[mask]),
        "halo_luma_outside": halo_luma,
        "gt_to_render_edge_dist_px": gt_to_render,
        "render_to_gt_edge_dist_px": render_to_gt,
        "edge_symmetric_dist_px": edge_symmetric,
        "hard_score": hard_score,
    }


def summarize_records(records: list[dict], topk: int) -> dict:
    def mean_key(key: str) -> float:
        if not records:
            return 0.0
        return float(np.mean([record.get(key, 0.0) for record in records]))

    ranked = sorted(records, key=lambda record: record["hard_score"], reverse=True)
    return {
        "n_samples": len(records),
        "mean_foreground_l1": mean_key("foreground_l1"),
        "mean_boundary_l1": mean_key("boundary_l1"),
        "mean_interior_l1": mean_key("interior_l1"),
        "mean_boundary_minus_interior_l1": mean_key("boundary_minus_interior_l1"),
        "mean_edge_symmetric_dist_px": mean_key("edge_symmetric_dist_px"),
        "mean_render_minus_gt_luma_fg": mean_key("render_minus_gt_luma_fg"),
        "mean_halo_luma_outside": mean_key("halo_luma_outside"),
        "mean_hard_score": mean_key("hard_score"),
        "top_hard_samples": ranked[: max(0, int(topk))],
    }


def _format_template(template: str, cam: str, frame: int) -> Path:
    try:
        return Path(template.format(cam=cam, frame=frame))
    except Exception as exc:
        raise ValueError(f"failed to format template {template!r}") from exc


def _analyze_render_file(render_path: Path, gt_template: str, mask_template: str, band_width: int) -> dict:
    match = RENDER_RE.match(render_path.name)
    if match is None:
        raise ValueError(f"unexpected render filename: {render_path.name}")

    cam = match.group("cam")
    frame = int(match.group("frame"))
    render = _read_rgb(render_path)
    size = (render.shape[1], render.shape[0])
    gt_path = _format_template(gt_template, cam, frame)
    mask_path = _format_template(mask_template, cam, frame)
    gt = _read_rgb(gt_path, size=size)
    mask = _read_mask(mask_path, size=size)

    record = analyze_one(render, gt, mask, band_width)
    record.update({
        "render": str(render_path),
        "gt": str(gt_path),
        "mask": str(mask_path),
        "cam": cam,
        "frame": frame,
    })
    return record


def _write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    public_records = [
        {key: value for key, value in record.items() if isinstance(value, (str, int, float))}
        for record in records
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not public_records:
            return
        writer = csv.DictWriter(handle, fieldnames=list(public_records[0].keys()))
        writer.writeheader()
        writer.writerows(public_records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze render foreground, boundary, color and halo quality.")
    parser.add_argument("--render-dir", required=True, type=Path)
    parser.add_argument("--gt-template", required=True)
    parser.add_argument("--mask-template", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--band-width", type=int, default=7)
    parser.add_argument("--topk", type=int, default=12)
    args = parser.parse_args()

    if not args.render_dir.exists():
        raise FileNotFoundError(args.render_dir)

    records = [
        _analyze_render_file(path, args.gt_template, args.mask_template, args.band_width)
        for path in sorted(args.render_dir.glob("render_c*_f*.png"))
    ]
    summary = summarize_records(records, args.topk)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.out_dir / "render_quality_summary.json"
    samples_path = args.out_dir / "render_quality_samples.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(samples_path, sorted(records, key=lambda record: record["hard_score"], reverse=True))

    print(f"wrote {summary_path}")
    print(f"wrote {samples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
