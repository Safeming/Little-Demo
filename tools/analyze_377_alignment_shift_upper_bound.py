#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
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


def _resize_rgb(rgb: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return rgb
    h, w = rgb.shape[:2]
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)


def _resize_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return mask
    h, w = mask.shape[:2]
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    resized = cv2.resize(mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def _boundary_band(mask: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    u8 = mask.astype(np.uint8)
    dilated = cv2.dilate(u8, kernel, iterations=1).astype(bool)
    eroded = cv2.erode(u8, kernel, iterations=1).astype(bool)
    return dilated & (~eroded)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


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


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _shift_rgb(rgb: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return rgb
    h, w = rgb.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0),
    )


def _metrics(
    render_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    mask: np.ndarray,
    band: np.ndarray,
    support: np.ndarray,
    distance_scale: float = 1.0,
) -> dict:
    abs_diff = np.abs(render_rgb - gt_rgb).mean(axis=2)
    interior = mask & (~band)
    render_edges = _edges(_luma(render_rgb), support)
    gt_edges = _edges(_luma(gt_rgb), support)
    gt_to_render = _edge_distance(gt_edges, render_edges) * distance_scale
    render_to_gt = _edge_distance(render_edges, gt_edges) * distance_scale
    edge_symmetric = 0.5 * (gt_to_render + render_to_gt)
    return {
        "fg_l1": _safe_mean(abs_diff[mask]),
        "boundary_l1": _safe_mean(abs_diff[band]),
        "interior_l1": _safe_mean(abs_diff[interior]),
        "boundary_edge_l1": _safe_mean(abs_diff[band & (gt_edges | render_edges)]),
        "gt_to_render_edge_dist_px": gt_to_render,
        "render_to_gt_edge_dist_px": render_to_gt,
        "edge_symmetric_dist_px": edge_symmetric,
    }


def _score(metrics: dict, baseline_interior_l1: float, edge_weight: float, interior_penalty: float) -> float:
    interior_regression = max(float(metrics["interior_l1"]) - baseline_interior_l1, 0.0)
    return (
        float(metrics["boundary_l1"])
        + edge_weight * float(metrics["edge_symmetric_dist_px"])
        + interior_penalty * interior_regression
    )


def _candidate_shifts(max_shift: int, step: int) -> list[tuple[int, int]]:
    step = max(1, int(step))
    max_shift = max(0, int(max_shift))
    values = list(range(-max_shift, max_shift + 1, step))
    if 0 not in values:
        values.append(0)
    values = sorted(set(values))
    return [(dx, dy) for dy in values for dx in values]


def analyze_sample(
    render_path: Path,
    dataset_root: Path,
    subject: str,
    band_width: int,
    shifts: list[tuple[int, int]],
    edge_weight: float,
    interior_penalty: float,
    analysis_scale: float,
) -> dict:
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
    render_rgb = _resize_rgb(render_rgb, analysis_scale)
    gt_rgb = _resize_rgb(gt_rgb, analysis_scale)
    mask = _resize_mask(mask, analysis_scale)
    distance_scale = 1.0 / max(float(analysis_scale), 1e-6)
    band = _boundary_band(mask, band_width)
    support = _boundary_band(mask, max(band_width * 2, 3)) | mask

    baseline = _metrics(render_rgb, gt_rgb, mask, band, support, distance_scale=distance_scale)
    baseline_score = _score(baseline, baseline["interior_l1"], edge_weight, interior_penalty)

    best = None
    best_shifted_rgb = None
    for dx, dy in shifts:
        shifted = _shift_rgb(render_rgb, dx, dy)
        shifted_metrics = _metrics(shifted, gt_rgb, mask, band, support, distance_scale=distance_scale)
        shifted_score = _score(shifted_metrics, baseline["interior_l1"], edge_weight, interior_penalty)
        if best is None or shifted_score < best["score"]:
            best = {
                "dx": int(dx),
                "dy": int(dy),
                "score": shifted_score,
                **shifted_metrics,
            }
            best_shifted_rgb = shifted

    assert best is not None
    return {
        "render": str(render_path),
        "cam": cam,
        "frame": frame,
        "fg_pixels": int(mask.sum()),
        "boundary_pixels": int(band.sum()),
        "baseline_score": baseline_score,
        "baseline": baseline,
        "oracle": best,
        "delta": {
            "score": best["score"] - baseline_score,
            "fg_l1": best["fg_l1"] - baseline["fg_l1"],
            "boundary_l1": best["boundary_l1"] - baseline["boundary_l1"],
            "interior_l1": best["interior_l1"] - baseline["interior_l1"],
            "edge_symmetric_dist_px": best["edge_symmetric_dist_px"] - baseline["edge_symmetric_dist_px"],
        },
        "_gt_path": str(gt_path),
        "_mask_path": str(mask_path),
        "_render_rgb": render_rgb,
        "_gt_rgb": gt_rgb,
        "_oracle_rgb": best_shifted_rgb,
        "_mask": mask,
        "_band": band,
        "_support": support,
        "_distance_scale": distance_scale,
    }


def _mean(records: list[dict], path: tuple[str, ...]) -> float:
    if not records:
        return 0.0
    values = []
    for record in records:
        current = record
        for key in path:
            current = current[key]
        values.append(float(current))
    return float(np.mean(values))


def _fixed_shift_metrics(
    records: list[dict],
    shifts: list[tuple[int, int]],
    edge_weight: float,
    interior_penalty: float,
) -> tuple[tuple[int, int], dict]:
    best_shift = (0, 0)
    best_summary = None
    for dx, dy in shifts:
        shifted_metrics = []
        scores = []
        for record in records:
            render_rgb = _shift_rgb(record["_render_rgb"], dx, dy)
            gt_rgb = record["_gt_rgb"]
            mask = record["_mask"]
            band = record["_band"]
            support = record["_support"]
            metrics = _metrics(
                render_rgb,
                gt_rgb,
                mask,
                band,
                support,
                distance_scale=float(record.get("_distance_scale", 1.0)),
            )
            shifted_metrics.append(metrics)
            scores.append(_score(metrics, record["baseline"]["interior_l1"], edge_weight, interior_penalty))
        summary = {
            "score": float(np.mean(scores)) if scores else 0.0,
            "fg_l1": float(np.mean([m["fg_l1"] for m in shifted_metrics])) if shifted_metrics else 0.0,
            "boundary_l1": float(np.mean([m["boundary_l1"] for m in shifted_metrics])) if shifted_metrics else 0.0,
            "interior_l1": float(np.mean([m["interior_l1"] for m in shifted_metrics])) if shifted_metrics else 0.0,
            "edge_symmetric_dist_px": float(np.mean([m["edge_symmetric_dist_px"] for m in shifted_metrics])) if shifted_metrics else 0.0,
        }
        if best_summary is None or summary["score"] < best_summary["score"]:
            best_shift = (int(dx), int(dy))
            best_summary = summary
    assert best_summary is not None
    return best_shift, best_summary


def _public_record(record: dict) -> dict:
    row = {
        "render": record["render"],
        "cam": record["cam"],
        "frame": record["frame"],
        "fg_pixels": record["fg_pixels"],
        "boundary_pixels": record["boundary_pixels"],
        "baseline_score": record["baseline_score"],
        "baseline_fg_l1": record["baseline"]["fg_l1"],
        "baseline_boundary_l1": record["baseline"]["boundary_l1"],
        "baseline_interior_l1": record["baseline"]["interior_l1"],
        "baseline_edge_symmetric_dist_px": record["baseline"]["edge_symmetric_dist_px"],
        "oracle_dx": record["oracle"]["dx"],
        "oracle_dy": record["oracle"]["dy"],
        "oracle_dx_original_est": record["oracle"]["dx"] * record["_distance_scale"],
        "oracle_dy_original_est": record["oracle"]["dy"] * record["_distance_scale"],
        "oracle_score": record["oracle"]["score"],
        "oracle_fg_l1": record["oracle"]["fg_l1"],
        "oracle_boundary_l1": record["oracle"]["boundary_l1"],
        "oracle_interior_l1": record["oracle"]["interior_l1"],
        "oracle_edge_symmetric_dist_px": record["oracle"]["edge_symmetric_dist_px"],
        "delta_score": record["delta"]["score"],
        "delta_fg_l1": record["delta"]["fg_l1"],
        "delta_boundary_l1": record["delta"]["boundary_l1"],
        "delta_interior_l1": record["delta"]["interior_l1"],
        "delta_edge_symmetric_dist_px": record["delta"]["edge_symmetric_dist_px"],
    }
    return row


def _rgb_to_image(rgb: np.ndarray, width: int) -> Image.Image:
    image = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))
    if image.size[0] != width:
        scale = width / image.size[0]
        image = image.resize((width, int(round(image.size[1] * scale))), Image.BILINEAR)
    return image


def _make_tile(record: dict, width: int = 260) -> Image.Image:
    gt = _rgb_to_image(record["_gt_rgb"], width)
    render = _rgb_to_image(record["_render_rgb"], width)
    oracle = _rgb_to_image(record["_oracle_rgb"], width)
    diff = np.abs(record["_oracle_rgb"] - record["_gt_rgb"]).mean(axis=2)
    diff_img = _rgb_to_image(np.repeat(np.clip(diff * 5.0, 0, 1)[..., None], 3, axis=2), width)

    mask = Image.fromarray(record["_mask"].astype(np.uint8) * 255).resize(gt.size, Image.NEAREST)
    band = _boundary_band(np.asarray(mask) > 0, 5)
    overlay = np.asarray(oracle).copy()
    overlay[band] = (255, 48, 48)
    overlay_img = Image.fromarray(overlay)

    label_h = 38
    tile = Image.new("RGB", (width * 5, gt.size[1] + label_h), (18, 18, 18))
    draw = ImageDraw.Draw(tile)
    for i, (name, image) in enumerate((("GT", gt), ("Render", render), ("BestShift", oracle), ("Diff x5", diff_img), ("Band", overlay_img))):
        tile.paste(image, (i * width, label_h))
        draw.text((i * width + 6, 6), name, fill=(240, 240, 240))
    draw.text(
        (6, 22),
        (
            f"c{record['cam']} f{record['frame']:06d} "
            f"shift=({record['oracle']['dx']},{record['oracle']['dy']}) "
            f"edge {record['baseline']['edge_symmetric_dist_px']:.2f}->{record['oracle']['edge_symmetric_dist_px']:.2f} "
            f"bd {record['baseline']['boundary_l1']:.4f}->{record['oracle']['boundary_l1']:.4f}"
        ),
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
    parser = argparse.ArgumentParser(
        description="Estimate a 2D shift upper bound for CoreView_377 hard contour alignment."
    )
    parser.add_argument("--render-exp", required=True, type=Path)
    parser.add_argument("--dataset-root", default=Path("data/ZJUMoCap"), type=Path)
    parser.add_argument("--subject", default="CoreView_377")
    parser.add_argument("--split-dir", default="test-view")
    parser.add_argument("--band-width", type=int, default=7)
    parser.add_argument("--max-shift", type=int, default=8)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--analysis-scale", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=0.015)
    parser.add_argument("--interior-penalty", type=float, default=0.60)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    render_dir = args.render_exp / args.split_dir / "renders"
    if not render_dir.exists():
        raise FileNotFoundError(render_dir)
    out_dir = args.out_dir or (args.render_exp / "diagnostics" / "alignment_shift_upper_bound")
    out_dir.mkdir(parents=True, exist_ok=True)

    shifts = _candidate_shifts(args.max_shift, args.step)
    records = [
        analyze_sample(
            path,
            args.dataset_root,
            args.subject,
            args.band_width,
            shifts,
            args.edge_weight,
            args.interior_penalty,
            float(args.analysis_scale),
        )
        for path in sorted(render_dir.glob("render_c*_f*.png"))
    ]
    records.sort(key=lambda item: item["baseline_score"], reverse=True)

    by_camera: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_camera[str(record["cam"])].append(record)

    camera_rows = []
    for cam, cam_records in sorted(by_camera.items(), key=lambda item: int(item[0])):
        fixed_shift, fixed = _fixed_shift_metrics(
            cam_records,
            shifts,
            args.edge_weight,
            args.interior_penalty,
        )
        dxs = [float(record["oracle"]["dx"]) for record in cam_records]
        dys = [float(record["oracle"]["dy"]) for record in cam_records]
        camera_rows.append({
            "cam": cam,
            "n_samples": len(cam_records),
            "baseline_score": _mean(cam_records, ("baseline_score",)),
            "baseline_fg_l1": _mean(cam_records, ("baseline", "fg_l1")),
            "baseline_boundary_l1": _mean(cam_records, ("baseline", "boundary_l1")),
            "baseline_interior_l1": _mean(cam_records, ("baseline", "interior_l1")),
            "baseline_edge_symmetric_dist_px": _mean(cam_records, ("baseline", "edge_symmetric_dist_px")),
            "oracle_score": _mean(cam_records, ("oracle", "score")),
            "oracle_fg_l1": _mean(cam_records, ("oracle", "fg_l1")),
            "oracle_boundary_l1": _mean(cam_records, ("oracle", "boundary_l1")),
            "oracle_interior_l1": _mean(cam_records, ("oracle", "interior_l1")),
            "oracle_edge_symmetric_dist_px": _mean(cam_records, ("oracle", "edge_symmetric_dist_px")),
            "oracle_mean_dx": float(np.mean(dxs)) if dxs else 0.0,
            "oracle_mean_dy": float(np.mean(dys)) if dys else 0.0,
            "oracle_mean_dx_original_est": float(np.mean(dxs) / max(float(args.analysis_scale), 1e-6)) if dxs else 0.0,
            "oracle_mean_dy_original_est": float(np.mean(dys) / max(float(args.analysis_scale), 1e-6)) if dys else 0.0,
            "oracle_std_dx": float(np.std(dxs)) if dxs else 0.0,
            "oracle_std_dy": float(np.std(dys)) if dys else 0.0,
            "fixed_dx": fixed_shift[0],
            "fixed_dy": fixed_shift[1],
            "fixed_dx_original_est": fixed_shift[0] / max(float(args.analysis_scale), 1e-6),
            "fixed_dy_original_est": fixed_shift[1] / max(float(args.analysis_scale), 1e-6),
            "fixed_score": fixed["score"],
            "fixed_fg_l1": fixed["fg_l1"],
            "fixed_boundary_l1": fixed["boundary_l1"],
            "fixed_interior_l1": fixed["interior_l1"],
            "fixed_edge_symmetric_dist_px": fixed["edge_symmetric_dist_px"],
        })

    public_records = [_public_record(record) for record in records]
    summary = {
        "render_exp": str(args.render_exp),
        "n_samples": len(records),
        "band_width": args.band_width,
        "max_shift": args.max_shift,
        "step": args.step,
        "analysis_scale": float(args.analysis_scale),
        "edge_weight": args.edge_weight,
        "interior_penalty": args.interior_penalty,
        "baseline": {
            "score": _mean(records, ("baseline_score",)),
            "fg_l1": _mean(records, ("baseline", "fg_l1")),
            "boundary_l1": _mean(records, ("baseline", "boundary_l1")),
            "interior_l1": _mean(records, ("baseline", "interior_l1")),
            "edge_symmetric_dist_px": _mean(records, ("baseline", "edge_symmetric_dist_px")),
        },
        "oracle_per_sample_shift": {
            "score": _mean(records, ("oracle", "score")),
            "fg_l1": _mean(records, ("oracle", "fg_l1")),
            "boundary_l1": _mean(records, ("oracle", "boundary_l1")),
            "interior_l1": _mean(records, ("oracle", "interior_l1")),
            "edge_symmetric_dist_px": _mean(records, ("oracle", "edge_symmetric_dist_px")),
        },
        "delta_oracle_minus_baseline": {
            "score": _mean(records, ("delta", "score")),
            "fg_l1": _mean(records, ("delta", "fg_l1")),
            "boundary_l1": _mean(records, ("delta", "boundary_l1")),
            "interior_l1": _mean(records, ("delta", "interior_l1")),
            "edge_symmetric_dist_px": _mean(records, ("delta", "edge_symmetric_dist_px")),
        },
        "camera_summary": camera_rows,
        "top_hard_samples": public_records[: args.topk],
    }

    (out_dir / "alignment_shift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "alignment_shift_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(public_records[0].keys()) if public_records else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(public_records)
    with (out_dir / "alignment_shift_camera.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(camera_rows[0].keys()) if camera_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(camera_rows)
    write_montage(records, out_dir / "top_shift_upper_bound.png", args.topk)

    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
