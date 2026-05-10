#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from analyze_377_reliable_teacher_confidence import (
    RENDER_RE,
    _boundary_band,
    _dog,
    _edge_distance,
    _edges,
    _erode,
    _grad_mag,
    _load_cam_params,
    _load_preprocessed,
    _luma,
    _read_render,
    _region_mask,
    _region_metrics,
    _safe_mean,
    _safe_percentile,
)


V211_REGIONS = OrderedDict(
    [
        ("face", (13,)),
        ("hair", (2,)),
        ("face_hair", (2, 13)),
        ("arms", (14, 15)),
        ("upper_cloth", (5, 6, 7, 11)),
        ("lower_cloth", (9, 10, 12)),
        ("legs", (16, 17)),
        ("shoes", (18, 19)),
        ("cloth_all", (5, 6, 7, 9, 10, 11, 12, 18, 19)),
        ("valid_roi", (2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)),
    ]
)

VISUAL_REGIONS = ("face_hair", "arms", "upper_cloth", "lower_cloth", "shoes")

REGION_COLORS = {
    "face": (255, 70, 70),
    "hair": (255, 170, 30),
    "face_hair": (255, 90, 80),
    "arms": (70, 210, 90),
    "upper_cloth": (70, 130, 255),
    "lower_cloth": (170, 80, 255),
    "legs": (80, 210, 190),
    "shoes": (255, 220, 55),
    "cloth_all": (115, 150, 255),
    "valid_roi": (220, 220, 220),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a region x view conflict map from train-view renders. "
            "The output is analysis-only and is meant to decide which regions "
            "are safe for visibility-gated high-frequency training."
        )
    )
    parser.add_argument("--render-exp", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/ZJUMoCap"))
    parser.add_argument("--parser-root", type=Path, default=Path("data/parsers_from_hulk_multiview"))
    parser.add_argument("--subject", default="CoreView_377")
    parser.add_argument("--split-dir", default="test-view")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--band-width", type=int, default=7)
    parser.add_argument("--region-erode", type=int, default=5)
    parser.add_argument("--min-region-pixels", type=int, default=96)
    parser.add_argument("--reliable-l1-thresh", type=float, default=0.075)
    parser.add_argument("--missing-hf-ratio", type=float, default=0.78)
    parser.add_argument("--topk", type=int, default=24)
    parser.add_argument("--montage-cams", type=int, default=5)
    parser.add_argument("--montage-cell-width", type=int, default=156)
    return parser.parse_args()


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _safe_std(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(arr.std())


def _safe_ptp(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, 90.0) - np.percentile(arr, 10.0))


def _teacher_score(metrics: dict, edge_px: float) -> float:
    pixels = float(metrics.get("pixels", 0.0))
    if pixels <= 0:
        return 0.0
    px_factor = min(1.0, math.sqrt(pixels / 5000.0))
    quality = float(metrics.get("quality_score", 0.0))
    reliable = float(metrics.get("reliable_ratio", 0.0))
    missing = float(metrics.get("missing_hf_ratio", 0.0))
    hf_ratio = min(1.15, float(metrics.get("hf_ratio", 0.0)))
    edge_ratio = min(1.15, float(metrics.get("edge_ratio", 0.0)))
    interior_l1 = float(metrics.get("interior_l1", 0.0))
    boundary_l1 = float(metrics.get("boundary_l1", 0.0))
    score = (
        0.50 * quality
        + 0.20 * reliable
        + 0.13 * hf_ratio
        + 0.12 * edge_ratio
        - 0.26 * missing
        - 1.15 * interior_l1
        - 0.55 * boundary_l1
        - 0.020 * min(edge_px, 8.0)
    )
    return float(px_factor * score)


def _need_score(metrics: dict) -> float:
    pixels = float(metrics.get("pixels", 0.0))
    if pixels <= 0:
        return 0.0
    px_factor = min(1.0, math.sqrt(pixels / 5000.0))
    gt_hf = float(metrics.get("gt_hf_mean", 0.0))
    hf_ratio = float(metrics.get("hf_ratio", 0.0))
    edge_ratio = float(metrics.get("edge_ratio", 0.0))
    missing = float(metrics.get("missing_hf_ratio", 0.0))
    hf_strength = min(1.0, gt_hf / 0.040)
    score = (
        0.42 * missing
        + 0.24 * _clip01(1.0 - hf_ratio)
        + 0.17 * _clip01(1.0 - edge_ratio)
        + 0.17 * hf_strength
    )
    return float(px_factor * score)


def analyze_sample(render_path: Path, args: argparse.Namespace, cam_params: dict) -> tuple[dict, list[dict]]:
    match = RENDER_RE.match(render_path.name)
    if match is None:
        raise ValueError(f"Unexpected render filename: {render_path.name}")
    cam = int(match.group("cam"))
    frame = int(match.group("frame"))

    render_rgb = _read_render(render_path)
    size = (render_rgb.shape[1], render_rgb.shape[0])
    gt_rgb, fg, parser_mask = _load_preprocessed(
        args.dataset_root,
        args.parser_root,
        args.subject,
        cam_params,
        cam,
        frame,
        size,
    )

    diff = np.abs(render_rgb - gt_rgb).mean(axis=2)
    fg_boundary = _boundary_band(fg, args.band_width)
    fg_interior = fg & (~fg_boundary)
    support = fg | _boundary_band(fg, max(args.band_width * 2, 3))

    render_gray = _luma(render_rgb)
    gt_gray = _luma(gt_rgb)
    render_hf = _dog(render_gray)
    gt_hf = _dog(gt_gray)
    render_grad = _grad_mag(render_gray)
    gt_grad = _grad_mag(gt_gray)
    gt_hf_thresh = _safe_percentile(gt_hf[fg], 70.0, default=0.02)
    gt_edges = _edges(gt_gray, support)
    render_edges = _edges(render_gray, support)
    gt_to_render = _edge_distance(gt_edges, render_edges)
    render_to_gt = _edge_distance(render_edges, gt_edges)
    edge_symmetric = 0.5 * (gt_to_render + render_to_gt)

    sample = {
        "render": str(render_path),
        "cam": cam,
        "frame": frame,
        "fg_pixels": int(fg.sum()),
        "parser_available": parser_mask is not None,
        "fg_l1": _safe_mean(diff[fg]),
        "fg_interior_l1": _safe_mean(diff[fg_interior]),
        "fg_boundary_l1": _safe_mean(diff[fg_boundary]),
        "edge_symmetric_dist_px": edge_symmetric,
        "gt_hf_thresh": gt_hf_thresh,
    }

    rows = []
    for region_name, labels in V211_REGIONS.items():
        region = _region_mask(parser_mask, fg, labels)
        metrics = _region_metrics(
            region,
            fg_boundary,
            diff,
            gt_hf,
            render_hf,
            gt_grad,
            render_grad,
            gt_edges,
            render_edges,
            gt_hf_thresh,
            args,
        )
        row = {
            "region": region_name,
            "cam": cam,
            "frame": frame,
            "render": str(render_path),
            "edge_symmetric_dist_px": edge_symmetric,
        }
        row.update(metrics)
        row["teacher_score"] = _teacher_score(metrics, edge_symmetric)
        row["need_score"] = _need_score(metrics)
        row["alignment_risk"] = _clip01((edge_symmetric - 2.65) / 1.35) + _clip01(
            (float(metrics.get("boundary_l1", 0.0)) - 0.064) / 0.020
        )
        rows.append(row)
    return sample, rows


def _aggregate_numeric(rows: list[dict], keys: tuple[str, ...]) -> dict:
    out = {}
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in rows]
        out[key] = float(np.mean(values)) if values else 0.0
        out[f"{key}_p90"] = float(np.percentile(values, 90.0)) if values else 0.0
    return out


def build_frame_region_rows(sample_region_rows: list[dict], min_region_pixels: int) -> list[dict]:
    grouped = defaultdict(list)
    for row in sample_region_rows:
        if int(row.get("pixels", 0)) >= min_region_pixels:
            grouped[(int(row["frame"]), str(row["region"]))].append(row)

    out = []
    for (frame, region), rows in sorted(grouped.items()):
        if not rows:
            continue
        teacher_values = [float(row["teacher_score"]) for row in rows]
        missing_values = [float(row["missing_hf_ratio"]) for row in rows]
        hf_values = [float(row["gt_hf_mean"]) for row in rows]
        hf_ratio_values = [float(row["hf_ratio"]) for row in rows]
        edge_values = [float(row["edge_symmetric_dist_px"]) for row in rows]
        boundary_values = [float(row["boundary_l1"]) for row in rows]
        quality_values = [float(row["quality_score"]) for row in rows]

        ranked = sorted(rows, key=lambda item: float(item["teacher_score"]), reverse=True)
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else ranked[0]
        top_gap = float(best["teacher_score"]) - float(second["teacher_score"])
        gt_hf_mean = float(np.mean(hf_values)) if hf_values else 0.0
        gt_hf_cv = _safe_std(hf_values) / max(gt_hf_mean, 1.0e-6)
        teacher_spread = _safe_ptp(teacher_values)
        missing_spread = _safe_ptp(missing_values)
        hf_ratio_spread = _safe_ptp(hf_ratio_values)
        edge_spread = _safe_ptp(edge_values)
        boundary_spread = _safe_ptp(boundary_values)
        quality_spread = _safe_ptp(quality_values)
        ambiguous_top = 1.0 - _clip01(top_gap / 0.070)
        conflict = (
            0.24 * min(gt_hf_cv / 1.20, 1.0)
            + 0.23 * min(missing_spread / 0.35, 1.0)
            + 0.17 * min(teacher_spread / 0.20, 1.0)
            + 0.12 * min(hf_ratio_spread / 0.45, 1.0)
            + 0.11 * min(edge_spread / 1.80, 1.0)
            + 0.08 * min(boundary_spread / 0.025, 1.0)
            + 0.05 * ambiguous_top
        )
        out.append(
            {
                "frame": frame,
                "region": region,
                "n_views": len(rows),
                "best_cam": int(best["cam"]),
                "second_cam": int(second["cam"]),
                "top_gap": top_gap,
                "best_teacher_score": float(best["teacher_score"]),
                "mean_teacher_score": float(np.mean(teacher_values)),
                "teacher_spread_p90_p10": teacher_spread,
                "mean_need_score": float(np.mean([float(row["need_score"]) for row in rows])),
                "max_need_score": float(max(float(row["need_score"]) for row in rows)),
                "gt_hf_mean": gt_hf_mean,
                "gt_hf_cv": float(gt_hf_cv),
                "missing_hf_mean": float(np.mean(missing_values)),
                "missing_hf_spread_p90_p10": missing_spread,
                "hf_ratio_mean": float(np.mean(hf_ratio_values)),
                "hf_ratio_spread_p90_p10": hf_ratio_spread,
                "edge_px_mean": float(np.mean(edge_values)),
                "edge_px_spread_p90_p10": edge_spread,
                "boundary_l1_mean": float(np.mean(boundary_values)),
                "boundary_l1_spread_p90_p10": boundary_spread,
                "quality_mean": float(np.mean(quality_values)),
                "quality_spread_p90_p10": quality_spread,
                "conflict_score": float(conflict),
                "top_cams": ",".join(str(int(row["cam"])) for row in ranked[:5]),
            }
        )
    return out


def build_region_camera_rows(sample_region_rows: list[dict], frame_region_rows: list[dict], min_region_pixels: int) -> list[dict]:
    top_counter = Counter((str(row["region"]), int(row["best_cam"])) for row in frame_region_rows)
    grouped = defaultdict(list)
    for row in sample_region_rows:
        if int(row.get("pixels", 0)) >= min_region_pixels:
            grouped[(str(row["region"]), int(row["cam"]))].append(row)

    keys = (
        "pixels",
        "teacher_score",
        "need_score",
        "quality_score",
        "interior_l1",
        "boundary_l1",
        "gt_hf_mean",
        "render_hf_mean",
        "hf_ratio",
        "edge_ratio",
        "reliable_ratio",
        "missing_hf_ratio",
        "edge_symmetric_dist_px",
        "alignment_risk",
    )
    out = []
    for (region, cam), rows in sorted(grouped.items()):
        result = {"region": region, "cam": cam, "samples": len(rows), "top_frame_count": top_counter[(region, cam)]}
        result.update(_aggregate_numeric(rows, keys))
        out.append(result)
    return out


def _entropy_norm(values: list[int]) -> float:
    if not values:
        return 0.0
    counts = np.asarray(list(Counter(values).values()), dtype=np.float32)
    probs = counts / max(float(counts.sum()), 1.0)
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1.0e-9))))
    return entropy / max(math.log(float(len(counts))), 1.0e-6) if len(counts) > 1 else 0.0


def _recommend_action(region: str, row: dict) -> str:
    if int(row.get("frames", 0)) < 8 or float(row.get("coverage", 0.0)) < 0.18:
        return "hold_low_coverage"
    # edge_px is image-level contour drift, so keep it visible in the report but
    # do not use it as a region-local decision gate.
    boundary_l1 = float(row.get("boundary_l1_mean", row.get("boundary_l1_mean_mean", 0.0)))
    if boundary_l1 > 0.074:
        return "alignment_first"
    if region in ("face", "hair", "face_hair") and float(row.get("conflict_score_mean", 0.0)) > 0.46:
        return "keep_anchor_or_alignment_probe"
    if (
        float(row.get("need_score_mean", 0.0)) > 0.16
        and float(row.get("best_teacher_score_mean", 0.0)) > 0.18
        and float(row.get("conflict_score_mean", 0.0)) > 0.32
    ):
        return "visibility_gated_hf_probe"
    if float(row.get("need_score_mean", 0.0)) > 0.13 and float(row.get("best_teacher_score_mean", 0.0)) > 0.14:
        return "conservative_teacher_probe"
    return "hold_or_metric_only"


def build_region_summary(frame_region_rows: list[dict], sample_region_rows: list[dict], min_region_pixels: int) -> list[dict]:
    sample_by_region = defaultdict(list)
    for row in sample_region_rows:
        if int(row.get("pixels", 0)) >= min_region_pixels:
            sample_by_region[str(row["region"])].append(row)

    frame_by_region = defaultdict(list)
    for row in frame_region_rows:
        frame_by_region[str(row["region"])].append(row)

    all_frames = {int(row["frame"]) for row in sample_region_rows}
    total_frames = max(len(all_frames), 1)
    out = []
    for region in V211_REGIONS:
        frames = frame_by_region.get(region, [])
        samples = sample_by_region.get(region, [])
        if frames:
            top_cams = [int(row["best_cam"]) for row in frames]
            frame_keys = (
                "best_teacher_score",
                "mean_teacher_score",
                "mean_need_score",
                "max_need_score",
                "gt_hf_mean",
                "gt_hf_cv",
                "missing_hf_mean",
                "missing_hf_spread_p90_p10",
                "hf_ratio_mean",
                "edge_px_mean",
                "boundary_l1_mean",
                "conflict_score",
            )
            row = {
                "region": region,
                "frames": len(frames),
                "samples": len(samples),
                "coverage": len(frames) / total_frames,
                "top_cam_entropy_norm": _entropy_norm(top_cams),
                "top_cams_ranked": ",".join(
                    str(cam) for cam, _count in Counter(top_cams).most_common(8)
                ),
            }
            for key in frame_keys:
                values = [float(item.get(key, 0.0)) for item in frames]
                out_key = key.replace("mean_", "").replace("best_", "best_")
                row[f"{out_key}_mean"] = float(np.mean(values)) if values else 0.0
                row[f"{out_key}_p90"] = float(np.percentile(values, 90.0)) if values else 0.0
        else:
            row = {
                "region": region,
                "frames": 0,
                "samples": 0,
                "coverage": 0.0,
                "top_cam_entropy_norm": 0.0,
                "top_cams_ranked": "",
            }
        if samples:
            row["sample_teacher_score_mean"] = float(np.mean([float(item["teacher_score"]) for item in samples]))
            row["sample_need_score_mean"] = float(np.mean([float(item["need_score"]) for item in samples]))
            row["sample_reliable_ratio_mean"] = float(np.mean([float(item["reliable_ratio"]) for item in samples]))
            row["sample_missing_hf_mean"] = float(np.mean([float(item["missing_hf_ratio"]) for item in samples]))
            row["sample_hf_ratio_mean"] = float(np.mean([float(item["hf_ratio"]) for item in samples]))
        else:
            row["sample_teacher_score_mean"] = 0.0
            row["sample_need_score_mean"] = 0.0
            row["sample_reliable_ratio_mean"] = 0.0
            row["sample_missing_hf_mean"] = 0.0
            row["sample_hf_ratio_mean"] = 0.0
        row["recommended_action"] = _recommend_action(region, row)
        out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    image = image.convert("RGB")
    scale = min(target_w / max(image.size[0], 1), target_h / max(image.size[1], 1))
    new_size = (max(1, int(round(image.size[0] * scale))), max(1, int(round(image.size[1] * scale))))
    resized = image.resize(new_size, Image.BILINEAR)
    canvas = Image.new("RGB", size, (18, 18, 18))
    canvas.paste(resized, ((target_w - new_size[0]) // 2, (target_h - new_size[1]) // 2))
    return canvas


def _region_crop_bounds(mask: np.ndarray, pad: int = 24) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    h, w = mask.shape[:2]
    if xs.size == 0 or ys.size == 0:
        return 0, 0, w, h
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    y1 = min(h, int(ys.max()) + pad + 1)
    return x0, y0, x1, y1


def _sample_cell(row: dict, region: str, args: argparse.Namespace, cam_params: dict) -> Image.Image:
    render = _read_render(Path(row["render"]))
    size = (render.shape[1], render.shape[0])
    gt, fg, parser_mask = _load_preprocessed(
        args.dataset_root,
        args.parser_root,
        args.subject,
        cam_params,
        int(row["cam"]),
        int(row["frame"]),
        size,
    )
    labels = V211_REGIONS[region]
    mask = _region_mask(parser_mask, fg, labels)
    interior = _erode(mask, args.region_erode)
    gt_gray = _luma(gt)
    render_gray = _luma(render)
    gt_hf = _dog(gt_gray)
    render_hf = _dog(render_gray)
    thresh = _safe_percentile(gt_hf[fg], 70.0, default=0.02)
    missing = interior & (gt_hf >= thresh) & (render_hf < gt_hf * args.missing_hf_ratio)

    color = np.asarray(REGION_COLORS.get(region, (255, 90, 80)), dtype=np.float32)
    gt_overlay = np.clip(gt * 255.0, 0, 255).astype(np.uint8)
    render_overlay = np.clip(render * 255.0, 0, 255).astype(np.uint8)
    gt_overlay[mask] = (0.55 * gt_overlay[mask].astype(np.float32) + 0.45 * color).astype(np.uint8)
    render_overlay[missing] = (0.50 * render_overlay[missing].astype(np.float32) + 0.50 * np.asarray((255, 30, 30))).astype(np.uint8)

    x0, y0, x1, y1 = _region_crop_bounds(mask, pad=26)
    gt_img = Image.fromarray(gt_overlay[y0:y1, x0:x1])
    render_img = Image.fromarray(render_overlay[y0:y1, x0:x1])
    panel_w = args.montage_cell_width
    panel_h = int(round(panel_w * 1.16))
    label_h = 38
    cell = Image.new("RGB", (panel_w * 2, panel_h + label_h), (12, 12, 12))
    draw = ImageDraw.Draw(cell)
    cell.paste(_fit_image(gt_img, (panel_w, panel_h)), (0, label_h))
    cell.paste(_fit_image(render_img, (panel_w, panel_h)), (panel_w, label_h))
    draw.text(
        (6, 5),
        f"c{int(row['cam']):02d} t={float(row['teacher_score']):.3f}",
        fill=(245, 245, 245),
    )
    draw.text(
        (6, 21),
        f"miss={float(row['missing_hf_ratio']):.3f} hf={float(row['hf_ratio']):.2f}",
        fill=(255, 220, 120),
    )
    return cell


def write_conflict_montage(
    path: Path,
    frame_region_rows: list[dict],
    sample_region_rows: list[dict],
    args: argparse.Namespace,
    cam_params: dict,
) -> None:
    sample_map = defaultdict(list)
    for row in sample_region_rows:
        sample_map[(int(row["frame"]), str(row["region"]))].append(row)

    candidates = [
        row
        for row in frame_region_rows
        if row["region"] in VISUAL_REGIONS and int(row.get("n_views", 0)) >= 4
    ]
    candidates.sort(
        key=lambda row: float(row["conflict_score"]) * (0.40 + float(row["max_need_score"])),
        reverse=True,
    )
    selected = candidates[: args.topk]
    if not selected:
        return

    rows_img = []
    for frame_region in selected:
        key = (int(frame_region["frame"]), str(frame_region["region"]))
        rows = sample_map.get(key, [])
        if not rows:
            continue
        top = sorted(rows, key=lambda item: float(item["teacher_score"]), reverse=True)[:2]
        missing = sorted(rows, key=lambda item: float(item["missing_hf_ratio"]), reverse=True)[:2]
        edge = sorted(rows, key=lambda item: float(item["edge_symmetric_dist_px"]), reverse=True)[:1]
        chosen = []
        seen = set()
        for item in top + missing + edge:
            cam = int(item["cam"])
            if cam in seen:
                continue
            chosen.append(item)
            seen.add(cam)
            if len(chosen) >= args.montage_cams:
                break
        cells = [_sample_cell(row, str(frame_region["region"]), args, cam_params) for row in chosen]
        cell_w = cells[0].size[0]
        cell_h = cells[0].size[1]
        label_w = 235
        row_img = Image.new("RGB", (label_w + cell_w * len(cells), cell_h), (10, 10, 10))
        draw = ImageDraw.Draw(row_img)
        draw.text(
            (8, 8),
            f"{frame_region['region']}\nf{int(frame_region['frame']):06d}\nconf={float(frame_region['conflict_score']):.3f}\nneed={float(frame_region['max_need_score']):.3f}\ntop={frame_region['top_cams']}",
            fill=(238, 238, 238),
            spacing=4,
        )
        for idx, cell in enumerate(cells):
            row_img.paste(cell, (label_w + idx * cell_w, 0))
        rows_img.append(row_img)

    if not rows_img:
        return
    width = max(img.size[0] for img in rows_img)
    height = sum(img.size[1] for img in rows_img)
    sheet = Image.new("RGB", (width, height), (8, 8, 8))
    y = 0
    for img in rows_img:
        sheet.paste(img, (0, y))
        y += img.size[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def write_summary(path: Path, region_rows: list[dict], camera_rows: list[dict], summary: dict) -> None:
    lines = [
        "# v211 Region View Conflict Summary",
        "",
        f"- generated_at: {summary['generated_at']}",
        f"- samples: {summary['n_samples']}",
        f"- regions: {', '.join(V211_REGIONS.keys())}",
        "",
        "## Region Decisions",
        "",
        "| region | action | frames | conflict | need | best_teacher | missing | hf_ratio | edge_px | top_cams |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in region_rows:
        edge_px = float(row.get("edge_px_mean", row.get("edge_px_mean_mean", 0.0)))
        lines.append(
            f"| {row['region']} | {row['recommended_action']} | {int(row.get('frames', 0))} | "
            f"{float(row.get('conflict_score_mean', 0.0)):.3f} | {float(row.get('need_score_mean', 0.0)):.3f} | "
            f"{float(row.get('best_teacher_score_mean', 0.0)):.3f} | {float(row.get('sample_missing_hf_mean', 0.0)):.3f} | "
            f"{float(row.get('sample_hf_ratio_mean', 0.0)):.3f} | {edge_px:.3f} | "
            f"{row.get('top_cams_ranked', '')} |"
        )

    lines.extend(["", "## Region Top Cameras", ""])
    lines.append("| region | cam | score | top_frames | missing | reliable | hf_ratio | boundary | edge_px |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for region in VISUAL_REGIONS:
        rows = [row for row in camera_rows if row["region"] == region]
        rows.sort(key=lambda item: float(item.get("teacher_score", 0.0)), reverse=True)
        for row in rows[:6]:
            lines.append(
                f"| {region} | {int(row['cam'])} | {float(row.get('teacher_score', 0.0)):.3f} | "
                f"{int(row.get('top_frame_count', 0))} | {float(row.get('missing_hf_ratio', 0.0)):.3f} | "
                f"{float(row.get('reliable_ratio', 0.0)):.3f} | {float(row.get('hf_ratio', 0.0)):.3f} | "
                f"{float(row.get('boundary_l1', 0.0)):.4f} | {float(row.get('edge_symmetric_dist_px', 0.0)):.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_training_plan(region_rows: list[dict], camera_rows: list[dict]) -> dict:
    plan = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "strict_anchor": "v198a",
        "recommended_regions": [],
        "blocked_regions": [],
        "camera_sets": {},
        "notes": [
            "Use this as an analysis gate. It is not a direct replacement for render metrics.",
            "High conflict with high need favors a visibility-gated HF residual/cache, not stronger shared canonical texture loss.",
        ],
    }
    for row in region_rows:
        region = str(row["region"])
        action = str(row["recommended_action"])
        if region in ("valid_roi", "cloth_all", "face", "hair", "legs"):
            continue
        cams = [item for item in camera_rows if item["region"] == region]
        cams.sort(key=lambda item: float(item.get("teacher_score", 0.0)), reverse=True)
        top_cams = [int(item["cam"]) for item in cams[:8]]
        plan["camera_sets"][region] = top_cams
        entry = {
            "region": region,
            "action": action,
            "top_cameras": top_cams,
            "conflict_score": float(row.get("conflict_score_mean", 0.0)),
            "need_score": float(row.get("need_score_mean", 0.0)),
            "best_teacher_score": float(row.get("best_teacher_score_mean", 0.0)),
        }
        if action in ("visibility_gated_hf_probe", "conservative_teacher_probe"):
            plan["recommended_regions"].append(entry)
        else:
            plan["blocked_regions"].append(entry)
    return plan


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cam_params = _load_cam_params(args.dataset_root, args.subject)

    render_paths = []
    for exp in args.render_exp:
        render_dir = exp / args.split_dir / "renders"
        if not render_dir.exists():
            raise FileNotFoundError(render_dir)
        render_paths.extend(sorted(render_dir.glob("render_c*_f*.png")))
    if not render_paths:
        raise SystemExit("No render images found.")

    samples = []
    sample_region_rows = []
    for idx, path in enumerate(render_paths, start=1):
        sample, rows = analyze_sample(path, args, cam_params)
        samples.append(sample)
        sample_region_rows.extend(rows)
        if idx % 500 == 0:
            print(f"analyzed {idx}/{len(render_paths)} renders", flush=True)

    frame_region_rows = build_frame_region_rows(sample_region_rows, args.min_region_pixels)
    region_camera_rows = build_region_camera_rows(sample_region_rows, frame_region_rows, args.min_region_pixels)
    region_summary = build_region_summary(frame_region_rows, sample_region_rows, args.min_region_pixels)
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_samples": len(samples),
        "n_sample_region_rows": len(sample_region_rows),
        "n_frame_region_rows": len(frame_region_rows),
        "out_dir": str(args.out_dir),
    }
    training_plan = build_training_plan(region_summary, region_camera_rows)
    summary["training_plan"] = training_plan

    (args.out_dir / "region_view_conflict_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.out_dir / "region_training_plan.json").write_text(
        json.dumps(training_plan, indent=2), encoding="utf-8"
    )
    _write_csv(args.out_dir / "sample_region_summary.tsv", sample_region_rows)
    _write_csv(args.out_dir / "frame_region_conflicts.tsv", frame_region_rows)
    _write_csv(args.out_dir / "region_camera_topk.tsv", region_camera_rows)
    _write_csv(args.out_dir / "region_view_summary.tsv", region_summary)
    write_summary(args.out_dir / "summary.md", region_summary, region_camera_rows, summary)
    write_conflict_montage(
        args.out_dir / "region_conflict_montage.png",
        frame_region_rows,
        sample_region_rows,
        args,
        cam_params,
    )

    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
