#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


RENDER_RE = re.compile(r"render_c(?P<cam>\d+)_f(?P<frame>\d+)\.png$")

REGIONS = OrderedDict(
    [
        ("face", (13,)),
        ("hair", (2,)),
        ("face_hair", (2, 13)),
        ("arms", (14, 15)),
        ("upper_cloth", (5, 6, 7, 11)),
        ("lower_cloth", (9, 10, 12)),
        ("valid_roi", (2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)),
    ]
)

REGION_WEIGHTS = {
    "face": 1.20,
    "hair": 0.85,
    "face_hair": 1.50,
    "arms": 0.90,
    "upper_cloth": 1.15,
    "lower_cloth": 0.55,
    "valid_roi": 0.35,
}

REGION_COLORS = {
    "face": (255, 70, 70),
    "hair": (255, 170, 30),
    "arms": (70, 210, 90),
    "upper_cloth": (70, 130, 255),
    "lower_cloth": (170, 80, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze v198a train-view renders for parser-ROI local teacher reliability."
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
    parser.add_argument("--topk", type=int, default=16)
    return parser.parse_args()


def _read_render(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _safe_percentile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    if values.size == 0:
        return float(default)
    return float(np.percentile(values, q))


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _dog(gray: np.ndarray, sigma: float = 1.15) -> np.ndarray:
    blur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    return np.abs(gray.astype(np.float32) - blur)


def _grad_mag(gray: np.ndarray) -> np.ndarray:
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _boundary_band(mask: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    u8 = mask.astype(np.uint8)
    dilated = cv2.dilate(u8, kernel, iterations=1).astype(bool)
    eroded = cv2.erode(u8, kernel, iterations=1).astype(bool)
    return dilated & (~eroded)


def _erode(mask: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


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


def _load_cam_params(dataset_root: Path, subject: str) -> dict:
    cam_path = dataset_root / subject / "cam_params.json"
    return json.loads(cam_path.read_text(encoding="utf-8"))


def _centered_intrinsics(cam: dict) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(cam["K"], dtype=np.float32).copy()
    dist = np.asarray(cam["D"], dtype=np.float32).ravel()
    k[0, 2] = 1024.0 / 2.0
    k[1, 2] = 1024.0 / 2.0
    return k, dist


def _load_parser(parser_root: Path, subject: str, cam: int, frame: int) -> np.ndarray | None:
    path = parser_root / subject / "mask_cihp" / f"Camera_B{int(cam)}" / f"{int(frame):06d}.png"
    if not path.exists():
        return None
    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def _load_preprocessed(
    dataset_root: Path,
    parser_root: Path,
    subject: str,
    cam_params: dict,
    cam: int,
    frame: int,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    cam_name = str(int(cam))
    frame_name = f"{int(frame):06d}"
    gt_path = dataset_root / subject / cam_name / f"{frame_name}.jpg"
    mask_path = dataset_root / subject / cam_name / f"{frame_name}.png"
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    image_bgr = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(gt_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hard_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if hard_mask is None:
        raise FileNotFoundError(mask_path)
    parser_mask = _load_parser(parser_root, subject, cam, frame)

    k, dist = _centered_intrinsics(cam_params[cam_name])
    image = cv2.undistort(image, k, dist, None)
    hard_mask = cv2.undistort(hard_mask, k, dist, None)
    if parser_mask is not None:
        parser_mask = cv2.undistort(parser_mask, k, dist, None)

    target_w, target_h = size
    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    hard_mask = cv2.resize(hard_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    if parser_mask is not None:
        parser_mask = cv2.resize(parser_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    fg = hard_mask > 0
    image = image.astype(np.float32) / 255.0
    image[~fg] = 0.0
    return image, fg, None if parser_mask is None else parser_mask.astype(np.int32)


def _region_mask(parser_mask: np.ndarray | None, fg_mask: np.ndarray, labels: tuple[int, ...]) -> np.ndarray:
    if parser_mask is None:
        return np.zeros_like(fg_mask, dtype=bool)
    return np.isin(parser_mask, np.asarray(labels, dtype=np.int32)) & fg_mask


def _region_metrics(
    region: np.ndarray,
    fg_boundary: np.ndarray,
    diff: np.ndarray,
    gt_hf: np.ndarray,
    render_hf: np.ndarray,
    gt_grad: np.ndarray,
    render_grad: np.ndarray,
    gt_edges: np.ndarray,
    render_edges: np.ndarray,
    gt_hf_thresh: float,
    args: argparse.Namespace,
) -> dict:
    region_px = int(region.sum())
    if region_px <= 0:
        return {
            "pixels": 0,
            "interior_pixels": 0,
            "region_l1": 0.0,
            "interior_l1": 0.0,
            "boundary_l1": 0.0,
            "gt_hf_mean": 0.0,
            "render_hf_mean": 0.0,
            "hf_ratio": 0.0,
            "gt_grad_mean": 0.0,
            "render_grad_mean": 0.0,
            "edge_ratio": 0.0,
            "reliable_ratio": 0.0,
            "missing_hf_ratio": 0.0,
            "boundary_fraction": 0.0,
            "quality_score": 0.0,
        }

    region_boundary = (_boundary_band(region, args.band_width) | fg_boundary) & region
    interior = _erode(region, args.region_erode) & (~fg_boundary)
    if int(interior.sum()) < args.min_region_pixels:
        interior = region & (~region_boundary)
    if int(interior.sum()) < args.min_region_pixels:
        interior = region

    high_gt = interior & (gt_hf >= gt_hf_thresh)
    reliable = high_gt & (diff <= args.reliable_l1_thresh)
    missing_hf = high_gt & (render_hf < gt_hf * args.missing_hf_ratio)

    gt_edge_count = int((gt_edges & interior).sum())
    render_edge_count = int((render_edges & interior).sum())
    edge_ratio = float(render_edge_count / max(gt_edge_count, 1))
    edge_ratio = min(edge_ratio, 2.0)

    gt_hf_high = gt_hf[high_gt]
    render_hf_high = render_hf[high_gt]
    if gt_hf_high.size > 0:
        hf_ratio = float(render_hf_high.mean() / max(float(gt_hf_high.mean()), 1.0e-6))
    else:
        hf_ratio = 0.0

    interior_l1 = _safe_mean(diff[interior])
    boundary_l1 = _safe_mean(diff[region_boundary])
    gt_hf_mean = _safe_mean(gt_hf[interior])
    render_hf_mean = _safe_mean(render_hf[interior])
    gt_grad_mean = _safe_mean(gt_grad[interior])
    render_grad_mean = _safe_mean(render_grad[interior])
    reliable_ratio = float(reliable.sum() / max(int(interior.sum()), 1))
    missing_ratio = float(missing_hf.sum() / max(int(interior.sum()), 1))
    boundary_fraction = float(region_boundary.sum() / max(region_px, 1))
    px_factor = min(1.0, math.sqrt(region_px / 6000.0))
    hf_factor = min(1.5, gt_hf_mean / 0.035) if gt_hf_mean > 0.0 else 0.0
    quality = px_factor * (
        0.42 * reliable_ratio
        + 0.22 * min(edge_ratio, 1.15)
        + 0.20 * min(hf_ratio, 1.2)
        + 0.16 * min(hf_factor, 1.2)
    )
    quality -= 1.75 * interior_l1 + 0.80 * boundary_l1 + 0.15 * boundary_fraction

    return {
        "pixels": region_px,
        "interior_pixels": int(interior.sum()),
        "region_l1": _safe_mean(diff[region]),
        "interior_l1": interior_l1,
        "boundary_l1": boundary_l1,
        "gt_hf_mean": gt_hf_mean,
        "render_hf_mean": render_hf_mean,
        "hf_ratio": hf_ratio,
        "gt_grad_mean": gt_grad_mean,
        "render_grad_mean": render_grad_mean,
        "edge_ratio": edge_ratio,
        "reliable_ratio": reliable_ratio,
        "missing_hf_ratio": missing_ratio,
        "boundary_fraction": boundary_fraction,
        "quality_score": float(quality),
    }


def analyze_sample(render_path: Path, args: argparse.Namespace, cam_params: dict) -> dict:
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

    region_records = {}
    weighted_score_sum = 0.0
    weighted_score_weight = 0.0
    for region_name, labels in REGIONS.items():
        mask = _region_mask(parser_mask, fg, labels)
        metrics = _region_metrics(
            mask,
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
        region_records[region_name] = metrics
        if metrics["pixels"] >= args.min_region_pixels:
            weight = REGION_WEIGHTS.get(region_name, 1.0)
            weighted_score_sum += weight * float(metrics["quality_score"])
            weighted_score_weight += weight

    edge_symmetric = 0.5 * (gt_to_render + render_to_gt)
    hard_score = _safe_mean(diff[fg_boundary]) + 0.015 * edge_symmetric
    if "valid_roi" in region_records:
        hard_score += 0.15 * float(region_records["valid_roi"]["missing_hf_ratio"])

    return {
        "render": str(render_path),
        "cam": cam,
        "frame": frame,
        "fg_pixels": int(fg.sum()),
        "parser_available": parser_mask is not None,
        "fg_l1": _safe_mean(diff[fg]),
        "fg_interior_l1": _safe_mean(diff[fg_interior]),
        "fg_boundary_l1": _safe_mean(diff[fg_boundary]),
        "edge_symmetric_dist_px": edge_symmetric,
        "gt_to_render_edge_dist_px": gt_to_render,
        "render_to_gt_edge_dist_px": render_to_gt,
        "gt_hf_thresh": gt_hf_thresh,
        "quality_score": float(weighted_score_sum / max(weighted_score_weight, 1.0)),
        "hard_score": float(hard_score),
        "regions": region_records,
    }


def _aggregate_numeric(records: list[dict], keys: list[str], weight_key: str | None = None) -> dict:
    out = {}
    for key in keys:
        values = []
        weights = []
        for record in records:
            value = record.get(key)
            if value is None:
                continue
            values.append(float(value))
            if weight_key is not None:
                weights.append(float(record.get(weight_key, 1.0)))
        if not values:
            out[key] = 0.0
        elif weight_key is not None:
            out[key] = float(np.average(np.asarray(values), weights=np.asarray(weights)))
        else:
            out[key] = float(np.mean(values))
    return out


def aggregate(records: list[dict]) -> tuple[list[dict], list[dict], dict]:
    by_cam = defaultdict(list)
    by_region = defaultdict(list)
    by_cam_region = defaultdict(list)
    for record in records:
        cam = int(record["cam"])
        by_cam[cam].append(record)
        for region, metrics in record["regions"].items():
            flat = dict(metrics)
            flat["cam"] = cam
            flat["region"] = region
            flat["frame"] = int(record["frame"])
            by_region[region].append(flat)
            by_cam_region[(cam, region)].append(flat)

    camera_rows = []
    for cam, cam_records in sorted(by_cam.items()):
        row = {
            "cam": cam,
            "samples": len(cam_records),
        }
        row.update(
            _aggregate_numeric(
                cam_records,
                [
                    "fg_l1",
                    "fg_interior_l1",
                    "fg_boundary_l1",
                    "edge_symmetric_dist_px",
                    "quality_score",
                    "hard_score",
                ],
            )
        )
        for region in ("face_hair", "arms", "upper_cloth", "lower_cloth", "valid_roi"):
            region_records = by_cam_region.get((cam, region), [])
            prefix = f"{region}_"
            agg = _aggregate_numeric(
                region_records,
                [
                    "pixels",
                    "interior_l1",
                    "boundary_l1",
                    "gt_hf_mean",
                    "render_hf_mean",
                    "hf_ratio",
                    "edge_ratio",
                    "reliable_ratio",
                    "missing_hf_ratio",
                    "quality_score",
                ],
            )
            for key, value in agg.items():
                row[prefix + key] = value
        camera_rows.append(row)

    region_rows = []
    for region, region_records in by_region.items():
        row = {"region": region, "samples": len(region_records)}
        row.update(
            _aggregate_numeric(
                region_records,
                [
                    "pixels",
                    "interior_l1",
                    "boundary_l1",
                    "gt_hf_mean",
                    "render_hf_mean",
                    "hf_ratio",
                    "edge_ratio",
                    "reliable_ratio",
                    "missing_hf_ratio",
                    "boundary_fraction",
                    "quality_score",
                ],
            )
        )
        region_rows.append(row)

    scores = [float(row["quality_score"]) for row in camera_rows]
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 1.0
    span = max(max_score - min_score, 1.0e-6)
    for row in camera_rows:
        norm = (float(row["quality_score"]) - min_score) / span
        penalty = 0.0
        penalty += max(0.0, float(row["fg_boundary_l1"]) - 0.070) * 4.0
        weight = 0.35 + 1.65 * norm - penalty
        row["recommended_weight"] = float(np.clip(weight, 0.25, 2.0))

    ranked = sorted(camera_rows, key=lambda item: (item["quality_score"], -item["hard_score"]), reverse=True)
    ranked_cams = [int(row["cam"]) for row in ranked]
    view_sets = {
        "top8": ranked_cams[:8],
        "top10": ranked_cams[:10],
        "top12": ranked_cams[:12],
        "drop_bad4": sorted(ranked_cams[:-4]),
        "drop_bad6": sorted(ranked_cams[:-6]),
        "bottom4": ranked_cams[-4:],
        "ranked": ranked_cams,
    }
    weights = {str(int(row["cam"])): round(float(row["recommended_weight"]), 4) for row in camera_rows}
    weight_str = "{" + ",".join(f"{int(row['cam'])}:{float(row['recommended_weight']):.4f}" for row in sorted(camera_rows, key=lambda r: int(r["cam"]))) + "}"
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_samples": len(records),
        "ranked_cameras": ranked_cams,
        "view_sets": view_sets,
        "camera_weights": weights,
        "camera_weights_omega": weight_str,
        "best_camera": ranked_cams[0] if ranked_cams else None,
        "worst_camera": ranked_cams[-1] if ranked_cams else None,
    }
    return camera_rows, region_rows, summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _public_sample_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "regions"}
        for region in ("face_hair", "arms", "upper_cloth", "lower_cloth", "valid_roi"):
            metrics = record["regions"].get(region, {})
            for key in ("pixels", "interior_l1", "hf_ratio", "edge_ratio", "reliable_ratio", "missing_hf_ratio", "quality_score"):
                row[f"{region}_{key}"] = metrics.get(key, 0.0)
        rows.append(row)
    return rows


def _overlay_regions(gt: np.ndarray, parser_mask: np.ndarray | None, fg: np.ndarray) -> Image.Image:
    base = np.clip(gt * 255.0, 0, 255).astype(np.uint8)
    overlay = base.copy()
    if parser_mask is not None:
        for name, color in REGION_COLORS.items():
            labels = REGIONS.get(name, ())
            if not labels:
                continue
            mask = np.isin(parser_mask, np.asarray(labels, dtype=np.int32)) & fg
            overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.asarray(color, dtype=np.float32)).astype(np.uint8)
    return Image.fromarray(overlay)


def _missing_overlay(gt: np.ndarray, missing_mask: np.ndarray) -> Image.Image:
    base = np.clip(gt * 255.0, 0, 255).astype(np.uint8)
    overlay = base.copy()
    overlay[missing_mask] = (255, 35, 35)
    return Image.fromarray(overlay)


def _make_tile(record: dict, args: argparse.Namespace, cam_params: dict, width: int = 260) -> Image.Image:
    render = _read_render(Path(record["render"]))
    size = (render.shape[1], render.shape[0])
    gt, fg, parser_mask = _load_preprocessed(
        args.dataset_root,
        args.parser_root,
        args.subject,
        cam_params,
        int(record["cam"]),
        int(record["frame"]),
        size,
    )
    diff = np.abs(render - gt).mean(axis=2)
    gt_gray = _luma(gt)
    render_gray = _luma(render)
    gt_hf = _dog(gt_gray)
    render_hf = _dog(render_gray)
    high_thresh = _safe_percentile(gt_hf[fg], 70.0, default=0.02)
    valid_roi = _region_mask(parser_mask, fg, REGIONS["valid_roi"])
    missing = valid_roi & (gt_hf >= high_thresh) & (render_hf < gt_hf * args.missing_hf_ratio)

    render_img = Image.fromarray(np.clip(render * 255.0, 0, 255).astype(np.uint8))
    gt_img = Image.fromarray(np.clip(gt * 255.0, 0, 255).astype(np.uint8))
    diff_img = Image.fromarray(np.clip(diff * 6.0 * 255.0, 0, 255).astype(np.uint8)).convert("RGB")
    parser_img = _overlay_regions(gt, parser_mask, fg)
    miss_img = _missing_overlay(gt, missing)

    panels = [
        ("GT", gt_img),
        ("Render", render_img),
        ("Diff x6", diff_img),
        ("Parser ROI", parser_img),
        ("HF Missing", miss_img),
    ]
    scale = width / panels[0][1].size[0]
    height = int(round(panels[0][1].size[1] * scale))
    label_h = 44
    tile = Image.new("RGB", (width * len(panels), height + label_h), (18, 18, 18))
    draw = ImageDraw.Draw(tile)
    for idx, (name, image) in enumerate(panels):
        resized = image.resize((width, height), Image.BILINEAR if name != "Parser ROI" else Image.NEAREST)
        tile.paste(resized, (idx * width, label_h))
        draw.text((idx * width + 6, 7), name, fill=(235, 235, 235))
    draw.text(
        (6, 25),
        f"c{int(record['cam']):02d} f{int(record['frame']):06d} hard={float(record['hard_score']):.4f} q={float(record['quality_score']):.4f}",
        fill=(255, 220, 120),
    )
    return tile


def write_montage(records: list[dict], args: argparse.Namespace, cam_params: dict, out_path: Path) -> None:
    chosen = sorted(records, key=lambda item: item["hard_score"], reverse=True)[: args.topk]
    if not chosen:
        return
    tiles = [_make_tile(record, args, cam_params) for record in chosen]
    width = max(tile.size[0] for tile in tiles)
    height = sum(tile.size[1] for tile in tiles)
    sheet = Image.new("RGB", (width, height), (12, 12, 12))
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y))
        y += tile.size[1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _write_markdown(path: Path, camera_rows: list[dict], region_rows: list[dict], summary: dict) -> None:
    ranked = sorted(camera_rows, key=lambda item: item["quality_score"], reverse=True)
    lines = [
        "# Reliable Teacher Confidence Summary",
        "",
        f"- samples: {summary['n_samples']}",
        f"- ranked_cameras: {summary['ranked_cameras']}",
        f"- top8: {summary['view_sets']['top8']}",
        f"- top12: {summary['view_sets']['top12']}",
        f"- drop_bad4: {summary['view_sets']['drop_bad4']}",
        f"- camera_weights_omega: `{summary['camera_weights_omega']}`",
        "",
        "## Camera Ranking",
        "",
        "| cam | quality | weight | fg_l1 | boundary_l1 | edge_px | valid_missing | valid_reliable |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked:
        lines.append(
            f"| {int(row['cam'])} | {float(row['quality_score']):.5f} | "
            f"{float(row['recommended_weight']):.3f} | {float(row['fg_l1']):.5f} | "
            f"{float(row['fg_boundary_l1']):.5f} | {float(row['edge_symmetric_dist_px']):.3f} | "
            f"{float(row.get('valid_roi_missing_hf_ratio', 0.0)):.5f} | "
            f"{float(row.get('valid_roi_reliable_ratio', 0.0)):.5f} |"
        )
    lines.extend(["", "## Region Means", ""])
    lines.append("| region | pixels | quality | hf_ratio | edge_ratio | reliable | missing | boundary_l1 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(region_rows, key=lambda item: REGION_WEIGHTS.get(item["region"], 1.0), reverse=True):
        lines.append(
            f"| {row['region']} | {float(row['pixels']):.1f} | {float(row['quality_score']):.5f} | "
            f"{float(row['hf_ratio']):.4f} | {float(row['edge_ratio']):.4f} | "
            f"{float(row['reliable_ratio']):.5f} | {float(row['missing_hf_ratio']):.5f} | "
            f"{float(row['boundary_l1']):.5f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    records = [analyze_sample(path, args, cam_params) for path in render_paths]
    records.sort(key=lambda item: (item["cam"], item["frame"]))
    camera_rows, region_rows, summary = aggregate(records)

    (args.out_dir / "reliable_teacher_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.out_dir / "camera_rows.json").write_text(json.dumps(camera_rows, indent=2), encoding="utf-8")
    (args.out_dir / "region_rows.json").write_text(json.dumps(region_rows, indent=2), encoding="utf-8")
    _write_csv(args.out_dir / "camera_summary.tsv", camera_rows)
    _write_csv(args.out_dir / "region_summary.tsv", region_rows)
    _write_csv(args.out_dir / "sample_summary.tsv", _public_sample_rows(records))
    (args.out_dir / "camera_weights_omega.txt").write_text(
        summary["camera_weights_omega"] + "\n", encoding="utf-8"
    )
    _write_markdown(args.out_dir / "summary.md", camera_rows, region_rows, summary)
    write_montage(records, args, cam_params, args.out_dir / "hard_samples_montage.png")

    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
