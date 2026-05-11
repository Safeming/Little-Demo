#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


RENDER_RE = re.compile(r"render_c(?P<cam>\d+)_f(?P<frame>\d+)\.png$")
DEFAULT_PANELS = (
    "gt",
    "gt_mask",
    "render",
    "outside_gt",
    "layer",
    "region",
    "compact_semantic",
    "semantic",
)
PANEL_DIRS = {
    "render": "renders",
    "layer": "binding_maps/layer",
    "region": "binding_maps/region",
    "thin": "binding_maps/thin",
    "semantic": "binding_maps/semantic",
    "compact_semantic": "binding_maps/compact_semantic",
    "body_prob": "binding_maps/body_prob",
    "soft_prob": "binding_maps/soft_prob",
    "cloth_prob": "binding_maps/cloth_prob",
    "temporal": "binding_maps/temporal",
}
PANEL_TITLES = {
    "gt": "GT",
    "gt_mask": "GT Mask",
    "render": "Render",
    "outside_gt": "Outside GT",
    "base_render": "Base",
    "base_diff": "Base Diff",
    "layer": "Layer",
    "region": "Region",
    "thin": "Thin",
    "semantic": "Semantic",
    "compact_semantic": "Compact",
    "body_prob": "Body Prob",
    "soft_prob": "Soft Prob",
    "cloth_prob": "Cloth Prob",
    "temporal": "Temporal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure head/upper-boundary render spill outside GT masks.")
    parser.add_argument("--render-exp", required=True, type=Path)
    parser.add_argument("--baseline-render-exp", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/ZJUMoCap"))
    parser.add_argument("--subject", default="CoreView_377")
    parser.add_argument("--split", default="test-view")
    parser.add_argument("--select", nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--render-fg-threshold", type=int, default=6)
    parser.add_argument("--head-bottom-ratio", type=float, default=0.36)
    parser.add_argument("--head-pad-ratio-x", type=float, default=0.18)
    parser.add_argument("--head-pad-ratio-y", type=float, default=0.05)
    parser.add_argument("--panel-width", type=int, default=230)
    parser.add_argument("--panels", nargs="*", default=list(DEFAULT_PANELS))
    return parser.parse_args()


def read_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    return np.asarray(mask, dtype=np.uint8) > 0


def find_gt_and_mask(dataset_root: Path, subject: str, cam: str, frame: str) -> tuple[Path, Path]:
    cam_dir = dataset_root / subject / str(int(cam))
    frame_i = int(frame)
    gt_candidates = (
        cam_dir / f"{frame_i:06d}.jpg",
        cam_dir / f"{frame_i}.jpg",
    )
    mask_candidates = (
        cam_dir / f"{frame_i:06d}.png",
        cam_dir / f"{frame_i}.png",
    )
    gt_path = next((path for path in gt_candidates if path.exists()), None)
    mask_path = next((path for path in mask_candidates if path.exists()), None)
    if gt_path is None:
        raise FileNotFoundError(f"Missing GT image for camera {cam}, frame {frame}")
    if mask_path is None:
        raise FileNotFoundError(f"Missing GT mask for camera {cam}, frame {frame}")
    return gt_path, mask_path


def bbox_from_mask(mask: np.ndarray, pad_x: int = 0, pad_y: int = 0) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    h, w = mask.shape
    if ys.size == 0:
        return 0, 0, w, h
    x1 = max(int(xs.min()) - pad_x, 0)
    x2 = min(int(xs.max()) + 1 + pad_x, w)
    y1 = max(int(ys.min()) - pad_y, 0)
    y2 = min(int(ys.max()) + 1 + pad_y, h)
    return x1, y1, x2, y2


def head_crop_from_mask(
    mask: np.ndarray,
    bottom_ratio: float,
    pad_ratio_x: float,
    pad_ratio_y: float,
) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    h, w = mask.shape
    if ys.size == 0:
        return 0, 0, w, max(1, int(round(0.4 * h)))
    y_top = int(ys.min())
    y_bottom = int(ys.max())
    x_left = int(xs.min())
    x_right = int(xs.max())
    body_h = max(y_bottom - y_top + 1, 1)
    body_w = max(x_right - x_left + 1, 1)
    pad_x = int(round(body_w * max(pad_ratio_x, 0.0)))
    pad_y = int(round(body_h * max(pad_ratio_y, 0.0)))
    x1 = max(x_left - pad_x, 0)
    x2 = min(x_right + 1 + pad_x, w)
    y1 = max(y_top - pad_y, 0)
    y2 = min(y_top + int(round(body_h * min(max(bottom_ratio, 0.1), 0.75))) + pad_y, h)
    y2 = max(y2, y1 + 1)
    return x1, y1, x2, y2


def render_foreground(rgb: np.ndarray, threshold: int) -> np.ndarray:
    return rgb.max(axis=2) > int(threshold)


def outside_overlay(render_rgb: np.ndarray, gt_mask: np.ndarray, render_fg: np.ndarray) -> np.ndarray:
    overlay = render_rgb.copy()
    outside = render_fg & (~gt_mask)
    overlay[outside] = (255, 32, 32)
    boundary = cv2.morphologyEx(gt_mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)).astype(bool)
    overlay[boundary] = (40, 220, 255)
    return overlay


def crop_image(image: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = crop
    return image[y1:y2, x1:x2]


def resize_panel(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w == width:
        return image
    height = max(1, int(round(h * (width / max(w, 1)))))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def panel_image(
    key: str,
    split_dir: Path,
    render_name: str,
    target_size: tuple[int, int],
    gt_rgb: np.ndarray,
    gt_mask: np.ndarray,
    render_rgb: np.ndarray,
    outside_rgb: np.ndarray,
    baseline_exp: Path | None,
) -> np.ndarray | None:
    if key == "gt":
        return gt_rgb
    if key == "gt_mask":
        return np.repeat((gt_mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
    if key == "render":
        return render_rgb
    if key == "outside_gt":
        return outside_rgb
    if key == "base_render":
        if baseline_exp is None:
            return None
        path = baseline_exp / split_dir.name / "renders" / render_name
        return read_rgb(path, size=target_size) if path.exists() else None
    if key == "base_diff":
        if baseline_exp is None:
            return None
        path = baseline_exp / split_dir.name / "renders" / render_name
        if not path.exists():
            return None
        base = read_rgb(path, size=target_size).astype(np.int16)
        diff = np.abs(render_rgb.astype(np.int16) - base).mean(axis=2)
        return np.repeat(np.clip(diff * 24.0, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    rel = PANEL_DIRS.get(key)
    if rel is None:
        return None
    path = split_dir / rel / render_name
    if not path.exists():
        return None
    return read_rgb(path, size=target_size)


def build_panel(
    images: list[tuple[str, np.ndarray]],
    crop: tuple[int, int, int, int],
    panel_width: int,
    footer: str,
) -> Image.Image:
    crops: list[tuple[str, np.ndarray]] = []
    for title, image in images:
        crops.append((title, resize_panel(crop_image(image, crop), panel_width)))
    if not crops:
        raise ValueError("No panel images to render")
    panel_h = max(item[1].shape[0] for item in crops)
    header_h = 28
    footer_h = 24
    gap = 8
    width = len(crops) * panel_width + (len(crops) - 1) * gap
    height = header_h + panel_h + footer_h
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for title, arr in crops:
        img = Image.fromarray(arr)
        if img.size[1] != panel_h:
            padded = Image.new("RGB", (panel_width, panel_h), (0, 0, 0))
            padded.paste(img, (0, 0))
            img = padded
        canvas.paste(img, (x, header_h))
        draw.text((x + 6, 7), title, fill=(20, 20, 20))
        x += panel_width + gap
    draw.text((6, header_h + panel_h + 5), footer, fill=(30, 30, 30))
    return canvas


def analyze_one(
    render_path: Path,
    split_dir: Path,
    args: argparse.Namespace,
    out_panel_dir: Path,
) -> dict:
    match = RENDER_RE.match(render_path.name)
    if match is None:
        raise ValueError(f"Unexpected render filename: {render_path.name}")
    cam = match.group("cam")
    frame = match.group("frame")
    gt_path, mask_path = find_gt_and_mask(args.dataset_root, args.subject, cam, frame)
    render_rgb = read_rgb(render_path)
    target_size = (render_rgb.shape[1], render_rgb.shape[0])
    gt_rgb = read_rgb(gt_path, size=target_size)
    gt_mask = read_mask(mask_path, size=target_size)
    fg = render_foreground(render_rgb, args.render_fg_threshold)
    outside = fg & (~gt_mask)
    head_crop = head_crop_from_mask(
        gt_mask,
        args.head_bottom_ratio,
        args.head_pad_ratio_x,
        args.head_pad_ratio_y,
    )
    full_crop = bbox_from_mask(gt_mask, pad_x=12, pad_y=12)
    x1, y1, x2, y2 = head_crop
    head_region = np.zeros_like(gt_mask, dtype=bool)
    head_region[y1:y2, x1:x2] = True
    outside_head = outside & head_region
    outside_rgb = outside_overlay(render_rgb, gt_mask, fg)

    baseline_mean_abs_diff = 0.0
    baseline_head_mean_abs_diff = 0.0
    if args.baseline_render_exp is not None:
        baseline_path = args.baseline_render_exp / args.split / "renders" / render_path.name
        if baseline_path.exists():
            baseline_rgb = read_rgb(baseline_path, size=target_size)
            diff = np.abs(render_rgb.astype(np.int16) - baseline_rgb.astype(np.int16)).mean(axis=2)
            baseline_mean_abs_diff = float(diff.mean())
            baseline_head_mean_abs_diff = float(diff[head_region].mean()) if bool(head_region.any()) else 0.0

    panel_entries = []
    for key in args.panels:
        image = panel_image(
            key,
            split_dir,
            render_path.name,
            target_size,
            gt_rgb,
            gt_mask,
            render_rgb,
            outside_rgb,
            args.baseline_render_exp,
        )
        if image is not None:
            panel_entries.append((PANEL_TITLES.get(key, key), image))
    footer = (
        f"{render_path.name} outside_head={int(outside_head.sum())} "
        f"outside_total={int(outside.sum())} base_head_diff={baseline_head_mean_abs_diff:.3f}"
    )
    build_panel(panel_entries, head_crop, args.panel_width, footer).save(out_panel_dir / render_path.name)

    return {
        "render": str(render_path),
        "cam": int(cam),
        "frame": int(frame),
        "head_crop_x1": head_crop[0],
        "head_crop_y1": head_crop[1],
        "head_crop_x2": head_crop[2],
        "head_crop_y2": head_crop[3],
        "full_crop_x1": full_crop[0],
        "full_crop_y1": full_crop[1],
        "full_crop_x2": full_crop[2],
        "full_crop_y2": full_crop[3],
        "gt_mask_pixels": int(gt_mask.sum()),
        "render_fg_pixels": int(fg.sum()),
        "outside_gt_pixels": int(outside.sum()),
        "outside_gt_head_pixels": int(outside_head.sum()),
        "outside_gt_head_ratio": float(outside_head.sum() / max(head_region.sum(), 1)),
        "outside_gt_total_ratio": float(outside.sum() / max(gt_mask.sum(), 1)),
        "baseline_mean_abs_diff_rgb": baseline_mean_abs_diff,
        "baseline_head_mean_abs_diff_rgb": baseline_head_mean_abs_diff,
    }


def iter_render_files(render_dir: Path, selected: list[str] | None) -> list[Path]:
    if selected:
        return [render_dir / item for item in selected]
    return sorted(render_dir.glob("render_c*_f*.png"))


def main() -> int:
    args = parse_args()
    split_dir = args.render_exp / args.split
    render_dir = split_dir / "renders"
    if not render_dir.exists():
        raise FileNotFoundError(render_dir)
    out_dir = args.out_dir or (args.render_exp / "diagnostics" / "head_silhouette")
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = out_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for render_path in iter_render_files(render_dir, args.select):
        if not render_path.exists():
            continue
        records.append(analyze_one(render_path, split_dir, args, panel_dir))

    records.sort(key=lambda item: item["outside_gt_head_pixels"], reverse=True)
    with (out_dir / "head_silhouette_metrics.tsv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(records[0].keys()) if records else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        if fieldnames:
            writer.writeheader()
            writer.writerows(records)

    if records:
        top_panels = [Image.open(panel_dir / Path(record["render"]).name).convert("RGB") for record in records[: min(8, len(records))]]
        width = max(image.size[0] for image in top_panels)
        height = sum(image.size[1] for image in top_panels)
        sheet = Image.new("RGB", (width, height), (245, 245, 245))
        y = 0
        for image in top_panels:
            sheet.paste(image, (0, y))
            y += image.size[1]
        sheet.save(out_dir / "head_silhouette_top.png")

    print(f"Saved {len(records)} head silhouette records to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
