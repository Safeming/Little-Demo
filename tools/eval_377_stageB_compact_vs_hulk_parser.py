#!/usr/bin/env python3
"""Evaluate StageB compact semantic masks against direct Hulk parser masks on held-out views."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image


RENDER_RE = re.compile(r"render_c(?P<cam>\d+)_f(?P<frame>\d+)\.png$")
REGIONS = ("hair", "face", "skin", "upper", "lower", "shoes")
DEFAULT_COMPACT_MAPPING = Path("configs/semantic/hulk_cihp_compact_6.json")
FALLBACK_PARSER_GROUPS = {
    "hair": (2,),
    "face": (13,),
    "skin": (14, 15, 16, 17),
    "upper": (5, 6, 7, 11),
    "lower": (9, 10, 12),
    "shoes": (8, 18, 19),
}
REGION_COLORS = {
    "hair": (52, 88, 190),
    "face": (238, 176, 118),
    "skin": (232, 196, 126),
    "upper": (66, 150, 210),
    "lower": (82, 178, 126),
    "shoes": (230, 202, 78),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare 3D compact masks with direct Hulk parser masks.")
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path(
            "exp/stageB/377_hulk_light_v224c_head_reliable_views_preserve_stageB_headfix_fixed_20260511_103654_bjt"
            "_v224c_parserhard_rgbclip_best_20260511_131226_bjt"
        ),
    )
    parser.add_argument("--split", default="test-view")
    parser.add_argument("--parser-root", type=Path, default=Path("data/parsers_from_hulk_multiview"))
    parser.add_argument("--subject", default="CoreView_377")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/ZJUMoCap"))
    parser.add_argument("--compact-mapping", type=Path, default=DEFAULT_COMPACT_MAPPING)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--select",
        nargs="*",
        default=[
            "render_c21_f000240.png",
            "render_c21_f000300.png",
            "render_c22_f000240.png",
            "render_c23_f000300.png",
            "render_c23_f000420.png",
        ],
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--panel-width", type=int, default=190)
    parser.add_argument("--header-height", type=int, default=30)
    parser.add_argument("--gap", type=int, default=6)
    return parser.parse_args()


def load_parser_groups(path: Path) -> dict[str, tuple[int, ...]]:
    if path.exists():
        data = json.loads(path.read_text())
        groups = {}
        for name in REGIONS:
            labels = data.get("groups", {}).get(name, FALLBACK_PARSER_GROUPS[name])
            groups[name] = tuple(int(label) for label in labels)
        return groups
    return dict(FALLBACK_PARSER_GROUPS)


def read_rgb(path: Path) -> np.ndarray:
    image = imageio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] > 3:
        image = image[..., :3]
    return image.astype(np.uint8)


def read_binary_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape_hw, dtype=bool)
    mask = imageio.imread(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape[:2] != shape_hw:
        mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def read_parser_mask(
    path: Path,
    shape_hw: tuple[int, int],
    K: np.ndarray | None = None,
    dist: np.ndarray | None = None,
) -> np.ndarray | None:
    if not path.exists():
        return None
    with Image.open(path) as img:
        mask = np.array(img)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if K is not None and dist is not None:
        mask = cv2.undistort(mask, K.astype(np.float32), dist.astype(np.float32).ravel(), None)
    if mask.shape[:2] != shape_hw:
        mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.int32)


def parser_regions(parser_mask: np.ndarray | None, shape_hw: tuple[int, int], parser_groups: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    regions = {name: np.zeros(shape_hw, dtype=bool) for name in REGIONS}
    if parser_mask is None:
        return regions
    for name in REGIONS:
        labels = np.asarray(parser_groups[name], dtype=parser_mask.dtype)
        regions[name] = np.isin(parser_mask, labels)
    return regions


def load_cameras(dataset_root: Path, subject: str) -> dict:
    path = dataset_root / subject / "cam_params.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def camera_intrinsics(cameras: dict, cam: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    cam_data = cameras.get(str(cam))
    if not cam_data:
        return None, None
    return np.asarray(cam_data["K"], dtype=np.float32), np.asarray(cam_data["D"], dtype=np.float32).ravel()


def component_count(mask: np.ndarray, min_area: int = 16) -> int:
    arr = mask.astype(np.uint8)
    num, _, stats, _ = cv2.connectedComponentsWithStats(arr, connectivity=8)
    count = 0
    for idx in range(1, num):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area:
            count += 1
    return count


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = int(np.logical_and(pred, target).sum())
    union = int(np.logical_or(pred, target).sum())
    pred_count = int(pred.sum())
    target_count = int(target.sum())
    return {
        "iou": float(inter / union) if union > 0 else 1.0,
        "precision": float(inter / pred_count) if pred_count > 0 else (1.0 if target_count == 0 else 0.0),
        "recall": float(inter / target_count) if target_count > 0 else 1.0,
        "pred_pixels": pred_count,
        "parser_pixels": target_count,
        "intersection_pixels": inter,
        "union_pixels": union,
        "pred_components": component_count(pred),
        "parser_components": component_count(target),
    }


def color_overlay(rgb: np.ndarray, pred: np.ndarray, parser: np.ndarray) -> np.ndarray:
    out = (rgb.astype(np.float32) * 0.35 + 18.0).clip(0, 255)
    pred_only = pred & ~parser
    parser_only = parser & ~pred
    both = pred & parser
    out[pred_only] = np.array([230, 64, 64], dtype=np.float32)
    out[parser_only] = np.array([64, 132, 238], dtype=np.float32)
    out[both] = np.array([70, 210, 118], dtype=np.float32)
    return out.astype(np.uint8)


def resize_width(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / float(image.shape[1])
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def draw_label(canvas: np.ndarray, text: str, left: int, width: int, header_height: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = left + max(4, (width - tw) // 2)
    y = max(th + 4, (header_height + th) // 2 - baseline)
    cv2.putText(canvas, text, (x, y), font, scale, (30, 30, 30), thickness, cv2.LINE_AA)


def montage(panels: list[tuple[str, np.ndarray]], width: int, header_height: int, gap: int) -> np.ndarray:
    resized = [(label, resize_width(image, width)) for label, image in panels]
    h = max(image.shape[0] for _, image in resized)
    canvas = np.full((h + header_height, len(resized) * width + (len(resized) - 1) * gap, 3), 255, dtype=np.uint8)
    left = 0
    for label, image in resized:
        if image.shape[0] != h:
            image = cv2.resize(image, (width, h), interpolation=cv2.INTER_LINEAR)
        canvas[header_height:, left : left + width] = image
        draw_label(canvas, label, left, width, header_height)
        left += width + gap
    return canvas


def stack_tiles(tiles: list[np.ndarray], gap: int) -> np.ndarray:
    if not tiles:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    w = max(tile.shape[1] for tile in tiles)
    h = sum(tile.shape[0] for tile in tiles) + gap * (len(tiles) - 1)
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)
    top = 0
    for tile in tiles:
        canvas[top : top + tile.shape[0], : tile.shape[1]] = tile
        top += tile.shape[0] + gap
    return canvas


def resolve_names(render_dir: Path, selected: list[str] | None, limit: int) -> list[str]:
    if selected:
        names = selected
    else:
        names = [p.name for p in sorted(render_dir.glob("render_c*_f*.png"))]
    if limit > 0:
        names = names[:limit]
    return names


def main() -> int:
    args = parse_args()
    split_dir = args.exp_dir / args.split
    render_dir = split_dir / "renders"
    compact_dir = split_dir / "semantic_editable_assets" / "compact_head_masks"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "panels").mkdir(exist_ok=True)
    cameras = load_cameras(args.dataset_root, args.subject)
    parser_groups = load_parser_groups(args.compact_mapping)

    rows = []
    tiles = []
    for render_name in resolve_names(render_dir, args.select, args.limit):
        match = RENDER_RE.match(render_name)
        if match is None:
            continue
        rgb_path = render_dir / render_name
        if not rgb_path.exists():
            continue
        rgb = read_rgb(rgb_path)
        shape_hw = rgb.shape[:2]
        cam = int(match.group("cam"))
        frame = int(match.group("frame"))
        parser_path = args.parser_root / args.subject / "mask_cihp" / f"Camera_B{cam}" / f"{frame:06d}.png"
        K, dist = camera_intrinsics(cameras, cam)
        parser_regs = parser_regions(read_parser_mask(parser_path, shape_hw, K=K, dist=dist), shape_hw, parser_groups)
        compact_regs = {
            name: read_binary_mask(compact_dir / name / render_name, shape_hw)
            for name in REGIONS
        }

        frame_panels = [("rgb", rgb)]
        for name in REGIONS:
            m = metrics(compact_regs[name], parser_regs[name])
            row = {"render_name": render_name, "region": name, "parser_path": str(parser_path), **m}
            rows.append(row)
            if name in ("hair", "face", "upper", "lower", "shoes"):
                overlay = color_overlay(rgb, compact_regs[name], parser_regs[name])
                frame_panels.append((f"{name} {m['iou']:.2f}", overlay))

        tile = montage(frame_panels, args.panel_width, args.header_height, args.gap)
        imageio.imwrite(args.out_dir / "panels" / render_name, tile)
        tiles.append(tile)

    with (args.out_dir / "per_frame_region_metrics.tsv").open("w", newline="") as f:
        fieldnames = [
            "render_name",
            "region",
            "iou",
            "precision",
            "recall",
            "pred_pixels",
            "parser_pixels",
            "intersection_pixels",
            "union_pixels",
            "pred_components",
            "parser_components",
            "parser_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {}
    for name in REGIONS:
        region_rows = [row for row in rows if row["region"] == name]
        if not region_rows:
            continue
        summary[name] = {
            key: float(np.mean([float(row[key]) for row in region_rows]))
            for key in ("iou", "precision", "recall", "pred_pixels", "parser_pixels", "pred_components", "parser_components")
        }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    imageio.imwrite(args.out_dir / "compact_vs_parser_stacked.png", stack_tiles(tiles, args.gap))
    print(f"Saved {len(rows)} region metrics to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
