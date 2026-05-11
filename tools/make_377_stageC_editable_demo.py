#!/usr/bin/env python3
"""Build the first StageC semantic editable-assets demo from exported StageB assets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image


RENDER_RE = re.compile(r"render_c(?P<cam>\d+)_f(?P<frame>\d+)\.png$")
REGION_COLORS = {
    "hair": (52, 88, 190),
    "face": (238, 176, 118),
    "skin": (232, 196, 126),
    "upper": (66, 150, 210),
    "lower": (82, 178, 126),
    "shoes": (230, 202, 78),
}
EDIT_COLORS = {
    "upper": np.array([0.08, 0.42, 0.88], dtype=np.float32),
    "lower": np.array([0.90, 0.28, 0.16], dtype=np.float32),
    "shoes": np.array([0.98, 0.84, 0.18], dtype=np.float32),
}
INTERPRETABILITY_PANELS = (
    ("rgb", "renders"),
    ("layer", "binding_maps/layer"),
    ("body_prob", "binding_maps/body_prob"),
    ("cloth_prob", "binding_maps/cloth_prob"),
    ("compact", "binding_maps/compact_semantic"),
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create StageC minimal semantic-editable demo panels.")
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path(
            "exp/stageB/377_hulk_light_v224c_head_reliable_views_preserve_stageB_headfix_fixed_20260511_103654_bjt"
            "_v224c_parserhard_rgbclip_best_20260511_131226_bjt"
        ),
    )
    parser.add_argument("--split", default="test-view")
    parser.add_argument("--output-dir", type=Path, default=None)
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
    parser.add_argument("--panel-width", type=int, default=220)
    parser.add_argument("--header-height", type=int, default=34)
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--edit-alpha", type=float, default=0.58)
    parser.add_argument("--overlay-alpha", type=float, default=0.48)
    parser.add_argument(
        "--mask-source",
        choices=("compact", "direct_parser"),
        default="compact",
        help="Use exported 3D compact masks or direct 2D Hulk parser masks for StageC edits.",
    )
    parser.add_argument(
        "--parser-root",
        type=Path,
        default=Path("data/parsers_from_hulk_multiview"),
        help="Root containing CoreView_377/mask_cihp/Camera_B*/frame.png.",
    )
    parser.add_argument("--subject", default="CoreView_377")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/ZJUMoCap"))
    parser.add_argument("--compact-mapping", type=Path, default=DEFAULT_COMPACT_MAPPING)
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


def read_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape_hw, dtype=np.float32)
    mask = imageio.imread(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape[:2] != shape_hw:
        mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)


def read_parser_index_mask(
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


def parser_mask_to_regions(
    parser_mask: np.ndarray | None,
    shape_hw: tuple[int, int],
    parser_groups: dict[str, tuple[int, ...]],
) -> dict[str, np.ndarray]:
    if parser_mask is None:
        return {name: np.zeros(shape_hw, dtype=np.float32) for name in REGIONS}
    regions = {}
    for name in REGIONS:
        labels = parser_groups[name]
        regions[name] = np.isin(parser_mask, np.asarray(labels, dtype=parser_mask.dtype)).astype(np.float32)
    return {name: mask.clip(0.0, 1.0) for name, mask in regions.items()}


def parse_render_name(render_name: str) -> tuple[int, int] | None:
    match = RENDER_RE.match(render_name)
    if match is None:
        return None
    return int(match.group("cam")), int(match.group("frame"))


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


def load_stagec_masks(
    args: argparse.Namespace,
    asset_dir: Path,
    render_name: str,
    shape_hw: tuple[int, int],
    parser_groups: dict[str, tuple[int, ...]],
) -> tuple[dict[str, np.ndarray], str]:
    if args.mask_source == "direct_parser":
        parsed = parse_render_name(render_name)
        if parsed is None:
            return parser_mask_to_regions(None, shape_hw, parser_groups), "direct_parser_missing_name"
        cam, frame = parsed
        parser_path = args.parser_root / args.subject / "mask_cihp" / f"Camera_B{cam}" / f"{frame:06d}.png"
        K, dist = camera_intrinsics(load_cameras(args.dataset_root, args.subject), cam)
        return parser_mask_to_regions(
            read_parser_index_mask(parser_path, shape_hw, K=K, dist=dist),
            shape_hw,
            parser_groups,
        ), str(parser_path)
    masks = {
        name: read_mask(asset_dir / "compact_head_masks" / name / render_name, shape_hw)
        for name in REGIONS
    }
    return masks, "compact_head_masks"


def write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, image.clip(0, 255).astype(np.uint8))


def resize_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    scale = width / float(image.shape[1])
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def draw_label(canvas: np.ndarray, text: str, left: int, width: int, header_height: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = left + max(5, (width - tw) // 2)
    y = max(th + 5, (header_height + th) // 2 - baseline)
    cv2.putText(canvas, text, (x, y), font, scale, (30, 30, 30), thickness, cv2.LINE_AA)


def montage(panels: list[tuple[str, np.ndarray]], panel_width: int, header_height: int, gap: int) -> np.ndarray:
    resized = [(label, resize_width(image, panel_width)) for label, image in panels]
    panel_h = max(image.shape[0] for _, image in resized)
    canvas_h = panel_h + header_height
    canvas_w = len(resized) * panel_width + max(0, len(resized) - 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    left = 0
    for label, image in resized:
        if image.shape[0] != panel_h:
            image = cv2.resize(image, (panel_width, panel_h), interpolation=cv2.INTER_LINEAR)
        canvas[header_height:, left : left + panel_width] = image
        draw_label(canvas, label, left, panel_width, header_height)
        left += panel_width + gap
    cv2.line(canvas, (0, header_height - 1), (canvas_w, header_height - 1), (220, 220, 220), 1)
    return canvas


def stack_tiles(tiles: Iterable[np.ndarray], gap: int) -> np.ndarray:
    tile_list = list(tiles)
    if not tile_list:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    width = max(tile.shape[1] for tile in tile_list)
    height = sum(tile.shape[0] for tile in tile_list) + gap * (len(tile_list) - 1)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    top = 0
    for tile in tile_list:
        canvas[top : top + tile.shape[0], : tile.shape[1]] = tile
        top += tile.shape[0] + gap
    return canvas


def mask_to_color(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    active = mask > 0.5
    out[active] = np.array(color, dtype=np.uint8)
    return out


def soft_mask(mask: np.ndarray) -> np.ndarray:
    smoothed = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=1.1)
    return smoothed.clip(0.0, 1.0)


def recolor_region(rgb: np.ndarray, mask: np.ndarray, target_rgb: np.ndarray, alpha: float) -> np.ndarray:
    base = rgb.astype(np.float32) / 255.0
    m = soft_mask(mask)[..., None] * float(alpha)
    luminance = np.maximum(base.mean(axis=2, keepdims=True), 0.05)
    target = (0.42 * target_rgb.reshape(1, 1, 3) + 0.58 * target_rgb.reshape(1, 1, 3) * (0.55 + luminance)).clip(0.0, 1.0)
    edited = base * (1.0 - m) + target * m
    return (edited * 255.0).clip(0, 255).astype(np.uint8)


def overlay_regions(rgb: np.ndarray, masks: dict[str, np.ndarray], alpha: float) -> np.ndarray:
    out = rgb.astype(np.float32)
    priority = ["skin", "upper", "lower", "shoes", "face", "hair"]
    for name in priority:
        mask = masks.get(name)
        if mask is None:
            continue
        color = np.array(REGION_COLORS[name], dtype=np.float32).reshape(1, 1, 3)
        m = soft_mask(mask)[..., None] * alpha
        out = out * (1.0 - m) + color * m
    return out.clip(0, 255).astype(np.uint8)


def cutout(rgb: np.ndarray, mask: np.ndarray, background: int = 245) -> np.ndarray:
    m = soft_mask(mask)[..., None]
    return (rgb.astype(np.float32) * m + float(background) * (1.0 - m)).clip(0, 255).astype(np.uint8)


def region_context(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    base = rgb.astype(np.float32)
    dim = (base * 0.36 + 22.0).clip(0, 255)
    m = soft_mask(mask)[..., None]
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    target = (base * (1.0 - alpha) + color_arr * alpha).clip(0, 255)
    out = dim * (1.0 - m) + target * m

    edge = cv2.morphologyEx((mask > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
    out[edge > 0] = color_arr.reshape(3)
    return out.clip(0, 255).astype(np.uint8)


def bbox_from_mask(mask: np.ndarray, pad: int = 32) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0.5)
    if ys.size == 0 or xs.size == 0:
        return None
    h, w = mask.shape[:2]
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    y1 = min(h, int(ys.max()) + pad + 1)
    return x0, y0, x1, y1


def crop_to_bbox(image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    if bbox is None:
        return image
    x0, y0, x1, y1 = bbox
    return image[y0:y1, x0:x1]


def compose_region_grid(rgb: np.ndarray, masks: dict[str, np.ndarray], panel_width: int, header_height: int, gap: int) -> np.ndarray:
    union = np.zeros(rgb.shape[:2], dtype=np.float32)
    for mask in masks.values():
        union = np.maximum(union, mask)
    bbox = bbox_from_mask(union, pad=48)
    panels = []
    for name in REGIONS:
        region = cutout(rgb, masks.get(name, np.zeros(rgb.shape[:2], dtype=np.float32)))
        panels.append((name, crop_to_bbox(region, bbox)))
    return montage(panels, panel_width, header_height, gap)


def compose_region_context_grid(
    rgb: np.ndarray,
    masks: dict[str, np.ndarray],
    panel_width: int,
    header_height: int,
    gap: int,
    alpha: float,
) -> np.ndarray:
    union = np.zeros(rgb.shape[:2], dtype=np.float32)
    for mask in masks.values():
        union = np.maximum(union, mask)
    bbox = bbox_from_mask(union, pad=48)
    panels = []
    for name in REGIONS:
        highlighted = region_context(
            rgb,
            masks.get(name, np.zeros(rgb.shape[:2], dtype=np.float32)),
            REGION_COLORS[name],
            alpha,
        )
        panels.append((name, crop_to_bbox(highlighted, bbox)))
    return montage(panels, panel_width, header_height, gap)


def resolve_render_names(split_dir: Path, selected: list[str] | None, limit: int) -> list[str]:
    if selected:
        names = selected
    else:
        names = [path.name for path in sorted((split_dir / "renders").glob("render_c*_f*.png"))]
    if limit > 0:
        names = names[:limit]
    return names


def main() -> int:
    args = parse_args()
    split_dir = args.exp_dir / args.split
    asset_dir = split_dir / "semantic_editable_assets"
    output_dir = args.output_dir or (args.exp_dir / args.split / "stageC_min_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not split_dir.exists():
        raise FileNotFoundError(split_dir)
    if not asset_dir.exists():
        raise FileNotFoundError(asset_dir)

    render_names = resolve_render_names(split_dir, args.select, args.limit)
    records = []
    edit_tiles = []
    interpret_tiles = []
    region_tiles = []
    parser_groups = load_parser_groups(args.compact_mapping)

    for render_name in render_names:
        if RENDER_RE.match(render_name) is None:
            continue
        rgb_path = split_dir / "renders" / render_name
        if not rgb_path.exists():
            continue
        rgb = read_rgb(rgb_path)
        shape_hw = rgb.shape[:2]
        masks, mask_source_detail = load_stagec_masks(args, asset_dir, render_name, shape_hw, parser_groups)

        upper_edit = recolor_region(rgb, masks["upper"], EDIT_COLORS["upper"], args.edit_alpha)
        lower_edit = recolor_region(rgb, masks["lower"], EDIT_COLORS["lower"], args.edit_alpha)
        cloth_edit = recolor_region(upper_edit, masks["lower"], EDIT_COLORS["lower"], args.edit_alpha)
        shoes_edit = recolor_region(rgb, masks["shoes"], EDIT_COLORS["shoes"], args.edit_alpha)
        overlay = overlay_regions(rgb, masks, args.overlay_alpha)
        regions = compose_region_context_grid(
            rgb,
            masks,
            max(120, args.panel_width // 2),
            args.header_height,
            max(4, args.gap // 2),
            args.overlay_alpha,
        )
        cutouts = compose_region_grid(rgb, masks, max(120, args.panel_width // 2), args.header_height, max(4, args.gap // 2))

        stem = Path(render_name).stem
        frame_dir = output_dir / "frames" / stem
        write_rgb(frame_dir / "rgb.png", rgb)
        write_rgb(frame_dir / "upper_color_edit.png", upper_edit)
        write_rgb(frame_dir / "lower_color_edit.png", lower_edit)
        write_rgb(frame_dir / "upper_lower_color_edit.png", cloth_edit)
        write_rgb(frame_dir / "shoes_color_edit.png", shoes_edit)
        write_rgb(frame_dir / "region_overlay.png", overlay)
        write_rgb(frame_dir / "region_context.png", regions)
        write_rgb(frame_dir / "region_cutouts.png", cutouts)
        for name, mask in masks.items():
            write_rgb(frame_dir / f"{name}_cutout.png", cutout(rgb, mask))
            write_rgb(frame_dir / f"{name}_mask_color.png", mask_to_color(mask, REGION_COLORS[name]))

        edit_tile = montage(
            [
                ("RGB", rgb),
                ("regions", overlay),
                ("upper edit", upper_edit),
                ("lower edit", lower_edit),
                ("upper+lower", cloth_edit),
                ("shoes edit", shoes_edit),
            ],
            args.panel_width,
            args.header_height,
            args.gap,
        )
        write_rgb(output_dir / "edit_montages" / render_name, edit_tile)
        edit_tiles.append(edit_tile)

        interpret_panels = []
        for title, rel_dir in INTERPRETABILITY_PANELS:
            panel_path = split_dir / rel_dir / render_name
            if panel_path.exists():
                interpret_panels.append((title, read_rgb(panel_path)))
        if interpret_panels:
            interp_tile = montage(interpret_panels, args.panel_width, args.header_height, args.gap)
            write_rgb(output_dir / "interpretability_montages" / render_name, interp_tile)
            interpret_tiles.append(interp_tile)

        write_rgb(output_dir / "region_cutout_montages" / render_name, regions)
        region_tiles.append(regions)

        coverage = {name: float((mask > 0.5).mean()) for name, mask in masks.items()}
        pixel_count = {name: int((mask > 0.5).sum()) for name, mask in masks.items()}
        records.append(
            {
                "render_name": render_name,
                "frame_dir": str(frame_dir.relative_to(output_dir)),
                "mask_source": args.mask_source,
                "mask_source_detail": mask_source_detail,
                "coverage": coverage,
                "pixel_count": pixel_count,
            }
        )

    write_rgb(output_dir / "stageC_edit_demo_stacked.png", stack_tiles(edit_tiles, args.gap))
    write_rgb(output_dir / "stageC_interpretability_stacked.png", stack_tiles(interpret_tiles, args.gap))
    write_rgb(output_dir / "stageC_region_cutouts_stacked.png", stack_tiles(region_tiles, args.gap))

    manifest = {
        "stage": "StageC_min_semantic_editable_demo",
        "source_exp": str(args.exp_dir),
        "split": args.split,
        "output_dir": str(output_dir),
        "mask_source": args.mask_source,
        "parser_root": str(args.parser_root),
        "dataset_root": str(args.dataset_root),
        "num_views": len(records),
        "edit_alpha": args.edit_alpha,
        "overlay_alpha": args.overlay_alpha,
        "regions": list(REGIONS),
        "outputs": {
            "edit_stacked": "stageC_edit_demo_stacked.png",
            "interpretability_stacked": "stageC_interpretability_stacked.png",
            "region_cutouts_stacked": "stageC_region_cutouts_stacked.png",
        },
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved StageC demo for {len(records)} views to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
