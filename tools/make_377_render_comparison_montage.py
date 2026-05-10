#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


RENDER_RE = re.compile(r"render_c(?P<cam>\d+)_f(?P<frame>\d+)\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GT + multi-experiment render comparison montages.")
    parser.add_argument("--render-exp", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--gt-root", type=Path, default=Path("data/ZJUMoCap/CoreView_377"))
    parser.add_argument("--split", default="test-view")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--select", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--crop", nargs=4, type=int, default=None)
    parser.add_argument("--panel-width", type=int, default=260)
    parser.add_argument("--header-height", type=int, default=38)
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--stack", action="store_true", help="Also write a vertically stacked single-sheet montage.")
    return parser.parse_args()


def _read_rgb(path: Path) -> np.ndarray:
    image = imageio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] > 3:
        image = image[..., :3]
    return image.astype(np.uint8)


def _find_gt(gt_root: Path, cam: str, frame: str) -> Path:
    cam_dir = gt_root / str(int(cam))
    frame_int = int(frame)
    candidates = [
        cam_dir / f"{frame_int:06d}.jpg",
        cam_dir / f"{frame_int:06d}.png",
        cam_dir / f"{frame_int}.jpg",
        cam_dir / f"{frame_int}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(candidates[0])


def _crop(image: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return image
    x1, y1, x2, y2 = crop
    x1 = max(0, min(x1, image.shape[1] - 1))
    y1 = max(0, min(y1, image.shape[0] - 1))
    x2 = max(x1 + 1, min(x2, image.shape[1]))
    y2 = max(y1 + 1, min(y2, image.shape[0]))
    return image[y1:y2, x1:x2]


def _resize(image: np.ndarray, panel_width: int) -> np.ndarray:
    scale = float(panel_width) / float(image.shape[1])
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (panel_width, height), interpolation=cv2.INTER_LINEAR)


def _draw_label(canvas: np.ndarray, text: str, left: int, width: int, header_height: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.68
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = left + max(6, (width - tw) // 2)
    y = max(th + 5, (header_height + th) // 2 - baseline)
    cv2.putText(canvas, text, (x, y), font, scale, (30, 30, 30), thickness, cv2.LINE_AA)


def _build_tile(panels: list[tuple[str, np.ndarray]], args: argparse.Namespace) -> np.ndarray:
    resized = [(label, _resize(_crop(image, tuple(args.crop) if args.crop else None), args.panel_width)) for label, image in panels]
    panel_h = max(image.shape[0] for _, image in resized)
    panel_w = args.panel_width
    canvas_h = panel_h + args.header_height
    canvas_w = len(resized) * panel_w + (len(resized) - 1) * args.gap
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    left = 0
    for label, image in resized:
        if image.shape[0] != panel_h:
            image = cv2.resize(image, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)
        canvas[args.header_height :, left : left + panel_w] = image
        _draw_label(canvas, label, left, panel_w, args.header_height)
        left += panel_w + args.gap
    cv2.line(canvas, (0, args.header_height - 1), (canvas_w, args.header_height - 1), (220, 220, 220), 1)
    return canvas


def main() -> int:
    args = parse_args()
    if len(args.labels) != len(args.render_exp):
        raise ValueError("--labels length must match --render-exp length")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    first_render_dir = args.render_exp[0] / args.split / "renders"
    if args.select:
        names = list(args.select)
    else:
        names = [path.name for path in sorted(first_render_dir.glob("render_c*_f*.png"))]
        if args.limit > 0:
            names = names[: args.limit]

    tiles = []
    for name in names:
        match = RENDER_RE.match(name)
        if match is None:
            continue
        cam, frame = match.group("cam"), match.group("frame")
        gt = _read_rgb(_find_gt(args.gt_root, cam, frame))
        panels = [("GT", gt)]
        missing = False
        for label, exp in zip(args.labels, args.render_exp):
            path = exp / args.split / "renders" / name
            if not path.exists():
                missing = True
                break
            panels.append((label, _read_rgb(path)))
        if missing:
            continue
        tile = _build_tile(panels, args)
        out_path = args.output_dir / name.replace("render_", "compare_")
        imageio.imwrite(out_path, tile)
        tiles.append(tile)

    if args.stack and tiles:
        width = max(tile.shape[1] for tile in tiles)
        height = sum(tile.shape[0] for tile in tiles) + args.gap * (len(tiles) - 1)
        sheet = np.full((height, width, 3), 245, dtype=np.uint8)
        top = 0
        for tile in tiles:
            sheet[top : top + tile.shape[0], : tile.shape[1]] = tile
            top += tile.shape[0] + args.gap
        imageio.imwrite(args.output_dir / "stacked_comparison.png", sheet)

    print(f"Saved {len(tiles)} comparison montages to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
