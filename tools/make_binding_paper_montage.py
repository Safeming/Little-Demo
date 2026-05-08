#!/usr/bin/env python3
"""Build paper-ready montage figures for binding interpretability."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import cv2
import imageio.v2 as imageio
import numpy as np

RENDER_PATTERN = re.compile(r"render_c(?P<cam>\d+)_f(?P<frame>\d+)\.png$")
PANEL_SPECS = {
    'gt': ('Ground Truth', None),
    'render': ('Render', 'renders'),
    'layer': ('Layer', 'binding_maps/layer'),
    'region': ('Region', 'binding_maps/region'),
    'thin': ('Thin', 'binding_maps/thin'),
    'semantic': ('Semantic', 'binding_maps/semantic'),
    'temporal': ('Temporal', 'binding_maps/temporal'),
    'body_prob': ('Body Prob', 'binding_maps/body_prob'),
    'soft_prob': ('Soft Prob', 'binding_maps/soft_prob'),
    'cloth_prob': ('Cloth Prob', 'binding_maps/cloth_prob'),
}
DEFAULT_PANELS = ['gt', 'render', 'layer', 'region', 'thin']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create paper-ready montage images from render and binding-map outputs.'
    )
    parser.add_argument('--exp-dir', type=Path, required=True,
                        help='Experiment directory containing split folders.')
    parser.add_argument('--gt-root', type=Path, default=Path('data/ZJUMoCap/CoreView_377'),
                        help='Ground-truth camera root, e.g. data/ZJUMoCap/CoreView_377.')
    parser.add_argument('--split', default='test-view',
                        help='Render split folder under the experiment directory.')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Optional output directory. Defaults to <exp-dir>/<split>/paper_montages.')
    parser.add_argument('--limit', type=int, default=0,
                        help='Optional positive limit on the number of montage images to create.')
    parser.add_argument('--select', nargs='*', default=None,
                        help='Optional explicit render filenames, e.g. render_c03_f000120.png.')
    parser.add_argument('--panels', nargs='*', default=DEFAULT_PANELS,
                        help='Panel keys to include: ' + ', '.join(PANEL_SPECS.keys()))
    parser.add_argument('--crop', nargs=4, type=int, default=None,
                        metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help='Optional crop box applied to every panel before montage export.')
    parser.add_argument('--font-scale', type=float, default=0.85,
                        help='OpenCV text scale for panel titles.')
    parser.add_argument('--panel-gap', type=int, default=10,
                        help='Gap in pixels between panels.')
    parser.add_argument('--header-height', type=int, default=44,
                        help='Height in pixels for the title bar above the panels.')
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    image = imageio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] > 3:
        image = image[..., :3]
    return image.astype(np.uint8)


def resize_to(image: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    height, width = size_hw
    if image.shape[0] == height and image.shape[1] == width:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def crop_and_resize(image: np.ndarray, crop: Optional[tuple[int, int, int, int]], size_hw: tuple[int, int]) -> np.ndarray:
    if crop is None:
        return resize_to(image, size_hw)
    x1, y1, x2, y2 = crop
    x1 = max(0, min(x1, image.shape[1] - 1))
    y1 = max(0, min(y1, image.shape[0] - 1))
    x2 = max(x1 + 1, min(x2, image.shape[1]))
    y2 = max(y1 + 1, min(y2, image.shape[0]))
    cropped = image[y1:y2, x1:x2]
    return resize_to(cropped, size_hw)


def find_gt_image(gt_root: Path, cam: str, frame: str) -> Optional[Path]:
    cam_dir = gt_root / str(int(cam))
    candidates = [
        cam_dir / f'{int(frame)}.jpg',
        cam_dir / f'{int(frame)}.png',
        cam_dir / f'{int(frame):06d}.jpg',
        cam_dir / f'{int(frame):06d}.png',
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def draw_label(canvas: np.ndarray, text: str, left: int, top: int, width: int, header_height: int, font_scale: float) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = left + max((width - text_w) // 2, 8)
    y = top + max((header_height + text_h) // 2 - baseline, text_h + 4)
    cv2.putText(canvas, text, (x, y), font, font_scale, (25, 25, 25), thickness, cv2.LINE_AA)


def build_montage(panels: list[tuple[str, np.ndarray]], header_height: int, panel_gap: int, font_scale: float) -> np.ndarray:
    panel_h, panel_w = panels[0][1].shape[:2]
    canvas_h = panel_h + header_height
    canvas_w = len(panels) * panel_w + (len(panels) - 1) * panel_gap
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    left = 0
    for title, image in panels:
        canvas[header_height:, left:left + panel_w] = image
        draw_label(canvas, title, left, 0, panel_w, header_height, font_scale)
        left += panel_w + panel_gap

    cv2.line(canvas, (0, header_height - 1), (canvas_w, header_height - 1), (220, 220, 220), 1)
    return canvas


def iter_render_files(render_dir: Path, selected: Optional[Iterable[str]] = None) -> list[Path]:
    if selected:
        return [render_dir / name for name in selected]
    return sorted(render_dir.glob('render_c*_f*.png'))


def resolve_panel_image(panel_key: str, split_dir: Path, render_name: str, gt_root: Path, cam: str, frame: str, target_hw: tuple[int, int], crop: Optional[tuple[int, int, int, int]]) -> tuple[str, np.ndarray]:
    title, relative_dir = PANEL_SPECS[panel_key]
    if panel_key == 'gt':
        gt_path = find_gt_image(gt_root, cam, frame)
        if gt_path is None:
            raise FileNotFoundError(f'No GT found for {render_name}')
        image = load_rgb(gt_path)
    else:
        image_path = split_dir / relative_dir / render_name
        if not image_path.exists():
            raise FileNotFoundError(f'Missing panel image: {image_path}')
        image = load_rgb(image_path)
    image = crop_and_resize(image, crop, target_hw)
    return title, image


def main() -> int:
    args = parse_args()
    split_dir = args.exp_dir / args.split
    render_dir = split_dir / 'renders'
    output_dir = args.output_dir or (split_dir / 'paper_montages')
    output_dir.mkdir(parents=True, exist_ok=True)

    if not render_dir.exists():
        raise FileNotFoundError(f'Missing render directory: {render_dir}')
    invalid = [panel for panel in args.panels if panel not in PANEL_SPECS]
    if invalid:
        raise ValueError(f'Unknown panel keys: {invalid}')

    render_files = iter_render_files(render_dir, args.select)
    if args.limit > 0:
        render_files = render_files[:args.limit]

    crop = tuple(args.crop) if args.crop is not None else None
    saved = 0
    for render_path in render_files:
        if not render_path.exists():
            continue
        match = RENDER_PATTERN.match(render_path.name)
        if match is None:
            continue
        render_img = load_rgb(render_path)
        target_hw = render_img.shape[:2]
        cam = match.group('cam')
        frame = match.group('frame')

        panels = []
        for panel_key in args.panels:
            title, image = resolve_panel_image(panel_key, split_dir, render_path.name, args.gt_root, cam, frame, target_hw, crop)
            panels.append((title, image))

        montage = build_montage(panels, args.header_height, args.panel_gap, args.font_scale)
        suffix = '_crop' if crop is not None else ''
        out_path = output_dir / render_path.name.replace('render_', f'montage{suffix}_')
        imageio.imwrite(out_path, montage)
        saved += 1

    print(f'Saved {saved} montage images to {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
