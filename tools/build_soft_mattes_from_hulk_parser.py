#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/remote-home/ming/3dgs-avatar-release-main')
DEFAULT_PARSER_ROOT = ROOT / 'data' / 'parsers_from_hulk_multiview'
DEFAULT_IMAGE_ROOT = ROOT / 'data' / 'ZJUMoCap'
DEFAULT_TARGET_ROOT = ROOT / 'data' / 'mattes_from_hulk_multiview'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build soft alpha mattes from Hulk multiview parser labels using an OpenCV trimap + GrabCut refinement pipeline.'
    )
    parser.add_argument('--subject', default='CoreView_377')
    parser.add_argument('--cameras', default='1-20', help='Camera ids, e.g. 1-20 or 1,2,5')
    parser.add_argument('--parser-root', type=Path, default=DEFAULT_PARSER_ROOT)
    parser.add_argument('--image-root', type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument('--target-root', type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument('--target-dirname', default='alpha')
    parser.add_argument('--mask-fallback-from-zju', action='store_true', default=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--grabcut-iters', type=int, default=2)
    parser.add_argument('--fg-erode', type=int, default=9)
    parser.add_argument('--bg-dilate', type=int, default=21)
    parser.add_argument('--boundary-width', type=int, default=9)
    parser.add_argument('--feather-sigma', type=float, default=2.2)
    parser.add_argument('--prob-fg-alpha', type=float, default=0.85)
    parser.add_argument('--prob-bg-alpha', type=float, default=0.15)
    parser.add_argument('--limit', type=int, default=0, help='Optional max frames per camera for smoke testing.')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def parse_camera_spec(spec: str) -> list[int]:
    cameras: set[int] = set()
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            start, end = chunk.split('-', 1)
            start_i = int(start)
            end_i = int(end)
            lo, hi = sorted((start_i, end_i))
            cameras.update(range(lo, hi + 1))
        else:
            cameras.add(int(chunk))
    return sorted(cameras)


def _odd(value: int) -> int:
    value = max(int(value), 1)
    return value if value % 2 == 1 else value + 1


def elliptical_kernel(size: int) -> np.ndarray:
    size = _odd(size)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def resolve_image_path(image_dir: Path, stem: str) -> Path | None:
    for suffix in ('.jpg', '.png', '.jpeg'):
        path = image_dir / f'{stem}{suffix}'
        if path.exists():
            return path
    return None


def build_trimap_masks(fg_mask: np.ndarray, fg_erode: int, bg_dilate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fg_mask = (fg_mask > 0).astype(np.uint8)
    sure_fg = cv2.erode(fg_mask, elliptical_kernel(fg_erode), iterations=1)
    dilated = cv2.dilate(fg_mask, elliptical_kernel(bg_dilate), iterations=1)
    sure_bg = (1 - dilated).astype(np.uint8)
    unknown = np.clip(1 - sure_fg - sure_bg, 0, 1).astype(np.uint8)
    return sure_fg, sure_bg, unknown


def run_grabcut(image_bgr: np.ndarray, sure_fg: np.ndarray, sure_bg: np.ndarray, unknown: np.ndarray, iterations: int) -> np.ndarray:
    mask = np.full(image_bgr.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    mask[(sure_fg > 0) | (unknown > 0)] = cv2.GC_PR_FGD
    mask[sure_bg > 0] = cv2.GC_BGD
    mask[sure_fg > 0] = cv2.GC_FGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image_bgr, mask, None, bgd, fgd, max(int(iterations), 1), cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask
    return mask


def soften_from_grabcut(gc_mask: np.ndarray, sure_fg: np.ndarray, sure_bg: np.ndarray, boundary_width: int, feather_sigma: float, prob_fg_alpha: float, prob_bg_alpha: float) -> np.ndarray:
    alpha = np.full(gc_mask.shape, float(prob_bg_alpha), dtype=np.float32)
    alpha[gc_mask == cv2.GC_BGD] = 0.0
    alpha[gc_mask == cv2.GC_PR_BGD] = float(prob_bg_alpha)
    alpha[gc_mask == cv2.GC_PR_FGD] = float(prob_fg_alpha)
    alpha[gc_mask == cv2.GC_FGD] = 1.0

    binary = np.isin(gc_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    width = _odd(boundary_width)
    band = cv2.dilate(binary, elliptical_kernel(width), iterations=1) - cv2.erode(binary, elliptical_kernel(width), iterations=1)
    if band.any():
        blurred = cv2.GaussianBlur(alpha, (0, 0), sigmaX=max(float(feather_sigma), 0.1), sigmaY=max(float(feather_sigma), 0.1))
        alpha[band > 0] = blurred[band > 0]

    alpha[sure_fg > 0] = 1.0
    alpha[sure_bg > 0] = 0.0
    return np.clip(alpha, 0.0, 1.0)


def build_soft_alpha(image_bgr: np.ndarray, seed_mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    fg_mask = (seed_mask > 0).astype(np.uint8)
    if fg_mask.sum() == 0:
        return np.zeros(parser_mask.shape, dtype=np.float32)

    sure_fg, sure_bg, unknown = build_trimap_masks(fg_mask, args.fg_erode, args.bg_dilate)
    gc_mask = run_grabcut(image_bgr, sure_fg, sure_bg, unknown, args.grabcut_iters)
    return soften_from_grabcut(
        gc_mask,
        sure_fg=sure_fg,
        sure_bg=sure_bg,
        boundary_width=args.boundary_width,
        feather_sigma=args.feather_sigma,
        prob_fg_alpha=args.prob_fg_alpha,
        prob_bg_alpha=args.prob_bg_alpha,
    )


def _camera_seed_paths(args: argparse.Namespace, camera_id: int, image_dir: Path) -> tuple[list[Path], str]:
    parser_dir = args.parser_root / args.subject / 'mask_cihp' / f'Camera_B{camera_id}'
    if parser_dir.exists():
        return sorted(parser_dir.glob('*.png')), 'parser'

    if not args.mask_fallback_from_zju:
        return [], 'missing'

    mask_paths = sorted(
        p for p in image_dir.glob('*.png')
        if resolve_image_path(image_dir, p.stem) is not None
    )
    return mask_paths, 'zju_mask'


def process_camera(args: argparse.Namespace, camera_id: int) -> tuple[int, int]:
    image_dir = args.image_root / args.subject / str(camera_id)
    target_dir = args.target_root / args.subject / args.target_dirname / f'Camera_B{camera_id}'

    if not image_dir.exists():
        print(f'[skip] missing image dir: {image_dir}')
        return 0, 0

    seed_paths, seed_kind = _camera_seed_paths(args, camera_id, image_dir)
    if not seed_paths:
        print(f'[skip] missing parser or fallback mask for camera {camera_id:02d}')
        return 0, 0

    if args.limit > 0:
        seed_paths = seed_paths[:args.limit]

    total = 0
    written = 0
    if not args.dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    for seed_path in seed_paths:
        total += 1
        target_path = target_dir / seed_path.name
        if target_path.exists() and not args.overwrite:
            continue

        image_path = resolve_image_path(image_dir, seed_path.stem)
        if image_path is None:
            print(f'[warn] missing RGB for {seed_path}')
            continue

        seed_mask = cv2.imread(seed_path.as_posix(), cv2.IMREAD_GRAYSCALE)
        image_bgr = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
        if seed_mask is None or image_bgr is None:
            print(f'[warn] failed to read pair: {seed_path} / {image_path}')
            continue

        alpha = build_soft_alpha(image_bgr, seed_mask, args)
        if not args.dry_run:
            cv2.imwrite(target_path.as_posix(), np.clip(alpha * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8))
        written += 1

    print(f'[camera {camera_id:02d}] seed={seed_kind} processed {written}/{total} frames')
    return total, written


def main() -> int:
    args = parse_args()
    cameras = parse_camera_spec(args.cameras)
    total = 0
    written = 0
    for camera_id in cameras:
        cam_total, cam_written = process_camera(args, camera_id)
        total += cam_total
        written += cam_written

    print(f'Finished {args.subject}: wrote {written} soft mattes from {total} seed masks into {args.target_root / args.subject / args.target_dirname}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
