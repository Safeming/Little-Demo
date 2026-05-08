#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BODY_LABELS = {13, 14, 15, 16, 17}
CLOTH_LABELS = {5, 6, 7, 9, 10, 11, 12}
HAIR_LABELS = {1, 2}
SHOE_LABELS = {18, 19}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Directly convert Hulk parsing PNGs into raw body/cloth/valid/uncertain masks.')
    parser.add_argument('--mask-dir', type=Path, required=True)
    parser.add_argument('--target-root', type=Path, default=Path('/remote-home/ming/3dgs-avatar-release-main/data/parsing_priors_from_hulk_raw'))
    parser.add_argument('--subject', required=True)
    parser.add_argument('--camera', required=True)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def main() -> int:
    args = parse_args()
    target = args.target_root / args.subject / args.camera
    body_dir = target / 'body'
    cloth_dir = target / 'cloth'
    valid_dir = target / 'valid'
    uncertain_dir = target / 'uncertain'
    for d in (body_dir, cloth_dir, valid_dir, uncertain_dir):
        ensure_dir(d)

    total = 0
    for mask_path in sorted(args.mask_dir.glob('*.png')):
        out_paths = [body_dir / mask_path.name, cloth_dir / mask_path.name, valid_dir / mask_path.name, uncertain_dir / mask_path.name]
        if (not args.overwrite) and all(p.exists() for p in out_paths):
            continue
        mask = read_mask(mask_path)
        hair = np.isin(mask, list(HAIR_LABELS)).astype(np.uint8)
        shoes = np.isin(mask, list(SHOE_LABELS)).astype(np.uint8)
        body = np.isin(mask, list(BODY_LABELS)).astype(np.uint8)
        cloth = np.isin(mask, list(CLOTH_LABELS)).astype(np.uint8)
        valid = ((body + cloth) > 0).astype(np.uint8)
        # raw direct conversion: only mark body-cloth contact as uncertain, no fallback logic.
        k = np.ones((3, 3), np.uint8)
        body_d = cv2.dilate(body, k)
        cloth_d = cv2.dilate(cloth, k)
        uncertain = ((body_d * cloth_d) > 0).astype(np.uint8)
        valid = (valid * (1 - hair) * (1 - shoes)).astype(np.uint8)
        body = (body * valid * (1 - uncertain)).astype(np.uint8)
        cloth = (cloth * valid * (1 - uncertain)).astype(np.uint8)
        cv2.imwrite(str(body_dir / mask_path.name), body * 255)
        cv2.imwrite(str(cloth_dir / mask_path.name), cloth * 255)
        cv2.imwrite(str(valid_dir / mask_path.name), valid * 255)
        cv2.imwrite(str(uncertain_dir / mask_path.name), uncertain * 255)
        total += 1
    print(f'Built Hulk raw priors for {args.subject} cam {args.camera}: {total} masks written to {target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
