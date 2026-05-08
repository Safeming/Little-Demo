#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare a minimal CIHP-style validation set for Hulk parsing inference.')
    parser.add_argument('--image-dir', type=Path, required=True, help='Directory containing input RGB frames, e.g. CoreView_377/1')
    parser.add_argument('--output-root', type=Path, required=True, help='Target root, e.g. /remote-home/ming/Hulk/data/zju377_cam1_cihp')
    parser.add_argument('--link', action='store_true', help='Use symlink instead of copying RGB images')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_dir = args.image_dir
    out_root = args.output_root
    img_out = out_root / 'instance-level_human_parsing' / 'Validation' / 'Images'
    lab_out = out_root / 'instance-level_human_parsing' / 'Validation' / 'Category_ids'
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)

    frames = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg'}])
    ids = []
    for src in frames:
        stem = src.stem
        ids.append(stem)
        dst_img = img_out / f'{stem}.jpg'
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        if args.link:
            dst_img.symlink_to(src)
        else:
            if src.suffix.lower() == '.jpg':
                shutil.copy2(src, dst_img)
            else:
                img = cv2.imread(str(src), cv2.IMREAD_COLOR)
                cv2.imwrite(str(dst_img), img)

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        dummy = np.zeros((h, w), dtype=np.uint8)
        cv2.imwrite(str(lab_out / f'{stem}.png'), dummy)

    val_id = out_root / 'instance-level_human_parsing' / 'Validation' / 'val_id.txt'
    val_id.write_text('\n'.join(ids) + ('\n' if ids else ''))
    print(f'Prepared {len(ids)} frames at {out_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
