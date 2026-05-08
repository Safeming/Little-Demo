#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path('/remote-home/ming/3dgs-avatar-release-main')
DEFAULT_ZJU_ROOT = ROOT / 'data' / 'ZJUMoCap'
DEFAULT_HULK_DATA_ROOT = Path('/remote-home/ming/Hulk/data/zju377_multiview_cihp')
PREP_SCRIPT = ROOT / 'tools' / 'prepare_hulk_cihp_infer.py'


def parse_camera_spec(spec: str) -> list[str]:
    cams: list[int] = []
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            start, end = chunk.split('-', 1)
            cams.extend(range(int(start), int(end) + 1))
        else:
            cams.append(int(chunk))
    unique = sorted(set(cams))
    return [str(cam) for cam in unique]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare CIHP-style Hulk inference inputs for multiple ZJU cameras.')
    parser.add_argument('--subject', default='CoreView_377')
    parser.add_argument('--cameras', default='1-23', help='Camera ids, e.g. 1-20 or 1,2,5')
    parser.add_argument('--zju-root', type=Path, default=DEFAULT_ZJU_ROOT)
    parser.add_argument('--hulk-data-root', type=Path, default=DEFAULT_HULK_DATA_ROOT)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--link', action='store_true', help='Symlink RGBs instead of copying.')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cameras = parse_camera_spec(args.cameras)
    subject_root = (args.zju_root / args.subject).resolve()
    if not subject_root.exists():
        raise FileNotFoundError(subject_root)

    for cam in cameras:
        image_dir = subject_root / cam
        if not image_dir.exists():
            raise FileNotFoundError(f'Missing camera directory: {image_dir}')
        out_root = args.hulk_data_root.resolve() / args.subject / f'cam{int(cam)}'
        command = [
            args.python,
            str(PREP_SCRIPT),
            '--image-dir',
            str(image_dir),
            '--output-root',
            str(out_root),
        ]
        if args.link:
            command.append('--link')
        print('Running:', ' '.join(command))
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=ROOT)
    print(f'Prepared Hulk multiview CIHP inputs for {args.subject}: {", ".join(cameras)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
