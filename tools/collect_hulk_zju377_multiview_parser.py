#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DEFAULT_HULK_ROOT = Path('/remote-home/ming/Hulk')
DEFAULT_TARGET_ROOT = Path('/remote-home/ming/3dgs-avatar-release-main/data/parsers_from_hulk_multiview')


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
    return [str(cam) for cam in sorted(set(cams))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Collect per-camera Hulk pseudo labels into 3DGS cihp_subject parser layout.')
    parser.add_argument('--subject', default='CoreView_377')
    parser.add_argument('--cameras', default='1-20', help='Camera ids, e.g. 1-20 or 1,2,5')
    parser.add_argument('--hulk-root', type=Path, default=DEFAULT_HULK_ROOT)
    parser.add_argument('--target-root', type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument('--exp-prefix', default='zju377_mv_hulk')
    parser.add_argument('--link', action='store_true', help='Symlink pseudo labels instead of copying them.')
    parser.add_argument('--allow-legacy-cam1', action='store_true', help='For cam1, fall back to the old shared pseudo_labels dir if no per-exp output exists.')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def resolve_source_dir(hulk_root: Path, expname: str, cam: str, allow_legacy_cam1: bool) -> Path:
    base = hulk_root / 'experiments' / 'release' / 'test_results' / 'checkpoints'
    per_exp = base / expname / 'pseudo_labels'
    if per_exp.exists():
        return per_exp
    if allow_legacy_cam1 and int(cam) == 1:
        legacy = base / 'pseudo_labels'
        if legacy.exists():
            return legacy
    raise FileNotFoundError(f'Missing Hulk pseudo labels for cam{cam}: {per_exp}')


def main() -> int:
    args = parse_args()
    cameras = parse_camera_spec(args.cameras)
    for cam in cameras:
        expname = f'{args.exp_prefix}_cam{int(cam)}'
        source_dir = resolve_source_dir(args.hulk_root.resolve(), expname, cam, args.allow_legacy_cam1)
        target_dir = args.target_root.resolve() / args.subject / 'mask_cihp' / f'Camera_B{int(cam)}'
        sources = sorted(source_dir.glob('*.png'))
        print(f'Collecting cam{cam}: {len(sources)} masks -> {target_dir}')
        if args.dry_run:
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        for src in sources:
            dst = target_dir / src.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.link:
                dst.symlink_to(src)
            else:
                shutil.copy2(src, dst)
    print(f'Built parser_root at {args.target_root.resolve() / args.subject / "mask_cihp"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
