#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path('/remote-home/ming/3dgs-avatar-release-main')
DEFAULT_HULK_ROOT = Path('/remote-home/ming/Hulk')
DEFAULT_TEMPLATE = DEFAULT_HULK_ROOT / 'experiments' / 'release' / 'Hulk_vit-B_zju377_cam1_cihp.yaml'
DEFAULT_TEST_CONFIG = DEFAULT_HULK_ROOT / 'experiments' / 'release' / 'vd_par_cihp_flip_test_zju377.yaml'
DEFAULT_DATA_ROOT = DEFAULT_HULK_ROOT / 'data' / 'zju377_multiview_cihp'
DEFAULT_GENERATED_DIR = DEFAULT_HULK_ROOT / 'experiments' / 'release' / 'generated_zju377_multiview'
DEFAULT_LOAD_PATH = DEFAULT_HULK_ROOT / 'experiments' / 'release' / 'checkpoints' / 'Hulk_vit-B' / 'ckpt_task18_iter_newest.pth.tar'
DATA_PATH_RE = re.compile(r'^(\s*data_path:\s*)(\S+)(.*)$')


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
    parser = argparse.ArgumentParser(description='Run Hulk parsing inference for multiple ZJU cameras.')
    parser.add_argument('--subject', default='CoreView_377')
    parser.add_argument('--cameras', default='1-20', help='Camera ids, e.g. 1-20 or 1,2,5')
    parser.add_argument('--hulk-root', type=Path, default=DEFAULT_HULK_ROOT)
    parser.add_argument('--template-config', type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument('--test-config', type=Path, default=DEFAULT_TEST_CONFIG)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--generated-config-dir', type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument('--load-path', type=Path, default=DEFAULT_LOAD_PATH)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--spec-ginfo-index', type=int, default=18)
    parser.add_argument('--exp-prefix', default='zju377_mv_hulk')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def write_camera_config(template_path: Path, output_path: Path, data_path: Path) -> None:
    text = template_path.read_text()
    replaced = 0
    lines = []
    for line in text.splitlines():
        match = DATA_PATH_RE.match(line)
        if match:
            lines.append(f"{match.group(1)}{data_path.as_posix()}{match.group(3)}")
            replaced += 1
        else:
            lines.append(line)
    if replaced == 0:
        raise RuntimeError(f'Failed to rewrite data_path in {template_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n')


def main() -> int:
    args = parse_args()
    cameras = parse_camera_spec(args.cameras)
    release_dir = args.hulk_root / 'experiments' / 'release'
    for cam in cameras:
        cam_data_root = args.data_root.resolve() / args.subject / f'cam{int(cam)}'
        if not cam_data_root.exists():
            raise FileNotFoundError(f'Missing prepared Hulk input root: {cam_data_root}')
        generated_config = args.generated_config_dir.resolve() / f'Hulk_vit-B_{args.subject}_cam{int(cam)}_cihp.yaml'
        write_camera_config(args.template_config.resolve(), generated_config, cam_data_root)
        expname = f'{args.exp_prefix}_cam{int(cam)}'
        command = [
            args.python,
            str((args.hulk_root / 'test_mae.py').resolve()),
            '--expname',
            expname,
            '--config',
            str(generated_config),
            '--test_config',
            str(args.test_config.resolve()),
            '--spec_ginfo_index',
            str(args.spec_ginfo_index),
            '--load-path',
            str(args.load_path.resolve()),
        ]
        print('Running:', ' '.join(command))
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=release_dir)
    print(f'Launched Hulk multiview parsing for {args.subject}: {", ".join(cameras)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
