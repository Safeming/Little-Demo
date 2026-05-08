#!/usr/bin/env python3
"""Convenience wrapper for continuous-frame temporal interpretability export."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MAPS = ['temporal', 'layer', 'region', 'thin', 'semantic']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run render.py in video mode to export continuous-frame temporal binding maps.'
    )
    parser.add_argument('--load-ckpt', type=Path, required=True,
                        help='Checkpoint path, e.g. exp/.../ckpt15000.pth')
    parser.add_argument('--dataset', default='zjumocap_377_mono')
    parser.add_argument('--rigid', default='explicit_binding')
    parser.add_argument('--non-rigid', default='hashgrid')
    parser.add_argument('--pose-correction', default='direct')
    parser.add_argument('--texture', default='shallow_mlp')
    parser.add_argument('--exp-dir', type=Path, required=True,
                        help='Output experiment directory for the video interpretability export.')
    parser.add_argument('--maps', nargs='*', default=DEFAULT_MAPS,
                        help='Binding map names to export in video mode.')
    parser.add_argument('--iteration', type=int, default=15000,
                        help='Iteration number used by render.py for bookkeeping.')
    parser.add_argument('--extra-override', action='append', default=[],
                        help='Additional Hydra override passed through verbatim. Repeatable.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    map_override = '[' + ','.join(args.maps) + ']'
    command = [
        sys.executable,
        'render.py',
        'mode=test',
        f'dataset={args.dataset}',
        f'rigid={args.rigid}',
        f'non_rigid={args.non_rigid}',
        f'pose_correction={args.pose_correction}',
        f'texture={args.texture}',
        'dataset.test_mode=video',
        'wandb_disable=true',
        '+export_interpretability=true',
        f'+binding_map_names={map_override}',
        f'+exp_dir={args.exp_dir.as_posix()}',
        f'load_ckpt={args.load_ckpt.as_posix()}',
        f'opt.iterations={args.iteration}',
    ]
    command.extend(args.extra_override)
    print('Running:', ' '.join(command))
    return subprocess.call(command)


if __name__ == '__main__':
    raise SystemExit(main())
