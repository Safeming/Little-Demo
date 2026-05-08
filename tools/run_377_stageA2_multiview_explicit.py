#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPTION_STACK = ['stageA_377_multiview_explicit_hq_v1']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Stage A2 multiview RGB reconstruction for CoreView_377: multiview + full image + original mask + no Hulk + no crop, while keeping explicit_binding.'
    )
    parser.add_argument('--exp-dir', type=Path, required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--dataset', default='zjumocap_377_multiview_hq')
    parser.add_argument('--dataset-root', default=str(ROOT / 'data' / 'ZJUMoCap'))
    parser.add_argument('--rigid', default='explicit_binding')
    parser.add_argument('--non-rigid', default='hashgrid')
    parser.add_argument('--pose-correction', default='direct')
    parser.add_argument('--texture', default='shallow_mlp')
    parser.add_argument('--option', action='append', default=[])
    parser.add_argument('--extra-override', action='append', default=[])
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    option_stack = list(dict.fromkeys(DEFAULT_OPTION_STACK + args.option))
    wandb_disable = 'false' if args.wandb else 'true'
    return [
        args.python,
        'train.py',
        'mode=train',
        f'dataset={args.dataset}',
        f'dataset.root_dir={args.dataset_root}',
        'dataset.preload=false',
        'dataset.parsing_prior.enable=false',
        'dataset.person_crop.enable=false',
        f'rigid={args.rigid}',
        f'non_rigid={args.non_rigid}',
        f'pose_correction={args.pose_correction}',
        f'texture={args.texture}',
        f'option=[{",".join(option_stack)}]',
        f'+exp_dir={args.exp_dir.resolve().as_posix()}',
        f'wandb_disable={wandb_disable}',
        *args.extra_override,
    ]


def main() -> int:
    args = parse_args()
    command = build_command(args)
    print('Running:', ' '.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True, cwd=ROOT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
