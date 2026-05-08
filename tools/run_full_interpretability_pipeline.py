#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiment_utils import find_checkpoint, infer_core_overrides, infer_iteration_from_checkpoint

DEFAULT_VIEW_MAPS = ['layer', 'region', 'body_prob', 'soft_prob', 'cloth_prob', 'semantic', 'temporal', 'thin']
DEFAULT_VIDEO_MAPS = ['temporal', 'layer', 'region', 'thin', 'semantic']
DEFAULT_MONTAGE_PANELS = ['gt', 'render', 'layer', 'region', 'body_prob', 'cloth_prob', 'thin', 'semantic']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the full v4.1 interpretability export pipeline.')
    parser.add_argument('--main-exp', type=Path, required=True,
                        help='Main training experiment directory, e.g. exp/...-main')
    parser.add_argument('--load-ckpt', type=Path, default=None,
                        help='Optional explicit checkpoint path. Overrides --checkpoint-mode.')
    parser.add_argument('--checkpoint-mode', choices=['latest', 'best'], default='latest',
                        help='How to choose a checkpoint when --load-ckpt is omitted.')
    parser.add_argument('--python', default=sys.executable,
                        help='Python interpreter used for render and helper scripts.')
    parser.add_argument('--view-exp-dir', type=Path, default=None,
                        help='Output directory for test-view interpretability export.')
    parser.add_argument('--video-exp-dir', type=Path, default=None,
                        help='Output directory for test-video interpretability export.')
    parser.add_argument('--view-maps', nargs='*', default=DEFAULT_VIEW_MAPS)
    parser.add_argument('--video-maps', nargs='*', default=DEFAULT_VIDEO_MAPS)
    parser.add_argument('--montage-panels', nargs='*', default=DEFAULT_MONTAGE_PANELS)
    parser.add_argument('--skip-view', action='store_true')
    parser.add_argument('--skip-video', action='store_true')
    parser.add_argument('--skip-montage', action='store_true')
    parser.add_argument('--copy-assets', action='store_true',
                        help='Copy keyframe assets into binding_analysis/selected_assets.')
    parser.add_argument('--extra-override', action='append', default=[],
                        help='Additional Hydra override passed to render.py. Repeatable.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing them.')
    return parser.parse_args()


def run(command: list[str], dry_run: bool) -> None:
    print('Running:', ' '.join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    main_exp = args.main_exp.resolve()
    ckpt = args.load_ckpt.resolve() if args.load_ckpt else find_checkpoint(main_exp, mode=args.checkpoint_mode).resolve()
    iteration = infer_iteration_from_checkpoint(ckpt) or 15000
    view_exp_dir = (args.view_exp_dir or Path(f'{main_exp.as_posix()}_interp_full')).resolve()
    video_exp_dir = (args.video_exp_dir or Path(f'{main_exp.as_posix()}_interp_video')).resolve()
    overrides = infer_core_overrides(main_exp)

    view_render = [
        args.python,
        'render.py',
        'mode=test',
        *overrides,
        'export_interpretability=true',
        f'++binding_map_names=[{",".join(args.view_maps)}]',
        f'+exp_dir={view_exp_dir.as_posix()}',
        f'load_ckpt={ckpt.as_posix()}',
        f'opt.iterations={iteration}',
        *args.extra_override,
    ]
    view_summary = [
        args.python,
        'tools/summarize_binding_interpretability.py',
        '--exp-dir', view_exp_dir.as_posix(),
        '--split', 'test-view',
    ]
    if args.copy_assets:
        view_summary.append('--copy-assets')
    view_montage = [
        args.python,
        'tools/make_binding_paper_montage.py',
        '--exp-dir', view_exp_dir.as_posix(),
        '--split', 'test-view',
        '--panels', *args.montage_panels,
    ]
    video_export = [
        args.python,
        'tools/export_temporal_binding_video.py',
        '--load-ckpt', ckpt.as_posix(),
        '--exp-dir', video_exp_dir.as_posix(),
        '--iteration', str(iteration),
        '--maps', *args.video_maps,
    ]
    for override in overrides:
        key, value = override.split('=', 1)
        flag = '--non-rigid' if key == 'non_rigid' else f'--{key.replace("_", "-")}'
        if key == 'wandb_disable':
            continue
        video_export.extend([flag, value])
    for override in args.extra_override:
        video_export.extend(['--extra-override', override])
    video_summary = [
        args.python,
        'tools/summarize_binding_interpretability.py',
        '--exp-dir', video_exp_dir.as_posix(),
        '--split', 'test-video',
    ]
    if args.copy_assets:
        video_summary.append('--copy-assets')

    if not args.skip_view:
        run(view_render, args.dry_run)
        run(view_summary, args.dry_run)
        if not args.skip_montage:
            run(view_montage, args.dry_run)
    if not args.skip_video:
        run(video_export, args.dry_run)
        run(video_summary, args.dry_run)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
