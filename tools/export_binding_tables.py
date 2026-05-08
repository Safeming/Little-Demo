#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from experiment_utils import collect_experiment_record, flatten_record, write_csv

METRIC_FIELDS = ['label', 'psnr', 'ssim', 'lpips', 'l1']
BINDING_FIELDS = [
    'label',
    'layer_rigid', 'layer_soft', 'layer_free',
    'region_body', 'region_soft', 'region_cloth',
    'semantic_stability', 'semantic_distance',
    'thin_score', 'temporal_slip', 'temporal_nonzero_frames',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export paper-ready CSV and LaTeX tables for experiment comparison.')
    parser.add_argument('--exp-dirs', type=Path, nargs='+', required=True,
                        help='Experiment directories to include.')
    parser.add_argument('--labels', nargs='*', default=None,
                        help='Optional custom labels aligned with --exp-dirs.')
    parser.add_argument('--split', default='test-view')
    parser.add_argument('--output-dir', type=Path, default=Path('exp/comparisons/latest/tables'))
    return parser.parse_args()


def to_latex(rows: list[dict[str, object]], fields: list[str]) -> str:
    header = ' & '.join(fields) + ' \\\\'
    lines = ['\\begin{tabular}{' + 'l' * len(fields) + '}', '\\toprule', header, '\\midrule']
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, '')
            if isinstance(value, float):
                values.append(f'{value:.4f}')
            else:
                values.append(str(value))
        lines.append(' & '.join(values) + ' \\\\')
    lines.extend(['\\bottomrule', '\\end{tabular}'])
    return '\n'.join(lines) + '\n'


def main() -> int:
    args = parse_args()
    if args.labels and len(args.labels) != len(args.exp_dirs):
        raise ValueError('--labels must match the number of --exp-dirs')

    rows = []
    for index, exp_dir in enumerate(args.exp_dirs):
        record = collect_experiment_record(exp_dir, args.split)
        if args.labels:
            record['label'] = args.labels[index]
        rows.append(flatten_record(record))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = [{field: row.get(field, '') for field in METRIC_FIELDS} for row in rows]
    binding_rows = [{field: row.get(field, '') for field in BINDING_FIELDS} for row in rows]

    write_csv(args.output_dir / 'metrics_table.csv', metric_rows, METRIC_FIELDS)
    write_csv(args.output_dir / 'binding_table.csv', binding_rows, BINDING_FIELDS)
    (args.output_dir / 'metrics_table.tex').write_text(to_latex(metric_rows, METRIC_FIELDS))
    (args.output_dir / 'binding_table.tex').write_text(to_latex(binding_rows, BINDING_FIELDS))

    print(f'Wrote tables to {args.output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
