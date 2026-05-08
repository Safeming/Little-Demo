#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiment_utils import collect_experiment_record, flatten_record, rows_to_markdown, write_csv, write_json

DEFAULT_FIELDS = [
    'label', 'psnr', 'ssim', 'lpips', 'l1',
    'layer_rigid', 'layer_soft', 'layer_free',
    'region_body', 'region_soft', 'region_cloth',
    'semantic_stability', 'thin_score', 'temporal_slip',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare experiment metrics and interpretability summaries.')
    parser.add_argument('--exp-dirs', type=Path, nargs='+', required=True,
                        help='Experiment directories to compare.')
    parser.add_argument('--labels', nargs='*', default=None,
                        help='Optional custom labels aligned with --exp-dirs.')
    parser.add_argument('--split', default='test-view',
                        help='Interpretability split used to look up binding_analysis/aggregate.json.')
    parser.add_argument('--output-dir', type=Path, default=Path('exp/comparisons/latest'),
                        help='Directory for comparison.json/csv/md.')
    parser.add_argument('--sort-by', default=None,
                        help='Optional numeric field to sort descending, e.g. psnr.')
    return parser.parse_args()


def maybe_sort(rows: list[dict[str, Any]], sort_by: str | None) -> list[dict[str, Any]]:
    if not sort_by:
        return rows
    return sorted(rows, key=lambda row: (row.get(sort_by) is not None, row.get(sort_by) or float('-inf')), reverse=True)


def main() -> int:
    args = parse_args()
    if args.labels and len(args.labels) != len(args.exp_dirs):
        raise ValueError('--labels must match the number of --exp-dirs')

    records = []
    for index, exp_dir in enumerate(args.exp_dirs):
        record = collect_experiment_record(exp_dir, args.split)
        if args.labels:
            record['label'] = args.labels[index]
        records.append(record)

    rows = [flatten_record(record) for record in records]
    rows = maybe_sort(rows, args.sort_by)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_json(args.output_dir / 'comparison.json', {'split': args.split, 'records': records, 'rows': rows})
    write_csv(args.output_dir / 'comparison.csv', rows, list(rows[0].keys()) if rows else ['label'])
    markdown = rows_to_markdown(rows, [field for field in DEFAULT_FIELDS if field in (rows[0].keys() if rows else DEFAULT_FIELDS)])
    (args.output_dir / 'comparison.md').write_text(markdown + '\n')

    print(f'Wrote comparison bundle to {args.output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
