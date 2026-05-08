#!/usr/bin/env python3
"""Aggregate interpretability summaries and auto-select keyframes."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Aggregate binding-map summaries and select representative keyframes.'
    )
    parser.add_argument('--exp-dir', type=Path, required=True,
                        help='Experiment directory containing split/binding_maps/summary.json.')
    parser.add_argument('--split', default='test-view',
                        help='Split folder, e.g. test-view or test-video.')
    parser.add_argument('--summary-path', type=Path, default=None,
                        help='Optional explicit summary.json path.')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Optional output directory. Defaults to <exp-dir>/<split>/binding_analysis.')
    parser.add_argument('--copy-assets', action='store_true',
                        help='Copy selected renders/maps/montages into the analysis folder for convenience.')
    return parser.parse_args()


def load_summary(path: Path) -> list[dict[str, Any]]:
    with open(path, 'r') as f:
        return json.load(f)


def metric(record: dict[str, Any], section: str, key: str, default: float = 0.0) -> float:
    return float(record.get(section, {}).get(key, default))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)


def std(values: Iterable[float], avg: float) -> float:
    values = list(values)
    if not values:
        return 0.0
    return math.sqrt(sum((v - avg) ** 2 for v in values) / len(values))


def quantile(values: Iterable[float], q: float) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float32), q))


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    layer_rigid = [metric(r, 'layer', 'rigid_ratio') for r in records]
    layer_soft = [metric(r, 'layer', 'soft_ratio') for r in records]
    layer_free = [metric(r, 'layer', 'free_ratio') for r in records]
    region_body = [metric(r, 'region', 'body_prob_mean', metric(r, 'region', 'body_ratio')) for r in records]
    region_soft = [metric(r, 'region', 'soft_prob_mean', metric(r, 'region', 'soft_ratio')) for r in records]
    region_cloth = [metric(r, 'region', 'cloth_prob_mean', metric(r, 'region', 'cloth_ratio')) for r in records]
    semantic_stability = [metric(r, 'semantic', 'semantic_stability_mean') for r in records]
    semantic_distance = [metric(r, 'semantic', 'semantic_distance_mean') for r in records]
    thin_score = [metric(r, 'thin', 'thin_score_mean') for r in records]
    temporal_slip = [metric(r, 'temporal', 'temporal_slip_mean') for r in records]
    temporal_nonzero = sum(v > 1e-8 for v in temporal_slip)

    layer_mean = {
        'rigid': mean(layer_rigid),
        'soft': mean(layer_soft),
        'free': mean(layer_free),
    }
    region_mean = {
        'body': mean(region_body),
        'soft': mean(region_soft),
        'cloth': mean(region_cloth),
    }

    return {
        'num_frames': len(records),
        'layer_mean': layer_mean,
        'layer_std': {
            'rigid': std(layer_rigid, layer_mean['rigid']),
            'soft': std(layer_soft, layer_mean['soft']),
            'free': std(layer_free, layer_mean['free']),
        },
        'region_mean': region_mean,
        'region_std': {
            'body': std(region_body, region_mean['body']),
            'soft': std(region_soft, region_mean['soft']),
            'cloth': std(region_cloth, region_mean['cloth']),
        },
        'semantic': {
            'stability_mean': mean(semantic_stability),
            'stability_p10': quantile(semantic_stability, 0.10),
            'stability_p50': quantile(semantic_stability, 0.50),
            'stability_p90': quantile(semantic_stability, 0.90),
            'distance_mean': mean(semantic_distance),
        },
        'thin': {
            'score_mean': mean(thin_score),
            'score_max': max(thin_score) if thin_score else 0.0,
            'score_p10': quantile(thin_score, 0.10),
            'score_p50': quantile(thin_score, 0.50),
            'score_p90': quantile(thin_score, 0.90),
            'score_p95': quantile(thin_score, 0.95),
        },
        'temporal': {
            'slip_mean': mean(temporal_slip),
            'slip_max': max(temporal_slip) if temporal_slip else 0.0,
            'slip_p90': quantile(temporal_slip, 0.90),
            'nonzero_frames': temporal_nonzero,
            'recommended_split': 'test-video' if temporal_nonzero == 0 else None,
        },
    }


def balance_score(values: list[float]) -> float:
    total = sum(values)
    if total <= 1e-8:
        return 0.0
    probs = [max(v / total, 1e-8) for v in values]
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy / math.log(len(probs))


def map_image_path(split_dir: Path, map_name: str, image_name: str) -> Path:
    root = split_dir / 'renders' if map_name == 'render' else split_dir / 'binding_maps' / map_name
    return root / f'render_{image_name}.png'


def map_saliency(split_dir: Path, map_name: str, image_name: str) -> float:
    path = map_image_path(split_dir, map_name, image_name)
    if not path.exists():
        return 0.0
    image = imageio.imread(path)
    if image.ndim == 2:
        gray = image.astype(np.float32) / 255.0
    else:
        gray = image[..., :3].astype(np.float32).mean(axis=2) / 255.0
    mask = gray > 1e-3
    if mask.sum() == 0:
        return 0.0
    values = gray[mask]
    return float(values.std())


def select_keyframes(records: list[dict[str, Any]], split_dir: Path) -> dict[str, Any]:
    def best_by(fn, available=None):
        candidates = records
        if available is not None:
            candidates = [record for record in records if available(record)]
        if not candidates:
            return {
                'image_name': None,
                'score': 0.0,
                'available': False,
            }
        record = max(candidates, key=fn)
        return {
            'image_name': record['image_name'],
            'score': float(fn(record)),
            'available': True,
        }

    def has_map(map_name: str):
        return lambda record: map_image_path(split_dir, map_name, record['image_name']).exists()

    return {
        'layer_balanced': best_by(lambda r: balance_score([
            metric(r, 'layer', 'rigid_ratio'), metric(r, 'layer', 'soft_ratio'), metric(r, 'layer', 'free_ratio')
        ])),
        'region_balanced': best_by(lambda r: balance_score([
            metric(r, 'region', 'body_prob_mean', metric(r, 'region', 'body_ratio')),
            metric(r, 'region', 'soft_prob_mean', metric(r, 'region', 'soft_ratio')),
            metric(r, 'region', 'cloth_prob_mean', metric(r, 'region', 'cloth_ratio')),
        ])),
        'layer_showcase': best_by(lambda r: map_saliency(split_dir, 'layer', r['image_name']), has_map('layer')),
        'region_showcase': best_by(lambda r: map_saliency(split_dir, 'region', r['image_name']), has_map('region')),
        'body_prob_showcase': best_by(lambda r: map_saliency(split_dir, 'body_prob', r['image_name']), has_map('body_prob')),
        'soft_prob_showcase': best_by(lambda r: map_saliency(split_dir, 'soft_prob', r['image_name']), has_map('soft_prob')),
        'cloth_prob_showcase': best_by(lambda r: map_saliency(split_dir, 'cloth_prob', r['image_name']), has_map('cloth_prob')),
        'thin_detail': best_by(lambda r: map_saliency(split_dir, 'thin', r['image_name']), has_map('thin')),
        'semantic_hard': best_by(lambda r: metric(r, 'semantic', 'semantic_distance_mean')),
        'semantic_showcase': best_by(lambda r: map_saliency(split_dir, 'semantic', r['image_name']), has_map('semantic')),
        'temporal_slip': best_by(lambda r: metric(r, 'temporal', 'temporal_slip_mean')),
        'temporal_showcase': best_by(lambda r: map_saliency(split_dir, 'temporal', r['image_name']), has_map('temporal')),
    }


def copy_selected_assets(keyframes: dict[str, Any], split_dir: Path, output_dir: Path) -> None:
    asset_roots = {
        'render': split_dir / 'renders',
        'layer': split_dir / 'binding_maps' / 'layer',
        'region': split_dir / 'binding_maps' / 'region',
        'body_prob': split_dir / 'binding_maps' / 'body_prob',
        'soft_prob': split_dir / 'binding_maps' / 'soft_prob',
        'cloth_prob': split_dir / 'binding_maps' / 'cloth_prob',
        'semantic': split_dir / 'binding_maps' / 'semantic',
        'temporal': split_dir / 'binding_maps' / 'temporal',
        'thin': split_dir / 'binding_maps' / 'thin',
        'montage': split_dir / 'paper_montages',
    }
    assets_dir = output_dir / 'selected_assets'
    assets_dir.mkdir(parents=True, exist_ok=True)
    for label, payload in keyframes.items():
        image_name = payload.get('image_name')
        if not image_name:
            continue
        target_dir = assets_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)
        file_name = f'render_{image_name}.png'
        montage_name = f'montage_{image_name}.png'
        for asset_name, root in asset_roots.items():
            src = root / (montage_name if asset_name == 'montage' else file_name)
            if src.exists():
                shutil.copy2(src, target_dir / src.name)


def main() -> int:
    args = parse_args()
    split_dir = args.exp_dir / args.split
    summary_path = args.summary_path or (split_dir / 'binding_maps' / 'summary.json')
    output_dir = args.output_dir or (split_dir / 'binding_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_summary(summary_path)
    aggregate_stats = aggregate(records)
    keyframes = select_keyframes(records, split_dir)

    with open(output_dir / 'aggregate.json', 'w') as f:
        json.dump(aggregate_stats, f, indent=2)
    with open(output_dir / 'keyframes.json', 'w') as f:
        json.dump(keyframes, f, indent=2)

    if args.copy_assets:
        copy_selected_assets(keyframes, split_dir, output_dir)

    print(f'Wrote aggregate stats to {output_dir / "aggregate.json"}')
    print(f'Wrote keyframe suggestions to {output_dir / "keyframes.json"}')
    if aggregate_stats['temporal']['recommended_split']:
        print('Temporal slip is zero in this summary; rerun interpretability export with dataset.test_mode=video for temporal maps.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
