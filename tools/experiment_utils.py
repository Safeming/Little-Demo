#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re

import numpy as np
from pathlib import Path
from typing import Any

CORE_OVERRIDE_KEYS = ('dataset', 'rigid', 'non_rigid', 'pose_correction', 'texture')
CKPT_PATTERN = re.compile(r'^ckpt(?P<step>\d+)\.pth$')


def read_json(path: Path) -> dict[str, Any] | list[Any]:
    with open(path, 'r') as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, '') for name in fieldnames})


def parse_scalar(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.6f}'
    return str(value)


def nested_get(payload: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_hydra_overrides(exp_dir: Path) -> list[str]:
    path = exp_dir / '.hydra' / 'overrides.yaml'
    if not path.exists():
        return []
    overrides = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith('- '):
            overrides.append(line[2:])
    return overrides


def infer_core_overrides(exp_dir: Path) -> list[str]:
    selected: dict[str, str] = {}
    for override in load_hydra_overrides(exp_dir):
        if '=' not in override:
            continue
        key, value = override.split('=', 1)
        if key in CORE_OVERRIDE_KEYS or key == 'wandb_disable':
            selected[key] = value
    if 'wandb_disable' not in selected:
        selected['wandb_disable'] = 'true'
    ordered = [key for key in CORE_OVERRIDE_KEYS if key in selected]
    if 'wandb_disable' in selected:
        ordered.append('wandb_disable')
    return [f'{key}={selected[key]}' for key in ordered]


def find_checkpoint(exp_dir: Path, mode: str = 'latest') -> Path:
    if exp_dir.is_file() and exp_dir.suffix == '.pth':
        return exp_dir
    best_ckpt = exp_dir / 'best_ckpt.pth'
    numbered = []
    for path in exp_dir.glob('ckpt*.pth'):
        match = CKPT_PATTERN.match(path.name)
        if match:
            numbered.append((int(match.group('step')), path))
    numbered.sort(key=lambda item: item[0])
    if mode == 'best' and best_ckpt.exists():
        return best_ckpt
    if numbered:
        return numbered[-1][1]
    if best_ckpt.exists():
        return best_ckpt
    raise FileNotFoundError(f'No checkpoint found in {exp_dir}')


def infer_iteration_from_checkpoint(path: Path) -> int | None:
    match = CKPT_PATTERN.match(path.name)
    if match:
        return int(match.group('step'))
    return None


def find_analysis_root(exp_dir: Path, split: str) -> Path | None:
    direct = exp_dir / split / 'binding_analysis'
    if direct.exists():
        return direct
    sibling_candidates = sorted(exp_dir.parent.glob(f'{exp_dir.name}_interp*'))
    for candidate in sibling_candidates:
        analysis = candidate / split / 'binding_analysis'
        if analysis.exists():
            return analysis
    return None


def find_render_root(exp_dir: Path, split: str) -> Path | None:
    direct = exp_dir / split / 'renders'
    if direct.exists():
        return direct
    sibling_candidates = sorted(exp_dir.parent.glob(f'{exp_dir.name}_interp*'))
    for candidate in sibling_candidates:
        renders = candidate / split / 'renders'
        if renders.exists():
            return renders
    return None


def load_best_metrics(exp_dir: Path) -> dict[str, Any] | None:
    path = exp_dir / 'best_test_metrics.json'
    if path.exists():
        return read_json(path)

    results_path = exp_dir / 'test-view' / 'results.npz'
    if results_path.exists():
        payload = np.load(results_path)
        metrics = {
            'iteration': infer_iteration_from_checkpoint(find_checkpoint(exp_dir, mode='latest')) if any(exp_dir.glob('ckpt*.pth')) else None,
            'psnr': float(payload['psnr']) if 'psnr' in payload else None,
            'ssim': float(payload['ssim']) if 'ssim' in payload else None,
            'lpips': float(payload['lpips']) if 'lpips' in payload else None,
            'l1': None,
            'checkpoint': find_checkpoint(exp_dir, mode='latest').name if any(exp_dir.glob('ckpt*.pth')) else None,
            'source': 'results.npz',
        }
        return metrics
    return None


def load_binding_aggregate(exp_dir: Path, split: str) -> tuple[dict[str, Any] | None, Path | None]:
    analysis_root = find_analysis_root(exp_dir, split)
    if analysis_root is None:
        return None, None
    aggregate_path = analysis_root / 'aggregate.json'
    if not aggregate_path.exists():
        return None, analysis_root
    return read_json(aggregate_path), analysis_root


def discover_label(exp_dir: Path) -> str:
    return exp_dir.name


def collect_experiment_record(exp_dir: Path, split: str) -> dict[str, Any]:
    exp_dir = exp_dir.resolve()
    metrics = load_best_metrics(exp_dir)
    aggregate, analysis_root = load_binding_aggregate(exp_dir, split)
    render_root = find_render_root(exp_dir, split)
    latest_ckpt = None
    best_ckpt = exp_dir / 'best_ckpt.pth'
    try:
        latest_ckpt = find_checkpoint(exp_dir, mode='latest')
    except FileNotFoundError:
        latest_ckpt = None
    return {
        'label': discover_label(exp_dir),
        'exp_dir': exp_dir.as_posix(),
        'split': split,
        'overrides': load_hydra_overrides(exp_dir),
        'metrics': metrics,
        'binding_analysis': aggregate,
        'paths': {
            'analysis_root': analysis_root.as_posix() if analysis_root else None,
            'render_root': render_root.as_posix() if render_root else None,
            'latest_ckpt': latest_ckpt.as_posix() if latest_ckpt else None,
            'best_ckpt': best_ckpt.as_posix() if best_ckpt.exists() else None,
        },
    }


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get('metrics') or {}
    analysis = record.get('binding_analysis') or {}
    row = {
        'label': record['label'],
        'exp_dir': record['exp_dir'],
        'split': record['split'],
        'best_iteration': nested_get(metrics, 'iteration'),
        'psnr': nested_get(metrics, 'psnr'),
        'ssim': nested_get(metrics, 'ssim'),
        'lpips': nested_get(metrics, 'lpips'),
        'l1': nested_get(metrics, 'l1'),
        'layer_rigid': nested_get(analysis, 'layer_mean', 'rigid'),
        'layer_soft': nested_get(analysis, 'layer_mean', 'soft'),
        'layer_free': nested_get(analysis, 'layer_mean', 'free'),
        'region_body': nested_get(analysis, 'region_mean', 'body'),
        'region_soft': nested_get(analysis, 'region_mean', 'soft'),
        'region_cloth': nested_get(analysis, 'region_mean', 'cloth'),
        'semantic_stability': nested_get(analysis, 'semantic', 'stability_mean'),
        'semantic_distance': nested_get(analysis, 'semantic', 'distance_mean'),
        'thin_score': nested_get(analysis, 'thin', 'score_mean'),
        'temporal_slip': nested_get(analysis, 'temporal', 'slip_mean'),
        'temporal_nonzero_frames': nested_get(analysis, 'temporal', 'nonzero_frames'),
        'analysis_root': nested_get(record, 'paths', 'analysis_root'),
        'render_root': nested_get(record, 'paths', 'render_root'),
        'latest_ckpt': nested_get(record, 'paths', 'latest_ckpt'),
        'best_ckpt': nested_get(record, 'paths', 'best_ckpt'),
    }
    return row


def rows_to_markdown(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    header = '| ' + ' | '.join(fieldnames) + ' |'
    separator = '| ' + ' | '.join(['---'] * len(fieldnames)) + ' |'
    body = []
    for row in rows:
        body.append('| ' + ' | '.join(parse_scalar(row.get(name, '')) for name in fieldnames) + ' |')
    return '\n'.join([header, separator] + body)
