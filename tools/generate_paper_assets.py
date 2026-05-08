#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import imageio.v2 as imageio
import numpy as np

from experiment_utils import collect_experiment_record, flatten_record, write_csv

GT_ROOT = Path('data/ZJUMoCap/CoreView_377')
OUTPUT_ROOT = Path('exp/paper_assets/v41_submission_pack')

TABLE1_ROWS = [
    ('baseline', Path('exp/zju_377_mono-direct-mlp_field-ingp-shallow_mlp-baseline_15k_main')),
    ('v3', Path('exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-expbind_v3_15k-0311-0717')),
    ('v4.1', Path('exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_15k-0311-1123-main')),
]
TABLE2_ROWS = [
    ('v4.1', Path('exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_15k-0311-1123-main')),
    ('w/o body-cloth', Path('exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-ablate_nobodycloth_15k-0312-1125')),
    ('w/o temporal', Path('exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-ablate_notemporal_15k-0312-1205')),
    ('w/o semantic', Path('exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-ablate_nosemantic_15k-0312-1251')),
]

INTERP_ROOTS = {
    'v4.1_view': Path('exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_15k-0311-1123-main_interp_full/test-view'),
    'v4.1_video': Path('exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_15k-0311-1123-main_interp_video/test-video'),
    'nobodycloth_view': Path('exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-ablate_nobodycloth_15k-0312-1125_interp_full/test-view'),
    'notemporal_view': Path('exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-ablate_notemporal_15k-0312-1205_interp_full/test-view'),
    'notemporal_video': Path('exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-ablate_notemporal_15k-0312-1205_interp_video/test-video'),
    'nosemantic_view': Path('exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-ablate_nosemantic_15k-0312-1251_interp_full/test-view'),
}

FIELDNAMES_TABLE1 = ['label', 'psnr', 'ssim', 'lpips']
FIELDNAMES_TABLE2 = ['label', 'layer_rigid', 'layer_soft', 'layer_free', 'region_body', 'region_soft', 'region_cloth', 'semantic_stability', 'thin_score', 'temporal_slip']


def load_npz_metrics(exp_dir: Path, split: str = 'test-view') -> dict[str, Any]:
    path = exp_dir / split / 'results.npz'
    if not path.exists():
        return {'psnr': None, 'ssim': None, 'lpips': None}
    payload = np.load(path)
    return {
        'psnr': float(payload['psnr']) if 'psnr' in payload else None,
        'ssim': float(payload['ssim']) if 'ssim' in payload else None,
        'lpips': float(payload['lpips']) if 'lpips' in payload else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate paper tables and figures for the v4.1 binding project.')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--skip-tables', action='store_true')
    parser.add_argument('--skip-figures', action='store_true')
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    image = imageio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] > 3:
        image = image[..., :3]
    return image.astype(np.uint8)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, image)


def resize_to(image: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if image.shape[:2] == hw:
        return image
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def crop_image(image: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return image
    x1, y1, x2, y2 = crop
    x1 = max(0, min(x1, image.shape[1] - 1))
    y1 = max(0, min(y1, image.shape[0] - 1))
    x2 = max(x1 + 1, min(x2, image.shape[1]))
    y2 = max(y1 + 1, min(y2, image.shape[0]))
    return image[y1:y2, x1:x2]


def draw_label(canvas: np.ndarray, text: str, left: int, top: int, width: int, header_h: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x = left + max((width - tw) // 2, 8)
    y = top + max((header_h + th) // 2 - base, th + 4)
    cv2.putText(canvas, text, (x, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)


def build_montage(panels: list[tuple[str, np.ndarray]], header_h: int = 40, gap: int = 10) -> np.ndarray:
    panel_h, panel_w = panels[0][1].shape[:2]
    canvas = np.full((panel_h + header_h, len(panels) * panel_w + (len(panels) - 1) * gap, 3), 255, dtype=np.uint8)
    left = 0
    for title, image in panels:
        canvas[header_h:, left:left + panel_w] = image
        draw_label(canvas, title, left, 0, panel_w, header_h)
        left += panel_w + gap
    cv2.line(canvas, (0, header_h - 1), (canvas.shape[1], header_h - 1), (220, 220, 220), 1)
    return canvas


def find_gt(image_name: str) -> Path:
    cam, frame = image_name.split('_')
    cam = str(int(cam[1:]))
    frame_num = int(frame[1:])
    for suffix in ('jpg', 'png'):
        for pattern in (f'{frame_num}.{suffix}', f'{frame_num:06d}.{suffix}'):
            path = GT_ROOT / cam / pattern
            if path.exists():
                return path
    raise FileNotFoundError(f'GT not found for {image_name}')


def get_image(split_root: Path, subdir: str, image_name: str) -> np.ndarray:
    path = split_root / subdir / f'render_{image_name}.png'
    return load_image(path)


def load_keyframes(path: Path) -> dict[str, Any]:
    with open(path, 'r') as handle:
        return json.load(handle)


def choose_image_name(split_root: Path, preferred_keys: Iterable[str]) -> str:
    keyframes = load_keyframes(split_root / 'binding_analysis' / 'keyframes.json')
    for key in preferred_keys:
        payload = keyframes.get(key)
        if payload and payload.get('available') and payload.get('image_name'):
            return payload['image_name']
    raise RuntimeError(f'No usable keyframe found in {split_root}')


def make_tables(output_dir: Path) -> None:
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    records1 = []
    for label, exp_dir in TABLE1_ROWS:
        metrics = load_npz_metrics(exp_dir, 'test-view')
        row = {'label': label, **metrics}
        records1.append({field: row.get(field) for field in FIELDNAMES_TABLE1})
    write_csv(tables_dir / 'table1_main_render_metrics.csv', records1, FIELDNAMES_TABLE1)

    records2 = []
    for label, exp_dir in TABLE2_ROWS:
        record = collect_experiment_record(exp_dir, 'test-view')
        record['label'] = label
        row = flatten_record(record)
        video_record = collect_experiment_record(exp_dir, 'test-video')
        video_row = flatten_record(video_record)
        row['temporal_slip'] = video_row.get('temporal_slip')
        records2.append({field: row.get(field) for field in FIELDNAMES_TABLE2})
    write_csv(tables_dir / 'table2_ablation_interpretability.csv', records2, FIELDNAMES_TABLE2)

    def latex(rows: list[dict[str, Any]], fields: list[str]) -> str:
        lines = ['\\begin{tabular}{' + 'l' * len(fields) + '}', '\\toprule', ' & '.join(fields) + ' \\\\', '\\midrule']
        for row in rows:
            vals = []
            for field in fields:
                value = row.get(field, '')
                if isinstance(value, float):
                    vals.append(f'{value:.4f}')
                else:
                    vals.append(str(value))
            lines.append(' & '.join(vals) + ' \\\\')
        lines.extend(['\\bottomrule', '\\end{tabular}'])
        return '\n'.join(lines) + '\n'

    (tables_dir / 'table1_main_render_metrics.tex').write_text(latex(records1, FIELDNAMES_TABLE1))
    (tables_dir / 'table2_ablation_interpretability.tex').write_text(latex(records2, FIELDNAMES_TABLE2))


def make_figure1(output_dir: Path) -> None:
    split_root = INTERP_ROOTS['v4.1_view']
    image_name = choose_image_name(split_root, ['region_showcase', 'layer_showcase', 'semantic_showcase'])
    gt = resize_to(load_image(find_gt(image_name)), (512, 512))
    panels = [
        ('GT', gt),
        ('Render', get_image(split_root, 'renders', image_name)),
        ('Layer', get_image(split_root, 'binding_maps/layer', image_name)),
        ('Region', get_image(split_root, 'binding_maps/region', image_name)),
        ('Body Prob', get_image(split_root, 'binding_maps/body_prob', image_name)),
        ('Semantic', get_image(split_root, 'binding_maps/semantic', image_name)),
    ]
    save_image(output_dir / 'figures' / 'figure1_method_overview.png', build_montage(panels))


def make_figure2(output_dir: Path) -> None:
    main_root = INTERP_ROOTS['v4.1_view']
    ablate_root = INTERP_ROOTS['nobodycloth_view']
    image_name = choose_image_name(main_root, ['region_showcase', 'body_prob_showcase', 'cloth_prob_showcase'])
    gt = resize_to(load_image(find_gt(image_name)), (512, 512))
    panels = [
        ('GT', gt),
        ('V4.1 Render', get_image(main_root, 'renders', image_name)),
        ('V4.1 Region', get_image(main_root, 'binding_maps/region', image_name)),
        ('w/o BC Render', get_image(ablate_root, 'renders', image_name)),
        ('w/o BC Region', get_image(ablate_root, 'binding_maps/region', image_name)),
        ('w/o BC Body Prob', get_image(ablate_root, 'binding_maps/body_prob', image_name)),
    ]
    save_image(output_dir / 'figures' / 'figure2_bodycloth_ablation.png', build_montage(panels))


def make_figure3(output_dir: Path) -> None:
    main_root = INTERP_ROOTS['v4.1_video']
    ablate_root = INTERP_ROOTS['notemporal_video']
    image_name = choose_image_name(main_root, ['temporal_slip', 'temporal_showcase'])
    gt = resize_to(load_image(find_gt(image_name)), (512, 512))
    panels = [
        ('GT', gt),
        ('V4.1 Render', get_image(main_root, 'renders', image_name)),
        ('V4.1 Temporal', get_image(main_root, 'binding_maps/temporal', image_name)),
        ('w/o Temp Render', get_image(ablate_root, 'renders', image_name)),
        ('w/o Temp Temporal', get_image(ablate_root, 'binding_maps/temporal', image_name)),
    ]
    save_image(output_dir / 'figures' / 'figure3_temporal_ablation.png', build_montage(panels))


def make_figure4(output_dir: Path) -> None:
    image_name = 'c08_f000060'
    crop = (170, 180, 360, 410)
    gt = crop_image(resize_to(load_image(find_gt(image_name)), (512, 512)), crop)
    baseline = crop_image(get_image(TABLE1_ROWS[0][1] / 'test-view', 'renders', image_name), crop)
    v41 = crop_image(get_image(TABLE1_ROWS[2][1] / 'test-view', 'renders', image_name), crop)
    nobody = crop_image(get_image(TABLE2_ROWS[1][1] / 'test-view', 'renders', image_name), crop)
    panels = [
        ('GT Crop', resize_to(gt, (256, 256))),
        ('Baseline Crop', resize_to(baseline, (256, 256))),
        ('V4.1 Crop', resize_to(v41, (256, 256))),
        ('w/o BC Crop', resize_to(nobody, (256, 256))),
    ]
    save_image(output_dir / 'figures' / 'figure4_failure_case.png', build_montage(panels))


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_tables:
        make_tables(output_dir)
    if not args.skip_figures:
        (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
        make_figure1(output_dir)
        make_figure2(output_dir)
        make_figure3(output_dir)
        make_figure4(output_dir)
    print(f'Wrote paper assets to {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
