import argparse
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

RAW_CIHP_LABEL_NAMES = {
    1: "hat",
    2: "hair",
    3: "glove",
    4: "sunglasses",
    5: "upper_clothes",
    6: "dress",
    7: "coat",
    8: "socks",
    9: "pants",
    10: "jumpsuits",
    11: "scarf",
    12: "skirt",
    13: "face",
    14: "left_arm",
    15: "right_arm",
    16: "left_leg",
    17: "right_leg",
    18: "left_shoe",
    19: "right_shoe",
}

COMPACT_GROUP_TEMPLATES = OrderedDict([
    ("hair", [2]),
    ("face", [13]),
    ("skin", [14, 15, 16, 17]),
    ("upper", [5, 7, 11]),
    ("onepiece", [6, 10]),
    ("lower", [9, 12]),
    ("shoes", [8, 18, 19]),
    ("accessory", [1, 3, 4]),
])


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Hulk pseudo labels and derive a compact semantic mapping.")
    parser.add_argument('--parser-root', type=Path, required=True)
    parser.add_argument('--subject', type=str, default='CoreView_377')
    parser.add_argument('--parser-layout', type=str, default='flat_png', choices=['flat_png', 'cihp_subject'])
    parser.add_argument('--camera', type=str, default='1')
    parser.add_argument('--frame-start', type=int, default=0)
    parser.add_argument('--frame-end', type=int, default=0)
    parser.add_argument('--frame-step', type=int, default=1)
    parser.add_argument('--min-frame-ratio', type=float, default=0.01)
    parser.add_argument('--min-pixel-ratio', type=float, default=0.0005)
    parser.add_argument('--min-present-pixels', type=float, default=48.0)
    parser.add_argument('--skip-empty-min-pixels', type=int, default=64)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def resolve_mask_paths(args):
    if args.parser_layout == 'flat_png':
        paths = sorted(args.parser_root.glob('*.png'))
    else:
        cam_token = f'Camera_B{int(args.camera)}'
        paths = sorted((args.parser_root / args.subject / 'mask_cihp' / cam_token).glob('*.png'))
    if args.frame_end > 0:
        paths = paths[args.frame_start:args.frame_end:args.frame_step]
    else:
        paths = paths[args.frame_start::args.frame_step]
    return paths


def compute_stats(paths):
    label_pixels = {label: 0 for label in RAW_CIHP_LABEL_NAMES}
    frames_present = {label: 0 for label in RAW_CIHP_LABEL_NAMES}
    total_fg_pixels = 0
    frame_names = []
    frame_label_pixels = []
    for path in paths:
        arr = np.array(Image.open(path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        fg = arr > 0
        total_fg_pixels += int(fg.sum())
        frame_names.append(path.stem)
        per_frame = {}
        unique, counts = np.unique(arr, return_counts=True)
        count_map = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
        for label in RAW_CIHP_LABEL_NAMES:
            px = count_map.get(label, 0)
            per_frame[label] = px
            label_pixels[label] += px
            if px > 0:
                frames_present[label] += 1
        frame_label_pixels.append(per_frame)
    total_frames = len(paths)
    stats = {}
    for label, name in RAW_CIHP_LABEL_NAMES.items():
        present = frames_present[label]
        pixels = label_pixels[label]
        mean_present = pixels / max(present, 1)
        stats[label] = {
            'name': name,
            'frames_present': int(present),
            'frame_ratio': float(present / max(total_frames, 1)),
            'pixels': int(pixels),
            'pixel_ratio': float(pixels / max(total_fg_pixels, 1)),
            'mean_pixels_when_present': float(mean_present),
        }
    return stats, frame_names, frame_label_pixels, total_frames, total_fg_pixels


def _label_is_active(label, group_name, stats, args):
    item = stats[label]
    base_active = (
        item['pixel_ratio'] >= args.min_pixel_ratio
        or (item['frame_ratio'] >= args.min_frame_ratio and item['mean_pixels_when_present'] >= args.min_present_pixels)
    )
    if not base_active:
        return False
    if group_name == 'onepiece':
        upper = stats[5]
        lower = stats[9]
        if item['frame_ratio'] > 0.85 and upper['frame_ratio'] > 0.85 and lower['frame_ratio'] > 0.85:
            if item['pixel_ratio'] < 0.15 * (upper['pixel_ratio'] + lower['pixel_ratio']):
                return False
    return True


def recommend_mapping(stats, args):
    active_labels = set()
    label_actions = {}
    groups = OrderedDict()
    ignored_labels = []
    for group_name, labels in COMPACT_GROUP_TEMPLATES.items():
        active = []
        for label in labels:
            if _label_is_active(label, group_name, stats, args):
                active.append(label)
                active_labels.add(label)
        if active:
            groups[group_name] = active
        for label in labels:
            if label in active:
                action = 'keep' if group_name in {'hair', 'face'} else 'merge'
                target = group_name
            else:
                action = 'ignore'
                target = None
                ignored_labels.append(label)
            label_actions[label] = {
                'name': RAW_CIHP_LABEL_NAMES[label],
                'action': action,
                'target': target,
            }
    ignored_labels = sorted(set(ignored_labels))
    return groups, label_actions, sorted(active_labels), ignored_labels


def find_empty_frames(frame_names, frame_label_pixels, active_labels, min_pixels):
    empty = []
    active_labels = tuple(active_labels)
    for name, counts in zip(frame_names, frame_label_pixels):
        total = sum(int(counts.get(label, 0)) for label in active_labels)
        if total < min_pixels:
            empty.append({'frame': name, 'active_pixels': int(total)})
    return empty


def write_outputs(args, stats, groups, label_actions, active_labels, ignored_labels, empty_frames, total_frames, total_fg_pixels):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compact = {
        'subject': args.subject,
        'parser_root': str(args.parser_root),
        'parser_layout': args.parser_layout,
        'camera': str(args.camera),
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'class_names': list(groups.keys()),
        'groups': {name: list(labels) for name, labels in groups.items()},
        'active_labels': list(active_labels),
        'ignore_labels': list(ignored_labels),
        'label_names': {str(k): v for k, v in RAW_CIHP_LABEL_NAMES.items()},
        'skip_empty_min_pixels': int(args.skip_empty_min_pixels),
        'empty_frames': empty_frames,
    }
    report = {
        'subject': args.subject,
        'total_frames': int(total_frames),
        'total_fg_pixels': int(total_fg_pixels),
        'label_stats': {str(k): v for k, v in stats.items()},
        'label_actions': {str(k): v for k, v in label_actions.items()},
        'recommended_compact_mapping': compact,
    }
    json_path = args.output_dir / f'{args.subject}_hulk_compact_mapping.json'
    report_path = args.output_dir / f'{args.subject}_hulk_label_stats.json'
    md_path = args.output_dir / f'{args.subject}_hulk_label_stats.md'
    json_path.write_text(json.dumps(compact, indent=2))
    report_path.write_text(json.dumps(report, indent=2))

    lines = []
    lines.append(f'# Hulk Label Stats: {args.subject}')
    lines.append('')
    lines.append(f'- total_frames: {total_frames}')
    lines.append(f'- total_fg_pixels: {total_fg_pixels}')
    lines.append(f'- active_compact_classes: {", ".join(groups.keys()) if groups else "(none)"}')
    lines.append(f'- ignore_labels: {", ".join(str(x) for x in ignored_labels) if ignored_labels else "(none)"}')
    lines.append(f'- empty_frames_below_{args.skip_empty_min_pixels}px: {len(empty_frames)}')
    lines.append('')
    lines.append('## Recommended Compact Mapping')
    lines.append('')
    for name, labels in groups.items():
        label_desc = ', '.join(f'{label}:{RAW_CIHP_LABEL_NAMES[label]}' for label in labels)
        lines.append(f'- {name}: {label_desc}')
    lines.append('')
    lines.append('## Label Actions')
    lines.append('')
    for label in sorted(RAW_CIHP_LABEL_NAMES):
        action = label_actions.get(label, {'action': 'ignore', 'target': None})
        suffix = f" -> {action['target']}" if action['target'] else ''
        s = stats[label]
        lines.append(f"- {label}:{RAW_CIHP_LABEL_NAMES[label]}: {action['action']}{suffix}; frame_ratio={s['frame_ratio']:.4f}, pixel_ratio={s['pixel_ratio']:.4f}, mean_pixels_when_present={s['mean_pixels_when_present']:.1f}")
    lines.append('')
    if empty_frames:
        lines.append('## Empty Frames')
        lines.append('')
        for item in empty_frames[:100]:
            lines.append(f"- {item['frame']}: active_pixels={item['active_pixels']}")
    md_path.write_text('\n'.join(lines) + '\n')
    return json_path, report_path, md_path


def main():
    args = parse_args()
    paths = resolve_mask_paths(args)
    if not paths:
        raise SystemExit('No parser masks found for the provided arguments.')
    stats, frame_names, frame_label_pixels, total_frames, total_fg_pixels = compute_stats(paths)
    groups, label_actions, active_labels, ignored_labels = recommend_mapping(stats, args)
    empty_frames = find_empty_frames(frame_names, frame_label_pixels, active_labels, args.skip_empty_min_pixels)
    json_path, report_path, md_path = write_outputs(args, stats, groups, label_actions, active_labels, ignored_labels, empty_frames, total_frames, total_fg_pixels)
    print(f'compact_mapping: {json_path}')
    print(f'stats_json: {report_path}')
    print(f'stats_md: {md_path}')
    print(f'active_classes: {list(groups.keys())}')
    print(f'empty_frames: {len(empty_frames)}')


if __name__ == '__main__':
    main()
