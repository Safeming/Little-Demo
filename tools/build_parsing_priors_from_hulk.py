#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from build_parsing_priors_from_cihp import (
    BODY_LABELS,
    CLOTH_LABELS,
    FACE_LABEL,
    HAIR_LABELS,
    SHOE_LABELS,
    VALID_IMAGE_SUFFIXES,
    adaptive_torso_zones,
    build_priors_for_frame,
    dilate,
    ensure_dir,
    keep_components,
    robust_close,
    robust_erode,
    robust_open,
    sample_pixels,
    skin_score_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert Hulk parsing outputs into Hulk-dominant body/cloth/valid/uncertain priors with old-rule fallback.')
    parser.add_argument('--mask-dir', type=Path, required=True, help='Directory of Hulk parsing PNGs.')
    parser.add_argument('--image-dir', type=Path, required=True, help='Directory of source RGB images for the same camera.')
    parser.add_argument('--subject', required=True, help='e.g. CoreView_377')
    parser.add_argument('--camera', required=True, help='e.g. 1')
    parser.add_argument('--target-root', type=Path, default=Path('/remote-home/ming/3dgs-avatar-release-main/data/parsing_priors_from_hulk_fallback'))
    parser.add_argument('--old-prior-root', type=Path, default=Path('/remote-home/ming/3dgs-avatar-release-main/data/parsing_priors_from_cihp'))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--sample-step', type=int, default=12)
    parser.add_argument('--max-samples-per-frame', type=int, default=2500)
    return parser.parse_args()


def find_image_path(image_dir: Path, stem: str) -> Path | None:
    for suffix in VALID_IMAGE_SUFFIXES:
        path = image_dir / f'{stem}{suffix}'
        if path.exists():
            return path
    return None


def read_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def read_binary(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(path)
    return (arr > 127).astype(np.uint8)


def compute_skin_statistics(mask_dir: Path, image_dir: Path, sample_step: int, max_samples_per_frame: int) -> dict[str, list[float]]:
    samples = []
    mask_paths = sorted(mask_dir.glob('*.png'))[::max(sample_step, 1)]
    for mask_path in mask_paths:
        image_path = find_image_path(image_dir, mask_path.stem)
        if image_path is None:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = read_mask(mask_path)
        if image is None:
            continue

        face_mask = (mask == FACE_LABEL).astype(np.uint8)
        skin_mask = np.isin(mask, BODY_LABELS).astype(np.uint8)
        if face_mask.sum() >= 48:
            sample_mask = robust_erode(face_mask, 3)
            if sample_mask.sum() < 24:
                sample_mask = face_mask
        else:
            sample_mask = robust_erode(skin_mask, 3)
            if sample_mask.sum() < 24:
                sample_mask = skin_mask
        feats = sample_pixels(image, sample_mask, max_samples_per_frame)
        if feats.size:
            samples.append(feats)

    if not samples:
        raise RuntimeError(f'Could not estimate skin statistics from {mask_dir}')

    data = np.concatenate(samples, axis=0)
    median = np.median(data, axis=0)
    mad = np.median(np.abs(data - median[None, :]), axis=0)
    scale = np.maximum(1.4826 * mad, np.array([8.0, 5.0, 5.0, 6.0, 6.0], dtype=np.float32))
    q05 = np.quantile(data[:, 0], 0.05)
    q95 = np.quantile(data[:, 0], 0.95)
    return {
        'median': median.astype(float).tolist(),
        'scale': scale.astype(float).tolist(),
        'y_range': [float(q05), float(q95)],
        'sample_count': int(data.shape[0]),
    }


def build_hulk_dominant_priors(mask: np.ndarray, image_bgr: np.ndarray, stats: dict[str, list[float]], old_body: np.ndarray, old_cloth: np.ndarray, old_uncertain: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fg = (mask > 0).astype(np.uint8)
    hair = np.isin(mask, HAIR_LABELS).astype(np.uint8)
    shoe = np.isin(mask, SHOE_LABELS).astype(np.uint8)
    raw_body = np.isin(mask, BODY_LABELS).astype(np.uint8)
    raw_cloth = np.isin(mask, CLOTH_LABELS).astype(np.uint8)

    zones = adaptive_torso_zones(mask, raw_body, raw_cloth)
    chest_core = zones['chest_core']
    chest_band = zones['chest_band']
    neck_core = zones['neck_core']
    upper_torso = zones['upper_torso']
    outer_torso = zones['outer_torso']
    strap_zone = zones['strap_zone']
    boundary_zone = zones['boundary_zone']
    torso_focus = zones['torso_focus']

    strong_skin, soft_skin, relaxed_skin = skin_score_mask(image_bgr, stats)
    strong_skin *= fg
    soft_skin *= fg
    relaxed_skin *= fg

    # Hulk raw segmentation is the default; old priors only patch the ambiguous torso/strap areas.
    body = (raw_body * (1 - hair) * (1 - shoe)).astype(np.uint8)
    cloth = (raw_cloth * (1 - hair) * (1 - shoe)).astype(np.uint8)

    strap_anchor = (raw_cloth * strap_zone * (1 - neck_core)).astype(np.uint8)
    strap_anchor = keep_components(robust_open(robust_close(strap_anchor, 3), 3), anchor=raw_cloth * strap_zone, min_area=10)

    chest_skin = (soft_skin * np.clip(chest_core + neck_core + 0.35 * chest_band, 0, 1) * (1 - strap_anchor) * (1 - hair) * (1 - shoe)).astype(np.uint8)
    torso_skin = (strong_skin * np.clip(chest_band + neck_core + 0.5 * upper_torso, 0, 1) * (1 - strap_anchor) * (1 - hair) * (1 - shoe)).astype(np.uint8)
    skin_patch = np.clip(chest_skin + torso_skin, 0, 1).astype(np.uint8)

    # Old body only falls back inside the torso/chest if Hulk missed exposed skin.
    old_body_torso = (old_body * np.clip(chest_band + neck_core + 0.35 * torso_focus, 0, 1)).astype(np.uint8)
    body = np.clip(body + skin_patch + old_body_torso, 0, 1).astype(np.uint8)

    # Old cloth only falls back on straps / outer torso where Hulk may under-segment thin cloth.
    old_cloth_strap = (old_cloth * np.clip(strap_zone + outer_torso, 0, 1) * (1 - body)).astype(np.uint8)
    cloth = np.clip(cloth + strap_anchor + old_cloth_strap, 0, 1).astype(np.uint8)

    # Remove direct conflicts, then keep stable connected regions.
    body = (body * (1 - cloth)).astype(np.uint8)
    cloth = (cloth * (1 - body)).astype(np.uint8)
    body = keep_components(body, anchor=np.clip(raw_body + neck_core + old_body_torso, 0, 1).astype(np.uint8), min_area=24)
    cloth = keep_components(cloth, anchor=np.clip(raw_cloth + strap_anchor + old_cloth_strap, 0, 1).astype(np.uint8), min_area=24)

    overlap_zone = (dilate(body, 3) * dilate(cloth, 3)).astype(np.uint8)
    color_conflict = ((soft_skin + relaxed_skin) > 0).astype(np.uint8) * raw_cloth * chest_band
    uncertain = np.clip(overlap_zone + color_conflict + (old_uncertain * boundary_zone), 0, 1).astype(np.uint8)
    uncertain = robust_open(uncertain, 3)

    valid = (fg * (1 - hair) * (1 - shoe) * np.clip(body + cloth, 0, 1)).astype(np.uint8)
    valid = (valid * (1 - uncertain)).astype(np.uint8)
    body = (body * valid).astype(np.uint8)
    cloth = (cloth * valid).astype(np.uint8)
    return body, cloth, valid, uncertain


def main() -> int:
    args = parse_args()
    if not args.mask_dir.exists():
        raise FileNotFoundError(f'Missing Hulk mask dir: {args.mask_dir}')
    if not args.image_dir.exists():
        raise FileNotFoundError(f'Missing image dir: {args.image_dir}')

    target_subject = args.target_root / args.subject / args.camera
    old_subject = args.old_prior_root / args.subject / args.camera
    body_out = target_subject / 'body'
    cloth_out = target_subject / 'cloth'
    valid_out = target_subject / 'valid'
    uncertain_out = target_subject / 'uncertain'
    for out_dir in (body_out, cloth_out, valid_out, uncertain_out):
        ensure_dir(out_dir)

    stats = compute_skin_statistics(args.mask_dir, args.image_dir, args.sample_step, args.max_samples_per_frame)
    ensure_dir(target_subject)
    (target_subject / 'skin_stats.json').write_text(json.dumps(stats, indent=2))

    total = 0
    for mask_path in sorted(args.mask_dir.glob('*.png')):
        body_path = body_out / mask_path.name
        cloth_path = cloth_out / mask_path.name
        valid_path = valid_out / mask_path.name
        uncertain_path = uncertain_out / mask_path.name
        if not args.overwrite and body_path.exists() and cloth_path.exists() and valid_path.exists() and uncertain_path.exists():
            continue

        image_path = find_image_path(args.image_dir, mask_path.stem)
        if image_path is None:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = read_mask(mask_path)
        if image is None:
            continue

        old_body = read_binary(old_subject / 'body' / mask_path.name)
        old_cloth = read_binary(old_subject / 'cloth' / mask_path.name)
        old_uncertain = read_binary(old_subject / 'uncertain' / mask_path.name)
        body, cloth, valid, uncertain = build_hulk_dominant_priors(mask, image, stats, old_body, old_cloth, old_uncertain)
        cv2.imwrite(str(body_path), body * 255)
        cv2.imwrite(str(cloth_path), cloth * 255)
        cv2.imwrite(str(valid_path), valid * 255)
        cv2.imwrite(str(uncertain_path), uncertain * 255)
        total += 1

    print(f'Built Hulk fallback priors for {args.subject} cam {args.camera}: {total} masks written to {target_subject}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
