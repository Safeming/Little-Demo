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
    parser = argparse.ArgumentParser(description='Build image-guided Hulk parsing priors with strap/chest refinement.')
    parser.add_argument('--mask-dir', type=Path, required=True)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--subject', required=True)
    parser.add_argument('--camera', required=True)
    parser.add_argument('--target-root', type=Path, default=Path('/remote-home/ming/3dgs-avatar-release-main/data/parsing_priors_from_hulk_refine'))
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


def _cloth_similarity(image_bgr: np.ndarray, cloth_seed: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ys, xs = np.where(cloth_seed > 0)
    if ys.size < 16:
        return np.zeros(lab.shape[:2], dtype=np.float32)
    feats = lab[ys, xs]
    center = np.median(feats, axis=0)
    mad = np.median(np.abs(feats - center[None, :]), axis=0)
    scale = np.maximum(1.4826 * mad, np.array([6.0, 4.0, 4.0], dtype=np.float32))
    z = np.abs((lab - center[None, None, :]) / scale[None, None, :])
    dist = np.sqrt((z ** 2).sum(axis=-1))
    sim = np.exp(-0.5 * np.square(dist / 2.0))
    return sim.astype(np.float32)


def _edge_response(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    q = max(float(np.quantile(mag, 0.92)), 1e-6)
    return np.clip(mag / q, 0.0, 1.0).astype(np.float32)


def build_refined_priors(mask: np.ndarray, image_bgr: np.ndarray, stats: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fg = (mask > 0).astype(np.uint8)
    hair = np.isin(mask, HAIR_LABELS).astype(np.uint8)
    shoe = np.isin(mask, SHOE_LABELS).astype(np.uint8)
    face = (mask == FACE_LABEL).astype(np.uint8)
    raw_body = np.isin(mask, BODY_LABELS).astype(np.uint8)
    raw_cloth = np.isin(mask, CLOTH_LABELS).astype(np.uint8)

    zones = adaptive_torso_zones(mask, raw_body, raw_cloth)
    torso_focus = zones['torso_focus']
    chest_core = zones['chest_core']
    chest_band = zones['chest_band']
    neck_core = zones['neck_core']
    upper_torso = zones['upper_torso']
    outer_torso = zones['outer_torso']
    strap_zone = zones['strap_zone']
    boundary_zone = zones['boundary_zone']

    strong_skin, soft_skin, relaxed_skin = skin_score_mask(image_bgr, stats)
    strong_skin *= fg
    soft_skin *= fg
    relaxed_skin *= fg

    cloth_seed = (raw_cloth * np.clip(strap_zone + outer_torso + upper_torso, 0, 1)).astype(np.uint8)
    if cloth_seed.sum() < 24:
        cloth_seed = raw_cloth
    cloth_sim = _cloth_similarity(image_bgr, cloth_seed)
    edge = _edge_response(image_bgr)

    body = (raw_body * (1 - hair) * (1 - shoe)).astype(np.uint8)
    cloth = (raw_cloth * (1 - hair) * (1 - shoe)).astype(np.uint8)

    # Recover thin straps / dark cloth details both inside and just outside the coarse cloth torso region.
    cloth_core = robust_erode(raw_cloth.astype(np.uint8), 5)
    cloth_ring = (raw_cloth * (1 - cloth_core)).astype(np.uint8)
    skin_neighbor = dilate(((raw_body + face) > 0).astype(np.uint8), 17)
    inner_strap = (raw_cloth * cloth_ring * np.clip(strap_zone + 0.65 * outer_torso, 0, 1) * skin_neighbor * (cloth_sim > 0.40).astype(np.uint8) * (edge > 0.10).astype(np.uint8)).astype(np.uint8)
    outer_search = (dilate(raw_cloth, 9) * np.clip(strap_zone + outer_torso, 0, 1) * (1 - cloth) * (1 - strong_skin) * (1 - face) * (1 - hair) * (1 - shoe)).astype(np.uint8)
    outer_strap = (outer_search * (cloth_sim > 0.52).astype(np.uint8) * (edge > 0.20).astype(np.uint8)).astype(np.uint8)
    thin_candidate = np.clip(inner_strap + outer_strap, 0, 1).astype(np.uint8)
    thin_candidate = keep_components(robust_open(robust_close(thin_candidate, 3), 3), anchor=np.clip((raw_cloth * strap_zone) + inner_strap, 0, 1).astype(np.uint8), min_area=6)
    thin_candidate = (thin_candidate * np.clip(strap_zone + outer_torso, 0, 1)).astype(np.uint8)

    # Recover exposed upper-chest skin where the parser swallowed skin into upper-clothes.
    skin_search = (np.clip(chest_core + neck_core + 0.40 * chest_band + 0.25 * torso_focus, 0, 1) * (1 - hair) * (1 - shoe)).astype(np.uint8)
    near_skin = dilate(((raw_body + face) > 0).astype(np.uint8), 11)
    skin_candidate = (skin_search * near_skin * (soft_skin > 0).astype(np.uint8) * (cloth_sim < 0.42).astype(np.uint8)).astype(np.uint8)
    skin_candidate = keep_components(robust_open(robust_close(skin_candidate, 3), 3), anchor=np.clip(raw_body + neck_core + face, 0, 1).astype(np.uint8), min_area=10)

    body = np.clip(body + skin_candidate, 0, 1).astype(np.uint8)
    cloth = np.clip(cloth + thin_candidate, 0, 1).astype(np.uint8)

    # Resolve competition conservatively.
    conflict = (dilate(body, 3) * dilate(cloth, 3)).astype(np.uint8)
    ambiguous = ((relaxed_skin > 0).astype(np.uint8) * (cloth_sim > 0.34).astype(np.uint8) * (cloth_sim < 0.62).astype(np.uint8) * np.clip(chest_band + boundary_zone, 0, 1).astype(np.uint8))
    uncertain = np.clip(conflict + ambiguous, 0, 1).astype(np.uint8)
    uncertain = robust_open(uncertain, 3)

    valid = (fg * (1 - hair) * (1 - shoe) * np.clip(body + cloth, 0, 1)).astype(np.uint8)
    body = (body * valid * (1 - uncertain)).astype(np.uint8)
    cloth = (cloth * valid * (1 - uncertain)).astype(np.uint8)
    thin_candidate = (thin_candidate * cloth).astype(np.uint8)
    return body, cloth, valid, uncertain, thin_candidate


def main() -> int:
    args = parse_args()
    target = args.target_root / args.subject / args.camera
    body_dir = target / 'body'
    cloth_dir = target / 'cloth'
    valid_dir = target / 'valid'
    uncertain_dir = target / 'uncertain'
    thin_dir = target / 'thin_cloth'
    for d in (body_dir, cloth_dir, valid_dir, uncertain_dir, thin_dir):
        ensure_dir(d)

    stats = compute_skin_statistics(args.mask_dir, args.image_dir, args.sample_step, args.max_samples_per_frame)
    (target / 'skin_stats.json').write_text(json.dumps(stats, indent=2))

    total = 0
    for mask_path in sorted(args.mask_dir.glob('*.png')):
        out_paths = [body_dir / mask_path.name, cloth_dir / mask_path.name, valid_dir / mask_path.name, uncertain_dir / mask_path.name, thin_dir / mask_path.name]
        if (not args.overwrite) and all(p.exists() for p in out_paths):
            continue
        image_path = find_image_path(args.image_dir, mask_path.stem)
        if image_path is None:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = read_mask(mask_path)
        if image is None:
            continue
        body, cloth, valid, uncertain, thin = build_refined_priors(mask, image, stats)
        cv2.imwrite(str(body_dir / mask_path.name), body * 255)
        cv2.imwrite(str(cloth_dir / mask_path.name), cloth * 255)
        cv2.imwrite(str(valid_dir / mask_path.name), valid * 255)
        cv2.imwrite(str(uncertain_dir / mask_path.name), uncertain * 255)
        cv2.imwrite(str(thin_dir / mask_path.name), thin * 255)
        total += 1

    print(f'Built Hulk refine priors for {args.subject} cam {args.camera}: {total} masks written to {target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
