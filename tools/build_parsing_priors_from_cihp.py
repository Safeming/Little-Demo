#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

FACE_LABEL = 13
BODY_LABELS = (13, 14, 15, 16, 17)
CLOTH_LABELS = (5, 6, 7, 9, 10, 11, 12)
HAIR_LABELS = (1, 2)
SHOE_LABELS = (18, 19)
VALID_IMAGE_SUFFIXES = ('.jpg', '.png', '.jpeg')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build body/cloth/valid priors from CIHP masks with face-skin recovery.')
    parser.add_argument('--source-root', type=Path, default=Path('/remote-home/ming/dataSet'))
    parser.add_argument('--target-root', type=Path, default=Path('/remote-home/ming/3dgs-avatar-release-main/data/parsing_priors_from_cihp'))
    parser.add_argument('--subject', required=True, help='e.g. CoreView_377')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--sample-step', type=int, default=12, help='Frame stride for subject skin-stat estimation.')
    parser.add_argument('--max-samples-per-frame', type=int, default=2500)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def robust_open(mask: np.ndarray, k: int) -> np.ndarray:
    kernel = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def robust_close(mask: np.ndarray, k: int) -> np.ndarray:
    kernel = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def dilate(mask: np.ndarray, k: int) -> np.ndarray:
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(mask, kernel)


def robust_erode(mask: np.ndarray, k: int) -> np.ndarray:
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(mask, kernel)


def keep_components(mask: np.ndarray, anchor: np.ndarray | None = None, min_area: int = 0) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    anchor_ids: set[int] = set()
    if anchor is not None and anchor.sum() > 0:
        anchor_ids = {int(idx) for idx in np.unique(labels[anchor > 0]) if idx > 0}
    for idx in range(1, n_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= min_area or idx in anchor_ids:
            keep[labels == idx] = 1
    return keep


def find_image_path(subject_root: Path, cam_dir_name: str, stem: str) -> Path | None:
    cam_dir = subject_root / cam_dir_name
    if not cam_dir.exists():
        return None
    for suffix in VALID_IMAGE_SUFFIXES:
        path = cam_dir / f'{stem}{suffix}'
        if path.exists():
            return path
    return None


def sample_pixels(image_bgr: np.ndarray, mask: np.ndarray, limit: int) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return np.empty((0, 5), dtype=np.float32)
    if ys.size > limit:
        step = max(ys.size // limit, 1)
        ys = ys[::step][:limit]
        xs = xs[::step][:limit]
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    feats = np.stack([
        ycrcb[ys, xs, 0],
        ycrcb[ys, xs, 1],
        ycrcb[ys, xs, 2],
        lab[ys, xs, 1],
        lab[ys, xs, 2],
    ], axis=-1)
    return feats.astype(np.float32)


def compute_skin_statistics(source_subject: Path, sample_step: int, max_samples_per_frame: int) -> dict[str, list[float]]:
    samples = []
    mask_root = source_subject / 'mask_cihp'
    cams = sorted([x for x in mask_root.iterdir() if x.is_dir()])
    for cam_dir in cams:
        mask_paths = sorted(cam_dir.glob('*.png'))[::max(sample_step, 1)]
        for mask_path in mask_paths:
            stem = mask_path.stem
            image_path = find_image_path(source_subject, cam_dir.name, stem)
            if image_path is None:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
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
        raise RuntimeError(f'Could not estimate skin statistics for {source_subject.name}')

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


def skin_score_mask(image_bgr: np.ndarray, stats: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    feats = np.stack([ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2], lab[..., 1], lab[..., 2]], axis=-1)
    median = np.asarray(stats['median'], dtype=np.float32)
    scale = np.asarray(stats['scale'], dtype=np.float32)
    z = np.abs((feats - median[None, None, :]) / scale[None, None, :])
    dist = np.sqrt((z ** 2).sum(axis=-1))
    y_low, y_high = stats['y_range']
    y = feats[..., 0]
    light_gate = (y >= max(0.0, y_low - 20.0)) & (y <= min(255.0, y_high + 20.0))
    strong = (dist < 2.6) & light_gate
    soft = (dist < 3.4) & light_gate
    relaxed = (dist < 4.15) & light_gate
    return strong.astype(np.uint8), soft.astype(np.uint8), relaxed.astype(np.uint8)


def adaptive_torso_zones(mask: np.ndarray, base_body: np.ndarray, base_cloth: np.ndarray) -> dict[str, np.ndarray]:
    fg = (mask > 0).astype(np.uint8)
    face = (mask == FACE_LABEL).astype(np.uint8)
    yy, xx = np.indices(mask.shape)

    face_bbox = None
    ys, xs = np.where(face > 0)
    if ys.size >= 12:
        face_bbox = (ys.min(), ys.max(), xs.min(), xs.max())

    fg_ys, fg_xs = np.where(fg > 0)
    if fg_ys.size == 0:
        zeros = np.zeros_like(fg)
        return {
            'torso_focus': zeros,
            'chest_core': zeros,
            'chest_band': zeros,
            'neck_core': zeros,
            'upper_torso': zeros,
            'outer_torso': zeros,
            'strap_zone': zeros,
            'boundary_zone': zeros,
        }

    if face_bbox is not None:
        fy0, fy1, fx0, fx1 = face_bbox
        face_h = max(fy1 - fy0 + 1, 6)
        face_w = max(fx1 - fx0 + 1, 6)
        cx = 0.5 * (fx0 + fx1)
        torso_top = fy1 + 0.06 * face_h
        torso_bottom = fy1 + 1.85 * face_h
        torso_half_w = max(0.60 * face_w, 12.0)
    else:
        y0, y1 = fg_ys.min(), fg_ys.max()
        x0, x1 = fg_xs.min(), fg_xs.max()
        h = max(y1 - y0 + 1, 8)
        w = max(x1 - x0 + 1, 8)
        cx = 0.5 * (x0 + x1)
        torso_top = y0 + 0.10 * h
        torso_bottom = y0 + 0.62 * h
        torso_half_w = max(0.22 * w, 12.0)

    torso_seed = ((yy >= torso_top) & (yy <= torso_bottom) & (np.abs(xx - cx) <= 1.35 * torso_half_w) & (fg > 0)).astype(np.uint8)
    seed_pixels = np.where((torso_seed > 0) & ((base_body > 0) | (base_cloth > 0)))
    if seed_pixels[0].size >= 24:
        cx = float(np.median(seed_pixels[1]))
        q10 = np.quantile(seed_pixels[1], 0.10)
        q90 = np.quantile(seed_pixels[1], 0.90)
        torso_half_w = max(0.5 * (q90 - q10), torso_half_w * 0.75, 10.0)

    torso_h = torso_bottom - torso_top
    torso_focus = ((yy >= torso_top) & (yy <= torso_bottom) & (np.abs(xx - cx) <= 0.72 * torso_half_w) & (fg > 0)).astype(np.uint8)
    chest_core = ((yy >= torso_top + 0.14 * torso_h) & (yy <= torso_top + 0.58 * torso_h) & (np.abs(xx - cx) <= 0.36 * torso_half_w) & (fg > 0)).astype(np.uint8)
    chest_band = ((yy >= torso_top + 0.04 * torso_h) & (yy <= torso_top + 0.68 * torso_h) & (np.abs(xx - cx) <= 0.50 * torso_half_w) & (fg > 0)).astype(np.uint8)
    neck_core = ((yy >= torso_top - 0.05 * torso_h) & (yy <= torso_top + 0.22 * torso_h) & (np.abs(xx - cx) <= 0.28 * torso_half_w) & (fg > 0)).astype(np.uint8)
    upper_torso = ((yy >= torso_top - 0.04 * torso_h) & (yy <= torso_top + 0.48 * torso_h) & (np.abs(xx - cx) <= 0.82 * torso_half_w) & (fg > 0)).astype(np.uint8)
    outer_torso = ((yy >= torso_top - 0.02 * torso_h) & (yy <= torso_top + 0.52 * torso_h) & (np.abs(xx - cx) >= 0.18 * torso_half_w) & (np.abs(xx - cx) <= 0.82 * torso_half_w) & (fg > 0)).astype(np.uint8)
    strap_zone = ((yy >= torso_top - 0.02 * torso_h) & (yy <= torso_top + 0.42 * torso_h) & (np.abs(xx - cx) >= 0.26 * torso_half_w) & (np.abs(xx - cx) <= 0.62 * torso_half_w) & (fg > 0)).astype(np.uint8)

    cloth_d = dilate(base_cloth.astype(np.uint8), 5)
    body_d = dilate(base_body.astype(np.uint8), 5)
    boundary_zone = (cloth_d * body_d).astype(np.uint8)
    boundary_zone = robust_close(boundary_zone, 5)
    return {
        'torso_focus': torso_focus,
        'chest_core': chest_core,
        'chest_band': chest_band,
        'neck_core': neck_core,
        'upper_torso': upper_torso,
        'outer_torso': outer_torso,
        'strap_zone': strap_zone,
        'boundary_zone': boundary_zone,
    }


def build_priors_for_frame(mask: np.ndarray, image_bgr: np.ndarray, stats: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fg = (mask > 0).astype(np.uint8)
    hair = np.isin(mask, HAIR_LABELS).astype(np.uint8)
    shoe = np.isin(mask, SHOE_LABELS).astype(np.uint8)
    base_body = np.isin(mask, BODY_LABELS).astype(np.uint8)
    base_cloth = np.isin(mask, CLOTH_LABELS).astype(np.uint8)
    zones = adaptive_torso_zones(mask, base_body, base_cloth)
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

    near_body = dilate(base_body, 13)
    recover_zone = np.clip((near_body * torso_focus) + chest_band + neck_core + 0.5 * upper_torso, 0, 1)
    strap_anchor = (base_cloth * strap_zone * (1 - neck_core)).astype(np.uint8)
    strap_anchor = keep_components(robust_open(robust_close(strap_anchor, 3), 3), anchor=base_cloth * strap_zone, min_area=18)
    cloth_guard = np.clip(strap_anchor + (base_cloth * outer_torso * (1 - chest_core)), 0, 1).astype(np.uint8)

    chest_center = np.clip(chest_core + neck_core, 0, 1).astype(np.uint8)
    chest_skin = (soft_skin * np.clip(chest_core + neck_core + 0.35 * chest_band, 0, 1) * (1 - cloth_guard) * (1 - hair) * (1 - shoe)).astype(np.uint8)
    center_skin = (relaxed_skin * chest_center * (1 - strap_anchor) * (1 - hair) * (1 - shoe)).astype(np.uint8)
    torso_skin = (strong_skin * recover_zone * (1 - cloth_guard) * (1 - hair) * (1 - shoe)).astype(np.uint8)
    recovered_skin = np.clip(torso_skin + chest_skin + center_skin, 0, 1).astype(np.uint8)
    recover_anchor = np.clip(dilate(base_body, 9) + neck_core + chest_core, 0, 1).astype(np.uint8)
    recovered_skin = keep_components(recovered_skin, anchor=recover_anchor, min_area=42)
    recovered_skin = robust_close(robust_open(recovered_skin, 3), 5)
    recovered_skin = (recovered_skin * torso_focus) + (base_body * (1 - torso_focus))
    recovered_skin = np.clip(recovered_skin, 0, 1).astype(np.uint8)

    body = np.clip(base_body + recovered_skin, 0, 1).astype(np.uint8)
    body = keep_components(body, anchor=np.clip(base_body + neck_core, 0, 1).astype(np.uint8), min_area=32)
    cloth = (base_cloth * (1 - recovered_skin) * (1 - hair) * (1 - shoe)).astype(np.uint8)
    cloth = np.clip(cloth + strap_anchor, 0, 1).astype(np.uint8)
    cloth = keep_components(cloth, anchor=np.clip(base_cloth + strap_anchor, 0, 1).astype(np.uint8), min_area=36)

    conflict = (base_cloth * soft_skin * chest_band).astype(np.uint8)
    uncertain = ((conflict + (dilate(body, 3) * dilate(cloth, 3))) > 1).astype(np.uint8)
    uncertain = (uncertain * boundary_zone).astype(np.uint8)
    uncertain = robust_open(uncertain, 3)

    valid = (fg * (1 - hair) * (1 - shoe) * (1 - uncertain) * np.clip(body + cloth, 0, 1)).astype(np.uint8)
    body = (body * valid).astype(np.uint8)
    cloth = (cloth * valid).astype(np.uint8)
    return body, cloth, valid, uncertain


def main() -> int:
    args = parse_args()
    source_subject = args.source_root / args.subject
    mask_root = source_subject / 'mask_cihp'
    if not mask_root.exists():
        raise FileNotFoundError(f'Missing source mask_cihp directory: {mask_root}')

    target_subject = args.target_root / args.subject
    ensure_dir(target_subject)
    stats = compute_skin_statistics(source_subject, args.sample_step, args.max_samples_per_frame)
    (target_subject / 'skin_stats.json').write_text(json.dumps(stats, indent=2))

    total = 0
    cams = sorted([x for x in mask_root.iterdir() if x.is_dir()])
    for cam_dir in cams:
        cam_name = cam_dir.name.replace('Camera_B', '')
        body_out = target_subject / cam_name / 'body'
        cloth_out = target_subject / cam_name / 'cloth'
        valid_out = target_subject / cam_name / 'valid'
        uncertain_out = target_subject / cam_name / 'uncertain'
        ensure_dir(body_out)
        ensure_dir(cloth_out)
        ensure_dir(valid_out)
        ensure_dir(uncertain_out)

        for mask_path in sorted(cam_dir.glob('*.png')):
            stem = mask_path.stem
            body_path = body_out / mask_path.name
            cloth_path = cloth_out / mask_path.name
            valid_path = valid_out / mask_path.name
            uncertain_path = uncertain_out / mask_path.name
            if not args.overwrite and body_path.exists() and cloth_path.exists() and valid_path.exists() and uncertain_path.exists():
                continue

            image_path = find_image_path(source_subject, cam_dir.name, stem)
            if image_path is None:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                continue

            body, cloth, valid, uncertain = build_priors_for_frame(mask, image, stats)
            cv2.imwrite(str(body_path), body * 255)
            cv2.imwrite(str(cloth_path), cloth * 255)
            cv2.imwrite(str(valid_path), valid * 255)
            cv2.imwrite(str(uncertain_path), uncertain * 255)
            total += 1

    print(f'Built parsing priors for {args.subject}: {total} masks written to {target_subject}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
