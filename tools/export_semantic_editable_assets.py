import glob
import json
import os
import re
from pathlib import Path

import cv2
import hydra
import imageio.v2 as imageio
import numpy as np
import torch
import torchvision
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

_VIEW_NAME_RE = re.compile(r'^c(?P<cam>\d+)_f(?P<frame>\d+)$')

COARSE_REGION_NAMES = ('skin', 'cloth', 'hair', 'face', 'uncertain')
FINE_REGION_NAMES = ('upper', 'lower', 'shoes', 'accessory')
ALL_REGION_NAMES = ('skin', 'hair', 'upper', 'lower', 'shoes', 'accessory', 'cloth', 'face', 'uncertain')
CIHP_PALETTE = np.array([
    [0, 0, 0], [128, 0, 0], [255, 0, 0], [0, 85, 0], [170, 0, 51],
    [255, 85, 0], [0, 0, 85], [0, 119, 221], [85, 85, 0], [0, 85, 85],
    [85, 51, 0], [52, 86, 128], [0, 128, 0], [0, 0, 255], [51, 170, 221],
    [0, 255, 255], [85, 255, 170], [170, 255, 85], [255, 255, 255], [220, 220, 220],
], dtype=np.uint8)
RAW_CIHP_LABEL_NAMES = {
    1: 'hat',
    2: 'hair',
    3: 'glove',
    4: 'sunglasses',
    5: 'upper_clothes',
    6: 'dress',
    7: 'coat',
    8: 'socks',
    9: 'pants',
    10: 'jumpsuits',
    11: 'scarf',
    12: 'skirt',
    13: 'face',
    14: 'left_arm',
    15: 'right_arm',
    16: 'left_leg',
    17: 'right_leg',
    18: 'left_shoe',
    19: 'right_shoe',
}
RAW_CIHP_NAME_TO_LABEL = {name: label_idx for label_idx, name in RAW_CIHP_LABEL_NAMES.items()}
RAW_CIHP_EXPORT_NAMES = tuple(RAW_CIHP_LABEL_NAMES.values())
RAW_CIHP_PREVIEW_ORDER = (
    'hair',
    'face',
    'upper_clothes',
    'coat',
    'dress',
    'pants',
    'skirt',
    'jumpsuits',
    'left_arm',
    'right_arm',
    'left_leg',
    'right_leg',
    'left_shoe',
    'right_shoe',
    'socks',
    'hat',
    'sunglasses',
    'glove',
    'scarf',
)
GROUPED_PARSER_PREVIEW_GROUPS = {
    'hair': ('hair',),
    'face': ('face',),
    'upper_wear': ('upper_clothes', 'coat', 'dress', 'jumpsuits', 'scarf'),
    'lower_wear': ('pants', 'skirt', 'dress', 'jumpsuits', 'socks'),
    'arms': ('left_arm', 'right_arm', 'glove'),
    'legs': ('left_leg', 'right_leg'),
    'shoes': ('left_shoe', 'right_shoe'),
    'head_acc': ('hat', 'sunglasses'),
}
RAW_CIHP_PREVIEW_MIN_AREAS = {
    'hat': 48,
    'glove': 48,
    'sunglasses': 48,
    'scarf': 48,
    'socks': 48,
    'coat': 64,
    'dress': 64,
    'jumpsuits': 64,
    'skirt': 48,
}
RAW_CIHP_PREVIEW_RELATIVE_MIN_FRACTIONS = {
    'coat': 0.08,
    'dress': 0.10,
    'jumpsuits': 0.10,
    'skirt': 0.06,
}
RAW_CIHP_CLOTH_STRUCTURE_NAMES = (
    'upper_clothes',
    'coat',
    'dress',
    'pants',
    'jumpsuits',
    'skirt',
    'socks',
    'scarf',
)
PARSER_LABEL_GROUPS = {
    'hair': (2,),
    'hat': (1,),
    'face': (13,),
    'skin': (13, 14, 15, 16, 17),
    'upper': (5, 7),
    'lower': (9, 12),
    'full_cloth': (6, 10),
    'shoes': (18, 19),
    'accessory': (1, 3, 4, 8, 11),
}


def _to_python(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _save_image(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(tensor.clamp(0.0, 1.0), path)


def _binary_morph(mask, op, ksize):
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    arr = (mask > 0.5).astype(np.uint8)
    if op == 'open':
        arr = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)
    elif op == 'close':
        arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
    elif op == 'dilate':
        arr = cv2.dilate(arr, kernel, iterations=1)
    elif op == 'erode':
        arr = cv2.erode(arr, kernel, iterations=1)
    return arr.astype(np.float32)


def _undistort_resize_image(image_path, mask_path, camera, out_hw, white_bg=False):
    K = np.array(camera['K'], dtype=np.float32).copy()
    dist = np.array(camera['D'], dtype=np.float32).ravel()
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    H, W = image.shape[:2]
    M = np.eye(3, dtype=np.float32)
    M[0, 2] = (K[0, 2] - W / 2) / K[0, 0]
    M[1, 2] = (K[1, 2] - H / 2) / K[1, 1]
    K[0, 2] = W / 2
    K[1, 2] = H / 2

    image = cv2.undistort(image, K, dist, None)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (mask.shape[1], mask.shape[0]), cv2.CV_32FC1)
    mask = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_NEAREST)
    out_h, out_w = out_hw
    image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    fg = (mask != 0).astype(np.float32)
    if white_bg:
        image[fg < 0.5] = 255.0
    else:
        image[fg < 0.5] = 0.0
    return image.astype(np.float32) / 255.0, fg, K, dist


def _load_prior_mask(path, K, dist, out_hw):
    if not path or not os.path.exists(path):
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = cv2.undistort(mask, K, dist, None)
    out_h, out_w = out_hw
    mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _resolve_parser_mask_path(subject, cam_name, frame_idx, parser_root, parser_layout='cihp_subject'):
    root = Path(parser_root)
    frame_name = f'{int(frame_idx):06d}.png'
    if parser_layout == 'cihp_subject':
        return root / subject / 'mask_cihp' / f'Camera_B{int(cam_name)}' / frame_name
    if parser_layout == 'flat_png':
        return root / frame_name
    raise ValueError(f'Unsupported parser layout: {parser_layout}')


def _load_parser_index_mask(subject, cam_name, frame_idx, K, dist, out_hw, parser_root, parser_layout='cihp_subject'):
    if not parser_root:
        return None
    mask_path = _resolve_parser_mask_path(subject, cam_name, frame_idx, parser_root, parser_layout)
    if not mask_path.exists():
        return None
    # Preserve palette-index PNG values from external parsers such as Hulk.
    with Image.open(mask_path) as img:
        mask = np.array(img)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = cv2.undistort(mask, K, dist, None)
    out_h, out_w = out_hw
    if mask.shape != (out_h, out_w):
        mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.int32)


def _mask_from_index(parser_mask, labels):
    if parser_mask is None:
        return None
    label_array = np.asarray(list(labels), dtype=parser_mask.dtype)
    mask = np.isin(parser_mask, label_array).astype(np.float32)
    return mask.clip(0.0, 1.0)


def _extract_parser_regions(parser_mask):
    if parser_mask is None:
        return {}
    return {name: _mask_from_index(parser_mask, labels) for name, labels in PARSER_LABEL_GROUPS.items()}


def _load_cihp_mask(subject, cam_name, frame_idx, labels, out_hw, cihp_root):
    mask_path = Path(cihp_root) / subject / 'mask_cihp' / f'Camera_B{int(cam_name)}' / f'{int(frame_idx):06d}.png'
    if not mask_path.exists():
        return None
    mask = imageio.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    label_array = np.asarray(list(labels), dtype=mask.dtype)
    mask = np.isin(mask, label_array).astype(np.float32)
    mask = _binary_morph(_binary_morph(mask, 'close', 5), 'open', 3)
    out_h, out_w = out_hw
    if mask.shape != (out_h, out_w):
        mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return mask.clip(0.0, 1.0)


def _mask_bbox(mask, threshold=0.5):
    if mask is None:
        return None
    ys, xs = np.where(mask > threshold)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _mask_components(mask):
    arr = (mask > 0.5).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(arr, connectivity=8)
    return num_labels, labels, stats


def _largest_component(mask):
    if mask is None:
        return None
    num_labels, labels, stats = _mask_components(mask)
    if num_labels <= 1:
        return mask.clip(0.0, 1.0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return (labels == keep).astype(np.float32)


def _keep_components_touching(mask, touch_mask, min_area=24):
    if mask is None:
        return None
    num_labels, labels, stats = _mask_components(mask)
    if num_labels <= 1:
        return mask.clip(0.0, 1.0)
    keep = np.zeros_like(mask, dtype=np.float32)
    touch = touch_mask > 0.5
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == label_id
        if np.any(comp & touch):
            keep[comp] = 1.0
    return keep


def _rgb_distance(image, mean_rgb, std_rgb):
    std_rgb = np.maximum(std_rgb, 0.035)
    return np.sqrt(np.sum(np.square((image - mean_rgb.reshape(1, 1, 3)) / std_rgb.reshape(1, 1, 3)), axis=2))


def _estimate_color_stats(image, seed_mask):
    if seed_mask is None or float((seed_mask > 0.5).sum()) < 24:
        return None
    pixels = image[seed_mask > 0.5]
    mean_rgb = pixels.mean(axis=0)
    std_rgb = pixels.std(axis=0)
    brightness = pixels.mean(axis=1)
    saturation = pixels.max(axis=1) - pixels.min(axis=1)
    warmth = 1.10 * pixels[:, 0] + 0.70 * pixels[:, 1] - 1.30 * pixels[:, 2]
    return {
        'mean_rgb': mean_rgb.astype(np.float32),
        'std_rgb': std_rgb.astype(np.float32),
        'brightness_mean': float(brightness.mean()),
        'brightness_std': float(brightness.std()),
        'saturation_mean': float(saturation.mean()),
        'saturation_std': float(saturation.std()),
        'warmth_mean': float(warmth.mean()),
        'warmth_std': float(warmth.std()),
    }


def _recover_skin_mask(image, fg_mask, skin_mask, cloth_mask, hair_mask, face_mask):
    base_skin = np.zeros_like(fg_mask) if skin_mask is None else np.clip(skin_mask, 0.0, 1.0)
    face_seed = np.zeros_like(fg_mask) if face_mask is None else np.clip(face_mask, 0.0, 1.0)
    seed = np.maximum(base_skin, face_seed)
    seed = _binary_morph(_binary_morph(seed, 'close', 5), 'open', 3)
    stats = _estimate_color_stats(image, seed)
    if stats is None:
        return base_skin

    bbox = _mask_bbox(fg_mask)
    if bbox is None:
        return base_skin
    y0, y1, x0, x1 = bbox
    h = max(y1 - y0 + 1, 8)
    w = max(x1 - x0 + 1, 8)
    yy, xx = np.mgrid[0:fg_mask.shape[0], 0:fg_mask.shape[1]].astype(np.float32)
    x_center = 0.5 * (x0 + x1)
    x_rel = np.abs(xx - x_center) / (0.5 * w + 1e-6)
    y_rel = (yy - float(y0)) / float(h)

    brightness = image.mean(axis=2)
    saturation = image.max(axis=2) - image.min(axis=2)
    warmth = 1.10 * image[..., 0] + 0.70 * image[..., 1] - 1.30 * image[..., 2]
    color_dist = _rgb_distance(image, stats['mean_rgb'], stats['std_rgb'])
    warm_margin = max(0.10, 2.5 * stats['warmth_std'])
    bright_margin = max(0.08, 2.5 * stats['brightness_std'])
    sat_margin = max(0.08, 2.5 * stats['saturation_std'])

    if face_mask is not None and float((face_mask > 0.5).sum()) >= 16:
        fy0, fy1, fx0, fx1 = _mask_bbox(face_mask)
        fh = max(fy1 - fy0 + 1, 6)
        fw = max(fx1 - fx0 + 1, 6)
        fx_center = 0.5 * (fx0 + fx1)
        face_x = np.abs(xx - fx_center) / (0.5 * fw + 1e-6)
        torso_band = ((yy > fy1 - 0.08 * fh) & (yy < fy1 + 2.3 * fh) & (face_x < 1.55)).astype(np.float32)
        upper_limb_band = ((yy > fy0 - 0.1 * fh) & (yy < fy1 + 4.8 * fh) & (face_x < 3.9)).astype(np.float32)
        geo_gate = np.maximum(torso_band, upper_limb_band)
    else:
        geo_gate = ((y_rel > 0.02) & (y_rel < 0.82) & (x_rel < 0.48)).astype(np.float32)

    cloth_block = np.zeros_like(fg_mask) if cloth_mask is None else np.clip(cloth_mask, 0.0, 1.0)
    hair_block = np.zeros_like(fg_mask) if hair_mask is None else np.clip(hair_mask, 0.0, 1.0)
    skin_like = (
        (color_dist < 2.75)
        & (np.abs(warmth - stats['warmth_mean']) < warm_margin)
        & (np.abs(brightness - stats['brightness_mean']) < bright_margin)
        & (np.abs(saturation - stats['saturation_mean']) < sat_margin)
        & (brightness > 0.12)
        & (saturation > 0.02)
    ).astype(np.float32)
    recovered = fg_mask * geo_gate * skin_like * (1.0 - 0.85 * cloth_block) * (1.0 - 0.90 * hair_block)
    recovered = np.maximum(recovered, base_skin)
    recovered = _binary_morph(_binary_morph(recovered, 'close', 5), 'open', 3)
    recovered = _keep_components_touching(recovered, _binary_morph(seed, 'dilate', 11), min_area=36)
    recovered = np.maximum(recovered, base_skin)
    return recovered.clip(0.0, 1.0)


def _refine_cloth_mask(fg_mask, cloth_mask, skin_mask, hair_mask, face_mask):
    if cloth_mask is None:
        return None
    refined = np.clip(cloth_mask, 0.0, 1.0) * np.clip(fg_mask, 0.0, 1.0)
    for blocker, scale in ((skin_mask, 0.92), (hair_mask, 0.75), (face_mask, 0.95)):
        if blocker is not None:
            refined = refined * (1.0 - scale * np.clip(blocker, 0.0, 1.0))
    refined = _binary_morph(_binary_morph(refined, 'close', 5), 'open', 3)
    return refined.clip(0.0, 1.0)


def _head_guided_face_hair(image, fg_mask, skin_mask, cloth_mask, hair_mask, face_mask):
    bbox = _mask_bbox(fg_mask)
    if bbox is None:
        return face_mask, hair_mask

    y0, y1, x0, x1 = bbox
    h = max(y1 - y0 + 1, 8)
    w = max(x1 - x0 + 1, 8)
    yy, xx = np.mgrid[0:fg_mask.shape[0], 0:fg_mask.shape[1]].astype(np.float32)
    y_rel = (yy - float(y0)) / float(h)
    x_rel = np.abs(xx - 0.5 * (x0 + x1)) / (0.5 * w + 1e-6)

    brightness = image.mean(axis=2)
    saturation = image.max(axis=2) - image.min(axis=2)
    warmth = 1.10 * image[..., 0] + 0.70 * image[..., 1] - 1.30 * image[..., 2]

    skin_hint = skin_mask if skin_mask is not None else np.zeros_like(fg_mask)
    if face_mask is None or face_mask.sum() < 64:
        face_geom = ((y_rel > 0.02) & (y_rel < 0.28) & (x_rel < 0.17)).astype(np.float32)
        face_color = ((brightness > 0.16) & (warmth > -0.01) & (saturation > 0.03)).astype(np.float32)
        face_fallback = fg_mask * face_geom * face_color * np.clip(skin_hint + 0.35, 0.0, 1.0)
        face_fallback = _binary_morph(_binary_morph(face_fallback, 'close', 5), 'open', 3)
        if face_fallback.sum() > 48:
            face_mask = face_fallback

    skin_seed = np.maximum(np.clip(skin_hint, 0.0, 1.0), np.clip(face_mask, 0.0, 1.0) if face_mask is not None else 0.0)
    skin_stats = _estimate_color_stats(image, skin_seed)
    cloth_block = np.zeros_like(fg_mask) if cloth_mask is None else np.clip(cloth_mask, 0.0, 1.0)
    base_hair = np.zeros_like(fg_mask) if hair_mask is None else np.clip(hair_mask, 0.0, 1.0)

    face_bbox = _mask_bbox(face_mask)
    if face_bbox is not None:
        fy0, fy1, fx0, fx1 = face_bbox
        fh = max(fy1 - fy0 + 1, 6)
        fw = max(fx1 - fx0 + 1, 6)
        fx_center = 0.5 * (fx0 + fx1)
        face_x = np.abs(xx - fx_center) / (0.5 * fw + 1e-6)
        hair_geom = ((yy > fy0 - 1.12 * fh) & (yy < fy1 + 0.18 * fh) & (face_x < 1.42)).astype(np.float32)
        crown_zone = ((yy > fy0 - 1.05 * fh) & (yy < fy0 + 0.12 * fh) & (face_x < 1.15)).astype(np.float32)
        side_zone = ((yy > fy0 - 0.04 * fh) & (yy < fy1 + 0.10 * fh) & (face_x > 0.82) & (face_x < 1.48)).astype(np.float32)
        upper_shell = ((yy > fy0 - 0.90 * fh) & (yy < fy0 + 0.22 * fh) & (face_x < 1.20)).astype(np.float32)
        face_core = _binary_morph(np.clip(face_mask, 0.0, 1.0), 'dilate', 11)
        lower_face = ((yy > fy0 + 0.18 * fh) & (yy < fy1 + 0.24 * fh) & (face_x < 0.90)).astype(np.float32)
        neck_block = ((yy > fy1 + 0.04 * fh) & (yy < fy1 + 0.52 * fh) & (face_x < 1.02)).astype(np.float32)
        seed_touch = np.maximum(_binary_morph(base_hair, 'dilate', 9), _binary_morph(crown_zone, 'dilate', 15))
    else:
        hair_geom = ((y_rel > -0.02) & (y_rel < 0.24) & (x_rel < 0.20)).astype(np.float32)
        crown_zone = ((y_rel > -0.02) & (y_rel < 0.14) & (x_rel < 0.17)).astype(np.float32)
        side_zone = ((y_rel > 0.03) & (y_rel < 0.20) & (x_rel > 0.10) & (x_rel < 0.22)).astype(np.float32)
        upper_shell = ((y_rel > -0.02) & (y_rel < 0.18) & (x_rel < 0.18)).astype(np.float32)
        face_core = np.zeros_like(fg_mask)
        lower_face = np.zeros_like(fg_mask)
        neck_block = np.zeros_like(fg_mask)
        seed_touch = np.maximum(_binary_morph(base_hair, 'dilate', 9), _binary_morph(crown_zone, 'dilate', 13))

    if skin_stats is not None:
        color_dist = _rgb_distance(image, skin_stats['mean_rgb'], skin_stats['std_rgb'])
        warm_margin = max(0.10, 2.0 * skin_stats['warmth_std'])
        bright_margin = max(0.12, 2.2 * skin_stats['brightness_std'])
        skin_like = (
            (color_dist < 2.35)
            & (np.abs(warmth - skin_stats['warmth_mean']) < warm_margin)
            & (np.abs(brightness - skin_stats['brightness_mean']) < bright_margin)
        ).astype(np.float32)
    else:
        skin_like = np.zeros_like(fg_mask)

    dark_hair = ((brightness < 0.38) & (saturation < 0.34)).astype(np.float32)
    dark_core = ((brightness < 0.28) & (saturation < 0.26)).astype(np.float32)
    non_skin = np.maximum(1.0 - skin_like, dark_core)
    shell_candidate = (fg_mask > 0.5) & (hair_geom > 0.5) & (face_core < 0.5) & ((upper_shell > 0.5) | (side_zone > 0.5))
    dark_candidate = (fg_mask > 0.5) & (hair_geom > 0.5) & (dark_hair > 0.5) & (non_skin > 0.5)
    candidate = np.logical_or.reduce([
        shell_candidate,
        dark_candidate,
        crown_zone > 0.5,
        seed_touch > 0.5,
    ]).astype(np.float32)
    candidate = candidate * fg_mask * (1.0 - 0.78 * lower_face) * (1.0 - 0.82 * neck_block * np.clip(skin_seed, 0.0, 1.0))
    candidate = candidate * (1.0 - 0.45 * cloth_block)
    candidate = _binary_morph(_binary_morph(candidate, 'close', 7), 'dilate', 5)
    candidate = _keep_components_touching(candidate, seed_touch, min_area=24)
    if candidate is None or candidate.sum() < 20:
        candidate = np.maximum(base_hair, crown_zone.astype(np.float32) * fg_mask)
    candidate = _binary_morph(_binary_morph(candidate, 'open', 3), 'close', 5)
    candidate = _keep_components_touching(candidate, np.maximum(seed_touch, _binary_morph(crown_zone, 'dilate', 17)), min_area=24)
    if candidate is None or candidate.sum() < 20:
        candidate = _largest_component(np.maximum(base_hair, candidate if candidate is not None else 0.0))

    if face_mask is not None:
        face_mask = _largest_component(_binary_morph(_binary_morph(face_mask, 'close', 5), 'open', 3)).clip(0.0, 1.0)
    candidate = np.clip(candidate if candidate is not None else base_hair, 0.0, 1.0)
    return face_mask, candidate



def _estimate_split_y(fg_mask, face_mask):
    face_bbox = _mask_bbox(face_mask)
    if face_bbox is not None:
        fy0, fy1, _, _ = face_bbox
        fh = max(fy1 - fy0 + 1, 6)
        return float(fy1 + 2.15 * fh)
    fg_bbox = _mask_bbox(fg_mask)
    if fg_bbox is None:
        return 0.5 * fg_mask.shape[0]
    y0, y1, _, _ = fg_bbox
    h = max(y1 - y0 + 1, 8)
    return float(y0 + 0.56 * h)


def _split_cloth_into_upper_lower(cloth_mask, fg_mask, face_mask, parser_regions):
    if cloth_mask is None:
        return None, None
    upper_seed = np.zeros_like(cloth_mask)
    lower_seed = np.zeros_like(cloth_mask)
    full_seed = np.zeros_like(cloth_mask)
    if parser_regions:
        upper_seed = np.clip(parser_regions.get('upper', np.zeros_like(cloth_mask)), 0.0, 1.0) * cloth_mask
        lower_seed = np.clip(parser_regions.get('lower', np.zeros_like(cloth_mask)), 0.0, 1.0) * cloth_mask
        full_seed = np.clip(parser_regions.get('full_cloth', np.zeros_like(cloth_mask)), 0.0, 1.0) * cloth_mask

    split_y = _estimate_split_y(fg_mask, face_mask)
    yy = np.mgrid[0:cloth_mask.shape[0], 0:cloth_mask.shape[1]][0].astype(np.float32)
    upper_half = (yy <= split_y).astype(np.float32)
    lower_half = (yy > split_y).astype(np.float32)

    occupied = np.clip(upper_seed + lower_seed, 0.0, 1.0)
    residual = cloth_mask * (1.0 - occupied)
    if full_seed.sum() > 0:
        residual = np.maximum(residual, full_seed)

    upper = np.clip(upper_seed + residual * upper_half, 0.0, 1.0)
    lower = np.clip(lower_seed + residual * lower_half, 0.0, 1.0)
    upper = _binary_morph(_binary_morph(upper, 'close', 5), 'open', 3)
    lower = _binary_morph(_binary_morph(lower, 'close', 5), 'open', 3)
    overlap = np.minimum(upper, lower)
    if overlap.sum() > 0:
        upper = upper * (1.0 - overlap)
        lower = lower * (1.0 - overlap)
    return upper.clip(0.0, 1.0), lower.clip(0.0, 1.0)



def _parser_rgb_tensor(parser_mask, fg_mask):
    if parser_mask is None:
        return torch.zeros(3, fg_mask.shape[0], fg_mask.shape[1], dtype=torch.float32)
    idx = np.clip(parser_mask.astype(np.int32), 0, len(CIHP_PALETTE) - 1)
    rgb = CIHP_PALETTE[idx].astype(np.float32) / 255.0
    if fg_mask is not None:
        rgb[fg_mask < 0.5] = 0.0
    return torch.from_numpy(rgb).permute(2, 0, 1).float()



def _compose_raw_parser_class_masks(parser_mask, fg_mask):
    if parser_mask is None:
        return {}
    out = {}
    base = fg_mask if fg_mask is not None else np.ones_like(parser_mask, dtype=np.float32)
    for label_idx, name in RAW_CIHP_LABEL_NAMES.items():
        mask = (parser_mask == label_idx).astype(np.float32) * base
        out[name] = mask.clip(0.0, 1.0)
    return out


def _compose_masks_direct_from_parser(fg_mask, parser_mask, valid_mask=None):
    # In direct-parser mode we trust the parser labels first and keep the full foreground.
    # The old valid/body-cloth prior truncates hair and shoes, which is exactly what we want to avoid here.
    base_mask = fg_mask
    if parser_mask is None:
        return {
            'foreground': fg_mask,
            'valid': base_mask,
            'skin': None,
            'cloth': None,
            'hair': None,
            'face': None,
            'uncertain': np.zeros_like(base_mask),
            'upper': None,
            'lower': None,
            'shoes': None,
            'accessory': None,
        }

    parser_regions = _extract_parser_regions(parser_mask)
    hair_mask = np.clip(parser_regions.get('hair', np.zeros_like(fg_mask)), 0.0, 1.0) * base_mask
    face_mask = np.clip(parser_regions.get('face', np.zeros_like(fg_mask)), 0.0, 1.0) * base_mask
    skin_mask = np.clip(parser_regions.get('skin', np.zeros_like(fg_mask)), 0.0, 1.0) * base_mask
    upper_seed = np.clip(parser_regions.get('upper', np.zeros_like(fg_mask)), 0.0, 1.0) * base_mask
    lower_seed = np.clip(parser_regions.get('lower', np.zeros_like(fg_mask)), 0.0, 1.0) * base_mask
    full_seed = np.clip(parser_regions.get('full_cloth', np.zeros_like(fg_mask)), 0.0, 1.0) * base_mask
    shoes_mask = np.clip(parser_regions.get('shoes', np.zeros_like(fg_mask)), 0.0, 1.0) * base_mask
    accessory_mask = np.clip(parser_regions.get('accessory', np.zeros_like(fg_mask)), 0.0, 1.0) * base_mask

    upper_mask, lower_mask = _split_cloth_into_upper_lower(np.clip(upper_seed + lower_seed + full_seed, 0.0, 1.0), fg_mask, face_mask, parser_regions)
    if upper_mask is None:
        upper_mask = upper_seed
    if lower_mask is None:
        lower_mask = lower_seed

    cloth_mask = np.clip(upper_mask + lower_mask + shoes_mask + accessory_mask, 0.0, 1.0)
    uncertain = np.zeros_like(base_mask)
    return {
        'foreground': fg_mask,
        'valid': base_mask,
        'skin': skin_mask.clip(0.0, 1.0),
        'cloth': cloth_mask.clip(0.0, 1.0),
        'hair': hair_mask.clip(0.0, 1.0),
        'face': face_mask.clip(0.0, 1.0),
        'uncertain': uncertain,
        'upper': upper_mask.clip(0.0, 1.0),
        'lower': lower_mask.clip(0.0, 1.0),
        'shoes': shoes_mask.clip(0.0, 1.0),
        'accessory': accessory_mask.clip(0.0, 1.0),
    }


def _compose_fine_masks(fg_mask, parser_mask, coarse_masks):
    parser_regions = _extract_parser_regions(parser_mask)
    cloth_mask = coarse_masks.get('cloth')
    skin_mask = coarse_masks.get('skin')
    hair_mask = coarse_masks.get('hair')
    face_mask = coarse_masks.get('face')
    valid_mask = coarse_masks.get('valid') if coarse_masks.get('valid') is not None else fg_mask

    upper_mask, lower_mask = _split_cloth_into_upper_lower(cloth_mask, fg_mask, face_mask, parser_regions)

    shoes_mask = None
    if parser_regions:
        shoes_mask = parser_regions.get('shoes')
        if shoes_mask is not None:
            shoes_mask = shoes_mask * fg_mask
            shoes_mask = _binary_morph(_binary_morph(shoes_mask, 'close', 3), 'open', 3)

    accessory_mask = None
    if parser_regions:
        accessory_mask = parser_regions.get('accessory')
        if accessory_mask is not None:
            blocker = np.zeros_like(accessory_mask)
            for mask in (skin_mask, hair_mask, face_mask, shoes_mask):
                if mask is not None:
                    blocker = np.maximum(blocker, np.clip(mask, 0.0, 1.0))
            accessory_mask = accessory_mask * fg_mask * (1.0 - 0.85 * blocker)
            if upper_mask is not None:
                accessory_mask = accessory_mask * (1.0 - 0.55 * np.clip(upper_mask, 0.0, 1.0))
            if lower_mask is not None:
                accessory_mask = accessory_mask * (1.0 - 0.55 * np.clip(lower_mask, 0.0, 1.0))
            accessory_mask = _binary_morph(_binary_morph(accessory_mask, 'close', 3), 'open', 3)

    if upper_mask is not None:
        upper_mask = (upper_mask * valid_mask).clip(0.0, 1.0)
    if lower_mask is not None:
        lower_mask = (lower_mask * valid_mask).clip(0.0, 1.0)

    return {
        'upper': upper_mask,
        'lower': lower_mask,
        'shoes': shoes_mask,
        'accessory': accessory_mask,
    }


def _build_uncertain_mask(base_mask, masks):
    valid_masks = [m for m in masks if m is not None and m.sum() > 0]
    if not valid_masks:
        return np.zeros_like(base_mask)

    shell_vote = np.zeros_like(base_mask)
    dilated_masks = []
    confident_cores = []
    for mask in valid_masks:
        dilated = _binary_morph(mask, 'dilate', 3)
        eroded = _binary_morph(mask, 'erode', 3)
        dilated_masks.append(dilated)
        confident_cores.append(_binary_morph(mask, 'erode', 5))
        shell_vote += np.clip(dilated - eroded, 0.0, 1.0)

    stacked = np.stack([(m > 0.25).astype(np.float32) for m in dilated_masks], axis=0)
    overlap = (stacked.sum(axis=0) >= 2).astype(np.float32)
    boundary = (shell_vote >= 2.0).astype(np.float32)
    core_union = np.maximum.reduce(confident_cores) if confident_cores else np.zeros_like(base_mask)
    uncertain = np.maximum(overlap, boundary) * base_mask * (1.0 - 0.90 * core_union)
    uncertain = _binary_morph(_binary_morph(uncertain, 'close', 3), 'open', 3)
    return (uncertain > 0.15).astype(np.float32)



def _masked_region_stats(image, mask, fg_mask=None):
    stats = {
        'count': 0,
        'coverage': 0.0,
        'rgb_mean': [0.0, 0.0, 0.0],
        'rgb_std': [0.0, 0.0, 0.0],
        'brightness_mean': 0.0,
        'brightness_std': 0.0,
        'saturation_mean': 0.0,
        'saturation_std': 0.0,
        'warmth_mean': 0.0,
        'warmth_std': 0.0,
    }
    if mask is None:
        return stats
    region = mask > 0.5
    if fg_mask is not None:
        region = region & (fg_mask > 0.5)
        fg_count = int((fg_mask > 0.5).sum())
    else:
        fg_count = int(region.size)
    count = int(region.sum())
    stats['count'] = count
    stats['coverage'] = float(count / max(fg_count, 1))
    if count == 0:
        return stats
    pixels = image[region]
    rgb_mean = pixels.mean(axis=0)
    rgb_std = pixels.std(axis=0)
    brightness = pixels.mean(axis=1)
    saturation = pixels.max(axis=1) - pixels.min(axis=1)
    warmth = 1.10 * pixels[:, 0] + 0.70 * pixels[:, 1] - 1.30 * pixels[:, 2]
    stats.update({
        'rgb_mean': rgb_mean.tolist(),
        'rgb_std': rgb_std.tolist(),
        'brightness_mean': float(brightness.mean()),
        'brightness_std': float(brightness.std()),
        'saturation_mean': float(saturation.mean()),
        'saturation_std': float(saturation.std()),
        'warmth_mean': float(warmth.mean()),
        'warmth_std': float(warmth.std()),
    })
    return stats


def _aggregate_region_appearance(records, region_name):
    total_count = 0.0
    coverage_sum = 0.0
    rgb_first = np.zeros(3, dtype=np.float64)
    rgb_second = np.zeros(3, dtype=np.float64)
    brightness_first = 0.0
    brightness_second = 0.0
    saturation_first = 0.0
    saturation_second = 0.0
    warmth_first = 0.0
    warmth_second = 0.0
    contributing = 0
    for record in records:
        stats = record['appearance'][region_name]
        count = float(stats['count'])
        if count <= 0:
            continue
        contributing += 1
        total_count += count
        coverage_sum += float(stats['coverage'])
        rgb_mean = np.asarray(stats['rgb_mean'], dtype=np.float64)
        rgb_std = np.asarray(stats['rgb_std'], dtype=np.float64)
        rgb_first += count * rgb_mean
        rgb_second += count * (np.square(rgb_std) + np.square(rgb_mean))
        for key, first, second in [
            ('brightness', brightness_first, brightness_second),
            ('saturation', saturation_first, saturation_second),
            ('warmth', warmth_first, warmth_second),
        ]:
            mean_v = float(stats[f'{key}_mean'])
            std_v = float(stats[f'{key}_std'])
            if key == 'brightness':
                brightness_first += count * mean_v
                brightness_second += count * (std_v ** 2 + mean_v ** 2)
            elif key == 'saturation':
                saturation_first += count * mean_v
                saturation_second += count * (std_v ** 2 + mean_v ** 2)
            else:
                warmth_first += count * mean_v
                warmth_second += count * (std_v ** 2 + mean_v ** 2)
    if total_count <= 0:
        return {
            'count': 0,
            'coverage_mean': 0.0,
            'rgb_mean': [0.0, 0.0, 0.0],
            'rgb_std': [0.0, 0.0, 0.0],
            'brightness_mean': 0.0,
            'brightness_std': 0.0,
            'saturation_mean': 0.0,
            'saturation_std': 0.0,
            'warmth_mean': 0.0,
            'warmth_std': 0.0,
        }
    rgb_mean = rgb_first / total_count
    rgb_var = np.maximum(rgb_second / total_count - np.square(rgb_mean), 0.0)
    brightness_mean = brightness_first / total_count
    brightness_var = max(brightness_second / total_count - brightness_mean ** 2, 0.0)
    saturation_mean = saturation_first / total_count
    saturation_var = max(saturation_second / total_count - saturation_mean ** 2, 0.0)
    warmth_mean = warmth_first / total_count
    warmth_var = max(warmth_second / total_count - warmth_mean ** 2, 0.0)
    return {
        'count': int(total_count),
        'coverage_mean': float(coverage_sum / max(contributing, 1)),
        'rgb_mean': rgb_mean.tolist(),
        'rgb_std': np.sqrt(rgb_var).tolist(),
        'brightness_mean': float(brightness_mean),
        'brightness_std': float(np.sqrt(brightness_var)),
        'saturation_mean': float(saturation_mean),
        'saturation_std': float(np.sqrt(saturation_var)),
        'warmth_mean': float(warmth_mean),
        'warmth_std': float(np.sqrt(warmth_var)),
    }



def _mask_color_tensor(mask, label_idx):
    if mask is None:
        return None
    color = CIHP_PALETTE[int(label_idx)].astype(np.float32) / 255.0
    rgb = mask[..., None].astype(np.float32) * color.reshape(1, 1, 3)
    return torch.from_numpy(rgb).permute(2, 0, 1).float()


def _remove_small_components(mask, min_area=24):
    if mask is None:
        return None
    arr = (mask > 0.5).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(arr, connectivity=8)
    if num_labels <= 1:
        return arr.astype(np.float32)
    keep = np.zeros_like(arr, dtype=np.uint8)
    for label_id in range(1, num_labels):
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= min_area:
            keep[labels == label_id] = 1
    return keep.astype(np.float32)


def _panel_label(text):
    return text.replace('_', ' ')


def _preview_min_area_for(name, default=24):
    return max(int(default), int(RAW_CIHP_PREVIEW_MIN_AREAS.get(name, default)))


def _clean_preview_mask(name, mask, min_area=24):
    if mask is None:
        return None
    return _remove_small_components(mask, min_area=_preview_min_area_for(name, default=min_area))


def _mask_area(mask):
    if mask is None:
        return 0.0
    return float((mask > 0.5).sum())


def _keep_preview_mask(name, mask, cleaned_masks, min_area=24):
    area = _mask_area(mask)
    if area < float(_preview_min_area_for(name, default=min_area)):
        return False
    relative_min = RAW_CIHP_PREVIEW_RELATIVE_MIN_FRACTIONS.get(name)
    if relative_min is not None:
        cloth_area = sum(_mask_area(cleaned_masks.get(member)) for member in RAW_CIHP_CLOTH_STRUCTURE_NAMES)
        if cloth_area > 0 and area < float(relative_min) * cloth_area:
            return False
    if name in ('dress', 'jumpsuits'):
        upper_area = _mask_area(cleaned_masks.get('upper_clothes')) + _mask_area(cleaned_masks.get('coat'))
        lower_area = _mask_area(cleaned_masks.get('pants')) + _mask_area(cleaned_masks.get('skirt'))
        if upper_area > 1.25 * area and lower_area > 1.25 * area:
            return False
    return True


def _label_panel_tensor(panel, label):
    arr = panel.permute(1, 2, 0).detach().cpu().numpy().copy()
    h, w = arr.shape[:2]
    bar_h = max(22, h // 12)
    arr[:bar_h] *= 0.12
    font_scale = max(0.45, min(0.90, bar_h / 26.0))
    thickness = 1 if bar_h < 30 else 2
    baseline_y = min(h - 4, int(bar_h * 0.72))
    cv2.putText(
        arr,
        _panel_label(label),
        (6, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (1.0, 1.0, 1.0),
        thickness,
        cv2.LINE_AA,
    )
    return torch.from_numpy(arr).permute(2, 0, 1).float()


def _panel_grid_tensors(panels, cols=4):
    if not panels:
        raise ValueError('panels must be non-empty')
    c, h, w = panels[0].shape
    rows = (len(panels) + cols - 1) // cols
    blank = torch.zeros(c, h, w, dtype=panels[0].dtype)
    filled = list(panels) + [blank] * (rows * cols - len(panels))
    row_tensors = []
    for row_idx in range(rows):
        row = filled[row_idx * cols:(row_idx + 1) * cols]
        row_tensors.append(torch.cat(row, dim=2))
    return torch.cat(row_tensors, dim=1)


def _build_grouped_parser_masks(raw_parser_masks, min_area=24):
    if not raw_parser_masks:
        return {}
    sample_mask = next(iter(raw_parser_masks.values()))
    cleaned_masks = {
        name: _clean_preview_mask(name, raw_parser_masks.get(name), min_area=min_area)
        for name in RAW_CIHP_LABEL_NAMES.values()
    }
    grouped = {}
    for group_name, member_names in GROUPED_PARSER_PREVIEW_GROUPS.items():
        merged = np.zeros_like(sample_mask, dtype=np.float32)
        for name in member_names:
            mask = cleaned_masks.get(name)
            if mask is not None and _keep_preview_mask(name, mask, cleaned_masks, min_area=min_area):
                merged = np.maximum(merged, np.clip(mask, 0.0, 1.0))
        merged = _remove_small_components(merged, min_area=min_area)
        if merged is not None and _mask_area(merged) >= float(min_area):
            grouped[group_name] = merged
    return grouped


def _direct_parser_preview_tensor(image, fg_mask, parser_mask, raw_parser_masks, min_area=24):
    rgb = torch.from_numpy(image).permute(2, 0, 1).float()
    panels = [
        _label_panel_tensor(rgb, 'render'),
        _label_panel_tensor(_parser_rgb_tensor(parser_mask, fg_mask), 'parser_map'),
    ]
    cleaned_masks = {
        name: _clean_preview_mask(name, raw_parser_masks.get(name), min_area=min_area)
        for name in RAW_CIHP_LABEL_NAMES.values()
    }
    for name in RAW_CIHP_PREVIEW_ORDER:
        mask = cleaned_masks.get(name)
        if mask is None or not _keep_preview_mask(name, mask, cleaned_masks, min_area=min_area):
            continue
        panel = _mask_color_tensor(mask, RAW_CIHP_NAME_TO_LABEL[name])
        if panel is None:
            continue
        panels.append(_label_panel_tensor(panel, name))
    return _panel_grid_tensors(panels, cols=4)


def _grouped_parser_preview_tensor(image, fg_mask, grouped_masks):
    rgb = torch.from_numpy(image).permute(2, 0, 1).float()
    panels = [_label_panel_tensor(rgb, 'render')]
    color_sources = {
        'hair': 2,
        'face': 13,
        'upper_wear': 5,
        'lower_wear': 9,
        'arms': 14,
        'legs': 16,
        'shoes': 18,
        'head_acc': 1,
    }
    for name in ('hair', 'face', 'upper_wear', 'lower_wear', 'arms', 'legs', 'shoes', 'head_acc'):
        mask = grouped_masks.get(name)
        if mask is None:
            continue
        panel = _mask_color_tensor(mask, color_sources[name])
        if panel is None:
            continue
        panels.append(_label_panel_tensor(panel, name))
    return _panel_grid_tensors(panels, cols=3)


def _compose_masks(image, fg_mask, skin_mask, cloth_mask, hair_mask, face_mask, valid_mask):
    semantic_fg = valid_mask if valid_mask is not None and valid_mask.sum() > 0 else fg_mask
    face_mask, hair_mask = _head_guided_face_hair(image, fg_mask, skin_mask, cloth_mask, hair_mask, face_mask)
    skin_mask = _recover_skin_mask(image, fg_mask, skin_mask, cloth_mask, hair_mask, face_mask)
    if face_mask is not None:
        skin_mask = face_mask if skin_mask is None else np.maximum(skin_mask, 0.82 * face_mask)
    cloth_mask = _refine_cloth_mask(fg_mask, cloth_mask, skin_mask, hair_mask, face_mask)
    face_mask, hair_mask = _head_guided_face_hair(image, fg_mask, skin_mask, cloth_mask, hair_mask, face_mask)
    if semantic_fg is not None:
        if skin_mask is not None:
            skin_mask = (skin_mask * semantic_fg).clip(0.0, 1.0)
        if cloth_mask is not None:
            cloth_mask = (cloth_mask * semantic_fg).clip(0.0, 1.0)
        if hair_mask is not None:
            hair_mask = (hair_mask * semantic_fg).clip(0.0, 1.0)
        if face_mask is not None:
            face_mask = (face_mask * semantic_fg).clip(0.0, 1.0)
    if hair_mask is not None and face_mask is not None:
        face_bbox = _mask_bbox(face_mask)
        if face_bbox is not None:
            fy0, fy1, fx0, fx1 = face_bbox
            fh = max(fy1 - fy0 + 1, 6)
            fw = max(fx1 - fx0 + 1, 6)
            yy, xx = np.mgrid[0:fg_mask.shape[0], 0:fg_mask.shape[1]].astype(np.float32)
            face_x = np.abs(xx - 0.5 * (fx0 + fx1)) / (0.5 * fw + 1e-6)
            lower_face = ((yy > fy0 + 0.20 * fh) & (yy < fy1 + 0.15 * fh) & (face_x < 0.80)).astype(np.float32)
            hair_mask = (hair_mask * (1.0 - 0.78 * lower_face)).clip(0.0, 1.0)
    base_mask = semantic_fg if semantic_fg is not None else fg_mask
    if base_mask is None:
        base_mask = fg_mask
    uncertain = _build_uncertain_mask(base_mask, [skin_mask, cloth_mask, hair_mask])
    if uncertain is not None and uncertain.sum() > 0:
        soften = 1.0 - 0.18 * uncertain
        if skin_mask is not None:
            skin_mask = (skin_mask * soften).clip(0.0, 1.0)
        if cloth_mask is not None:
            cloth_mask = (cloth_mask * soften).clip(0.0, 1.0)
        if hair_mask is not None:
            hair_mask = (hair_mask * soften).clip(0.0, 1.0)
    return {
        'foreground': fg_mask,
        'valid': base_mask,
        'skin': skin_mask,
        'cloth': cloth_mask,
        'hair': hair_mask,
        'face': face_mask,
        'uncertain': uncertain,
    }


def _preview_tensor(image, masks, parser_mask=None, direct_parser_mode=False, raw_parser_masks=None, min_area=24):
    rgb = torch.from_numpy(image).permute(2, 0, 1).float()
    if direct_parser_mode and parser_mask is not None:
        return _direct_parser_preview_tensor(image, masks.get('foreground'), parser_mask, raw_parser_masks or {}, min_area=min_area)
    panels = [_label_panel_tensor(rgb, 'render')]
    for name in ('skin', 'hair', 'upper', 'lower', 'shoes', 'accessory'):
        mask = masks.get(name)
        if mask is None:
            panel = torch.zeros(3, *rgb.shape[1:], dtype=rgb.dtype)
        else:
            panel = torch.from_numpy(mask).float().unsqueeze(0).repeat(3, 1, 1)
        panels.append(_label_panel_tensor(panel, name))
    return torch.cat(panels, dim=2)


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(config):
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    dataset_cfg = config.dataset
    if dataset_cfg.name != 'zjumocap':
        raise ValueError('This standalone exporter currently supports zjumocap only.')
    if config.mode != 'test':
        raise ValueError('Use mode=test for this exporter.')

    root_dir = dataset_cfg.root_dir
    if not os.path.isabs(root_dir):
        root_dir = os.path.join('/remote-home/ming/3dgs-avatar-release-main', root_dir)
    subject = dataset_cfg.subject
    subject_dir = os.path.join(root_dir, subject)
    with open(os.path.join(subject_dir, 'cam_params.json'), 'r') as f:
        cameras = json.load(f)

    test_mode = dataset_cfg.test_mode
    cam_names = dataset_cfg.test_views[test_mode]
    if len(cam_names) == 0:
        cam_names = cameras['all_cam_names']
    start_frame, end_frame, step = dataset_cfg.test_frames[test_mode]
    model_files = sorted(glob.glob(os.path.join(subject_dir, 'models', '*.npz')))
    if end_frame == 0:
        end_frame = len(model_files)
    frame_ids = list(range(len(model_files)))[start_frame:end_frame:step]
    out_h, out_w = dataset_cfg.img_hw

    config.suffix = f'{config.mode}-{test_mode}'
    asset_root = os.path.join(config.exp_dir, config.suffix, config.get('semantic_editable_assets_dirname', 'semantic_editable_assets'))
    mask_root = os.path.join(asset_root, 'masks')
    motion_root = os.path.join(asset_root, 'motions')
    source_root = os.path.join(asset_root, 'source_rgb')
    preview_root = os.path.join(asset_root, 'preview')
    parser_preview_root = os.path.join(asset_root, 'parser_preview')
    grouped_preview_root = os.path.join(asset_root, 'grouped_preview')
    raw_mask_root = os.path.join(asset_root, 'raw_masks')
    os.makedirs(mask_root, exist_ok=True)
    os.makedirs(motion_root, exist_ok=True)
    os.makedirs(source_root, exist_ok=True)
    os.makedirs(preview_root, exist_ok=True)
    os.makedirs(parser_preview_root, exist_ok=True)
    os.makedirs(grouped_preview_root, exist_ok=True)
    os.makedirs(raw_mask_root, exist_ok=True)

    prior_root = dataset_cfg.parsing_prior.root_dir if dataset_cfg.parsing_prior.enable else ''
    cihp_root = config.get('binding_map_cihp_root', '/remote-home/ming/dataSet')
    preview_min_area = int(config.get('semantic_editable_preview_min_area', 24))

    view_records = []
    motion_records = []
    pbar = tqdm(total=len(cam_names) * len(frame_ids), desc='Export semantic editable assets')
    for cam_name in cam_names:
        camera = cameras[str(cam_name)]
        cam_dir = os.path.join(subject_dir, str(cam_name))
        for frame_idx in frame_ids:
            image_path = os.path.join(cam_dir, f'{frame_idx:06d}.jpg')
            mask_path = os.path.join(cam_dir, f'{frame_idx:06d}.png')
            image, fg_mask, K, dist = _undistort_resize_image(image_path, mask_path, camera, (out_h, out_w), white_bg=dataset_cfg.white_background)
            body_prior_path = os.path.join(prior_root, subject, str(cam_name), 'body', f'{frame_idx:06d}.png') if prior_root else None
            cloth_prior_path = os.path.join(prior_root, subject, str(cam_name), 'cloth', f'{frame_idx:06d}.png') if prior_root else None
            valid_prior_path = os.path.join(prior_root, subject, str(cam_name), 'valid', f'{frame_idx:06d}.png') if prior_root else None
            skin_mask = _load_prior_mask(body_prior_path, K, dist, (out_h, out_w))
            cloth_mask = _load_prior_mask(cloth_prior_path, K, dist, (out_h, out_w))
            valid_mask = _load_prior_mask(valid_prior_path, K, dist, (out_h, out_w))
            parser_root = config.get('semantic_editable_parser_root', '') or cihp_root
            parser_layout = config.get('semantic_editable_parser_layout', 'cihp_subject')
            parser_mask = _load_parser_index_mask(subject, cam_name, frame_idx, K, dist, (out_h, out_w), parser_root, parser_layout)
            hair_mask = _mask_from_index(parser_mask, PARSER_LABEL_GROUPS['hair']) if parser_mask is not None else _load_cihp_mask(subject, cam_name, frame_idx, labels=(2,), out_hw=(out_h, out_w), cihp_root=cihp_root)
            face_mask = _mask_from_index(parser_mask, PARSER_LABEL_GROUPS['face']) if parser_mask is not None else _load_cihp_mask(subject, cam_name, frame_idx, labels=(13,), out_hw=(out_h, out_w), cihp_root=cihp_root)
            direct_parser_mode = bool(config.get('semantic_editable_direct_parser_mode', False))
            raw_parser_masks = {}
            if direct_parser_mode and parser_mask is not None:
                masks = _compose_masks_direct_from_parser(fg_mask, parser_mask, valid_mask)
                raw_parser_masks = _compose_raw_parser_class_masks(parser_mask, fg_mask)
            else:
                masks = _compose_masks(image, fg_mask, skin_mask, cloth_mask, hair_mask, face_mask, valid_mask)
                masks.update(_compose_fine_masks(fg_mask, parser_mask, masks))

            image_name = f'c{int(cam_name):02d}_f{frame_idx:06d}'
            rgb_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            _save_image(rgb_tensor, os.path.join(source_root, f'render_{image_name}.png'))
            _save_image(
                _preview_tensor(
                    image,
                    masks,
                    parser_mask=parser_mask,
                    direct_parser_mode=direct_parser_mode,
                    raw_parser_masks=raw_parser_masks,
                    min_area=preview_min_area,
                ),
                os.path.join(preview_root, f'render_{image_name}.png'),
            )
            if direct_parser_mode and raw_parser_masks:
                grouped_masks = _build_grouped_parser_masks(raw_parser_masks, min_area=preview_min_area)
                _save_image(
                    _grouped_parser_preview_tensor(image, fg_mask, grouped_masks),
                    os.path.join(grouped_preview_root, f'render_{image_name}.png'),
                )
            if parser_mask is not None:
                _save_image(_parser_rgb_tensor(parser_mask, fg_mask), os.path.join(parser_preview_root, f'render_{image_name}.png'))

            mask_files = {}
            for name, mask in masks.items():
                if mask is None:
                    continue
                mask_t = torch.from_numpy(mask).float().unsqueeze(0)
                out_path = os.path.join(mask_root, name, f'render_{image_name}.png')
                _save_image(mask_t, out_path)
                mask_files[name] = os.path.relpath(out_path, asset_root)
            raw_mask_files = {}
            for name, mask in raw_parser_masks.items():
                mask_t = torch.from_numpy(mask).float().unsqueeze(0)
                out_path = os.path.join(raw_mask_root, name, f'render_{image_name}.png')
                _save_image(mask_t, out_path)
                raw_mask_files[name] = os.path.relpath(out_path, asset_root)

            model = np.load(model_files[frame_idx])
            motion = {
                'frame_id': int(frame_idx),
                'cam_id': int(cam_name),
                'trans': model['trans'].astype(np.float32),
                'root_orient': model['root_orient'].astype(np.float32),
                'pose_body': model['pose_body'].astype(np.float32),
                'pose_hand': model['pose_hand'].astype(np.float32),
                'bone_transforms': model['bone_transforms'].astype(np.float32),
            }
            motion_path = os.path.join(motion_root, f'{image_name}.npz')
            np.savez_compressed(motion_path, **motion)

            appearance = {
                region: _masked_region_stats(image, masks.get(region), fg_mask=masks.get('foreground'))
                for region in ALL_REGION_NAMES
            }
            view_records.append({
                'image_name': image_name,
                'frame_id': int(frame_idx),
                'cam_id': int(cam_name),
                'motion_file': os.path.relpath(motion_path, asset_root),
                'mask_files': mask_files,
                'raw_mask_files': raw_mask_files,
                'appearance': appearance,
            })
            motion_records.append(motion)
            pbar.update(1)
    pbar.close()

    with open(os.path.join(asset_root, 'view_records.json'), 'w') as f:
        json.dump(view_records, f, indent=2)

    np.savez_compressed(
        os.path.join(asset_root, 'motion_bank.npz'),
        image_names=np.asarray([record['image_name'] for record in view_records]),
        frame_ids=np.asarray([record['frame_id'] for record in view_records], dtype=np.int32),
        cam_ids=np.asarray([record['cam_id'] for record in view_records], dtype=np.int32),
        trans=np.stack([record['trans'] for record in motion_records], axis=0),
        root_orient=np.stack([record['root_orient'] for record in motion_records], axis=0),
        pose_body=np.stack([record['pose_body'] for record in motion_records], axis=0),
        pose_hand=np.stack([record['pose_hand'] for record in motion_records], axis=0),
        bone_transforms=np.stack([record['bone_transforms'] for record in motion_records], axis=0),
    )

    appearance_bank = {
        'subject': subject,
        'dataset_name': config.get('dataset_name', ''),
        'split': config.suffix,
        'num_views': len(view_records),
        'regions': {
            region_name: _aggregate_region_appearance(view_records, region_name)
            for region_name in ALL_REGION_NAMES
        },
    }
    with open(os.path.join(asset_root, 'appearance_bank.json'), 'w') as f:
        json.dump(appearance_bank, f, indent=2)

    meta = {
        'subject': subject,
        'dataset_name': config.get('dataset_name', ''),
        'split': config.suffix,
        'num_views': len(view_records),
        'asset_root': asset_root,
        'coarse_regions': list(COARSE_REGION_NAMES),
        'fine_regions': list(FINE_REGION_NAMES),
        'all_regions': list(ALL_REGION_NAMES),
        'raw_parser_regions': list(RAW_CIHP_EXPORT_NAMES),
    }
    with open(os.path.join(asset_root, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'Semantic-editable assets exported to: {asset_root}')


if __name__ == '__main__':
    main()
