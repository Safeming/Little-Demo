#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
import re
import math
import shutil
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
import imageio.v2 as imageio
import numpy as np
import torch
import torchvision
import wandb
from omegaconf import OmegaConf
from tqdm import trange

from gaussian_renderer import render, rasterize_gaussians
from scene import GaussianModel, Scene
from utils.general_utils import Evaluator, PSEvaluator, fix_random
from utils.pytorch3d_compat import ops
from tools.export_semantic_editable_assets import (
    CIHP_PALETTE,
    _build_grouped_parser_masks,
    _compose_raw_parser_class_masks,
    _direct_parser_preview_tensor,
    _grouped_parser_preview_tensor,
    _label_panel_tensor,
    _mask_color_tensor,
    _panel_grid_tensors,
)


def _configure_torch_threads_from_env():
    raw = os.environ.get('TORCH_NUM_THREADS') or os.environ.get('OMP_NUM_THREADS')
    if not raw:
        return
    try:
        threads = max(1, int(raw))
    except ValueError:
        return
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(threads, 4)))
    except RuntimeError:
        pass


_configure_torch_threads_from_env()
try:
    cv2.setNumThreads(max(1, int(os.environ.get('OPENCV_FOR_THREADS_NUM', os.environ.get('OMP_NUM_THREADS', '1')))))
except Exception:
    pass


LAYER_COLORS = torch.tensor([
    [0.96, 0.28, 0.22],
    [0.98, 0.83, 0.25],
    [0.22, 0.55, 0.96],
], dtype=torch.float32)
REGION_COLORS = torch.tensor([
    [0.96, 0.36, 0.20],
    [0.42, 0.86, 0.34],
    [0.30, 0.62, 0.98],
], dtype=torch.float32)
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
COMPACT_SEMANTIC_NAMES = ('hair', 'face', 'skin', 'upper', 'lower', 'shoes')
COMPACT_SEMANTIC_LABEL_IDS = {
    'hair': 2,
    'face': 13,
    'skin': 14,
    'upper': 5,
    'lower': 9,
    'shoes': 18,
}
COMPACT_SEMANTIC_COLORS = torch.from_numpy(
    (CIHP_PALETTE[[2, 13, 14, 5, 9, 18]].astype(np.float32) / 255.0)
).float()


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


def _snapshot_hydra_run(config, output_dir):
    hydra_dir = os.path.join(output_dir, '.hydra')
    os.makedirs(hydra_dir, exist_ok=True)

    runtime_dir = None
    try:
        runtime_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        runtime_dir = None

    if runtime_dir:
        src_hydra_dir = os.path.join(runtime_dir, '.hydra')
        for name in ('config.yaml', 'hydra.yaml', 'overrides.yaml'):
            src = os.path.join(src_hydra_dir, name)
            dst = os.path.join(hydra_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, dst)

    if not os.path.exists(os.path.join(hydra_dir, 'config.yaml')):
        OmegaConf.save(config=config, f=os.path.join(hydra_dir, 'config.yaml'), resolve=False)

    hydra_cfg = None
    try:
        hydra_cfg = HydraConfig.get()
    except Exception:
        hydra_cfg = None

    if hydra_cfg is not None and not os.path.exists(os.path.join(hydra_dir, 'hydra.yaml')):
        OmegaConf.save(config=hydra_cfg, f=os.path.join(hydra_dir, 'hydra.yaml'), resolve=False)

    overrides_path = os.path.join(hydra_dir, 'overrides.yaml')
    if not os.path.exists(overrides_path):
        overrides = []
        if hydra_cfg is not None:
            overrides = list(getattr(hydra_cfg.overrides, 'task', []) or [])
        with open(overrides_path, 'w') as handle:
            for override in overrides:
                handle.write(f'- {override}\n')


def _semantic_asset_root(config):
    return os.path.join(
        config.exp_dir,
        config.suffix,
        config.get('semantic_editable_assets_dirname', 'semantic_editable_assets'),
    )


def _prepare_mask_for_asset(mask, ref_image):
    if mask is None:
        return None
    if mask.dim() == 4:
        mask_2d = mask[0, 0]
    elif mask.dim() == 3:
        mask_2d = mask[0]
    else:
        mask_2d = mask
    mask_2d = mask_2d.to(device=ref_image.device, dtype=ref_image.dtype).clamp(0.0, 1.0)
    if mask_2d.shape != ref_image.shape[-2:]:
        mask_2d = torch.nn.functional.interpolate(
            mask_2d.unsqueeze(0).unsqueeze(0),
            size=ref_image.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )[0, 0]
    return mask_2d.clamp(0.0, 1.0)


def _save_mask_png(mask, path):
    if mask is None:
        return
    torchvision.utils.save_image(mask.unsqueeze(0), path)


def _save_tensor_png(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(tensor.clamp(0.0, 1.0), path)


def _torch_image_to_numpy(image):
    return image.detach().permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0).astype(np.float32)


def _torch_mask_to_numpy(mask, ref_image=None):
    if mask is None:
        return None
    if ref_image is not None:
        mask = _prepare_mask_for_asset(mask, ref_image)
    elif mask.dim() == 4:
        mask = mask[0, 0]
    elif mask.dim() == 3:
        mask = mask[0]
    return mask.detach().cpu().numpy().astype(np.float32)


def _parser_rgb_tensor_from_index_mask(parser_mask_np, fg_mask_np=None):
    idx = np.clip(parser_mask_np.astype(np.int32), 0, len(CIHP_PALETTE) - 1)
    rgb = CIHP_PALETTE[idx].astype(np.float32) / 255.0
    if fg_mask_np is not None:
        rgb[fg_mask_np < 0.5] = 0.0
    return torch.from_numpy(rgb).permute(2, 0, 1).float()


def _compact_semantic_palette(names):
    colors = []
    for idx, name in enumerate(names):
        label_idx = COMPACT_SEMANTIC_LABEL_IDS.get(name)
        if label_idx is not None and 0 <= int(label_idx) < len(CIHP_PALETTE):
            color = torch.from_numpy(CIHP_PALETTE[int(label_idx)].astype(np.float32) / 255.0).float()
        else:
            color = COMPACT_SEMANTIC_COLORS[idx % len(COMPACT_SEMANTIC_COLORS)].clone()
        colors.append(color)
    if not colors:
        return COMPACT_SEMANTIC_COLORS.clone()
    return torch.stack(colors, dim=0)


def _get_compact_semantic_probs(pc):
    probs = getattr(pc, 'binding_compact_semantic_probs', None)
    names = tuple(getattr(pc, 'binding_compact_semantic_names', COMPACT_SEMANTIC_NAMES))
    if probs is None:
        compact_ids = getattr(pc, 'binding_compact_semantic_ids', None)
        if compact_ids is None:
            return None, names
        num_classes = max(len(names), int(compact_ids.max().item()) + 1 if compact_ids.numel() > 0 else 0)
        probs = torch.nn.functional.one_hot(compact_ids.long().clamp_min(0), num_classes=num_classes).float()
    return probs.detach(), names


def _rasterize_prob_channels(view, pc, pipe, background, probs):
    if probs is None or probs.numel() == 0:
        return None, None
    rendered_chunks = []
    opacity = None
    num_channels = probs.shape[-1]
    for start in range(0, num_channels, 3):
        chunk = probs[:, start:start + 3]
        if chunk.shape[-1] < 3:
            pad = torch.zeros(chunk.shape[0], 3 - chunk.shape[-1], device=chunk.device, dtype=chunk.dtype)
            chunk = torch.cat([chunk, pad], dim=-1)
        pkg = rasterize_gaussians(
            view,
            pc,
            pipe,
            background,
            colors_precomp=chunk,
            return_opacity=opacity is None,
        )
        rendered = pkg['render'].clamp(0.0, 1.0)
        rendered_chunks.append(rendered[:min(3, num_channels - start)])
        if opacity is None:
            opacity = pkg['opacity_render'].clamp(0.0, 1.0)
    return torch.cat(rendered_chunks, dim=0), opacity


def _view_fg_mask_tensor(view, ref_image):
    mask = _prepare_mask_for_asset(getattr(view, 'original_mask', None), ref_image)
    if mask is None:
        return torch.ones_like(ref_image[0])
    return (mask > 0.5).to(dtype=ref_image.dtype)


def _binding_map_support_mask(view, ref_image, opacity=None, opacity_threshold=0.06, close_kernel=3, erode_kernel=0):
    fg_mask = _view_fg_mask_tensor(view, ref_image).unsqueeze(0)
    support = fg_mask
    if opacity is not None:
        support = support * (opacity[:1] > float(opacity_threshold)).to(dtype=ref_image.dtype)
    support = _binary_close(support.unsqueeze(0), close_kernel)[0]
    if erode_kernel and erode_kernel > 1:
        support = _binary_erode(support.unsqueeze(0), erode_kernel)[0]
    return support.clamp(0.0, 1.0)


def _weighted_box_blur(image, weight, kernel_size=5):
    kernel_size = max(int(kernel_size), 1)
    if kernel_size <= 1:
        return image
    pad = kernel_size // 2
    weighted = image * weight
    numer = torch.nn.functional.avg_pool2d(weighted.unsqueeze(0), kernel_size, stride=1, padding=pad)[0]
    denom = torch.nn.functional.avg_pool2d(weight.unsqueeze(0), kernel_size, stride=1, padding=pad)[0].clamp_min(1e-4)
    return numer / denom


def _prepare_render_export_image(config, view, rendering, opacity=None):
    image = rendering.clamp(0.0, 1.0)
    if opacity is None or not bool(config.get('render_export_refine', True)):
        return image

    opacity = opacity[:1].to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)
    support = _binding_map_support_mask(
        view,
        image,
        opacity=opacity,
        opacity_threshold=float(config.get('render_export_opacity_threshold', 0.06)),
        close_kernel=int(config.get('render_export_close_kernel', 3)),
        erode_kernel=int(config.get('render_export_erode_kernel', 0)),
    )
    fg_mask = _view_fg_mask_tensor(view, image).unsqueeze(0)
    fill_support = _binary_close(
        torch.maximum(fg_mask, support).unsqueeze(0),
        int(config.get('render_export_fill_close_kernel', 5)),
    )[0].clamp(0.0, 1.0)

    power = float(config.get('render_export_opacity_power', 0.85))
    floor = float(config.get('render_export_opacity_floor', 0.30))
    floor = max(min(floor, 1.0), 1e-3)
    power = max(power, 0.0)

    compensated = image * opacity.clamp_min(floor).pow(-power)
    compensated = compensated.clamp(0.0, 1.0)

    blur_kernel = int(config.get('render_export_fill_blur_kernel', 7))
    blurred = _weighted_box_blur(compensated, fill_support, kernel_size=blur_kernel).clamp(0.0, 1.0)
    fill_threshold = float(config.get('render_export_fill_opacity_threshold', 0.55))
    fill_low = float(config.get('render_export_fill_opacity_low', 0.18))
    hole_alpha = ((fill_threshold - opacity) / max(fill_threshold - fill_low, 1e-6)).clamp(0.0, 1.0)
    hole_alpha = hole_alpha * fill_support
    hole_alpha = hole_alpha * (1.0 - (opacity > float(config.get('render_export_keep_core_threshold', 0.72))).to(dtype=image.dtype))
    hole_alpha = _binary_dilate(hole_alpha.unsqueeze(0), int(config.get('render_export_fill_dilate_kernel', 3)))[0].clamp(0.0, 1.0)

    refined = compensated * (1.0 - hole_alpha) + blurred * hole_alpha
    refined = refined.clamp(0.0, 1.0)

    if config.dataset.white_background:
        bg = torch.ones_like(image)
    else:
        bg = torch.zeros_like(image)
    return (refined * fill_support + bg * (1.0 - fill_support)).clamp(0.0, 1.0)


def _compact_hard_assignment(compact_probs_2d, compact_names, fg_mask=None, opacity=None, opacity_threshold=0.05, confidence_threshold=0.0):
    if compact_probs_2d is None or compact_probs_2d.numel() == 0:
        return {}, None, None

    device = compact_probs_2d.device
    dtype = compact_probs_2d.dtype
    h, w = compact_probs_2d.shape[-2:]
    support = torch.ones(h, w, device=device, dtype=torch.bool)

    if fg_mask is not None:
        if fg_mask.dim() == 3:
            fg_mask = fg_mask[0]
        support = support & (fg_mask.to(device=device) > 0.5)

    if opacity is not None:
        if opacity.dim() == 3:
            opacity_2d = opacity[0]
        elif opacity.dim() == 2:
            opacity_2d = opacity
        else:
            opacity_2d = opacity.squeeze()
        support = support & (opacity_2d.to(device=device) > float(opacity_threshold))

    confidence, class_ids = compact_probs_2d.max(dim=0)
    valid = support & (confidence > float(confidence_threshold))

    hard_masks = {}
    for class_idx, class_name in enumerate(compact_names):
        hard_masks[class_name] = ((class_ids == class_idx) & valid).to(dtype=dtype).unsqueeze(0)
    return hard_masks, class_ids, valid.to(dtype=dtype)


def _compact_semantic_map_tensor(compact_names, class_ids_2d, valid_mask_2d=None):
    palette = _compact_semantic_palette(compact_names).to(device=class_ids_2d.device)
    colors = palette[class_ids_2d.long()].permute(2, 0, 1).to(dtype=palette.dtype)
    if valid_mask_2d is not None:
        if valid_mask_2d.dim() == 3:
            valid_mask_2d = valid_mask_2d[0]
        colors = colors * valid_mask_2d.to(device=colors.device, dtype=colors.dtype).unsqueeze(0)
    return colors.clamp(0.0, 1.0)


def _compact_head_preview_tensor(image_np, compact_names, hard_masks, compact_map_tensor):
    rgb = torch.from_numpy(image_np).permute(2, 0, 1).float()
    panels = [
        _label_panel_tensor(rgb, 'source_rgb'),
        _label_panel_tensor(compact_map_tensor, 'compact_map'),
    ]
    for class_name in compact_names:
        mask_t = hard_masks.get(class_name)
        if mask_t is None or mask_t.sum().item() < 8.0:
            continue
        mask_np = mask_t[0].detach().cpu().numpy().astype('float32')
        label_idx = COMPACT_SEMANTIC_LABEL_IDS.get(class_name, 2)
        panel = _mask_color_tensor(mask_np, label_idx)
        if panel is None:
            continue
        panels.append(_label_panel_tensor(panel, class_name))
    return _panel_grid_tensors(panels, cols=4)


def _semantic_editable_use_direct_parser(config):
    if bool(config.get('semantic_editable_direct_parser_mode', False)):
        return True
    dataset_cfg = config.get('dataset', None)
    if dataset_cfg is None:
        return False
    parsing_prior = dataset_cfg.get('parsing_prior', None)
    if parsing_prior is None:
        return False
    return bool(parsing_prior.get('use_direct_parser_labels', False))


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
    if image is None or mask is None:
        return stats

    image = image[:3]
    if mask.dim() == 4:
        mask_2d = mask[0, 0]
    elif mask.dim() == 3:
        mask_2d = mask[0]
    else:
        mask_2d = mask
    mask_2d = mask_2d > 0.5
    if fg_mask is not None:
        if fg_mask.dim() == 4:
            fg_mask = fg_mask[0, 0]
        elif fg_mask.dim() == 3:
            fg_mask = fg_mask[0]
        mask_2d = mask_2d & (fg_mask > 0.5)
        fg_count = int((fg_mask > 0.5).sum().item())
    else:
        fg_count = int(mask_2d.numel())

    count = int(mask_2d.sum().item())
    stats['count'] = count
    stats['coverage'] = float(count / max(fg_count, 1))
    if count == 0:
        return stats

    pixels = image[:, mask_2d].transpose(0, 1)
    rgb_mean = pixels.mean(dim=0)
    rgb_std = pixels.std(dim=0, unbiased=False)
    brightness = pixels.mean(dim=-1)
    saturation = pixels.max(dim=-1).values - pixels.min(dim=-1).values
    warmth = 1.10 * pixels[:, 0] + 0.70 * pixels[:, 1] - 1.30 * pixels[:, 2]

    stats.update({
        'rgb_mean': _to_python(rgb_mean),
        'rgb_std': _to_python(rgb_std),
        'brightness_mean': float(brightness.mean().item()),
        'brightness_std': float(brightness.std(unbiased=False).item()),
        'saturation_mean': float(saturation.mean().item()),
        'saturation_std': float(saturation.std(unbiased=False).item()),
        'warmth_mean': float(warmth.mean().item()),
        'warmth_std': float(warmth.std(unbiased=False).item()),
    })
    return stats


def _compose_semantic_masks(view, render_pkg, config):
    image = view.original_image[:3]
    fg_mask = _prepare_mask_for_asset(getattr(view, 'original_mask', None), image)
    valid_mask = _prepare_mask_for_asset(getattr(view, 'parsing_valid_mask', None), image)
    skin_mask = _prepare_mask_for_asset(getattr(view, 'parsing_body_mask', None), image)
    cloth_mask = _prepare_mask_for_asset(getattr(view, 'parsing_cloth_mask', None), image)
    hair_mask = _prepare_mask_for_asset(_load_cihp_hair_mask(view, image, config, labels=(2,)), image)
    face_mask = _prepare_mask_for_asset(_load_cihp_hair_mask(view, image, config, labels=(13,)), image)

    semantic_fg = valid_mask if valid_mask is not None and valid_mask.sum().item() > 0 else fg_mask
    if face_mask is not None:
        skin_mask = face_mask if skin_mask is None else torch.maximum(skin_mask, face_mask)
    if semantic_fg is not None:
        if skin_mask is not None:
            skin_mask = (skin_mask * semantic_fg).clamp(0.0, 1.0)
        if cloth_mask is not None:
            cloth_mask = (cloth_mask * semantic_fg).clamp(0.0, 1.0)
        if hair_mask is not None:
            hair_mask = (hair_mask * semantic_fg).clamp(0.0, 1.0)
        if face_mask is not None:
            face_mask = (face_mask * semantic_fg).clamp(0.0, 1.0)
    if hair_mask is not None and face_mask is not None:
        hair_mask = (hair_mask * (1.0 - face_mask)).clamp(0.0, 1.0)

    base_mask = semantic_fg if semantic_fg is not None else fg_mask
    if base_mask is None:
        base_mask = torch.ones_like(image[0])

    occupied = torch.zeros_like(base_mask)
    for candidate in (skin_mask, cloth_mask, hair_mask):
        if candidate is not None:
            occupied = torch.maximum(occupied, candidate)
    uncertain_mask = (base_mask - occupied).clamp(0.0, 1.0)

    return {
        'foreground': fg_mask,
        'valid': base_mask,
        'skin': skin_mask,
        'cloth': cloth_mask,
        'hair': hair_mask,
        'face': face_mask,
        'uncertain': uncertain_mask,
    }


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

    for record in records:
        stats = record['appearance'][region_name]
        count = float(stats['count'])
        if count <= 0:
            continue
        total_count += count
        coverage_sum += float(stats['coverage'])
        rgb_mean = np.asarray(stats['rgb_mean'], dtype=np.float64)
        rgb_std = np.asarray(stats['rgb_std'], dtype=np.float64)
        rgb_first += count * rgb_mean
        rgb_second += count * (np.square(rgb_std) + np.square(rgb_mean))
        brightness_mean = float(stats['brightness_mean'])
        brightness_std = float(stats['brightness_std'])
        brightness_first += count * brightness_mean
        brightness_second += count * (brightness_std ** 2 + brightness_mean ** 2)
        saturation_mean = float(stats['saturation_mean'])
        saturation_std = float(stats['saturation_std'])
        saturation_first += count * saturation_mean
        saturation_second += count * (saturation_std ** 2 + saturation_mean ** 2)
        warmth_mean = float(stats['warmth_mean'])
        warmth_std = float(stats['warmth_std'])
        warmth_first += count * warmth_mean
        warmth_second += count * (warmth_std ** 2 + warmth_mean ** 2)

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

    contributing = max(sum(1 for record in records if record['appearance'][region_name]['count'] > 0), 1)
    return {
        'count': int(total_count),
        'coverage_mean': float(coverage_sum / contributing),
        'rgb_mean': rgb_mean.tolist(),
        'rgb_std': np.sqrt(rgb_var).tolist(),
        'brightness_mean': float(brightness_mean),
        'brightness_std': float(np.sqrt(brightness_var)),
        'saturation_mean': float(saturation_mean),
        'saturation_std': float(np.sqrt(saturation_var)),
        'warmth_mean': float(warmth_mean),
        'warmth_std': float(np.sqrt(warmth_var)),
    }


def _aggregate_binding_summary(records):
    aggregated = {}
    counts = {}
    for record in records:
        binding = record.get('binding', {})
        if not binding:
            continue
        for map_name, stats in binding.items():
            if map_name == 'image_name' or not isinstance(stats, dict):
                continue
            target = aggregated.setdefault(map_name, {})
            target_counts = counts.setdefault(map_name, {})
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    target[key] = target.get(key, 0.0) + float(value)
                    target_counts[key] = target_counts.get(key, 0) + 1
    for map_name, stats in aggregated.items():
        for key, value in list(stats.items()):
            stats[key] = value / max(counts[map_name].get(key, 1), 1)
    return aggregated


def _finalize_semantic_editable_assets(config, asset_records):
    if not asset_records:
        return

    asset_root = _semantic_asset_root(config)
    os.makedirs(asset_root, exist_ok=True)
    views_json = [record['json'] for record in asset_records]
    with open(os.path.join(asset_root, 'view_records.json'), 'w') as f:
        json.dump(views_json, f, indent=2)

    rots = np.stack([record['motion']['rots'] for record in asset_records], axis=0)
    jtrs = np.stack([record['motion']['Jtrs'] for record in asset_records], axis=0)
    bone_transforms = np.stack([record['motion']['bone_transforms'] for record in asset_records], axis=0)
    frame_ids = np.asarray([record['motion']['frame_id'] for record in asset_records], dtype=np.int32)
    cam_ids = np.asarray([record['motion']['cam_id'] for record in asset_records], dtype=np.int32)
    image_names = np.asarray([record['json']['image_name'] for record in asset_records])
    np.savez_compressed(
        os.path.join(asset_root, 'motion_bank.npz'),
        image_names=image_names,
        frame_ids=frame_ids,
        cam_ids=cam_ids,
        rots=rots,
        Jtrs=jtrs,
        bone_transforms=bone_transforms,
    )

    appearance_bank = {
        'subject': config.dataset.get('subject', ''),
        'dataset_name': config.get('dataset_name', ''),
        'split': config.suffix,
        'num_views': len(asset_records),
        'regions': {
            region_name: _aggregate_region_appearance(views_json, region_name)
            for region_name in ('skin', 'hair', 'cloth', 'face', 'uncertain')
        },
        'binding_summary_mean': _aggregate_binding_summary(views_json),
    }
    with open(os.path.join(asset_root, 'appearance_bank.json'), 'w') as f:
        json.dump(appearance_bank, f, indent=2)

    mask_modes = {record['json'].get('mask_export_mode', 'coarse_regions') for record in asset_records}
    subject_meta = {
        'subject': config.dataset.get('subject', ''),
        'dataset_name': config.get('dataset_name', ''),
        'split': config.suffix,
        'num_views': len(asset_records),
        'frame_range': [int(frame_ids.min()), int(frame_ids.max())] if frame_ids.size > 0 else [-1, -1],
        'cam_ids': sorted({int(v) for v in cam_ids.tolist()}),
        'asset_root': asset_root,
        'mask_export_mode': 'fine_parser' if 'fine_parser' in mask_modes else 'coarse_regions',
    }
    with open(os.path.join(asset_root, 'meta.json'), 'w') as f:
        json.dump(subject_meta, f, indent=2)


def _export_semantic_editable_assets(config, view, render_pkg, binding_record, asset_records):
    if not config.get('export_semantic_editable_assets', False):
        return

    asset_root = _semantic_asset_root(config)
    mask_root = os.path.join(asset_root, 'masks')
    compact_mask_root = os.path.join(asset_root, 'compact_head_masks')
    coarse_mask_root = os.path.join(asset_root, 'coarse_masks')
    raw_mask_root = os.path.join(asset_root, 'raw_masks')
    motion_root = os.path.join(asset_root, 'motions')
    source_root = os.path.join(asset_root, 'source_rgb')
    preview_root = os.path.join(asset_root, 'preview')
    compact_preview_root = os.path.join(asset_root, 'compact_head_preview')
    grouped_preview_root = os.path.join(asset_root, 'grouped_preview')
    parser_preview_root = os.path.join(asset_root, 'parser_preview')
    os.makedirs(mask_root, exist_ok=True)
    os.makedirs(compact_mask_root, exist_ok=True)
    os.makedirs(coarse_mask_root, exist_ok=True)
    os.makedirs(raw_mask_root, exist_ok=True)
    os.makedirs(motion_root, exist_ok=True)
    os.makedirs(source_root, exist_ok=True)
    os.makedirs(preview_root, exist_ok=True)
    os.makedirs(compact_preview_root, exist_ok=True)
    os.makedirs(grouped_preview_root, exist_ok=True)
    os.makedirs(parser_preview_root, exist_ok=True)

    image = view.original_image[:3]
    image_np = _torch_image_to_numpy(image)
    masks = _compose_semantic_masks(view, render_pkg, config)
    parser_mask_2d = _prepare_view_parser_mask(view, image)
    direct_parser_mode = _semantic_editable_use_direct_parser(config) and parser_mask_2d is not None
    preview_min_area = int(config.get('semantic_editable_preview_min_area', 24))

    fg_mask_np = _torch_mask_to_numpy(masks.get('foreground'), ref_image=image)
    valid_mask_np = _torch_mask_to_numpy(masks.get('valid'), ref_image=image)
    if fg_mask_np is None:
        fg_mask_np = valid_mask_np if valid_mask_np is not None else np.ones(image_np.shape[:2], dtype=np.float32)

    fine_mask_paths = {}
    coarse_mask_paths = {}
    raw_mask_paths = {}
    compact_mask_paths = {}
    preview_file = None
    compact_preview_file = None
    grouped_preview_file = None
    parser_preview_file = None

    if config.get('semantic_editable_save_view_masks', True):
        _save_tensor_png(image, os.path.join(source_root, f'render_{view.image_name}.png'))

        for name, mask in masks.items():
            if mask is None:
                continue
            out_dir = os.path.join(coarse_mask_root, name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'render_{view.image_name}.png')
            _save_mask_png(mask, out_path)
            coarse_mask_paths[name] = os.path.relpath(out_path, asset_root)

        if parser_mask_2d is not None:
            parser_mask_np = parser_mask_2d.detach().cpu().numpy().astype(np.int32)
            raw_parser_masks = _compose_raw_parser_class_masks(parser_mask_np, fg_mask_np)
            if direct_parser_mode:
                for label_name, raw_mask_np in raw_parser_masks.items():
                    if float((raw_mask_np > 0.5).sum()) <= 0:
                        continue
                    mask_tensor = torch.from_numpy(raw_mask_np).float().unsqueeze(0)
                    main_dir = os.path.join(mask_root, label_name)
                    os.makedirs(main_dir, exist_ok=True)
                    main_path = os.path.join(main_dir, f'render_{view.image_name}.png')
                    _save_mask_png(mask_tensor, main_path)
                    fine_mask_paths[label_name] = os.path.relpath(main_path, asset_root)

                    raw_dir = os.path.join(raw_mask_root, label_name)
                    os.makedirs(raw_dir, exist_ok=True)
                    raw_path = os.path.join(raw_dir, f'render_{view.image_name}.png')
                    _save_mask_png(mask_tensor, raw_path)
                    raw_mask_paths[label_name] = os.path.relpath(raw_path, asset_root)

                preview_path = os.path.join(preview_root, f'render_{view.image_name}.png')
                preview_tensor = _direct_parser_preview_tensor(
                    image_np,
                    fg_mask_np,
                    parser_mask_np,
                    raw_parser_masks,
                    min_area=preview_min_area,
                )
                _save_tensor_png(preview_tensor, preview_path)
                preview_file = os.path.relpath(preview_path, asset_root)

                grouped_masks = _build_grouped_parser_masks(raw_parser_masks, min_area=preview_min_area)
                grouped_preview_path = os.path.join(grouped_preview_root, f'render_{view.image_name}.png')
                grouped_preview_tensor = _grouped_parser_preview_tensor(image_np, fg_mask_np, grouped_masks)
                _save_tensor_png(grouped_preview_tensor, grouped_preview_path)
                grouped_preview_file = os.path.relpath(grouped_preview_path, asset_root)
            else:
                for label_idx, label_name in RAW_CIHP_LABEL_NAMES.items():
                    raw_mask = (parser_mask_2d == float(label_idx)).to(dtype=image.dtype).unsqueeze(0)
                    if raw_mask.sum().item() <= 0:
                        continue
                    out_dir = os.path.join(raw_mask_root, label_name)
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f'render_{view.image_name}.png')
                    _save_mask_png(raw_mask, out_path)
                    raw_mask_paths[label_name] = os.path.relpath(out_path, asset_root)

            parser_preview_path = os.path.join(parser_preview_root, f'render_{view.image_name}.png')
            _save_tensor_png(_parser_rgb_tensor_from_index_mask(parser_mask_np, fg_mask_np), parser_preview_path)
            parser_preview_file = os.path.relpath(parser_preview_path, asset_root)

        if not direct_parser_mode:
            fine_mask_paths = dict(coarse_mask_paths)

        compact_hard_masks = {}
        compact_map_tensor = None
        compact_names = tuple()
        if bool(config.get('semantic_editable_export_compact_head', True)):
            compact_probs, compact_names = _get_compact_semantic_probs(render_pkg['deformed_gaussian'])
            compact_probs_2d = None
            compact_opacity = None
            if compact_probs is not None:
                compact_background = torch.zeros(3, device=image.device, dtype=image.dtype)
                compact_probs_2d, compact_opacity = _rasterize_prob_channels(
                    view,
                    render_pkg['deformed_gaussian'],
                    config.pipeline,
                    compact_background,
                    compact_probs,
                )
            if compact_probs_2d is not None and compact_probs_2d.shape[0] > 0:
                compact_hard_masks, compact_ids_2d, compact_valid_2d = _compact_hard_assignment(
                    compact_probs_2d,
                    compact_names,
                    fg_mask=_view_fg_mask_tensor(view, image),
                    opacity=compact_opacity,
                    opacity_threshold=float(config.get('semantic_editable_compact_opacity_threshold', 0.05)),
                    confidence_threshold=float(config.get('semantic_editable_compact_confidence_threshold', 0.0)),
                )
                compact_map_tensor = _compact_semantic_map_tensor(compact_names, compact_ids_2d, compact_valid_2d)
                for class_name, class_mask in compact_hard_masks.items():
                    if class_mask.sum().item() <= 1e-4:
                        continue
                    out_dir = os.path.join(compact_mask_root, class_name)
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f'render_{view.image_name}.png')
                    _save_mask_png(class_mask, out_path)
                    compact_mask_paths[class_name] = os.path.relpath(out_path, asset_root)

                compact_preview_path = os.path.join(compact_preview_root, f'render_{view.image_name}.png')
                compact_preview_tensor = _compact_head_preview_tensor(
                    image_np,
                    compact_names,
                    compact_hard_masks,
                    compact_map_tensor,
                )
                _save_tensor_png(compact_preview_tensor, compact_preview_path)
                compact_preview_file = os.path.relpath(compact_preview_path, asset_root)

    motion = {
        'frame_id': int(getattr(view, 'frame_id', -1)),
        'cam_id': int(getattr(view, 'cam_id', -1)),
        'rots': view.rots.detach().cpu().numpy().astype(np.float32),
        'Jtrs': view.Jtrs.detach().cpu().numpy().astype(np.float32),
        'bone_transforms': view.bone_transforms.detach().cpu().numpy().astype(np.float32),
    }

    if config.get('semantic_editable_save_motion_npz', True):
        motion_path = os.path.join(motion_root, f'{view.image_name}.npz')
        np.savez_compressed(
            motion_path,
            frame_id=motion['frame_id'],
            cam_id=motion['cam_id'],
            rots=motion['rots'],
            Jtrs=motion['Jtrs'],
            bone_transforms=motion['bone_transforms'],
        )
        motion_file = os.path.relpath(motion_path, asset_root)
    else:
        motion_file = None

    fg_mask = masks.get('valid', None)
    appearance = {
        region_name: _masked_region_stats(image, masks.get(region_name), fg_mask=fg_mask)
        for region_name in ('skin', 'hair', 'cloth', 'face', 'uncertain')
    }
    appearance_compact = {}
    for class_name, class_mask in compact_hard_masks.items():
        appearance_compact[class_name] = _masked_region_stats(image, class_mask, fg_mask=fg_mask)

    record = {
        'image_name': view.image_name,
        'frame_id': motion['frame_id'],
        'cam_id': motion['cam_id'],
        'motion_file': motion_file,
        'mask_export_mode': 'fine_parser' if direct_parser_mode else 'coarse_regions',
        'mask_files': fine_mask_paths,
        'coarse_mask_files': coarse_mask_paths,
        'raw_mask_files': raw_mask_paths,
        'compact_head_mask_files': compact_mask_paths,
        'appearance': appearance,
        'appearance_compact': appearance_compact,
    }
    if preview_file is not None:
        record['preview_file'] = preview_file
    if compact_preview_file is not None:
        record['compact_head_preview_file'] = compact_preview_file
    if grouped_preview_file is not None:
        record['grouped_preview_file'] = grouped_preview_file
    if parser_preview_file is not None:
        record['parser_preview_file'] = parser_preview_file
    if config.get('semantic_editable_include_binding_summary', True) and binding_record is not None:
        record['binding'] = binding_record

    asset_records.append({
        'json': record,
        'motion': motion,
    })

def _heatmap_rgb(values):
    values = values.clamp(0.0, 1.0)
    r = torch.clamp(1.5 * values - 0.5, 0.0, 1.0)
    g = torch.clamp(1.0 - torch.abs(2.0 * values - 1.0), 0.0, 1.0)
    b = torch.clamp(1.5 * (1.0 - values) - 0.5, 0.0, 1.0)
    return torch.stack([r, g, b], dim=-1)


def _stability_rgb(stability):
    stability = stability.clamp(0.0, 1.0)
    red = torch.tensor([0.92, 0.25, 0.18], device=stability.device, dtype=stability.dtype)
    green = torch.tensor([0.18, 0.82, 0.34], device=stability.device, dtype=stability.dtype)
    yellow = torch.tensor([0.95, 0.86, 0.22], device=stability.device, dtype=stability.dtype)
    mid = stability <= 0.5
    colors = torch.empty(stability.shape[0], 3, device=stability.device, dtype=stability.dtype)
    if mid.any():
        t = (stability[mid] / 0.5).unsqueeze(-1)
        colors[mid] = red * (1.0 - t) + yellow * t
    if (~mid).any():
        t = ((stability[~mid] - 0.5) / 0.5).unsqueeze(-1)
        colors[~mid] = yellow * (1.0 - t) + green * t
    return colors


def _thin_rgb(thin_score):
    thin_score = thin_score.clamp(0.0, 1.0)
    low = torch.tensor([0.15, 0.22, 0.75], device=thin_score.device, dtype=thin_score.dtype)
    high = torch.tensor([0.98, 0.18, 0.86], device=thin_score.device, dtype=thin_score.dtype)
    return low * (1.0 - thin_score.unsqueeze(-1)) + high * thin_score.unsqueeze(-1)


def _mask_bbox(mask):
    if mask is None:
        return None
    if mask.dim() == 4:
        mask_2d = mask[0, 0]
    elif mask.dim() == 3:
        mask_2d = mask[0]
    else:
        mask_2d = mask
    if mask_2d.sum().item() < 4:
        return None
    ys, xs = torch.where(mask_2d > 0.05)
    if ys.numel() == 0:
        return None
    return ys.min().float(), ys.max().float(), xs.min().float(), xs.max().float()


def _appearance_guided_torso_refine(image, render_rgb, opacity, map_name, face_mask=None, cloth_mask=None):
    if render_rgb is None or map_name not in {'region', 'body_prob', 'cloth_prob'}:
        return image

    if opacity is not None and opacity.dim() == 3:
        fg = opacity[0] > 0.02
    else:
        fg = render_rgb.mean(dim=0) > 0.02
    if fg.sum().item() < 64:
        return image

    ys, xs = torch.where(fg)
    y0 = ys.min().float()
    y1 = ys.max().float()
    x0 = xs.min().float()
    x1 = xs.max().float()
    h = (y1 - y0).clamp_min(8.0)
    w = (x1 - x0).clamp_min(8.0)

    yy, xx = torch.meshgrid(
        torch.arange(image.shape[1], device=image.device, dtype=image.dtype),
        torch.arange(image.shape[2], device=image.device, dtype=image.dtype),
        indexing='ij',
    )
    y_rel = (yy - y0) / h
    x_rel = (xx - (x0 + x1) * 0.5).abs() / (0.5 * w + 1e-6)

    upper_torso = torch.sigmoid((y_rel - 0.14) / 0.05) * torch.sigmoid((0.52 - y_rel) / 0.07)
    chest_core = upper_torso * torch.sigmoid((0.22 - x_rel) / 0.05)
    strap_band = upper_torso * torch.sigmoid((x_rel - 0.16) / 0.04) * torch.sigmoid((0.46 - x_rel) / 0.05)

    brightness = render_rgb.mean(dim=0)
    saturation = render_rgb.max(dim=0).values - render_rgb.min(dim=0).values
    warmth = 1.10 * render_rgb[0] + 0.70 * render_rgb[1] - 1.30 * render_rgb[2]

    torso_skin = fg.float() * chest_core * torch.sigmoid((brightness - 0.18) / 0.05) * torch.sigmoid((warmth - 0.02) / 0.05)
    neck_band = torch.sigmoid((y_rel - 0.03) / 0.03) * torch.sigmoid((0.26 - y_rel) / 0.05)
    neck_core = neck_band * torch.sigmoid((0.20 - x_rel) / 0.045)
    neck_skin = fg.float() * neck_core * torch.sigmoid((brightness - 0.16) / 0.045) * torch.sigmoid((warmth - 0.00) / 0.045)
    collar_skin = fg.float() * torch.sigmoid((y_rel - 0.16) / 0.03) * torch.sigmoid((0.34 - y_rel) / 0.04)
    collar_skin = collar_skin * torch.sigmoid((0.18 - x_rel) / 0.04)
    collar_skin = collar_skin * torch.sigmoid((brightness - 0.20) / 0.04) * torch.sigmoid((warmth - 0.04) / 0.04)
    skin_hint = torch.clamp(torch.maximum(torch.maximum(torso_skin, 1.15 * neck_skin), 1.05 * collar_skin), 0.0, 1.0)
    cloth_hint = fg.float() * strap_band * torch.sigmoid((0.50 - brightness) / 0.08) * torch.sigmoid((0.24 - saturation) / 0.06)
    cloth_hint = cloth_hint * (1.0 - 0.82 * skin_hint)

    face_bbox = _mask_bbox(face_mask)
    if face_bbox is not None:
        fy0, fy1, fx0, fx1 = face_bbox
        face_h = (fy1 - fy0).clamp_min(6.0)
        face_w = (fx1 - fx0).clamp_min(6.0)
        face_center_x = 0.5 * (fx0 + fx1)
        face_x_rel = (xx - face_center_x).abs() / (0.5 * face_w + 1e-6)
        neck_anchor = fy1 + 0.02 * face_h
        neck_from_face = fg.float() * torch.sigmoid((yy - neck_anchor) / 3.0) * torch.sigmoid(((neck_anchor + 0.22 * face_h) - yy) / 4.5)
        neck_from_face = neck_from_face * torch.sigmoid((0.24 - face_x_rel) / 0.05)
        neck_from_face = neck_from_face * torch.sigmoid((brightness - 0.17) / 0.04) * torch.sigmoid((warmth - 0.01) / 0.04)

        chest_anchor = fy1 + 0.18 * face_h
        chest_span = 0.38 * h
        chest_from_face = fg.float() * torch.sigmoid((yy - chest_anchor) / 5.0) * torch.sigmoid(((chest_anchor + chest_span) - yy) / 7.0)
        chest_from_face = chest_from_face * torch.sigmoid((0.18 - face_x_rel) / 0.04)
        chest_from_face = chest_from_face * torch.sigmoid((brightness - 0.21) / 0.04) * torch.sigmoid((warmth - 0.05) / 0.04)
        if cloth_mask is not None:
            cloth_gate = cloth_mask[0] if cloth_mask.dim() == 3 else cloth_mask[0, 0]
            chest_from_face = chest_from_face * (1.0 - 0.45 * cloth_gate)
        skin_hint = torch.clamp(torch.maximum(torch.maximum(skin_hint, 1.30 * neck_from_face), 1.18 * chest_from_face), 0.0, 1.0)

    if map_name == 'region':
        body_color = REGION_COLORS[0].to(image.device, image.dtype).view(3, 1, 1)
        cloth_color = REGION_COLORS[2].to(image.device, image.dtype).view(3, 1, 1)
        body_alpha = (0.62 * skin_hint).clamp(0.0, 0.74).unsqueeze(0)
        cloth_alpha = (0.78 * cloth_hint).clamp(0.0, 0.82).unsqueeze(0)
        image = image * (1.0 - body_alpha) + body_color * body_alpha
        image = image * (1.0 - cloth_alpha) + cloth_color * cloth_alpha
        return image

    high = image.new_tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
    low = image.new_tensor([0.0, 0.0, 1.0]).view(3, 1, 1)
    if map_name == 'body_prob':
        hi_alpha = (0.64 * skin_hint).clamp(0.0, 0.76).unsqueeze(0)
        lo_alpha = (0.36 * cloth_hint).clamp(0.0, 0.44).unsqueeze(0)
        image = image * (1.0 - hi_alpha) + high * hi_alpha
        image = image * (1.0 - lo_alpha) + low * lo_alpha
        return image

    hi_alpha = (0.66 * cloth_hint).clamp(0.0, 0.78).unsqueeze(0)
    lo_alpha = (0.46 * skin_hint).clamp(0.0, 0.54).unsqueeze(0)
    image = image * (1.0 - hi_alpha) + high * hi_alpha
    image = image * (1.0 - lo_alpha) + low * lo_alpha
    return image



def _appearance_guided_detail_refine(image, render_rgb, opacity, map_name, body_mask=None, cloth_mask=None, hair_mask=None, shoe_mask=None):
    if render_rgb is None or map_name != 'thin':
        return image

    if opacity is not None and opacity.dim() == 3:
        fg = (opacity[0] > 0.02).to(dtype=image.dtype)
    else:
        fg = (render_rgb.mean(dim=0) > 0.02).to(dtype=image.dtype)
    if fg.sum().item() < 64:
        return image

    def _prep(mask):
        if mask is None:
            return torch.zeros_like(fg)
        if mask.dim() == 4:
            mask = mask[0, 0]
        elif mask.dim() == 3:
            mask = mask[0]
        if mask.shape[-2:] != image.shape[-2:]:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=image.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )[0, 0]
        return mask.clamp(0.0, 1.0) * fg

    body = _prep(body_mask)
    cloth = _prep(cloth_mask)
    hair = _prep(hair_mask)
    shoe = _prep(shoe_mask)

    body = body * (1.0 - 0.88 * cloth) * (1.0 - 0.96 * hair)
    cloth = cloth * (1.0 - 0.72 * body) * (1.0 - 0.92 * hair)

    body_core = _binary_erode((body > 0.40).to(dtype=image.dtype).unsqueeze(0).unsqueeze(0), 5)[0, 0] * fg
    cloth_core = _binary_erode((cloth > 0.42).to(dtype=image.dtype).unsqueeze(0).unsqueeze(0), 5)[0, 0] * fg
    body_shell = _binary_dilate((body > 0.18).to(dtype=image.dtype).unsqueeze(0).unsqueeze(0), 3)[0, 0] * fg
    cloth_shell = _binary_dilate((cloth > 0.18).to(dtype=image.dtype).unsqueeze(0).unsqueeze(0), 3)[0, 0] * fg
    body_edge = (body_shell - body_core).clamp(0.0, 1.0)
    cloth_edge = (cloth_shell - cloth_core).clamp(0.0, 1.0)
    boundary = (body_shell * cloth_shell).clamp(0.0, 1.0)
    fg_edge = (_binary_dilate(fg.unsqueeze(0).unsqueeze(0), 3)[0, 0] - _binary_erode(fg.unsqueeze(0).unsqueeze(0), 3)[0, 0]).clamp(0.0, 1.0) * fg

    gray = render_rgb.mean(dim=0)
    blur = torch.nn.functional.avg_pool2d(gray.unsqueeze(0).unsqueeze(0), kernel_size=5, stride=1, padding=2)[0, 0]
    highfreq = (gray - blur).abs()
    dx = torch.zeros_like(gray)
    dy = torch.zeros_like(gray)
    dx[:, 1:] = (gray[:, 1:] - gray[:, :-1]).abs()
    dy[1:, :] = (gray[1:, :] - gray[:-1, :]).abs()
    grad = torch.sqrt(dx * dx + dy * dy + 1e-8)
    sat = render_rgb.max(dim=0).values - render_rgb.min(dim=0).values
    darkness = 1.0 - gray

    edge_score = _sigmoid_score((grad - 0.055) / 0.016)
    highfreq_score = _sigmoid_score((highfreq - 0.035) / 0.012)
    sat_score = _sigmoid_score((sat - 0.10) / 0.035)
    dark_score = _sigmoid_score((darkness - 0.52) / 0.08)

    ys, xs = torch.where(fg > 0.5)
    y0 = ys.min().float()
    y1 = ys.max().float()
    x0 = xs.min().float()
    x1 = xs.max().float()
    h = (y1 - y0).clamp_min(8.0)
    w = (x1 - x0).clamp_min(8.0)
    yy, xx = torch.meshgrid(
        torch.arange(image.shape[1], device=image.device, dtype=image.dtype),
        torch.arange(image.shape[2], device=image.device, dtype=image.dtype),
        indexing='ij',
    )
    y_rel = (yy - y0) / h
    x_rel = (xx - (x0 + x1) * 0.5).abs() / (0.5 * w + 1e-6)

    shoulder_band = cloth * torch.sigmoid((y_rel - 0.10) / 0.03) * torch.sigmoid((0.42 - y_rel) / 0.04)
    lower_cloth_band = cloth * torch.sigmoid((y_rel - 0.38) / 0.05) * torch.sigmoid((0.92 - y_rel) / 0.06)
    cloth_top_band = cloth * torch.sigmoid((y_rel - 0.28) / 0.035) * torch.sigmoid((0.68 - y_rel) / 0.05)
    center_band = torch.sigmoid((0.32 - x_rel) / 0.08)
    dark_line = dark_score * highfreq_score
    narrow_line = edge_score * _sigmoid_score((0.18 - cloth_edge - boundary) / 0.06)

    cloth_detail = cloth * torch.clamp(0.80 * cloth_edge + 0.44 * boundary + 0.22 * fg_edge, 0.0, 1.0)
    cloth_interior_line = torch.clamp(0.74 * cloth_top_band * center_band * dark_line + 0.26 * cloth_top_band * dark_line * narrow_line, 0.0, 1.0)
    strap_edge = shoulder_band * torch.clamp(0.74 * edge_score + 0.54 * highfreq_score + 0.22 * fg_edge, 0.0, 1.0)
    hem_edge = lower_cloth_band * torch.clamp(0.70 * edge_score + 0.44 * cloth_edge + 0.18 * fg_edge, 0.0, 1.0)
    shoe_detail = shoe * torch.clamp(0.74 * fg_edge + 0.42 * edge_score + 0.16 * dark_score, 0.0, 1.0)
    hair_detail = hair * torch.clamp(0.34 * fg_edge + 0.22 * edge_score, 0.0, 1.0)
    accessory_detail = torch.clamp(
        0.88 * edge_score * (0.52 * cloth_detail + 0.58 * shoe_detail + 0.20 * hair_detail)
        + 0.60 * highfreq_score * (0.30 * cloth_edge + 0.24 * shoe + 0.08 * hair)
        + 0.34 * strap_edge
        + 0.28 * hem_edge
        + 0.42 * cloth_interior_line
        + 0.16 * sat_score * cloth_edge
        + 0.20 * dark_score * shoe,
        0.0,
        1.0,
    )
    body_outline = torch.clamp(body_edge * (1.0 - 0.92 * cloth_shell) * (1.0 - 0.96 * shoe) * (1.0 - 0.96 * hair), 0.0, 1.0)
    interior_suppress = torch.clamp(0.96 * body_core + 0.58 * cloth_core * (1.0 - 0.82 * cloth_edge) * (1.0 - 0.80 * boundary), 0.0, 1.0)
    detail = torch.clamp(accessory_detail * (1.0 - 0.88 * interior_suppress), 0.0, 1.0)
    detail = detail * (1.0 - 0.52 * body_outline)
    detail = torch.maximum(detail, 0.28 * boundary * edge_score * torch.maximum(cloth, shoe))
    detail = torch.maximum(detail, 0.28 * strap_edge)
    detail = torch.maximum(detail, 0.34 * cloth_interior_line)
    detail = detail * _sigmoid_score((detail - 0.14) / 0.045)
    detail = detail.pow(1.04) * fg

    low = image.new_tensor([0.15, 0.22, 0.75]).view(3, 1, 1)
    high = image.new_tensor([0.98, 0.18, 0.86]).view(3, 1, 1)
    return (low * (1.0 - detail.unsqueeze(0)) + high * detail.unsqueeze(0)) * fg.unsqueeze(0)


def _bodyprob_guided_refine(image, render_rgb, opacity, map_name, face_mask=None, body_mask=None, cloth_mask=None, hair_mask=None, shoe_mask=None):
    if render_rgb is None or map_name not in {'region', 'layer', 'body_prob', 'cloth_prob'}:
        return image

    if opacity is not None and opacity.dim() == 3:
        fg = opacity[0] > 0.02
    else:
        fg = render_rgb.mean(dim=0) > 0.02
    if fg.sum().item() < 64:
        return image

    fg_float = fg.float()

    def _mask2d(mask):
        if mask is None:
            return None
        if mask.dim() == 4:
            mask = mask[0, 0]
        elif mask.dim() == 3:
            mask = mask[0]
        if mask.shape[-2:] != image.shape[-2:]:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=image.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )[0, 0]
        return mask.clamp(0.0, 1.0) * fg_float

    def _palette_scores(img, palette):
        pixels = img.permute(1, 2, 0).reshape(-1, 3)
        dists = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(dim=-1)
        scores = torch.softmax(-14.0 * dists, dim=-1)
        return scores.reshape(img.shape[1], img.shape[2], palette.shape[0]).permute(2, 0, 1) * fg_float.unsqueeze(0)

    def _hard_assign(scores, palette):
        valid = (scores.sum(dim=0, keepdim=True) > 1e-6).float() * fg_float.unsqueeze(0)
        winner = scores.argmax(dim=0)
        hard = torch.nn.functional.one_hot(winner, num_classes=scores.shape[0]).permute(2, 0, 1).to(dtype=image.dtype)
        hard = hard * valid
        return torch.einsum('chw,cd->dhw', hard, palette)

    face_alpha = _mask2d(face_mask)
    body_alpha = _mask2d(body_mask)
    cloth_alpha = _mask2d(cloth_mask)
    hair_alpha = _mask2d(hair_mask)
    shoe_alpha = _mask2d(shoe_mask)

    zeros = torch.zeros_like(fg_float)
    if face_alpha is None:
        face_alpha = zeros
    if body_alpha is None:
        body_alpha = zeros
    if cloth_alpha is None:
        cloth_alpha = zeros
    if hair_alpha is None:
        hair_alpha = zeros
    if shoe_alpha is None:
        shoe_alpha = zeros

    ys, xs = torch.where(fg)
    y0 = ys.min().float()
    y1 = ys.max().float()
    x0 = xs.min().float()
    x1 = xs.max().float()
    h = (y1 - y0).clamp_min(8.0)
    w = (x1 - x0).clamp_min(8.0)

    yy, xx = torch.meshgrid(
        torch.arange(image.shape[1], device=image.device, dtype=image.dtype),
        torch.arange(image.shape[2], device=image.device, dtype=image.dtype),
        indexing='ij',
    )
    y_rel = (yy - y0) / h
    x_rel = (xx - (x0 + x1) * 0.5).abs() / (0.5 * w + 1e-6)

    brightness = render_rgb.mean(dim=0)
    saturation = render_rgb.max(dim=0).values - render_rgb.min(dim=0).values
    warmth = 1.10 * render_rgb[0] + 0.71 * render_rgb[1] - 1.24 * render_rgb[2]

    skin_base = torch.sigmoid((brightness - 0.17) / 0.04) * torch.sigmoid((warmth - 0.01) / 0.04)
    skin_clean = skin_base * torch.sigmoid((0.30 - saturation) / 0.05)
    upper_band = fg_float * torch.sigmoid((y_rel - 0.03) / 0.03) * torch.sigmoid((0.42 - y_rel) / 0.06)
    torso_band = fg_float * torch.sigmoid((y_rel - 0.10) / 0.04) * torch.sigmoid((0.50 - y_rel) / 0.08)

    parsing_body = upper_band * skin_clean
    parsing_body = torch.maximum(parsing_body, body_alpha * (0.52 + 0.48 * skin_clean))
    parsing_body = torch.maximum(parsing_body, 0.58 * face_alpha * skin_clean)
    parsing_body = parsing_body * (1.0 - 0.78 * cloth_alpha)

    neck_ref = upper_band * torch.sigmoid((y_rel - 0.05) / 0.02) * torch.sigmoid((0.23 - y_rel) / 0.03)
    neck_ref = neck_ref * torch.sigmoid((0.18 - x_rel) / 0.035) * skin_clean
    chest_ref = upper_band * torch.sigmoid((y_rel - 0.15) / 0.025) * torch.sigmoid((0.33 - y_rel) / 0.04)
    chest_ref = chest_ref * torch.sigmoid((0.13 - x_rel) / 0.03)
    chest_ref = chest_ref * torch.sigmoid((brightness - 0.21) / 0.035) * torch.sigmoid((warmth - 0.05) / 0.035)
    clavicle_ref = upper_band * torch.sigmoid((y_rel - 0.11) / 0.02) * torch.sigmoid((0.29 - y_rel) / 0.035)
    clavicle_ref = clavicle_ref * torch.sigmoid((x_rel - 0.06) / 0.02) * torch.sigmoid((0.24 - x_rel) / 0.03)
    clavicle_ref = clavicle_ref * torch.sigmoid((brightness - 0.205) / 0.03) * torch.sigmoid((warmth - 0.04) / 0.03)
    clavicle_ref = clavicle_ref * (1.0 - 0.35 * cloth_alpha)

    face_bbox = _mask_bbox(face_alpha)
    if face_bbox is not None:
        fy0, fy1, fx0, fx1 = face_bbox
        face_h = (fy1 - fy0).clamp_min(6.0)
        face_w = (fx1 - fx0).clamp_min(6.0)
        face_center_x = 0.5 * (fx0 + fx1)
        face_x_rel = (xx - face_center_x).abs() / (0.5 * face_w + 1e-6)

        neck_anchor = fy1 + 0.01 * face_h
        neck_from_face = fg_float * torch.sigmoid((yy - neck_anchor) / 2.8) * torch.sigmoid(((neck_anchor + 0.26 * face_h) - yy) / 4.0)
        neck_from_face = neck_from_face * torch.sigmoid((0.22 - face_x_rel) / 0.045)
        neck_from_face = neck_from_face * torch.sigmoid((brightness - 0.17) / 0.035) * torch.sigmoid((warmth - 0.01) / 0.035)

        chest_anchor = fy1 + 0.16 * face_h
        chest_from_face = fg_float * torch.sigmoid((yy - chest_anchor) / 4.0) * torch.sigmoid(((chest_anchor + 0.34 * h) - yy) / 6.0)
        chest_from_face = chest_from_face * torch.sigmoid((0.12 - face_x_rel) / 0.03)
        chest_from_face = chest_from_face * torch.sigmoid((brightness - 0.215) / 0.03) * torch.sigmoid((warmth - 0.05) / 0.03)
        chest_from_face = chest_from_face * (1.0 - 0.55 * cloth_alpha)

        neck_ref = torch.maximum(neck_ref, 1.35 * neck_from_face)
        chest_ref = torch.maximum(chest_ref, 1.25 * chest_from_face)

    body_ref = torch.clamp(torch.maximum(parsing_body, torch.maximum(torch.maximum(1.06 * neck_ref, 1.10 * chest_ref), 1.20 * clavicle_ref)), 0.0, 1.0)
    chest_open = torch.clamp(torch.maximum(1.34 * chest_ref, 1.16 * neck_ref) * (1.0 - 0.34 * cloth_alpha), 0.0, 1.0)
    body_ref = torch.clamp(torch.maximum(body_ref, chest_open), 0.0, 1.0)
    cloth_ref = torch.clamp(cloth_alpha * (1.0 - 0.56 * body_ref) * (1.0 - 0.46 * chest_open), 0.0, 1.0)
    soft_boundary = (_binary_dilate(body_ref.unsqueeze(0).unsqueeze(0), 5)[0, 0] * _binary_dilate(cloth_ref.unsqueeze(0).unsqueeze(0), 5)[0, 0]).clamp(0.0, 1.0)
    soft_boundary = soft_boundary * (1.0 - 0.78 * body_ref) * (1.0 - 0.80 * cloth_ref)
    soft_misc = torch.maximum(hair_alpha, shoe_alpha)
    exposed_skin = torch.clamp(torch.maximum(face_alpha, torch.maximum(neck_ref, torch.maximum(chest_ref, 0.88 * clavicle_ref))), 0.0, 1.0)
    body_core = _binary_erode((body_ref > 0.40).to(dtype=image.dtype).unsqueeze(0).unsqueeze(0), 3)[0, 0] * fg_float
    cloth_core = _binary_erode((cloth_ref > 0.42).to(dtype=image.dtype).unsqueeze(0).unsqueeze(0), 3)[0, 0] * fg_float
    body_shell = _binary_dilate((body_ref > 0.28).to(dtype=image.dtype).unsqueeze(0).unsqueeze(0), 5)[0, 0] * fg_float
    cloth_shell = _binary_dilate((cloth_ref > 0.30).to(dtype=image.dtype).unsqueeze(0).unsqueeze(0), 5)[0, 0] * fg_float
    body_edge = (body_shell - body_core).clamp(0.0, 1.0)
    cloth_edge = (cloth_shell - cloth_core).clamp(0.0, 1.0)
    upper_skin_ref = torch.clamp(torch.maximum(exposed_skin, body_ref), 0.0, 1.0)
    body_bridge = _binary_close((upper_skin_ref > 0.22).to(dtype=image.dtype).unsqueeze(0).unsqueeze(0), 5)[0, 0] * fg_float
    body_bridge = body_bridge * (1.0 - 0.82 * cloth_core) * (1.0 - 0.48 * cloth_ref)
    waist_skin = torso_band * torch.sigmoid((y_rel - 0.30) / 0.04) * torch.sigmoid((0.74 - y_rel) / 0.06)
    waist_skin = waist_skin * torch.sigmoid((x_rel - 0.08) / 0.03) * torch.sigmoid((0.36 - x_rel) / 0.05)
    waist_skin = waist_skin * skin_clean * (1.0 - 0.78 * cloth_ref)
    body_bridge = torch.maximum(body_bridge, 0.85 * waist_skin)
    cloth_bridge = _binary_close((cloth_ref > 0.24).to(dtype=image.dtype).unsqueeze(0).unsqueeze(0), 5)[0, 0] * fg_float
    cloth_bridge = cloth_bridge * (1.0 - 0.84 * body_core) * (1.0 - 0.36 * upper_skin_ref)

    if map_name == 'region':
        palette = REGION_COLORS.to(device=image.device, dtype=image.dtype)
        base_scores = _palette_scores(image, palette)
        soft_ref = torch.clamp(
            torch.maximum(1.00 * soft_boundary, 0.90 * soft_misc)
            + 0.18 * torch.minimum(body_edge, cloth_edge)
            + 0.10 * torch.maximum(body_edge, cloth_edge),
            0.0,
            1.0,
        )
        body_score = torch.clamp(
            0.62 * base_scores[0] + 1.22 * body_ref + 0.76 * chest_open + 0.38 * body_core + 0.52 * exposed_skin + 0.34 * body_bridge + 0.16 * waist_skin + 0.08 * body_edge
            - 0.34 * cloth_ref - 0.28 * cloth_core - 0.16 * cloth_bridge - 0.44 * soft_ref,
            0.0,
            2.35,
        )
        cloth_score = torch.clamp(
            0.68 * base_scores[2] + 0.98 * cloth_ref + 0.28 * cloth_core + 0.16 * cloth_bridge
            - 0.38 * body_ref - 0.84 * chest_open - 0.42 * exposed_skin - 0.42 * body_bridge - 0.46 * soft_ref,
            0.0,
            2.00,
        )
        soft_score = torch.clamp(
            0.46 * base_scores[1] + 0.98 * soft_ref + 0.14 * torch.minimum(body_ref, cloth_ref) + 0.18 * torch.minimum(body_bridge, cloth_bridge)
            - 0.12 * exposed_skin - 0.10 * body_bridge,
            0.0,
            1.75,
        )
        scores = torch.stack([body_score, soft_score, cloth_score], dim=0) * fg_float.unsqueeze(0)
        region_image = _hard_assign(scores, palette) * fg_float.unsqueeze(0)
        body_tight = torch.clamp(torch.maximum(torch.maximum(1.05 * exposed_skin, 1.08 * chest_open), 0.94 * body_bridge) * (1.0 - 0.92 * cloth_core) * (1.0 - 0.56 * cloth_ref), 0.0, 1.0)
        cloth_tight = torch.clamp(torch.maximum(cloth_core, 0.82 * cloth_bridge) * (1.0 - 0.90 * body_core) * (1.0 - 0.48 * upper_skin_ref), 0.0, 1.0)
        transition_tight = torch.clamp(torch.minimum(body_edge, cloth_edge) * (1.0 - 0.65 * body_tight) * (1.0 - 0.65 * cloth_tight), 0.0, 1.0)
        body_alpha = (0.40 * body_tight).clamp(0.0, 0.46).unsqueeze(0)
        cloth_alpha = (0.20 * cloth_tight).clamp(0.0, 0.24).unsqueeze(0)
        soft_alpha = (0.14 * transition_tight).clamp(0.0, 0.18).unsqueeze(0)
        region_image = region_image * (1.0 - body_alpha) + palette[0].view(3, 1, 1) * body_alpha
        region_image = region_image * (1.0 - cloth_alpha) + palette[2].view(3, 1, 1) * cloth_alpha
        region_image = region_image * (1.0 - soft_alpha) + palette[1].view(3, 1, 1) * soft_alpha
        return region_image * fg_float.unsqueeze(0)

    if map_name == 'layer':
        palette = LAYER_COLORS.to(device=image.device, dtype=image.dtype)
        base_scores = _palette_scores(image, palette)
        silhouette = torch.sigmoid((x_rel - 0.16) / 0.040)
        torso_center = torso_band * torch.sigmoid((0.16 - x_rel) / 0.040)
        limb_ref = body_ref * torch.maximum(torch.sigmoid((x_rel - 0.21) / 0.045), torch.sigmoid((y_rel - 0.49) / 0.05))
        rigid_core = 0.56 * base_scores[0] + 0.80 * limb_ref + 0.58 * torso_center * torch.maximum(body_core, 0.85 * exposed_skin)
        rigid_ref = torch.clamp(1.04 * neck_ref + 0.30 * face_alpha + 0.46 * chest_ref + 0.30 * clavicle_ref + rigid_core, 0.0, 2.0)
        torso_soft = torch.maximum(0.72 * soft_boundary, 0.24 * cloth_ref * (1.0 - silhouette))
        soft_ref = torch.clamp(
            0.84 * base_scores[1] + 0.82 * torso_soft + 0.18 * cloth_ref + 0.12 * torch.minimum(body_ref, cloth_ref)
            - 0.16 * torso_center * torch.maximum(body_core, exposed_skin),
            0.0,
            1.75,
        )
        free_band = torch.maximum(cloth_edge * torch.maximum(silhouette, torch.sigmoid((y_rel - 0.56) / 0.05)), 0.90 * soft_misc)
        free_ref = torch.clamp(0.64 * base_scores[2] + 1.00 * free_band + 0.12 * soft_boundary, 0.0, 1.6)
        torso_support = torch.clamp(torso_center * (1.0 - 0.75 * soft_boundary) * (1.0 - 0.90 * free_band), 0.0, 1.0)
        rigid_ref = torch.clamp(rigid_ref + 0.42 * torso_support, 0.0, 2.2)
        soft_ref = torch.clamp(soft_ref - 0.18 * torso_support, 0.0, 1.7)
        rigid_ref = rigid_ref * (1.0 - 0.34 * free_ref)
        soft_ref = soft_ref * (1.0 - 0.12 * free_ref)
        scores = torch.stack([rigid_ref, soft_ref, free_ref], dim=0) * fg_float.unsqueeze(0)
        layer_image = _hard_assign(scores, palette) * fg_float.unsqueeze(0)
        torso_core = torch.clamp(torso_center * torch.sigmoid((0.46 - y_rel) / 0.08) * (1.0 - 0.85 * cloth_edge) * (1.0 - 0.96 * soft_misc), 0.0, 1.0)
        rigid_alpha = (0.82 * torso_core).clamp(0.0, 0.88).unsqueeze(0)
        rigid_color = palette[0].view(3, 1, 1)
        layer_image = layer_image * (1.0 - rigid_alpha) + rigid_color * rigid_alpha
        return layer_image * fg_float.unsqueeze(0)

    high = image.new_tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
    low = image.new_tensor([0.0, 0.0, 1.0]).view(3, 1, 1)
    if map_name == 'body_prob':
        hi_alpha = torch.clamp(0.86 * body_ref + 0.24 * body_core + 0.28 * chest_open + 0.18 * exposed_skin, 0.0, 0.98).unsqueeze(0)
        image = image * (1.0 - hi_alpha) + high * hi_alpha
        return image * fg_float.unsqueeze(0)

    hi_alpha = torch.clamp(0.82 * cloth_ref + 0.18 * cloth_core - 0.28 * body_ref - 0.36 * chest_open, 0.0, 0.92).unsqueeze(0)
    image = image * (1.0 - hi_alpha) + high * hi_alpha
    lo_alpha = torch.clamp(torch.maximum(0.96 * body_ref + 0.22 * chest_open, 0.84 * soft_misc), 0.0, 0.98).unsqueeze(0)
    image = image * (1.0 - lo_alpha) + low * lo_alpha
    return image * fg_float.unsqueeze(0)


def _appearance_guided_head_refine(image, render_rgb, opacity, map_name):
    if render_rgb is None or map_name not in {'layer', 'region', 'body_prob', 'cloth_prob'}:
        return image

    if opacity is not None and opacity.dim() == 3:
        fg = opacity[0] > 0.02
    else:
        fg = render_rgb.mean(dim=0) > 0.02
    if fg.sum().item() < 64:
        return image

    ys, xs = torch.where(fg)
    y0 = ys.min().float()
    y1 = ys.max().float()
    x0 = xs.min().float()
    x1 = xs.max().float()
    h = (y1 - y0).clamp_min(8.0)
    w = (x1 - x0).clamp_min(8.0)

    yy, xx = torch.meshgrid(
        torch.arange(image.shape[1], device=image.device, dtype=image.dtype),
        torch.arange(image.shape[2], device=image.device, dtype=image.dtype),
        indexing='ij',
    )
    y_rel = (yy - y0) / h
    x_rel = (xx - (x0 + x1) * 0.5).abs() / (0.5 * w + 1e-6)

    head_band = fg.float() * torch.sigmoid((0.27 - y_rel) / 0.030)
    crown_band = head_band * torch.sigmoid((0.14 - y_rel) / 0.024)
    forehead_band = fg.float() * torch.sigmoid((y_rel - 0.08) / 0.024) * torch.sigmoid((0.25 - y_rel) / 0.032)
    center_face = torch.sigmoid((0.34 - x_rel) / 0.055)
    side_hair = torch.sigmoid((x_rel - 0.18) / 0.040)

    brightness = render_rgb.mean(dim=0)
    saturation = render_rgb.max(dim=0).values - render_rgb.min(dim=0).values
    warmth = 1.10 * render_rgb[0] + 0.72 * render_rgb[1] - 1.20 * render_rgb[2]

    dark_hint = torch.sigmoid((0.22 - brightness) / 0.045)
    neutral_hint = torch.sigmoid((0.20 - saturation) / 0.045)
    skin_hint = torch.sigmoid((brightness - 0.16) / 0.045) * torch.sigmoid((warmth - 0.00) / 0.045)

    face_hint = torch.clamp(forehead_band * center_face * skin_hint, 0.0, 1.0)
    hair_hint = torch.clamp(crown_band * (0.62 + 0.38 * side_hair) + 0.34 * head_band * dark_hint * neutral_hint, 0.0, 1.0)
    hair_hint = hair_hint * (1.0 - 0.92 * face_hint)

    if map_name == 'region':
        soft_color = REGION_COLORS[1].to(image.device, image.dtype).view(3, 1, 1)
        body_color = REGION_COLORS[0].to(image.device, image.dtype).view(3, 1, 1)
        body_alpha = (0.78 * face_hint).clamp(0.0, 0.86).unsqueeze(0)
        hair_alpha = (0.92 * hair_hint * (1.0 - 0.88 * face_hint)).clamp(0.0, 0.98).unsqueeze(0)
        image = image * (1.0 - body_alpha) + body_color * body_alpha
        image = image * (1.0 - hair_alpha) + soft_color * hair_alpha
        return image

    if map_name == 'layer':
        free_color = LAYER_COLORS[2].to(image.device, image.dtype).view(3, 1, 1)
        rigid_color = LAYER_COLORS[0].to(image.device, image.dtype).view(3, 1, 1)
        rigid_alpha = (0.34 * face_hint).clamp(0.0, 0.42).unsqueeze(0)
        hair_alpha = (0.90 * hair_hint * (1.0 - 0.90 * face_hint)).clamp(0.0, 0.96).unsqueeze(0)
        image = image * (1.0 - rigid_alpha) + rigid_color * rigid_alpha
        image = image * (1.0 - hair_alpha) + free_color * hair_alpha
        return image

    high = image.new_tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
    low = image.new_tensor([0.0, 0.0, 1.0]).view(3, 1, 1)
    if map_name == 'body_prob':
        hi_alpha = (0.74 * face_hint).clamp(0.0, 0.82).unsqueeze(0)
        lo_alpha = (0.94 * hair_hint).clamp(0.0, 0.98).unsqueeze(0)
        image = image * (1.0 - hi_alpha) + high * hi_alpha
        image = image * (1.0 - lo_alpha) + low * lo_alpha
        return image

    lo_alpha = (0.78 * torch.maximum(face_hint, hair_hint)).clamp(0.0, 0.92).unsqueeze(0)
    image = image * (1.0 - lo_alpha) + low * lo_alpha
    return image



_VIEW_NAME_RE = re.compile(r'c(?P<cam>\d+)_f(?P<frame>\d+)$')


def _prepare_view_parser_mask(view, image):
    parser_mask = getattr(view, 'parsing_parser_mask', None)
    if parser_mask is None:
        return None
    if parser_mask.dim() == 3:
        parser_mask_2d = parser_mask[0]
    elif parser_mask.dim() == 2:
        parser_mask_2d = parser_mask
    else:
        parser_mask_2d = parser_mask.squeeze()
    parser_mask_2d = parser_mask_2d.to(device=image.device, dtype=image.dtype)
    if parser_mask_2d.shape != image.shape[-2:]:
        parser_mask_2d = torch.nn.functional.interpolate(
            parser_mask_2d.unsqueeze(0).unsqueeze(0),
            size=image.shape[-2:],
            mode='nearest',
        )[0, 0]
    return parser_mask_2d.round()


def _load_view_parser_label_mask(view, image, labels=(2,)):
    parser_mask_2d = _prepare_view_parser_mask(view, image)
    if parser_mask_2d is None:
        return None
    label_mask = torch.zeros_like(parser_mask_2d)
    for label in labels:
        label_mask = torch.maximum(label_mask, (parser_mask_2d == float(label)).to(dtype=image.dtype))
    return label_mask.unsqueeze(0).clamp(0.0, 1.0)


def _load_cihp_hair_mask(view, image, config, labels=(2,)):
    parser_mask = _load_view_parser_label_mask(view, image, labels=labels)
    if parser_mask is not None:
        return parser_mask
    image_name = getattr(view, 'image_name', '')
    match = _VIEW_NAME_RE.match(image_name)
    if match is None:
        return None

    subject = None
    dataset_cfg = config.get('dataset', None)
    if dataset_cfg is not None:
        subject = dataset_cfg.get('subject', None)
    if not subject:
        return None

    base_root = Path(config.get('binding_map_cihp_root', '/remote-home/ming/dataSet'))
    cam = int(match.group('cam'))
    frame = int(match.group('frame'))
    mask_path = base_root / subject / 'mask_cihp' / f'Camera_B{cam}' / f'{frame:06d}.png'
    if not mask_path.exists():
        return None

    mask_np = imageio.imread(mask_path)
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    label_array = np.asarray(list(labels), dtype=mask_np.dtype)
    hair = torch.from_numpy(np.isin(mask_np, label_array).astype(np.float32)).to(device=image.device, dtype=image.dtype)
    hair = hair.unsqueeze(0).unsqueeze(0)
    hair = _binary_open(_binary_close(hair, 5), 3)
    if hair.shape[-2:] != image.shape[-2:]:
        hair = torch.nn.functional.interpolate(hair, size=image.shape[-2:], mode='bilinear', align_corners=False)
    return hair.clamp(0.0, 1.0)


def _apply_cihp_parsing_override(image, hair_mask, body_mask, cloth_mask, map_name, face_mask=None, shoe_mask=None):
    if map_name not in {'layer', 'region', 'body_prob', 'cloth_prob'}:
        return image
    if hair_mask is None and body_mask is None and cloth_mask is None and face_mask is None and shoe_mask is None:
        return image

    def _prep(mask, dilate=3, erode=0):
        if mask is None:
            return None
        mask = mask.to(device=image.device, dtype=image.dtype)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
        elif mask.dim() != 4:
            mask = mask.reshape(1, 1, *mask.shape[-2:])
        if erode > 0:
            mask = _binary_erode(mask, erode)
        if dilate > 0:
            mask = _binary_dilate(mask, dilate)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
        mask = mask.clamp(0.0, 1.0)
        target_hw = image.shape[-2:]
        if mask.shape[-2:] != target_hw:
            mask = torch.nn.functional.interpolate(mask, size=target_hw, mode='bilinear', align_corners=False)
        if image.dim() == 3 and mask.dim() == 4:
            mask = mask[0]
        return mask

    hair_alpha = _prep(hair_mask, dilate=0, erode=4)
    body_alpha = _prep(body_mask, dilate=5)
    cloth_alpha = _prep(cloth_mask, dilate=5)
    face_alpha = _prep(face_mask, dilate=2)
    shoe_alpha = _prep(shoe_mask, dilate=3)

    face_bbox = None
    face_h = None
    face_w = None
    yy = None
    xx = None
    if hair_alpha is not None and face_alpha is not None:
        face_bbox = _mask_bbox(face_alpha)
        if face_bbox is not None:
            fy0, fy1, fx0, fx1 = face_bbox
            face_h = (fy1 - fy0).clamp_min(6.0)
            face_w = (fx1 - fx0).clamp_min(6.0)
            mask_h, mask_w = hair_alpha.shape[-2:]
            yy, xx = torch.meshgrid(
                torch.arange(mask_h, device=image.device, dtype=image.dtype),
                torch.arange(mask_w, device=image.device, dtype=image.dtype),
                indexing='ij',
            )
            face_x_rel = (xx - 0.5 * (fx0 + fx1)).abs() / (0.5 * face_w + 1e-6)
            face_dilate = _binary_dilate(face_alpha.unsqueeze(0), 5)[0]
            forehead_cut = torch.sigmoid(((fy0 + 0.16 * face_h) - yy) / 3.0)
            side_keep = torch.sigmoid((face_x_rel - 0.34) / 0.05)
            hair_keep = torch.maximum(forehead_cut, side_keep)
            hair_alpha = hair_alpha * hair_keep * (1.0 - 0.90 * face_dilate * (1.0 - side_keep))
    elif face_alpha is not None:
        face_bbox = _mask_bbox(face_alpha)
        if face_bbox is not None:
            fy0, fy1, fx0, fx1 = face_bbox
            face_h = (fy1 - fy0).clamp_min(6.0)
            face_w = (fx1 - fx0).clamp_min(6.0)
            mask_h, mask_w = face_alpha.shape[-2:]
            yy, xx = torch.meshgrid(
                torch.arange(mask_h, device=image.device, dtype=image.dtype),
                torch.arange(mask_w, device=image.device, dtype=image.dtype),
                indexing='ij',
            )

    if body_alpha is not None and cloth_alpha is not None:
        boundary = (_binary_dilate(body_alpha.unsqueeze(0), 5)[0] * _binary_dilate(cloth_alpha.unsqueeze(0), 5)[0]).clamp(0.0, 1.0)
        boundary = boundary * (1.0 - 0.92 * body_alpha) * (1.0 - 0.92 * cloth_alpha)
    else:
        boundary = None

    chest_body_alpha = None
    if body_alpha is not None and face_bbox is not None and yy is not None and xx is not None:
        fy0, fy1, fx0, fx1 = face_bbox
        face_cx = 0.5 * (fx0 + fx1)
        chest_y = torch.sigmoid((yy - (fy1 + 0.02 * face_h)) / (0.10 * face_h + 1e-6))
        chest_y = chest_y * torch.sigmoid((((fy1 + 1.15 * face_h)) - yy) / (0.18 * face_h + 1e-6))
        chest_x = torch.sigmoid(((0.48 * face_w) - (xx - face_cx).abs()) / (0.10 * face_w + 1e-6))
        chest_body_alpha = body_alpha * chest_y * chest_x
        if cloth_alpha is not None:
            chest_body_alpha = chest_body_alpha * (1.0 - 0.55 * cloth_alpha)
        chest_body_alpha = chest_body_alpha.clamp(0.0, 1.0)

    if hair_alpha is not None and body_alpha is not None:
        hair_alpha = hair_alpha * (1.0 - 0.98 * body_alpha)
    if hair_alpha is not None and cloth_alpha is not None:
        hair_alpha = hair_alpha * (1.0 - 0.98 * cloth_alpha)
    if body_alpha is not None and cloth_alpha is not None:
        body_alpha = body_alpha * (1.0 - 0.88 * cloth_alpha)
        cloth_alpha = cloth_alpha * (1.0 - 0.88 * body_alpha)
    if shoe_alpha is not None:
        if body_alpha is not None:
            body_alpha = body_alpha * (1.0 - 0.98 * shoe_alpha)
        if cloth_alpha is not None:
            cloth_alpha = cloth_alpha * (1.0 - 0.70 * shoe_alpha)
        if hair_alpha is not None:
            hair_alpha = hair_alpha * (1.0 - 0.98 * shoe_alpha)

    if map_name == 'region':
        body_color = REGION_COLORS[0].to(image.device, image.dtype).view(3, 1, 1)
        soft_color = REGION_COLORS[1].to(image.device, image.dtype).view(3, 1, 1)
        cloth_color = REGION_COLORS[2].to(image.device, image.dtype).view(3, 1, 1)
        if body_alpha is not None:
            image = image * (1.0 - 0.95 * body_alpha) + body_color * (0.95 * body_alpha)
        if cloth_alpha is not None:
            image = image * (1.0 - 0.95 * cloth_alpha) + cloth_color * (0.95 * cloth_alpha)
        if chest_body_alpha is not None:
            image = image * (1.0 - 0.98 * chest_body_alpha) + body_color * (0.98 * chest_body_alpha)
        if boundary is not None:
            image = image * (1.0 - 0.72 * boundary) + soft_color * (0.72 * boundary)
        if hair_alpha is not None:
            image = image * (1.0 - 0.98 * hair_alpha) + soft_color * (0.98 * hair_alpha)
        if shoe_alpha is not None:
            image = image * (1.0 - 0.95 * shoe_alpha) + soft_color * (0.95 * shoe_alpha)
        return image

    if map_name == 'layer':
        rigid_color = LAYER_COLORS[0].to(image.device, image.dtype).view(3, 1, 1)
        soft_color = LAYER_COLORS[1].to(image.device, image.dtype).view(3, 1, 1)
        free_color = LAYER_COLORS[2].to(image.device, image.dtype).view(3, 1, 1)
        body_rigid = None
        if body_alpha is not None:
            body_rigid = (0.70 * body_alpha * (1.0 - 0.72 * cloth_alpha)).clamp(0.0, 0.82)
            image = image * (1.0 - body_rigid) + rigid_color * body_rigid
        if cloth_alpha is not None:
            cloth_free = (0.86 * cloth_alpha * (1.0 - 0.46 * body_alpha)).clamp(0.0, 0.94)
            image = image * (1.0 - cloth_free) + free_color * cloth_free
        if boundary is not None:
            layer_soft = (0.44 * boundary * (1.0 - 0.55 * torch.maximum(body_alpha, cloth_alpha))).clamp(0.0, 0.52)
            image = image * (1.0 - layer_soft) + soft_color * layer_soft
        if hair_alpha is not None:
            image = image * (1.0 - 0.96 * hair_alpha) + free_color * (0.96 * hair_alpha)
        if shoe_alpha is not None:
            image = image * (1.0 - 0.94 * shoe_alpha) + free_color * (0.94 * shoe_alpha)
        return image

    high = image.new_tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
    low = image.new_tensor([0.0, 0.0, 1.0]).view(3, 1, 1)
    if map_name == 'body_prob':
        if body_alpha is not None:
            image = image * (1.0 - 0.90 * body_alpha) + high * (0.90 * body_alpha)
        if face_alpha is not None:
            image = image * (1.0 - 0.55 * face_alpha) + high * (0.55 * face_alpha)
        suppress = None
        if cloth_alpha is not None and hair_alpha is not None:
            suppress = torch.maximum(cloth_alpha, hair_alpha)
        else:
            suppress = cloth_alpha if cloth_alpha is not None else hair_alpha
        if suppress is not None:
            image = image * (1.0 - 0.96 * suppress) + low * (0.96 * suppress)
        if shoe_alpha is not None:
            image = image * (1.0 - 0.96 * shoe_alpha) + low * (0.96 * shoe_alpha)
        return image

    if cloth_alpha is not None:
        image = image * (1.0 - 0.92 * cloth_alpha) + high * (0.92 * cloth_alpha)
    suppress = None
    if body_alpha is not None and hair_alpha is not None:
        suppress = torch.maximum(body_alpha, hair_alpha)
    else:
        suppress = body_alpha if body_alpha is not None else hair_alpha
    if suppress is not None:
        image = image * (1.0 - 0.96 * suppress) + low * (0.96 * suppress)
    if shoe_alpha is not None:
        image = image * (1.0 - 0.96 * shoe_alpha) + low * (0.96 * shoe_alpha)
    return image


def _region_follow_bodyprob(region_image, body_prob_image, cloth_prob_image, face_mask=None):
    if face_mask is None:
        return region_image
    if face_mask.dim() == 4:
        face_mask = face_mask[0, 0]
    elif face_mask.dim() == 3:
        face_mask = face_mask[0]
    face_bbox = _mask_bbox(face_mask)
    if face_bbox is None:
        return region_image

    fy0, fy1, fx0, fx1 = face_bbox
    face_h = (fy1 - fy0).clamp_min(6.0)
    face_w = (fx1 - fx0).clamp_min(6.0)
    h, w = region_image.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=region_image.device, dtype=region_image.dtype),
        torch.arange(w, device=region_image.device, dtype=region_image.dtype),
        indexing='ij',
    )
    cx = 0.5 * (fx0 + fx1)
    chest_y = torch.sigmoid((yy - (fy1 + 0.04 * face_h)) / (0.11 * face_h + 1e-6))
    chest_y = chest_y * torch.sigmoid((((fy1 + 1.30 * face_h)) - yy) / (0.22 * face_h + 1e-6))
    chest_x = torch.sigmoid(((0.54 * face_w) - (xx - cx).abs()) / (0.12 * face_w + 1e-6))
    chest_focus = chest_y * chest_x

    body_dom = torch.sigmoid(((body_prob_image[0] - body_prob_image[2]) - 0.02) / 0.06)
    cloth_low = torch.sigmoid(((cloth_prob_image[2] - cloth_prob_image[0]) + 0.04) / 0.06)
    rescue = (chest_focus * body_dom * cloth_low).clamp(0.0, 1.0)
    body_color = REGION_COLORS[0].to(region_image.device, region_image.dtype).view(3, 1, 1)
    rescue = torch.clamp(rescue + 0.22 * chest_focus * body_dom, 0.0, 1.0)
    return region_image * (1.0 - 0.99 * rescue.unsqueeze(0)) + body_color * (0.99 * rescue.unsqueeze(0))


def _stabilize_visual_map(image, map_name, opacity, face_mask=None, body_mask=None, cloth_mask=None, hair_mask=None, shoe_mask=None):
    if map_name not in {'layer', 'region', 'semantic', 'thin'}:
        return image

    if opacity is not None and opacity.dim() == 3:
        fg = (opacity[0] > 0.03).to(dtype=image.dtype)
    else:
        fg = (image.sum(dim=0) > 0.05).to(dtype=image.dtype)
    if fg.sum().item() < 64:
        return image

    def _prep(mask):
        if mask is None:
            return torch.zeros_like(fg)
        if mask.dim() == 4:
            mask = mask[0, 0]
        elif mask.dim() == 3:
            mask = mask[0]
        if mask.shape[-2:] != image.shape[-2:]:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=image.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )[0, 0]
        return mask.clamp(0.0, 1.0) * fg

    def _palette_scores(img, palette):
        pixels = img.permute(1, 2, 0).reshape(-1, 3)
        dists = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(dim=-1)
        scores = torch.softmax(-14.0 * dists, dim=-1)
        return scores.reshape(img.shape[1], img.shape[2], palette.shape[0]).permute(2, 0, 1) * fg.unsqueeze(0)

    def _normalize_scores(scores):
        scores = scores.clamp_min(1e-6)
        return scores / scores.sum(dim=0, keepdim=True).clamp_min(1e-6)

    def _hard_assign(scores, palette):
        valid = (scores.sum(dim=0, keepdim=True) > 1e-6).to(dtype=image.dtype) * fg.unsqueeze(0)
        winner = scores.argmax(dim=0)
        hard = torch.nn.functional.one_hot(winner, num_classes=scores.shape[0]).permute(2, 0, 1).to(dtype=image.dtype)
        hard = hard * valid
        return torch.einsum('chw,cd->dhw', hard, palette)

    def _neighbor_consensus(scores, kernel_size=5, blend_low=0.18, blend_high=0.42):
        probs = _normalize_scores(scores)
        pooled = torch.nn.functional.avg_pool2d(
            probs.unsqueeze(0),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )[0]
        confidence = probs.max(dim=0).values
        blend = ((blend_high - confidence) / max(blend_high - blend_low, 1e-6)).clamp(0.0, 1.0)
        blend = blend * fg
        probs = _normalize_scores(probs * (1.0 - blend.unsqueeze(0)) + pooled * blend.unsqueeze(0))
        return probs

    face = _prep(face_mask)
    body = _prep(body_mask)
    cloth = _prep(cloth_mask)
    hair = _prep(hair_mask)
    shoe = _prep(shoe_mask)

    body = torch.maximum(body, face)
    soft_misc = torch.maximum(hair, shoe)
    raw_body = body
    body = body * (1.0 - 0.90 * cloth) * (1.0 - 0.98 * soft_misc)
    cloth = cloth * (1.0 - 0.72 * raw_body) * (1.0 - 0.96 * soft_misc)

    body_core = _binary_erode((body > 0.36).to(dtype=image.dtype).unsqueeze(0).unsqueeze(0), 5)[0, 0] * fg
    cloth_core = _binary_erode((cloth > 0.38).to(dtype=image.dtype).unsqueeze(0).unsqueeze(0), 5)[0, 0] * fg
    body_shell = _binary_dilate((body > 0.18).to(dtype=image.dtype).unsqueeze(0).unsqueeze(0), 5)[0, 0] * fg
    cloth_shell = _binary_dilate((cloth > 0.18).to(dtype=image.dtype).unsqueeze(0).unsqueeze(0), 5)[0, 0] * fg
    fg_edge = (_binary_dilate(fg.unsqueeze(0).unsqueeze(0), 5)[0, 0] - _binary_erode(fg.unsqueeze(0).unsqueeze(0), 5)[0, 0]).clamp(0.0, 1.0) * fg
    boundary = (body_shell * cloth_shell).clamp(0.0, 1.0)
    body_edge = (body_shell - body_core).clamp(0.0, 1.0)
    cloth_edge = (cloth_shell - cloth_core).clamp(0.0, 1.0)

    ys, xs = torch.where(fg > 0.5)
    y0 = ys.min().float()
    y1 = ys.max().float()
    x0 = xs.min().float()
    x1 = xs.max().float()
    h = (y1 - y0).clamp_min(8.0)
    w = (x1 - x0).clamp_min(8.0)
    yy, xx = torch.meshgrid(
        torch.arange(image.shape[1], device=image.device, dtype=image.dtype),
        torch.arange(image.shape[2], device=image.device, dtype=image.dtype),
        indexing='ij',
    )
    y_rel = (yy - y0) / h
    x_rel = (xx - (x0 + x1) * 0.5).abs() / (0.5 * w + 1e-6)

    if map_name == 'region':
        palette = REGION_COLORS.to(device=image.device, dtype=image.dtype)
        base = _palette_scores(image, palette)
        body_fill = _binary_close((torch.maximum(body_core, 0.78 * body) > 0.14).to(dtype=image.dtype).unsqueeze(0).unsqueeze(0), 5)[0, 0] * fg
        cloth_fill = _binary_close((torch.maximum(cloth_core, 0.78 * cloth) > 0.14).to(dtype=image.dtype).unsqueeze(0).unsqueeze(0), 5)[0, 0] * fg
        body_fill = torch.maximum(body_fill * (1.0 - 0.78 * cloth_core) * (1.0 - 0.96 * soft_misc), body_core)
        cloth_fill = torch.maximum(cloth_fill * (1.0 - 0.74 * body_core) * (1.0 - 0.92 * soft_misc), cloth_core)
        soft_band = torch.clamp(
            1.15 * boundary
            + 1.00 * soft_misc
            + 0.42 * torch.minimum(body_edge, cloth_edge)
            + 0.22 * fg_edge,
            0.0,
            1.8,
        )
        body_score = torch.clamp(
            1.10 * base[0]
            + 1.85 * body_core
            + 0.95 * body
            + 0.65 * face
            + 0.75 * body_fill
            - 1.15 * cloth_core
            - 0.25 * cloth_fill
            - 1.45 * soft_misc
            - 0.70 * soft_band,
            0.0,
            3.4,
        )
        cloth_score = torch.clamp(
            1.10 * base[2]
            + 1.90 * cloth_core
            + 1.05 * cloth
            + 0.72 * cloth_fill
            - 1.00 * body_core
            - 0.30 * body_fill
            - 1.40 * face
            - 1.20 * hair
            - 0.55 * shoe
            - 0.62 * soft_band,
            0.0,
            3.4,
        )
        soft_score = torch.clamp(
            0.95 * base[1]
            + 1.35 * soft_band
            + 0.35 * torch.maximum(body_edge, cloth_edge)
            + 0.12 * torch.minimum(body_fill, cloth_fill),
            0.0,
            3.1,
        )
        region_probs = _neighbor_consensus(torch.stack([body_score, soft_score, cloth_score], dim=0) * fg.unsqueeze(0), kernel_size=5)
        region_probs[0] = torch.where(face > 0.35, region_probs[0] + 0.75 * face, region_probs[0])
        region_probs[1] = torch.where(soft_misc > 0.25, region_probs[1] + 0.95 * soft_misc, region_probs[1])
        region_probs[0] = region_probs[0] + 0.65 * body_fill
        region_probs[2] = region_probs[2] + 0.60 * cloth_fill
        region_probs[1] = region_probs[1] + 0.20 * boundary
        region_probs = _normalize_scores(region_probs)
        body_force = (body_fill * (1.0 - 0.82 * cloth_core)).clamp(0.0, 1.0)
        cloth_force = (cloth_fill * (1.0 - 0.78 * body_core)).clamp(0.0, 1.0)
        body_target = torch.stack([torch.ones_like(body_force), torch.zeros_like(body_force), torch.zeros_like(body_force)], dim=0)
        cloth_target = torch.stack([torch.zeros_like(cloth_force), torch.zeros_like(cloth_force), torch.ones_like(cloth_force)], dim=0)
        body_alpha = (0.32 * body_force).clamp(0.0, 0.34)
        cloth_alpha = (0.28 * cloth_force * (1.0 - 0.55 * body_force)).clamp(0.0, 0.30)
        region_probs = region_probs * (1.0 - body_alpha.unsqueeze(0)) + body_target * body_alpha.unsqueeze(0)
        region_probs = region_probs * (1.0 - cloth_alpha.unsqueeze(0)) + cloth_target * cloth_alpha.unsqueeze(0)
        region_probs = _normalize_scores(region_probs)
        region = _hard_assign(region_probs * fg.unsqueeze(0), palette)
        return region * fg.unsqueeze(0)

    if map_name == 'semantic':
        red = image.new_tensor([0.92, 0.25, 0.18])
        yellow = image.new_tensor([0.95, 0.86, 0.22])
        green = image.new_tensor([0.18, 0.82, 0.34])
        palette = torch.stack([red, yellow, green], dim=0)
        base = _palette_scores(image, palette)
        stable_fill = _binary_close((torch.maximum(body_core, 0.82 * body) > 0.12).to(dtype=image.dtype).unsqueeze(0).unsqueeze(0), 5)[0, 0] * fg
        stable_fill = torch.maximum(stable_fill * (1.0 - 0.82 * boundary) * (1.0 - 0.96 * soft_misc), body_core)
        mid_band = torch.clamp(1.25 * boundary + 0.20 * torch.minimum(body_edge, cloth_edge) + 0.08 * fg_edge, 0.0, 1.8)
        unstable = torch.clamp(0.92 * cloth + 1.08 * soft_misc + 0.30 * cloth_edge + 0.12 * fg_edge, 0.0, 2.0) * (1.0 - 0.68 * stable_fill)
        low_score = torch.clamp(
            0.70 * base[0]
            + 1.40 * unstable
            - 0.60 * stable_fill
            - 0.28 * mid_band,
            0.0,
            3.2,
        )
        mid_score = torch.clamp(
            0.68 * base[1]
            + 1.35 * mid_band
            + 0.16 * torch.minimum(stable_fill, unstable),
            0.0,
            3.0,
        )
        high_score = torch.clamp(
            0.70 * base[2]
            + 1.85 * stable_fill
            + 0.55 * face
            - 0.62 * unstable
            - 0.48 * mid_band,
            0.0,
            3.4,
        )
        semantic_probs = _neighbor_consensus(torch.stack([low_score, mid_score, high_score], dim=0) * fg.unsqueeze(0), kernel_size=5)
        semantic_probs[2] = semantic_probs[2] + 0.80 * stable_fill + 0.34 * face
        semantic_probs[1] = semantic_probs[1] + 0.32 * mid_band
        semantic_probs[0] = semantic_probs[0] + 0.88 * soft_misc + 0.38 * cloth_edge
        semantic_probs = _normalize_scores(semantic_probs)
        stable_force = stable_fill.clamp(0.0, 1.0)
        unstable_force = torch.clamp(soft_misc + 0.72 * cloth_edge + 0.45 * cloth * (1.0 - 0.80 * body), 0.0, 1.0)
        mid_force = torch.clamp(boundary + 0.18 * torch.minimum(body_edge, cloth_edge), 0.0, 1.0)
        low_target = torch.stack([torch.ones_like(unstable_force), torch.zeros_like(unstable_force), torch.zeros_like(unstable_force)], dim=0)
        mid_target = torch.stack([torch.zeros_like(mid_force), torch.ones_like(mid_force), torch.zeros_like(mid_force)], dim=0)
        high_target = torch.stack([torch.zeros_like(stable_force), torch.zeros_like(stable_force), torch.ones_like(stable_force)], dim=0)
        high_alpha = (0.46 * stable_force).clamp(0.0, 0.52)
        low_alpha = (0.50 * unstable_force * (1.0 - 0.55 * stable_force)).clamp(0.0, 0.56)
        mid_alpha = (0.34 * mid_force * (1.0 - 0.60 * torch.maximum(stable_force, unstable_force))).clamp(0.0, 0.38)
        semantic_probs = semantic_probs * (1.0 - high_alpha.unsqueeze(0)) + high_target * high_alpha.unsqueeze(0)
        semantic_probs = semantic_probs * (1.0 - low_alpha.unsqueeze(0)) + low_target * low_alpha.unsqueeze(0)
        semantic_probs = semantic_probs * (1.0 - mid_alpha.unsqueeze(0)) + mid_target * mid_alpha.unsqueeze(0)
        semantic_probs = _normalize_scores(semantic_probs)
        semantic = _hard_assign(semantic_probs * fg.unsqueeze(0), palette)
        return semantic * fg.unsqueeze(0)

    if map_name == 'thin':
        low = image.new_tensor([0.15, 0.22, 0.75])
        high = image.new_tensor([0.98, 0.18, 0.86])
        palette = torch.stack([low, high], dim=0)
        base = _palette_scores(image, palette)
        boundary_hint = torch.clamp(1.35 * boundary + 0.95 * cloth_edge + 0.45 * fg_edge * cloth_shell + 0.25 * soft_misc * fg_edge, 0.0, 2.0)
        suppress = torch.clamp(0.85 * body_core + 0.55 * cloth_core * (1.0 - boundary), 0.0, 1.5)
        high_score = torch.clamp(0.60 * base[1] + 1.60 * boundary_hint - 1.20 * suppress, 0.0, 3.4)
        low_score = torch.clamp(0.80 * base[0] + 0.90 * suppress + 0.20 * body + 0.18 * cloth, 0.0, 3.2)
        thin_probs = _neighbor_consensus(torch.stack([low_score, high_score], dim=0) * fg.unsqueeze(0), kernel_size=5)
        high_force = torch.clamp(0.72 * boundary + 0.62 * cloth_edge + 0.38 * soft_misc * fg_edge, 0.0, 1.0)
        low_force = torch.clamp(body_core + 0.70 * cloth_core * (1.0 - boundary), 0.0, 1.0)
        low_target = torch.stack([torch.ones_like(low_force), torch.zeros_like(low_force)], dim=0)
        high_target = torch.stack([torch.zeros_like(high_force), torch.ones_like(high_force)], dim=0)
        high_alpha = (0.54 * high_force).clamp(0.0, 0.58)
        low_alpha = (0.44 * low_force * (1.0 - 0.65 * high_force)).clamp(0.0, 0.48)
        thin_probs = thin_probs * (1.0 - high_alpha.unsqueeze(0)) + high_target * high_alpha.unsqueeze(0)
        thin_probs = thin_probs * (1.0 - low_alpha.unsqueeze(0)) + low_target * low_alpha.unsqueeze(0)
        thin_probs = _normalize_scores(thin_probs)
        thin_map = low.view(3, 1, 1) * thin_probs[0].unsqueeze(0) + high.view(3, 1, 1) * thin_probs[1].unsqueeze(0)
        return thin_map * fg.unsqueeze(0)

    palette = LAYER_COLORS.to(device=image.device, dtype=image.dtype)
    rigid_color = palette[0].view(3, 1, 1)
    soft_color = palette[1].view(3, 1, 1)
    free_color = palette[2].view(3, 1, 1)
    base = _palette_scores(image, palette)
    torso_center = fg * torch.sigmoid((y_rel - 0.10) / 0.04) * torch.sigmoid((0.54 - y_rel) / 0.05) * torch.sigmoid((0.30 - x_rel) / 0.06)
    limb_core = body_core * (1.0 - 0.58 * torso_center)
    cloth_hem = cloth_edge * torch.maximum(torch.sigmoid((y_rel - 0.38) / 0.05), torch.sigmoid((x_rel - 0.30) / 0.05))
    torso_attached = cloth_shell * torso_center * (1.0 - 0.86 * boundary) * (1.0 - 0.97 * soft_misc)
    rigid_fill = torch.maximum(body_core, 0.96 * torso_attached)
    free_prior = torch.clamp(1.35 * hair + 1.25 * shoe + 1.55 * cloth_hem, 0.0, 2.4)
    soft_prior = torch.clamp(
        1.35 * boundary
        + 0.44 * body_edge
        + 0.34 * cloth_edge
        + 0.12 * torch.minimum(body_shell, cloth_shell),
        0.0,
        2.3,
    )
    rigid_score = torch.clamp(
        0.72 * base[0]
        + 2.15 * limb_core
        + 1.85 * rigid_fill
        + 1.32 * torso_center * (1.0 - 0.88 * cloth_hem)
        + 0.84 * face
        - 0.22 * cloth_edge
        - 1.08 * free_prior
        - 0.62 * soft_prior,
        0.0,
        3.6,
    )
    free_score = torch.clamp(
        0.72 * base[2]
        + 2.05 * free_prior
        + 0.18 * fg_edge * torch.maximum(cloth_shell, soft_misc)
        - 1.02 * rigid_fill
        - 0.72 * torso_attached,
        0.0,
        3.4,
    )
    soft_score = torch.clamp(
        0.66 * base[1]
        + 1.55 * soft_prior
        + 0.12 * cloth_edge
        + 0.12 * body_edge
        - 0.74 * rigid_fill
        - 0.28 * free_prior,
        0.0,
        3.0,
    )
    layer_probs = _neighbor_consensus(torch.stack([rigid_score, soft_score, free_score], dim=0) * fg.unsqueeze(0), kernel_size=5)
    layer_probs[2] = torch.where(soft_misc > 0.25, layer_probs[2] + 1.16 * soft_misc, layer_probs[2])
    layer_probs[1] = torch.where(boundary > 0.16, layer_probs[1] + 0.48 * boundary, layer_probs[1])
    layer_probs[0] = torch.where(face > 0.30, layer_probs[0] + 0.66 * face, layer_probs[0])
    layer_probs = _normalize_scores(layer_probs)
    rigid_force = (rigid_fill * (1.0 - 0.78 * cloth_hem)).clamp(0.0, 1.0)
    free_force = torch.clamp(torch.maximum(soft_misc, cloth_hem), 0.0, 1.0)
    soft_force = torch.clamp(boundary + 0.18 * torch.minimum(body_edge, cloth_edge), 0.0, 1.0)
    rigid_target = torch.stack([
        torch.ones_like(rigid_force),
        torch.zeros_like(rigid_force),
        torch.zeros_like(rigid_force),
    ], dim=0)
    free_target = torch.stack([
        torch.zeros_like(free_force),
        torch.zeros_like(free_force),
        torch.ones_like(free_force),
    ], dim=0)
    soft_target = torch.stack([
        torch.zeros_like(soft_force),
        torch.ones_like(soft_force),
        torch.zeros_like(soft_force),
    ], dim=0)
    rigid_alpha = (0.60 * rigid_force).clamp(0.0, 0.60)
    free_alpha = (0.50 * free_force * (1.0 - 0.72 * rigid_force)).clamp(0.0, 0.60)
    soft_alpha = (0.30 * soft_force * (1.0 - 0.70 * torch.maximum(rigid_force, free_force))).clamp(0.0, 0.32)
    layer_probs = layer_probs * (1.0 - rigid_alpha.unsqueeze(0)) + rigid_target * rigid_alpha.unsqueeze(0)
    layer_probs = layer_probs * (1.0 - free_alpha.unsqueeze(0)) + free_target * free_alpha.unsqueeze(0)
    layer_probs = layer_probs * (1.0 - soft_alpha.unsqueeze(0)) + soft_target * soft_alpha.unsqueeze(0)
    layer_probs = _normalize_scores(layer_probs)
    layer = _hard_assign(layer_probs * fg.unsqueeze(0), palette)
    body_outline = (fg_edge * body_shell * (1.0 - 0.82 * cloth_shell)).clamp(0.0, 1.0)
    cloth_outline = (fg_edge * cloth_hem).clamp(0.0, 1.0)
    soft_outline = (fg_edge * boundary).clamp(0.0, 1.0)
    layer = layer * (1.0 - 0.34 * body_outline.unsqueeze(0)) + rigid_color * (0.34 * body_outline.unsqueeze(0))
    layer = layer * (1.0 - 0.28 * cloth_outline.unsqueeze(0)) + free_color * (0.28 * cloth_outline.unsqueeze(0))
    layer = layer * (1.0 - 0.18 * soft_outline.unsqueeze(0)) + soft_color * (0.18 * soft_outline.unsqueeze(0))
    return layer * fg.unsqueeze(0)


def _binding_map_names(config):
    names = config.get(
        'binding_map_names',
        ['layer', 'region', 'compact_semantic', 'body_prob', 'soft_prob', 'cloth_prob', 'semantic', 'temporal', 'thin'],
    )
    if isinstance(names, str):
        names = [name.strip() for name in names.split(',') if name.strip()]
    else:
        names = list(names)
    valid = {'layer', 'region', 'compact_semantic', 'body_prob', 'soft_prob', 'cloth_prob', 'semantic', 'temporal', 'thin'}
    return [name for name in names if name in valid]


def _estimate_scale(values, scale=None, quantile=0.95):
    values = values.detach()
    if values.numel() == 0:
        return values.new_tensor(1.0)
    if scale is None:
        return torch.quantile(values, quantile).clamp_min(1e-6)
    return torch.as_tensor(scale, device=values.device, dtype=values.dtype).clamp_min(1e-6)


def _get_layer_probs(pc):
    probs = getattr(pc, 'binding_weights', None)
    if probs is not None:
        return probs.detach()
    layer_ids = getattr(pc, 'binding_layer_ids', torch.zeros(pc.get_xyz.shape[0], device=pc.get_xyz.device, dtype=torch.long))
    return torch.nn.functional.one_hot(layer_ids.long().clamp_min(0), num_classes=3).float()


def _get_region_probs(pc):
    probs = getattr(pc, 'binding_region_probs', None)
    if probs is not None:
        return probs.detach()
    region_ids = getattr(pc, 'binding_region_ids', torch.zeros(pc.get_xyz.shape[0], device=pc.get_xyz.device, dtype=torch.long))
    return torch.nn.functional.one_hot(region_ids.long().clamp_min(0), num_classes=3).float()


def _resolve_visibility(pc, visibility_filter=None, radii=None):
    device = pc.get_xyz.device
    point_count = pc.get_xyz.shape[0]
    if visibility_filter is not None and visibility_filter.shape[0] == point_count:
        visible_mask = visibility_filter.detach().bool()
    else:
        visible_mask = torch.ones(point_count, device=device, dtype=torch.bool)

    weights = visible_mask.float()
    if radii is not None and radii.shape[0] == point_count:
        weights = weights * radii.detach().float().clamp_min(0.0)

    if float(weights.sum().item()) <= 0.0:
        weights = visible_mask.float()
    if float(weights.sum().item()) <= 0.0:
        visible_mask = torch.ones(point_count, device=device, dtype=torch.bool)
        weights = torch.ones(point_count, device=device, dtype=torch.float32)
    return visible_mask, weights


def _visible_values(values, visible_mask):
    if values.numel() == 0:
        return values
    if visible_mask.any():
        return values[visible_mask]
    return values


def _weighted_mean(values, weights):
    if values.numel() == 0:
        shape = values.shape[1:] if values.dim() > 1 else ()
        return torch.zeros(shape, device=weights.device, dtype=weights.dtype)
    denom = weights.sum().clamp_min(1e-6)
    view_shape = (weights.shape[0],) + (1,) * (values.dim() - 1)
    return (values * weights.view(view_shape)).sum(dim=0) / denom


def _weighted_hist(ids, num_classes, weights):
    hist = torch.zeros(num_classes, device=weights.device, dtype=weights.dtype)
    hist.scatter_add_(0, ids.long().clamp(0, num_classes - 1), weights)
    return hist


def _confidence_from_probs(probs):
    if probs.shape[-1] <= 1:
        return torch.ones(probs.shape[0], device=probs.device, dtype=probs.dtype)
    uniform = 1.0 / probs.shape[-1]
    confidence = (probs.max(dim=-1).values - uniform) / max(1.0 - uniform, 1e-6)
    return confidence.clamp(0.0, 1.0)


def _categorical_display_strength(probs, config, prefix):
    top_prob = probs.max(dim=-1).values
    confidence = _confidence_from_probs(probs)
    if prefix == 'layer':
        default_low, default_high = 0.46, 0.62
    elif prefix == 'region':
        default_low, default_high = 0.54, 0.70
    else:
        default_low, default_high = 0.60, 0.74
    low_thresh = float(config.get(f'binding_map_{prefix}_display_low', config.get('binding_map_display_low', default_low)))
    high_thresh = float(config.get(f'binding_map_{prefix}_display_high', config.get('binding_map_display_high', default_high)))
    low_thresh = min(low_thresh, high_thresh - 1e-3)
    blend = ((top_prob - low_thresh) / max(high_thresh - low_thresh, 1e-6)).clamp(0.0, 1.0)
    strong = ((top_prob - high_thresh) / max(1.0 - high_thresh, 1e-6)).clamp(0.0, 1.0)
    display_floor = float(config.get(f'binding_map_{prefix}_display_floor', config.get('binding_map_display_floor', 0.16 if prefix == 'region' else 0.20)))
    display = display_floor + (1.0 - display_floor) * ((0.18 + 0.82 * confidence) * blend)
    return top_prob, confidence, blend, strong, display.clamp(0.0, 1.0)


def _windowed_scalar_display(values, scale, gamma, config, prefix, default_low=0.20, default_high=0.90, default_threshold=0.35):
    raw = torch.clamp(values / scale, 0.0, 1.0)
    low = float(config.get(f'binding_map_{prefix}_display_low', config.get('binding_map_scalar_display_low', default_low)))
    high = float(config.get(f'binding_map_{prefix}_display_high', config.get('binding_map_scalar_display_high', default_high)))
    threshold = float(config.get(f'binding_map_{prefix}_display_threshold', config.get('binding_map_scalar_display_threshold', default_threshold)))
    low = min(low, high - 1e-3)
    windowed = ((raw - low) / max(high - low, 1e-6)).clamp(0.0, 1.0)
    visible = torch.where(windowed > threshold, ((windowed - threshold) / max(1.0 - threshold, 1e-6)).clamp(0.0, 1.0), torch.zeros_like(windowed))
    return visible.pow(gamma)


def _mix_prob_colors(probs, palette, config, prefix):
    mix_power = float(config.get(f'binding_map_{prefix}_mix_power', config.get('binding_map_mix_power', 0.65)))
    gray = float(config.get(f'binding_map_{prefix}_gray', config.get('binding_map_gray', 0.18)))
    weak_tint = float(config.get(f'binding_map_{prefix}_weak_tint', config.get('binding_map_weak_tint', 0.34)))

    vis_probs = probs.clamp_min(1e-6)
    if mix_power != 1.0:
        vis_probs = vis_probs.pow(mix_power)
        vis_probs = vis_probs / vis_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    top_prob, _, blend, strong, display_conf = _categorical_display_strength(probs, config, prefix)
    mixed_colors = torch.matmul(vis_probs, palette.to(probs.device))
    gray_color = torch.full_like(mixed_colors, gray) * (0.88 + 0.12 * top_prob.unsqueeze(-1))
    weak_colors = gray_color * (1.0 - weak_tint) + mixed_colors * weak_tint
    mid_colors = weak_colors * (1.0 - blend.unsqueeze(-1)) + mixed_colors * blend.unsqueeze(-1)
    colors = mid_colors * (1.0 - strong.unsqueeze(-1)) + mixed_colors * strong.unsqueeze(-1)
    colors = weak_colors * (1.0 - display_conf.unsqueeze(-1)) + colors * display_conf.unsqueeze(-1)
    return colors, display_conf


def _label_confidence_colors(probs, palette, config, prefix):
    gray = float(config.get(f'binding_map_{prefix}_gray', config.get('binding_map_gray', 0.18)))
    weak_tint = float(config.get(f'binding_map_{prefix}_weak_tint', config.get('binding_map_weak_tint', 0.34)))

    label_ids = probs.argmax(dim=-1)
    base_colors = palette.to(probs.device)[label_ids]
    top_prob, _, blend, strong, display_conf = _categorical_display_strength(probs, config, prefix)
    gray_color = torch.full_like(base_colors, gray) * (0.88 + 0.12 * top_prob.unsqueeze(-1))
    weak_colors = gray_color * (1.0 - weak_tint) + base_colors * weak_tint
    mid_colors = weak_colors * (1.0 - blend.unsqueeze(-1)) + base_colors * blend.unsqueeze(-1)
    colors = mid_colors * (1.0 - strong.unsqueeze(-1)) + base_colors * strong.unsqueeze(-1)
    colors = weak_colors * (1.0 - display_conf.unsqueeze(-1)) + colors * display_conf.unsqueeze(-1)
    return colors, display_conf, label_ids


def _binary_erode(mask, kernel_size=3):
    pad = kernel_size // 2
    return 1.0 - torch.nn.functional.max_pool2d(1.0 - mask, kernel_size, stride=1, padding=pad)


def _binary_dilate(mask, kernel_size=3):
    pad = kernel_size // 2
    return torch.nn.functional.max_pool2d(mask, kernel_size, stride=1, padding=pad)


def _binary_open(mask, kernel_size=3):
    return _binary_dilate(_binary_erode(mask, kernel_size), kernel_size)


def _binary_close(mask, kernel_size=3):
    return _binary_erode(_binary_dilate(mask, kernel_size), kernel_size)


def _light_label_morphology(image, palette, config, prefix):
    enabled = bool(config.get(f'binding_map_{prefix}_morphology', config.get('binding_map_morphology', True)))
    if not enabled:
        return image
    kernel_size = int(config.get(f'binding_map_{prefix}_morph_kernel', config.get('binding_map_morph_kernel', 3)))
    iterations = int(config.get(f'binding_map_{prefix}_morph_iter', config.get('binding_map_morph_iter', 1)))
    fg_threshold = float(config.get(f'binding_map_{prefix}_morph_fg_threshold', config.get('binding_map_morph_fg_threshold', 0.05)))
    if kernel_size <= 1 or iterations <= 0:
        return image

    device = image.device
    palette = palette.to(device=device, dtype=image.dtype)
    img = image.unsqueeze(0)
    fg = (img.sum(dim=1, keepdim=True) > fg_threshold).float()
    if fg.max().item() <= 0:
        return image

    pixels = img.permute(0, 2, 3, 1).reshape(-1, 3)
    dists = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(dim=-1)
    labels = dists.argmin(dim=-1).reshape(1, img.shape[2], img.shape[3])

    class_masks = []
    for cid in range(palette.shape[0]):
        mask = (labels == cid).float().unsqueeze(1) * fg
        refined = mask
        for _ in range(iterations):
            refined = _binary_open(refined, kernel_size)
            refined = _binary_close(refined, kernel_size)
        class_masks.append(refined)

    masks = torch.cat(class_masks, dim=1)
    masks = masks * fg
    winner = masks.argmax(dim=1, keepdim=True)
    hard = torch.zeros_like(masks)
    hard.scatter_(1, winner, 1.0)
    hard = hard * fg

    refined = torch.einsum('bchw,ck->bkhw', hard, palette)
    blend = float(config.get(f'binding_map_{prefix}_morph_blend', config.get('binding_map_morph_blend', 0.22)))
    return (1.0 - blend) * image + blend * refined.squeeze(0)


def _common_visibility_stats(pc, visible_mask, weights):
    return {
        'visible_ratio': float(visible_mask.float().mean().item()),
        'visible_count': int(visible_mask.sum().item()),
        'point_count': int(pc.get_xyz.shape[0]),
        'weight_sum': float(weights.sum().item()),
    }


def _binding_positions(pc):
    canonical_xyz = getattr(pc, 'canonical_xyz', None)
    if canonical_xyz is not None and canonical_xyz.shape[0] == pc.get_xyz.shape[0]:
        return canonical_xyz.detach()
    return pc.get_xyz.detach()


def _smooth_prob_tensor(probs, positions, config, prefix):
    smooth_k = int(config.get(f'binding_map_{prefix}_smooth_k', config.get('binding_map_smooth_k', 8)))
    smooth_alpha = float(config.get(f'binding_map_{prefix}_smooth_alpha', config.get('binding_map_smooth_alpha', 0.25)))
    confidence_aware = bool(config.get(f'binding_map_{prefix}_confidence_aware', config.get('binding_map_confidence_aware', True)))
    if smooth_k <= 0 or smooth_alpha <= 0.0 or probs.shape[0] <= 1:
        return probs

    k = min(smooth_k + 1, probs.shape[0])
    knn = ops.knn_points(positions.unsqueeze(0), positions.unsqueeze(0), K=k)
    if k <= 1:
        return probs

    nn_idx = knn.idx[0, :, 1:]
    nn_dists = torch.sqrt(knn.dists[0, :, 1:].clamp_min(1e-8))
    scale = torch.quantile(nn_dists.detach().reshape(-1), 0.5).clamp_min(1e-6)
    nn_weights = torch.exp(-nn_dists / scale)
    neigh = probs[nn_idx]
    smooth = (neigh * nn_weights.unsqueeze(-1)).sum(dim=1) / nn_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

    if confidence_aware:
        alpha = smooth_alpha * (1.0 - _confidence_from_probs(probs))
    else:
        alpha = probs.new_full((probs.shape[0],), smooth_alpha)
    return probs * (1.0 - alpha.unsqueeze(-1)) + smooth * alpha.unsqueeze(-1)


def _smooth_scalar_tensor(values, positions, config, prefix):
    smooth_k = int(config.get(f'binding_map_{prefix}_smooth_k', config.get('binding_map_scalar_smooth_k', config.get('binding_map_smooth_k', 8))))
    smooth_alpha = float(config.get(f'binding_map_{prefix}_smooth_alpha', config.get('binding_map_scalar_smooth_alpha', config.get('binding_map_smooth_alpha', 0.25))))
    if smooth_k <= 0 or smooth_alpha <= 0.0 or values.shape[0] <= 1:
        return values

    k = min(smooth_k + 1, values.shape[0])
    knn = ops.knn_points(positions.unsqueeze(0), positions.unsqueeze(0), K=k)
    if k <= 1:
        return values

    nn_idx = knn.idx[0, :, 1:]
    nn_dists = torch.sqrt(knn.dists[0, :, 1:].clamp_min(1e-8))
    scale = torch.quantile(nn_dists.detach().reshape(-1), 0.5).clamp_min(1e-6)
    nn_weights = torch.exp(-nn_dists / scale)
    neigh = values[nn_idx]
    smooth = (neigh * nn_weights).sum(dim=1) / nn_weights.sum(dim=1).clamp_min(1e-6)
    return values * (1.0 - smooth_alpha) + smooth * smooth_alpha


def _clean_prob_tensor(probs, positions, config, prefix):
    clean_k = int(config.get(f'binding_map_{prefix}_clean_k', config.get('binding_map_clean_k', 14)))
    clean_alpha = float(config.get(f'binding_map_{prefix}_clean_alpha', config.get('binding_map_clean_alpha', 0.55)))
    if clean_k <= 0 or clean_alpha <= 0.0 or probs.shape[0] <= 1:
        return probs

    k = min(clean_k + 1, probs.shape[0])
    knn = ops.knn_points(positions.unsqueeze(0), positions.unsqueeze(0), K=k)
    if k <= 1:
        return probs

    nn_idx = knn.idx[0, :, 1:]
    nn_dists = torch.sqrt(knn.dists[0, :, 1:].clamp_min(1e-8))
    scale = torch.quantile(nn_dists.detach().reshape(-1), 0.55).clamp_min(1e-6)
    nn_weights = torch.exp(-nn_dists / scale)
    neigh = probs[nn_idx]
    smooth = (neigh * nn_weights.unsqueeze(-1)).sum(dim=1) / nn_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    confidence = _confidence_from_probs(probs)
    alpha = clean_alpha * (1.0 - confidence).pow(1.35)
    blended = probs * (1.0 - alpha.unsqueeze(-1)) + smooth * alpha.unsqueeze(-1)
    return _renorm_probs(blended)


def _clean_scalar_tensor(values, positions, config, prefix):
    clean_k = int(config.get(f'binding_map_{prefix}_clean_k', config.get('binding_map_scalar_clean_k', config.get('binding_map_clean_k', 12))))
    clean_alpha = float(config.get(f'binding_map_{prefix}_clean_alpha', config.get('binding_map_scalar_clean_alpha', config.get('binding_map_clean_alpha', 0.22))))
    if clean_k <= 0 or clean_alpha <= 0.0 or values.shape[0] <= 1:
        return values

    k = min(clean_k + 1, values.shape[0])
    knn = ops.knn_points(positions.unsqueeze(0), positions.unsqueeze(0), K=k)
    if k <= 1:
        return values

    nn_idx = knn.idx[0, :, 1:]
    nn_dists = torch.sqrt(knn.dists[0, :, 1:].clamp_min(1e-8))
    scale = torch.quantile(nn_dists.detach().reshape(-1), 0.55).clamp_min(1e-6)
    nn_weights = torch.exp(-nn_dists / scale)
    neigh = values[nn_idx]
    smooth = (neigh * nn_weights).sum(dim=1) / nn_weights.sum(dim=1).clamp_min(1e-6)

    contrast = (values - smooth).abs()
    norm = contrast / contrast.quantile(0.8).clamp_min(1e-6)
    alpha = clean_alpha * (1.0 - norm.clamp(0.0, 1.0)).pow(1.2)
    return values * (1.0 - alpha) + smooth * alpha


def _rigid_cfg(config):
    try:
        return config.model.deformer.rigid
    except Exception:
        return {}


def _sigmoid_score(x):
    return torch.sigmoid(x)


def _joint_mask(joint_ids, indices):
    mask = torch.zeros_like(joint_ids, dtype=torch.float32)
    if not indices:
        return mask
    for idx in indices:
        mask = torch.where(joint_ids == idx, torch.ones_like(mask), mask)
    return mask


def _binding_aux(pc):
    n = pc.get_xyz.shape[0]
    device = pc.get_xyz.device
    dominant_joint = getattr(pc, 'binding_dominant_joint', torch.zeros(n, device=device, dtype=torch.long)).long()
    confidence = getattr(pc, 'binding_anchor_confidence', torch.full((n,), 0.5, device=device))
    surface_distance = getattr(pc, 'binding_surface_distance', torch.zeros(n, device=device))
    semantic_distance = getattr(pc, 'binding_semantic_distance', torch.zeros(n, device=device))
    thin_score = getattr(pc, 'binding_thin_score', torch.zeros(n, device=device))
    return dominant_joint, confidence, surface_distance, semantic_distance, thin_score


def _head_hairline_override(pc, positions, config):
    dominant_joint, _, _, _, _ = _binding_aux(pc)
    head_mask = _joint_mask(dominant_joint, [15])
    if head_mask.max().item() <= 0:
        return head_mask

    head_points = head_mask > 0.5
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    if head_points.any():
        head_top = torch.quantile(y[head_points], 0.82)
        head_bottom = torch.quantile(y[head_points], 0.10)
        head_height = (head_top - head_bottom).abs().clamp_min(0.02)
        head_center_x = torch.quantile(x[head_points], 0.50)
        head_center_z = torch.quantile(z[head_points], 0.50)
        head_span_x = (torch.quantile(x[head_points], 0.88) - torch.quantile(x[head_points], 0.12)).abs().clamp_min(0.018)
        head_span_z = (torch.quantile(z[head_points], 0.88) - torch.quantile(z[head_points], 0.12)).abs().clamp_min(0.018)
        radial = torch.sqrt(
            ((x - head_center_x) / (0.44 * head_span_x + 1e-6)).pow(2)
            + ((z - head_center_z) / (0.44 * head_span_z + 1e-6)).pow(2)
        )
        periphery = torch.sigmoid((radial - 0.94) / 0.12)
        central = torch.sigmoid((0.66 - radial) / 0.10)
        # Use a dome-shaped hairline: higher at the forehead center, lower near temples and ears.
        hairline_ratio = (0.70 + 0.08 * central - 0.06 * periphery).clamp(0.62, 0.82)
        hairline_y = head_bottom + hairline_ratio * head_height
        hairline_band = torch.sigmoid((y - hairline_y) / (0.020 * head_height + 1e-6))
        crown_cap = torch.sigmoid((y - (head_bottom + 0.82 * head_height)) / (0.018 * head_height + 1e-6))
        override = torch.maximum(hairline_band * (0.64 + 0.36 * periphery), crown_cap)
    else:
        override = torch.zeros_like(head_mask)

    return (head_mask * override).clamp(0.0, 1.0)


def _head_hair_support(pc, positions, config):
    dominant_joint, confidence, surface_distance, semantic_distance, thin_score = _binding_aux(pc)
    head_mask = _joint_mask(dominant_joint, [15])
    if head_mask.max().item() <= 0:
        return head_mask

    rigid_cfg = _rigid_cfg(config)
    width = max(float(rigid_cfg.get('region_transition_width', 0.015)), 1e-6)
    body_conf_thr = float(rigid_cfg.get('body_confidence_threshold', 0.7))
    body_surface_thr = float(rigid_cfg.get('body_surface_threshold', 0.018))
    body_sem_thr = float(rigid_cfg.get('body_semantic_threshold', 0.012))

    body_contact = _sigmoid_score((confidence - body_conf_thr) / width) * _sigmoid_score((body_surface_thr - surface_distance) / width)
    body_semantic = _sigmoid_score((body_sem_thr - semantic_distance) / width)

    head_points = head_mask > 0.5
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    if head_points.any():
        head_top = torch.quantile(y[head_points], 0.80)
        head_bottom = torch.quantile(y[head_points], 0.12)
        head_height = (head_top - head_bottom).abs().clamp_min(0.02)
        head_center_x = torch.quantile(x[head_points], 0.50)
        head_center_z = torch.quantile(z[head_points], 0.50)
        head_span_x = (torch.quantile(x[head_points], 0.88) - torch.quantile(x[head_points], 0.12)).abs().clamp_min(0.018)
        head_span_z = (torch.quantile(z[head_points], 0.88) - torch.quantile(z[head_points], 0.12)).abs().clamp_min(0.018)
        topness = torch.sigmoid((y - (head_bottom + 0.64 * head_height)) / (0.09 * head_height + 1e-6))
        top_cap = torch.sigmoid((y - (head_bottom + 0.72 * head_height)) / (0.07 * head_height + 1e-6))
        upper_band = torch.sigmoid((y - (head_bottom + 0.44 * head_height)) / (0.11 * head_height + 1e-6))
        lower_band = torch.sigmoid(((head_bottom + 0.52 * head_height) - y) / (0.09 * head_height + 1e-6))
        radial = torch.sqrt(
            ((x - head_center_x) / (0.44 * head_span_x + 1e-6)).pow(2)
            + ((z - head_center_z) / (0.44 * head_span_z + 1e-6)).pow(2)
        )
        periphery = torch.sigmoid((radial - 1.02) / 0.16)
        central = torch.sigmoid((0.72 - radial) / 0.14)
    else:
        topness = torch.zeros_like(head_mask)
        top_cap = torch.zeros_like(head_mask)
        upper_band = torch.zeros_like(head_mask)
        lower_band = torch.zeros_like(head_mask)
        periphery = torch.zeros_like(head_mask)
        central = torch.zeros_like(head_mask)

    crown_hair = topness * (0.58 + 0.42 * periphery)
    crown_core_hair = top_cap * central
    side_hair = upper_band * periphery
    face_core = central * lower_band * torch.clamp(0.60 * body_contact + 0.40 * body_semantic, 0.0, 1.0)
    jaw_face = lower_band * torch.clamp(0.52 * body_contact + 0.28 * body_semantic + 0.20 * (1.0 - periphery), 0.0, 1.0)

    support = head_mask * torch.clamp(
        0.22 * crown_hair
        + 0.18 * crown_core_hair
        + 0.16 * side_hair
        + 0.10 * topness * periphery
        + 0.14 * (1.0 - body_contact)
        + 0.10 * (1.0 - body_semantic)
        + 0.08 * thin_score
        + 0.08 * _sigmoid_score((surface_distance - 0.72 * body_surface_thr) / width)
        - 0.46 * face_core
        - 0.22 * jaw_face,
        0.0,
        1.0,
    )
    hairline_override = _head_hairline_override(pc, positions, config)
    forehead_release = head_mask * central * (1.0 - hairline_override)
    forehead_release = forehead_release * torch.clamp(0.35 + 0.65 * torch.maximum(body_contact, body_semantic), 0.0, 1.0)
    support = support * (1.0 - 0.92 * forehead_release)
    support = torch.maximum(support, 0.96 * hairline_override)
    return support.clamp(0.0, 1.0)


def _torso_strap_chest_priors(dominant_joint, positions):
    torso_mask = _joint_mask(dominant_joint, [3, 6, 9, 12])
    shoulder_mask = torch.clamp(torso_mask + _joint_mask(dominant_joint, [13, 14, 16, 17]), 0.0, 1.0)
    if torso_mask.max().item() <= 0:
        zeros = torso_mask
        return zeros, zeros

    torso_points = torso_mask > 0.5
    x = positions[:, 0]
    y = positions[:, 1]
    if torso_points.any():
        torso_top = torch.quantile(y[torso_points], 0.90)
        torso_bottom = torch.quantile(y[torso_points], 0.18)
        torso_height = (torso_top - torso_bottom).abs().clamp_min(0.04)
        torso_center_x = torch.quantile(x[torso_points], 0.50)
        torso_span_x = (torch.quantile(x[torso_points], 0.88) - torch.quantile(x[torso_points], 0.12)).abs().clamp_min(0.025)
        upper_band = torch.sigmoid((y - (torso_bottom + 0.46 * torso_height)) / (0.09 * torso_height + 1e-6))
        upper_band = upper_band * torch.sigmoid(((torso_bottom + 0.97 * torso_height) - y) / (0.11 * torso_height + 1e-6))
        clavicle_band = torch.sigmoid(((torso_bottom + 0.90 * torso_height) - y) / (0.07 * torso_height + 1e-6))
        side_dist = (x - torso_center_x).abs()
        strap_side = torch.sigmoid((side_dist - 0.18 * torso_span_x) / (0.05 * torso_span_x + 1e-6))
        strap_side = strap_side * torch.sigmoid(((0.46 * torso_span_x) - side_dist) / (0.06 * torso_span_x + 1e-6))
        chest_center = torch.sigmoid(((0.30 * torso_span_x) - side_dist) / (0.06 * torso_span_x + 1e-6))
    else:
        upper_band = torch.zeros_like(torso_mask)
        clavicle_band = torch.zeros_like(torso_mask)
        strap_side = torch.zeros_like(torso_mask)
        chest_center = torch.zeros_like(torso_mask)

    strap_prior = shoulder_mask * upper_band * clavicle_band * strap_side
    chest_prior = torso_mask * upper_band * chest_center
    return strap_prior.clamp(0.0, 1.0), chest_prior.clamp(0.0, 1.0)


def _renorm_probs(probs):
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def _sharpen_probs(probs, power):
    if power <= 1.0:
        return _renorm_probs(probs)
    return _renorm_probs(probs.clamp_min(1e-6).pow(power))


def _exclusive_bodycloth_probs(probs, pair_strength=0.68, soft_share=0.28, interior_suppress=0.40):
    probs = _renorm_probs(probs)
    body = probs[:, 0]
    soft = probs[:, 1]
    cloth = probs[:, 2]

    shared = torch.sqrt((body * cloth).clamp_min(0.0))
    body_orig = body
    cloth_orig = cloth

    body = torch.clamp(body_orig * (1.0 - pair_strength * cloth_orig), 0.0, 1.0)
    cloth = torch.clamp(cloth_orig * (1.0 - pair_strength * body_orig), 0.0, 1.0)
    soft = torch.clamp(soft + soft_share * shared, 0.0, 1.0)

    probs = _renorm_probs(torch.stack([body, soft, cloth], dim=-1))
    body = probs[:, 0]
    soft = probs[:, 1]
    cloth = probs[:, 2]

    dominance_gap = (body - cloth).abs()
    dominant_max = torch.maximum(body, cloth)
    interior = _sigmoid_score((dominant_max - 0.44) / 0.05) * _sigmoid_score((dominance_gap - 0.16) / 0.05)
    soft = soft * (1.0 - interior_suppress * interior)
    probs = _renorm_probs(torch.stack([body, soft, cloth], dim=-1))
    return probs


def _calibrated_region_probs(pc, region_probs, config):
    rigid_cfg = _rigid_cfg(config)
    width = max(float(rigid_cfg.get('region_transition_width', 0.015)), 1e-6)
    body_conf_thr = float(rigid_cfg.get('body_confidence_threshold', 0.7))
    body_surface_thr = float(rigid_cfg.get('body_surface_threshold', 0.018))
    body_sem_thr = float(rigid_cfg.get('body_semantic_threshold', 0.012))
    cloth_surface_thr = float(rigid_cfg.get('cloth_surface_threshold', 0.02))
    cloth_sem_thr = float(rigid_cfg.get('cloth_semantic_threshold', 0.04))

    dominant_joint, confidence, surface_distance, semantic_distance, thin_score = _binding_aux(pc)
    positions = _binding_positions(pc)
    shorts_core_mask = _joint_mask(dominant_joint, [0, 1, 2])
    upper_leg_mask = _joint_mask(dominant_joint, [4, 5])
    lower_leg_mask = _joint_mask(dominant_joint, [7, 8])
    foot_mask = _joint_mask(dominant_joint, [10, 11])
    torso_mask = _joint_mask(dominant_joint, [3, 6, 9, 12])
    arm_mask = _joint_mask(dominant_joint, [13, 14, 16, 17, 18, 19, 20, 21, 22, 23])
    distal_arm_mask = _joint_mask(dominant_joint, [18, 19, 20, 21, 22, 23])
    hand_mask = _joint_mask(dominant_joint, [20, 21, 22, 23])
    head_mask = _joint_mask(dominant_joint, [15])
    head_hair = _head_hair_support(pc, positions, config)
    head_core = (head_mask * (1.0 - 0.93 * head_hair)).clamp(0.0, 1.0)
    strap_prior, chest_prior = _torso_strap_chest_priors(dominant_joint, positions)

    body_contact = _sigmoid_score((confidence - body_conf_thr) / width) * _sigmoid_score((body_surface_thr - surface_distance) / width)
    body_semantic = _sigmoid_score((body_sem_thr - semantic_distance) / width)
    cloth_surface = _sigmoid_score((surface_distance - cloth_surface_thr) / width)
    cloth_semantic = _sigmoid_score((semantic_distance - cloth_sem_thr) / width)

    shorts_core_evidence = shorts_core_mask * torch.clamp(0.75 * cloth_semantic + 0.10 * cloth_surface + 0.15 * thin_score, 0.0, 1.0)
    upper_leg_cloth = upper_leg_mask * torch.clamp(0.62 * cloth_semantic + 0.18 * cloth_surface + 0.12 * thin_score - 0.28 * body_contact, 0.0, 1.0)
    shorts_evidence = torch.clamp(shorts_core_evidence + 0.65 * upper_leg_cloth, 0.0, 1.0)
    torso_cloth = torso_mask * torch.clamp(0.70 * cloth_semantic + 0.10 * cloth_surface + 0.20 * thin_score, 0.0, 1.0) * (1.0 - 0.45 * body_contact)
    arm_edge_cloth = distal_arm_mask * torch.clamp(0.40 * cloth_semantic + 0.22 * cloth_surface + 0.18 * thin_score - 0.16 * body_contact, 0.0, 1.0)
    leg_body_support = torch.clamp(upper_leg_mask + lower_leg_mask, 0.0, 1.0)
    leg_body_support = leg_body_support * _sigmoid_score((confidence - 0.42) / width) * _sigmoid_score(((body_surface_thr * 1.95) - surface_distance) / width)
    leg_body_support = leg_body_support * (1.0 - 0.38 * torch.clamp(0.75 * cloth_semantic + 0.25 * thin_score, 0.0, 1.0))
    hand_body_support = hand_mask * _sigmoid_score((confidence - 0.34) / width) * _sigmoid_score(((body_surface_thr * 1.80) - surface_distance) / width)
    hand_body_support = hand_body_support * (0.78 + 0.22 * body_semantic) * (1.0 - 0.28 * cloth_semantic)
    foot_cloth_support = foot_mask * torch.clamp(0.56 + 0.22 * cloth_surface + 0.18 * thin_score + 0.12 * (1.0 - body_contact), 0.0, 1.0)
    lower_leg_cloth_residual = lower_leg_mask * torch.clamp(
        0.48 * cloth_semantic + 0.16 * cloth_surface + 0.18 * thin_score - 0.10 * body_contact,
        0.0,
        1.0,
    )
    torso_body = torso_mask * body_contact * (0.55 + 0.45 * body_semantic)
    arm_body = arm_mask * body_contact * (0.60 + 0.40 * body_semantic)
    head_body = head_core * body_contact * (0.65 + 0.35 * body_semantic)
    head_body_support = head_core * torch.clamp(0.58 + 0.24 * body_contact + 0.20 * body_semantic - 0.18 * cloth_semantic, 0.0, 1.0)
    head_hair_soft = head_hair * torch.clamp(0.72 + 0.12 * thin_score + 0.10 * (1.0 - body_contact), 0.0, 1.0)

    body_prob = region_probs[:, 0]
    soft_prob = region_probs[:, 1]
    cloth_prob = region_probs[:, 2]

    shorts_body_transfer = 0.55 * shorts_evidence * body_prob
    torso_body_transfer = 0.30 * torso_cloth * body_prob
    shorts_soft_transfer = 0.30 * shorts_evidence * soft_prob
    torso_soft_transfer = 0.18 * torso_cloth * soft_prob

    body_prob = torch.clamp(body_prob - shorts_body_transfer - torso_body_transfer, min=0.0)
    soft_prob = torch.clamp(soft_prob - shorts_soft_transfer - torso_soft_transfer + 0.12 * (shorts_body_transfer + torso_body_transfer), min=0.0)
    cloth_prob = torch.clamp(
        cloth_prob
        + 0.88 * shorts_body_transfer
        + 0.82 * torso_body_transfer
        + shorts_soft_transfer
        + torso_soft_transfer
        + 0.30 * shorts_evidence
        + 0.18 * torso_cloth,
        min=0.0,
    )

    foot_body_transfer = 0.92 * foot_cloth_support * body_prob
    foot_soft_transfer = 0.34 * foot_cloth_support * soft_prob
    hand_cloth_transfer = 0.52 * hand_body_support * cloth_prob
    hand_soft_transfer = 0.20 * hand_body_support * soft_prob
    body_prob = torch.clamp(body_prob - foot_body_transfer, min=0.0)
    soft_prob = torch.clamp(soft_prob - foot_soft_transfer + 0.10 * foot_body_transfer, min=0.0)
    cloth_prob = torch.clamp(cloth_prob + 0.90 * foot_body_transfer + foot_soft_transfer + 0.25 * foot_cloth_support, min=0.0)

    head_cloth_transfer = 0.40 * head_body_support * cloth_prob
    head_body_transfer = 0.60 * head_hair_soft * body_prob
    body_prob = torch.clamp(body_prob - head_body_transfer + head_cloth_transfer + 0.10 * head_body_support, min=0.0)
    soft_prob = soft_prob + 0.80 * head_body_transfer + 0.22 * head_hair_soft
    cloth_prob = torch.clamp(cloth_prob - head_cloth_transfer - 0.74 * head_hair_soft * cloth_prob, min=0.0)

    torso_overlap = torch.sqrt((region_probs[:, 0] * region_probs[:, 2]).clamp_min(0.0))
    strap_cloth_hint = strap_prior * torch.clamp(0.24 * region_probs[:, 2] + 0.30 * cloth_semantic + 0.24 * thin_score + 0.14 * torso_overlap + 0.10 * cloth_surface - 0.08 * body_contact, 0.0, 1.0)
    chest_body_hint = chest_prior * torch.clamp(0.34 * body_contact + 0.28 * body_semantic + 0.18 * region_probs[:, 0] - 0.16 * cloth_semantic - 0.18 * thin_score - 0.10 * region_probs[:, 2], 0.0, 1.0)
    body_prob = torch.clamp(body_prob + 0.34 * chest_body_hint - 0.18 * strap_cloth_hint, min=0.0)
    soft_prob = soft_prob + 0.06 * torch.minimum(strap_cloth_hint, chest_body_hint)
    cloth_prob = torch.clamp(cloth_prob + 0.34 * strap_cloth_hint - 0.24 * chest_body_hint, min=0.0)

    body_boost = 0.34 * torso_body + 0.32 * arm_body + 0.82 * leg_body_support + 0.42 * hand_body_support + 0.26 * head_body + 0.12 * head_body_support
    lower_leg_cloth_transfer = 0.72 * leg_body_support * cloth_prob
    lower_leg_soft_transfer = 0.18 * leg_body_support * soft_prob
    body_prob = body_prob + body_boost + lower_leg_cloth_transfer + lower_leg_soft_transfer + 0.18 * leg_body_support + hand_cloth_transfer + hand_soft_transfer
    soft_prob = torch.clamp(soft_prob - lower_leg_soft_transfer - hand_soft_transfer, min=0.0)
    cloth_prob = torch.clamp(cloth_prob - lower_leg_cloth_transfer - hand_cloth_transfer, min=0.0)
    soft_prob = soft_prob * (1.0 - 0.35 * (shorts_evidence + torso_cloth).clamp(max=1.0))
    soft_prob = soft_prob * (1.0 - 0.18 * body_boost.clamp(max=1.0)).clamp(min=0.04) + 0.34 * head_hair_soft
    cloth_prob = cloth_prob * (1.0 - 0.68 * leg_body_support - 0.42 * hand_body_support - 0.34 * head_core).clamp(min=0.01)
    cloth_prob = cloth_prob + 0.10 * lower_leg_cloth_residual * (1.0 - leg_body_support) + 0.12 * arm_edge_cloth * (1.0 - hand_body_support)

    calibrated = torch.stack([body_prob, soft_prob, cloth_prob], dim=-1)
    calibrated = _exclusive_bodycloth_probs(calibrated, pair_strength=0.74, soft_share=0.26, interior_suppress=0.42)
    return _sharpen_probs(calibrated, 2.32)


def _calibrated_layer_probs(pc, layer_probs, region_probs, config):
    rigid_cfg = _rigid_cfg(config)
    width = max(float(rigid_cfg.get('region_transition_width', 0.015)), 1e-6)
    body_conf_thr = float(rigid_cfg.get('body_confidence_threshold', 0.7))
    body_surface_thr = float(rigid_cfg.get('body_surface_threshold', 0.018))
    cloth_surface_thr = float(rigid_cfg.get('cloth_surface_threshold', 0.02))

    dominant_joint, confidence, surface_distance, semantic_distance, thin_score = _binding_aux(pc)
    positions = _binding_positions(pc)
    shorts_core_mask = _joint_mask(dominant_joint, [0, 1, 2])
    shorts_mask = _joint_mask(dominant_joint, [0, 1, 2, 4, 5])
    upper_leg_mask = _joint_mask(dominant_joint, [4, 5])
    lower_leg_mask = _joint_mask(dominant_joint, [7, 8])
    foot_mask = _joint_mask(dominant_joint, [10, 11])
    torso_mask = _joint_mask(dominant_joint, [3, 6, 9, 12])
    arm_mask = _joint_mask(dominant_joint, [13, 14, 16, 17, 18, 19, 20, 21, 22, 23])
    distal_arm_mask = _joint_mask(dominant_joint, [18, 19, 20, 21, 22, 23])
    hand_mask = _joint_mask(dominant_joint, [20, 21, 22, 23])
    head_mask = _joint_mask(dominant_joint, [15])
    head_hair = _head_hair_support(pc, positions, config)
    head_core = (head_mask * (1.0 - 0.93 * head_hair)).clamp(0.0, 1.0)

    body_contact = _sigmoid_score((confidence - body_conf_thr) / width) * _sigmoid_score((body_surface_thr - surface_distance) / width)
    cloth_surface = _sigmoid_score((surface_distance - cloth_surface_thr) / width)
    cloth_semantic = _sigmoid_score((semantic_distance - float(rigid_cfg.get('cloth_semantic_threshold', 0.04))) / width)

    shorts_evidence = shorts_mask * torch.clamp(0.86 * cloth_semantic + 0.14 * cloth_surface + 0.12 * thin_score + 0.12 * region_probs[:, 2], 0.0, 1.0)
    shorts_core_uniform = shorts_core_mask * torch.clamp(0.60 * region_probs[:, 2] + 0.32 * cloth_semantic + 0.18 * thin_score, 0.0, 1.0)
    torso_free = torso_mask * torch.clamp(0.58 * cloth_semantic + 0.14 * cloth_surface + 0.22 * thin_score + 0.14 * region_probs[:, 2], 0.0, 1.0)
    cloth_edge_mask = torch.clamp(torso_mask + distal_arm_mask, 0.0, 1.0)
    cloth_edge_free = cloth_edge_mask * torch.clamp(0.48 * cloth_surface + 0.40 * cloth_semantic + 0.32 * thin_score + 0.24 * region_probs[:, 2] - 0.14 * region_probs[:, 0], 0.0, 1.0)
    foot_free = foot_mask * torch.clamp(0.78 + 0.16 * cloth_surface + 0.14 * thin_score + 0.16 * (1.0 - body_contact), 0.0, 1.0)
    head_free = head_hair * torch.clamp(0.72 + 0.18 * thin_score + 0.16 * (1.0 - body_contact), 0.0, 1.0)

    torso_rigid = torso_mask * body_contact
    arm_rigid = arm_mask * body_contact
    distal_arm_rigid = distal_arm_mask * _sigmoid_score((confidence - 0.36) / width) * _sigmoid_score(((body_surface_thr * 1.85) - surface_distance) / width) * (1.0 - 0.22 * cloth_semantic)
    head_rigid = head_core * body_contact
    lower_leg_body = lower_leg_mask * torch.clamp(0.60 + 0.36 * body_contact + 0.34 * region_probs[:, 0] + 0.22 * _sigmoid_score(((body_surface_thr * 2.00) - surface_distance) / width), 0.0, 1.0) * (1.0 - 0.10 * region_probs[:, 2])
    hand_body = hand_mask * _sigmoid_score((confidence - 0.34) / width) * _sigmoid_score(((body_surface_thr * 1.60) - surface_distance) / width) * (0.72 + 0.28 * region_probs[:, 0]) * (1.0 - 0.24 * region_probs[:, 2])
    upper_leg_soft = upper_leg_mask * torch.clamp(0.24 * region_probs[:, 2] + 0.18 * cloth_semantic + 0.24 * (1.0 - body_contact), 0.0, 1.0)

    rigid_prob = layer_probs[:, 0]
    soft_prob = layer_probs[:, 1]
    free_prob = layer_probs[:, 2]

    rigid_evidence = torch.clamp(1.04 * torso_rigid + 0.66 * arm_rigid + 0.52 * distal_arm_rigid + 1.78 * lower_leg_body + 0.58 * hand_body + 0.56 * head_rigid + 0.22 * region_probs[:, 0] - 0.28 * foot_free - 0.40 * head_free, 0.0, 1.0)
    free_evidence = torch.clamp(1.22 * shorts_evidence + 0.36 * shorts_core_uniform + 0.42 * torso_free + 0.74 * cloth_edge_free + 0.94 * foot_free + 0.76 * head_free + 0.30 * cloth_surface + 0.34 * thin_score + 0.18 * region_probs[:, 2] - 0.86 * lower_leg_body, 0.0, 1.0)

    soft_to_rigid = 0.74 * soft_prob * rigid_evidence
    soft_to_free = 0.82 * soft_prob * free_evidence
    shorts_soft_to_free = 0.34 * soft_prob * shorts_evidence
    lower_leg_free_to_rigid = 0.72 * free_prob * lower_leg_body

    rigid_prob = rigid_prob + soft_to_rigid + lower_leg_free_to_rigid + 0.52 * rigid_evidence + 0.18 * region_probs[:, 0] * (1.0 - free_evidence) + 0.34 * lower_leg_body
    free_prob = free_prob + soft_to_free + shorts_soft_to_free + 0.46 * free_evidence + 0.14 * region_probs[:, 2] * (1.0 - rigid_evidence) + 0.34 * head_free
    soft_prob = torch.clamp(soft_prob - soft_to_rigid - soft_to_free - shorts_soft_to_free + 0.10 * upper_leg_soft + 0.12 * head_hair, min=0.0)
    free_prob = torch.clamp(free_prob - lower_leg_free_to_rigid, min=0.0)

    soft_prob = soft_prob * (1.0 - 0.24 * (rigid_evidence + free_evidence).clamp(max=1.0)).clamp(min=0.02)
    free_prob = free_prob * (1.0 - 0.78 * lower_leg_body).clamp(min=0.03) + 0.12 * shorts_core_uniform

    calibrated = torch.stack([rigid_prob, soft_prob, free_prob], dim=-1)
    calibrated = _renorm_probs(calibrated)
    dominance_gap = (calibrated[:, 0] - calibrated[:, 2]).abs()
    dominant_max = torch.maximum(calibrated[:, 0], calibrated[:, 2])
    interior = _sigmoid_score((dominant_max - 0.42) / 0.05) * _sigmoid_score((dominance_gap - 0.14) / 0.05)
    calibrated[:, 1] = calibrated[:, 1] * (1.0 - 0.34 * interior)
    calibrated[:, 0] = calibrated[:, 0] + 0.12 * interior * _sigmoid_score((calibrated[:, 0] - calibrated[:, 2] - 0.04) / 0.04)
    calibrated[:, 2] = calibrated[:, 2] + 0.12 * interior * _sigmoid_score((calibrated[:, 2] - calibrated[:, 0] - 0.04) / 0.04)
    return _sharpen_probs(calibrated, 3.15)


def _foot_neighbor_support(pc, positions, region_probs, config):
    if region_probs.shape[0] <= 1:
        return region_probs.new_zeros(region_probs.shape[0])

    k = min(int(config.get('binding_map_foot_neighbor_k', 20)) + 1, region_probs.shape[0])
    if k <= 1:
        return region_probs.new_zeros(region_probs.shape[0])

    dominant_joint, confidence, surface_distance, _, _ = _binding_aux(pc)
    foot_seed = _joint_mask(dominant_joint, [10, 11])
    foot_seed = torch.clamp(foot_seed * (0.75 + 0.25 * region_probs[:, 2]) * (0.70 + 0.30 * (1.0 - region_probs[:, 0])), 0.0, 1.0)

    knn = ops.knn_points(positions.unsqueeze(0), positions.unsqueeze(0), K=k)
    nn_idx = knn.idx[0, :, 1:]
    nn_dists = torch.sqrt(knn.dists[0, :, 1:].clamp_min(1e-8))
    scale = torch.quantile(nn_dists.detach().reshape(-1), 0.55).clamp_min(1e-6)
    nn_weights = torch.exp(-nn_dists / scale)

    neigh = foot_seed[nn_idx]
    neigh_score = (neigh * nn_weights).sum(dim=1) / nn_weights.sum(dim=1).clamp_min(1e-6)
    cloth_neigh = (region_probs[:, 2][nn_idx] * nn_weights).sum(dim=1) / nn_weights.sum(dim=1).clamp_min(1e-6)

    # Shoes are usually low, close to foot seeds, cloth-dominant, and weak in body contact.
    low_surface = torch.sigmoid(((float(_rigid_cfg(config).get('body_surface_threshold', 0.018)) * 2.4) - surface_distance) / max(float(_rigid_cfg(config).get('region_transition_width', 0.015)), 1e-6))
    y_coord = positions[:, 1]
    seed_mask = foot_seed > 0.18
    if seed_mask.any():
        foot_top = torch.quantile(y_coord[seed_mask], 0.88)
        foot_bottom = torch.quantile(y_coord[seed_mask], 0.12)
        foot_height = (foot_top - foot_bottom).abs().clamp_min(0.015)
        low_band = torch.sigmoid((foot_top + 0.30 * foot_height - y_coord) / (0.65 * foot_height + 1e-6))
    else:
        low_band = torch.ones_like(neigh_score)

    support = neigh_score * (0.42 + 0.34 * region_probs[:, 2] + 0.24 * cloth_neigh) * (1.0 - 0.55 * region_probs[:, 0]) * (0.55 + 0.45 * low_surface) * (0.35 + 0.65 * low_band)
    support = torch.maximum(
        support,
        0.74 * cloth_neigh * low_band * (1.0 - 0.62 * region_probs[:, 0]),
    )
    return support.clamp(0.0, 1.0)


def _region_neighbor_consensus(pc, region_probs, positions, config):
    if region_probs.shape[0] <= 1:
        return region_probs

    foot_neighbor = _foot_neighbor_support(pc, positions, region_probs, config)
    head_hairline = _head_hairline_override(pc, positions, config)
    head_hair = torch.maximum(_head_hair_support(pc, positions, config), head_hairline)

    k = min(int(config.get('binding_map_region_consensus_k', 18)) + 1, region_probs.shape[0])
    if k <= 1:
        return region_probs

    knn = ops.knn_points(positions.unsqueeze(0), positions.unsqueeze(0), K=k)
    nn_idx = knn.idx[0, :, 1:]
    nn_dists = torch.sqrt(knn.dists[0, :, 1:].clamp_min(1e-8))
    scale = torch.quantile(nn_dists.detach().reshape(-1), 0.55).clamp_min(1e-6)
    nn_weights = torch.exp(-nn_dists / scale)
    neigh = region_probs[nn_idx]
    neigh_mean = (neigh * nn_weights.unsqueeze(-1)).sum(dim=1) / nn_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    neigh_mean = _renorm_probs(neigh_mean)

    neigh_top = neigh_mean.max(dim=-1).values
    neigh_sorted, neigh_ids = neigh_mean.sort(dim=-1, descending=True)
    neigh_gap = neigh_sorted[:, 0] - neigh_sorted[:, 1]
    entropy = -(neigh_mean.clamp_min(1e-6) * neigh_mean.clamp_min(1e-6).log()).sum(dim=-1) / math.log(3.0)

    dominant = neigh_ids[:, 0]
    one_hot = torch.zeros_like(region_probs)
    one_hot.scatter_(1, dominant.unsqueeze(-1), 1.0)

    dominant_joint, _, _, _, _ = _binding_aux(pc)
    foot_mask = _joint_mask(dominant_joint, [10, 11]).unsqueeze(-1)
    hand_mask = _joint_mask(dominant_joint, [20, 21, 22, 23])
    lower_leg_mask = _joint_mask(dominant_joint, [7, 8])
    head_mask = _joint_mask(dominant_joint, [15])
    head_core_mask = (head_mask * (1.0 - 0.93 * head_hair)).unsqueeze(-1)
    head_hair_mask = head_hair.unsqueeze(-1)
    foot_cloth = torch.zeros_like(region_probs)
    foot_cloth[:, 2] = 1.0
    head_body = torch.zeros_like(region_probs)
    head_body[:, 0] = 1.0
    head_soft = torch.zeros_like(region_probs)
    head_soft[:, 1] = 1.0
    one_hot = torch.where(foot_mask > 0, foot_cloth, one_hot)
    one_hot = torch.where(head_core_mask > 0, head_body, one_hot)
    one_hot = torch.where(head_hair_mask > 0, head_soft, one_hot)

    consensus_strength = (0.72 * neigh_top + 0.28 * neigh_gap).clamp(0.0, 1.0)
    mixed_strength = (0.75 * entropy + 0.25 * (1.0 - neigh_gap.clamp(0.0, 1.0))).clamp(0.0, 1.0)

    snap_alpha = float(config.get('binding_map_region_consensus_alpha', 0.64)) * consensus_strength * (1.0 - 0.55 * mixed_strength)
    soft_alpha = float(config.get('binding_map_region_soft_mix_alpha', 0.54)) * torch.clamp(mixed_strength - 0.18, min=0.0) / 0.82

    probs = region_probs * (1.0 - snap_alpha.unsqueeze(-1)) + one_hot * snap_alpha.unsqueeze(-1)
    probs[:, 0] = probs[:, 0] * (1.0 - 0.18 * soft_alpha)
    probs[:, 2] = probs[:, 2] * (1.0 - 0.18 * soft_alpha)
    probs[:, 1] = probs[:, 1] + soft_alpha

    shoe_alpha = float(config.get('binding_map_region_shoe_alpha', 0.78)) * foot_neighbor
    probs[:, 0] = probs[:, 0] * (1.0 - 0.92 * shoe_alpha)
    probs[:, 1] = probs[:, 1] * (1.0 - 0.28 * shoe_alpha)
    probs[:, 2] = probs[:, 2] + shoe_alpha

    hand_body_alpha = 0.42 * hand_mask * _sigmoid_score((probs[:, 0] - probs[:, 2] + 0.02) / 0.04) * (1.0 - 0.75 * foot_neighbor)
    probs[:, 2] = probs[:, 2] * (1.0 - 0.65 * hand_body_alpha)
    probs[:, 1] = probs[:, 1] * (1.0 - 0.34 * hand_body_alpha)
    probs[:, 0] = probs[:, 0] + hand_body_alpha

    lower_leg_body_alpha = 0.22 * lower_leg_mask * _sigmoid_score((probs[:, 0] - probs[:, 2] - 0.02) / 0.05) * (1.0 - 0.65 * foot_neighbor)
    probs[:, 2] = probs[:, 2] * (1.0 - 0.38 * lower_leg_body_alpha)
    probs[:, 1] = probs[:, 1] * (1.0 - 0.24 * lower_leg_body_alpha)
    probs[:, 0] = probs[:, 0] + lower_leg_body_alpha

    probs[:, 0] = probs[:, 0] * (1.0 - 0.88 * head_hair)
    probs[:, 2] = probs[:, 2] * (1.0 - 0.96 * head_hair)
    probs[:, 1] = probs[:, 1] + 0.90 * head_hair
    probs = _renorm_probs(probs)
    hairline_alpha = head_hairline.clamp(0.0, 1.0).unsqueeze(-1)
    probs = probs * (1.0 - hairline_alpha) + head_soft * hairline_alpha
    return _renorm_probs(probs)


def _display_region_probs(pc, region_probs, positions, config):
    region_probs = _region_neighbor_consensus(pc, region_probs, positions, config)
    dominant_joint, confidence, surface_distance, semantic_distance, thin_score = _binding_aux(pc)
    strap_prior, chest_prior = _torso_strap_chest_priors(dominant_joint, positions)
    rigid_cfg = _rigid_cfg(config)
    width = max(float(rigid_cfg.get('region_transition_width', 0.015)), 1e-6)
    body_surface_thr = float(rigid_cfg.get('body_surface_threshold', 0.018))
    cloth_surface_thr = float(rigid_cfg.get('cloth_surface_threshold', 0.02))
    body_sem_thr = float(rigid_cfg.get('body_semantic_threshold', 0.012))
    cloth_sem_thr = float(rigid_cfg.get('cloth_semantic_threshold', 0.04))
    body_contact = _sigmoid_score((confidence - float(rigid_cfg.get('body_confidence_threshold', 0.7))) / width)
    body_contact = body_contact * _sigmoid_score((body_surface_thr - surface_distance) / width)
    body_semantic = _sigmoid_score((body_sem_thr - semantic_distance) / width)
    cloth_surface = _sigmoid_score((surface_distance - cloth_surface_thr) / width)
    cloth_semantic = _sigmoid_score((semantic_distance - cloth_sem_thr) / width)
    head_hair = _head_hair_support(pc, positions, config)
    body = region_probs[:, 0] * (1.0 - 0.88 * head_hair)
    soft = region_probs[:, 1] + 0.86 * head_hair
    cloth = region_probs[:, 2] * (1.0 - 0.96 * head_hair)

    body_scalar = _calibrated_scalar_values('body_prob', region_probs, _get_layer_probs(pc), pc, config)
    cloth_scalar = _calibrated_scalar_values('cloth_prob', region_probs, _get_layer_probs(pc), pc, config)
    body_gap = body_scalar - cloth_scalar
    body_scalar_support = _sigmoid_score((body_gap - 0.04) / 0.035)
    cloth_scalar_support = _sigmoid_score((-body_gap - 0.05) / 0.040)

    torso_mask = _joint_mask(dominant_joint, [3, 6, 9, 12])
    upper_limb_mask = _joint_mask(dominant_joint, [13, 14, 16, 17, 18, 19])
    lower_limb_mask = _joint_mask(dominant_joint, [4, 5, 7, 8])
    hand_mask = _joint_mask(dominant_joint, [20, 21, 22, 23])
    foot_mask = _joint_mask(dominant_joint, [10, 11])
    body_zone = torch.clamp(torso_mask + upper_limb_mask + lower_limb_mask + hand_mask, 0.0, 1.0)
    body_scalar_peak = _sigmoid_score((body_scalar - 0.46) / 0.060)
    cloth_scalar_peak = _sigmoid_score((cloth_scalar - 0.48) / 0.060)
    body_scalar_clean = _sigmoid_score((body_gap - 0.08) / 0.026)
    exposed_body_support = body_scalar_support * (0.62 * torso_mask + 0.28 * upper_limb_mask + 0.26 * lower_limb_mask + 0.34 * hand_mask)
    exposed_body_support = exposed_body_support * (0.68 + 0.32 * body_contact) * (0.62 + 0.38 * body_semantic)
    exposed_body_support = exposed_body_support * (1.0 - 0.58 * cloth_scalar_support)
    body_anchor_support = body_zone * body_scalar_peak * body_scalar_clean
    body_anchor_support = body_anchor_support * (0.72 + 0.28 * body_contact) * (0.66 + 0.34 * body_semantic)
    body_anchor_support = body_anchor_support * (1.0 - 0.70 * cloth_scalar_peak) * (1.0 - 0.96 * head_hair) * (1.0 - 0.92 * foot_mask)
    body_anchor_support = body_anchor_support * (1.0 - 0.22 * thin_score)

    chest_body_hint = chest_prior * torch.clamp(0.42 * body_contact + 0.24 * body_semantic + 0.22 * body + 0.34 * body_scalar_support - 0.14 * cloth_semantic - 0.10 * thin_score, 0.0, 1.0)
    chest_open_body = chest_prior * torch.clamp(0.54 * body_scalar_peak + 0.36 * body_scalar_clean + 0.18 * body_contact + 0.12 * body_semantic + 0.10 * body - 0.28 * cloth_scalar_peak - 0.14 * cloth_semantic - 0.10 * thin_score, 0.0, 1.0)
    strap_cloth_hint = strap_prior * torch.clamp(0.38 * cloth_semantic + 0.22 * cloth_surface + 0.16 * thin_score + 0.12 * cloth + 0.20 * cloth_scalar_support - 0.12 * body_contact - 0.14 * body_scalar_support - 0.18 * chest_open_body, 0.0, 1.0)

    body = torch.clamp(body + 0.32 * chest_body_hint + 0.54 * chest_open_body + 0.42 * exposed_body_support + 0.52 * body_anchor_support - 0.14 * strap_cloth_hint - 0.12 * cloth_scalar_support, 0.0, 1.0)
    cloth = torch.clamp(cloth + 0.34 * strap_cloth_hint + 0.10 * cloth_scalar_support - 0.28 * chest_body_hint - 0.40 * chest_open_body - 0.28 * exposed_body_support - 0.44 * body_anchor_support, 0.0, 1.0)
    soft = torch.clamp(soft + 0.05 * torch.minimum(chest_body_hint, strap_cloth_hint) + 0.08 * torch.sqrt((exposed_body_support * cloth).clamp_min(0.0)) - 0.16 * body_anchor_support - 0.18 * chest_open_body, 0.0, 1.0)
    region_probs = _renorm_probs(torch.stack([body, soft, cloth], dim=-1))
    body = region_probs[:, 0]
    soft = region_probs[:, 1]
    cloth = region_probs[:, 2]

    body_snap = body_anchor_support * (1.0 - 0.75 * cloth) * (1.0 - 0.72 * soft)
    region_probs[:, 0] = region_probs[:, 0] + 0.30 * body_snap
    region_probs[:, 1] = region_probs[:, 1] * (1.0 - 0.18 * body_snap)
    region_probs[:, 2] = region_probs[:, 2] * (1.0 - 0.34 * body_snap)
    region_probs = _renorm_probs(region_probs)
    body = region_probs[:, 0]
    soft = region_probs[:, 1]
    cloth = region_probs[:, 2]

    dominance_gap = (body - cloth).abs()
    overlap = torch.sqrt((body * cloth).clamp_min(0.0))
    boundary_band = overlap * _sigmoid_score((0.18 - dominance_gap) / 0.045)

    # Render soft as a narrow transition band near body/cloth interfaces,
    # instead of letting it flood large interior regions.
    soft_display = torch.clamp(0.01 * soft + 1.12 * boundary_band, 0.0, 1.0)
    suppress = (0.98 * soft_display).clamp(0.0, 0.995)

    body_display = torch.clamp(body * (1.0 - suppress), 0.0, 1.0)
    cloth_display = torch.clamp(cloth * (1.0 - suppress), 0.0, 1.0)

    strong_body = _sigmoid_score((body - cloth - 0.07) / 0.035)
    strong_cloth = _sigmoid_score((cloth - body - 0.07) / 0.035)
    scalar_body_snap = _sigmoid_score((body_gap - 0.06) / 0.030)
    body_display = body_display + 0.16 * strong_body * (1.0 - soft_display) + 0.18 * scalar_body_snap * (1.0 - cloth_display) + 0.18 * body_anchor_support * (1.0 - cloth_display)
    cloth_display = cloth_display + 0.16 * strong_cloth * (1.0 - soft_display)
    cloth_display = cloth_display * (1.0 - 0.22 * scalar_body_snap) * (1.0 - 0.18 * body_anchor_support)

    dominance_gap = (body_display - cloth_display).abs()
    dominant_max = torch.maximum(body_display, cloth_display)
    snap = _sigmoid_score((dominant_max - 0.30) / 0.035) * _sigmoid_score((dominance_gap - 0.06) / 0.030)
    body_mask = (body_display >= cloth_display).float()
    cloth_mask = 1.0 - body_mask
    body_display = body_display * (1.0 - snap) + body_mask * snap
    cloth_display = cloth_display * (1.0 - snap) + cloth_mask * snap
    soft_display = soft_display * (1.0 - 0.96 * snap)

    display = torch.stack([body_display, soft_display, cloth_display], dim=-1)
    display = _exclusive_bodycloth_probs(display, pair_strength=0.82, soft_share=0.18, interior_suppress=0.72)
    return _sharpen_probs(display, float(config.get('binding_map_region_display_sharpen', 2.30)))



def _enhanced_thin_values(pc, region_probs, layer_probs, config):
    rigid_cfg = _rigid_cfg(config)
    width = max(float(rigid_cfg.get('region_transition_width', 0.015)), 1e-6)
    body_surface_thr = float(rigid_cfg.get('body_surface_threshold', 0.018))
    body_sem_thr = float(rigid_cfg.get('body_semantic_threshold', 0.012))
    cloth_surface_thr = float(rigid_cfg.get('cloth_surface_threshold', 0.02))
    cloth_sem_thr = float(rigid_cfg.get('cloth_semantic_threshold', 0.04))
    dominant_joint, confidence, surface_distance, semantic_distance, thin_score = _binding_aux(pc)

    body_contact = _sigmoid_score((confidence - float(rigid_cfg.get('body_confidence_threshold', 0.7))) / width)
    body_contact = body_contact * _sigmoid_score((body_surface_thr - surface_distance) / width)
    semantic_core = _sigmoid_score((body_sem_thr - semantic_distance) / width)
    cloth_surface = _sigmoid_score((surface_distance - cloth_surface_thr) / width)
    cloth_semantic = _sigmoid_score((semantic_distance - cloth_sem_thr) / width)
    cloth = region_probs[:, 2]
    body = region_probs[:, 0]
    free = layer_probs[:, 2]
    soft = region_probs[:, 1]
    cloth_body_gap = (cloth - body).abs()
    overlap = torch.sqrt((cloth * (0.36 * soft + 0.14 * free)).clamp_min(0.0))
    boundary_band = overlap * _sigmoid_score((0.14 - cloth_body_gap) / 0.034)

    torso_mask = _joint_mask(dominant_joint, [3, 6, 9, 12])
    shoulder_mask = torch.clamp(torso_mask + _joint_mask(dominant_joint, [13, 14, 16, 17, 18, 19]), 0.0, 1.0)
    shorts_mask = _joint_mask(dominant_joint, [0, 1, 2, 4, 5])
    distal_mask = _joint_mask(dominant_joint, [7, 8, 10, 11, 20, 21, 22, 23])
    edge_mask = torch.clamp(shorts_mask + torso_mask + distal_mask, 0.0, 1.0)

    cloth_interior = torch.clamp(
        cloth * _sigmoid_score((cloth - body - 0.10) / 0.038) * _sigmoid_score((0.08 - boundary_band) / 0.026),
        0.0,
        1.0,
    )
    body_interior = torch.clamp(
        body * body_contact * semantic_core * _sigmoid_score((0.08 - boundary_band) / 0.024) * (1.0 - 0.42 * free),
        0.0,
        1.0,
    )
    strap_hint = shoulder_mask * torch.clamp(
        0.72 * boundary_band + 0.14 * cloth_surface + 0.14 * cloth_semantic + 0.08 * thin_score + 0.12 * free - 0.42 * body_interior - 0.28 * cloth_interior,
        0.0,
        1.0,
    )
    trim_hint = edge_mask * torch.clamp(
        0.70 * boundary_band + 0.16 * free + 0.10 * cloth_surface + 0.08 * cloth_semantic + 0.04 * (1.0 - confidence) - 0.34 * body_interior - 0.42 * cloth_interior,
        0.0,
        1.0,
    )
    distal_hint = distal_mask * torch.clamp(0.28 * boundary_band + 0.16 * free + 0.08 * thin_score - 0.22 * body_interior, 0.0, 1.0)
    base = torch.clamp(0.10 * thin_score + 0.06 * boundary_band, 0.0, 1.0)
    values = torch.clamp(
        base
        + 0.74 * trim_hint * (1.0 - base)
        + 0.66 * strap_hint * (1.0 - base)
        + 0.34 * distal_hint * (1.0 - base)
        - 0.34 * cloth_interior
        - 0.46 * body_interior
        - 0.16 * torso_mask * body_contact * semantic_core,
        0.0,
        1.0,
    )
    values = values * _sigmoid_score((values - 0.16) / 0.045)
    return values


def _enhanced_semantic_stability(pc, region_probs, layer_probs, positions, config):
    rigid_cfg = _rigid_cfg(config)
    width = max(float(rigid_cfg.get('region_transition_width', 0.015)), 1e-6)
    dominant_joint, confidence, surface_distance, semantic_distance, thin_score = _binding_aux(pc)

    torso_mask = _joint_mask(dominant_joint, [3, 6, 9, 12])
    head_mask = _joint_mask(dominant_joint, [15])
    upper_limb_mask = _joint_mask(dominant_joint, [4, 5, 13, 14, 16, 17, 18, 19])
    lower_limb_mask = _joint_mask(dominant_joint, [7, 8])
    distal_mask = _joint_mask(dominant_joint, [10, 11, 20, 21, 22, 23])

    body_surface_thr = float(rigid_cfg.get('body_surface_threshold', 0.018))
    body_sem_thr = float(rigid_cfg.get('body_semantic_threshold', 0.012))
    body_contact = _sigmoid_score((confidence - float(rigid_cfg.get('body_confidence_threshold', 0.7))) / width)
    body_contact = body_contact * _sigmoid_score((body_surface_thr - surface_distance) / width)
    semantic_core = _sigmoid_score((body_sem_thr - semantic_distance) / width)

    scale = semantic_distance.quantile(0.82).clamp_min(1e-6)
    base_stability = 1.0 - torch.clamp(semantic_distance / scale, 0.0, 1.0).pow(0.72)
    torso_floor = torso_mask * body_contact * (0.60 + 0.40 * semantic_core) * (0.72 + 0.28 * torch.maximum(region_probs[:, 0], 0.65 * region_probs[:, 1]))
    head_floor = head_mask * body_contact * (0.56 + 0.44 * semantic_core)
    stable_core = torch.clamp(0.94 * torso_floor + 0.78 * head_floor + 0.18 * region_probs[:, 0] + 0.10 * layer_probs[:, 0], 0.0, 1.0)
    limb_core = torch.clamp(
        0.16 * upper_limb_mask * body_contact * (0.68 + 0.32 * semantic_core) * (0.64 + 0.36 * region_probs[:, 0])
        + 0.14 * lower_limb_mask * body_contact * (0.64 + 0.36 * semantic_core) * (0.70 + 0.30 * region_probs[:, 0]),
        0.0,
        1.0,
    )
    body_cloth_mix = 2.0 * torch.minimum(region_probs[:, 0], region_probs[:, 2])
    edge_mix = torch.clamp(0.72 * body_cloth_mix + 0.44 * layer_probs[:, 1] + 0.26 * thin_score, 0.0, 1.0)
    unstable_shell = torch.clamp(
        0.26 * distal_mask
        + 0.14 * thin_score
        + 0.16 * region_probs[:, 2]
        + 0.14 * layer_probs[:, 2]
        + 0.24 * edge_mix,
        0.0,
        1.0,
    ) * (1.0 - 0.68 * stable_core)

    stability = torch.clamp(base_stability + 0.34 * stable_core + 0.18 * limb_core + 0.08 * region_probs[:, 0] - 0.30 * unstable_shell, 0.0, 1.0)
    stability = torch.maximum(stability, 0.96 * stable_core)
    mid_band = torch.clamp(edge_mix * (1.0 - torch.maximum(stable_core, unstable_shell)), 0.0, 1.0)
    stability = stability * (1.0 - 0.34 * mid_band) + 0.56 * (0.34 * mid_band)
    stability = stability * (1.0 - 0.46 * unstable_shell) + 0.16 * (0.46 * unstable_shell)
    stability = _clean_scalar_tensor(stability, positions, config, 'semantic')
    stability = _smooth_scalar_tensor(stability, positions, config, 'semantic')
    return stability


def _calibrated_scalar_values(map_name, region_probs, layer_probs, pc, config):
    body = region_probs[:, 0]
    soft = region_probs[:, 1]
    cloth = region_probs[:, 2]
    rigid = layer_probs[:, 0]
    layer_soft = layer_probs[:, 1]
    free = layer_probs[:, 2]
    rigid_cfg = _rigid_cfg(config)
    width = max(float(rigid_cfg.get('region_transition_width', 0.015)), 1e-6)
    body_surface_thr = float(rigid_cfg.get('body_surface_threshold', 0.018))
    dominant_joint, confidence, surface_distance, semantic_distance, thin_score = _binding_aux(pc)
    positions = _binding_positions(pc)
    head_hair = _head_hair_support(pc, positions, config)
    head_core = (_joint_mask(dominant_joint, [15]) * (1.0 - 0.93 * head_hair)).clamp(0.0, 1.0)
    strap_prior, chest_prior = _torso_strap_chest_priors(dominant_joint, positions)
    cloth_surface_thr = float(rigid_cfg.get('cloth_surface_threshold', 0.02))
    cloth_sem_thr = float(rigid_cfg.get('cloth_semantic_threshold', 0.04))
    cloth_surface = _sigmoid_score((surface_distance - cloth_surface_thr) / width)
    cloth_semantic = _sigmoid_score((semantic_distance - cloth_sem_thr) / width)
    boundary_support = torch.sqrt((body * cloth).clamp_min(0.0))

    upper_leg_mask = _joint_mask(dominant_joint, [4, 5])
    lower_leg_mask = _joint_mask(dominant_joint, [7, 8])
    leg_body_mask = torch.clamp(upper_leg_mask + lower_leg_mask, 0.0, 1.0)
    foot_mask = _joint_mask(dominant_joint, [10, 11])
    hand_mask = _joint_mask(dominant_joint, [20, 21, 22, 23])
    torso_mask = _joint_mask(dominant_joint, [3, 6, 9, 12])
    head_mask = _joint_mask(dominant_joint, [15])
    shoulder_cloth_mask = torch.clamp(torso_mask + _joint_mask(dominant_joint, [13, 14, 16, 17, 18, 19]), 0.0, 1.0)
    edge_cloth_mask = torch.clamp(_joint_mask(dominant_joint, [0, 1, 2]) + torso_mask + _joint_mask(dominant_joint, [18, 19, 20, 21, 22, 23]), 0.0, 1.0)
    head_body_support = head_core * _sigmoid_score((confidence - 0.34) / width) * _sigmoid_score(((body_surface_thr * 1.80) - surface_distance) / width)

    leg_body_support = leg_body_mask * _sigmoid_score((confidence - 0.44) / width) * _sigmoid_score(((body_surface_thr * 1.85) - surface_distance) / width)
    leg_body_support = leg_body_support * (1.0 - 0.45 * cloth)
    lower_leg_detail_support = lower_leg_mask * _sigmoid_score((confidence - 0.36) / width) * _sigmoid_score(((body_surface_thr * 1.95) - surface_distance) / width)
    lower_leg_detail_support = lower_leg_detail_support * (0.72 + 0.28 * body) * (1.0 - 0.40 * cloth)
    hand_body_support = hand_mask * _sigmoid_score((confidence - 0.34) / width) * _sigmoid_score(((body_surface_thr * 1.60) - surface_distance) / width)
    hand_body_support = hand_body_support * (0.76 + 0.24 * body) * (1.0 - 0.34 * cloth)
    hand_detail_support = hand_mask * torch.clamp(0.56 * body + 0.24 * rigid + 0.18 * (1.0 - cloth) + 0.14 * (1.0 - thin_score), 0.0, 1.0) * _sigmoid_score((confidence - 0.30) / width)
    foot_cloth_support = foot_mask * torch.clamp(0.62 + 0.22 * cloth + 0.16 * free + 0.18 * boundary_support - 0.08 * rigid, 0.0, 1.0)
    edge_cloth_support = edge_cloth_mask * torch.clamp(0.22 * cloth + 0.20 * free + 0.22 * thin_score + 0.18 * cloth_surface + 0.16 * cloth_semantic + 0.12 * boundary_support, 0.0, 1.0)
    strap_cloth_support = shoulder_cloth_mask * torch.clamp(0.44 * boundary_support + 0.34 * cloth + 0.28 * free + 0.28 * thin_score + 0.22 * cloth_surface + 0.20 * cloth_semantic - 0.18 * body, 0.0, 1.0)

    core_body_support = (0.24 * torso_mask + 0.18 * head_core) * _sigmoid_score((confidence - 0.46) / width) * _sigmoid_score(((body_surface_thr * 1.75) - surface_distance) / width)

    if map_name == 'body_prob':
        strap_body_penalty = strap_prior * torch.clamp(0.26 * cloth + 0.22 * strap_cloth_support + 0.12 * thin_score, 0.0, 1.0)
        chest_body_bonus = chest_prior * torch.clamp(0.34 * body + 0.28 * core_body_support + 0.16 * body, 0.0, 1.0)
        values = torch.clamp(body + 0.34 * rigid - 0.40 * cloth - 0.08 * free + 0.34 * core_body_support + 0.34 * leg_body_support + 0.52 * hand_body_support + 0.32 * hand_detail_support + 0.28 * lower_leg_detail_support + 0.16 * head_body_support + 0.24 * chest_body_bonus - 0.46 * foot_cloth_support - 0.12 * strap_cloth_support - 0.20 * strap_body_penalty - 0.60 * head_hair, 0.0, 1.0)
        return values
    if map_name == 'cloth_prob':
        strap_cloth_bonus = strap_prior * torch.clamp(0.30 * cloth + 0.24 * strap_cloth_support + 0.14 * thin_score + 0.10 * boundary_support, 0.0, 1.0)
        chest_cloth_penalty = chest_prior * torch.clamp(0.34 * body + 0.20 * core_body_support + 0.14 * boundary_support, 0.0, 1.0)
        values = torch.clamp(cloth + 0.36 * free - 0.82 * body - 0.22 * rigid - 0.30 * leg_body_support - 0.54 * hand_body_support - 0.20 * lower_leg_detail_support - 0.60 * head_body_support - 1.10 * head_hair + 0.48 * foot_cloth_support + 0.38 * edge_cloth_support + 0.70 * strap_cloth_support + 0.24 * strap_cloth_bonus + 0.16 * boundary_support - 0.24 * chest_cloth_penalty, 0.0, 1.0)
        return values
    if map_name == 'soft_prob':
        values = torch.clamp(soft + 0.20 * layer_soft - 0.12 * rigid - 0.12 * free + 0.28 * head_hair, 0.0, 1.0)
        return values
    raise ValueError(map_name)


def _map_colors_and_stats(pc, map_name, config, visibility_filter=None, radii=None):
    device = pc.get_xyz.device
    positions = _binding_positions(pc)
    layer_probs_raw = _get_layer_probs(pc)
    region_probs_raw = _get_region_probs(pc)
    region_probs = _calibrated_region_probs(pc, region_probs_raw, config)
    layer_probs = _calibrated_layer_probs(pc, layer_probs_raw, region_probs, config)
    layer_probs_vis = _smooth_prob_tensor(layer_probs, positions, config, 'layer')
    region_probs_vis = _smooth_prob_tensor(region_probs, positions, config, 'region')
    visible_mask, weights = _resolve_visibility(pc, visibility_filter=visibility_filter, radii=radii)
    common_stats = _common_visibility_stats(pc, visible_mask, weights)

    if map_name == 'layer':
        layer_probs_vis = _smooth_prob_tensor(layer_probs_vis, positions, config, 'layer_vis')
        layer_probs_vis = _clean_prob_tensor(layer_probs_vis, positions, config, 'layer')
        dominant_joint, _, _, _, _ = _binding_aux(pc)
        lower_leg_mask = _joint_mask(dominant_joint, [7, 8])
        limb_skin_mask = _joint_mask(dominant_joint, [4, 5, 7, 8, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23])
        torso_mask = _joint_mask(dominant_joint, [3, 6, 9, 12])
        shorts_mask = _joint_mask(dominant_joint, [0, 1, 2, 4, 5])
        head_hairline = _head_hairline_override(pc, positions, config)
        head_hair = torch.maximum(_head_hair_support(pc, positions, config), head_hairline)
        body_scalar = _calibrated_scalar_values('body_prob', region_probs, layer_probs, pc, config)
        cloth_scalar = _calibrated_scalar_values('cloth_prob', region_probs, layer_probs, pc, config)
        lower_leg_region_body = _sigmoid_score((region_probs[:, 0] - region_probs[:, 2] + 0.00) / 0.025)
        lower_leg_layer_body = _sigmoid_score((layer_probs_vis[:, 0] - layer_probs_vis[:, 2] + 0.00) / 0.030)
        lower_leg_skin = _sigmoid_score((body_scalar - cloth_scalar - 0.06) / 0.045)
        limb_skin_support = limb_skin_mask * _sigmoid_score((body_scalar - cloth_scalar - 0.08) / 0.040) * _sigmoid_score((region_probs[:, 0] - region_probs[:, 2] - 0.02) / 0.045) * (1.0 - 0.75 * head_hair)
        torso_body_rigid = torso_mask * _sigmoid_score((body_scalar - cloth_scalar - 0.02) / 0.035) * _sigmoid_score((region_probs[:, 0] - region_probs[:, 2] - 0.01) / 0.035) * (1.0 - 0.82 * head_hair)
        torso_cloth_free = torso_mask * _sigmoid_score((cloth_scalar - body_scalar + 0.00) / 0.035) * _sigmoid_score((region_probs[:, 2] - region_probs[:, 0] + 0.00) / 0.035) * (1.0 - 0.82 * head_hair)
        torso_soft_band = torso_mask * _sigmoid_score((0.10 - (body_scalar - cloth_scalar).abs()) / 0.05) * torch.clamp(2.0 * torch.minimum(region_probs[:, 0], region_probs[:, 2]), 0.0, 1.0) * (1.0 - 0.82 * head_hair)
        lower_leg_rigid_alpha = 0.92 * lower_leg_mask * torch.maximum(torch.maximum(lower_leg_region_body, lower_leg_layer_body), lower_leg_skin)
        shorts_free_alpha = 0.26 * shorts_mask * _sigmoid_score((region_probs[:, 2] - region_probs[:, 0] + 0.02) / 0.04)
        layer_probs_vis[:, 2] = layer_probs_vis[:, 2] * (1.0 - 0.90 * lower_leg_rigid_alpha - 0.34 * limb_skin_support - 0.88 * torso_body_rigid).clamp(min=0.0)
        layer_probs_vis[:, 1] = layer_probs_vis[:, 1] * (1.0 - 0.56 * lower_leg_rigid_alpha - 0.30 * shorts_free_alpha - 0.34 * limb_skin_support - 0.58 * torso_body_rigid - 0.62 * torso_cloth_free).clamp(min=0.0) + 0.18 * torso_soft_band
        layer_probs_vis[:, 0] = layer_probs_vis[:, 0] + 1.10 * lower_leg_rigid_alpha + 0.34 * limb_skin_support + 0.96 * torso_body_rigid
        layer_probs_vis[:, 2] = layer_probs_vis[:, 2] + 0.78 * torso_cloth_free + shorts_free_alpha
        lower_leg_snap_alpha = 0.58 * lower_leg_mask * lower_leg_skin
        lower_leg_target = torch.stack([
            torch.full_like(lower_leg_snap_alpha, 0.94),
            torch.full_like(lower_leg_snap_alpha, 0.06),
            torch.zeros_like(lower_leg_snap_alpha),
        ], dim=-1)
        layer_probs_vis = layer_probs_vis * (1.0 - lower_leg_snap_alpha.unsqueeze(-1)) + lower_leg_target * lower_leg_snap_alpha.unsqueeze(-1)
        limb_snap_alpha = 0.26 * limb_skin_support
        limb_target = torch.stack([
            torch.full_like(limb_snap_alpha, 0.90),
            torch.full_like(limb_snap_alpha, 0.10),
            torch.zeros_like(limb_snap_alpha),
        ], dim=-1)
        layer_probs_vis = layer_probs_vis * (1.0 - limb_snap_alpha.unsqueeze(-1)) + limb_target * limb_snap_alpha.unsqueeze(-1)
        torso_rigid_alpha = 0.46 * torso_body_rigid
        torso_rigid_target = torch.stack([
            torch.full_like(torso_rigid_alpha, 0.90),
            torch.full_like(torso_rigid_alpha, 0.08),
            torch.full_like(torso_rigid_alpha, 0.02),
        ], dim=-1)
        layer_probs_vis = layer_probs_vis * (1.0 - torso_rigid_alpha.unsqueeze(-1)) + torso_rigid_target * torso_rigid_alpha.unsqueeze(-1)
        torso_free_alpha = 0.56 * torso_cloth_free
        torso_free_target = torch.stack([
            torch.full_like(torso_free_alpha, 0.04),
            torch.full_like(torso_free_alpha, 0.08),
            torch.full_like(torso_free_alpha, 0.88),
        ], dim=-1)
        layer_probs_vis = layer_probs_vis * (1.0 - torso_free_alpha.unsqueeze(-1)) + torso_free_target * torso_free_alpha.unsqueeze(-1)
        torso_soft_alpha = 0.12 * torso_soft_band * (1.0 - 0.75 * torch.maximum(torso_body_rigid, torso_cloth_free))
        torso_soft_target = torch.stack([
            torch.full_like(torso_soft_alpha, 0.10),
            torch.full_like(torso_soft_alpha, 0.78),
            torch.full_like(torso_soft_alpha, 0.12),
        ], dim=-1)
        layer_probs_vis = layer_probs_vis * (1.0 - torso_soft_alpha.unsqueeze(-1)) + torso_soft_target * torso_soft_alpha.unsqueeze(-1)
        layer_probs_vis[:, 0] = layer_probs_vis[:, 0] * (1.0 - 0.88 * head_hair)
        layer_probs_vis[:, 1] = layer_probs_vis[:, 1] * (1.0 - 0.24 * head_hair)
        layer_probs_vis[:, 2] = layer_probs_vis[:, 2] + 0.92 * head_hair
        layer_probs_vis = _renorm_probs(layer_probs_vis)
        hairline_alpha = head_hairline.clamp(0.0, 1.0).unsqueeze(-1)
        head_free_target = torch.stack([
            torch.zeros_like(head_hairline),
            torch.zeros_like(head_hairline),
            torch.ones_like(head_hairline),
        ], dim=-1)
        layer_probs_vis = layer_probs_vis * (1.0 - hairline_alpha) + head_free_target * hairline_alpha
        layer_probs_vis = _renorm_probs(layer_probs_vis)
        layer_color_mode = config.get('binding_map_layer_color_mode', 'label_confidence')
        if layer_color_mode == 'mix':
            colors, confidence = _mix_prob_colors(layer_probs_vis, LAYER_COLORS, config, 'layer')
            layer_ids = layer_probs_vis.argmax(dim=-1)
        else:
            colors, confidence, layer_ids = _label_confidence_colors(layer_probs_vis, LAYER_COLORS, config, 'layer')
        hist = _weighted_hist(layer_ids, 3, weights)
        mean_probs = _weighted_mean(layer_probs_vis, weights)
        stats = dict(common_stats)
        stats.update({
            'rigid_ratio': float((hist[0] / hist.sum().clamp_min(1e-6)).item()),
            'soft_ratio': float((hist[1] / hist.sum().clamp_min(1e-6)).item()),
            'free_ratio': float((hist[2] / hist.sum().clamp_min(1e-6)).item()),
            'rigid_prob_mean': float(mean_probs[0].item()),
            'soft_prob_mean': float(mean_probs[1].item()),
            'free_prob_mean': float(mean_probs[2].item()),
            'layer_confidence_mean': float(_weighted_mean(confidence, weights).item()),
        })
        return colors, stats

    if map_name == 'region':
        region_probs_vis = _smooth_prob_tensor(region_probs_vis, positions, config, 'region_vis')
        region_probs_vis = _display_region_probs(pc, region_probs_vis, positions, config)
        region_probs_vis = _clean_prob_tensor(region_probs_vis, positions, config, 'region')

        dominant_joint, confidence_aux, surface_distance_aux, semantic_distance_aux, thin_score_aux = _binding_aux(pc)
        strap_prior, chest_prior = _torso_strap_chest_priors(dominant_joint, positions)
        body_peer = _calibrated_scalar_values('body_prob', region_probs_vis, layer_probs, pc, config)
        cloth_peer = _calibrated_scalar_values('cloth_prob', region_probs_vis, layer_probs, pc, config)
        chest_gap = body_peer - cloth_peer
        torso_mask = _joint_mask(dominant_joint, [3, 6, 9, 12])
        chest_focus = chest_prior
        torso_points = torso_mask > 0.5
        if torso_points.any():
            x = positions[:, 0]
            y = positions[:, 1]
            torso_top = torch.quantile(y[torso_points], 0.90)
            torso_bottom = torch.quantile(y[torso_points], 0.16)
            torso_height = (torso_top - torso_bottom).abs().clamp_min(0.04)
            torso_center_x = torch.quantile(x[torso_points], 0.50)
            torso_span_x = (torch.quantile(x[torso_points], 0.90) - torch.quantile(x[torso_points], 0.10)).abs().clamp_min(0.03)
            chest_band_y = _sigmoid_score((y - (torso_bottom + 0.46 * torso_height)) / (0.08 * torso_height + 1e-6))
            chest_band_y = chest_band_y * _sigmoid_score((((torso_bottom + 0.92 * torso_height)) - y) / (0.10 * torso_height + 1e-6))
            chest_band_x = _sigmoid_score(((0.34 * torso_span_x) - (x - torso_center_x).abs()) / (0.08 * torso_span_x + 1e-6))
            chest_focus = torch.maximum(chest_focus, torso_mask * chest_band_y * chest_band_x)
        chest_rescue = chest_focus * _sigmoid_score((body_peer - 0.30) / 0.032) * _sigmoid_score((chest_gap + 0.02) / 0.015)
        chest_rescue = chest_rescue * (1.0 - 0.42 * strap_prior) * (1.0 - 0.08 * thin_score_aux)
        region_probs_vis[:, 0] = region_probs_vis[:, 0] + 1.18 * chest_rescue
        region_probs_vis[:, 1] = region_probs_vis[:, 1] * (1.0 - 0.36 * chest_rescue)
        region_probs_vis[:, 2] = region_probs_vis[:, 2] * (1.0 - 0.94 * chest_rescue)
        region_probs_vis = _renorm_probs(region_probs_vis)

        region_probs_vis = _sharpen_probs(region_probs_vis, float(config.get('binding_map_region_final_sharpen', 1.82)))
        region_color_mode = config.get('binding_map_region_color_mode', 'label_confidence')
        if region_color_mode == 'mix':
            colors, confidence = _mix_prob_colors(region_probs_vis, REGION_COLORS, config, 'region')
            region_ids = region_probs_vis.argmax(dim=-1)
        else:
            colors, confidence, region_ids = _label_confidence_colors(region_probs_vis, REGION_COLORS, config, 'region')
        hist = _weighted_hist(region_ids, 3, weights)
        mean_probs = _weighted_mean(region_probs_vis, weights)
        stats = dict(common_stats)
        stats.update({
            'body_ratio': float((hist[0] / hist.sum().clamp_min(1e-6)).item()),
            'soft_ratio': float((hist[1] / hist.sum().clamp_min(1e-6)).item()),
            'cloth_ratio': float((hist[2] / hist.sum().clamp_min(1e-6)).item()),
            'body_prob_mean': float(mean_probs[0].item()),
            'soft_prob_mean': float(mean_probs[1].item()),
            'cloth_prob_mean': float(mean_probs[2].item()),
            'region_confidence_mean': float(_weighted_mean(confidence, weights).item()),
        })
        return colors, stats

    if map_name == 'compact_semantic':
        compact_probs, compact_names = _get_compact_semantic_probs(pc)
        if compact_probs is None:
            compact_probs = torch.zeros(pc.get_xyz.shape[0], len(COMPACT_SEMANTIC_NAMES), device=device)
            compact_probs[:, 0] = 1.0
            compact_names = COMPACT_SEMANTIC_NAMES
        compact_palette = _compact_semantic_palette(compact_names).to(device=device, dtype=compact_probs.dtype)
        compact_probs_vis = _smooth_prob_tensor(compact_probs, positions, config, 'compact_semantic')
        compact_probs_vis = _clean_prob_tensor(compact_probs_vis, positions, config, 'compact_semantic')
        compact_probs_vis = _sharpen_probs(
            compact_probs_vis,
            float(config.get('binding_map_compact_semantic_sharpen', 1.35)),
        )
        compact_color_mode = config.get('binding_map_compact_semantic_color_mode', 'label_confidence')
        if compact_color_mode == 'mix':
            colors, confidence = _mix_prob_colors(compact_probs_vis, compact_palette, config, 'compact_semantic')
            compact_ids = compact_probs_vis.argmax(dim=-1)
        else:
            colors, confidence, compact_ids = _label_confidence_colors(compact_probs_vis, compact_palette, config, 'compact_semantic')
        hist = _weighted_hist(compact_ids, compact_probs_vis.shape[-1], weights)
        mean_probs = _weighted_mean(compact_probs_vis, weights)
        stats = dict(common_stats)
        for class_idx, class_name in enumerate(compact_names):
            ratio = hist[class_idx] / hist.sum().clamp_min(1e-6)
            stats[f'{class_name}_ratio'] = float(ratio.item())
            stats[f'{class_name}_prob_mean'] = float(mean_probs[class_idx].item())
        stats['compact_confidence_mean'] = float(_weighted_mean(confidence, weights).item())
        return colors, stats

    if map_name in {'body_prob', 'soft_prob', 'cloth_prob'}:
        idx = {'body_prob': 0, 'soft_prob': 1, 'cloth_prob': 2}[map_name]
        raw_values = _calibrated_scalar_values(map_name, region_probs, layer_probs, pc, config)
        values = _smooth_scalar_tensor(raw_values, positions, config, map_name)
        values = _clean_scalar_tensor(values, positions, config, map_name)
        foot_neighbor = _foot_neighbor_support(pc, positions, region_probs, config)
        dominant_joint, confidence, surface_distance, semantic_distance_aux, thin_score_aux = _binding_aux(pc)
        hand_mask = _joint_mask(dominant_joint, [20, 21, 22, 23])
        lower_leg_mask = _joint_mask(dominant_joint, [7, 8])
        head_hairline = _head_hairline_override(pc, positions, config)
        head_hair = torch.maximum(_head_hair_support(pc, positions, config), head_hairline)
        head_core = (_joint_mask(dominant_joint, [15]) * (1.0 - 0.93 * head_hair)).clamp(0.0, 1.0)
        shoulder_mask = torch.clamp(_joint_mask(dominant_joint, [3, 6, 9, 12]) + _joint_mask(dominant_joint, [13, 14, 16, 17, 18, 19]), 0.0, 1.0)
        strap_prior, chest_prior = _torso_strap_chest_priors(dominant_joint, positions)
        rigid_cfg = _rigid_cfg(config)
        width = max(float(rigid_cfg.get('region_transition_width', 0.015)), 1e-6)
        body_surface_thr = float(rigid_cfg.get('body_surface_threshold', 0.018))
        cloth_surface_thr = float(rigid_cfg.get('cloth_surface_threshold', 0.02))
        body_sem_thr = float(rigid_cfg.get('body_semantic_threshold', 0.012))
        cloth_sem_thr = float(rigid_cfg.get('cloth_semantic_threshold', 0.04))
        body_contact = _sigmoid_score((confidence - float(rigid_cfg.get('body_confidence_threshold', 0.7))) / width)
        body_contact = body_contact * _sigmoid_score((body_surface_thr - surface_distance) / width)
        body_semantic = _sigmoid_score((body_sem_thr - semantic_distance_aux) / width)
        cloth_surface = _sigmoid_score((surface_distance - cloth_surface_thr) / width)
        cloth_semantic = _sigmoid_score((semantic_distance_aux - cloth_sem_thr) / width)
        if map_name == 'body_prob':
            hand_boost = 0.16 * hand_mask * _sigmoid_score((region_probs[:, 0] - region_probs[:, 2] + 0.02) / 0.04)
            lower_leg_boost = 0.12 * lower_leg_mask * _sigmoid_score((region_probs[:, 0] - region_probs[:, 2] + 0.04) / 0.05) * (1.0 - foot_neighbor)
            head_boost = 0.10 * head_core * _sigmoid_score((region_probs[:, 0] - region_probs[:, 2] + 0.01) / 0.03)
            cloth_peer = torch.clamp(_calibrated_scalar_values('cloth_prob', region_probs, layer_probs, pc, config), 0.0, 1.0)
            exclusivity = (1.0 - 0.94 * cloth_peer).clamp(0.03, 1.0)
            chest_rescue = 0.48 * chest_prior * torch.clamp(0.56 * body_contact + 0.30 * body_semantic + 0.18 * region_probs[:, 0] - 0.10 * cloth_semantic, 0.0, 1.0)
            strap_penalty = 0.14 * strap_prior * torch.clamp(0.24 * region_probs[:, 2] + 0.18 * cloth_semantic + 0.10 * thin_score_aux, 0.0, 1.0)
            values = torch.clamp((values - 0.58 * foot_neighbor + hand_boost + lower_leg_boost + head_boost - 0.50 * head_hair - 0.08 * shoulder_mask * cloth_peer) * exclusivity + 0.12 * region_probs[:, 0] * (1.0 - cloth_peer) + chest_rescue - strap_penalty, 0.0, 1.0)
            values = torch.clamp(values * (1.0 - 0.22 * cloth_peer) + 0.14 * _sigmoid_score((values - cloth_peer - 0.02) / 0.04), 0.0, 1.0)
            values = values * (1.0 - 0.995 * head_hairline)
        elif map_name == 'cloth_prob':
            strap_boost = 0.24 * shoulder_mask * _sigmoid_score((region_probs[:, 2] - region_probs[:, 0] + 0.02) / 0.04) * (0.5 + 0.5 * layer_probs[:, 2])
            body_peer = _calibrated_scalar_values('body_prob', region_probs, layer_probs, pc, config)
            body_peer = torch.clamp(body_peer, 0.0, 1.0)
            exclusivity = (1.0 - 1.02 * body_peer).clamp(0.01, 1.0)
            strap_rescue = 0.34 * strap_prior * torch.clamp(0.34 * cloth_semantic + 0.20 * cloth_surface + 0.14 * thin_score_aux + 0.10 * region_probs[:, 2], 0.0, 1.0)
            chest_penalty = 0.40 * chest_prior * torch.clamp(0.44 * body_contact + 0.22 * body_semantic + 0.18 * region_probs[:, 0], 0.0, 1.0)
            values = torch.clamp((values + 0.36 * foot_neighbor + strap_boost + strap_rescue - 0.96 * head_hair - 0.26 * head_core - 0.10 * hand_mask * body_peer - chest_penalty) * exclusivity + 0.12 * region_probs[:, 2] * (1.0 - body_peer), 0.0, 1.0)
            values = torch.clamp(values * (1.0 - 0.28 * body_peer) + 0.16 * _sigmoid_score((values - body_peer - 0.02) / 0.04), 0.0, 1.0)
            values = values * (1.0 - 0.998 * head_hairline)
        elif map_name == 'soft_prob':
            body_peer = torch.clamp(_calibrated_scalar_values('body_prob', region_probs, layer_probs, pc, config), 0.0, 1.0)
            cloth_peer = torch.clamp(_calibrated_scalar_values('cloth_prob', region_probs, layer_probs, pc, config), 0.0, 1.0)
            transition_mix = torch.minimum(body_peer, cloth_peer)
            region_mix = torch.minimum(region_probs[:, 0], region_probs[:, 2])
            body_core = _sigmoid_score((body_peer - 0.54) / 0.05) * (1.0 - 0.82 * cloth_peer)
            cloth_core = _sigmoid_score((cloth_peer - 0.56) / 0.05) * (1.0 - 0.80 * body_peer)
            joint_soft = torch.clamp(
                _joint_mask(dominant_joint, [3, 6, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23])
                + 0.72 * _joint_mask(dominant_joint, [7, 8]),
                0.0,
                1.0,
            )
            boundary_soft = torch.clamp(
                0.58 * transition_mix
                + 0.42 * region_mix
                + 0.20 * thin_score_aux
                + 0.16 * cloth_surface * body_contact
                + 0.12 * body_semantic * cloth_semantic,
                0.0,
                1.0,
            )
            values = torch.clamp(
                0.14 * values
                + 0.58 * region_probs[:, 1]
                + 0.82 * boundary_soft
                + 0.24 * joint_soft
                - 0.58 * body_core
                - 0.62 * cloth_core
                - 0.22 * head_hair
                - 0.18 * foot_neighbor,
                0.0,
                1.0,
            )
            values = torch.clamp((values - 0.10) / 0.58, 0.0, 1.0)
            values = values * _sigmoid_score((values - 0.16) / 0.07)
            values = values.pow(1.08)
        visible_values = _visible_values(values, visible_mask)
        quantile = float(config.get(f'binding_map_{map_name}_quantile', config.get('binding_map_prob_quantile', 0.95)))
        gamma = float(config.get(f'binding_map_{map_name}_gamma', config.get('binding_map_prob_gamma', 0.65)))
        scale = _estimate_scale(
            visible_values,
            scale=config.get(f'binding_map_{map_name}_scale', config.get('binding_map_prob_scale', None)),
            quantile=quantile,
        )
        normalized = _windowed_scalar_display(values, scale, gamma, config, map_name, default_low=0.18, default_high=0.88, default_threshold=0.30)
        colors = _heatmap_rgb(normalized)
        stats = dict(common_stats)
        stats.update({
            'mean': float(_weighted_mean(values, weights).item()),
            'max': float(visible_values.max().item()),
            'scale': float(scale.item()),
            'quantile': quantile,
            'gamma': gamma,
        })
        return colors, stats

    if map_name == 'semantic':
        semantic_distance = getattr(pc, 'binding_semantic_distance', torch.zeros(pc.get_xyz.shape[0], device=device))
        stability = _enhanced_semantic_stability(pc, region_probs, layer_probs, positions, config)
        dominant_joint, confidence, surface_distance, semantic_distance_aux, _ = _binding_aux(pc)
        torso_mask = _joint_mask(dominant_joint, [3, 6, 9, 12])
        head_mask = _joint_mask(dominant_joint, [15])
        body_surface_thr = float(_rigid_cfg(config).get('body_surface_threshold', 0.018))
        width = max(float(_rigid_cfg(config).get('region_transition_width', 0.015)), 1e-6)
        body_contact = _sigmoid_score((confidence - float(_rigid_cfg(config).get('body_confidence_threshold', 0.7))) / width)
        body_contact = body_contact * _sigmoid_score((body_surface_thr - surface_distance) / width)
        semantic_core = _sigmoid_score((float(_rigid_cfg(config).get('body_semantic_threshold', 0.012)) - semantic_distance_aux) / width)
        stability = torch.clamp(stability + 0.20 * torso_mask * body_contact * semantic_core + 0.10 * head_mask * body_contact * semantic_core, 0.0, 1.0)
        stable_alpha = _sigmoid_score((stability - 0.70) / 0.05)
        unstable_alpha = _sigmoid_score((0.34 - stability) / 0.05)
        mid_alpha = _sigmoid_score((0.12 - (stability - 0.56).abs()) / 0.05) * (1.0 - torch.maximum(stable_alpha, unstable_alpha))
        stability = stability * (1.0 - 0.44 * stable_alpha) + 0.92 * (0.44 * stable_alpha)
        stability = stability * (1.0 - 0.40 * unstable_alpha) + 0.14 * (0.40 * unstable_alpha)
        stability = stability * (1.0 - 0.26 * mid_alpha) + 0.56 * (0.26 * mid_alpha)
        stability_vis = _windowed_scalar_display(stability, stability.new_tensor(1.0), 0.84, config, 'semantic', default_low=0.32, default_high=0.90, default_threshold=0.34)
        colors = _stability_rgb(stability_vis)
        visible_values = _visible_values(semantic_distance, visible_mask)
        semantic_quantile = float(config.get('binding_map_semantic_quantile', 0.82))
        semantic_scale = _estimate_scale(visible_values, config.get('binding_map_semantic_scale', None), quantile=semantic_quantile)
        stats = dict(common_stats)
        stats.update({
            'semantic_distance_mean': float(_weighted_mean(semantic_distance, weights).item()),
            'semantic_distance_max': float(visible_values.max().item()),
            'semantic_stability_mean': float(_weighted_mean(stability, weights).item()),
            'semantic_scale': float(semantic_scale.item()),
        })
        return colors, stats

    if map_name == 'temporal':
        temporal_slip = getattr(pc, 'binding_temporal_slip', torch.zeros(pc.get_xyz.shape[0], device=device))
        dominant_joint, confidence, surface_distance, semantic_distance_aux, thin_score_aux = _binding_aux(pc)
        rigid_cfg = _rigid_cfg(config)
        width = max(float(rigid_cfg.get('region_transition_width', 0.015)), 1e-6)
        body_surface_thr = float(rigid_cfg.get('body_surface_threshold', 0.018))
        cloth_surface_thr = float(rigid_cfg.get('cloth_surface_threshold', 0.02))
        body_sem_thr = float(rigid_cfg.get('body_semantic_threshold', 0.012))
        cloth_sem_thr = float(rigid_cfg.get('cloth_semantic_threshold', 0.04))
        body_contact = _sigmoid_score((confidence - float(rigid_cfg.get('body_confidence_threshold', 0.7))) / width)
        body_contact = body_contact * _sigmoid_score((body_surface_thr - surface_distance) / width)
        cloth_surface = _sigmoid_score((surface_distance - cloth_surface_thr) / width)
        body_semantic = _sigmoid_score((body_sem_thr - semantic_distance_aux) / width)
        cloth_semantic = _sigmoid_score((semantic_distance_aux - cloth_sem_thr) / width)
        transition_mix = torch.minimum(region_probs[:, 0], region_probs[:, 2])
        uncertainty = torch.clamp(1.0 - confidence, 0.0, 1.0)
        joint_transition = torch.clamp(
            _joint_mask(dominant_joint, [3, 6, 9, 12, 13, 14, 16, 17, 18, 19]) + 0.68 * _joint_mask(dominant_joint, [7, 8]),
            0.0,
            1.0,
        )
        proxy_temporal = torch.clamp(
            0.44 * transition_mix
            + 0.24 * region_probs[:, 1]
            + 0.18 * uncertainty
            + 0.16 * thin_score_aux
            + 0.12 * torch.minimum(body_contact, cloth_surface)
            + 0.10 * torch.minimum(body_semantic, cloth_semantic)
            + 0.10 * joint_transition,
            0.0,
            1.0,
        )
        proxy_temporal = proxy_temporal * (1.0 - 0.48 * _sigmoid_score((confidence - 0.88) / 0.04))
        temporal_quantile = float(config.get('binding_map_temporal_quantile', 0.95))
        temporal_gamma = float(config.get('binding_map_temporal_gamma', 0.7))
        visible_values = _visible_values(temporal_slip, visible_mask)
        temporal_values = temporal_slip
        using_proxy_temporal = False
        if float(visible_values.max().item()) <= 1e-8:
            using_proxy_temporal = True
            temporal_values = torch.clamp((proxy_temporal - 0.10) / 0.28, 0.0, 1.0)
            temporal_values = temporal_values.pow(1.14)
            visible_values = _visible_values(temporal_values, visible_mask)
        else:
            temporal_values = torch.clamp(torch.maximum(temporal_slip, 0.24 * proxy_temporal), 0.0, 1.0)
            visible_values = _visible_values(temporal_values, visible_mask)
        temporal_scale_cfg = config.get('binding_map_temporal_scale', None)
        if temporal_scale_cfg is None and not using_proxy_temporal:
            temporal_scale_cfg = 0.02
        temporal_scale = _estimate_scale(visible_values, temporal_scale_cfg, quantile=temporal_quantile)
        normalized = _windowed_scalar_display(
            temporal_values,
            temporal_scale,
            0.82 if using_proxy_temporal else temporal_gamma,
            config,
            'temporal',
            default_low=0.26 if using_proxy_temporal else 0.18,
            default_high=0.92 if using_proxy_temporal else 0.88,
            default_threshold=0.40 if using_proxy_temporal else 0.32,
        )
        colors = _heatmap_rgb(normalized)
        stats = dict(common_stats)
        stats.update({
            'temporal_slip_mean': float(_weighted_mean(temporal_values, weights).item()),
            'temporal_slip_max': float(visible_values.max().item()),
            'temporal_scale': float(temporal_scale.item()),
        })
        return colors, stats

    if map_name == 'thin':
        thin_score = _enhanced_thin_values(pc, region_probs, layer_probs, config)
        thin_score = _smooth_scalar_tensor(thin_score, positions, config, 'thin')
        thin_score = _clean_scalar_tensor(thin_score, positions, config, 'thin')
        thin_score = torch.clamp(thin_score - float(config.get('binding_map_thin_bias', 0.12)), 0.0, 1.0)
        thin_score = torch.clamp((thin_score - 0.12) / 0.62, 0.0, 1.0)
        thin_score = thin_score * _sigmoid_score((thin_score - 0.14) / 0.08)
        thin_score = thin_score.pow(1.18)
        visible_values = _visible_values(thin_score, visible_mask)
        thin_scale = _estimate_scale(visible_values, config.get('binding_map_thin_scale', None), quantile=float(config.get('binding_map_thin_quantile', 0.80)))
        thin_vis = _windowed_scalar_display(thin_score, thin_scale, float(config.get('binding_map_thin_gamma', 0.78)), config, 'thin', default_low=0.24, default_high=0.90, default_threshold=0.34)
        colors = _thin_rgb(thin_vis)
        stats = dict(common_stats)
        stats.update({
            'thin_score_mean': float(_weighted_mean(thin_score, weights).item()),
            'thin_score_max': float(visible_values.max().item()),
            'thin_scale': float(thin_scale.item()),
        })
        return colors, stats

    raise ValueError(f'Unknown binding map: {map_name}')


def _export_binding_maps(config, view, render_pkg, background, summary_records):
    if not config.get('export_interpretability', False):
        return

    pc = render_pkg['deformed_gaussian']
    opacity = render_pkg.get('opacity_render', None)
    visibility_filter = render_pkg.get('visibility_filter', None)
    radii = render_pkg.get('radii', None)
    map_root = os.path.join(config.exp_dir, config.suffix, 'binding_maps')
    os.makedirs(map_root, exist_ok=True)

    record = {'image_name': view.image_name}
    for map_name in _binding_map_names(config):
        colors, stats = _map_colors_and_stats(
            pc,
            map_name,
            config,
            visibility_filter=visibility_filter,
            radii=radii,
        )
        map_pkg = rasterize_gaussians(
            view,
            pc,
            config.pipeline,
            background,
            colors_precomp=colors,
            return_opacity=False,
        )
        image = map_pkg['render']
        if opacity is not None and config.get('binding_map_use_opacity_mask', True):
            image = image * opacity
        if map_name == 'compact_semantic':
            compact_probs, compact_names = _get_compact_semantic_probs(pc)
            if compact_probs is not None:
                compact_background = torch.zeros(3, device=image.device, dtype=image.dtype)
                compact_probs_2d, compact_opacity = _rasterize_prob_channels(
                    view,
                    pc,
                    config.pipeline,
                    compact_background,
                    compact_probs,
                )
                compact_hard_masks, compact_ids_2d, compact_valid_2d = _compact_hard_assignment(
                    compact_probs_2d,
                    compact_names,
                    fg_mask=_view_fg_mask_tensor(view, image),
                    opacity=compact_opacity,
                    opacity_threshold=float(config.get('binding_map_compact_semantic_opacity_threshold', 0.05)),
                    confidence_threshold=float(config.get('binding_map_compact_semantic_confidence_threshold', 0.0)),
                )
                image = _compact_semantic_map_tensor(compact_names, compact_ids_2d, compact_valid_2d)
        if map_name == 'layer':
            image = _light_label_morphology(image, LAYER_COLORS, config, 'layer')
        elif map_name == 'region':
            image = _light_label_morphology(image, REGION_COLORS, config, 'region')
        render_rgb = render_pkg.get('render', None)
        hair_mask_2d = _load_cihp_hair_mask(view, image, config, labels=(2,))
        face_mask_2d = _load_cihp_hair_mask(view, image, config, labels=(13,))
        body_mask_2d = _load_cihp_hair_mask(view, image, config, labels=(13, 14, 15, 16, 17))
        cloth_mask_2d = _load_cihp_hair_mask(view, image, config, labels=(5, 6, 7, 9, 10, 11, 12))
        shoe_mask_2d = _load_cihp_hair_mask(view, image, config, labels=(18, 19))
        if map_name != 'compact_semantic':
            image = _appearance_guided_head_refine(image, render_rgb, opacity, map_name)
            image = _apply_cihp_parsing_override(image, hair_mask_2d, body_mask_2d, cloth_mask_2d, map_name, face_mask=face_mask_2d, shoe_mask=shoe_mask_2d)
            image = _appearance_guided_torso_refine(image, render_rgb, opacity, map_name, face_mask=face_mask_2d, cloth_mask=cloth_mask_2d)
            image = _bodyprob_guided_refine(image, render_rgb, opacity, map_name, face_mask=face_mask_2d, body_mask=body_mask_2d, cloth_mask=cloth_mask_2d, hair_mask=hair_mask_2d, shoe_mask=shoe_mask_2d)
        if map_name == 'region':
            aux_maps = {}
            for aux_name in ('body_prob', 'cloth_prob'):
                aux_colors, _ = _map_colors_and_stats(pc, aux_name, config, visibility_filter=visibility_filter, radii=radii)
                aux_pkg = rasterize_gaussians(view, pc, config.pipeline, background, colors_precomp=aux_colors, return_opacity=False)
                aux_image = aux_pkg['render']
                if opacity is not None and config.get('binding_map_use_opacity_mask', True):
                    aux_image = aux_image * opacity
                aux_image = _appearance_guided_head_refine(aux_image, render_rgb, opacity, aux_name)
                aux_image = _apply_cihp_parsing_override(aux_image, hair_mask_2d, body_mask_2d, cloth_mask_2d, aux_name, face_mask=face_mask_2d, shoe_mask=shoe_mask_2d)
                aux_image = _appearance_guided_torso_refine(aux_image, render_rgb, opacity, aux_name, face_mask=face_mask_2d, cloth_mask=cloth_mask_2d)
                aux_image = _bodyprob_guided_refine(aux_image, render_rgb, opacity, aux_name, face_mask=face_mask_2d, body_mask=body_mask_2d, cloth_mask=cloth_mask_2d, hair_mask=hair_mask_2d, shoe_mask=shoe_mask_2d)
                aux_maps[aux_name] = aux_image
            image = _region_follow_bodyprob(image, aux_maps['body_prob'], aux_maps['cloth_prob'], face_mask=face_mask_2d)
        if map_name != 'compact_semantic':
            image = _stabilize_visual_map(image, map_name, opacity, face_mask=face_mask_2d, body_mask=body_mask_2d, cloth_mask=cloth_mask_2d, hair_mask=hair_mask_2d, shoe_mask=shoe_mask_2d)
            image = _appearance_guided_detail_refine(image, render_rgb, opacity, map_name, body_mask=body_mask_2d, cloth_mask=cloth_mask_2d, hair_mask=hair_mask_2d, shoe_mask=shoe_mask_2d)
        if map_name in {'layer', 'region'}:
            hard_fg = _binding_map_support_mask(
                view,
                image,
                opacity=opacity,
                opacity_threshold=float(config.get('binding_map_hard_fg_opacity_threshold', 0.06)),
                close_kernel=3,
                erode_kernel=3,
            )
            image = image * hard_fg
            if map_name == 'layer':
                image = _light_label_morphology(image, LAYER_COLORS, config, 'layer')
            else:
                image = _light_label_morphology(image, REGION_COLORS, config, 'region')
        else:
            support = _binding_map_support_mask(
                view,
                image,
                opacity=opacity,
                opacity_threshold=float(config.get('binding_map_opacity_threshold', 0.06)),
                close_kernel=3,
                erode_kernel=0,
            )
            image = image * support
        out_dir = os.path.join(map_root, map_name)
        os.makedirs(out_dir, exist_ok=True)
        torchvision.utils.save_image(image, os.path.join(out_dir, f'render_{view.image_name}.png'))
        record[map_name] = {key: _to_python(value) for key, value in stats.items()}

    summary_records.append(record)


def _render_schedule_local_iteration(config):
    value = config.get('render_schedule_local_iteration', '__default__')
    if value == '__default__':
        opt = config.get('opt', None)
        return int(opt.iterations) if opt is not None and 'iterations' in opt else None
    if value is None or str(value).lower() in ('none', 'null', 'global'):
        return None
    return int(value)


def _set_texture_schedule_context(scene, local_iteration=None, schedule_iteration=None):
    texture = getattr(getattr(scene, 'converter', None), 'texture', None)
    setter = getattr(texture, 'set_schedule_context', None)
    if callable(setter):
        setter(
            local_iteration=local_iteration,
            schedule_iteration=schedule_iteration,
        )


def _render_split(config, scene, background, evaluate=False, iteration=None):
    render_path = os.path.join(config.exp_dir, config.suffix, 'renders')
    os.makedirs(render_path, exist_ok=True)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    evaluator = PSEvaluator() if config.dataset.name == 'people_snapshot' else Evaluator()
    psnrs, ssims, lpipss, times = [], [], [], []
    summary_records = []
    asset_records = []
    need_opacity = bool(config.get('export_interpretability', False) or config.get('export_semantic_editable_assets', False))
    render_iteration = int(config.opt.iterations if iteration is None else iteration)
    render_local_iteration = _render_schedule_local_iteration(config)

    for idx in trange(len(scene.test_dataset), desc='Rendering progress'):
        view = scene.test_dataset[idx]
        iter_start.record()
        _set_texture_schedule_context(
            scene,
            local_iteration=render_local_iteration,
            schedule_iteration=render_iteration,
        )
        render_pkg = render(
            view,
            render_iteration,
            scene,
            config.pipeline,
            background,
            compute_loss=False,
            return_opacity=need_opacity,
        )
        iter_end.record()
        torch.cuda.synchronize()
        elapsed = iter_start.elapsed_time(iter_end)

        rendering = render_pkg['render']
        export_render = _prepare_render_export_image(
            config,
            view,
            rendering,
            opacity=render_pkg.get('opacity_render', None),
        )
        torchvision.utils.save_image(export_render, os.path.join(render_path, f'render_{view.image_name}.png'))

        if evaluate:
            gt = view.original_image[:3, :, :]
            wandb_img = [
                wandb.Image(rendering[None], caption='render_{}'.format(view.image_name)),
                wandb.Image(gt[None], caption='gt_{}'.format(view.image_name)),
            ]
            wandb.log({'test_images': wandb_img})
            if config.evaluate:
                metrics = evaluator(rendering, gt)
                psnrs.append(metrics['psnr'])
                ssims.append(metrics['ssim'])
                lpipss.append(metrics['lpips'])
            else:
                psnrs.append(torch.tensor([0.], device='cuda'))
                ssims.append(torch.tensor([0.], device='cuda'))
                lpipss.append(torch.tensor([0.], device='cuda'))
        else:
            wandb_img = [wandb.Image(rendering[None], caption='render_{}'.format(view.image_name))]
            wandb.log({'test_images': wandb_img})

        prev_summary_len = len(summary_records)
        _export_binding_maps(config, view, render_pkg, background, summary_records)
        binding_record = summary_records[-1] if len(summary_records) > prev_summary_len else None
        _export_semantic_editable_assets(config, view, render_pkg, binding_record, asset_records)
        times.append(elapsed)

    if summary_records:
        map_root = os.path.join(config.exp_dir, config.suffix, 'binding_maps')
        with open(os.path.join(map_root, 'summary.json'), 'w') as f:
            json.dump(summary_records, f, indent=2)
    if asset_records:
        _finalize_semantic_editable_assets(config, asset_records)

    results = {'time': float(np.mean(times[1:])) if len(times) > 1 else float(np.mean(times))}
    if evaluate:
        results.update({
            'psnr': torch.mean(torch.stack(psnrs)),
            'ssim': torch.mean(torch.stack(ssims)),
            'lpips': torch.mean(torch.stack(lpipss)),
        })
    return results


def predict(config):
    with torch.set_grad_enabled(False):
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, 'ckpt' + str(config.opt.iterations) + '.pth')
        loaded_iteration = scene.load_checkpoint(load_ckpt)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

        results = _render_split(config, scene, background, evaluate=False, iteration=loaded_iteration)
        wandb.log({'metrics/time': results['time']})
        np.savez(os.path.join(config.exp_dir, config.suffix, 'results.npz'), time=results['time'])


def test(config):
    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, 'ckpt' + str(config.opt.iterations) + '.pth')
        loaded_iteration = scene.load_checkpoint(load_ckpt)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

        results = _render_split(config, scene, background, evaluate=True, iteration=loaded_iteration)
        wandb.log({
            'metrics/psnr': results['psnr'],
            'metrics/ssim': results['ssim'],
            'metrics/lpips': results['lpips'],
            'metrics/time': results['time'],
        })
        np.savez(
            os.path.join(config.exp_dir, config.suffix, 'results.npz'),
            psnr=results['psnr'].cpu().numpy(),
            ssim=results['ssim'].cpu().numpy(),
            lpips=results['lpips'].cpu().numpy(),
            time=results['time'],
        )


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)
    _snapshot_hydra_run(config, config.exp_dir)

    if config.mode == 'test':
        config.suffix = config.mode + '-' + config.dataset.test_mode
    elif config.mode == 'predict':
        predict_seq = config.dataset.predict_seq
        if config.dataset.name == 'zjumocap':
            predict_dict = {0: 'dance0', 1: 'dance1', 2: 'flipping', 3: 'canonical'}
        else:
            predict_dict = {0: 'rotation', 1: 'dance2'}
        config.suffix = config.mode + '-' + predict_dict[predict_seq]
    else:
        raise ValueError
    if config.dataset.freeview:
        config.suffix = config.suffix + '-freeview'

    wandb_name = config.name + '-' + config.suffix
    wandb.init(
        mode='disabled' if config.wandb_disable else None,
        name=wandb_name,
        project='gaussian-splatting-avatar-test',
        entity='fast-avatar',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )

    fix_random(config.seed)

    if config.mode == 'test':
        test(config)
    elif config.mode == 'predict':
        predict(config)
    else:
        raise ValueError


if __name__ == '__main__':
    main()
