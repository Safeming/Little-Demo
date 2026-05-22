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

import os
import json
import shutil
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, rasterize_gaussians
from scene import Scene, GaussianModel
from utils.general_utils import fix_random, Evaluator, PSEvaluator
from tqdm import tqdm
from utils.loss_utils import full_aiap_loss
from utils.graphics_utils import geom_transform_points

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
try:
    import wandb
except ImportError:
    class _WandbFallback:
        class Image:
            def __init__(self, *args, **kwargs):
                pass

        class Histogram:
            def __init__(self, *args, **kwargs):
                pass

        class Settings:
            def __init__(self, *args, **kwargs):
                pass

        @staticmethod
        def init(*args, **kwargs):
            return None

        @staticmethod
        def log(*args, **kwargs):
            return None

    wandb = _WandbFallback()
import lpips
from utils.pytorch3d_compat import ops
from pathlib import Path
from utils.adopted_geometry import apply_explicit_binding_render_preset


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


PARSER_FACE_LABELS = (13,)
PARSER_ARM_LABELS = (14, 15)
PARSER_UPPER_TORSO_LABELS = (5, 6, 7, 11)
PARSER_LOWER_CLOTH_LABELS = (9, 10, 12)
COMPACT_SEMANTIC_CLASS_NAMES = ('hair', 'face', 'skin', 'upper', 'lower', 'shoes')


def _apply_stageB_semantic_adapter_only_train_policy(config):
    opt = config.get('opt', None)
    if opt is None or not bool(opt.get('stageB_semantic_adapter_only_train', False)):
        return

    allowed_lambda_keys = {'lambda_binding_semantic_asset_adapter_reg'}
    for key in list(opt.keys()):
        if str(key).startswith('lambda_') and key not in allowed_lambda_keys:
            opt[key] = 0.0

    zero_lr_keys = (
        'position_lr_init',
        'position_lr_final',
        'feature_lr',
        'opacity_lr',
        'scaling_lr',
        'rotation_lr',
        'boundary_opacity_residual_lr',
        'boundary_scaling_residual_lr',
        'boundary_cov_residual_lr',
        'binding_layer_logits_lr',
        'semantic_region_logits_lr',
        'semantic_compact_logits_lr',
        'pose_correction_lr',
        'rigid_lr',
        'non_rigid_lr',
        'nr_latent_lr',
        'texture_lr',
        'tex_latent_lr',
        'camera_affine_lr',
        'camera_geometry_lr',
    )
    for key in zero_lr_keys:
        opt[key] = 0.0

    opt['stageB_semantic_loss_enable'] = True
    opt['percent_dense'] = 0.0
    opt['densify_from_iter'] = 999999999
    opt['densify_until_iter'] = 0
    opt['boundary_aware_enable'] = False
    opt['boundary_tag_enable'] = False
    opt['boundary_live_score_cache_enable'] = False
    opt['boundary_signed_routing_enable'] = False
    config.render_export_refine = False

    resume_cfg = config.get('resume', None)
    if resume_cfg is not None:
        resume_cfg['disable_densify_on_resume'] = True
        resume_cfg['disable_opacity_reset_on_resume'] = True
        resume_cfg['restore_gaussian_optimizer_state'] = False
        resume_cfg['restore_converter_optimizer_state'] = False
        resume_cfg['restore_converter_scheduler_state'] = False
        resume_cfg['clear_boundary_tags_on_resume'] = False
        resume_cfg['clear_binding_state_on_resume'] = False
    print(
        "StageB semantic adapter-only train policy active: "
        f"semantic_lr=({opt.get('semantic_region_logits_lr', 0.0)}, "
        f"{opt.get('semantic_compact_logits_lr', 0.0)}), "
        f"semantic_asset_lr=({opt.get('semantic_asset_region_logits_lr', 0.0)}, "
        f"{opt.get('semantic_asset_compact_logits_lr', 0.0)}), "
        f"texture_lr={opt.get('texture_lr', 0.0)}, "
        f"lambda_l1={opt.get('lambda_l1', 0.0)}, "
        f"lambda_semantic_asset_adapter_reg={opt.get('lambda_binding_semantic_asset_adapter_reg', 0.0)}"
    )

HEAD_JOINT_INDEX = 15
NECK_JOINT_INDEX = 12
LEFT_COLLAR_JOINT_INDEX = 13
RIGHT_COLLAR_JOINT_INDEX = 14
LEFT_SHOULDER_JOINT_INDEX = 16
RIGHT_SHOULDER_JOINT_INDEX = 17
LEFT_ELBOW_JOINT_INDEX = 18
RIGHT_ELBOW_JOINT_INDEX = 19


def C(iteration, value):
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = OmegaConf.to_container(value)
        if not isinstance(value, list):
            raise TypeError('Scalar specification only supports list, got', type(value))
        value_list = [0] + value
        i = 0
        current_step = iteration
        while i < len(value_list):
            if current_step >= value_list[i]:
                i += 2
            else:
                break
        value = value_list[i - 1]
    return value


def _plain_mapping(value):
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        value = value.strip()
        if value.startswith('{') and value.endswith('}'):
            value = value[1:-1]
        parsed = {}
        for item in value.split(','):
            item = item.strip()
            if not item:
                continue
            if ':' not in item:
                continue
            key, raw = item.split(':', 1)
            key = key.strip().strip('"\'')
            raw = raw.strip().strip('"\'')
            try:
                parsed[key] = float(raw)
            except ValueError:
                parsed[key] = raw
        return parsed
    return dict(value) if isinstance(value, dict) else {}


def _lookup_float_mapping(mapping, key, default=1.0):
    if not isinstance(mapping, dict):
        return float(default)
    candidates = []
    if key is not None:
        candidates.extend([key, str(key)])
        try:
            candidates.extend([int(key), str(int(key))])
        except (TypeError, ValueError):
            pass
    for candidate in candidates:
        if candidate in mapping:
            try:
                return float(mapping[candidate])
            except (TypeError, ValueError):
                return float(default)
    return float(default)


def _camera_id_from_data(data):
    for attr in ('cam_id', 'camera_id', 'cam_idx', 'view_id', 'view_index'):
        value = getattr(data, attr, None)
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = value.detach().cpu().item()
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                continue
            value = value[0]
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    image_name = getattr(data, 'image_name', None)
    if isinstance(image_name, (list, tuple)):
        image_name = image_name[0] if image_name else None
    if isinstance(image_name, str):
        # Dataset image names follow cXX_fYYYYYY; keep a defensive parser for old samples.
        for token in image_name.replace('-', '_').split('_'):
            if len(token) >= 2 and token[0] == 'c' and token[1:].isdigit():
                return int(token[1:])
    return None


def _reliable_view_supervision_debug(data, opt):
    enabled = bool(opt.get('reliable_view_supervision_enable', False))
    camera_id = _camera_id_from_data(data)
    default_weight = float(opt.get('reliable_view_default_highfreq_weight', 1.0))
    unknown_weight = float(opt.get('reliable_view_unknown_highfreq_weight', default_weight))
    camera_weights = _plain_mapping(opt.get('reliable_view_camera_quality_weights', {}))
    raw_weight = _lookup_float_mapping(
        camera_weights,
        camera_id,
        unknown_weight if camera_id is None else default_weight,
    )

    if enabled:
        weight = raw_weight
        power = float(opt.get('reliable_view_highfreq_power', 1.0))
        if power > 0.0 and power != 1.0:
            weight = max(weight, 0.0) ** power
        min_weight = float(opt.get('reliable_view_highfreq_min_weight', 0.0))
        max_weight = float(opt.get('reliable_view_highfreq_max_weight', 0.0))
        if min_weight > 0.0:
            weight = max(weight, min_weight)
        if max_weight > 0.0:
            weight = min(weight, max_weight)
    else:
        weight = 1.0

    return {
        'enabled': enabled,
        'camera_id': float(camera_id if camera_id is not None else -1),
        'raw_weight': float(raw_weight),
        'weight': float(weight),
    }


def _bounded_probability(raw_weights, min_prob=0.0, max_prob=0.0):
    weights = np.asarray(raw_weights, dtype=np.float64)
    weights = np.maximum(weights, 1.0e-8)
    prob = weights / weights.sum()
    n = prob.shape[0]
    if n <= 0:
        return prob

    min_prob = float(min_prob or 0.0)
    max_prob = float(max_prob or 0.0)
    if min_prob > 0.0 and min_prob * n < 1.0:
        prob = np.maximum(prob, min_prob)
        prob = prob / prob.sum()
    if max_prob > 0.0:
        max_prob = max(max_prob, 1.0 / n)
        for _ in range(12):
            over = prob > max_prob
            if not np.any(over):
                break
            excess = float((prob[over] - max_prob).sum())
            prob[over] = max_prob
            under = ~over
            under_sum = float(prob[under].sum())
            if under_sum <= 0.0:
                break
            prob[under] += excess * prob[under] / under_sum
        prob = prob / prob.sum()
    return prob.astype(np.float64)


class FrameBalancedCameraWeightedSampler:
    def __init__(self, dataset, opt):
        self.dataset = dataset
        self.opt = opt
        self.camera_weights = _plain_mapping(opt.get('train_sample_camera_weights', {}))
        self.min_prob = float(opt.get('train_sample_camera_min_prob', 0.0))
        self.max_prob = float(opt.get('train_sample_camera_max_prob', 0.0))
        self.log_interval = int(opt.get('train_sample_log_interval', 500))
        self.accumulation_steps = max(1, int(opt.get('train_sample_accumulation_steps', 1)))
        self.accumulation_without_replacement = bool(
            opt.get('train_sample_accumulation_without_replacement', True)
        )
        self.groups = {}
        self.frame_stack = []
        self.pending_indices = []
        self.pending_cameras = []
        self.pending_frame_id = None
        self.current_accumulation_size = 1
        self.current_accumulation_step = 1
        self.sample_count = 0
        self.camera_counts = {}
        self._build_groups()

    def _sample_meta(self, idx):
        raw_data = getattr(self.dataset, 'data', None)
        if raw_data is not None and idx < len(raw_data):
            item = raw_data[idx]
            frame_id = item.get('frame_idx', item.get('frame_id', None))
            cam_id = item.get('cam_name', item.get('cam_id', item.get('cam_idx', None)))
            if frame_id is not None and cam_id is not None:
                return int(frame_id), str(int(cam_id))
        sample = self.dataset[idx]
        frame_id = getattr(sample, 'frame_id', None)
        cam_id = getattr(sample, 'cam_id', None)
        if frame_id is None or cam_id is None:
            raise RuntimeError('frame-balanced sampler requires frame_id and cam_id metadata.')
        return int(frame_id), str(int(cam_id))

    def _build_groups(self):
        frame_to_items = {}
        for idx in range(len(self.dataset)):
            frame_id, cam_id = self._sample_meta(idx)
            frame_to_items.setdefault(frame_id, []).append((idx, cam_id))
            self.camera_counts.setdefault(cam_id, 0)
        if not frame_to_items:
            raise RuntimeError('frame-balanced sampler found no training samples.')

        for frame_id, items in frame_to_items.items():
            indices = [idx for idx, _ in items]
            cams = [cam for _, cam in items]
            raw_weights = [
                float(self.camera_weights.get(cam, self.camera_weights.get(str(int(cam)), 1.0)))
                for cam in cams
            ]
            prob = _bounded_probability(
                raw_weights,
                min_prob=self.min_prob,
                max_prob=self.max_prob,
            )
            self.groups[frame_id] = {
                'indices': indices,
                'cameras': cams,
                'prob': prob,
            }
        self.frames = sorted(self.groups.keys())
        self._reset_frame_stack()
        camera_names = sorted(self.camera_counts.keys(), key=lambda value: int(value))
        print(
            '[TrainSampler] mode=frame_balanced_camera_weighted '
            f'frames={len(self.frames)} samples={len(self.dataset)} cameras={camera_names} '
            f'min_prob={self.min_prob:.4f} max_prob={self.max_prob:.4f} '
            f'accumulation_steps={self.accumulation_steps}'
        )

    def _reset_frame_stack(self):
        self.frame_stack = list(self.frames)

    def _queue_next_frame_samples(self):
        if not self.frame_stack:
            self._reset_frame_stack()
        frame_pos = randint(0, len(self.frame_stack) - 1)
        frame_id = self.frame_stack.pop(frame_pos)
        group = self.groups[frame_id]
        sample_size = min(self.accumulation_steps, len(group['indices']))
        replace = not self.accumulation_without_replacement or sample_size > len(group['indices'])
        choices = np.random.choice(
            len(group['indices']),
            size=sample_size,
            replace=replace,
            p=group['prob'],
        )
        self.pending_indices = [group['indices'][int(choice)] for choice in choices]
        self.pending_cameras = [group['cameras'][int(choice)] for choice in choices]
        self.pending_frame_id = frame_id
        self.current_accumulation_size = max(1, len(self.pending_indices))
        self.current_accumulation_step = 0

    def next_index(self):
        if not self.pending_indices:
            self._queue_next_frame_samples()
        data_idx = self.pending_indices.pop(0)
        cam_id = self.pending_cameras.pop(0)
        self.current_accumulation_step += 1
        self.sample_count += 1
        self.camera_counts[cam_id] = self.camera_counts.get(cam_id, 0) + 1
        return data_idx

    def gradient_accumulation_scale(self):
        return 1.0 / float(max(1, self.current_accumulation_size))

    def should_optimizer_step(self):
        return self.current_accumulation_step >= self.current_accumulation_size

    def maybe_log(self, iteration):
        if self.log_interval <= 0 or self.sample_count <= 0:
            return
        if iteration == 1 or iteration % self.log_interval == 0:
            ordered = sorted(self.camera_counts.items(), key=lambda item: int(item[0]))
            summary = ','.join(f'{cam}:{count}' for cam, count in ordered)
            print(
                '[TrainSampler] '
                f'iter={iteration} samples={self.sample_count} '
                f'accum_step={self.current_accumulation_step}/{self.current_accumulation_size} '
                f'frame={self.pending_frame_id} camera_counts={summary}'
            )


def _build_train_sampler(dataset, opt):
    mode = str(opt.get('train_sample_mode', 'random')).lower()
    if mode in ('random', 'default', 'legacy', 'none'):
        return None
    if mode == 'frame_balanced_camera_weighted':
        return FrameBalancedCameraWeightedSampler(dataset, opt)
    raise ValueError(f'Unsupported train_sample_mode: {mode}')


def save_best_checkpoint(scene, iteration, metrics, metric_name='psnr_fg', metric_source='test'):
    ckpt_path = scene.save_checkpoint(iteration, filename='best_ckpt.pth', verbose=False)
    summary = {
        'iteration': int(iteration),
        'selection_metric': metric_name,
        'selection_source': metric_source,
    }
    summary.update({k: float(v) for k, v in metrics.items()})
    summary['checkpoint'] = os.path.basename(ckpt_path)
    with open(os.path.join(scene.save_dir, 'best_test_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    metric_value = float(summary.get(metric_name, summary.get('psnr', 0.0)))
    print("\n[ITER {}] Saving Best Checkpoint ({} {} {:.6f})".format(iteration, metric_source, metric_name, metric_value))


def _get_mask_from_data(data, source='original', fallback='original'):
    attr_map = {
        'original': 'original_mask',
        'hard': 'hard_mask',
        'soft': 'soft_mask',
    }
    attr = attr_map.get(str(source), str(source))
    value = getattr(data, attr, None)
    if torch.is_tensor(value):
        mask = value.cuda().float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask[:1]
        else:
            mask = mask.reshape(1, *mask.shape[-2:])
        return mask.clamp(0.0, 1.0)
    if fallback is None or fallback == source:
        return None
    return _get_mask_from_data(data, fallback, fallback=None)


def _foreground_mask_from_data(data):
    fg_mask = _get_mask_from_data(data, 'hard', fallback='original')
    if fg_mask is None:
        fg_mask = data.original_mask.cuda()
    if fg_mask.dim() == 2:
        fg_mask = fg_mask.unsqueeze(0)
    elif fg_mask.dim() == 3:
        fg_mask = fg_mask[:1]
    else:
        fg_mask = fg_mask.reshape(1, *fg_mask.shape[-2:])
    return (fg_mask > 0.5).float()


def _soft_transition_mask(mask, low, high):
    if mask is None:
        return None
    low = float(low)
    high = float(high)
    if high <= low:
        return torch.zeros_like(mask)
    return ((mask > low) & (mask < high)).float()


def _foreground_metrics(image, gt_image, fg_mask, evaluator):
    fg_mask = fg_mask.to(device=image.device, dtype=image.dtype)
    fg_norm = (fg_mask.sum() * image.shape[0]).clamp_min(1.0)
    l1_fg = (torch.abs(image - gt_image) * fg_mask).sum().double() / fg_norm.double()

    valid_mask_2d = (fg_mask[0] > 0.5)
    valid_mask_3d = valid_mask_2d.unsqueeze(0).expand_as(image)
    psnr_fg = evaluator.psnr(image, gt_image, valid_mask=valid_mask_3d)
    ssim_fg = evaluator.ssim(image, gt_image, valid_mask=valid_mask_2d)
    lpips_fg = evaluator.lpips(image, gt_image, valid_mask=valid_mask_2d).mean()
    return {
        'l1_fg': l1_fg,
        'psnr_fg': psnr_fg,
        'ssim_fg': ssim_fg,
        'lpips_fg': lpips_fg,
    }



def _normalize_kernel_size(kernel_size):
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def _masked_l1_loss(image, gt_image, mask):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1 and image.shape[0] != 1:
        mask = mask.expand(image.shape[0], -1, -1)
    mask = mask.to(device=image.device, dtype=image.dtype)
    norm = mask.sum().clamp_min(1.0)
    return (torch.abs(image - gt_image) * mask).sum() / norm


def _masked_channel_stats(image, mask):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    mask = mask.to(device=image.device, dtype=image.dtype)
    if mask.shape[0] == 1 and image.shape[0] != 1:
        mask = mask.expand(image.shape[0], -1, -1)
    denom = mask.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    mean = (image * mask).sum(dim=(1, 2), keepdim=True) / denom
    var = (((image - mean) * mask) ** 2).sum(dim=(1, 2), keepdim=True) / denom
    return mean, torch.sqrt(var.clamp_min(1.0e-6)), denom


def _photometric_corrected_gt_image(image, gt_image, fg_mask, config_opt, iteration=0):
    if not bool(config_opt.get('photometric_correction_enable', False)):
        return gt_image, {}

    correction_mask = (fg_mask > 0.5).float().to(device=image.device, dtype=image.dtype)
    erode_kernel_size = int(config_opt.get('photometric_correction_erode_kernel_size', 0))
    if erode_kernel_size > 1:
        eroded_mask = _binary_erode(correction_mask, erode_kernel_size).clamp(0.0, 1.0)
        if eroded_mask.sum().item() > 0:
            correction_mask = eroded_mask

    min_pixels = float(config_opt.get('photometric_correction_min_pixels', 128))
    active_pixels = float(correction_mask.sum().item())
    if active_pixels < min_pixels:
        return gt_image, {'active_pixels': active_pixels, 'strength': 0.0}

    pred_mean, pred_std, _ = _masked_channel_stats(image.detach(), correction_mask)
    gt_mean, gt_std, _ = _masked_channel_stats(gt_image.detach(), correction_mask)
    min_scale = float(config_opt.get('photometric_correction_min_scale', 0.70))
    max_scale = float(config_opt.get('photometric_correction_max_scale', 1.45))
    max_shift = float(config_opt.get('photometric_correction_max_shift', 0.16))
    scale = (pred_std / gt_std.clamp_min(1.0e-4)).clamp(min_scale, max_scale)
    shift = (pred_mean - gt_mean * scale).clamp(-max_shift, max_shift)
    aligned_gt = (gt_image * scale + shift).clamp(0.0, 1.0)

    strength = float(C(iteration, config_opt.get('photometric_correction_strength', 1.0)))
    strength = max(0.0, min(1.0, strength))
    blend_mask = correction_mask.clamp(0.0, 1.0)
    corrected = gt_image * (1.0 - strength * blend_mask) + aligned_gt * (strength * blend_mask)
    return corrected.clamp(0.0, 1.0), {
        'active_pixels': active_pixels,
        'strength': strength,
        'scale_mean': float(scale.mean().item()),
        'shift_abs_mean': float(shift.abs().mean().item()),
    }


def _contour_uncertainty_weight_mask(fg_mask, config_opt):
    if not bool(config_opt.get('contour_uncertainty_enable', False)):
        return None, {}

    fg_mask = (fg_mask > 0.5).float()
    band_width = int(config_opt.get('contour_uncertainty_band_width', 0))
    outer_width = int(config_opt.get('contour_uncertainty_outer_width', 0))
    uncertainty = torch.zeros_like(fg_mask)
    if band_width > 1:
        uncertainty = torch.maximum(uncertainty, _foreground_boundary_mask(fg_mask, band_width))
    if outer_width > 1:
        uncertainty = torch.maximum(uncertainty, _foreground_outer_shell_mask(fg_mask, 1, outer_width))
    min_weight = float(config_opt.get('contour_uncertainty_min_weight', 0.35))
    min_weight = max(0.0, min(1.0, min_weight))
    weight = torch.ones_like(fg_mask, dtype=fg_mask.dtype, device=fg_mask.device)
    weight = weight * (1.0 - uncertainty.clamp(0.0, 1.0) * (1.0 - min_weight))
    return weight.clamp(min_weight, 1.0), {
        'uncertain_pixels': float(uncertainty.sum().item()),
        'min_weight': min_weight,
    }


def _luma_gradient_magnitude(image):
    weights = image.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    luma = (image * weights).sum(dim=0, keepdim=True)
    grad_x = F.pad((luma[:, :, 1:] - luma[:, :, :-1]).abs(), (0, 1, 0, 0))
    grad_y = F.pad((luma[:, 1:, :] - luma[:, :-1, :]).abs(), (0, 0, 0, 1))
    return (grad_x + grad_y).clamp_min(0.0)


def _alignment_aware_contour_weight_mask(image, gt_image, fg_mask, region_mask, config_opt):
    if not bool(config_opt.get('alignment_aware_contour_enable', False)):
        return None, {}
    if region_mask is None:
        return None, {}

    fg_mask = (fg_mask > 0.5).float().to(device=image.device, dtype=image.dtype)
    region_mask = region_mask.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0) * fg_mask
    if float(region_mask.sum().item()) <= 0.0:
        return None, {}

    band_width = int(config_opt.get('alignment_aware_contour_band_width', 9))
    contour_band = _foreground_boundary_mask(fg_mask, band_width).to(device=image.device, dtype=image.dtype)
    contour_band = (contour_band * region_mask).clamp(0.0, 1.0)
    if float(contour_band.sum().item()) <= 0.0:
        return None, {}

    gt_grad = _luma_gradient_magnitude(gt_image.detach())
    edge_focus = _masked_normalize_map(gt_grad, contour_band)
    error_map = (image.detach() - gt_image.detach()).abs().mean(dim=0, keepdim=True)
    error_focus = _masked_normalize_map(error_map, contour_band)

    stable_boost = float(config_opt.get('alignment_aware_contour_stable_boost', 0.35))
    suppress_strength = float(config_opt.get('alignment_aware_contour_misaligned_suppress', 0.25))
    error_power = float(config_opt.get('alignment_aware_contour_error_power', 1.25))
    edge_protect = float(config_opt.get('alignment_aware_contour_edge_protect', 0.65))
    min_weight = float(config_opt.get('alignment_aware_contour_min_weight', 0.72))
    max_weight = float(config_opt.get('alignment_aware_contour_max_weight', 1.45))

    stable = edge_focus * (1.0 - error_focus).clamp(0.0, 1.0)
    suppress = error_focus.clamp(0.0, 1.0).pow(error_power) * (1.0 - edge_protect * edge_focus).clamp(0.0, 1.0)
    local_weight = 1.0 + stable_boost * stable - suppress_strength * suppress
    local_weight = local_weight.clamp(min_weight, max_weight)

    weight = torch.ones_like(contour_band)
    weight = weight * (1.0 - contour_band) + local_weight * contour_band
    return weight.clamp(min_weight, max_weight), {
        'pixels': float(contour_band.sum().item()),
        'mean_weight': float(((weight - 1.0).abs() * contour_band).sum().item() / max(float(contour_band.sum().item()), 1.0)),
        'stable_mean': float((stable * contour_band).sum().item() / max(float(contour_band.sum().item()), 1.0)),
        'suppress_mean': float((suppress * contour_band).sum().item() / max(float(contour_band.sum().item()), 1.0)),
        'min_weight': min_weight,
        'max_weight': max_weight,
    }


def _masked_binary_cross_entropy(prediction, target, mask, channel_weight=None, positive_weight=None):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1 and prediction.shape[0] != 1:
        mask = mask.expand(prediction.shape[0], -1, -1)
    mask = mask.to(device=prediction.device, dtype=prediction.dtype)

    if not torch.is_tensor(target):
        target = torch.full_like(prediction, float(target))
    else:
        target = target.to(device=prediction.device, dtype=prediction.dtype)
        if target.dim() == 2:
            target = target.unsqueeze(0)
        if target.shape[0] == 1 and prediction.shape[0] != 1:
            target = target.expand_as(prediction)

    norm = mask.sum().clamp_min(1.0)
    prediction = prediction.clamp(1.0e-3, 1.0 - 1.0e-3)
    loss = F.binary_cross_entropy(prediction, target, reduction='none')
    if positive_weight is not None:
        positive_weight = positive_weight.to(device=prediction.device, dtype=prediction.dtype)
        if positive_weight.dim() == 1:
            positive_weight = positive_weight.view(-1, 1, 1)
        loss = loss * (1.0 + (positive_weight - 1.0) * target)
    if channel_weight is not None:
        channel_weight = channel_weight.to(device=prediction.device, dtype=prediction.dtype)
        if channel_weight.dim() == 1:
            channel_weight = channel_weight.view(-1, 1, 1)
        loss = loss * channel_weight
    return (loss * mask).sum() / norm


def _boundary_regularization_positions(scene):
    canonical_xyz = getattr(scene.gaussians, 'canonical_xyz', None)
    if torch.is_tensor(canonical_xyz) and canonical_xyz.shape == scene.gaussians.get_xyz.shape:
        return canonical_xyz.detach()
    return scene.gaussians.get_xyz.detach()


def _boundary_residual_smoothness_loss(values, positions, point_mask, k=8, distance_quantile=0.5):
    if not torch.is_tensor(values) or values.numel() == 0:
        device = positions.device if torch.is_tensor(positions) else 'cuda'
        return torch.tensor(0.0, device=device)
    if not torch.is_tensor(positions) or positions.shape[0] != values.shape[0]:
        return values.new_tensor(0.0)
    if point_mask is None:
        return values.new_tensor(0.0)

    active_idx = torch.nonzero(point_mask.reshape(-1) > 0.0, as_tuple=False).squeeze(-1)
    if active_idx.numel() <= 1:
        return values.new_tensor(0.0)

    k = int(k)
    if k <= 0:
        return values.new_tensor(0.0)

    active_values = values[active_idx]
    active_positions = positions[active_idx].detach()
    knn_k = min(k + 1, active_positions.shape[0])
    if knn_k <= 1:
        return values.new_tensor(0.0)

    knn = ops.knn_points(active_positions.unsqueeze(0), active_positions.unsqueeze(0), K=knn_k)
    nn_idx = knn.idx[0, :, 1:]
    if nn_idx.numel() == 0:
        return values.new_tensor(0.0)

    nn_dists = torch.sqrt(knn.dists[0, :, 1:].clamp_min(1e-8))
    distance_quantile = float(min(max(distance_quantile, 0.05), 0.95))
    scale = torch.quantile(nn_dists.detach().reshape(-1), distance_quantile).clamp_min(1e-6)
    nn_weights = torch.exp(-nn_dists / scale)
    neigh = active_values[nn_idx]
    smooth = (neigh * nn_weights.unsqueeze(-1)).sum(dim=1) / nn_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    diff = active_values - smooth
    return diff.pow(2).sum(dim=-1).mean()


def _smooth_point_scores(values, positions, k=8, distance_quantile=0.5):
    if not torch.is_tensor(values) or values.numel() == 0:
        return values
    if not torch.is_tensor(positions) or positions.shape[0] != values.shape[0]:
        return values

    k = int(k)
    if k <= 0 or values.shape[0] <= 1:
        return values

    active_positions = positions.detach()
    knn_k = min(k + 1, active_positions.shape[0])
    if knn_k <= 1:
        return values

    knn = ops.knn_points(active_positions.unsqueeze(0), active_positions.unsqueeze(0), K=knn_k)
    nn_idx = knn.idx[0, :, 1:]
    if nn_idx.numel() == 0:
        return values

    nn_dists = torch.sqrt(knn.dists[0, :, 1:].clamp_min(1e-8))
    distance_quantile = float(min(max(distance_quantile, 0.05), 0.95))
    scale = torch.quantile(nn_dists.detach().reshape(-1), distance_quantile).clamp_min(1e-6)
    nn_weights = torch.exp(-nn_dists / scale)
    neigh = values[nn_idx]
    if values.dim() == 1:
        smooth = (neigh * nn_weights).sum(dim=1) / nn_weights.sum(dim=1).clamp_min(1e-6)
    else:
        smooth = (neigh * nn_weights.unsqueeze(-1)).sum(dim=1) / nn_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return smooth


def _get_boundary_tag_candidate_score(boundary_score, deformed_gaussian, config):
    if boundary_score is None:
        return None

    score = boundary_score
    power = float(config.opt.get('boundary_tag_score_power', 1.0))
    if power != 1.0:
        score = score.pow(power)

    min_score = float(config.opt.get('boundary_tag_min_score', 0.0))
    if min_score > 0.0:
        score = torch.where(score >= min_score, score, torch.zeros_like(score))

    thin_suppress_weight = float(config.opt.get('boundary_tag_thin_suppress_weight', 0.0))
    thin_score = getattr(deformed_gaussian, 'binding_thin_score', None)
    if thin_suppress_weight > 0.0 and torch.is_tensor(thin_score) and thin_score.numel() == score.numel():
        thin_score = thin_score.detach().float().clamp(0.0, 1.0)
        suppress = (1.0 - thin_suppress_weight * thin_score).clamp(0.0, 1.0)
        score = score * suppress

    smooth_blend = float(config.opt.get('boundary_tag_score_smooth_blend', 0.0))
    smooth_k = int(config.opt.get('boundary_tag_score_smooth_k', 0))
    if smooth_blend > 0.0 and smooth_k > 0:
        smooth_positions = getattr(deformed_gaussian, 'canonical_xyz', None)
        if not torch.is_tensor(smooth_positions) or smooth_positions.shape[0] != score.shape[0]:
            smooth_positions = getattr(deformed_gaussian, 'get_xyz', None)
        if torch.is_tensor(smooth_positions) and smooth_positions.shape[0] == score.shape[0]:
            smooth_score = _smooth_point_scores(
                score,
                smooth_positions,
                k=smooth_k,
                distance_quantile=float(config.opt.get('boundary_tag_score_smooth_distance_quantile', 0.5)),
            ).clamp(0.0, 1.0)
            score = torch.lerp(score, smooth_score, min(max(smooth_blend, 0.0), 1.0))

    support_k = int(config.opt.get('boundary_tag_support_k', 0))
    support_threshold = float(config.opt.get('boundary_tag_support_threshold', 0.0))
    if support_k > 0 and support_threshold > 0.0:
        support_positions = getattr(deformed_gaussian, 'canonical_xyz', None)
        if not torch.is_tensor(support_positions) or support_positions.shape[0] != score.shape[0]:
            support_positions = getattr(deformed_gaussian, 'get_xyz', None)
        if torch.is_tensor(support_positions) and support_positions.shape[0] == score.shape[0]:
            support_signal = (score > 0.0).float()
            support_score = _smooth_point_scores(
                support_signal,
                support_positions,
                k=support_k,
                distance_quantile=float(config.opt.get('boundary_tag_support_distance_quantile', 0.5)),
            ).clamp(0.0, 1.0)
            support_score = ((support_score - support_threshold) / max(1.0 - support_threshold, 1.0e-6)).clamp(0.0, 1.0)
            support_power = float(config.opt.get('boundary_tag_support_power', 1.0))
            if support_power != 1.0:
                support_score = support_score.pow(support_power)
            score = score * support_score

    return score.clamp(0.0, 1.0)


def _get_boundary_score_tensor(deformed_gaussian, prefer_mixed=True):
    if deformed_gaussian is None:
        return None

    attr_names = ['binding_boundary_mixed_score', 'binding_boundary_score'] if bool(prefer_mixed) else [
        'binding_boundary_score',
        'binding_boundary_mixed_score',
    ]
    for attr_name in attr_names:
        score = getattr(deformed_gaussian, attr_name, None)
        if torch.is_tensor(score):
            return score.detach().float().reshape(-1).clamp(0.0, 1.0)
    return None


def _get_boundary_aware_score(deformed_gaussian, config):
    if not bool(config.opt.get('boundary_aware_enable', False)):
        return None
    score = _get_boundary_score_tensor(deformed_gaussian, prefer_mixed=True)
    if not torch.is_tensor(score) or score.numel() == 0:
        return None
    score = score.detach().float().clamp(0.0, 1.0)
    power = float(config.opt.get('boundary_aware_score_power', 1.0))
    min_keep = float(config.opt.get('boundary_aware_min_keep', 0.0))
    threshold = float(config.opt.get('boundary_aware_threshold', 0.0))
    if power != 1.0:
        score = score.pow(power)
    if threshold > 0.0:
        score = torch.where(score >= threshold, score, torch.zeros_like(score))
    if min_keep > 0.0:
        score = torch.clamp(score, min=min_keep, max=1.0)
    return score


def _boundary_live_score_cache_enabled(config):
    return bool(config.opt.get('boundary_live_score_cache_enable', False))


def _texture_boundary_head_enabled(config):
    try:
        boundary_cfg = (
            config.get('model', {})
            .get('texture', {})
            .get('structured_trunk', {})
            .get('output_head', {})
            .get('local_color', {})
            .get('owner', {})
            .get('head', {})
            .get('boundary', {})
        )
        return bool(boundary_cfg.get('enable', False))
    except Exception:
        return False


def _validate_boundary_live_score_cache_config(config):
    if not _boundary_live_score_cache_enabled(config):
        return
    require_head = bool(
        config.opt.get('boundary_live_score_cache_require_texture_boundary_head', True)
    )
    if require_head and not _texture_boundary_head_enabled(config):
        raise ValueError(
            'boundary_live_score_cache_enable=true requires '
            'model.texture.structured_trunk.output_head.local_color.owner.head.'
            'boundary.enable=true. Disable '
            'opt.boundary_live_score_cache_require_texture_boundary_head only for '
            'non-texture diagnostic runs.'
        )


def _boundary_live_score_cache_key(data, config):
    mode = str(config.opt.get('boundary_live_score_cache_key', 'camera')).lower()
    image_name = str(getattr(data, 'image_name', ''))
    cam_id = str(getattr(data, 'cam_id', 'unknown_cam'))
    frame_id = str(getattr(data, 'frame_id', 'unknown_frame'))
    if mode in ('image', 'view', 'camera_frame', 'frame_camera'):
        return image_name or f'c{cam_id}_f{frame_id}'
    if mode == 'frame':
        return f'f{frame_id}'
    if mode == 'camera':
        return f'c{cam_id}'
    return image_name or f'c{cam_id}_f{frame_id}'


def _apply_boundary_live_score_cache(scene, data, cache, config):
    if not _boundary_live_score_cache_enabled(config):
        return None, False
    if not hasattr(scene.gaussians, 'set_live_boundary_score_state'):
        return None, False
    key = _boundary_live_score_cache_key(data, config)
    cached_score = cache.get(key)
    point_count = int(scene.gaussians.get_xyz.shape[0])
    if torch.is_tensor(cached_score) and cached_score.numel() == point_count:
        scene.gaussians.set_live_boundary_score_state(
            cached_score.to(device=scene.gaussians.get_xyz.device, dtype=torch.float32)
        )
        return key, True
    if cached_score is not None:
        cache.pop(key, None)
    if bool(config.opt.get('boundary_live_score_cache_clear_missing', True)):
        scene.gaussians.set_live_boundary_score_state(None)
    return key, False


def _update_boundary_live_score_cache(cache, key, boundary_score, config):
    if key is None or not _boundary_live_score_cache_enabled(config):
        return
    if not torch.is_tensor(boundary_score) or boundary_score.numel() == 0:
        return
    score = boundary_score.detach().reshape(-1).float().clamp(0.0, 1.0).cpu()
    ema = float(config.opt.get('boundary_live_score_cache_ema', 0.0))
    old_score = cache.get(key)
    if torch.is_tensor(old_score) and old_score.shape == score.shape and ema > 0.0:
        ema = min(max(ema, 0.0), 0.999)
        score = old_score.float() * ema + score * (1.0 - ema)
    cache[key] = score
    max_entries = int(config.opt.get('boundary_live_score_cache_max_entries', 64))
    while max_entries > 0 and len(cache) > max_entries:
        cache.pop(next(iter(cache)))


def _select_boundary_subset(boundary_score, mode='threshold', threshold=0.6, topk_ratio=0.08, min_ratio=0.0, binary=True):
    if boundary_score is None:
        return None

    subset = torch.zeros_like(boundary_score)
    if mode == 'threshold':
        subset = (boundary_score >= float(threshold)).float()
    elif mode == 'topk_ratio':
        topk_ratio = min(max(float(topk_ratio), 0.0), 1.0)
        if topk_ratio > 0.0:
            k = max(int(np.ceil(boundary_score.numel() * topk_ratio)), 1)
            cutoff = torch.topk(boundary_score, k=k, sorted=True).values[-1]
            subset = (boundary_score >= cutoff).float()
    else:
        raise ValueError(f'Unsupported boundary subset mode: {mode}')

    min_ratio = min(max(float(min_ratio), 0.0), 1.0)
    if min_ratio > 0.0 and float(subset.mean().item()) < min_ratio:
        k = max(int(np.ceil(boundary_score.numel() * min_ratio)), 1)
        cutoff = torch.topk(boundary_score, k=k, sorted=True).values[-1]
        subset = torch.maximum(subset, (boundary_score >= cutoff).float())

    if bool(binary):
        return subset
    return subset * boundary_score


def _make_boundary_subset_score(boundary_score, config):
    if boundary_score is None:
        return None
    if not bool(config.opt.get('boundary_subset_enable', False)):
        return boundary_score

    return _select_boundary_subset(
        boundary_score,
        mode=str(config.opt.get('boundary_subset_mode', 'threshold')),
        threshold=float(config.opt.get('boundary_subset_threshold', 0.6)),
        topk_ratio=float(config.opt.get('boundary_subset_topk_ratio', 0.08)),
        min_ratio=float(config.opt.get('boundary_subset_min_ratio', 0.0)),
        binary=bool(config.opt.get('boundary_subset_binary', True)),
    )


def _build_boundary_tag_mask(boundary_score, deformed_gaussian, config):
    candidate_score = _get_boundary_tag_candidate_score(boundary_score, deformed_gaussian, config)
    if candidate_score is None:
        return None
    return _select_boundary_subset(
        candidate_score,
        mode=str(config.opt.get('boundary_tag_mode', config.opt.get('boundary_subset_mode', 'threshold'))),
        threshold=float(config.opt.get('boundary_tag_threshold', config.opt.get('boundary_subset_threshold', 0.6))),
        topk_ratio=float(config.opt.get('boundary_tag_topk_ratio', config.opt.get('boundary_subset_topk_ratio', 0.08))),
        min_ratio=float(config.opt.get('boundary_tag_min_ratio', config.opt.get('boundary_subset_min_ratio', 0.0))),
        binary=bool(config.opt.get('boundary_tag_binary', True)),
    )


def _clear_scene_binding_state(scene):
    if hasattr(scene.gaussians, 'clear_binding_state'):
        scene.gaussians.clear_binding_state()
    rigid = getattr(getattr(scene.converter, 'deformer', None), 'rigid', None)
    if rigid is not None:
        if hasattr(rigid, 'binding_cache'):
            rigid.binding_cache = {}
        if hasattr(rigid, 'temporal_cache'):
            rigid.temporal_cache.clear()


def _set_texture_schedule_context(scene, local_iteration=None, schedule_iteration=None):
    texture = getattr(getattr(scene, 'converter', None), 'texture', None)
    setter = getattr(texture, 'set_schedule_context', None)
    if callable(setter):
        setter(
            local_iteration=local_iteration,
            schedule_iteration=schedule_iteration,
        )


def _maybe_refresh_resume_binding(scene, config, checkpoint_active, local_iteration, schedule_iteration):
    if not checkpoint_active:
        return False
    resume_cfg = config.get('resume', None)
    if resume_cfg is None:
        return False

    refresh_interval = int(resume_cfg.get('binding_refresh_interval', 0))
    refresh_until_iter = int(resume_cfg.get('binding_refresh_until_iter', 0))
    refresh_init_iter = int(resume_cfg.get('binding_refresh_init_iter', 0))
    if refresh_interval <= 0 or local_iteration < refresh_init_iter:
        return False
    if refresh_until_iter > 0 and local_iteration > refresh_until_iter:
        return False
    if (local_iteration - refresh_init_iter) % refresh_interval != 0:
        return False

    _clear_scene_binding_state(scene)
    print(
        f'[LOCAL ITER {local_iteration} | ITER {schedule_iteration}] '
        'cleared binding cache for resume rebind.'
    )
    return True


def _maybe_refresh_boundary_tags(scene, boundary_score, deformed_gaussian, config, iteration, local_iteration=None):
    if not bool(config.opt.get('boundary_tag_enable', False)):
        return scene.gaussians.get_boundary_tags()
    if boundary_score is None:
        return scene.gaussians.get_boundary_tags()

    scene.gaussians.ensure_boundary_state_matches_points(verbose=False)

    boundary_tag_iteration = local_iteration if (
        local_iteration is not None
        and bool(config.opt.get('boundary_tag_schedule_use_local_iteration', False))
    ) else iteration

    init_iter = int(config.opt.get('boundary_tag_init_iter', 0))
    if boundary_tag_iteration < init_iter:
        return scene.gaussians.get_boundary_tags()

    update_interval = int(config.opt.get('boundary_tag_update_interval', 0))
    update_until_iter = int(config.opt.get('boundary_tag_update_until_iter', init_iter))
    has_tags = scene.gaussians.has_boundary_tags()
    should_update = not has_tags
    if not should_update and update_interval > 0 and boundary_tag_iteration <= update_until_iter:
        should_update = ((boundary_tag_iteration - init_iter) % update_interval == 0)
    if not should_update:
        return scene.gaussians.get_boundary_tags()

    boundary_tag = _build_boundary_tag_mask(boundary_score, deformed_gaussian, config)
    if boundary_tag is None:
        return scene.gaussians.get_boundary_tags()
    if has_tags and bool(config.opt.get('boundary_tag_union_with_existing', False)):
        boundary_tag = torch.maximum(scene.gaussians.get_boundary_tags().to(boundary_tag.device), boundary_tag)

    scene.gaussians.set_boundary_tags(boundary_tag)
    if bool(config.opt.get('boundary_tag_verbose', True)):
        tag_fraction = float((boundary_tag > 0).float().mean().item())
        tag_count = int((boundary_tag > 0).sum().item())
        print(f'[ITER {iteration}] boundary-tagged subset updated: {tag_count} / {boundary_tag.numel()} ({tag_fraction:.4f})')
    return scene.gaussians.get_boundary_tags()


def _get_boundary_effective_score(scene, boundary_score, deformed_gaussian, config, iteration, local_iteration=None):
    if bool(config.opt.get('boundary_tag_enable', False)):
        boundary_tag = _maybe_refresh_boundary_tags(
            scene,
            boundary_score,
            deformed_gaussian,
            config,
            iteration,
            local_iteration=local_iteration,
        )
        if boundary_tag is None:
            return None
        boundary_tag = boundary_tag.to(device=boundary_score.device if boundary_score is not None else scene.gaussians.get_xyz.device, dtype=torch.float32)
        if bool(config.opt.get('boundary_tag_use_score_within_subset', False)) and boundary_score is not None:
            candidate_score = _get_boundary_tag_candidate_score(boundary_score, deformed_gaussian, config)
            if candidate_score is not None:
                return boundary_tag * candidate_score
        return boundary_tag
    return _make_boundary_subset_score(boundary_score, config)


def _project_points_to_pixel_coords(points, camera):
    if not torch.is_tensor(points) or points.numel() == 0:
        return None, None, None

    ndc = geom_transform_points(points.detach(), camera.full_proj_transform)
    width = int(camera.image_width)
    height = int(camera.image_height)
    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)

    px = ((ndc[:, 0] + 1.0) * 0.5 * float(max(width - 1, 1))).round().long()
    py = ((1.0 - (ndc[:, 1] + 1.0) * 0.5) * float(max(height - 1, 1))).round().long()

    valid = torch.isfinite(ndc).all(dim=1)
    valid = valid & (ndc[:, 2] > 0.0)
    valid = valid & (px >= 0) & (px < width) & (py >= 0) & (py < height)

    px = px.clamp(0, max_x)
    py = py.clamp(0, max_y)
    return px, py, valid


def _sample_mask_at_pixels(mask, px, py, valid):
    if mask is None or px is None or py is None or valid is None:
        return None

    if mask.dim() == 3:
        mask_2d = mask[0]
    elif mask.dim() == 2:
        mask_2d = mask
    else:
        mask_2d = mask.reshape(mask.shape[-2], mask.shape[-1])

    sampled = torch.zeros((px.shape[0],), dtype=torch.float32, device=px.device)
    if bool(valid.any().item()):
        sampled[valid] = mask_2d[py[valid], px[valid]].to(device=px.device, dtype=torch.float32)
    return sampled


def _postprocess_boundary_point_score(score, positions, config):
    if not torch.is_tensor(score):
        return score

    score = score.detach().float().clamp(0.0, 1.0)
    gain = float(config.opt.get('boundary_image_error_score_gain', 1.0))
    if gain != 1.0:
        score = (score * gain).clamp(0.0, 1.0)

    power = float(config.opt.get('boundary_image_error_score_power', 1.0))
    if power != 1.0:
        score = score.pow(power)

    min_score = float(config.opt.get('boundary_image_error_score_min', 0.0))
    if min_score > 0.0:
        score = torch.where(score >= min_score, score, torch.zeros_like(score))

    smooth_blend = float(config.opt.get('boundary_image_error_score_smooth_blend', 0.0))
    smooth_k = int(config.opt.get('boundary_image_error_score_smooth_k', 0))
    if (
        smooth_blend > 0.0
        and smooth_k > 0
        and torch.is_tensor(positions)
        and positions.shape[0] == score.shape[0]
    ):
        smooth = _smooth_point_scores(
            score,
            positions,
            k=smooth_k,
            distance_quantile=float(config.opt.get('boundary_image_error_score_smooth_distance_quantile', 0.5)),
        ).clamp(0.0, 1.0)
        score = torch.lerp(score, smooth, min(max(smooth_blend, 0.0), 1.0))

    return score.clamp(0.0, 1.0)


def _build_boundary_image_error_point_score(
    deformed_gaussian,
    camera,
    opacity,
    gt_mask_boundary,
    visibility_filter,
    config,
    iteration=0,
):
    image_score_enable = bool(config.opt.get('boundary_image_error_score_enable', False))
    signed_enable = bool(config.opt.get('boundary_image_error_score_signed_enable', False))
    if (not image_score_enable and not signed_enable) or opacity is None or gt_mask_boundary is None:
        return None, None
    if deformed_gaussian is None:
        return None, None

    point_positions = getattr(deformed_gaussian, 'get_xyz', None)
    if not torch.is_tensor(point_positions):
        point_positions = getattr(deformed_gaussian, '_xyz', None)
    if not torch.is_tensor(point_positions) or point_positions.numel() == 0:
        return None, None

    opacity_2d = opacity[:1].detach().float().clamp(0.0, 1.0)
    target_2d = gt_mask_boundary[:1].detach().float().clamp(0.0, 1.0)

    pred_threshold = float(config.opt.get('boundary_image_error_pred_threshold', 0.42))
    target_threshold = float(config.opt.get('boundary_image_error_target_threshold', 0.50))
    band_width = int(config.opt.get(
        'boundary_image_error_score_band_width',
        config.opt.get('boundary_band_width', 4),
    ))
    focus_dilate = int(config.opt.get('boundary_image_error_score_focus_dilate', 0))

    pred_binary = (opacity_2d >= pred_threshold).float()
    target_binary = (target_2d >= target_threshold).float()
    pred_boundary = _foreground_boundary_mask(pred_binary, band_width)
    target_boundary = _foreground_boundary_mask(target_binary, band_width)

    focus_mask = torch.clamp(pred_boundary + target_boundary, 0.0, 1.0)
    if focus_dilate > 1:
        focus_mask = _binary_dilate(focus_mask, focus_dilate).clamp(0.0, 1.0)

    boundary_disagreement = torch.abs(pred_boundary - target_boundary)
    opacity_error = torch.abs(opacity_2d - target_2d)
    candidate_map = torch.maximum(boundary_disagreement, opacity_error * focus_mask).clamp(0.0, 1.0)

    px, py, proj_valid = _project_points_to_pixel_coords(point_positions, camera)
    if px is None or py is None or proj_valid is None:
        return None, None

    if torch.is_tensor(visibility_filter) and visibility_filter.shape[0] == proj_valid.shape[0]:
        proj_valid = proj_valid & visibility_filter.to(device=proj_valid.device, dtype=torch.bool)

    score_positions = getattr(deformed_gaussian, 'canonical_xyz', None)
    if not torch.is_tensor(score_positions) or score_positions.shape[0] != point_positions.shape[0]:
        score_positions = point_positions

    sampled_candidate = _sample_mask_at_pixels(candidate_map, px, py, proj_valid)
    if sampled_candidate is None:
        return None, None
    sampled_candidate = _postprocess_boundary_point_score(sampled_candidate, score_positions, config)

    if not signed_enable:
        return sampled_candidate, proj_valid

    under_map = torch.maximum(
        (target_boundary - pred_boundary).clamp_min(0.0),
        ((target_2d - opacity_2d).clamp_min(0.0) * focus_mask),
    ).clamp(0.0, 1.0)
    over_map = torch.maximum(
        (pred_boundary - target_boundary).clamp_min(0.0),
        ((opacity_2d - target_2d).clamp_min(0.0) * focus_mask),
    ).clamp(0.0, 1.0)

    sampled_under = _sample_mask_at_pixels(under_map, px, py, proj_valid)
    sampled_over = _sample_mask_at_pixels(over_map, px, py, proj_valid)
    sampled_under = _postprocess_boundary_point_score(sampled_under, score_positions, config)
    sampled_over = _postprocess_boundary_point_score(sampled_over, score_positions, config)
    sampled_candidate = torch.maximum(sampled_candidate, torch.maximum(sampled_under, sampled_over))
    return {
        'candidate': sampled_candidate,
        'under': sampled_under,
        'over': sampled_over,
    }, proj_valid


def _mix_boundary_prior_with_image_score(boundary_prior_score, boundary_image_score, boundary_image_valid, config, iteration=0):
    prior = boundary_prior_score
    image = boundary_image_score
    if not torch.is_tensor(prior) and not torch.is_tensor(image):
        return None
    if not torch.is_tensor(prior):
        return image.detach().float().clamp(0.0, 1.0)
    if not torch.is_tensor(image):
        return prior.detach().float().clamp(0.0, 1.0)

    prior = prior.detach().float().clamp(0.0, 1.0)
    image = image.detach().float().clamp(0.0, 1.0)
    if image.shape[0] != prior.shape[0]:
        return prior

    image_mix = float(C(iteration, config.opt.get('boundary_image_error_score_mix', 1.0)))
    image_mix = min(max(image_mix, 0.0), 1.0)
    prior_floor = float(config.opt.get('boundary_image_error_score_prior_floor', 0.0))

    if torch.is_tensor(boundary_image_valid) and boundary_image_valid.shape[0] == prior.shape[0]:
        valid_mask = boundary_image_valid.to(device=prior.device, dtype=torch.bool)
        image = torch.where(valid_mask, image, prior)

    mixed = torch.lerp(prior, torch.maximum(prior, image), image_mix)
    if prior_floor > 0.0:
        mixed = torch.maximum(mixed, prior * prior_floor)
    return mixed.clamp(0.0, 1.0)


def _build_directional_boundary_effective_score(
    boundary_effective_score,
    directional_image_score,
    candidate_image_score,
    config,
    direction='grow',
):
    if not torch.is_tensor(boundary_effective_score) or not torch.is_tensor(directional_image_score):
        return None
    if directional_image_score.shape[0] != boundary_effective_score.shape[0]:
        return None

    directional = directional_image_score.detach().float().clamp(0.0, 1.0)
    if (
        torch.is_tensor(candidate_image_score)
        and candidate_image_score.shape[0] == directional.shape[0]
    ):
        candidate = candidate_image_score.detach().float().clamp(0.0, 1.0)
        share = directional / candidate.clamp_min(1.0e-6)
    else:
        share = directional

    if str(direction).lower() == 'shrink':
        share_gain = float(config.opt.get(
            'boundary_signed_shrink_share_gain',
            config.opt.get('boundary_signed_share_gain', 1.0),
        ))
        share_power = float(config.opt.get(
            'boundary_signed_shrink_share_power',
            config.opt.get('boundary_signed_share_power', 1.0),
        ))
    else:
        share_gain = float(config.opt.get('boundary_signed_share_gain', 1.0))
        share_power = float(config.opt.get('boundary_signed_share_power', 1.0))

    share = (share * share_gain).clamp(0.0, 1.0)
    if share_power != 1.0:
        share = share.pow(share_power)
    return (boundary_effective_score.detach().float().clamp(0.0, 1.0) * share).clamp(0.0, 1.0)


def _make_boundary_param_mask(param, base_score, param_scale):
    if base_score is None:
        return None
    scale = float(param_scale)
    if scale <= 0.0:
        score = torch.zeros_like(base_score)
    elif scale >= 1.0:
        score = base_score
    else:
        score = torch.clamp(base_score * scale, 0.0, 1.0)
    if param.dim() == 1:
        return score
    view_shape = (score.shape[0],) + (1,) * (param.dim() - 1)
    return score.view(view_shape)


def _register_boundary_grad_hooks(scene, boundary_score, config):
    if boundary_score is None:
        return [], []

    hooks = []
    gaussian = scene.gaussians
    param_specs = [
        (gaussian._xyz, config.opt.get('boundary_aware_xyz_scale', 0.35)),
        (gaussian._features_dc, config.opt.get('boundary_aware_feature_dc_scale', 0.0)),
        (gaussian._features_rest, config.opt.get('boundary_aware_feature_rest_scale', 0.0)),
        (gaussian._opacity, config.opt.get('boundary_aware_opacity_scale', 1.0)),
        (gaussian._scaling, config.opt.get('boundary_aware_scaling_scale', 0.7)),
        (gaussian._rotation, config.opt.get('boundary_aware_rotation_scale', 0.25)),
        (gaussian._boundary_opacity_residual, config.opt.get('boundary_aware_boundary_opacity_residual_scale', 1.0)),
        (gaussian._boundary_scaling_residual, config.opt.get('boundary_aware_boundary_scaling_residual_scale', 1.0)),
    ]
    for param, param_scale in param_specs:
        mask = _make_boundary_param_mask(param, boundary_score, param_scale)
        if mask is None:
            continue
        hooks.append(param.register_hook(lambda grad, mask=mask: grad * mask))

    frozen_converter_params = []
    if bool(config.opt.get('boundary_aware_freeze_converter_for_boundary_loss', True)):
        for param in scene.converter.parameters():
            if param.requires_grad:
                param.requires_grad_(False)
                frozen_converter_params.append(param)

    return hooks, frozen_converter_params


def _remove_grad_hooks(hooks):
    for hook in hooks:
        hook.remove()


def _remove_boundary_grad_hooks(hooks, frozen_converter_params):
    _remove_grad_hooks(hooks)
    for param in frozen_converter_params:
        param.requires_grad_(True)


def _binary_dilate(mask, kernel_size):
    kernel_size = _normalize_kernel_size(kernel_size)
    if kernel_size <= 1:
        return mask
    return F.max_pool2d(mask.unsqueeze(0), kernel_size, stride=1, padding=kernel_size // 2).squeeze(0)


def _binary_erode(mask, kernel_size):
    kernel_size = _normalize_kernel_size(kernel_size)
    if kernel_size <= 1:
        return mask
    return 1.0 - _binary_dilate(1.0 - mask, kernel_size)


def _foreground_boundary_mask(fg_mask, kernel_size):
    kernel_size = _normalize_kernel_size(kernel_size)
    fg_mask = (fg_mask > 0.5).float()
    if kernel_size <= 1:
        return torch.zeros_like(fg_mask)
    dilated = _binary_dilate(fg_mask, kernel_size)
    eroded = _binary_erode(fg_mask, kernel_size)
    return (dilated - eroded).clamp(0.0, 1.0)


def _foreground_outer_ring_mask(fg_mask, kernel_size):
    kernel_size = _normalize_kernel_size(kernel_size)
    fg_mask = (fg_mask > 0.5).float()
    if kernel_size <= 1:
        return torch.zeros_like(fg_mask)
    dilated = _binary_dilate(fg_mask, kernel_size)
    return (dilated - fg_mask).clamp(0.0, 1.0)


def _detail_interior_mask(region_mask, fg_mask, config_opt=None, region_name=''):
    if region_mask is None:
        return None
    interior = region_mask.float().clamp(0.0, 1.0)
    if config_opt is None:
        return interior

    region_prefix = f'{region_name}_' if region_name else ''
    boundary_keep_source = interior
    erode_kernel_size = int(
        config_opt.get(
            f'{region_prefix}detail_interior_erode_kernel_size',
            config_opt.get('detail_interior_erode_kernel_size', 0),
        )
    )
    if erode_kernel_size > 1:
        interior = _binary_erode(interior, erode_kernel_size).clamp(0.0, 1.0)

    boundary_exclude_width = int(
        config_opt.get(
            f'{region_prefix}detail_interior_exclude_boundary_width',
            config_opt.get('detail_interior_exclude_boundary_width', 0),
        )
    )
    if boundary_exclude_width > 1:
        interior = interior * (1.0 - _foreground_boundary_mask(fg_mask, boundary_exclude_width))

    boundary_keep_weight = float(
        config_opt.get(
            f'{region_prefix}detail_boundary_keep_weight',
            config_opt.get('detail_boundary_keep_weight', 0.0),
        )
    )
    boundary_keep_width = int(
        config_opt.get(
            f'{region_prefix}detail_boundary_keep_width',
            config_opt.get('detail_boundary_keep_width', boundary_exclude_width),
        )
    )
    boundary_keep_use_pre_erode = bool(
        config_opt.get(
            f'{region_prefix}detail_boundary_keep_use_pre_erode',
            config_opt.get('detail_boundary_keep_use_pre_erode', True),
        )
    )
    if boundary_keep_weight > 0.0 and boundary_keep_width > 1:
        keep_source = boundary_keep_source if boundary_keep_use_pre_erode else interior
        boundary_keep = keep_source * _foreground_boundary_mask(fg_mask, boundary_keep_width)
        boundary_keep = boundary_keep * float(min(max(boundary_keep_weight, 0.0), 1.0))
        interior = torch.maximum(interior, boundary_keep)

    return interior.clamp(0.0, 1.0)


def _build_local_focus_core_mask(region_mask, fg_mask, config_opt=None, region_name=''):
    if region_mask is None:
        return None

    source_mask = region_mask.float().clamp(0.0, 1.0) * fg_mask
    if config_opt is None:
        return source_mask

    region_prefix = f'{region_name}_' if region_name else ''
    core_mask = source_mask

    erode_kernel_size = int(config_opt.get(f'{region_prefix}erode_kernel_size', 0))
    if erode_kernel_size > 1 and core_mask.sum().item() > 0:
        eroded_mask = _binary_erode(core_mask, erode_kernel_size).clamp(0.0, 1.0) * fg_mask
        if eroded_mask.sum().item() > 0:
            core_mask = eroded_mask

    center_width_ratio = float(config_opt.get(f'{region_prefix}center_width_ratio', 1.0))
    top_trim_ratio = float(config_opt.get(f'{region_prefix}top_trim_ratio', 0.0))
    bottom_trim_ratio = float(config_opt.get(f'{region_prefix}bottom_trim_ratio', 0.0))
    window_min_pixels = int(config_opt.get(f'{region_prefix}window_min_pixels', 8))
    fallback_to_source = bool(config_opt.get(f'{region_prefix}fallback_to_source', True))

    if center_width_ratio < 0.999 or top_trim_ratio > 0.0 or bottom_trim_ratio > 0.0:
        center_width_ratio = min(max(center_width_ratio, 0.15), 1.0)
        top_trim_ratio = min(max(top_trim_ratio, 0.0), 0.70)
        bottom_trim_ratio = min(max(bottom_trim_ratio, 0.0), 0.70)

        if top_trim_ratio + bottom_trim_ratio >= 0.90:
            total_trim = max(top_trim_ratio + bottom_trim_ratio, 1.0e-6)
            scale = 0.90 / total_trim
            top_trim_ratio *= scale
            bottom_trim_ratio *= scale

        y1, y2, x1, x2 = _foreground_bbox_from_mask(core_mask, padding=0)
        region_h = max(y2 - y1, 1)
        region_w = max(x2 - x1, 1)
        center_x = 0.5 * float(x1 + x2)
        half_width = max(int(round(region_w * center_width_ratio * 0.5)), 1)
        left = max(int(round(center_x - half_width)), x1)
        right = min(int(round(center_x + half_width)), x2)
        top = min(max(int(round(y1 + region_h * top_trim_ratio)), y1), y2 - 1)
        bottom = max(min(int(round(y2 - region_h * bottom_trim_ratio)), y2), top + 1)

        window_mask = torch.zeros_like(core_mask)
        window_mask[:, top:bottom, left:right] = 1.0
        cropped_core_mask = core_mask * window_mask * fg_mask
        if cropped_core_mask.sum().item() >= max(window_min_pixels, 1):
            core_mask = cropped_core_mask
        elif not fallback_to_source:
            core_mask = cropped_core_mask

    return core_mask.clamp(0.0, 1.0)


def _foreground_outer_shell_mask(fg_mask, inner_kernel_size, outer_kernel_size):
    inner_kernel_size = _normalize_kernel_size(inner_kernel_size)
    outer_kernel_size = _normalize_kernel_size(outer_kernel_size)
    fg_mask = (fg_mask > 0.5).float()
    if outer_kernel_size <= 1 or outer_kernel_size <= inner_kernel_size:
        return torch.zeros_like(fg_mask)
    outer_dilated = _binary_dilate(fg_mask, outer_kernel_size)
    if inner_kernel_size <= 1:
        inner_dilated = fg_mask
    else:
        inner_dilated = _binary_dilate(fg_mask, inner_kernel_size)
    return (outer_dilated - inner_dilated).clamp(0.0, 1.0)


def _foreground_outer_shell_weight_mask(fg_mask, inner_kernel_size, outer_kernel_size, min_weight=0.25):
    inner_kernel_size = _normalize_kernel_size(inner_kernel_size)
    outer_kernel_size = _normalize_kernel_size(outer_kernel_size)
    fg_mask = (fg_mask > 0.5).float()
    if outer_kernel_size <= 1 or outer_kernel_size <= inner_kernel_size:
        return torch.zeros_like(fg_mask)

    min_weight = float(min(max(min_weight, 0.0), 1.0))
    shell_mask = torch.zeros_like(fg_mask)
    prev = fg_mask if inner_kernel_size <= 1 else _binary_dilate(fg_mask, inner_kernel_size)
    shell_steps = list(range(inner_kernel_size + 2, outer_kernel_size + 1, 2))
    if not shell_steps:
        return _foreground_outer_shell_mask(fg_mask, inner_kernel_size, outer_kernel_size)

    denom = max(len(shell_steps) - 1, 1)
    for idx, kernel_size in enumerate(shell_steps):
        current = _binary_dilate(fg_mask, kernel_size)
        ring = (current - prev).clamp(0.0, 1.0)
        t = float(idx) / float(denom)
        weight = 1.0 - t * (1.0 - min_weight)
        shell_mask = torch.maximum(shell_mask, ring * weight)
        prev = current
    return shell_mask


def _foreground_outer_spike_mask(opacity, outer_shell_mask, support_kernel_size=7, opacity_threshold=0.08, support_threshold=0.35, power=1.0):
    if opacity is None or outer_shell_mask is None:
        ref = outer_shell_mask if outer_shell_mask is not None else opacity
        if torch.is_tensor(ref):
            return torch.zeros_like(ref)
        return None

    support_kernel_size = _normalize_kernel_size(support_kernel_size)
    opacity_threshold = float(min(max(opacity_threshold, 0.0), 1.0))
    support_threshold = float(min(max(support_threshold, 0.0), 1.0))
    power = float(max(power, 1.0e-6))

    shell_weight = outer_shell_mask.to(device=opacity.device, dtype=opacity.dtype).clamp(0.0, 1.0)
    shell_binary = (shell_weight > 0.0).float()
    if shell_binary.sum().item() <= 0:
        return torch.zeros_like(shell_weight)

    opacity_detached = opacity.detach().to(dtype=shell_weight.dtype)
    pred_outer = ((opacity_detached >= opacity_threshold).float() * shell_binary)
    if pred_outer.sum().item() <= 0:
        return torch.zeros_like(shell_weight)

    if support_kernel_size > 1:
        pred_support = F.avg_pool2d(pred_outer.unsqueeze(0), support_kernel_size, stride=1, padding=support_kernel_size // 2).squeeze(0)
        shell_support = F.avg_pool2d(shell_binary.unsqueeze(0), support_kernel_size, stride=1, padding=support_kernel_size // 2).squeeze(0).clamp_min(1.0e-6)
        support_ratio = (pred_support / shell_support).clamp(0.0, 1.0)
    else:
        support_ratio = pred_outer

    if support_threshold <= 0.0:
        isolation_score = torch.ones_like(support_ratio)
    else:
        isolation_score = ((support_threshold - support_ratio) / max(support_threshold, 1.0e-6)).clamp(0.0, 1.0)
    if power != 1.0:
        isolation_score = isolation_score.pow(power)

    opacity_score = ((opacity_detached - opacity_threshold) / max(1.0 - opacity_threshold, 1.0e-6)).clamp(0.0, 1.0)
    return (shell_weight * pred_outer * isolation_score * opacity_score).clamp(0.0, 1.0)


def _foreground_outer_fragment_mask(opacity, outer_shell_mask, region_mask=None, opacity_threshold=0.08, component_area_min=2, component_area_max=160, fill_ratio_max=0.60):
    if opacity is None or outer_shell_mask is None:
        ref = outer_shell_mask if outer_shell_mask is not None else opacity
        if torch.is_tensor(ref):
            return torch.zeros_like(ref)
        return None

    opacity_threshold = float(min(max(opacity_threshold, 0.0), 1.0))
    component_area_min = max(int(component_area_min), 1)
    component_area_max = max(int(component_area_max), component_area_min)
    fill_ratio_max = float(min(max(fill_ratio_max, 0.0), 1.0))

    shell_weight = outer_shell_mask.to(device=opacity.device, dtype=opacity.dtype).clamp(0.0, 1.0)
    candidate = ((opacity.detach().to(dtype=shell_weight.dtype) >= opacity_threshold).float() * (shell_weight > 0.0).float())
    if region_mask is not None:
        candidate = candidate * (region_mask.to(device=opacity.device, dtype=shell_weight.dtype) > 0.0).float()
    if candidate.sum().item() <= 0:
        return torch.zeros_like(shell_weight)

    candidate_np = (candidate[0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_np, connectivity=8)
    if num_labels <= 1:
        return torch.zeros_like(shell_weight)

    selected = np.zeros_like(candidate_np, dtype=np.float32)
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < component_area_min or area > component_area_max:
            continue
        width = max(int(stats[idx, cv2.CC_STAT_WIDTH]), 1)
        height = max(int(stats[idx, cv2.CC_STAT_HEIGHT]), 1)
        fill_ratio = float(area) / float(width * height)
        if fill_ratio > fill_ratio_max:
            continue
        selected[labels == idx] = 1.0

    if float(selected.sum()) <= 0.0:
        return torch.zeros_like(shell_weight)

    selected_mask = torch.from_numpy(selected).to(device=opacity.device, dtype=shell_weight.dtype).unsqueeze(0)
    return (selected_mask * shell_weight).clamp(0.0, 1.0)


def _foreground_outer_bead_mask(opacity, outer_shell_mask, region_mask=None, opacity_threshold=0.08, opening_kernel_size=5, support_kernel_size=7, support_threshold=0.55, power=1.0):
    if opacity is None or outer_shell_mask is None:
        ref = outer_shell_mask if outer_shell_mask is not None else opacity
        if torch.is_tensor(ref):
            return torch.zeros_like(ref)
        return None

    opening_kernel_size = _normalize_kernel_size(opening_kernel_size)
    support_kernel_size = _normalize_kernel_size(support_kernel_size)
    opacity_threshold = float(min(max(opacity_threshold, 0.0), 1.0))
    support_threshold = float(min(max(support_threshold, 0.0), 1.0))
    power = float(max(power, 1.0e-6))

    shell_weight = outer_shell_mask.to(device=opacity.device, dtype=opacity.dtype).clamp(0.0, 1.0)
    shell_binary = (shell_weight > 0.0).float()
    candidate = (opacity.detach().to(dtype=shell_weight.dtype) >= opacity_threshold).float() * shell_binary
    if region_mask is not None:
        candidate = candidate * (region_mask.to(device=opacity.device, dtype=shell_weight.dtype) > 0.0).float()
    if candidate.sum().item() <= 0:
        return torch.zeros_like(shell_weight)

    if opening_kernel_size > 1:
        opened = _binary_dilate(_binary_erode(candidate, opening_kernel_size), opening_kernel_size).clamp(0.0, 1.0)
    else:
        opened = candidate
    bead_binary = (candidate - opened).clamp(0.0, 1.0)
    if bead_binary.sum().item() <= 0:
        return torch.zeros_like(shell_weight)

    if support_kernel_size > 1:
        pred_support = F.avg_pool2d(candidate.unsqueeze(0), support_kernel_size, stride=1, padding=support_kernel_size // 2).squeeze(0)
        shell_support = F.avg_pool2d(shell_binary.unsqueeze(0), support_kernel_size, stride=1, padding=support_kernel_size // 2).squeeze(0).clamp_min(1.0e-6)
        support_ratio = (pred_support / shell_support).clamp(0.0, 1.0)
    else:
        support_ratio = candidate

    if support_threshold <= 0.0:
        support_score = torch.ones_like(support_ratio)
    else:
        support_score = ((support_threshold - support_ratio) / max(support_threshold, 1.0e-6)).clamp(0.0, 1.0)
    if power != 1.0:
        support_score = support_score.pow(power)

    opacity_score = ((opacity.detach().to(dtype=shell_weight.dtype) - opacity_threshold) / max(1.0 - opacity_threshold, 1.0e-6)).clamp(0.0, 1.0)
    return (bead_binary * shell_weight * support_score * opacity_score).clamp(0.0, 1.0)


def _binary_reconstruct(seed, mask, max_steps=16):
    mask = (mask > 0.5).float()
    current = (seed > 0.5).float() * mask
    if mask.sum().item() <= 0 or current.sum().item() <= 0:
        return torch.zeros_like(mask)

    max_steps = max(int(max_steps), 1)
    for _ in range(max_steps):
        grown = (_binary_dilate(current, 3) * mask).clamp(0.0, 1.0)
        if (grown - current).clamp_min(0.0).sum().item() <= 0:
            break
        current = grown
    return current


def _foreground_outer_chain_mask(
    opacity,
    outer_shell_mask,
    region_mask=None,
    opacity_threshold=0.08,
    support_kernel_size=7,
    seed_support_threshold=0.65,
    propagate_support_threshold=0.40,
    anchor_weight_threshold=0.70,
    max_steps=16,
    power=1.0,
):
    if opacity is None or outer_shell_mask is None:
        ref = outer_shell_mask if outer_shell_mask is not None else opacity
        if torch.is_tensor(ref):
            return torch.zeros_like(ref)
        return None

    support_kernel_size = _normalize_kernel_size(support_kernel_size)
    opacity_threshold = float(min(max(opacity_threshold, 0.0), 1.0))
    seed_support_threshold = float(min(max(seed_support_threshold, 0.0), 1.0))
    propagate_support_threshold = float(min(max(propagate_support_threshold, 0.0), 1.0))
    anchor_weight_threshold = float(min(max(anchor_weight_threshold, 0.0), 1.0))
    power = float(max(power, 1.0e-6))

    shell_weight = outer_shell_mask.to(device=opacity.device, dtype=opacity.dtype).clamp(0.0, 1.0)
    shell_binary = (shell_weight > 0.0).float()
    candidate = (opacity.detach().to(dtype=shell_weight.dtype) >= opacity_threshold).float() * shell_binary
    if region_mask is not None:
        candidate = candidate * (region_mask.to(device=opacity.device, dtype=shell_weight.dtype) > 0.0).float()
    if candidate.sum().item() <= 0:
        return torch.zeros_like(shell_weight)

    if support_kernel_size > 1:
        pred_support = F.avg_pool2d(candidate.unsqueeze(0), support_kernel_size, stride=1, padding=support_kernel_size // 2).squeeze(0)
        shell_support = F.avg_pool2d(shell_binary.unsqueeze(0), support_kernel_size, stride=1, padding=support_kernel_size // 2).squeeze(0).clamp_min(1.0e-6)
        support_ratio = (pred_support / shell_support).clamp(0.0, 1.0)
    else:
        support_ratio = candidate

    anchor_binary = candidate * (shell_weight >= anchor_weight_threshold).float()
    seed_binary = anchor_binary
    if seed_support_threshold <= 0.0:
        seed_binary = torch.maximum(seed_binary, candidate)
    else:
        seed_binary = torch.maximum(seed_binary, candidate * (support_ratio >= seed_support_threshold).float())
    if seed_binary.sum().item() <= 0:
        return torch.zeros_like(shell_weight)

    if propagate_support_threshold <= 0.0:
        eligible_binary = candidate
    else:
        eligible_binary = candidate * torch.maximum(
            (support_ratio >= propagate_support_threshold).float(),
            (shell_weight >= anchor_weight_threshold).float(),
        )

    keep_binary = _binary_reconstruct(seed_binary, eligible_binary, max_steps=max_steps)
    chain_binary = (candidate - keep_binary).clamp(0.0, 1.0)
    if chain_binary.sum().item() <= 0:
        return torch.zeros_like(shell_weight)

    if seed_support_threshold <= 0.0:
        chain_score = torch.ones_like(support_ratio)
    else:
        chain_score = ((seed_support_threshold - support_ratio) / max(seed_support_threshold, 1.0e-6)).clamp(0.0, 1.0)
    if power != 1.0:
        chain_score = chain_score.pow(power)

    opacity_score = ((opacity.detach().to(dtype=shell_weight.dtype) - opacity_threshold) / max(1.0 - opacity_threshold, 1.0e-6)).clamp(0.0, 1.0)
    return (chain_binary * shell_weight * chain_score * opacity_score).clamp(0.0, 1.0)


def _foreground_arm_boundary_tail_mask(
    opacity,
    fg_mask,
    region_mask=None,
    opacity_threshold=0.06,
    boundary_band_kernel_size=5,
    outer_shell_start_width=1,
    outer_shell_end_width=11,
    opening_kernel_size=7,
    closing_kernel_size=5,
    support_kernel_size=9,
    support_threshold=0.72,
    component_area_min=2,
    component_area_max=36,
    aspect_ratio_min=1.25,
    fill_ratio_max=0.62,
    touch_ratio_min=0.18,
    power=1.0,
):
    if opacity is None or fg_mask is None:
        ref = fg_mask if fg_mask is not None else opacity
        if torch.is_tensor(ref):
            return torch.zeros_like(ref)
        return None

    opacity_threshold = float(min(max(opacity_threshold, 0.0), 1.0))
    boundary_band_kernel_size = _normalize_kernel_size(boundary_band_kernel_size)
    outer_shell_start_width = max(int(outer_shell_start_width), 0)
    outer_shell_end_width = max(int(outer_shell_end_width), outer_shell_start_width)
    opening_kernel_size = _normalize_kernel_size(opening_kernel_size)
    closing_kernel_size = _normalize_kernel_size(closing_kernel_size)
    support_kernel_size = _normalize_kernel_size(support_kernel_size)
    support_threshold = float(min(max(support_threshold, 0.0), 1.0))
    component_area_min = max(int(component_area_min), 1)
    component_area_max = max(int(component_area_max), component_area_min)
    aspect_ratio_min = float(max(aspect_ratio_min, 1.0))
    fill_ratio_max = float(min(max(fill_ratio_max, 0.0), 1.0))
    touch_ratio_min = float(min(max(touch_ratio_min, 0.0), 1.0))
    power = float(max(power, 1.0e-6))

    fg_binary = (fg_mask.to(device=opacity.device, dtype=opacity.dtype) > 0.5).float()
    if region_mask is None:
        region_binary = torch.ones_like(fg_binary)
    else:
        region_binary = (region_mask.to(device=opacity.device, dtype=opacity.dtype) > 0.0).float()
    if region_binary.sum().item() <= 0:
        return torch.zeros_like(fg_binary)

    opacity_detached = opacity.detach().to(dtype=fg_binary.dtype)
    pred_binary = (opacity_detached >= opacity_threshold).float() * region_binary
    if pred_binary.sum().item() <= 0:
        return torch.zeros_like(fg_binary)

    boundary_band = _foreground_boundary_mask(fg_binary, boundary_band_kernel_size) * region_binary
    if outer_shell_end_width > 1:
        outer_shell = _foreground_outer_shell_mask(
            fg_binary,
            outer_shell_start_width,
            outer_shell_end_width,
        ) * region_binary
    else:
        outer_shell = torch.zeros_like(boundary_band)
    tail_band = torch.maximum(boundary_band, outer_shell).clamp(0.0, 1.0)
    if tail_band.sum().item() <= 0:
        return torch.zeros_like(fg_binary)

    smooth_binary = pred_binary
    if opening_kernel_size > 1:
        smooth_binary = _binary_dilate(_binary_erode(smooth_binary, opening_kernel_size), opening_kernel_size).clamp(0.0, 1.0)
    if closing_kernel_size > 1:
        smooth_binary = _binary_erode(_binary_dilate(smooth_binary, closing_kernel_size), closing_kernel_size).clamp(0.0, 1.0)
    smooth_binary = smooth_binary * region_binary

    tail_binary = ((pred_binary - smooth_binary).clamp(0.0, 1.0) * tail_band)
    if tail_binary.sum().item() <= 0:
        return torch.zeros_like(fg_binary)

    if support_kernel_size > 1:
        pred_support = F.avg_pool2d(pred_binary.unsqueeze(0), support_kernel_size, stride=1, padding=support_kernel_size // 2).squeeze(0)
        region_support = F.avg_pool2d(region_binary.unsqueeze(0), support_kernel_size, stride=1, padding=support_kernel_size // 2).squeeze(0).clamp_min(1.0e-6)
        support_ratio = (pred_support / region_support).clamp(0.0, 1.0)
    else:
        support_ratio = pred_binary

    tail_np = (tail_binary[0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    if int(tail_np.sum()) <= 0:
        return torch.zeros_like(fg_binary)
    outer_np = (outer_shell[0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    boundary_np = (boundary_band[0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    support_np = support_ratio[0].detach().cpu().numpy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tail_np, connectivity=8)
    if num_labels <= 1:
        return torch.zeros_like(fg_binary)

    selected = np.zeros_like(tail_np, dtype=np.float32)
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < component_area_min or area > component_area_max:
            continue

        width = max(int(stats[idx, cv2.CC_STAT_WIDTH]), 1)
        height = max(int(stats[idx, cv2.CC_STAT_HEIGHT]), 1)
        major = max(width, height)
        minor = max(min(width, height), 1)
        aspect_ratio = float(major) / float(minor)
        fill_ratio = float(area) / float(width * height)

        comp = (labels == idx)
        outer_touch = float(outer_np[comp].mean()) if np.any(comp) else 0.0
        boundary_touch = float(boundary_np[comp].mean()) if np.any(comp) else 0.0
        support_mean = float(support_np[comp].mean()) if np.any(comp) else 1.0

        if aspect_ratio < aspect_ratio_min and fill_ratio > fill_ratio_max:
            continue
        if max(outer_touch, boundary_touch) < touch_ratio_min:
            continue
        if support_mean > support_threshold:
            continue

        selected[comp] = 1.0

    if float(selected.sum()) <= 0.0:
        return torch.zeros_like(fg_binary)

    selected_mask = torch.from_numpy(selected).to(device=opacity.device, dtype=fg_binary.dtype).unsqueeze(0)
    if support_threshold <= 0.0:
        support_score = torch.ones_like(support_ratio)
    else:
        support_score = ((support_threshold - support_ratio) / max(support_threshold, 1.0e-6)).clamp(0.0, 1.0)
    if power != 1.0:
        support_score = support_score.pow(power)

    opacity_score = ((opacity_detached - opacity_threshold) / max(1.0 - opacity_threshold, 1.0e-6)).clamp(0.0, 1.0)
    return (selected_mask * tail_band * support_score * opacity_score).clamp(0.0, 1.0)


def _foreground_arm_boundary_fringe_mask(
    opacity,
    fg_mask,
    region_mask=None,
    opacity_threshold=0.055,
    boundary_band_kernel_size=5,
    outer_shell_start_width=1,
    outer_shell_end_width=13,
    opening_kernel_size=9,
    closing_kernel_size=7,
    support_kernel_size=11,
    support_threshold=0.78,
    component_area_min=4,
    component_area_max=120,
    fill_ratio_max=0.82,
    touch_ratio_min=0.22,
    power=1.0,
):
    if opacity is None or fg_mask is None:
        ref = fg_mask if fg_mask is not None else opacity
        if torch.is_tensor(ref):
            return torch.zeros_like(ref)
        return None

    opacity_threshold = float(min(max(opacity_threshold, 0.0), 1.0))
    boundary_band_kernel_size = _normalize_kernel_size(boundary_band_kernel_size)
    outer_shell_start_width = max(int(outer_shell_start_width), 0)
    outer_shell_end_width = max(int(outer_shell_end_width), outer_shell_start_width)
    opening_kernel_size = _normalize_kernel_size(opening_kernel_size)
    closing_kernel_size = _normalize_kernel_size(closing_kernel_size)
    support_kernel_size = _normalize_kernel_size(support_kernel_size)
    support_threshold = float(min(max(support_threshold, 0.0), 1.0))
    component_area_min = max(int(component_area_min), 1)
    component_area_max = max(int(component_area_max), component_area_min)
    fill_ratio_max = float(min(max(fill_ratio_max, 0.0), 1.0))
    touch_ratio_min = float(min(max(touch_ratio_min, 0.0), 1.0))
    power = float(max(power, 1.0e-6))

    fg_binary = (fg_mask.to(device=opacity.device, dtype=opacity.dtype) > 0.5).float()
    if region_mask is None:
        region_binary = torch.ones_like(fg_binary)
    else:
        region_binary = (region_mask.to(device=opacity.device, dtype=opacity.dtype) > 0.0).float()
    if region_binary.sum().item() <= 0:
        return torch.zeros_like(fg_binary)

    opacity_detached = opacity.detach().to(dtype=fg_binary.dtype)
    pred_binary = (opacity_detached >= opacity_threshold).float() * region_binary
    if pred_binary.sum().item() <= 0:
        return torch.zeros_like(fg_binary)

    boundary_band = _foreground_boundary_mask(fg_binary, boundary_band_kernel_size) * region_binary
    if outer_shell_end_width > 1:
        outer_shell = _foreground_outer_shell_mask(
            fg_binary,
            outer_shell_start_width,
            outer_shell_end_width,
        ) * region_binary
    else:
        outer_shell = torch.zeros_like(boundary_band)
    fringe_band = torch.maximum(boundary_band, outer_shell).clamp(0.0, 1.0)
    if fringe_band.sum().item() <= 0:
        return torch.zeros_like(fg_binary)

    coarse_binary = pred_binary
    if opening_kernel_size > 1:
        coarse_binary = _binary_dilate(_binary_erode(coarse_binary, opening_kernel_size), opening_kernel_size).clamp(0.0, 1.0)
    if closing_kernel_size > 1:
        coarse_binary = _binary_erode(_binary_dilate(coarse_binary, closing_kernel_size), closing_kernel_size).clamp(0.0, 1.0)
    coarse_binary = coarse_binary * region_binary

    fringe_binary = ((pred_binary - coarse_binary).clamp(0.0, 1.0) * fringe_band)
    if fringe_binary.sum().item() <= 0:
        return torch.zeros_like(fg_binary)

    if support_kernel_size > 1:
        pred_support = F.avg_pool2d(pred_binary.unsqueeze(0), support_kernel_size, stride=1, padding=support_kernel_size // 2).squeeze(0)
        region_support = F.avg_pool2d(region_binary.unsqueeze(0), support_kernel_size, stride=1, padding=support_kernel_size // 2).squeeze(0).clamp_min(1.0e-6)
        support_ratio = (pred_support / region_support).clamp(0.0, 1.0)
    else:
        support_ratio = pred_binary

    fringe_np = (fringe_binary[0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    if int(fringe_np.sum()) <= 0:
        return torch.zeros_like(fg_binary)
    outer_np = (outer_shell[0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    boundary_np = (boundary_band[0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    support_np = support_ratio[0].detach().cpu().numpy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fringe_np, connectivity=8)
    if num_labels <= 1:
        return torch.zeros_like(fg_binary)

    selected = np.zeros_like(fringe_np, dtype=np.float32)
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < component_area_min or area > component_area_max:
            continue

        width = max(int(stats[idx, cv2.CC_STAT_WIDTH]), 1)
        height = max(int(stats[idx, cv2.CC_STAT_HEIGHT]), 1)
        fill_ratio = float(area) / float(width * height)
        comp = (labels == idx)
        outer_touch = float(outer_np[comp].mean()) if np.any(comp) else 0.0
        boundary_touch = float(boundary_np[comp].mean()) if np.any(comp) else 0.0
        support_mean = float(support_np[comp].mean()) if np.any(comp) else 1.0

        if fill_ratio > fill_ratio_max:
            continue
        if max(outer_touch, boundary_touch) < touch_ratio_min:
            continue
        if support_mean > support_threshold:
            continue

        selected[comp] = 1.0

    if float(selected.sum()) <= 0.0:
        return torch.zeros_like(fg_binary)

    selected_mask = torch.from_numpy(selected).to(device=opacity.device, dtype=fg_binary.dtype).unsqueeze(0)
    if support_threshold <= 0.0:
        support_score = torch.ones_like(support_ratio)
    else:
        support_score = ((support_threshold - support_ratio) / max(support_threshold, 1.0e-6)).clamp(0.0, 1.0)
    if power != 1.0:
        support_score = support_score.pow(power)

    opacity_score = ((opacity_detached - opacity_threshold) / max(1.0 - opacity_threshold, 1.0e-6)).clamp(0.0, 1.0)
    return (selected_mask * fringe_band * support_score * opacity_score).clamp(0.0, 1.0)


def _foreground_arm_boundary_attached_fragment_mask(
    opacity,
    fg_mask,
    region_mask=None,
    opacity_threshold=0.055,
    boundary_band_kernel_size=5,
    outer_shell_start_width=1,
    outer_shell_end_width=11,
    component_area_min=1,
    component_area_max=64,
    fill_ratio_max=0.88,
    touch_ratio_min=0.18,
    power=1.0,
):
    if opacity is None or fg_mask is None:
        ref = fg_mask if fg_mask is not None else opacity
        if torch.is_tensor(ref):
            return torch.zeros_like(ref)
        return None

    opacity_threshold = float(min(max(opacity_threshold, 0.0), 1.0))
    boundary_band_kernel_size = _normalize_kernel_size(boundary_band_kernel_size)
    outer_shell_start_width = max(int(outer_shell_start_width), 0)
    outer_shell_end_width = max(int(outer_shell_end_width), outer_shell_start_width)
    component_area_min = max(int(component_area_min), 1)
    component_area_max = max(int(component_area_max), component_area_min)
    fill_ratio_max = float(min(max(fill_ratio_max, 0.0), 1.0))
    touch_ratio_min = float(min(max(touch_ratio_min, 0.0), 1.0))
    power = float(max(power, 1.0e-6))

    fg_binary = (fg_mask.to(device=opacity.device, dtype=opacity.dtype) > 0.5).float()
    if region_mask is None:
        region_binary = torch.ones_like(fg_binary)
    else:
        region_binary = (region_mask.to(device=opacity.device, dtype=opacity.dtype) > 0.0).float()
    if region_binary.sum().item() <= 0:
        return torch.zeros_like(fg_binary)

    opacity_detached = opacity.detach().to(dtype=fg_binary.dtype)
    pred_binary = (opacity_detached >= opacity_threshold).float() * region_binary
    if pred_binary.sum().item() <= 0:
        return torch.zeros_like(fg_binary)

    boundary_band = _foreground_boundary_mask(fg_binary, boundary_band_kernel_size) * region_binary
    if outer_shell_end_width > 1:
        outer_shell = _foreground_outer_shell_mask(
            fg_binary,
            outer_shell_start_width,
            outer_shell_end_width,
        ) * region_binary
    else:
        outer_shell = torch.zeros_like(boundary_band)
    attached_band = torch.maximum(boundary_band, outer_shell).clamp(0.0, 1.0)
    if attached_band.sum().item() <= 0:
        return torch.zeros_like(fg_binary)

    attached_binary = ((pred_binary - fg_binary).clamp(0.0, 1.0) * attached_band)
    if attached_binary.sum().item() <= 0:
        return torch.zeros_like(fg_binary)

    attached_np = (attached_binary[0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    if int(attached_np.sum()) <= 0:
        return torch.zeros_like(fg_binary)
    outer_np = (outer_shell[0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    boundary_np = (boundary_band[0].detach().cpu().numpy() > 0.5).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(attached_np, connectivity=8)
    if num_labels <= 1:
        return torch.zeros_like(fg_binary)

    selected = np.zeros_like(attached_np, dtype=np.float32)
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < component_area_min or area > component_area_max:
            continue

        width = max(int(stats[idx, cv2.CC_STAT_WIDTH]), 1)
        height = max(int(stats[idx, cv2.CC_STAT_HEIGHT]), 1)
        fill_ratio = float(area) / float(width * height)
        comp = (labels == idx)
        outer_touch = float(outer_np[comp].mean()) if np.any(comp) else 0.0
        boundary_touch = float(boundary_np[comp].mean()) if np.any(comp) else 0.0

        if fill_ratio > fill_ratio_max:
            continue
        if max(outer_touch, boundary_touch) < touch_ratio_min:
            continue
        selected[comp] = 1.0

    if float(selected.sum()) <= 0.0:
        return torch.zeros_like(fg_binary)

    selected_mask = torch.from_numpy(selected).to(device=opacity.device, dtype=fg_binary.dtype).unsqueeze(0)
    opacity_score = ((opacity_detached - opacity_threshold) / max(1.0 - opacity_threshold, 1.0e-6)).clamp(0.0, 1.0)
    if power != 1.0:
        opacity_score = opacity_score.pow(power)
    return (selected_mask * attached_band * opacity_score).clamp(0.0, 1.0)


def _select_small_binary_components(mask, area_min=1, area_max=64):
    if mask is None or not torch.is_tensor(mask):
        return None

    area_min = max(int(area_min), 1)
    area_max = max(int(area_max), area_min)
    binary = (mask > 0.5).float()
    if binary.sum().item() <= 0:
        return torch.zeros_like(binary)

    mask_np = (binary[0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    if num_labels <= 1:
        return torch.zeros_like(binary)

    selected = np.zeros_like(mask_np, dtype=np.float32)
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < area_min or area > area_max:
            continue
        selected[labels == idx] = 1.0

    if float(selected.sum()) <= 0.0:
        return torch.zeros_like(binary)
    return torch.from_numpy(selected).to(device=mask.device, dtype=mask.dtype).unsqueeze(0)


def _foreground_arm_boundary_roughness_masks(
    opacity,
    fg_mask,
    region_mask=None,
    opacity_threshold=0.06,
    boundary_band_kernel_size=5,
    opening_kernel_size=3,
    closing_kernel_size=5,
    stipple_area_min=1,
    stipple_area_max=24,
    notch_area_min=1,
    notch_area_max=24,
    stipple_power=1.0,
    notch_power=1.0,
):
    if opacity is None or fg_mask is None:
        ref = fg_mask if fg_mask is not None else opacity
        if torch.is_tensor(ref):
            zero = torch.zeros_like(ref)
            return zero, zero
        return None, None

    opacity_threshold = float(min(max(opacity_threshold, 0.0), 1.0))
    boundary_band_kernel_size = _normalize_kernel_size(boundary_band_kernel_size)
    opening_kernel_size = _normalize_kernel_size(opening_kernel_size)
    closing_kernel_size = _normalize_kernel_size(closing_kernel_size)
    stipple_power = float(max(stipple_power, 1.0e-6))
    notch_power = float(max(notch_power, 1.0e-6))

    fg_binary = (fg_mask.to(device=opacity.device, dtype=opacity.dtype) > 0.5).float()
    if region_mask is None:
        region_binary = torch.ones_like(fg_binary)
    else:
        region_binary = (region_mask.to(device=opacity.device, dtype=opacity.dtype) > 0.0).float()
    if region_binary.sum().item() <= 0:
        zero = torch.zeros_like(fg_binary)
        return zero, zero

    opacity_detached = opacity.detach().to(dtype=fg_binary.dtype)
    pred_binary = (opacity_detached >= opacity_threshold).float() * region_binary

    smooth_binary = pred_binary
    if opening_kernel_size > 1:
        smooth_binary = _binary_dilate(_binary_erode(smooth_binary, opening_kernel_size), opening_kernel_size).clamp(0.0, 1.0)
    if closing_kernel_size > 1:
        smooth_binary = _binary_erode(_binary_dilate(smooth_binary, closing_kernel_size), closing_kernel_size).clamp(0.0, 1.0)
    smooth_binary = smooth_binary * region_binary

    boundary_band = _foreground_boundary_mask(fg_binary, boundary_band_kernel_size) * region_binary
    inner_band = _foreground_inner_ring_mask(fg_binary, boundary_band_kernel_size) * region_binary
    if boundary_band.sum().item() <= 0:
        zero = torch.zeros_like(fg_binary)
        return zero, zero

    stipple_binary = ((pred_binary - smooth_binary).clamp(0.0, 1.0) * boundary_band)
    notch_binary = ((smooth_binary - pred_binary).clamp(0.0, 1.0) * inner_band)

    stipple_binary = _select_small_binary_components(
        stipple_binary,
        area_min=stipple_area_min,
        area_max=stipple_area_max,
    )
    notch_binary = _select_small_binary_components(
        notch_binary,
        area_min=notch_area_min,
        area_max=notch_area_max,
    )
    if stipple_binary is None or notch_binary is None:
        zero = torch.zeros_like(fg_binary)
        return zero, zero

    opacity_score = ((opacity_detached - opacity_threshold) / max(1.0 - opacity_threshold, 1.0e-6)).clamp(0.0, 1.0)
    if stipple_power != 1.0:
        opacity_score = opacity_score.pow(stipple_power)
    missing_score = (1.0 - opacity_detached).clamp(0.0, 1.0)
    if notch_power != 1.0:
        missing_score = missing_score.pow(notch_power)

    stipple_mask = (stipple_binary * boundary_band * opacity_score).clamp(0.0, 1.0)
    notch_mask = (notch_binary * inner_band * missing_score).clamp(0.0, 1.0)
    return stipple_mask, notch_mask


def _foreground_arm_hole_and_gap_masks(
    opacity,
    fg_mask,
    region_mask=None,
    opacity_threshold=0.08,
    gap_opacity_threshold=None,
    inner_band_kernel_size=5,
    inner_region_dilate=11,
    closing_kernel_size=9,
    hole_area_min=1,
    hole_area_max=48,
    gap_area_min=2,
    gap_area_max=96,
    hole_power=1.0,
    gap_power=1.0,
):
    if opacity is None or fg_mask is None:
        ref = fg_mask if fg_mask is not None else opacity
        if torch.is_tensor(ref):
            zero = torch.zeros_like(ref)
            return zero, zero
        return None, None

    opacity_threshold = float(min(max(opacity_threshold, 0.0), 1.0))
    if gap_opacity_threshold is None:
        gap_opacity_threshold = opacity_threshold
    gap_opacity_threshold = float(min(max(gap_opacity_threshold, 0.0), 1.0))
    inner_band_kernel_size = _normalize_kernel_size(inner_band_kernel_size)
    inner_region_dilate = _normalize_kernel_size(inner_region_dilate)
    closing_kernel_size = _normalize_kernel_size(closing_kernel_size)
    hole_area_min = max(int(hole_area_min), 1)
    hole_area_max = max(int(hole_area_max), hole_area_min)
    gap_area_min = max(int(gap_area_min), 1)
    gap_area_max = max(int(gap_area_max), gap_area_min)
    hole_power = float(max(hole_power, 1.0e-6))
    gap_power = float(max(gap_power, 1.0e-6))

    fg_binary = (fg_mask.to(device=opacity.device, dtype=opacity.dtype) > 0.5).float()
    if region_mask is None:
        region_binary = torch.ones_like(fg_binary)
    else:
        region_binary = (region_mask.to(device=opacity.device, dtype=opacity.dtype) > 0.0).float()
    region_binary = region_binary * fg_binary
    if region_binary.sum().item() <= 0:
        zero = torch.zeros_like(fg_binary)
        return zero, zero

    opacity_detached = opacity.detach().to(dtype=fg_binary.dtype)
    pred_binary = (opacity_detached >= opacity_threshold).float() * region_binary
    pred_binary_gap = (opacity_detached >= gap_opacity_threshold).float() * region_binary
    missing_binary = (region_binary - pred_binary).clamp(0.0, 1.0)
    if missing_binary.sum().item() <= 0:
        zero = torch.zeros_like(fg_binary)
        return zero, zero

    inner_band = _foreground_inner_ring_mask(fg_binary, inner_band_kernel_size) * region_binary
    if inner_region_dilate > 1:
        inner_region = _binary_dilate(inner_band, inner_region_dilate).clamp(0.0, 1.0) * region_binary
    else:
        inner_region = inner_band
    if inner_region.sum().item() <= 0:
        zero = torch.zeros_like(fg_binary)
        return zero, zero

    hole_binary = _select_small_binary_components(
        missing_binary * inner_region,
        area_min=hole_area_min,
        area_max=hole_area_max,
    )

    closed_binary = pred_binary_gap
    if closing_kernel_size > 1:
        closed_binary = _binary_erode(_binary_dilate(closed_binary, closing_kernel_size), closing_kernel_size).clamp(0.0, 1.0)
    gap_binary = _select_small_binary_components(
        (closed_binary - pred_binary_gap).clamp(0.0, 1.0) * inner_region,
        area_min=gap_area_min,
        area_max=gap_area_max,
    )

    if hole_binary is None or gap_binary is None:
        zero = torch.zeros_like(fg_binary)
        return zero, zero

    missing_score = (1.0 - opacity_detached).clamp(0.0, 1.0)
    hole_score = missing_score if hole_power == 1.0 else missing_score.pow(hole_power)
    gap_score = missing_score if gap_power == 1.0 else missing_score.pow(gap_power)

    hole_mask = (hole_binary * inner_region * hole_score).clamp(0.0, 1.0)
    gap_mask = (gap_binary * inner_region * gap_score).clamp(0.0, 1.0)
    return hole_mask, gap_mask


def _foreground_region_pinhole_mask(
    opacity,
    fg_mask,
    region_mask=None,
    opacity_threshold=0.08,
    closing_kernel_size=7,
    support_kernel_size=9,
    support_threshold=0.72,
    core_erode_kernel_size=3,
    area_min=1,
    area_max=32,
    power=1.0,
):
    if opacity is None or fg_mask is None:
        ref = fg_mask if fg_mask is not None else opacity
        if torch.is_tensor(ref):
            return torch.zeros_like(ref)
        return None

    opacity_threshold = float(min(max(opacity_threshold, 0.0), 1.0))
    closing_kernel_size = _normalize_kernel_size(closing_kernel_size)
    support_kernel_size = _normalize_kernel_size(support_kernel_size)
    support_threshold = float(min(max(support_threshold, 0.0), 1.0))
    core_erode_kernel_size = _normalize_kernel_size(core_erode_kernel_size)
    area_min = max(int(area_min), 1)
    area_max = max(int(area_max), area_min)
    power = float(max(power, 1.0e-6))

    fg_binary = (fg_mask.to(device=opacity.device, dtype=opacity.dtype) > 0.5).float()
    if region_mask is None:
        region_binary = fg_binary
    else:
        region_binary = (region_mask.to(device=opacity.device, dtype=opacity.dtype) > 0.0).float() * fg_binary
    if region_binary.sum().item() <= 0:
        return torch.zeros_like(fg_binary)

    core_region = region_binary
    if core_erode_kernel_size > 1:
        eroded_region = _binary_erode(region_binary, core_erode_kernel_size).clamp(0.0, 1.0) * region_binary
        if eroded_region.sum().item() > 0:
            core_region = eroded_region

    opacity_detached = opacity.detach().to(dtype=fg_binary.dtype)
    pred_binary = (opacity_detached >= opacity_threshold).float() * region_binary
    if pred_binary.sum().item() <= 0:
        return torch.zeros_like(fg_binary)

    closed_binary = pred_binary
    if closing_kernel_size > 1:
        closed_binary = _binary_erode(_binary_dilate(closed_binary, closing_kernel_size), closing_kernel_size).clamp(0.0, 1.0)
    pinhole_binary = (closed_binary - pred_binary).clamp(0.0, 1.0) * core_region
    pinhole_binary = _select_small_binary_components(
        pinhole_binary,
        area_min=area_min,
        area_max=area_max,
    )
    if pinhole_binary is None or pinhole_binary.sum().item() <= 0:
        return torch.zeros_like(fg_binary)

    if support_kernel_size > 1:
        pred_support = F.avg_pool2d(pred_binary.unsqueeze(0), support_kernel_size, stride=1, padding=support_kernel_size // 2).squeeze(0)
        region_support = F.avg_pool2d(region_binary.unsqueeze(0), support_kernel_size, stride=1, padding=support_kernel_size // 2).squeeze(0).clamp_min(1.0e-6)
        support_ratio = (pred_support / region_support).clamp(0.0, 1.0)
    else:
        support_ratio = pred_binary

    if support_threshold <= 0.0:
        support_score = torch.ones_like(support_ratio)
    else:
        support_score = ((support_ratio - support_threshold) / max(1.0 - support_threshold, 1.0e-6)).clamp(0.0, 1.0)

    missing_score = (1.0 - opacity_detached).clamp(0.0, 1.0)
    if power != 1.0:
        missing_score = missing_score.pow(power)
    return (pinhole_binary * core_region * support_score * missing_score).clamp(0.0, 1.0)


def _foreground_small_disagreement_component_masks(
    opacity,
    target_mask,
    region_mask=None,
    fp_region_mask=None,
    fn_region_mask=None,
    pred_threshold=0.08,
    target_threshold=0.5,
    fp_area_min=1,
    fp_area_max=48,
    fn_area_min=1,
    fn_area_max=48,
    fp_power=1.0,
    fn_power=1.0,
):
    if opacity is None or target_mask is None:
        ref = target_mask if target_mask is not None else opacity
        if torch.is_tensor(ref):
            zero = torch.zeros_like(ref)
            return zero, zero
        return None, None

    pred_threshold = float(min(max(pred_threshold, 0.0), 1.0))
    target_threshold = float(min(max(target_threshold, 0.0), 1.0))
    fp_area_min = max(int(fp_area_min), 1)
    fp_area_max = max(int(fp_area_max), fp_area_min)
    fn_area_min = max(int(fn_area_min), 1)
    fn_area_max = max(int(fn_area_max), fn_area_min)
    fp_power = float(max(fp_power, 1.0e-6))
    fn_power = float(max(fn_power, 1.0e-6))

    target_binary = (target_mask.to(device=opacity.device, dtype=opacity.dtype) >= target_threshold).float()
    if region_mask is None:
        region_binary = torch.ones_like(target_binary)
    else:
        region_binary = (region_mask.to(device=opacity.device, dtype=opacity.dtype) > 0.0).float()
    if region_binary.sum().item() <= 0:
        zero = torch.zeros_like(target_binary)
        return zero, zero

    fp_region = region_binary
    if fp_region_mask is not None:
        fp_region = fp_region * (fp_region_mask.to(device=opacity.device, dtype=opacity.dtype) > 0.0).float()
    fn_region = region_binary
    if fn_region_mask is not None:
        fn_region = fn_region * (fn_region_mask.to(device=opacity.device, dtype=opacity.dtype) > 0.0).float()

    opacity_detached = opacity.detach().to(dtype=target_binary.dtype)
    pred_binary = (opacity_detached >= pred_threshold).float()
    fp_binary = (pred_binary - target_binary).clamp(0.0, 1.0) * fp_region
    fn_binary = (target_binary - pred_binary).clamp(0.0, 1.0) * fn_region

    fp_binary = _select_small_binary_components(
        fp_binary,
        area_min=fp_area_min,
        area_max=fp_area_max,
    )
    fn_binary = _select_small_binary_components(
        fn_binary,
        area_min=fn_area_min,
        area_max=fn_area_max,
    )
    if fp_binary is None or fn_binary is None:
        zero = torch.zeros_like(target_binary)
        return zero, zero

    fp_score = ((opacity_detached - pred_threshold) / max(1.0 - pred_threshold, 1.0e-6)).clamp(0.0, 1.0)
    fn_score = (target_binary - opacity_detached).clamp(0.0, 1.0)
    if fp_power != 1.0:
        fp_score = fp_score.pow(fp_power)
    if fn_power != 1.0:
        fn_score = fn_score.pow(fn_power)

    fp_mask = (fp_binary * fp_region * fp_score).clamp(0.0, 1.0)
    fn_mask = (fn_binary * fn_region * fn_score).clamp(0.0, 1.0)
    return fp_mask, fn_mask


def _small_luma_difference_component_masks(
    image,
    gt_image,
    region_mask=None,
    darker_region_mask=None,
    brighter_region_mask=None,
    darker_threshold=0.08,
    brighter_threshold=0.08,
    darker_area_min=1,
    darker_area_max=32,
    brighter_area_min=1,
    brighter_area_max=32,
    darker_power=1.0,
    brighter_power=1.0,
):
    if image is None or gt_image is None or not torch.is_tensor(image) or not torch.is_tensor(gt_image):
        ref = image if torch.is_tensor(image) else gt_image
        if torch.is_tensor(ref):
            zero = torch.zeros_like(ref[:1])
            return zero, zero
        return None, None

    darker_threshold = float(min(max(darker_threshold, 0.0), 1.0))
    brighter_threshold = float(min(max(brighter_threshold, 0.0), 1.0))
    darker_area_min = max(int(darker_area_min), 1)
    darker_area_max = max(int(darker_area_max), darker_area_min)
    brighter_area_min = max(int(brighter_area_min), 1)
    brighter_area_max = max(int(brighter_area_max), brighter_area_min)
    darker_power = float(max(darker_power, 1.0e-6))
    brighter_power = float(max(brighter_power, 1.0e-6))

    if region_mask is None:
        region_binary = torch.ones_like(image[:1])
    else:
        region_binary = (region_mask.to(device=image.device, dtype=image.dtype) > 0.0).float()
    if region_binary.sum().item() <= 0:
        zero = torch.zeros_like(image[:1])
        return zero, zero

    darker_region = region_binary
    if darker_region_mask is not None:
        darker_region = darker_region * (darker_region_mask.to(device=image.device, dtype=image.dtype) > 0.0).float()
    brighter_region = region_binary
    if brighter_region_mask is not None:
        brighter_region = brighter_region * (brighter_region_mask.to(device=image.device, dtype=image.dtype) > 0.0).float()

    luma_weights = image.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    pred_luma = (image.detach() * luma_weights).sum(dim=0, keepdim=True)
    gt_luma = (gt_image.detach() * luma_weights).sum(dim=0, keepdim=True)
    dark_diff = (gt_luma - pred_luma).clamp(0.0, 1.0)
    bright_diff = (pred_luma - gt_luma).clamp(0.0, 1.0)

    darker_binary = (dark_diff >= darker_threshold).float() * darker_region
    brighter_binary = (bright_diff >= brighter_threshold).float() * brighter_region
    darker_binary = _select_small_binary_components(
        darker_binary,
        area_min=darker_area_min,
        area_max=darker_area_max,
    )
    brighter_binary = _select_small_binary_components(
        brighter_binary,
        area_min=brighter_area_min,
        area_max=brighter_area_max,
    )
    if darker_binary is None or brighter_binary is None:
        zero = torch.zeros_like(image[:1])
        return zero, zero

    dark_score = ((dark_diff - darker_threshold) / max(1.0 - darker_threshold, 1.0e-6)).clamp(0.0, 1.0)
    bright_score = ((bright_diff - brighter_threshold) / max(1.0 - brighter_threshold, 1.0e-6)).clamp(0.0, 1.0)
    if darker_power != 1.0:
        dark_score = dark_score.pow(darker_power)
    if brighter_power != 1.0:
        bright_score = bright_score.pow(brighter_power)

    darker_mask = (darker_binary * darker_region * dark_score).clamp(0.0, 1.0)
    brighter_mask = (brighter_binary * brighter_region * bright_score).clamp(0.0, 1.0)
    return darker_mask, brighter_mask


def _foreground_inner_ring_mask(fg_mask, kernel_size):
    kernel_size = _normalize_kernel_size(kernel_size)
    fg_mask = (fg_mask > 0.5).float()
    if kernel_size <= 1:
        return torch.zeros_like(fg_mask)
    eroded = _binary_erode(fg_mask, kernel_size)
    return (fg_mask - eroded).clamp(0.0, 1.0)


def _foreground_bbox_from_mask(fg_mask, padding=0):
    valid = fg_mask[0] > 0.5
    coords = torch.nonzero(valid, as_tuple=False)
    h, w = fg_mask.shape[-2:]
    if coords.numel() == 0:
        return 0, h, 0, w

    padding = max(int(padding), 0)
    y1 = max(int(coords[:, 0].min().item()) - padding, 0)
    y2 = min(int(coords[:, 0].max().item()) + 1 + padding, h)
    x1 = max(int(coords[:, 1].min().item()) - padding, 0)
    x2 = min(int(coords[:, 1].max().item()) + 1 + padding, w)
    return y1, y2, x1, x2


def _perceptual_stable_region_mask(
    mask,
    fg_mask,
    config_opt,
    region_name='global',
    image=None,
    gt_image=None,
):
    if mask is None:
        return None
    stable = mask.float().clamp(0.0, 1.0)
    if config_opt is None:
        return stable

    prefix = f'{region_name}_' if region_name and region_name != 'global' else ''
    exclude_width = int(
        config_opt.get(
            f'{prefix}perceptual_exclude_boundary_width',
            config_opt.get('perceptual_exclude_boundary_width', 0),
        )
    )
    if exclude_width > 1:
        boundary = _foreground_boundary_mask(fg_mask, exclude_width)
        stable = (stable * (1.0 - boundary)).clamp(0.0, 1.0)

    adaptive_enable = bool(
        config_opt.get(
            f'{prefix}perceptual_adaptive_boundary_exclude_enable',
            config_opt.get('perceptual_adaptive_boundary_exclude_enable', False),
        )
    )
    adaptive_width = int(
        config_opt.get(
            f'{prefix}perceptual_adaptive_boundary_width',
            config_opt.get('perceptual_adaptive_boundary_width', exclude_width),
        )
    )
    if adaptive_enable and adaptive_width > 1 and image is not None and gt_image is not None:
        adaptive_band = _foreground_boundary_mask(fg_mask, adaptive_width).to(
            device=image.device,
            dtype=image.dtype,
        )
        adaptive_band = (adaptive_band * mask.to(device=image.device, dtype=image.dtype)).clamp(0.0, 1.0)
        if adaptive_band.sum().item() > 0:
            gt_grad = _masked_normalize_map(_luma_gradient_magnitude(gt_image.detach()), adaptive_band)
            err = _masked_normalize_map((image.detach() - gt_image.detach()).abs().mean(dim=0, keepdim=True), adaptive_band)
            edge_protect = float(
                config_opt.get(
                    f'{prefix}perceptual_adaptive_edge_protect',
                    config_opt.get('perceptual_adaptive_edge_protect', 0.35),
                )
            )
            err_threshold = float(
                config_opt.get(
                    f'{prefix}perceptual_adaptive_error_threshold',
                    config_opt.get('perceptual_adaptive_error_threshold', 0.42),
                )
            )
            err_width = float(
                config_opt.get(
                    f'{prefix}perceptual_adaptive_error_width',
                    config_opt.get('perceptual_adaptive_error_width', 0.18),
                )
            )
            reliability_power = float(
                config_opt.get(
                    f'{prefix}perceptual_adaptive_reliability_power',
                    config_opt.get('perceptual_adaptive_reliability_power', 1.0),
                )
            )
            min_weight = float(
                config_opt.get(
                    f'{prefix}perceptual_adaptive_min_weight',
                    config_opt.get('perceptual_adaptive_min_weight', 0.0),
                )
            )
            err_width = max(err_width, 1.0e-6)
            unreliable = torch.sigmoid((err - err_threshold) / err_width)
            protected_unreliable = (unreliable * (1.0 - edge_protect * gt_grad)).clamp(0.0, 1.0)
            reliability = (1.0 - protected_unreliable).clamp(0.0, 1.0)
            if reliability_power > 0.0 and abs(reliability_power - 1.0) > 1.0e-6:
                reliability = reliability.pow(reliability_power)
            min_weight = max(0.0, min(1.0, min_weight))
            if min_weight > 0.0:
                reliability = reliability.clamp(min_weight, 1.0)
            stable = (
                stable * (1.0 - adaptive_band)
                + stable * adaptive_band * reliability.to(device=stable.device, dtype=stable.dtype)
            ).clamp(0.0, 1.0)

    erode_kernel = int(
        config_opt.get(
            f'{prefix}perceptual_stable_erode_kernel_size',
            config_opt.get('perceptual_stable_erode_kernel_size', 0),
        )
    )
    if erode_kernel > 1 and stable.sum().item() > 0:
        eroded = _binary_erode(stable, erode_kernel).clamp(0.0, 1.0) * fg_mask
        if eroded.sum().item() > 0:
            stable = eroded

    min_pixels = int(
        config_opt.get(
            f'{prefix}perceptual_stable_min_pixels',
            config_opt.get('perceptual_stable_min_pixels', 32),
        )
    )
    if stable.sum().item() < max(min_pixels, 1):
        fallback = bool(
            config_opt.get(
                f'{prefix}perceptual_stable_fallback_to_source',
                config_opt.get('perceptual_stable_fallback_to_source', True),
            )
        )
        if fallback:
            stable = mask.float().clamp(0.0, 1.0)
    return stable.clamp(0.0, 1.0)


def _ensure_lpips_min_size(pred_image, gt_image, min_size=32):
    min_size = max(int(min_size), 1)
    if pred_image.shape[-2] <= 0 or pred_image.shape[-1] <= 0:
        return None, None

    target_h = max(int(pred_image.shape[-2]), min_size)
    target_w = max(int(pred_image.shape[-1]), min_size)
    if target_h == int(pred_image.shape[-2]) and target_w == int(pred_image.shape[-1]):
        return pred_image, gt_image

    def _resize(image):
        added_batch = False
        if image.dim() == 3:
            image = image.unsqueeze(0)
            added_batch = True
        image = F.interpolate(image, size=(target_h, target_w), mode='bilinear', align_corners=False)
        if added_batch:
            image = image.squeeze(0)
        return image

    return _resize(pred_image), _resize(gt_image)


def _region_opt_value(config_opt, region_name, key, default):
    if config_opt is None:
        return default
    region_prefix = f'{region_name}_' if region_name else ''
    return config_opt.get(f'{region_prefix}{key}', config_opt.get(key, default))


def _parser_region_mask(data, fg_mask, label_ids):
    parser_mask = getattr(data, 'parsing_parser_mask', None)
    if parser_mask is None or not torch.is_tensor(parser_mask):
        return None
    parser_mask = parser_mask[:1].to(device=fg_mask.device, dtype=fg_mask.dtype)
    region_mask = torch.zeros_like(fg_mask)
    for label_id in label_ids:
        region_mask = torch.maximum(region_mask, (parser_mask == float(label_id)).float())
    region_mask = region_mask * fg_mask
    if region_mask.sum().item() < 8:
        return None
    return region_mask


def _rasterize_probability_channels(data, pc, pipe, background, probs):
    if probs is None or not torch.is_tensor(probs) or probs.numel() == 0:
        return None, None
    rendered_chunks = []
    opacity = None
    channel_count = probs.shape[-1]
    for start in range(0, channel_count, 3):
        chunk = probs[:, start:start + 3]
        if chunk.shape[-1] < 3:
            pad = torch.zeros(
                chunk.shape[0],
                3 - chunk.shape[-1],
                dtype=chunk.dtype,
                device=chunk.device,
            )
            chunk = torch.cat([chunk, pad], dim=-1)
        pkg = rasterize_gaussians(
            data,
            pc,
            pipe,
            background,
            colors_precomp=chunk,
            return_opacity=opacity is None,
        )
        rendered_chunks.append(pkg['render'][:min(3, channel_count - start)].clamp(0.0, 1.0))
        if opacity is None:
            opacity = pkg.get('opacity_render', None)
            if opacity is not None:
                opacity = opacity[:1].clamp(0.0, 1.0)
    return torch.cat(rendered_chunks, dim=0), opacity


def _semantic_class_weight_tensor(value, class_count, device, dtype):
    if value is None:
        return None
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.strip().strip('[]').split(',') if item.strip()]
        try:
            value = [float(item) for item in raw_items]
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        weights = [float(value)] * int(class_count)
    elif isinstance(value, (list, tuple)):
        weights = [float(item) for item in value[:class_count]]
    else:
        return None
    if len(weights) < class_count:
        weights.extend([1.0] * (class_count - len(weights)))
    return torch.as_tensor(weights[:class_count], device=device, dtype=dtype)


def _semantic_index_list(value, default, class_count):
    if value is None:
        value = default
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.strip().strip('[]').split(',') if item.strip()]
        try:
            value = [int(item) for item in raw_items]
        except ValueError:
            value = default
    if isinstance(value, (int, float)):
        value = [int(value)]
    elif isinstance(value, (list, tuple)):
        value = [int(item) for item in value]
    else:
        value = list(default)
    return [idx for idx in value if 0 <= idx < int(class_count)]


def _sum_semantic_indices(channels, indices):
    if not indices:
        return torch.zeros_like(channels[:1])
    return channels[indices].sum(dim=0, keepdim=True).clamp(0.0, 1.0)


def _normalize_probability_channels(channels, opacity=None, eps=1.0e-6):
    if opacity is not None and torch.is_tensor(opacity):
        denom = opacity[:1].to(device=channels.device, dtype=channels.dtype)
    else:
        denom = channels.sum(dim=0, keepdim=True)
    return (channels / denom.clamp_min(eps)).clamp(0.0, 1.0)


def _masked_dice_loss(pred, target, mask, channel_weight=None):
    pred = pred.clamp(0.0, 1.0)
    target = target.to(device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)
    if mask.shape[0] == 1 and pred.shape[0] != 1:
        mask = mask.expand(pred.shape[0], -1, -1)
    if target.shape[0] == 1 and pred.shape[0] != 1:
        target = target.expand_as(pred)
    intersection = (pred * target * mask).sum(dim=(1, 2))
    denom = ((pred + target) * mask).sum(dim=(1, 2)).clamp_min(1.0)
    loss = 1.0 - (2.0 * intersection + 1.0) / (denom + 1.0)
    if channel_weight is not None:
        channel_weight = channel_weight.to(device=pred.device, dtype=pred.dtype).flatten()[: loss.shape[0]]
        if channel_weight.numel() < loss.shape[0]:
            channel_weight = torch.cat([channel_weight, torch.ones(loss.shape[0] - channel_weight.numel(), device=pred.device, dtype=pred.dtype)])
        return (loss * channel_weight).sum() / channel_weight.sum().clamp_min(1.0e-6)
    return loss.mean()


def _masked_multiclass_cross_entropy(pred, target, mask, class_weight=None):
    pred = pred.clamp_min(1.0e-6)
    pred = pred / pred.sum(dim=0, keepdim=True).clamp_min(1.0e-6)
    target = target.to(device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    class_count = min(pred.shape[0], target.shape[0])
    if class_count <= 0:
        return pred.new_tensor(0.0)
    pred = pred[:class_count]
    target = target[:class_count]

    valid = mask[:1] * (target.sum(dim=0, keepdim=True) > 0.5).to(dtype=pred.dtype)
    if valid.sum().item() <= 0:
        return pred.new_tensor(0.0)

    target_ids = torch.argmax(target, dim=0)
    per_pixel = -torch.log(pred.gather(0, target_ids.unsqueeze(0)).clamp_min(1.0e-6))
    pixel_weight = torch.ones_like(per_pixel)
    if class_weight is not None:
        class_weight = class_weight.to(device=pred.device, dtype=pred.dtype).flatten()[:class_count]
        if class_weight.numel() < class_count:
            class_weight = torch.cat([
                class_weight,
                torch.ones(class_count - class_weight.numel(), device=pred.device, dtype=pred.dtype),
            ])
        pixel_weight = class_weight[target_ids].unsqueeze(0)
    weighted_valid = valid * pixel_weight
    return (per_pixel * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)


def _semantic_parser_valid_mask(data, fg_mask, opt, opacity=None):
    valid = getattr(data, 'parsing_valid_mask', None)
    if torch.is_tensor(valid):
        valid = valid[:1].to(device=fg_mask.device, dtype=fg_mask.dtype).clamp(0.0, 1.0)
    else:
        valid = fg_mask.clone()

    uncertain = getattr(data, 'parsing_uncertain_mask', None)
    if torch.is_tensor(uncertain) and bool(opt.get('stageB_semantic_ignore_uncertain', True)):
        valid = valid * (1.0 - uncertain[:1].to(device=fg_mask.device, dtype=fg_mask.dtype).clamp(0.0, 1.0))

    boundary_width = int(opt.get('stageB_semantic_ignore_boundary_width', 7))
    if boundary_width > 1:
        valid = valid * (1.0 - _foreground_boundary_mask(fg_mask, boundary_width)).clamp(0.0, 1.0)

    if opacity is not None and bool(opt.get('stageB_semantic_use_opacity_support', True)):
        opacity_threshold = float(opt.get('stageB_semantic_opacity_threshold', 0.04))
        valid = valid * (opacity[:1].to(device=fg_mask.device, dtype=fg_mask.dtype) > opacity_threshold).float()

    return (valid * fg_mask).clamp(0.0, 1.0)


def _compact_semantic_class_names(data, class_count):
    names = getattr(data, 'parsing_compact_class_names', None)
    if isinstance(names, str):
        names = [part.strip() for part in names.split(',') if part.strip()]
    elif isinstance(names, (list, tuple)):
        names = list(names)
    else:
        names = list(COMPACT_SEMANTIC_CLASS_NAMES)
    if len(names) < class_count:
        names.extend([f'class{idx}' for idx in range(len(names), class_count)])
    return names[:class_count]


def _compact_semantic_stat_name(name, index):
    raw = str(name) if name is not None else f'class{index}'
    safe = ''.join(ch.lower() if ch.isalnum() else '_' for ch in raw).strip('_')
    return safe or f'class{index}'


def _update_stageB_semantic_compact_debug_stats(stats, data, pred_prob, target, valid, opt):
    if pred_prob is None or target is None:
        return
    class_count = min(pred_prob.shape[0], target.shape[0])
    if class_count <= 0:
        return
    threshold = float(opt.get('stageB_semantic_debug_pred_threshold', 0.5))
    names = _compact_semantic_class_names(data, class_count)
    with torch.no_grad():
        pred = pred_prob[:class_count].detach()
        tgt = target[:class_count].detach()
        valid_mask = valid[:1].detach().to(device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)
        valid_pixels = valid_mask.sum().clamp_min(1.0)
        pred_hard = (pred >= threshold).to(pred.dtype) * valid_mask
        tgt_valid = tgt * valid_mask
        intersection = (pred_hard * tgt_valid).sum(dim=(1, 2))
        pred_pixels = pred_hard.sum(dim=(1, 2))
        target_pixels = tgt_valid.sum(dim=(1, 2))
        union = (pred_pixels + target_pixels - intersection).clamp_min(1.0)
        precision = intersection / pred_pixels.clamp_min(1.0)
        recall = intersection / target_pixels.clamp_min(1.0)
        iou = intersection / union
        mean_prob = (pred * valid_mask).sum(dim=(1, 2)) / valid_pixels
        target_ratio = target_pixels / valid_pixels
        stats['compact_debug_threshold'] = threshold
        for idx, name in enumerate(names):
            prefix = f"compact_{_compact_semantic_stat_name(name, idx)}"
            stats[f'{prefix}_iou'] = float(iou[idx].item())
            stats[f'{prefix}_precision'] = float(precision[idx].item())
            stats[f'{prefix}_recall'] = float(recall[idx].item())
            stats[f'{prefix}_pred_pixels'] = float(pred_pixels[idx].item())
            stats[f'{prefix}_target_pixels'] = float(target_pixels[idx].item())
            stats[f'{prefix}_mean_prob'] = float(mean_prob[idx].item())
            stats[f'{prefix}_target_ratio'] = float(target_ratio[idx].item())


def _compute_stageB_semantic_parser_loss(data, render_pkg, pipe, background, fg_mask, opt):
    zero = fg_mask.new_tensor(0.0)
    stats = {
        'enabled': 0.0,
        'valid_pixels': 0.0,
        'body_cloth': 0.0,
        'compact': 0.0,
        'compact_ce': 0.0,
        'parent': 0.0,
        'exclusive': 0.0,
        'smooth': 0.0,
    }
    if not bool(opt.get('stageB_semantic_loss_enable', False)):
        return zero, stats

    pc = render_pkg["deformed_gaussian"]
    region_probs = getattr(pc, 'binding_region_probs_asset_raw', None)
    if region_probs is None:
        region_probs = getattr(pc, 'binding_region_probs_raw', None)
    if region_probs is None:
        region_probs = getattr(pc, 'binding_region_probs', None)
    if region_probs is None:
        return zero, stats

    compact_probs = getattr(pc, 'binding_compact_semantic_probs_asset_raw', None)
    if compact_probs is None:
        compact_probs = getattr(pc, 'binding_compact_semantic_probs_raw', None)
    if compact_probs is None:
        compact_probs = getattr(pc, 'binding_compact_semantic_probs', None)

    opacity = render_pkg.get("opacity_render", None)
    valid_mask = _semantic_parser_valid_mask(data, fg_mask, opt, opacity=opacity)
    min_pixels = float(opt.get('stageB_semantic_min_valid_pixels', 64))
    valid_pixels = float(valid_mask.sum().detach().item())
    stats['enabled'] = 1.0
    stats['valid_pixels'] = valid_pixels
    if valid_pixels < min_pixels:
        return zero, stats

    semantic_background = torch.zeros(3, dtype=region_probs.dtype, device=region_probs.device)
    region_2d, semantic_opacity = _rasterize_probability_channels(
        data,
        pc,
        pipe,
        semantic_background,
        region_probs,
    )
    if region_2d is None:
        return zero, stats
    region_pred_prob = _normalize_probability_channels(
        region_2d,
        opacity=semantic_opacity,
        eps=float(opt.get('stageB_semantic_prob_normalize_eps', 1.0e-6)),
    )
    if semantic_opacity is not None and opacity is None:
        valid_mask = _semantic_parser_valid_mask(data, fg_mask, opt, opacity=semantic_opacity)
        valid_pixels = float(valid_mask.sum().detach().item())
        stats['valid_pixels'] = valid_pixels
        if valid_pixels < min_pixels:
            return zero, stats

    total = zero
    compact_2d = None
    compact_target = None
    compact_pred_prob = None
    compact_valid = valid_mask.to(device=region_2d.device, dtype=region_2d.dtype)
    compact_class_count = 0
    compact_parent_body_indices = []
    compact_parent_cloth_indices = []
    parent_target_from_compact = None

    compact_masks = getattr(data, 'parsing_compact_masks', None)
    if torch.is_tensor(compact_probs) and torch.is_tensor(compact_masks):
        compact_background = torch.zeros(3, dtype=compact_probs.dtype, device=compact_probs.device)
        compact_2d, compact_opacity = _rasterize_probability_channels(
            data,
            pc,
            pipe,
            compact_background,
            compact_probs,
        )
        if compact_2d is not None:
            compact_target = compact_masks.to(device=compact_2d.device, dtype=compact_2d.dtype).clamp(0.0, 1.0)
            compact_class_count = min(compact_2d.shape[0], compact_target.shape[0])
            if compact_class_count > 0:
                compact_2d = compact_2d[:compact_class_count]
                compact_target = compact_target[:compact_class_count]
                compact_valid = valid_mask.to(device=compact_2d.device, dtype=compact_2d.dtype)
                compact_pred_prob = _normalize_probability_channels(
                    compact_2d,
                    opacity=compact_opacity,
                    eps=float(opt.get('stageB_semantic_prob_normalize_eps', 1.0e-6)),
                )
                compact_parent_body_indices = _semantic_index_list(
                    opt.get('stageB_semantic_parent_body_indices', None),
                    default=(0, 1, 2),
                    class_count=compact_class_count,
                )
                compact_parent_cloth_indices = _semantic_index_list(
                    opt.get('stageB_semantic_parent_cloth_indices', None),
                    default=(3, 4, 5),
                    class_count=compact_class_count,
                )
                body_from_target = _sum_semantic_indices(compact_target, compact_parent_body_indices)
                cloth_from_target = _sum_semantic_indices(compact_target, compact_parent_cloth_indices)
                if body_from_target.sum().item() > 0 or cloth_from_target.sum().item() > 0:
                    parent_target_from_compact = torch.cat([body_from_target, cloth_from_target], dim=0)

    body_mask = getattr(data, 'parsing_body_mask', None)
    cloth_mask = getattr(data, 'parsing_cloth_mask', None)
    if torch.is_tensor(body_mask) and torch.is_tensor(cloth_mask):
        body_target = body_mask[:1].to(device=region_2d.device, dtype=region_2d.dtype).clamp(0.0, 1.0)
        cloth_target = cloth_mask[:1].to(device=region_2d.device, dtype=region_2d.dtype).clamp(0.0, 1.0)
        if parent_target_from_compact is not None and bool(opt.get('stageB_semantic_parent_target_use_compact_groups', True)):
            body_target = torch.maximum(body_target, parent_target_from_compact[:1].to(device=region_2d.device, dtype=region_2d.dtype))
            cloth_target = torch.maximum(cloth_target, parent_target_from_compact[1:2].to(device=region_2d.device, dtype=region_2d.dtype))
        parent_pred = torch.stack([region_pred_prob[0], region_pred_prob[2]], dim=0)
        parent_target = torch.cat([body_target, cloth_target], dim=0)
        parent_valid = valid_mask.to(device=region_2d.device, dtype=region_2d.dtype)
        bce_weight = float(opt.get('stageB_semantic_body_cloth_bce_weight', 1.0))
        dice_weight = float(opt.get('stageB_semantic_body_cloth_dice_weight', 0.75))
        body_cloth_loss = (
            bce_weight * _masked_binary_cross_entropy(parent_pred, parent_target, parent_valid)
            + dice_weight * _masked_dice_loss(parent_pred, parent_target, parent_valid)
        )
        total = total + float(opt.get('stageB_semantic_body_cloth_weight', 1.0)) * body_cloth_loss
        stats['body_cloth'] = float(body_cloth_loss.detach().item())

    compact_loss = zero
    if compact_pred_prob is not None and compact_target is not None and compact_class_count > 0:
        compact_class_weight = _semantic_class_weight_tensor(
            opt.get('stageB_semantic_compact_class_weights', None),
            compact_class_count,
            compact_pred_prob.device,
            compact_pred_prob.dtype,
        )
        compact_positive_weight = _semantic_class_weight_tensor(
            opt.get('stageB_semantic_compact_positive_weights', None),
            compact_class_count,
            compact_pred_prob.device,
            compact_pred_prob.dtype,
        )
        compact_bce_loss = _masked_binary_cross_entropy(
            compact_pred_prob,
            compact_target,
            compact_valid,
            channel_weight=compact_class_weight,
            positive_weight=compact_positive_weight,
        )
        compact_dice_loss = _masked_dice_loss(
            compact_pred_prob,
            compact_target,
            compact_valid,
            channel_weight=compact_class_weight,
        )
        compact_ce_loss = _masked_multiclass_cross_entropy(
            compact_pred_prob,
            compact_target,
            compact_valid,
            class_weight=compact_class_weight,
        )
        compact_loss = (
            float(opt.get('stageB_semantic_compact_bce_weight', 1.0)) * compact_bce_loss
            + float(opt.get('stageB_semantic_compact_dice_weight', 0.75)) * compact_dice_loss
            + float(opt.get('stageB_semantic_compact_ce_weight', 0.0)) * compact_ce_loss
        )
        total = total + float(opt.get('stageB_semantic_compact_weight', 1.0)) * compact_loss
        stats['compact'] = float(compact_loss.detach().item())
        stats['compact_ce'] = float(compact_ce_loss.detach().item())
        _update_stageB_semantic_compact_debug_stats(
            stats,
            data,
            compact_pred_prob,
            compact_target,
            compact_valid,
            opt,
        )

        if bool(opt.get('stageB_semantic_parent_consistency_enable', True)) and compact_class_count > 0:
            compact_body = _sum_semantic_indices(compact_pred_prob, compact_parent_body_indices)
            compact_cloth = _sum_semantic_indices(compact_pred_prob, compact_parent_cloth_indices)
            parent_from_compact = torch.cat([compact_body, compact_cloth], dim=0)
            parent_from_region = torch.stack([region_pred_prob[0], region_pred_prob[2]], dim=0)
            parent_loss = ((parent_from_region - parent_from_compact).abs() * compact_valid).sum() / compact_valid.sum().clamp_min(1.0)
            total = total + float(opt.get('stageB_semantic_parent_consistency_weight', 0.35)) * parent_loss
            stats['parent'] = float(parent_loss.detach().item())

    exclusive_weight = float(opt.get('stageB_semantic_exclusive_weight', 0.0))
    if exclusive_weight > 0.0:
        exclusive_loss = (
            region_pred_prob[0:1]
            * region_pred_prob[2:3]
            * valid_mask.to(device=region_pred_prob.device, dtype=region_pred_prob.dtype)
        ).sum() / valid_mask.sum().clamp_min(1.0)
        total = total + exclusive_weight * exclusive_loss
        stats['exclusive'] = float(exclusive_loss.detach().item())

    smooth_weight = float(opt.get('stageB_semantic_adapter_smooth_weight', 0.0))
    if smooth_weight > 0.0:
        positions = getattr(pc, 'canonical_xyz', None)
        if not torch.is_tensor(positions) or positions.shape[0] != region_probs.shape[0]:
            positions = pc.get_xyz
        smooth_loss = _boundary_residual_smoothness_loss(
            region_probs,
            positions,
            torch.ones((region_probs.shape[0],), dtype=torch.bool, device=region_probs.device),
            k=int(opt.get('stageB_semantic_adapter_smooth_k', 8)),
            distance_quantile=float(opt.get('stageB_semantic_adapter_smooth_distance_quantile', 0.5)),
        )
        total = total + smooth_weight * smooth_loss
        stats['smooth'] = float(smooth_loss.detach().item())

    return total, stats


def _make_pixel_grid(mask):
    h, w = mask.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=mask.device, dtype=mask.dtype),
        torch.arange(w, device=mask.device, dtype=mask.dtype),
        indexing='ij',
    )
    return yy.unsqueeze(0), xx.unsqueeze(0)


def _ellipse_region_mask(fg_mask, center_x, center_y, radius_x, radius_y):
    yy, xx = _make_pixel_grid(fg_mask)
    center_x = torch.as_tensor(center_x, device=fg_mask.device, dtype=fg_mask.dtype)
    center_y = torch.as_tensor(center_y, device=fg_mask.device, dtype=fg_mask.dtype)
    radius_x = torch.as_tensor(radius_x, device=fg_mask.device, dtype=fg_mask.dtype).clamp_min(1.0)
    radius_y = torch.as_tensor(radius_y, device=fg_mask.device, dtype=fg_mask.dtype).clamp_min(1.0)
    score = ((xx - center_x) / radius_x).pow(2) + ((yy - center_y) / radius_y).pow(2)
    return (score <= 1.0).float() * fg_mask


def _capsule_region_mask(fg_mask, p0, p1, radius):
    yy, xx = _make_pixel_grid(fg_mask)
    x0 = torch.as_tensor(p0[0], device=fg_mask.device, dtype=fg_mask.dtype)
    y0 = torch.as_tensor(p0[1], device=fg_mask.device, dtype=fg_mask.dtype)
    x1 = torch.as_tensor(p1[0], device=fg_mask.device, dtype=fg_mask.dtype)
    y1 = torch.as_tensor(p1[1], device=fg_mask.device, dtype=fg_mask.dtype)
    radius = torch.as_tensor(radius, device=fg_mask.device, dtype=fg_mask.dtype).clamp_min(1.0)

    vx = x1 - x0
    vy = y1 - y0
    denom = (vx * vx + vy * vy).clamp_min(1e-6)
    t = (((xx - x0) * vx + (yy - y0) * vy) / denom).clamp(0.0, 1.0)
    proj_x = x0 + t * vx
    proj_y = y0 + t * vy
    dist2 = (xx - proj_x).pow(2) + (yy - proj_y).pow(2)
    return (dist2 <= radius * radius).float() * fg_mask


def _project_world_points_to_image(data, points_world):
    if points_world is None or not torch.is_tensor(points_world) or points_world.numel() == 0:
        return None, None
    points_world = points_world.to(device=data.full_proj_transform.device, dtype=data.full_proj_transform.dtype)
    projected = geom_transform_points(points_world, data.full_proj_transform)
    if projected.dim() != 2 or projected.shape[-1] < 2:
        return None, None

    pix = torch.empty(projected.shape[0], 2, device=projected.device, dtype=projected.dtype)
    pix[:, 0] = (projected[:, 0] + 1.0) * 0.5 * float(data.image_width - 1)
    pix[:, 1] = (1.0 - projected[:, 1]) * 0.5 * float(data.image_height - 1)
    valid = torch.isfinite(pix).all(dim=-1)
    return pix, valid


def _heuristic_face_region_mask(fg_mask):
    valid = fg_mask[0] > 0.5
    coords = torch.nonzero(valid, as_tuple=False)
    if coords.numel() == 0:
        return torch.zeros_like(fg_mask)

    y_top = coords[:, 0].min().float()
    y_bottom = coords[:, 0].max().float()
    x_left = coords[:, 1].min().float()
    x_right = coords[:, 1].max().float()
    body_h = (y_bottom - y_top + 1.0).clamp_min(1.0)
    body_w = (x_right - x_left + 1.0).clamp_min(1.0)

    upper_coords = coords[coords[:, 0].float() <= (y_top + 0.30 * body_h)]
    if upper_coords.numel() > 0:
        center_x = upper_coords[:, 1].float().median()
    else:
        center_x = 0.5 * (x_left + x_right)
    center_y = y_top + 0.18 * body_h
    radius_x = torch.maximum(0.12 * body_w, torch.tensor(8.0, device=fg_mask.device, dtype=fg_mask.dtype))
    radius_y = torch.maximum(0.14 * body_h, torch.tensor(10.0, device=fg_mask.device, dtype=fg_mask.dtype))
    return _ellipse_region_mask(fg_mask, center_x, center_y, radius_x, radius_y)


def _joint_guided_face_region_mask(data, fg_mask):
    posed_joints = getattr(data, 'posed_joints', None)
    projected, valid = _project_world_points_to_image(data, posed_joints)
    required = [NECK_JOINT_INDEX, HEAD_JOINT_INDEX, LEFT_COLLAR_JOINT_INDEX, RIGHT_COLLAR_JOINT_INDEX, LEFT_SHOULDER_JOINT_INDEX, RIGHT_SHOULDER_JOINT_INDEX]
    if projected is None or valid is None or any(idx >= projected.shape[0] for idx in required):
        return None
    if not bool(valid[required].all().item()):
        return None

    head = projected[HEAD_JOINT_INDEX]
    neck = projected[NECK_JOINT_INDEX]
    left_shoulder = projected[LEFT_SHOULDER_JOINT_INDEX]
    right_shoulder = projected[RIGHT_SHOULDER_JOINT_INDEX]
    left_collar = projected[LEFT_COLLAR_JOINT_INDEX]
    right_collar = projected[RIGHT_COLLAR_JOINT_INDEX]

    shoulder_width = torch.norm(left_shoulder - right_shoulder).clamp_min(16.0)
    collar_width = torch.norm(left_collar - right_collar).clamp_min(12.0)
    head_span = torch.norm(head - neck).clamp_min(10.0)
    center = 0.58 * head + 0.42 * neck
    radius_x = torch.maximum(0.22 * shoulder_width, 0.58 * head_span)
    radius_x = torch.maximum(radius_x, 0.18 * collar_width)
    radius_y = torch.maximum(0.28 * shoulder_width, 0.78 * head_span)

    face_mask = _ellipse_region_mask(fg_mask, center[0], center[1], radius_x, radius_y)
    if face_mask.sum().item() < 16:
        return None
    return face_mask


def _resolve_face_region_from_data(data, fg_mask, config_opt=None):
    region_source = 'auto' if config_opt is None else str(config_opt.get('face_region_source', 'auto'))
    parser_dilate = 5 if config_opt is None else int(config_opt.get('face_region_parser_dilate', 5))

    parser_face_mask = _parser_region_mask(data, fg_mask, PARSER_FACE_LABELS)
    if parser_face_mask is not None:
        if parser_dilate > 1:
            parser_face_mask = _binary_dilate(parser_face_mask, parser_dilate).clamp(0.0, 1.0)
        parser_face_mask = parser_face_mask * fg_mask

    joint_face_mask = None
    if region_source not in ('parser_only', 'heuristic_only'):
        joint_face_mask = _joint_guided_face_region_mask(data, fg_mask)

    heuristic_face_mask = None

    def _get_heuristic_face_mask():
        nonlocal heuristic_face_mask
        if heuristic_face_mask is None:
            heuristic_face_mask = _heuristic_face_region_mask(fg_mask)
        return heuristic_face_mask

    if region_source == 'parser_only':
        if parser_face_mask is not None:
            region = parser_face_mask
            actual_source = 'parser_only'
        else:
            region = torch.zeros_like(fg_mask)
            actual_source = 'parser_missing_empty'
    elif region_source == 'joint_only':
        if joint_face_mask is not None:
            region = joint_face_mask
            actual_source = 'joint_only'
        else:
            region = _get_heuristic_face_mask()
            actual_source = 'joint_fallback_heuristic'
    elif region_source == 'heuristic_only':
        region = _get_heuristic_face_mask()
        actual_source = 'heuristic_only'
    elif region_source == 'union':
        if parser_face_mask is not None and joint_face_mask is not None:
            region = torch.maximum(parser_face_mask, joint_face_mask).clamp(0.0, 1.0) * fg_mask
            actual_source = 'union_parser_joint'
        elif parser_face_mask is not None:
            region = parser_face_mask
            actual_source = 'union_parser'
        elif joint_face_mask is not None:
            region = joint_face_mask
            actual_source = 'union_joint'
        else:
            region = _get_heuristic_face_mask()
            actual_source = 'union_heuristic'
    elif region_source == 'parser_prefer':
        if parser_face_mask is not None:
            region = parser_face_mask
            actual_source = 'parser_prefer_parser'
        elif joint_face_mask is not None:
            region = joint_face_mask
            actual_source = 'parser_prefer_joint_fallback'
        else:
            region = _get_heuristic_face_mask()
            actual_source = 'parser_prefer_heuristic_fallback'
    else:
        if parser_face_mask is not None:
            region = parser_face_mask
            actual_source = 'auto_parser'
        elif joint_face_mask is not None:
            region = joint_face_mask
            actual_source = 'auto_joint'
        else:
            region = _get_heuristic_face_mask()
            actual_source = 'auto_heuristic'

    meta = {
        'requested_source': region_source,
        'actual_source': actual_source,
        'has_parser': parser_face_mask is not None,
        'has_joint': joint_face_mask is not None,
        'region_pixels': float(region.sum().item()),
        'parser_pixels': float(parser_face_mask.sum().item()) if parser_face_mask is not None else 0.0,
        'joint_pixels': float(joint_face_mask.sum().item()) if joint_face_mask is not None else 0.0,
        'heuristic_pixels': float(heuristic_face_mask.sum().item()) if heuristic_face_mask is not None else 0.0,
    }
    return region.clamp(0.0, 1.0), meta


def _head_outer_shell_region_from_data(data, fg_mask, face_mask=None, config_opt=None):
    source = None if face_mask is None else face_mask.float().clamp(0.0, 1.0)
    fallback_used = False
    if source is None or source.sum().item() < 8:
        source = _joint_guided_face_region_mask(data, fg_mask)
        if source is None or source.sum().item() < 8:
            source = _heuristic_face_region_mask(fg_mask)
            fallback_used = True

    region = source.float().clamp(0.0, 1.0)
    if config_opt is not None:
        dilate = int(config_opt.get('silhouette_head_outer_region_dilate', 17))
    else:
        dilate = 17
    if dilate > 1:
        region = _binary_dilate(region, dilate).clamp(0.0, 1.0)

    valid = fg_mask[0] > 0.5
    coords = torch.nonzero(valid, as_tuple=False)
    if coords.numel() > 0:
        y_top = coords[:, 0].min().float()
        y_bottom = coords[:, 0].max().float()
        body_h = (y_bottom - y_top + 1.0).clamp_min(1.0)
        yy, _ = _make_pixel_grid(fg_mask)
        bottom_ratio = 0.34
        if config_opt is not None:
            bottom_ratio = float(config_opt.get('silhouette_head_outer_bottom_ratio', 0.34))
        bottom_ratio = min(max(bottom_ratio, 0.12), 0.65)
        upper_band = (yy <= (y_top + bottom_ratio * body_h)).float()
        region = region * upper_band

    if config_opt is not None and bool(config_opt.get('silhouette_head_outer_use_fg_clip', True)):
        clip = _binary_dilate(fg_mask, int(config_opt.get('silhouette_head_outer_fg_clip_dilate', 35))).clamp(0.0, 1.0)
        region = region * clip

    region = region.clamp(0.0, 1.0)
    meta = {
        'region_pixels': float(region.sum().item()),
        'fallback_used': fallback_used,
    }
    return region, meta


def _face_region_effective_min_pixels(config_opt, region_meta=None):
    base_min_pixels = 48 if config_opt is None else int(config_opt.get('face_region_min_pixels', 48))
    if config_opt is None or region_meta is None:
        return base_min_pixels
    if not bool(config_opt.get('face_region_source_aware_validity_enable', False)):
        return base_min_pixels

    actual_source = str(region_meta.get('actual_source', ''))
    parser_backed_sources = {
        'parser_only',
        'parser_prefer_parser',
        'union_parser',
        'union_parser_joint',
        'auto_parser',
    }
    joint_backed_sources = {
        'joint_only',
        'parser_prefer_joint_fallback',
        'union_joint',
        'auto_joint',
    }
    heuristic_backed_sources = {
        'heuristic_only',
        'joint_fallback_heuristic',
        'parser_prefer_heuristic_fallback',
        'union_heuristic',
        'auto_heuristic',
    }

    if actual_source in parser_backed_sources:
        min_pixels = int(config_opt.get('face_region_min_pixels_parser', max(base_min_pixels // 2, 1)))
    elif actual_source in joint_backed_sources:
        min_pixels = int(config_opt.get('face_region_min_pixels_joint', base_min_pixels))
    elif actual_source in heuristic_backed_sources:
        min_pixels = int(config_opt.get('face_region_min_pixels_heuristic', base_min_pixels))
    else:
        min_pixels = base_min_pixels

    floor = int(config_opt.get('face_region_min_pixels_floor', 1))
    return max(min_pixels, floor)


def _maybe_log_face_region_debug(config_opt, schedule_iteration, region_meta, region_valid, min_pixels):
    if config_opt is None or region_meta is None:
        return
    if not bool(config_opt.get('face_region_debug_enable', False)):
        return

    interval = int(config_opt.get('face_region_debug_interval', 0))
    warmup_iters = int(config_opt.get('face_region_debug_warmup_iters', 0))
    if schedule_iteration < warmup_iters:
        return
    if interval > 0 and schedule_iteration % interval != 0:
        return

    requested_source = str(region_meta.get('requested_source', ''))
    actual_source = str(region_meta.get('actual_source', ''))
    has_parser = int(bool(region_meta.get('has_parser', False)))
    has_joint = int(bool(region_meta.get('has_joint', False)))
    region_pixels = float(region_meta.get('region_pixels', 0.0))
    parser_pixels = float(region_meta.get('parser_pixels', 0.0))
    joint_pixels = float(region_meta.get('joint_pixels', 0.0))
    heuristic_pixels = float(region_meta.get('heuristic_pixels', 0.0))

    print(
        (
            f"[FaceROI] iter={schedule_iteration} requested={requested_source} actual={actual_source} "
            f"has_parser={has_parser} has_joint={has_joint} region_pixels={region_pixels:.1f} "
            f"parser_pixels={parser_pixels:.1f} joint_pixels={joint_pixels:.1f} "
            f"heuristic_pixels={heuristic_pixels:.1f} min_pixels={int(min_pixels)} valid={int(region_valid)}"
        ),
        flush=True,
    )


def _heuristic_shoulder_arm_region_mask(fg_mask, face_mask=None):
    valid = fg_mask[0] > 0.5
    coords = torch.nonzero(valid, as_tuple=False)
    if coords.numel() == 0:
        return torch.zeros_like(fg_mask)

    y_top = coords[:, 0].min().float()
    y_bottom = coords[:, 0].max().float()
    x_left = coords[:, 1].min().float()
    x_right = coords[:, 1].max().float()
    body_h = (y_bottom - y_top + 1.0).clamp_min(1.0)
    body_w = (x_right - x_left + 1.0).clamp_min(1.0)
    yy, xx = _make_pixel_grid(fg_mask)
    x_center = 0.5 * (x_left + x_right)
    x_rel = (xx - x_center) / (0.5 * body_w + 1e-6)
    y_rel = (yy - y_top) / (body_h + 1e-6)

    side_band = (torch.abs(x_rel) > 0.14) & (torch.abs(x_rel) < 0.62)
    upper_band = (y_rel > 0.15) & (y_rel < 0.58)
    clavicle_band = (torch.abs(x_rel) < 0.52) & (y_rel > 0.14) & (y_rel < 0.30)
    region = ((side_band & upper_band) | clavicle_band).float() * fg_mask
    if face_mask is not None:
        region = region * (1.0 - 0.75 * face_mask)
    return region.clamp(0.0, 1.0)


def _joint_guided_shoulder_arm_region_mask(data, fg_mask, face_mask=None):
    posed_joints = getattr(data, 'posed_joints', None)
    projected, valid = _project_world_points_to_image(data, posed_joints)
    required = [LEFT_COLLAR_JOINT_INDEX, RIGHT_COLLAR_JOINT_INDEX, LEFT_SHOULDER_JOINT_INDEX, RIGHT_SHOULDER_JOINT_INDEX, LEFT_ELBOW_JOINT_INDEX, RIGHT_ELBOW_JOINT_INDEX]
    if projected is None or valid is None or any(idx >= projected.shape[0] for idx in required):
        return None
    if not bool(valid[required].all().item()):
        return None

    left_collar = projected[LEFT_COLLAR_JOINT_INDEX]
    right_collar = projected[RIGHT_COLLAR_JOINT_INDEX]
    left_shoulder = projected[LEFT_SHOULDER_JOINT_INDEX]
    right_shoulder = projected[RIGHT_SHOULDER_JOINT_INDEX]
    left_elbow = projected[LEFT_ELBOW_JOINT_INDEX]
    right_elbow = projected[RIGHT_ELBOW_JOINT_INDEX]

    shoulder_width = torch.norm(left_shoulder - right_shoulder).clamp_min(20.0)
    radius = torch.clamp(0.16 * shoulder_width, min=9.0, max=42.0)
    region = torch.zeros_like(fg_mask)

    for p0, p1, scale in (
        (left_collar, left_shoulder, 0.95),
        (right_collar, right_shoulder, 0.95),
        (left_shoulder, left_elbow, 1.05),
        (right_shoulder, right_elbow, 1.05),
        (left_collar, right_collar, 0.65),
    ):
        region = torch.maximum(region, _capsule_region_mask(fg_mask, p0, p1, radius * scale))

    for center in (left_collar, right_collar, left_shoulder, right_shoulder):
        region = torch.maximum(region, _ellipse_region_mask(fg_mask, center[0], center[1], 1.15 * radius, 0.95 * radius))

    yy, _ = _make_pixel_grid(fg_mask)
    y_top = torch.min(torch.stack([left_collar[1], right_collar[1], left_shoulder[1], right_shoulder[1]])) - 1.4 * radius
    y_bottom = torch.max(torch.stack([left_elbow[1], right_elbow[1], left_shoulder[1], right_shoulder[1]])) + 1.4 * radius
    vertical_band = ((yy >= y_top) & (yy <= y_bottom)).float()
    region = region * vertical_band * fg_mask
    if face_mask is not None:
        region = region * (1.0 - 0.75 * face_mask)

    if region.sum().item() < 24:
        return None
    return region.clamp(0.0, 1.0)


def _resolve_shoulder_arm_region_from_data(data, fg_mask, face_mask=None, config_opt=None):
    region_source = 'union' if config_opt is None else str(config_opt.get('shoulder_arm_region_source', 'union'))
    parser_dilate = 9 if config_opt is None else int(config_opt.get('shoulder_arm_region_parser_dilate', 9))

    parser_arm_mask = _parser_region_mask(data, fg_mask, PARSER_ARM_LABELS)
    if parser_arm_mask is not None:
        if parser_dilate > 1:
            parser_arm_mask = _binary_dilate(parser_arm_mask, parser_dilate).clamp(0.0, 1.0)
        parser_arm_mask = parser_arm_mask * fg_mask
        if face_mask is not None:
            parser_arm_mask = parser_arm_mask * (1.0 - 0.75 * face_mask)

    joint_arm_mask = None
    if region_source not in ('parser_only', 'heuristic_only'):
        joint_arm_mask = _joint_guided_shoulder_arm_region_mask(data, fg_mask, face_mask=face_mask)

    heuristic_arm_mask = None

    def _get_heuristic_arm_mask():
        nonlocal heuristic_arm_mask
        if heuristic_arm_mask is None:
            heuristic_arm_mask = _heuristic_shoulder_arm_region_mask(fg_mask, face_mask=face_mask)
        return heuristic_arm_mask

    if region_source == 'parser_only':
        if parser_arm_mask is not None:
            region = parser_arm_mask
            actual_source = 'parser_only'
        else:
            region = torch.zeros_like(fg_mask)
            actual_source = 'parser_missing_empty'
    elif region_source == 'joint_only':
        if joint_arm_mask is not None:
            region = joint_arm_mask
            actual_source = 'joint_only'
        else:
            region = _get_heuristic_arm_mask()
            actual_source = 'joint_fallback_heuristic'
    elif region_source == 'parser_prefer':
        if parser_arm_mask is not None:
            region = parser_arm_mask
            actual_source = 'parser_prefer_parser'
        elif joint_arm_mask is not None:
            region = joint_arm_mask
            actual_source = 'parser_prefer_joint_fallback'
        else:
            region = _get_heuristic_arm_mask()
            actual_source = 'parser_prefer_heuristic_fallback'
    elif region_source == 'heuristic_only':
        region = _get_heuristic_arm_mask()
        actual_source = 'heuristic_only'
    elif parser_arm_mask is not None and joint_arm_mask is not None:
        region = torch.maximum(parser_arm_mask, joint_arm_mask).clamp(0.0, 1.0) * fg_mask
        actual_source = 'union_parser_joint'
    elif parser_arm_mask is not None:
        region = parser_arm_mask
        actual_source = 'union_parser'
    elif joint_arm_mask is not None:
        region = joint_arm_mask
        actual_source = 'union_joint'
    else:
        region = _get_heuristic_arm_mask()
        actual_source = 'union_heuristic'

    meta = {
        'requested_source': region_source,
        'actual_source': actual_source,
        'has_parser': parser_arm_mask is not None,
        'has_joint': joint_arm_mask is not None,
        'region_pixels': float(region.sum().item()),
        'parser_pixels': float(parser_arm_mask.sum().item()) if parser_arm_mask is not None else 0.0,
        'joint_pixels': float(joint_arm_mask.sum().item()) if joint_arm_mask is not None else 0.0,
        'heuristic_pixels': float(heuristic_arm_mask.sum().item()) if heuristic_arm_mask is not None else 0.0,
    }
    return region, meta


def _shoulder_arm_region_effective_min_pixels(config_opt, region_meta=None):
    base_min_pixels = 96 if config_opt is None else int(config_opt.get('shoulder_arm_region_min_pixels', 96))
    if config_opt is None or region_meta is None:
        return base_min_pixels
    if not bool(config_opt.get('shoulder_arm_region_source_aware_validity_enable', False)):
        return base_min_pixels

    actual_source = str(region_meta.get('actual_source', ''))
    parser_backed_sources = {
        'parser_only',
        'parser_prefer_parser',
        'union_parser',
        'union_parser_joint',
    }
    joint_backed_sources = {
        'joint_only',
        'parser_prefer_joint_fallback',
        'union_joint',
    }
    heuristic_backed_sources = {
        'heuristic_only',
        'joint_fallback_heuristic',
        'parser_prefer_heuristic_fallback',
        'union_heuristic',
    }

    if actual_source in parser_backed_sources:
        min_pixels = int(config_opt.get('shoulder_arm_region_min_pixels_parser', 40))
    elif actual_source in joint_backed_sources:
        min_pixels = int(config_opt.get('shoulder_arm_region_min_pixels_joint', base_min_pixels))
    elif actual_source in heuristic_backed_sources:
        min_pixels = int(config_opt.get('shoulder_arm_region_min_pixels_heuristic', base_min_pixels))
    else:
        min_pixels = base_min_pixels

    floor = int(config_opt.get('shoulder_arm_region_min_pixels_floor', 1))
    return max(min_pixels, floor)


def _maybe_log_shoulder_arm_region_debug(config_opt, schedule_iteration, region_meta, region_valid, min_pixels):
    if config_opt is None or region_meta is None:
        return
    if not bool(config_opt.get('shoulder_arm_region_debug_enable', False)):
        return

    interval = int(config_opt.get('shoulder_arm_region_debug_interval', 0))
    warmup_iters = int(config_opt.get('shoulder_arm_region_debug_warmup_iters', 0))
    if schedule_iteration < warmup_iters:
        return
    if interval > 0 and schedule_iteration % interval != 0:
        return

    requested_source = str(region_meta.get('requested_source', ''))
    actual_source = str(region_meta.get('actual_source', ''))
    has_parser = int(bool(region_meta.get('has_parser', False)))
    has_joint = int(bool(region_meta.get('has_joint', False)))
    region_pixels = float(region_meta.get('region_pixels', 0.0))
    parser_pixels = float(region_meta.get('parser_pixels', 0.0))
    joint_pixels = float(region_meta.get('joint_pixels', 0.0))
    heuristic_pixels = float(region_meta.get('heuristic_pixels', 0.0))

    print(
        (
            f"[ShoulderROI] iter={schedule_iteration} requested={requested_source} actual={actual_source} "
            f"has_parser={has_parser} has_joint={has_joint} region_pixels={region_pixels:.1f} "
            f"parser_pixels={parser_pixels:.1f} joint_pixels={joint_pixels:.1f} "
            f"heuristic_pixels={heuristic_pixels:.1f} min_pixels={int(min_pixels)} valid={int(region_valid)}"
        ),
        flush=True,
    )


def _select_shoulder_supervision_basis_mask(shoulder_arm_mask, shoulder_focus_mask, shoulder_collar_mask, fg_mask, mode='arm'):
    alias_map = {
        'arm': 'arm',
        'shoulder_arm': 'arm',
        'focus': 'focus',
        'shoulder_focus': 'focus',
        'collar': 'collar',
        'shoulder_collar': 'collar',
        'focus_union_collar': 'focus_union_collar',
        'focus_collar': 'focus_union_collar',
        'collar_focus': 'focus_union_collar',
        'shoulder_focus_union_collar': 'focus_union_collar',
    }
    resolved_mode = alias_map.get(str(mode).lower(), 'arm')

    arm_mask = shoulder_arm_mask.clamp(0.0, 1.0) * fg_mask
    focus_mask = shoulder_focus_mask.clamp(0.0, 1.0) * fg_mask
    collar_mask = shoulder_collar_mask.clamp(0.0, 1.0) * fg_mask
    basis_masks = {
        'arm': arm_mask,
        'focus': focus_mask,
        'collar': collar_mask,
        'focus_union_collar': torch.maximum(focus_mask, collar_mask).clamp(0.0, 1.0) * fg_mask,
    }
    return basis_masks[resolved_mode], resolved_mode


def _materialize_shoulder_supervision_pattern(
    basis_mask,
    fg_mask,
    pattern_mode='raw',
    boundary_mask=None,
    outer_shell_mask=None,
    target_mask=None,
    opacity=None,
    disagreement_threshold=0.08,
):
    alias_map = {
        'raw': 'raw',
        'region': 'raw',
        'boundary': 'boundary',
        'outer_shell': 'outer_shell',
        'shell': 'outer_shell',
        'error_band': 'error_band',
        'disagreement': 'error_band',
        'boundary_or_error': 'boundary_or_error',
        'boundary_plus_error': 'boundary_or_error',
    }
    resolved_pattern = alias_map.get(str(pattern_mode).lower(), 'raw')
    zero_mask = torch.zeros_like(fg_mask)
    final_mask = basis_mask

    if resolved_pattern == 'boundary':
        final_mask = basis_mask * boundary_mask if boundary_mask is not None else zero_mask
    elif resolved_pattern == 'outer_shell':
        final_mask = basis_mask * outer_shell_mask if outer_shell_mask is not None else zero_mask
    elif resolved_pattern in {'error_band', 'boundary_or_error'}:
        error_band = zero_mask
        if torch.is_tensor(target_mask) and torch.is_tensor(opacity):
            pred_binary = (opacity.detach() >= float(disagreement_threshold)).float()
            target_binary = (target_mask.to(device=pred_binary.device, dtype=pred_binary.dtype) >= 0.5).float()
            error_band = (pred_binary - target_binary).abs() * basis_mask
        if resolved_pattern == 'error_band':
            final_mask = error_band
        else:
            boundary_component = basis_mask * boundary_mask if boundary_mask is not None else zero_mask
            final_mask = torch.maximum(boundary_component, error_band)

    final_mask = final_mask.clamp(0.0, 1.0) * fg_mask
    return final_mask, resolved_pattern


def _build_local_region_supervision_mask(
    region_mask,
    fg_mask,
    pattern_mode='raw',
    boundary_mask=None,
    outer_shell_mask=None,
    target_mask=None,
    opacity=None,
    basis_dilate=0,
    final_dilate=0,
    disagreement_threshold=0.08,
):
    basis_mask = region_mask.clamp(0.0, 1.0) * fg_mask
    if int(basis_dilate) > 1 and basis_mask.sum().item() > 0:
        basis_mask = _binary_dilate(basis_mask, int(basis_dilate)).clamp(0.0, 1.0) * fg_mask

    final_mask, resolved_pattern = _materialize_shoulder_supervision_pattern(
        basis_mask,
        fg_mask,
        pattern_mode=pattern_mode,
        boundary_mask=boundary_mask,
        outer_shell_mask=outer_shell_mask,
        target_mask=target_mask,
        opacity=opacity,
        disagreement_threshold=disagreement_threshold,
    )
    if int(final_dilate) > 1 and final_mask.sum().item() > 0:
        final_mask = _binary_dilate(final_mask, int(final_dilate)).clamp(0.0, 1.0)
        support_mask = basis_mask if resolved_pattern != 'raw' else fg_mask
        final_mask = final_mask * support_mask

    meta = {
        'basis_mode': 'region',
        'pattern_mode': resolved_pattern,
        'basis_pixels': float(basis_mask.sum().item()),
        'final_pixels': float(final_mask.sum().item()),
    }
    return final_mask.clamp(0.0, 1.0), meta


def _build_shoulder_supervision_mask(
    shoulder_arm_mask,
    shoulder_focus_mask,
    shoulder_collar_mask,
    fg_mask,
    basis_mode='arm',
    pattern_mode='raw',
    boundary_mask=None,
    outer_shell_mask=None,
    target_mask=None,
    opacity=None,
    basis_dilate=0,
    final_dilate=0,
    disagreement_threshold=0.08,
):
    basis_mask, resolved_basis = _select_shoulder_supervision_basis_mask(
        shoulder_arm_mask,
        shoulder_focus_mask,
        shoulder_collar_mask,
        fg_mask,
        mode=basis_mode,
    )
    if int(basis_dilate) > 1 and basis_mask.sum().item() > 0:
        basis_mask = _binary_dilate(basis_mask, int(basis_dilate)).clamp(0.0, 1.0) * fg_mask

    final_mask, resolved_pattern = _materialize_shoulder_supervision_pattern(
        basis_mask,
        fg_mask,
        pattern_mode=pattern_mode,
        boundary_mask=boundary_mask,
        outer_shell_mask=outer_shell_mask,
        target_mask=target_mask,
        opacity=opacity,
        disagreement_threshold=disagreement_threshold,
    )
    if int(final_dilate) > 1 and final_mask.sum().item() > 0:
        final_mask = _binary_dilate(final_mask, int(final_dilate)).clamp(0.0, 1.0)
        support_mask = basis_mask if resolved_pattern != 'raw' else fg_mask
        final_mask = final_mask * support_mask

    meta = {
        'basis_mode': resolved_basis,
        'pattern_mode': resolved_pattern,
        'basis_pixels': float(basis_mask.sum().item()),
        'final_pixels': float(final_mask.sum().item()),
    }
    return final_mask.clamp(0.0, 1.0), meta


def _mask_bbox_stats(mask, padding=0):
    if not torch.is_tensor(mask):
        return 0.0, 0.0
    pixels = float(mask.sum().item())
    if pixels <= 0.0:
        return 0.0, 0.0
    y1, y2, x1, x2 = _foreground_bbox_from_mask(mask, padding=padding)
    bbox_area = float(max(y2 - y1, 0) * max(x2 - x1, 0))
    fill_ratio = pixels / max(bbox_area, 1.0)
    return bbox_area, fill_ratio


def _maybe_log_shoulder_local_mask_debug(config_opt, schedule_iteration, debug_info):
    if config_opt is None or debug_info is None:
        return
    if not bool(config_opt.get('shoulder_local_mask_debug_enable', False)):
        return

    interval = int(config_opt.get('shoulder_local_mask_debug_interval', 0))
    warmup_iters = int(config_opt.get('shoulder_local_mask_debug_warmup_iters', 0))
    if schedule_iteration < warmup_iters:
        return
    if interval > 0 and schedule_iteration % interval != 0:
        return

    print(
        (
            f"[ShoulderLocalMask] iter={schedule_iteration} "
            f"arm={debug_info.get('arm_pixels', 0.0):.1f} focus={debug_info.get('focus_pixels', 0.0):.1f} collar={debug_info.get('collar_pixels', 0.0):.1f} "
            f"image={debug_info.get('image_basis', 'arm')}/{debug_info.get('image_pattern', 'raw')}:{debug_info.get('image_pixels', 0.0):.1f} "
            f"percep={debug_info.get('perceptual_basis', 'arm')}/{debug_info.get('perceptual_pattern', 'raw')}:{debug_info.get('perceptual_pixels', 0.0):.1f} "
            f"percep_bbox={debug_info.get('perceptual_bbox_area', 0.0):.1f} percep_fill={debug_info.get('perceptual_fill_ratio', 0.0):.4f} "
            f"boundary={debug_info.get('boundary_basis', 'arm')}/{debug_info.get('boundary_pattern', 'boundary')}:{debug_info.get('boundary_pixels', 0.0):.1f} "
            f"region={debug_info.get('region_basis', 'arm')}/{debug_info.get('region_pattern', 'raw')}:{debug_info.get('region_pixels', 0.0):.1f} "
            f"disagree={debug_info.get('disagreement_basis', 'arm')}/{debug_info.get('disagreement_pattern', 'error_band')}:{debug_info.get('disagreement_pixels', 0.0):.1f} "
            f"outer={debug_info.get('outer_basis', 'arm')}/{debug_info.get('outer_pattern', 'outer_shell')}:{debug_info.get('outer_pixels', 0.0):.1f}"
        ),
        flush=True,
    )


def _maybe_log_shoulder_local_loss_diag(config_opt, schedule_iteration, diag_info):
    if config_opt is None or diag_info is None:
        return
    if not bool(config_opt.get('shoulder_local_loss_diag_enable', False)):
        return

    interval = int(config_opt.get('shoulder_local_loss_diag_interval', 0))
    warmup_iters = int(config_opt.get('shoulder_local_loss_diag_warmup_iters', 0))
    if schedule_iteration < warmup_iters:
        return
    if interval > 0 and schedule_iteration % interval != 0:
        return

    local_image = float(diag_info.get('local_image', 0.0))
    global_image = float(diag_info.get('global_image', 0.0))
    local_boundary = float(diag_info.get('local_boundary', 0.0))
    global_boundary = float(diag_info.get('global_boundary', 0.0))
    base_total = max(float(diag_info.get('base_total', 0.0)), 1.0e-8)
    total_loss = max(float(diag_info.get('total_loss', 0.0)), 1.0e-8)
    local_total = local_image + local_boundary
    print(
        (
            f"[ShoulderLocalDiag] iter={schedule_iteration} "
            f"local_image={local_image:.6f} global_image={global_image:.6f} image_share_base={local_image / base_total:.4f} "
            f"local_boundary={local_boundary:.6f} global_boundary={global_boundary:.6f} "
            f"local_total_share={local_total / total_loss:.4f} total_loss={total_loss:.6f}"
        ),
        flush=True,
    )


def _maybe_log_waist_region_debug(config_opt, schedule_iteration, region_meta, region_valid, min_pixels):
    if config_opt is None or region_meta is None:
        return
    if not bool(config_opt.get('waist_region_debug_enable', False)):
        return

    interval = int(config_opt.get('waist_region_debug_interval', 0))
    warmup_iters = int(config_opt.get('waist_region_debug_warmup_iters', 0))
    if schedule_iteration < warmup_iters:
        return
    if interval > 0 and schedule_iteration % interval != 0:
        return

    print(
        (
            f"[WaistROI] iter={schedule_iteration} "
            f"requested={region_meta.get('requested_source', '')} actual={region_meta.get('actual_source', '')} "
            f"mode={region_meta.get('region_mode', '')} has_parser={int(bool(region_meta.get('has_parser', False)))} "
            f"parser_pixels={float(region_meta.get('parser_pixels', 0.0)):.1f} "
            f"region_pixels={float(region_meta.get('region_pixels', 0.0)):.1f} "
            f"min_pixels={int(min_pixels)} valid={int(region_valid)}"
        ),
        flush=True,
    )


def _maybe_log_upper_torso_region_debug(config_opt, schedule_iteration, region_meta, region_valid, min_pixels):
    if config_opt is None or region_meta is None:
        return
    if not bool(config_opt.get('upper_torso_region_debug_enable', False)):
        return

    interval = int(config_opt.get('upper_torso_region_debug_interval', 0))
    warmup_iters = int(config_opt.get('upper_torso_region_debug_warmup_iters', 0))
    if schedule_iteration < warmup_iters:
        return
    if interval > 0 and schedule_iteration % interval != 0:
        return

    print(
        (
            f"[UpperTorsoROI] iter={schedule_iteration} "
            f"requested={region_meta.get('requested_source', '')} actual={region_meta.get('actual_source', '')} "
            f"has_parser={int(bool(region_meta.get('has_parser', False)))} "
            f"has_joint={int(bool(region_meta.get('has_joint', False)))} "
            f"parser_pixels={float(region_meta.get('parser_pixels', 0.0)):.1f} "
            f"joint_pixels={float(region_meta.get('joint_pixels', 0.0)):.1f} "
            f"heuristic_pixels={float(region_meta.get('heuristic_pixels', 0.0)):.1f} "
            f"region_pixels={float(region_meta.get('region_pixels', 0.0)):.1f} "
            f"min_pixels={int(min_pixels)} valid={int(region_valid)}"
        ),
        flush=True,
    )


def _maybe_log_waist_local_loss_diag(config_opt, schedule_iteration, diag_info):
    if config_opt is None or diag_info is None:
        return
    if not bool(config_opt.get('waist_local_loss_diag_enable', False)):
        return

    interval = int(config_opt.get('waist_local_loss_diag_interval', 0))
    warmup_iters = int(config_opt.get('waist_local_loss_diag_warmup_iters', 0))
    if schedule_iteration < warmup_iters:
        return
    if interval > 0 and schedule_iteration % interval != 0:
        return

    waist_image = float(diag_info.get('waist_image', 0.0))
    shoulder_image = float(diag_info.get('shoulder_image', 0.0))
    global_image = float(diag_info.get('global_image', 0.0))
    region_pixels = float(diag_info.get('region_pixels', 0.0))
    perceptual_pixels = float(diag_info.get('perceptual_pixels', 0.0))
    base_total = max(float(diag_info.get('base_total', 0.0)), 1.0e-8)
    total_loss = max(float(diag_info.get('total_loss', 0.0)), 1.0e-8)
    print(
        (
            f"[WaistLocalDiag] iter={schedule_iteration} "
            f"waist_image={waist_image:.6f} shoulder_image={shoulder_image:.6f} global_image={global_image:.6f} "
            f"waist_share_base={waist_image / base_total:.4f} waist_share_total={waist_image / total_loss:.4f} "
            f"region_pixels={region_pixels:.1f} perceptual_pixels={perceptual_pixels:.1f}"
        ),
        flush=True,
    )


def _mask_active_pixels(mask, threshold=0.05):
    if not torch.is_tensor(mask):
        return 0.0
    return float((mask > float(threshold)).sum().item())


def _mask_overlap_sum(mask_a, mask_b):
    if not torch.is_tensor(mask_a) or not torch.is_tensor(mask_b):
        return 0.0
    mask_a = mask_a.float()
    mask_b = mask_b.to(device=mask_a.device, dtype=mask_a.dtype)
    return float((mask_a * mask_b).sum().item())


def _optional_float(value):
    if torch.is_tensor(value):
        return float(value.detach().item())
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_optional_float(value, fmt='.4f', default='na'):
    value = _optional_float(value)
    if value is None:
        return default
    return format(value, fmt)


def _collect_texture_clarity_stats(texture_module):
    if texture_module is None:
        return {}

    stats = {}
    attr_map = {
        'structured_trunk_shared_abs_mean': 'last_structured_trunk_shared_abs_mean',
        'structured_trunk_shared_residual_abs_mean': 'last_structured_trunk_shared_residual_abs_mean',
        'structured_trunk_carrier_abs_mean': 'last_structured_trunk_carrier_abs_mean',
        'structured_trunk_structure_abs_mean': 'last_structured_trunk_structure_abs_mean',
        'structured_trunk_structure_raw_abs_mean': 'last_structured_trunk_structure_raw_abs_mean',
        'structured_trunk_structure_residual_abs_mean': 'last_structured_trunk_structure_residual_abs_mean',
        'structured_trunk_local_abs_mean': 'last_structured_trunk_local_abs_mean',
        'structured_trunk_local_raw_abs_mean': 'last_structured_trunk_local_raw_abs_mean',
        'structured_trunk_local_gate_mean': 'last_structured_trunk_local_gate_mean',
        'structured_trunk_local_residual_abs_mean': 'last_structured_trunk_local_residual_abs_mean',
        'structured_trunk_total_abs_mean': 'last_structured_trunk_total_abs_mean',
        'structured_trunk_head_abs_mean': 'last_structured_trunk_head_abs_mean',
        'structured_trunk_head_color_abs_mean': 'last_structured_trunk_head_color_abs_mean',
        'structured_trunk_head_gate_mean': 'last_structured_trunk_head_gate_mean',
        'structured_trunk_head_gate_boost_mean': 'last_structured_trunk_head_gate_boost_mean',
        'structured_trunk_head_local_color_abs_mean': 'last_structured_trunk_head_local_color_abs_mean',
        'structured_trunk_head_fusion_abs_mean': 'last_structured_trunk_head_fusion_abs_mean',
        'structured_trunk_owner_abs_mean': 'last_structured_trunk_owner_abs_mean',
        'structured_trunk_owner_input_abs_mean': 'last_structured_trunk_owner_input_abs_mean',
        'structured_trunk_owner_color_abs_mean': 'last_structured_trunk_owner_color_abs_mean',
        'structured_trunk_owner_support_mean': 'last_structured_trunk_owner_support_mean',
        'structured_trunk_owner_gate_mean': 'last_structured_trunk_owner_gate_mean',
        'structured_trunk_owner_takeover_mean': 'last_structured_trunk_owner_takeover_mean',
        'structured_trunk_owner_takeover_legacy_scale_mean': 'last_structured_trunk_owner_takeover_legacy_scale_mean',
        'structured_trunk_owner_boundary_abs_mean': 'last_structured_trunk_owner_boundary_abs_mean',
        'structured_trunk_owner_boundary_input_abs_mean': 'last_structured_trunk_owner_boundary_input_abs_mean',
        'structured_trunk_owner_boundary_color_abs_mean': 'last_structured_trunk_owner_boundary_color_abs_mean',
        'structured_trunk_owner_boundary_focus_mean': 'last_structured_trunk_owner_boundary_focus_mean',
        'structured_trunk_owner_boundary_gate_mean': 'last_structured_trunk_owner_boundary_gate_mean',
        'structured_trunk_owner_boundary_takeover_mean': 'last_structured_trunk_owner_boundary_takeover_mean',
        'structured_trunk_scaffold_abs_mean': 'last_structured_trunk_scaffold_abs_mean',
        'structured_trunk_coarse_abs_mean': 'last_structured_trunk_coarse_abs_mean',
        'structured_trunk_hf_abs_mean': 'last_structured_trunk_hf_abs_mean',
        'structured_trunk_hf_color_abs_mean': 'last_structured_trunk_hf_color_abs_mean',
        'structured_trunk_hf_gate_mean': 'last_structured_trunk_hf_gate_mean',
        'structured_trunk_hf_local_color_abs_mean': 'last_structured_trunk_hf_local_color_abs_mean',
        'structured_trunk_hf_fusion_abs_mean': 'last_structured_trunk_hf_fusion_abs_mean',
        'structured_trunk_hf_region_gain_mean': 'last_structured_trunk_hf_region_gain_mean',
        'structured_trunk_coarse_region_scale_mean': 'last_structured_trunk_coarse_region_scale_mean',
        'structured_trunk_region_support_mean': 'last_structured_trunk_region_support_mean',
        'detail_residual_abs_mean': 'last_detail_residual_abs_mean',
        'detail_tiny_repair_abs_mean': 'last_detail_tiny_repair_abs_mean',
        'detail_scale': 'last_detail_scale',
        'detail_schedule_iteration': 'last_detail_schedule_iteration',
        'detail_gate_mean': 'last_detail_gate_mean',
        'detail_gate_fraction': 'last_detail_gate_fraction',
        'detail_high_freq_residual_abs_mean': 'last_detail_high_freq_residual_abs_mean',
        'detail_high_freq_scale': 'last_detail_high_freq_scale',
        'detail_high_freq_gate_mean': 'last_detail_high_freq_gate_mean',
        'detail_high_freq_gate_fraction': 'last_detail_high_freq_gate_fraction',
        'detail_high_freq_point_gate_mean': 'last_detail_high_freq_point_gate_mean',
        'detail_high_freq_point_gate_fraction': 'last_detail_high_freq_point_gate_fraction',
        'detail_high_freq_carrier_abs_mean': 'last_detail_high_freq_carrier_abs_mean',
        'detail_high_freq_chroma_abs_mean': 'last_detail_high_freq_chroma_abs_mean',
        'detail_high_freq_luma_abs_mean': 'last_detail_high_freq_luma_abs_mean',
        'detail_high_freq_face_abs_mean': 'last_detail_high_freq_face_abs_mean',
        'detail_high_freq_face_raw_abs_mean': 'last_detail_high_freq_face_raw_abs_mean',
        'detail_high_freq_face_after_gate_abs_mean': 'last_detail_high_freq_face_after_gate_abs_mean',
        'detail_high_freq_face_gate_mean': 'last_detail_high_freq_face_gate_mean',
        'detail_high_freq_face_gate_fraction': 'last_detail_high_freq_face_gate_fraction',
        'detail_high_freq_face_point_gate_mean': 'last_detail_high_freq_face_point_gate_mean',
        'detail_high_freq_face_point_gate_fraction': 'last_detail_high_freq_face_point_gate_fraction',
        'detail_high_freq_face_local_abs_mean': 'last_detail_high_freq_face_local_abs_mean',
        'detail_high_freq_face_local_raw_abs_mean': 'last_detail_high_freq_face_local_raw_abs_mean',
        'detail_high_freq_face_extra_local_abs_mean': 'last_detail_high_freq_face_extra_local_abs_mean',
        'detail_high_freq_face_extra_local_raw_abs_mean': 'last_detail_high_freq_face_extra_local_raw_abs_mean',
        'detail_high_freq_face_extra_local_gate_mean': 'last_detail_high_freq_face_extra_local_gate_mean',
        'detail_high_freq_structure_abs_mean': 'last_detail_high_freq_structure_abs_mean',
        'detail_high_freq_structure_raw_abs_mean': 'last_detail_high_freq_structure_raw_abs_mean',
        'detail_high_freq_boundary_floor_mean': 'last_detail_high_freq_boundary_floor_mean',
        'detail_high_freq_view_conflict_scale': 'last_detail_high_freq_view_conflict_scale',
        'detail_high_freq_view_conflict_abs_mean': 'last_detail_high_freq_view_conflict_abs_mean',
        'detail_high_freq_view_conflict_raw_abs_mean': 'last_detail_high_freq_view_conflict_raw_abs_mean',
        'detail_high_freq_view_conflict_gate_mean': 'last_detail_high_freq_view_conflict_gate_mean',
        'detail_high_freq_view_conflict_gate_fraction': 'last_detail_high_freq_view_conflict_gate_fraction',
        'detail_high_freq_view_conflict_point_gate_mean': 'last_detail_high_freq_view_conflict_point_gate_mean',
        'detail_high_freq_view_conflict_point_gate_fraction': 'last_detail_high_freq_view_conflict_point_gate_fraction',
        'detail_high_freq_view_conflict_boundary_suppress_mean': 'last_detail_high_freq_view_conflict_boundary_suppress_mean',
    }
    for key, attr_name in attr_map.items():
        value = _optional_float(getattr(texture_module, attr_name, None))
        if value is not None:
            stats[key] = value
    return stats


def _build_owner_local_detail_boost(config_opt, schedule_iteration, texture_module):
    region_names = ('face', 'shoulder_arm', 'upper_torso', 'upper_torso_core')
    edge_region_names = ('face', 'shoulder_arm')
    boundary_region_names = ('shoulder_arm', 'upper_torso')
    boost = {
        'enabled': False,
        'takeover_mean': _optional_float(
            getattr(texture_module, 'last_structured_trunk_owner_takeover_mean', None)
        ) if texture_module is not None else None,
        'legacy_scale_mean': _optional_float(
            getattr(texture_module, 'last_structured_trunk_owner_takeover_legacy_scale_mean', None)
        ) if texture_module is not None else None,
        'takeover_signal': 0.0,
        'legacy_signal': 0.0,
        'legacy_mix': 0.0,
        'ownership_signal': 0.0,
        'detail_scales': {name: 1.0 for name in region_names},
        'luma_scales': {name: 1.0 for name in region_names},
        'patch_scales': {name: 1.0 for name in region_names},
        'edge_scales': {name: 1.0 for name in edge_region_names},
        'boundary_scales': {name: 1.0 for name in boundary_region_names},
    }
    if config_opt is None or texture_module is None:
        return boost
    if not bool(config_opt.get('owner_local_detail_boost_enable', False)):
        return boost

    warmup_iters = int(config_opt.get('owner_local_detail_boost_warmup_iters', 0))
    if schedule_iteration < warmup_iters:
        return boost

    takeover_mean = boost['takeover_mean']
    if takeover_mean is None:
        return boost

    takeover_floor = float(config_opt.get('owner_local_detail_boost_takeover_floor', 0.35))
    takeover_gain = max(
        float(C(schedule_iteration, config_opt.get('owner_local_detail_boost_takeover_gain', 1.0))),
        0.0,
    )
    takeover_power = float(config_opt.get('owner_local_detail_boost_takeover_power', 1.0))
    takeover_signal = (takeover_mean - takeover_floor) / max(1.0 - takeover_floor, 1.0e-6)
    takeover_signal = max(0.0, min(1.0, takeover_signal * takeover_gain))
    if takeover_power != 1.0 and takeover_signal > 0.0:
        takeover_signal = takeover_signal ** takeover_power
    boost['takeover_signal'] = takeover_signal

    legacy_mix = float(C(schedule_iteration, config_opt.get('owner_local_detail_boost_legacy_mix', 0.0)))
    legacy_mix = max(0.0, min(1.0, legacy_mix))
    boost['legacy_mix'] = legacy_mix

    legacy_signal = 0.0
    legacy_scale_mean = boost['legacy_scale_mean']
    if legacy_scale_mean is not None:
        legacy_scale_mean = max(0.0, min(1.0, legacy_scale_mean))
        legacy_floor = float(config_opt.get('owner_local_detail_boost_legacy_floor', 0.0))
        legacy_power = float(config_opt.get('owner_local_detail_boost_legacy_power', 1.0))
        legacy_signal = (1.0 - legacy_scale_mean - legacy_floor) / max(1.0 - legacy_floor, 1.0e-6)
        legacy_signal = max(0.0, min(1.0, legacy_signal))
        if legacy_power != 1.0 and legacy_signal > 0.0:
            legacy_signal = legacy_signal ** legacy_power
    boost['legacy_signal'] = legacy_signal

    ownership_signal = (1.0 - legacy_mix) * takeover_signal + legacy_mix * legacy_signal
    min_signal = float(config_opt.get('owner_local_detail_boost_min_signal', 0.0))
    if ownership_signal < min_signal:
        ownership_signal = 0.0
    ownership_signal = max(0.0, min(1.0, ownership_signal))
    boost['ownership_signal'] = ownership_signal

    detail_max_extra = max(
        0.0,
        float(C(schedule_iteration, config_opt.get('owner_local_detail_boost_detail_max_extra', 0.0))),
    )
    luma_max_extra = max(
        0.0,
        float(C(schedule_iteration, config_opt.get('owner_local_detail_boost_luma_max_extra', detail_max_extra))),
    )
    patch_max_extra = max(
        0.0,
        float(C(schedule_iteration, config_opt.get('owner_local_detail_boost_patch_max_extra', detail_max_extra))),
    )
    edge_max_extra = max(
        0.0,
        float(C(schedule_iteration, config_opt.get('owner_local_detail_boost_edge_max_extra', 0.0))),
    )
    boundary_max_extra = max(
        0.0,
        float(C(schedule_iteration, config_opt.get('owner_local_detail_boost_boundary_max_extra', 0.0))),
    )
    if ownership_signal <= 0.0 or max(
        detail_max_extra,
        luma_max_extra,
        patch_max_extra,
        edge_max_extra,
        boundary_max_extra,
    ) <= 0.0:
        return boost

    region_strengths = {
        'face': float(config_opt.get('owner_local_detail_boost_face_strength', 1.0)),
        'shoulder_arm': float(config_opt.get('owner_local_detail_boost_shoulder_strength', 1.0)),
        'upper_torso': float(config_opt.get('owner_local_detail_boost_upper_torso_strength', 1.0)),
        'upper_torso_core': float(config_opt.get('owner_local_detail_boost_upper_torso_core_strength', 1.0)),
    }
    for region_name, region_strength in region_strengths.items():
        region_strength = max(region_strength, 0.0)
        boost['detail_scales'][region_name] = (
            1.0 + detail_max_extra * region_strength * ownership_signal
        )
        boost['luma_scales'][region_name] = (
            1.0 + luma_max_extra * region_strength * ownership_signal
        )
        boost['patch_scales'][region_name] = (
            1.0 + patch_max_extra * region_strength * ownership_signal
        )

    edge_region_strengths = {
        'face': float(
            config_opt.get(
                'owner_local_detail_boost_face_edge_strength',
                region_strengths['face'],
            )
        ),
        'shoulder_arm': float(
            config_opt.get(
                'owner_local_detail_boost_shoulder_edge_strength',
                region_strengths['shoulder_arm'],
            )
        ),
    }
    for region_name, region_strength in edge_region_strengths.items():
        region_strength = max(region_strength, 0.0)
        boost['edge_scales'][region_name] = (
            1.0 + edge_max_extra * region_strength * ownership_signal
        )

    boundary_region_strengths = {
        'shoulder_arm': float(
            config_opt.get(
                'owner_local_detail_boost_shoulder_boundary_strength',
                region_strengths['shoulder_arm'],
            )
        ),
        'upper_torso': float(
            config_opt.get(
                'owner_local_detail_boost_upper_torso_boundary_strength',
                region_strengths['upper_torso'],
            )
        ),
    }
    for region_name, region_strength in boundary_region_strengths.items():
        region_strength = max(region_strength, 0.0)
        boost['boundary_scales'][region_name] = (
            1.0 + boundary_max_extra * region_strength * ownership_signal
        )

    boost['enabled'] = True
    return boost


def _maybe_log_clarity_debug(config_opt, schedule_iteration, debug_info):
    if config_opt is None or debug_info is None:
        return
    if not bool(config_opt.get('clarity_debug_enable', False)):
        return

    interval = int(config_opt.get('clarity_debug_interval', 0))
    warmup_iters = int(config_opt.get('clarity_debug_warmup_iters', 0))
    if schedule_iteration < warmup_iters:
        return
    if interval > 0 and schedule_iteration % interval != 0:
        return

    print(
        (
            f"[ClarityMask] iter={schedule_iteration} "
            f"face={debug_info.get('face_source', '')} roi={float(debug_info.get('face_roi_pixels', 0.0)):.1f} "
            f"detail={float(debug_info.get('face_detail_pixels', 0.0)):.1f}/{float(debug_info.get('face_detail_active_pixels', 0.0)):.0f} "
            f"boundary={float(debug_info.get('face_detail_boundary_overlap', 0.0)):.1f} "
            f"shoulder={debug_info.get('shoulder_source', '')} roi={float(debug_info.get('shoulder_roi_pixels', 0.0)):.1f} "
            f"detail={float(debug_info.get('shoulder_detail_pixels', 0.0)):.1f}/{float(debug_info.get('shoulder_detail_active_pixels', 0.0)):.0f} "
            f"boundary={float(debug_info.get('shoulder_detail_boundary_overlap', 0.0)):.1f} "
            f"torso={debug_info.get('upper_torso_source', '')} roi={float(debug_info.get('upper_torso_roi_pixels', 0.0)):.1f} "
            f"detail={float(debug_info.get('upper_torso_detail_pixels', 0.0)):.1f}/{float(debug_info.get('upper_torso_detail_active_pixels', 0.0)):.0f} "
            f"boundary={float(debug_info.get('upper_torso_detail_boundary_overlap', 0.0)):.1f} "
            f"core={float(debug_info.get('upper_torso_core_pixels', 0.0)):.1f}/{float(debug_info.get('upper_torso_core_active_pixels', 0.0)):.0f} "
            f"supervise={float(debug_info.get('upper_torso_boundary_pixels', 0.0)):.1f}/{float(debug_info.get('upper_torso_outer_pixels', 0.0)):.1f} "
            f"waist_roi={float(debug_info.get('waist_roi_pixels', 0.0)):.1f} "
            f"waist_detail={float(debug_info.get('waist_detail_pixels', 0.0)):.1f}/{float(debug_info.get('waist_detail_active_pixels', 0.0)):.0f}"
        ),
        flush=True,
    )
    print(
        (
            f"[ClarityLoss] iter={schedule_iteration} "
            f"fullframe={float(debug_info.get('fullframe_image', 0.0)):.6f} "
            f"face={float(debug_info.get('face_image', 0.0)):.6f} "
            f"shoulder={float(debug_info.get('shoulder_image', 0.0)):.6f} "
            f"torso={float(debug_info.get('upper_torso_image', 0.0)):.6f} "
            f"waist={float(debug_info.get('waist_image', 0.0)):.6f} "
            f"local_share={float(debug_info.get('local_share', 0.0)):.4f} "
            f"face_dog={float(debug_info.get('face_luma_dog', 0.0)):.6f} "
            f"shoulder_dog={float(debug_info.get('shoulder_luma_dog', 0.0)):.6f} "
            f"torso_dog={float(debug_info.get('upper_torso_luma_dog', 0.0)):.6f} "
            f"torso_core_dog={float(debug_info.get('upper_torso_core_luma_dog', 0.0)):.6f} "
            f"face_patch={float(debug_info.get('face_patch', 0.0)):.6f} "
            f"shoulder_patch={float(debug_info.get('shoulder_patch', 0.0)):.6f} "
            f"torso_patch={float(debug_info.get('upper_torso_patch', 0.0)):.6f} "
            f"torso_core_patch={float(debug_info.get('upper_torso_core_patch', 0.0)):.6f} "
            f"total={float(debug_info.get('total_loss', 0.0)):.6f}"
        ),
        flush=True,
    )
    owner_boost_enabled = bool(debug_info.get('owner_local_detail_boost_enabled', False))
    owner_boost_signal = float(debug_info.get('owner_local_detail_boost_signal', 0.0))
    if owner_boost_enabled or owner_boost_signal > 0.0:
        print(
            (
                f"[ClarityOwnerBoost] iter={schedule_iteration} "
                f"signal={owner_boost_signal:.4f} "
                f"takeover={float(debug_info.get('owner_local_detail_boost_takeover', 0.0)):.4f} "
                f"takeover_signal={float(debug_info.get('owner_local_detail_boost_takeover_signal', 0.0)):.4f} "
                f"legacy_scale={float(debug_info.get('owner_local_detail_boost_legacy_scale', 0.0)):.4f} "
                f"legacy_signal={float(debug_info.get('owner_local_detail_boost_legacy_signal', 0.0)):.4f} "
                f"legacy_mix={float(debug_info.get('owner_local_detail_boost_legacy_mix', 0.0)):.4f} "
                f"face={float(debug_info.get('owner_local_detail_boost_face_detail_scale', 1.0)):.3f}/"
                f"{float(debug_info.get('owner_local_detail_boost_face_luma_scale', 1.0)):.3f}/"
                f"{float(debug_info.get('owner_local_detail_boost_face_patch_scale', 1.0)):.3f} "
                f"shoulder={float(debug_info.get('owner_local_detail_boost_shoulder_detail_scale', 1.0)):.3f}/"
                f"{float(debug_info.get('owner_local_detail_boost_shoulder_luma_scale', 1.0)):.3f}/"
                f"{float(debug_info.get('owner_local_detail_boost_shoulder_patch_scale', 1.0)):.3f} "
                f"torso={float(debug_info.get('owner_local_detail_boost_upper_torso_luma_scale', 1.0)):.3f}/"
                f"{float(debug_info.get('owner_local_detail_boost_upper_torso_patch_scale', 1.0)):.3f} "
                f"core={float(debug_info.get('owner_local_detail_boost_upper_torso_core_luma_scale', 1.0)):.3f}/"
                f"{float(debug_info.get('owner_local_detail_boost_upper_torso_core_patch_scale', 1.0)):.3f} "
                f"edge={float(debug_info.get('owner_local_detail_boost_face_edge_scale', 1.0)):.3f}/"
                f"{float(debug_info.get('owner_local_detail_boost_shoulder_edge_scale', 1.0)):.3f} "
                f"boundary={float(debug_info.get('owner_local_detail_boost_shoulder_boundary_scale', 1.0)):.3f}/"
                f"{float(debug_info.get('owner_local_detail_boost_upper_torso_boundary_scale', 1.0)):.3f}"
            ),
            flush=True,
        )

    texture_stats = debug_info.get('texture_stats', {})
    if texture_stats:
        print(
            (
                f"[ClarityTexTrunk] iter={schedule_iteration} "
                f"trunk={_format_optional_float(texture_stats.get('structured_trunk_total_abs_mean'), '.5f')} "
                f"shared={_format_optional_float(texture_stats.get('structured_trunk_shared_abs_mean'), '.5f')} "
                f"shared_res={_format_optional_float(texture_stats.get('structured_trunk_shared_residual_abs_mean'), '.5f')} "
                f"carrier={_format_optional_float(texture_stats.get('structured_trunk_carrier_abs_mean'), '.5f')} "
                f"struct_raw={_format_optional_float(texture_stats.get('structured_trunk_structure_raw_abs_mean'), '.5f')} "
                f"struct={_format_optional_float(texture_stats.get('structured_trunk_structure_abs_mean'), '.5f')} "
                f"struct_res={_format_optional_float(texture_stats.get('structured_trunk_structure_residual_abs_mean'), '.5f')} "
                f"local_raw={_format_optional_float(texture_stats.get('structured_trunk_local_raw_abs_mean'), '.5f')} "
                f"local={_format_optional_float(texture_stats.get('structured_trunk_local_abs_mean'), '.5f')} "
                f"local_res={_format_optional_float(texture_stats.get('structured_trunk_local_residual_abs_mean'), '.5f')} "
                f"local_gate={_format_optional_float(texture_stats.get('structured_trunk_local_gate_mean'), '.4f')} "
                f"head_color={_format_optional_float(texture_stats.get('structured_trunk_head_color_abs_mean'), '.5f')} "
                f"head={_format_optional_float(texture_stats.get('structured_trunk_head_abs_mean'), '.5f')} "
                f"head_gate={_format_optional_float(texture_stats.get('structured_trunk_head_gate_mean'), '.4f')} "
                f"head_boost={_format_optional_float(texture_stats.get('structured_trunk_head_gate_boost_mean'), '.4f')} "
                f"head_local={_format_optional_float(texture_stats.get('structured_trunk_head_local_color_abs_mean'), '.5f')} "
                f"owner_in={_format_optional_float(texture_stats.get('structured_trunk_owner_input_abs_mean'), '.5f')} "
                f"owner_color={_format_optional_float(texture_stats.get('structured_trunk_owner_color_abs_mean'), '.5f')} "
                f"owner={_format_optional_float(texture_stats.get('structured_trunk_owner_abs_mean'), '.5f')} "
                f"owner_support={_format_optional_float(texture_stats.get('structured_trunk_owner_support_mean'), '.4f')} "
                f"owner_gate={_format_optional_float(texture_stats.get('structured_trunk_owner_gate_mean'), '.4f')} "
                f"owner_takeover={_format_optional_float(texture_stats.get('structured_trunk_owner_takeover_mean'), '.4f')} "
                f"owner_legacy={_format_optional_float(texture_stats.get('structured_trunk_owner_takeover_legacy_scale_mean'), '.4f')} "
                f"boundary_in={_format_optional_float(texture_stats.get('structured_trunk_owner_boundary_input_abs_mean'), '.5f')} "
                f"boundary_color={_format_optional_float(texture_stats.get('structured_trunk_owner_boundary_color_abs_mean'), '.5f')} "
                f"boundary={_format_optional_float(texture_stats.get('structured_trunk_owner_boundary_abs_mean'), '.5f')} "
                f"boundary_focus={_format_optional_float(texture_stats.get('structured_trunk_owner_boundary_focus_mean'), '.4f')} "
                f"boundary_gate={_format_optional_float(texture_stats.get('structured_trunk_owner_boundary_gate_mean'), '.4f')} "
                f"boundary_takeover={_format_optional_float(texture_stats.get('structured_trunk_owner_boundary_takeover_mean'), '.4f')} "
                f"head_fuse={_format_optional_float(texture_stats.get('structured_trunk_head_fusion_abs_mean'), '.5f')} "
                f"hf_fuse={_format_optional_float(texture_stats.get('structured_trunk_hf_fusion_abs_mean'), '.5f')} "
                f"support={_format_optional_float(texture_stats.get('structured_trunk_region_support_mean'), '.4f')} "
                f"coarse_region={_format_optional_float(texture_stats.get('structured_trunk_coarse_region_scale_mean'), '.4f')} "
                f"hf_region={_format_optional_float(texture_stats.get('structured_trunk_hf_region_gain_mean'), '.4f')}"
            ),
            flush=True,
        )
        print(
            (
                f"[ClarityTexHF] iter={schedule_iteration} "
                f"tiny={_format_optional_float(texture_stats.get('detail_tiny_repair_abs_mean'), '.5f')} "
                f"detail_scale={_format_optional_float(texture_stats.get('detail_scale'), '.3f')} "
                f"hf_scale={_format_optional_float(texture_stats.get('detail_high_freq_scale'), '.3f')} "
                f"hf_gate={_format_optional_float(texture_stats.get('detail_high_freq_gate_mean'), '.4f')} "
                f"hf_gate_frac={_format_optional_float(texture_stats.get('detail_high_freq_gate_fraction'), '.4f')} "
                f"hf_point={_format_optional_float(texture_stats.get('detail_high_freq_point_gate_mean'), '.4f')} "
                f"hf_floor={_format_optional_float(texture_stats.get('detail_high_freq_boundary_floor_mean'), '.4f')} "
                f"carrier={_format_optional_float(texture_stats.get('detail_high_freq_carrier_abs_mean'), '.5f')} "
                f"struct_raw={_format_optional_float(texture_stats.get('detail_high_freq_structure_raw_abs_mean'), '.5f')} "
                f"struct={_format_optional_float(texture_stats.get('detail_high_freq_structure_abs_mean'), '.5f')} "
                f"luma={_format_optional_float(texture_stats.get('detail_high_freq_luma_abs_mean'), '.5f')} "
                f"face_local_raw={_format_optional_float(texture_stats.get('detail_high_freq_face_local_raw_abs_mean'), '.5f')} "
                f"face_local={_format_optional_float(texture_stats.get('detail_high_freq_face_local_abs_mean'), '.5f')} "
                f"extra_local_raw={_format_optional_float(texture_stats.get('detail_high_freq_face_extra_local_raw_abs_mean'), '.5f')} "
                f"extra_local={_format_optional_float(texture_stats.get('detail_high_freq_face_extra_local_abs_mean'), '.5f')} "
                f"extra_gate={_format_optional_float(texture_stats.get('detail_high_freq_face_extra_local_gate_mean'), '.4f')} "
                f"face_raw={_format_optional_float(texture_stats.get('detail_high_freq_face_raw_abs_mean'), '.5f')} "
                f"face_gated={_format_optional_float(texture_stats.get('detail_high_freq_face_after_gate_abs_mean'), '.5f')} "
                f"face={_format_optional_float(texture_stats.get('detail_high_freq_face_abs_mean'), '.5f')} "
                f"face_gate={_format_optional_float(texture_stats.get('detail_high_freq_face_gate_mean'), '.4f')} "
                f"face_point={_format_optional_float(texture_stats.get('detail_high_freq_face_point_gate_mean'), '.4f')} "
                f"view_conflict_scale={_format_optional_float(texture_stats.get('detail_high_freq_view_conflict_scale'), '.3f')} "
                f"view_conflict={_format_optional_float(texture_stats.get('detail_high_freq_view_conflict_abs_mean'), '.5f')} "
                f"view_conflict_raw={_format_optional_float(texture_stats.get('detail_high_freq_view_conflict_raw_abs_mean'), '.5f')} "
                f"view_conflict_gate={_format_optional_float(texture_stats.get('detail_high_freq_view_conflict_gate_mean'), '.4f')} "
                f"view_conflict_point={_format_optional_float(texture_stats.get('detail_high_freq_view_conflict_point_gate_mean'), '.4f')}"
            ),
            flush=True,
        )
    extra_local_debug = str(debug_info.get('texture_extra_local_debug', '') or '').strip()
    if extra_local_debug:
        print(f"[ClarityTexLocal] iter={schedule_iteration} {extra_local_debug}", flush=True)
    trunk_debug = str(debug_info.get('texture_trunk_debug', '') or '').strip()
    if trunk_debug:
        print(f"[ClarityTexTrunkDbg] iter={schedule_iteration} {trunk_debug}", flush=True)
    structure_debug = str(debug_info.get('texture_structure_debug', '') or '').strip()
    if structure_debug:
        print(f"[ClarityTexStruct] iter={schedule_iteration} {structure_debug}", flush=True)


def _grad_abs_stats(loss_term, target):
    if not torch.is_tensor(loss_term) or not torch.is_tensor(target):
        return None
    if not loss_term.requires_grad or not target.requires_grad:
        return None
    if float(loss_term.detach().abs().item()) <= 0.0:
        return None
    grad = torch.autograd.grad(loss_term, target, retain_graph=True, allow_unused=True)[0]
    if grad is None:
        return None
    grad_abs = grad.detach().abs()
    return {
        'mean': float(grad_abs.mean().item()),
        'max': float(grad_abs.max().item()),
    }


def _maybe_log_shoulder_local_grad_probe(
    config_opt,
    schedule_iteration,
    image,
    opacity_bce,
    global_image_term,
    local_image_term,
    global_boundary_term,
    local_boundary_term,
):
    if config_opt is None:
        return
    if not bool(config_opt.get('shoulder_local_loss_grad_probe_enable', False)):
        return

    interval = int(config_opt.get('shoulder_local_loss_grad_probe_interval', 0))
    warmup_iters = int(config_opt.get('shoulder_local_loss_grad_probe_warmup_iters', 0))
    if schedule_iteration < warmup_iters:
        return
    if interval > 0 and schedule_iteration % interval != 0:
        return

    global_image_stats = _grad_abs_stats(global_image_term, image)
    local_image_stats = _grad_abs_stats(local_image_term, image)
    global_boundary_stats = _grad_abs_stats(global_boundary_term, opacity_bce) if torch.is_tensor(opacity_bce) else None
    local_boundary_stats = _grad_abs_stats(local_boundary_term, opacity_bce) if torch.is_tensor(opacity_bce) else None

    def _mean(stats):
        return 0.0 if stats is None else float(stats.get('mean', 0.0))

    def _max(stats):
        return 0.0 if stats is None else float(stats.get('max', 0.0))

    global_image_mean = _mean(global_image_stats)
    local_image_mean = _mean(local_image_stats)
    global_boundary_mean = _mean(global_boundary_stats)
    local_boundary_mean = _mean(local_boundary_stats)

    print(
        (
            f"[ShoulderLocalGrad] iter={schedule_iteration} "
            f"img_local_mean={local_image_mean:.6e} img_global_mean={global_image_mean:.6e} "
            f"img_ratio={local_image_mean / max(global_image_mean, 1.0e-12):.4f} "
            f"img_local_max={_max(local_image_stats):.6e} img_global_max={_max(global_image_stats):.6e} "
            f"opacity_local_mean={local_boundary_mean:.6e} opacity_global_mean={global_boundary_mean:.6e} "
            f"opacity_ratio={local_boundary_mean / max(global_boundary_mean, 1.0e-12):.4f} "
            f"opacity_local_max={_max(local_boundary_stats):.6e} opacity_global_max={_max(global_boundary_stats):.6e}"
        ),
        flush=True,
    )


def _heuristic_shoulder_focus_region_mask(fg_mask, face_mask=None):
    valid = fg_mask[0] > 0.5
    coords = torch.nonzero(valid, as_tuple=False)
    if coords.numel() == 0:
        return torch.zeros_like(fg_mask)

    y_top = coords[:, 0].min().float()
    y_bottom = coords[:, 0].max().float()
    x_left = coords[:, 1].min().float()
    x_right = coords[:, 1].max().float()
    body_h = (y_bottom - y_top + 1.0).clamp_min(1.0)
    body_w = (x_right - x_left + 1.0).clamp_min(1.0)
    yy, xx = _make_pixel_grid(fg_mask)
    x_center = 0.5 * (x_left + x_right)
    x_rel = (xx - x_center) / (0.5 * body_w + 1e-6)
    y_rel = (yy - y_top) / (body_h + 1e-6)

    shoulder_band = (torch.abs(x_rel) > 0.14) & (torch.abs(x_rel) < 0.62) & (y_rel > 0.08) & (y_rel < 0.32)
    upper_arm_band = (torch.abs(x_rel) > 0.20) & (torch.abs(x_rel) < 0.70) & (y_rel > 0.16) & (y_rel < 0.46)
    outer_side = torch.abs(x_rel) > 0.28
    region = ((shoulder_band | (upper_arm_band & outer_side)).float() * fg_mask).clamp(0.0, 1.0)
    if face_mask is not None:
        region = region * (1.0 - 0.85 * face_mask)
    return region.clamp(0.0, 1.0)


def _joint_guided_shoulder_focus_region_mask(data, fg_mask, face_mask=None):
    posed_joints = getattr(data, 'posed_joints', None)
    projected, valid = _project_world_points_to_image(data, posed_joints)
    required = [
        NECK_JOINT_INDEX,
        LEFT_COLLAR_JOINT_INDEX,
        RIGHT_COLLAR_JOINT_INDEX,
        LEFT_SHOULDER_JOINT_INDEX,
        RIGHT_SHOULDER_JOINT_INDEX,
        LEFT_ELBOW_JOINT_INDEX,
        RIGHT_ELBOW_JOINT_INDEX,
    ]
    if projected is None or valid is None or any(idx >= projected.shape[0] for idx in required):
        return None
    if not bool(valid[required].all().item()):
        return None

    neck = projected[NECK_JOINT_INDEX]
    left_collar = projected[LEFT_COLLAR_JOINT_INDEX]
    right_collar = projected[RIGHT_COLLAR_JOINT_INDEX]
    left_shoulder = projected[LEFT_SHOULDER_JOINT_INDEX]
    right_shoulder = projected[RIGHT_SHOULDER_JOINT_INDEX]
    left_elbow = projected[LEFT_ELBOW_JOINT_INDEX]
    right_elbow = projected[RIGHT_ELBOW_JOINT_INDEX]

    yy, xx = _make_pixel_grid(fg_mask)
    shoulder_width = torch.norm(left_shoulder - right_shoulder).clamp_min(20.0)
    radius = torch.clamp(0.14 * shoulder_width, min=8.0, max=30.0)
    region = torch.zeros_like(fg_mask)

    side_specs = (
        (left_collar, left_shoulder, left_elbow, -1.0),
        (right_collar, right_shoulder, right_elbow, 1.0),
    )
    for collar, shoulder, elbow, direction in side_specs:
        upper_arm_mid = 0.68 * shoulder + 0.32 * elbow
        neck_side = 0.62 * collar + 0.38 * neck

        side_region = torch.zeros_like(fg_mask)
        for p0, p1, scale in (
            (collar, shoulder, 0.82),
            (shoulder, upper_arm_mid, 0.92),
        ):
            side_region = torch.maximum(side_region, _capsule_region_mask(fg_mask, p0, p1, radius * scale))

        for center, scale_x, scale_y in (
            (shoulder, 1.20, 0.95),
            (collar, 0.95, 0.85),
            (neck_side, 0.78, 0.78),
        ):
            side_region = torch.maximum(
                side_region,
                _ellipse_region_mask(fg_mask, center[0], center[1], radius * scale_x, radius * scale_y),
            )

        if direction < 0.0:
            outer_selector = (xx <= shoulder[0] + 0.30 * radius).float()
        else:
            outer_selector = (xx >= shoulder[0] - 0.30 * radius).float()
        y_top = torch.min(torch.stack([neck[1], collar[1], shoulder[1]])) - 1.2 * radius
        y_bottom = torch.max(torch.stack([collar[1], shoulder[1], upper_arm_mid[1]])) + 1.1 * radius
        vertical_band = ((yy >= y_top) & (yy <= y_bottom)).float()
        side_region = side_region * outer_selector * vertical_band * fg_mask
        region = torch.maximum(region, side_region)

    torso_core = _capsule_region_mask(fg_mask, left_collar, right_collar, 0.70 * radius)
    region = region * (1.0 - 0.60 * torso_core)
    if face_mask is not None:
        region = region * (1.0 - 0.85 * face_mask)
    region = region.clamp(0.0, 1.0) * fg_mask
    if region.sum().item() < 16:
        return None
    return region.clamp(0.0, 1.0)


def _shoulder_focus_region_mask_from_data(data, fg_mask, face_mask=None):
    joint_focus_mask = _joint_guided_shoulder_focus_region_mask(data, fg_mask, face_mask=face_mask)
    if joint_focus_mask is not None:
        return joint_focus_mask
    return _heuristic_shoulder_focus_region_mask(fg_mask, face_mask=face_mask)


def _heuristic_shoulder_collar_region_mask(fg_mask, face_mask=None):
    valid = fg_mask[0] > 0.5
    coords = torch.nonzero(valid, as_tuple=False)
    if coords.numel() == 0:
        return torch.zeros_like(fg_mask)

    y_top = coords[:, 0].min().float()
    y_bottom = coords[:, 0].max().float()
    x_left = coords[:, 1].min().float()
    x_right = coords[:, 1].max().float()
    body_h = (y_bottom - y_top + 1.0).clamp_min(1.0)
    body_w = (x_right - x_left + 1.0).clamp_min(1.0)
    yy, xx = _make_pixel_grid(fg_mask)
    x_center = 0.5 * (x_left + x_right)
    x_rel = (xx - x_center) / (0.5 * body_w + 1e-6)
    y_rel = (yy - y_top) / (body_h + 1e-6)

    outer_shoulder = (torch.abs(x_rel) > 0.18) & (torch.abs(x_rel) < 0.58) & (y_rel > 0.08) & (y_rel < 0.26)
    collar_side = (torch.abs(x_rel) > 0.10) & (torch.abs(x_rel) < 0.42) & (y_rel > 0.07) & (y_rel < 0.20)
    upper_selector = y_rel < 0.30
    region = ((outer_shoulder | collar_side).float() * upper_selector.float() * fg_mask).clamp(0.0, 1.0)

    torso_core = ((torch.abs(x_rel) < 0.16) & (y_rel > 0.10) & (y_rel < 0.26)).float() * fg_mask
    region = region * (1.0 - 0.82 * torso_core)
    if face_mask is not None:
        region = region * (1.0 - 0.88 * face_mask)
    return region.clamp(0.0, 1.0)


def _joint_guided_shoulder_collar_region_mask(data, fg_mask, face_mask=None):
    posed_joints = getattr(data, 'posed_joints', None)
    projected, valid = _project_world_points_to_image(data, posed_joints)
    required = [
        NECK_JOINT_INDEX,
        LEFT_COLLAR_JOINT_INDEX,
        RIGHT_COLLAR_JOINT_INDEX,
        LEFT_SHOULDER_JOINT_INDEX,
        RIGHT_SHOULDER_JOINT_INDEX,
    ]
    if projected is None or valid is None or any(idx >= projected.shape[0] for idx in required):
        return None
    if not bool(valid[required].all().item()):
        return None

    neck = projected[NECK_JOINT_INDEX]
    left_collar = projected[LEFT_COLLAR_JOINT_INDEX]
    right_collar = projected[RIGHT_COLLAR_JOINT_INDEX]
    left_shoulder = projected[LEFT_SHOULDER_JOINT_INDEX]
    right_shoulder = projected[RIGHT_SHOULDER_JOINT_INDEX]

    yy, xx = _make_pixel_grid(fg_mask)
    shoulder_width = torch.norm(left_shoulder - right_shoulder).clamp_min(20.0)
    radius = torch.clamp(0.11 * shoulder_width, min=6.0, max=18.0)
    region = torch.zeros_like(fg_mask)

    side_specs = (
        (left_collar, left_shoulder, -1.0),
        (right_collar, right_shoulder, 1.0),
    )
    for collar, shoulder, direction in side_specs:
        neck_side = 0.64 * collar + 0.36 * neck

        side_region = torch.zeros_like(fg_mask)
        for p0, p1, scale in (
            (neck_side, collar, 0.74),
            (collar, shoulder, 0.84),
        ):
            side_region = torch.maximum(side_region, _capsule_region_mask(fg_mask, p0, p1, radius * scale))

        for center, scale_x, scale_y in (
            (shoulder, 1.00, 0.76),
            (collar, 0.84, 0.68),
            (neck_side, 0.68, 0.62),
        ):
            side_region = torch.maximum(
                side_region,
                _ellipse_region_mask(fg_mask, center[0], center[1], radius * scale_x, radius * scale_y),
            )

        if direction < 0.0:
            outer_selector = (xx <= shoulder[0] + 0.14 * radius).float()
        else:
            outer_selector = (xx >= shoulder[0] - 0.14 * radius).float()
        y_top = torch.min(torch.stack([neck[1], collar[1], shoulder[1]])) - 1.0 * radius
        y_bottom = torch.max(torch.stack([collar[1], shoulder[1]])) + 0.85 * radius
        vertical_band = ((yy >= y_top) & (yy <= y_bottom)).float()
        side_region = side_region * outer_selector * vertical_band * fg_mask
        region = torch.maximum(region, side_region)

    torso_core = _capsule_region_mask(fg_mask, left_collar, right_collar, 0.54 * radius)
    neck_core = _ellipse_region_mask(fg_mask, neck[0], neck[1] + 0.22 * radius, 0.82 * radius, 0.72 * radius)
    region = region * (1.0 - 0.82 * torso_core) * (1.0 - 0.40 * neck_core)
    if face_mask is not None:
        region = region * (1.0 - 0.88 * face_mask)
    region = region.clamp(0.0, 1.0) * fg_mask
    if region.sum().item() < 10:
        return None
    return region.clamp(0.0, 1.0)


def _shoulder_collar_region_mask_from_data(data, fg_mask, face_mask=None):
    joint_collar_mask = _joint_guided_shoulder_collar_region_mask(data, fg_mask, face_mask=face_mask)
    if joint_collar_mask is not None:
        return joint_collar_mask
    return _heuristic_shoulder_collar_region_mask(fg_mask, face_mask=face_mask)


def _heuristic_upper_torso_region_mask(fg_mask, face_mask=None, shoulder_arm_mask=None):
    valid = fg_mask[0] > 0.5
    coords = torch.nonzero(valid, as_tuple=False)
    if coords.numel() == 0:
        return torch.zeros_like(fg_mask)

    y_top = coords[:, 0].min().float()
    y_bottom = coords[:, 0].max().float()
    x_left = coords[:, 1].min().float()
    x_right = coords[:, 1].max().float()
    body_h = (y_bottom - y_top + 1.0).clamp_min(1.0)
    body_w = (x_right - x_left + 1.0).clamp_min(1.0)
    yy, xx = _make_pixel_grid(fg_mask)
    x_center = 0.5 * (x_left + x_right)
    x_rel = (xx - x_center) / (0.5 * body_w + 1.0e-6)
    y_rel = (yy - y_top) / (body_h + 1.0e-6)

    torso_core = (torch.abs(x_rel) < 0.42) & (y_rel > 0.18) & (y_rel < 0.54)
    clavicle_cap = (torch.abs(x_rel) < 0.30) & (y_rel > 0.08) & (y_rel < 0.24)
    region = (torso_core | clavicle_cap).float() * fg_mask
    if face_mask is not None:
        region = region * (1.0 - 0.88 * face_mask)
    if shoulder_arm_mask is not None:
        region = region * (1.0 - 0.55 * shoulder_arm_mask)
    return region.clamp(0.0, 1.0)


def _joint_guided_upper_torso_region_mask(data, fg_mask, face_mask=None, shoulder_arm_mask=None):
    posed_joints = getattr(data, 'posed_joints', None)
    projected, valid = _project_world_points_to_image(data, posed_joints)
    required = [
        NECK_JOINT_INDEX,
        LEFT_COLLAR_JOINT_INDEX,
        RIGHT_COLLAR_JOINT_INDEX,
        LEFT_SHOULDER_JOINT_INDEX,
        RIGHT_SHOULDER_JOINT_INDEX,
    ]
    if projected is None or valid is None or any(idx >= projected.shape[0] for idx in required):
        return None
    if not bool(valid[required].all().item()):
        return None

    neck = projected[NECK_JOINT_INDEX]
    left_collar = projected[LEFT_COLLAR_JOINT_INDEX]
    right_collar = projected[RIGHT_COLLAR_JOINT_INDEX]
    left_shoulder = projected[LEFT_SHOULDER_JOINT_INDEX]
    right_shoulder = projected[RIGHT_SHOULDER_JOINT_INDEX]
    center = 0.5 * (left_collar + right_collar)
    shoulder_width = torch.norm(left_shoulder - right_shoulder).clamp_min(20.0)
    radius = torch.clamp(0.18 * shoulder_width, min=9.0, max=28.0)

    valid_fg = fg_mask[0] > 0.5
    coords = torch.nonzero(valid_fg, as_tuple=False)
    if coords.numel() == 0:
        return None
    body_bottom = coords[:, 0].max().float()

    yy, xx = _make_pixel_grid(fg_mask)
    region = torch.zeros_like(fg_mask)
    region = torch.maximum(region, _capsule_region_mask(fg_mask, left_collar, right_collar, 0.88 * radius))
    region = torch.maximum(
        region,
        _ellipse_region_mask(
            fg_mask,
            center[0],
            center[1] + 1.25 * radius,
            0.85 * shoulder_width,
            2.10 * radius,
        ),
    )
    region = torch.maximum(
        region,
        _ellipse_region_mask(
            fg_mask,
            center[0],
            center[1] + 2.40 * radius,
            0.62 * shoulder_width,
            1.55 * radius,
        ),
    )

    y_top = torch.min(torch.stack([neck[1], left_collar[1], right_collar[1]])) - 0.5 * radius
    y_bottom = torch.minimum(center[1] + 4.2 * radius, body_bottom - 0.18 * radius)
    vertical_band = ((yy >= y_top) & (yy <= y_bottom)).float()
    center_band = (torch.abs(xx - center[0]) <= 0.82 * shoulder_width).float()
    region = region * vertical_band * center_band * fg_mask
    if face_mask is not None:
        region = region * (1.0 - 0.90 * face_mask)
    if shoulder_arm_mask is not None:
        region = region * (1.0 - 0.48 * shoulder_arm_mask)
    region = region.clamp(0.0, 1.0)
    if region.sum().item() < 18:
        return None
    return region


def _resolve_upper_torso_region_from_data(data, fg_mask, face_mask=None, shoulder_arm_mask=None, config_opt=None):
    region_source = 'parser_prefer' if config_opt is None else str(config_opt.get('upper_torso_region_source', 'parser_prefer'))
    parser_dilate = 7 if config_opt is None else int(config_opt.get('upper_torso_region_parser_dilate', 7))

    parser_upper_mask = _parser_region_mask(data, fg_mask, PARSER_UPPER_TORSO_LABELS)
    if parser_upper_mask is not None:
        if parser_dilate > 1:
            parser_upper_mask = _binary_dilate(parser_upper_mask, parser_dilate).clamp(0.0, 1.0)
        parser_upper_mask = parser_upper_mask * fg_mask
        if face_mask is not None:
            parser_upper_mask = parser_upper_mask * (1.0 - 0.88 * face_mask)
        if shoulder_arm_mask is not None:
            parser_upper_mask = parser_upper_mask * (1.0 - 0.45 * shoulder_arm_mask)

    joint_upper_mask = None
    if region_source not in ('parser_only', 'heuristic_only'):
        joint_upper_mask = _joint_guided_upper_torso_region_mask(
            data,
            fg_mask,
            face_mask=face_mask,
            shoulder_arm_mask=shoulder_arm_mask,
        )

    heuristic_upper_mask = None

    def _get_heuristic_upper_mask():
        nonlocal heuristic_upper_mask
        if heuristic_upper_mask is None:
            heuristic_upper_mask = _heuristic_upper_torso_region_mask(
                fg_mask,
                face_mask=face_mask,
                shoulder_arm_mask=shoulder_arm_mask,
            )
        return heuristic_upper_mask

    if region_source == 'parser_only':
        if parser_upper_mask is not None:
            region = parser_upper_mask
            actual_source = 'parser_only'
        else:
            region = torch.zeros_like(fg_mask)
            actual_source = 'parser_missing_empty'
    elif region_source == 'joint_only':
        if joint_upper_mask is not None:
            region = joint_upper_mask
            actual_source = 'joint_only'
        else:
            region = _get_heuristic_upper_mask()
            actual_source = 'joint_fallback_heuristic'
    elif region_source == 'heuristic_only':
        region = _get_heuristic_upper_mask()
        actual_source = 'heuristic_only'
    elif region_source == 'union':
        if parser_upper_mask is not None and joint_upper_mask is not None:
            region = torch.maximum(parser_upper_mask, joint_upper_mask).clamp(0.0, 1.0) * fg_mask
            actual_source = 'union_parser_joint'
        elif parser_upper_mask is not None:
            region = parser_upper_mask
            actual_source = 'union_parser'
        elif joint_upper_mask is not None:
            region = joint_upper_mask
            actual_source = 'union_joint'
        else:
            region = _get_heuristic_upper_mask()
            actual_source = 'union_heuristic'
    else:
        if parser_upper_mask is not None:
            region = parser_upper_mask
            actual_source = 'parser_prefer_parser'
        elif joint_upper_mask is not None:
            region = joint_upper_mask
            actual_source = 'parser_prefer_joint_fallback'
        else:
            region = _get_heuristic_upper_mask()
            actual_source = 'parser_prefer_heuristic_fallback'

    meta = {
        'requested_source': region_source,
        'actual_source': actual_source,
        'has_parser': parser_upper_mask is not None,
        'has_joint': joint_upper_mask is not None,
        'region_pixels': float(region.sum().item()),
        'parser_pixels': float(parser_upper_mask.sum().item()) if parser_upper_mask is not None else 0.0,
        'joint_pixels': float(joint_upper_mask.sum().item()) if joint_upper_mask is not None else 0.0,
        'heuristic_pixels': float(heuristic_upper_mask.sum().item()) if heuristic_upper_mask is not None else 0.0,
    }
    return region.clamp(0.0, 1.0), meta


def _heuristic_waist_region_mask(fg_mask):
    valid = fg_mask[0] > 0.5
    coords = torch.nonzero(valid, as_tuple=False)
    if coords.numel() == 0:
        return torch.zeros_like(fg_mask)

    y_top = coords[:, 0].min().float()
    y_bottom = coords[:, 0].max().float()
    x_left = coords[:, 1].min().float()
    x_right = coords[:, 1].max().float()
    body_h = (y_bottom - y_top + 1.0).clamp_min(1.0)
    body_w = (x_right - x_left + 1.0).clamp_min(1.0)
    yy, xx = _make_pixel_grid(fg_mask)
    x_center = 0.5 * (x_left + x_right)
    x_rel = (xx - x_center) / (0.5 * body_w + 1.0e-6)
    y_rel = (yy - y_top) / (body_h + 1.0e-6)

    center_band = torch.abs(x_rel) < 0.44
    waist_band = (y_rel > 0.46) & (y_rel < 0.70)
    return (center_band & waist_band).float() * fg_mask


def _waist_top_band_mask(lower_cloth_mask, fg_mask, config_opt=None):
    if lower_cloth_mask is None or not torch.is_tensor(lower_cloth_mask):
        return None

    coords = torch.nonzero(lower_cloth_mask[0] > 0.5, as_tuple=False)
    if coords.numel() == 0:
        return None

    y_top = coords[:, 0].min().float()
    y_bottom = coords[:, 0].max().float()
    x_left = coords[:, 1].min().float()
    x_right = coords[:, 1].max().float()
    lower_h = (y_bottom - y_top + 1.0).clamp_min(1.0)
    band_ratio = 0.30 if config_opt is None else float(config_opt.get('waist_region_band_height_ratio', 0.30))
    band_min = 10 if config_opt is None else int(config_opt.get('waist_region_band_height_min_pixels', 10))
    band_max = 28 if config_opt is None else int(config_opt.get('waist_region_band_height_max_pixels', 28))
    top_pad = 4 if config_opt is None else int(config_opt.get('waist_region_top_pad_pixels', 4))
    bottom_pad = 3 if config_opt is None else int(config_opt.get('waist_region_bottom_pad_pixels', 3))
    center_focus_enable = False if config_opt is None else bool(config_opt.get('waist_region_center_focus_enable', False))
    center_width_ratio = 0.62 if config_opt is None else float(config_opt.get('waist_region_center_width_ratio', 0.62))
    final_dilate = 0 if config_opt is None else int(config_opt.get('waist_region_dilate', 0))

    yy, xx = _make_pixel_grid(fg_mask)
    band_height = torch.clamp(band_ratio * lower_h, min=float(band_min), max=float(band_max))
    top_band = ((yy >= (y_top - float(top_pad))) & (yy <= (y_top + band_height + float(bottom_pad)))).float()
    region = lower_cloth_mask * top_band

    if center_focus_enable:
        x_center = 0.5 * (x_left + x_right)
        half_width = 0.5 * center_width_ratio * (x_right - x_left + 1.0)
        center_band = ((xx >= (x_center - half_width)) & (xx <= (x_center + half_width))).float()
        region = region * center_band

    if final_dilate > 1:
        region = _binary_dilate(region, final_dilate).clamp(0.0, 1.0)
    region = region.clamp(0.0, 1.0) * fg_mask
    if region.sum().item() < 8:
        return None
    return region


def _resolve_waist_region_from_data(data, fg_mask, config_opt=None):
    region_source = 'parser_prefer' if config_opt is None else str(config_opt.get('waist_region_source', 'parser_prefer'))
    region_mode = 'top_band' if config_opt is None else str(config_opt.get('waist_region_mode', 'top_band'))
    parser_dilate = 5 if config_opt is None else int(config_opt.get('waist_region_parser_dilate', 5))

    parser_lower_mask = _parser_region_mask(data, fg_mask, PARSER_LOWER_CLOTH_LABELS)
    if parser_lower_mask is not None and parser_dilate > 1:
        parser_lower_mask = _binary_dilate(parser_lower_mask, parser_dilate).clamp(0.0, 1.0) * fg_mask

    def _parser_region():
        if parser_lower_mask is None:
            return None
        if region_mode == 'lower_cloth':
            return parser_lower_mask
        return _waist_top_band_mask(parser_lower_mask, fg_mask, config_opt=config_opt)

    if region_source == 'parser_only':
        region = _parser_region()
        actual_source = 'parser_only' if region is not None else 'parser_missing_empty'
    elif region_source == 'heuristic_only':
        region = _heuristic_waist_region_mask(fg_mask)
        actual_source = 'heuristic_only'
    else:
        region = _parser_region()
        if region is not None:
            actual_source = 'parser_prefer_parser'
        else:
            region = _heuristic_waist_region_mask(fg_mask)
            actual_source = 'parser_prefer_heuristic_fallback'

    if region is None:
        region = torch.zeros_like(fg_mask)

    meta = {
        'requested_source': region_source,
        'actual_source': actual_source,
        'region_mode': region_mode,
        'has_parser': parser_lower_mask is not None,
        'parser_pixels': float(parser_lower_mask.sum().item()) if parser_lower_mask is not None else 0.0,
        'region_pixels': float(region.sum().item()),
    }
    return region.clamp(0.0, 1.0), meta


def _masked_gradient_l1_loss(image, gt_image, mask):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    mask = mask.to(device=image.device, dtype=image.dtype)

    diff_x = image[:, :, 1:] - image[:, :, :-1]
    gt_diff_x = gt_image[:, :, 1:] - gt_image[:, :, :-1]
    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    if mask_x.shape[0] == 1 and diff_x.shape[0] != 1:
        mask_x = mask_x.expand(diff_x.shape[0], -1, -1)

    diff_y = image[:, 1:, :] - image[:, :-1, :]
    gt_diff_y = gt_image[:, 1:, :] - gt_image[:, :-1, :]
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]
    if mask_y.shape[0] == 1 and diff_y.shape[0] != 1:
        mask_y = mask_y.expand(diff_y.shape[0], -1, -1)

    norm = mask_x.sum() + mask_y.sum()
    norm = norm.clamp_min(1.0)
    loss_x = (torch.abs(diff_x - gt_diff_x) * mask_x).sum()
    loss_y = (torch.abs(diff_y - gt_diff_y) * mask_y).sum()
    return (loss_x + loss_y) / norm


def _resize_chw_tensor(tensor, size, mode):
    kwargs = {}
    if mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
        kwargs['align_corners'] = False
    return F.interpolate(tensor.unsqueeze(0), size=size, mode=mode, **kwargs).squeeze(0)


def _downsample_for_detail_loss(image, gt_image, mask, scale):
    if scale <= 1:
        return image, gt_image, mask

    height, width = image.shape[-2:]
    target_h = max(height // scale, 1)
    target_w = max(width // scale, 1)
    target_size = (target_h, target_w)
    image_ds = _resize_chw_tensor(image, target_size, mode='area')
    gt_image_ds = _resize_chw_tensor(gt_image, target_size, mode='area')

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    mask_ds = _resize_chw_tensor(mask.to(device=image.device, dtype=image.dtype), target_size, mode='bilinear')
    return image_ds, gt_image_ds, mask_ds.clamp(0.0, 1.0)


def _masked_highpass_l1_loss(image, gt_image, mask, blur_kernel=5):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    mask = mask.to(device=image.device, dtype=image.dtype)

    kernel = max(int(blur_kernel), 1)
    kernel = min(kernel, min(image.shape[-2:]))
    if kernel % 2 == 0:
        kernel = max(kernel - 1, 1)
    if kernel <= 1:
        return image.new_tensor(0.0)

    pad = kernel // 2
    image_4d = image.unsqueeze(0)
    gt_image_4d = gt_image.unsqueeze(0)
    mask_4d = mask.unsqueeze(0)
    image_blur = F.avg_pool2d(image_4d, kernel_size=kernel, stride=1, padding=pad)
    gt_image_blur = F.avg_pool2d(gt_image_4d, kernel_size=kernel, stride=1, padding=pad)
    image_high = image_4d - image_blur
    gt_image_high = gt_image_4d - gt_image_blur
    norm = mask_4d.sum().clamp_min(1.0)
    return (torch.abs(image_high - gt_image_high) * mask_4d).sum() / norm


def _resolve_detail_scales(value, default=(1, 2, 4)):
    if value is None:
        return list(default)
    if isinstance(value, (int, float)):
        return [max(int(value), 1)]
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, list):
        return list(default)

    scales = []
    for item in value:
        scale = max(int(item), 1)
        if scale not in scales:
            scales.append(scale)
    return scales if scales else list(default)


def _masked_multiscale_detail_loss(
    image,
    gt_image,
    mask,
    scales=(1, 2, 4),
    blur_kernel=5,
    scale_decay=0.5,
    gradient_mix=0.25,
):
    scales = _resolve_detail_scales(scales)
    total_loss = image.new_tensor(0.0)
    total_weight = 0.0

    for level_idx, scale in enumerate(scales):
        image_level, gt_image_level, mask_level = _downsample_for_detail_loss(image, gt_image, mask, scale)
        if min(image_level.shape[-2:]) < 2:
            continue

        level_weight = float(scale_decay) ** level_idx
        level_loss = _masked_highpass_l1_loss(
            image_level,
            gt_image_level,
            mask_level,
            blur_kernel=blur_kernel,
        )
        if gradient_mix > 0.0:
            level_loss = level_loss + float(gradient_mix) * _masked_gradient_l1_loss(
                image_level,
                gt_image_level,
                mask_level,
            )
        total_loss = total_loss + level_weight * level_loss
        total_weight += level_weight

    if total_weight <= 0.0:
        return image.new_tensor(0.0)
    return total_loss / total_weight


def _luma_from_chw(image):
    weights = image.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    return (image * weights).sum(dim=0, keepdim=True)


def _normalize_blur_kernel(kernel, spatial_shape):
    kernel = max(int(kernel), 1)
    kernel = min(kernel, int(min(spatial_shape)))
    if kernel % 2 == 0:
        kernel = max(kernel - 1, 1)
    return kernel


def _avg_blur_chw(image, kernel):
    kernel = _normalize_blur_kernel(kernel, image.shape[-2:])
    if kernel <= 1:
        return image
    pad = kernel // 2
    return F.avg_pool2d(image.unsqueeze(0), kernel_size=kernel, stride=1, padding=pad).squeeze(0)


def _luma_dog_response(image, small_kernel=3, large_kernel=9):
    luma = _luma_from_chw(image)
    small_kernel = _normalize_blur_kernel(small_kernel, luma.shape[-2:])
    large_kernel = _normalize_blur_kernel(max(int(large_kernel), small_kernel + 2), luma.shape[-2:])
    if large_kernel <= small_kernel:
        large_kernel = _normalize_blur_kernel(small_kernel + 2, luma.shape[-2:])
    blur_small = _avg_blur_chw(luma, small_kernel)
    blur_large = _avg_blur_chw(luma, large_kernel)
    return blur_small - blur_large


def _masked_normalize_map(values, mask):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    mask = mask.to(device=values.device, dtype=values.dtype)
    weighted = values.abs() * mask
    max_value = weighted.max()
    if float(max_value.item()) <= 1.0e-6:
        return torch.zeros_like(weighted)
    return (weighted / max_value.clamp_min(1.0e-6)).clamp(0.0, 1.0)


def _detail_focus_map(gt_response, error_response, mask, error_mix=0.0, focus_power=1.0):
    gt_focus = _masked_normalize_map(gt_response, mask)
    error_mix = float(min(max(error_mix, 0.0), 1.0))
    if error_mix > 0.0 and error_response is not None:
        error_focus = _masked_normalize_map(error_response, mask)
        focus = (1.0 - error_mix) * gt_focus + error_mix * error_focus
    else:
        focus = gt_focus
    if focus_power != 1.0:
        focus = focus.pow(float(focus_power))
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    return (focus * mask.to(device=focus.device, dtype=focus.dtype)).clamp(0.0, 1.0)


def _masked_weighted_luma_dog_loss(
    image,
    gt_image,
    mask,
    small_kernel=3,
    large_kernel=9,
    base_weight=0.25,
    focus_scale=2.0,
    focus_power=1.5,
    error_mix=0.0,
):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    mask = mask.to(device=image.device, dtype=image.dtype)
    if float(mask.sum().item()) <= 0.0:
        return image.new_tensor(0.0)

    pred_response = _luma_dog_response(image, small_kernel=small_kernel, large_kernel=large_kernel)
    gt_response = _luma_dog_response(gt_image, small_kernel=small_kernel, large_kernel=large_kernel)
    error_response = (image.detach() - gt_image.detach()).abs().mean(dim=0, keepdim=True)
    focus_map = _detail_focus_map(
        gt_response.detach(),
        error_response,
        mask,
        error_mix=error_mix,
        focus_power=focus_power,
    )
    weight_map = (float(base_weight) + float(focus_scale) * focus_map).clamp_min(0.0) * mask
    norm = weight_map.sum().clamp_min(1.0)
    return (torch.abs(pred_response - gt_response) * weight_map).sum() / norm


def _masked_multiscale_weighted_luma_dog_loss(
    image,
    gt_image,
    mask,
    scales=(1, 2, 4),
    scale_decay=0.5,
    small_kernel=3,
    large_kernel=9,
    base_weight=0.25,
    focus_scale=2.0,
    focus_power=1.5,
    error_mix=0.0,
):
    scales = _resolve_detail_scales(scales)
    total_loss = image.new_tensor(0.0)
    total_weight = 0.0

    for level_idx, scale in enumerate(scales):
        image_level, gt_image_level, mask_level = _downsample_for_detail_loss(image, gt_image, mask, scale)
        if min(image_level.shape[-2:]) < 2:
            continue

        level_weight = float(scale_decay) ** level_idx
        level_loss = _masked_weighted_luma_dog_loss(
            image_level,
            gt_image_level,
            mask_level,
            small_kernel=small_kernel,
            large_kernel=large_kernel,
            base_weight=base_weight,
            focus_scale=focus_scale,
            focus_power=focus_power,
            error_mix=error_mix,
        )
        total_loss = total_loss + level_weight * level_loss
        total_weight += level_weight

    if total_weight <= 0.0:
        return image.new_tensor(0.0)
    return total_loss / total_weight


def _crop_centered_patch(image, center_y, center_x, patch_size):
    patch_size = max(int(patch_size), 1)
    height, width = image.shape[-2:]
    patch_h = min(patch_size, height)
    patch_w = min(patch_size, width)
    y0 = min(max(int(center_y) - patch_h // 2, 0), max(height - patch_h, 0))
    x0 = min(max(int(center_x) - patch_w // 2, 0), max(width - patch_w, 0))
    return image[:, y0:y0 + patch_h, x0:x0 + patch_w]


def _select_salient_patch_centers(score_map, mask, topk=1, suppress_radius=0):
    if score_map.dim() == 3:
        score_map = score_map[0]
    if mask.dim() == 3:
        mask = mask[0]
    score_map = score_map.detach().clone()
    valid_mask = (mask > 0.0).to(device=score_map.device)
    if not bool(valid_mask.any().item()):
        return []

    score_map = score_map * valid_mask.float()
    suppress_radius = max(int(suppress_radius), 0)
    centers = []
    height, width = score_map.shape[-2:]
    for _ in range(max(int(topk), 0)):
        flat_idx = int(torch.argmax(score_map.reshape(-1)).item())
        peak = float(score_map.reshape(-1)[flat_idx].item())
        if peak <= 0.0:
            break
        center_y = flat_idx // width
        center_x = flat_idx % width
        centers.append((center_y, center_x))
        y0 = max(center_y - suppress_radius, 0)
        y1 = min(center_y + suppress_radius + 1, height)
        x0 = max(center_x - suppress_radius, 0)
        x1 = min(center_x + suppress_radius + 1, width)
        score_map[y0:y1, x0:x1] = 0.0
    return centers


def _masked_hard_patch_perceptual_loss(
    image,
    gt_image,
    mask,
    loss_fn_vgg,
    patch_size=64,
    topk=1,
    min_size=32,
    suppress_radius=0,
    masked=True,
    small_kernel=3,
    large_kernel=9,
    focus_power=1.5,
    error_mix=0.5,
    min_mask_coverage=0.0,
):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    mask = mask.to(device=image.device, dtype=image.dtype)
    if float(mask.sum().item()) <= 0.0:
        return image.new_tensor(0.0), 0

    gt_response = _luma_dog_response(gt_image.detach(), small_kernel=small_kernel, large_kernel=large_kernel)
    error_response = (image.detach() - gt_image.detach()).abs().mean(dim=0, keepdim=True)
    focus_map = _detail_focus_map(
        gt_response,
        error_response,
        mask,
        error_mix=error_mix,
        focus_power=focus_power,
    )
    centers = _select_salient_patch_centers(
        focus_map,
        mask,
        topk=topk,
        suppress_radius=suppress_radius if suppress_radius > 0 else patch_size // 2,
    )
    if len(centers) <= 0:
        return image.new_tensor(0.0), 0

    total_loss = image.new_tensor(0.0)
    valid_count = 0
    for center_y, center_x in centers:
        pred_patch = _crop_centered_patch(image, center_y, center_x, patch_size)
        gt_patch = _crop_centered_patch(gt_image, center_y, center_x, patch_size)
        if masked:
            mask_patch = _crop_centered_patch(mask, center_y, center_x, patch_size)
            min_mask_coverage = float(min_mask_coverage)
            if min_mask_coverage > 0.0:
                coverage = float((mask_patch > 0.05).float().mean().item())
                if coverage < min_mask_coverage:
                    continue
            pred_patch = pred_patch * mask_patch
            gt_patch = gt_patch * mask_patch
        pred_patch, gt_patch = _ensure_lpips_min_size(pred_patch, gt_patch, min_size=min_size)
        if pred_patch is None:
            continue
        total_loss = total_loss + loss_fn_vgg(
            pred_patch.unsqueeze(0),
            gt_patch.unsqueeze(0),
            normalize=True,
        ).mean()
        valid_count += 1

    if valid_count <= 0:
        return image.new_tensor(0.0), 0
    return total_loss / float(valid_count), valid_count


def _zero_nonfinite_grads(optimizer):
    had_nonfinite = False
    for group in optimizer.param_groups:
        for param in group.get('params', []):
            if param.grad is None:
                continue
            finite_mask = torch.isfinite(param.grad)
            if torch.all(finite_mask):
                continue
            param.grad = torch.where(finite_mask, param.grad, torch.zeros_like(param.grad))
            had_nonfinite = True
    return had_nonfinite


def _snapshot_hydra_run(config, output_dir):
    hydra_dir = os.path.join(output_dir, '.hydra')
    os.makedirs(hydra_dir, exist_ok=True)

    runtime_dir = None
    try:
        runtime_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        runtime_dir = None

    copied = False
    if runtime_dir:
        src_hydra_dir = os.path.join(runtime_dir, '.hydra')
        for name in ('config.yaml', 'hydra.yaml', 'overrides.yaml'):
            src = os.path.join(src_hydra_dir, name)
            dst = os.path.join(hydra_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied = True

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

def training(config):
    model = config.model
    dataset = config.dataset
    opt = config.opt
    pipe = config.pipeline
    testing_iterations = config.test_iterations
    testing_interval = config.test_interval
    saving_iterations = config.save_iterations
    checkpoint_iterations = config.checkpoint_iterations
    checkpoint = config.start_checkpoint
    debug_from = config.debug_from

    # define lpips
    lpips_type = config.opt.get('lpips_type', 'vgg')
    loss_fn_vgg = lpips.LPIPS(net=lpips_type).cuda() # for training
    evaluator = PSEvaluator() if dataset.name == 'people_snapshot' else Evaluator()

    first_iter = 0
    gaussians = GaussianModel(model.gaussian)
    scene = Scene(config, gaussians, config.exp_dir)
    scene.train()

    gaussians.training_setup(opt)
    loaded_iteration = 0
    if checkpoint:
        loaded_iteration = scene.load_checkpoint(checkpoint) or 0
        gaussians.training_setup(opt)
        gaussians.ensure_boundary_state_matches_points(verbose=True)
        print(f"Loaded checkpoint {checkpoint} (iteration {loaded_iteration})")

    resume_cfg = config.get('resume', None)
    disable_densify_on_resume = bool(resume_cfg.get('disable_densify_on_resume', False)) if resume_cfg else False
    disable_opacity_reset_on_resume = bool(resume_cfg.get('disable_opacity_reset_on_resume', False)) if resume_cfg else False
    use_checkpoint_iteration_as_offset = bool(resume_cfg.get('use_checkpoint_iteration_as_offset', False)) if resume_cfg else False
    iteration_offset = loaded_iteration if checkpoint and use_checkpoint_iteration_as_offset else 0
    allow_densify = not (checkpoint and disable_densify_on_resume)
    allow_opacity_reset = not (checkpoint and disable_opacity_reset_on_resume)
    if checkpoint and resume_cfg and bool(resume_cfg.get('require_no_densify_on_resume', False)):
        densify_until_iter = int(opt.get('densify_until_iter', 0))
        percent_dense = float(opt.get('percent_dense', 0.0))
        if allow_densify or densify_until_iter > 0 or percent_dense > 0.0:
            raise ValueError(
                'resume.require_no_densify_on_resume=true requires '
                'resume.disable_densify_on_resume=true, opt.densify_until_iter=0, '
                'and opt.percent_dense=0.0.'
            )
    if checkpoint and not allow_densify:
        print('Resume safety: densification disabled for checkpoint finetune.')
    if checkpoint and not allow_opacity_reset:
        print('Resume safety: opacity resets disabled for checkpoint finetune.')
    if iteration_offset > 0:
        print(f'Resume safety: using iteration offset {iteration_offset} for schedules and checkpoint naming.')
    if checkpoint and resume_cfg and bool(resume_cfg.get('clear_boundary_tags_on_resume', False)):
        scene.gaussians.clear_boundary_tags()
        print('Resume safety: cleared boundary tags for resume retagging.')
    if checkpoint and resume_cfg and bool(resume_cfg.get('clear_binding_state_on_resume', False)):
        _clear_scene_binding_state(scene)
        print('Resume safety: cleared binding state for resume rebind.')

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    boundary_live_score_cache = {}
    _validate_boundary_live_score_cache_config(config)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    data_stack = None
    train_sampler = _build_train_sampler(scene.train_dataset, opt)
    ema_loss_for_log = 0.0
    best_metric_name = str(config.get('best_metric', 'psnr_fg'))
    best_metric_mode = str(config.get('best_metric_mode', 'max')).lower()
    best_metric_source = str(config.get('best_metric_source', 'auto')).lower()
    if best_metric_mode not in ('max', 'min'):
        raise ValueError(f'Unsupported best_metric_mode: {best_metric_mode}')
    best_test_score = float('-inf') if best_metric_mode == 'max' else float('inf')
    diagnostic_validate_at_start = bool(opt.get('diagnostic_validate_at_start', False))
    diagnostic_no_backward = bool(opt.get('diagnostic_no_backward', False))
    diagnostic_skip_optimizer_step = bool(opt.get('diagnostic_skip_optimizer_step', False))
    diagnostic_log_interval = int(opt.get('diagnostic_log_interval', 100))
    if diagnostic_validate_at_start:
        raw_interval_iteration = opt.get('diagnostic_validate_interval_iteration', '__default__')
        if raw_interval_iteration == '__default__':
            diagnostic_interval_iteration = 0
        elif str(raw_interval_iteration).lower() in ('global', 'none', 'null'):
            diagnostic_interval_iteration = None
        else:
            diagnostic_interval_iteration = int(raw_interval_iteration)
        diagnostic_check_iteration = (
            iteration_offset if diagnostic_interval_iteration is None else diagnostic_interval_iteration
        )
        print('[TrainDiagnostic] running validation immediately after checkpoint load before train loop.')
        diagnostic_metrics = validation(
            iteration_offset,
            [diagnostic_check_iteration],
            0,
            scene,
            evaluator,
            (pipe, background),
            interval_iteration=diagnostic_interval_iteration,
        )
        metrics_path = str(opt.get('diagnostic_validation_metrics_path', '') or '')
        if metrics_path and diagnostic_metrics is not None:
            metrics_dir = os.path.dirname(metrics_path)
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
            with open(metrics_path, 'w') as f:
                json.dump(diagnostic_metrics, f, indent=2)
        if bool(opt.get('diagnostic_exit_after_start_validation', False)):
            print('[TrainDiagnostic] diagnostic_exit_after_start_validation=true: exiting before train loop.')
            return
    if diagnostic_no_backward:
        print('[TrainDiagnostic] diagnostic_no_backward=true: train loop will forward/evaluate but skip backward.')
    if diagnostic_skip_optimizer_step:
        print('[TrainDiagnostic] diagnostic_skip_optimizer_step=true: train loop will skip scene.optimize().')
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        local_iteration = iteration
        schedule_iteration = iteration + iteration_offset

        iter_start.record()

        gaussians.update_learning_rate(schedule_iteration)
        _maybe_refresh_resume_binding(scene, config, checkpoint is not None, local_iteration, schedule_iteration)
        _set_texture_schedule_context(
            scene,
            local_iteration=local_iteration,
            schedule_iteration=schedule_iteration,
        )

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if schedule_iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a training data point.
        if train_sampler is not None:
            data_idx = train_sampler.next_index()
            train_sampler.maybe_log(schedule_iteration)
            gradient_accumulation_scale = float(train_sampler.gradient_accumulation_scale())
            accumulation_should_step = bool(train_sampler.should_optimizer_step())
        else:
            if not data_stack:
                data_stack = list(range(len(scene.train_dataset)))
            data_idx = data_stack.pop(randint(0, len(data_stack)-1))
            gradient_accumulation_scale = 1.0
            accumulation_should_step = True
        data = scene.train_dataset[data_idx]

        # Render
        if (schedule_iteration - 1) == debug_from:
            pipe.debug = True

        lambda_mask = C(schedule_iteration, config.opt.lambda_mask)
        lambda_mask_boundary = C(schedule_iteration, config.opt.get('lambda_mask_boundary', 0.0))
        lambda_mask_boundary_hard = C(schedule_iteration, config.opt.get('lambda_mask_boundary_hard', 0.0))
        lambda_mask_shoulder_arm_boundary_hard = C(schedule_iteration, config.opt.get('lambda_mask_shoulder_arm_boundary_hard', 0.0))
        lambda_mask_upper_torso_boundary_hard = C(schedule_iteration, config.opt.get('lambda_mask_upper_torso_boundary_hard', 0.0))
        lambda_mask_shoulder_arm_region_hard = C(schedule_iteration, config.opt.get('lambda_mask_shoulder_arm_region_hard', 0.0))
        lambda_mask_shoulder_arm_disagreement_hard = C(schedule_iteration, config.opt.get('lambda_mask_shoulder_arm_disagreement_hard', 0.0))
        lambda_mask_shoulder_focus_small_fp_hard = C(schedule_iteration, config.opt.get('lambda_mask_shoulder_focus_small_fp_hard', 0.0))
        lambda_mask_shoulder_focus_small_fn_hard = C(schedule_iteration, config.opt.get('lambda_mask_shoulder_focus_small_fn_hard', 0.0))
        lambda_silhouette_outer = C(schedule_iteration, config.opt.get('lambda_silhouette_outer', 0.0))
        lambda_silhouette_outer_shell = C(schedule_iteration, config.opt.get('lambda_silhouette_outer_shell', 0.0))
        lambda_silhouette_head_outer_shell = C(schedule_iteration, config.opt.get('lambda_silhouette_head_outer_shell', 0.0))
        lambda_silhouette_shoulder_arm_outer_shell = C(schedule_iteration, config.opt.get('lambda_silhouette_shoulder_arm_outer_shell', 0.0))
        lambda_silhouette_upper_torso_outer_shell = C(schedule_iteration, config.opt.get('lambda_silhouette_upper_torso_outer_shell', 0.0))
        lambda_silhouette_outer_spike = C(schedule_iteration, config.opt.get('lambda_silhouette_outer_spike', 0.0))
        lambda_silhouette_outer_fragment = C(schedule_iteration, config.opt.get('lambda_silhouette_outer_fragment', 0.0))
        lambda_silhouette_outer_bead = C(schedule_iteration, config.opt.get('lambda_silhouette_outer_bead', 0.0))
        lambda_silhouette_outer_chain = C(schedule_iteration, config.opt.get('lambda_silhouette_outer_chain', 0.0))
        lambda_silhouette_arm_stipple = C(schedule_iteration, config.opt.get('lambda_silhouette_arm_stipple', 0.0))
        lambda_silhouette_arm_tail = C(schedule_iteration, config.opt.get('lambda_silhouette_arm_tail', 0.0))
        lambda_silhouette_arm_fringe = C(schedule_iteration, config.opt.get('lambda_silhouette_arm_fringe', 0.0))
        lambda_silhouette_arm_attached_fragment = C(schedule_iteration, config.opt.get('lambda_silhouette_arm_attached_fragment', 0.0))
        lambda_silhouette_shoulder_attached_fragment = C(schedule_iteration, config.opt.get('lambda_silhouette_shoulder_attached_fragment', 0.0))
        lambda_silhouette_arm_notch = C(schedule_iteration, config.opt.get('lambda_silhouette_arm_notch', 0.0))
        lambda_silhouette_arm_hole = C(schedule_iteration, config.opt.get('lambda_silhouette_arm_hole', 0.0))
        lambda_silhouette_arm_gap = C(schedule_iteration, config.opt.get('lambda_silhouette_arm_gap', 0.0))
        lambda_silhouette_shoulder_bead = C(schedule_iteration, config.opt.get('lambda_silhouette_shoulder_bead', 0.0))
        lambda_silhouette_shoulder_chain = C(schedule_iteration, config.opt.get('lambda_silhouette_shoulder_chain', 0.0))
        lambda_silhouette_shoulder_hole = C(schedule_iteration, config.opt.get('lambda_silhouette_shoulder_hole', 0.0))
        lambda_silhouette_shoulder_gap = C(schedule_iteration, config.opt.get('lambda_silhouette_shoulder_gap', 0.0))
        lambda_silhouette_shoulder_pinhole = C(schedule_iteration, config.opt.get('lambda_silhouette_shoulder_pinhole', 0.0))
        lambda_silhouette_inner = C(schedule_iteration, config.opt.get('lambda_silhouette_inner', 0.0))
        image_pattern_mode = str(config.opt.get('shoulder_arm_image_region_pattern', 'raw')).lower()
        perceptual_pattern_mode = str(config.opt.get('shoulder_arm_perceptual_region_pattern', 'raw')).lower()
        opacity_required_patterns = {'error_band', 'boundary_or_error', 'boundary_plus_error'}
        shoulder_image_loss_active = (
            C(schedule_iteration, config.opt.get('lambda_l1_shoulder_arm', 0.0)) > 0.0
            or C(schedule_iteration, config.opt.get('lambda_edge_shoulder_arm', 0.0)) > 0.0
        )
        shoulder_perceptual_loss_active = C(schedule_iteration, config.opt.get('lambda_perceptual_shoulder_arm', 0.0)) > 0.0
        stageB_semantic_loss_enabled = bool(config.opt.get('stageB_semantic_loss_enable', False))
        use_mask = any(
            value > 0.
            for value in (
                lambda_mask,
                lambda_mask_boundary,
                lambda_mask_boundary_hard,
                lambda_mask_shoulder_arm_boundary_hard,
                lambda_mask_upper_torso_boundary_hard,
                lambda_mask_shoulder_arm_region_hard,
                lambda_mask_shoulder_arm_disagreement_hard,
                lambda_mask_shoulder_focus_small_fp_hard,
                lambda_mask_shoulder_focus_small_fn_hard,
                lambda_silhouette_outer,
                lambda_silhouette_outer_shell,
                lambda_silhouette_head_outer_shell,
                lambda_silhouette_shoulder_arm_outer_shell,
                lambda_silhouette_upper_torso_outer_shell,
                lambda_silhouette_outer_spike,
                lambda_silhouette_outer_fragment,
                lambda_silhouette_outer_bead,
                lambda_silhouette_outer_chain,
                lambda_silhouette_arm_stipple,
                lambda_silhouette_arm_tail,
                lambda_silhouette_arm_fringe,
                lambda_silhouette_arm_attached_fragment,
                lambda_silhouette_arm_notch,
                lambda_silhouette_arm_hole,
                lambda_silhouette_arm_gap,
                lambda_silhouette_shoulder_bead,
                lambda_silhouette_shoulder_chain,
                lambda_silhouette_shoulder_attached_fragment,
                lambda_silhouette_shoulder_hole,
                lambda_silhouette_shoulder_gap,
                lambda_silhouette_shoulder_pinhole,
                lambda_silhouette_inner,
            )
        )
        use_mask = use_mask or (shoulder_image_loss_active and image_pattern_mode in opacity_required_patterns)
        use_mask = use_mask or (shoulder_perceptual_loss_active and perceptual_pattern_mode in opacity_required_patterns)
        use_mask = use_mask or (
            stageB_semantic_loss_enabled
            and bool(config.opt.get('stageB_semantic_use_opacity_support', True))
        )
        boundary_live_cache_key, boundary_live_cache_hit = _apply_boundary_live_score_cache(
            scene,
            data,
            boundary_live_score_cache,
            config,
        )
        render_pkg = render(data, schedule_iteration, scene, pipe, background, compute_loss=True, return_opacity=use_mask)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        opacity = render_pkg.get("opacity_render", None) if use_mask else None
        live_boundary_prior_score = _get_boundary_score_tensor(
            render_pkg["deformed_gaussian"],
            prefer_mixed=False,
        )

        # Loss
        gt_image = data.original_image.cuda()
        hard_gt_mask = _get_mask_from_data(data, 'hard', fallback='original')
        soft_gt_mask = _get_mask_from_data(data, 'soft', fallback=None)
        foreground_mask_source = str(config.opt.get('foreground_mask_source', 'hard'))
        global_mask_source = str(config.opt.get('global_mask_source', 'original'))
        boundary_target_mask_source = str(config.opt.get('boundary_target_mask_source', global_mask_source))
        fg_mask_source = _get_mask_from_data(data, foreground_mask_source, fallback='hard')
        if fg_mask_source is None:
            fg_mask_source = _get_mask_from_data(data, 'original', fallback=None)
        fg_mask = (fg_mask_source > 0.5).float()
        gt_mask = _get_mask_from_data(data, global_mask_source, fallback='original')
        if gt_mask is None:
            gt_mask = data.original_mask.cuda().float().clamp(0.0, 1.0)
        gt_mask_boundary = _get_mask_from_data(data, boundary_target_mask_source, fallback=global_mask_source)
        if gt_mask_boundary is None:
            gt_mask_boundary = hard_gt_mask if hard_gt_mask is not None else gt_mask
        live_boundary_image_score_raw, live_boundary_image_valid = _build_boundary_image_error_point_score(
            render_pkg["deformed_gaussian"],
            data,
            opacity,
            gt_mask_boundary,
            visibility_filter,
            config,
            iteration=schedule_iteration,
        )
        live_boundary_image_score = live_boundary_image_score_raw
        live_boundary_image_under_score = None
        live_boundary_image_over_score = None
        if isinstance(live_boundary_image_score_raw, dict):
            live_boundary_image_score = live_boundary_image_score_raw.get('candidate')
            live_boundary_image_under_score = live_boundary_image_score_raw.get('under')
            live_boundary_image_over_score = live_boundary_image_score_raw.get('over')
        if torch.is_tensor(live_boundary_image_score):
            setattr(
                render_pkg["deformed_gaussian"],
                "binding_boundary_image_score",
                live_boundary_image_score.detach(),
            )
        if torch.is_tensor(live_boundary_image_under_score):
            setattr(
                render_pkg["deformed_gaussian"],
                "binding_boundary_image_under_score",
                live_boundary_image_under_score.detach(),
            )
        if torch.is_tensor(live_boundary_image_over_score):
            setattr(
                render_pkg["deformed_gaussian"],
                "binding_boundary_image_over_score",
                live_boundary_image_over_score.detach(),
            )
        live_boundary_score = _mix_boundary_prior_with_image_score(
            live_boundary_prior_score,
            live_boundary_image_score,
            live_boundary_image_valid,
            config,
            iteration=schedule_iteration,
        )
        if torch.is_tensor(live_boundary_score):
            setattr(
                render_pkg["deformed_gaussian"],
                "binding_boundary_mixed_score",
                live_boundary_score.detach(),
            )
        live_boundary_subset_score = None
        if (
            torch.is_tensor(live_boundary_score)
            and live_boundary_score.shape[0] == scene.gaussians.get_xyz.shape[0]
        ):
            live_boundary_score = live_boundary_score.detach().float().clamp(0.0, 1.0)
            live_boundary_subset_score = _build_boundary_tag_mask(
                live_boundary_score,
                render_pkg["deformed_gaussian"],
                config,
            )
            if (
                torch.is_tensor(live_boundary_subset_score)
                and live_boundary_subset_score.shape[0] == live_boundary_score.shape[0]
            ):
                candidate_score = _get_boundary_tag_candidate_score(
                    live_boundary_score,
                    render_pkg["deformed_gaussian"],
                    config,
                )
                if torch.is_tensor(candidate_score) and candidate_score.shape[0] == live_boundary_score.shape[0]:
                    live_boundary_subset_score = live_boundary_subset_score.to(candidate_score.device) * candidate_score
        if hasattr(scene.gaussians, "set_live_boundary_score_state"):
            scene.gaussians.set_live_boundary_score_state(live_boundary_subset_score)
        _update_boundary_live_score_cache(
            boundary_live_score_cache,
            boundary_live_cache_key,
            live_boundary_subset_score,
            config,
        )
        opacity_bce = torch.clamp(opacity, 1.e-3, 1.-1.e-3) if opacity is not None else None
        boundary_region_source = str(config.opt.get('boundary_region_source', 'binary'))
        boundary_band_width = int(config.opt.get('boundary_band_width', 0))
        if boundary_region_source == 'soft_alpha' and soft_gt_mask is not None:
            boundary_mask = _soft_transition_mask(
                soft_gt_mask,
                config.opt.get('boundary_soft_low', 0.05),
                config.opt.get('boundary_soft_high', 0.95),
            )
        else:
            boundary_mask = _foreground_boundary_mask(fg_mask, boundary_band_width) if boundary_band_width > 1 else torch.zeros_like(fg_mask)
        silhouette_outer_ring_width = int(config.opt.get('silhouette_outer_ring_width', 0))
        silhouette_outer_shell_start_width = int(config.opt.get('silhouette_outer_shell_start_width', silhouette_outer_ring_width))
        silhouette_outer_shell_end_width = int(config.opt.get('silhouette_outer_shell_end_width', 0))
        silhouette_inner_ring_width = int(config.opt.get('silhouette_inner_ring_width', 0))
        outer_ring_mask = _foreground_outer_ring_mask(fg_mask, silhouette_outer_ring_width) if silhouette_outer_ring_width > 1 else torch.zeros_like(fg_mask)
        if silhouette_outer_shell_end_width > 1:
            if bool(config.opt.get('silhouette_outer_shell_soft_weights', False)):
                outer_shell_mask = _foreground_outer_shell_weight_mask(
                    fg_mask,
                    silhouette_outer_shell_start_width,
                    silhouette_outer_shell_end_width,
                    min_weight=float(config.opt.get('silhouette_outer_shell_weight_min', 0.25)),
                )
            else:
                outer_shell_mask = _foreground_outer_shell_mask(fg_mask, silhouette_outer_shell_start_width, silhouette_outer_shell_end_width)
        else:
            outer_shell_mask = torch.zeros_like(fg_mask)
        outer_spike_mask = torch.zeros_like(fg_mask)
        inner_ring_mask = _foreground_inner_ring_mask(fg_mask, silhouette_inner_ring_width) if silhouette_inner_ring_width > 1 else torch.zeros_like(fg_mask)
        face_mask, face_region_meta = _resolve_face_region_from_data(
            data,
            fg_mask,
            config_opt=config.opt,
        )
        head_outer_region_mask, head_outer_region_meta = _head_outer_shell_region_from_data(
            data,
            fg_mask,
            face_mask=face_mask,
            config_opt=config.opt,
        )
        head_outer_region_valid = head_outer_region_mask.sum().item() >= float(config.opt.get('silhouette_head_outer_min_pixels', 16))
        shoulder_arm_mask, shoulder_arm_region_meta = _resolve_shoulder_arm_region_from_data(
            data,
            fg_mask,
            face_mask=face_mask,
            config_opt=config.opt,
        )
        shoulder_focus_mask = _shoulder_focus_region_mask_from_data(data, fg_mask, face_mask=face_mask)
        shoulder_collar_mask = _shoulder_collar_region_mask_from_data(data, fg_mask, face_mask=face_mask)
        upper_torso_mask, upper_torso_region_meta = _resolve_upper_torso_region_from_data(
            data,
            fg_mask,
            face_mask=face_mask,
            shoulder_arm_mask=shoulder_arm_mask,
            config_opt=config.opt,
        )
        waist_mask, waist_region_meta = _resolve_waist_region_from_data(
            data,
            fg_mask,
            config_opt=config.opt,
        )

        lambda_l1 = C(schedule_iteration, config.opt.lambda_l1)
        lambda_dssim = C(schedule_iteration, config.opt.lambda_dssim)
        lambda_l1_fg = C(schedule_iteration, config.opt.get('lambda_l1_fg', 0.0))
        lambda_l1_boundary = C(schedule_iteration, config.opt.get('lambda_l1_boundary', 0.0))
        lambda_l1_face = C(schedule_iteration, config.opt.get('lambda_l1_face', 0.0))
        lambda_perceptual_face = C(schedule_iteration, config.opt.get('lambda_perceptual_face', 0.0))
        lambda_perceptual_face_patch = C(schedule_iteration, config.opt.get('lambda_perceptual_face_patch', 0.0))
        lambda_edge_face = C(schedule_iteration, config.opt.get('lambda_edge_face', 0.0))
        lambda_detail_face = C(schedule_iteration, config.opt.get('lambda_detail_face', 0.0))
        lambda_detail_face_luma_dog = C(schedule_iteration, config.opt.get('lambda_detail_face_luma_dog', 0.0))
        lambda_l1_shoulder_arm = C(schedule_iteration, config.opt.get('lambda_l1_shoulder_arm', 0.0))
        lambda_edge_shoulder_arm = C(schedule_iteration, config.opt.get('lambda_edge_shoulder_arm', 0.0))
        lambda_perceptual_shoulder_arm = C(schedule_iteration, config.opt.get('lambda_perceptual_shoulder_arm', 0.0))
        lambda_perceptual_shoulder_arm_patch = C(schedule_iteration, config.opt.get('lambda_perceptual_shoulder_arm_patch', 0.0))
        lambda_detail_shoulder_arm = C(schedule_iteration, config.opt.get('lambda_detail_shoulder_arm', 0.0))
        lambda_detail_shoulder_arm_luma_dog = C(schedule_iteration, config.opt.get('lambda_detail_shoulder_arm_luma_dog', 0.0))
        lambda_detail_upper_torso_luma_dog = C(schedule_iteration, config.opt.get('lambda_detail_upper_torso_luma_dog', 0.0))
        lambda_perceptual_upper_torso_patch = C(schedule_iteration, config.opt.get('lambda_perceptual_upper_torso_patch', 0.0))
        lambda_detail_upper_torso_core_luma_dog = C(schedule_iteration, config.opt.get('lambda_detail_upper_torso_core_luma_dog', 0.0))
        lambda_perceptual_upper_torso_core_patch = C(schedule_iteration, config.opt.get('lambda_perceptual_upper_torso_core_patch', 0.0))
        lambda_l1_shoulder_focus_dark_outlier = C(schedule_iteration, config.opt.get('lambda_l1_shoulder_focus_dark_outlier', 0.0))
        lambda_l1_shoulder_focus_bright_outlier = C(schedule_iteration, config.opt.get('lambda_l1_shoulder_focus_bright_outlier', 0.0))
        lambda_l1_waist = C(schedule_iteration, config.opt.get('lambda_l1_waist', 0.0))
        lambda_edge_waist = C(schedule_iteration, config.opt.get('lambda_edge_waist', 0.0))
        lambda_perceptual_waist = C(schedule_iteration, config.opt.get('lambda_perceptual_waist', 0.0))
        lambda_perceptual_waist_patch = C(schedule_iteration, config.opt.get('lambda_perceptual_waist_patch', 0.0))
        lambda_detail_waist = C(schedule_iteration, config.opt.get('lambda_detail_waist', 0.0))
        lambda_detail_waist_luma_dog = C(schedule_iteration, config.opt.get('lambda_detail_waist_luma_dog', 0.0))
        owner_local_detail_boost = _build_owner_local_detail_boost(
            config.opt,
            schedule_iteration,
            getattr(scene.converter, 'texture', None),
        )
        lambda_edge_face *= owner_local_detail_boost['edge_scales']['face']
        lambda_detail_face *= owner_local_detail_boost['detail_scales']['face']
        lambda_detail_face_luma_dog *= owner_local_detail_boost['luma_scales']['face']
        lambda_perceptual_face_patch *= owner_local_detail_boost['patch_scales']['face']
        lambda_edge_shoulder_arm *= owner_local_detail_boost['edge_scales']['shoulder_arm']
        lambda_detail_shoulder_arm *= owner_local_detail_boost['detail_scales']['shoulder_arm']
        lambda_detail_shoulder_arm_luma_dog *= owner_local_detail_boost['luma_scales']['shoulder_arm']
        lambda_perceptual_shoulder_arm_patch *= owner_local_detail_boost['patch_scales']['shoulder_arm']
        lambda_mask_shoulder_arm_boundary_hard *= owner_local_detail_boost['boundary_scales']['shoulder_arm']
        lambda_detail_upper_torso_luma_dog *= owner_local_detail_boost['luma_scales']['upper_torso']
        lambda_perceptual_upper_torso_patch *= owner_local_detail_boost['patch_scales']['upper_torso']
        lambda_mask_upper_torso_boundary_hard *= owner_local_detail_boost['boundary_scales']['upper_torso']
        lambda_detail_upper_torso_core_luma_dog *= owner_local_detail_boost['luma_scales']['upper_torso_core']
        lambda_perceptual_upper_torso_core_patch *= owner_local_detail_boost['patch_scales']['upper_torso_core']
        reliable_view_debug = _reliable_view_supervision_debug(data, config.opt)
        reliable_highfreq_weight = float(reliable_view_debug.get('weight', 1.0))
        if bool(reliable_view_debug.get('enabled', False)):
            if bool(config.opt.get('reliable_view_apply_edge', True)):
                lambda_edge_face *= reliable_highfreq_weight
                lambda_edge_shoulder_arm *= reliable_highfreq_weight
                lambda_edge_waist *= reliable_highfreq_weight
            if bool(config.opt.get('reliable_view_apply_detail', True)):
                lambda_detail_face *= reliable_highfreq_weight
                lambda_detail_shoulder_arm *= reliable_highfreq_weight
                lambda_detail_waist *= reliable_highfreq_weight
            if bool(config.opt.get('reliable_view_apply_luma_dog', True)):
                lambda_detail_face_luma_dog *= reliable_highfreq_weight
                lambda_detail_shoulder_arm_luma_dog *= reliable_highfreq_weight
                lambda_detail_upper_torso_luma_dog *= reliable_highfreq_weight
                lambda_detail_upper_torso_core_luma_dog *= reliable_highfreq_weight
                lambda_detail_waist_luma_dog *= reliable_highfreq_weight
            if bool(config.opt.get('reliable_view_apply_patch_perceptual', True)):
                lambda_perceptual_face_patch *= reliable_highfreq_weight
                lambda_perceptual_shoulder_arm_patch *= reliable_highfreq_weight
                lambda_perceptual_upper_torso_patch *= reliable_highfreq_weight
                lambda_perceptual_upper_torso_core_patch *= reliable_highfreq_weight
                lambda_perceptual_waist_patch *= reliable_highfreq_weight
            if bool(config.opt.get('reliable_view_apply_region_perceptual', False)):
                lambda_perceptual_face *= reliable_highfreq_weight
                lambda_perceptual_shoulder_arm *= reliable_highfreq_weight
                lambda_perceptual_waist *= reliable_highfreq_weight
        detail_multiscale_scales = config.opt.get('detail_multiscale_scales', [1, 2, 4])
        detail_highpass_kernel = int(config.opt.get('detail_highpass_kernel', 5))
        detail_scale_decay = float(config.opt.get('detail_scale_decay', 0.5))
        detail_gradient_mix = float(config.opt.get('detail_gradient_mix', 0.25))
        boundary_aware_enable = bool(config.opt.get('boundary_aware_enable', False))
        gate_l1_boundary = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_l1_boundary', False))
        gate_mask_boundary = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_mask_boundary', True))
        gate_mask_boundary_hard = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_mask_boundary_hard', True))
        gate_mask_shoulder_arm_boundary_hard = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_mask_shoulder_arm_boundary_hard', True))
        gate_mask_upper_torso_boundary_hard = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_mask_upper_torso_boundary_hard', True))
        gate_mask_shoulder_arm_region_hard = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_mask_shoulder_arm_region_hard', True))
        gate_mask_shoulder_arm_disagreement_hard = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_mask_shoulder_arm_disagreement_hard', True))
        gate_mask_shoulder_focus_small_fp_hard = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_mask_shoulder_focus_small_fp_hard', True))
        gate_mask_shoulder_focus_small_fn_hard = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_mask_shoulder_focus_small_fn_hard', True))
        gate_silhouette_outer = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_outer', True))
        gate_silhouette_outer_shell = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_outer_shell', True))
        gate_silhouette_head_outer_shell = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_head_outer_shell', True))
        gate_silhouette_shoulder_arm_outer_shell = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_shoulder_arm_outer_shell', True))
        gate_silhouette_upper_torso_outer_shell = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_upper_torso_outer_shell', True))
        gate_silhouette_outer_spike = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_outer_spike', True))
        gate_silhouette_outer_fragment = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_outer_fragment', True))
        gate_silhouette_outer_bead = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_outer_bead', True))
        gate_silhouette_outer_chain = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_outer_chain', True))
        gate_silhouette_arm_stipple = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_arm_stipple', True))
        gate_silhouette_arm_tail = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_arm_tail', True))
        gate_silhouette_arm_fringe = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_arm_fringe', True))
        gate_silhouette_arm_attached_fragment = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_arm_attached_fragment', True))
        gate_silhouette_shoulder_attached_fragment = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_shoulder_attached_fragment', True))
        gate_silhouette_arm_notch = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_arm_notch', True))
        gate_silhouette_arm_hole = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_arm_hole', True))
        gate_silhouette_arm_gap = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_arm_gap', True))
        gate_silhouette_shoulder_bead = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_shoulder_bead', True))
        gate_silhouette_shoulder_chain = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_shoulder_chain', True))
        gate_silhouette_shoulder_hole = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_shoulder_hole', True))
        gate_silhouette_shoulder_gap = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_shoulder_gap', True))
        gate_silhouette_shoulder_pinhole = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_shoulder_pinhole', True))
        gate_silhouette_inner = boundary_aware_enable and bool(config.opt.get('boundary_aware_gate_silhouette_inner', True))
        loss_l1 = torch.tensor(0.).cuda()
        loss_dssim = torch.tensor(0.).cuda()
        loss_l1_fg = torch.tensor(0.).cuda()
        loss_l1_boundary = torch.tensor(0.).cuda()
        loss_l1_face = torch.tensor(0.).cuda()
        loss_perceptual_face = torch.tensor(0.).cuda()
        loss_edge_face = torch.tensor(0.).cuda()
        loss_detail_face = torch.tensor(0.).cuda()
        loss_detail_face_luma_dog = torch.tensor(0.).cuda()
        loss_perceptual_face_patch = torch.tensor(0.).cuda()
        loss_l1_shoulder_arm = torch.tensor(0.).cuda()
        loss_edge_shoulder_arm = torch.tensor(0.).cuda()
        loss_perceptual_shoulder_arm = torch.tensor(0.).cuda()
        loss_detail_shoulder_arm = torch.tensor(0.).cuda()
        loss_detail_shoulder_arm_luma_dog = torch.tensor(0.).cuda()
        loss_perceptual_shoulder_arm_patch = torch.tensor(0.).cuda()
        loss_detail_upper_torso_luma_dog = torch.tensor(0.).cuda()
        loss_perceptual_upper_torso_patch = torch.tensor(0.).cuda()
        loss_detail_upper_torso_core_luma_dog = torch.tensor(0.).cuda()
        loss_perceptual_upper_torso_core_patch = torch.tensor(0.).cuda()
        loss_l1_shoulder_focus_dark_outlier = torch.tensor(0.).cuda()
        loss_l1_shoulder_focus_bright_outlier = torch.tensor(0.).cuda()
        loss_l1_waist = torch.tensor(0.).cuda()
        loss_edge_waist = torch.tensor(0.).cuda()
        loss_perceptual_waist = torch.tensor(0.).cuda()
        loss_detail_waist = torch.tensor(0.).cuda()
        loss_detail_waist_luma_dog = torch.tensor(0.).cuda()
        loss_perceptual_waist_patch = torch.tensor(0.).cuda()
        face_patch_count = 0
        shoulder_arm_patch_count = 0
        upper_torso_patch_count = 0
        upper_torso_core_patch_count = 0
        waist_patch_count = 0
        face_region_pixels = float(face_region_meta.get('region_pixels', face_mask.sum().item()))
        face_region_min_pixels = _face_region_effective_min_pixels(config.opt, face_region_meta)
        shoulder_arm_region_pixels = float(shoulder_arm_region_meta.get('region_pixels', shoulder_arm_mask.sum().item()))
        shoulder_arm_region_min_pixels = _shoulder_arm_region_effective_min_pixels(config.opt, shoulder_arm_region_meta)
        shoulder_focus_region_min_pixels = int(config.opt.get('shoulder_focus_region_min_pixels', 40))
        shoulder_collar_region_min_pixels = int(config.opt.get('shoulder_collar_region_min_pixels', 18))
        upper_torso_region_pixels = float(upper_torso_region_meta.get('region_pixels', upper_torso_mask.sum().item()))
        upper_torso_region_min_pixels = int(config.opt.get('upper_torso_region_min_pixels', 36))
        waist_region_min_pixels = int(config.opt.get('waist_region_min_pixels', 24))
        face_region_valid = face_region_pixels >= face_region_min_pixels
        shoulder_arm_region_valid = shoulder_arm_region_pixels >= shoulder_arm_region_min_pixels
        shoulder_focus_region_valid = shoulder_focus_mask.sum().item() >= shoulder_focus_region_min_pixels
        shoulder_collar_region_valid = shoulder_collar_mask.sum().item() >= shoulder_collar_region_min_pixels
        upper_torso_region_valid = upper_torso_region_pixels >= upper_torso_region_min_pixels
        waist_region_valid = waist_mask.sum().item() >= waist_region_min_pixels
        _maybe_log_face_region_debug(
            config.opt,
            schedule_iteration,
            face_region_meta,
            face_region_valid,
            face_region_min_pixels,
        )
        _maybe_log_shoulder_arm_region_debug(
            config.opt,
            schedule_iteration,
            shoulder_arm_region_meta,
            shoulder_arm_region_valid,
            shoulder_arm_region_min_pixels,
        )
        _maybe_log_waist_region_debug(
            config.opt,
            schedule_iteration,
            waist_region_meta,
            waist_region_valid,
            waist_region_min_pixels,
        )
        _maybe_log_upper_torso_region_debug(
            config.opt,
            schedule_iteration,
            upper_torso_region_meta,
            upper_torso_region_valid,
            upper_torso_region_min_pixels,
        )
        shoulder_supervision_target_mask = hard_gt_mask if hard_gt_mask is not None else gt_mask
        arm_boundary_target = hard_gt_mask if hard_gt_mask is not None else gt_mask_boundary
        arm_region_target = hard_gt_mask if hard_gt_mask is not None else gt_mask
        shoulder_arm_image_mask, shoulder_arm_image_region_meta = _build_shoulder_supervision_mask(
            shoulder_arm_mask,
            shoulder_focus_mask,
            shoulder_collar_mask,
            fg_mask,
            basis_mode=str(config.opt.get('shoulder_arm_image_region_mode', 'arm')),
            pattern_mode=str(config.opt.get('shoulder_arm_image_region_pattern', 'raw')),
            boundary_mask=boundary_mask,
            outer_shell_mask=outer_shell_mask,
            target_mask=shoulder_supervision_target_mask,
            opacity=opacity_bce,
            basis_dilate=int(config.opt.get('shoulder_arm_image_region_dilate', 0)),
            final_dilate=int(config.opt.get('shoulder_arm_image_region_pattern_dilate', 0)),
            disagreement_threshold=float(config.opt.get('shoulder_arm_image_region_opacity_threshold', config.opt.get('shoulder_arm_disagreement_opacity_threshold', 0.08))),
        )
        shoulder_arm_image_region_min_pixels = int(config.opt.get('shoulder_arm_image_region_min_pixels', shoulder_arm_region_min_pixels))
        shoulder_arm_image_region_valid = shoulder_arm_image_mask.sum().item() >= shoulder_arm_image_region_min_pixels
        shoulder_arm_perceptual_region_mask, shoulder_arm_perceptual_region_meta = _build_shoulder_supervision_mask(
            shoulder_arm_mask,
            shoulder_focus_mask,
            shoulder_collar_mask,
            fg_mask,
            basis_mode=str(config.opt.get('shoulder_arm_perceptual_region_mode', 'arm')),
            pattern_mode=str(config.opt.get('shoulder_arm_perceptual_region_pattern', 'raw')),
            boundary_mask=boundary_mask,
            outer_shell_mask=outer_shell_mask,
            target_mask=shoulder_supervision_target_mask,
            opacity=opacity_bce,
            basis_dilate=int(config.opt.get('shoulder_arm_perceptual_region_dilate', 0)),
            final_dilate=int(config.opt.get('shoulder_arm_perceptual_region_pattern_dilate', 0)),
            disagreement_threshold=float(config.opt.get('shoulder_arm_perceptual_region_opacity_threshold', config.opt.get('shoulder_arm_disagreement_opacity_threshold', 0.08))),
        )
        shoulder_arm_perceptual_region_min_pixels = int(config.opt.get('shoulder_arm_perceptual_region_min_pixels', shoulder_arm_region_min_pixels))
        shoulder_arm_perceptual_region_valid = shoulder_arm_perceptual_region_mask.sum().item() >= shoulder_arm_perceptual_region_min_pixels
        shoulder_arm_boundary_supervision_mask, shoulder_arm_boundary_region_meta = _build_shoulder_supervision_mask(
            shoulder_arm_mask,
            shoulder_focus_mask,
            shoulder_collar_mask,
            fg_mask,
            basis_mode=str(config.opt.get('shoulder_arm_boundary_region_mode', 'arm')),
            pattern_mode=str(config.opt.get('shoulder_arm_boundary_region_pattern', 'boundary')),
            boundary_mask=boundary_mask,
            outer_shell_mask=outer_shell_mask,
            target_mask=arm_boundary_target,
            opacity=opacity_bce,
            basis_dilate=int(config.opt.get('shoulder_arm_boundary_region_dilate', 23)),
            final_dilate=int(config.opt.get('shoulder_arm_boundary_region_pattern_dilate', 0)),
            disagreement_threshold=float(config.opt.get('shoulder_arm_boundary_region_opacity_threshold', config.opt.get('shoulder_arm_disagreement_opacity_threshold', 0.08))),
        )
        shoulder_arm_region_hard_supervision_mask, shoulder_arm_region_hard_meta = _build_shoulder_supervision_mask(
            shoulder_arm_mask,
            shoulder_focus_mask,
            shoulder_collar_mask,
            fg_mask,
            basis_mode=str(config.opt.get('shoulder_arm_region_hard_mode', 'arm')),
            pattern_mode=str(config.opt.get('shoulder_arm_region_hard_pattern', 'raw')),
            boundary_mask=boundary_mask,
            outer_shell_mask=outer_shell_mask,
            target_mask=arm_region_target,
            opacity=opacity_bce,
            basis_dilate=int(config.opt.get('shoulder_arm_region_hard_dilate', 19)),
            final_dilate=int(config.opt.get('shoulder_arm_region_hard_pattern_dilate', 0)),
            disagreement_threshold=float(config.opt.get('shoulder_arm_region_hard_opacity_threshold', config.opt.get('shoulder_arm_disagreement_opacity_threshold', 0.08))),
        )
        shoulder_arm_disagreement_supervision_mask, shoulder_arm_disagreement_meta = _build_shoulder_supervision_mask(
            shoulder_arm_mask,
            shoulder_focus_mask,
            shoulder_collar_mask,
            fg_mask,
            basis_mode=str(config.opt.get('shoulder_arm_disagreement_region_mode', 'arm')),
            pattern_mode=str(config.opt.get('shoulder_arm_disagreement_region_pattern', 'error_band')),
            boundary_mask=boundary_mask,
            outer_shell_mask=outer_shell_mask,
            target_mask=arm_region_target,
            opacity=opacity_bce,
            basis_dilate=int(config.opt.get('shoulder_arm_disagreement_region_dilate', 25)),
            final_dilate=int(config.opt.get('shoulder_arm_disagreement_mask_dilate', 7)),
            disagreement_threshold=float(config.opt.get('shoulder_arm_disagreement_opacity_threshold', 0.08)),
        )
        shoulder_arm_outer_shell_supervision_mask, shoulder_arm_outer_shell_meta = _build_shoulder_supervision_mask(
            shoulder_arm_mask,
            shoulder_focus_mask,
            shoulder_collar_mask,
            fg_mask,
            basis_mode=str(config.opt.get('shoulder_arm_outer_shell_region_mode', 'arm')),
            pattern_mode=str(config.opt.get('shoulder_arm_outer_shell_region_pattern', 'outer_shell')),
            boundary_mask=boundary_mask,
            outer_shell_mask=outer_shell_mask,
            target_mask=arm_boundary_target,
            opacity=opacity_bce,
            basis_dilate=int(config.opt.get('shoulder_arm_outer_shell_region_dilate', 25)),
            final_dilate=int(config.opt.get('shoulder_arm_outer_shell_region_pattern_dilate', 0)),
            disagreement_threshold=float(config.opt.get('shoulder_arm_outer_shell_region_opacity_threshold', config.opt.get('shoulder_arm_disagreement_opacity_threshold', 0.08))),
        )
        upper_torso_boundary_target = hard_gt_mask if hard_gt_mask is not None else gt_mask_boundary
        upper_torso_boundary_supervision_mask, upper_torso_boundary_region_meta = _build_local_region_supervision_mask(
            upper_torso_mask,
            fg_mask,
            pattern_mode=str(config.opt.get('upper_torso_boundary_region_pattern', 'boundary')),
            boundary_mask=boundary_mask,
            outer_shell_mask=outer_shell_mask,
            target_mask=upper_torso_boundary_target,
            opacity=opacity_bce,
            basis_dilate=int(config.opt.get('upper_torso_boundary_region_dilate', 11)),
            final_dilate=int(config.opt.get('upper_torso_boundary_region_pattern_dilate', 0)),
            disagreement_threshold=float(config.opt.get('upper_torso_boundary_region_opacity_threshold', 0.08)),
        )
        upper_torso_outer_shell_supervision_mask, upper_torso_outer_shell_meta = _build_local_region_supervision_mask(
            upper_torso_mask,
            fg_mask,
            pattern_mode=str(config.opt.get('upper_torso_outer_shell_region_pattern', 'outer_shell')),
            boundary_mask=boundary_mask,
            outer_shell_mask=outer_shell_mask,
            target_mask=upper_torso_boundary_target,
            opacity=opacity_bce,
            basis_dilate=int(config.opt.get('upper_torso_outer_shell_region_dilate', 13)),
            final_dilate=int(config.opt.get('upper_torso_outer_shell_region_pattern_dilate', 0)),
            disagreement_threshold=float(config.opt.get('upper_torso_outer_shell_region_opacity_threshold', 0.08)),
        )
        upper_torso_image_mask = upper_torso_mask
        upper_torso_image_dilate = int(config.opt.get('upper_torso_image_region_dilate', 0))
        if upper_torso_image_dilate > 1:
            upper_torso_image_mask = _binary_dilate(upper_torso_image_mask, upper_torso_image_dilate).clamp(0.0, 1.0) * fg_mask
        upper_torso_image_region_min_pixels = int(
            config.opt.get('upper_torso_image_region_min_pixels', upper_torso_region_min_pixels)
        )
        upper_torso_image_region_valid = (
            upper_torso_image_mask.sum().item() >= upper_torso_image_region_min_pixels
        )
        waist_image_mask = waist_mask
        waist_image_dilate = int(config.opt.get('waist_image_region_dilate', 0))
        if waist_image_dilate > 1:
            waist_image_mask = _binary_dilate(waist_image_mask, waist_image_dilate).clamp(0.0, 1.0) * fg_mask
        waist_image_region_min_pixels = int(config.opt.get('waist_image_region_min_pixels', waist_region_min_pixels))
        waist_image_region_valid = waist_image_mask.sum().item() >= waist_image_region_min_pixels
        face_detail_mask = _detail_interior_mask(face_mask, fg_mask, config.opt, region_name='face')
        face_detail_region_min_pixels = int(config.opt.get('face_detail_region_min_pixels', face_region_min_pixels))
        face_detail_validity_threshold = float(
            _region_opt_value(config.opt, 'face', 'detail_region_validity_threshold', config.opt.get('detail_region_validity_threshold', 0.05))
        )
        face_detail_active_pixels = _mask_active_pixels(face_detail_mask, threshold=face_detail_validity_threshold)
        face_detail_region_valid = face_detail_active_pixels >= face_detail_region_min_pixels
        shoulder_arm_detail_mask = _detail_interior_mask(
            shoulder_arm_image_mask,
            fg_mask,
            config.opt,
            region_name='shoulder_arm',
        )
        shoulder_arm_detail_region_min_pixels = int(
            config.opt.get('shoulder_arm_detail_region_min_pixels', shoulder_arm_image_region_min_pixels)
        )
        shoulder_arm_detail_validity_threshold = float(
            _region_opt_value(
                config.opt,
                'shoulder_arm',
                'detail_region_validity_threshold',
                config.opt.get('detail_region_validity_threshold', 0.05),
            )
        )
        shoulder_arm_detail_active_pixels = _mask_active_pixels(
            shoulder_arm_detail_mask,
            threshold=shoulder_arm_detail_validity_threshold,
        )
        shoulder_arm_detail_region_valid = (
            shoulder_arm_detail_active_pixels >= shoulder_arm_detail_region_min_pixels
        )
        upper_torso_detail_mask = _detail_interior_mask(
            upper_torso_image_mask,
            fg_mask,
            config.opt,
            region_name='upper_torso',
        )
        upper_torso_detail_region_min_pixels = int(
            config.opt.get('upper_torso_detail_region_min_pixels', upper_torso_image_region_min_pixels)
        )
        upper_torso_detail_validity_threshold = float(
            _region_opt_value(
                config.opt,
                'upper_torso',
                'detail_region_validity_threshold',
                config.opt.get('detail_region_validity_threshold', 0.05),
            )
        )
        upper_torso_detail_active_pixels = _mask_active_pixels(
            upper_torso_detail_mask,
            threshold=upper_torso_detail_validity_threshold,
        )
        upper_torso_detail_region_valid = (
            upper_torso_detail_active_pixels >= upper_torso_detail_region_min_pixels
        )
        upper_torso_core_mask = _build_local_focus_core_mask(
            upper_torso_detail_mask,
            fg_mask,
            config.opt,
            region_name='upper_torso_core',
        )
        upper_torso_core_region_min_pixels = int(
            config.opt.get('upper_torso_core_region_min_pixels', max(16, upper_torso_detail_region_min_pixels // 3))
        )
        upper_torso_core_validity_threshold = float(
            _region_opt_value(
                config.opt,
                'upper_torso_core',
                'region_validity_threshold',
                config.opt.get('detail_region_validity_threshold', 0.05),
            )
        )
        upper_torso_core_active_pixels = _mask_active_pixels(
            upper_torso_core_mask,
            threshold=upper_torso_core_validity_threshold,
        )
        upper_torso_core_region_valid = (
            upper_torso_core_active_pixels >= upper_torso_core_region_min_pixels
        )
        waist_detail_mask = _detail_interior_mask(waist_image_mask, fg_mask, config.opt, region_name='waist')
        waist_detail_region_min_pixels = int(config.opt.get('waist_detail_region_min_pixels', waist_image_region_min_pixels))
        waist_detail_validity_threshold = float(
            _region_opt_value(config.opt, 'waist', 'detail_region_validity_threshold', config.opt.get('detail_region_validity_threshold', 0.05))
        )
        waist_detail_active_pixels = _mask_active_pixels(waist_detail_mask, threshold=waist_detail_validity_threshold)
        waist_detail_region_valid = waist_detail_active_pixels >= waist_detail_region_min_pixels
        upper_torso_perceptual_used_mask = upper_torso_mask
        upper_torso_core_perceptual_used_mask = upper_torso_core_mask
        waist_perceptual_used_mask = waist_mask
        waist_perceptual_region_min_pixels = int(config.opt.get('waist_perceptual_region_min_pixels', waist_region_min_pixels))
        loss_gt_image, photometric_correction_debug = _photometric_corrected_gt_image(
            image,
            gt_image,
            fg_mask,
            config.opt,
            iteration=schedule_iteration,
        )
        contour_uncertainty_weight, contour_uncertainty_debug = _contour_uncertainty_weight_mask(
            fg_mask,
            config.opt,
        )
        fullframe_image_loss_mask = contour_uncertainty_weight
        fg_image_loss_mask = fg_mask
        boundary_image_loss_mask = boundary_mask
        face_image_loss_mask = face_mask
        shoulder_arm_l1_loss_mask = shoulder_arm_image_mask
        shoulder_arm_edge_loss_mask = shoulder_arm_image_mask
        upper_torso_luma_loss_mask = upper_torso_detail_mask
        upper_torso_core_luma_loss_mask = upper_torso_core_mask
        waist_l1_loss_mask = waist_image_mask
        waist_edge_loss_mask = waist_image_mask
        if contour_uncertainty_weight is not None:
            contour_uncertainty_weight = contour_uncertainty_weight.to(device=image.device, dtype=image.dtype)
            fg_image_loss_mask = (fg_image_loss_mask * contour_uncertainty_weight).clamp(0.0, 1.0)
            boundary_image_loss_mask = (boundary_image_loss_mask * contour_uncertainty_weight).clamp(0.0, 1.0)
            face_image_loss_mask = (face_image_loss_mask * contour_uncertainty_weight).clamp(0.0, 1.0)
            shoulder_arm_l1_loss_mask = (shoulder_arm_l1_loss_mask * contour_uncertainty_weight).clamp(0.0, 1.0)
            shoulder_arm_edge_loss_mask = (shoulder_arm_edge_loss_mask * contour_uncertainty_weight).clamp(0.0, 1.0)
            upper_torso_luma_loss_mask = (upper_torso_luma_loss_mask * contour_uncertainty_weight).clamp(0.0, 1.0)
            upper_torso_core_luma_loss_mask = (upper_torso_core_luma_loss_mask * contour_uncertainty_weight).clamp(0.0, 1.0)
            waist_l1_loss_mask = (waist_l1_loss_mask * contour_uncertainty_weight).clamp(0.0, 1.0)
            waist_edge_loss_mask = (waist_edge_loss_mask * contour_uncertainty_weight).clamp(0.0, 1.0)
        alignment_contour_debug = {}
        alignment_contour_weight, alignment_contour_debug = _alignment_aware_contour_weight_mask(
            image,
            loss_gt_image,
            fg_mask,
            shoulder_arm_image_mask,
            config.opt,
        )
        if alignment_contour_weight is not None:
            alignment_contour_weight = alignment_contour_weight.to(device=image.device, dtype=image.dtype)
            shoulder_arm_edge_loss_mask = (shoulder_arm_edge_loss_mask * alignment_contour_weight).clamp(
                0.0,
                float(config.opt.get('alignment_aware_contour_max_weight', 1.45)),
            )
            if bool(config.opt.get('alignment_aware_contour_apply_luma', False)):
                shoulder_arm_detail_mask = (shoulder_arm_detail_mask * alignment_contour_weight).clamp(
                    0.0,
                    float(config.opt.get('alignment_aware_contour_max_weight', 1.45)),
                )
            if bool(config.opt.get('alignment_aware_contour_apply_l1', False)):
                shoulder_arm_l1_loss_mask = (shoulder_arm_l1_loss_mask * alignment_contour_weight).clamp(
                    0.0,
                    float(config.opt.get('alignment_aware_contour_max_weight', 1.45)),
                )
        if bool(config.opt.get('alignment_aware_contour_apply_waist', False)):
            waist_alignment_weight, waist_alignment_debug = _alignment_aware_contour_weight_mask(
                image,
                loss_gt_image,
                fg_mask,
                waist_image_mask,
                config.opt,
            )
            if waist_alignment_weight is not None:
                waist_alignment_weight = waist_alignment_weight.to(device=image.device, dtype=image.dtype)
                waist_edge_loss_mask = (waist_edge_loss_mask * waist_alignment_weight).clamp(
                    0.0,
                    float(config.opt.get('alignment_aware_contour_max_weight', 1.45)),
                )
                if bool(config.opt.get('alignment_aware_contour_apply_luma', False)):
                    waist_detail_mask = (waist_detail_mask * waist_alignment_weight).clamp(
                        0.0,
                        float(config.opt.get('alignment_aware_contour_max_weight', 1.45)),
                    )
                alignment_contour_debug['waist_pixels'] = float(waist_alignment_debug.get('pixels', 0.0))
                alignment_contour_debug['waist_mean_weight'] = float(waist_alignment_debug.get('mean_weight', 0.0))
        photo_debug_interval = int(config.opt.get('photometric_contour_debug_interval', 0))
        if photo_debug_interval > 0 and schedule_iteration % photo_debug_interval == 0:
            camera_affine_debug = getattr(scene.converter.camera_affine, 'last_debug', {})
            print(
                (
                    f"[PhotometricContour] iter={schedule_iteration} "
                    f"photo_strength={float(photometric_correction_debug.get('strength', 0.0)):.3f} "
                    f"photo_pixels={float(photometric_correction_debug.get('active_pixels', 0.0)):.1f} "
                    f"photo_scale={float(photometric_correction_debug.get('scale_mean', 1.0)):.3f} "
                    f"photo_shift={float(photometric_correction_debug.get('shift_abs_mean', 0.0)):.4f} "
                    f"uncertain_pixels={float(contour_uncertainty_debug.get('uncertain_pixels', 0.0)):.1f} "
                    f"uncertain_min_w={float(contour_uncertainty_debug.get('min_weight', 1.0)):.3f} "
                    f"align_pixels={float(alignment_contour_debug.get('pixels', 0.0)):.1f} "
                    f"align_mean_w={float(alignment_contour_debug.get('mean_weight', 0.0)):.4f} "
                    f"align_stable={float(alignment_contour_debug.get('stable_mean', 0.0)):.4f} "
                    f"align_suppress={float(alignment_contour_debug.get('suppress_mean', 0.0)):.4f} "
                    f"align_waist_pixels={float(alignment_contour_debug.get('waist_pixels', 0.0)):.1f} "
                    f"cam_affine={camera_affine_debug.get('camera', 'na')}:"
                    f"{float(camera_affine_debug.get('active', 0.0)):.0f}/"
                    f"{float(camera_affine_debug.get('strength', 0.0)):.3f}/"
                    f"{float(camera_affine_debug.get('scale_delta_abs_mean', 0.0)):.4f}/"
                    f"{float(camera_affine_debug.get('shift_abs_mean', 0.0)):.4f}"
                ),
                flush=True,
            )
        if lambda_l1 > 0.:
            if fullframe_image_loss_mask is not None:
                loss_l1 = _masked_l1_loss(image, loss_gt_image, fullframe_image_loss_mask)
            else:
                loss_l1 = l1_loss(image, loss_gt_image)
        if lambda_dssim > 0.:
            loss_dssim = 1.0 - ssim(image, loss_gt_image)
        if lambda_l1_fg > 0.:
            loss_l1_fg = _masked_l1_loss(image, loss_gt_image, fg_image_loss_mask)
        if lambda_l1_boundary > 0. and boundary_mask.sum().item() > 0:
            loss_l1_boundary = _masked_l1_loss(image, loss_gt_image, boundary_image_loss_mask)
        if lambda_l1_face > 0. and face_region_valid:
            loss_l1_face = _masked_l1_loss(image, loss_gt_image, face_image_loss_mask)
        if lambda_edge_face > 0. and face_region_valid:
            loss_edge_face = _masked_gradient_l1_loss(image, loss_gt_image, face_image_loss_mask)
        if lambda_detail_face > 0. and face_detail_region_valid:
            loss_detail_face = _masked_multiscale_detail_loss(
                image,
                loss_gt_image,
                face_detail_mask,
                scales=detail_multiscale_scales,
                blur_kernel=detail_highpass_kernel,
                scale_decay=detail_scale_decay,
                gradient_mix=detail_gradient_mix,
            )
        if lambda_detail_face_luma_dog > 0. and face_detail_region_valid:
            loss_detail_face_luma_dog = _masked_multiscale_weighted_luma_dog_loss(
                image,
                loss_gt_image,
                face_detail_mask,
                scales=_region_opt_value(config.opt, 'face', 'detail_luma_dog_scales', detail_multiscale_scales),
                scale_decay=float(_region_opt_value(config.opt, 'face', 'detail_luma_dog_scale_decay', detail_scale_decay)),
                small_kernel=int(_region_opt_value(config.opt, 'face', 'detail_luma_dog_small_kernel', 3)),
                large_kernel=int(_region_opt_value(config.opt, 'face', 'detail_luma_dog_large_kernel', 9)),
                base_weight=float(_region_opt_value(config.opt, 'face', 'detail_luma_dog_base_weight', 0.20)),
                focus_scale=float(_region_opt_value(config.opt, 'face', 'detail_luma_dog_focus_scale', 2.4)),
                focus_power=float(_region_opt_value(config.opt, 'face', 'detail_luma_dog_focus_power', 1.5)),
                error_mix=float(_region_opt_value(config.opt, 'face', 'detail_luma_dog_error_mix', 0.15)),
            )
        if lambda_l1_shoulder_arm > 0. and shoulder_arm_image_region_valid:
            loss_l1_shoulder_arm = _masked_l1_loss(image, loss_gt_image, shoulder_arm_l1_loss_mask)
        if lambda_edge_shoulder_arm > 0. and shoulder_arm_image_region_valid:
            loss_edge_shoulder_arm = _masked_gradient_l1_loss(image, loss_gt_image, shoulder_arm_edge_loss_mask)
        if lambda_detail_shoulder_arm > 0. and shoulder_arm_detail_region_valid:
            loss_detail_shoulder_arm = _masked_multiscale_detail_loss(
                image,
                loss_gt_image,
                shoulder_arm_detail_mask,
                scales=detail_multiscale_scales,
                blur_kernel=detail_highpass_kernel,
                scale_decay=detail_scale_decay,
                gradient_mix=detail_gradient_mix,
            )
        if lambda_detail_shoulder_arm_luma_dog > 0. and shoulder_arm_detail_region_valid:
            loss_detail_shoulder_arm_luma_dog = _masked_multiscale_weighted_luma_dog_loss(
                image,
                loss_gt_image,
                shoulder_arm_detail_mask,
                scales=_region_opt_value(config.opt, 'shoulder_arm', 'detail_luma_dog_scales', detail_multiscale_scales),
                scale_decay=float(_region_opt_value(config.opt, 'shoulder_arm', 'detail_luma_dog_scale_decay', detail_scale_decay)),
                small_kernel=int(_region_opt_value(config.opt, 'shoulder_arm', 'detail_luma_dog_small_kernel', 3)),
                large_kernel=int(_region_opt_value(config.opt, 'shoulder_arm', 'detail_luma_dog_large_kernel', 9)),
                base_weight=float(_region_opt_value(config.opt, 'shoulder_arm', 'detail_luma_dog_base_weight', 0.20)),
                focus_scale=float(_region_opt_value(config.opt, 'shoulder_arm', 'detail_luma_dog_focus_scale', 2.1)),
                focus_power=float(_region_opt_value(config.opt, 'shoulder_arm', 'detail_luma_dog_focus_power', 1.45)),
                error_mix=float(_region_opt_value(config.opt, 'shoulder_arm', 'detail_luma_dog_error_mix', 0.18)),
            )
        if lambda_detail_upper_torso_luma_dog > 0. and upper_torso_detail_region_valid:
            loss_detail_upper_torso_luma_dog = _masked_multiscale_weighted_luma_dog_loss(
                image,
                loss_gt_image,
                upper_torso_luma_loss_mask,
                scales=_region_opt_value(config.opt, 'upper_torso', 'detail_luma_dog_scales', detail_multiscale_scales),
                scale_decay=float(_region_opt_value(config.opt, 'upper_torso', 'detail_luma_dog_scale_decay', detail_scale_decay)),
                small_kernel=int(_region_opt_value(config.opt, 'upper_torso', 'detail_luma_dog_small_kernel', 3)),
                large_kernel=int(_region_opt_value(config.opt, 'upper_torso', 'detail_luma_dog_large_kernel', 9)),
                base_weight=float(_region_opt_value(config.opt, 'upper_torso', 'detail_luma_dog_base_weight', 0.18)),
                focus_scale=float(_region_opt_value(config.opt, 'upper_torso', 'detail_luma_dog_focus_scale', 2.0)),
                focus_power=float(_region_opt_value(config.opt, 'upper_torso', 'detail_luma_dog_focus_power', 1.38)),
                error_mix=float(_region_opt_value(config.opt, 'upper_torso', 'detail_luma_dog_error_mix', 0.18)),
            )
        if lambda_detail_upper_torso_core_luma_dog > 0. and upper_torso_core_region_valid:
            loss_detail_upper_torso_core_luma_dog = _masked_multiscale_weighted_luma_dog_loss(
                image,
                loss_gt_image,
                upper_torso_core_luma_loss_mask,
                scales=_region_opt_value(config.opt, 'upper_torso_core', 'detail_luma_dog_scales', detail_multiscale_scales),
                scale_decay=float(_region_opt_value(config.opt, 'upper_torso_core', 'detail_luma_dog_scale_decay', detail_scale_decay)),
                small_kernel=int(_region_opt_value(config.opt, 'upper_torso_core', 'detail_luma_dog_small_kernel', 3)),
                large_kernel=int(_region_opt_value(config.opt, 'upper_torso_core', 'detail_luma_dog_large_kernel', 9)),
                base_weight=float(_region_opt_value(config.opt, 'upper_torso_core', 'detail_luma_dog_base_weight', 0.20)),
                focus_scale=float(_region_opt_value(config.opt, 'upper_torso_core', 'detail_luma_dog_focus_scale', 2.4)),
                focus_power=float(_region_opt_value(config.opt, 'upper_torso_core', 'detail_luma_dog_focus_power', 1.48)),
                error_mix=float(_region_opt_value(config.opt, 'upper_torso_core', 'detail_luma_dog_error_mix', 0.18)),
            )
        if lambda_l1_waist > 0. and waist_image_region_valid:
            loss_l1_waist = _masked_l1_loss(image, loss_gt_image, waist_l1_loss_mask)
        if lambda_edge_waist > 0. and waist_image_region_valid:
            loss_edge_waist = _masked_gradient_l1_loss(image, loss_gt_image, waist_edge_loss_mask)
        if lambda_detail_waist > 0. and waist_detail_region_valid:
            loss_detail_waist = _masked_multiscale_detail_loss(
                image,
                loss_gt_image,
                waist_detail_mask,
                scales=detail_multiscale_scales,
                blur_kernel=detail_highpass_kernel,
                scale_decay=detail_scale_decay,
                gradient_mix=detail_gradient_mix,
            )
        if lambda_detail_waist_luma_dog > 0. and waist_detail_region_valid:
            loss_detail_waist_luma_dog = _masked_multiscale_weighted_luma_dog_loss(
                image,
                loss_gt_image,
                waist_detail_mask,
                scales=_region_opt_value(config.opt, 'waist', 'detail_luma_dog_scales', detail_multiscale_scales),
                scale_decay=float(_region_opt_value(config.opt, 'waist', 'detail_luma_dog_scale_decay', detail_scale_decay)),
                small_kernel=int(_region_opt_value(config.opt, 'waist', 'detail_luma_dog_small_kernel', 3)),
                large_kernel=int(_region_opt_value(config.opt, 'waist', 'detail_luma_dog_large_kernel', 9)),
                base_weight=float(_region_opt_value(config.opt, 'waist', 'detail_luma_dog_base_weight', 0.18)),
                focus_scale=float(_region_opt_value(config.opt, 'waist', 'detail_luma_dog_focus_scale', 1.9)),
                focus_power=float(_region_opt_value(config.opt, 'waist', 'detail_luma_dog_focus_power', 1.35)),
                error_mix=float(_region_opt_value(config.opt, 'waist', 'detail_luma_dog_error_mix', 0.18)),
            )
        if shoulder_focus_region_valid and (lambda_l1_shoulder_focus_dark_outlier > 0. or lambda_l1_shoulder_focus_bright_outlier > 0.):
            shoulder_photo_region = shoulder_focus_mask
            shoulder_photo_region_dilate = int(config.opt.get('shoulder_focus_photo_region_dilate', 3))
            if shoulder_photo_region_dilate > 1:
                shoulder_photo_region = _binary_dilate(shoulder_photo_region, shoulder_photo_region_dilate).clamp(0.0, 1.0) * fg_mask

            shoulder_target_binary = hard_gt_mask if hard_gt_mask is not None else fg_mask
            shoulder_target_binary = (shoulder_target_binary.to(device=image.device, dtype=image.dtype) >= 0.5).float()

            shoulder_dark_region = shoulder_photo_region * shoulder_target_binary
            shoulder_dark_core_erode = int(config.opt.get('shoulder_focus_dark_core_erode_kernel_size', 3))
            if shoulder_dark_core_erode > 1 and shoulder_dark_region.sum().item() > 0:
                dark_core = _binary_erode(shoulder_dark_region, shoulder_dark_core_erode).clamp(0.0, 1.0) * shoulder_dark_region
                if dark_core.sum().item() > 0:
                    shoulder_dark_region = dark_core

            shoulder_bright_region = _foreground_outer_shell_mask(
                shoulder_target_binary,
                int(config.opt.get('shoulder_focus_bright_outer_shell_start_width', 1)),
                int(config.opt.get('shoulder_focus_bright_outer_shell_end_width', 17)),
            ) * shoulder_photo_region
            if shoulder_bright_region.sum().item() <= 0:
                shoulder_bright_region = shoulder_photo_region

            shoulder_dark_mask, shoulder_bright_mask = _small_luma_difference_component_masks(
                image,
                gt_image,
                region_mask=shoulder_photo_region,
                darker_region_mask=shoulder_dark_region,
                brighter_region_mask=shoulder_bright_region,
                darker_threshold=float(config.opt.get('shoulder_focus_dark_luma_threshold', 0.085)),
                brighter_threshold=float(config.opt.get('shoulder_focus_bright_luma_threshold', 0.065)),
                darker_area_min=int(config.opt.get('shoulder_focus_dark_area_min', 1)),
                darker_area_max=int(config.opt.get('shoulder_focus_dark_area_max', 28)),
                brighter_area_min=int(config.opt.get('shoulder_focus_bright_area_min', 1)),
                brighter_area_max=int(config.opt.get('shoulder_focus_bright_area_max', 40)),
                darker_power=float(config.opt.get('shoulder_focus_dark_power', 1.8)),
                brighter_power=float(config.opt.get('shoulder_focus_bright_power', 1.4)),
            )
            if lambda_l1_shoulder_focus_dark_outlier > 0. and shoulder_dark_mask is not None and shoulder_dark_mask.sum().item() > 0:
                loss_l1_shoulder_focus_dark_outlier = _masked_l1_loss(image, gt_image, shoulder_dark_mask)
            if lambda_l1_shoulder_focus_bright_outlier > 0. and shoulder_bright_mask is not None and shoulder_bright_mask.sum().item() > 0:
                loss_l1_shoulder_focus_bright_outlier = _masked_l1_loss(image, gt_image, shoulder_bright_mask)
        base_loss = (
            lambda_l1 * loss_l1
            + lambda_dssim * loss_dssim
            + lambda_l1_fg * loss_l1_fg
            + lambda_l1_face * loss_l1_face
            + lambda_edge_face * loss_edge_face
            + lambda_detail_face * loss_detail_face
            + lambda_detail_face_luma_dog * loss_detail_face_luma_dog
            + lambda_l1_shoulder_arm * loss_l1_shoulder_arm
            + lambda_edge_shoulder_arm * loss_edge_shoulder_arm
            + lambda_detail_shoulder_arm * loss_detail_shoulder_arm
            + lambda_detail_shoulder_arm_luma_dog * loss_detail_shoulder_arm_luma_dog
            + lambda_detail_upper_torso_luma_dog * loss_detail_upper_torso_luma_dog
            + lambda_detail_upper_torso_core_luma_dog * loss_detail_upper_torso_core_luma_dog
            + lambda_l1_waist * loss_l1_waist
            + lambda_edge_waist * loss_edge_waist
            + lambda_detail_waist * loss_detail_waist
            + lambda_detail_waist_luma_dog * loss_detail_waist_luma_dog
            + lambda_l1_shoulder_focus_dark_outlier * loss_l1_shoulder_focus_dark_outlier
            + lambda_l1_shoulder_focus_bright_outlier * loss_l1_shoulder_focus_bright_outlier
        )
        loss_stageB_semantic, stageB_semantic_stats = _compute_stageB_semantic_parser_loss(
            data,
            render_pkg,
            pipe,
            background,
            fg_mask,
            config.opt,
        )
        base_loss += loss_stageB_semantic
        stageB_semantic_debug_interval = int(config.opt.get('stageB_semantic_debug_interval', 0) or 0)
        if (
            stageB_semantic_debug_interval > 0
            and stageB_semantic_stats.get('enabled', 0.0) > 0.0
            and schedule_iteration % stageB_semantic_debug_interval == 0
        ):
            debug_regions = config.opt.get('stageB_semantic_debug_regions', None)
            if debug_regions is None:
                debug_regions = COMPACT_SEMANTIC_CLASS_NAMES
            elif isinstance(debug_regions, str):
                debug_regions = [part.strip() for part in debug_regions.strip('[]').split(',') if part.strip()]
            cam_id = _camera_id_from_data(data)
            frame_id = getattr(data, 'frame_id', None)
            if torch.is_tensor(frame_id):
                frame_id = int(frame_id.detach().cpu().item()) if frame_id.numel() == 1 else -1
            try:
                frame_id = int(frame_id)
            except (TypeError, ValueError):
                frame_id = -1
            region_parts = []
            for region_name in debug_regions:
                key = _compact_semantic_stat_name(region_name, len(region_parts))
                region_parts.append(
                    (
                        f"{key}:"
                        f"iou={float(stageB_semantic_stats.get(f'compact_{key}_iou', 0.0)):.3f},"
                        f"p={float(stageB_semantic_stats.get(f'compact_{key}_precision', 0.0)):.3f},"
                        f"r={float(stageB_semantic_stats.get(f'compact_{key}_recall', 0.0)):.3f},"
                        f"pred={float(stageB_semantic_stats.get(f'compact_{key}_pred_pixels', 0.0)):.0f},"
                        f"tgt={float(stageB_semantic_stats.get(f'compact_{key}_target_pixels', 0.0)):.0f},"
                        f"prob={float(stageB_semantic_stats.get(f'compact_{key}_mean_prob', 0.0)):.3f}"
                    )
                )
            print(
                "[StageBSemanticDbg] "
                f"iter={schedule_iteration} cam={cam_id if cam_id is not None else -1} frame={frame_id} "
                f"valid={float(stageB_semantic_stats.get('valid_pixels', 0.0)):.0f} "
                f"loss={float(loss_stageB_semantic.detach().item()):.5f} "
                f"body_cloth={float(stageB_semantic_stats.get('body_cloth', 0.0)):.5f} "
                f"compact={float(stageB_semantic_stats.get('compact', 0.0)):.5f} "
                f"ce={float(stageB_semantic_stats.get('compact_ce', 0.0)):.5f} "
                f"parent={float(stageB_semantic_stats.get('parent', 0.0)):.5f} "
                f"exclusive={float(stageB_semantic_stats.get('exclusive', 0.0)):.5f} "
                f"thr={float(stageB_semantic_stats.get('compact_debug_threshold', 0.5)):.2f} | "
                + " | ".join(region_parts),
                flush=True,
            )

        lambda_binding_semantic_adapter_reg = C(
            schedule_iteration,
            config.opt.get(
                'lambda_binding_semantic_asset_adapter_reg',
                config.opt.get('lambda_binding_semantic_adapter_reg', 0.0),
            ),
        )
        if lambda_binding_semantic_adapter_reg > 0.0 and hasattr(scene.gaussians, 'semantic_asset_logits_adapter_regularization'):
            loss_binding_semantic_adapter_reg = scene.gaussians.semantic_asset_logits_adapter_regularization()
            base_loss += lambda_binding_semantic_adapter_reg * loss_binding_semantic_adapter_reg
        else:
            loss_binding_semantic_adapter_reg = torch.tensor(0., device=image.device)

        boundary_loss = torch.tensor(0.).cuda()
        if gate_l1_boundary:
            boundary_loss += lambda_l1_boundary * loss_l1_boundary
        else:
            base_loss += lambda_l1_boundary * loss_l1_boundary

        # perceptual loss
        lambda_perceptual = C(schedule_iteration, config.opt.get('lambda_perceptual', 0.))
        if (
            bool(reliable_view_debug.get('enabled', False))
            and bool(config.opt.get('reliable_view_apply_global_perceptual', False))
        ):
            lambda_perceptual *= reliable_highfreq_weight
        if lambda_perceptual > 0:
            crop_pad = int(config.opt.get('perceptual_crop_pad', 0))
            perceptual_mask = fg_mask
            perceptual_mask_dilate = int(config.opt.get('perceptual_mask_dilate', 0))
            if perceptual_mask_dilate > 1:
                perceptual_mask = _binary_dilate(perceptual_mask, perceptual_mask_dilate).clamp(0.0, 1.0) * fg_mask
            perceptual_mask = _perceptual_stable_region_mask(
                perceptual_mask,
                fg_mask,
                config.opt,
                region_name='global',
                image=image,
                gt_image=loss_gt_image,
            )
            y1, y2, x1, x2 = _foreground_bbox_from_mask(perceptual_mask, padding=crop_pad)
            fg_image = image[:, y1:y2, x1:x2]
            gt_fg_image = loss_gt_image[:, y1:y2, x1:x2]
            if bool(config.opt.get('perceptual_masked', False)):
                perceptual_mask_crop = perceptual_mask[:, y1:y2, x1:x2].to(device=image.device, dtype=image.dtype)
                fg_image = fg_image * perceptual_mask_crop
                gt_fg_image = gt_fg_image * perceptual_mask_crop
            fg_image, gt_fg_image = _ensure_lpips_min_size(
                fg_image,
                gt_fg_image,
                min_size=int(config.opt.get('perceptual_min_size', config.opt.get('lpips_min_size', 32))),
            )
            if fg_image is not None:
                loss_perceptual = loss_fn_vgg(fg_image, gt_fg_image, normalize=True).mean()
                base_loss += lambda_perceptual * loss_perceptual
            else:
                loss_perceptual = torch.tensor(0., device=image.device)
        else:
            loss_perceptual = torch.tensor(0.)

        if lambda_perceptual_face > 0. and face_region_valid:
            face_crop_pad = int(config.opt.get('face_perceptual_crop_pad', 12))
            face_perceptual_mask = face_mask
            face_mask_dilate = int(config.opt.get('face_perceptual_mask_dilate', 0))
            if face_mask_dilate > 1:
                face_perceptual_mask = _binary_dilate(face_perceptual_mask, face_mask_dilate).clamp(0.0, 1.0) * fg_mask
            face_perceptual_mask = _perceptual_stable_region_mask(
                face_perceptual_mask,
                fg_mask,
                config.opt,
                region_name='face',
                image=image,
                gt_image=loss_gt_image,
            )
            fy1, fy2, fx1, fx2 = _foreground_bbox_from_mask(face_perceptual_mask, padding=face_crop_pad)
            face_image = image[:, fy1:fy2, fx1:fx2]
            gt_face_image = loss_gt_image[:, fy1:fy2, fx1:fx2]
            if bool(config.opt.get('face_perceptual_masked', False)):
                face_mask_crop = face_perceptual_mask[:, fy1:fy2, fx1:fx2].to(device=image.device, dtype=image.dtype)
                face_image = face_image * face_mask_crop
                gt_face_image = gt_face_image * face_mask_crop
            face_image, gt_face_image = _ensure_lpips_min_size(
                face_image,
                gt_face_image,
                min_size=int(config.opt.get('face_perceptual_min_size', config.opt.get('lpips_min_size', 32))),
            )
            if face_image is not None:
                loss_perceptual_face = loss_fn_vgg(face_image, gt_face_image, normalize=True).mean()
                base_loss += lambda_perceptual_face * loss_perceptual_face
        if lambda_perceptual_face_patch > 0. and face_detail_region_valid:
            face_patch_mask = _perceptual_stable_region_mask(
                face_detail_mask,
                fg_mask,
                config.opt,
                region_name='face',
                image=image,
                gt_image=loss_gt_image,
            )
            loss_perceptual_face_patch, face_patch_count = _masked_hard_patch_perceptual_loss(
                image,
                loss_gt_image,
                face_patch_mask,
                loss_fn_vgg,
                patch_size=int(_region_opt_value(config.opt, 'face', 'patch_perceptual_size', 56)),
                topk=int(_region_opt_value(config.opt, 'face', 'patch_perceptual_topk', 2)),
                min_size=int(_region_opt_value(config.opt, 'face', 'patch_perceptual_min_size', config.opt.get('lpips_min_size', 32))),
                suppress_radius=int(_region_opt_value(config.opt, 'face', 'patch_perceptual_suppress_radius', 20)),
                masked=bool(_region_opt_value(config.opt, 'face', 'patch_perceptual_masked', True)),
                small_kernel=int(_region_opt_value(config.opt, 'face', 'patch_perceptual_small_kernel', 3)),
                large_kernel=int(_region_opt_value(config.opt, 'face', 'patch_perceptual_large_kernel', 11)),
                focus_power=float(_region_opt_value(config.opt, 'face', 'patch_perceptual_focus_power', 1.4)),
                error_mix=float(_region_opt_value(config.opt, 'face', 'patch_perceptual_error_mix', 0.60)),
                min_mask_coverage=float(_region_opt_value(config.opt, 'face', 'patch_perceptual_min_mask_coverage', config.opt.get('patch_perceptual_min_mask_coverage', 0.0))),
            )
            base_loss += lambda_perceptual_face_patch * loss_perceptual_face_patch

        shoulder_arm_perceptual_used_mask = shoulder_arm_perceptual_region_mask
        if lambda_perceptual_shoulder_arm > 0. and shoulder_arm_perceptual_region_valid:
            shoulder_arm_crop_pad = int(config.opt.get('shoulder_arm_perceptual_crop_pad', 12))
            shoulder_arm_perceptual_mask = shoulder_arm_perceptual_region_mask
            shoulder_arm_mask_dilate = int(config.opt.get('shoulder_arm_perceptual_mask_dilate', 0))
            if shoulder_arm_mask_dilate > 1:
                shoulder_arm_perceptual_mask = _binary_dilate(shoulder_arm_perceptual_mask, shoulder_arm_mask_dilate).clamp(0.0, 1.0) * fg_mask
            shoulder_arm_perceptual_mask = _perceptual_stable_region_mask(
                shoulder_arm_perceptual_mask,
                fg_mask,
                config.opt,
                region_name='shoulder_arm',
                image=image,
                gt_image=loss_gt_image,
            )
            shoulder_arm_perceptual_used_mask = shoulder_arm_perceptual_mask
            if shoulder_arm_perceptual_mask.sum().item() > 0:
                sy1, sy2, sx1, sx2 = _foreground_bbox_from_mask(shoulder_arm_perceptual_mask, padding=shoulder_arm_crop_pad)
                shoulder_arm_image = image[:, sy1:sy2, sx1:sx2]
                gt_shoulder_arm_image = loss_gt_image[:, sy1:sy2, sx1:sx2]
                if bool(config.opt.get('shoulder_arm_perceptual_masked', False)):
                    shoulder_arm_mask_crop = shoulder_arm_perceptual_mask[:, sy1:sy2, sx1:sx2].to(device=image.device, dtype=image.dtype)
                    shoulder_arm_image = shoulder_arm_image * shoulder_arm_mask_crop
                    gt_shoulder_arm_image = gt_shoulder_arm_image * shoulder_arm_mask_crop
                shoulder_arm_image, gt_shoulder_arm_image = _ensure_lpips_min_size(
                    shoulder_arm_image,
                    gt_shoulder_arm_image,
                    min_size=int(config.opt.get('shoulder_arm_perceptual_min_size', config.opt.get('lpips_min_size', 32))),
                )
                if shoulder_arm_image is not None:
                    loss_perceptual_shoulder_arm = loss_fn_vgg(shoulder_arm_image, gt_shoulder_arm_image, normalize=True).mean()
                    base_loss += lambda_perceptual_shoulder_arm * loss_perceptual_shoulder_arm
        if lambda_perceptual_shoulder_arm_patch > 0. and shoulder_arm_detail_region_valid:
            shoulder_arm_patch_mask = _perceptual_stable_region_mask(
                shoulder_arm_detail_mask,
                fg_mask,
                config.opt,
                region_name='shoulder_arm',
                image=image,
                gt_image=loss_gt_image,
            )
            loss_perceptual_shoulder_arm_patch, shoulder_arm_patch_count = _masked_hard_patch_perceptual_loss(
                image,
                loss_gt_image,
                shoulder_arm_patch_mask,
                loss_fn_vgg,
                patch_size=int(_region_opt_value(config.opt, 'shoulder_arm', 'patch_perceptual_size', 64)),
                topk=int(_region_opt_value(config.opt, 'shoulder_arm', 'patch_perceptual_topk', 1)),
                min_size=int(_region_opt_value(config.opt, 'shoulder_arm', 'patch_perceptual_min_size', config.opt.get('lpips_min_size', 32))),
                suppress_radius=int(_region_opt_value(config.opt, 'shoulder_arm', 'patch_perceptual_suppress_radius', 24)),
                masked=bool(_region_opt_value(config.opt, 'shoulder_arm', 'patch_perceptual_masked', True)),
                small_kernel=int(_region_opt_value(config.opt, 'shoulder_arm', 'patch_perceptual_small_kernel', 3)),
                large_kernel=int(_region_opt_value(config.opt, 'shoulder_arm', 'patch_perceptual_large_kernel', 11)),
                focus_power=float(_region_opt_value(config.opt, 'shoulder_arm', 'patch_perceptual_focus_power', 1.35)),
                error_mix=float(_region_opt_value(config.opt, 'shoulder_arm', 'patch_perceptual_error_mix', 0.65)),
                min_mask_coverage=float(_region_opt_value(config.opt, 'shoulder_arm', 'patch_perceptual_min_mask_coverage', config.opt.get('patch_perceptual_min_mask_coverage', 0.0))),
            )
            base_loss += lambda_perceptual_shoulder_arm_patch * loss_perceptual_shoulder_arm_patch
        if lambda_perceptual_upper_torso_patch > 0. and upper_torso_detail_region_valid:
            upper_torso_patch_mask = _perceptual_stable_region_mask(
                upper_torso_detail_mask,
                fg_mask,
                config.opt,
                region_name='upper_torso',
                image=image,
                gt_image=loss_gt_image,
            )
            loss_perceptual_upper_torso_patch, upper_torso_patch_count = _masked_hard_patch_perceptual_loss(
                image,
                loss_gt_image,
                upper_torso_patch_mask,
                loss_fn_vgg,
                patch_size=int(_region_opt_value(config.opt, 'upper_torso', 'patch_perceptual_size', 56)),
                topk=int(_region_opt_value(config.opt, 'upper_torso', 'patch_perceptual_topk', 1)),
                min_size=int(_region_opt_value(config.opt, 'upper_torso', 'patch_perceptual_min_size', config.opt.get('lpips_min_size', 32))),
                suppress_radius=int(_region_opt_value(config.opt, 'upper_torso', 'patch_perceptual_suppress_radius', 20)),
                masked=bool(_region_opt_value(config.opt, 'upper_torso', 'patch_perceptual_masked', True)),
                small_kernel=int(_region_opt_value(config.opt, 'upper_torso', 'patch_perceptual_small_kernel', 3)),
                large_kernel=int(_region_opt_value(config.opt, 'upper_torso', 'patch_perceptual_large_kernel', 11)),
                focus_power=float(_region_opt_value(config.opt, 'upper_torso', 'patch_perceptual_focus_power', 1.32)),
                error_mix=float(_region_opt_value(config.opt, 'upper_torso', 'patch_perceptual_error_mix', 0.58)),
                min_mask_coverage=float(_region_opt_value(config.opt, 'upper_torso', 'patch_perceptual_min_mask_coverage', config.opt.get('patch_perceptual_min_mask_coverage', 0.0))),
            )
            upper_torso_perceptual_used_mask = upper_torso_patch_mask
            base_loss += lambda_perceptual_upper_torso_patch * loss_perceptual_upper_torso_patch
        if lambda_perceptual_upper_torso_core_patch > 0. and upper_torso_core_region_valid:
            upper_torso_core_patch_mask = _perceptual_stable_region_mask(
                upper_torso_core_mask,
                fg_mask,
                config.opt,
                region_name='upper_torso_core',
                image=image,
                gt_image=loss_gt_image,
            )
            loss_perceptual_upper_torso_core_patch, upper_torso_core_patch_count = _masked_hard_patch_perceptual_loss(
                image,
                loss_gt_image,
                upper_torso_core_patch_mask,
                loss_fn_vgg,
                patch_size=int(_region_opt_value(config.opt, 'upper_torso_core', 'patch_perceptual_size', 52)),
                topk=int(_region_opt_value(config.opt, 'upper_torso_core', 'patch_perceptual_topk', 2)),
                min_size=int(_region_opt_value(config.opt, 'upper_torso_core', 'patch_perceptual_min_size', config.opt.get('lpips_min_size', 32))),
                suppress_radius=int(_region_opt_value(config.opt, 'upper_torso_core', 'patch_perceptual_suppress_radius', 18)),
                masked=bool(_region_opt_value(config.opt, 'upper_torso_core', 'patch_perceptual_masked', True)),
                small_kernel=int(_region_opt_value(config.opt, 'upper_torso_core', 'patch_perceptual_small_kernel', 3)),
                large_kernel=int(_region_opt_value(config.opt, 'upper_torso_core', 'patch_perceptual_large_kernel', 11)),
                focus_power=float(_region_opt_value(config.opt, 'upper_torso_core', 'patch_perceptual_focus_power', 1.40)),
                error_mix=float(_region_opt_value(config.opt, 'upper_torso_core', 'patch_perceptual_error_mix', 0.60)),
                min_mask_coverage=float(_region_opt_value(config.opt, 'upper_torso_core', 'patch_perceptual_min_mask_coverage', config.opt.get('patch_perceptual_min_mask_coverage', 0.0))),
            )
            upper_torso_core_perceptual_used_mask = upper_torso_core_patch_mask
            base_loss += lambda_perceptual_upper_torso_core_patch * loss_perceptual_upper_torso_core_patch

        if lambda_perceptual_waist > 0. and waist_region_valid:
            waist_perceptual_mask = waist_mask
            waist_mask_dilate = int(config.opt.get('waist_perceptual_mask_dilate', 0))
            if waist_mask_dilate > 1:
                waist_perceptual_mask = _binary_dilate(waist_perceptual_mask, waist_mask_dilate).clamp(0.0, 1.0) * fg_mask
            waist_perceptual_mask = _perceptual_stable_region_mask(
                waist_perceptual_mask,
                fg_mask,
                config.opt,
                region_name='waist',
                image=image,
                gt_image=loss_gt_image,
            )
            waist_perceptual_used_mask = waist_perceptual_mask
            waist_perceptual_region_valid = waist_perceptual_mask.sum().item() >= waist_perceptual_region_min_pixels
            if waist_perceptual_region_valid:
                waist_crop_pad = int(config.opt.get('waist_perceptual_crop_pad', 8))
                wy1, wy2, wx1, wx2 = _foreground_bbox_from_mask(waist_perceptual_mask, padding=waist_crop_pad)
                waist_image = image[:, wy1:wy2, wx1:wx2]
                gt_waist_image = loss_gt_image[:, wy1:wy2, wx1:wx2]
                if bool(config.opt.get('waist_perceptual_masked', False)):
                    waist_mask_crop = waist_perceptual_mask[:, wy1:wy2, wx1:wx2].to(device=image.device, dtype=image.dtype)
                    waist_image = waist_image * waist_mask_crop
                    gt_waist_image = gt_waist_image * waist_mask_crop
                waist_image, gt_waist_image = _ensure_lpips_min_size(
                    waist_image,
                    gt_waist_image,
                    min_size=int(config.opt.get('waist_perceptual_min_size', config.opt.get('lpips_min_size', 32))),
                )
                if waist_image is not None:
                    loss_perceptual_waist = loss_fn_vgg(waist_image, gt_waist_image, normalize=True).mean()
                    base_loss += lambda_perceptual_waist * loss_perceptual_waist
        if lambda_perceptual_waist_patch > 0. and waist_detail_region_valid:
            waist_patch_mask = _perceptual_stable_region_mask(
                waist_detail_mask,
                fg_mask,
                config.opt,
                region_name='waist',
                image=image,
                gt_image=loss_gt_image,
            )
            loss_perceptual_waist_patch, waist_patch_count = _masked_hard_patch_perceptual_loss(
                image,
                loss_gt_image,
                waist_patch_mask,
                loss_fn_vgg,
                patch_size=int(_region_opt_value(config.opt, 'waist', 'patch_perceptual_size', 56)),
                topk=int(_region_opt_value(config.opt, 'waist', 'patch_perceptual_topk', 1)),
                min_size=int(_region_opt_value(config.opt, 'waist', 'patch_perceptual_min_size', config.opt.get('lpips_min_size', 32))),
                suppress_radius=int(_region_opt_value(config.opt, 'waist', 'patch_perceptual_suppress_radius', 20)),
                masked=bool(_region_opt_value(config.opt, 'waist', 'patch_perceptual_masked', True)),
                small_kernel=int(_region_opt_value(config.opt, 'waist', 'patch_perceptual_small_kernel', 3)),
                large_kernel=int(_region_opt_value(config.opt, 'waist', 'patch_perceptual_large_kernel', 11)),
                focus_power=float(_region_opt_value(config.opt, 'waist', 'patch_perceptual_focus_power', 1.30)),
                error_mix=float(_region_opt_value(config.opt, 'waist', 'patch_perceptual_error_mix', 0.55)),
                min_mask_coverage=float(_region_opt_value(config.opt, 'waist', 'patch_perceptual_min_mask_coverage', config.opt.get('patch_perceptual_min_mask_coverage', 0.0))),
            )
            base_loss += lambda_perceptual_waist_patch * loss_perceptual_waist_patch

        # mask loss
        loss_mask_boundary = torch.tensor(0.).cuda()
        loss_mask_boundary_hard = torch.tensor(0.).cuda()
        loss_mask_shoulder_arm_boundary_hard = torch.tensor(0.).cuda()
        loss_mask_upper_torso_boundary_hard = torch.tensor(0.).cuda()
        loss_mask_shoulder_arm_region_hard = torch.tensor(0.).cuda()
        loss_mask_shoulder_arm_disagreement_hard = torch.tensor(0.).cuda()
        loss_mask_shoulder_focus_small_fp_hard = torch.tensor(0.).cuda()
        loss_mask_shoulder_focus_small_fn_hard = torch.tensor(0.).cuda()
        loss_silhouette_outer = torch.tensor(0.).cuda()
        loss_silhouette_outer_shell = torch.tensor(0.).cuda()
        loss_silhouette_head_outer_shell = torch.tensor(0.).cuda()
        loss_silhouette_shoulder_arm_outer_shell = torch.tensor(0.).cuda()
        loss_silhouette_upper_torso_outer_shell = torch.tensor(0.).cuda()
        loss_silhouette_outer_spike = torch.tensor(0.).cuda()
        loss_silhouette_outer_fragment = torch.tensor(0.).cuda()
        loss_silhouette_outer_bead = torch.tensor(0.).cuda()
        loss_silhouette_outer_chain = torch.tensor(0.).cuda()
        loss_silhouette_arm_stipple = torch.tensor(0.).cuda()
        loss_silhouette_arm_tail = torch.tensor(0.).cuda()
        loss_silhouette_arm_fringe = torch.tensor(0.).cuda()
        loss_silhouette_arm_attached_fragment = torch.tensor(0.).cuda()
        loss_silhouette_shoulder_attached_fragment = torch.tensor(0.).cuda()
        loss_silhouette_arm_notch = torch.tensor(0.).cuda()
        loss_silhouette_arm_hole = torch.tensor(0.).cuda()
        loss_silhouette_arm_gap = torch.tensor(0.).cuda()
        loss_silhouette_shoulder_bead = torch.tensor(0.).cuda()
        loss_silhouette_shoulder_chain = torch.tensor(0.).cuda()
        loss_silhouette_shoulder_hole = torch.tensor(0.).cuda()
        loss_silhouette_shoulder_gap = torch.tensor(0.).cuda()
        loss_silhouette_shoulder_pinhole = torch.tensor(0.).cuda()
        loss_silhouette_inner = torch.tensor(0.).cuda()
        if not use_mask:
            loss_mask = torch.tensor(0.).cuda()
        else:
            if config.opt.mask_loss_type == 'bce':
                loss_mask = F.binary_cross_entropy(opacity_bce, gt_mask)
                if lambda_mask_boundary > 0. and boundary_mask.sum().item() > 0:
                    boundary_term = F.binary_cross_entropy(opacity_bce, gt_mask_boundary, reduction='none')
                    loss_mask_boundary = (boundary_term * boundary_mask).sum() / boundary_mask.sum().clamp_min(1.0)
            elif config.opt.mask_loss_type == 'l1':
                loss_mask = F.l1_loss(opacity, gt_mask)
                if lambda_mask_boundary > 0. and boundary_mask.sum().item() > 0:
                    boundary_term = torch.abs(opacity - gt_mask_boundary)
                    loss_mask_boundary = (boundary_term * boundary_mask).sum() / boundary_mask.sum().clamp_min(1.0)
            else:
                raise ValueError

            if lambda_mask_boundary_hard > 0. and boundary_mask.sum().item() > 0:
                loss_mask_boundary_hard = _masked_binary_cross_entropy(opacity_bce, gt_mask_boundary, boundary_mask)
            if lambda_mask_shoulder_arm_boundary_hard > 0. and shoulder_arm_boundary_supervision_mask.sum().item() > 0:
                loss_mask_shoulder_arm_boundary_hard = _masked_binary_cross_entropy(opacity_bce, arm_boundary_target, shoulder_arm_boundary_supervision_mask)
            if lambda_mask_upper_torso_boundary_hard > 0. and upper_torso_boundary_supervision_mask.sum().item() > 0:
                loss_mask_upper_torso_boundary_hard = _masked_binary_cross_entropy(opacity_bce, upper_torso_boundary_target, upper_torso_boundary_supervision_mask)
            if lambda_mask_shoulder_arm_region_hard > 0. and shoulder_arm_region_hard_supervision_mask.sum().item() > 0:
                loss_mask_shoulder_arm_region_hard = _masked_binary_cross_entropy(opacity_bce, arm_region_target, shoulder_arm_region_hard_supervision_mask)
            if lambda_mask_shoulder_arm_disagreement_hard > 0. and shoulder_arm_disagreement_supervision_mask.sum().item() > 0:
                loss_mask_shoulder_arm_disagreement_hard = _masked_binary_cross_entropy(opacity_bce, arm_region_target, shoulder_arm_disagreement_supervision_mask)
            if lambda_silhouette_shoulder_arm_outer_shell > 0. and shoulder_arm_outer_shell_supervision_mask.sum().item() > 0:
                loss_silhouette_shoulder_arm_outer_shell = _masked_binary_cross_entropy(opacity_bce, 0.0, shoulder_arm_outer_shell_supervision_mask)
            if lambda_silhouette_upper_torso_outer_shell > 0. and upper_torso_outer_shell_supervision_mask.sum().item() > 0:
                loss_silhouette_upper_torso_outer_shell = _masked_binary_cross_entropy(opacity_bce, 0.0, upper_torso_outer_shell_supervision_mask)
            shoulder_small_region_mode = str(config.opt.get('shoulder_focus_small_disagreement_region_mode', 'focus'))
            shoulder_small_region = shoulder_focus_mask
            shoulder_small_region_valid = shoulder_focus_region_valid
            if shoulder_small_region_mode == 'collar':
                shoulder_small_region = shoulder_collar_mask
                shoulder_small_region_valid = shoulder_collar_region_valid
            if shoulder_small_region_valid and (lambda_mask_shoulder_focus_small_fp_hard > 0. or lambda_mask_shoulder_focus_small_fn_hard > 0.):
                shoulder_focus_region = shoulder_small_region
                shoulder_focus_region_dilate = int(config.opt.get('shoulder_focus_small_disagreement_region_dilate', 5))
                if shoulder_focus_region_dilate > 1:
                    shoulder_focus_region = _binary_dilate(shoulder_focus_region, shoulder_focus_region_dilate).clamp(0.0, 1.0)

                shoulder_focus_target = hard_gt_mask if hard_gt_mask is not None else gt_mask
                shoulder_focus_target_binary = (shoulder_focus_target >= 0.5).float()
                shoulder_focus_outer_region = _foreground_outer_shell_mask(
                    shoulder_focus_target_binary,
                    int(config.opt.get('shoulder_focus_small_fp_outer_shell_start_width', 1)),
                    int(config.opt.get('shoulder_focus_small_fp_outer_shell_end_width', 17)),
                ) * shoulder_focus_region
                shoulder_focus_inner_region = _foreground_inner_ring_mask(
                    shoulder_focus_target_binary,
                    int(config.opt.get('shoulder_focus_small_fn_inner_ring_width', 5)),
                ) * shoulder_focus_region * shoulder_focus_target_binary
                shoulder_focus_inner_region_dilate = int(config.opt.get('shoulder_focus_small_fn_region_dilate', 11))
                if shoulder_focus_inner_region_dilate > 1 and shoulder_focus_inner_region.sum().item() > 0:
                    shoulder_focus_inner_region = _binary_dilate(shoulder_focus_inner_region, shoulder_focus_inner_region_dilate).clamp(0.0, 1.0) * shoulder_focus_region * shoulder_focus_target_binary

                shoulder_small_fp_mask, shoulder_small_fn_mask = _foreground_small_disagreement_component_masks(
                    opacity_bce,
                    shoulder_focus_target,
                    region_mask=shoulder_focus_region,
                    fp_region_mask=shoulder_focus_outer_region,
                    fn_region_mask=shoulder_focus_inner_region,
                    pred_threshold=float(config.opt.get('shoulder_focus_small_disagreement_opacity_threshold', 0.075)),
                    target_threshold=float(config.opt.get('shoulder_focus_small_disagreement_target_threshold', 0.5)),
                    fp_area_min=int(config.opt.get('shoulder_focus_small_fp_area_min', 1)),
                    fp_area_max=int(config.opt.get('shoulder_focus_small_fp_area_max', 40)),
                    fn_area_min=int(config.opt.get('shoulder_focus_small_fn_area_min', 1)),
                    fn_area_max=int(config.opt.get('shoulder_focus_small_fn_area_max', 40)),
                    fp_power=float(config.opt.get('shoulder_focus_small_fp_power', 1.6)),
                    fn_power=float(config.opt.get('shoulder_focus_small_fn_power', 1.6)),
                )
                if lambda_mask_shoulder_focus_small_fp_hard > 0. and shoulder_small_fp_mask is not None and shoulder_small_fp_mask.sum().item() > 0:
                    loss_mask_shoulder_focus_small_fp_hard = _masked_binary_cross_entropy(opacity_bce, 0.0, shoulder_small_fp_mask)
                if lambda_mask_shoulder_focus_small_fn_hard > 0. and shoulder_small_fn_mask is not None and shoulder_small_fn_mask.sum().item() > 0:
                    loss_mask_shoulder_focus_small_fn_hard = _masked_binary_cross_entropy(opacity_bce, 1.0, shoulder_small_fn_mask)
            if lambda_silhouette_outer > 0. and outer_ring_mask.sum().item() > 0:
                loss_silhouette_outer = _masked_binary_cross_entropy(opacity_bce, 0.0, outer_ring_mask)
            if lambda_silhouette_outer_shell > 0. and outer_shell_mask.sum().item() > 0:
                loss_silhouette_outer_shell = _masked_binary_cross_entropy(opacity_bce, 0.0, outer_shell_mask)
            if lambda_silhouette_head_outer_shell > 0. and outer_shell_mask.sum().item() > 0 and head_outer_region_valid:
                head_outer_shell_mask = outer_shell_mask * head_outer_region_mask
                if head_outer_shell_mask.sum().item() > 0:
                    loss_silhouette_head_outer_shell = _masked_binary_cross_entropy(opacity_bce, 0.0, head_outer_shell_mask)
            if lambda_silhouette_outer_spike > 0. and outer_shell_mask.sum().item() > 0:
                outer_spike_mask = _foreground_outer_spike_mask(
                    opacity_bce,
                    outer_shell_mask,
                    support_kernel_size=int(config.opt.get('silhouette_outer_spike_support_kernel_size', 7)),
                    opacity_threshold=float(config.opt.get('silhouette_outer_spike_opacity_threshold', 0.08)),
                    support_threshold=float(config.opt.get('silhouette_outer_spike_support_threshold', 0.35)),
                    power=float(config.opt.get('silhouette_outer_spike_power', 1.0)),
                )
                if outer_spike_mask is not None and outer_spike_mask.sum().item() > 0:
                    loss_silhouette_outer_spike = _masked_binary_cross_entropy(opacity_bce, 0.0, outer_spike_mask)
            if lambda_silhouette_outer_fragment > 0. and outer_shell_mask.sum().item() > 0 and shoulder_arm_region_valid:
                fragment_region = shoulder_arm_mask
                fragment_region_dilate = int(config.opt.get('silhouette_outer_fragment_region_dilate', 13))
                if fragment_region_dilate > 1:
                    fragment_region = _binary_dilate(fragment_region, fragment_region_dilate).clamp(0.0, 1.0)
                outer_fragment_mask = _foreground_outer_fragment_mask(
                    opacity_bce,
                    outer_shell_mask,
                    region_mask=fragment_region,
                    opacity_threshold=float(config.opt.get('silhouette_outer_fragment_opacity_threshold', 0.08)),
                    component_area_min=int(config.opt.get('silhouette_outer_fragment_area_min', 2)),
                    component_area_max=int(config.opt.get('silhouette_outer_fragment_area_max', 160)),
                    fill_ratio_max=float(config.opt.get('silhouette_outer_fragment_fill_ratio_max', 0.60)),
                )
                if outer_fragment_mask is not None and outer_fragment_mask.sum().item() > 0:
                    loss_silhouette_outer_fragment = _masked_binary_cross_entropy(opacity_bce, 0.0, outer_fragment_mask)
            if lambda_silhouette_outer_bead > 0. and outer_shell_mask.sum().item() > 0 and shoulder_arm_region_valid:
                bead_region = shoulder_arm_mask
                bead_region_dilate = int(config.opt.get('silhouette_outer_bead_region_dilate', 17))
                if bead_region_dilate > 1:
                    bead_region = _binary_dilate(bead_region, bead_region_dilate).clamp(0.0, 1.0)
                outer_bead_mask = _foreground_outer_bead_mask(
                    opacity_bce,
                    outer_shell_mask,
                    region_mask=bead_region,
                    opacity_threshold=float(config.opt.get('silhouette_outer_bead_opacity_threshold', 0.08)),
                    opening_kernel_size=int(config.opt.get('silhouette_outer_bead_opening_kernel_size', 5)),
                    support_kernel_size=int(config.opt.get('silhouette_outer_bead_support_kernel_size', 7)),
                    support_threshold=float(config.opt.get('silhouette_outer_bead_support_threshold', 0.55)),
                    power=float(config.opt.get('silhouette_outer_bead_power', 1.0)),
                )
                if outer_bead_mask is not None and outer_bead_mask.sum().item() > 0:
                    loss_silhouette_outer_bead = _masked_binary_cross_entropy(opacity_bce, 0.0, outer_bead_mask)
            if lambda_silhouette_outer_chain > 0. and outer_shell_mask.sum().item() > 0 and shoulder_arm_region_valid:
                chain_region = shoulder_arm_mask
                chain_region_dilate = int(config.opt.get('silhouette_outer_chain_region_dilate', 19))
                if chain_region_dilate > 1:
                    chain_region = _binary_dilate(chain_region, chain_region_dilate).clamp(0.0, 1.0)
                outer_chain_mask = _foreground_outer_chain_mask(
                    opacity_bce,
                    outer_shell_mask,
                    region_mask=chain_region,
                    opacity_threshold=float(config.opt.get('silhouette_outer_chain_opacity_threshold', 0.08)),
                    support_kernel_size=int(config.opt.get('silhouette_outer_chain_support_kernel_size', 7)),
                    seed_support_threshold=float(config.opt.get('silhouette_outer_chain_seed_support_threshold', 0.65)),
                    propagate_support_threshold=float(config.opt.get('silhouette_outer_chain_propagate_support_threshold', 0.40)),
                    anchor_weight_threshold=float(config.opt.get('silhouette_outer_chain_anchor_weight_threshold', 0.70)),
                    max_steps=int(config.opt.get('silhouette_outer_chain_max_steps', 16)),
                    power=float(config.opt.get('silhouette_outer_chain_power', 1.0)),
                )
                if outer_chain_mask is not None and outer_chain_mask.sum().item() > 0:
                    loss_silhouette_outer_chain = _masked_binary_cross_entropy(opacity_bce, 0.0, outer_chain_mask)
            if shoulder_arm_region_valid and any(
                value > 0.
                for value in (
                    lambda_silhouette_arm_stipple,
                    lambda_silhouette_arm_tail,
                    lambda_silhouette_arm_fringe,
                    lambda_silhouette_arm_attached_fragment,
                    lambda_silhouette_arm_notch,
                    lambda_silhouette_arm_hole,
                    lambda_silhouette_arm_gap,
                )
            ):
                roughness_region = shoulder_arm_mask
                roughness_region_dilate = int(config.opt.get('silhouette_arm_roughness_region_dilate', 19))
                if roughness_region_dilate > 1:
                    roughness_region = _binary_dilate(roughness_region, roughness_region_dilate).clamp(0.0, 1.0)
                arm_stipple_mask, arm_notch_mask = None, None
                if lambda_silhouette_arm_stipple > 0. or lambda_silhouette_arm_notch > 0.:
                    arm_stipple_mask, arm_notch_mask = _foreground_arm_boundary_roughness_masks(
                        opacity_bce,
                        fg_mask,
                        region_mask=roughness_region,
                        opacity_threshold=float(config.opt.get('silhouette_arm_roughness_opacity_threshold', 0.06)),
                        boundary_band_kernel_size=int(config.opt.get('silhouette_arm_roughness_band_kernel_size', 5)),
                        opening_kernel_size=int(config.opt.get('silhouette_arm_roughness_open_kernel_size', 3)),
                        closing_kernel_size=int(config.opt.get('silhouette_arm_roughness_close_kernel_size', 5)),
                        stipple_area_min=int(config.opt.get('silhouette_arm_stipple_area_min', 1)),
                        stipple_area_max=int(config.opt.get('silhouette_arm_stipple_area_max', 24)),
                        notch_area_min=int(config.opt.get('silhouette_arm_notch_area_min', 1)),
                        notch_area_max=int(config.opt.get('silhouette_arm_notch_area_max', 24)),
                        stipple_power=float(config.opt.get('silhouette_arm_stipple_power', 1.0)),
                        notch_power=float(config.opt.get('silhouette_arm_notch_power', 1.0)),
                    )
                if lambda_silhouette_arm_stipple > 0. and arm_stipple_mask is not None and arm_stipple_mask.sum().item() > 0:
                    loss_silhouette_arm_stipple = _masked_binary_cross_entropy(opacity_bce, 0.0, arm_stipple_mask)
                if lambda_silhouette_arm_tail > 0.:
                    tail_region = roughness_region
                    tail_region_dilate = int(config.opt.get('silhouette_arm_tail_region_dilate', 21))
                    if tail_region_dilate > 1:
                        tail_region = _binary_dilate(tail_region, tail_region_dilate).clamp(0.0, 1.0)
                    arm_tail_mask = _foreground_arm_boundary_tail_mask(
                        opacity_bce,
                        fg_mask,
                        region_mask=tail_region,
                        opacity_threshold=float(config.opt.get('silhouette_arm_tail_opacity_threshold', 0.06)),
                        boundary_band_kernel_size=int(config.opt.get('silhouette_arm_tail_boundary_kernel_size', 5)),
                        outer_shell_start_width=int(config.opt.get('silhouette_arm_tail_outer_shell_start_width', 1)),
                        outer_shell_end_width=int(config.opt.get('silhouette_arm_tail_outer_shell_end_width', 11)),
                        opening_kernel_size=int(config.opt.get('silhouette_arm_tail_open_kernel_size', 7)),
                        closing_kernel_size=int(config.opt.get('silhouette_arm_tail_close_kernel_size', 5)),
                        support_kernel_size=int(config.opt.get('silhouette_arm_tail_support_kernel_size', 9)),
                        support_threshold=float(config.opt.get('silhouette_arm_tail_support_threshold', 0.72)),
                        component_area_min=int(config.opt.get('silhouette_arm_tail_area_min', 2)),
                        component_area_max=int(config.opt.get('silhouette_arm_tail_area_max', 36)),
                        aspect_ratio_min=float(config.opt.get('silhouette_arm_tail_aspect_ratio_min', 1.25)),
                        fill_ratio_max=float(config.opt.get('silhouette_arm_tail_fill_ratio_max', 0.62)),
                        touch_ratio_min=float(config.opt.get('silhouette_arm_tail_touch_ratio_min', 0.18)),
                        power=float(config.opt.get('silhouette_arm_tail_power', 1.0)),
                    )
                    if arm_tail_mask is not None and arm_tail_mask.sum().item() > 0:
                        loss_silhouette_arm_tail = _masked_binary_cross_entropy(opacity_bce, 0.0, arm_tail_mask)
                if lambda_silhouette_arm_fringe > 0.:
                    fringe_region = roughness_region
                    fringe_region_dilate = int(config.opt.get('silhouette_arm_fringe_region_dilate', 23))
                    if fringe_region_dilate > 1:
                        fringe_region = _binary_dilate(fringe_region, fringe_region_dilate).clamp(0.0, 1.0)
                    arm_fringe_mask = _foreground_arm_boundary_fringe_mask(
                        opacity_bce,
                        fg_mask,
                        region_mask=fringe_region,
                        opacity_threshold=float(config.opt.get('silhouette_arm_fringe_opacity_threshold', 0.055)),
                        boundary_band_kernel_size=int(config.opt.get('silhouette_arm_fringe_boundary_kernel_size', 5)),
                        outer_shell_start_width=int(config.opt.get('silhouette_arm_fringe_outer_shell_start_width', 1)),
                        outer_shell_end_width=int(config.opt.get('silhouette_arm_fringe_outer_shell_end_width', 13)),
                        opening_kernel_size=int(config.opt.get('silhouette_arm_fringe_open_kernel_size', 9)),
                        closing_kernel_size=int(config.opt.get('silhouette_arm_fringe_close_kernel_size', 7)),
                        support_kernel_size=int(config.opt.get('silhouette_arm_fringe_support_kernel_size', 11)),
                        support_threshold=float(config.opt.get('silhouette_arm_fringe_support_threshold', 0.78)),
                        component_area_min=int(config.opt.get('silhouette_arm_fringe_area_min', 4)),
                        component_area_max=int(config.opt.get('silhouette_arm_fringe_area_max', 120)),
                        fill_ratio_max=float(config.opt.get('silhouette_arm_fringe_fill_ratio_max', 0.82)),
                        touch_ratio_min=float(config.opt.get('silhouette_arm_fringe_touch_ratio_min', 0.22)),
                        power=float(config.opt.get('silhouette_arm_fringe_power', 1.0)),
                    )
                    if arm_fringe_mask is not None and arm_fringe_mask.sum().item() > 0:
                        loss_silhouette_arm_fringe = _masked_binary_cross_entropy(opacity_bce, 0.0, arm_fringe_mask)
                if lambda_silhouette_arm_attached_fragment > 0.:
                    attached_region = roughness_region
                    attached_region_dilate = int(config.opt.get('silhouette_arm_attached_fragment_region_dilate', 17))
                    if attached_region_dilate > 1:
                        attached_region = _binary_dilate(attached_region, attached_region_dilate).clamp(0.0, 1.0)
                    arm_attached_fragment_mask = _foreground_arm_boundary_attached_fragment_mask(
                        opacity_bce,
                        fg_mask,
                        region_mask=attached_region,
                        opacity_threshold=float(config.opt.get('silhouette_arm_attached_fragment_opacity_threshold', 0.055)),
                        boundary_band_kernel_size=int(config.opt.get('silhouette_arm_attached_fragment_boundary_kernel_size', 5)),
                        outer_shell_start_width=int(config.opt.get('silhouette_arm_attached_fragment_outer_shell_start_width', 1)),
                        outer_shell_end_width=int(config.opt.get('silhouette_arm_attached_fragment_outer_shell_end_width', 11)),
                        component_area_min=int(config.opt.get('silhouette_arm_attached_fragment_area_min', 1)),
                        component_area_max=int(config.opt.get('silhouette_arm_attached_fragment_area_max', 64)),
                        fill_ratio_max=float(config.opt.get('silhouette_arm_attached_fragment_fill_ratio_max', 0.88)),
                        touch_ratio_min=float(config.opt.get('silhouette_arm_attached_fragment_touch_ratio_min', 0.18)),
                        power=float(config.opt.get('silhouette_arm_attached_fragment_power', 1.0)),
                    )
                    if arm_attached_fragment_mask is not None and arm_attached_fragment_mask.sum().item() > 0:
                        loss_silhouette_arm_attached_fragment = _masked_binary_cross_entropy(opacity_bce, 0.0, arm_attached_fragment_mask)
                if lambda_silhouette_arm_notch > 0. and arm_notch_mask is not None and arm_notch_mask.sum().item() > 0:
                    loss_silhouette_arm_notch = _masked_binary_cross_entropy(opacity_bce, 1.0, arm_notch_mask)
                if lambda_silhouette_arm_hole > 0. or lambda_silhouette_arm_gap > 0.:
                    hole_gap_region = roughness_region
                    hole_gap_region_dilate = int(config.opt.get('silhouette_arm_hole_region_dilate', 0))
                    if hole_gap_region_dilate > 1:
                        hole_gap_region = _binary_dilate(hole_gap_region, hole_gap_region_dilate).clamp(0.0, 1.0)
                    arm_hole_mask, arm_gap_mask = _foreground_arm_hole_and_gap_masks(
                        opacity_bce,
                        fg_mask,
                        region_mask=hole_gap_region,
                        opacity_threshold=float(config.opt.get('silhouette_arm_hole_opacity_threshold', config.opt.get('silhouette_arm_gap_opacity_threshold', 0.08))),
                        gap_opacity_threshold=float(config.opt.get('silhouette_arm_gap_opacity_threshold', config.opt.get('silhouette_arm_hole_opacity_threshold', 0.08))),
                        inner_band_kernel_size=int(config.opt.get('silhouette_arm_hole_inner_band_kernel_size', config.opt.get('silhouette_arm_gap_inner_band_kernel_size', 5))),
                        inner_region_dilate=int(config.opt.get('silhouette_arm_hole_inner_region_dilate', config.opt.get('silhouette_arm_gap_inner_region_dilate', 11))),
                        closing_kernel_size=int(config.opt.get('silhouette_arm_gap_close_kernel_size', 9)),
                        hole_area_min=int(config.opt.get('silhouette_arm_hole_area_min', 1)),
                        hole_area_max=int(config.opt.get('silhouette_arm_hole_area_max', 48)),
                        gap_area_min=int(config.opt.get('silhouette_arm_gap_area_min', 2)),
                        gap_area_max=int(config.opt.get('silhouette_arm_gap_area_max', 96)),
                        hole_power=float(config.opt.get('silhouette_arm_hole_power', 1.0)),
                        gap_power=float(config.opt.get('silhouette_arm_gap_power', 1.0)),
                    )
                if lambda_silhouette_arm_hole > 0. and arm_hole_mask is not None and arm_hole_mask.sum().item() > 0:
                    loss_silhouette_arm_hole = _masked_binary_cross_entropy(opacity_bce, 1.0, arm_hole_mask)
                if lambda_silhouette_arm_gap > 0. and arm_gap_mask is not None and arm_gap_mask.sum().item() > 0:
                    loss_silhouette_arm_gap = _masked_binary_cross_entropy(opacity_bce, 1.0, arm_gap_mask)
            shoulder_cleanup_region_mode = str(config.opt.get('shoulder_cleanup_region_mode', 'focus'))
            shoulder_cleanup_region_base = shoulder_focus_mask
            shoulder_cleanup_region_valid = shoulder_focus_region_valid
            if shoulder_cleanup_region_mode == 'collar':
                shoulder_cleanup_region_base = shoulder_collar_mask
                shoulder_cleanup_region_valid = shoulder_collar_region_valid
            if shoulder_cleanup_region_valid and any(
                value > 0.
                for value in (
                    lambda_silhouette_shoulder_attached_fragment,
                    lambda_silhouette_shoulder_bead,
                    lambda_silhouette_shoulder_chain,
                    lambda_silhouette_shoulder_hole,
                    lambda_silhouette_shoulder_gap,
                    lambda_silhouette_shoulder_pinhole,
                )
            ):
                shoulder_cleanup_region = shoulder_cleanup_region_base
                shoulder_cleanup_region_dilate = int(config.opt.get('silhouette_shoulder_focus_region_dilate', 7))
                if shoulder_cleanup_region_dilate > 1:
                    shoulder_cleanup_region = _binary_dilate(shoulder_cleanup_region, shoulder_cleanup_region_dilate).clamp(0.0, 1.0)
                if lambda_silhouette_shoulder_bead > 0. or lambda_silhouette_shoulder_chain > 0.:
                    shoulder_outer_region = shoulder_cleanup_region
                    shoulder_outer_region_dilate = int(config.opt.get('silhouette_shoulder_outer_region_dilate', 0))
                    if shoulder_outer_region_dilate > 1:
                        shoulder_outer_region = _binary_dilate(shoulder_outer_region, shoulder_outer_region_dilate).clamp(0.0, 1.0)
                    shoulder_outer_shell_mask = _foreground_outer_shell_mask(
                        fg_mask,
                        int(config.opt.get('silhouette_shoulder_outer_shell_start_width', 1)),
                        int(config.opt.get('silhouette_shoulder_outer_shell_end_width', 15)),
                    ) * shoulder_outer_region
                    if lambda_silhouette_shoulder_bead > 0. and shoulder_outer_shell_mask.sum().item() > 0:
                        shoulder_bead_mask = _foreground_outer_bead_mask(
                            opacity_bce,
                            shoulder_outer_shell_mask,
                            region_mask=shoulder_outer_region,
                            opacity_threshold=float(config.opt.get('silhouette_shoulder_bead_opacity_threshold', 0.035)),
                            opening_kernel_size=int(config.opt.get('silhouette_shoulder_bead_opening_kernel_size', 5)),
                            support_kernel_size=int(config.opt.get('silhouette_shoulder_bead_support_kernel_size', 7)),
                            support_threshold=float(config.opt.get('silhouette_shoulder_bead_support_threshold', 0.78)),
                            power=float(config.opt.get('silhouette_shoulder_bead_power', 1.0)),
                        )
                        if shoulder_bead_mask is not None and shoulder_bead_mask.sum().item() > 0:
                            loss_silhouette_shoulder_bead = _masked_binary_cross_entropy(opacity_bce, 0.0, shoulder_bead_mask)
                    if lambda_silhouette_shoulder_chain > 0. and shoulder_outer_shell_mask.sum().item() > 0:
                        shoulder_chain_mask = _foreground_outer_chain_mask(
                            opacity_bce,
                            shoulder_outer_shell_mask,
                            region_mask=shoulder_outer_region,
                            opacity_threshold=float(config.opt.get('silhouette_shoulder_chain_opacity_threshold', 0.035)),
                            support_kernel_size=int(config.opt.get('silhouette_shoulder_chain_support_kernel_size', 7)),
                            seed_support_threshold=float(config.opt.get('silhouette_shoulder_chain_seed_support_threshold', 0.74)),
                            propagate_support_threshold=float(config.opt.get('silhouette_shoulder_chain_propagate_support_threshold', 0.46)),
                            anchor_weight_threshold=float(config.opt.get('silhouette_shoulder_chain_anchor_weight_threshold', 0.68)),
                            max_steps=int(config.opt.get('silhouette_shoulder_chain_max_steps', 20)),
                            power=float(config.opt.get('silhouette_shoulder_chain_power', 1.0)),
                        )
                        if shoulder_chain_mask is not None and shoulder_chain_mask.sum().item() > 0:
                            loss_silhouette_shoulder_chain = _masked_binary_cross_entropy(opacity_bce, 0.0, shoulder_chain_mask)
                if lambda_silhouette_shoulder_attached_fragment > 0.:
                    shoulder_attached_region = shoulder_cleanup_region
                    shoulder_attached_region_dilate = int(config.opt.get('silhouette_shoulder_attached_fragment_region_dilate', 5))
                    if shoulder_attached_region_dilate > 1:
                        shoulder_attached_region = _binary_dilate(shoulder_attached_region, shoulder_attached_region_dilate).clamp(0.0, 1.0)
                    shoulder_attached_fragment_mask = _foreground_arm_boundary_attached_fragment_mask(
                        opacity_bce,
                        fg_mask,
                        region_mask=shoulder_attached_region,
                        opacity_threshold=float(config.opt.get('silhouette_shoulder_attached_fragment_opacity_threshold', 0.04)),
                        boundary_band_kernel_size=int(config.opt.get('silhouette_shoulder_attached_fragment_boundary_kernel_size', 5)),
                        outer_shell_start_width=int(config.opt.get('silhouette_shoulder_attached_fragment_outer_shell_start_width', 1)),
                        outer_shell_end_width=int(config.opt.get('silhouette_shoulder_attached_fragment_outer_shell_end_width', 13)),
                        component_area_min=int(config.opt.get('silhouette_shoulder_attached_fragment_area_min', 1)),
                        component_area_max=int(config.opt.get('silhouette_shoulder_attached_fragment_area_max', 72)),
                        fill_ratio_max=float(config.opt.get('silhouette_shoulder_attached_fragment_fill_ratio_max', 0.94)),
                        touch_ratio_min=float(config.opt.get('silhouette_shoulder_attached_fragment_touch_ratio_min', 0.08)),
                        power=float(config.opt.get('silhouette_shoulder_attached_fragment_power', 1.0)),
                    )
                    if shoulder_attached_fragment_mask is not None and shoulder_attached_fragment_mask.sum().item() > 0:
                        loss_silhouette_shoulder_attached_fragment = _masked_binary_cross_entropy(opacity_bce, 0.0, shoulder_attached_fragment_mask)
                if lambda_silhouette_shoulder_hole > 0. or lambda_silhouette_shoulder_gap > 0.:
                    shoulder_hole_gap_region = shoulder_cleanup_region
                    shoulder_hole_gap_region_dilate = int(config.opt.get('silhouette_shoulder_hole_region_dilate', 0))
                    if shoulder_hole_gap_region_dilate > 1:
                        shoulder_hole_gap_region = _binary_dilate(shoulder_hole_gap_region, shoulder_hole_gap_region_dilate).clamp(0.0, 1.0)
                    shoulder_hole_mask, shoulder_gap_mask = _foreground_arm_hole_and_gap_masks(
                        opacity_bce,
                        fg_mask,
                        region_mask=shoulder_hole_gap_region,
                        opacity_threshold=float(config.opt.get('silhouette_shoulder_hole_opacity_threshold', config.opt.get('silhouette_shoulder_gap_opacity_threshold', 0.07))),
                        gap_opacity_threshold=float(config.opt.get('silhouette_shoulder_gap_opacity_threshold', config.opt.get('silhouette_shoulder_hole_opacity_threshold', 0.07))),
                        inner_band_kernel_size=int(config.opt.get('silhouette_shoulder_hole_inner_band_kernel_size', config.opt.get('silhouette_shoulder_gap_inner_band_kernel_size', 5))),
                        inner_region_dilate=int(config.opt.get('silhouette_shoulder_hole_inner_region_dilate', config.opt.get('silhouette_shoulder_gap_inner_region_dilate', 9))),
                        closing_kernel_size=int(config.opt.get('silhouette_shoulder_gap_close_kernel_size', 9)),
                        hole_area_min=int(config.opt.get('silhouette_shoulder_hole_area_min', 1)),
                        hole_area_max=int(config.opt.get('silhouette_shoulder_hole_area_max', 48)),
                        gap_area_min=int(config.opt.get('silhouette_shoulder_gap_area_min', 2)),
                        gap_area_max=int(config.opt.get('silhouette_shoulder_gap_area_max', 96)),
                        hole_power=float(config.opt.get('silhouette_shoulder_hole_power', 1.0)),
                        gap_power=float(config.opt.get('silhouette_shoulder_gap_power', 1.0)),
                    )
                    if lambda_silhouette_shoulder_hole > 0. and shoulder_hole_mask is not None and shoulder_hole_mask.sum().item() > 0:
                        loss_silhouette_shoulder_hole = _masked_binary_cross_entropy(opacity_bce, 1.0, shoulder_hole_mask)
                    if lambda_silhouette_shoulder_gap > 0. and shoulder_gap_mask is not None and shoulder_gap_mask.sum().item() > 0:
                        loss_silhouette_shoulder_gap = _masked_binary_cross_entropy(opacity_bce, 1.0, shoulder_gap_mask)
                if lambda_silhouette_shoulder_pinhole > 0.:
                    shoulder_pinhole_region = shoulder_cleanup_region
                    shoulder_pinhole_region_dilate = int(config.opt.get('silhouette_shoulder_pinhole_region_dilate', 0))
                    if shoulder_pinhole_region_dilate > 1:
                        shoulder_pinhole_region = _binary_dilate(shoulder_pinhole_region, shoulder_pinhole_region_dilate).clamp(0.0, 1.0)
                    shoulder_pinhole_mask = _foreground_region_pinhole_mask(
                        opacity_bce,
                        fg_mask,
                        region_mask=shoulder_pinhole_region,
                        opacity_threshold=float(config.opt.get('silhouette_shoulder_pinhole_opacity_threshold', 0.07)),
                        closing_kernel_size=int(config.opt.get('silhouette_shoulder_pinhole_close_kernel_size', 7)),
                        support_kernel_size=int(config.opt.get('silhouette_shoulder_pinhole_support_kernel_size', 9)),
                        support_threshold=float(config.opt.get('silhouette_shoulder_pinhole_support_threshold', 0.72)),
                        core_erode_kernel_size=int(config.opt.get('silhouette_shoulder_pinhole_core_erode_kernel_size', 3)),
                        area_min=int(config.opt.get('silhouette_shoulder_pinhole_area_min', 1)),
                        area_max=int(config.opt.get('silhouette_shoulder_pinhole_area_max', 24)),
                        power=float(config.opt.get('silhouette_shoulder_pinhole_power', 1.0)),
                    )
                    if shoulder_pinhole_mask is not None and shoulder_pinhole_mask.sum().item() > 0:
                        loss_silhouette_shoulder_pinhole = _masked_binary_cross_entropy(opacity_bce, 1.0, shoulder_pinhole_mask)
            if lambda_silhouette_inner > 0. and inner_ring_mask.sum().item() > 0:
                loss_silhouette_inner = _masked_binary_cross_entropy(opacity_bce, 1.0, inner_ring_mask)
        shoulder_perceptual_bbox_area, shoulder_perceptual_fill_ratio = _mask_bbox_stats(
            shoulder_arm_perceptual_used_mask,
            padding=int(config.opt.get('shoulder_arm_perceptual_crop_pad', 12)),
        )
        waist_perceptual_bbox_area, waist_perceptual_fill_ratio = _mask_bbox_stats(
            waist_perceptual_used_mask,
            padding=int(config.opt.get('waist_perceptual_crop_pad', 8)),
        )
        _maybe_log_shoulder_local_mask_debug(
            config.opt,
            schedule_iteration,
            {
                'arm_pixels': float(shoulder_arm_mask.sum().item()),
                'focus_pixels': float(shoulder_focus_mask.sum().item()),
                'collar_pixels': float(shoulder_collar_mask.sum().item()),
                'image_basis': shoulder_arm_image_region_meta.get('basis_mode', 'arm'),
                'image_pattern': shoulder_arm_image_region_meta.get('pattern_mode', 'raw'),
                'image_pixels': float(shoulder_arm_image_mask.sum().item()),
                'perceptual_basis': shoulder_arm_perceptual_region_meta.get('basis_mode', 'arm'),
                'perceptual_pattern': shoulder_arm_perceptual_region_meta.get('pattern_mode', 'raw'),
                'perceptual_pixels': float(shoulder_arm_perceptual_used_mask.sum().item()),
                'perceptual_bbox_area': shoulder_perceptual_bbox_area,
                'perceptual_fill_ratio': shoulder_perceptual_fill_ratio,
                'boundary_basis': shoulder_arm_boundary_region_meta.get('basis_mode', 'arm'),
                'boundary_pattern': shoulder_arm_boundary_region_meta.get('pattern_mode', 'boundary'),
                'boundary_pixels': float(shoulder_arm_boundary_supervision_mask.sum().item()),
                'region_basis': shoulder_arm_region_hard_meta.get('basis_mode', 'arm'),
                'region_pattern': shoulder_arm_region_hard_meta.get('pattern_mode', 'raw'),
                'region_pixels': float(shoulder_arm_region_hard_supervision_mask.sum().item()),
                'disagreement_basis': shoulder_arm_disagreement_meta.get('basis_mode', 'arm'),
                'disagreement_pattern': shoulder_arm_disagreement_meta.get('pattern_mode', 'error_band'),
                'disagreement_pixels': float(shoulder_arm_disagreement_supervision_mask.sum().item()),
                'outer_basis': shoulder_arm_outer_shell_meta.get('basis_mode', 'arm'),
                'outer_pattern': shoulder_arm_outer_shell_meta.get('pattern_mode', 'outer_shell'),
                'outer_pixels': float(shoulder_arm_outer_shell_supervision_mask.sum().item()),
            },
        )
        waist_debug_enable = bool(config.opt.get('waist_region_debug_enable', False))
        waist_debug_interval = int(config.opt.get('waist_region_debug_interval', 0))
        waist_debug_warmup_iters = int(config.opt.get('waist_region_debug_warmup_iters', 0))
        if (
            waist_debug_enable
            and schedule_iteration >= waist_debug_warmup_iters
            and (waist_debug_interval <= 0 or schedule_iteration % waist_debug_interval == 0)
        ):
            print(
                (
                    f"[WaistLocalMask] iter={schedule_iteration} "
                    f"image_pixels={float(waist_image_mask.sum().item()):.1f} "
                    f"perceptual_pixels={float(waist_perceptual_used_mask.sum().item()):.1f} "
                    f"perceptual_bbox={waist_perceptual_bbox_area:.1f} "
                    f"perceptual_fill={waist_perceptual_fill_ratio:.4f}"
                ),
                flush=True,
            )
        upper_torso_debug_enable = bool(config.opt.get('upper_torso_region_debug_enable', False))
        upper_torso_debug_interval = int(config.opt.get('upper_torso_region_debug_interval', 0))
        upper_torso_debug_warmup_iters = int(config.opt.get('upper_torso_region_debug_warmup_iters', 0))
        if (
            upper_torso_debug_enable
            and schedule_iteration >= upper_torso_debug_warmup_iters
            and (upper_torso_debug_interval <= 0 or schedule_iteration % upper_torso_debug_interval == 0)
        ):
            print(
                (
                    f"[UpperTorsoLocalMask] iter={schedule_iteration} "
                    f"image_pixels={float(upper_torso_image_mask.sum().item()):.1f} "
                    f"detail_pixels={float(upper_torso_detail_mask.sum().item()):.1f} "
                    f"core_pixels={float(upper_torso_core_mask.sum().item()):.1f} "
                    f"perceptual_pixels={float(upper_torso_perceptual_used_mask.sum().item()):.1f} "
                    f"core_perceptual_pixels={float(upper_torso_core_perceptual_used_mask.sum().item()):.1f} "
                    f"boundary_pixels={float(upper_torso_boundary_supervision_mask.sum().item()):.1f} "
                    f"outer_pixels={float(upper_torso_outer_shell_supervision_mask.sum().item()):.1f}"
                ),
                flush=True,
            )
        base_loss += lambda_mask * loss_mask
        if gate_mask_boundary:
            boundary_loss += lambda_mask_boundary * loss_mask_boundary
        else:
            base_loss += lambda_mask_boundary * loss_mask_boundary
        if gate_mask_boundary_hard:
            boundary_loss += lambda_mask_boundary_hard * loss_mask_boundary_hard
        else:
            base_loss += lambda_mask_boundary_hard * loss_mask_boundary_hard
        if gate_mask_shoulder_arm_boundary_hard:
            boundary_loss += lambda_mask_shoulder_arm_boundary_hard * loss_mask_shoulder_arm_boundary_hard
        else:
            base_loss += lambda_mask_shoulder_arm_boundary_hard * loss_mask_shoulder_arm_boundary_hard
        if gate_mask_upper_torso_boundary_hard:
            boundary_loss += lambda_mask_upper_torso_boundary_hard * loss_mask_upper_torso_boundary_hard
        else:
            base_loss += lambda_mask_upper_torso_boundary_hard * loss_mask_upper_torso_boundary_hard
        if gate_mask_shoulder_arm_region_hard:
            boundary_loss += lambda_mask_shoulder_arm_region_hard * loss_mask_shoulder_arm_region_hard
        else:
            base_loss += lambda_mask_shoulder_arm_region_hard * loss_mask_shoulder_arm_region_hard
        if gate_mask_shoulder_arm_disagreement_hard:
            boundary_loss += lambda_mask_shoulder_arm_disagreement_hard * loss_mask_shoulder_arm_disagreement_hard
        else:
            base_loss += lambda_mask_shoulder_arm_disagreement_hard * loss_mask_shoulder_arm_disagreement_hard
        if gate_mask_shoulder_focus_small_fp_hard:
            boundary_loss += lambda_mask_shoulder_focus_small_fp_hard * loss_mask_shoulder_focus_small_fp_hard
        else:
            base_loss += lambda_mask_shoulder_focus_small_fp_hard * loss_mask_shoulder_focus_small_fp_hard
        if gate_mask_shoulder_focus_small_fn_hard:
            boundary_loss += lambda_mask_shoulder_focus_small_fn_hard * loss_mask_shoulder_focus_small_fn_hard
        else:
            base_loss += lambda_mask_shoulder_focus_small_fn_hard * loss_mask_shoulder_focus_small_fn_hard
        if gate_silhouette_outer:
            boundary_loss += lambda_silhouette_outer * loss_silhouette_outer
        else:
            base_loss += lambda_silhouette_outer * loss_silhouette_outer
        if gate_silhouette_outer_shell:
            boundary_loss += lambda_silhouette_outer_shell * loss_silhouette_outer_shell
        else:
            base_loss += lambda_silhouette_outer_shell * loss_silhouette_outer_shell
        if gate_silhouette_head_outer_shell:
            boundary_loss += lambda_silhouette_head_outer_shell * loss_silhouette_head_outer_shell
        else:
            base_loss += lambda_silhouette_head_outer_shell * loss_silhouette_head_outer_shell
        if gate_silhouette_shoulder_arm_outer_shell:
            boundary_loss += lambda_silhouette_shoulder_arm_outer_shell * loss_silhouette_shoulder_arm_outer_shell
        else:
            base_loss += lambda_silhouette_shoulder_arm_outer_shell * loss_silhouette_shoulder_arm_outer_shell
        if gate_silhouette_upper_torso_outer_shell:
            boundary_loss += lambda_silhouette_upper_torso_outer_shell * loss_silhouette_upper_torso_outer_shell
        else:
            base_loss += lambda_silhouette_upper_torso_outer_shell * loss_silhouette_upper_torso_outer_shell
        if gate_silhouette_outer_spike:
            boundary_loss += lambda_silhouette_outer_spike * loss_silhouette_outer_spike
        else:
            base_loss += lambda_silhouette_outer_spike * loss_silhouette_outer_spike
        if gate_silhouette_outer_fragment:
            boundary_loss += lambda_silhouette_outer_fragment * loss_silhouette_outer_fragment
        else:
            base_loss += lambda_silhouette_outer_fragment * loss_silhouette_outer_fragment
        if gate_silhouette_outer_bead:
            boundary_loss += lambda_silhouette_outer_bead * loss_silhouette_outer_bead
        else:
            base_loss += lambda_silhouette_outer_bead * loss_silhouette_outer_bead
        if gate_silhouette_outer_chain:
            boundary_loss += lambda_silhouette_outer_chain * loss_silhouette_outer_chain
        else:
            base_loss += lambda_silhouette_outer_chain * loss_silhouette_outer_chain
        if gate_silhouette_arm_stipple:
            boundary_loss += lambda_silhouette_arm_stipple * loss_silhouette_arm_stipple
        else:
            base_loss += lambda_silhouette_arm_stipple * loss_silhouette_arm_stipple
        if gate_silhouette_arm_tail:
            boundary_loss += lambda_silhouette_arm_tail * loss_silhouette_arm_tail
        else:
            base_loss += lambda_silhouette_arm_tail * loss_silhouette_arm_tail
        if gate_silhouette_arm_fringe:
            boundary_loss += lambda_silhouette_arm_fringe * loss_silhouette_arm_fringe
        else:
            base_loss += lambda_silhouette_arm_fringe * loss_silhouette_arm_fringe
        if gate_silhouette_arm_attached_fragment:
            boundary_loss += lambda_silhouette_arm_attached_fragment * loss_silhouette_arm_attached_fragment
        else:
            base_loss += lambda_silhouette_arm_attached_fragment * loss_silhouette_arm_attached_fragment
        if gate_silhouette_shoulder_attached_fragment:
            boundary_loss += lambda_silhouette_shoulder_attached_fragment * loss_silhouette_shoulder_attached_fragment
        else:
            base_loss += lambda_silhouette_shoulder_attached_fragment * loss_silhouette_shoulder_attached_fragment
        if gate_silhouette_arm_notch:
            boundary_loss += lambda_silhouette_arm_notch * loss_silhouette_arm_notch
        else:
            base_loss += lambda_silhouette_arm_notch * loss_silhouette_arm_notch
        if gate_silhouette_arm_hole:
            boundary_loss += lambda_silhouette_arm_hole * loss_silhouette_arm_hole
        else:
            base_loss += lambda_silhouette_arm_hole * loss_silhouette_arm_hole
        if gate_silhouette_arm_gap:
            boundary_loss += lambda_silhouette_arm_gap * loss_silhouette_arm_gap
        else:
            base_loss += lambda_silhouette_arm_gap * loss_silhouette_arm_gap
        if gate_silhouette_shoulder_bead:
            boundary_loss += lambda_silhouette_shoulder_bead * loss_silhouette_shoulder_bead
        else:
            base_loss += lambda_silhouette_shoulder_bead * loss_silhouette_shoulder_bead
        if gate_silhouette_shoulder_chain:
            boundary_loss += lambda_silhouette_shoulder_chain * loss_silhouette_shoulder_chain
        else:
            base_loss += lambda_silhouette_shoulder_chain * loss_silhouette_shoulder_chain
        if gate_silhouette_shoulder_hole:
            boundary_loss += lambda_silhouette_shoulder_hole * loss_silhouette_shoulder_hole
        else:
            base_loss += lambda_silhouette_shoulder_hole * loss_silhouette_shoulder_hole
        if gate_silhouette_shoulder_gap:
            boundary_loss += lambda_silhouette_shoulder_gap * loss_silhouette_shoulder_gap
        else:
            base_loss += lambda_silhouette_shoulder_gap * loss_silhouette_shoulder_gap
        if gate_silhouette_shoulder_pinhole:
            boundary_loss += lambda_silhouette_shoulder_pinhole * loss_silhouette_shoulder_pinhole
        else:
            base_loss += lambda_silhouette_shoulder_pinhole * loss_silhouette_shoulder_pinhole
        if gate_silhouette_inner:
            boundary_loss += lambda_silhouette_inner * loss_silhouette_inner
        else:
            base_loss += lambda_silhouette_inner * loss_silhouette_inner

        boundary_shrink_loss = torch.tensor(0.).cuda()
        boundary_grow_loss = torch.tensor(0.).cuda()
        if gate_mask_shoulder_focus_small_fp_hard:
            boundary_shrink_loss += lambda_mask_shoulder_focus_small_fp_hard * loss_mask_shoulder_focus_small_fp_hard
        if gate_silhouette_outer:
            boundary_shrink_loss += lambda_silhouette_outer * loss_silhouette_outer
        if gate_silhouette_outer_shell:
            boundary_shrink_loss += lambda_silhouette_outer_shell * loss_silhouette_outer_shell
        if gate_silhouette_head_outer_shell:
            boundary_shrink_loss += lambda_silhouette_head_outer_shell * loss_silhouette_head_outer_shell
        if gate_silhouette_shoulder_arm_outer_shell:
            boundary_shrink_loss += lambda_silhouette_shoulder_arm_outer_shell * loss_silhouette_shoulder_arm_outer_shell
        if gate_silhouette_upper_torso_outer_shell:
            boundary_shrink_loss += lambda_silhouette_upper_torso_outer_shell * loss_silhouette_upper_torso_outer_shell
        if gate_silhouette_outer_spike:
            boundary_shrink_loss += lambda_silhouette_outer_spike * loss_silhouette_outer_spike
        if gate_silhouette_outer_fragment:
            boundary_shrink_loss += lambda_silhouette_outer_fragment * loss_silhouette_outer_fragment
        if gate_silhouette_outer_bead:
            boundary_shrink_loss += lambda_silhouette_outer_bead * loss_silhouette_outer_bead
        if gate_silhouette_outer_chain:
            boundary_shrink_loss += lambda_silhouette_outer_chain * loss_silhouette_outer_chain
        if gate_silhouette_arm_stipple:
            boundary_shrink_loss += lambda_silhouette_arm_stipple * loss_silhouette_arm_stipple
        if gate_silhouette_arm_tail:
            boundary_shrink_loss += lambda_silhouette_arm_tail * loss_silhouette_arm_tail
        if gate_silhouette_arm_fringe:
            boundary_shrink_loss += lambda_silhouette_arm_fringe * loss_silhouette_arm_fringe
        if gate_silhouette_arm_attached_fragment:
            boundary_shrink_loss += lambda_silhouette_arm_attached_fragment * loss_silhouette_arm_attached_fragment
        if gate_silhouette_shoulder_attached_fragment:
            boundary_shrink_loss += lambda_silhouette_shoulder_attached_fragment * loss_silhouette_shoulder_attached_fragment
        if gate_silhouette_shoulder_bead:
            boundary_shrink_loss += lambda_silhouette_shoulder_bead * loss_silhouette_shoulder_bead
        if gate_silhouette_shoulder_chain:
            boundary_shrink_loss += lambda_silhouette_shoulder_chain * loss_silhouette_shoulder_chain

        if gate_mask_shoulder_focus_small_fn_hard:
            boundary_grow_loss += lambda_mask_shoulder_focus_small_fn_hard * loss_mask_shoulder_focus_small_fn_hard
        if gate_silhouette_arm_notch:
            boundary_grow_loss += lambda_silhouette_arm_notch * loss_silhouette_arm_notch
        if gate_silhouette_arm_hole:
            boundary_grow_loss += lambda_silhouette_arm_hole * loss_silhouette_arm_hole
        if gate_silhouette_arm_gap:
            boundary_grow_loss += lambda_silhouette_arm_gap * loss_silhouette_arm_gap
        if gate_silhouette_shoulder_hole:
            boundary_grow_loss += lambda_silhouette_shoulder_hole * loss_silhouette_shoulder_hole
        if gate_silhouette_shoulder_gap:
            boundary_grow_loss += lambda_silhouette_shoulder_gap * loss_silhouette_shoulder_gap
        if gate_silhouette_shoulder_pinhole:
            boundary_grow_loss += lambda_silhouette_shoulder_pinhole * loss_silhouette_shoulder_pinhole
        if gate_silhouette_inner:
            boundary_grow_loss += lambda_silhouette_inner * loss_silhouette_inner
        boundary_mixed_loss = boundary_loss - boundary_shrink_loss - boundary_grow_loss

        # skinning loss
        lambda_skinning = C(schedule_iteration, config.opt.lambda_skinning)
        if lambda_skinning > 0:
            loss_skinning = scene.get_skinning_loss()
            base_loss += lambda_skinning * loss_skinning
        else:
            loss_skinning = torch.tensor(0.).cuda()

        lambda_aiap_xyz = C(schedule_iteration, config.opt.get('lambda_aiap_xyz', 0.))
        lambda_aiap_cov = C(schedule_iteration, config.opt.get('lambda_aiap_cov', 0.))
        aiap_interval = max(int(config.opt.get('aiap_interval', 1)), 1)
        aiap_max_points = int(config.opt.get('aiap_max_points', 0))
        should_compute_aiap = (schedule_iteration % aiap_interval == 0)
        if (lambda_aiap_xyz > 0. or lambda_aiap_cov > 0.) and should_compute_aiap:
            loss_aiap_xyz, loss_aiap_cov = full_aiap_loss(
                scene.gaussians,
                render_pkg["deformed_gaussian"],
                max_points=aiap_max_points,
            )
        else:
            loss_aiap_xyz = torch.tensor(0.).cuda()
            loss_aiap_cov = torch.tensor(0.).cuda()
        base_loss += lambda_aiap_xyz * loss_aiap_xyz
        base_loss += lambda_aiap_cov * loss_aiap_cov

        loss_boundary_opacity_residual_reg = torch.tensor(0.).cuda()
        loss_boundary_scaling_residual_reg = torch.tensor(0.).cuda()
        loss_boundary_opacity_residual_smooth = torch.tensor(0.).cuda()
        loss_boundary_scaling_residual_smooth = torch.tensor(0.).cuda()
        lambda_boundary_opacity_residual_reg = C(schedule_iteration, config.opt.get('lambda_boundary_opacity_residual_reg', 0.0))
        lambda_boundary_scaling_residual_reg = C(schedule_iteration, config.opt.get('lambda_boundary_scaling_residual_reg', 0.0))
        lambda_boundary_opacity_residual_smooth = C(schedule_iteration, config.opt.get('lambda_boundary_opacity_residual_smooth', 0.0))
        lambda_boundary_scaling_residual_smooth = C(schedule_iteration, config.opt.get('lambda_boundary_scaling_residual_smooth', 0.0))
        boundary_tags_for_reg = scene.gaussians.get_boundary_tags()
        if boundary_tags_for_reg is not None:
            boundary_mask_reg = boundary_tags_for_reg.unsqueeze(-1)
            if lambda_boundary_opacity_residual_reg > 0.0:
                loss_boundary_opacity_residual_reg = ((scene.gaussians._boundary_opacity_residual * boundary_mask_reg) ** 2).sum() / boundary_mask_reg.sum().clamp_min(1.0)
                base_loss += lambda_boundary_opacity_residual_reg * loss_boundary_opacity_residual_reg
            if lambda_boundary_scaling_residual_reg > 0.0:
                loss_boundary_scaling_residual_reg = ((scene.gaussians._boundary_scaling_residual * boundary_mask_reg) ** 2).sum() / boundary_mask_reg.sum().clamp_min(1.0)
                base_loss += lambda_boundary_scaling_residual_reg * loss_boundary_scaling_residual_reg
            if lambda_boundary_opacity_residual_smooth > 0.0 or lambda_boundary_scaling_residual_smooth > 0.0:
                smooth_positions = _boundary_regularization_positions(scene)
                smooth_k = int(config.opt.get('boundary_residual_smooth_k', 8))
                smooth_quantile = float(config.opt.get('boundary_residual_smooth_distance_quantile', 0.5))
                point_mask = boundary_tags_for_reg > 0
                if lambda_boundary_opacity_residual_smooth > 0.0:
                    loss_boundary_opacity_residual_smooth = _boundary_residual_smoothness_loss(
                        scene.gaussians._boundary_opacity_residual,
                        smooth_positions,
                        point_mask,
                        k=smooth_k,
                        distance_quantile=smooth_quantile,
                    )
                    base_loss += lambda_boundary_opacity_residual_smooth * loss_boundary_opacity_residual_smooth
                if lambda_boundary_scaling_residual_smooth > 0.0:
                    loss_boundary_scaling_residual_smooth = _boundary_residual_smoothness_loss(
                        scene.gaussians._boundary_scaling_residual,
                        smooth_positions,
                        point_mask,
                        k=smooth_k,
                        distance_quantile=smooth_quantile,
                    )
                    base_loss += lambda_boundary_scaling_residual_smooth * loss_boundary_scaling_residual_smooth

        # regularization
        loss_reg = render_pkg["loss_reg"]
        for name, value in loss_reg.items():
            lbd = opt.get(f"lambda_{name}", 0.)
            lbd = C(schedule_iteration, lbd)
            base_loss += lbd * value

        weighted_fullframe_image_term = (
            lambda_l1 * loss_l1
            + lambda_dssim * loss_dssim
            + lambda_l1_fg * loss_l1_fg
            + lambda_perceptual * loss_perceptual
        )
        weighted_face_image_term = (
            + lambda_l1_face * loss_l1_face
            + lambda_edge_face * loss_edge_face
            + lambda_detail_face * loss_detail_face
            + lambda_detail_face_luma_dog * loss_detail_face_luma_dog
            + lambda_perceptual_face * loss_perceptual_face
            + lambda_perceptual_face_patch * loss_perceptual_face_patch
        )
        weighted_global_image_term = weighted_fullframe_image_term + weighted_face_image_term
        weighted_shoulder_image_term = (
            lambda_l1_shoulder_arm * loss_l1_shoulder_arm
            + lambda_edge_shoulder_arm * loss_edge_shoulder_arm
            + lambda_detail_shoulder_arm * loss_detail_shoulder_arm
            + lambda_detail_shoulder_arm_luma_dog * loss_detail_shoulder_arm_luma_dog
            + lambda_perceptual_shoulder_arm * loss_perceptual_shoulder_arm
            + lambda_perceptual_shoulder_arm_patch * loss_perceptual_shoulder_arm_patch
            + lambda_l1_shoulder_focus_dark_outlier * loss_l1_shoulder_focus_dark_outlier
            + lambda_l1_shoulder_focus_bright_outlier * loss_l1_shoulder_focus_bright_outlier
        )
        weighted_upper_torso_image_term = (
            lambda_detail_upper_torso_luma_dog * loss_detail_upper_torso_luma_dog
            + lambda_perceptual_upper_torso_patch * loss_perceptual_upper_torso_patch
            + lambda_detail_upper_torso_core_luma_dog * loss_detail_upper_torso_core_luma_dog
            + lambda_perceptual_upper_torso_core_patch * loss_perceptual_upper_torso_core_patch
        )
        weighted_waist_image_term = (
            lambda_l1_waist * loss_l1_waist
            + lambda_edge_waist * loss_edge_waist
            + lambda_detail_waist * loss_detail_waist
            + lambda_detail_waist_luma_dog * loss_detail_waist_luma_dog
            + lambda_perceptual_waist * loss_perceptual_waist
            + lambda_perceptual_waist_patch * loss_perceptual_waist_patch
        )
        weighted_local_image_term = (
            weighted_shoulder_image_term
            + weighted_upper_torso_image_term
            + weighted_waist_image_term
        )
        weighted_global_boundary_term = (
            lambda_l1_boundary * loss_l1_boundary
            + lambda_mask_boundary * loss_mask_boundary
            + lambda_mask_boundary_hard * loss_mask_boundary_hard
            + lambda_silhouette_outer * loss_silhouette_outer
            + lambda_silhouette_outer_shell * loss_silhouette_outer_shell
            + lambda_silhouette_head_outer_shell * loss_silhouette_head_outer_shell
            + lambda_silhouette_outer_spike * loss_silhouette_outer_spike
            + lambda_silhouette_outer_fragment * loss_silhouette_outer_fragment
            + lambda_silhouette_outer_bead * loss_silhouette_outer_bead
            + lambda_silhouette_outer_chain * loss_silhouette_outer_chain
            + lambda_silhouette_inner * loss_silhouette_inner
        )
        weighted_shoulder_boundary_term = (
            lambda_mask_shoulder_arm_boundary_hard * loss_mask_shoulder_arm_boundary_hard
            + lambda_mask_shoulder_arm_region_hard * loss_mask_shoulder_arm_region_hard
            + lambda_mask_shoulder_arm_disagreement_hard * loss_mask_shoulder_arm_disagreement_hard
            + lambda_mask_shoulder_focus_small_fp_hard * loss_mask_shoulder_focus_small_fp_hard
            + lambda_mask_shoulder_focus_small_fn_hard * loss_mask_shoulder_focus_small_fn_hard
            + lambda_silhouette_shoulder_arm_outer_shell * loss_silhouette_shoulder_arm_outer_shell
            + lambda_silhouette_arm_stipple * loss_silhouette_arm_stipple
            + lambda_silhouette_arm_tail * loss_silhouette_arm_tail
            + lambda_silhouette_arm_fringe * loss_silhouette_arm_fringe
            + lambda_silhouette_arm_attached_fragment * loss_silhouette_arm_attached_fragment
            + lambda_silhouette_arm_notch * loss_silhouette_arm_notch
            + lambda_silhouette_arm_hole * loss_silhouette_arm_hole
            + lambda_silhouette_arm_gap * loss_silhouette_arm_gap
            + lambda_silhouette_shoulder_attached_fragment * loss_silhouette_shoulder_attached_fragment
            + lambda_silhouette_shoulder_bead * loss_silhouette_shoulder_bead
            + lambda_silhouette_shoulder_chain * loss_silhouette_shoulder_chain
            + lambda_silhouette_shoulder_hole * loss_silhouette_shoulder_hole
            + lambda_silhouette_shoulder_gap * loss_silhouette_shoulder_gap
            + lambda_silhouette_shoulder_pinhole * loss_silhouette_shoulder_pinhole
        )
        weighted_upper_torso_boundary_term = (
            lambda_mask_upper_torso_boundary_hard * loss_mask_upper_torso_boundary_hard
            + lambda_silhouette_upper_torso_outer_shell * loss_silhouette_upper_torso_outer_shell
        )
        weighted_local_boundary_term = (
            weighted_shoulder_boundary_term
            + weighted_upper_torso_boundary_term
        )
        loss = base_loss + boundary_loss
        boundary_score = _get_boundary_aware_score(render_pkg["deformed_gaussian"], config)
        boundary_effective_score = _get_boundary_effective_score(
            scene,
            boundary_score,
            render_pkg["deformed_gaussian"],
            config,
            schedule_iteration,
            local_iteration=local_iteration,
        )
        boundary_signed_routing_enable = bool(config.opt.get('boundary_signed_routing_enable', False))
        boundary_image_candidate_score = getattr(render_pkg["deformed_gaussian"], 'binding_boundary_image_score', None)
        boundary_image_under_score = getattr(render_pkg["deformed_gaussian"], 'binding_boundary_image_under_score', None)
        boundary_image_over_score = getattr(render_pkg["deformed_gaussian"], 'binding_boundary_image_over_score', None)
        boundary_grow_effective_score = (
            _build_directional_boundary_effective_score(
                boundary_effective_score,
                boundary_image_under_score,
                boundary_image_candidate_score,
                config=config,
                direction='grow',
            )
            if boundary_signed_routing_enable else None
        )
        boundary_shrink_effective_score = (
            _build_directional_boundary_effective_score(
                boundary_effective_score,
                boundary_image_over_score,
                boundary_image_candidate_score,
                config=config,
                direction='shrink',
            )
            if boundary_signed_routing_enable else None
        )
        boundary_loss_value = float(boundary_loss.detach().abs().item()) if torch.is_tensor(boundary_loss) else 0.0
        boundary_mixed_loss_value = float(boundary_mixed_loss.detach().abs().item()) if torch.is_tensor(boundary_mixed_loss) else 0.0
        boundary_grow_loss_value = float(boundary_grow_loss.detach().abs().item()) if torch.is_tensor(boundary_grow_loss) else 0.0
        boundary_shrink_loss_value = float(boundary_shrink_loss.detach().abs().item()) if torch.is_tensor(boundary_shrink_loss) else 0.0
        use_boundary_aware_backward = (
            boundary_aware_enable
            and boundary_effective_score is not None
            and float(boundary_effective_score.max().item()) > 0.0
            and boundary_loss_value > 0.0
        )
        face_boundary_debug_width = max(
            int(_region_opt_value(config.opt, 'face', 'detail_interior_exclude_boundary_width', config.opt.get('detail_interior_exclude_boundary_width', 0))),
            int(_region_opt_value(config.opt, 'face', 'detail_boundary_keep_width', config.opt.get('detail_boundary_keep_width', 0))),
        )
        shoulder_boundary_debug_width = max(
            int(
                _region_opt_value(
                    config.opt,
                    'shoulder_arm',
                    'detail_interior_exclude_boundary_width',
                    config.opt.get('detail_interior_exclude_boundary_width', 0),
                )
            ),
            int(_region_opt_value(config.opt, 'shoulder_arm', 'detail_boundary_keep_width', config.opt.get('detail_boundary_keep_width', 0))),
        )
        upper_torso_boundary_debug_width = max(
            int(_region_opt_value(config.opt, 'upper_torso', 'detail_interior_exclude_boundary_width', config.opt.get('detail_interior_exclude_boundary_width', 0))),
            int(_region_opt_value(config.opt, 'upper_torso', 'detail_boundary_keep_width', config.opt.get('detail_boundary_keep_width', 0))),
        )
        waist_boundary_debug_width = max(
            int(_region_opt_value(config.opt, 'waist', 'detail_interior_exclude_boundary_width', config.opt.get('detail_interior_exclude_boundary_width', 0))),
            int(_region_opt_value(config.opt, 'waist', 'detail_boundary_keep_width', config.opt.get('detail_boundary_keep_width', 0))),
        )
        face_detail_boundary_band = (
            _foreground_boundary_mask(fg_mask, face_boundary_debug_width)
            if face_boundary_debug_width > 1 else torch.zeros_like(face_detail_mask)
        )
        shoulder_detail_boundary_band = (
            _foreground_boundary_mask(fg_mask, shoulder_boundary_debug_width)
            if shoulder_boundary_debug_width > 1 else torch.zeros_like(shoulder_arm_detail_mask)
        )
        upper_torso_detail_boundary_band = (
            _foreground_boundary_mask(fg_mask, upper_torso_boundary_debug_width)
            if upper_torso_boundary_debug_width > 1 else torch.zeros_like(upper_torso_detail_mask)
        )
        waist_detail_boundary_band = (
            _foreground_boundary_mask(fg_mask, waist_boundary_debug_width)
            if waist_boundary_debug_width > 1 else torch.zeros_like(waist_detail_mask)
        )
        texture_clarity_stats = _collect_texture_clarity_stats(getattr(scene.converter, 'texture', None))
        total_loss_value = max(float(loss.detach().item()), 1.0e-8)
        local_clarity_share = (
            weighted_face_image_term
            + weighted_local_image_term
            + weighted_local_boundary_term
        ).detach().item() / total_loss_value
        _maybe_log_clarity_debug(
            config.opt,
            schedule_iteration,
            {
                'face_source': face_region_meta.get('actual_source', ''),
                'face_roi_pixels': face_region_pixels,
                'face_detail_pixels': float(face_detail_mask.sum().item()),
                'face_detail_active_pixels': face_detail_active_pixels,
                'face_detail_boundary_overlap': _mask_overlap_sum(face_detail_mask, face_detail_boundary_band),
                'shoulder_source': shoulder_arm_region_meta.get('actual_source', ''),
                'shoulder_roi_pixels': shoulder_arm_region_pixels,
                'shoulder_detail_pixels': float(shoulder_arm_detail_mask.sum().item()),
                'shoulder_detail_active_pixels': shoulder_arm_detail_active_pixels,
                'shoulder_detail_boundary_overlap': _mask_overlap_sum(shoulder_arm_detail_mask, shoulder_detail_boundary_band),
                'upper_torso_source': upper_torso_region_meta.get('actual_source', ''),
                'upper_torso_roi_pixels': upper_torso_region_pixels,
                'upper_torso_detail_pixels': float(upper_torso_detail_mask.sum().item()),
                'upper_torso_detail_active_pixels': upper_torso_detail_active_pixels,
                'upper_torso_detail_boundary_overlap': _mask_overlap_sum(upper_torso_detail_mask, upper_torso_detail_boundary_band),
                'upper_torso_core_pixels': float(upper_torso_core_mask.sum().item()),
                'upper_torso_core_active_pixels': upper_torso_core_active_pixels,
                'upper_torso_boundary_pixels': float(upper_torso_boundary_supervision_mask.sum().item()),
                'upper_torso_outer_pixels': float(upper_torso_outer_shell_supervision_mask.sum().item()),
                'waist_roi_pixels': float(waist_mask.sum().item()),
                'waist_detail_pixels': float(waist_detail_mask.sum().item()),
                'waist_detail_active_pixels': waist_detail_active_pixels,
                'waist_detail_boundary_overlap': _mask_overlap_sum(waist_detail_mask, waist_detail_boundary_band),
                'fullframe_image': float(weighted_fullframe_image_term.detach().item()),
                'face_image': float(weighted_face_image_term.detach().item()),
                'shoulder_image': float(weighted_shoulder_image_term.detach().item()),
                'upper_torso_image': float(weighted_upper_torso_image_term.detach().item()),
                'waist_image': float(weighted_waist_image_term.detach().item()),
                'local_share': local_clarity_share,
                'face_luma_dog': float((lambda_detail_face_luma_dog * loss_detail_face_luma_dog).detach().item()),
                'shoulder_luma_dog': float((lambda_detail_shoulder_arm_luma_dog * loss_detail_shoulder_arm_luma_dog).detach().item()),
                'upper_torso_luma_dog': float((lambda_detail_upper_torso_luma_dog * loss_detail_upper_torso_luma_dog).detach().item()),
                'upper_torso_core_luma_dog': float((lambda_detail_upper_torso_core_luma_dog * loss_detail_upper_torso_core_luma_dog).detach().item()),
                'face_patch': float((lambda_perceptual_face_patch * loss_perceptual_face_patch).detach().item()),
                'shoulder_patch': float((lambda_perceptual_shoulder_arm_patch * loss_perceptual_shoulder_arm_patch).detach().item()),
                'upper_torso_patch': float((lambda_perceptual_upper_torso_patch * loss_perceptual_upper_torso_patch).detach().item()),
                'upper_torso_core_patch': float((lambda_perceptual_upper_torso_core_patch * loss_perceptual_upper_torso_core_patch).detach().item()),
                'total_loss': total_loss_value,
                'owner_local_detail_boost_enabled': float(int(bool(owner_local_detail_boost.get('enabled', False)))),
                'owner_local_detail_boost_signal': float(owner_local_detail_boost.get('ownership_signal', 0.0)),
                'owner_local_detail_boost_takeover': float(owner_local_detail_boost.get('takeover_mean') or 0.0),
                'owner_local_detail_boost_takeover_signal': float(owner_local_detail_boost.get('takeover_signal', 0.0)),
                'owner_local_detail_boost_legacy_scale': float(owner_local_detail_boost.get('legacy_scale_mean') or 0.0),
                'owner_local_detail_boost_legacy_signal': float(owner_local_detail_boost.get('legacy_signal', 0.0)),
                'owner_local_detail_boost_legacy_mix': float(owner_local_detail_boost.get('legacy_mix', 0.0)),
                'owner_local_detail_boost_face_detail_scale': float(owner_local_detail_boost['detail_scales']['face']),
                'owner_local_detail_boost_face_luma_scale': float(owner_local_detail_boost['luma_scales']['face']),
                'owner_local_detail_boost_face_patch_scale': float(owner_local_detail_boost['patch_scales']['face']),
                'owner_local_detail_boost_face_edge_scale': float(owner_local_detail_boost['edge_scales']['face']),
                'owner_local_detail_boost_shoulder_detail_scale': float(owner_local_detail_boost['detail_scales']['shoulder_arm']),
                'owner_local_detail_boost_shoulder_luma_scale': float(owner_local_detail_boost['luma_scales']['shoulder_arm']),
                'owner_local_detail_boost_shoulder_patch_scale': float(owner_local_detail_boost['patch_scales']['shoulder_arm']),
                'owner_local_detail_boost_shoulder_edge_scale': float(owner_local_detail_boost['edge_scales']['shoulder_arm']),
                'owner_local_detail_boost_shoulder_boundary_scale': float(owner_local_detail_boost['boundary_scales']['shoulder_arm']),
                'owner_local_detail_boost_upper_torso_luma_scale': float(owner_local_detail_boost['luma_scales']['upper_torso']),
                'owner_local_detail_boost_upper_torso_patch_scale': float(owner_local_detail_boost['patch_scales']['upper_torso']),
                'owner_local_detail_boost_upper_torso_boundary_scale': float(owner_local_detail_boost['boundary_scales']['upper_torso']),
                'owner_local_detail_boost_upper_torso_core_luma_scale': float(owner_local_detail_boost['luma_scales']['upper_torso_core']),
                'owner_local_detail_boost_upper_torso_core_patch_scale': float(owner_local_detail_boost['patch_scales']['upper_torso_core']),
                'texture_stats': texture_clarity_stats,
                'texture_trunk_debug': getattr(scene.converter.texture, 'last_structured_trunk_debug', ''),
                'texture_extra_local_debug': getattr(scene.converter.texture, 'last_detail_high_freq_face_extra_local_debug', ''),
                'texture_structure_debug': getattr(scene.converter.texture, 'last_detail_high_freq_structure_debug', ''),
            },
        )

        if not torch.isfinite(loss):
            print(f"[ITER {schedule_iteration}] Non-finite loss detected; skipping iteration.")
            scene.gaussians.optimizer.zero_grad(set_to_none=True)
            scene.converter.optimizer.zero_grad(set_to_none=True)
            removed = scene.gaussians.prune_nonfinite_points(verbose=True)
            if removed <= 0:
                torch.cuda.empty_cache()
            continue

        _maybe_log_shoulder_local_loss_diag(
            config.opt,
            schedule_iteration,
            {
                'local_image': float(weighted_shoulder_image_term.detach().item()),
                'global_image': float(weighted_global_image_term.detach().item()),
                'local_boundary': float(weighted_local_boundary_term.detach().item()),
                'global_boundary': float(weighted_global_boundary_term.detach().item()),
                'base_total': float(base_loss.detach().item()),
                'total_loss': float(loss.detach().item()),
            },
        )
        _maybe_log_waist_local_loss_diag(
            config.opt,
            schedule_iteration,
            {
                'waist_image': float(weighted_waist_image_term.detach().item()),
                'shoulder_image': float(weighted_shoulder_image_term.detach().item()),
                'global_image': float(weighted_global_image_term.detach().item()),
                'region_pixels': float(waist_image_mask.sum().item()),
                'perceptual_pixels': float(waist_perceptual_used_mask.sum().item()),
                'base_total': float(base_loss.detach().item()),
                'total_loss': float(loss.detach().item()),
            },
        )
        _maybe_log_shoulder_local_grad_probe(
            config.opt,
            schedule_iteration,
            image,
            opacity_bce,
            weighted_global_image_term,
            weighted_shoulder_image_term,
            weighted_global_boundary_term,
            weighted_local_boundary_term,
        )

        if diagnostic_no_backward:
            scene.gaussians.optimizer.zero_grad(set_to_none=True)
            scene.converter.optimizer.zero_grad(set_to_none=True)
            if diagnostic_log_interval > 0 and schedule_iteration % diagnostic_log_interval == 0:
                print(f"[TrainDiagnostic] iter={schedule_iteration} skipped backward.")
        elif use_boundary_aware_backward:
            base_loss_value = float(base_loss.detach().abs().item()) if torch.is_tensor(base_loss) else 0.0
            if base_loss_value > 0.0:
                (base_loss * gradient_accumulation_scale).backward(retain_graph=True)
            directional_boundary_specs = []
            boundary_signed_mixed_loss_scale = float(C(
                schedule_iteration,
                config.opt.get('boundary_signed_mixed_loss_scale', 1.0),
            ))
            boundary_signed_shrink_loss_scale = float(C(
                schedule_iteration,
                config.opt.get('boundary_signed_shrink_loss_scale', 1.0),
            ))
            boundary_signed_grow_loss_scale = float(C(
                schedule_iteration,
                config.opt.get('boundary_signed_grow_loss_scale', 1.0),
            ))
            if (
                boundary_signed_routing_enable
                and boundary_signed_mixed_loss_scale > 0.0
                and boundary_mixed_loss_value > 0.0
                and boundary_effective_score is not None
                and float(boundary_effective_score.max().item()) > 0.0
            ):
                directional_boundary_specs.append((
                    boundary_mixed_loss,
                    boundary_effective_score,
                    boundary_signed_mixed_loss_scale,
                ))
            if (
                boundary_signed_routing_enable
                and boundary_signed_shrink_loss_scale > 0.0
                and boundary_shrink_loss_value > 0.0
                and boundary_shrink_effective_score is not None
                and float(boundary_shrink_effective_score.max().item()) > 0.0
            ):
                directional_boundary_specs.append((
                    boundary_shrink_loss,
                    boundary_shrink_effective_score,
                    boundary_signed_shrink_loss_scale,
                ))
            if (
                boundary_signed_routing_enable
                and boundary_signed_grow_loss_scale > 0.0
                and boundary_grow_loss_value > 0.0
                and boundary_grow_effective_score is not None
                and float(boundary_grow_effective_score.max().item()) > 0.0
            ):
                directional_boundary_specs.append((
                    boundary_grow_loss,
                    boundary_grow_effective_score,
                    boundary_signed_grow_loss_scale,
                ))

            if directional_boundary_specs:
                for idx, (boundary_term, boundary_term_score, boundary_term_scale) in enumerate(directional_boundary_specs):
                    if boundary_term_scale != 1.0:
                        boundary_term = boundary_term * boundary_term_scale
                    if gradient_accumulation_scale != 1.0:
                        boundary_term = boundary_term * gradient_accumulation_scale
                    hooks, frozen_converter_params = _register_boundary_grad_hooks(scene, boundary_term_score, config)
                    try:
                        boundary_term.backward(retain_graph=idx < len(directional_boundary_specs) - 1)
                    finally:
                        _remove_boundary_grad_hooks(hooks, frozen_converter_params)
            else:
                hooks, frozen_converter_params = _register_boundary_grad_hooks(scene, boundary_effective_score, config)
                try:
                    (boundary_loss * gradient_accumulation_scale).backward()
                finally:
                    _remove_boundary_grad_hooks(hooks, frozen_converter_params)
        else:
            (loss * gradient_accumulation_scale).backward()

        if not diagnostic_no_backward:
            gauss_bad_grad = _zero_nonfinite_grads(scene.gaussians.optimizer)
            converter_bad_grad = _zero_nonfinite_grads(scene.converter.optimizer)
            if gauss_bad_grad or converter_bad_grad:
                print(f"[ITER {schedule_iteration}] Non-finite gradients detected; zeroed before optimizer step.")

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            elapsed = iter_start.elapsed_time(iter_end)
            log_loss = {
                'loss/l1_loss': loss_l1.item(),
                'loss/l1_fg_loss': loss_l1_fg.item(),
                'loss/l1_boundary_loss': loss_l1_boundary.item(),
                'loss/l1_face_loss': loss_l1_face.item(),
                'loss/edge_face_loss': loss_edge_face.item(),
                'loss/detail_face_loss': loss_detail_face.item(),
                'loss/detail_face_luma_dog_loss': loss_detail_face_luma_dog.item(),
                'loss/perceptual_face_patch_loss': loss_perceptual_face_patch.item(),
                'loss/l1_shoulder_arm_loss': loss_l1_shoulder_arm.item(),
                'loss/edge_shoulder_arm_loss': loss_edge_shoulder_arm.item(),
                'loss/detail_shoulder_arm_loss': loss_detail_shoulder_arm.item(),
                'loss/detail_shoulder_arm_luma_dog_loss': loss_detail_shoulder_arm_luma_dog.item(),
                'loss/perceptual_shoulder_arm_loss': loss_perceptual_shoulder_arm.item(),
                'loss/perceptual_shoulder_arm_patch_loss': loss_perceptual_shoulder_arm_patch.item(),
                'loss/detail_upper_torso_luma_dog_loss': loss_detail_upper_torso_luma_dog.item(),
                'loss/perceptual_upper_torso_patch_loss': loss_perceptual_upper_torso_patch.item(),
                'loss/detail_upper_torso_core_luma_dog_loss': loss_detail_upper_torso_core_luma_dog.item(),
                'loss/perceptual_upper_torso_core_patch_loss': loss_perceptual_upper_torso_core_patch.item(),
                'loss/l1_waist_loss': loss_l1_waist.item(),
                'loss/edge_waist_loss': loss_edge_waist.item(),
                'loss/detail_waist_loss': loss_detail_waist.item(),
                'loss/detail_waist_luma_dog_loss': loss_detail_waist_luma_dog.item(),
                'loss/perceptual_waist_loss': loss_perceptual_waist.item(),
                'loss/perceptual_waist_patch_loss': loss_perceptual_waist_patch.item(),
                'loss/l1_shoulder_focus_dark_outlier_loss': loss_l1_shoulder_focus_dark_outlier.item(),
                'loss/l1_shoulder_focus_bright_outlier_loss': loss_l1_shoulder_focus_bright_outlier.item(),
                'loss/ssim_loss': loss_dssim.item(),
                'loss/perceptual_loss': loss_perceptual.item(),
                'loss/perceptual_face_loss': loss_perceptual_face.item(),
                'loss/mask_loss': loss_mask.item(),
                'loss/mask_boundary_loss': loss_mask_boundary.item(),
                'loss/mask_boundary_hard_loss': loss_mask_boundary_hard.item(),
                'loss/mask_shoulder_arm_boundary_hard_loss': loss_mask_shoulder_arm_boundary_hard.item(),
                'loss/mask_upper_torso_boundary_hard_loss': loss_mask_upper_torso_boundary_hard.item(),
                'loss/mask_shoulder_arm_region_hard_loss': loss_mask_shoulder_arm_region_hard.item(),
                'loss/mask_shoulder_arm_disagreement_hard_loss': loss_mask_shoulder_arm_disagreement_hard.item(),
                'loss/mask_shoulder_focus_small_fp_hard_loss': loss_mask_shoulder_focus_small_fp_hard.item(),
                'loss/mask_shoulder_focus_small_fn_hard_loss': loss_mask_shoulder_focus_small_fn_hard.item(),
                'loss/silhouette_outer_loss': loss_silhouette_outer.item(),
                'loss/silhouette_outer_shell_loss': loss_silhouette_outer_shell.item(),
                'loss/silhouette_head_outer_shell_loss': loss_silhouette_head_outer_shell.item(),
                'loss/silhouette_shoulder_arm_outer_shell_loss': loss_silhouette_shoulder_arm_outer_shell.item(),
                'loss/silhouette_upper_torso_outer_shell_loss': loss_silhouette_upper_torso_outer_shell.item(),
                'loss/silhouette_outer_spike_loss': loss_silhouette_outer_spike.item(),
                'loss/silhouette_outer_fragment_loss': loss_silhouette_outer_fragment.item(),
                'loss/silhouette_outer_bead_loss': loss_silhouette_outer_bead.item(),
                'loss/silhouette_outer_chain_loss': loss_silhouette_outer_chain.item(),
                'loss/silhouette_arm_stipple_loss': loss_silhouette_arm_stipple.item(),
                'loss/silhouette_arm_tail_loss': loss_silhouette_arm_tail.item(),
                'loss/silhouette_arm_fringe_loss': loss_silhouette_arm_fringe.item(),
                'loss/silhouette_arm_attached_fragment_loss': loss_silhouette_arm_attached_fragment.item(),
                'loss/silhouette_shoulder_attached_fragment_loss': loss_silhouette_shoulder_attached_fragment.item(),
                'loss/silhouette_arm_notch_loss': loss_silhouette_arm_notch.item(),
                'loss/silhouette_arm_hole_loss': loss_silhouette_arm_hole.item(),
                'loss/silhouette_arm_gap_loss': loss_silhouette_arm_gap.item(),
                'loss/silhouette_shoulder_bead_loss': loss_silhouette_shoulder_bead.item(),
                'loss/silhouette_shoulder_chain_loss': loss_silhouette_shoulder_chain.item(),
                'loss/silhouette_shoulder_hole_loss': loss_silhouette_shoulder_hole.item(),
                'loss/silhouette_shoulder_gap_loss': loss_silhouette_shoulder_gap.item(),
                'loss/silhouette_shoulder_pinhole_loss': loss_silhouette_shoulder_pinhole.item(),
                'loss/silhouette_inner_loss': loss_silhouette_inner.item(),
                'loss/loss_skinning': loss_skinning.item(),
                'loss/xyz_aiap_loss': loss_aiap_xyz.item(),
                'loss/cov_aiap_loss': loss_aiap_cov.item(),
                'loss/boundary_opacity_residual_reg': loss_boundary_opacity_residual_reg.item(),
                'loss/boundary_scaling_residual_reg': loss_boundary_scaling_residual_reg.item(),
                'loss/boundary_opacity_residual_smooth': loss_boundary_opacity_residual_smooth.item(),
                'loss/boundary_scaling_residual_smooth': loss_boundary_scaling_residual_smooth.item(),
                'loss/stageB_semantic_loss': loss_stageB_semantic.item(),
                'loss/stageB_semantic_enabled': float(stageB_semantic_stats.get('enabled', 0.0)),
                'loss/stageB_semantic_valid_pixels': float(stageB_semantic_stats.get('valid_pixels', 0.0)),
                'loss/stageB_semantic_body_cloth': float(stageB_semantic_stats.get('body_cloth', 0.0)),
                'loss/stageB_semantic_compact': float(stageB_semantic_stats.get('compact', 0.0)),
                'loss/stageB_semantic_compact_ce': float(stageB_semantic_stats.get('compact_ce', 0.0)),
                'loss/stageB_semantic_parent': float(stageB_semantic_stats.get('parent', 0.0)),
                'loss/stageB_semantic_exclusive': float(stageB_semantic_stats.get('exclusive', 0.0)),
                'loss/stageB_semantic_smooth': float(stageB_semantic_stats.get('smooth', 0.0)),
                'loss/binding_semantic_adapter_reg': loss_binding_semantic_adapter_reg.item(),
                'loss/boundary_total_loss': boundary_loss.item(),
                'loss/boundary_mixed_loss': boundary_mixed_loss.item(),
                'loss/boundary_grow_loss': boundary_grow_loss.item(),
                'loss/boundary_shrink_loss': boundary_shrink_loss.item(),
                'loss/base_total_loss': base_loss.item(),
                'loss/fullframe_image_weighted': weighted_fullframe_image_term.item(),
                'loss/face_local_image_weighted': weighted_face_image_term.item(),
                'loss/global_image_weighted': weighted_global_image_term.item(),
                'loss/shoulder_local_image_weighted': weighted_shoulder_image_term.item(),
                'loss/upper_torso_local_image_weighted': weighted_upper_torso_image_term.item(),
                'loss/waist_local_image_weighted': weighted_waist_image_term.item(),
                'loss/local_image_weighted_combined': weighted_local_image_term.item(),
                'loss/global_boundary_weighted': weighted_global_boundary_term.item(),
                'loss/shoulder_local_boundary_weighted': weighted_shoulder_boundary_term.item(),
                'loss/upper_torso_local_boundary_weighted': weighted_upper_torso_boundary_term.item(),
                'loss/local_boundary_weighted_combined': weighted_local_boundary_term.item(),
                'loss/photometric_correction_strength': float(photometric_correction_debug.get('strength', 0.0)),
                'loss/photometric_correction_pixels': float(photometric_correction_debug.get('active_pixels', 0.0)),
                'loss/photometric_correction_scale_mean': float(photometric_correction_debug.get('scale_mean', 1.0)),
                'loss/photometric_correction_shift_abs_mean': float(photometric_correction_debug.get('shift_abs_mean', 0.0)),
                'loss/contour_uncertainty_pixels': float(contour_uncertainty_debug.get('uncertain_pixels', 0.0)),
                'loss/contour_uncertainty_min_weight': float(contour_uncertainty_debug.get('min_weight', 1.0)),
                'loss/camera_affine_reg': loss_reg.get('camera_affine_reg', torch.tensor(0.0, device='cuda')).item(),
                'loss/camera_affine_active': float(getattr(scene.converter.camera_affine, 'last_debug', {}).get('active', 0.0)),
                'loss/camera_affine_strength': float(getattr(scene.converter.camera_affine, 'last_debug', {}).get('strength', 0.0)),
                'loss/camera_affine_scale_delta_abs_mean': float(getattr(scene.converter.camera_affine, 'last_debug', {}).get('scale_delta_abs_mean', 0.0)),
                'loss/camera_affine_shift_abs_mean': float(getattr(scene.converter.camera_affine, 'last_debug', {}).get('shift_abs_mean', 0.0)),
                'loss/camera_geometry_reg': loss_reg.get('camera_geometry_reg', torch.tensor(0.0, device='cuda')).item(),
                'loss/camera_geometry_active': float(getattr(scene.converter.camera_geometry, 'last_debug', {}).get('active', 0.0)),
                'loss/camera_geometry_strength': float(getattr(scene.converter.camera_geometry, 'last_debug', {}).get('strength', 0.0)),
                'loss/camera_geometry_rot_deg_abs_mean': float(getattr(scene.converter.camera_geometry, 'last_debug', {}).get('rot_deg_abs_mean', 0.0)),
                'loss/camera_geometry_trans_abs_mean': float(getattr(scene.converter.camera_geometry, 'last_debug', {}).get('trans_abs_mean', 0.0)),
                'loss/local_clarity_total_share': local_clarity_share,
                'loss/face_local_total_share': weighted_face_image_term.item() / max(loss.item(), 1.0e-8),
                'loss/shoulder_local_total_share': (weighted_shoulder_image_term + weighted_shoulder_boundary_term).item() / max(loss.item(), 1.0e-8),
                'loss/upper_torso_local_total_share': (weighted_upper_torso_image_term + weighted_upper_torso_boundary_term).item() / max(loss.item(), 1.0e-8),
                'loss/waist_local_total_share': weighted_waist_image_term.item() / max(loss.item(), 1.0e-8),
                'loss/owner_local_detail_boost_enabled': float(int(bool(owner_local_detail_boost.get('enabled', False)))),
                'loss/owner_local_detail_boost_signal': float(owner_local_detail_boost.get('ownership_signal', 0.0)),
                'loss/owner_local_detail_boost_takeover': float(owner_local_detail_boost.get('takeover_mean') or 0.0),
                'loss/owner_local_detail_boost_takeover_signal': float(owner_local_detail_boost.get('takeover_signal', 0.0)),
                'loss/owner_local_detail_boost_legacy_scale': float(owner_local_detail_boost.get('legacy_scale_mean') or 0.0),
                'loss/owner_local_detail_boost_legacy_signal': float(owner_local_detail_boost.get('legacy_signal', 0.0)),
                'loss/owner_local_detail_boost_legacy_mix': float(owner_local_detail_boost.get('legacy_mix', 0.0)),
                'loss/owner_local_detail_boost_face_detail_scale': float(owner_local_detail_boost['detail_scales']['face']),
                'loss/owner_local_detail_boost_face_luma_scale': float(owner_local_detail_boost['luma_scales']['face']),
                'loss/owner_local_detail_boost_face_patch_scale': float(owner_local_detail_boost['patch_scales']['face']),
                'loss/owner_local_detail_boost_face_edge_scale': float(owner_local_detail_boost['edge_scales']['face']),
                'loss/owner_local_detail_boost_shoulder_detail_scale': float(owner_local_detail_boost['detail_scales']['shoulder_arm']),
                'loss/owner_local_detail_boost_shoulder_luma_scale': float(owner_local_detail_boost['luma_scales']['shoulder_arm']),
                'loss/owner_local_detail_boost_shoulder_patch_scale': float(owner_local_detail_boost['patch_scales']['shoulder_arm']),
                'loss/owner_local_detail_boost_shoulder_edge_scale': float(owner_local_detail_boost['edge_scales']['shoulder_arm']),
                'loss/owner_local_detail_boost_shoulder_boundary_scale': float(owner_local_detail_boost['boundary_scales']['shoulder_arm']),
                'loss/owner_local_detail_boost_upper_torso_luma_scale': float(owner_local_detail_boost['luma_scales']['upper_torso']),
                'loss/owner_local_detail_boost_upper_torso_patch_scale': float(owner_local_detail_boost['patch_scales']['upper_torso']),
                'loss/owner_local_detail_boost_upper_torso_boundary_scale': float(owner_local_detail_boost['boundary_scales']['upper_torso']),
                'loss/owner_local_detail_boost_upper_torso_core_luma_scale': float(owner_local_detail_boost['luma_scales']['upper_torso_core']),
                'loss/owner_local_detail_boost_upper_torso_core_patch_scale': float(owner_local_detail_boost['patch_scales']['upper_torso_core']),
                'loss/reliable_view_enabled': float(int(bool(reliable_view_debug.get('enabled', False)))),
                'loss/reliable_view_camera_id': float(reliable_view_debug.get('camera_id', -1.0)),
                'loss/reliable_view_raw_highfreq_weight': float(reliable_view_debug.get('raw_weight', 1.0)),
                'loss/reliable_view_highfreq_weight': float(reliable_view_debug.get('weight', 1.0)),
                'loss/face_region_pixels': face_region_pixels,
                'loss/face_region_min_pixels': float(face_region_min_pixels),
                'loss/face_region_has_parser': float(int(bool(face_region_meta.get('has_parser', False)))),
                'loss/face_region_has_joint': float(int(bool(face_region_meta.get('has_joint', False)))),
                'loss/head_outer_region_pixels': float(head_outer_region_meta.get('region_pixels', head_outer_region_mask.sum().item())),
                'loss/head_outer_region_fallback': float(int(bool(head_outer_region_meta.get('fallback_used', False)))),
                'loss/upper_torso_region_pixels': upper_torso_region_pixels,
                'loss/upper_torso_region_min_pixels': float(upper_torso_region_min_pixels),
                'loss/upper_torso_region_has_parser': float(int(bool(upper_torso_region_meta.get('has_parser', False)))),
                'loss/upper_torso_region_has_joint': float(int(bool(upper_torso_region_meta.get('has_joint', False)))),
                'loss/face_image_region_pixels': float(face_mask.sum().item()),
                'loss/face_detail_region_active_pixels': face_detail_active_pixels,
                'loss/face_detail_boundary_overlap_pixels': _mask_overlap_sum(face_detail_mask, face_detail_boundary_band),
                'loss/shoulder_image_region_pixels': float(shoulder_arm_image_mask.sum().item()),
                'loss/shoulder_perceptual_region_pixels': float(shoulder_arm_perceptual_used_mask.sum().item()),
                'loss/shoulder_boundary_region_pixels': float(shoulder_arm_boundary_supervision_mask.sum().item()),
                'loss/shoulder_region_hard_pixels': float(shoulder_arm_region_hard_supervision_mask.sum().item()),
                'loss/shoulder_disagreement_region_pixels': float(shoulder_arm_disagreement_supervision_mask.sum().item()),
                'loss/shoulder_outer_shell_region_pixels': float(shoulder_arm_outer_shell_supervision_mask.sum().item()),
                'loss/shoulder_detail_region_pixels': float(shoulder_arm_detail_mask.sum().item()),
                'loss/shoulder_detail_region_active_pixels': shoulder_arm_detail_active_pixels,
                'loss/shoulder_detail_boundary_overlap_pixels': _mask_overlap_sum(shoulder_arm_detail_mask, shoulder_detail_boundary_band),
                'loss/upper_torso_image_region_pixels': float(upper_torso_image_mask.sum().item()),
                'loss/upper_torso_boundary_region_pixels': float(upper_torso_boundary_supervision_mask.sum().item()),
                'loss/upper_torso_outer_shell_region_pixels': float(upper_torso_outer_shell_supervision_mask.sum().item()),
                'loss/upper_torso_detail_region_pixels': float(upper_torso_detail_mask.sum().item()),
                'loss/upper_torso_detail_region_active_pixels': upper_torso_detail_active_pixels,
                'loss/upper_torso_detail_boundary_overlap_pixels': _mask_overlap_sum(upper_torso_detail_mask, upper_torso_detail_boundary_band),
                'loss/upper_torso_perceptual_region_pixels': float(upper_torso_perceptual_used_mask.sum().item()),
                'loss/upper_torso_core_region_pixels': float(upper_torso_core_mask.sum().item()),
                'loss/upper_torso_core_region_active_pixels': upper_torso_core_active_pixels,
                'loss/upper_torso_core_perceptual_region_pixels': float(upper_torso_core_perceptual_used_mask.sum().item()),
                'loss/waist_image_region_pixels': float(waist_image_mask.sum().item()),
                'loss/waist_detail_region_pixels': float(waist_detail_mask.sum().item()),
                'loss/waist_detail_region_active_pixels': waist_detail_active_pixels,
                'loss/waist_detail_boundary_overlap_pixels': _mask_overlap_sum(waist_detail_mask, waist_detail_boundary_band),
                'loss/waist_perceptual_region_pixels': float(waist_perceptual_used_mask.sum().item()),
                'loss/face_detail_region_pixels': float(face_detail_mask.sum().item()),
                'loss/face_patch_count': float(face_patch_count),
                'loss/shoulder_arm_patch_count': float(shoulder_arm_patch_count),
                'loss/upper_torso_patch_count': float(upper_torso_patch_count),
                'loss/upper_torso_core_patch_count': float(upper_torso_core_patch_count),
                'loss/waist_patch_count': float(waist_patch_count),
                'loss/total_loss': loss.item(),
                'iter_time': elapsed,
            }
            for key, value in stageB_semantic_stats.items():
                if key.startswith('compact_') and key not in ('compact', 'compact_ce'):
                    log_loss[f'loss/stageB_semantic_{key}'] = float(value)
            structured_trunk_total_abs_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_total_abs_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_total_abs_mean):
                log_loss['loss/texture_structured_trunk_total_abs_mean'] = (
                    structured_trunk_total_abs_mean.item()
                )
            structured_trunk_shared_abs_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_shared_abs_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_shared_abs_mean):
                log_loss['loss/texture_structured_trunk_shared_abs_mean'] = (
                    structured_trunk_shared_abs_mean.item()
                )
            structured_trunk_carrier_abs_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_carrier_abs_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_carrier_abs_mean):
                log_loss['loss/texture_structured_trunk_carrier_abs_mean'] = (
                    structured_trunk_carrier_abs_mean.item()
                )
            structured_trunk_structure_abs_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_structure_abs_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_structure_abs_mean):
                log_loss['loss/texture_structured_trunk_structure_abs_mean'] = (
                    structured_trunk_structure_abs_mean.item()
                )
            structured_trunk_structure_raw_abs_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_structure_raw_abs_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_structure_raw_abs_mean):
                log_loss['loss/texture_structured_trunk_structure_raw_abs_mean'] = (
                    structured_trunk_structure_raw_abs_mean.item()
                )
            structured_trunk_local_abs_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_local_abs_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_local_abs_mean):
                log_loss['loss/texture_structured_trunk_local_abs_mean'] = (
                    structured_trunk_local_abs_mean.item()
                )
            structured_trunk_local_raw_abs_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_local_raw_abs_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_local_raw_abs_mean):
                log_loss['loss/texture_structured_trunk_local_raw_abs_mean'] = (
                    structured_trunk_local_raw_abs_mean.item()
                )
            structured_trunk_local_gate_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_local_gate_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_local_gate_mean):
                log_loss['loss/texture_structured_trunk_local_gate_mean'] = (
                    structured_trunk_local_gate_mean.item()
                )
            structured_trunk_owner_input_abs_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_owner_input_abs_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_owner_input_abs_mean):
                log_loss['loss/texture_structured_trunk_owner_input_abs_mean'] = (
                    structured_trunk_owner_input_abs_mean.item()
                )
            structured_trunk_owner_abs_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_owner_abs_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_owner_abs_mean):
                log_loss['loss/texture_structured_trunk_owner_abs_mean'] = (
                    structured_trunk_owner_abs_mean.item()
                )
            structured_trunk_owner_color_abs_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_owner_color_abs_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_owner_color_abs_mean):
                log_loss['loss/texture_structured_trunk_owner_color_abs_mean'] = (
                    structured_trunk_owner_color_abs_mean.item()
                )
            structured_trunk_owner_support_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_owner_support_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_owner_support_mean):
                log_loss['loss/texture_structured_trunk_owner_support_mean'] = (
                    structured_trunk_owner_support_mean.item()
                )
            structured_trunk_owner_gate_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_owner_gate_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_owner_gate_mean):
                log_loss['loss/texture_structured_trunk_owner_gate_mean'] = (
                    structured_trunk_owner_gate_mean.item()
                )
            structured_trunk_owner_takeover_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_owner_takeover_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_owner_takeover_mean):
                log_loss['loss/texture_structured_trunk_owner_takeover_mean'] = (
                    structured_trunk_owner_takeover_mean.item()
                )
            structured_trunk_owner_takeover_legacy_scale_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_owner_takeover_legacy_scale_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_owner_takeover_legacy_scale_mean):
                log_loss['loss/texture_structured_trunk_owner_takeover_legacy_scale_mean'] = (
                    structured_trunk_owner_takeover_legacy_scale_mean.item()
                )
            structured_trunk_owner_boundary_abs_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_owner_boundary_abs_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_owner_boundary_abs_mean):
                log_loss['loss/texture_structured_trunk_owner_boundary_abs_mean'] = (
                    structured_trunk_owner_boundary_abs_mean.item()
                )
            structured_trunk_owner_boundary_input_abs_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_owner_boundary_input_abs_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_owner_boundary_input_abs_mean):
                log_loss['loss/texture_structured_trunk_owner_boundary_input_abs_mean'] = (
                    structured_trunk_owner_boundary_input_abs_mean.item()
                )
            structured_trunk_owner_boundary_color_abs_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_owner_boundary_color_abs_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_owner_boundary_color_abs_mean):
                log_loss['loss/texture_structured_trunk_owner_boundary_color_abs_mean'] = (
                    structured_trunk_owner_boundary_color_abs_mean.item()
                )
            structured_trunk_owner_boundary_focus_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_owner_boundary_focus_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_owner_boundary_focus_mean):
                log_loss['loss/texture_structured_trunk_owner_boundary_focus_mean'] = (
                    structured_trunk_owner_boundary_focus_mean.item()
                )
            structured_trunk_owner_boundary_gate_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_owner_boundary_gate_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_owner_boundary_gate_mean):
                log_loss['loss/texture_structured_trunk_owner_boundary_gate_mean'] = (
                    structured_trunk_owner_boundary_gate_mean.item()
                )
            structured_trunk_owner_boundary_takeover_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_owner_boundary_takeover_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_owner_boundary_takeover_mean):
                log_loss['loss/texture_structured_trunk_owner_boundary_takeover_mean'] = (
                    structured_trunk_owner_boundary_takeover_mean.item()
                )
            structured_trunk_scaffold_abs_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_scaffold_abs_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_scaffold_abs_mean):
                log_loss['loss/texture_structured_trunk_scaffold_abs_mean'] = (
                    structured_trunk_scaffold_abs_mean.item()
                )
            structured_trunk_coarse_abs_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_coarse_abs_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_coarse_abs_mean):
                log_loss['loss/texture_structured_trunk_coarse_abs_mean'] = (
                    structured_trunk_coarse_abs_mean.item()
                )
            structured_trunk_hf_abs_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_hf_abs_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_hf_abs_mean):
                log_loss['loss/texture_structured_trunk_hf_abs_mean'] = (
                    structured_trunk_hf_abs_mean.item()
                )
            structured_trunk_hf_color_abs_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_hf_color_abs_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_hf_color_abs_mean):
                log_loss['loss/texture_structured_trunk_hf_color_abs_mean'] = (
                    structured_trunk_hf_color_abs_mean.item()
                )
            structured_trunk_hf_gate_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_hf_gate_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_hf_gate_mean):
                log_loss['loss/texture_structured_trunk_hf_gate_mean'] = (
                    structured_trunk_hf_gate_mean.item()
                )
            structured_trunk_hf_local_color_abs_mean = getattr(
                scene.converter.texture,
                'last_structured_trunk_hf_local_color_abs_mean',
                None,
            )
            if torch.is_tensor(structured_trunk_hf_local_color_abs_mean):
                log_loss['loss/texture_structured_trunk_hf_local_color_abs_mean'] = (
                    structured_trunk_hf_local_color_abs_mean.item()
                )
            detail_residual_abs_mean = getattr(scene.converter.texture, 'last_detail_residual_abs_mean', None)
            if torch.is_tensor(detail_residual_abs_mean):
                log_loss['loss/texture_detail_residual_abs_mean'] = detail_residual_abs_mean.item()
                log_loss['loss/texture_detail_residual_scale'] = float(getattr(scene.converter.texture, 'last_detail_scale', 0.0))
                log_loss['loss/texture_detail_schedule_iteration'] = float(
                    getattr(scene.converter.texture, 'last_detail_schedule_iteration', 0.0)
                )
            detail_gate_mean = getattr(scene.converter.texture, 'last_detail_gate_mean', None)
            if torch.is_tensor(detail_gate_mean):
                log_loss['loss/texture_detail_gate_mean'] = detail_gate_mean.item()
            detail_gate_fraction = getattr(scene.converter.texture, 'last_detail_gate_fraction', None)
            if torch.is_tensor(detail_gate_fraction):
                log_loss['loss/texture_detail_gate_fraction'] = detail_gate_fraction.item()
            detail_high_freq_residual_abs_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_residual_abs_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_residual_abs_mean):
                log_loss['loss/texture_detail_high_freq_residual_abs_mean'] = (
                    detail_high_freq_residual_abs_mean.item()
                )
                log_loss['loss/texture_detail_high_freq_scale'] = float(
                    getattr(scene.converter.texture, 'last_detail_high_freq_scale', 0.0)
                )
            detail_high_freq_gate_mean = getattr(scene.converter.texture, 'last_detail_high_freq_gate_mean', None)
            if torch.is_tensor(detail_high_freq_gate_mean):
                log_loss['loss/texture_detail_high_freq_gate_mean'] = detail_high_freq_gate_mean.item()
            detail_high_freq_gate_fraction = getattr(scene.converter.texture, 'last_detail_high_freq_gate_fraction', None)
            if torch.is_tensor(detail_high_freq_gate_fraction):
                log_loss['loss/texture_detail_high_freq_gate_fraction'] = (
                    detail_high_freq_gate_fraction.item()
                )
            detail_high_freq_point_gate_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_point_gate_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_point_gate_mean):
                log_loss['loss/texture_detail_high_freq_point_gate_mean'] = (
                    detail_high_freq_point_gate_mean.item()
                )
            detail_high_freq_point_gate_fraction = getattr(
                scene.converter.texture,
                'last_detail_high_freq_point_gate_fraction',
                None,
            )
            if torch.is_tensor(detail_high_freq_point_gate_fraction):
                log_loss['loss/texture_detail_high_freq_point_gate_fraction'] = (
                    detail_high_freq_point_gate_fraction.item()
                )
            detail_high_freq_boundary_floor_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_boundary_floor_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_boundary_floor_mean):
                log_loss['loss/texture_detail_high_freq_boundary_floor_mean'] = (
                    detail_high_freq_boundary_floor_mean.item()
                )
            detail_high_freq_view_conflict_abs_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_view_conflict_abs_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_view_conflict_abs_mean):
                log_loss['loss/texture_detail_high_freq_view_conflict_abs_mean'] = (
                    detail_high_freq_view_conflict_abs_mean.item()
                )
                log_loss['loss/texture_detail_high_freq_view_conflict_scale'] = float(
                    getattr(scene.converter.texture, 'last_detail_high_freq_view_conflict_scale', 0.0)
                )
            detail_high_freq_view_conflict_gate_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_view_conflict_gate_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_view_conflict_gate_mean):
                log_loss['loss/texture_detail_high_freq_view_conflict_gate_mean'] = (
                    detail_high_freq_view_conflict_gate_mean.item()
                )
            detail_high_freq_view_conflict_point_gate_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_view_conflict_point_gate_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_view_conflict_point_gate_mean):
                log_loss['loss/texture_detail_high_freq_view_conflict_point_gate_mean'] = (
                    detail_high_freq_view_conflict_point_gate_mean.item()
                )
            detail_high_freq_view_conflict_boundary_suppress_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_view_conflict_boundary_suppress_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_view_conflict_boundary_suppress_mean):
                log_loss['loss/texture_detail_high_freq_view_conflict_boundary_suppress_mean'] = (
                    detail_high_freq_view_conflict_boundary_suppress_mean.item()
                )
            detail_high_freq_carrier_abs_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_carrier_abs_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_carrier_abs_mean):
                log_loss['loss/texture_detail_high_freq_carrier_abs_mean'] = (
                    detail_high_freq_carrier_abs_mean.item()
                )
            detail_high_freq_structure_abs_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_structure_abs_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_structure_abs_mean):
                log_loss['loss/texture_detail_high_freq_structure_abs_mean'] = (
                    detail_high_freq_structure_abs_mean.item()
                )
            detail_high_freq_structure_raw_abs_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_structure_raw_abs_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_structure_raw_abs_mean):
                log_loss['loss/texture_detail_high_freq_structure_raw_abs_mean'] = (
                    detail_high_freq_structure_raw_abs_mean.item()
                )
            detail_high_freq_chroma_abs_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_chroma_abs_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_chroma_abs_mean):
                log_loss['loss/texture_detail_high_freq_chroma_abs_mean'] = (
                    detail_high_freq_chroma_abs_mean.item()
                )
            detail_high_freq_luma_abs_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_luma_abs_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_luma_abs_mean):
                log_loss['loss/texture_detail_high_freq_luma_abs_mean'] = (
                    detail_high_freq_luma_abs_mean.item()
                )
            detail_high_freq_face_abs_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_face_abs_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_face_abs_mean):
                log_loss['loss/texture_detail_high_freq_face_abs_mean'] = (
                    detail_high_freq_face_abs_mean.item()
                )
            detail_high_freq_face_local_abs_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_face_local_abs_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_face_local_abs_mean):
                log_loss['loss/texture_detail_high_freq_face_local_abs_mean'] = (
                    detail_high_freq_face_local_abs_mean.item()
                )
            detail_high_freq_face_local_raw_abs_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_face_local_raw_abs_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_face_local_raw_abs_mean):
                log_loss['loss/texture_detail_high_freq_face_local_raw_abs_mean'] = (
                    detail_high_freq_face_local_raw_abs_mean.item()
                )
            detail_high_freq_face_extra_local_abs_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_face_extra_local_abs_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_face_extra_local_abs_mean):
                log_loss['loss/texture_detail_high_freq_face_extra_local_abs_mean'] = (
                    detail_high_freq_face_extra_local_abs_mean.item()
                )
            detail_high_freq_face_extra_local_raw_abs_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_face_extra_local_raw_abs_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_face_extra_local_raw_abs_mean):
                log_loss['loss/texture_detail_high_freq_face_extra_local_raw_abs_mean'] = (
                    detail_high_freq_face_extra_local_raw_abs_mean.item()
                )
            detail_high_freq_face_extra_local_gate_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_face_extra_local_gate_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_face_extra_local_gate_mean):
                log_loss['loss/texture_detail_high_freq_face_extra_local_gate_mean'] = (
                    detail_high_freq_face_extra_local_gate_mean.item()
                )
            detail_high_freq_face_gate_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_face_gate_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_face_gate_mean):
                log_loss['loss/texture_detail_high_freq_face_gate_mean'] = (
                    detail_high_freq_face_gate_mean.item()
                )
            detail_high_freq_face_gate_fraction = getattr(
                scene.converter.texture,
                'last_detail_high_freq_face_gate_fraction',
                None,
            )
            if torch.is_tensor(detail_high_freq_face_gate_fraction):
                log_loss['loss/texture_detail_high_freq_face_gate_fraction'] = (
                    detail_high_freq_face_gate_fraction.item()
                )
            detail_high_freq_face_point_gate_mean = getattr(
                scene.converter.texture,
                'last_detail_high_freq_face_point_gate_mean',
                None,
            )
            if torch.is_tensor(detail_high_freq_face_point_gate_mean):
                log_loss['loss/texture_detail_high_freq_face_point_gate_mean'] = (
                    detail_high_freq_face_point_gate_mean.item()
                )
            detail_high_freq_face_point_gate_fraction = getattr(
                scene.converter.texture,
                'last_detail_high_freq_face_point_gate_fraction',
                None,
            )
            if torch.is_tensor(detail_high_freq_face_point_gate_fraction):
                log_loss['loss/texture_detail_high_freq_face_point_gate_fraction'] = (
                    detail_high_freq_face_point_gate_fraction.item()
                )
            if boundary_score is not None:
                log_loss['loss/boundary_score_mean'] = boundary_score.mean().item()
                log_loss['loss/boundary_score_max'] = boundary_score.max().item()
            if _boundary_live_score_cache_enabled(config):
                log_loss['loss/boundary_live_cache_hit'] = float(int(boundary_live_cache_hit))
                log_loss['loss/boundary_live_cache_entries'] = float(len(boundary_live_score_cache))
            boundary_prior_score = _get_boundary_score_tensor(
                render_pkg["deformed_gaussian"],
                prefer_mixed=False,
            )
            if boundary_prior_score is not None:
                log_loss['loss/boundary_prior_mean'] = boundary_prior_score.detach().float().mean().item()
                log_loss['loss/boundary_prior_max'] = boundary_prior_score.detach().float().max().item()
            boundary_image_score = getattr(render_pkg["deformed_gaussian"], 'binding_boundary_image_score', None)
            if torch.is_tensor(boundary_image_score):
                log_loss['loss/boundary_image_mean'] = boundary_image_score.detach().float().mean().item()
                log_loss['loss/boundary_image_max'] = boundary_image_score.detach().float().max().item()
            boundary_image_under_score = getattr(render_pkg["deformed_gaussian"], 'binding_boundary_image_under_score', None)
            if torch.is_tensor(boundary_image_under_score):
                log_loss['loss/boundary_image_under_mean'] = boundary_image_under_score.detach().float().mean().item()
                log_loss['loss/boundary_image_under_max'] = boundary_image_under_score.detach().float().max().item()
            boundary_image_over_score = getattr(render_pkg["deformed_gaussian"], 'binding_boundary_image_over_score', None)
            if torch.is_tensor(boundary_image_over_score):
                log_loss['loss/boundary_image_over_mean'] = boundary_image_over_score.detach().float().mean().item()
                log_loss['loss/boundary_image_over_max'] = boundary_image_over_score.detach().float().max().item()
            if boundary_effective_score is not None:
                log_loss['loss/boundary_effective_mean'] = boundary_effective_score.mean().item()
                log_loss['loss/boundary_effective_max'] = boundary_effective_score.max().item()
                if bool(config.opt.get('boundary_subset_enable', False)):
                    log_loss['loss/boundary_subset_fraction'] = (boundary_effective_score > 0).float().mean().item()
            if boundary_grow_effective_score is not None:
                log_loss['loss/boundary_grow_effective_mean'] = boundary_grow_effective_score.mean().item()
                log_loss['loss/boundary_grow_effective_max'] = boundary_grow_effective_score.max().item()
            if boundary_shrink_effective_score is not None:
                log_loss['loss/boundary_shrink_effective_mean'] = boundary_shrink_effective_score.mean().item()
                log_loss['loss/boundary_shrink_effective_max'] = boundary_shrink_effective_score.max().item()
            boundary_tags = scene.gaussians.get_boundary_tags()
            if boundary_tags is not None:
                log_loss['loss/boundary_tag_fraction'] = (boundary_tags > 0).float().mean().item()
                log_loss['loss/boundary_tag_mean'] = boundary_tags.mean().item()
                log_loss['loss/boundary_opacity_residual_abs_mean'] = (scene.gaussians._boundary_opacity_residual.abs() * boundary_tags.unsqueeze(-1)).sum().item() / boundary_tags.sum().clamp_min(1.0).item()
                log_loss['loss/boundary_scaling_residual_abs_mean'] = (scene.gaussians._boundary_scaling_residual.abs() * boundary_tags.unsqueeze(-1)).sum().item() / boundary_tags.sum().clamp_min(1.0).item()
            log_loss.update({
                'loss/loss_' + k: v for k, v in loss_reg.items()
            })
            wandb.log(log_loss)

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if (
                _boundary_live_score_cache_enabled(config)
                and bool(config.opt.get('boundary_live_score_cache_clear_before_eval', True))
                and hasattr(scene.gaussians, 'set_live_boundary_score_state')
            ):
                scene.gaussians.set_live_boundary_score_state(None)
            val_metrics = validation(schedule_iteration, testing_iterations, testing_interval, scene, evaluator, (pipe, background), interval_iteration=iteration)
            test_metrics = None
            selected_metric_source = None
            if val_metrics is not None:
                metric_sources = ['best_eval', 'test'] if best_metric_source == 'auto' else [best_metric_source, 'test']
                seen_sources = set()
                for metric_source in metric_sources:
                    if metric_source in seen_sources:
                        continue
                    seen_sources.add(metric_source)
                    metric_values = val_metrics.get(metric_source)
                    if metric_values is not None:
                        test_metrics = metric_values
                        selected_metric_source = metric_source
                        break
            if test_metrics is not None:
                fallback_score = float('-inf') if best_metric_mode == 'max' else float('inf')
                current_score = float(test_metrics.get(best_metric_name, test_metrics.get('psnr', fallback_score)))
                is_better = current_score > best_test_score if best_metric_mode == 'max' else current_score < best_test_score
                if is_better:
                    best_test_score = current_score
                    save_best_checkpoint(
                        scene,
                        schedule_iteration,
                        test_metrics,
                        metric_name=best_metric_name,
                        metric_source=selected_metric_source or 'test',
                    )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(schedule_iteration))
                scene.save(schedule_iteration)

            # Densification
            if allow_densify and schedule_iteration < opt.densify_until_iter and schedule_iteration > model.gaussian.delay:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if schedule_iteration > opt.densify_from_iter and schedule_iteration % opt.densification_interval == 0:
                    size_threshold = 20 if schedule_iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt, scene, size_threshold, iteration=schedule_iteration)
                    densify_debug_verbose = bool(model.gaussian.get('binding_densify_debug_verbose', False))
                    if densify_debug_verbose:
                        binding_state = scene.gaussians.get_binding_state() if hasattr(scene.gaussians, 'get_binding_state') else {}
                        refresh_mask = binding_state.get('anchor_refresh_mask', None) if isinstance(binding_state, dict) else None
                        risky_child_mask = binding_state.get('densify_risky_child_mask', None) if isinstance(binding_state, dict) else None
                        refresh_count = int(refresh_mask.to(dtype=torch.bool).sum().item()) if torch.is_tensor(refresh_mask) else 0
                        risky_count = int(risky_child_mask.to(dtype=torch.bool).sum().item()) if torch.is_tensor(risky_child_mask) else 0
                        overlap_count = 0
                        if (
                            torch.is_tensor(refresh_mask)
                            and torch.is_tensor(risky_child_mask)
                            and refresh_mask.shape[0] == risky_child_mask.shape[0]
                        ):
                            overlap_count = int(
                                (
                                    refresh_mask.to(dtype=torch.bool)
                                    & risky_child_mask.to(dtype=torch.bool)
                                ).sum().item()
                            )
                        print(
                            '[Train] post-densify binding state '
                            f'iter={schedule_iteration} points={scene.gaussians.get_xyz.shape[0]} '
                            f'refresh={refresh_count} risky={risky_count} overlap={overlap_count}'
                        )
                        if hasattr(scene.gaussians, '_binding_source_joint_debug_summary'):
                            print(
                                '[Train] post-densify source joints '
                                f'iter={schedule_iteration} '
                                f'{scene.gaussians._binding_source_joint_debug_summary(binding_state)}'
                            )
                    if bool(model.gaussian.get('binding_densify_immediate_refresh_enable', True)):
                        rigid_deformer = getattr(getattr(scene.converter, 'deformer', None), 'rigid', None)
                        if rigid_deformer is not None and hasattr(rigid_deformer, 'refresh_pending_binding'):
                            refresh_binding = rigid_deformer.refresh_pending_binding(
                                scene.gaussians.get_xyz.detach(),
                                schedule_iteration,
                                canonical_owner=scene.gaussians,
                            )
                            if hasattr(rigid_deformer, 'consume_latest_subset_refresh_info'):
                                refresh_info = rigid_deformer.consume_latest_subset_refresh_info()
                                if densify_debug_verbose:
                                    info_refresh_mask = refresh_info.get('refresh_mask', None) if isinstance(refresh_info, dict) else None
                                    info_refresh_count = int(info_refresh_mask.to(dtype=torch.bool).sum().item()) if torch.is_tensor(info_refresh_mask) else 0
                                    print(
                                        '[Train] immediate binding refresh result '
                                        f'iter={schedule_iteration} refreshed={int(refresh_binding is not None)} '
                                        f'consumed={int(bool(refresh_info))} consumed_refresh={info_refresh_count}'
                                    )
                                if refresh_info:
                                    gaussians.apply_post_rebind_child_correction(
                                        refresh_info,
                                        iteration=schedule_iteration,
                                    )
                
                if allow_opacity_reset and (schedule_iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and schedule_iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()

            # Optimizer step
            if diagnostic_skip_optimizer_step:
                if diagnostic_log_interval > 0 and schedule_iteration % diagnostic_log_interval == 0:
                    print(f"[TrainDiagnostic] iter={schedule_iteration} skipped optimizer step.")
                scene.gaussians.optimizer.zero_grad(set_to_none=True)
                scene.converter.optimizer.zero_grad(set_to_none=True)
            elif iteration < opt.iterations and accumulation_should_step:
                scene.optimize(schedule_iteration)

            if bool(model.gaussian.get('binding_stale_refresh_enable', False)):
                stale_refresh_points = gaussians.mark_stale_binding_points_for_refresh(
                    iteration=schedule_iteration,
                )
                if stale_refresh_points > 0:
                    rigid_deformer = getattr(getattr(scene.converter, 'deformer', None), 'rigid', None)
                    if rigid_deformer is not None and hasattr(rigid_deformer, 'refresh_pending_binding'):
                        rigid_deformer.refresh_pending_binding(
                            scene.gaussians.get_xyz.detach(),
                            schedule_iteration,
                            canonical_owner=scene.gaussians,
                        )
                        if hasattr(rigid_deformer, 'consume_latest_subset_refresh_info'):
                            refresh_info = rigid_deformer.consume_latest_subset_refresh_info()
                            if refresh_info:
                                gaussians.apply_post_rebind_child_correction(
                                    refresh_info,
                                    iteration=schedule_iteration,
                                )

            if iteration in checkpoint_iterations:
                scene.save_checkpoint(schedule_iteration)

def validation(iteration, testing_iterations, testing_interval, scene : Scene, evaluator, renderArgs, interval_iteration=None):
    # Report test and samples of training set
    check_iteration = iteration if interval_iteration is None else interval_iteration
    if testing_interval > 0:
        if not check_iteration % testing_interval == 0:
            return None
    else:
        if not check_iteration in testing_iterations:
            return None

    scene.eval()
    torch.cuda.empty_cache()
    image_log_limit = int(scene.cfg.get('validation_image_log_limit', 3))
    train_stride = max(1, len(scene.train_dataset) // 10)
    validation_configs = []
    if getattr(scene, 'best_eval_dataset', None) is not None:
        validation_configs.append({
            'name': 'best_eval',
            'dataset': scene.best_eval_dataset,
            'cameras': list(range(len(scene.best_eval_dataset))),
            'log_images': False,
        })
    validation_configs.extend((
        {
            'name': 'test',
            'dataset': scene.test_dataset,
            'cameras': list(range(len(scene.test_dataset))),
            'log_images': True,
        },
        {
            'name': 'train',
            'dataset': scene.train_dataset,
            'cameras': [idx for idx in range(0, len(scene.train_dataset), train_stride)],
            'log_images': True,
        },
    ))
    metrics_summary = {}

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0
            l1_fg_test = 0.0
            psnr_fg_test = 0.0
            ssim_fg_test = 0.0
            lpips_fg_test = 0.0
            examples = []
            eval_dataset = config['dataset']
            for idx, data_idx in enumerate(config['cameras']):
                with torch.no_grad():
                    data = eval_dataset[data_idx]
                    _set_texture_schedule_context(
                        scene,
                        local_iteration=check_iteration if interval_iteration is not None else None,
                        schedule_iteration=iteration,
                    )
                    render_pkg = render(data, iteration, scene, *renderArgs, compute_loss=False, return_opacity=True)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(data.original_image.to("cuda"), 0.0, 1.0)
                    opacity_image = torch.clamp(render_pkg["opacity_render"], 0.0, 1.0)
                    fg_mask = _foreground_mask_from_data(data)

                    if config.get('log_images', False) and image_log_limit != 0 and idx < image_log_limit:
                        wandb_img = wandb.Image(opacity_image[None],
                                                caption=config['name'] + "_view_{}/render_opacity".format(data.image_name))
                        examples.append(wandb_img)
                        wandb_img = wandb.Image(image[None], caption=config['name'] + "_view_{}/render".format(data.image_name))
                        examples.append(wandb_img)
                        wandb_img = wandb.Image(gt_image[None], caption=config['name'] + "_view_{}/ground_truth".format(
                            data.image_name))
                        examples.append(wandb_img)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    metrics_test = evaluator(image, gt_image)
                    metrics_fg = _foreground_metrics(image, gt_image, fg_mask, evaluator)
                    psnr_test += metrics_test["psnr"]
                    ssim_test += metrics_test["ssim"]
                    lpips_test += metrics_test["lpips"]
                    l1_fg_test += metrics_fg['l1_fg']
                    psnr_fg_test += metrics_fg['psnr_fg']
                    ssim_fg_test += metrics_fg['ssim_fg']
                    lpips_fg_test += metrics_fg['lpips_fg']
                    del render_pkg, image, gt_image, opacity_image, fg_mask, metrics_test, metrics_fg

                if examples:
                    wandb.log({config['name'] + "_images": examples})
                    examples.clear()

            psnr_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])
            l1_fg_test /= len(config['cameras'])
            psnr_fg_test /= len(config['cameras'])
            ssim_fg_test /= len(config['cameras'])
            lpips_fg_test /= len(config['cameras'])
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} FG_PSNR {} FG_LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, psnr_fg_test, lpips_fg_test))
            wandb.log({
                config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                config['name'] + '/loss_viewpoint - psnr': psnr_test,
                config['name'] + '/loss_viewpoint - ssim': ssim_test,
                config['name'] + '/loss_viewpoint - lpips': lpips_test,
                config['name'] + '/loss_viewpoint_fg - l1_loss': l1_fg_test,
                config['name'] + '/loss_viewpoint_fg - psnr': psnr_fg_test,
                config['name'] + '/loss_viewpoint_fg - ssim': ssim_fg_test,
                config['name'] + '/loss_viewpoint_fg - lpips': lpips_fg_test,
            })
            metrics_summary[config['name']] = {
                'l1': float(l1_test),
                'psnr': float(psnr_test),
                'ssim': float(ssim_test),
                'lpips': float(lpips_test),
                'l1_fg': float(l1_fg_test),
                'psnr_fg': float(psnr_fg_test),
                'ssim_fg': float(ssim_fg_test),
                'lpips_fg': float(lpips_fg_test),
            }

    wandb.log({'scene/opacity_histogram': wandb.Histogram(scene.gaussians.get_opacity.detach().cpu())})
    wandb.log({'total_points': scene.gaussians.get_xyz.shape[0]})
    torch.cuda.empty_cache()
    scene.train()
    return metrics_summary

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))
    OmegaConf.set_struct(config, False) # allow adding new values to config
    apply_explicit_binding_render_preset(config, repo_root=Path(__file__).resolve().parent)
    _apply_stageB_semantic_adapter_only_train_policy(config)

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)
    _snapshot_hydra_run(config, config.exp_dir)
    config.checkpoint_iterations.append(config.opt.iterations)

    # set wandb logger
    wandb_name = config.name
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project='gaussian-splatting-avatar',
        entity='fast-avatar',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )

    print("Optimizing " + config.exp_dir)

    # Initialize system state (RNG)
    fix_random(config.seed)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(config.detect_anomaly)
    training(config)

    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
