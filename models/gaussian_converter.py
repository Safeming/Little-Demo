import torch
import torch.nn as nn
import numpy as np
import copy
from fnmatch import fnmatch
from omegaconf import OmegaConf
from .deformer import get_deformer
from .pose_correction import get_pose_correction
from .texture import get_texture


def _resolve_scheduled_scalar(iteration, value, default=0.0):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)

    value_list = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value_list, list):
        return float(value)
    if len(value_list) == 0:
        return default
    if len(value_list) % 2 == 0:
        raise ValueError(f"Scheduled scalar expects [v0, step1, v1, ...], got {value_list}")

    schedule = [0] + value_list
    index = 0
    while index < len(schedule):
        if iteration >= schedule[index]:
            index += 2
        else:
            break
    return float(schedule[index - 1])


def _cfg_list(value):
    if value is None:
        return []
    value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _name_matches(name, patterns):
    return any(fnmatch(name, pattern) for pattern in patterns)


def _filter_named_parameters(named_parameters, include_patterns=None, exclude_patterns=None):
    include_patterns = _cfg_list(include_patterns)
    exclude_patterns = _cfg_list(exclude_patterns)
    selected = []
    selected_names = []
    skipped_names = []
    for name, param in named_parameters:
        include = True
        if include_patterns:
            include = _name_matches(name, include_patterns)
        if include and exclude_patterns and _name_matches(name, exclude_patterns):
            include = False
        if include:
            selected.append(param)
            selected_names.append(name)
        else:
            skipped_names.append(name)
    return selected, selected_names, skipped_names


def _append_param_group(groups, params, lr, name=None, weight_decay=None):
    group = {'params': list(params), 'lr': lr}
    if name is not None:
        group['name'] = name
    if weight_decay is not None:
        group['weight_decay'] = weight_decay
    groups.append(group)


class CameraAffineCorrection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.enabled = bool(cfg.opt.get('camera_affine_enable', False))
        self.max_camera_id = max(0, int(cfg.opt.get('camera_affine_max_camera_id', 32)))
        self.scale_max_delta = float(cfg.opt.get('camera_affine_scale_max_delta', 0.08))
        self.shift_max_abs = float(cfg.opt.get('camera_affine_shift_max_abs', 0.04))
        self.clamp_colors = bool(cfg.opt.get('camera_affine_clamp_colors', True))
        self.apply_unknown = bool(cfg.opt.get('camera_affine_apply_unknown', False))
        self.train_camera_ids = self._normalize_camera_ids(cfg.opt.get('camera_affine_train_camera_ids', None))
        self.train_camera_id_set = set(self.train_camera_ids)
        self.last_debug = {}

        if self.enabled:
            # Index 0 is reserved for unknown/invalid cameras. Zeros mean identity:
            # scale = 1, shift = 0.
            self.scale_raw = nn.Parameter(torch.zeros(self.max_camera_id + 1, 3))
            self.shift_raw = nn.Parameter(torch.zeros(self.max_camera_id + 1, 3))
        else:
            self.register_parameter('scale_raw', None)
            self.register_parameter('shift_raw', None)

    @staticmethod
    def _normalize_camera_ids(value):
        if value is None:
            return []
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if isinstance(value, (int, float, str)):
            value = [value]
        ids = []
        for item in value:
            try:
                ids.append(int(item))
            except (TypeError, ValueError):
                continue
        return sorted(set(ids))

    def _camera_id(self, camera):
        cam_id = getattr(camera, 'cam_id', None)
        if cam_id is None:
            return None
        try:
            return int(cam_id)
        except (TypeError, ValueError):
            return None

    def _camera_index(self, camera):
        cam_id = self._camera_id(camera)
        if cam_id is None:
            return None, 'unknown'
        if self.train_camera_id_set and cam_id not in self.train_camera_id_set:
            return None, str(cam_id)
        if cam_id < 0 or cam_id > self.max_camera_id:
            return (0 if self.apply_unknown else None), str(cam_id)
        return cam_id, str(cam_id)

    def _effective_params(self, index):
        scale_delta = self.scale_max_delta * torch.tanh(self.scale_raw[index])
        shift = self.shift_max_abs * torch.tanh(self.shift_raw[index])
        scale = 1.0 + scale_delta
        return scale, shift, scale_delta

    def forward(self, colors, camera, iteration=0):
        if (
            not self.enabled
            or self.scale_raw is None
            or self.shift_raw is None
            or colors is None
        ):
            self.last_debug = {}
            return colors

        index, cam_label = self._camera_index(camera)
        if index is None:
            self.last_debug = {
                'camera': cam_label,
                'active': 0.0,
                'strength': 0.0,
                'scale_delta_abs_mean': 0.0,
                'shift_abs_mean': 0.0,
            }
            return colors

        strength = _resolve_scheduled_scalar(
            iteration,
            self.cfg.opt.get('camera_affine_strength', 1.0),
            default=1.0,
        )
        strength = max(0.0, min(1.0, strength))
        if strength <= 0.0:
            self.last_debug = {
                'camera': cam_label,
                'active': 1.0,
                'strength': 0.0,
                'scale_delta_abs_mean': 0.0,
                'shift_abs_mean': 0.0,
            }
            return colors

        scale, shift, scale_delta = self._effective_params(index)
        corrected = colors * scale.view(1, 3) + shift.view(1, 3)
        if strength < 1.0:
            corrected = colors * (1.0 - strength) + corrected * strength
        if self.clamp_colors:
            corrected = corrected.clamp(0.0, 1.0)

        self.last_debug = {
            'camera': cam_label,
            'active': 1.0,
            'strength': strength,
            'scale_delta_abs_mean': float(scale_delta.detach().abs().mean().item()),
            'shift_abs_mean': float(shift.detach().abs().mean().item()),
        }
        return corrected

    def regularization(self, camera):
        if not self.enabled or self.scale_raw is None or self.shift_raw is None:
            ref = next(self.parameters(), None)
            return torch.tensor(0.0, device=ref.device if ref is not None else 'cuda')

        index, _ = self._camera_index(camera)
        if index is None:
            return self.scale_raw.sum() * 0.0

        _, shift, scale_delta = self._effective_params(index)
        scale_norm = scale_delta / max(self.scale_max_delta, 1.0e-6)
        shift_norm = shift / max(self.shift_max_abs, 1.0e-6)
        scale_weight = float(self.cfg.opt.get('camera_affine_scale_reg_weight', 1.0))
        shift_weight = float(self.cfg.opt.get('camera_affine_shift_reg_weight', 1.0))
        return scale_weight * (scale_norm ** 2).mean() + shift_weight * (shift_norm ** 2).mean()


class CameraGeometryCorrection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.enabled = bool(cfg.opt.get('camera_geometry_enable', False))
        self.max_camera_id = max(0, int(cfg.opt.get('camera_geometry_max_camera_id', 32)))
        self.rot_max_deg = float(cfg.opt.get('camera_geometry_rot_max_deg', 0.12))
        self.trans_max = float(cfg.opt.get('camera_geometry_trans_max', 0.003))
        self.apply_unknown = bool(cfg.opt.get('camera_geometry_apply_unknown', False))
        self.train_camera_ids = self._normalize_camera_ids(cfg.opt.get('camera_geometry_train_camera_ids', None))
        self.train_camera_id_set = set(self.train_camera_ids)
        self.last_debug = {}

        if self.enabled:
            self.rot_raw = nn.Parameter(torch.zeros(self.max_camera_id + 1, 3))
            self.trans_raw = nn.Parameter(torch.zeros(self.max_camera_id + 1, 3))
        else:
            self.register_parameter('rot_raw', None)
            self.register_parameter('trans_raw', None)

    @staticmethod
    def _normalize_camera_ids(value):
        if value is None:
            return []
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if isinstance(value, (int, float, str)):
            value = [value]
        ids = []
        for item in value:
            try:
                ids.append(int(item))
            except (TypeError, ValueError):
                continue
        return sorted(set(ids))

    def _camera_id(self, camera):
        cam_id = getattr(camera, 'cam_id', None)
        if cam_id is None:
            return None
        try:
            return int(cam_id)
        except (TypeError, ValueError):
            return None

    def _camera_index(self, camera):
        cam_id = self._camera_id(camera)
        if cam_id is None:
            return None, 'unknown'
        if self.train_camera_id_set and cam_id not in self.train_camera_id_set:
            return None, str(cam_id)
        if cam_id < 0 or cam_id > self.max_camera_id:
            return (0 if self.apply_unknown else None), str(cam_id)
        return cam_id, str(cam_id)

    def _effective_params(self, index):
        rot_max_rad = self.rot_max_deg * np.pi / 180.0
        rot_vec = rot_max_rad * torch.tanh(self.rot_raw[index])
        trans = self.trans_max * torch.tanh(self.trans_raw[index])
        return rot_vec, trans

    @staticmethod
    def _rotation_from_rotvec(rot_vec):
        theta = torch.linalg.norm(rot_vec).clamp_min(1.0e-12)
        axis = rot_vec / theta
        zero = torch.zeros((), dtype=rot_vec.dtype, device=rot_vec.device)
        k = torch.stack((
            torch.stack((zero, -axis[2], axis[1])),
            torch.stack((axis[2], zero, -axis[0])),
            torch.stack((-axis[1], axis[0], zero)),
        ))
        eye = torch.eye(3, dtype=rot_vec.dtype, device=rot_vec.device)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        return eye + sin_theta * k + (1.0 - cos_theta) * (k @ k)

    def forward(self, gaussians, camera, iteration=0):
        if (
            not self.enabled
            or self.rot_raw is None
            or self.trans_raw is None
            or gaussians is None
        ):
            self.last_debug = {}
            return gaussians

        index, cam_label = self._camera_index(camera)
        if index is None:
            self.last_debug = {
                'camera': cam_label,
                'active': 0.0,
                'strength': 0.0,
                'rot_deg_abs_mean': 0.0,
                'trans_abs_mean': 0.0,
            }
            return gaussians

        strength = _resolve_scheduled_scalar(
            iteration,
            self.cfg.opt.get('camera_geometry_strength', 1.0),
            default=1.0,
        )
        strength = max(0.0, min(1.0, strength))
        if strength <= 0.0:
            self.last_debug = {
                'camera': cam_label,
                'active': 1.0,
                'strength': 0.0,
                'rot_deg_abs_mean': 0.0,
                'trans_abs_mean': 0.0,
            }
            return gaussians

        rot_vec, trans = self._effective_params(index)
        rot_vec = rot_vec * strength
        trans = trans * strength
        rotation = self._rotation_from_rotvec(rot_vec)

        # Keep dynamic binding/semantic attributes that texture heads depend on;
        # GaussianModel.clone() only copies a fixed subset.
        corrected = copy.copy(gaussians)
        center = getattr(camera, 'camera_center', None)
        xyz = gaussians.get_xyz
        if torch.is_tensor(center):
            center = center.to(device=xyz.device, dtype=xyz.dtype).view(1, 3)
            corrected._xyz = (xyz - center) @ rotation.transpose(0, 1) + center + trans.view(1, 3)
        else:
            corrected._xyz = xyz @ rotation.transpose(0, 1) + trans.view(1, 3)
        if hasattr(gaussians, 'rotation_precomp'):
            corrected.rotation_precomp = torch.matmul(rotation.view(1, 3, 3), gaussians.rotation_precomp)

        self.last_debug = {
            'camera': cam_label,
            'active': 1.0,
            'strength': strength,
            'rot_deg_abs_mean': float((rot_vec.detach().abs().mean() * 180.0 / np.pi).item()),
            'trans_abs_mean': float(trans.detach().abs().mean().item()),
        }
        return corrected

    def regularization(self, camera):
        if not self.enabled or self.rot_raw is None or self.trans_raw is None:
            ref = next(self.parameters(), None)
            return torch.tensor(0.0, device=ref.device if ref is not None else 'cuda')

        index, _ = self._camera_index(camera)
        if index is None:
            return self.rot_raw.sum() * 0.0

        rot_vec, trans = self._effective_params(index)
        rot_max_rad = max(self.rot_max_deg * np.pi / 180.0, 1.0e-8)
        trans_max = max(self.trans_max, 1.0e-8)
        rot_weight = float(self.cfg.opt.get('camera_geometry_rot_reg_weight', 1.0))
        trans_weight = float(self.cfg.opt.get('camera_geometry_trans_reg_weight', 1.0))
        return rot_weight * ((rot_vec / rot_max_rad) ** 2).mean() + trans_weight * ((trans / trans_max) ** 2).mean()


class GaussianConverter(nn.Module):
    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg
        self.metadata = metadata

        self.pose_correction = get_pose_correction(cfg.model.pose_correction, metadata)
        self.deformer = get_deformer(cfg.model.deformer, metadata)
        self.texture = get_texture(cfg.model.texture, metadata)
        self.camera_affine = CameraAffineCorrection(cfg)
        self.camera_geometry = CameraGeometryCorrection(cfg)

        self.optimizer, self.scheduler = None, None
        self._latent_weight_decay_cfg = self.cfg.opt.get('latent_weight_decay', 0.05)
        self._latent_weight_decay_group_indices = []
        self.set_optimizer()

    def set_optimizer(self):
        latent_weight_decay = _resolve_scheduled_scalar(0, self._latent_weight_decay_cfg, default=0.05)
        opt_params = []
        _append_param_group(opt_params, self.deformer.rigid.parameters(), self.cfg.opt.get('rigid_lr', 0.), name='rigid')
        # {'params': self.deformer.non_rigid.parameters(), 'lr': self.cfg.opt.get('non_rigid_lr', 0.)},
        _append_param_group(
            opt_params,
            [p for n, p in self.deformer.non_rigid.named_parameters() if 'latent' not in n],
            self.cfg.opt.get('non_rigid_lr', 0.),
            name='non_rigid',
        )
        _append_param_group(
            opt_params,
            [p for n, p in self.deformer.non_rigid.named_parameters() if 'latent' in n],
            self.cfg.opt.get('nr_latent_lr', 0.),
            name='non_rigid_latent',
            weight_decay=latent_weight_decay,
        )
        pose_params_getter = getattr(self.pose_correction, 'trainable_parameters', None)
        pose_params = pose_params_getter() if callable(pose_params_getter) else self.pose_correction.parameters()
        _append_param_group(opt_params, pose_params, self.cfg.opt.get('pose_correction_lr', 0.), name='pose_correction')

        texture_named = list(self.texture.named_parameters())
        texture_trainable_patterns = self.cfg.opt.get('texture_trainable_name_patterns', None)
        texture_frozen_patterns = self.cfg.opt.get('texture_frozen_name_patterns', None)
        texture_nonlatent, texture_nonlatent_names, texture_nonlatent_skipped = _filter_named_parameters(
            [(n, p) for n, p in texture_named if 'latent' not in n],
            include_patterns=texture_trainable_patterns,
            exclude_patterns=texture_frozen_patterns,
        )
        texture_latent, texture_latent_names, texture_latent_skipped = _filter_named_parameters(
            [(n, p) for n, p in texture_named if 'latent' in n],
            include_patterns=texture_trainable_patterns,
            exclude_patterns=texture_frozen_patterns,
        )
        _append_param_group(opt_params, texture_nonlatent, self.cfg.opt.get('texture_lr', 0.), name='texture')
        _append_param_group(
            opt_params,
            texture_latent,
            self.cfg.opt.get('tex_latent_lr', 0.),
            name='texture_latent',
            weight_decay=latent_weight_decay,
        )
        camera_affine_params = list(self.camera_affine.parameters())
        if camera_affine_params:
            _append_param_group(
                opt_params,
                camera_affine_params,
                self.cfg.opt.get('camera_affine_lr', 0.0),
                name='camera_affine',
            )
        camera_geometry_params = list(self.camera_geometry.parameters())
        if camera_geometry_params:
            _append_param_group(
                opt_params,
                camera_geometry_params,
                self.cfg.opt.get('camera_geometry_lr', 0.0),
                name='camera_geometry',
            )
        if texture_trainable_patterns or texture_frozen_patterns:
            print(
                '[GaussianConverter] texture optimizer filter: '
                f'trainable_nonlatent={len(texture_nonlatent_names)} '
                f'trainable_latent={len(texture_latent_names)} '
                f'skipped={len(texture_nonlatent_skipped) + len(texture_latent_skipped)} '
                f'patterns={_cfg_list(texture_trainable_patterns)} '
                f'exclude={_cfg_list(texture_frozen_patterns)}'
            )
            if texture_nonlatent_names:
                preview = ','.join(texture_nonlatent_names[:8])
                suffix = '...' if len(texture_nonlatent_names) > 8 else ''
                print(f'[GaussianConverter] texture trainable preview: {preview}{suffix}')
        self._latent_weight_decay_group_indices = [idx for idx, group in enumerate(opt_params) if 'weight_decay' in group]
        self.optimizer = torch.optim.Adam(params=opt_params, lr=0.001, eps=1e-15)

        gamma = self.cfg.opt.lr_ratio ** (1. / self.cfg.opt.iterations)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

    def forward(self, gaussians, camera, iteration, compute_loss=True):
        loss_reg = {}
        # loss_reg.update(gaussians.get_opacity_loss())
        camera, loss_reg_pose = self.pose_correction(camera, iteration)

        # pose augmentation
        pose_noise = self.cfg.pipeline.get('pose_noise', 0.)
        if self.training and pose_noise > 0 and np.random.uniform() <= 0.5:
            camera = camera.copy()
            camera.rots = camera.rots + torch.randn(camera.rots.shape, device=camera.rots.device) * pose_noise

        deformed_gaussians, loss_reg_deformer = self.deformer(gaussians, camera, iteration, compute_loss)
        deformed_gaussians = self.camera_geometry(deformed_gaussians, camera, iteration=iteration)

        loss_reg.update(loss_reg_pose)
        loss_reg.update(loss_reg_deformer)
        loss_reg['camera_geometry_reg'] = self.camera_geometry.regularization(camera)

        color_precompute = self.texture(deformed_gaussians, camera, iteration=iteration)
        color_precompute = self.camera_affine(color_precompute, camera, iteration=iteration)
        loss_reg['camera_affine_reg'] = self.camera_affine.regularization(camera)

        return deformed_gaussians, loss_reg, color_precompute

    def optimize(self, iteration=0):
        grad_clip = self.cfg.opt.get('grad_clip', 0.)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        latent_weight_decay = _resolve_scheduled_scalar(iteration, self._latent_weight_decay_cfg, default=0.05)
        for group_idx in self._latent_weight_decay_group_indices:
            self.optimizer.param_groups[group_idx]['weight_decay'] = latent_weight_decay
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

    def on_partial_load(self, missing_keys=None, unexpected_keys=None):
        texture_hook = getattr(self.texture, 'on_partial_load', None)
        if callable(texture_hook):
            texture_hook(missing_keys=missing_keys, unexpected_keys=unexpected_keys)
        if missing_keys and any('camera_affine.' in key for key in missing_keys):
            print('[GaussianConverter] camera affine initialized as identity for partial checkpoint load.')
