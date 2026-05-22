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
import torch
from omegaconf import OmegaConf
from models import GaussianConverter
from scene.gaussian_model import GaussianModel
from dataset import load_dataset


def _cfg_to_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
        if isinstance(value, list):
            return value
    return [value]


def _validate_partial_state_keys(keys, patterns, key_type):
    if not keys or not patterns:
        return
    invalid = []
    for key in keys:
        if not any(pattern in key for pattern in patterns):
            invalid.append(key)
    if invalid:
        raise RuntimeError(
            f"Partial converter load saw unexpected {key_type} keys: {invalid}. "
            f"Allowed patterns: {patterns}"
        )


def _state_key_matches_patterns(key, patterns):
    return any(pattern in key for pattern in patterns)


def _frame_source_indices_from_metadata(metadata):
    frame_dict = metadata.get('frame_dict', {}) if isinstance(metadata, dict) else {}
    if not frame_dict:
        return None
    try:
        items = [(int(frame_id), int(local_idx)) for frame_id, local_idx in frame_dict.items()]
    except Exception:
        return None
    if not items:
        return None
    count = len(items)
    ordered = [None] * count
    for frame_id, local_idx in items:
        if local_idx < 0 or local_idx >= count:
            return None
        ordered[local_idx] = frame_id
    if any(frame_id is None for frame_id in ordered):
        return None
    return torch.tensor(ordered, dtype=torch.long)


def _is_frame_embedding_key(key):
    return (
        key.startswith('pose_correction.')
        or key.startswith('texture.latent.')
        or key.startswith('deformer.non_rigid.latent.')
    )


def _adapt_partial_converter_tensor(key, checkpoint_value, model_value, frame_source_indices=None):
    if not (torch.is_tensor(checkpoint_value) and torch.is_tensor(model_value)):
        return None, None
    if checkpoint_value.shape == model_value.shape:
        return checkpoint_value, 'exact'
    if (
        frame_source_indices is not None
        and _is_frame_embedding_key(key)
        and checkpoint_value.ndim >= 1
        and model_value.ndim >= 1
        and checkpoint_value.shape[1:] == model_value.shape[1:]
        and model_value.shape[0] == int(frame_source_indices.numel())
        and checkpoint_value.shape[0] > int(frame_source_indices.max().item())
    ):
        indices = frame_source_indices.to(device=checkpoint_value.device)
        adapted = checkpoint_value.index_select(0, indices)
        return adapted, f'frame_gather[{checkpoint_value.shape[0]}->{model_value.shape[0]}]'
    if checkpoint_value.ndim != 2 or model_value.ndim != 2:
        return None, None
    if not key.startswith('texture.structured_trunk_output_head_'):
        return None, None
    if not key.endswith('.lin0.weight'):
        return None, None

    src_out, src_in = checkpoint_value.shape
    dst_out, dst_in = model_value.shape
    if src_out != dst_out or src_in > dst_in:
        return None, None

    adapted = model_value.detach().clone()
    adapted[:, :src_in].copy_(
        checkpoint_value.to(device=model_value.device, dtype=model_value.dtype)
    )
    return adapted, f'prefix_copy[{src_out}x{src_in}->{dst_out}x{dst_in}]'


def _prepare_partial_converter_state_dict(module, checkpoint_state_dict, allowed_patterns, metadata=None):
    model_state_dict = module.state_dict()
    filtered_state_dict = {}
    mismatched_missing_keys = []
    adapted_mismatch_logs = []
    frame_source_indices = _frame_source_indices_from_metadata(metadata)

    for key, value in checkpoint_state_dict.items():
        model_value = model_state_dict.get(key, None)
        if model_value is None:
            filtered_state_dict[key] = value
            continue
        if not (torch.is_tensor(value) and torch.is_tensor(model_value)):
            filtered_state_dict[key] = value
            continue
        if value.shape == model_value.shape:
            filtered_state_dict[key] = value
            continue
        adapted_value, adapt_reason = _adapt_partial_converter_tensor(
            key,
            value,
            model_value,
            frame_source_indices=frame_source_indices,
        )
        if adapted_value is not None:
            filtered_state_dict[key] = adapted_value
            adapted_mismatch_logs.append(
                f"{key}: {tuple(value.shape)} -> {tuple(model_value.shape)} via {adapt_reason}"
            )
            continue
        if not _state_key_matches_patterns(key, allowed_patterns):
            raise RuntimeError(
                "Partial converter load hit a shape mismatch on a non-allowed key: "
                f"{key} checkpoint_shape={tuple(value.shape)} model_shape={tuple(model_value.shape)}. "
                f"Allowed patterns: {allowed_patterns}"
            )

        mismatched_missing_keys.append(key)

    return filtered_state_dict, mismatched_missing_keys, adapted_mismatch_logs


class Scene:

    gaussians : GaussianModel

    def __init__(self, cfg, gaussians : GaussianModel, save_dir : str):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.cfg = cfg

        self.save_dir = save_dir
        self.gaussians = gaussians

        self.train_dataset = load_dataset(cfg.dataset, split='train')
        self.metadata = self.train_dataset.metadata
        self.best_eval_split = str(cfg.get('best_eval_split', '') or '').lower()
        self.best_eval_dataset = None
        if cfg.mode == 'train':
            self.test_dataset = load_dataset(cfg.dataset, split='val')
            if self.best_eval_split not in ('', 'none', 'val'):
                self.best_eval_dataset = load_dataset(cfg.dataset, split=self.best_eval_split)
            elif self.best_eval_split == 'val':
                self.best_eval_dataset = self.test_dataset
        elif cfg.mode == 'test':
            self.test_dataset = load_dataset(cfg.dataset, split='test')
        elif cfg.mode == 'predict':
            self.test_dataset = load_dataset(cfg.dataset, split='predict')
        else:
            raise ValueError

        self.cameras_extent = self.metadata['cameras_extent']

        self.gaussians.create_from_pcd(self.test_dataset.readPointCloud(), spatial_lr_scale=self.cameras_extent)

        self.converter = GaussianConverter(cfg, self.metadata).cuda()

    def train(self):
        self.converter.train()

    def eval(self):
        self.converter.eval()

    def optimize(self, iteration):
        gaussians_delay = self.cfg.model.gaussian.get('delay', 0)
        if iteration >= gaussians_delay:
            self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        self.converter.optimize(iteration)

    def convert_gaussians(self, viewpoint_camera, iteration, compute_loss=True):
        return self.converter(self.gaussians, viewpoint_camera, iteration, compute_loss)

    def get_skinning_loss(self):
        loss_reg = self.converter.deformer.rigid.regularization()
        loss_skinning = loss_reg.get('loss_skinning', torch.tensor(0.).cuda())
        return loss_skinning

    def save(self, iteration):
        point_cloud_path = os.path.join(self.save_dir, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_checkpoint(self, iteration, filename=None, verbose=True):
        if filename is None:
            filename = "ckpt" + str(iteration) + ".pth"
        save_path = filename if os.path.isabs(filename) else os.path.join(self.save_dir, filename)
        if verbose:
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
        torch.save((self.gaussians.capture(),
                    self.converter.state_dict(),
                    self.converter.optimizer.state_dict(),
                    self.converter.scheduler.state_dict(),
                    iteration), save_path)
        return save_path

    def load_checkpoint(self, path):
        (gaussian_params, converter_sd, converter_opt_sd, converter_scd_sd, first_iter) = torch.load(path)
        resume_cfg = self.cfg.get('resume', None)
        self.gaussians.restore(gaussian_params, self.cfg.opt, resume_cfg=resume_cfg)
        allow_partial_converter_load = bool(resume_cfg.get('allow_partial_converter_load', False)) if resume_cfg else False
        restore_converter_optimizer_state = bool(resume_cfg.get('restore_converter_optimizer_state', False)) if resume_cfg else False
        restore_converter_scheduler_state = bool(resume_cfg.get('restore_converter_scheduler_state', False)) if resume_cfg else False
        preserve_converter_config_lrs = bool(resume_cfg.get('restore_converter_optimizer_preserve_config_lrs', False)) if resume_cfg else False
        if allow_partial_converter_load:
            missing_patterns = _cfg_to_list(
                resume_cfg.get('partial_converter_missing_keys_allow_patterns', [])
            ) if resume_cfg else []
            filtered_converter_sd, mismatched_missing_keys, adapted_mismatch_logs = (
                _prepare_partial_converter_state_dict(
                    self.converter,
                    converter_sd,
                    missing_patterns,
                    metadata=self.metadata,
                )
            )
            incompatible = self.converter.load_state_dict(filtered_converter_sd, strict=False)
            missing_keys = list(incompatible.missing_keys)
            unexpected_keys = list(incompatible.unexpected_keys)
            unexpected_patterns = _cfg_to_list(
                resume_cfg.get('partial_converter_unexpected_keys_allow_patterns', [])
            ) if resume_cfg else []
            if mismatched_missing_keys:
                missing_keys.extend(mismatched_missing_keys)
            missing_keys = sorted(set(missing_keys))
            unexpected_keys = sorted(set(unexpected_keys))
            _validate_partial_state_keys(missing_keys, missing_patterns, 'missing')
            _validate_partial_state_keys(unexpected_keys, unexpected_patterns, 'unexpected')
            for log_line in adapted_mismatch_logs:
                print(f'Resume safety: adapted mismatched converter key: {log_line}')
            if mismatched_missing_keys:
                print(
                    'Resume safety: skipped mismatched converter keys: '
                    f'{mismatched_missing_keys}'
                )
            if missing_keys:
                print(f'Resume safety: partial converter load missing keys: {missing_keys}')
            if unexpected_keys:
                print(f'Resume safety: partial converter load unexpected keys: {unexpected_keys}')
            partial_load_hook = getattr(self.converter, 'on_partial_load', None)
            if callable(partial_load_hook):
                partial_load_hook(missing_keys=missing_keys, unexpected_keys=unexpected_keys)
        else:
            self.converter.load_state_dict(converter_sd)
        if restore_converter_optimizer_state:
            try:
                current_lrs = [group.get('lr', None) for group in self.converter.optimizer.param_groups]
                current_wds = [group.get('weight_decay', None) for group in self.converter.optimizer.param_groups]
                current_names = [group.get('name', None) for group in self.converter.optimizer.param_groups]
                if len(converter_opt_sd.get('param_groups', [])) == len(self.converter.optimizer.param_groups):
                    self.converter.optimizer.load_state_dict(converter_opt_sd)
                    if preserve_converter_config_lrs:
                        for group, lr, weight_decay, name in zip(
                            self.converter.optimizer.param_groups,
                            current_lrs,
                            current_wds,
                            current_names,
                        ):
                            if lr is not None:
                                group['lr'] = lr
                            if weight_decay is not None:
                                group['weight_decay'] = weight_decay
                            if name is not None:
                                group['name'] = name
                    print('Resume safety: restored converter optimizer state.')
                else:
                    print('[Scene] converter optimizer param group count changed; skipping optimizer state restore.')
            except Exception as exc:
                print(f'[Scene] failed to restore converter optimizer state ({exc}); continuing with fresh optimizer state.')
        if restore_converter_scheduler_state:
            try:
                self.converter.scheduler.load_state_dict(converter_scd_sd)
                print('Resume safety: restored converter scheduler state.')
            except Exception as exc:
                print(f'[Scene] failed to restore converter scheduler state ({exc}); continuing with fresh scheduler state.')
        return int(first_iter)
