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

import torch
import numpy as np
import json
import csv
from collections.abc import Sequence
from utils.general_utils import inverse_sigmoid
from torch import nn
import torch.nn.functional as F
import os
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, geom_transform_points
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.pytorch3d_compat import ops


def _constant_lr_func(lr):
    def helper(_iteration):
        return float(lr)
    return helper


def resolve_schedule_value(iteration, value, default=None):
    if value is None:
        return default
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        seq = list(value)
    else:
        return value
    if len(seq) == 0:
        return default
    if len(seq) == 1:
        return seq[0]
    current = seq[0]
    pairs = seq[1:]
    for idx in range(0, len(pairs) - 1, 2):
        start_iter = int(pairs[idx])
        scheduled_value = pairs[idx + 1]
        if iteration >= start_iter:
            current = scheduled_value
    return current


class GaussianModel:
    _covariance_signed_point_cache = {}
    _covariance_signed_component_cache = {}

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, cfg):
        self.cfg = cfg

        # two modes: SH coefficient or feature
        self.use_sh = cfg.use_sh
        self.active_sh_degree = 0
        if self.use_sh:
            self.max_sh_degree = cfg.sh_degree
            self.feature_dim = (self.max_sh_degree + 1) ** 2
        else:
            self.feature_dim = cfg.feature_dim

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._boundary_tag = torch.empty(0)
        self._boundary_opacity_residual = torch.empty(0)
        self._boundary_scaling_residual = torch.empty(0)
        self._boundary_cov_residual = torch.empty(0)
        self._binding_layer_logits_residual = torch.empty(0)
        self._semantic_region_logits_residual = torch.empty(0)
        self._semantic_compact_logits_residual = torch.empty(0)
        self._semantic_asset_region_logits_residual = torch.empty(0)
        self._semantic_asset_compact_logits_residual = torch.empty(0)
        self._next_lineage_uid = 0
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.binding_state = {}
        self.setup_functions()

    def _boundary_cov_residual_channels(self):
        try:
            return max(int(self.cfg.get('boundary_cov_residual_channels', 1)), 1)
        except Exception:
            return 1

    def clone(self):
        cloned = GaussianModel(self.cfg)
        cloned._next_lineage_uid = self._next_lineage_uid

        properties = ["active_sh_degree",
                      "non_rigid_feature",
                      "canonical_xyz",
                      "binding_weights",
                      "binding_distance",
                      "binding_surface_distance",
                      "binding_boundary_score",
                      "binding_boundary_live_score",
                      "binding_boundary_mixed_score",
                      "binding_anchor_ids",
                      "binding_anchor_face_ids",
                      "binding_barycentric",
                      "binding_layer_ids",
                      "binding_region_probs",
                      "binding_region_probs_raw",
                      "binding_region_probs_asset",
                      "binding_region_probs_asset_raw",
                      "binding_region_ids",
                      "binding_region_ids_asset",
                      "binding_compact_semantic_probs",
                      "binding_compact_semantic_probs_raw",
                      "binding_compact_semantic_probs_asset",
                      "binding_compact_semantic_probs_asset_raw",
                      "binding_compact_semantic_ids",
                      "binding_compact_semantic_ids_asset",
                      "binding_compact_semantic_names",
                      "binding_semantic_score",
                      "binding_semantic_distance",
                      "binding_thin_score",
                      "binding_part_rigid_prior",
                      "binding_part_free_prior",
                      "binding_temporal_slip",
                      ]
        for property in properties:
            if hasattr(self, property):
                setattr(cloned, property, getattr(self, property))

        parameters = ["_xyz",
                      "_features_dc",
                      "_features_rest",
                      "_scaling",
                      "_rotation",
                      "_opacity",
                      "_boundary_tag",
                      "_boundary_opacity_residual",
                      "_boundary_scaling_residual",
                      "_boundary_cov_residual",
                      "_binding_layer_logits_residual",
                      "_semantic_region_logits_residual",
                      "_semantic_compact_logits_residual",
                      "_semantic_asset_region_logits_residual",
                      "_semantic_asset_compact_logits_residual"]
        for parameter in parameters:
            setattr(cloned, parameter, getattr(self, parameter) + 0.)

        if self.has_binding_state():
            cloned.set_binding_state(self.get_binding_state())

        return cloned

    def has_binding_state(self):
        return len(getattr(self, 'binding_state', {})) > 0

    def get_binding_state(self):
        return getattr(self, 'binding_state', {})

    def _capture_binding_state(self):
        binding_state = self.get_binding_state()
        if not isinstance(binding_state, dict) or len(binding_state) == 0:
            return {}
        return {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in binding_state.items()
        }

    def clear_binding_state(self):
        self.binding_state = {}

    def set_binding_state(self, binding_state):
        if binding_state is None or len(binding_state) == 0:
            self.clear_binding_state()
            return
        self.binding_state = {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in binding_state.items()
        }
        point_count = None
        device = None
        for value in self.binding_state.values():
            if torch.is_tensor(value):
                point_count = value.shape[0]
                device = value.device
                break
        if point_count is not None and 'anchor_refresh_mask' not in self.binding_state:
            self.binding_state['anchor_refresh_mask'] = torch.zeros((point_count,), dtype=torch.bool, device=device)

    def has_boundary_tag_state(self):
        return (
            torch.is_tensor(self._boundary_tag)
            and self._boundary_tag.numel() == self.get_xyz.shape[0]
            and self._boundary_tag.numel() > 0
        )

    def has_boundary_tags(self):
        return (
            self.has_boundary_tag_state()
            and bool((self._boundary_tag > 0).any().item())
        )

    def get_live_boundary_score_state(self):
        boundary_score = getattr(self, 'binding_boundary_live_score', None)
        if not torch.is_tensor(boundary_score):
            boundary_score = getattr(self, 'binding_boundary_score', None)
        if (
            not torch.is_tensor(boundary_score)
            or boundary_score.shape[0] != self.get_xyz.shape[0]
            or boundary_score.numel() <= 0
        ):
            return None
        device = self._xyz.device if torch.is_tensor(self._xyz) and self._xyz.numel() > 0 else boundary_score.device
        return boundary_score.detach().to(device=device, dtype=torch.float32).reshape(-1).clamp(0.0, 1.0)

    def set_live_boundary_score_state(self, boundary_score):
        if boundary_score is None:
            if hasattr(self, 'binding_boundary_live_score'):
                delattr(self, 'binding_boundary_live_score')
            return
        if not torch.is_tensor(boundary_score):
            boundary_score = torch.tensor(boundary_score, dtype=torch.float32)
        boundary_score = boundary_score.detach().reshape(-1).float().clamp(0.0, 1.0)
        if self.get_xyz.numel() > 0 and boundary_score.shape[0] != self.get_xyz.shape[0]:
            raise ValueError(
                f'live boundary score shape mismatch: got {boundary_score.shape[0]}, '
                f'expected {self.get_xyz.shape[0]}'
            )
        device = self._xyz.device if torch.is_tensor(self._xyz) and self._xyz.numel() > 0 else boundary_score.device
        setattr(self, 'binding_boundary_live_score', boundary_score.to(device=device))

    def _slice_live_boundary_score_state(self, mask, repeats=1):
        boundary_score = self.get_live_boundary_score_state()
        if boundary_score is None:
            return None
        selected = boundary_score[mask]
        if repeats != 1:
            selected = selected.repeat(repeats)
        return selected.clone()

    def _append_live_boundary_score_state(self, extension_score):
        if extension_score is None:
            return
        if not torch.is_tensor(extension_score):
            extension_score = torch.tensor(extension_score, dtype=torch.float32)
        extension_score = extension_score.detach().reshape(-1).float().clamp(0.0, 1.0)

        point_count = int(self.get_xyz.shape[0]) if torch.is_tensor(self._xyz) and self._xyz.ndim >= 2 else 0
        device = self._xyz.device if torch.is_tensor(self._xyz) and self._xyz.numel() > 0 else extension_score.device
        extension_count = int(extension_score.shape[0])
        old_count = max(point_count - extension_count, 0)

        base_score = self.get_live_boundary_score_state()
        if torch.is_tensor(base_score) and base_score.shape[0] == old_count:
            base_score = base_score.to(device=device, dtype=torch.float32)
        else:
            base_score = torch.zeros((old_count,), dtype=torch.float32, device=device)

        if old_count == 0:
            self.set_live_boundary_score_state(extension_score.to(device=device))
            return
        self.set_live_boundary_score_state(
            torch.cat((base_score, extension_score.to(device=device)), dim=0)
        )

    def get_boundary_tag_state(self):
        if not self.has_boundary_tag_state():
            return None
        return self._boundary_tag

    def get_boundary_tags(self):
        if not self.has_boundary_tags():
            return None
        return self._boundary_tag

    def get_boundary_residual_support_state(self):
        point_count = int(self.get_xyz.shape[0]) if torch.is_tensor(self._xyz) and self._xyz.ndim >= 2 else 0
        if point_count <= 0:
            return None

        device = self._xyz.device if torch.is_tensor(self._xyz) and self._xyz.numel() > 0 else None
        support = None

        boundary_tag = self.get_boundary_tag_state()
        if torch.is_tensor(boundary_tag) and boundary_tag.shape[0] == point_count:
            support = boundary_tag.detach().to(device=device, dtype=torch.float32).reshape(-1).clamp(0.0, 1.0)

        opacity_eps = float(self.cfg.get('boundary_residual_support_opacity_epsilon', 1.0e-8))
        if (
            torch.is_tensor(self._boundary_opacity_residual)
            and self._boundary_opacity_residual.shape[0] == point_count
            and self._boundary_opacity_residual.numel() > 0
        ):
            opacity_support = (
                self._boundary_opacity_residual.detach().abs().amax(dim=-1) > opacity_eps
            ).to(device=device, dtype=torch.float32)
            support = opacity_support if support is None else torch.maximum(support, opacity_support)

        scaling_eps = float(self.cfg.get('boundary_residual_support_scaling_epsilon', 1.0e-8))
        if (
            torch.is_tensor(self._boundary_scaling_residual)
            and self._boundary_scaling_residual.shape[0] == point_count
            and self._boundary_scaling_residual.numel() > 0
        ):
            scaling_support = (
                torch.norm(self._boundary_scaling_residual.detach(), dim=-1) > scaling_eps
            ).to(device=device, dtype=torch.float32)
            support = scaling_support if support is None else torch.maximum(support, scaling_support)

        cov_eps = float(self.cfg.get('boundary_residual_support_cov_epsilon', 1.0e-8))
        if (
            torch.is_tensor(self._boundary_cov_residual)
            and self._boundary_cov_residual.shape[0] == point_count
            and self._boundary_cov_residual.numel() > 0
        ):
            cov_support = (
                self._boundary_cov_residual.detach().abs().amax(dim=-1) > cov_eps
            ).to(device=device, dtype=torch.float32)
            support = cov_support if support is None else torch.maximum(support, cov_support)

        if support is None or not bool((support > 0).any().item()):
            return None
        return support.clamp(0.0, 1.0)

    def get_boundary_residual_support(self):
        support = self.get_boundary_residual_support_state()
        if support is None:
            return None
        if not bool((support > 0).any().item()):
            return None
        return support

    def get_boundary_support_role_mask(self):
        point_count = int(self.get_xyz.shape[0]) if torch.is_tensor(self._xyz) and self._xyz.ndim >= 2 else 0
        if point_count <= 0 or not self.has_binding_state():
            return None
        role = self.binding_state.get('boundary_support_role', None)
        if not torch.is_tensor(role) or role.shape[0] != point_count:
            return None
        return role.to(device=self._xyz.device, dtype=torch.float32).reshape(-1) > 0.5

    def clear_boundary_tags(self):
        device = self._xyz.device if torch.is_tensor(self._xyz) and self._xyz.numel() > 0 else None
        self._boundary_tag = torch.empty(0, device=device)

    def _resize_pointwise_state(self, value, point_count, tail_shape=(), dtype=torch.float32, device=None, fill_value=0.0):
        target_shape = (int(point_count),) + tuple(tail_shape)
        resized = torch.full(target_shape, fill_value, dtype=dtype, device=device)
        if not torch.is_tensor(value) or value.numel() <= 0:
            return resized

        source = value.detach().to(device=device, dtype=dtype)
        if tuple(source.shape[1:]) != tuple(tail_shape):
            return resized

        copy_count = min(int(source.shape[0]), int(point_count))
        if copy_count > 0:
            resized[:copy_count] = source[:copy_count]
        return resized

    def ensure_boundary_state_matches_points(self, verbose=False):
        point_count = int(self.get_xyz.shape[0]) if torch.is_tensor(self._xyz) and self._xyz.ndim >= 2 else 0
        device = self._xyz.device if torch.is_tensor(self._xyz) and self._xyz.numel() > 0 else None
        changed = []

        if point_count <= 0:
            if torch.is_tensor(self._boundary_tag) and self._boundary_tag.numel() > 0:
                self.clear_boundary_tags()
                changed.append("boundary_tag")
            if not torch.is_tensor(self._boundary_opacity_residual) or self._boundary_opacity_residual.numel() > 0:
                self._boundary_opacity_residual = torch.empty(0, 1, device=device)
                changed.append("boundary_opacity_residual")
            if not torch.is_tensor(self._boundary_scaling_residual) or self._boundary_scaling_residual.numel() > 0:
                self._boundary_scaling_residual = torch.empty(0, 3, device=device)
                changed.append("boundary_scaling_residual")
            if not torch.is_tensor(self._boundary_cov_residual) or self._boundary_cov_residual.numel() > 0:
                self._boundary_cov_residual = torch.empty(0, self._boundary_cov_residual_channels(), device=device)
                changed.append("boundary_cov_residual")
            return len(changed) > 0

        if not torch.is_tensor(self._boundary_tag) or self._boundary_tag.shape[0] != point_count:
            self._boundary_tag = self._resize_pointwise_state(
                self._boundary_tag,
                point_count,
                dtype=torch.float32,
                device=device,
            )
            changed.append("boundary_tag")
        else:
            self._boundary_tag = self._boundary_tag.to(device=device, dtype=torch.float)

        expected_opacity_shape = (point_count, self._opacity.shape[1] if self._opacity.ndim > 1 else 1)
        if (
            not torch.is_tensor(self._boundary_opacity_residual)
            or tuple(self._boundary_opacity_residual.shape) != expected_opacity_shape
        ):
            boundary_opacity_residual = self._resize_pointwise_state(
                self._boundary_opacity_residual,
                point_count,
                tail_shape=expected_opacity_shape[1:],
                dtype=self._opacity.dtype,
                device=device,
            )
            self._boundary_opacity_residual = nn.Parameter(boundary_opacity_residual.requires_grad_(True))
            changed.append("boundary_opacity_residual")

        expected_scaling_shape = tuple(self._scaling.shape)
        if (
            not torch.is_tensor(self._boundary_scaling_residual)
            or tuple(self._boundary_scaling_residual.shape) != expected_scaling_shape
        ):
            boundary_scaling_residual = self._resize_pointwise_state(
                self._boundary_scaling_residual,
                point_count,
                tail_shape=expected_scaling_shape[1:],
                dtype=self._scaling.dtype,
                device=device,
            )
            self._boundary_scaling_residual = nn.Parameter(boundary_scaling_residual.requires_grad_(True))
            changed.append("boundary_scaling_residual")

        expected_cov_shape = (point_count, self._boundary_cov_residual_channels())
        if (
            not torch.is_tensor(self._boundary_cov_residual)
            or tuple(self._boundary_cov_residual.shape) != expected_cov_shape
        ):
            boundary_cov_residual = torch.zeros(expected_cov_shape, dtype=self._scaling.dtype, device=device)
            if torch.is_tensor(self._boundary_cov_residual) and self._boundary_cov_residual.numel() > 0:
                source = self._boundary_cov_residual.detach().to(device=device, dtype=self._scaling.dtype)
                if source.ndim == 1:
                    source = source.reshape(-1, 1)
                copy_rows = min(int(source.shape[0]), point_count)
                copy_cols = min(int(source.shape[1]), expected_cov_shape[1]) if source.ndim > 1 else 0
                if copy_rows > 0 and copy_cols > 0:
                    boundary_cov_residual[:copy_rows, :copy_cols] = source[:copy_rows, :copy_cols]
            self._boundary_cov_residual = nn.Parameter(boundary_cov_residual.requires_grad_(True))
            changed.append("boundary_cov_residual")

        if verbose and changed:
            print(
                "[GaussianModel] boundary state resynced for "
                f"{point_count} points: {', '.join(changed)}"
            )
        return len(changed) > 0

    def _semantic_adapter_enabled(self):
        return bool(self.cfg.get('semantic_logits_adapter_enable', False))

    def _semantic_asset_adapter_enabled(self):
        return bool(self.cfg.get('semantic_asset_logits_adapter_enable', False))

    def _layer_logits_adapter_enabled(self):
        return bool(self.cfg.get('binding_layer_logits_adapter_enable', False))

    def _ensure_layer_logits_adapter_state_matches_points(self, verbose=False):
        point_count = int(self.get_xyz.shape[0]) if torch.is_tensor(self._xyz) and self._xyz.ndim >= 2 else 0
        device = self._xyz.device if torch.is_tensor(self._xyz) and self._xyz.numel() > 0 else None
        expected_shape = (point_count, 3)
        if point_count <= 0:
            if not torch.is_tensor(self._binding_layer_logits_residual) or self._binding_layer_logits_residual.numel() > 0:
                self._binding_layer_logits_residual = torch.empty(0, 3, device=device)
                if verbose:
                    print("[GaussianModel] binding layer logits adapter cleared for empty model")
                return True
            return False
        if (
            not torch.is_tensor(self._binding_layer_logits_residual)
            or tuple(self._binding_layer_logits_residual.shape) != expected_shape
        ):
            residual = self._resize_pointwise_state(
                self._binding_layer_logits_residual,
                point_count,
                tail_shape=expected_shape[1:],
                dtype=self._xyz.dtype,
                device=device,
            )
            self._binding_layer_logits_residual = nn.Parameter(residual.requires_grad_(True))
            if verbose:
                print(
                    "[GaussianModel] binding layer logits adapter resynced for "
                    f"{point_count} points"
                )
            return True
        return False

    def apply_binding_layer_logits_adapter(self, layer_weights, boundary_score=None):
        if not self._layer_logits_adapter_enabled():
            return layer_weights
        self._ensure_layer_logits_adapter_state_matches_points(verbose=False)
        if not torch.is_tensor(layer_weights) or layer_weights.shape != self._binding_layer_logits_residual.shape:
            return layer_weights
        delta = self._binding_layer_logits_residual.to(device=layer_weights.device, dtype=layer_weights.dtype)
        max_delta = float(self.cfg.get('binding_layer_logits_adapter_max_delta', 0.75))
        if max_delta > 0.0:
            delta = torch.tanh(delta) * max_delta
        boundary_min = float(self.cfg.get('binding_layer_logits_adapter_boundary_min', 0.0))
        if boundary_min > 0.0 and torch.is_tensor(boundary_score) and boundary_score.shape[0] == layer_weights.shape[0]:
            gate = torch.sigmoid(
                (boundary_score.to(device=layer_weights.device, dtype=layer_weights.dtype).reshape(-1) - boundary_min)
                / 0.04
            ).reshape(-1, 1)
            delta = delta * gate
        return F.softmax(torch.log(layer_weights.clamp_min(1.0e-6)) + delta, dim=-1)

    def binding_layer_logits_adapter_regularization(self):
        if not self._layer_logits_adapter_enabled():
            device = self._xyz.device if torch.is_tensor(self._xyz) else "cuda"
            return torch.tensor(0.0, device=device)
        self._ensure_layer_logits_adapter_state_matches_points(verbose=False)
        return self._binding_layer_logits_residual.pow(2).mean()

    def _ensure_semantic_logits_residual_pair_matches_points(
        self,
        region_attr,
        compact_attr,
        region_label,
        compact_label,
        verbose_label,
        verbose=False,
    ):
        point_count = int(self.get_xyz.shape[0]) if torch.is_tensor(self._xyz) and self._xyz.ndim >= 2 else 0
        device = self._xyz.device if torch.is_tensor(self._xyz) and self._xyz.numel() > 0 else None
        changed = []

        region_shape = (point_count, 3)
        compact_shape = (point_count, int(self.cfg.get('semantic_logits_adapter_compact_classes', 6)))
        region_value = getattr(self, region_attr)
        compact_value = getattr(self, compact_attr)

        if point_count <= 0:
            if not torch.is_tensor(region_value) or region_value.numel() > 0:
                setattr(self, region_attr, torch.empty(0, 3, device=device))
                changed.append(region_label)
            if not torch.is_tensor(compact_value) or compact_value.numel() > 0:
                setattr(self, compact_attr, torch.empty(0, compact_shape[1], device=device))
                changed.append(compact_label)
            return len(changed) > 0

        if (
            not torch.is_tensor(region_value)
            or tuple(region_value.shape) != region_shape
        ):
            region_residual = self._resize_pointwise_state(
                region_value,
                point_count,
                tail_shape=region_shape[1:],
                dtype=self._xyz.dtype,
                device=device,
            )
            setattr(self, region_attr, nn.Parameter(region_residual.requires_grad_(True)))
            changed.append(region_label)

        if (
            not torch.is_tensor(compact_value)
            or tuple(compact_value.shape) != compact_shape
        ):
            compact_residual = self._resize_pointwise_state(
                compact_value,
                point_count,
                tail_shape=compact_shape[1:],
                dtype=self._xyz.dtype,
                device=device,
            )
            setattr(self, compact_attr, nn.Parameter(compact_residual.requires_grad_(True)))
            changed.append(compact_label)

        if verbose and changed:
            print(
                f"[GaussianModel] {verbose_label} state resynced for "
                f"{point_count} points: {', '.join(changed)}"
            )
        return len(changed) > 0

    def _ensure_semantic_adapter_state_matches_points(self, verbose=False):
        return self._ensure_semantic_logits_residual_pair_matches_points(
            "_semantic_region_logits_residual",
            "_semantic_compact_logits_residual",
            "semantic_region_logits_residual",
            "semantic_compact_logits_residual",
            "semantic adapter",
            verbose=verbose,
        )

    def _ensure_semantic_asset_adapter_state_matches_points(self, verbose=False):
        return self._ensure_semantic_logits_residual_pair_matches_points(
            "_semantic_asset_region_logits_residual",
            "_semantic_asset_compact_logits_residual",
            "semantic_asset_region_logits_residual",
            "semantic_asset_compact_logits_residual",
            "semantic asset adapter",
            verbose=verbose,
        )

    def _apply_semantic_logits_residual_pair(
        self,
        region_probs,
        compact_probs,
        region_residual,
        compact_residual,
        max_delta,
    ):
        if torch.is_tensor(region_probs) and region_probs.shape == region_residual.shape:
            delta = region_residual.to(device=region_probs.device, dtype=region_probs.dtype)
            if max_delta > 0.0:
                delta = torch.tanh(delta) * max_delta
            region_probs = F.softmax(torch.log(region_probs.clamp_min(1e-6)) + delta, dim=-1)

        if torch.is_tensor(compact_probs) and compact_probs.shape == compact_residual.shape:
            delta = compact_residual.to(device=compact_probs.device, dtype=compact_probs.dtype)
            if max_delta > 0.0:
                delta = torch.tanh(delta) * max_delta
            compact_probs = F.softmax(torch.log(compact_probs.clamp_min(1e-6)) + delta, dim=-1)

        return region_probs, compact_probs

    def apply_semantic_logits_adapter(self, region_probs, compact_probs):
        if not self._semantic_adapter_enabled():
            return region_probs, compact_probs
        self._ensure_semantic_adapter_state_matches_points(verbose=False)

        max_delta = float(self.cfg.get('semantic_logits_adapter_max_delta', 1.25))
        return self._apply_semantic_logits_residual_pair(
            region_probs,
            compact_probs,
            self._semantic_region_logits_residual,
            self._semantic_compact_logits_residual,
            max_delta,
        )

    def apply_semantic_logits_adapter_for_supervision(self, region_probs, compact_probs):
        return self.apply_semantic_asset_logits_adapter(region_probs, compact_probs)

    def apply_semantic_asset_logits_adapter(self, region_probs, compact_probs):
        if not self._semantic_asset_adapter_enabled():
            return region_probs, compact_probs
        self._ensure_semantic_asset_adapter_state_matches_points(verbose=False)

        max_delta = float(self.cfg.get(
            'semantic_asset_logits_adapter_max_delta',
            self.cfg.get('semantic_logits_adapter_max_delta', 1.25),
        ))
        return self._apply_semantic_logits_residual_pair(
            region_probs,
            compact_probs,
            self._semantic_asset_region_logits_residual,
            self._semantic_asset_compact_logits_residual,
            max_delta,
        )

    def semantic_logits_adapter_regularization(self):
        if not self._semantic_adapter_enabled():
            device = self._xyz.device if torch.is_tensor(self._xyz) else "cuda"
            return torch.tensor(0.0, device=device)
        self._ensure_semantic_adapter_state_matches_points(verbose=False)
        reg = self._semantic_region_logits_residual.pow(2).mean()
        reg = reg + self._semantic_compact_logits_residual.pow(2).mean()
        return reg

    def semantic_asset_logits_adapter_regularization(self):
        if not self._semantic_asset_adapter_enabled():
            device = self._xyz.device if torch.is_tensor(self._xyz) else "cuda"
            return torch.tensor(0.0, device=device)
        self._ensure_semantic_asset_adapter_state_matches_points(verbose=False)
        reg = self._semantic_asset_region_logits_residual.pow(2).mean()
        reg = reg + self._semantic_asset_compact_logits_residual.pow(2).mean()
        return reg

    def reset_boundary_residuals(self):
        if self.get_xyz.numel() <= 0:
            self._boundary_opacity_residual = torch.empty(0, device=self._xyz.device if torch.is_tensor(self._xyz) else None)
            self._boundary_scaling_residual = torch.empty(0, 3, device=self._xyz.device if torch.is_tensor(self._xyz) else None)
            self._boundary_cov_residual = torch.empty(0, self._boundary_cov_residual_channels(), device=self._xyz.device if torch.is_tensor(self._xyz) else None)
            return
        device = self._xyz.device
        self._boundary_opacity_residual = nn.Parameter(torch.zeros((self.get_xyz.shape[0], 1), dtype=torch.float, device=device).requires_grad_(True))
        self._boundary_scaling_residual = nn.Parameter(torch.zeros((self.get_xyz.shape[0], 3), dtype=torch.float, device=device).requires_grad_(True))
        self._boundary_cov_residual = nn.Parameter(torch.zeros((self.get_xyz.shape[0], self._boundary_cov_residual_channels()), dtype=torch.float, device=device).requires_grad_(True))

    def set_boundary_tags(self, boundary_tag):
        self.ensure_boundary_state_matches_points(verbose=False)
        if boundary_tag is None:
            self.clear_boundary_tags()
            return
        if not torch.is_tensor(boundary_tag):
            boundary_tag = torch.tensor(boundary_tag, dtype=torch.float32)
        boundary_tag = boundary_tag.detach().reshape(-1).float().clamp(0.0, 1.0)
        if self.get_xyz.numel() > 0 and boundary_tag.shape[0] != self.get_xyz.shape[0]:
            raise ValueError(f'boundary_tag shape mismatch: got {boundary_tag.shape[0]}, expected {self.get_xyz.shape[0]}')
        device = self._xyz.device if torch.is_tensor(self._xyz) and self._xyz.numel() > 0 else boundary_tag.device
        self._boundary_tag = boundary_tag.to(device=device)

    def _slice_boundary_tags(self, mask, repeats=1):
        boundary_tag = self.get_boundary_tag_state()
        if boundary_tag is None:
            return None
        selected = boundary_tag[mask]
        if repeats != 1:
            selected = selected.repeat(repeats)
        return selected.clone()

    def _append_boundary_tags(self, extension_tags):
        if extension_tags is None:
            return
        if not torch.is_tensor(extension_tags):
            extension_tags = torch.tensor(extension_tags, dtype=torch.float32)
        extension_tags = extension_tags.detach().reshape(-1).float().clamp(0.0, 1.0)

        point_count = int(self.get_xyz.shape[0]) if torch.is_tensor(self._xyz) and self._xyz.ndim >= 2 else 0
        device = self._xyz.device if torch.is_tensor(self._xyz) and self._xyz.numel() > 0 else extension_tags.device
        extension_count = int(extension_tags.shape[0])
        old_count = max(point_count - extension_count, 0)

        if torch.is_tensor(self._boundary_tag) and self._boundary_tag.shape[0] == old_count:
            base_tags = self._boundary_tag.to(device=device, dtype=torch.float32)
        elif self.has_boundary_tag_state():
            base_tags = self._boundary_tag.to(device=device, dtype=torch.float32)
            if base_tags.shape[0] == point_count:
                self._boundary_tag = base_tags
                return
            base_tags = torch.zeros((old_count,), dtype=torch.float32, device=device)
        else:
            base_tags = torch.zeros((old_count,), dtype=torch.float32, device=device)

        if old_count == 0:
            self._boundary_tag = extension_tags.to(device=device)
            return
        self._boundary_tag = torch.cat((base_tags, extension_tags.to(device=device)), dim=0)

    def set_fwd_transform(self, T_fwd):
        self.fwd_transform = T_fwd

    def color_by_opacity(self):
        cloned = self.clone()
        cloned._features_dc = self.get_opacity.unsqueeze(-1).expand(-1,-1,3)
        cloned._features_rest = torch.zeros_like(cloned._features_rest)
        return cloned

    def capture(self):
        self.ensure_boundary_state_matches_points(verbose=False)
        self._ensure_layer_logits_adapter_state_matches_points(verbose=False)
        self._ensure_semantic_adapter_state_matches_points(verbose=False)
        self._ensure_semantic_asset_adapter_state_matches_points(verbose=False)
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._boundary_tag,
            self._boundary_opacity_residual,
            self._boundary_scaling_residual,
            self._boundary_cov_residual,
            self._binding_layer_logits_residual,
            self._semantic_region_logits_residual,
            self._semantic_compact_logits_residual,
            self._semantic_asset_region_logits_residual,
            self._semantic_asset_compact_logits_residual,
            self._capture_binding_state(),
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            {} if self.optimizer is None else self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args, resume_cfg=None):
        if len(model_args) == 12:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
            self.clear_boundary_tags()
            self._boundary_opacity_residual = nn.Parameter(torch.zeros_like(self._opacity).requires_grad_(True))
            self._boundary_scaling_residual = nn.Parameter(torch.zeros_like(self._scaling).requires_grad_(True))
            self._boundary_cov_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], self._boundary_cov_residual_channels()), dtype=self._opacity.dtype, device=self._opacity.device).requires_grad_(True))
            self._binding_layer_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
            self._semantic_region_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
            self._semantic_compact_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 6), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
            self._semantic_asset_region_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
            self._semantic_asset_compact_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 6), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
        elif len(model_args) == 13:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._boundary_tag,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
            self._boundary_opacity_residual = nn.Parameter(torch.zeros_like(self._opacity).requires_grad_(True))
            self._boundary_scaling_residual = nn.Parameter(torch.zeros_like(self._scaling).requires_grad_(True))
            self._boundary_cov_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], self._boundary_cov_residual_channels()), dtype=self._opacity.dtype, device=self._opacity.device).requires_grad_(True))
            self._binding_layer_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
            self._semantic_region_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
            self._semantic_compact_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 6), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
            self._semantic_asset_region_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
            self._semantic_asset_compact_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 6), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
        elif len(model_args) == 15:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._boundary_tag,
            self._boundary_opacity_residual,
            self._boundary_scaling_residual,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
            self._semantic_region_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
            self._semantic_compact_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 6), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
            self._boundary_cov_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], self._boundary_cov_residual_channels()), dtype=self._opacity.dtype, device=self._opacity.device).requires_grad_(True))
            self._binding_layer_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
            self._semantic_asset_region_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
            self._semantic_asset_compact_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 6), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
        elif len(model_args) == 17:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._boundary_tag,
            self._boundary_opacity_residual,
            self._boundary_scaling_residual,
            self._semantic_region_logits_residual,
            self._semantic_compact_logits_residual,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
            self._boundary_cov_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], self._boundary_cov_residual_channels()), dtype=self._opacity.dtype, device=self._opacity.device).requires_grad_(True))
            self._binding_layer_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
            self._semantic_asset_region_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
            self._semantic_asset_compact_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 6), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
        elif len(model_args) == 18:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._boundary_tag,
            self._boundary_opacity_residual,
            self._boundary_scaling_residual,
            self._semantic_region_logits_residual,
            self._semantic_compact_logits_residual,
            binding_state,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
            self.set_binding_state(binding_state)
            self._boundary_cov_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], self._boundary_cov_residual_channels()), dtype=self._opacity.dtype, device=self._opacity.device).requires_grad_(True))
            self._binding_layer_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
            self._semantic_asset_region_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
            self._semantic_asset_compact_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 6), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
        elif len(model_args) == 19:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._boundary_tag,
            self._boundary_opacity_residual,
            self._boundary_scaling_residual,
            self._boundary_cov_residual,
            self._semantic_region_logits_residual,
            self._semantic_compact_logits_residual,
            binding_state,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
            self.set_binding_state(binding_state)
            self._binding_layer_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
            self._semantic_asset_region_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
            self._semantic_asset_compact_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 6), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
        elif len(model_args) == 20:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._boundary_tag,
            self._boundary_opacity_residual,
            self._boundary_scaling_residual,
            self._boundary_cov_residual,
            self._binding_layer_logits_residual,
            self._semantic_region_logits_residual,
            self._semantic_compact_logits_residual,
            binding_state,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
            self.set_binding_state(binding_state)
            self._semantic_asset_region_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 3), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
            self._semantic_asset_compact_logits_residual = nn.Parameter(torch.zeros((self._xyz.shape[0], 6), dtype=self._xyz.dtype, device=self._xyz.device).requires_grad_(True))
        elif len(model_args) == 22:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._boundary_tag,
            self._boundary_opacity_residual,
            self._boundary_scaling_residual,
            self._boundary_cov_residual,
            self._binding_layer_logits_residual,
            self._semantic_region_logits_residual,
            self._semantic_compact_logits_residual,
            self._semantic_asset_region_logits_residual,
            self._semantic_asset_compact_logits_residual,
            binding_state,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
            self.set_binding_state(binding_state)
        else:
            raise ValueError(f'Unexpected GaussianModel checkpoint format with {len(model_args)} entries.')
        self.ensure_boundary_state_matches_points(verbose=True)
        self._ensure_layer_logits_adapter_state_matches_points(verbose=True)
        self._ensure_semantic_adapter_state_matches_points(verbose=True)
        self._ensure_semantic_asset_adapter_state_matches_points(verbose=True)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer = None

    def training_setup(self, training_args):
        self.ensure_boundary_state_matches_points(verbose=False)
        self._ensure_layer_logits_adapter_state_matches_points(verbose=False)
        self._ensure_semantic_adapter_state_matches_points(verbose=False)
        self._ensure_semantic_asset_adapter_state_matches_points(verbose=False)
        self.percent_dense = float(training_args.get('percent_dense', 0.0))
        device = self.get_xyz.device
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        if not torch.is_tensor(self.max_radii2D) or self.max_radii2D.shape[0] != self.get_xyz.shape[0]:
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0],), device=device)

        feature_ratio = 20.0 if self.use_sh else 1.0
        params = [
            {'params': [self._xyz], 'lr': float(training_args.get('position_lr_init', 0.0)) * float(self.spatial_lr_scale or 1.0), "name": "xyz"},
            {'params': [self._features_dc], 'lr': float(training_args.get('feature_lr', 0.0)), "name": "f_dc"},
            {'params': [self._features_rest], 'lr': float(training_args.get('feature_lr', 0.0)) / feature_ratio, "name": "f_rest"},
            {'params': [self._opacity], 'lr': float(training_args.get('opacity_lr', 0.0)), "name": "opacity"},
            {'params': [self._scaling], 'lr': float(training_args.get('scaling_lr', 0.0)), "name": "scaling"},
            {'params': [self._rotation], 'lr': float(training_args.get('rotation_lr', 0.0)), "name": "rotation"},
            {'params': [self._boundary_opacity_residual], 'lr': float(training_args.get('boundary_opacity_residual_lr', 0.0)), "name": "boundary_opacity_residual"},
            {'params': [self._boundary_scaling_residual], 'lr': float(training_args.get('boundary_scaling_residual_lr', 0.0)), "name": "boundary_scaling_residual"},
            {'params': [self._boundary_cov_residual], 'lr': float(training_args.get('boundary_cov_residual_lr', 0.0)), "name": "boundary_cov_residual"},
            {'params': [self._binding_layer_logits_residual], 'lr': float(training_args.get('binding_layer_logits_lr', 0.0)), "name": "binding_layer_logits_residual"},
            {'params': [self._semantic_region_logits_residual], 'lr': float(training_args.get('semantic_region_logits_lr', 0.0)), "name": "semantic_region_logits_residual"},
            {'params': [self._semantic_compact_logits_residual], 'lr': float(training_args.get('semantic_compact_logits_lr', 0.0)), "name": "semantic_compact_logits_residual"},
            {'params': [self._semantic_asset_region_logits_residual], 'lr': float(training_args.get('semantic_asset_region_logits_lr', 0.0)), "name": "semantic_asset_region_logits_residual"},
            {'params': [self._semantic_asset_compact_logits_residual], 'lr': float(training_args.get('semantic_asset_compact_logits_lr', 0.0)), "name": "semantic_asset_compact_logits_residual"},
        ]
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = _constant_lr_func(params[0]['lr'])

    def update_learning_rate(self, iteration):
        if self.optimizer is None:
            return 0.0
        lr = 0.0
        for param_group in self.optimizer.param_groups:
            if param_group.get("name") == "xyz":
                lr = float(self.xyz_scheduler_args(iteration)) if hasattr(self, "xyz_scheduler_args") else float(param_group.get("lr", 0.0))
                param_group['lr'] = lr
                break
        return lr

    def prune_nonfinite_points(self, verbose=False):
        tensors = [self._xyz, self._features_dc, self._features_rest, self._scaling, self._rotation, self._opacity]
        has_bad = any(torch.is_tensor(t) and t.numel() > 0 and not bool(torch.isfinite(t.detach()).all().item()) for t in tensors)
        if has_bad and verbose:
            print("[GaussianModel] non-finite values detected; no automatic prune is available in the v338 train path.")
        return 0

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        return None

    def densify_and_prune(self, *args, **kwargs):
        return None

    def reset_opacity(self):
        if self.optimizer is None or not torch.is_tensor(self._opacity):
            return
        with torch.no_grad():
            self._opacity.copy_(inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)))

    def mark_stale_binding_points_for_refresh(self, *args, **kwargs):
        return 0

    def apply_post_rebind_child_correction(self, *args, **kwargs):
        return None

    @property
    def get_scaling(self):
        raw_scaling = self._scaling
        boundary_residual_support = self.get_boundary_residual_support()
        if (
            torch.is_tensor(self._boundary_scaling_residual)
            and self._boundary_scaling_residual.numel() == self._scaling.numel()
            and boundary_residual_support is not None
        ):
            raw_scaling = raw_scaling + self._boundary_scaling_residual * boundary_residual_support.unsqueeze(-1)
        return self.scaling_activation(raw_scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        raw_opacity = self._opacity
        boundary_residual_support = self.get_boundary_residual_support()
        if (
            torch.is_tensor(self._boundary_opacity_residual)
            and self._boundary_opacity_residual.numel() == self._opacity.numel()
            and boundary_residual_support is not None
        ):
            raw_opacity = raw_opacity + self._boundary_opacity_residual * boundary_residual_support.unsqueeze(-1)
        return self.opacity_activation(raw_opacity)
    
    @staticmethod
    def _orthogonalize_covariance_rotation(rotation):
        if not torch.is_tensor(rotation) or rotation.numel() == 0 or rotation.shape[-2:] != (3, 3):
            return rotation
        try:
            u, _, vh = torch.linalg.svd(rotation)
            orth = torch.matmul(u, vh)
            det = torch.det(orth)
            if torch.is_tensor(det):
                bad = det < 0
                if bool(bad.any()):
                    u = u.clone()
                    u[bad, :, -1] *= -1
                    orth = torch.matmul(u, vh)
            return orth
        except Exception:
            return rotation

    @staticmethod
    def _reduce_scaling_to_scalar(scaling, reduce_mode):
        reduce_mode = str(reduce_mode or "geom").lower()
        eps = 1.0e-8
        if reduce_mode == "mean":
            return scaling.mean(dim=-1, keepdim=True)
        if reduce_mode == "max":
            return scaling.max(dim=-1, keepdim=True).values
        if reduce_mode == "min":
            return scaling.min(dim=-1, keepdim=True).values
        return torch.exp(torch.log(torch.clamp(scaling, min=eps)).mean(dim=-1, keepdim=True))

    @staticmethod
    def _apply_anisotropy_clamp(scaling, clamp_ratio):
        clamp_ratio = float(clamp_ratio or 0.0)
        if clamp_ratio <= 1.0:
            return scaling
        eps = 1.0e-8
        log_scaling = torch.log(torch.clamp(scaling, min=eps))
        log_center = log_scaling.mean(dim=-1, keepdim=True)
        log_limit = float(np.log(clamp_ratio))
        return torch.exp(torch.clamp(log_scaling - log_center, -log_limit, log_limit) + log_center)

    @staticmethod
    def _load_covariance_signed_point_ids(point_json, camera=None):
        path = str(point_json or "")
        if not path:
            return (), ()
        image_name = GaussianModel._camera_image_name(camera)
        cache_key = f"{path}|{image_name}"
        cached = GaussianModel._covariance_signed_point_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            data = {}
        image_data = None
        by_image = data.get("by_image", None) if isinstance(data, dict) else None
        if image_name and isinstance(by_image, dict):
            image_data = by_image.get(image_name, None)
        source = image_data if isinstance(image_data, dict) else data
        shrink_ids = tuple(int(idx) for idx in source.get("shrink_point_ids", []) if int(idx) >= 0)
        grow_ids = tuple(int(idx) for idx in source.get("grow_point_ids", []) if int(idx) >= 0)
        cached = (shrink_ids, grow_ids)
        GaussianModel._covariance_signed_point_cache[cache_key] = cached
        return cached

    @staticmethod
    def _ids_to_mask(ids, point_count, device):
        mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
        if not ids:
            return mask
        idx = torch.tensor(ids, dtype=torch.long, device=device)
        idx = idx[(idx >= 0) & (idx < point_count)]
        if idx.numel() > 0:
            mask[idx] = True
        return mask

    @staticmethod
    def _parse_covariance_id_list(value, name_to_id=None):
        if value is None:
            return ()
        if torch.is_tensor(value):
            if value.numel() == 0:
                return ()
            return tuple(int(x) for x in value.detach().cpu().reshape(-1).tolist())
        if isinstance(value, (list, tuple, set)):
            parsed = []
            for item in value:
                parsed.extend(GaussianModel._parse_covariance_id_list(item, name_to_id=name_to_id))
            return tuple(parsed)
        if isinstance(value, (int, np.integer)):
            return (int(value),)
        text = str(value or "").strip()
        if not text or text.lower() in ("none", "null", "all", "*"):
            return ()
        for ch in "[](){}":
            text = text.replace(ch, " ")
        tokens = [tok.strip() for tok in text.replace(";", ",").split(",") if tok.strip()]
        parsed = []
        name_to_id = name_to_id or {}
        for token in tokens:
            lowered = token.lower()
            if lowered in name_to_id:
                parsed.append(int(name_to_id[lowered]))
                continue
            try:
                parsed.append(int(float(token)))
            except ValueError:
                continue
        return tuple(parsed)

    @staticmethod
    def _values_to_mask(values, allowed_ids):
        if not torch.is_tensor(values):
            return None
        allowed_ids = tuple(int(x) for x in (allowed_ids or ()))
        if not allowed_ids:
            return torch.ones((values.shape[0],), dtype=torch.bool, device=values.device)
        allowed = torch.tensor(allowed_ids, dtype=values.dtype, device=values.device)
        return (values.reshape(-1).unsqueeze(-1) == allowed.reshape(1, -1)).any(dim=-1)

    @staticmethod
    def _bool_attr_mask(value, point_count, device, default=False):
        if torch.is_tensor(value) and value.shape[0] == point_count:
            return value.to(device=device).reshape(-1).bool()
        return torch.full((point_count,), bool(default), dtype=torch.bool, device=device)

    @staticmethod
    def _scalar_attr(value, point_count, device, default=0.0):
        if torch.is_tensor(value) and value.shape[0] == point_count:
            return value.to(device=device).reshape(-1).float()
        return torch.full((point_count,), float(default), dtype=torch.float32, device=device)

    @staticmethod
    def _long_attr(value, point_count, device, default=0):
        if torch.is_tensor(value) and value.shape[0] == point_count:
            return value.to(device=device).reshape(-1).long()
        return torch.full((point_count,), int(default), dtype=torch.long, device=device)

    @staticmethod
    def _topk_bool_mask(base_mask, score, max_points):
        if not torch.is_tensor(base_mask) or base_mask.numel() == 0:
            return base_mask
        max_points = int(max_points)
        if max_points < 0:
            return base_mask
        if max_points == 0:
            return torch.zeros_like(base_mask, dtype=torch.bool)
        active_count = int(base_mask.sum().item())
        if active_count <= max_points:
            return base_mask
        score = score.to(device=base_mask.device).reshape(-1).float()
        ranked = torch.where(base_mask, score, torch.full_like(score, -float("inf")))
        top_values, top_idx = torch.topk(ranked, k=min(max_points, int(ranked.numel())))
        keep = top_values > -float("inf")
        mask = torch.zeros_like(base_mask, dtype=torch.bool)
        if bool(keep.any().item()):
            mask[top_idx[keep]] = True
        return mask

    @staticmethod
    def _camera_image_name(camera):
        if camera is None:
            return ""
        image_name = getattr(camera, "image_name", "")
        if isinstance(image_name, (list, tuple)):
            image_name = image_name[0] if image_name else ""
        if isinstance(image_name, str) and image_name:
            return image_name
        cam_id = getattr(camera, "cam_id", getattr(camera, "uid", ""))
        frame_id = getattr(camera, "frame_id", getattr(camera, "frame_idx", ""))
        try:
            cam_int = int(cam_id)
            frame_int = int(frame_id)
            return f"c{cam_int:02d}_f{frame_int:06d}"
        except Exception:
            return ""

    @staticmethod
    def _project_covariance_points(points, camera):
        if camera is None or not torch.is_tensor(points) or points.numel() == 0:
            return None, None
        try:
            ndc = geom_transform_points(points.detach(), camera.full_proj_transform.to(device=points.device, dtype=points.dtype))
            width = int(camera.image_width)
            height = int(camera.image_height)
        except Exception:
            return None, None
        px = (ndc[:, 0] + 1.0) * 0.5 * float(max(width - 1, 1))
        py = (1.0 - (ndc[:, 1] + 1.0) * 0.5) * float(max(height - 1, 1))
        valid = torch.isfinite(ndc).all(dim=1)
        valid = valid & (ndc[:, 2] > 0.0)
        valid = valid & (px >= 0.0) & (px <= float(max(width - 1, 0)))
        valid = valid & (py >= 0.0) & (py <= float(max(height - 1, 0)))
        return torch.stack((px, py), dim=-1), valid

    @staticmethod
    def _load_covariance_signed_components(component_csv, point_csv=""):
        path = str(component_csv or "")
        if not path:
            return {}
        point_path = str(point_csv or "")
        cache_key = f"{path}|{point_path}"
        cached = GaussianModel._covariance_signed_component_cache.get(cache_key)
        if cached is not None:
            return cached
        point_stats = {}
        if point_path:
            try:
                with open(point_path, "r", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        try:
                            point_idx = int(row.get("point_idx", -1))
                            point_stats[point_idx] = {
                                "layer_id": int(float(row.get("layer_id", 0) or 0)),
                                "region_id": int(float(row.get("region_id", 0) or 0)),
                                "dominant_joint": int(float(row.get("dominant_joint", -1) or -1)),
                                "surface_distance": float(row.get("surface_distance", 0.0) or 0.0),
                                "thin_score": float(row.get("thin_score", 0.0) or 0.0),
                                "boundary_score": float(row.get("boundary_score", 0.0) or 0.0),
                            }
                        except ValueError:
                            continue
            except Exception:
                point_stats = {}

        def _parse_ids(text):
            ids = []
            for token in str(text or "").replace(",", ";").split(";"):
                token = token.strip()
                if not token:
                    continue
                try:
                    ids.append(int(token))
                except ValueError:
                    continue
            return ids

        def _common_values(stats, key, max_values=4):
            counts = {}
            for item in stats:
                value = int(item[key])
                counts[value] = counts.get(value, 0) + 1
            if not counts:
                return []
            ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
            return [int(value) for value, _ in ordered[: int(max_values)]]

        def _range_values(stats, key, pad):
            values = [float(item[key]) for item in stats if np.isfinite(float(item[key]))]
            if not values:
                return None, None
            return max(0.0, min(values) - float(pad)), max(values) + float(pad)

        by_image = {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    image_name = str(row.get("image_name", "")).strip()
                    direction = str(row.get("direction", "")).strip().lower()
                    if not image_name or direction not in ("outer", "inner"):
                        continue
                    try:
                        record = {
                            "direction": direction,
                            "area": float(row.get("area", 0.0) or 0.0),
                            "cx": float(row.get("centroid_x", 0.0) or 0.0),
                            "cy": float(row.get("centroid_y", 0.0) or 0.0),
                            "x": float(row.get("bbox_x", 0.0) or 0.0),
                            "y": float(row.get("bbox_y", 0.0) or 0.0),
                            "w": float(row.get("bbox_w", 0.0) or 0.0),
                            "h": float(row.get("bbox_h", 0.0) or 0.0),
                            "near_score_sum": float(row.get("near_score_sum", 0.0) or 0.0),
                            "top_point_ids": str(row.get("top_point_ids", "") or ""),
                        }
                    except ValueError:
                        continue
                    top_stats = [point_stats[idx] for idx in _parse_ids(row.get("top_point_ids", "")) if idx in point_stats]
                    if top_stats:
                        surface_min, surface_max = _range_values(top_stats, "surface_distance", 0.012)
                        thin_min, thin_max = _range_values(top_stats, "thin_score", 0.22)
                        boundary_min, _ = _range_values(top_stats, "boundary_score", 0.18)
                        record.update({
                            "sig_layer_ids": _common_values(top_stats, "layer_id", max_values=3),
                            "sig_region_ids": _common_values(top_stats, "region_id", max_values=3),
                            "sig_joint_ids": _common_values(top_stats, "dominant_joint", max_values=5),
                            "sig_surface_min": surface_min,
                            "sig_surface_max": surface_max,
                            "sig_thin_min": thin_min,
                            "sig_thin_max": thin_max,
                            "sig_boundary_min": boundary_min,
                        })
                    by_image.setdefault(image_name, {"outer": [], "inner": []})[direction].append(record)
        except Exception:
            by_image = {}
        for item in by_image.values():
            for direction in ("outer", "inner"):
                item[direction].sort(key=lambda rec: (rec.get("area", 0.0), rec.get("near_score_sum", 0.0)), reverse=True)
        GaussianModel._covariance_signed_component_cache[cache_key] = by_image
        return by_image

    @staticmethod
    def _component_spatial_mask_and_normals(
        xy,
        valid,
        camera,
        component_csv,
        point_csv,
        direction,
        layer_ids=None,
        region_ids=None,
        joint_ids=None,
        surface=None,
        thin=None,
        boundary=None,
        signature_enable=False,
        pad_px=10.0,
        ellipse_scale=1.25,
        max_components=16,
        min_area=1.0,
        top_ids_enable=False,
        top_ids_only=False,
    ):
        point_count = int(xy.shape[0]) if torch.is_tensor(xy) else 0
        if point_count <= 0:
            return None, None, False
        device = xy.device
        mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
        normals = torch.zeros((point_count, 2), dtype=xy.dtype, device=device)
        counts = torch.zeros((point_count, 1), dtype=xy.dtype, device=device)
        scores = torch.zeros((point_count,), dtype=xy.dtype, device=device)
        records_by_image = GaussianModel._load_covariance_signed_components(component_csv, point_csv=point_csv)
        image_name = GaussianModel._camera_image_name(camera)
        records = records_by_image.get(image_name, {}).get(str(direction).lower(), [])
        records = [rec for rec in records if float(rec.get("area", 0.0)) >= float(min_area)]
        records = records[: max(int(max_components), 0)]
        if not records:
            return mask, normals, scores, False

        pad_px = float(pad_px)
        ellipse_scale = max(float(ellipse_scale), 1.0e-6)
        for rec in records:
            cx = float(rec["cx"])
            cy = float(rec["cy"])
            half_w = max(0.5 * float(rec["w"]) * ellipse_scale + pad_px, 1.0)
            half_h = max(0.5 * float(rec["h"]) * ellipse_scale + pad_px, 1.0)
            dx = xy[:, 0] - cx
            dy = xy[:, 1] - cy
            ellipse_dist2 = (dx / half_w) ** 2 + (dy / half_h) ** 2
            inside = ellipse_dist2 <= 1.0
            inside = inside & valid.bool()
            feature_score = torch.ones((point_count,), dtype=xy.dtype, device=device)
            if bool(signature_enable) and rec.get("sig_layer_ids", None) is not None:
                if layer_ids is not None and rec.get("sig_layer_ids"):
                    inside = inside & GaussianModel._values_to_mask(layer_ids, rec["sig_layer_ids"])
                if region_ids is not None and rec.get("sig_region_ids"):
                    inside = inside & GaussianModel._values_to_mask(region_ids, rec["sig_region_ids"])
                if joint_ids is not None and rec.get("sig_joint_ids"):
                    inside = inside & GaussianModel._values_to_mask(joint_ids, rec["sig_joint_ids"])
                if surface is not None:
                    sig_min = rec.get("sig_surface_min", None)
                    sig_max = rec.get("sig_surface_max", None)
                    if sig_min is not None:
                        inside = inside & (surface.reshape(-1) >= float(sig_min))
                    if sig_max is not None:
                        inside = inside & (surface.reshape(-1) <= float(sig_max))
                    if sig_min is not None and sig_max is not None:
                        center = 0.5 * (float(sig_min) + float(sig_max))
                        radius = max(0.5 * (float(sig_max) - float(sig_min)), 1.0e-6)
                        feature_score = feature_score * (1.0 - torch.abs(surface.reshape(-1) - center) / radius).clamp(0.0, 1.0)
                if thin is not None:
                    sig_min = rec.get("sig_thin_min", None)
                    sig_max = rec.get("sig_thin_max", None)
                    if sig_min is not None:
                        inside = inside & (thin.reshape(-1) >= float(sig_min))
                    if sig_max is not None:
                        inside = inside & (thin.reshape(-1) <= float(sig_max))
                    if sig_min is not None and sig_max is not None:
                        center = 0.5 * (float(sig_min) + float(sig_max))
                        radius = max(0.5 * (float(sig_max) - float(sig_min)), 1.0e-6)
                        feature_score = feature_score * (0.35 + 0.65 * (1.0 - torch.abs(thin.reshape(-1) - center) / radius).clamp(0.0, 1.0))
                if boundary is not None and rec.get("sig_boundary_min", None) is not None:
                    inside = inside & (boundary.reshape(-1) >= float(rec["sig_boundary_min"]))
            if bool(top_ids_enable):
                top_ids = []
                for value in str(rec.get("top_point_ids", "") or "").split(";"):
                    value = value.strip()
                    if not value:
                        continue
                    try:
                        idx = int(value)
                    except Exception:
                        continue
                    if 0 <= idx < point_count:
                        top_ids.append(idx)
                if top_ids:
                    top_mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
                    top_mask[torch.tensor(top_ids, dtype=torch.long, device=device)] = True
                    if bool(top_ids_only):
                        inside = inside & top_mask
                    else:
                        inside = inside | (top_mask & valid.bool())
            if not bool(inside.any().item()):
                continue
            local = torch.stack((dx, dy), dim=-1)
            local = F.normalize(local, dim=-1, eps=1.0e-6)
            spatial_score = (1.0 - torch.sqrt(ellipse_dist2.clamp_min(0.0))).clamp(0.0, 1.0)
            component_weight = float(rec.get("near_score_sum", 1.0) or 1.0)
            component_weight = max(component_weight, 1.0)
            local_score = spatial_score * feature_score * float(np.log1p(component_weight))
            mask = mask | inside
            normals[inside] = normals[inside] + local[inside]
            counts[inside] = counts[inside] + 1.0
            scores[inside] = torch.maximum(scores[inside], local_score[inside])
        has_records = True
        active = counts.reshape(-1) > 0.0
        if bool(active.any().item()):
            normals[active] = F.normalize(normals[active] / counts[active].clamp_min(1.0), dim=-1, eps=1.0e-6)
        return mask, normals, scores, has_records

    @staticmethod
    def _apply_signed_masks_to_scaling(
        scaling,
        shrink_mask,
        grow_mask,
        shrink_factor=1.0,
        grow_factor=1.0,
        anisotropic_axis="all",
    ):
        shrink_factor = float(shrink_factor or 1.0)
        grow_factor = float(grow_factor or 1.0)
        if np.isclose(shrink_factor, 1.0) and np.isclose(grow_factor, 1.0):
            return scaling
        edited = scaling.clone()
        axis = str(anisotropic_axis or "all").lower()

        def _apply(mask, factor):
            if not torch.is_tensor(mask) or not bool(mask.any().item()) or np.isclose(float(factor), 1.0):
                return
            if factor <= 0.0:
                return
            if axis in ("max", "major", "largest"):
                selected = edited[mask]
                axis_idx = selected.argmax(dim=-1)
                rows = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                edited[rows, axis_idx] = edited[rows, axis_idx] * factor
            elif axis in ("min", "minor", "smallest"):
                selected = edited[mask]
                axis_idx = selected.argmin(dim=-1)
                rows = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                edited[rows, axis_idx] = edited[rows, axis_idx] * factor
            else:
                edited[mask] = edited[mask] * factor

        _apply(shrink_mask, shrink_factor)
        _apply(grow_mask, grow_factor)
        return edited

    def _apply_signed_point_scaling(
        self,
        scaling,
        signed_point_json="",
        shrink_factor=1.0,
        grow_factor=1.0,
        max_shrink_points=-1,
        max_grow_points=-1,
        anisotropic_axis="all",
    ):
        if not signed_point_json:
            return scaling
        shrink_factor = float(shrink_factor or 1.0)
        grow_factor = float(grow_factor or 1.0)
        if np.isclose(shrink_factor, 1.0) and np.isclose(grow_factor, 1.0):
            return scaling
        shrink_ids, grow_ids = self._load_covariance_signed_point_ids(signed_point_json)
        max_shrink_points = int(max_shrink_points)
        max_grow_points = int(max_grow_points)
        if max_shrink_points >= 0:
            shrink_ids = shrink_ids[:max_shrink_points]
        if max_grow_points >= 0:
            grow_ids = grow_ids[:max_grow_points]
        if not shrink_ids and not grow_ids:
            return scaling

        edited = scaling.clone()
        point_count = int(edited.shape[0])
        shrink_mask = self._ids_to_mask(shrink_ids, point_count, edited.device)
        grow_mask = self._ids_to_mask(grow_ids, point_count, edited.device)
        axis = str(anisotropic_axis or "all").lower()

        def _apply(mask, factor):
            if not bool(mask.any().item()) or np.isclose(float(factor), 1.0):
                return
            if factor <= 0.0:
                return
            if axis in ("max", "major", "largest"):
                selected = edited[mask]
                axis_idx = selected.argmax(dim=-1)
                rows = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                edited[rows, axis_idx] = edited[rows, axis_idx] * factor
            elif axis in ("min", "minor", "smallest"):
                selected = edited[mask]
                axis_idx = selected.argmin(dim=-1)
                rows = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                edited[rows, axis_idx] = edited[rows, axis_idx] * factor
            else:
                edited[mask] = edited[mask] * factor

        _apply(shrink_mask, shrink_factor)
        _apply(grow_mask, grow_factor)
        return edited

    def _dynamic_signed_covariance_masks(
        self,
        camera=None,
        component_csv="",
        point_csv="",
        component_signature_enable=False,
        over_layer_ids=(),
        over_region_ids=(),
        over_joint_ids=(),
        under_layer_ids=(),
        under_region_ids=(),
        under_joint_ids=(),
        boundary_min=0.0,
        surface_min=None,
        surface_max=None,
        component_pad_px=10.0,
        component_ellipse_scale=1.25,
        component_max_over=16,
        component_max_under=16,
        component_min_area=1.0,
        component_required=False,
        component_top_ids_enable=False,
        component_top_ids_only=False,
        max_over_points=-1,
        max_under_points=-1,
    ):
        point_count = int(self.get_xyz.shape[0])
        device = self.get_xyz.device
        empty = torch.zeros((point_count,), dtype=torch.bool, device=device)
        zero_normals = torch.zeros((point_count, 2), dtype=self.get_xyz.dtype, device=device)
        zero_scores = torch.zeros((point_count,), dtype=self.get_xyz.dtype, device=device)
        if point_count <= 0:
            return empty, empty, zero_normals, zero_normals, zero_scores, zero_scores

        layer_ids = self._long_attr(getattr(self, "binding_layer_ids", None), point_count, device, default=0)
        region_ids = self._long_attr(getattr(self, "binding_region_ids", None), point_count, device, default=0)
        joint_ids = self._long_attr(getattr(self, "binding_dominant_joint", None), point_count, device, default=-1)
        boundary = self._scalar_attr(getattr(self, "binding_boundary_score", None), point_count, device, default=0.0)
        surface = self._scalar_attr(getattr(self, "binding_surface_distance", None), point_count, device, default=0.0)
        thin = self._scalar_attr(getattr(self, "binding_thin_score", None), point_count, device, default=0.0)

        layer_names = {"rigid": 0, "soft": 1, "free": 2}
        region_names = {"body": 0, "soft": 1, "cloth": 2}
        over_layer_ids = self._parse_covariance_id_list(over_layer_ids, name_to_id=layer_names)
        under_layer_ids = self._parse_covariance_id_list(under_layer_ids, name_to_id=layer_names)
        over_region_ids = self._parse_covariance_id_list(over_region_ids, name_to_id=region_names)
        under_region_ids = self._parse_covariance_id_list(under_region_ids, name_to_id=region_names)
        over_joint_ids = self._parse_covariance_id_list(over_joint_ids)
        under_joint_ids = self._parse_covariance_id_list(under_joint_ids)

        over_mask = self._values_to_mask(layer_ids, over_layer_ids)
        over_mask = over_mask & self._values_to_mask(region_ids, over_region_ids)
        over_mask = over_mask & self._values_to_mask(joint_ids, over_joint_ids)
        under_mask = self._values_to_mask(layer_ids, under_layer_ids)
        under_mask = under_mask & self._values_to_mask(region_ids, under_region_ids)
        under_mask = under_mask & self._values_to_mask(joint_ids, under_joint_ids)

        boundary_min = float(boundary_min or 0.0)
        if boundary_min > 0.0:
            over_mask = over_mask & (boundary >= boundary_min)
            under_mask = under_mask & (boundary >= boundary_min)

        if surface_min is not None and str(surface_min) != "":
            surface_min = float(surface_min)
            over_mask = over_mask & (surface >= surface_min)
            under_mask = under_mask & (surface >= surface_min)
        if surface_max is not None and str(surface_max) != "":
            surface_max = float(surface_max)
            over_mask = over_mask & (surface <= surface_max)
            under_mask = under_mask & (surface <= surface_max)

        xy, valid = self._project_covariance_points(self.get_xyz, camera)
        over_normals = zero_normals
        under_normals = zero_normals.clone()
        over_component_score = torch.zeros((point_count,), dtype=torch.float32, device=device)
        under_component_score = torch.zeros((point_count,), dtype=torch.float32, device=device)
        if xy is not None and torch.is_tensor(valid):
            over_component, over_normals, over_component_score, has_over_records = self._component_spatial_mask_and_normals(
                xy,
                valid,
                camera,
                component_csv,
                point_csv,
                "outer",
                layer_ids=layer_ids,
                region_ids=region_ids,
                joint_ids=joint_ids,
                surface=surface,
                thin=thin,
                boundary=boundary,
                signature_enable=component_signature_enable,
                pad_px=component_pad_px,
                ellipse_scale=component_ellipse_scale,
                max_components=component_max_over,
                min_area=component_min_area,
                top_ids_enable=component_top_ids_enable,
                top_ids_only=component_top_ids_only,
            )
            under_component, under_normals, under_component_score, has_under_records = self._component_spatial_mask_and_normals(
                xy,
                valid,
                camera,
                component_csv,
                point_csv,
                "inner",
                layer_ids=layer_ids,
                region_ids=region_ids,
                joint_ids=joint_ids,
                surface=surface,
                thin=thin,
                boundary=boundary,
                signature_enable=component_signature_enable,
                pad_px=component_pad_px,
                ellipse_scale=component_ellipse_scale,
                max_components=component_max_under,
                min_area=component_min_area,
                top_ids_enable=component_top_ids_enable,
                top_ids_only=component_top_ids_only,
            )
            if has_over_records and over_component is not None:
                over_mask = over_mask & over_component
            elif bool(component_required):
                over_mask = torch.zeros_like(over_mask, dtype=torch.bool)
            else:
                over_mask = over_mask & valid.bool()
            if has_under_records and under_component is not None:
                under_mask = under_mask & under_component
            elif bool(component_required):
                under_mask = torch.zeros_like(under_mask, dtype=torch.bool)
            else:
                under_mask = under_mask & valid.bool()

        scale = self.get_scaling.detach().float()
        scale_score = (scale.mean(dim=-1) / scale.mean(dim=-1).detach().quantile(0.90).clamp_min(1.0e-6)).clamp(0.0, 2.0)
        opacity_score = self.get_opacity.detach().reshape(-1).float().clamp(0.0, 1.0)
        over_score = over_component_score.float()
        under_score = under_component_score.float()
        fallback_over = boundary.clamp(0.0, 1.0) + 0.15 * surface.clamp_min(0.0)
        fallback_under = boundary.clamp(0.0, 1.0) - 0.10 * surface.clamp_min(0.0)
        over_score = torch.where(over_score > 0.0, over_score * (0.35 + 0.45 * opacity_score + 0.20 * scale_score), fallback_over)
        under_score = torch.where(under_score > 0.0, under_score * (0.35 + 0.45 * opacity_score + 0.20 * scale_score), fallback_under)
        over_mask = self._topk_bool_mask(over_mask, over_score, max_over_points)
        under_mask = self._topk_bool_mask(under_mask, under_score, max_under_points)
        overlap = over_mask & under_mask
        if bool(overlap.any().item()):
            over_wins = over_score >= under_score
            over_mask = over_mask & (~overlap | over_wins)
            under_mask = under_mask & (~overlap | ~over_wins)
        return over_mask, under_mask, over_normals, under_normals, over_score, under_score

    def _score_weighted_normal_factor(
        self,
        mask,
        score,
        base_factor=1.0,
        power=1.0,
        min_weight=0.0,
        quantile=0.90,
    ):
        if torch.is_tensor(base_factor):
            return base_factor
        base_factor = float(base_factor or 1.0)
        point_count = int(self.get_xyz.shape[0])
        device = self.get_xyz.device
        dtype = self.get_xyz.dtype
        factor = torch.ones((point_count,), dtype=dtype, device=device)
        if (
            point_count <= 0
            or np.isclose(base_factor, 1.0)
            or not torch.is_tensor(mask)
            or mask.shape[0] != point_count
            or not bool(mask.any().item())
        ):
            return factor
        if not torch.is_tensor(score) or score.shape[0] != point_count:
            factor[mask.bool()] = base_factor
            return factor

        active = mask.bool()
        active_score = score.to(device=device, dtype=dtype).reshape(-1)[active].clamp_min(0.0)
        if active_score.numel() == 0:
            return factor
        if bool((active_score > 0).any().item()):
            quantile = float(min(max(quantile, 0.10), 1.0))
            denom = torch.quantile(active_score.detach(), quantile).clamp_min(1.0e-6)
            weight = (active_score / denom).clamp(0.0, 1.0)
        else:
            weight = torch.ones_like(active_score)
        power = max(float(power or 1.0), 1.0e-6)
        if not np.isclose(power, 1.0):
            weight = weight.pow(power)
        min_weight = float(min(max(min_weight, 0.0), 1.0))
        if min_weight > 0.0:
            weight = min_weight + (1.0 - min_weight) * weight
        factor[active] = 1.0 + (base_factor - 1.0) * weight
        return factor

    def _covariance_matrix_for_mode(
        self,
        scaling,
        scaling_modifier,
        rotation,
        mode,
        polar_det_min=0.0,
        polar_det_max=0.0,
        polar_det_power=1.0,
        polar_anisotropy_clamp=1.25,
    ):
        mode = str(mode or "default").lower()
        if mode in ("default", "anisotropic", "aniso"):
            return self._covariance_matrix_from_scaling_rotation(scaling, scaling_modifier, rotation)
        if mode in ("orthogonalized", "orth", "orth_rotation"):
            return self._covariance_matrix_from_scaling_rotation(
                scaling,
                scaling_modifier,
                self._orthogonalize_covariance_rotation(rotation),
            )
        if mode in ("canonical_rotation", "canonical", "raw_quaternion"):
            return self._covariance_matrix_from_scaling_rotation(scaling, scaling_modifier, self._rotation)
        if mode.startswith("rotation_isotropic_") or mode.startswith("rot_isotropic_"):
            reduce_mode = mode.rsplit("_", 1)[-1]
            scalar = self._reduce_scaling_to_scalar(scaling, reduce_mode).repeat(1, 3)
            return self._covariance_matrix_from_scaling_rotation(scalar, scaling_modifier, rotation)
        if mode.startswith("world_isotropic_") or mode.startswith("iso_"):
            reduce_mode = mode.rsplit("_", 1)[-1]
            scalar = scaling_modifier * self._reduce_scaling_to_scalar(scaling, reduce_mode).reshape(-1)
            cov = torch.zeros((scaling.shape[0], 3, 3), dtype=scaling.dtype, device=scaling.device)
            cov[:, 0, 0] = scalar * scalar
            cov[:, 1, 1] = scalar * scalar
            cov[:, 2, 2] = scalar * scalar
            return cov
        if mode in ("polar_det", "polar_volume", "binding_stable", "binding_stable_det"):
            return self._covariance_matrix_from_polar_stabilized_transform(
                scaling,
                scaling_modifier,
                rotation,
                mode="polar_det",
                det_min=polar_det_min,
                det_max=polar_det_max,
                det_power=polar_det_power,
                anisotropy_clamp=polar_anisotropy_clamp,
            )
        if mode in ("polar_svd_clamp", "svd_clamp", "polar_aniso_clamp"):
            return self._covariance_matrix_from_polar_stabilized_transform(
                scaling,
                scaling_modifier,
                rotation,
                mode="polar_svd_clamp",
                det_min=polar_det_min,
                det_max=polar_det_max,
                det_power=polar_det_power,
                anisotropy_clamp=polar_anisotropy_clamp,
            )
        return self._covariance_matrix_from_scaling_rotation(scaling, scaling_modifier, rotation)

    def _binding_covariance_guard_gate(
        self,
        point_count,
        device,
        boundary_min=0.08,
        layer_ids="soft,free",
        region_ids="cloth,soft",
        joint_ids="",
        thin_min=None,
        surface_min=None,
        surface_max=None,
        power=1.0,
        max_points=-1,
    ):
        dtype = self.get_xyz.dtype
        if point_count <= 0:
            return (
                torch.zeros((point_count,), dtype=torch.bool, device=device),
                torch.zeros((point_count,), dtype=dtype, device=device),
            )

        layer_names = {"rigid": 0, "soft": 1, "free": 2}
        region_names = {"body": 0, "soft": 1, "cloth": 2}
        layer = self._long_attr(getattr(self, "binding_layer_ids", None), point_count, device, default=0)
        region = self._long_attr(getattr(self, "binding_region_ids", None), point_count, device, default=0)
        joint = self._long_attr(getattr(self, "binding_dominant_joint", None), point_count, device, default=-1)
        boundary = self._scalar_attr(getattr(self, "binding_boundary_score", None), point_count, device, default=0.0).to(dtype=dtype)
        thin = self._scalar_attr(getattr(self, "binding_thin_score", None), point_count, device, default=0.0).to(dtype=dtype)
        surface = self._scalar_attr(getattr(self, "binding_surface_distance", None), point_count, device, default=0.0).to(dtype=dtype)

        mask = boundary >= float(boundary_min or 0.0)
        parsed_layers = self._parse_covariance_id_list(layer_ids, name_to_id=layer_names)
        if parsed_layers:
            mask = mask & self._values_to_mask(layer, parsed_layers)
        parsed_regions = self._parse_covariance_id_list(region_ids, name_to_id=region_names)
        if parsed_regions:
            mask = mask & self._values_to_mask(region, parsed_regions)
        parsed_joints = self._parse_covariance_id_list(joint_ids)
        if parsed_joints:
            mask = mask & self._values_to_mask(joint, parsed_joints)

        if thin_min is not None and str(thin_min) != "":
            mask = mask & (thin >= float(thin_min))
        if surface_min is not None and str(surface_min) != "":
            mask = mask & (surface >= float(surface_min))
        if surface_max is not None and str(surface_max) != "":
            mask = mask & (surface <= float(surface_max))

        score = boundary.clamp(0.0, 1.0)
        if bool(mask.any().item()):
            active_score = score[mask]
            denom = torch.quantile(active_score.detach(), 0.90).clamp_min(1.0e-6)
            weight = (score / denom).clamp(0.0, 1.0)
        else:
            weight = torch.zeros_like(score)
        power = max(float(power or 1.0), 1.0e-6)
        if not np.isclose(power, 1.0):
            weight = weight.pow(power)
        mask = self._topk_bool_mask(mask, score, max_points)
        weight = torch.where(mask, weight, torch.zeros_like(weight))
        return mask, weight

    def _apply_binding_covariance_guard(
        self,
        covariance,
        scaling,
        scaling_modifier,
        rotation,
        mode="canonical_blend",
        strength=0.5,
        boundary_min=0.08,
        layer_ids="soft,free",
        region_ids="cloth,soft",
        joint_ids="",
        thin_min=None,
        surface_min=None,
        surface_max=None,
        power=1.0,
        max_points=-1,
        anisotropy_clamp=1.25,
    ):
        if covariance.numel() == 0:
            return covariance
        point_count = int(covariance.shape[0])
        mask, weight = self._binding_covariance_guard_gate(
            point_count,
            covariance.device,
            boundary_min=boundary_min,
            layer_ids=layer_ids,
            region_ids=region_ids,
            joint_ids=joint_ids,
            thin_min=thin_min,
            surface_min=surface_min,
            surface_max=surface_max,
            power=power,
            max_points=max_points,
        )
        if not bool(mask.any().item()):
            return covariance

        guard_mode = str(mode or "canonical_blend").lower()
        if guard_mode in ("canonical", "canonical_blend", "canonical_rotation"):
            target_cov = self._covariance_matrix_for_mode(scaling, scaling_modifier, rotation, "canonical_rotation")
        elif guard_mode in ("orth", "orthogonalized", "orthogonalized_blend"):
            target_cov = self._covariance_matrix_for_mode(scaling, scaling_modifier, rotation, "orthogonalized")
        elif guard_mode in ("rot_iso", "rotation_isotropic", "rotation_isotropic_geom"):
            target_cov = self._covariance_matrix_for_mode(scaling, scaling_modifier, rotation, "rotation_isotropic_geom")
        elif guard_mode in ("world_iso", "world_isotropic", "world_isotropic_geom"):
            target_cov = self._covariance_matrix_for_mode(scaling, scaling_modifier, rotation, "world_isotropic_geom")
        elif guard_mode in ("aniso_clamp", "anisotropy_clamp", "clamp"):
            guarded_scaling = self._apply_anisotropy_clamp(scaling, anisotropy_clamp)
            target_cov = self._covariance_matrix_for_mode(guarded_scaling, scaling_modifier, rotation, "default")
        elif guard_mode in ("polar_det", "polar_volume", "binding_stable", "binding_stable_det"):
            target_cov = self._covariance_matrix_for_mode(scaling, scaling_modifier, rotation, "polar_det")
        elif guard_mode in ("polar_svd_clamp", "svd_clamp", "polar_aniso_clamp"):
            target_cov = self._covariance_matrix_for_mode(
                scaling,
                scaling_modifier,
                rotation,
                "polar_svd_clamp",
                polar_anisotropy_clamp=anisotropy_clamp,
            )
        else:
            return covariance

        blend = (weight * float(strength or 0.0)).clamp(0.0, 1.0).reshape(-1, 1, 1)
        edited = covariance.clone()
        edited[mask] = covariance[mask] * (1.0 - blend[mask]) + target_cov[mask] * blend[mask]
        eye = torch.eye(3, dtype=edited.dtype, device=edited.device).reshape(1, 3, 3)
        edited[mask] = 0.5 * (edited[mask] + edited[mask].transpose(1, 2)) + eye * 1.0e-10
        return edited

    def _signed_dynamic_guard_target_covariance(
        self,
        scaling,
        scaling_modifier,
        rotation,
        mode,
        anisotropy_clamp=1.20,
    ):
        guard_mode = str(mode or "none").lower()
        if guard_mode in ("", "none", "off", "disabled", "identity"):
            return None
        if guard_mode in ("canonical", "canonical_blend", "canonical_rotation"):
            return self._covariance_matrix_for_mode(scaling, scaling_modifier, rotation, "canonical_rotation")
        if guard_mode in ("orth", "orthogonalized", "orthogonalized_blend"):
            return self._covariance_matrix_for_mode(scaling, scaling_modifier, rotation, "orthogonalized")
        if guard_mode in ("rot_iso", "rotation_isotropic", "rotation_isotropic_geom"):
            return self._covariance_matrix_for_mode(scaling, scaling_modifier, rotation, "rotation_isotropic_geom")
        if guard_mode in ("world_iso", "world_isotropic", "world_isotropic_geom"):
            return self._covariance_matrix_for_mode(scaling, scaling_modifier, rotation, "world_isotropic_geom")
        if guard_mode in ("aniso_clamp", "anisotropy_clamp", "clamp"):
            guarded_scaling = self._apply_anisotropy_clamp(scaling, anisotropy_clamp)
            return self._covariance_matrix_for_mode(guarded_scaling, scaling_modifier, rotation, "default")
        if guard_mode in ("polar_det", "polar_volume", "binding_stable", "binding_stable_det"):
            return self._covariance_matrix_for_mode(scaling, scaling_modifier, rotation, "polar_det")
        if guard_mode in ("polar_svd_clamp", "svd_clamp", "polar_aniso_clamp"):
            return self._covariance_matrix_for_mode(
                scaling,
                scaling_modifier,
                rotation,
                "polar_svd_clamp",
                polar_anisotropy_clamp=anisotropy_clamp,
            )
        return None

    def _signed_dynamic_guard_weight(
        self,
        mask,
        score,
        strength=0.0,
        power=1.0,
        quantile=0.90,
        min_weight=0.0,
    ):
        if not torch.is_tensor(mask) or mask.numel() == 0:
            return None
        active = mask.bool()
        if not bool(active.any().item()):
            return None
        dtype = self.get_xyz.dtype
        device = self.get_xyz.device
        weight = torch.zeros((mask.shape[0],), dtype=dtype, device=device)
        if torch.is_tensor(score) and score.shape[0] == mask.shape[0]:
            active_score = score.to(device=device, dtype=dtype).reshape(-1)[active].clamp_min(0.0)
            if bool((active_score > 0.0).any().item()):
                quantile = float(min(max(quantile, 0.10), 1.0))
                denom = torch.quantile(active_score.detach(), quantile).clamp_min(1.0e-6)
                local_weight = (active_score / denom).clamp(0.0, 1.0)
            else:
                local_weight = torch.ones_like(active_score)
        else:
            local_weight = torch.ones((int(active.sum().item()),), dtype=dtype, device=device)
        power = max(float(power or 1.0), 1.0e-6)
        if not np.isclose(power, 1.0):
            local_weight = local_weight.pow(power)
        min_weight = float(min(max(min_weight, 0.0), 1.0))
        if min_weight > 0.0:
            local_weight = min_weight + (1.0 - min_weight) * local_weight
        weight[active] = local_weight * float(strength or 0.0)
        return weight.clamp(0.0, 1.0)

    def _apply_signed_dynamic_covariance_guard(
        self,
        covariance,
        scaling,
        scaling_modifier,
        rotation,
        shrink_mask,
        grow_mask,
        shrink_score=None,
        grow_score=None,
        shrink_mode="aniso_clamp",
        grow_mode="canonical_blend",
        shrink_strength=0.0,
        grow_strength=0.0,
        power=1.0,
        quantile=0.90,
        min_weight=0.0,
        anisotropy_clamp=1.20,
    ):
        if covariance.numel() == 0:
            return covariance
        edited = covariance
        eye = None

        def _apply(mask, score, mode, strength):
            nonlocal edited, eye
            strength = float(strength or 0.0)
            if strength <= 0.0 or not torch.is_tensor(mask) or not bool(mask.any().item()):
                return
            target = self._signed_dynamic_guard_target_covariance(
                scaling,
                scaling_modifier,
                rotation,
                mode,
                anisotropy_clamp=anisotropy_clamp,
            )
            if target is None:
                return
            weight = self._signed_dynamic_guard_weight(
                mask,
                score,
                strength=strength,
                power=power,
                quantile=quantile,
                min_weight=min_weight,
            )
            if weight is None:
                return
            if edited is covariance:
                edited = covariance.clone()
            if eye is None:
                eye = torch.eye(3, dtype=edited.dtype, device=edited.device).reshape(1, 3, 3)
            blend = weight.reshape(-1, 1, 1)
            active = mask.bool()
            edited[active] = covariance[active] * (1.0 - blend[active]) + target[active] * blend[active]
            edited[active] = 0.5 * (edited[active] + edited[active].transpose(1, 2)) + eye * 1.0e-10

        _apply(shrink_mask, shrink_score, shrink_mode, shrink_strength)
        _apply(grow_mask, grow_score, grow_mode, grow_strength)
        return edited

    def _signed_dynamic_offset_weight(
        self,
        mask,
        score,
        base_px=0.0,
        power=1.0,
        quantile=0.90,
        min_weight=0.0,
    ):
        point_count = int(self.get_xyz.shape[0])
        dtype = self.get_xyz.dtype
        device = self.get_xyz.device
        weight = torch.zeros((point_count,), dtype=dtype, device=device)
        base_px = float(base_px or 0.0)
        if (
            base_px <= 0.0
            or not torch.is_tensor(mask)
            or mask.shape[0] != point_count
            or not bool(mask.any().item())
        ):
            return weight
        active = mask.bool()
        if torch.is_tensor(score) and score.shape[0] == point_count:
            active_score = score.to(device=device, dtype=dtype).reshape(-1)[active].clamp_min(0.0)
            if bool((active_score > 0.0).any().item()):
                quantile = float(min(max(quantile, 0.10), 1.0))
                denom = torch.quantile(active_score.detach(), quantile).clamp_min(1.0e-6)
                local_weight = (active_score / denom).clamp(0.0, 1.0)
            else:
                local_weight = torch.ones_like(active_score)
        else:
            local_weight = torch.ones((int(active.sum().item()),), dtype=dtype, device=device)
        power = max(float(power or 1.0), 1.0e-6)
        if not np.isclose(power, 1.0):
            local_weight = local_weight.pow(power)
        min_weight = float(min(max(min_weight, 0.0), 1.0))
        if min_weight > 0.0:
            local_weight = min_weight + (1.0 - min_weight) * local_weight
        weight[active] = local_weight * base_px
        return weight

    def _screen_world_delta_from_pixel_shift(
        self,
        points,
        camera,
        shift_px,
        eps=1.0e-3,
        damping=1.0e-5,
        max_world_step=0.003,
    ):
        if (
            camera is None
            or not torch.is_tensor(points)
            or points.numel() == 0
            or not torch.is_tensor(shift_px)
            or shift_px.numel() == 0
        ):
            return torch.zeros_like(points)
        eps = float(eps or 1.0e-3)
        if eps <= 0.0:
            eps = 1.0e-3
        base_xy, base_valid = self._project_covariance_points(points.detach(), camera)
        if base_xy is None or not torch.is_tensor(base_valid):
            return torch.zeros_like(points)
        point_count = int(points.shape[0])
        basis = torch.eye(3, dtype=points.dtype, device=points.device).reshape(1, 3, 3)
        shifted_points = points.detach().reshape(point_count, 1, 3) + basis * eps
        shifted_xy, shifted_valid = self._project_covariance_points(shifted_points.reshape(-1, 3), camera)
        if shifted_xy is None or not torch.is_tensor(shifted_valid):
            return torch.zeros_like(points)
        shifted_xy = shifted_xy.reshape(point_count, 3, 2)
        shifted_valid = shifted_valid.reshape(point_count, 3)
        valid = base_valid.bool() & shifted_valid.bool().all(dim=1)
        jac = ((shifted_xy - base_xy.reshape(point_count, 1, 2)) / eps).permute(0, 2, 1).float()
        jj_t = torch.bmm(jac, jac.transpose(1, 2))
        eye = torch.eye(2, device=points.device, dtype=torch.float32).reshape(1, 2, 2)
        try:
            inv = torch.linalg.inv(jj_t + float(damping or 0.0) * eye)
            rhs = shift_px.to(device=points.device, dtype=torch.float32).reshape(point_count, 2, 1)
            world = torch.bmm(jac.transpose(1, 2), torch.bmm(inv, rhs)).squeeze(-1)
        except RuntimeError:
            return torch.zeros_like(points)
        world = torch.nan_to_num(world.to(dtype=points.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        world = torch.where(valid.reshape(-1, 1), world, torch.zeros_like(world))
        max_world_step = float(max_world_step or 0.0)
        if max_world_step > 0.0:
            norm = torch.norm(world, dim=-1, keepdim=True).clamp_min(1.0e-12)
            world = world * torch.clamp(max_world_step / norm, max=1.0)
        return world

    def get_signed_center_offset(
        self,
        camera=None,
        signed_point_json="",
        signed_dynamic_enable=False,
        signed_dynamic_component_csv="",
        signed_dynamic_point_csv="",
        signed_dynamic_component_signature_enable=False,
        signed_dynamic_over_layer_ids="soft,free",
        signed_dynamic_over_region_ids="cloth",
        signed_dynamic_over_joint_ids="6,9,12,14,15",
        signed_dynamic_under_layer_ids="soft",
        signed_dynamic_under_region_ids="body,cloth,soft",
        signed_dynamic_under_joint_ids="4,7,8",
        signed_dynamic_boundary_min=0.0,
        signed_dynamic_surface_min=None,
        signed_dynamic_surface_max=None,
        signed_dynamic_component_pad_px=10.0,
        signed_dynamic_component_ellipse_scale=1.25,
        signed_dynamic_component_max_over=16,
        signed_dynamic_component_max_under=16,
        signed_dynamic_component_min_area=1.0,
        signed_dynamic_component_required=False,
        signed_dynamic_component_top_ids_enable=False,
        signed_dynamic_component_top_ids_only=False,
        signed_dynamic_max_over_points=-1,
        signed_dynamic_max_under_points=-1,
        signed_max_shrink_points=-1,
        signed_max_grow_points=-1,
        outer_offset_px=0.0,
        inner_offset_px=0.0,
        outer_direction="view_center",
        inner_direction="component_center",
        score_weight_power=1.0,
        score_weight_min=0.0,
        score_weight_quantile=0.90,
        jacobian_eps=1.0e-3,
        jacobian_damping=1.0e-5,
        max_world_step=0.003,
    ):
        point_count = int(self.get_xyz.shape[0])
        offset = torch.zeros_like(self.get_xyz)
        if camera is None or point_count <= 0:
            return offset
        outer_offset_px = float(outer_offset_px or 0.0)
        inner_offset_px = float(inner_offset_px or 0.0)
        if outer_offset_px <= 0.0 and inner_offset_px <= 0.0:
            return offset

        if bool(signed_dynamic_enable):
            (
                shrink_mask,
                grow_mask,
                shrink_normals,
                grow_normals,
                shrink_score,
                grow_score,
            ) = self._dynamic_signed_covariance_masks(
                camera=camera,
                component_csv=signed_dynamic_component_csv,
                point_csv=signed_dynamic_point_csv,
                component_signature_enable=signed_dynamic_component_signature_enable,
                over_layer_ids=signed_dynamic_over_layer_ids,
                over_region_ids=signed_dynamic_over_region_ids,
                over_joint_ids=signed_dynamic_over_joint_ids,
                under_layer_ids=signed_dynamic_under_layer_ids,
                under_region_ids=signed_dynamic_under_region_ids,
                under_joint_ids=signed_dynamic_under_joint_ids,
                boundary_min=signed_dynamic_boundary_min,
                surface_min=signed_dynamic_surface_min,
                surface_max=signed_dynamic_surface_max,
                component_pad_px=signed_dynamic_component_pad_px,
                component_ellipse_scale=signed_dynamic_component_ellipse_scale,
                component_max_over=signed_dynamic_component_max_over,
                component_max_under=signed_dynamic_component_max_under,
                component_min_area=signed_dynamic_component_min_area,
                component_required=signed_dynamic_component_required,
                component_top_ids_enable=signed_dynamic_component_top_ids_enable,
                component_top_ids_only=signed_dynamic_component_top_ids_only,
                max_over_points=signed_dynamic_max_over_points,
                max_under_points=signed_dynamic_max_under_points,
            )
        else:
            shrink_mask = torch.zeros((point_count,), dtype=torch.bool, device=self.get_xyz.device)
            grow_mask = torch.zeros_like(shrink_mask)
            shrink_normals = torch.zeros((point_count, 2), dtype=self.get_xyz.dtype, device=self.get_xyz.device)
            grow_normals = torch.zeros_like(shrink_normals)
            shrink_score = torch.zeros((point_count,), dtype=self.get_xyz.dtype, device=self.get_xyz.device)
            grow_score = torch.zeros_like(shrink_score)
        if signed_point_json:
            shrink_ids, grow_ids = self._load_covariance_signed_point_ids(signed_point_json, camera=camera)
            max_shrink = int(signed_max_shrink_points)
            max_grow = int(signed_max_grow_points)
            if max_shrink >= 0:
                shrink_ids = shrink_ids[:max_shrink]
            if max_grow >= 0:
                grow_ids = grow_ids[:max_grow]
            point_shrink = self._ids_to_mask(shrink_ids, point_count, self.get_xyz.device)
            point_grow = self._ids_to_mask(grow_ids, point_count, self.get_xyz.device)
            if torch.is_tensor(point_shrink) and bool(point_shrink.any().item()):
                shrink_mask = shrink_mask | point_shrink
                shrink_score = torch.maximum(
                    shrink_score.to(device=self.get_xyz.device, dtype=self.get_xyz.dtype),
                    point_shrink.to(device=self.get_xyz.device, dtype=self.get_xyz.dtype),
                )
            if torch.is_tensor(point_grow) and bool(point_grow.any().item()):
                grow_mask = grow_mask | point_grow
                grow_score = torch.maximum(
                    grow_score.to(device=self.get_xyz.device, dtype=self.get_xyz.dtype),
                    point_grow.to(device=self.get_xyz.device, dtype=self.get_xyz.dtype),
                )
        xy, valid = self._project_covariance_points(self.get_xyz.detach(), camera)
        if xy is None or not torch.is_tensor(valid):
            return offset
        if bool(valid.any().item()):
            view_center = xy[valid.bool()].mean(dim=0)
        else:
            view_center = xy.mean(dim=0)

        def _direction(mask, normals, mode):
            mode = str(mode or "component_center").lower()
            if mode in ("view_center", "center", "subject_center"):
                direction = xy - view_center.reshape(1, 2)
            else:
                direction = normals.to(device=xy.device, dtype=xy.dtype) if torch.is_tensor(normals) else torch.zeros_like(xy)
            weak = torch.linalg.norm(direction, dim=-1) < 1.0e-6
            if bool(weak.any().item()):
                direction = direction.clone()
                direction[weak, 0] = xy[weak, 0] - view_center[0]
                direction[weak, 1] = xy[weak, 1] - view_center[1]
            weak = torch.linalg.norm(direction, dim=-1) < 1.0e-6
            if bool(weak.any().item()):
                direction = direction.clone()
                direction[weak, 0] = 1.0
                direction[weak, 1] = 0.0
            return F.normalize(direction, dim=-1, eps=1.0e-6)

        screen_shift = torch.zeros((point_count, 2), dtype=self.get_xyz.dtype, device=self.get_xyz.device)
        if outer_offset_px > 0.0 and torch.is_tensor(shrink_mask) and bool(shrink_mask.any().item()):
            weight = self._signed_dynamic_offset_weight(
                shrink_mask,
                shrink_score,
                base_px=outer_offset_px,
                power=score_weight_power,
                quantile=score_weight_quantile,
                min_weight=score_weight_min,
            )
            inward = -_direction(shrink_mask, shrink_normals, outer_direction)
            screen_shift = screen_shift + inward * weight.reshape(-1, 1)
        if inner_offset_px > 0.0 and torch.is_tensor(grow_mask) and bool(grow_mask.any().item()):
            weight = self._signed_dynamic_offset_weight(
                grow_mask,
                grow_score,
                base_px=inner_offset_px,
                power=score_weight_power,
                quantile=score_weight_quantile,
                min_weight=score_weight_min,
            )
            toward_gap = -_direction(grow_mask, grow_normals, inner_direction)
            screen_shift = screen_shift + toward_gap * weight.reshape(-1, 1)
        active = (torch.linalg.norm(screen_shift, dim=-1) > 1.0e-7) & valid.bool()
        if not bool(active.any().item()):
            return offset
        active_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
        active_world = self._screen_world_delta_from_pixel_shift(
            self.get_xyz.detach()[active_idx],
            camera,
            screen_shift[active_idx],
            eps=jacobian_eps,
            damping=jacobian_damping,
            max_world_step=max_world_step,
        )
        offset[active_idx] = active_world.to(device=offset.device, dtype=offset.dtype)
        return offset.detach()

    @staticmethod
    def _covariance_matrix_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(float(scaling_modifier) * scaling, rotation)
        return L @ L.transpose(1, 2)

    @staticmethod
    def _covariance_matrix_from_linear_scaling(linear, scaling, scaling_modifier):
        L = linear * (float(scaling_modifier) * scaling).reshape(-1, 1, 3)
        return L @ L.transpose(1, 2)

    @staticmethod
    def _polar_decompose_linear_transform(linear):
        if not torch.is_tensor(linear) or linear.numel() == 0 or linear.shape[-2:] != (3, 3):
            return linear, None, None, None
        try:
            u, singular, vh = torch.linalg.svd(linear)
            orth = torch.matmul(u, vh)
            det = torch.det(orth)
            if torch.is_tensor(det):
                bad = det < 0
                if bool(bad.any()):
                    u = u.clone()
                    u[bad, :, -1] *= -1.0
                    orth = torch.matmul(u, vh)
            return orth, singular.clamp_min(1.0e-6), u, vh
        except Exception:
            return GaussianModel._orthogonalize_covariance_rotation(linear), None, None, None

    @staticmethod
    def _covariance_matrix_from_polar_stabilized_transform(
        scaling,
        scaling_modifier,
        linear,
        mode="polar_det",
        det_min=0.0,
        det_max=0.0,
        det_power=1.0,
        anisotropy_clamp=1.25,
    ):
        orth, singular, u, vh = GaussianModel._polar_decompose_linear_transform(linear)
        if singular is None:
            return GaussianModel._covariance_matrix_from_scaling_rotation(
                scaling,
                scaling_modifier,
                orth,
            )

        eps = 1.0e-6
        singular = singular.clamp_min(eps)
        geom = torch.exp(torch.log(singular).mean(dim=-1, keepdim=True))
        det_power = float(det_power or 1.0)
        if not np.isclose(det_power, 1.0):
            geom = geom.pow(det_power)
        det_min = float(det_min or 0.0)
        det_max = float(det_max or 0.0)
        if det_min > 0.0:
            geom = geom.clamp_min(det_min)
        if det_max > 0.0:
            geom = geom.clamp_max(det_max)

        mode = str(mode or "polar_det").lower()
        if mode in ("polar_svd_clamp", "svd_clamp", "polar_aniso_clamp"):
            clamp_ratio = float(anisotropy_clamp or 0.0)
            if clamp_ratio > 1.0:
                log_ratio = torch.log(singular / geom.clamp_min(eps))
                log_limit = float(np.log(clamp_ratio))
                stable_singular = geom * torch.exp(torch.clamp(log_ratio, -log_limit, log_limit))
            else:
                stable_singular = singular
            stable_linear = torch.matmul(u * stable_singular.reshape(-1, 1, 3), vh)
        else:
            stable_linear = orth * geom.reshape(-1, 1, 1)

        return GaussianModel._covariance_matrix_from_linear_scaling(
            stable_linear,
            scaling,
            scaling_modifier,
        )

    @staticmethod
    def _camera_screen_world_axes(camera, device, dtype):
        view = getattr(camera, "world_view_transform", None) if camera is not None else None
        if not torch.is_tensor(view) or view.shape[0] < 3 or view.shape[1] < 3:
            right = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
            up = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
            return right, up
        try:
            view_inv = torch.inverse(view.to(device=device, dtype=dtype))
            right = view_inv[0, :3]
            up = view_inv[1, :3]
        except RuntimeError:
            linear = view[:3, :3].to(device=device, dtype=dtype)
            right = linear[:, 0]
            up = linear[:, 1]
        right = F.normalize(right.reshape(1, 3), dim=-1, eps=1.0e-6).reshape(3)
        up = F.normalize(up.reshape(1, 3), dim=-1, eps=1.0e-6).reshape(3)
        return right, up

    def _apply_screen_space_covariance_actuator(
        self,
        covariance,
        camera,
        shrink_mask,
        grow_mask,
        shrink_normals=None,
        grow_normals=None,
        normal_shrink_factor=1.0,
        normal_grow_factor=1.0,
        tangent_factor=1.0,
    ):
        if camera is None or covariance.numel() == 0:
            return covariance
        if torch.is_tensor(normal_shrink_factor):
            normal_shrink_factor = normal_shrink_factor.to(device=covariance.device, dtype=covariance.dtype)
            shrink_changed = bool((torch.abs(normal_shrink_factor - 1.0) > 1.0e-7).any().item())
        else:
            normal_shrink_factor = float(normal_shrink_factor or 1.0)
            shrink_changed = not np.isclose(normal_shrink_factor, 1.0)
        if torch.is_tensor(normal_grow_factor):
            normal_grow_factor = normal_grow_factor.to(device=covariance.device, dtype=covariance.dtype)
            grow_changed = bool((torch.abs(normal_grow_factor - 1.0) > 1.0e-7).any().item())
        else:
            normal_grow_factor = float(normal_grow_factor or 1.0)
            grow_changed = not np.isclose(normal_grow_factor, 1.0)
        tangent_factor = float(tangent_factor or 1.0)
        if (
            not shrink_changed
            and not grow_changed
            and np.isclose(tangent_factor, 1.0)
        ):
            return covariance
        edited = covariance.clone()
        right, up = self._camera_screen_world_axes(camera, edited.device, edited.dtype)

        def _directions(mask, normals_2d):
            if not torch.is_tensor(mask) or not bool(mask.any().item()):
                return None
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            xy = None
            valid = None
            if torch.is_tensor(normals_2d) and normals_2d.shape[0] == edited.shape[0]:
                n2 = normals_2d.to(device=edited.device, dtype=edited.dtype)[idx]
            else:
                xy, valid = self._project_covariance_points(self.get_xyz, camera)
                if xy is None:
                    n2 = torch.zeros((idx.numel(), 2), dtype=edited.dtype, device=edited.device)
                    n2[:, 0] = 1.0
                else:
                    center = xy[valid].mean(dim=0) if bool(valid.any().item()) else xy.mean(dim=0)
                    n2 = xy[idx] - center.reshape(1, 2)
            weak = torch.linalg.norm(n2, dim=-1) < 1.0e-6
            if bool(weak.any().item()):
                n2 = n2.clone()
                if xy is None:
                    xy, valid = self._project_covariance_points(self.get_xyz, camera)
                if xy is not None:
                    center = xy[valid].mean(dim=0) if torch.is_tensor(valid) and bool(valid.any().item()) else xy.mean(dim=0)
                    fallback = xy[idx] - center.reshape(1, 2)
                    n2[weak] = fallback[weak]
                still_weak = torch.linalg.norm(n2, dim=-1) < 1.0e-6
                if bool(still_weak.any().item()):
                    n2[still_weak, 0] = 1.0
                    n2[still_weak, 1] = 0.0
            n2 = F.normalize(n2, dim=-1, eps=1.0e-6)
            normal = F.normalize(n2[:, 0:1] * right.reshape(1, 3) - n2[:, 1:2] * up.reshape(1, 3), dim=-1, eps=1.0e-6)
            tangent = F.normalize(-n2[:, 1:2] * right.reshape(1, 3) - n2[:, 0:1] * up.reshape(1, 3), dim=-1, eps=1.0e-6)
            return idx, normal, tangent

        def _apply(mask, normals_2d, normal_factor):
            dirs = _directions(mask, normals_2d)
            if dirs is None:
                return
            idx, normal, tangent = dirs
            cov = edited[idx]
            eye = torch.eye(3, dtype=cov.dtype, device=cov.device).reshape(1, 3, 3)
            if torch.is_tensor(normal_factor):
                factors = normal_factor[idx].reshape(-1, 1, 1).to(device=cov.device, dtype=cov.dtype)
                normal_op = normal.unsqueeze(-1) * normal.unsqueeze(-2)
                transform = eye + (factors - 1.0) * normal_op
                cov = transform @ cov @ transform.transpose(1, 2)
            elif not np.isclose(normal_factor, 1.0):
                normal_op = normal.unsqueeze(-1) * normal.unsqueeze(-2)
                transform = eye + (normal_factor - 1.0) * normal_op
                cov = transform @ cov @ transform.transpose(1, 2)
            if not np.isclose(tangent_factor, 1.0):
                tangent_op = tangent.unsqueeze(-1) * tangent.unsqueeze(-2)
                transform = eye + (tangent_factor - 1.0) * tangent_op
                cov = transform @ cov @ transform.transpose(1, 2)
            cov = 0.5 * (cov + cov.transpose(1, 2)) + eye * 1.0e-10
            edited[idx] = cov

        _apply(shrink_mask, shrink_normals, normal_shrink_factor)
        _apply(grow_mask, grow_normals, normal_grow_factor)
        return edited

    def _boundary_covariance_normal_factors(
        self,
        shrink_mask,
        grow_mask,
        normal_shrink_factor=1.0,
        normal_grow_factor=1.0,
        enable=False,
        max_abs=0.12,
    ):
        if (
            not bool(enable)
            or not torch.is_tensor(self._boundary_cov_residual)
            or self._boundary_cov_residual.numel() == 0
        ):
            return normal_shrink_factor, normal_grow_factor
        point_count = int(self.get_xyz.shape[0])
        if self._boundary_cov_residual.shape[0] != point_count:
            return normal_shrink_factor, normal_grow_factor
        residual = self._boundary_cov_residual.to(device=self.get_xyz.device, dtype=self.get_xyz.dtype)
        if residual.ndim == 1:
            residual = residual.reshape(-1, 1)
        shrink_residual = residual[:, 0]
        grow_residual = residual[:, 1] if residual.shape[1] > 1 else residual[:, 0]
        max_abs = float(max_abs or 0.0)
        if max_abs > 0.0:
            shrink_residual = torch.tanh(shrink_residual) * max_abs
            grow_residual = torch.tanh(grow_residual) * max_abs

        def _factor(base_factor, mask, residual_values):
            if not torch.is_tensor(mask) or mask.shape[0] != point_count:
                return base_factor
            if torch.is_tensor(base_factor):
                factor = base_factor.to(device=residual_values.device, dtype=residual_values.dtype).reshape(-1).clone()
                if factor.shape[0] != point_count:
                    return base_factor
            else:
                factor = torch.full((point_count,), float(base_factor), dtype=residual_values.dtype, device=residual_values.device)
            active = mask.bool()
            if bool(active.any().item()):
                factor[active] = torch.clamp(factor[active] + residual_values[active], min=0.25, max=2.50)
            return factor

        shrink_factor = _factor(normal_shrink_factor, shrink_mask, shrink_residual)
        grow_factor = _factor(normal_grow_factor, grow_mask, grow_residual)
        return shrink_factor, grow_factor

    def get_covariance(
        self,
        scaling_modifier=1,
        mode="default",
        anisotropy_clamp=0.0,
        isotropic_reduce="geom",
        polar_det_min=0.0,
        polar_det_max=0.0,
        polar_det_power=1.0,
        polar_anisotropy_clamp=1.25,
        signed_point_json="",
        signed_shrink_factor=1.0,
        signed_grow_factor=1.0,
        signed_max_shrink_points=-1,
        signed_max_grow_points=-1,
        signed_anisotropic_axis="all",
        signed_point_screen_actuator_enable=False,
        signed_dynamic_enable=False,
        signed_dynamic_component_csv="",
        signed_dynamic_point_csv="",
        signed_dynamic_component_signature_enable=False,
        signed_dynamic_over_layer_ids="soft,free",
        signed_dynamic_over_region_ids="cloth",
        signed_dynamic_over_joint_ids="6,9,12,14,15",
        signed_dynamic_under_layer_ids="soft",
        signed_dynamic_under_region_ids="body,cloth,soft",
        signed_dynamic_under_joint_ids="4,7,8",
        signed_dynamic_boundary_min=0.0,
        signed_dynamic_surface_min=None,
        signed_dynamic_surface_max=None,
        signed_dynamic_component_pad_px=10.0,
        signed_dynamic_component_ellipse_scale=1.25,
        signed_dynamic_component_max_over=16,
        signed_dynamic_component_max_under=16,
        signed_dynamic_component_min_area=1.0,
        signed_dynamic_component_required=False,
        signed_dynamic_component_top_ids_enable=False,
        signed_dynamic_component_top_ids_only=False,
        signed_dynamic_score_weighting_enable=False,
        signed_dynamic_score_weight_power=1.0,
        signed_dynamic_score_weight_min=0.0,
        signed_dynamic_score_weight_quantile=0.90,
        signed_dynamic_max_over_points=-1,
        signed_dynamic_max_under_points=-1,
        signed_dynamic_guard_enable=False,
        signed_dynamic_guard_shrink_mode="aniso_clamp",
        signed_dynamic_guard_grow_mode="canonical_blend",
        signed_dynamic_guard_shrink_strength=0.0,
        signed_dynamic_guard_grow_strength=0.0,
        signed_dynamic_guard_power=1.0,
        signed_dynamic_guard_quantile=0.90,
        signed_dynamic_guard_min_weight=0.0,
        signed_dynamic_guard_anisotropy_clamp=1.20,
        signed_screen_actuator_enable=False,
        signed_screen_normal_shrink_factor=1.0,
        signed_screen_normal_grow_factor=1.0,
        signed_screen_tangent_factor=1.0,
        boundary_cov_residual_enable=False,
        boundary_cov_residual_max_abs=0.12,
        binding_covariance_guard_enable=False,
        binding_covariance_guard_mode="canonical_blend",
        binding_covariance_guard_strength=0.5,
        binding_covariance_guard_boundary_min=0.08,
        binding_covariance_guard_layer_ids="soft,free",
        binding_covariance_guard_region_ids="cloth,soft",
        binding_covariance_guard_joint_ids="",
        binding_covariance_guard_thin_min=None,
        binding_covariance_guard_surface_min=None,
        binding_covariance_guard_surface_max=None,
        binding_covariance_guard_power=1.0,
        binding_covariance_guard_max_points=-1,
        binding_covariance_guard_anisotropy_clamp=1.25,
        camera=None,
    ):
        mode = str(mode or "default").lower()
        scaling = self._apply_anisotropy_clamp(self.get_scaling, anisotropy_clamp)
        dynamic_shrink_mask = None
        dynamic_grow_mask = None
        shrink_normals_2d = None
        grow_normals_2d = None
        dynamic_shrink_score = None
        dynamic_grow_score = None
        point_shrink_mask = None
        point_grow_mask = None
        if bool(signed_point_screen_actuator_enable) and signed_point_json:
            shrink_ids, grow_ids = self._load_covariance_signed_point_ids(signed_point_json, camera=camera)
            max_shrink = int(signed_max_shrink_points)
            max_grow = int(signed_max_grow_points)
            if max_shrink >= 0:
                shrink_ids = shrink_ids[:max_shrink]
            if max_grow >= 0:
                grow_ids = grow_ids[:max_grow]
            point_shrink_mask = self._ids_to_mask(shrink_ids, int(self.get_xyz.shape[0]), self.get_xyz.device)
            point_grow_mask = self._ids_to_mask(grow_ids, int(self.get_xyz.shape[0]), self.get_xyz.device)

        if bool(signed_dynamic_enable):
            (
                dynamic_shrink_mask,
                dynamic_grow_mask,
                shrink_normals_2d,
                grow_normals_2d,
                dynamic_shrink_score,
                dynamic_grow_score,
            ) = self._dynamic_signed_covariance_masks(
                camera=camera,
                component_csv=signed_dynamic_component_csv,
                point_csv=signed_dynamic_point_csv,
                component_signature_enable=signed_dynamic_component_signature_enable,
                over_layer_ids=signed_dynamic_over_layer_ids,
                over_region_ids=signed_dynamic_over_region_ids,
                over_joint_ids=signed_dynamic_over_joint_ids,
                under_layer_ids=signed_dynamic_under_layer_ids,
                under_region_ids=signed_dynamic_under_region_ids,
                under_joint_ids=signed_dynamic_under_joint_ids,
                boundary_min=signed_dynamic_boundary_min,
                surface_min=signed_dynamic_surface_min,
                surface_max=signed_dynamic_surface_max,
                component_pad_px=signed_dynamic_component_pad_px,
                component_ellipse_scale=signed_dynamic_component_ellipse_scale,
                component_max_over=signed_dynamic_component_max_over,
                component_max_under=signed_dynamic_component_max_under,
                component_min_area=signed_dynamic_component_min_area,
                component_required=signed_dynamic_component_required,
                component_top_ids_enable=signed_dynamic_component_top_ids_enable,
                component_top_ids_only=signed_dynamic_component_top_ids_only,
                max_over_points=signed_dynamic_max_over_points,
                max_under_points=signed_dynamic_max_under_points,
            )
        if bool(signed_point_screen_actuator_enable):
            if point_shrink_mask is not None:
                dynamic_shrink_mask = point_shrink_mask if dynamic_shrink_mask is None else (dynamic_shrink_mask | point_shrink_mask)
                dynamic_shrink_score = point_shrink_mask.to(dtype=self.get_xyz.dtype, device=self.get_xyz.device)
            if point_grow_mask is not None:
                dynamic_grow_mask = point_grow_mask if dynamic_grow_mask is None else (dynamic_grow_mask | point_grow_mask)
                dynamic_grow_score = point_grow_mask.to(dtype=self.get_xyz.dtype, device=self.get_xyz.device)
        scaling = self._apply_signed_point_scaling(
            scaling,
            signed_point_json=signed_point_json,
            shrink_factor=signed_shrink_factor,
            grow_factor=signed_grow_factor,
            max_shrink_points=signed_max_shrink_points,
            max_grow_points=signed_max_grow_points,
            anisotropic_axis=signed_anisotropic_axis,
        )
        if bool(signed_dynamic_enable) and not bool(signed_screen_actuator_enable):
            scaling = self._apply_signed_masks_to_scaling(
                scaling,
                dynamic_shrink_mask,
                dynamic_grow_mask,
                shrink_factor=signed_shrink_factor,
                grow_factor=signed_grow_factor,
                anisotropic_axis=signed_anisotropic_axis,
            )
        rotation = self.rotation_precomp if hasattr(self, 'rotation_precomp') else self._rotation

        use_screen_actuator = bool(signed_screen_actuator_enable) and (
            bool(signed_dynamic_enable) or bool(signed_point_screen_actuator_enable)
        )
        if use_screen_actuator:
            if bool(signed_dynamic_score_weighting_enable):
                signed_screen_normal_shrink_factor = self._score_weighted_normal_factor(
                    dynamic_shrink_mask,
                    dynamic_shrink_score,
                    base_factor=signed_screen_normal_shrink_factor,
                    power=signed_dynamic_score_weight_power,
                    min_weight=signed_dynamic_score_weight_min,
                    quantile=signed_dynamic_score_weight_quantile,
                )
                signed_screen_normal_grow_factor = self._score_weighted_normal_factor(
                    dynamic_grow_mask,
                    dynamic_grow_score,
                    base_factor=signed_screen_normal_grow_factor,
                    power=signed_dynamic_score_weight_power,
                    min_weight=signed_dynamic_score_weight_min,
                    quantile=signed_dynamic_score_weight_quantile,
                )
            normal_shrink_factor, normal_grow_factor = self._boundary_covariance_normal_factors(
                dynamic_shrink_mask,
                dynamic_grow_mask,
                normal_shrink_factor=signed_screen_normal_shrink_factor,
                normal_grow_factor=signed_screen_normal_grow_factor,
                enable=boundary_cov_residual_enable,
                max_abs=boundary_cov_residual_max_abs,
            )
            if mode in ("default", "anisotropic", "aniso"):
                cov = self._covariance_matrix_from_scaling_rotation(scaling, scaling_modifier, rotation)
            elif mode in ("orthogonalized", "orth", "orth_rotation"):
                cov = self._covariance_matrix_from_scaling_rotation(
                    scaling,
                    scaling_modifier,
                    self._orthogonalize_covariance_rotation(rotation),
                )
            elif mode in ("canonical_rotation", "canonical", "raw_quaternion"):
                cov = self._covariance_matrix_from_scaling_rotation(scaling, scaling_modifier, self._rotation)
            elif mode.startswith("rotation_isotropic_") or mode.startswith("rot_isotropic_"):
                reduce_mode = mode.rsplit("_", 1)[-1]
                scalar = self._reduce_scaling_to_scalar(scaling, reduce_mode).repeat(1, 3)
                cov = self._covariance_matrix_from_scaling_rotation(scalar, scaling_modifier, rotation)
            elif mode.startswith("world_isotropic_") or mode.startswith("iso_"):
                reduce_mode = mode.rsplit("_", 1)[-1]
                scalar = scaling_modifier * self._reduce_scaling_to_scalar(scaling, reduce_mode).reshape(-1)
                cov = torch.zeros((scaling.shape[0], 3, 3), dtype=scaling.dtype, device=scaling.device)
                cov[:, 0, 0] = scalar * scalar
                cov[:, 1, 1] = scalar * scalar
                cov[:, 2, 2] = scalar * scalar
            elif mode in ("polar_det", "polar_volume", "binding_stable", "binding_stable_det"):
                cov = self._covariance_matrix_from_polar_stabilized_transform(
                    scaling,
                    scaling_modifier,
                    rotation,
                    mode="polar_det",
                    det_min=polar_det_min,
                    det_max=polar_det_max,
                    det_power=polar_det_power,
                    anisotropy_clamp=polar_anisotropy_clamp,
                )
            elif mode in ("polar_svd_clamp", "svd_clamp", "polar_aniso_clamp"):
                cov = self._covariance_matrix_from_polar_stabilized_transform(
                    scaling,
                    scaling_modifier,
                    rotation,
                    mode="polar_svd_clamp",
                    det_min=polar_det_min,
                    det_max=polar_det_max,
                    det_power=polar_det_power,
                    anisotropy_clamp=polar_anisotropy_clamp,
                )
            else:
                cov = self._covariance_matrix_from_scaling_rotation(scaling, scaling_modifier, rotation)
            cov = self._apply_screen_space_covariance_actuator(
                cov,
                camera,
                dynamic_shrink_mask,
                dynamic_grow_mask,
                shrink_normals=shrink_normals_2d,
                grow_normals=grow_normals_2d,
                normal_shrink_factor=normal_shrink_factor,
                normal_grow_factor=normal_grow_factor,
                tangent_factor=signed_screen_tangent_factor,
            )
            if bool(signed_dynamic_guard_enable):
                cov = self._apply_signed_dynamic_covariance_guard(
                    cov,
                    scaling,
                    scaling_modifier,
                    rotation,
                    dynamic_shrink_mask,
                    dynamic_grow_mask,
                    shrink_score=dynamic_shrink_score,
                    grow_score=dynamic_grow_score,
                    shrink_mode=signed_dynamic_guard_shrink_mode,
                    grow_mode=signed_dynamic_guard_grow_mode,
                    shrink_strength=signed_dynamic_guard_shrink_strength,
                    grow_strength=signed_dynamic_guard_grow_strength,
                    power=signed_dynamic_guard_power,
                    quantile=signed_dynamic_guard_quantile,
                    min_weight=signed_dynamic_guard_min_weight,
                    anisotropy_clamp=signed_dynamic_guard_anisotropy_clamp,
                )
            if bool(binding_covariance_guard_enable):
                cov = self._apply_binding_covariance_guard(
                    cov,
                    scaling,
                    scaling_modifier,
                    rotation,
                    mode=binding_covariance_guard_mode,
                    strength=binding_covariance_guard_strength,
                    boundary_min=binding_covariance_guard_boundary_min,
                    layer_ids=binding_covariance_guard_layer_ids,
                    region_ids=binding_covariance_guard_region_ids,
                    joint_ids=binding_covariance_guard_joint_ids,
                    thin_min=binding_covariance_guard_thin_min,
                    surface_min=binding_covariance_guard_surface_min,
                    surface_max=binding_covariance_guard_surface_max,
                    power=binding_covariance_guard_power,
                    max_points=binding_covariance_guard_max_points,
                    anisotropy_clamp=binding_covariance_guard_anisotropy_clamp,
                )
            return strip_symmetric(cov)

        cov = self._covariance_matrix_for_mode(
            scaling,
            scaling_modifier,
            rotation,
            mode,
            polar_det_min=polar_det_min,
            polar_det_max=polar_det_max,
            polar_det_power=polar_det_power,
            polar_anisotropy_clamp=polar_anisotropy_clamp,
        )
        if bool(binding_covariance_guard_enable):
            cov = self._apply_binding_covariance_guard(
                cov,
                scaling,
                scaling_modifier,
                rotation,
                mode=binding_covariance_guard_mode,
                strength=binding_covariance_guard_strength,
                boundary_min=binding_covariance_guard_boundary_min,
                layer_ids=binding_covariance_guard_layer_ids,
                region_ids=binding_covariance_guard_region_ids,
                joint_ids=binding_covariance_guard_joint_ids,
                thin_min=binding_covariance_guard_thin_min,
                surface_min=binding_covariance_guard_surface_min,
                surface_max=binding_covariance_guard_surface_max,
                power=binding_covariance_guard_power,
                max_points=binding_covariance_guard_max_points,
                anisotropy_clamp=binding_covariance_guard_anisotropy_clamp,
            )
        return strip_symmetric(cov)

    def oneupSHdegree(self):
        if not self.use_sh:
            return
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def get_opacity_loss(self):
        # opacity classification loss
        opacity = self.get_opacity
        eps = 1e-6
        loss_opacity_cls = -(opacity * torch.log(opacity + eps) + (1 - opacity) * torch.log(1 - opacity + eps)).mean()
        return {'opacity': loss_opacity_cls}

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale=1.):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        if self.use_sh:
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
        else:
            features = torch.zeros((fused_color.shape[0], 1, self.feature_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._boundary_tag = torch.zeros((fused_point_cloud.shape[0],), dtype=torch.float, device="cuda")
        self._boundary_opacity_residual = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda").requires_grad_(True))
        self._boundary_scaling_residual = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda").requires_grad_(True))
        self._boundary_cov_residual = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], self._boundary_cov_residual_channels()), dtype=torch.float, device="cuda").requires_grad_(True))
        self._binding_layer_logits_residual = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda").requires_grad_(True))
        self._semantic_region_logits_residual = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda").requires_grad_(True))
        self._semantic_compact_logits_residual = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 6), dtype=torch.float, device="cuda").requires_grad_(True))
        self._semantic_asset_region_logits_residual = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda").requires_grad_(True))
        self._semantic_asset_compact_logits_residual = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 6), dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def construct_list_of_attributes(self):
        attrs = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            attrs.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            attrs.append('f_rest_{}'.format(i))
        attrs.append('opacity')
        attrs.append('boundary_tag')
        attrs.append('boundary_opacity_residual')
        for i in range(self._boundary_scaling_residual.shape[1]):
            attrs.append('boundary_scale_residual_{}'.format(i))
        cov_channels = self._boundary_cov_residual_channels()
        attrs.append('boundary_cov_residual')
        for i in range(1, cov_channels):
            attrs.append(f'boundary_cov_residual_{i}')
        for i in range(3):
            attrs.append('binding_layer_logits_residual_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            attrs.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            attrs.append('rot_{}'.format(i))
        return attrs

    def save_ply(self, path):
        self.ensure_boundary_state_matches_points(verbose=False)
        self._ensure_layer_logits_adapter_state_matches_points(verbose=False)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        boundary_tag = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        if self.has_boundary_tag_state():
            boundary_tag = self._boundary_tag.detach().unsqueeze(-1).cpu().numpy()
        boundary_opacity_residual = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        if torch.is_tensor(self._boundary_opacity_residual) and self._boundary_opacity_residual.numel() > 0:
            boundary_opacity_residual = self._boundary_opacity_residual.detach().cpu().numpy()
        boundary_scaling_residual = np.zeros((xyz.shape[0], 3), dtype=np.float32)
        if torch.is_tensor(self._boundary_scaling_residual) and self._boundary_scaling_residual.numel() > 0:
            boundary_scaling_residual = self._boundary_scaling_residual.detach().cpu().numpy()
        boundary_cov_residual = np.zeros((xyz.shape[0], self._boundary_cov_residual_channels()), dtype=np.float32)
        if torch.is_tensor(self._boundary_cov_residual) and self._boundary_cov_residual.numel() > 0:
            source_cov = self._boundary_cov_residual.detach().cpu().numpy()
            copy_cols = min(source_cov.shape[1] if source_cov.ndim > 1 else 1, boundary_cov_residual.shape[1])
            boundary_cov_residual[:, :copy_cols] = source_cov.reshape(source_cov.shape[0], -1)[:, :copy_cols]
        binding_layer_logits_residual = np.zeros((xyz.shape[0], 3), dtype=np.float32)
        if torch.is_tensor(self._binding_layer_logits_residual) and self._binding_layer_logits_residual.numel() > 0:
            source_layer = self._binding_layer_logits_residual.detach().cpu().numpy()
            copy_cols = min(source_layer.shape[1] if source_layer.ndim > 1 else 1, 3)
            binding_layer_logits_residual[:, :copy_cols] = source_layer.reshape(source_layer.shape[0], -1)[:, :copy_cols]
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        attributes = np.concatenate((
            xyz, normals, f_dc, f_rest, opacities, boundary_tag,
            boundary_opacity_residual, boundary_scaling_residual, boundary_cov_residual,
            binding_layer_logits_residual, scale, rotation,
        ), axis=1)
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, 'vertex')]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)
        vertex = plydata.elements[0]
        names = vertex.data.dtype.names

        xyz = np.stack((np.asarray(vertex['x']), np.asarray(vertex['y']), np.asarray(vertex['z'])), axis=1)
        opacities = np.asarray(vertex['opacity'])[..., np.newaxis]
        boundary_tag = np.asarray(vertex['boundary_tag'])[..., np.newaxis] if 'boundary_tag' in names else np.zeros_like(opacities)
        boundary_opacity_residual = (
            np.asarray(vertex['boundary_opacity_residual'])[..., np.newaxis]
            if 'boundary_opacity_residual' in names else np.zeros_like(opacities)
        )

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(vertex['f_dc_0'])
        features_dc[:, 1, 0] = np.asarray(vertex['f_dc_1'])
        features_dc[:, 2, 0] = np.asarray(vertex['f_dc_2'])

        extra_f_names = sorted([p.name for p in vertex.properties if p.name.startswith('f_rest_')], key=lambda x: int(x.split('_')[-1]))
        expected_rest = 3 * (self.max_sh_degree + 1) ** 2 - 3
        if len(extra_f_names) != expected_rest:
            raise ValueError(f'Unexpected SH rest channel count in {path}: got {len(extra_f_names)}, expected {expected_rest}')
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(vertex[attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = sorted([p.name for p in vertex.properties if p.name.startswith('scale_')], key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(vertex[attr_name])

        boundary_scale_names = sorted([p.name for p in vertex.properties if p.name.startswith('boundary_scale_residual_')], key=lambda x: int(x.split('_')[-1]))
        boundary_scaling_residual = np.zeros((xyz.shape[0], 3), dtype=np.float32)
        for idx, attr_name in enumerate(boundary_scale_names[:3]):
            boundary_scaling_residual[:, idx] = np.asarray(vertex[attr_name])

        cov_channels = self._boundary_cov_residual_channels()
        boundary_cov_residual = np.zeros((xyz.shape[0], cov_channels), dtype=np.float32)
        if 'boundary_cov_residual' in names:
            boundary_cov_residual[:, 0] = np.asarray(vertex['boundary_cov_residual'])
        for idx in range(1, cov_channels):
            attr_name = f'boundary_cov_residual_{idx}'
            if attr_name in names:
                boundary_cov_residual[:, idx] = np.asarray(vertex[attr_name])

        binding_layer_logits_residual = np.zeros((xyz.shape[0], 3), dtype=np.float32)
        for idx in range(3):
            attr_name = f'binding_layer_logits_residual_{idx}'
            if attr_name in names:
                binding_layer_logits_residual[:, idx] = np.asarray(vertex[attr_name])

        rot_names = sorted([p.name for p in vertex.properties if p.name.startswith('rot')], key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(vertex[attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device='cuda').requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device='cuda').transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device='cuda').transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device='cuda').requires_grad_(True))
        self._boundary_tag = torch.tensor(boundary_tag[:, 0], dtype=torch.float, device='cuda')
        self._boundary_opacity_residual = nn.Parameter(torch.tensor(boundary_opacity_residual, dtype=torch.float, device='cuda').requires_grad_(True))
        self._boundary_scaling_residual = nn.Parameter(torch.tensor(boundary_scaling_residual, dtype=torch.float, device='cuda').requires_grad_(True))
        self._boundary_cov_residual = nn.Parameter(torch.tensor(boundary_cov_residual, dtype=torch.float, device='cuda').requires_grad_(True))
        self._binding_layer_logits_residual = nn.Parameter(torch.tensor(binding_layer_logits_residual, dtype=torch.float, device='cuda').requires_grad_(True))
        self._semantic_region_logits_residual = nn.Parameter(torch.zeros((xyz.shape[0], 3), dtype=torch.float, device='cuda').requires_grad_(True))
        self._semantic_compact_logits_residual = nn.Parameter(torch.zeros((xyz.shape[0], 6), dtype=torch.float, device='cuda').requires_grad_(True))
        self._semantic_asset_region_logits_residual = nn.Parameter(torch.zeros((xyz.shape[0], 3), dtype=torch.float, device='cuda').requires_grad_(True))
        self._semantic_asset_compact_logits_residual = nn.Parameter(torch.zeros((xyz.shape[0], 6), dtype=torch.float, device='cuda').requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device='cuda').requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device='cuda').requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device='cuda')
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.optimizer = None
        self.active_sh_degree = self.max_sh_degree
