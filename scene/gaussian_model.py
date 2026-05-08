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
from collections.abc import Sequence
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import torch.nn.functional as F
import os
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.pytorch3d_compat import ops

import trimesh
import igl


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
    def _arm_joint_ids(self):
        return [13, 14, 16, 17, 18, 19, 20, 21, 22, 23]

    def _binding_lineage_ids_from_state(self, binding_state, point_count, device):
        if not binding_state:
            return None
        lineage_ids = binding_state.get('densify_root_lineage_id', None)
        if not torch.is_tensor(lineage_ids) or lineage_ids.shape[0] != point_count:
            lineage_ids = binding_state.get('densify_lineage_id', None)
        if not torch.is_tensor(lineage_ids) or lineage_ids.shape[0] != point_count:
            return None
        return lineage_ids.to(device=device, dtype=torch.long)

    def _binding_arm_gate_mask_from_state(
        self,
        binding_state,
        point_count,
        device,
        iteration=0,
        arm_gate_mode='current',
        require_source_consensus=False,
    ):
        if not binding_state:
            return None

        def _state_tensor(key):
            value = binding_state.get(key, None)
            if torch.is_tensor(value) and value.shape[0] == point_count:
                return value.to(device=device)
            return None

        allowed_joint_ids = self._arm_joint_ids()
        current_arm_mask = None
        dominant_joint = _state_tensor('dominant_joint')
        if dominant_joint is not None:
            arm_joint_ids = torch.tensor(
                allowed_joint_ids,
                device=device,
                dtype=dominant_joint.dtype,
            )
            current_arm_mask = (
                dominant_joint.unsqueeze(-1) == arm_joint_ids.unsqueeze(0)
            ).any(dim=-1)

        source_arm_mask = self._source_joint_firewall_mask_from_source_joints(
            source_parent_joint=_state_tensor('source_parent_joint'),
            source_root_parent_joint=_state_tensor('source_root_parent_joint'),
            device=device,
            iteration=iteration,
            allowed_joint_ids=allowed_joint_ids,
            require_consensus=bool(require_source_consensus),
        )

        arm_gate_mode = str(arm_gate_mode or 'current').lower()
        if arm_gate_mode == 'current':
            return current_arm_mask
        if arm_gate_mode == 'source':
            return source_arm_mask if source_arm_mask is not None else current_arm_mask
        if arm_gate_mode == 'source_or_current':
            if source_arm_mask is not None and current_arm_mask is not None:
                return source_arm_mask | current_arm_mask
            if source_arm_mask is not None:
                return source_arm_mask
            return current_arm_mask
        if arm_gate_mode == 'source_and_current':
            if source_arm_mask is not None and current_arm_mask is not None:
                return source_arm_mask & current_arm_mask
            if source_arm_mask is not None:
                return source_arm_mask
            return current_arm_mask
        return current_arm_mask

    def _binding_lineage_offender_mask(self, binding_state, point_count, device, iteration=0):
        if not bool(self.cfg.get('binding_densify_lineage_offender_filter_enable', True)):
            return torch.zeros((point_count,), dtype=torch.bool, device=device)
        if point_count <= 0:
            return torch.zeros((point_count,), dtype=torch.bool, device=device)

        lineage_offender_score_accum, lineage_offender_count_accum = self.get_lineage_offender_state()
        full_point_count = int(self.get_xyz.shape[0]) if torch.is_tensor(self._xyz) and self._xyz.ndim >= 2 else 0
        if (
            lineage_offender_score_accum is None
            or lineage_offender_count_accum is None
            or full_point_count <= 0
            or lineage_offender_score_accum.shape[0] != full_point_count
            or lineage_offender_count_accum.shape[0] != full_point_count
        ):
            return torch.zeros((point_count,), dtype=torch.bool, device=device)

        lineage_offender_mean = (
            lineage_offender_score_accum.to(device=device)
            / lineage_offender_count_accum.to(device=device).clamp_min(1.0)
        )
        lineage_offender_min_observations = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_lineage_offender_min_observations', 2.0),
            default=2.0,
        ))
        lineage_offender_score_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_lineage_offender_score_threshold', 0.45),
            default=0.45,
        ))
        full_mask = (
            lineage_offender_count_accum.to(device=device) >= lineage_offender_min_observations
        ) & (
            lineage_offender_mean >= lineage_offender_score_threshold
        )
        if not bool(full_mask.any().item()):
            return torch.zeros((point_count,), dtype=torch.bool, device=device)

        if point_count == full_point_count:
            return full_mask.to(device=device, dtype=torch.bool)

        full_binding_state = self.get_binding_state()
        full_lineage_ids = self._binding_lineage_ids_from_state(full_binding_state, full_point_count, device)
        subset_lineage_ids = self._binding_lineage_ids_from_state(binding_state, point_count, device)
        if full_lineage_ids is None or subset_lineage_ids is None:
            return torch.zeros((point_count,), dtype=torch.bool, device=device)

        flagged_lineage_ids = torch.unique(full_lineage_ids[full_mask & (full_lineage_ids >= 0)], sorted=True)
        if flagged_lineage_ids.numel() == 0:
            return torch.zeros((point_count,), dtype=torch.bool, device=device)
        return torch.isin(subset_lineage_ids, flagged_lineage_ids)

    def _allowed_newborn_firewall_joint_ids(self, iteration=0):
        joint_ids_cfg = self.cfg.get(
            'binding_densify_source_firewall_joint_ids',
            self._arm_joint_ids(),
        )
        if joint_ids_cfg is None:
            return []
        if isinstance(joint_ids_cfg, Sequence) and not isinstance(joint_ids_cfg, str):
            raw_joint_ids = list(joint_ids_cfg)
            # Plain joint-id lists like [13, 14, 16, ...] should stay literal.
            if all(not isinstance(x, Sequence) or isinstance(x, str) for x in raw_joint_ids):
                if all(isinstance(x, (bool, int, float, np.integer, np.floating)) for x in raw_joint_ids):
                    return [int(x) for x in raw_joint_ids]
            resolved = resolve_schedule_value(iteration, joint_ids_cfg, default=joint_ids_cfg)
            if isinstance(resolved, Sequence) and not isinstance(resolved, str):
                return [int(x) for x in resolved]
            return [int(resolved)]
        return [int(joint_ids_cfg)]

    def _source_joint_firewall_mask_from_parent_indices(
        self,
        point_count,
        device,
        source_parent_index=None,
        source_root_parent_index=None,
        iteration=0,
        allowed_joint_ids=None,
        require_consensus=False,
    ):
        firewall_enable = bool(self.cfg.get('binding_densify_source_firewall_enable', False))
        if not firewall_enable or point_count <= 0:
            return None
        binding_state = self.get_binding_state()
        if not binding_state:
            return None

        dominant_joint = binding_state.get('dominant_joint', None)
        if not torch.is_tensor(dominant_joint) or dominant_joint.shape[0] != self.get_xyz.shape[0]:
            return None
        dominant_joint = dominant_joint.to(device=device)

        if allowed_joint_ids is None:
            allowed_joint_ids = self._allowed_newborn_firewall_joint_ids(iteration=iteration)
        if len(allowed_joint_ids) == 0:
            return None
        allowed_joint_ids = torch.tensor(allowed_joint_ids, device=device, dtype=dominant_joint.dtype)

        def _mask_from_parent(parent_index):
            if not torch.is_tensor(parent_index) or parent_index.shape[0] != point_count:
                return None, 0, 0
            parent_index = parent_index.to(device=device, dtype=torch.long)
            valid_parent = (parent_index >= 0) & (parent_index < dominant_joint.shape[0])
            if not bool(valid_parent.any().item()):
                return None, 0, 0
            mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
            parent_joint = dominant_joint[parent_index[valid_parent]]
            mask[valid_parent] = (parent_joint.unsqueeze(-1) == allowed_joint_ids.unsqueeze(0)).any(dim=-1)
            return mask, int(valid_parent.sum().item()), int(mask.sum().item())

        source_mask, source_valid_count, source_match_count = _mask_from_parent(source_parent_index)
        root_mask, root_valid_count, root_match_count = _mask_from_parent(source_root_parent_index)
        if source_mask is None:
            source_mask = root_mask
        elif root_mask is not None:
            if require_consensus:
                source_mask &= root_mask
            else:
                source_mask |= root_mask
        if source_mask is None:
            return None

        if bool(self.cfg.get('binding_densify_debug_verbose', False)):
            print(
                '[GaussianModel] source firewall mask '
                f'iter={iteration} allowed={allowed_joint_ids.tolist()} '
                f'kept={int(source_mask.sum().item())}/{point_count} '
                f'consensus={int(bool(require_consensus))} '
                f'parent_valid={source_valid_count} parent_match={source_match_count} '
                f'root_valid={root_valid_count} root_match={root_match_count}'
            )
        return source_mask

    def _source_joint_firewall_mask_from_source_joints(
        self,
        source_parent_joint=None,
        source_root_parent_joint=None,
        device=None,
        iteration=0,
        allowed_joint_ids=None,
        require_consensus=False,
    ):
        firewall_enable = bool(self.cfg.get('binding_densify_source_firewall_enable', False))
        if not firewall_enable:
            return None

        point_count = None
        if torch.is_tensor(source_parent_joint):
            point_count = int(source_parent_joint.shape[0])
        elif torch.is_tensor(source_root_parent_joint):
            point_count = int(source_root_parent_joint.shape[0])
        if point_count is None or point_count <= 0:
            return None
        if device is None:
            if torch.is_tensor(source_parent_joint):
                device = source_parent_joint.device
            elif torch.is_tensor(source_root_parent_joint):
                device = source_root_parent_joint.device
            else:
                device = self._xyz.device

        if allowed_joint_ids is None:
            allowed_joint_ids = self._allowed_newborn_firewall_joint_ids(iteration=iteration)
        if len(allowed_joint_ids) == 0:
            return None
        allowed_joint_ids = torch.tensor(allowed_joint_ids, device=device, dtype=torch.long)

        def _mask_from_source_joint(source_joint):
            if not torch.is_tensor(source_joint) or source_joint.shape[0] != point_count:
                return None, 0, 0
            source_joint = source_joint.to(device=device, dtype=torch.long).reshape(-1)
            valid = source_joint >= 0
            if not bool(valid.any().item()):
                return None, 0, 0
            mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
            mask[valid] = (source_joint[valid].unsqueeze(-1) == allowed_joint_ids.unsqueeze(0)).any(dim=-1)
            return mask, int(valid.sum().item()), int(mask.sum().item())

        source_mask, source_valid_count, source_match_count = _mask_from_source_joint(source_parent_joint)
        root_mask, root_valid_count, root_match_count = _mask_from_source_joint(source_root_parent_joint)
        if source_mask is None:
            source_mask = root_mask
        elif root_mask is not None:
            if require_consensus:
                source_mask &= root_mask
            else:
                source_mask |= root_mask
        if source_mask is None:
            return None

        if bool(self.cfg.get('binding_densify_debug_verbose', False)):
            print(
                '[GaussianModel] source firewall mask '
                f'iter={iteration} allowed={allowed_joint_ids.tolist()} '
                f'kept={int(source_mask.sum().item())}/{point_count} '
                f'consensus={int(bool(require_consensus))} '
                f'parent_valid={source_valid_count} parent_match={source_match_count} '
                f'root_valid={root_valid_count} root_match={root_match_count} '
                f'mode=source_joint'
            )
        return source_mask

    def _binding_joint_gate_mask(self, point_indices, device, joint_ids):
        if not torch.is_tensor(point_indices):
            return None
        point_indices = point_indices.to(device=device, dtype=torch.long).reshape(-1)
        if point_indices.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool, device=device)

        binding_state = self.get_binding_state()
        if not binding_state:
            return None
        dominant_joint = binding_state.get('dominant_joint', None)
        if not torch.is_tensor(dominant_joint) or dominant_joint.shape[0] != self.get_xyz.shape[0]:
            return None
        dominant_joint = dominant_joint.to(device=device)
        joint_ids = torch.tensor(list(joint_ids), device=device, dtype=dominant_joint.dtype)
        return (dominant_joint[point_indices].unsqueeze(-1) == joint_ids.unsqueeze(0)).any(dim=-1)

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
        self._offender_score_accum = torch.empty(0)
        self._offender_count_accum = torch.empty(0)
        self._offender_refill_score = torch.empty(0)
        self._lineage_offender_score_accum = torch.empty(0)
        self._lineage_offender_count_accum = torch.empty(0)
        self._next_lineage_uid = 0
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.binding_state = {}
        self.setup_functions()

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
                      "binding_region_ids",
                      "binding_compact_semantic_probs",
                      "binding_compact_semantic_ids",
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
                      "_boundary_scaling_residual"]
        for parameter in parameters:
            setattr(cloned, parameter, getattr(self, parameter) + 0.)

        if self.has_binding_state():
            cloned.set_binding_state(self.get_binding_state())

        return cloned

    def has_binding_state(self):
        return len(getattr(self, 'binding_state', {})) > 0

    def get_binding_state(self):
        return getattr(self, 'binding_state', {})

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
        self._ensure_binding_lineage_state(self.binding_state)

    def _allocate_lineage_ids(self, count, device):
        count = int(count)
        if count <= 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        start = int(self._next_lineage_uid)
        end = start + count
        self._next_lineage_uid = end
        return torch.arange(start, end, dtype=torch.long, device=device)

    def _ensure_binding_lineage_state(self, binding_state):
        if not binding_state:
            return binding_state

        point_count = None
        device = None
        for value in binding_state.values():
            if torch.is_tensor(value):
                point_count = value.shape[0]
                device = value.device
                break
        if point_count is None or point_count <= 0:
            return binding_state

        lineage_id = binding_state.get('densify_lineage_id', None)
        if not torch.is_tensor(lineage_id) or lineage_id.shape[0] != point_count:
            lineage_id = self._allocate_lineage_ids(point_count, device)
        else:
            lineage_id = lineage_id.to(device=device, dtype=torch.long)
            if lineage_id.numel() > 0:
                self._next_lineage_uid = max(self._next_lineage_uid, int(lineage_id.max().item()) + 1)

        root_lineage_id = binding_state.get('densify_root_lineage_id', None)
        if not torch.is_tensor(root_lineage_id) or root_lineage_id.shape[0] != point_count:
            root_lineage_id = lineage_id.clone()
        else:
            root_lineage_id = root_lineage_id.to(device=device, dtype=torch.long)
            root_lineage_id = torch.where(root_lineage_id >= 0, root_lineage_id, lineage_id)
            if root_lineage_id.numel() > 0:
                self._next_lineage_uid = max(self._next_lineage_uid, int(root_lineage_id.max().item()) + 1)

        binding_state['densify_lineage_id'] = lineage_id
        binding_state['densify_root_lineage_id'] = root_lineage_id
        return binding_state

    def _slice_binding_state(self, mask, repeats=1):
        if not self.has_binding_state():
            return {}

        sliced_state = {}
        for key, value in self.binding_state.items():
            if not torch.is_tensor(value) or value.shape[0] != mask.shape[0]:
                continue
            selected = value[mask]
            if repeats != 1:
                repeat_shape = (repeats,) + (1,) * (selected.dim() - 1)
                selected = selected.repeat(*repeat_shape)
            sliced_state[key] = selected.clone()
        return sliced_state

    def _update_binding_offsets(self, binding_state, delta_xyz):
        if not binding_state:
            return binding_state

        updated_state = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in binding_state.items()
        }

        if 'bound_xyz' in updated_state:
            updated_state['bound_xyz'] = updated_state['bound_xyz'] + delta_xyz
        if 'local_offset' in updated_state:
            updated_state['local_offset'] = updated_state['local_offset'] + delta_xyz

        if 'anchor_normal' in updated_state:
            normals = updated_state['anchor_normal']
            delta_normal_mag = torch.sum(delta_xyz * normals, dim=-1, keepdim=True)
            delta_normal = delta_normal_mag * normals
            delta_tangent = delta_xyz - delta_normal

            if 'normal_offset' in updated_state:
                updated_state['normal_offset'] = updated_state['normal_offset'] + delta_normal
            if 'tangent_offset' in updated_state:
                updated_state['tangent_offset'] = updated_state['tangent_offset'] + delta_tangent
            if 'normal_offset' in updated_state:
                updated_state['surface_distance'] = torch.norm(updated_state['normal_offset'], dim=-1)
            else:
                updated_state['surface_distance'] = torch.abs(delta_normal_mag.squeeze(-1))

        return updated_state

    def _append_binding_state(self, extension_state):
        if not extension_state or not self.has_binding_state():
            return

        appended_state = dict(self.binding_state)
        base_count = None
        base_device = None
        for value in appended_state.values():
            if torch.is_tensor(value):
                base_count = int(value.shape[0])
                base_device = value.device
                break
        if base_count is None:
            return

        extension_count = None
        extension_device = None
        for value in extension_state.values():
            if torch.is_tensor(value):
                extension_count = int(value.shape[0])
                extension_device = value.device
                break
        if extension_count is None:
            return
        original_base_count = int(base_count)
        original_base_device = base_device

        def _append_mask_key(key):
            base_mask = appended_state.get(key, None)
            if not torch.is_tensor(base_mask) or base_mask.shape[0] != original_base_count:
                base_mask = torch.zeros((original_base_count,), dtype=torch.bool, device=original_base_device)
            else:
                base_mask = base_mask.to(device=original_base_device, dtype=torch.bool)

            ext_mask = extension_state.get(key, None)
            if not torch.is_tensor(ext_mask) or ext_mask.shape[0] != extension_count:
                ext_mask = torch.zeros((extension_count,), dtype=torch.bool, device=extension_device)
            else:
                ext_mask = ext_mask.to(device=original_base_device, dtype=torch.bool)

            appended_state[key] = torch.cat((base_mask, ext_mask), dim=0)

        _append_mask_key('anchor_refresh_mask')
        _append_mask_key('densify_risky_child_mask')
        missing_index_fill_keys = {
            'densify_parent_index',
            'densify_root_parent_index',
            'source_parent_joint',
            'source_root_parent_joint',
        }

        for key, value in extension_state.items():
            if not torch.is_tensor(value):
                continue
            if key in ('anchor_refresh_mask', 'densify_risky_child_mask'):
                continue
            if key not in appended_state:
                base_shape = (original_base_count,) + tuple(value.shape[1:])
                if key in missing_index_fill_keys and value.dtype in (torch.int32, torch.int64):
                    appended_state[key] = torch.full(
                        base_shape,
                        -1,
                        dtype=value.dtype,
                        device=original_base_device,
                    )
                else:
                    appended_state[key] = torch.zeros(
                        base_shape,
                        dtype=value.dtype,
                        device=original_base_device,
                    )
            appended_state[key] = torch.cat((appended_state[key], value.to(appended_state[key].device)), dim=0)
        self.binding_state = appended_state

    def _binding_mask_debug_summary(self, binding_state):
        if not binding_state:
            return 'empty'
        point_count = 0
        refresh_count = 0
        risky_count = 0
        overlap_count = 0
        for value in binding_state.values():
            if torch.is_tensor(value):
                point_count = int(value.shape[0])
                break
        refresh_mask = binding_state.get('anchor_refresh_mask', None)
        risky_mask = binding_state.get('densify_risky_child_mask', None)
        if torch.is_tensor(refresh_mask):
            refresh_mask = refresh_mask.to(dtype=torch.bool)
            refresh_count = int(refresh_mask.sum().item())
        else:
            refresh_mask = None
        if torch.is_tensor(risky_mask):
            risky_mask = risky_mask.to(dtype=torch.bool)
            risky_count = int(risky_mask.sum().item())
        else:
            risky_mask = None
        if refresh_mask is not None and risky_mask is not None and refresh_mask.shape[0] == risky_mask.shape[0]:
            overlap_count = int((refresh_mask & risky_mask).sum().item())
        return (
            f'points={point_count} refresh={refresh_count} '
            f'risky={risky_count} overlap={overlap_count}'
        )

    def _binding_source_joint_debug_summary(self, binding_state):
        if not binding_state:
            return 'source_parent=missing source_root=missing'

        def _summarize(key):
            value = binding_state.get(key, None)
            if not torch.is_tensor(value):
                return f'{key}=missing'
            valid = value.to(dtype=torch.long) >= 0
            return f'{key}=present valid={int(valid.sum().item())}/{int(value.shape[0])}'

        return (
            f'{_summarize("source_parent_joint")} '
            f'{_summarize("source_root_parent_joint")}'
        )

    def _mark_risky_binding_points_for_refresh(self, binding_state, boundary_tags=None, iteration=0):
        if not binding_state:
            return binding_state

        updated_state = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in binding_state.items()
        }
        point_count = None
        for value in updated_state.values():
            if torch.is_tensor(value):
                point_count = value.shape[0]
                break
        if point_count is None or point_count == 0:
            return updated_state

        device = None
        for value in updated_state.values():
            if torch.is_tensor(value):
                device = value.device
                break

        refresh_mask = updated_state.get('anchor_refresh_mask', None)
        if not torch.is_tensor(refresh_mask) or refresh_mask.shape[0] != point_count:
            refresh_mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
        else:
            refresh_mask = refresh_mask.to(device=device, dtype=torch.bool)

        boundary_mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
        if torch.is_tensor(boundary_tags) and boundary_tags.shape[0] == point_count:
            boundary_mask = boundary_tags.to(device=device).reshape(-1) > 0.0

        semantic_distance_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_refresh_semantic_distance_threshold', 0.03),
            default=0.03,
        ))
        surface_distance_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_refresh_surface_distance_threshold', 0.012),
            default=0.012,
        ))
        confidence_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_refresh_confidence_threshold', 0.6),
            default=0.6,
        ))
        weight_gap_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_refresh_weight_gap_threshold', 0.35),
            default=0.35,
        ))
        boundary_only = bool(self.cfg.get('binding_refresh_boundary_only', True))
        require_boundary_tag = bool(self.cfg.get('binding_refresh_require_boundary_tag', False))
        force_on_boundary = bool(self.cfg.get('binding_refresh_force_on_boundary', False))

        semantic_distance = updated_state.get('semantic_distance', None)
        surface_distance = updated_state.get('surface_distance', None)
        confidence = updated_state.get('anchor_confidence', None)
        anchor_weights = updated_state.get('anchor_weights', None)

        risk_metric = torch.zeros((point_count,), dtype=torch.bool, device=device)
        if torch.is_tensor(semantic_distance) and semantic_distance.shape[0] == point_count:
            risk_metric |= semantic_distance.to(device=device) > semantic_distance_threshold
        if torch.is_tensor(surface_distance) and surface_distance.shape[0] == point_count:
            risk_metric |= surface_distance.to(device=device) > surface_distance_threshold
        if torch.is_tensor(confidence) and confidence.shape[0] == point_count:
            risk_metric |= confidence.to(device=device) < confidence_threshold
        if torch.is_tensor(anchor_weights) and anchor_weights.shape[0] == point_count and anchor_weights.ndim == 2 and anchor_weights.shape[1] >= 2:
            top2 = torch.topk(anchor_weights.to(device=device), k=2, dim=-1).values
            risk_metric |= (top2[:, 0] - top2[:, 1]) < weight_gap_threshold

        refresh_gate = torch.ones((point_count,), dtype=torch.bool, device=device)
        if boundary_only:
            if torch.is_tensor(boundary_tags) and boundary_tags.shape[0] == point_count:
                refresh_gate = boundary_mask
            elif require_boundary_tag:
                refresh_gate = torch.zeros((point_count,), dtype=torch.bool, device=device)
        refresh_mask |= refresh_gate & risk_metric
        if force_on_boundary:
            refresh_mask |= boundary_mask
        updated_state['anchor_refresh_mask'] = refresh_mask
        return updated_state

    def _clear_newborn_binding_flags(self, binding_state):
        if not binding_state:
            return binding_state

        updated_state = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in binding_state.items()
        }
        point_count = None
        device = None
        for value in updated_state.values():
            if torch.is_tensor(value):
                point_count = value.shape[0]
                device = value.device
                break
        if point_count is None or point_count == 0:
            return updated_state

        for key in ('anchor_refresh_mask', 'densify_risky_child_mask'):
            state_value = updated_state.get(key, None)
            if torch.is_tensor(state_value) and state_value.shape[0] == point_count:
                updated_state[key] = torch.zeros_like(state_value, dtype=torch.bool, device=device)
            else:
                updated_state[key] = torch.zeros((point_count,), dtype=torch.bool, device=device)
        return updated_state

    def _annotate_densified_binding_lineage(self, binding_state, parent_index, iteration=0):
        if not binding_state or not torch.is_tensor(parent_index):
            return binding_state

        updated_state = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in binding_state.items()
        }
        point_count = None
        device = None
        for value in updated_state.values():
            if torch.is_tensor(value):
                point_count = value.shape[0]
                device = value.device
                break
        if point_count is None or point_count == 0:
            return updated_state

        parent_index = parent_index.to(device=device, dtype=torch.long).reshape(-1)
        if parent_index.shape[0] != point_count:
            return updated_state

        previous_lineage_id = updated_state.get('densify_lineage_id', None)
        if torch.is_tensor(previous_lineage_id) and previous_lineage_id.shape[0] == point_count:
            previous_lineage_id = previous_lineage_id.to(device=device, dtype=torch.long)
        else:
            previous_lineage_id = None
        previous_root_lineage_id = updated_state.get('densify_root_lineage_id', None)
        if torch.is_tensor(previous_root_lineage_id) and previous_root_lineage_id.shape[0] == point_count:
            previous_root_lineage_id = previous_root_lineage_id.to(device=device, dtype=torch.long)
        else:
            previous_root_lineage_id = None
        previous_dominant_joint = updated_state.get('dominant_joint', None)
        if torch.is_tensor(previous_dominant_joint) and previous_dominant_joint.shape[0] == point_count:
            previous_dominant_joint = previous_dominant_joint.to(device=device, dtype=torch.long)
        else:
            previous_dominant_joint = None
        previous_source_root_parent_joint = updated_state.get('source_root_parent_joint', None)
        if (
            torch.is_tensor(previous_source_root_parent_joint)
            and previous_source_root_parent_joint.shape[0] == point_count
        ):
            previous_source_root_parent_joint = previous_source_root_parent_joint.to(
                device=device,
                dtype=torch.long,
            )
        else:
            previous_source_root_parent_joint = None

        updated_state['densify_parent_index'] = parent_index.clone()

        root_parent_index = updated_state.get('densify_root_parent_index', None)
        if torch.is_tensor(root_parent_index) and root_parent_index.shape[0] == point_count:
            root_parent_index = root_parent_index.to(device=device, dtype=torch.long)
            updated_state['densify_root_parent_index'] = torch.where(
                root_parent_index >= 0,
                root_parent_index,
                parent_index,
            )
        else:
            updated_state['densify_root_parent_index'] = parent_index.clone()

        updated_state['densify_lineage_id'] = self._allocate_lineage_ids(point_count, device)
        if previous_root_lineage_id is not None:
            updated_state['densify_root_lineage_id'] = previous_root_lineage_id.clone()
        elif previous_lineage_id is not None:
            updated_state['densify_root_lineage_id'] = previous_lineage_id.clone()
        else:
            updated_state['densify_root_lineage_id'] = updated_state['densify_lineage_id'].clone()

        if previous_dominant_joint is not None:
            updated_state['source_parent_joint'] = previous_dominant_joint.clone()
        else:
            updated_state['source_parent_joint'] = torch.full(
                (point_count,),
                -1,
                dtype=torch.long,
                device=device,
            )
        if previous_source_root_parent_joint is not None:
            updated_state['source_root_parent_joint'] = previous_source_root_parent_joint.clone()
        elif previous_dominant_joint is not None:
            updated_state['source_root_parent_joint'] = previous_dominant_joint.clone()
        else:
            updated_state['source_root_parent_joint'] = torch.full(
                (point_count,),
                -1,
                dtype=torch.long,
                device=device,
            )

        updated_state['densify_birth_iter'] = torch.full(
            (point_count,),
            int(iteration),
            dtype=torch.long,
            device=device,
        )
        return updated_state

    def _binding_boundary_arm_risk_mask(
        self,
        binding_state,
        point_count,
        device,
        iteration=0,
        boundary_tags=None,
        boundary_threshold=0.05,
        arm_only=True,
        semantic_scale=1.0,
        surface_scale=1.0,
        confidence_margin=0.0,
        weight_gap_scale=1.0,
        include_refresh_mask=True,
        boundary_only=True,
        arm_gate_mode='current',
    ):
        debug_verbose = bool(self.cfg.get('binding_densify_debug_verbose', False))
        if not binding_state:
            return torch.zeros((point_count,), dtype=torch.bool, device=device)

        def _state_tensor(key):
            value = binding_state.get(key, None)
            if torch.is_tensor(value) and value.shape[0] == point_count:
                return value.to(device=device)
            return None

        gate_mask = torch.ones((point_count,), dtype=torch.bool, device=device)
        require_boundary_tag = bool(self.cfg.get('binding_risk_gate_require_boundary_tag', False))
        if boundary_only:
            if torch.is_tensor(boundary_tags) and boundary_tags.shape[0] == point_count:
                gate_mask &= boundary_tags.to(device=device).reshape(-1) > boundary_threshold
            elif require_boundary_tag:
                gate_mask &= False

        if arm_only:
            arm_mask = self._binding_arm_gate_mask_from_state(
                binding_state,
                point_count,
                device,
                iteration=iteration,
                arm_gate_mode=arm_gate_mode,
                require_source_consensus=bool(self.cfg.get(
                    'binding_densify_child_immediate_refresh_arm_gate_require_source_consensus',
                    False,
                )),
            )
            if arm_mask is None:
                gate_mask &= False
            else:
                gate_mask &= arm_mask.to(device=device, dtype=torch.bool)

        semantic_distance_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_refresh_semantic_distance_threshold', 0.03),
            default=0.03,
        ))
        surface_distance_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_refresh_surface_distance_threshold', 0.012),
            default=0.012,
        ))
        confidence_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_refresh_confidence_threshold', 0.6),
            default=0.6,
        ))
        weight_gap_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_refresh_weight_gap_threshold', 0.35),
            default=0.35,
        ))

        risk_mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
        if include_refresh_mask:
            refresh_mask = _state_tensor('anchor_refresh_mask')
            if refresh_mask is not None:
                risk_mask |= refresh_mask.bool()

        semantic_distance = _state_tensor('semantic_distance')
        if semantic_distance is not None:
            risk_mask |= semantic_distance > (semantic_distance_threshold * semantic_scale)

        surface_distance = _state_tensor('surface_distance')
        if surface_distance is not None:
            risk_mask |= surface_distance > (surface_distance_threshold * surface_scale)

        confidence = _state_tensor('anchor_confidence')
        if confidence is not None:
            risk_mask |= confidence < min(0.999, confidence_threshold + confidence_margin)

        anchor_weights = _state_tensor('anchor_weights')
        if anchor_weights is not None and anchor_weights.ndim == 2 and anchor_weights.shape[1] >= 2:
            top2 = torch.topk(anchor_weights, k=2, dim=-1).values
            risk_mask |= (top2[:, 0] - top2[:, 1]) < (weight_gap_threshold * weight_gap_scale)

        lineage_offender_mask = self._binding_lineage_offender_mask(
            binding_state,
            point_count,
            device,
            iteration=iteration,
        )
        if torch.is_tensor(lineage_offender_mask) and lineage_offender_mask.shape[0] == point_count:
            risk_mask |= lineage_offender_mask.to(device=device, dtype=torch.bool)

        final_mask = gate_mask & risk_mask
        if debug_verbose:
            print(
                '[GaussianModel] binding boundary-arm risk mask '
                f'iter={iteration} points={point_count} gate={int(gate_mask.sum().item())} '
                f'risk={int(risk_mask.sum().item())} final={int(final_mask.sum().item())} '
                f'arm_only={int(bool(arm_only))} boundary_only={int(bool(boundary_only))} '
                f'include_refresh={int(bool(include_refresh_mask))} '
                f'arm_gate_mode={arm_gate_mode}'
            )
        return final_mask

    def _attenuate_densified_children(
        self,
        new_binding_state,
        new_boundary_tags,
        new_opacity,
        new_scaling,
        new_features_dc,
        new_features_rest,
        new_boundary_opacity_residual,
        new_boundary_scaling_residual,
        iteration=0,
        extra_risk_mask=None,
        refresh_risk_mask=None,
    ):
        if new_opacity.shape[0] == 0:
            return (
                new_binding_state,
                new_opacity,
                new_scaling,
                new_features_dc,
                new_features_rest,
                new_boundary_opacity_residual,
                new_boundary_scaling_residual,
            )

        point_count = new_opacity.shape[0]
        device = new_opacity.device
        use_child_risk_mask = bool(self.cfg.get('binding_densify_child_attenuate_enable', False))
        attenuation_mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
        if use_child_risk_mask:
            attenuation_mask |= self._binding_boundary_arm_risk_mask(
                new_binding_state,
                point_count,
                device,
                iteration=iteration,
                boundary_tags=new_boundary_tags,
                boundary_threshold=float(resolve_schedule_value(
                    iteration,
                    self.cfg.get('binding_densify_child_boundary_threshold', 0.05),
                    default=0.05,
                )),
                arm_only=bool(self.cfg.get('binding_densify_child_arm_only', True)),
                semantic_scale=float(resolve_schedule_value(
                    iteration,
                    self.cfg.get('binding_densify_child_semantic_distance_scale', 0.85),
                    default=0.85,
                )),
                surface_scale=float(resolve_schedule_value(
                    iteration,
                    self.cfg.get('binding_densify_child_surface_distance_scale', 0.85),
                    default=0.85,
                )),
                confidence_margin=float(resolve_schedule_value(
                    iteration,
                    self.cfg.get('binding_densify_child_confidence_margin', 0.06),
                    default=0.06,
                )),
                weight_gap_scale=float(resolve_schedule_value(
                    iteration,
                    self.cfg.get('binding_densify_child_weight_gap_scale', 1.0),
                    default=1.0,
                )),
                include_refresh_mask=True,
            )
        if torch.is_tensor(extra_risk_mask) and extra_risk_mask.shape[0] == point_count:
            attenuation_mask |= extra_risk_mask.to(device=device, dtype=torch.bool)

        immediate_refresh_mask = None
        if torch.is_tensor(refresh_risk_mask) and refresh_risk_mask.shape[0] == point_count:
            immediate_refresh_mask = refresh_risk_mask.to(device=device, dtype=torch.bool)

        mark_refresh_from_attenuation = bool(
            self.cfg.get('binding_densify_child_mark_refresh_from_attenuation', False)
        )
        mark_risky_from_attenuation = bool(
            self.cfg.get('binding_densify_child_mark_risky_from_attenuation', False)
        )
        refresh_assignment_mask = immediate_refresh_mask
        if refresh_assignment_mask is None and mark_refresh_from_attenuation and bool(attenuation_mask.any().item()):
            refresh_assignment_mask = attenuation_mask.clone()
        risky_child_assignment_mask = immediate_refresh_mask
        if risky_child_assignment_mask is None and mark_risky_from_attenuation and bool(attenuation_mask.any().item()):
            risky_child_assignment_mask = attenuation_mask.clone()

        if (
            not bool(attenuation_mask.any().item())
            and refresh_assignment_mask is None
            and risky_child_assignment_mask is None
        ):
            return (
                new_binding_state,
                new_opacity,
                new_scaling,
                new_features_dc,
                new_features_rest,
                new_boundary_opacity_residual,
                new_boundary_scaling_residual,
            )

        opacity_factor = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_child_opacity_factor', 0.55),
            default=0.55,
        ))
        scale_factor = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_child_scale_factor', 0.88),
            default=0.88,
        ))
        feature_dc_factor = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_child_feature_dc_factor', 0.75),
            default=0.75,
        ))
        feature_rest_factor = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_child_feature_rest_factor', 0.65),
            default=0.65,
        ))

        attenuated_opacity = new_opacity
        attenuated_scaling = new_scaling
        attenuated_features_dc = new_features_dc
        attenuated_features_rest = new_features_rest
        attenuated_boundary_opacity_residual = new_boundary_opacity_residual
        attenuated_boundary_scaling_residual = new_boundary_scaling_residual

        if bool(attenuation_mask.any().item()):
            attenuated_opacity = new_opacity.clone()
            actual_opacity = self.opacity_activation(attenuated_opacity[attenuation_mask]) * opacity_factor
            actual_opacity = actual_opacity.clamp(1e-4, 1.0 - 1e-4)
            attenuated_opacity[attenuation_mask] = self.inverse_opacity_activation(actual_opacity)

            attenuated_scaling = new_scaling.clone()
            actual_scaling = self.scaling_activation(attenuated_scaling[attenuation_mask]) * scale_factor
            actual_scaling = actual_scaling.clamp_min(1e-6)
            attenuated_scaling[attenuation_mask] = self.scaling_inverse_activation(actual_scaling)

            attenuated_features_dc = new_features_dc.clone()
            attenuated_features_dc[attenuation_mask] = attenuated_features_dc[attenuation_mask] * feature_dc_factor

            attenuated_features_rest = new_features_rest.clone()
            attenuated_features_rest[attenuation_mask] = attenuated_features_rest[attenuation_mask] * feature_rest_factor

            attenuated_boundary_opacity_residual = new_boundary_opacity_residual.clone()
            attenuated_boundary_opacity_residual[attenuation_mask] = 0.0

            attenuated_boundary_scaling_residual = new_boundary_scaling_residual.clone()
            attenuated_boundary_scaling_residual[attenuation_mask] = 0.0

        updated_binding_state = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in new_binding_state.items()
        }
        if risky_child_assignment_mask is not None:
            updated_binding_state['densify_risky_child_mask'] = risky_child_assignment_mask.clone()
        if refresh_assignment_mask is not None:
            refresh_mask = updated_binding_state.get('anchor_refresh_mask', None)
            if not torch.is_tensor(refresh_mask) or refresh_mask.shape[0] != point_count:
                refresh_mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
            else:
                refresh_mask = refresh_mask.to(device=device, dtype=torch.bool)
            refresh_mask[refresh_assignment_mask] = True
            updated_binding_state['anchor_refresh_mask'] = refresh_mask

        if bool(self.cfg.get('binding_densify_child_verbose', False)):
            attenuated_count = int(attenuation_mask.sum().item())
            refresh_count = int(refresh_assignment_mask.sum().item()) if refresh_assignment_mask is not None else 0
            if attenuated_count > 0:
                print(
                    f'[GaussianModel] attenuated {attenuated_count} newborn children '
                    f'at iter {iteration}'
                )
            if refresh_count > 0:
                print(
                    f'[GaussianModel] marked {refresh_count} newborn children for immediate refresh '
                    f'at iter {iteration}'
                )

        return (
            updated_binding_state,
            attenuated_opacity,
            attenuated_scaling,
            attenuated_features_dc,
            attenuated_features_rest,
            attenuated_boundary_opacity_residual,
            attenuated_boundary_scaling_residual,
        )

    def _apply_directional_split_to_risky_children(self, parent_xyz, child_xyz, binding_state, risk_mask, iteration=0):
        if child_xyz.shape[0] == 0 or not torch.is_tensor(risk_mask) or not bool(risk_mask.any().item()):
            return child_xyz
        if not binding_state:
            return child_xyz

        anchor_normal = binding_state.get('anchor_normal', None)
        if not torch.is_tensor(anchor_normal) or anchor_normal.shape != child_xyz.shape:
            return child_xyz

        device = child_xyz.device
        risk_mask = risk_mask.to(device=device, dtype=torch.bool)
        normals = F.normalize(anchor_normal.to(device=device), dim=-1)
        delta = child_xyz - parent_xyz
        delta_normal_mag = torch.sum(delta * normals, dim=-1, keepdim=True)
        delta_normal = delta_normal_mag * normals
        delta_tangent = delta - delta_normal

        normal_factor = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_risky_child_normal_factor', 0.18),
            default=0.18,
        ))
        outward_normal_factor = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_risky_child_outward_normal_factor', 0.08),
            default=0.08,
        ))
        tangent_factor = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_risky_child_tangent_factor', 1.0),
            default=1.0,
        ))
        plane_pull = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_risky_child_plane_pull', 0.30),
            default=0.30,
        ))

        adjusted_delta_normal = delta_normal * normal_factor
        outward_mask = delta_normal_mag > 0
        adjusted_delta_normal[outward_mask.expand_as(adjusted_delta_normal)] = (
            delta_normal[outward_mask.expand_as(delta_normal)] * outward_normal_factor
        )

        adjusted_delta = delta_tangent * tangent_factor + adjusted_delta_normal
        adjusted_xyz = child_xyz.clone()
        adjusted_xyz[risk_mask] = parent_xyz[risk_mask] + adjusted_delta[risk_mask]

        normal_offset = binding_state.get('normal_offset', None)
        if plane_pull > 0.0 and torch.is_tensor(normal_offset) and normal_offset.shape == child_xyz.shape:
            adjusted_xyz[risk_mask] = adjusted_xyz[risk_mask] - normal_offset.to(device=device)[risk_mask] * plane_pull
        return adjusted_xyz

    def _predict_densified_child_risk_mask(
        self,
        new_binding_state,
        new_boundary_tags,
        iteration=0,
        extra_risk_mask=None,
    ):
        if not new_binding_state:
            return None
        point_count = 0
        for value in new_binding_state.values():
            if torch.is_tensor(value):
                point_count = int(value.shape[0])
                device = value.device
                break
        if point_count <= 0:
            return None

        risk_mask = self._binding_boundary_arm_risk_mask(
            new_binding_state,
            point_count,
            device,
            iteration=iteration,
            boundary_tags=new_boundary_tags,
            boundary_threshold=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_child_boundary_threshold', 0.05),
                default=0.05,
            )),
            arm_only=bool(self.cfg.get('binding_densify_child_arm_only', True)),
            semantic_scale=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_child_semantic_distance_scale', 0.85),
                default=0.85,
            )),
            surface_scale=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_child_surface_distance_scale', 0.85),
                default=0.85,
            )),
            confidence_margin=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_child_confidence_margin', 0.06),
                default=0.06,
            )),
            weight_gap_scale=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_child_weight_gap_scale', 1.0),
                default=1.0,
            )),
            include_refresh_mask=True,
        )

        if torch.is_tensor(extra_risk_mask) and extra_risk_mask.shape[0] == point_count:
            risk_mask |= extra_risk_mask.to(device=device, dtype=torch.bool)

        if bool(self.cfg.get('binding_densify_predictive_directional_split_verbose', False)):
            risky_count = int(risk_mask.sum().item())
            if risky_count > 0:
                print(
                    f'[GaussianModel] predictive risky newborn children {risky_count} '
                    f'at iter {iteration}'
                )
        return risk_mask

    def _predict_densified_child_immediate_refresh_mask(
        self,
        new_binding_state,
        new_boundary_tags,
        iteration=0,
        extra_risk_mask=None,
    ):
        if not bool(self.cfg.get('binding_densify_child_immediate_refresh_enable', True)):
            return None
        if not new_binding_state:
            return None

        point_count = 0
        for value in new_binding_state.values():
            if torch.is_tensor(value):
                point_count = int(value.shape[0])
                device = value.device
                break
        if point_count <= 0:
            return None

        refresh_mask = self._binding_boundary_arm_risk_mask(
            new_binding_state,
            point_count,
            device,
            iteration=iteration,
            boundary_tags=new_boundary_tags,
            boundary_only=bool(self.cfg.get('binding_densify_child_immediate_refresh_boundary_only', False)),
            boundary_threshold=float(resolve_schedule_value(
                iteration,
                self.cfg.get(
                    'binding_densify_child_immediate_refresh_boundary_threshold',
                    self.cfg.get('binding_densify_child_boundary_threshold', 0.05),
                ),
                default=0.05,
            )),
            arm_only=bool(self.cfg.get(
                'binding_densify_child_immediate_refresh_arm_only',
                self.cfg.get('binding_densify_child_arm_only', True),
            )),
            semantic_scale=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_child_immediate_refresh_semantic_distance_scale', 1.15),
                default=1.15,
            )),
            surface_scale=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_child_immediate_refresh_surface_distance_scale', 1.15),
                default=1.15,
            )),
            confidence_margin=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_child_immediate_refresh_confidence_margin', -0.05),
                default=-0.05,
            )),
            weight_gap_scale=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_child_immediate_refresh_weight_gap_scale', 0.8),
                default=0.8,
            )),
            include_refresh_mask=False,
            arm_gate_mode=self.cfg.get(
                'binding_densify_child_immediate_refresh_arm_gate_mode',
                'source_or_current',
            ),
        )

        applied_extra_risk_mask = None
        if (
            torch.is_tensor(extra_risk_mask)
            and extra_risk_mask.shape[0] == point_count
            and bool(self.cfg.get('binding_densify_child_immediate_refresh_include_extra_risk_mask', False))
        ):
            applied_extra_risk_mask = extra_risk_mask.to(device=device, dtype=torch.bool)
            refresh_mask |= applied_extra_risk_mask

        refresh_score = self._binding_risk_score(
            new_binding_state,
            point_count,
            device,
            iteration=iteration,
            include_refresh_mask=False,
        )
        if torch.is_tensor(applied_extra_risk_mask) and applied_extra_risk_mask.shape[0] == point_count:
            refresh_score = refresh_score + applied_extra_risk_mask.float() * float(
                resolve_schedule_value(
                    iteration,
                    self.cfg.get('binding_densify_child_immediate_refresh_extra_risk_score_bonus', 0.35),
                    default=0.35,
                )
            )
        refresh_mask, local_lineage_score = self._restrict_newborn_refresh_to_local_lineage_band(
            new_binding_state,
            refresh_mask,
            new_boundary_tags,
            iteration=iteration,
        )
        if local_lineage_score is not None and local_lineage_score.shape[0] == point_count:
            refresh_score = refresh_score + local_lineage_score.to(device=device, dtype=torch.float32) * float(
                resolve_schedule_value(
                    iteration,
                    self.cfg.get('binding_densify_child_immediate_refresh_local_lineage_score_bonus', 0.75),
                    default=0.75,
                )
            )
        refresh_mask = self._cap_mask_by_group_score(
            refresh_mask,
            new_binding_state.get('densify_root_lineage_id', None),
            refresh_score,
            max_points_per_group=int(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_child_immediate_refresh_max_points_per_lineage', 48),
                default=48,
            )),
        )
        refresh_mask = self._cap_mask_by_score(
            refresh_mask,
            score=refresh_score,
            max_points=int(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_child_immediate_refresh_max_points', 512),
                default=512,
            )),
            max_ratio=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_child_immediate_refresh_max_ratio', 0.03),
                default=0.03,
            )),
            min_score=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_child_immediate_refresh_min_score', 0.10),
                default=0.10,
            )),
        )

        if bool(self.cfg.get('binding_densify_debug_verbose', False)):
            refresh_boundary_threshold = float(resolve_schedule_value(
                iteration,
                self.cfg.get(
                    'binding_densify_child_immediate_refresh_boundary_threshold',
                    self.cfg.get('binding_densify_child_boundary_threshold', 0.05),
                ),
                default=0.05,
            ))
            print(
                '[GaussianModel] immediate-refresh prediction '
                f'iter={iteration} points={point_count} refresh={int(refresh_mask.sum().item())} '
                f'boundary_threshold={refresh_boundary_threshold:.4f} '
                f'extra_risk={int(applied_extra_risk_mask.sum().item()) if torch.is_tensor(applied_extra_risk_mask) else 0}'
            )
        if bool(self.cfg.get('binding_densify_child_immediate_refresh_verbose', False)):
            refresh_count = int(refresh_mask.sum().item())
            if refresh_count > 0:
                print(
                    f'[GaussianModel] immediate-refresh newborn children {refresh_count} '
                    f'at iter {iteration}'
                )
        return refresh_mask

    def _restrict_newborn_refresh_to_local_lineage_band(
        self,
        new_binding_state,
        refresh_mask,
        new_boundary_tags,
        iteration=0,
    ):
        if refresh_mask is None or refresh_mask.numel() == 0 or not bool(refresh_mask.any().item()):
            return refresh_mask, None
        if not bool(self.cfg.get('binding_densify_child_immediate_refresh_local_lineage_enable', True)):
            return refresh_mask, None
        if not new_binding_state:
            return refresh_mask, None

        point_count = refresh_mask.shape[0]
        device = refresh_mask.device

        def _state_tensor(key):
            value = new_binding_state.get(key, None)
            if torch.is_tensor(value) and value.shape[0] == point_count:
                return value.to(device=device)
            return None

        local_mask = refresh_mask.clone()
        local_score = torch.zeros((point_count,), dtype=torch.float32, device=device)
        source_firewall_kept = int(local_mask.sum().item())
        boundary_band_kept = 0
        age_kept = 0
        parent_locality_kept = 0
        pre_boundary_gate_kept = int(local_mask.sum().item())

        child_boundary_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get(
                'binding_densify_child_immediate_refresh_boundary_threshold',
                self.cfg.get('binding_densify_child_boundary_threshold', 0.05),
            ),
            default=0.05,
        ))
        parent_boundary_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get(
                'binding_densify_child_immediate_refresh_parent_boundary_threshold',
                child_boundary_threshold,
            ),
            default=child_boundary_threshold,
        ))

        boundary_band_mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
        boundary_score = torch.zeros((point_count,), dtype=torch.float32, device=device)
        child_boundary_support_mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
        direct_parent_boundary_kept = 0
        root_parent_boundary_kept = 0
        if torch.is_tensor(new_boundary_tags) and new_boundary_tags.shape[0] == point_count:
            child_boundary = new_boundary_tags.to(device=device, dtype=torch.float32).reshape(-1)
            child_boundary_support_mask = child_boundary > child_boundary_threshold
            boundary_band_mask |= child_boundary_support_mask
            boundary_score = torch.maximum(boundary_score, child_boundary)

        source_parent_index = _state_tensor('densify_parent_index')
        source_root_parent_index = _state_tensor('densify_root_parent_index')
        source_firewall_mask = self._source_joint_firewall_mask_from_parent_indices(
            point_count,
            device,
            source_parent_index=source_parent_index,
            source_root_parent_index=source_root_parent_index,
            iteration=iteration,
            require_consensus=bool(self.cfg.get(
                'binding_densify_child_immediate_refresh_require_source_consensus',
                True,
            )),
        )
        if source_firewall_mask is not None:
            local_mask &= source_firewall_mask
        source_firewall_kept = int(local_mask.sum().item())
        pre_boundary_gate_kept = source_firewall_kept
        current_xyz = self.get_xyz.detach() if self.get_xyz.numel() > 0 else None
        current_scaling = self.get_scaling.detach() if self.get_xyz.numel() > 0 else None
        current_boundary_tags = self.get_boundary_tags()
        boundary_signal_source = 'tag'
        if (
            current_xyz is None
            or not torch.is_tensor(current_boundary_tags)
            or current_boundary_tags.shape[0] != current_xyz.shape[0]
        ):
            current_boundary_tags = self.get_live_boundary_score_state()
            boundary_signal_source = 'live_score' if torch.is_tensor(current_boundary_tags) else 'none'
        live_boundary_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get(
                'binding_densify_child_immediate_refresh_live_boundary_threshold',
                0.75,
            ),
            default=0.75,
        ))
        active_parent_boundary_threshold = (
            live_boundary_threshold if boundary_signal_source == 'live_score' else parent_boundary_threshold
        )
        use_newborn_local_support = bool(
            self.cfg.get('binding_densify_child_immediate_refresh_newborn_local_support_enable', True)
        ) and boundary_signal_source != 'tag'
        newborn_local_support_candidates = 0
        newborn_local_support_kept = 0

        def _parent_boundary_mask(parent_index):
            parent_mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
            parent_score = torch.zeros((point_count,), dtype=torch.float32, device=device)
            if use_newborn_local_support:
                return parent_mask, parent_score
            if (
                current_xyz is None
                or not torch.is_tensor(parent_index)
                or parent_index.shape[0] != point_count
                or not torch.is_tensor(current_boundary_tags)
                or current_boundary_tags.shape[0] != current_xyz.shape[0]
            ):
                return parent_mask, parent_score
            valid_parent = (parent_index >= 0) & (parent_index < current_xyz.shape[0])
            if not bool(valid_parent.any().item()):
                return parent_mask, parent_score
            boundary_value = current_boundary_tags.to(device=device, dtype=torch.float32).reshape(-1)
            parent_boundary = torch.zeros((point_count,), dtype=torch.float32, device=device)
            parent_boundary[valid_parent] = boundary_value[parent_index[valid_parent].long()]
            parent_mask = parent_boundary > active_parent_boundary_threshold
            return parent_mask, parent_boundary

        child_xyz = _state_tensor('bound_xyz')
        distance_score = torch.zeros((point_count,), dtype=torch.float32, device=device)
        require_parent_locality = bool(self.cfg.get('binding_densify_child_immediate_refresh_require_parent_locality', True))
        parent_radius_scale = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_child_immediate_refresh_parent_radius_scale', 2.5),
            default=2.5,
        ))
        parent_radius_bias = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_child_immediate_refresh_parent_radius_bias', 0.006),
            default=0.006,
        ))

        def _parent_locality_support(parent_index):
            parent_mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
            parent_score = torch.zeros((point_count,), dtype=torch.float32, device=device)
            if (
                current_xyz is None
                or current_scaling is None
                or not torch.is_tensor(parent_index)
                or parent_index.shape[0] != point_count
                or not torch.is_tensor(child_xyz)
                or child_xyz.shape != (point_count, 3)
            ):
                return parent_mask, parent_score
            valid_parent = (parent_index >= 0) & (parent_index < current_xyz.shape[0])
            if not bool(valid_parent.any().item()):
                return parent_mask, parent_score
            parent_xyz = current_xyz[parent_index[valid_parent].long()].to(device=device)
            parent_scale = current_scaling[parent_index[valid_parent].long()].to(device=device).amax(dim=-1)
            parent_radius = parent_scale * parent_radius_scale
            parent_radius = (parent_radius + parent_radius_bias).clamp_min(1.0e-6)
            child_parent_dist = torch.norm(child_xyz[valid_parent].to(device=device) - parent_xyz, dim=-1)
            local_valid = child_parent_dist <= parent_radius
            parent_mask[valid_parent] = local_valid
            parent_score[valid_parent] = (1.0 - child_parent_dist / parent_radius).clamp(0.0, 1.0)
            return parent_mask, parent_score

        parent_boundary_mask, parent_boundary_score = _parent_boundary_mask(source_parent_index)
        root_parent_boundary_mask, root_parent_boundary_score = _parent_boundary_mask(source_root_parent_index)
        parent_distance_mask, parent_distance_score = _parent_locality_support(source_parent_index)
        root_parent_distance_mask, root_parent_distance_score = _parent_locality_support(source_root_parent_index)

        direct_parent_boundary_support = parent_boundary_mask.clone()
        root_parent_boundary_support = root_parent_boundary_mask.clone()
        root_boundary_fallback_enable = bool(
            self.cfg.get('binding_densify_child_immediate_refresh_root_boundary_fallback_enable', True)
        )
        root_boundary_fallback_only_when_direct_missing = bool(
            self.cfg.get(
                'binding_densify_child_immediate_refresh_root_boundary_fallback_only_when_direct_missing',
                True,
            )
        )
        if require_parent_locality:
            direct_parent_boundary_support &= parent_distance_mask
            root_parent_boundary_support &= root_parent_distance_mask
        if not root_boundary_fallback_enable:
            root_parent_boundary_support &= False
        elif root_boundary_fallback_only_when_direct_missing:
            root_parent_boundary_support &= ~direct_parent_boundary_support

        boundary_band_mask |= direct_parent_boundary_support | root_parent_boundary_support
        boundary_score = torch.maximum(
            boundary_score,
            torch.where(
                direct_parent_boundary_support,
                parent_boundary_score,
                torch.zeros_like(parent_boundary_score),
            ),
        )
        boundary_score = torch.maximum(
            boundary_score,
            torch.where(
                root_parent_boundary_support,
                root_parent_boundary_score,
                torch.zeros_like(root_parent_boundary_score),
            ),
        )
        direct_parent_boundary_kept = int(direct_parent_boundary_support.sum().item())
        root_parent_boundary_kept = int(root_parent_boundary_support.sum().item())

        preferred_parent_distance_mask = parent_distance_mask
        preferred_parent_distance_score = parent_distance_score
        if (
            not torch.is_tensor(source_parent_index)
            or source_parent_index.shape[0] != point_count
            or not bool(((source_parent_index >= 0) & (source_parent_index < self.get_xyz.shape[0])).any().item())
        ):
            preferred_parent_distance_mask = root_parent_distance_mask
            preferred_parent_distance_score = root_parent_distance_score

        newborn_local_support_mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
        newborn_local_support_score = torch.zeros((point_count,), dtype=torch.float32, device=device)
        if use_newborn_local_support:
            newborn_risk_score = self._binding_risk_score(
                new_binding_state,
                point_count,
                device,
                iteration=iteration,
                include_refresh_mask=False,
            )
            newborn_local_support_mask = refresh_mask.clone()
            if source_firewall_mask is not None:
                newborn_local_support_mask &= source_firewall_mask
            if bool(self.cfg.get(
                'binding_densify_child_immediate_refresh_newborn_local_support_require_parent_locality',
                require_parent_locality,
            )):
                newborn_local_support_mask &= preferred_parent_distance_mask
            if bool(self.cfg.get(
                'binding_densify_child_immediate_refresh_newborn_local_support_require_child_boundary',
                False,
            )):
                newborn_local_support_mask &= child_boundary_support_mask
            min_newborn_local_support_risk = float(resolve_schedule_value(
                iteration,
                self.cfg.get(
                    'binding_densify_child_immediate_refresh_newborn_local_support_min_risk_score',
                    0.0,
                ),
                default=0.0,
            ))
            if min_newborn_local_support_risk > 0.0:
                newborn_local_support_mask &= newborn_risk_score >= min_newborn_local_support_risk
            newborn_local_support_candidates = int(newborn_local_support_mask.sum().item())
            if newborn_local_support_candidates > 0:
                newborn_local_support_priority = newborn_risk_score.clone()
                newborn_local_support_priority = newborn_local_support_priority + (
                    preferred_parent_distance_score
                    * float(resolve_schedule_value(
                        iteration,
                        self.cfg.get(
                            'binding_densify_child_immediate_refresh_newborn_local_support_locality_bonus',
                            0.75,
                        ),
                        default=0.75,
                    ))
                )
                if torch.is_tensor(new_boundary_tags) and new_boundary_tags.shape[0] == point_count:
                    newborn_local_support_priority = newborn_local_support_priority + (
                        new_boundary_tags.to(device=device, dtype=torch.float32).reshape(-1).clamp(0.0, 1.0)
                        * float(resolve_schedule_value(
                            iteration,
                            self.cfg.get(
                                'binding_densify_child_immediate_refresh_newborn_local_support_child_boundary_bonus',
                                0.5,
                            ),
                            default=0.5,
                        ))
                    )
                newborn_local_support_mask = self._cap_mask_by_group_score(
                    newborn_local_support_mask,
                    new_binding_state.get('densify_root_lineage_id', None),
                    newborn_local_support_priority,
                    max_points_per_group=int(resolve_schedule_value(
                        iteration,
                        self.cfg.get(
                            'binding_densify_child_immediate_refresh_newborn_local_support_max_points_per_lineage',
                            8,
                        ),
                        default=8,
                    )),
                )
                newborn_local_support_mask = self._cap_mask_by_score(
                    newborn_local_support_mask,
                    score=newborn_local_support_priority,
                    max_points=int(resolve_schedule_value(
                        iteration,
                        self.cfg.get(
                            'binding_densify_child_immediate_refresh_newborn_local_support_max_points',
                            192,
                        ),
                        default=192,
                    )),
                    max_ratio=float(resolve_schedule_value(
                        iteration,
                        self.cfg.get(
                            'binding_densify_child_immediate_refresh_newborn_local_support_max_ratio',
                            0.015,
                        ),
                        default=0.015,
                    )),
                    min_score=None,
                )
                newborn_local_support_score = torch.where(
                    newborn_local_support_mask,
                    (
                        newborn_local_support_priority
                        / (1.0 + newborn_local_support_priority.abs())
                    ).clamp(0.0, 1.0),
                    torch.zeros_like(newborn_local_support_priority),
                )
                boundary_band_mask |= newborn_local_support_mask
                boundary_score = torch.maximum(boundary_score, newborn_local_support_score)
                newborn_local_support_kept = int(newborn_local_support_mask.sum().item())

        require_local_boundary_band = bool(
            self.cfg.get('binding_densify_child_immediate_refresh_require_local_boundary_band', True)
        )
        if require_local_boundary_band:
            # `require_local_boundary_band=true` should be fail-closed:
            # if the newborn and its source lineage expose no local boundary signal,
            # this point should not enter the immediate-refresh path.
            local_mask &= boundary_band_mask
        boundary_band_kept = int(local_mask.sum().item())

        max_child_age = int(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_child_immediate_refresh_max_child_age', 2),
            default=2,
        ))
        densify_birth_iter = _state_tensor('densify_birth_iter')
        if max_child_age >= 0 and torch.is_tensor(densify_birth_iter) and densify_birth_iter.shape[0] == point_count:
            child_age = max(iteration, 0) - densify_birth_iter.to(device=device, dtype=torch.long)
            local_mask &= child_age <= max_child_age
        age_kept = int(local_mask.sum().item())

        locality_support_mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
        locality_support_score = torch.zeros((point_count,), dtype=torch.float32, device=device)
        if bool(child_boundary_support_mask.any().item()):
            locality_support_mask |= preferred_parent_distance_mask
            locality_support_score = torch.maximum(locality_support_score, preferred_parent_distance_score)
        locality_support_mask |= direct_parent_boundary_support | root_parent_boundary_support
        locality_support_mask |= newborn_local_support_mask
        locality_support_score = torch.maximum(locality_support_score, parent_distance_score)
        locality_support_score = torch.maximum(locality_support_score, root_parent_distance_score)
        locality_support_score = torch.maximum(locality_support_score, newborn_local_support_score)
        distance_score = locality_support_score
        if require_parent_locality:
            # Parent-locality is now tied to the support path itself:
            # direct-parent boundary support needs direct locality, while
            # root fallback must also satisfy root-locality rather than
            # inheriting the direct parent's locality by accident.
            local_mask &= locality_support_mask
        parent_locality_kept = int(local_mask.sum().item())

        local_score = torch.maximum(local_score, boundary_score.clamp(0.0, 1.0))
        local_score = local_score + distance_score
        if bool(self.cfg.get('binding_densify_debug_verbose', False)):
            print(
                '[GaussianModel] immediate-refresh local-lineage band '
                f'iter={iteration} input={int(refresh_mask.sum().item())} '
                f'boundary_source={boundary_signal_source} '
                f'parent_boundary_threshold={active_parent_boundary_threshold:.4f} '
                f'firewall_kept={source_firewall_kept} '
                f'pre_boundary_gate={pre_boundary_gate_kept} '
                f'boundary_candidates={int(boundary_band_mask.sum().item())} '
                f'direct_boundary_kept={direct_parent_boundary_kept} '
                f'root_boundary_kept={root_parent_boundary_kept} '
                f'newborn_support_candidates={newborn_local_support_candidates} '
                f'newborn_support_kept={newborn_local_support_kept} '
                f'boundary_kept={boundary_band_kept} '
                f'age_kept={age_kept} '
                f'parent_locality_kept={parent_locality_kept} '
                f'output={int(local_mask.sum().item())}'
            )
        return local_mask, local_score

    def _preserve_split_parents_for_risky_boundary_points(self, selected_pts_mask, iteration=0):
        if not bool(self.cfg.get('binding_densify_keep_parent_for_risky_split_enable', True)):
            return torch.zeros_like(selected_pts_mask, dtype=torch.bool)
        if selected_pts_mask is None or selected_pts_mask.numel() == 0 or not bool(selected_pts_mask.any().item()):
            return torch.zeros_like(selected_pts_mask, dtype=torch.bool)

        binding_state = self.get_binding_state()
        if not binding_state:
            return torch.zeros_like(selected_pts_mask, dtype=torch.bool)

        point_count = selected_pts_mask.shape[0]
        device = selected_pts_mask.device
        preserve_mask = self._binding_boundary_arm_risk_mask(
            binding_state,
            point_count,
            device,
            iteration=iteration,
            boundary_tags=self.get_boundary_tag_state(),
            boundary_threshold=float(resolve_schedule_value(
                iteration,
                self.cfg.get(
                    'binding_densify_keep_parent_boundary_threshold',
                    self.cfg.get('binding_densify_boundary_threshold', 0.05),
                ),
                default=0.05,
            )),
            arm_only=bool(self.cfg.get('binding_densify_keep_parent_arm_only', True)),
            semantic_scale=float(resolve_schedule_value(
                iteration,
                self.cfg.get(
                    'binding_densify_keep_parent_semantic_distance_scale',
                    self.cfg.get('binding_densify_semantic_distance_scale', 0.8),
                ),
                default=0.8,
            )),
            surface_scale=float(resolve_schedule_value(
                iteration,
                self.cfg.get(
                    'binding_densify_keep_parent_surface_distance_scale',
                    self.cfg.get('binding_densify_surface_distance_scale', 0.8),
                ),
                default=0.8,
            )),
            confidence_margin=float(resolve_schedule_value(
                iteration,
                self.cfg.get(
                    'binding_densify_keep_parent_confidence_margin',
                    self.cfg.get('binding_densify_confidence_margin', 0.08),
                ),
                default=0.08,
            )),
            weight_gap_scale=float(resolve_schedule_value(
                iteration,
                self.cfg.get(
                    'binding_densify_keep_parent_weight_gap_scale',
                    self.cfg.get('binding_densify_weight_gap_scale', 1.0),
                ),
                default=1.0,
            )),
            include_refresh_mask=True,
        )
        preserve_mask &= selected_pts_mask
        if bool(self.cfg.get('binding_densify_keep_parent_verbose', False)) and bool(preserve_mask.any().item()):
            selected_count = int(selected_pts_mask.sum().item())
            preserve_count = int(preserve_mask.sum().item())
            print(
                f'[GaussianModel] preserving {preserve_count} / {selected_count} risky split parents '
                f'at iter {iteration}'
            )
        return preserve_mask

    def _filter_densify_candidates(self, selected_pts_mask, iteration=0):
        if not bool(self.cfg.get('binding_densify_risk_filter_enable', False)):
            return selected_pts_mask
        if selected_pts_mask is None or selected_pts_mask.numel() == 0:
            return selected_pts_mask

        start_iter = int(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_risk_filter_start_iter', 0),
            default=0,
        ))
        end_iter = int(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_risk_filter_end_iter', -1),
            default=-1,
        ))
        if iteration < start_iter:
            return selected_pts_mask
        if end_iter >= 0 and iteration > end_iter:
            return selected_pts_mask

        binding_state = self.get_binding_state()
        if not binding_state:
            return selected_pts_mask

        point_count = selected_pts_mask.shape[0]
        device = selected_pts_mask.device

        def _state_tensor(key):
            value = binding_state.get(key, None)
            if torch.is_tensor(value) and value.shape[0] == point_count:
                return value.to(device=device)
            return None

        boundary_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_boundary_threshold', 0.05),
            default=0.05,
        ))
        semantic_scale = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_semantic_distance_scale', 0.8),
            default=0.8,
        ))
        surface_scale = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_surface_distance_scale', 0.8),
            default=0.8,
        ))
        confidence_margin = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_confidence_margin', 0.08),
            default=0.08,
        ))
        weight_gap_scale = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_weight_gap_scale', 1.0),
            default=1.0,
        ))

        semantic_distance_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_refresh_semantic_distance_threshold', 0.03),
            default=0.03,
        ))
        surface_distance_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_refresh_surface_distance_threshold', 0.012),
            default=0.012,
        ))
        confidence_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_refresh_confidence_threshold', 0.6),
            default=0.6,
        ))
        weight_gap_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_refresh_weight_gap_threshold', 0.35),
            default=0.35,
        ))

        gate_mask = torch.ones_like(selected_pts_mask, dtype=torch.bool)
        require_boundary_tag = bool(self.cfg.get('binding_densify_require_boundary_tag', False))
        if bool(self.cfg.get('binding_densify_boundary_only', True)):
            boundary_tags = self.get_boundary_tag_state()
            if torch.is_tensor(boundary_tags) and boundary_tags.shape[0] == point_count:
                gate_mask &= boundary_tags.to(device=device).reshape(-1) > boundary_threshold
            elif require_boundary_tag:
                gate_mask &= False

        if bool(self.cfg.get('binding_densify_arm_only', True)):
            dominant_joint = _state_tensor('dominant_joint')
            if dominant_joint is None:
                return selected_pts_mask
            arm_joint_ids = torch.tensor([13, 14, 16, 17, 18, 19, 20, 21, 22, 23], device=device, dtype=dominant_joint.dtype)
            gate_mask &= (dominant_joint.unsqueeze(-1) == arm_joint_ids.unsqueeze(0)).any(dim=-1)

        base_risk_mask = torch.zeros_like(selected_pts_mask, dtype=torch.bool)
        if bool(self.cfg.get('binding_densify_block_refresh_points', True)):
            refresh_mask = _state_tensor('anchor_refresh_mask')
            if refresh_mask is not None:
                base_risk_mask |= refresh_mask.bool()

        semantic_distance = _state_tensor('semantic_distance')
        if semantic_distance is not None:
            base_risk_mask |= semantic_distance > (semantic_distance_threshold * semantic_scale)

        surface_distance = _state_tensor('surface_distance')
        if surface_distance is not None:
            base_risk_mask |= surface_distance > (surface_distance_threshold * surface_scale)

        confidence = _state_tensor('anchor_confidence')
        if confidence is not None:
            base_risk_mask |= confidence < min(0.999, confidence_threshold + confidence_margin)

        anchor_weights = _state_tensor('anchor_weights')
        if anchor_weights is not None and anchor_weights.ndim == 2 and anchor_weights.shape[1] >= 2:
            top2 = torch.topk(anchor_weights, k=2, dim=-1).values
            base_risk_mask |= (top2[:, 0] - top2[:, 1]) < (weight_gap_threshold * weight_gap_scale)

        lineage_block_mask = self._binding_lineage_offender_mask(
            binding_state,
            point_count,
            device,
            iteration=iteration,
        )
        if not torch.is_tensor(lineage_block_mask) or lineage_block_mask.shape[0] != point_count:
            lineage_block_mask = torch.zeros_like(selected_pts_mask, dtype=torch.bool)

        lineage_reentry_mask = torch.zeros_like(selected_pts_mask, dtype=torch.bool)
        if (
            bool(self.cfg.get('binding_densify_lineage_offender_allow_recent_newborn_reentry', True))
            and bool(lineage_block_mask.any().item())
        ):
            densify_birth_iter = _state_tensor('densify_birth_iter')
            if torch.is_tensor(densify_birth_iter) and densify_birth_iter.shape[0] == point_count:
                reentry_max_child_age = int(resolve_schedule_value(
                    iteration,
                    self.cfg.get('binding_densify_lineage_offender_reentry_max_child_age', 250),
                    default=250,
                ))
                child_age = max(iteration, 0) - densify_birth_iter.to(device=device, dtype=torch.long)
                recent_newborn_mask = child_age <= reentry_max_child_age
                reentry_arm_mask = self._binding_arm_gate_mask_from_state(
                    binding_state,
                    point_count,
                    device,
                    iteration=iteration,
                    arm_gate_mode=self.cfg.get(
                        'binding_densify_lineage_offender_reentry_arm_gate_mode',
                        'source_or_current',
                    ),
                    require_source_consensus=bool(self.cfg.get(
                        'binding_densify_child_immediate_refresh_arm_gate_require_source_consensus',
                        False,
                    )),
                )
                if reentry_arm_mask is None:
                    reentry_arm_mask = gate_mask
                lineage_reentry_mask = (
                    selected_pts_mask
                    & gate_mask
                    & lineage_block_mask
                    & recent_newborn_mask
                    & reentry_arm_mask.to(device=device, dtype=torch.bool)
                )

        effective_lineage_block_mask = lineage_block_mask & (~lineage_reentry_mask)
        block_mask = selected_pts_mask & gate_mask & (base_risk_mask | effective_lineage_block_mask)
        if not bool(block_mask.any().item()):
            return selected_pts_mask

        if bool(self.cfg.get('binding_densify_filter_verbose', False)):
            selected_count = int(selected_pts_mask.sum().item())
            blocked_count = int(block_mask.sum().item())
            if blocked_count > 0:
                log_msg = (
                    f'[GaussianModel] densify risk filter blocked {blocked_count} / '
                    f'{selected_count} candidates at iter {iteration}'
                )
                lineage_blocked_count = int((selected_pts_mask & gate_mask & effective_lineage_block_mask).sum().item())
                if lineage_blocked_count > 0:
                    log_msg += f' (lineage offender blocked {lineage_blocked_count})'
                lineage_reentry_count = int(lineage_reentry_mask.sum().item())
                if lineage_reentry_count > 0:
                    log_msg += f' (recent newborn reentry {lineage_reentry_count})'
                print(log_msg)
        return selected_pts_mask & (~block_mask)

    def _augment_clone_densify_candidates(self, selected_pts_mask, scene_extent, iteration=0):
        if not bool(self.cfg.get('binding_densify_clone_candidate_seed_enable', True)):
            return selected_pts_mask
        if selected_pts_mask is None or selected_pts_mask.numel() == 0:
            return selected_pts_mask

        start_iter = int(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_clone_candidate_seed_start_iter', 0),
            default=0,
        ))
        end_iter = int(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_clone_candidate_seed_end_iter', -1),
            default=-1,
        ))
        if iteration < start_iter:
            return selected_pts_mask
        if end_iter >= 0 and iteration > end_iter:
            return selected_pts_mask

        binding_state = self.get_binding_state()
        if not binding_state:
            return selected_pts_mask

        point_count = selected_pts_mask.shape[0]
        device = selected_pts_mask.device

        def _state_tensor(key):
            value = binding_state.get(key, None)
            if torch.is_tensor(value) and value.shape[0] == point_count:
                return value.to(device=device)
            return None

        densify_birth_iter = _state_tensor('densify_birth_iter')
        if densify_birth_iter is None:
            return selected_pts_mask

        child_age = max(iteration, 0) - densify_birth_iter.to(device=device, dtype=torch.long)
        min_child_age = int(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_clone_candidate_seed_min_child_age', 0),
            default=0,
        ))
        max_child_age = int(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_clone_candidate_seed_max_child_age', 600),
            default=600,
        ))
        recent_mask = child_age >= min_child_age
        if max_child_age >= 0:
            recent_mask &= child_age <= max_child_age
        if not bool(recent_mask.any().item()):
            return selected_pts_mask

        scale_mask = (
            torch.max(self.get_scaling, dim=1).values
            <= (self.percent_dense * scene_extent)
        )
        seed_mask = (~selected_pts_mask) & recent_mask & scale_mask
        if not bool(seed_mask.any().item()):
            return selected_pts_mask

        boundary_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get(
                'binding_densify_clone_candidate_seed_boundary_threshold',
                self.cfg.get(
                    'binding_densify_child_immediate_refresh_boundary_threshold',
                    self.cfg.get('binding_densify_child_boundary_threshold', 0.05),
                ),
            ),
            default=0.05,
        ))
        arm_gate_mode = self.cfg.get(
            'binding_densify_clone_candidate_seed_arm_gate_mode',
            self.cfg.get(
                'binding_densify_clone_immediate_refresh_parent_risk_arm_gate_mode',
                self.cfg.get(
                    'binding_densify_child_immediate_refresh_arm_gate_mode',
                    'source_or_current',
                ),
            ),
        )
        seed_mask &= self._binding_boundary_arm_risk_mask(
            binding_state,
            point_count,
            device,
            iteration=iteration,
            boundary_tags=self.get_boundary_tag_state(),
            boundary_only=bool(self.cfg.get(
                'binding_densify_clone_candidate_seed_boundary_only',
                False,
            )),
            boundary_threshold=boundary_threshold,
            arm_only=bool(self.cfg.get(
                'binding_densify_clone_candidate_seed_arm_only',
                True,
            )),
            semantic_scale=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_clone_candidate_seed_semantic_distance_scale', 1.0),
                default=1.0,
            )),
            surface_scale=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_clone_candidate_seed_surface_distance_scale', 1.0),
                default=1.0,
            )),
            confidence_margin=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_clone_candidate_seed_confidence_margin', 0.0),
                default=0.0,
            )),
            weight_gap_scale=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_clone_candidate_seed_weight_gap_scale', 1.0),
                default=1.0,
            )),
            include_refresh_mask=bool(self.cfg.get(
                'binding_densify_clone_candidate_seed_include_refresh_mask',
                True,
            )),
            arm_gate_mode=arm_gate_mode,
        )
        if not bool(seed_mask.any().item()):
            return selected_pts_mask

        seed_priority = self._binding_risk_score(
            binding_state,
            point_count,
            device,
            iteration=iteration,
            include_refresh_mask=bool(self.cfg.get(
                'binding_densify_clone_candidate_seed_include_refresh_mask',
                True,
            )),
        )

        if max_child_age != 0:
            recent_score = (
                1.0
                - child_age.float().clamp_min(0.0)
                / float(max(max_child_age, 1))
            ).clamp(0.0, 1.0)
            seed_priority = seed_priority + recent_score * float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_clone_candidate_seed_age_score_bonus', 0.25),
                default=0.25,
            ))

        source_firewall_mask = self._source_joint_firewall_mask_from_source_joints(
            source_parent_joint=_state_tensor('source_parent_joint'),
            source_root_parent_joint=_state_tensor('source_root_parent_joint'),
            device=device,
            iteration=iteration,
            allowed_joint_ids=self._allowed_newborn_firewall_joint_ids(iteration=iteration),
            require_consensus=bool(self.cfg.get(
                'binding_densify_clone_candidate_seed_require_source_consensus',
                False,
            )),
        )
        if source_firewall_mask is not None:
            seed_priority = seed_priority + source_firewall_mask.to(
                device=device,
                dtype=torch.float32,
            ) * float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_clone_candidate_seed_source_score_bonus', 0.25),
                default=0.25,
            ))

        seed_mask = self._cap_mask_by_group_score(
            seed_mask,
            self._binding_lineage_ids_from_state(binding_state, point_count, device),
            seed_priority,
            max_points_per_group=int(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_clone_candidate_seed_max_points_per_lineage', 24),
                default=24,
            )),
        )
        seed_mask = self._cap_mask_by_score(
            seed_mask,
            score=seed_priority,
            max_points=int(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_clone_candidate_seed_max_points', 256),
                default=256,
            )),
            max_ratio=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_clone_candidate_seed_max_ratio', 0.01),
                default=0.01,
            )),
            min_score=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_clone_candidate_seed_min_risk_score', 0.05),
                default=0.05,
            )),
        )
        if not bool(seed_mask.any().item()):
            return selected_pts_mask

        augmented_mask = selected_pts_mask | seed_mask
        verbose = bool(self.cfg.get('binding_densify_debug_verbose', False)) or bool(
            self.cfg.get('binding_densify_clone_candidate_seed_verbose', False)
        )
        if verbose:
            print(
                '[GaussianModel] clone candidate seed augmentation '
                f'iter={iteration} base={int(selected_pts_mask.sum().item())} '
                f'recent={int(recent_mask.sum().item())} '
                f'seed={int(seed_mask.sum().item())} '
                f'augmented={int(augmented_mask.sum().item())} '
                f'source_match={int(source_firewall_mask.sum().item()) if torch.is_tensor(source_firewall_mask) else 0}'
            )
        return augmented_mask

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

    def clear_boundary_tags(self):
        device = self._xyz.device if torch.is_tensor(self._xyz) and self._xyz.numel() > 0 else None
        self._boundary_tag = torch.empty(0, device=device)

    def has_offender_state(self):
        point_count = int(self.get_xyz.shape[0]) if torch.is_tensor(self._xyz) and self._xyz.ndim >= 2 else 0
        return (
            torch.is_tensor(self._offender_score_accum)
            and torch.is_tensor(self._offender_count_accum)
            and self._offender_score_accum.shape[0] == point_count
            and self._offender_count_accum.shape[0] == point_count
        )

    def get_offender_state(self):
        if not self.has_offender_state():
            return None, None
        return self._offender_score_accum, self._offender_count_accum

    def clear_offender_state(self):
        device = self._xyz.device if torch.is_tensor(self._xyz) and self._xyz.numel() > 0 else None
        self._offender_score_accum = torch.empty(0, device=device)
        self._offender_count_accum = torch.empty(0, device=device)

    def has_offender_refill_state(self):
        point_count = int(self.get_xyz.shape[0]) if torch.is_tensor(self._xyz) and self._xyz.ndim >= 2 else 0
        return (
            torch.is_tensor(self._offender_refill_score)
            and self._offender_refill_score.shape[0] == point_count
        )

    def get_offender_refill_score(self):
        if not self.has_offender_refill_state():
            return None
        return self._offender_refill_score

    def clear_offender_refill_state(self):
        device = self._xyz.device if torch.is_tensor(self._xyz) and self._xyz.numel() > 0 else None
        self._offender_refill_score = torch.empty(0, device=device)

    def has_lineage_offender_state(self):
        point_count = int(self.get_xyz.shape[0]) if torch.is_tensor(self._xyz) and self._xyz.ndim >= 2 else 0
        return (
            torch.is_tensor(self._lineage_offender_score_accum)
            and torch.is_tensor(self._lineage_offender_count_accum)
            and self._lineage_offender_score_accum.shape[0] == point_count
            and self._lineage_offender_count_accum.shape[0] == point_count
        )

    def get_lineage_offender_state(self):
        if not self.has_lineage_offender_state():
            return None, None
        return self._lineage_offender_score_accum, self._lineage_offender_count_accum

    def clear_lineage_offender_state(self):
        device = self._xyz.device if torch.is_tensor(self._xyz) and self._xyz.numel() > 0 else None
        self._lineage_offender_score_accum = torch.empty(0, device=device)
        self._lineage_offender_count_accum = torch.empty(0, device=device)

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

    def ensure_offender_state_matches_points(self, verbose=False):
        point_count = int(self.get_xyz.shape[0]) if torch.is_tensor(self._xyz) and self._xyz.ndim >= 2 else 0
        device = self._xyz.device if torch.is_tensor(self._xyz) and self._xyz.numel() > 0 else None
        changed = []

        if point_count <= 0:
            if torch.is_tensor(self._offender_score_accum) and self._offender_score_accum.numel() > 0:
                self._offender_score_accum = torch.empty(0, device=device)
                changed.append('offender_score_accum')
            if torch.is_tensor(self._offender_count_accum) and self._offender_count_accum.numel() > 0:
                self._offender_count_accum = torch.empty(0, device=device)
                changed.append('offender_count_accum')
            if torch.is_tensor(self._offender_refill_score) and self._offender_refill_score.numel() > 0:
                self._offender_refill_score = torch.empty(0, device=device)
                changed.append('offender_refill_score')
            if torch.is_tensor(self._lineage_offender_score_accum) and self._lineage_offender_score_accum.numel() > 0:
                self._lineage_offender_score_accum = torch.empty(0, device=device)
                changed.append('lineage_offender_score_accum')
            if torch.is_tensor(self._lineage_offender_count_accum) and self._lineage_offender_count_accum.numel() > 0:
                self._lineage_offender_count_accum = torch.empty(0, device=device)
                changed.append('lineage_offender_count_accum')
            return len(changed) > 0

        if not torch.is_tensor(self._offender_score_accum) or self._offender_score_accum.shape[0] != point_count:
            self._offender_score_accum = self._resize_pointwise_state(
                self._offender_score_accum,
                point_count,
                dtype=torch.float32,
                device=device,
            )
            changed.append('offender_score_accum')
        else:
            self._offender_score_accum = self._offender_score_accum.to(device=device, dtype=torch.float32)

        if not torch.is_tensor(self._offender_count_accum) or self._offender_count_accum.shape[0] != point_count:
            self._offender_count_accum = self._resize_pointwise_state(
                self._offender_count_accum,
                point_count,
                dtype=torch.float32,
                device=device,
            )
            changed.append('offender_count_accum')
        else:
            self._offender_count_accum = self._offender_count_accum.to(device=device, dtype=torch.float32)

        if not torch.is_tensor(self._offender_refill_score) or self._offender_refill_score.shape[0] != point_count:
            self._offender_refill_score = self._resize_pointwise_state(
                self._offender_refill_score,
                point_count,
                dtype=torch.float32,
                device=device,
            )
            changed.append('offender_refill_score')
        else:
            self._offender_refill_score = self._offender_refill_score.to(device=device, dtype=torch.float32)

        if not torch.is_tensor(self._lineage_offender_score_accum) or self._lineage_offender_score_accum.shape[0] != point_count:
            self._lineage_offender_score_accum = self._resize_pointwise_state(
                self._lineage_offender_score_accum,
                point_count,
                dtype=torch.float32,
                device=device,
            )
            changed.append('lineage_offender_score_accum')
        else:
            self._lineage_offender_score_accum = self._lineage_offender_score_accum.to(device=device, dtype=torch.float32)

        if not torch.is_tensor(self._lineage_offender_count_accum) or self._lineage_offender_count_accum.shape[0] != point_count:
            self._lineage_offender_count_accum = self._resize_pointwise_state(
                self._lineage_offender_count_accum,
                point_count,
                dtype=torch.float32,
                device=device,
            )
            changed.append('lineage_offender_count_accum')
        else:
            self._lineage_offender_count_accum = self._lineage_offender_count_accum.to(device=device, dtype=torch.float32)

        if verbose and changed:
            print(
                '[GaussianModel] offender state resynced for '
                f'{point_count} points: {", ".join(changed)}'
            )
        return len(changed) > 0

    def accumulate_offender_scores(self, offender_score, offender_count=None):
        if offender_score is None:
            return
        self.ensure_offender_state_matches_points(verbose=False)
        score = offender_score.detach().reshape(-1).float().clamp_min(0.0)
        if score.shape[0] != self.get_xyz.shape[0]:
            raise ValueError(f'offender_score shape mismatch: got {score.shape[0]}, expected {self.get_xyz.shape[0]}')
        if offender_count is None:
            count = (score > 0).float()
        else:
            count = offender_count.detach().reshape(-1).float().clamp_min(0.0)
            if count.shape[0] != self.get_xyz.shape[0]:
                raise ValueError(f'offender_count shape mismatch: got {count.shape[0]}, expected {self.get_xyz.shape[0]}')
        self._offender_score_accum = self._offender_score_accum + score.to(self._offender_score_accum.device)
        self._offender_count_accum = self._offender_count_accum + count.to(self._offender_count_accum.device)

    def set_offender_refill_score(self, refill_score, blend_mode='max', decay=1.0):
        self.ensure_offender_state_matches_points(verbose=False)
        if refill_score is None:
            if decay < 1.0 and self.has_offender_refill_state():
                self._offender_refill_score = self._offender_refill_score * float(max(decay, 0.0))
            return

        score = refill_score.detach().reshape(-1).float().clamp(0.0, 1.0)
        if score.shape[0] != self.get_xyz.shape[0]:
            raise ValueError(f'offender_refill_score shape mismatch: got {score.shape[0]}, expected {self.get_xyz.shape[0]}')

        if decay < 1.0:
            self._offender_refill_score = self._offender_refill_score * float(max(decay, 0.0))

        score = score.to(self._offender_refill_score.device)
        blend_mode = str(blend_mode).lower()
        if blend_mode == 'overwrite':
            self._offender_refill_score = score
        elif blend_mode == 'add':
            self._offender_refill_score = (self._offender_refill_score + score).clamp(0.0, 1.0)
        else:
            self._offender_refill_score = torch.maximum(self._offender_refill_score, score)

    def accumulate_lineage_offender_scores(self, offender_score, offender_count=None):
        if offender_score is None:
            return
        self.ensure_offender_state_matches_points(verbose=False)
        score = offender_score.detach().reshape(-1).float().clamp_min(0.0)
        if score.shape[0] != self.get_xyz.shape[0]:
            raise ValueError(f'lineage offender_score shape mismatch: got {score.shape[0]}, expected {self.get_xyz.shape[0]}')
        if offender_count is None:
            count = (score > 0).float()
        else:
            count = offender_count.detach().reshape(-1).float().clamp_min(0.0)
            if count.shape[0] != self.get_xyz.shape[0]:
                raise ValueError(f'lineage offender_count shape mismatch: got {count.shape[0]}, expected {self.get_xyz.shape[0]}')
        self._lineage_offender_score_accum = self._lineage_offender_score_accum + score.to(self._lineage_offender_score_accum.device)
        self._lineage_offender_count_accum = self._lineage_offender_count_accum + count.to(self._lineage_offender_count_accum.device)

    def _append_offender_state(self, extension_count):
        extension_count = int(extension_count)
        if extension_count <= 0:
            return
        point_count = int(self.get_xyz.shape[0]) if torch.is_tensor(self._xyz) and self._xyz.ndim >= 2 else 0
        old_count = max(point_count - extension_count, 0)
        device = self._xyz.device if torch.is_tensor(self._xyz) and self._xyz.numel() > 0 else None

        if torch.is_tensor(self._offender_score_accum) and self._offender_score_accum.shape[0] == old_count:
            score_accum = self._offender_score_accum.to(device=device, dtype=torch.float32)
        else:
            score_accum = torch.zeros((old_count,), dtype=torch.float32, device=device)

        if torch.is_tensor(self._offender_count_accum) and self._offender_count_accum.shape[0] == old_count:
            count_accum = self._offender_count_accum.to(device=device, dtype=torch.float32)
        else:
            count_accum = torch.zeros((old_count,), dtype=torch.float32, device=device)

        if torch.is_tensor(self._offender_refill_score) and self._offender_refill_score.shape[0] == old_count:
            refill_score = self._offender_refill_score.to(device=device, dtype=torch.float32)
        else:
            refill_score = torch.zeros((old_count,), dtype=torch.float32, device=device)

        self._offender_score_accum = torch.cat(
            (score_accum, torch.zeros((extension_count,), dtype=score_accum.dtype, device=device)),
            dim=0,
        )
        self._offender_count_accum = torch.cat(
            (count_accum, torch.zeros((extension_count,), dtype=count_accum.dtype, device=device)),
            dim=0,
        )
        self._offender_refill_score = torch.cat(
            (refill_score, torch.zeros((extension_count,), dtype=refill_score.dtype, device=device)),
            dim=0,
        )
        if torch.is_tensor(self._lineage_offender_score_accum) and self._lineage_offender_score_accum.shape[0] == old_count:
            lineage_score_accum = self._lineage_offender_score_accum.to(device=device, dtype=torch.float32)
        else:
            lineage_score_accum = torch.zeros((old_count,), dtype=torch.float32, device=device)

        if torch.is_tensor(self._lineage_offender_count_accum) and self._lineage_offender_count_accum.shape[0] == old_count:
            lineage_count_accum = self._lineage_offender_count_accum.to(device=device, dtype=torch.float32)
        else:
            lineage_count_accum = torch.zeros((old_count,), dtype=torch.float32, device=device)

        self._lineage_offender_score_accum = torch.cat(
            (lineage_score_accum, torch.zeros((extension_count,), dtype=lineage_score_accum.dtype, device=device)),
            dim=0,
        )
        self._lineage_offender_count_accum = torch.cat(
            (lineage_count_accum, torch.zeros((extension_count,), dtype=lineage_count_accum.dtype, device=device)),
            dim=0,
        )

    def _replace_optimizer_parameter(self, name, tensor):
        param = nn.Parameter(tensor.requires_grad_(True))
        if self.optimizer is None:
            return param

        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue

            old_param = group["params"][0]
            stored_state = self.optimizer.state.pop(old_param, None)
            group["params"][0] = param
            if stored_state is not None:
                if "exp_avg" in stored_state:
                    stored_state["exp_avg"] = torch.zeros_like(param)
                if "exp_avg_sq" in stored_state:
                    stored_state["exp_avg_sq"] = torch.zeros_like(param)
                self.optimizer.state[param] = stored_state
            break
        return param

    def _zero_optimizer_state_rows(self, name, row_mask):
        if self.optimizer is None or row_mask is None:
            return
        row_mask = row_mask.reshape(-1).bool()
        if row_mask.numel() == 0 or not bool(row_mask.any().item()):
            return
        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue
            param = group["params"][0]
            stored_state = self.optimizer.state.get(param, None)
            if stored_state is None:
                return
            for state_key in ("exp_avg", "exp_avg_sq"):
                value = stored_state.get(state_key, None)
                if torch.is_tensor(value) and value.shape[0] == row_mask.shape[0]:
                    value[row_mask] = 0
            return

    def _slice_refresh_info_tensor(self, refresh_info, key, target_mask=None, device=None, dtype=None):
        if not isinstance(refresh_info, dict) or not refresh_info:
            return None
        refresh_mask = refresh_info.get('refresh_mask', None)
        value = refresh_info.get(key, None)
        if not torch.is_tensor(refresh_mask) or not torch.is_tensor(value):
            return None
        if refresh_mask.shape[0] != self.get_xyz.shape[0]:
            return None

        if device is None:
            device = self._xyz.device
        refresh_mask = refresh_mask.to(device=device, dtype=torch.bool)
        refresh_idx = torch.nonzero(refresh_mask, as_tuple=False).squeeze(-1)
        if value.shape[0] != refresh_idx.shape[0]:
            return None

        value = value.to(device=device)
        if dtype is not None:
            value = value.to(dtype=dtype)

        if target_mask is None:
            return value
        if not torch.is_tensor(target_mask) or target_mask.shape[0] != refresh_mask.shape[0]:
            return None

        target_on_refresh = target_mask.to(device=device, dtype=torch.bool)[refresh_idx]
        if target_on_refresh.shape[0] != value.shape[0]:
            return None
        return value[target_on_refresh]

    def _build_post_rebind_keep_prior_structural_mask(
        self,
        kept_prior_child_mask,
        kept_prior_best_face_changed_mask=None,
        kept_prior_best_joint_changed_mask=None,
        kept_prior_best_anchor_shift=None,
        min_best_shift=0.0,
    ):
        if not torch.is_tensor(kept_prior_child_mask):
            return None

        device = kept_prior_child_mask.device
        base_mask = kept_prior_child_mask.to(device=device, dtype=torch.bool)
        evidence_mask = torch.zeros_like(base_mask)
        evidence_available = False
        best_shift_mask = None
        require_best_joint_change = bool(self.cfg.get(
            'binding_densify_postrebind_keep_prior_require_best_joint_change',
            True,
        ))
        allow_face_change_fallback = bool(self.cfg.get(
            'binding_densify_postrebind_keep_prior_allow_face_change_fallback',
            False,
        ))
        joint_evidence_mask = None
        face_evidence_mask = None

        if (
            torch.is_tensor(kept_prior_best_anchor_shift)
            and kept_prior_best_anchor_shift.shape[0] == base_mask.shape[0]
        ):
            best_shift_mask = kept_prior_best_anchor_shift.to(device=device, dtype=torch.float32) > float(min_best_shift)

        if (
            torch.is_tensor(kept_prior_best_joint_changed_mask)
            and kept_prior_best_joint_changed_mask.shape[0] == base_mask.shape[0]
        ):
            joint_mask = kept_prior_best_joint_changed_mask.to(device=device, dtype=torch.bool)
            if best_shift_mask is not None:
                joint_mask &= best_shift_mask
            joint_evidence_mask = joint_mask
            evidence_mask |= joint_mask
            evidence_available = True

        if (
            torch.is_tensor(kept_prior_best_face_changed_mask)
            and kept_prior_best_face_changed_mask.shape[0] == base_mask.shape[0]
        ):
            face_mask = kept_prior_best_face_changed_mask.to(device=device, dtype=torch.bool)
            if best_shift_mask is not None:
                face_mask &= best_shift_mask
            face_evidence_mask = face_mask
            if not require_best_joint_change:
                evidence_mask |= face_mask
                evidence_available = True

        if (
            require_best_joint_change
            and allow_face_change_fallback
            and not evidence_available
            and face_evidence_mask is not None
        ):
            evidence_mask |= face_evidence_mask
            evidence_available = True

        if (
            not require_best_joint_change
            and not evidence_available
            and best_shift_mask is not None
        ):
            evidence_mask |= best_shift_mask
            evidence_available = True

        if not evidence_available:
            return torch.zeros_like(base_mask)
        return base_mask & evidence_mask

    def _expand_post_rebind_switched_support_mask(
        self,
        support_mask,
        switched_mask,
        refresh_info,
        target_mask,
        device,
    ):
        if (
            not torch.is_tensor(support_mask)
            or not torch.is_tensor(switched_mask)
            or not torch.is_tensor(target_mask)
        ):
            return None

        if support_mask.shape[0] != switched_mask.shape[0]:
            return None
        if target_mask.shape[0] != self.get_xyz.shape[0]:
            return None

        mode = str(self.cfg.get(
            'binding_densify_postrebind_reset_switched_support_mode',
            'lineage_or_parent',
        )).strip().lower()
        if mode in {'', 'none', 'off', 'disabled'}:
            return support_mask & switched_mask

        support_mask = support_mask.to(device=device, dtype=torch.bool)
        switched_mask = switched_mask.to(device=device, dtype=torch.bool)
        if not bool(support_mask.any().item()) or not bool(switched_mask.any().item()):
            return support_mask & switched_mask

        expanded_mask = support_mask.clone()
        use_lineage = mode in {'lineage', 'lineage_or_parent', 'lineage_or_root_parent', 'all'}
        use_parent = mode in {'parent', 'lineage_or_parent', 'all'}
        use_root_parent = mode in {'root_parent', 'lineage_or_root_parent', 'all'}

        if use_lineage:
            selected_lineage_id = self._slice_refresh_info_tensor(
                refresh_info,
                'source_root_lineage_id',
                target_mask=target_mask,
                device=device,
                dtype=torch.long,
            )
            if torch.is_tensor(selected_lineage_id) and selected_lineage_id.shape[0] == support_mask.shape[0]:
                support_lineage_mask = support_mask & (selected_lineage_id >= 0)
                if bool(support_lineage_mask.any().item()):
                    support_lineage_ids = selected_lineage_id[support_lineage_mask].unique()
                    expanded_mask |= torch.isin(selected_lineage_id, support_lineage_ids)

        def _expand_by_parent_key(key):
            parent_index = self._slice_refresh_info_tensor(
                refresh_info,
                key,
                target_mask=target_mask,
                device=device,
                dtype=torch.long,
            )
            if not torch.is_tensor(parent_index) or parent_index.shape[0] != support_mask.shape[0]:
                return None
            support_parent_mask = support_mask & (parent_index >= 0)
            if not bool(support_parent_mask.any().item()):
                return None
            support_parent_ids = parent_index[support_parent_mask].unique()
            return torch.isin(parent_index, support_parent_ids)

        if use_parent:
            expanded_parent_mask = _expand_by_parent_key('source_parent_index')
            if (
                torch.is_tensor(expanded_parent_mask)
                and expanded_parent_mask.shape[0] == expanded_mask.shape[0]
            ):
                expanded_mask |= expanded_parent_mask
        if use_root_parent:
            expanded_root_parent_mask = _expand_by_parent_key('source_root_parent_index')
            if (
                torch.is_tensor(expanded_root_parent_mask)
                and expanded_root_parent_mask.shape[0] == expanded_mask.shape[0]
            ):
                expanded_mask |= expanded_root_parent_mask

        expanded_mask &= switched_mask
        if (
            not bool(expanded_mask.any().item())
            and bool(self.cfg.get(
                'binding_densify_postrebind_reset_switched_support_fallback_to_switched',
                True,
            ))
        ):
            return switched_mask.clone()
        return expanded_mask

    def _resolve_post_rebind_target_mask(self, refresh_info, iteration=0):
        if not isinstance(refresh_info, dict) or not refresh_info:
            return None
        if self.get_xyz.numel() <= 0:
            return None

        refresh_mask = refresh_info.get('refresh_mask', None)
        if not torch.is_tensor(refresh_mask) or refresh_mask.shape[0] != self.get_xyz.shape[0]:
            return None

        device = self._xyz.device
        # Clone here: `.to(...)` may return the same tensor when dtype/device
        # already match, and the target-building path mutates `target_mask`
        # in-place. If we alias `refresh_info['refresh_mask']`, later slices
        # see a shrunk refresh subset and all per-refresh tensors shape-mismatch.
        target_mask = refresh_mask.to(device=device, dtype=torch.bool).clone()
        risky_child_mask = self._slice_refresh_info_tensor(
            refresh_info,
            'risky_child_mask',
            device=device,
            dtype=torch.bool,
        )
        if torch.is_tensor(risky_child_mask) and risky_child_mask.shape[0] == int(target_mask.sum().item()):
            selected_idx = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)
            risky_full = torch.zeros_like(target_mask)
            risky_full[selected_idx] = risky_child_mask
            target_mask &= risky_full
        anchor_shift = self._slice_refresh_info_tensor(
            refresh_info,
            'anchor_shift',
            device=device,
            dtype=torch.float32,
        )
        kept_prior_child_mask = self._slice_refresh_info_tensor(
            refresh_info,
            'kept_prior_child_mask',
            device=device,
            dtype=torch.bool,
        )
        kept_prior_best_face_changed_mask = self._slice_refresh_info_tensor(
            refresh_info,
            'kept_prior_best_face_changed_mask',
            device=device,
            dtype=torch.bool,
        )
        kept_prior_best_joint_changed_mask = self._slice_refresh_info_tensor(
            refresh_info,
            'kept_prior_best_joint_changed_mask',
            device=device,
            dtype=torch.bool,
        )
        kept_prior_best_anchor_shift = self._slice_refresh_info_tensor(
            refresh_info,
            'kept_prior_best_anchor_shift',
            device=device,
            dtype=torch.float32,
        )
        shift_score = None
        if torch.is_tensor(anchor_shift) and anchor_shift.shape[0] == int(target_mask.sum().item()):
            selected_idx = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)
            shift_score = torch.zeros_like(target_mask, dtype=torch.float32)
            shift_score[selected_idx] = anchor_shift
        keep_prior_min_best_shift = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_postrebind_keep_prior_min_best_shift', 0.03),
            default=0.03,
        ))
        keep_prior_structural_mask = torch.zeros_like(refresh_mask, dtype=torch.bool, device=device)
        keep_prior_best_shift_score = None
        refresh_idx = torch.nonzero(refresh_mask.to(device=device, dtype=torch.bool), as_tuple=False).squeeze(-1)
        if (
            torch.is_tensor(kept_prior_child_mask)
            and kept_prior_child_mask.shape[0] == refresh_idx.shape[0]
        ):
            keep_prior_structural_refresh = self._build_post_rebind_keep_prior_structural_mask(
                kept_prior_child_mask=kept_prior_child_mask,
                kept_prior_best_face_changed_mask=kept_prior_best_face_changed_mask,
                kept_prior_best_joint_changed_mask=kept_prior_best_joint_changed_mask,
                kept_prior_best_anchor_shift=kept_prior_best_anchor_shift,
                min_best_shift=keep_prior_min_best_shift,
            )
            if torch.is_tensor(keep_prior_structural_refresh) and keep_prior_structural_refresh.shape[0] == refresh_idx.shape[0]:
                keep_prior_structural_mask[refresh_idx] = keep_prior_structural_refresh
            if (
                torch.is_tensor(kept_prior_best_anchor_shift)
                and kept_prior_best_anchor_shift.shape[0] == refresh_idx.shape[0]
            ):
                keep_prior_best_shift_score = torch.zeros_like(target_mask, dtype=torch.float32)
                keep_prior_best_shift_score[refresh_idx] = kept_prior_best_anchor_shift
        selected_idx = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)
        initial_target_count = int(selected_idx.shape[0])
        sliced_source_parent_joint = self._slice_refresh_info_tensor(
            refresh_info,
            'source_parent_joint',
            target_mask=target_mask,
            device=device,
            dtype=torch.long,
        )
        sliced_source_root_parent_joint = self._slice_refresh_info_tensor(
            refresh_info,
            'source_root_parent_joint',
            target_mask=target_mask,
            device=device,
            dtype=torch.long,
        )
        source_firewall_mask = self._source_joint_firewall_mask_from_source_joints(
            source_parent_joint=sliced_source_parent_joint,
            source_root_parent_joint=sliced_source_root_parent_joint,
            device=device,
            iteration=iteration,
            require_consensus=bool(self.cfg.get('binding_densify_postrebind_target_require_source_consensus', True)),
        )
        if source_firewall_mask is None:
            sliced_source_parent_index = self._slice_refresh_info_tensor(
                refresh_info,
                'source_parent_index',
                target_mask=target_mask,
                device=device,
                dtype=torch.long,
            )
            sliced_source_root_parent_index = self._slice_refresh_info_tensor(
                refresh_info,
                'source_root_parent_index',
                target_mask=target_mask,
                device=device,
                dtype=torch.long,
            )
            source_firewall_mask = self._source_joint_firewall_mask_from_parent_indices(
                point_count=int(selected_idx.shape[0]),
                device=device,
                source_parent_index=sliced_source_parent_index,
                source_root_parent_index=sliced_source_root_parent_index,
                iteration=iteration,
                require_consensus=bool(self.cfg.get('binding_densify_postrebind_target_require_source_consensus', True)),
            )
        if source_firewall_mask is not None and source_firewall_mask.shape[0] == selected_idx.shape[0]:
            target_mask[selected_idx] &= source_firewall_mask
        after_source_count = int(target_mask.sum().item())
        if bool(self.cfg.get('binding_densify_postrebind_target_arm_only', True)):
            arm_gate_mask = None
            binding_state = self.get_binding_state()
            point_count = self.get_xyz.shape[0]
            if binding_state and point_count > 0:
                arm_gate_mask = self._binding_arm_gate_mask_from_state(
                    binding_state,
                    point_count,
                    device,
                    iteration=iteration,
                    arm_gate_mode=self.cfg.get(
                        'binding_densify_postrebind_target_arm_gate_mode',
                        'source_or_current',
                    ),
                    require_source_consensus=bool(self.cfg.get(
                        'binding_densify_postrebind_target_require_source_consensus',
                        True,
                    )),
                )
                if torch.is_tensor(arm_gate_mask) and arm_gate_mask.shape[0] == point_count:
                    target_mask &= arm_gate_mask.to(device=device, dtype=torch.bool)
                    arm_gate_mask = None
            if arm_gate_mask is None:
                selected_idx = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)
                arm_gate_mask = self._binding_joint_gate_mask(
                    selected_idx,
                    device=device,
                    joint_ids=self.cfg.get(
                        'binding_densify_postrebind_target_joint_ids',
                        self._arm_joint_ids(),
                    ),
                )
                if arm_gate_mask is not None and arm_gate_mask.shape[0] == selected_idx.shape[0]:
                    target_mask[selected_idx] &= arm_gate_mask
                else:
                    target_mask[selected_idx] &= False
        after_arm_count = int(target_mask.sum().item())
        keep_prior_bonus_mask = torch.zeros_like(target_mask)
        target_mask = self._cap_mask_by_score(
            target_mask,
            score=shift_score,
            max_points=int(self.cfg.get('binding_densify_postrebind_target_max_points', 384)),
            max_ratio=float(self.cfg.get('binding_densify_postrebind_target_max_ratio', 0.0)),
            min_score=float(self.cfg.get('binding_densify_postrebind_target_min_shift', 0.008)),
        )
        keep_prior_max_points = int(self.cfg.get('binding_densify_postrebind_target_keep_prior_max_points', 0))
        if keep_prior_max_points > 0 and bool(keep_prior_structural_mask.any().item()):
            keep_prior_candidates = keep_prior_structural_mask & (~target_mask)
            if bool(keep_prior_candidates.any().item()):
                keep_prior_bonus_mask = self._cap_mask_by_score(
                    keep_prior_candidates,
                    score=keep_prior_best_shift_score,
                    max_points=keep_prior_max_points,
                    max_ratio=0.0,
                    min_score=keep_prior_min_best_shift,
                )
                if keep_prior_bonus_mask is not None:
                    target_mask |= keep_prior_bonus_mask
        if bool(self.cfg.get('binding_densify_debug_verbose', False)):
            print(
                '[GaussianModel] post-rebind target mask '
                f'iter={iteration} initial={initial_target_count} '
                f'after_source={after_source_count} after_arm={after_arm_count} '
                f'final={int(target_mask.sum().item())} '
                f'keep_prior_structural={int(keep_prior_structural_mask.sum().item())} '
                f'keep_prior_bonus={int(keep_prior_bonus_mask.sum().item())} '
                f'consensus={int(bool(self.cfg.get("binding_densify_postrebind_target_require_source_consensus", True)))}'
            )
        return target_mask

    def _cap_mask_by_score(self, mask, score=None, max_points=0, max_ratio=0.0, min_score=None):
        if mask is None or mask.numel() == 0 or not bool(mask.any().item()):
            return mask

        candidate_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if candidate_idx.numel() == 0:
            return mask

        if torch.is_tensor(score) and score.shape[0] == mask.shape[0]:
            candidate_score = score[candidate_idx].float()
        else:
            candidate_score = torch.ones((candidate_idx.shape[0],), dtype=torch.float32, device=mask.device)

        if min_score is not None:
            keep = candidate_score >= float(min_score)
            if not bool(keep.any().item()):
                best = torch.argmax(candidate_score)
                keep = torch.zeros_like(candidate_score, dtype=torch.bool)
                keep[best] = True
            candidate_idx = candidate_idx[keep]
            candidate_score = candidate_score[keep]

        if candidate_idx.numel() == 0:
            return torch.zeros_like(mask, dtype=torch.bool)

        limit = int(max_points)
        if max_ratio is not None and float(max_ratio) > 0.0:
            ratio_limit = max(int(np.ceil(float(mask.shape[0]) * float(max_ratio))), 1)
            limit = ratio_limit if limit <= 0 else min(limit, ratio_limit)
        if limit > 0 and candidate_idx.numel() > limit:
            topk = torch.topk(candidate_score, k=limit, sorted=False).indices
            candidate_idx = candidate_idx[topk]

        capped_mask = torch.zeros_like(mask, dtype=torch.bool)
        capped_mask[candidate_idx] = True
        return capped_mask

    def _cap_mask_by_group_score(self, mask, group=None, score=None, max_points_per_group=0):
        if (
            mask is None
            or mask.numel() == 0
            or not bool(mask.any().item())
            or max_points_per_group is None
            or int(max_points_per_group) <= 0
            or not torch.is_tensor(group)
            or group.shape[0] != mask.shape[0]
        ):
            return mask

        candidate_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if candidate_idx.numel() == 0:
            return mask

        group = group.to(device=mask.device, dtype=torch.long)
        if torch.is_tensor(score) and score.shape[0] == mask.shape[0]:
            candidate_score = score[candidate_idx].float()
        else:
            candidate_score = torch.ones((candidate_idx.shape[0],), dtype=torch.float32, device=mask.device)

        kept_idx = []
        candidate_group = group[candidate_idx]
        for group_id in candidate_group.unique().tolist():
            group_id = int(group_id)
            group_mask = candidate_group == group_id
            group_idx = candidate_idx[group_mask]
            if group_idx.numel() == 0:
                continue
            if group_id < 0 or group_idx.numel() <= int(max_points_per_group):
                kept_idx.append(group_idx)
                continue
            group_score = candidate_score[group_mask]
            topk = torch.topk(group_score, k=int(max_points_per_group), sorted=False).indices
            kept_idx.append(group_idx[topk])

        if not kept_idx:
            return torch.zeros_like(mask, dtype=torch.bool)

        capped_mask = torch.zeros_like(mask, dtype=torch.bool)
        capped_mask[torch.cat(kept_idx, dim=0)] = True
        return capped_mask

    def _binding_risk_score(self, binding_state, point_count, device, iteration=0, include_refresh_mask=False):
        if not binding_state:
            return torch.zeros((point_count,), dtype=torch.float32, device=device)

        def _state_tensor(key):
            value = binding_state.get(key, None)
            if torch.is_tensor(value) and value.shape[0] == point_count:
                return value.to(device=device)
            return None

        semantic_distance_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_refresh_semantic_distance_threshold', 0.03),
            default=0.03,
        ))
        surface_distance_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_refresh_surface_distance_threshold', 0.012),
            default=0.012,
        ))
        confidence_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_refresh_confidence_threshold', 0.6),
            default=0.6,
        ))
        weight_gap_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_refresh_weight_gap_threshold', 0.35),
            default=0.35,
        ))

        risk_score = torch.zeros((point_count,), dtype=torch.float32, device=device)

        semantic_distance = _state_tensor('semantic_distance')
        if semantic_distance is not None:
            risk_score += F.relu(
                semantic_distance.float() / max(semantic_distance_threshold, 1.0e-6) - 1.0
            )

        surface_distance = _state_tensor('surface_distance')
        if surface_distance is not None:
            risk_score += F.relu(
                surface_distance.float() / max(surface_distance_threshold, 1.0e-6) - 1.0
            )

        confidence = _state_tensor('anchor_confidence')
        if confidence is not None:
            risk_score += F.relu(
                (confidence_threshold - confidence.float()) / max(confidence_threshold, 1.0e-6)
            )

        anchor_weights = _state_tensor('anchor_weights')
        if anchor_weights is not None and anchor_weights.ndim == 2 and anchor_weights.shape[1] >= 2:
            top2 = torch.topk(anchor_weights.float(), k=2, dim=-1).values
            risk_score += F.relu(
                (weight_gap_threshold - (top2[:, 0] - top2[:, 1])) / max(weight_gap_threshold, 1.0e-6)
            )

        if include_refresh_mask:
            refresh_mask = _state_tensor('anchor_refresh_mask')
            if refresh_mask is not None:
                refresh_boost = float(self.cfg.get('binding_postrebind_refresh_score_boost', 0.25))
                risk_score += refresh_mask.float() * refresh_boost

        lineage_offender_mask = self._binding_lineage_offender_mask(
            binding_state,
            point_count,
            device,
            iteration=iteration,
        )
        if torch.is_tensor(lineage_offender_mask) and lineage_offender_mask.shape[0] == point_count:
            lineage_bonus = float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_lineage_offender_score_bonus', 0.35),
                default=0.35,
            ))
            risk_score += lineage_offender_mask.to(device=device, dtype=torch.float32) * lineage_bonus

        return risk_score

    def _candidate_local_support_score(self, candidate_mask, iteration=0):
        if candidate_mask is None or candidate_mask.numel() == 0 or not bool(candidate_mask.any().item()):
            return None, None
        if self.get_xyz.numel() <= 0:
            return None, None

        positions = self.get_xyz.detach()
        point_count = positions.shape[0]
        candidate_idx = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
        if candidate_idx.numel() == 0:
            return None, None

        support_k = int(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_stale_refresh_support_k', 8),
            default=8,
        ))
        support_k = min(max(support_k, 1), max(point_count - 1, 1))

        query = positions[candidate_idx]
        knn_k = min(point_count, support_k + 1)
        knn = ops.knn_points(
            query.unsqueeze(0),
            positions.unsqueeze(0),
            K=knn_k,
        )
        nn_dists = knn.dists[0].clamp_min(0.0).sqrt()
        nn_idx = knn.idx[0]
        if knn_k > 1:
            nn_dists = nn_dists[:, 1:]
            nn_idx = nn_idx[:, 1:]

        if nn_dists.numel() == 0:
            return torch.zeros_like(candidate_idx, dtype=torch.float32, device=positions.device), candidate_idx

        neighbor_opacity = self.get_opacity.detach().reshape(-1)[nn_idx].float().clamp(0.0, 1.0)
        local_scale = self.get_scaling.detach().amax(dim=-1)[candidate_idx].float().clamp_min(1.0e-6)
        scale_multiplier = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_stale_refresh_support_scale_multiplier', 2.0),
            default=2.0,
        ))
        support_radius = (local_scale * max(scale_multiplier, 1.0e-6)).unsqueeze(-1)
        distance_score = torch.exp(-nn_dists / support_radius).mean(dim=-1)
        opacity_score = neighbor_opacity.mean(dim=-1)
        support_score = (distance_score * opacity_score).clamp(0.0, 1.0)
        return support_score, candidate_idx

    def mark_stale_binding_points_for_refresh(self, iteration=0):
        if not bool(self.cfg.get('binding_stale_refresh_enable', False)):
            return 0
        if self.get_xyz.numel() <= 0 or not self.has_binding_state():
            return 0

        start_iter = int(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_stale_refresh_start_iter', 0),
            default=0,
        ))
        refresh_interval = int(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_stale_refresh_interval', 0),
            default=0,
        ))
        end_iter = int(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_stale_refresh_until_iter', -1),
            default=-1,
        ))
        if refresh_interval <= 0 or iteration < start_iter:
            return 0
        if end_iter >= 0 and iteration > end_iter:
            return 0
        if (iteration - start_iter) % refresh_interval != 0:
            return 0

        binding_state = self.get_binding_state()
        point_count = self.get_xyz.shape[0]
        device = self._xyz.device
        boundary_tags = self.get_boundary_tag_state()
        include_refresh_mask = bool(self.cfg.get('binding_stale_refresh_include_refresh_mask', False))

        candidate_mask = self._binding_boundary_arm_risk_mask(
            binding_state,
            point_count,
            device,
            iteration=iteration,
            boundary_tags=boundary_tags,
            boundary_threshold=float(resolve_schedule_value(
                iteration,
                self.cfg.get(
                    'binding_stale_refresh_boundary_threshold',
                    self.cfg.get('binding_densify_boundary_threshold', 0.05),
                ),
                default=0.05,
            )),
            arm_only=bool(self.cfg.get('binding_stale_refresh_arm_only', True)),
            semantic_scale=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_stale_refresh_semantic_distance_scale', 1.0),
                default=1.0,
            )),
            surface_scale=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_stale_refresh_surface_distance_scale', 1.0),
                default=1.0,
            )),
            confidence_margin=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_stale_refresh_confidence_margin', 0.0),
                default=0.0,
            )),
            weight_gap_scale=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_stale_refresh_weight_gap_scale', 1.0),
                default=1.0,
            )),
            include_refresh_mask=include_refresh_mask,
        )
        if not bool(candidate_mask.any().item()):
            return 0

        opacity = self.get_opacity.detach().reshape(-1).float().clamp(0.0, 1.0)
        min_opacity = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_stale_refresh_min_opacity', 0.0),
            default=0.0,
        ))
        max_opacity = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_stale_refresh_max_opacity', 1.0),
            default=1.0,
        ))
        candidate_mask &= opacity >= min_opacity
        if max_opacity < 1.0:
            candidate_mask &= opacity <= max_opacity
        if not bool(candidate_mask.any().item()):
            return 0

        support_score, candidate_idx = self._candidate_local_support_score(candidate_mask, iteration=iteration)
        if support_score is not None and candidate_idx is not None:
            max_support_score = float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_stale_refresh_max_support_score', 1.0),
                default=1.0,
            ))
            if max_support_score < 1.0:
                support_mask = support_score <= max_support_score
                filtered_mask = torch.zeros_like(candidate_mask)
                filtered_mask[candidate_idx[support_mask]] = True
                candidate_mask &= filtered_mask
        if not bool(candidate_mask.any().item()):
            return 0

        priority = self._binding_risk_score(
            binding_state,
            point_count,
            device,
            iteration=iteration,
            include_refresh_mask=include_refresh_mask,
        )
        if support_score is not None and candidate_idx is not None:
            low_support_bonus = float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_stale_refresh_low_support_bonus', 0.5),
                default=0.5,
            ))
            priority = priority.clone()
            priority[candidate_idx] = priority[candidate_idx] + (1.0 - support_score) * low_support_bonus
        min_risk_score = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_stale_refresh_min_risk_score', 0.0),
            default=0.0,
        ))
        if min_risk_score > 0.0:
            candidate_mask &= priority >= min_risk_score
        if not bool(candidate_mask.any().item()):
            return 0

        candidate_idx = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
        limit = int(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_stale_refresh_max_points', 0),
            default=0,
        ))
        max_ratio = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_stale_refresh_max_ratio', 0.0),
            default=0.0,
        ))
        if max_ratio > 0.0:
            ratio_limit = max(int(np.ceil(float(point_count) * max_ratio)), 1)
            limit = ratio_limit if limit <= 0 else min(limit, ratio_limit)
        if limit > 0 and candidate_idx.numel() > limit:
            candidate_priority = priority[candidate_idx]
            candidate_idx = candidate_idx[torch.topk(candidate_priority, k=limit, sorted=False).indices]
            candidate_mask = torch.zeros_like(candidate_mask)
            candidate_mask[candidate_idx] = True

        updated_state = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in binding_state.items()
        }
        refresh_mask = updated_state.get('anchor_refresh_mask', None)
        if not torch.is_tensor(refresh_mask) or refresh_mask.shape[0] != point_count:
            refresh_mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
        else:
            refresh_mask = refresh_mask.to(device=device, dtype=torch.bool)
        refresh_mask[candidate_mask] = True
        updated_state['anchor_refresh_mask'] = refresh_mask
        self.binding_state = updated_state

        marked_count = int(candidate_mask.sum().item())
        if marked_count > 0 and bool(self.cfg.get('binding_stale_refresh_verbose', False)):
            print(
                f'[GaussianModel] marked {marked_count} stale risky binding points for refresh '
                f'at iter {iteration}'
            )
        return marked_count

    def _build_post_rebind_unresolved_prune_mask(self, refresh_info, iteration=0):
        if not bool(self.cfg.get('binding_densify_postrebind_prune_enable', False)):
            return None

        target_mask = self._resolve_post_rebind_target_mask(refresh_info, iteration=iteration)
        if target_mask is None or not bool(target_mask.any().item()):
            return None

        point_count = target_mask.shape[0]
        device = target_mask.device
        candidate_mask = target_mask.clone()
        binding_state = self.get_binding_state()
        if binding_state:
            candidate_mask &= self._binding_boundary_arm_risk_mask(
                binding_state,
                point_count,
                device,
                iteration=iteration,
                boundary_tags=self.get_boundary_tag_state(),
                boundary_threshold=float(resolve_schedule_value(
                    iteration,
                    self.cfg.get(
                        'binding_densify_postrebind_prune_boundary_threshold',
                        self.cfg.get('binding_densify_boundary_threshold', 0.05),
                    ),
                    default=0.05,
                )),
                arm_only=bool(self.cfg.get('binding_densify_postrebind_prune_arm_only', True)),
                semantic_scale=float(resolve_schedule_value(
                    iteration,
                    self.cfg.get('binding_densify_postrebind_prune_semantic_distance_scale', 1.0),
                    default=1.0,
                )),
                surface_scale=float(resolve_schedule_value(
                    iteration,
                    self.cfg.get('binding_densify_postrebind_prune_surface_distance_scale', 1.0),
                    default=1.0,
                )),
                confidence_margin=float(resolve_schedule_value(
                    iteration,
                    self.cfg.get('binding_densify_postrebind_prune_confidence_margin', 0.0),
                    default=0.0,
                )),
                weight_gap_scale=float(resolve_schedule_value(
                    iteration,
                    self.cfg.get('binding_densify_postrebind_prune_weight_gap_scale', 1.0),
                    default=1.0,
                )),
                include_refresh_mask=False,
            )

        opacity_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_postrebind_prune_opacity_threshold', 0.08),
            default=0.08,
        ))
        opacity = self.get_opacity.detach().reshape(-1).float().clamp(0.0, 1.0)
        candidate_mask &= opacity <= opacity_threshold

        if not bool(candidate_mask.any().item()):
            return None

        priority = self._binding_risk_score(
            binding_state,
            point_count,
            device,
            iteration=iteration,
            include_refresh_mask=False,
        )
        priority = priority + (1.0 - opacity) * float(
            self.cfg.get('binding_densify_postrebind_prune_low_opacity_bonus', 0.5)
        )

        candidate_idx = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
        limit = int(self.cfg.get('binding_densify_postrebind_prune_max_points', 0))
        max_ratio = float(self.cfg.get('binding_densify_postrebind_prune_max_ratio', 0.0))
        if max_ratio > 0.0:
            ratio_limit = max(int(np.ceil(float(point_count) * max_ratio)), 1)
            limit = ratio_limit if limit <= 0 else min(limit, ratio_limit)

        if limit > 0 and candidate_idx.numel() > limit:
            candidate_priority = priority[candidate_idx]
            candidate_idx = candidate_idx[torch.topk(candidate_priority, k=limit, sorted=False).indices]
            candidate_mask = torch.zeros_like(candidate_mask)
            candidate_mask[candidate_idx] = True

        return candidate_mask

    def _resolve_post_rebind_source_parent_mask(self, refresh_info, child_mask, iteration=0):
        if not isinstance(refresh_info, dict) or not refresh_info:
            return None
        if child_mask is None or self.get_xyz.numel() <= 0:
            return None

        child_mask = child_mask.to(device=self._xyz.device, dtype=torch.bool)
        if child_mask.shape[0] != self.get_xyz.shape[0] or not bool(child_mask.any().item()):
            return None

        target_mask = self._resolve_post_rebind_target_mask(refresh_info, iteration=iteration)
        if target_mask is None or not bool(target_mask.any().item()):
            return None

        selected_idx = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)
        if selected_idx.numel() == 0:
            return None

        source_parent_index = self._slice_refresh_info_tensor(
            refresh_info,
            'source_parent_index',
            target_mask=target_mask,
            device=self._xyz.device,
            dtype=torch.long,
        )
        if not torch.is_tensor(source_parent_index) or source_parent_index.shape[0] != selected_idx.shape[0]:
            source_parent_index = self._slice_refresh_info_tensor(
                refresh_info,
                'source_root_parent_index',
                target_mask=target_mask,
                device=self._xyz.device,
                dtype=torch.long,
            )
        if not torch.is_tensor(source_parent_index) or source_parent_index.shape[0] != selected_idx.shape[0]:
            return None

        source_parent_index = source_parent_index.to(device=self._xyz.device, dtype=torch.long)
        selected_child_mask = child_mask[selected_idx]
        if not bool(selected_child_mask.any().item()):
            return None

        parent_idx = source_parent_index[selected_child_mask]
        valid_parent = (parent_idx >= 0) & (parent_idx < self.get_xyz.shape[0])
        if not bool(valid_parent.any().item()):
            return None

        parent_mask = torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device=self._xyz.device)
        parent_mask[parent_idx[valid_parent].unique()] = True
        return parent_mask

    def _accumulate_post_rebind_lineage_offender_scores(self, refresh_info, correction_mask=None, prune_mask=None, iteration=0):
        if not bool(self.cfg.get('binding_densify_postrebind_lineage_offender_enable', True)):
            return None
        if not isinstance(refresh_info, dict) or not refresh_info:
            return None
        if self.get_xyz.numel() <= 0 or not self.has_binding_state():
            return None

        target_mask = self._resolve_post_rebind_target_mask(refresh_info, iteration=iteration)
        if target_mask is None or not bool(target_mask.any().item()):
            return None

        point_count = self.get_xyz.shape[0]
        device = self._xyz.device
        selected_idx = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)
        if selected_idx.numel() == 0:
            return None

        source_root_lineage_id = self._slice_refresh_info_tensor(
            refresh_info,
            'source_root_lineage_id',
            target_mask=target_mask,
            device=device,
            dtype=torch.long,
        )
        if not torch.is_tensor(source_root_lineage_id) or source_root_lineage_id.shape[0] != selected_idx.shape[0]:
            return None

        stats = {
            'target_children': int(selected_idx.numel()),
            'still_risky_children': 0,
            'scored_children': 0,
            'active_lineages': 0,
            'lineage_points': 0,
            'mean_child_risk_score': 0.0,
            'max_child_risk_score': 0.0,
        }

        binding_state = self.get_binding_state()
        current_root_lineage_id = binding_state.get('densify_root_lineage_id', None)
        if not torch.is_tensor(current_root_lineage_id) or current_root_lineage_id.shape[0] != point_count:
            current_root_lineage_id = binding_state.get('densify_lineage_id', None)
        if not torch.is_tensor(current_root_lineage_id) or current_root_lineage_id.shape[0] != point_count:
            return stats
        current_root_lineage_id = current_root_lineage_id.to(device=device, dtype=torch.long)

        current_risk_mask = self._binding_boundary_arm_risk_mask(
            binding_state,
            point_count,
            device,
            iteration=iteration,
            boundary_tags=self.get_boundary_tag_state(),
            boundary_threshold=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_postrebind_lineage_offender_boundary_threshold', 0.04),
                default=0.04,
            )),
            arm_only=bool(self.cfg.get('binding_densify_postrebind_lineage_offender_arm_only', True)),
            semantic_scale=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_postrebind_lineage_offender_semantic_distance_scale', 1.0),
                default=1.0,
            )),
            surface_scale=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_postrebind_lineage_offender_surface_distance_scale', 1.0),
                default=1.0,
            )),
            confidence_margin=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_postrebind_lineage_offender_confidence_margin', 0.0),
                default=0.0,
            )),
            weight_gap_scale=float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_postrebind_lineage_offender_weight_gap_scale', 1.0),
                default=1.0,
            )),
            include_refresh_mask=False,
        )
        current_risk_score = self._binding_risk_score(
            binding_state,
            point_count,
            device,
            iteration=iteration,
            include_refresh_mask=False,
        )

        unresolved_child_mask = current_risk_mask[selected_idx]
        stats['still_risky_children'] = int(unresolved_child_mask.sum().item())
        if bool(unresolved_child_mask.any().item()):
            unresolved_scores = current_risk_score[selected_idx][unresolved_child_mask]
            stats['mean_child_risk_score'] = float(unresolved_scores.mean().item())
            stats['max_child_risk_score'] = float(unresolved_scores.max().item())

        selected_score = torch.zeros((selected_idx.shape[0],), dtype=torch.float32, device=device)
        unresolved_base_score = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_postrebind_lineage_offender_unresolved_base_score', 0.5),
            default=0.5,
        ))
        unresolved_risk_scale = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_postrebind_lineage_offender_unresolved_risk_scale', 0.12),
            default=0.12,
        ))
        if bool(unresolved_child_mask.any().item()):
            unresolved_score = (
                unresolved_base_score
                + current_risk_score[selected_idx] * unresolved_risk_scale
            ).clamp(0.0, 1.0)
            selected_score = torch.where(
                unresolved_child_mask,
                unresolved_score,
                selected_score,
            )

        correction_score = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_postrebind_lineage_offender_correction_score', 0.35),
            default=0.35,
        ))
        prune_score = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_postrebind_lineage_offender_prune_score', 0.75),
            default=0.75,
        ))
        if torch.is_tensor(correction_mask) and correction_mask.shape[0] == point_count:
            selected_score = torch.where(
                correction_mask[selected_idx].to(device=device, dtype=torch.bool),
                torch.full_like(selected_score, correction_score),
                selected_score,
            )
        if torch.is_tensor(prune_mask) and prune_mask.shape[0] == point_count:
            selected_score = torch.maximum(
                selected_score,
                torch.where(
                    prune_mask[selected_idx].to(device=device, dtype=torch.bool),
                    torch.full_like(selected_score, prune_score),
                    torch.zeros_like(selected_score),
                ),
            )
        active_child_mask = selected_score > 0.0
        if not bool(active_child_mask.any().item()):
            return stats

        active_lineage_ids = source_root_lineage_id[active_child_mask]
        active_lineage_scores = selected_score[active_child_mask]
        valid_lineage_mask = active_lineage_ids >= 0
        if not bool(valid_lineage_mask.any().item()):
            return stats
        active_lineage_ids = active_lineage_ids[valid_lineage_mask]
        active_lineage_scores = active_lineage_scores[valid_lineage_mask]
        stats['scored_children'] = int(active_child_mask.sum().item())

        offender_score = torch.zeros((point_count,), dtype=torch.float32, device=device)
        offender_count = torch.zeros((point_count,), dtype=torch.float32, device=device)
        unique_lineage_ids = active_lineage_ids.unique()
        stats['active_lineages'] = int(unique_lineage_ids.numel())
        for lineage_id in unique_lineage_ids.tolist():
            lineage_id = int(lineage_id)
            current_lineage_mask = (current_root_lineage_id == lineage_id) & current_risk_mask
            if not bool(current_lineage_mask.any().item()):
                continue
            lineage_score = float(active_lineage_scores[active_lineage_ids == lineage_id].max().item())
            offender_score[current_lineage_mask] = torch.maximum(
                offender_score[current_lineage_mask],
                torch.full_like(offender_score[current_lineage_mask], lineage_score),
            )
            offender_count[current_lineage_mask] = offender_count[current_lineage_mask] + 1.0

        if not bool((offender_count > 0).any().item()):
            return stats
        self.accumulate_lineage_offender_scores(offender_score, offender_count)
        stats['lineage_points'] = int((offender_count > 0).sum().item())
        return stats

    def _apply_post_rebind_source_parent_cleanup(self, parent_mask, iteration=0):
        if not bool(self.cfg.get('binding_densify_postrebind_source_parent_cleanup_enable', False)):
            return 0
        if self.get_xyz.numel() <= 0:
            return 0

        parent_mask = parent_mask.to(device=self._xyz.device, dtype=torch.bool)
        if parent_mask.shape[0] != self.get_xyz.shape[0] or not bool(parent_mask.any().item()):
            return 0

        if self.has_binding_state():
            parent_mask &= self._binding_boundary_arm_risk_mask(
                self.binding_state,
                self.get_xyz.shape[0],
                self._xyz.device,
                iteration=iteration,
                boundary_tags=self.get_boundary_tag_state(),
                boundary_threshold=float(resolve_schedule_value(
                    iteration,
                    self.cfg.get('binding_densify_postrebind_source_parent_boundary_threshold', 0.04),
                    default=0.04,
                )),
                arm_only=bool(self.cfg.get('binding_densify_postrebind_source_parent_arm_only', True)),
                semantic_scale=float(resolve_schedule_value(
                    iteration,
                    self.cfg.get('binding_densify_postrebind_source_parent_semantic_distance_scale', 1.0),
                    default=1.0,
                )),
                surface_scale=float(resolve_schedule_value(
                    iteration,
                    self.cfg.get('binding_densify_postrebind_source_parent_surface_distance_scale', 1.0),
                    default=1.0,
                )),
                confidence_margin=float(resolve_schedule_value(
                    iteration,
                    self.cfg.get('binding_densify_postrebind_source_parent_confidence_margin', 0.0),
                    default=0.0,
                )),
                weight_gap_scale=float(resolve_schedule_value(
                    iteration,
                    self.cfg.get('binding_densify_postrebind_source_parent_weight_gap_scale', 1.0),
                    default=1.0,
                )),
                include_refresh_mask=True,
            )
        if not bool(parent_mask.any().item()):
            return 0

        opacity_factor = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_postrebind_source_parent_opacity_factor', 0.3),
            default=0.3,
        ))
        scale_factor = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_postrebind_source_parent_scale_factor', 0.9),
            default=0.9,
        ))
        feature_dc_factor = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_postrebind_source_parent_feature_dc_factor', 0.8),
            default=0.8,
        ))
        feature_rest_factor = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_postrebind_source_parent_feature_rest_factor', 0.25),
            default=0.25,
        ))

        with torch.no_grad():
            self._features_dc.data[parent_mask] *= feature_dc_factor
            self._features_rest.data[parent_mask] *= feature_rest_factor

            actual_opacity = self.opacity_activation(self._opacity.data[parent_mask]) * opacity_factor
            actual_opacity = actual_opacity.clamp(1e-4, 1.0 - 1e-4)
            self._opacity.data[parent_mask] = self.inverse_opacity_activation(actual_opacity)

            actual_scaling = self.scaling_activation(self._scaling.data[parent_mask]) * scale_factor
            actual_scaling = actual_scaling.clamp_min(1e-6)
            self._scaling.data[parent_mask] = self.scaling_inverse_activation(actual_scaling)

            self._boundary_opacity_residual.data[parent_mask] = 0
            self._boundary_scaling_residual.data[parent_mask] = 0

        self._zero_optimizer_state_rows("f_dc", parent_mask)
        self._zero_optimizer_state_rows("f_rest", parent_mask)
        self._zero_optimizer_state_rows("opacity", parent_mask)
        self._zero_optimizer_state_rows("scaling", parent_mask)
        self._zero_optimizer_state_rows("boundary_opacity_residual", parent_mask)
        self._zero_optimizer_state_rows("boundary_scaling_residual", parent_mask)

        if self.has_binding_state():
            refresh_mask = self.binding_state.get('anchor_refresh_mask', None)
            if torch.is_tensor(refresh_mask) and refresh_mask.shape[0] == self.get_xyz.shape[0]:
                refresh_mask = refresh_mask.to(device=self._xyz.device, dtype=torch.bool)
            else:
                refresh_mask = torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device=self._xyz.device)
            refresh_mask[parent_mask] = True
            self.binding_state['anchor_refresh_mask'] = refresh_mask

            risky_child_state = self.binding_state.get('densify_risky_child_mask', None)
            if torch.is_tensor(risky_child_state) and risky_child_state.shape[0] == self.get_xyz.shape[0]:
                self.binding_state['densify_risky_child_mask'] = (
                    risky_child_state.to(device=self._xyz.device, dtype=torch.bool) & (~parent_mask)
                )

        return int(parent_mask.sum().item())

    def apply_post_rebind_child_correction(self, refresh_info, iteration=0):
        if not isinstance(refresh_info, dict) or not refresh_info:
            return 0
        enable_reset = bool(self.cfg.get('binding_densify_postrebind_reset_enable', False))
        enable_prune = bool(self.cfg.get('binding_densify_postrebind_prune_enable', False))
        if not enable_reset and not enable_prune:
            return 0
        if self.get_xyz.numel() <= 0:
            return 0

        target_mask = self._resolve_post_rebind_target_mask(refresh_info, iteration=iteration)
        if target_mask is None or not bool(target_mask.any().item()):
            return 0
        device = self._xyz.device
        target_count = int(target_mask.sum().item())

        selected_idx = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)
        changed_mask = torch.zeros((selected_idx.shape[0],), dtype=torch.bool, device=device)
        risky_child_mask = self._slice_refresh_info_tensor(
            refresh_info,
            'risky_child_mask',
            target_mask=target_mask,
            device=device,
            dtype=torch.bool,
        )
        risky_child_count = 0
        if torch.is_tensor(risky_child_mask) and risky_child_mask.shape[0] == selected_idx.shape[0]:
            risky_child_count = int(risky_child_mask.sum().item())
        anchor_shift = self._slice_refresh_info_tensor(
            refresh_info,
            'anchor_shift',
            target_mask=target_mask,
            device=device,
            dtype=torch.float32,
        )
        anchor_face_changed = self._slice_refresh_info_tensor(
            refresh_info,
            'anchor_face_changed',
            target_mask=target_mask,
            device=device,
            dtype=torch.bool,
        )
        switched_child_mask = self._slice_refresh_info_tensor(
            refresh_info,
            'switched_child_mask',
            target_mask=target_mask,
            device=device,
            dtype=torch.bool,
        )
        kept_prior_child_mask = self._slice_refresh_info_tensor(
            refresh_info,
            'kept_prior_child_mask',
            target_mask=target_mask,
            device=device,
            dtype=torch.bool,
        )
        kept_prior_best_face_changed_mask = self._slice_refresh_info_tensor(
            refresh_info,
            'kept_prior_best_face_changed_mask',
            target_mask=target_mask,
            device=device,
            dtype=torch.bool,
        )
        kept_prior_best_joint_changed_mask = self._slice_refresh_info_tensor(
            refresh_info,
            'kept_prior_best_joint_changed_mask',
            target_mask=target_mask,
            device=device,
            dtype=torch.bool,
        )
        kept_prior_best_anchor_shift = self._slice_refresh_info_tensor(
            refresh_info,
            'kept_prior_best_anchor_shift',
            target_mask=target_mask,
            device=device,
            dtype=torch.float32,
        )
        best_joint_changed_mask = self._slice_refresh_info_tensor(
            refresh_info,
            'best_joint_changed_mask',
            target_mask=target_mask,
            device=device,
            dtype=torch.bool,
        )
        dominant_joint_changed = self._slice_refresh_info_tensor(
            refresh_info,
            'dominant_joint_changed',
            target_mask=target_mask,
            device=device,
            dtype=torch.bool,
        )
        anchor_shift_mean = 0.0
        anchor_shift_max = 0.0
        face_changed_mask = None
        face_changed_count = 0
        switched_count = 0
        kept_prior_count = 0
        kept_prior_structural_mask = torch.zeros((selected_idx.shape[0],), dtype=torch.bool, device=device)
        kept_prior_structural_count = 0
        joint_changed_mask = None
        joint_changed_count = 0
        if torch.is_tensor(anchor_face_changed) and anchor_face_changed.shape[0] == selected_idx.shape[0]:
            face_changed_mask = anchor_face_changed
            face_changed_count = int(face_changed_mask.sum().item())
        switched_mask_available = (
            torch.is_tensor(switched_child_mask)
            and switched_child_mask.shape[0] == selected_idx.shape[0]
        )
        if switched_mask_available:
            switched_count = int(switched_child_mask.sum().item())
        kept_prior_mask_available = (
            torch.is_tensor(kept_prior_child_mask)
            and kept_prior_child_mask.shape[0] == selected_idx.shape[0]
        )
        if kept_prior_mask_available:
            kept_prior_count = int(kept_prior_child_mask.sum().item())
        if torch.is_tensor(best_joint_changed_mask) and best_joint_changed_mask.shape[0] == selected_idx.shape[0]:
            joint_changed_mask = best_joint_changed_mask
            joint_changed_count = int(joint_changed_mask.sum().item())
        elif torch.is_tensor(dominant_joint_changed) and dominant_joint_changed.shape[0] == selected_idx.shape[0]:
            joint_changed_mask = dominant_joint_changed
            joint_changed_count = int(joint_changed_mask.sum().item())

        reset_min_shift = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_postrebind_reset_min_shift', 0.012),
            default=0.012,
        ))
        shift_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_postrebind_anchor_shift_threshold', 0.01),
            default=0.01,
        ))
        face_change_shift_threshold = float(resolve_schedule_value(
            iteration,
            self.cfg.get(
                'binding_densify_postrebind_reset_face_change_shift_threshold',
                max(reset_min_shift * 2.5, shift_threshold),
            ),
            default=max(reset_min_shift * 2.5, shift_threshold),
        ))
        keep_prior_min_best_shift = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_postrebind_keep_prior_min_best_shift', 0.03),
            default=0.03,
        ))
        if kept_prior_mask_available:
            built_keep_prior_structural_mask = self._build_post_rebind_keep_prior_structural_mask(
                kept_prior_child_mask=kept_prior_child_mask,
                kept_prior_best_face_changed_mask=kept_prior_best_face_changed_mask,
                kept_prior_best_joint_changed_mask=kept_prior_best_joint_changed_mask,
                kept_prior_best_anchor_shift=kept_prior_best_anchor_shift,
                min_best_shift=keep_prior_min_best_shift,
            )
            if (
                torch.is_tensor(built_keep_prior_structural_mask)
                and built_keep_prior_structural_mask.shape[0] == selected_idx.shape[0]
            ):
                kept_prior_structural_mask = built_keep_prior_structural_mask.to(device=device, dtype=torch.bool)
                kept_prior_structural_count = int(kept_prior_structural_mask.sum().item())
        effective_anchor_shift = torch.zeros((selected_idx.shape[0],), dtype=torch.float32, device=device)
        if torch.is_tensor(anchor_shift) and anchor_shift.shape[0] == selected_idx.shape[0]:
            effective_anchor_shift = torch.maximum(effective_anchor_shift, anchor_shift.to(device=device, dtype=torch.float32))
        if (
            torch.is_tensor(kept_prior_best_anchor_shift)
            and kept_prior_best_anchor_shift.shape[0] == selected_idx.shape[0]
            and bool(kept_prior_structural_mask.any().item())
        ):
            effective_anchor_shift = torch.where(
                kept_prior_structural_mask,
                torch.maximum(
                    effective_anchor_shift,
                    kept_prior_best_anchor_shift.to(device=device, dtype=torch.float32),
                ),
                effective_anchor_shift,
            )
        if effective_anchor_shift.numel() > 0:
            anchor_shift_mean = float(effective_anchor_shift.mean().item())
            anchor_shift_max = float(effective_anchor_shift.max().item())
        shift_changed_mask = effective_anchor_shift > shift_threshold
        changed_mask |= shift_changed_mask
        after_shift_count = int(shift_changed_mask.sum().item())

        joint_change_gate_mask = torch.zeros((selected_idx.shape[0],), dtype=torch.bool, device=device)
        switched_gate_mask = torch.zeros((selected_idx.shape[0],), dtype=torch.bool, device=device)
        kept_prior_gate_mask = torch.zeros((selected_idx.shape[0],), dtype=torch.bool, device=device)
        face_change_gate_mask = torch.zeros((selected_idx.shape[0],), dtype=torch.bool, device=device)
        switched_support_mask = torch.zeros((selected_idx.shape[0],), dtype=torch.bool, device=device)
        structural_signal_available = False
        require_switched_signal = bool(self.cfg.get(
            'binding_densify_postrebind_reset_require_switched_signal',
            True,
        ))
        keep_prior_assist_only = bool(self.cfg.get(
            'binding_densify_postrebind_reset_keep_prior_assist_only',
            True,
        ))
        if switched_mask_available:
            switched_gate_mask |= switched_child_mask
            structural_signal_available = True
        if bool(self.cfg.get('binding_densify_postrebind_reset_include_kept_prior', True)) and bool(kept_prior_structural_mask.any().item()):
            kept_prior_gate_mask |= kept_prior_structural_mask
            if keep_prior_assist_only and switched_mask_available:
                kept_prior_gate_mask &= switched_gate_mask
            structural_signal_available = True
        if joint_changed_mask is not None:
            joint_change_gate_mask |= joint_changed_mask
            structural_signal_available = True
        if face_changed_mask is not None:
            structural_signal_available = True
            if effective_anchor_shift.shape[0] == selected_idx.shape[0]:
                face_change_gate_mask = face_changed_mask & (effective_anchor_shift > face_change_shift_threshold)
                # Face-id churn is only a rescue signal for points that the
                # rigid refresh explicitly marked as switched children.
                if switched_mask_available:
                    face_change_gate_mask &= switched_gate_mask
                if (
                    bool(self.cfg.get('binding_densify_postrebind_reset_face_rescue_require_joint_signal', True))
                    and joint_changed_mask is not None
                ):
                    face_change_gate_mask &= joint_change_gate_mask

        if require_switched_signal and switched_mask_available:
            structural_support_mask = joint_change_gate_mask | kept_prior_gate_mask | face_change_gate_mask
            if bool(structural_support_mask.any().item()):
                expanded_switched_support_mask = self._expand_post_rebind_switched_support_mask(
                    support_mask=structural_support_mask,
                    switched_mask=switched_gate_mask,
                    refresh_info=refresh_info,
                    target_mask=target_mask,
                    device=device,
                )
                if (
                    torch.is_tensor(expanded_switched_support_mask)
                    and expanded_switched_support_mask.shape[0] == selected_idx.shape[0]
                ):
                    switched_support_mask = expanded_switched_support_mask.to(device=device, dtype=torch.bool)
                else:
                    switched_support_mask = structural_support_mask & switched_gate_mask
            if bool(switched_support_mask.any().item()):
                structural_change_mask = switched_support_mask
            else:
                structural_change_mask = switched_gate_mask.clone()
        else:
            structural_change_mask = joint_change_gate_mask | switched_gate_mask | kept_prior_gate_mask | face_change_gate_mask
        face_rescue_count = int(face_change_gate_mask.sum().item())
        switched_support_count = int(switched_support_mask.sum().item())

        require_joint_change = bool(self.cfg.get('binding_densify_postrebind_reset_require_joint_change', False))
        if require_joint_change or bool(self.cfg.get('binding_densify_postrebind_joint_change_only', False)):
            if structural_signal_available:
                changed_mask = structural_change_mask
            elif not bool(shift_changed_mask.any().item()):
                changed_mask &= False
        elif structural_signal_available:
            changed_mask |= structural_change_mask
        after_joint_gate_count = int(changed_mask.sum().item())

        current_risky_count = 0
        risk_fallback_count = 0
        if bool(self.cfg.get('binding_densify_postrebind_reset_require_still_risky', True)):
            binding_state = self.get_binding_state()
            point_count = self.get_xyz.shape[0]
            effective_risk_mask = None
            if binding_state and point_count > 0:
                current_risk_mask = self._binding_boundary_arm_risk_mask(
                    binding_state,
                    point_count,
                    device,
                    iteration=iteration,
                    boundary_tags=self.get_boundary_tag_state(),
                    boundary_threshold=float(resolve_schedule_value(
                        iteration,
                        self.cfg.get('binding_densify_postrebind_reset_boundary_threshold', 0.05),
                        default=0.05,
                    )),
                    arm_only=bool(self.cfg.get('binding_densify_postrebind_reset_arm_only', True)),
                    semantic_scale=float(resolve_schedule_value(
                        iteration,
                        self.cfg.get('binding_densify_postrebind_reset_semantic_distance_scale', 1.0),
                        default=1.0,
                    )),
                    surface_scale=float(resolve_schedule_value(
                        iteration,
                        self.cfg.get('binding_densify_postrebind_reset_surface_distance_scale', 1.0),
                        default=1.0,
                    )),
                    confidence_margin=float(resolve_schedule_value(
                        iteration,
                        self.cfg.get('binding_densify_postrebind_reset_confidence_margin', 0.0),
                        default=0.0,
                    )),
                    weight_gap_scale=float(resolve_schedule_value(
                        iteration,
                        self.cfg.get('binding_densify_postrebind_reset_weight_gap_scale', 1.0),
                        default=1.0,
                    )),
                    include_refresh_mask=False,
                )
                effective_risk_mask = current_risk_mask[selected_idx]
                current_risky_count = int(effective_risk_mask.sum().item())
            if (
                torch.is_tensor(risky_child_mask)
                and risky_child_mask.shape[0] == selected_idx.shape[0]
                and (
                    effective_risk_mask is None
                    or not bool(effective_risk_mask.any().item())
                )
            ):
                effective_risk_mask = risky_child_mask
                risk_fallback_count = int(effective_risk_mask.sum().item())
            if effective_risk_mask is not None:
                changed_mask &= effective_risk_mask.to(device=device, dtype=torch.bool)
            else:
                changed_mask &= False
        after_risk_gate_count = int(changed_mask.sum().item())
        persistent_mask = self._build_post_rebind_persistent_newborn_mask(
            selected_idx,
            refresh_info=refresh_info,
            target_mask=target_mask,
            iteration=iteration,
        )
        if persistent_mask is not None and persistent_mask.shape[0] == selected_idx.shape[0]:
            changed_mask &= persistent_mask
        changed_count = int(changed_mask.sum().item())
        if bool(self.cfg.get('binding_densify_debug_verbose', False)):
            print(
                '[GaussianModel] post-rebind reset gates '
                f'iter={iteration} target={target_count} risky={risky_child_count} '
                f'shift_candidates={after_shift_count} face_changed={face_changed_count} '
                f'switched={switched_count} '
                f'kept_prior={kept_prior_count} '
                f'kept_prior_structural={kept_prior_structural_count} '
                f'face_rescue={face_rescue_count} '
                f'joint_changed={joint_changed_count} '
                f'switched_support={switched_support_count} '
                f'require_joint_change={int(require_joint_change)} '
                f'after_joint_gate={after_joint_gate_count} '
                f'current_risky={current_risky_count} risk_fallback={risk_fallback_count} '
                f'after_risk_gate={after_risk_gate_count} '
                f'final={changed_count}'
            )

        correction_mask = torch.zeros_like(target_mask)
        corrected_count = 0
        if enable_reset and bool(changed_mask.any().item()):
            correction_mask[selected_idx[changed_mask]] = True
            correction_score = None
            if effective_anchor_shift.shape[0] == selected_idx.shape[0]:
                correction_score = torch.zeros_like(correction_mask, dtype=torch.float32)
                correction_score[selected_idx] = effective_anchor_shift
            correction_mask = self._cap_mask_by_score(
                correction_mask,
                score=correction_score,
                max_points=int(self.cfg.get('binding_densify_postrebind_reset_max_points', 256)),
                max_ratio=float(self.cfg.get('binding_densify_postrebind_reset_max_ratio', 0.0)),
                min_score=reset_min_shift,
            )

            opacity_factor = float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_postrebind_opacity_factor', 0.72),
                default=0.72,
            ))
            scale_factor = float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_postrebind_scale_factor', 0.97),
                default=0.97,
            ))
            feature_dc_factor = float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_postrebind_feature_dc_factor', 0.98),
                default=0.98,
            ))
            feature_rest_factor = float(resolve_schedule_value(
                iteration,
                self.cfg.get('binding_densify_postrebind_feature_rest_factor', 0.75),
                default=0.75,
            ))
            update_appearance = bool(self.cfg.get('binding_densify_postrebind_reset_appearance_enable', False))

            with torch.no_grad():
                if update_appearance:
                    self._features_dc.data[correction_mask] *= feature_dc_factor
                    self._features_rest.data[correction_mask] *= feature_rest_factor

                actual_opacity = self.opacity_activation(self._opacity.data[correction_mask]) * opacity_factor
                actual_opacity = actual_opacity.clamp(1e-4, 1.0 - 1e-4)
                self._opacity.data[correction_mask] = self.inverse_opacity_activation(actual_opacity)

                actual_scaling = self.scaling_activation(self._scaling.data[correction_mask]) * scale_factor
                actual_scaling = actual_scaling.clamp_min(1e-6)
                self._scaling.data[correction_mask] = self.scaling_inverse_activation(actual_scaling)

                self._boundary_opacity_residual.data[correction_mask] = 0
                self._boundary_scaling_residual.data[correction_mask] = 0

            if update_appearance:
                self._zero_optimizer_state_rows("f_dc", correction_mask)
                self._zero_optimizer_state_rows("f_rest", correction_mask)
            self._zero_optimizer_state_rows("opacity", correction_mask)
            self._zero_optimizer_state_rows("scaling", correction_mask)
            self._zero_optimizer_state_rows("boundary_opacity_residual", correction_mask)
            self._zero_optimizer_state_rows("boundary_scaling_residual", correction_mask)

            if self.has_binding_state():
                risky_child_state = self.binding_state.get('densify_risky_child_mask', None)
                if torch.is_tensor(risky_child_state) and risky_child_state.shape[0] == self.get_xyz.shape[0]:
                    self.binding_state['densify_risky_child_mask'] = risky_child_state.to(device=device, dtype=torch.bool) & (~correction_mask)

            corrected_count = int(correction_mask.sum().item())

        pruned_count = 0
        prune_mask = self._build_post_rebind_unresolved_prune_mask(refresh_info, iteration=iteration)
        lineage_offender_stats = self._accumulate_post_rebind_lineage_offender_scores(
            refresh_info,
            correction_mask=correction_mask,
            prune_mask=prune_mask,
            iteration=iteration,
        )
        if lineage_offender_stats is None:
            lineage_offender_stats = {
                'target_children': target_count,
                'still_risky_children': 0,
                'scored_children': 0,
                'active_lineages': 0,
                'lineage_points': 0,
                'mean_child_risk_score': 0.0,
                'max_child_risk_score': 0.0,
            }
        lineage_offender_points = int(lineage_offender_stats.get('lineage_points', 0))
        parent_cleanup_count = 0
        cleanup_child_mask = correction_mask.clone()
        if prune_mask is not None and prune_mask.shape[0] == cleanup_child_mask.shape[0]:
            cleanup_child_mask |= prune_mask.to(device=device, dtype=torch.bool)
        parent_cleanup_mask = self._resolve_post_rebind_source_parent_mask(
            refresh_info,
            cleanup_child_mask,
            iteration=iteration,
        )
        if parent_cleanup_mask is not None and bool(parent_cleanup_mask.any().item()):
            parent_cleanup_count = self._apply_post_rebind_source_parent_cleanup(
                parent_cleanup_mask,
                iteration=iteration,
            )
        if prune_mask is not None and bool(prune_mask.any().item()):
            pruned_count = int(prune_mask.sum().item())
            self.prune_points(prune_mask)

        if bool(self.cfg.get('binding_densify_postrebind_verbose', False)):
            print(
                '[GaussianModel] post-rebind summary '
                f'iter={iteration} target={target_count} risky={risky_child_count} '
                f'changed={changed_count} shift_mean={anchor_shift_mean:.5f} shift_max={anchor_shift_max:.5f} '
                f'still_risky={int(lineage_offender_stats.get("still_risky_children", 0))} '
                f'scored_children={int(lineage_offender_stats.get("scored_children", 0))} '
                f'active_lineages={int(lineage_offender_stats.get("active_lineages", 0))} '
                f'lineage_points={lineage_offender_points} corrected={corrected_count} '
                f'pruned={pruned_count} parent_cleanup={parent_cleanup_count} '
                f'risk_mean={float(lineage_offender_stats.get("mean_child_risk_score", 0.0)):.5f} '
                f'risk_max={float(lineage_offender_stats.get("max_child_risk_score", 0.0)):.5f}'
            )
            if corrected_count > 0:
                print(
                    f'[GaussianModel] post-rebind corrected {corrected_count} risky newborn children '
                    f'at iter {iteration}'
                )
            if pruned_count > 0:
                print(
                    f'[GaussianModel] post-rebind pruned {pruned_count} unresolved risky newborn children '
                    f'at iter {iteration}'
                )
            if parent_cleanup_count > 0:
                print(
                    f'[GaussianModel] post-rebind suppressed {parent_cleanup_count} source parents '
                    f'at iter {iteration}'
                )
            if lineage_offender_points > 0:
                print(
                    f'[GaussianModel] post-rebind tagged {lineage_offender_points} lineage-linked offender points '
                    f'at iter {iteration}'
                )
        return corrected_count + pruned_count + parent_cleanup_count

    def _build_post_rebind_persistent_newborn_mask(self, selected_idx, refresh_info=None, target_mask=None, iteration=0):
        if not bool(self.cfg.get('binding_densify_postrebind_reset_persistent_only', True)):
            return None
        if selected_idx is None or selected_idx.numel() == 0 or not self.has_binding_state():
            return None

        binding_state = self.get_binding_state()
        point_count = self.get_xyz.shape[0]
        device = self._xyz.device
        persistent_mask = torch.ones((selected_idx.shape[0],), dtype=torch.bool, device=device)
        has_constraint = False

        densify_birth_iter = binding_state.get('densify_birth_iter', None)
        min_child_age = int(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_postrebind_reset_min_child_age', 0),
            default=0,
        ))
        if (
            min_child_age > 0
            and torch.is_tensor(densify_birth_iter)
            and densify_birth_iter.shape[0] == point_count
        ):
            child_age = max(iteration, 0) - densify_birth_iter.to(device=device, dtype=torch.long)[selected_idx]
            persistent_mask &= child_age >= min_child_age
            has_constraint = True

        min_lineage_observations = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_postrebind_reset_min_lineage_observations', 1.0),
            default=1.0,
        ))
        selected_lineage_id = None
        selected_lineage_valid_count = 0
        selected_lineage_unique = 0
        lineage_count_ready = False
        lineage_mode = 'none'
        if target_mask is not None:
            selected_lineage_id = self._slice_refresh_info_tensor(
                refresh_info,
                'source_root_lineage_id',
                target_mask=target_mask,
                device=device,
                dtype=torch.long,
            )
        if torch.is_tensor(selected_lineage_id) and selected_lineage_id.shape[0] == selected_idx.shape[0]:
            valid_selected_lineage = selected_lineage_id >= 0
            selected_lineage_valid_count = int(valid_selected_lineage.sum().item())
            if bool(valid_selected_lineage.any().item()):
                selected_lineage_unique = int(selected_lineage_id[valid_selected_lineage].unique().numel())
        if min_lineage_observations > 0.0:
            offender_count_accum = None
            if (
                torch.is_tensor(self._lineage_offender_count_accum)
                and self._lineage_offender_count_accum.shape[0] == point_count
            ):
                offender_count_accum = self._lineage_offender_count_accum.to(device=device, dtype=torch.float32)
                lineage_count_ready = True

            lineage_mask = None
            if torch.is_tensor(selected_lineage_id) and selected_lineage_id.shape[0] == selected_idx.shape[0]:
                current_root_lineage_id = binding_state.get('densify_root_lineage_id', None)
                if not torch.is_tensor(current_root_lineage_id) or current_root_lineage_id.shape[0] != point_count:
                    current_root_lineage_id = binding_state.get('densify_lineage_id', None)
                if torch.is_tensor(current_root_lineage_id) and current_root_lineage_id.shape[0] == point_count:
                    current_root_lineage_id = current_root_lineage_id.to(device=device, dtype=torch.long)
                else:
                    current_root_lineage_id = None

                lineage_mask = torch.zeros((selected_idx.shape[0],), dtype=torch.bool, device=device)
                for lineage_id in selected_lineage_id.unique().tolist():
                    lineage_id = int(lineage_id)
                    current_selected = selected_lineage_id == lineage_id
                    if not bool(current_selected.any().item()):
                        continue

                    # Cold-start fail-open: the current target batch itself counts as
                    # one lineage observation. Historical offender counts only add support.
                    lineage_observations = float(current_selected.sum().item())
                    if lineage_count_ready:
                        if lineage_id < 0:
                            lineage_observations = max(
                                lineage_observations,
                                float(offender_count_accum[selected_idx[current_selected]].max().item()),
                            )
                        elif current_root_lineage_id is not None:
                            lineage_points = current_root_lineage_id == lineage_id
                            if bool(lineage_points.any().item()):
                                lineage_observations = max(
                                    lineage_observations,
                                    float(offender_count_accum[lineage_points].max().item()),
                                )
                    lineage_mask[current_selected] = lineage_observations >= min_lineage_observations
                lineage_mode = 'selected_lineage'
            elif lineage_count_ready:
                lineage_mask = offender_count_accum[selected_idx] >= min_lineage_observations
                lineage_mode = 'selected_points'

            if lineage_mask is not None:
                persistent_mask &= lineage_mask
                has_constraint = True
            elif lineage_count_ready:
                persistent_mask &= offender_count_accum[selected_idx] >= min_lineage_observations
                lineage_mode = 'selected_points'
                has_constraint = True
            else:
                persistent_mask &= False
                lineage_mode = 'missing_state'
                has_constraint = True

        if bool(self.cfg.get('binding_densify_postrebind_verbose', False)) and has_constraint:
            print(
                '[GaussianModel] post-rebind persistent gate '
                f'iter={iteration} selected={int(selected_idx.shape[0])} '
                f'kept={int(persistent_mask.sum().item())} '
                f'lineage_valid={selected_lineage_valid_count} '
                f'unique_lineages={selected_lineage_unique} '
                f'lineage_count_ready={int(lineage_count_ready)} '
                f'mode={lineage_mode}'
            )
        return persistent_mask if has_constraint else None

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
            self._boundary_opacity_residual = self._replace_optimizer_parameter(
                "boundary_opacity_residual",
                boundary_opacity_residual,
            )
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
            self._boundary_scaling_residual = self._replace_optimizer_parameter(
                "boundary_scaling_residual",
                boundary_scaling_residual,
            )
            changed.append("boundary_scaling_residual")

        if verbose and changed:
            print(
                "[GaussianModel] boundary state resynced for "
                f"{point_count} points: {', '.join(changed)}"
            )
        return len(changed) > 0

    def reset_boundary_residuals(self):
        if self.get_xyz.numel() <= 0:
            self._boundary_opacity_residual = torch.empty(0, device=self._xyz.device if torch.is_tensor(self._xyz) else None)
            self._boundary_scaling_residual = torch.empty(0, 3, device=self._xyz.device if torch.is_tensor(self._xyz) else None)
            return
        device = self._xyz.device
        self._boundary_opacity_residual = nn.Parameter(torch.zeros((self.get_xyz.shape[0], 1), dtype=torch.float, device=device).requires_grad_(True))
        self._boundary_scaling_residual = nn.Parameter(torch.zeros((self.get_xyz.shape[0], 3), dtype=torch.float, device=device).requires_grad_(True))

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

    def _prune_binding_state(self, valid_points_mask):
        if not self.has_binding_state():
            return
        old_point_count = int(valid_points_mask.shape[0])
        device = valid_points_mask.device
        kept_old_idx = torch.nonzero(valid_points_mask, as_tuple=False).squeeze(-1)
        new_point_count = int(kept_old_idx.shape[0])
        old_to_new = torch.full((old_point_count,), -1, dtype=torch.long, device=device)
        if new_point_count > 0:
            old_to_new[kept_old_idx] = torch.arange(new_point_count, device=device, dtype=torch.long)

        remap_index_keys = {
            'densify_parent_index',
            'densify_root_parent_index',
        }
        remap_debug_stats = {}
        pruned_state = {}
        for key, value in self.binding_state.items():
            if not torch.is_tensor(value) or value.shape[0] != old_point_count:
                continue

            pruned_value = value[valid_points_mask]
            if key in remap_index_keys:
                remapped_value = torch.full_like(pruned_value, -1, dtype=torch.long, device=device)
                parent_index = pruned_value.to(device=device, dtype=torch.long).reshape(-1)
                valid_parent = (parent_index >= 0) & (parent_index < old_point_count)
                if bool(valid_parent.any().item()):
                    remapped_value[valid_parent] = old_to_new[parent_index[valid_parent]]
                remap_debug_stats[key] = {
                    'valid_before': int(valid_parent.sum().item()),
                    'valid_after': int((remapped_value >= 0).sum().item()),
                }
                pruned_value = remapped_value
            pruned_state[key] = pruned_value

        self.binding_state = pruned_state
        if bool(self.cfg.get('binding_densify_debug_verbose', False)) and remap_debug_stats:
            debug_parts = []
            for key in ('densify_parent_index', 'densify_root_parent_index'):
                stats = remap_debug_stats.get(key, None)
                if stats is None:
                    continue
                debug_parts.append(
                    f'{key}={stats["valid_before"]}->{stats["valid_after"]}'
                )
            if debug_parts:
                print(
                    '[GaussianModel] prune binding index remap '
                    f'kept={new_point_count}/{old_point_count} ' + ' '.join(debug_parts)
                )

    def set_fwd_transform(self, T_fwd):
        self.fwd_transform = T_fwd

    def color_by_opacity(self):
        cloned = self.clone()
        cloned._features_dc = self.get_opacity.unsqueeze(-1).expand(-1,-1,3)
        cloned._features_rest = torch.zeros_like(cloned._features_rest)
        return cloned

    def capture(self):
        self.ensure_boundary_state_matches_points(verbose=False)
        self.ensure_offender_state_matches_points(verbose=False)
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
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
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
        else:
            raise ValueError(f'Unexpected GaussianModel checkpoint format with {len(model_args)} entries.')
        self.ensure_boundary_state_matches_points(verbose=True)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        try:
            if len(opt_dict.get('param_groups', [])) == len(self.optimizer.param_groups):
                self.optimizer.load_state_dict(opt_dict)
            else:
                print('[GaussianModel] optimizer param group count changed; skipping optimizer state restore.')
        except Exception as exc:
            print(f'[GaussianModel] failed to restore optimizer state ({exc}); continuing with fresh optimizer state.')

    @property
    def get_scaling(self):
        raw_scaling = self._scaling
        if torch.is_tensor(self._boundary_scaling_residual) and self._boundary_scaling_residual.numel() == self._scaling.numel() and self.has_boundary_tags():
            raw_scaling = raw_scaling + self._boundary_scaling_residual * self._boundary_tag.unsqueeze(-1)
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
        if torch.is_tensor(self._boundary_opacity_residual) and self._boundary_opacity_residual.numel() == self._opacity.numel() and self.has_boundary_tags():
            raw_opacity = raw_opacity + self._boundary_opacity_residual * self._boundary_tag.unsqueeze(-1)
        return self.opacity_activation(raw_opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        if hasattr(self, 'rotation_precomp'):
            return self.covariance_activation(self.get_scaling, scaling_modifier, self.rotation_precomp)
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

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
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.ensure_boundary_state_matches_points(verbose=False)
        self.ensure_offender_state_matches_points(verbose=False)
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        feature_ratio = 20.0 if self.use_sh else 1.0
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / feature_ratio, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._boundary_opacity_residual], 'lr': training_args.get('boundary_opacity_residual_lr', 0.0), "name": "boundary_opacity_residual"},
            {'params': [self._boundary_scaling_residual], 'lr': training_args.get('boundary_scaling_residual_lr', 0.0), "name": "boundary_scaling_residual"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        l.append('boundary_tag')
        l.append('boundary_opacity_residual')
        for i in range(self._boundary_scaling_residual.shape[1]):
            l.append('boundary_scale_residual_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        self.ensure_boundary_state_matches_points(verbose=False)
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
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, boundary_tag, boundary_opacity_residual, boundary_scaling_residual, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_offender_subset(self, reset_mask, opacity_factor=0.35, scaling_factor=0.75, max_opacity=None, max_scaling=None, boundary_tag_value=None):
        if reset_mask is None:
            return 0
        if not torch.is_tensor(reset_mask):
            reset_mask = torch.tensor(reset_mask, device=self._xyz.device if torch.is_tensor(self._xyz) else 'cuda')
        reset_mask = reset_mask.detach().reshape(-1).bool()
        point_count = int(self.get_xyz.shape[0]) if torch.is_tensor(self._xyz) and self._xyz.ndim >= 2 else 0
        if point_count <= 0:
            return 0
        if reset_mask.shape[0] != point_count:
            raise ValueError(f'reset_mask shape mismatch: got {reset_mask.shape[0]}, expected {point_count}')
        reset_count = int(reset_mask.sum().item())
        if reset_count <= 0:
            return 0

        self.ensure_boundary_state_matches_points(verbose=False)
        self.ensure_offender_state_matches_points(verbose=False)

        device = self._xyz.device
        reset_mask = reset_mask.to(device=device)
        current_opacity = self.get_opacity.detach().clone()
        current_scaling = self.get_scaling.detach().clone()

        if torch.is_tensor(self._boundary_opacity_residual) and self._boundary_opacity_residual.shape[0] == point_count:
            new_boundary_opacity_residual = self._boundary_opacity_residual.detach().clone()
            new_boundary_opacity_residual[reset_mask] = 0.0
            optimizable_tensors = self.replace_tensor_to_optimizer(new_boundary_opacity_residual, "boundary_opacity_residual")
            self._boundary_opacity_residual = optimizable_tensors["boundary_opacity_residual"]

        if torch.is_tensor(self._boundary_scaling_residual) and self._boundary_scaling_residual.shape[0] == point_count:
            new_boundary_scaling_residual = self._boundary_scaling_residual.detach().clone()
            new_boundary_scaling_residual[reset_mask] = 0.0
            optimizable_tensors = self.replace_tensor_to_optimizer(new_boundary_scaling_residual, "boundary_scaling_residual")
            self._boundary_scaling_residual = optimizable_tensors["boundary_scaling_residual"]

        min_opacity = 1.0e-4
        target_opacity = current_opacity.clamp(min=min_opacity, max=1.0 - 1.0e-6)
        opacity_factor = float(min(max(opacity_factor, 0.0), 1.0))
        if opacity_factor != 1.0:
            target_opacity[reset_mask] = target_opacity[reset_mask] * opacity_factor
        if max_opacity is not None:
            max_opacity = float(min(max(max_opacity, min_opacity), 1.0 - 1.0e-6))
            target_opacity[reset_mask] = torch.minimum(
                target_opacity[reset_mask],
                torch.full_like(target_opacity[reset_mask], max_opacity),
            )
        new_opacity = self._opacity.detach().clone()
        new_opacity[reset_mask] = inverse_sigmoid(target_opacity[reset_mask].clamp(min=min_opacity, max=1.0 - 1.0e-6))
        optimizable_tensors = self.replace_tensor_to_optimizer(new_opacity, "opacity")
        self._opacity = optimizable_tensors["opacity"]

        min_scaling = 1.0e-4
        target_scaling = current_scaling.clamp_min(min_scaling)
        scaling_factor = float(max(scaling_factor, 1.0e-4))
        if scaling_factor != 1.0:
            target_scaling[reset_mask] = target_scaling[reset_mask] * scaling_factor
        if max_scaling is not None:
            max_scaling = float(max(max_scaling, min_scaling))
            target_scaling[reset_mask] = torch.minimum(
                target_scaling[reset_mask],
                torch.full_like(target_scaling[reset_mask], max_scaling),
            )
        new_scaling = self._scaling.detach().clone()
        new_scaling[reset_mask] = self.scaling_inverse_activation(target_scaling[reset_mask].clamp_min(min_scaling))
        optimizable_tensors = self.replace_tensor_to_optimizer(new_scaling, "scaling")
        self._scaling = optimizable_tensors["scaling"]

        if boundary_tag_value is not None and torch.is_tensor(self._boundary_tag) and self._boundary_tag.shape[0] == point_count:
            self._boundary_tag = self._boundary_tag.detach().clone()
            self._boundary_tag[reset_mask] = float(boundary_tag_value)

        if torch.is_tensor(self.xyz_gradient_accum) and self.xyz_gradient_accum.shape[0] == point_count:
            self.xyz_gradient_accum[reset_mask] = 0.0
        if torch.is_tensor(self.denom) and self.denom.shape[0] == point_count:
            self.denom[reset_mask] = 0.0
        if torch.is_tensor(self.max_radii2D) and self.max_radii2D.shape[0] == point_count:
            self.max_radii2D[reset_mask] = 0.0
        if self.has_offender_state():
            self._offender_score_accum[reset_mask] = 0.0
            self._offender_count_accum[reset_mask] = 0.0
        if self.has_lineage_offender_state():
            self._lineage_offender_score_accum[reset_mask] = 0.0
            self._lineage_offender_count_accum[reset_mask] = 0.0
        return reset_count

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        boundary_tag = np.asarray(plydata.elements[0]["boundary_tag"])[..., np.newaxis] if "boundary_tag" in plydata.elements[0].data.dtype.names else np.zeros_like(opacities)
        boundary_opacity_residual = np.asarray(plydata.elements[0]["boundary_opacity_residual"])[..., np.newaxis] if "boundary_opacity_residual" in plydata.elements[0].data.dtype.names else np.zeros_like(opacities)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        boundary_scale_residual_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("boundary_scale_residual_")]
        boundary_scale_residual_names = sorted(boundary_scale_residual_names, key = lambda x: int(x.split('_')[-1]))
        boundary_scaling_residual = np.zeros((xyz.shape[0], 3), dtype=np.float32)
        for idx, attr_name in enumerate(boundary_scale_residual_names[:3]):
            boundary_scaling_residual[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._boundary_tag = torch.tensor(boundary_tag[:, 0], dtype=torch.float, device="cuda")
        self._boundary_opacity_residual = nn.Parameter(torch.tensor(boundary_opacity_residual, dtype=torch.float, device="cuda").requires_grad_(True))
        self._boundary_scaling_residual = nn.Parameter(torch.tensor(boundary_scaling_residual, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                if stored_state is not None:
                    self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        self.ensure_boundary_state_matches_points(verbose=False)
        self.ensure_offender_state_matches_points(verbose=False)
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._boundary_opacity_residual = optimizable_tensors["boundary_opacity_residual"]
        self._boundary_scaling_residual = optimizable_tensors["boundary_scaling_residual"]
        self._boundary_opacity_residual = optimizable_tensors["boundary_opacity_residual"]
        self._boundary_scaling_residual = optimizable_tensors["boundary_scaling_residual"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.has_boundary_tag_state():
            self._boundary_tag = self._boundary_tag[valid_points_mask]
        live_boundary_score = self.get_live_boundary_score_state()
        if live_boundary_score is not None:
            self.set_live_boundary_score_state(live_boundary_score[valid_points_mask])
        elif hasattr(self, 'binding_boundary_live_score'):
            delattr(self, 'binding_boundary_live_score')
        self._prune_binding_state(valid_points_mask)
        if bool(self.cfg.get('binding_densify_debug_verbose', False)) and self.has_binding_state():
            print(
                '[GaussianModel] binding state after prune '
                f'{self._binding_mask_debug_summary(self.binding_state)}'
            )
        if self.has_offender_state():
            self._offender_score_accum = self._offender_score_accum[valid_points_mask]
            self._offender_count_accum = self._offender_count_accum[valid_points_mask]
        if self.has_offender_refill_state():
            self._offender_refill_score = self._offender_refill_score[valid_points_mask]
        if self.has_lineage_offender_state():
            self._lineage_offender_score_accum = self._lineage_offender_score_accum[valid_points_mask]
            self._lineage_offender_count_accum = self._lineage_offender_count_accum[valid_points_mask]

    def prune_nonfinite_points(self, verbose=True):
        if self._xyz.numel() == 0:
            return 0
        invalid_mask = ~torch.isfinite(self._xyz).all(dim=1)
        invalid_mask |= ~torch.isfinite(self._features_dc.reshape(self._features_dc.shape[0], -1)).all(dim=1)
        invalid_mask |= ~torch.isfinite(self._features_rest.reshape(self._features_rest.shape[0], -1)).all(dim=1)
        invalid_mask |= ~torch.isfinite(self._opacity.reshape(self._opacity.shape[0], -1)).all(dim=1)
        invalid_mask |= ~torch.isfinite(self._scaling.reshape(self._scaling.shape[0], -1)).all(dim=1)
        invalid_mask |= ~torch.isfinite(self._rotation.reshape(self._rotation.shape[0], -1)).all(dim=1)
        removed = int(invalid_mask.sum().item())
        if removed <= 0:
            return 0
        if removed >= self._xyz.shape[0]:
            raise RuntimeError('All Gaussian points became non-finite.')
        self.prune_points(invalid_mask)
        if verbose:
            print(f'[GaussianModel] pruned {removed} non-finite points.')
        return removed

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_binding_state=None, new_boundary_tags=None, new_boundary_opacity_residual=None, new_boundary_scaling_residual=None, new_live_boundary_score=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "boundary_opacity_residual": new_boundary_opacity_residual,
        "boundary_scaling_residual": new_boundary_scaling_residual}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._boundary_opacity_residual = optimizable_tensors["boundary_opacity_residual"]
        self._boundary_scaling_residual = optimizable_tensors["boundary_scaling_residual"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._append_binding_state(new_binding_state)
        self._append_boundary_tags(new_boundary_tags)
        self._append_live_boundary_score_state(new_live_boundary_score)
        self._append_offender_state(new_xyz.shape[0])
        self.ensure_boundary_state_matches_points(verbose=False)
        self.ensure_offender_state_matches_points(verbose=False)
        if bool(self.cfg.get('binding_densify_debug_verbose', False)) and self.has_binding_state():
            print(
                '[GaussianModel] binding state after append '
                f'{self._binding_mask_debug_summary(self.binding_state)}'
            )

    def _append_conservative_clones_for_risky_parents(self, selected_pts_mask, iteration=0):
        if selected_pts_mask is None or selected_pts_mask.numel() == 0 or not bool(selected_pts_mask.any().item()):
            return 0

        parent_idx = torch.nonzero(selected_pts_mask, as_tuple=False).squeeze(-1)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_boundary_opacity_residual = self._boundary_opacity_residual[selected_pts_mask]
        new_boundary_scaling_residual = self._boundary_scaling_residual[selected_pts_mask]
        new_binding_state = self._clear_newborn_binding_flags(self._slice_binding_state(selected_pts_mask))
        new_boundary_tags = self._slice_boundary_tags(selected_pts_mask)
        new_live_boundary_score = self._slice_live_boundary_score_state(selected_pts_mask)
        new_binding_state = self._annotate_densified_binding_lineage(
            new_binding_state,
            parent_idx,
            iteration=iteration,
        )

        clone_risk_mask = torch.ones((new_xyz.shape[0],), dtype=torch.bool, device=new_xyz.device)
        (
            new_binding_state,
            new_opacities,
            new_scaling,
            new_features_dc,
            new_features_rest,
            new_boundary_opacity_residual,
            new_boundary_scaling_residual,
        ) = self._attenuate_densified_children(
            new_binding_state,
            new_boundary_tags,
            new_opacities,
            new_scaling,
            new_features_dc,
            new_features_rest,
            new_boundary_opacity_residual,
            new_boundary_scaling_residual,
            iteration=iteration,
            extra_risk_mask=clone_risk_mask,
            refresh_risk_mask=clone_risk_mask,
        )

        clone_opacity_factor = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_risky_parent_clone_opacity_factor', 0.72),
            default=0.72,
        ))
        clone_scale_factor = float(resolve_schedule_value(
            iteration,
            self.cfg.get('binding_densify_risky_parent_clone_scale_factor', 0.82),
            default=0.82,
        ))

        actual_opacity = self.opacity_activation(new_opacities) * clone_opacity_factor
        actual_opacity = actual_opacity.clamp(1e-4, 1.0 - 1e-4)
        new_opacities = self.inverse_opacity_activation(actual_opacity)

        actual_scaling = self.scaling_activation(new_scaling) * clone_scale_factor
        actual_scaling = actual_scaling.clamp_min(1e-6)
        new_scaling = self.scaling_inverse_activation(actual_scaling)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_binding_state=new_binding_state,
            new_boundary_tags=new_boundary_tags,
            new_boundary_opacity_residual=new_boundary_opacity_residual,
            new_boundary_scaling_residual=new_boundary_scaling_residual,
            new_live_boundary_score=new_live_boundary_score,
        )

        clone_count = int(new_xyz.shape[0])
        if bool(self.cfg.get('binding_densify_risky_parent_clone_verbose', True)):
            print(
                f'[GaussianModel] diverted {clone_count} risky split parents to conservative clone path '
                f'at iter {iteration}'
            )
        return clone_count

    def densify_and_split(self, grads, grad_threshold, scene_extent, iteration=0, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        selected_pts_mask = self._filter_densify_candidates(selected_pts_mask, iteration=iteration)
        preserve_parent_mask = self._preserve_split_parents_for_risky_boundary_points(
            selected_pts_mask,
            iteration=iteration,
        )
        safe_selected_pts_mask = selected_pts_mask & (~preserve_parent_mask)
        risky_parent_idx = torch.nonzero(preserve_parent_mask, as_tuple=False).squeeze(-1)

        if bool(safe_selected_pts_mask.any().item()):
            safe_parent_idx = torch.nonzero(safe_selected_pts_mask, as_tuple=False).squeeze(-1)
            stds = self.get_scaling[safe_selected_pts_mask].repeat(N,1)
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[safe_selected_pts_mask]).repeat(N,1,1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[safe_selected_pts_mask].repeat(N, 1)
            new_scaling = self.scaling_inverse_activation(self.get_scaling[safe_selected_pts_mask].repeat(N,1) / (0.8*N))
            new_rotation = self._rotation[safe_selected_pts_mask].repeat(N,1)
            new_features_dc = self._features_dc[safe_selected_pts_mask].repeat(N,1,1)
            new_features_rest = self._features_rest[safe_selected_pts_mask].repeat(N,1,1)
            new_opacity = self._opacity[safe_selected_pts_mask].repeat(N,1)
            new_boundary_opacity_residual = self._boundary_opacity_residual[safe_selected_pts_mask].repeat(N,1)
            new_binding_state = self._clear_newborn_binding_flags(
                self._slice_binding_state(safe_selected_pts_mask, repeats=N)
            )
            new_boundary_tags = self._slice_boundary_tags(safe_selected_pts_mask, repeats=N)
            new_live_boundary_score = self._slice_live_boundary_score_state(safe_selected_pts_mask, repeats=N)
            new_binding_state = self._annotate_densified_binding_lineage(
                new_binding_state,
                safe_parent_idx.repeat(N),
                iteration=iteration,
            )
            base_new_binding_state = new_binding_state
            new_binding_state = base_new_binding_state
            parent_xyz = self.get_xyz[safe_selected_pts_mask].repeat(N, 1)
            new_binding_state = self._update_binding_offsets(base_new_binding_state, new_xyz - parent_xyz)
            predictive_risk_mask = None
            if bool(self.cfg.get('binding_densify_directional_split_enable', True)) and bool(
                self.cfg.get('binding_densify_predictive_directional_split_enable', True)
            ):
                predictive_risk_mask = self._predict_densified_child_risk_mask(
                    new_binding_state,
                    new_boundary_tags,
                    iteration=iteration,
                )
                if torch.is_tensor(predictive_risk_mask) and bool(predictive_risk_mask.any().item()):
                    new_xyz = self._apply_directional_split_to_risky_children(
                        parent_xyz,
                        new_xyz,
                        new_binding_state,
                        predictive_risk_mask,
                        iteration=iteration,
                    )
                    new_binding_state = self._update_binding_offsets(base_new_binding_state, new_xyz - parent_xyz)
            immediate_refresh_mask = self._predict_densified_child_immediate_refresh_mask(
                new_binding_state,
                new_boundary_tags,
                iteration=iteration,
            )
            new_boundary_scaling_residual = self._boundary_scaling_residual[safe_selected_pts_mask].repeat(N,1)
            (
                new_binding_state,
                new_opacity,
                new_scaling,
                new_features_dc,
                new_features_rest,
                new_boundary_opacity_residual,
                new_boundary_scaling_residual,
            ) = self._attenuate_densified_children(
                new_binding_state,
                new_boundary_tags,
                new_opacity,
                new_scaling,
                new_features_dc,
                new_features_rest,
                new_boundary_opacity_residual,
                new_boundary_scaling_residual,
                iteration=iteration,
                extra_risk_mask=predictive_risk_mask,
                refresh_risk_mask=immediate_refresh_mask,
            )
            if bool(self.cfg.get('binding_densify_debug_verbose', False)):
                print(
                    '[GaussianModel] newborn split binding state '
                    f'iter={iteration} {self._binding_mask_debug_summary(new_binding_state)}'
                )

            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_binding_state=new_binding_state, new_boundary_tags=new_boundary_tags, new_boundary_opacity_residual=new_boundary_opacity_residual, new_boundary_scaling_residual=new_boundary_scaling_residual, new_live_boundary_score=new_live_boundary_score)

        if bool(self.cfg.get('binding_densify_risky_parent_clone_enable', True)) and risky_parent_idx.numel() > 0:
            risky_clone_mask = torch.zeros((self.get_xyz.shape[0],), device="cuda", dtype=torch.bool)
            risky_clone_mask[risky_parent_idx] = True
            self._append_conservative_clones_for_risky_parents(
                risky_clone_mask,
                iteration=iteration,
            )

        if bool(safe_selected_pts_mask.any().item()):
            prune_filter = torch.zeros((self.get_xyz.shape[0],), device="cuda", dtype=torch.bool)
            prune_filter[:n_init_points] = safe_selected_pts_mask
            self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, iteration=0):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        selected_pts_mask = self._filter_densify_candidates(selected_pts_mask, iteration=iteration)
        selected_pts_mask = self._augment_clone_densify_candidates(
            selected_pts_mask,
            scene_extent,
            iteration=iteration,
        )
        parent_idx = torch.nonzero(selected_pts_mask, as_tuple=False).squeeze(-1)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_boundary_opacity_residual = self._boundary_opacity_residual[selected_pts_mask]
        new_boundary_scaling_residual = self._boundary_scaling_residual[selected_pts_mask]
        parent_binding_state = self._slice_binding_state(selected_pts_mask)
        new_binding_state = self._clear_newborn_binding_flags(parent_binding_state)
        new_boundary_tags = self._slice_boundary_tags(selected_pts_mask)
        new_live_boundary_score = self._slice_live_boundary_score_state(selected_pts_mask)
        new_binding_state = self._annotate_densified_binding_lineage(
            new_binding_state,
            parent_idx,
            iteration=iteration,
        )
        clone_parent_risk_mask = None
        if bool(self.cfg.get('binding_densify_clone_immediate_refresh_parent_risk_enable', False)):
            point_count = int(new_xyz.shape[0])
            device = new_xyz.device
            if point_count > 0:
                clone_parent_risk_mask = self._binding_boundary_arm_risk_mask(
                    parent_binding_state,
                    point_count,
                    device,
                    iteration=iteration,
                    boundary_tags=new_boundary_tags,
                    boundary_only=bool(self.cfg.get(
                        'binding_densify_clone_immediate_refresh_parent_risk_boundary_only',
                        True,
                    )),
                    boundary_threshold=float(resolve_schedule_value(
                        iteration,
                        self.cfg.get(
                            'binding_densify_child_immediate_refresh_boundary_threshold',
                            self.cfg.get('binding_densify_child_boundary_threshold', 0.05),
                        ),
                        default=0.05,
                    )),
                    arm_only=bool(self.cfg.get(
                        'binding_densify_child_immediate_refresh_arm_only',
                        self.cfg.get('binding_densify_child_arm_only', True),
                    )),
                    semantic_scale=float(resolve_schedule_value(
                        iteration,
                        self.cfg.get('binding_densify_child_immediate_refresh_semantic_distance_scale', 1.15),
                        default=1.15,
                    )),
                    surface_scale=float(resolve_schedule_value(
                        iteration,
                        self.cfg.get('binding_densify_child_immediate_refresh_surface_distance_scale', 1.15),
                        default=1.15,
                    )),
                    confidence_margin=float(resolve_schedule_value(
                        iteration,
                        self.cfg.get('binding_densify_child_immediate_refresh_confidence_margin', -0.05),
                        default=-0.05,
                    )),
                    weight_gap_scale=float(resolve_schedule_value(
                        iteration,
                        self.cfg.get('binding_densify_child_immediate_refresh_weight_gap_scale', 0.8),
                        default=0.8,
                    )),
                    include_refresh_mask=bool(self.cfg.get(
                        'binding_densify_clone_immediate_refresh_parent_risk_include_refresh_mask',
                        True,
                    )),
                    arm_gate_mode=self.cfg.get(
                        'binding_densify_clone_immediate_refresh_parent_risk_arm_gate_mode',
                        self.cfg.get(
                            'binding_densify_child_immediate_refresh_arm_gate_mode',
                            'source_or_current',
                        ),
                    ),
                )
                if bool(self.cfg.get('binding_densify_debug_verbose', False)):
                    print(
                        '[GaussianModel] clone parent-risk bootstrap '
                        f'iter={iteration} points={point_count} '
                        f'parent_risk={int(clone_parent_risk_mask.sum().item()) if torch.is_tensor(clone_parent_risk_mask) else 0}'
                    )
        immediate_refresh_mask = self._predict_densified_child_immediate_refresh_mask(
            new_binding_state,
            new_boundary_tags,
            iteration=iteration,
            extra_risk_mask=clone_parent_risk_mask,
        )
        (
            new_binding_state,
            new_opacities,
            new_scaling,
            new_features_dc,
            new_features_rest,
            new_boundary_opacity_residual,
            new_boundary_scaling_residual,
        ) = self._attenuate_densified_children(
            new_binding_state,
            new_boundary_tags,
            new_opacities,
            new_scaling,
            new_features_dc,
            new_features_rest,
            new_boundary_opacity_residual,
            new_boundary_scaling_residual,
            iteration=iteration,
            refresh_risk_mask=immediate_refresh_mask,
        )
        if bool(self.cfg.get('binding_densify_debug_verbose', False)):
            print(
                '[GaussianModel] newborn clone binding state '
                f'iter={iteration} {self._binding_mask_debug_summary(new_binding_state)}'
            )

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_binding_state=new_binding_state, new_boundary_tags=new_boundary_tags, new_boundary_opacity_residual=new_boundary_opacity_residual, new_boundary_scaling_residual=new_boundary_scaling_residual, new_live_boundary_score=new_live_boundary_score)

    def densify_and_prune(self, opt, scene, max_screen_size, iteration=0):
        extent = scene.cameras_extent

        self.prune_nonfinite_points(verbose=True)

        max_grad = opt.densify_grad_threshold
        min_opacity = opt.opacity_threshold

        grads = self.xyz_gradient_accum / self.denom
        grads[~torch.isfinite(grads)] = 0.0

        self.densify_and_clone(grads, max_grad, extent, iteration=iteration)
        self.densify_and_split(grads, max_grad, extent, iteration=iteration)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
