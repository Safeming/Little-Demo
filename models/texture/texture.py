import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf

from utils.sh_utils import eval_sh, eval_sh_bases, augm_rots
from utils.general_utils import build_rotation
from models.network_utils import VanillaCondMLP


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


def _normalize_weighted_id_specs(spec):
    if spec is None:
        return []

    if OmegaConf.is_config(spec):
        spec_list = OmegaConf.to_container(spec, resolve=True)
    else:
        spec_list = spec
    if not isinstance(spec_list, list):
        spec_list = [spec_list]

    normalized = []
    for item in spec_list:
        if item is None:
            continue
        if isinstance(item, (int, float)):
            normalized.append((int(item), 1.0))
            continue
        if isinstance(item, (list, tuple)):
            if len(item) <= 0:
                continue
            weight = 1.0 if len(item) < 2 or item[1] is None else float(item[1])
            normalized.append((int(item[0]), weight))
            continue
        try:
            normalized.append((int(item), 1.0))
        except (TypeError, ValueError):
            continue
    return normalized


def _normalize_weighted_name_specs(spec):
    if spec is None:
        return []

    if OmegaConf.is_config(spec):
        spec_list = OmegaConf.to_container(spec, resolve=True)
    else:
        spec_list = spec
    if not isinstance(spec_list, list):
        spec_list = [spec_list]

    normalized = []
    for item in spec_list:
        if item is None:
            continue
        if isinstance(item, str):
            name = item.strip()
            if name:
                normalized.append((name, 1.0))
            continue
        if isinstance(item, (list, tuple)):
            if len(item) <= 0 or item[0] is None:
                continue
            name = str(item[0]).strip()
            if not name:
                continue
            weight = 1.0 if len(item) < 2 or item[1] is None else float(item[1])
            normalized.append((name, weight))
    return normalized


def _weighted_id_mask(ids, spec):
    if not torch.is_tensor(ids) or ids.numel() <= 0:
        return None

    members = _normalize_weighted_id_specs(spec)
    if len(members) <= 0:
        return None

    mask = torch.zeros((ids.shape[0],), device=ids.device, dtype=torch.float32)
    for member_id, weight in members:
        if weight <= 0.0:
            continue
        mask = torch.maximum(mask, (ids == member_id).float() * float(weight))
    return mask.clamp(0.0, 1.0)


def _weighted_name_mask(ids, name_table, spec):
    if not torch.is_tensor(ids) or ids.numel() <= 0 or name_table is None:
        return None

    members = _normalize_weighted_name_specs(spec)
    if len(members) <= 0:
        return None

    if OmegaConf.is_config(name_table):
        name_table = OmegaConf.to_container(name_table, resolve=True)
    if not isinstance(name_table, (list, tuple)):
        return None

    name_to_id = {}
    for idx, name in enumerate(name_table):
        if name is None:
            continue
        name_to_id[str(name).strip().lower()] = int(idx)

    id_members = []
    for name, weight in members:
        if weight <= 0.0:
            continue
        mapped_id = name_to_id.get(str(name).strip().lower(), None)
        if mapped_id is not None:
            id_members.append((mapped_id, weight))
    if len(id_members) <= 0:
        return None

    return _weighted_id_mask(ids, id_members)


def _combine_gate_terms(terms, mode='max'):
    valid_terms = [term.clamp(0.0, 1.0) for term in terms if torch.is_tensor(term)]
    if len(valid_terms) <= 0:
        return None

    mode = str(mode or 'max').lower()
    if mode in ('max', 'union', 'or'):
        gate = valid_terms[0]
        for term in valid_terms[1:]:
            gate = torch.maximum(gate, term)
        return gate.clamp(0.0, 1.0)
    if mode in ('min', 'intersection', 'and'):
        gate = valid_terms[0]
        for term in valid_terms[1:]:
            gate = torch.minimum(gate, term)
        return gate.clamp(0.0, 1.0)
    if mode in ('sum', 'add'):
        return torch.stack(valid_terms, dim=0).sum(dim=0).clamp(0.0, 1.0)
    if mode in ('mul', 'product'):
        gate = valid_terms[0]
        for term in valid_terms[1:]:
            gate = gate * term
        return gate.clamp(0.0, 1.0)
    raise ValueError(f"Unsupported detail point gate combine mode: {mode}")


def _normalize_named_scalar_cfg(spec):
    if spec is None:
        return {}

    if OmegaConf.is_config(spec):
        spec = OmegaConf.to_container(spec, resolve=True)

    normalized = {}
    if isinstance(spec, dict):
        items = spec.items()
    elif isinstance(spec, (list, tuple)):
        items = []
        for item in spec:
            if item is None:
                continue
            if isinstance(item, dict):
                name = item.get('name', item.get('key', None))
                value = item.get('value', item.get('scale', item.get('gain', item.get('weight', None))))
                if name is None or value is None:
                    continue
                items.append((name, value))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                items.append((item[0], item[1]))
    else:
        return {}

    for name, value in items:
        if name is None or value is None:
            continue
        key = str(name).strip()
        if not key:
            continue
        try:
            normalized[key] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _compose_named_region_scalar(
    region_gates,
    named_scales,
    template,
    base=1.0,
    additive=True,
    min_value=None,
    max_value=None,
    named_modulators=None,
):
    scalar = template.new_full((template.shape[0], 1), float(base))
    if not isinstance(region_gates, dict) or not isinstance(named_scales, dict):
        return scalar

    for name, scale in named_scales.items():
        if scale == 0.0:
            continue
        gate = region_gates.get(str(name), None)
        if not torch.is_tensor(gate):
            continue
        gate = gate.to(device=template.device, dtype=template.dtype)
        if gate.dim() == 1:
            gate = gate.unsqueeze(-1)
        modulator = None
        if isinstance(named_modulators, dict):
            modulator = named_modulators.get(str(name), None)
        if torch.is_tensor(modulator):
            modulator = modulator.to(device=template.device, dtype=template.dtype)
            if modulator.dim() == 1:
                modulator = modulator.unsqueeze(-1)
            gate = gate * modulator
        if additive:
            scalar = scalar + gate * float(scale)
        else:
            scalar = scalar - gate * float(scale)

    if min_value is not None:
        scalar = scalar.clamp(min=float(min_value))
    if max_value is not None and float(max_value) > 0.0:
        scalar = scalar.clamp(max=float(max_value))
    return scalar


def _build_named_region_support_modulators(
    region_supports,
    template,
    offset=0.0,
    gain=1.0,
    power=1.0,
    base_mix=0.0,
    max_value=1.0,
):
    modulators = {}
    if not isinstance(region_supports, dict):
        return modulators

    offset = float(offset)
    gain = float(gain)
    power = float(power)
    base_mix = float(base_mix)
    max_value = float(max_value)
    for name, support in region_supports.items():
        if not torch.is_tensor(support):
            continue
        support = support.to(device=template.device, dtype=template.dtype)
        if support.dim() == 1:
            support = support.unsqueeze(-1)
        modulator = torch.sigmoid((support - offset) * gain)
        if power != 1.0:
            modulator = modulator.clamp(min=0.0).pow(power)
        if base_mix != 0.0:
            modulator = base_mix + (1.0 - base_mix) * modulator
        if max_value > 0.0:
            modulator = modulator.clamp(max=max_value)
        modulators[str(name)] = modulator.clamp(0.0, 1.0)
    return modulators


def _lookup_named_tensor(named_tensors, *names):
    if not isinstance(named_tensors, dict):
        return None
    for name in names:
        if name is None:
            continue
        value = named_tensors.get(str(name), None)
        if torch.is_tensor(value):
            return value
    return None


def _iter_named_linear_layers(module):
    if module is None:
        return []

    layers = []
    layer_idx = 0
    while hasattr(module, f"lin{layer_idx}"):
        layer = getattr(module, f"lin{layer_idx}")
        if not isinstance(layer, nn.Linear):
            return []
        layers.append((f"lin{layer_idx}", layer))
        layer_idx += 1
    return layers


def _copy_linear_layer_params(src_layer, dst_layer, allow_output_adapter=False):
    if src_layer.weight.shape == dst_layer.weight.shape and src_layer.bias.shape == dst_layer.bias.shape:
        dst_layer.weight.data.copy_(src_layer.weight.data)
        dst_layer.bias.data.copy_(src_layer.bias.data)
        return True

    if not allow_output_adapter or src_layer.weight.shape[1] != dst_layer.weight.shape[1]:
        return False

    src_out, dst_out = src_layer.weight.shape[0], dst_layer.weight.shape[0]
    if src_out == 1 and dst_out > 1:
        dst_layer.weight.data.copy_(src_layer.weight.data.repeat(dst_out, 1))
        dst_layer.bias.data.copy_(src_layer.bias.data.repeat(dst_out))
        return True
    if src_out > 1 and dst_out == 1:
        dst_layer.weight.data.copy_(src_layer.weight.data.mean(dim=0, keepdim=True))
        dst_layer.bias.data.copy_(src_layer.bias.data.mean(dim=0, keepdim=True))
        return True
    if src_out > dst_out:
        dst_layer.weight.data.copy_(src_layer.weight.data[:dst_out])
        dst_layer.bias.data.copy_(src_layer.bias.data[:dst_out])
        return True
    if src_out < dst_out:
        repeat_count = int(np.ceil(float(dst_out) / float(src_out)))
        dst_layer.weight.data.copy_(src_layer.weight.data.repeat(repeat_count, 1)[:dst_out])
        dst_layer.bias.data.copy_(src_layer.bias.data.repeat(repeat_count)[:dst_out])
        return True
    return False


def _copy_mlp_parameters(src_module, dst_module):
    src_layers = _iter_named_linear_layers(src_module)
    dst_layers = _iter_named_linear_layers(dst_module)
    if len(src_layers) <= 0 or len(src_layers) != len(dst_layers):
        return False, "layer_mismatch"

    with torch.no_grad():
        for layer_idx, ((src_name, src_layer), (dst_name, dst_layer)) in enumerate(zip(src_layers, dst_layers)):
            copied = _copy_linear_layer_params(
                src_layer,
                dst_layer,
                allow_output_adapter=layer_idx == len(src_layers) - 1,
            )
            if not copied:
                return False, f"shape_mismatch:{src_name}->{dst_name}"
    return True, "ok"


def _copy_mlp_parameters_with_input_subset(src_module, dst_module, input_columns, output_scale=1.0):
    src_layers = _iter_named_linear_layers(src_module)
    dst_layers = _iter_named_linear_layers(dst_module)
    if len(src_layers) <= 0 or len(src_layers) != len(dst_layers):
        return False, "layer_mismatch"

    input_columns = [int(idx) for idx in input_columns]
    if len(input_columns) <= 0:
        return False, "empty_input_subset"

    with torch.no_grad():
        for layer_idx, ((src_name, src_layer), (dst_name, dst_layer)) in enumerate(zip(src_layers, dst_layers)):
            if layer_idx == 0:
                if (
                    src_layer.weight.shape[0] != dst_layer.weight.shape[0]
                    or len(input_columns) > dst_layer.weight.shape[1]
                    or src_layer.bias.shape != dst_layer.bias.shape
                ):
                    return False, f"shape_mismatch:{src_name}->{dst_name}"
                dst_layer.weight.data.zero_()
                dst_layer.weight.data[:, :len(input_columns)].copy_(src_layer.weight.data[:, input_columns])
                dst_layer.bias.data.copy_(src_layer.bias.data)
                continue

            copied = _copy_linear_layer_params(
                src_layer,
                dst_layer,
                allow_output_adapter=False,
            )
            if not copied:
                return False, f"shape_mismatch:{src_name}->{dst_name}"

        if abs(float(output_scale) - 1.0) > 1.0e-8:
            dst_last = dst_layers[-1][1]
            dst_last.weight.data.mul_(float(output_scale))
            dst_last.bias.data.mul_(float(output_scale))
    return True, "ok"


def _clone_cfg_node(cfg_node):
    if cfg_node is None:
        return OmegaConf.create({})
    if OmegaConf.is_config(cfg_node):
        return OmegaConf.create(OmegaConf.to_container(cfg_node, resolve=True))
    return OmegaConf.create(cfg_node)


def _encoded_feature_dim(input_dims, multires, include_input=True):
    multires = int(multires)
    input_dims = int(input_dims)
    if multires <= 0:
        return input_dims if include_input else 0
    base_dim = input_dims if include_input else 0
    return base_dim + 2 * input_dims * multires


def _flatten_feature_tensor(values):
    if not torch.is_tensor(values):
        return None
    if values.ndim == 0:
        return values.reshape(1, 1)
    if values.ndim == 1:
        return values.unsqueeze(-1)
    if values.shape[0] == 0:
        trailing_dim = 1
        for dim in values.shape[1:]:
            trailing_dim *= int(dim)
        return values.new_zeros((0, trailing_dim))
    return values.reshape(values.shape[0], -1)


def _fourier_encode(values, multires, include_input=True, frequency_scale=1.0):
    if not torch.is_tensor(values):
        return None

    values = _flatten_feature_tensor(values)

    multires = int(multires)
    if multires <= 0:
        if include_input:
            return values
        return values.new_zeros((values.shape[0], 0))

    if values.shape[0] == 0:
        encoded_dim = _encoded_feature_dim(
            values.shape[1],
            multires,
            include_input=include_input,
        )
        return values.new_zeros((0, encoded_dim))

    outputs = []
    if include_input:
        outputs.append(values)

    freq_bands = (2.0 ** torch.arange(multires, device=values.device, dtype=values.dtype)) * float(frequency_scale)
    expanded = values.unsqueeze(-1) * freq_bands.view(1, 1, -1)
    expanded = expanded.reshape(values.shape[0], -1)
    outputs.append(torch.sin(expanded))
    outputs.append(torch.cos(expanded))
    return torch.cat(outputs, dim=-1)


def _default_face_local_region_cfg():
    return {
        'enable': True,
        'semantic_name_weights': [
            ['face', 1.00],
            ['skin', 1.00],
            ['upper', 0.35],
        ],
        'semantic_id_weights': [
            [1, 1.00],
            [2, 1.00],
            [3, 0.35],
        ],
        'joint_id_weights': [
            [15, 1.00],
            [12, 1.00],
            [13, 1.00],
            [14, 1.00],
            [16, 0.40],
            [17, 0.40],
            [18, 0.10],
            [19, 0.10],
        ],
        'combine_mode': 'max',
        'fallback_to_full': True,
        'fallback_to_global': True,
        'min_gate_mean': 0.002,
    }


def _normalize_local_carrier_cfg_list(spec):
    if spec is None:
        return []

    if OmegaConf.is_config(spec):
        spec = OmegaConf.to_container(spec, resolve=True)
    if isinstance(spec, dict):
        spec = [spec]
    if not isinstance(spec, list):
        return []

    normalized = []
    for idx, item in enumerate(spec):
        if not isinstance(item, dict):
            continue
        item_dict = dict(item)
        name = str(item_dict.get('name', f'region_{idx}')).strip() or f'region_{idx}'
        safe_key = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in name.lower())
        safe_key = safe_key.strip('_') or f'region_{idx}'
        item_dict['name'] = name
        item_dict['_key'] = safe_key
        normalized.append(OmegaConf.create(item_dict))
    return normalized


def _normalize_local_output_head_override_cfgs(spec):
    if spec is None:
        return {}

    if OmegaConf.is_config(spec):
        spec = OmegaConf.to_container(spec, resolve=True)
    if isinstance(spec, dict):
        spec = [spec]
    if not isinstance(spec, list):
        return {}

    normalized = {}
    for idx, item in enumerate(spec):
        if not isinstance(item, dict):
            continue
        item_dict = dict(item)
        name = str(item_dict.get('name', f'region_{idx}')).strip() or f'region_{idx}'
        safe_key = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in name.lower())
        safe_key = safe_key.strip('_') or f'region_{idx}'
        item_dict['name'] = name
        item_dict['_key'] = safe_key
        item_dict['raw_scale'] = float(item_dict.get('raw_scale', 1.0))
        item_dict['gated_scale'] = float(item_dict.get('gated_scale', 1.0))
        item_dict['gate_feature_scale'] = float(item_dict.get('gate_feature_scale', 1.0))
        item_dict['gate_bias'] = float(item_dict.get('gate_bias', 0.0))
        item_dict['gate_boost'] = float(item_dict.get('gate_boost', 0.0))
        item_dict['color_scale'] = float(item_dict.get('color_scale', 1.0))
        if item_dict.get('soft_gate_min', None) is not None:
            item_dict['soft_gate_min'] = float(item_dict['soft_gate_min'])
        normalized[safe_key] = OmegaConf.create(item_dict)
    return normalized


_STRUCTURE_CARRIER_ATTR_DIMS = {
    'binding_weights': 3,
    'binding_weights_raw': 3,
    'binding_region_probs': 3,
    'binding_region_probs_raw': 3,
    'binding_compact_semantic_probs': 6,
    'binding_compact_semantic_probs_raw': 6,
    'binding_surface_distance': 1,
    'binding_distance': 1,
    'binding_boundary_score': 1,
    'binding_boundary_live_score': 1,
    'binding_boundary_mixed_score': 1,
    'binding_semantic_score': 1,
    'binding_semantic_distance': 1,
    'binding_thin_score': 1,
    'binding_part_rigid_prior': 1,
    'binding_part_free_prior': 1,
    'binding_non_rigid_delta': 1,
    'binding_non_rigid_rigid_carry': 1,
    'binding_non_rigid_soft_carry': 1,
    'binding_non_rigid_geometry_shift': 1,
    'binding_non_rigid_layer_sharpen_gate': 1,
    'binding_non_rigid_layer_sharpen_power': 1,
}


def _normalize_structure_carrier_cfg_list(spec, default_detach=True):
    if spec is None:
        return []

    if OmegaConf.is_config(spec):
        spec = OmegaConf.to_container(spec, resolve=True)
    if isinstance(spec, dict):
        spec = spec.get('features', [spec])
    if isinstance(spec, str):
        spec = [{'attr': spec}]
    if not isinstance(spec, list):
        return []

    normalized = []
    for idx, item in enumerate(spec):
        if isinstance(item, str):
            item_dict = {'attr': item}
        elif isinstance(item, dict):
            item_dict = dict(item)
        else:
            continue

        attr_name = str(item_dict.get('attr', item_dict.get('name', ''))).strip()
        if not attr_name:
            continue
        feature_name = str(item_dict.get('name', attr_name)).strip() or attr_name
        dims = int(item_dict.get('dims', _STRUCTURE_CARRIER_ATTR_DIMS.get(attr_name, 0)))
        if dims <= 0:
            continue
        item_dict['attr'] = attr_name
        item_dict['name'] = feature_name
        item_dict['dims'] = dims
        item_dict['detach'] = bool(item_dict.get('detach', default_detach))
        item_dict['multires'] = int(item_dict.get('multires', 0))
        item_dict['include_input'] = bool(item_dict.get('include_input', True))
        item_dict['frequency_scale'] = float(item_dict.get('frequency_scale', 1.0))
        item_dict['scale'] = float(item_dict.get('scale', 1.0))
        item_dict['center'] = float(item_dict.get('center', 0.0))
        item_dict['weight'] = float(item_dict.get('weight', 1.0))
        if item_dict.get('clamp', None) is not None:
            item_dict['clamp'] = float(item_dict['clamp'])
        item_dict['abs'] = bool(item_dict.get('abs', False))
        normalized.append(OmegaConf.create(item_dict))
    return normalized

class ColorPrecompute(nn.Module):
    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg
        self.metadata = metadata

    def forward(self, gaussians, camera, iteration=0):
        raise NotImplementedError

class SH2RGB(ColorPrecompute):
    def __init__(self, cfg, metadata):
        super().__init__(cfg, metadata)
        
    def forward(self, gaussians, camera, iteration=0):
        shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
        dir_pp = (gaussians.get_xyz - camera.camera_center.repeat(gaussians.get_features.shape[0], 1))
        if self.cfg.cano_view_dir:
            T_fwd = gaussians.fwd_transform
            R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
            dir_pp = torch.matmul(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
            view_noise_scale = _resolve_scheduled_scalar(iteration, self.cfg.get('view_noise', 0.))
            if self.training and view_noise_scale > 0.:
                view_noise = torch.tensor(augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
                                          dtype=torch.float32,
                                          device=dir_pp.device).transpose(0, 1)
                dir_pp = torch.matmul(dir_pp, view_noise)

        dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-12)
        sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        return colors_precomp
        
class ColorMLP(ColorPrecompute):
    def __init__(self, cfg, metadata):
        super().__init__(cfg, metadata)
        d_in = cfg.feature_dim

        self.use_xyz = cfg.get('use_xyz', False)
        self.use_cov = cfg.get('use_cov', False)
        self.use_normal = cfg.get('use_normal', False)
        self.sh_degree = cfg.get('sh_degree', 0)
        self.cano_view_dir = cfg.get('cano_view_dir', False)
        self.non_rigid_dim = cfg.get('non_rigid_dim', 0)
        self.latent_dim = cfg.get('latent_dim', 0)

        if self.use_xyz:
            d_in += 3
        if self.use_cov:
            d_in += 6 # only upper triangle suffice
        if self.use_normal:
            d_in += 3 # quasi-normal by smallest eigenvector...
        if self.sh_degree > 0:
            d_in += (self.sh_degree + 1) ** 2 - 1
            self.sh_embed = lambda dir: eval_sh_bases(self.sh_degree, dir)[..., 1:]
        if self.non_rigid_dim > 0:
            d_in += self.non_rigid_dim
        if self.latent_dim > 0:
            d_in += self.latent_dim
            self.frame_dict = metadata['frame_dict']
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)

        d_out = 3
        self.mlp = VanillaCondMLP(d_in, 0, d_out, cfg.mlp)
        self.structured_trunk_cfg = cfg.get('structured_trunk', None)
        self.structured_trunk_enable = bool(
            self.structured_trunk_cfg.get('enable', False)
        ) if self.structured_trunk_cfg is not None else False
        self.structured_trunk_context_proj = None
        self.structured_trunk_carrier_proj = None
        self.structured_trunk_shared_proj = None
        self.structured_trunk_structure_cfg = None
        self.structured_trunk_structure_enable = False
        self.structured_trunk_structure_detach = True
        self.structured_trunk_structure_inject_scale = 0.0
        self.structured_trunk_structure_features = []
        self.structured_trunk_structure_proj = None
        self.structured_trunk_local_cfgs = []
        self.structured_trunk_local_projs = nn.ModuleDict()
        self.structured_trunk_use_input_context = True
        self.structured_trunk_input_context_dim = 0
        self.structured_trunk_use_canonical_xyz = True
        self.structured_trunk_use_view_dir = True
        self.structured_trunk_xyz_multires = 0
        self.structured_trunk_xyz_include_input = True
        self.structured_trunk_xyz_scale = 1.0
        self.structured_trunk_view_multires = 0
        self.structured_trunk_view_include_input = True
        self.structured_trunk_view_scale = 1.0
        self.structured_trunk_use_modulated_carrier = True
        self.structured_trunk_inject_scale = 0.0
        self.structured_trunk_local_gate_to_region = True
        self.structured_trunk_shared_mlp = None
        self.structured_trunk_shared_mlp_scale = 1.0
        self.structured_trunk_structure_mlp = None
        self.structured_trunk_structure_mlp_scale = 1.0
        self.structured_trunk_local_mlp_cfg = None
        self.structured_trunk_local_mlp_scale = 1.0
        self.structured_trunk_local_mlps = nn.ModuleDict()
        self.last_structured_trunk_shared_abs_mean = None
        self.last_structured_trunk_shared_residual_abs_mean = None
        self.last_structured_trunk_carrier_abs_mean = None
        self.last_structured_trunk_structure_abs_mean = None
        self.last_structured_trunk_structure_raw_abs_mean = None
        self.last_structured_trunk_structure_residual_abs_mean = None
        self.last_structured_trunk_local_abs_mean = None
        self.last_structured_trunk_local_raw_abs_mean = None
        self.last_structured_trunk_local_gate_mean = None
        self.last_structured_trunk_local_residual_abs_mean = None
        self.last_structured_trunk_total_abs_mean = None
        self.last_structured_trunk_debug = ''
        self.structured_trunk_output_head_cfg = None
        self.structured_trunk_output_head_enable = False
        self.structured_trunk_output_head_mode = 'rgb'
        self.structured_trunk_output_head_compose_mode = 'residual'
        self.structured_trunk_output_head_disable_input_residual = False
        self.structured_trunk_output_head_component_dim = d_in
        self.structured_trunk_output_head_use_base_input = True
        self.structured_trunk_output_head_use_shared = True
        self.structured_trunk_output_head_use_structure = True
        self.structured_trunk_output_head_use_local = True
        self.structured_trunk_output_head_use_local_raw_feature = False
        self.structured_trunk_output_head_use_local_gated_feature = True
        self.structured_trunk_output_head_use_local_gate_feature = True
        self.structured_trunk_output_head_local_soft_gate_min = 0.0
        self.structured_trunk_output_head_scale_cfg = 1.0
        self.structured_trunk_output_head_max_residual = 0.0
        self.structured_trunk_output_head_gate_bias = 0.0
        self.structured_trunk_output_head_gate_gain = 1.0
        self.structured_trunk_output_head_min_gate = 0.0
        self.structured_trunk_output_head_max_gate = 1.0
        self.structured_trunk_output_head_chroma_center = False
        self.structured_trunk_output_head_band_luma_scale = 1.0
        self.structured_trunk_output_head_band_chroma_scale = 1.0
        self.structured_trunk_output_head_local_region_overrides = {}
        self.structured_trunk_output_head_local_color_cfg = None
        self.structured_trunk_output_head_local_color_enable = False
        self.structured_trunk_output_head_local_color_use_base_feature = False
        self.structured_trunk_output_head_local_color_use_shared_feature = False
        self.structured_trunk_output_head_local_color_use_structure_feature = False
        self.structured_trunk_output_head_local_color_use_raw_feature = True
        self.structured_trunk_output_head_local_color_use_gated_feature = True
        self.structured_trunk_output_head_local_color_use_gate_feature = True
        self.structured_trunk_output_head_local_color_gate_with_region = True
        self.structured_trunk_output_head_local_color_init_from_output_head = True
        self.structured_trunk_output_head_local_color_init_scale = 0.35
        self.structured_trunk_output_head_local_color_scale_cfg = 1.0
        self.structured_trunk_output_head_local_color_owner_cfg = None
        self.structured_trunk_output_head_local_color_owner_enable = False
        self.structured_trunk_output_head_local_color_owner_scale_cfg = 1.0
        self.structured_trunk_output_head_local_color_owner_use_support = True
        self.structured_trunk_output_head_local_color_owner_use_region_gate = False
        self.structured_trunk_output_head_local_color_owner_gate_base = 0.0
        self.structured_trunk_output_head_local_color_owner_head_cfg = None
        self.structured_trunk_output_head_local_color_owner_head_enable = False
        self.structured_trunk_output_head_local_color_owner_head_mode = 'rgb'
        self.structured_trunk_output_head_local_color_owner_head_compose_mode = 'gated_color'
        self.structured_trunk_output_head_local_color_owner_head_scale_cfg = 1.0
        self.structured_trunk_output_head_local_color_owner_head_gate_bias = 0.0
        self.structured_trunk_output_head_local_color_owner_head_gate_gain = 1.0
        self.structured_trunk_output_head_local_color_owner_head_min_gate = 0.0
        self.structured_trunk_output_head_local_color_owner_head_max_gate = 1.0
        self.structured_trunk_output_head_local_color_owner_head_chroma_center = False
        self.structured_trunk_output_head_local_color_owner_head_band_luma_scale = 1.0
        self.structured_trunk_output_head_local_color_owner_head_band_chroma_scale = 1.0
        self.structured_trunk_output_head_local_color_owner_head_use_local_color_input = True
        self.structured_trunk_output_head_local_color_owner_head_use_local_color_output = True
        self.structured_trunk_output_head_local_color_owner_head_local_color_output_scale_cfg = 1.0
        self.structured_trunk_output_head_local_color_owner_head_use_local_geometry_raw = True
        self.structured_trunk_output_head_local_color_owner_head_use_support_feature = True
        self.structured_trunk_output_head_local_color_owner_head_use_region_gate_feature = False
        self.structured_trunk_output_head_local_color_owner_head_init_from_local_color = True
        self.structured_trunk_output_head_local_color_owner_head_init_scale = 0.5
        self.structured_trunk_output_head_local_color_owner_head_takeover_cfg = None
        self.structured_trunk_output_head_local_color_owner_head_takeover_enable = False
        self.structured_trunk_output_head_local_color_owner_head_takeover_scale_cfg = 1.0
        self.structured_trunk_output_head_local_color_owner_head_takeover_strength_cfg = 1.0
        self.structured_trunk_output_head_local_color_owner_head_takeover_base_mix = 0.0
        self.structured_trunk_output_head_local_color_owner_head_takeover_max = 1.0
        self.structured_trunk_output_head_local_color_owner_head_takeover_use_support = True
        self.structured_trunk_output_head_local_color_owner_head_takeover_support_detach = True
        self.structured_trunk_output_head_local_color_owner_head_takeover_support_offset = 0.0
        self.structured_trunk_output_head_local_color_owner_head_takeover_support_gain = 1.0
        self.structured_trunk_output_head_local_color_owner_head_takeover_support_power = 1.0
        self.structured_trunk_output_head_local_color_owner_head_takeover_region_strength_cfg = {}
        self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_cfg = None
        self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_enable = False
        self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_power = 1.0
        self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_min_scale = 0.0
        self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_apply_to_coarse = True
        self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_apply_to_hf = True
        self.structured_trunk_output_head_local_color_owner_head_boundary_cfg = None
        self.structured_trunk_output_head_local_color_owner_head_boundary_enable = False
        self.structured_trunk_output_head_local_color_owner_head_boundary_mode = 'rgb'
        self.structured_trunk_output_head_local_color_owner_head_boundary_compose_mode = 'gated_color'
        self.structured_trunk_output_head_local_color_owner_head_boundary_scale_cfg = 1.0
        self.structured_trunk_output_head_local_color_owner_head_boundary_gate_bias = 0.0
        self.structured_trunk_output_head_local_color_owner_head_boundary_gate_gain = 1.0
        self.structured_trunk_output_head_local_color_owner_head_boundary_min_gate = 0.0
        self.structured_trunk_output_head_local_color_owner_head_boundary_max_gate = 1.0
        self.structured_trunk_output_head_local_color_owner_head_boundary_contrib_gate_mode = 'multiply'
        self.structured_trunk_output_head_local_color_owner_head_boundary_contrib_min_gate = 0.0
        self.structured_trunk_output_head_local_color_owner_head_boundary_contrib_scale_cfg = 1.0
        self.structured_trunk_output_head_local_color_owner_head_boundary_chroma_center = False
        self.structured_trunk_output_head_local_color_owner_head_boundary_band_luma_scale = 1.0
        self.structured_trunk_output_head_local_color_owner_head_boundary_band_chroma_scale = 1.0
        self.structured_trunk_output_head_local_color_owner_head_boundary_use_local_color_input = True
        self.structured_trunk_output_head_local_color_owner_head_boundary_use_local_color_output = False
        self.structured_trunk_output_head_local_color_owner_head_boundary_local_color_output_scale_cfg = 1.0
        self.structured_trunk_output_head_local_color_owner_head_boundary_use_local_geometry_raw = True
        self.structured_trunk_output_head_local_color_owner_head_boundary_use_support_feature = False
        self.structured_trunk_output_head_local_color_owner_head_boundary_use_region_gate_feature = True
        self.structured_trunk_output_head_local_color_owner_head_boundary_use_boundary_feature = True
        self.structured_trunk_output_head_local_color_owner_head_boundary_focus_source = 'score'
        self.structured_trunk_output_head_local_color_owner_head_boundary_focus_threshold = 0.0
        self.structured_trunk_output_head_local_color_owner_head_boundary_focus_power = 1.0
        self.structured_trunk_output_head_local_color_owner_head_boundary_focus_min = 0.0
        self.structured_trunk_output_head_local_color_owner_head_boundary_focus_max = 1.0
        self.structured_trunk_output_head_local_color_owner_head_boundary_focus_use_region_gate = True
        self.structured_trunk_output_head_local_color_owner_head_boundary_focus_use_support = False
        self.structured_trunk_output_head_local_color_owner_head_boundary_focus_support_detach = True
        self.structured_trunk_output_head_local_color_owner_head_boundary_init_from_local_color = True
        self.structured_trunk_output_head_local_color_owner_head_boundary_init_scale = 0.35
        self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_cfg = None
        self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_enable = False
        self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_scale_cfg = 1.0
        self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_max = 1.0
        self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_use_region_gate = True
        self.structured_trunk_output_head_local_color_use_current_xyz_feature = False
        self.structured_trunk_output_head_local_color_current_xyz_multires = 6
        self.structured_trunk_output_head_local_color_current_xyz_include_input = True
        self.structured_trunk_output_head_local_color_current_xyz_scale = 1.0
        self.structured_trunk_output_head_local_color_use_current_radius_feature = False
        self.structured_trunk_output_head_local_color_current_radius_multires = 4
        self.structured_trunk_output_head_local_color_current_radius_include_input = True
        self.structured_trunk_output_head_local_color_current_radius_scale = 1.0
        self.structured_trunk_output_head_local_input_slices = {}
        self.structured_trunk_output_head_local_color_input_dims = {}
        self.structured_trunk_output_head_local_color_owner_head_input_dims = {}
        self.structured_trunk_output_head_local_color_owner_head_boundary_input_dims = {}
        self.structured_trunk_output_head_base_proj = None
        self.structured_trunk_output_head_shared_fusion_proj = None
        self.structured_trunk_output_head_structure_fusion_proj = None
        self.structured_trunk_output_head_local_fusion_projs = nn.ModuleDict()
        self.structured_trunk_output_head_local_geometry_fusion_projs = nn.ModuleDict()
        self.structured_trunk_output_head_local_color_mlps = nn.ModuleDict()
        self.structured_trunk_output_head_local_color_owner_head_mlps = nn.ModuleDict()
        self.structured_trunk_output_head_local_color_owner_head_gate_mlps = nn.ModuleDict()
        self.structured_trunk_output_head_local_color_owner_head_boundary_mlps = nn.ModuleDict()
        self.structured_trunk_output_head_local_color_owner_head_boundary_gate_mlps = nn.ModuleDict()
        self.structured_trunk_output_head_mlp = None
        self.structured_trunk_output_head_gate_mlp = None
        self.structured_trunk_output_head_dual_head_cfg = None
        self.structured_trunk_output_head_dual_head_enable = False
        self.structured_trunk_output_head_base_scaffold_scale_cfg = 1.0
        self.structured_trunk_output_head_coarse_scale_cfg = 1.0
        self.structured_trunk_output_head_coarse_region_suppress_cfg = {}
        self.structured_trunk_output_head_coarse_region_min_scale = 0.0
        self.structured_trunk_output_head_region_support_cfg = None
        self.structured_trunk_output_head_region_support_enable = False
        self.structured_trunk_output_head_region_support_source = 'hybrid'
        self.structured_trunk_output_head_region_support_detach = True
        self.structured_trunk_output_head_region_support_offset = 0.0
        self.structured_trunk_output_head_region_support_gain = 1.0
        self.structured_trunk_output_head_region_support_power = 1.0
        self.structured_trunk_output_head_region_support_base_mix = 0.0
        self.structured_trunk_output_head_region_support_max = 1.0
        self.structured_trunk_output_head_hf_head_cfg = None
        self.structured_trunk_output_head_hf_head_enable = False
        self.structured_trunk_output_head_hf_head_mode = 'rgb'
        self.structured_trunk_output_head_hf_head_compose_mode = 'gated_color'
        self.structured_trunk_output_head_hf_head_scale_cfg = 1.0
        self.structured_trunk_output_head_hf_head_max_residual = 0.0
        self.structured_trunk_output_head_hf_head_gate_bias = 0.0
        self.structured_trunk_output_head_hf_head_gate_gain = 1.0
        self.structured_trunk_output_head_hf_head_min_gate = 0.0
        self.structured_trunk_output_head_hf_head_max_gate = 1.0
        self.structured_trunk_output_head_hf_head_chroma_center = False
        self.structured_trunk_output_head_hf_head_band_luma_scale = 1.0
        self.structured_trunk_output_head_hf_head_band_chroma_scale = 1.0
        self.structured_trunk_output_head_hf_head_use_local_color = True
        self.structured_trunk_output_head_hf_head_local_color_scale_cfg = 1.0
        self.structured_trunk_output_head_hf_head_reuse_output_gate = True
        self.structured_trunk_output_head_hf_head_init_from_output_head = True
        self.structured_trunk_output_head_hf_head_init_scale = 0.15
        self.structured_trunk_output_head_hf_head_region_boost_cfg = {}
        self.structured_trunk_output_head_hf_head_region_boost_max = 0.0
        self.structured_trunk_output_head_hf_head_use_output_fusion = True
        self.structured_trunk_output_head_hf_head_use_shared_raw = False
        self.structured_trunk_output_head_hf_head_use_structure_raw = False
        self.structured_trunk_output_head_hf_head_use_local_geometry_raw = False
        self.structured_trunk_output_head_hf_head_shared_input_dim = 0
        self.structured_trunk_output_head_hf_head_structure_raw_dim = 0
        self.structured_trunk_output_head_hf_head_local_geometry_raw_dim = 0
        self.structured_trunk_output_head_hf_head_local_geometry_keys = []
        self.structured_trunk_output_head_fusion_dim = 0
        self.structured_trunk_output_head_hf_input_dim = 0
        self.structured_trunk_output_head_hf_head_mlp = None
        self.structured_trunk_output_head_hf_head_gate_mlp = None
        self.last_structured_trunk_head_abs_mean = None
        self.last_structured_trunk_head_color_abs_mean = None
        self.last_structured_trunk_head_gate_mean = None
        self.last_structured_trunk_head_gate_boost_mean = None
        self.last_structured_trunk_head_local_color_abs_mean = None
        self.last_structured_trunk_head_fusion_abs_mean = None
        self.last_structured_trunk_head_debug = ''
        self.last_structured_trunk_owner_abs_mean = None
        self.last_structured_trunk_owner_input_abs_mean = None
        self.last_structured_trunk_owner_color_abs_mean = None
        self.last_structured_trunk_owner_support_mean = None
        self.last_structured_trunk_owner_gate_mean = None
        self.last_structured_trunk_owner_takeover_mean = None
        self.last_structured_trunk_owner_takeover_legacy_scale_mean = None
        self.last_structured_trunk_owner_boundary_abs_mean = None
        self.last_structured_trunk_owner_boundary_input_abs_mean = None
        self.last_structured_trunk_owner_boundary_color_abs_mean = None
        self.last_structured_trunk_owner_boundary_focus_mean = None
        self.last_structured_trunk_owner_boundary_gate_mean = None
        self.last_structured_trunk_owner_boundary_takeover_mean = None
        self.last_structured_trunk_scaffold_abs_mean = None
        self.last_structured_trunk_coarse_abs_mean = None
        self.last_structured_trunk_hf_abs_mean = None
        self.last_structured_trunk_hf_color_abs_mean = None
        self.last_structured_trunk_hf_gate_mean = None
        self.last_structured_trunk_hf_local_color_abs_mean = None
        self.last_structured_trunk_hf_fusion_abs_mean = None
        self.last_structured_trunk_hf_region_gain_mean = None
        self.last_structured_trunk_coarse_region_scale_mean = None
        self.last_structured_trunk_region_support_mean = None
        self.last_structured_trunk_hf_debug = ''
        self.last_structured_trunk_owner_takeover_debug = ''
        self.last_structured_trunk_owner_boundary_debug = ''
        self.structured_trunk_output_head_global_input_slices = {}
        if self.structured_trunk_enable:
            self.structured_trunk_use_input_context = bool(
                self.structured_trunk_cfg.get('use_input_context', True)
            )
            self.structured_trunk_input_context_dim = int(
                self.structured_trunk_cfg.get(
                    'input_context_dim',
                    self.structured_trunk_cfg.get('context_dim', 32),
                )
            )
            self.structured_trunk_use_canonical_xyz = bool(
                self.structured_trunk_cfg.get('use_canonical_xyz', True)
            )
            self.structured_trunk_use_view_dir = bool(
                self.structured_trunk_cfg.get('use_view_dir', True)
            )
            self.structured_trunk_xyz_multires = int(
                self.structured_trunk_cfg.get('xyz_multires', 6)
            )
            self.structured_trunk_xyz_include_input = bool(
                self.structured_trunk_cfg.get('xyz_include_input', True)
            )
            self.structured_trunk_xyz_scale = float(
                self.structured_trunk_cfg.get('xyz_frequency_scale', 1.0)
            )
            self.structured_trunk_view_multires = int(
                self.structured_trunk_cfg.get('view_multires', 4)
            )
            self.structured_trunk_view_include_input = bool(
                self.structured_trunk_cfg.get('view_include_input', True)
            )
            self.structured_trunk_view_scale = float(
                self.structured_trunk_cfg.get('view_frequency_scale', 1.0)
            )
            self.structured_trunk_use_modulated_carrier = bool(
                self.structured_trunk_cfg.get('use_modulated_carrier', True)
            )
            self.structured_trunk_inject_scale = float(
                self.structured_trunk_cfg.get('inject_scale', 0.35)
            )
            self.structured_trunk_local_gate_to_region = bool(
                self.structured_trunk_cfg.get('local_gate_to_region', True)
            )

            context_dim = 0
            if self.structured_trunk_use_input_context:
                if self.structured_trunk_input_context_dim > 0:
                    self.structured_trunk_context_proj = nn.Linear(
                        d_in,
                        self.structured_trunk_input_context_dim,
                    )
                    nn.init.normal_(
                        self.structured_trunk_context_proj.weight,
                        mean=0.0,
                        std=0.02,
                    )
                    nn.init.constant_(self.structured_trunk_context_proj.bias, 0.0)
                    context_dim = self.structured_trunk_input_context_dim
                else:
                    context_dim = d_in

            raw_carrier_dim = 0
            if self.structured_trunk_use_canonical_xyz:
                raw_carrier_dim += _encoded_feature_dim(
                    3,
                    self.structured_trunk_xyz_multires,
                    include_input=self.structured_trunk_xyz_include_input,
                )
            if self.structured_trunk_use_view_dir:
                raw_carrier_dim += _encoded_feature_dim(
                    3,
                    self.structured_trunk_view_multires,
                    include_input=self.structured_trunk_view_include_input,
                )

            shared_input_dim = 0
            if context_dim > 0:
                shared_input_dim += context_dim
            if raw_carrier_dim > 0:
                shared_input_dim += raw_carrier_dim
            if (
                self.structured_trunk_use_modulated_carrier
                and raw_carrier_dim > 0
                and context_dim > 0
            ):
                self.structured_trunk_carrier_proj = nn.Linear(
                    raw_carrier_dim,
                    context_dim,
                )
                nn.init.normal_(
                    self.structured_trunk_carrier_proj.weight,
                    mean=0.0,
                    std=0.02,
                )
                nn.init.constant_(self.structured_trunk_carrier_proj.bias, 0.0)
                shared_input_dim += context_dim

            if shared_input_dim > 0:
                self.structured_trunk_shared_proj = nn.Linear(shared_input_dim, d_in)
                nn.init.constant_(self.structured_trunk_shared_proj.weight, 0.0)
                nn.init.constant_(self.structured_trunk_shared_proj.bias, 0.0)
                self.structured_trunk_shared_mlp, self.structured_trunk_shared_mlp_scale = (
                    self._build_structured_trunk_residual_mlp(
                        self.structured_trunk_cfg.get('shared_mlp', None),
                        shared_input_dim,
                        d_in,
                    )
                )

            self.structured_trunk_structure_cfg = self.structured_trunk_cfg.get(
                'structure_carrier',
                None,
            )
            self.structured_trunk_structure_enable = bool(
                self.structured_trunk_structure_cfg.get('enable', False)
            ) if self.structured_trunk_structure_cfg is not None else False
            if self.structured_trunk_structure_enable:
                self.structured_trunk_structure_detach = bool(
                    self.structured_trunk_structure_cfg.get('detach', True)
                )
                self.structured_trunk_structure_inject_scale = float(
                    self.structured_trunk_structure_cfg.get('inject_scale', 0.35)
                )
                self.structured_trunk_structure_features = _normalize_structure_carrier_cfg_list(
                    self.structured_trunk_structure_cfg.get('features', None),
                    default_detach=self.structured_trunk_structure_detach,
                )
                structure_raw_dim = self._encoded_structure_carrier_dim(
                    self.structured_trunk_structure_features
                )
                if structure_raw_dim > 0:
                    self.structured_trunk_structure_proj = nn.Linear(
                        structure_raw_dim,
                        d_in,
                    )
                    nn.init.constant_(self.structured_trunk_structure_proj.weight, 0.0)
                    nn.init.constant_(self.structured_trunk_structure_proj.bias, 0.0)
                    self.structured_trunk_structure_mlp, self.structured_trunk_structure_mlp_scale = (
                        self._build_structured_trunk_residual_mlp(
                            self.structured_trunk_cfg.get('structure_mlp', None),
                            structure_raw_dim,
                            d_in,
                        )
                    )
                else:
                    self.structured_trunk_structure_enable = False

            self.structured_trunk_local_mlp_cfg = self.structured_trunk_cfg.get(
                'local_mlp',
                None,
            )
            if self.structured_trunk_local_mlp_cfg is not None:
                self.structured_trunk_local_mlp_scale = float(
                    self.structured_trunk_local_mlp_cfg.get('scale', 1.0)
                )
            local_cfgs = _normalize_local_carrier_cfg_list(
                self.structured_trunk_cfg.get('local_carriers', None)
            )
            for local_cfg in local_cfgs:
                local_raw_dim = self._encoded_local_carrier_dim(local_cfg)
                if local_raw_dim <= 0:
                    continue
                local_key = str(local_cfg.get('_key', ''))
                if not local_key:
                    continue
                local_proj = nn.Linear(local_raw_dim, d_in)
                nn.init.constant_(local_proj.weight, 0.0)
                nn.init.constant_(local_proj.bias, 0.0)
                self.structured_trunk_local_projs[local_key] = local_proj
                local_mlp, _ = self._build_structured_trunk_residual_mlp(
                    self.structured_trunk_local_mlp_cfg,
                    local_raw_dim,
                    d_in,
                )
                if local_mlp is not None:
                    self.structured_trunk_local_mlps[local_key] = local_mlp
                self.structured_trunk_local_cfgs.append(local_cfg)
            self.structured_trunk_output_head_cfg = self.structured_trunk_cfg.get(
                'output_head',
                self.structured_trunk_cfg.get('trunk_rgb_head', None),
            )
            self.structured_trunk_output_head_enable = bool(
                self.structured_trunk_output_head_cfg.get('enable', False)
            ) if self.structured_trunk_output_head_cfg is not None else False
            if self.structured_trunk_output_head_enable:
                self.structured_trunk_output_head_mode = str(
                    self.structured_trunk_output_head_cfg.get('mode', 'rgb')
                ).lower()
                if self.structured_trunk_output_head_mode not in ('rgb', 'band'):
                    raise ValueError(
                        f"Unsupported structured trunk output head mode: {self.structured_trunk_output_head_mode}"
                    )
                self.structured_trunk_output_head_compose_mode = str(
                    self.structured_trunk_output_head_cfg.get(
                        'compose_mode',
                        self.structured_trunk_output_head_cfg.get('path_mode', 'residual'),
                    )
                ).lower()
                if self.structured_trunk_output_head_compose_mode not in (
                    'residual',
                    'gated_color',
                    'logit_gate',
                    'color',
                ):
                    raise ValueError(
                        "Unsupported structured trunk output head compose mode: "
                        f"{self.structured_trunk_output_head_compose_mode}"
                    )
                self.structured_trunk_output_head_disable_input_residual = bool(
                    self.structured_trunk_output_head_cfg.get('disable_input_residual', True)
                )
                self.structured_trunk_output_head_component_dim = int(
                    self.structured_trunk_output_head_cfg.get('component_dim', d_in)
                )
                self.structured_trunk_output_head_use_base_input = bool(
                    self.structured_trunk_output_head_cfg.get('use_base_input', True)
                )
                self.structured_trunk_output_head_use_shared = bool(
                    self.structured_trunk_output_head_cfg.get('use_shared', True)
                )
                self.structured_trunk_output_head_use_structure = bool(
                    self.structured_trunk_output_head_cfg.get('use_structure', True)
                )
                self.structured_trunk_output_head_use_local = bool(
                    self.structured_trunk_output_head_cfg.get('use_local', True)
                )
                self.structured_trunk_output_head_use_local_raw_feature = bool(
                    self.structured_trunk_output_head_cfg.get('use_local_raw_feature', False)
                )
                self.structured_trunk_output_head_use_local_gated_feature = bool(
                    self.structured_trunk_output_head_cfg.get('use_local_gated_feature', True)
                )
                if (
                    self.structured_trunk_output_head_use_local
                    and not self.structured_trunk_output_head_use_local_raw_feature
                    and not self.structured_trunk_output_head_use_local_gated_feature
                ):
                    self.structured_trunk_output_head_use_local_gated_feature = True
                self.structured_trunk_output_head_use_local_gate_feature = bool(
                    self.structured_trunk_output_head_cfg.get('use_local_gate_feature', True)
                )
                self.structured_trunk_output_head_local_soft_gate_min = float(
                    self.structured_trunk_output_head_cfg.get('local_soft_gate_min', 0.0)
                )
                self.structured_trunk_output_head_scale_cfg = self.structured_trunk_output_head_cfg.get(
                    'scale',
                    1.0,
                )
                self.structured_trunk_output_head_max_residual = float(
                    self.structured_trunk_output_head_cfg.get('max_residual', 0.18)
                )
                self.structured_trunk_output_head_gate_bias = float(
                    self.structured_trunk_output_head_cfg.get('gate_bias', 0.0)
                )
                self.structured_trunk_output_head_gate_gain = float(
                    self.structured_trunk_output_head_cfg.get('gate_gain', 1.0)
                )
                self.structured_trunk_output_head_min_gate = float(
                    self.structured_trunk_output_head_cfg.get('min_gate', 0.0)
                )
                self.structured_trunk_output_head_max_gate = float(
                    self.structured_trunk_output_head_cfg.get('max_gate', 1.0)
                )
                self.structured_trunk_output_head_chroma_center = bool(
                    self.structured_trunk_output_head_cfg.get('chroma_center', False)
                )
                self.structured_trunk_output_head_band_luma_scale = float(
                    self.structured_trunk_output_head_cfg.get('band_luma_scale', 0.70)
                )
                self.structured_trunk_output_head_band_chroma_scale = float(
                    self.structured_trunk_output_head_cfg.get('band_chroma_scale', 0.55)
                )
                self.structured_trunk_output_head_local_region_overrides = (
                    _normalize_local_output_head_override_cfgs(
                        self.structured_trunk_output_head_cfg.get(
                            'local_region_overrides',
                            None,
                        )
                    )
                )
                self.structured_trunk_output_head_local_color_cfg = (
                    self.structured_trunk_output_head_cfg.get('local_color', None)
                )
                self.structured_trunk_output_head_local_color_enable = bool(
                    self.structured_trunk_output_head_local_color_cfg.get('enable', False)
                ) if self.structured_trunk_output_head_local_color_cfg is not None else False
                if self.structured_trunk_output_head_local_color_enable:
                    self.structured_trunk_output_head_local_color_use_base_feature = bool(
                        self.structured_trunk_output_head_local_color_cfg.get(
                            'use_base_feature',
                            False,
                        )
                    )
                    self.structured_trunk_output_head_local_color_use_shared_feature = bool(
                        self.structured_trunk_output_head_local_color_cfg.get(
                            'use_shared_feature',
                            False,
                        )
                    )
                    self.structured_trunk_output_head_local_color_use_structure_feature = bool(
                        self.structured_trunk_output_head_local_color_cfg.get(
                            'use_structure_feature',
                            False,
                        )
                    )
                    self.structured_trunk_output_head_local_color_use_raw_feature = bool(
                        self.structured_trunk_output_head_local_color_cfg.get(
                            'use_raw_feature',
                            True,
                        )
                    )
                    self.structured_trunk_output_head_local_color_use_gated_feature = bool(
                        self.structured_trunk_output_head_local_color_cfg.get(
                            'use_gated_feature',
                            True,
                        )
                    )
                    self.structured_trunk_output_head_local_color_use_gate_feature = bool(
                        self.structured_trunk_output_head_local_color_cfg.get(
                            'use_gate_feature',
                            True,
                        )
                    )
                    self.structured_trunk_output_head_local_color_gate_with_region = bool(
                        self.structured_trunk_output_head_local_color_cfg.get(
                            'gate_with_region',
                            True,
                        )
                    )
                    self.structured_trunk_output_head_local_color_init_from_output_head = bool(
                        self.structured_trunk_output_head_local_color_cfg.get(
                            'init_from_output_head',
                            True,
                        )
                    )
                    self.structured_trunk_output_head_local_color_init_scale = float(
                        self.structured_trunk_output_head_local_color_cfg.get(
                            'init_scale',
                            0.35,
                        )
                    )
                    self.structured_trunk_output_head_local_color_scale_cfg = (
                        self.structured_trunk_output_head_local_color_cfg.get(
                            'scale',
                            1.0,
                        )
                    )
                    self.structured_trunk_output_head_local_color_owner_cfg = (
                        self.structured_trunk_output_head_local_color_cfg.get(
                            'owner',
                            None,
                        )
                    )
                    self.structured_trunk_output_head_local_color_owner_enable = bool(
                        self.structured_trunk_output_head_local_color_owner_cfg.get(
                            'enable',
                            False,
                        )
                    ) if self.structured_trunk_output_head_local_color_owner_cfg is not None else False
                    if self.structured_trunk_output_head_local_color_owner_enable:
                        self.structured_trunk_output_head_local_color_owner_scale_cfg = (
                            self.structured_trunk_output_head_local_color_owner_cfg.get(
                                'scale',
                                1.0,
                            )
                        )
                        self.structured_trunk_output_head_local_color_owner_use_support = bool(
                            self.structured_trunk_output_head_local_color_owner_cfg.get(
                                'use_support',
                                True,
                            )
                        )
                        self.structured_trunk_output_head_local_color_owner_use_region_gate = bool(
                            self.structured_trunk_output_head_local_color_owner_cfg.get(
                                'use_region_gate',
                                False,
                            )
                        )
                        self.structured_trunk_output_head_local_color_owner_gate_base = float(
                            self.structured_trunk_output_head_local_color_owner_cfg.get(
                                'gate_base',
                                0.0,
                            )
                        )
                        self.structured_trunk_output_head_local_color_owner_head_cfg = (
                            self.structured_trunk_output_head_local_color_owner_cfg.get(
                                'head',
                                None,
                            )
                        )
                        self.structured_trunk_output_head_local_color_owner_head_enable = bool(
                            self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                'enable',
                                False,
                            )
                        ) if self.structured_trunk_output_head_local_color_owner_head_cfg is not None else False
                        if self.structured_trunk_output_head_local_color_owner_head_enable:
                            owner_head_default_mode = self.structured_trunk_output_head_mode
                            owner_head_default_compose_mode = (
                                self.structured_trunk_output_head_compose_mode
                            )
                            owner_head_default_gate_gain = (
                                self.structured_trunk_output_head_gate_gain
                            )
                            owner_head_default_gate_bias = (
                                self.structured_trunk_output_head_gate_bias
                            )
                            owner_head_default_min_gate = (
                                self.structured_trunk_output_head_min_gate
                            )
                            owner_head_default_max_gate = (
                                self.structured_trunk_output_head_max_gate
                            )
                            owner_head_default_chroma_center = (
                                self.structured_trunk_output_head_chroma_center
                            )
                            owner_head_default_band_luma_scale = (
                                self.structured_trunk_output_head_band_luma_scale
                            )
                            owner_head_default_band_chroma_scale = (
                                self.structured_trunk_output_head_band_chroma_scale
                            )
                            owner_dual_head_cfg = self.structured_trunk_output_head_cfg.get(
                                'dual_head',
                                None,
                            )
                            if (
                                owner_dual_head_cfg is not None
                                and bool(owner_dual_head_cfg.get('enable', False))
                            ):
                                owner_hf_cfg = owner_dual_head_cfg.get(
                                    'hf_head',
                                    None,
                                )
                                if owner_hf_cfg is not None and bool(
                                    owner_hf_cfg.get('enable', False)
                                ):
                                    owner_head_default_mode = str(
                                        owner_hf_cfg.get(
                                            'mode',
                                            owner_head_default_mode,
                                        )
                                    ).lower()
                                    owner_head_default_compose_mode = str(
                                        owner_hf_cfg.get(
                                            'compose_mode',
                                            owner_hf_cfg.get(
                                                'path_mode',
                                                owner_head_default_compose_mode,
                                            ),
                                        )
                                    ).lower()
                                    owner_head_default_gate_gain = float(
                                        owner_hf_cfg.get(
                                            'gate_gain',
                                            owner_head_default_gate_gain,
                                        )
                                    )
                                    owner_head_default_gate_bias = float(
                                        owner_hf_cfg.get(
                                            'gate_bias',
                                            owner_head_default_gate_bias,
                                        )
                                    )
                                    owner_head_default_min_gate = float(
                                        owner_hf_cfg.get(
                                            'min_gate',
                                            owner_head_default_min_gate,
                                        )
                                    )
                                    owner_head_default_max_gate = float(
                                        owner_hf_cfg.get(
                                            'max_gate',
                                            owner_head_default_max_gate,
                                        )
                                    )
                                    owner_head_default_chroma_center = bool(
                                        owner_hf_cfg.get(
                                            'chroma_center',
                                            owner_head_default_chroma_center,
                                        )
                                    )
                                    owner_head_default_band_luma_scale = float(
                                        owner_hf_cfg.get(
                                            'band_luma_scale',
                                            owner_head_default_band_luma_scale,
                                        )
                                    )
                                    owner_head_default_band_chroma_scale = float(
                                        owner_hf_cfg.get(
                                            'band_chroma_scale',
                                            owner_head_default_band_chroma_scale,
                                        )
                                    )
                            self.structured_trunk_output_head_local_color_owner_head_mode = str(
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'mode',
                                    owner_head_default_mode,
                                )
                            ).lower()
                            if self.structured_trunk_output_head_local_color_owner_head_mode not in (
                                'rgb',
                                'band',
                            ):
                                raise ValueError(
                                    "Unsupported structured trunk owner head mode: "
                                    f"{self.structured_trunk_output_head_local_color_owner_head_mode}"
                                )
                            self.structured_trunk_output_head_local_color_owner_head_compose_mode = str(
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'compose_mode',
                                    self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                        'path_mode',
                                        owner_head_default_compose_mode,
                                    ),
                                )
                            ).lower()
                            if self.structured_trunk_output_head_local_color_owner_head_compose_mode not in (
                                'residual',
                                'gated_color',
                                'logit_gate',
                                'color',
                            ):
                                raise ValueError(
                                    "Unsupported structured trunk owner head compose mode: "
                                    f"{self.structured_trunk_output_head_local_color_owner_head_compose_mode}"
                                )
                            self.structured_trunk_output_head_local_color_owner_head_scale_cfg = (
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'scale',
                                    self.structured_trunk_output_head_local_color_owner_scale_cfg,
                                )
                            )
                            self.structured_trunk_output_head_local_color_owner_head_gate_gain = float(
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'gate_gain',
                                    owner_head_default_gate_gain,
                                )
                            )
                            self.structured_trunk_output_head_local_color_owner_head_gate_bias = float(
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'gate_bias',
                                    owner_head_default_gate_bias,
                                )
                            )
                            self.structured_trunk_output_head_local_color_owner_head_min_gate = float(
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'min_gate',
                                    owner_head_default_min_gate,
                                )
                            )
                            self.structured_trunk_output_head_local_color_owner_head_max_gate = float(
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'max_gate',
                                    owner_head_default_max_gate,
                                )
                            )
                            self.structured_trunk_output_head_local_color_owner_head_chroma_center = bool(
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'chroma_center',
                                    owner_head_default_chroma_center,
                                )
                            )
                            self.structured_trunk_output_head_local_color_owner_head_band_luma_scale = float(
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'band_luma_scale',
                                    owner_head_default_band_luma_scale,
                                )
                            )
                            self.structured_trunk_output_head_local_color_owner_head_band_chroma_scale = float(
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'band_chroma_scale',
                                    owner_head_default_band_chroma_scale,
                                )
                            )
                            self.structured_trunk_output_head_local_color_owner_head_use_local_color_input = bool(
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'use_local_color_input',
                                    True,
                                )
                            )
                            self.structured_trunk_output_head_local_color_owner_head_use_local_color_output = bool(
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'use_local_color_output',
                                    True,
                                )
                            )
                            self.structured_trunk_output_head_local_color_owner_head_local_color_output_scale_cfg = (
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'local_color_output_scale',
                                    1.0,
                                )
                            )
                            self.structured_trunk_output_head_local_color_owner_head_use_local_geometry_raw = bool(
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'use_local_geometry_raw',
                                    True,
                                )
                            )
                            self.structured_trunk_output_head_local_color_owner_head_use_support_feature = bool(
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'use_support_feature',
                                    self.structured_trunk_output_head_local_color_owner_use_support,
                                )
                            )
                            self.structured_trunk_output_head_local_color_owner_head_use_region_gate_feature = bool(
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'use_region_gate_feature',
                                    False,
                                )
                            )
                            self.structured_trunk_output_head_local_color_owner_head_init_from_local_color = bool(
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'init_from_local_color',
                                    True,
                                )
                            )
                            self.structured_trunk_output_head_local_color_owner_head_init_scale = float(
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'init_scale',
                                    0.5,
                                )
                            )
                            self.structured_trunk_output_head_local_color_owner_head_takeover_cfg = (
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'takeover',
                                    None,
                                )
                            )
                            self.structured_trunk_output_head_local_color_owner_head_takeover_enable = bool(
                                self.structured_trunk_output_head_local_color_owner_head_takeover_cfg.get(
                                    'enable',
                                    False,
                                )
                            ) if self.structured_trunk_output_head_local_color_owner_head_takeover_cfg is not None else False
                            if self.structured_trunk_output_head_local_color_owner_head_takeover_enable:
                                takeover_cfg = (
                                    self.structured_trunk_output_head_local_color_owner_head_takeover_cfg
                                )
                                self.structured_trunk_output_head_local_color_owner_head_takeover_scale_cfg = (
                                    takeover_cfg.get(
                                        'scale',
                                        self.structured_trunk_output_head_local_color_owner_head_scale_cfg,
                                    )
                                )
                                self.structured_trunk_output_head_local_color_owner_head_takeover_strength_cfg = (
                                    takeover_cfg.get('strength', 1.0)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_takeover_base_mix = float(
                                    takeover_cfg.get('base_mix', 0.0)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_takeover_max = float(
                                    takeover_cfg.get('max', 1.0)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_takeover_use_support = bool(
                                    takeover_cfg.get('use_support', True)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_takeover_support_detach = bool(
                                    takeover_cfg.get('support_detach', True)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_takeover_support_offset = float(
                                    takeover_cfg.get('support_offset', 0.0)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_takeover_support_gain = float(
                                    takeover_cfg.get('support_gain', 1.0)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_takeover_support_power = float(
                                    takeover_cfg.get('support_power', 1.0)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_takeover_region_strength_cfg = (
                                    _normalize_named_scalar_cfg(
                                        takeover_cfg.get('region_strength', None)
                                    )
                                )
                                self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_cfg = (
                                    takeover_cfg.get('legacy_decay', None)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_enable = bool(
                                    self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_cfg.get(
                                        'enable',
                                        False,
                                    )
                                ) if self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_cfg is not None else False
                                if self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_enable:
                                    legacy_decay_cfg = (
                                        self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_cfg
                                    )
                                    self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_power = float(
                                        legacy_decay_cfg.get('power', 1.0)
                                    )
                                    self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_min_scale = float(
                                        legacy_decay_cfg.get('min_scale', 0.0)
                                    )
                                    self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_apply_to_coarse = bool(
                                        legacy_decay_cfg.get('apply_to_coarse', True)
                                    )
                                    self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_apply_to_hf = bool(
                                        legacy_decay_cfg.get('apply_to_hf', True)
                                    )
                            self.structured_trunk_output_head_local_color_owner_head_boundary_cfg = (
                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                    'boundary',
                                    None,
                                )
                            )
                            self.structured_trunk_output_head_local_color_owner_head_boundary_enable = bool(
                                self.structured_trunk_output_head_local_color_owner_head_boundary_cfg.get(
                                    'enable',
                                    False,
                                )
                            ) if self.structured_trunk_output_head_local_color_owner_head_boundary_cfg is not None else False
                            if self.structured_trunk_output_head_local_color_owner_head_boundary_enable:
                                boundary_cfg = (
                                    self.structured_trunk_output_head_local_color_owner_head_boundary_cfg
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_mode = str(
                                    boundary_cfg.get(
                                        'mode',
                                        self.structured_trunk_output_head_local_color_owner_head_mode,
                                    )
                                ).lower()
                                self.structured_trunk_output_head_local_color_owner_head_boundary_compose_mode = str(
                                    boundary_cfg.get(
                                        'compose_mode',
                                        self.structured_trunk_output_head_local_color_owner_head_compose_mode,
                                    )
                                ).lower()
                                self.structured_trunk_output_head_local_color_owner_head_boundary_scale_cfg = (
                                    boundary_cfg.get(
                                        'scale',
                                        self.structured_trunk_output_head_local_color_owner_head_scale_cfg,
                                    )
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_gate_gain = float(
                                    boundary_cfg.get(
                                        'gate_gain',
                                        self.structured_trunk_output_head_local_color_owner_head_gate_gain,
                                    )
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_gate_bias = float(
                                    boundary_cfg.get(
                                        'gate_bias',
                                        self.structured_trunk_output_head_local_color_owner_head_gate_bias,
                                    )
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_min_gate = float(
                                    boundary_cfg.get(
                                        'min_gate',
                                        self.structured_trunk_output_head_local_color_owner_head_min_gate,
                                    )
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_max_gate = float(
                                    boundary_cfg.get(
                                        'max_gate',
                                        self.structured_trunk_output_head_local_color_owner_head_max_gate,
                                    )
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_contrib_gate_mode = str(
                                    boundary_cfg.get('contrib_gate_mode', 'multiply')
                                ).lower()
                                self.structured_trunk_output_head_local_color_owner_head_boundary_contrib_min_gate = float(
                                    boundary_cfg.get('contrib_min_gate', 0.0)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_contrib_scale_cfg = (
                                    boundary_cfg.get('contrib_scale', 1.0)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_chroma_center = bool(
                                    boundary_cfg.get(
                                        'chroma_center',
                                        self.structured_trunk_output_head_local_color_owner_head_chroma_center,
                                    )
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_band_luma_scale = float(
                                    boundary_cfg.get(
                                        'band_luma_scale',
                                        self.structured_trunk_output_head_local_color_owner_head_band_luma_scale,
                                    )
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_band_chroma_scale = float(
                                    boundary_cfg.get(
                                        'band_chroma_scale',
                                        self.structured_trunk_output_head_local_color_owner_head_band_chroma_scale,
                                    )
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_use_local_color_input = bool(
                                    boundary_cfg.get(
                                        'use_local_color_input',
                                        self.structured_trunk_output_head_local_color_owner_head_use_local_color_input,
                                    )
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_use_local_color_output = bool(
                                    boundary_cfg.get(
                                        'use_local_color_output',
                                        False,
                                    )
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_local_color_output_scale_cfg = (
                                    boundary_cfg.get(
                                        'local_color_output_scale',
                                        self.structured_trunk_output_head_local_color_owner_head_local_color_output_scale_cfg,
                                    )
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_use_local_geometry_raw = bool(
                                    boundary_cfg.get(
                                        'use_local_geometry_raw',
                                        self.structured_trunk_output_head_local_color_owner_head_use_local_geometry_raw,
                                    )
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_use_support_feature = bool(
                                    boundary_cfg.get(
                                        'use_support_feature',
                                        False,
                                    )
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_use_region_gate_feature = bool(
                                    boundary_cfg.get(
                                        'use_region_gate_feature',
                                        True,
                                    )
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_use_boundary_feature = bool(
                                    boundary_cfg.get(
                                        'use_boundary_feature',
                                        True,
                                    )
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_focus_source = str(
                                    boundary_cfg.get('focus_source', 'score')
                                ).lower()
                                self.structured_trunk_output_head_local_color_owner_head_boundary_focus_threshold = float(
                                    boundary_cfg.get('focus_threshold', 0.0)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_focus_power = float(
                                    boundary_cfg.get('focus_power', 1.0)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_focus_min = float(
                                    boundary_cfg.get('focus_min', 0.0)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_focus_max = float(
                                    boundary_cfg.get('focus_max', 1.0)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_focus_use_region_gate = bool(
                                    boundary_cfg.get('focus_use_region_gate', True)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_focus_use_support = bool(
                                    boundary_cfg.get('focus_use_support', False)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_focus_support_detach = bool(
                                    boundary_cfg.get('focus_support_detach', True)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_init_from_local_color = bool(
                                    boundary_cfg.get('init_from_local_color', True)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_init_scale = float(
                                    boundary_cfg.get('init_scale', 0.35)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_cfg = (
                                    boundary_cfg.get('takeover', None)
                                )
                                self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_enable = bool(
                                    self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_cfg.get(
                                        'enable',
                                        False,
                                    )
                                ) if self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_cfg is not None else False
                                if self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_enable:
                                    boundary_takeover_cfg = (
                                        self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_cfg
                                    )
                                    self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_scale_cfg = (
                                        boundary_takeover_cfg.get('scale', 1.0)
                                    )
                                    self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_max = float(
                                        boundary_takeover_cfg.get('max', 1.0)
                                    )
                                    self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_use_region_gate = bool(
                                        boundary_takeover_cfg.get('use_region_gate', True)
                                    )
                                if not (
                                    self.structured_trunk_output_head_local_color_owner_head_boundary_use_local_color_input
                                    or self.structured_trunk_output_head_local_color_owner_head_boundary_use_local_color_output
                                    or self.structured_trunk_output_head_local_color_owner_head_boundary_use_local_geometry_raw
                                    or self.structured_trunk_output_head_local_color_owner_head_boundary_use_support_feature
                                    or self.structured_trunk_output_head_local_color_owner_head_boundary_use_region_gate_feature
                                    or self.structured_trunk_output_head_local_color_owner_head_boundary_use_boundary_feature
                                ):
                                    self.structured_trunk_output_head_local_color_owner_head_boundary_use_boundary_feature = True
                            if not (
                                self.structured_trunk_output_head_local_color_owner_head_use_local_color_input
                                or self.structured_trunk_output_head_local_color_owner_head_use_local_color_output
                                or self.structured_trunk_output_head_local_color_owner_head_use_local_geometry_raw
                                or self.structured_trunk_output_head_local_color_owner_head_use_support_feature
                                or self.structured_trunk_output_head_local_color_owner_head_use_region_gate_feature
                            ):
                                self.structured_trunk_output_head_local_color_owner_head_use_local_color_input = True
                    self.structured_trunk_output_head_local_color_use_current_xyz_feature = bool(
                        self.structured_trunk_output_head_local_color_cfg.get(
                            'use_current_xyz_feature',
                            False,
                        )
                    )
                    self.structured_trunk_output_head_local_color_current_xyz_multires = int(
                        self.structured_trunk_output_head_local_color_cfg.get(
                            'current_xyz_multires',
                            6,
                        )
                    )
                    self.structured_trunk_output_head_local_color_current_xyz_include_input = bool(
                        self.structured_trunk_output_head_local_color_cfg.get(
                            'current_xyz_include_input',
                            True,
                        )
                    )
                    self.structured_trunk_output_head_local_color_current_xyz_scale = float(
                        self.structured_trunk_output_head_local_color_cfg.get(
                            'current_xyz_frequency_scale',
                            1.0,
                        )
                    )
                    self.structured_trunk_output_head_local_color_use_current_radius_feature = bool(
                        self.structured_trunk_output_head_local_color_cfg.get(
                            'use_current_radius_feature',
                            False,
                        )
                    )
                    self.structured_trunk_output_head_local_color_current_radius_multires = int(
                        self.structured_trunk_output_head_local_color_cfg.get(
                            'current_radius_multires',
                            4,
                        )
                    )
                    self.structured_trunk_output_head_local_color_current_radius_include_input = bool(
                        self.structured_trunk_output_head_local_color_cfg.get(
                            'current_radius_include_input',
                            True,
                        )
                    )
                    self.structured_trunk_output_head_local_color_current_radius_scale = float(
                        self.structured_trunk_output_head_local_color_cfg.get(
                            'current_radius_frequency_scale',
                            1.0,
                        )
                    )
                if not (
                        self.structured_trunk_output_head_local_color_use_base_feature
                        or self.structured_trunk_output_head_local_color_use_shared_feature
                        or self.structured_trunk_output_head_local_color_use_structure_feature
                        or self.structured_trunk_output_head_local_color_use_raw_feature
                        or self.structured_trunk_output_head_local_color_use_gated_feature
                        or self.structured_trunk_output_head_local_color_use_gate_feature
                        or self.structured_trunk_output_head_local_color_use_current_xyz_feature
                        or self.structured_trunk_output_head_local_color_use_current_radius_feature
                    ):
                        self.structured_trunk_output_head_local_color_use_gated_feature = True

                self.structured_trunk_output_head_dual_head_cfg = (
                    self.structured_trunk_output_head_cfg.get('dual_head', None)
                )
                self.structured_trunk_output_head_dual_head_enable = bool(
                    self.structured_trunk_output_head_dual_head_cfg.get('enable', False)
                ) if self.structured_trunk_output_head_dual_head_cfg is not None else False
                if self.structured_trunk_output_head_dual_head_enable:
                    self.structured_trunk_output_head_base_scaffold_scale_cfg = (
                        self.structured_trunk_output_head_dual_head_cfg.get(
                            'base_scaffold_scale',
                            1.0,
                        )
                    )
                    self.structured_trunk_output_head_coarse_scale_cfg = (
                        self.structured_trunk_output_head_dual_head_cfg.get(
                            'coarse_scale',
                            1.0,
                        )
                    )
                    self.structured_trunk_output_head_coarse_region_suppress_cfg = (
                        _normalize_named_scalar_cfg(
                            self.structured_trunk_output_head_dual_head_cfg.get(
                                'coarse_region_suppress',
                                None,
                            )
                        )
                    )
                    self.structured_trunk_output_head_coarse_region_min_scale = float(
                        self.structured_trunk_output_head_dual_head_cfg.get(
                            'coarse_region_min_scale',
                            0.0,
                        )
                    )
                    self.structured_trunk_output_head_region_support_cfg = (
                        self.structured_trunk_output_head_dual_head_cfg.get(
                            'region_support',
                            None,
                        )
                    )
                    self.structured_trunk_output_head_region_support_enable = bool(
                        self.structured_trunk_output_head_region_support_cfg.get(
                            'enable',
                            False,
                        )
                    ) if self.structured_trunk_output_head_region_support_cfg is not None else False
                    if self.structured_trunk_output_head_region_support_enable:
                        self.structured_trunk_output_head_region_support_source = str(
                            self.structured_trunk_output_head_region_support_cfg.get(
                                'source',
                                'hybrid',
                            )
                        ).lower()
                        self.structured_trunk_output_head_region_support_detach = bool(
                            self.structured_trunk_output_head_region_support_cfg.get(
                                'detach',
                                True,
                            )
                        )
                        self.structured_trunk_output_head_region_support_offset = float(
                            self.structured_trunk_output_head_region_support_cfg.get(
                                'offset',
                                0.0,
                            )
                        )
                        self.structured_trunk_output_head_region_support_gain = float(
                            self.structured_trunk_output_head_region_support_cfg.get(
                                'gain',
                                1.0,
                            )
                        )
                        self.structured_trunk_output_head_region_support_power = float(
                            self.structured_trunk_output_head_region_support_cfg.get(
                                'power',
                                1.0,
                            )
                        )
                        self.structured_trunk_output_head_region_support_base_mix = float(
                            self.structured_trunk_output_head_region_support_cfg.get(
                                'base_mix',
                                0.0,
                            )
                        )
                        self.structured_trunk_output_head_region_support_max = float(
                            self.structured_trunk_output_head_region_support_cfg.get(
                                'max',
                                1.0,
                            )
                        )
                    self.structured_trunk_output_head_hf_head_cfg = (
                        self.structured_trunk_output_head_dual_head_cfg.get(
                            'hf_head',
                            None,
                        )
                    )
                    self.structured_trunk_output_head_hf_head_enable = bool(
                        self.structured_trunk_output_head_hf_head_cfg.get('enable', True)
                    ) if self.structured_trunk_output_head_hf_head_cfg is not None else False
                    if self.structured_trunk_output_head_hf_head_enable:
                        self.structured_trunk_output_head_hf_head_mode = str(
                            self.structured_trunk_output_head_hf_head_cfg.get(
                                'mode',
                                self.structured_trunk_output_head_mode,
                            )
                        ).lower()
                        if self.structured_trunk_output_head_hf_head_mode not in ('rgb', 'band'):
                            raise ValueError(
                                "Unsupported structured trunk dual-head high-frequency mode: "
                                f"{self.structured_trunk_output_head_hf_head_mode}"
                            )
                        self.structured_trunk_output_head_hf_head_compose_mode = str(
                            self.structured_trunk_output_head_hf_head_cfg.get(
                                'compose_mode',
                                'gated_color',
                            )
                        ).lower()
                        if self.structured_trunk_output_head_hf_head_compose_mode not in (
                            'residual',
                            'gated_color',
                            'logit_gate',
                            'color',
                        ):
                            raise ValueError(
                                "Unsupported structured trunk dual-head high-frequency compose mode: "
                                f"{self.structured_trunk_output_head_hf_head_compose_mode}"
                            )
                        self.structured_trunk_output_head_hf_head_scale_cfg = (
                            self.structured_trunk_output_head_hf_head_cfg.get(
                                'scale',
                                1.0,
                            )
                        )
                        self.structured_trunk_output_head_hf_head_max_residual = float(
                            self.structured_trunk_output_head_hf_head_cfg.get(
                                'max_residual',
                                self.structured_trunk_output_head_max_residual,
                            )
                        )
                        self.structured_trunk_output_head_hf_head_gate_bias = float(
                            self.structured_trunk_output_head_hf_head_cfg.get(
                                'gate_bias',
                                self.structured_trunk_output_head_gate_bias,
                            )
                        )
                        self.structured_trunk_output_head_hf_head_gate_gain = float(
                            self.structured_trunk_output_head_hf_head_cfg.get(
                                'gate_gain',
                                self.structured_trunk_output_head_gate_gain,
                            )
                        )
                        self.structured_trunk_output_head_hf_head_min_gate = float(
                            self.structured_trunk_output_head_hf_head_cfg.get(
                                'min_gate',
                                self.structured_trunk_output_head_min_gate,
                            )
                        )
                        self.structured_trunk_output_head_hf_head_max_gate = float(
                            self.structured_trunk_output_head_hf_head_cfg.get(
                                'max_gate',
                                self.structured_trunk_output_head_max_gate,
                            )
                        )
                        self.structured_trunk_output_head_hf_head_chroma_center = bool(
                            self.structured_trunk_output_head_hf_head_cfg.get(
                                'chroma_center',
                                self.structured_trunk_output_head_chroma_center,
                            )
                        )
                        self.structured_trunk_output_head_hf_head_band_luma_scale = float(
                            self.structured_trunk_output_head_hf_head_cfg.get(
                                'band_luma_scale',
                                self.structured_trunk_output_head_band_luma_scale,
                            )
                        )
                        self.structured_trunk_output_head_hf_head_band_chroma_scale = float(
                            self.structured_trunk_output_head_hf_head_cfg.get(
                                'band_chroma_scale',
                                self.structured_trunk_output_head_hf_head_band_chroma_scale,
                            )
                        )
                        self.structured_trunk_output_head_hf_head_use_local_color = bool(
                            self.structured_trunk_output_head_hf_head_cfg.get(
                                'use_local_color',
                                True,
                            )
                        )
                        self.structured_trunk_output_head_hf_head_local_color_scale_cfg = (
                            self.structured_trunk_output_head_hf_head_cfg.get(
                                'local_color_scale',
                                1.0,
                            )
                        )
                        self.structured_trunk_output_head_hf_head_reuse_output_gate = bool(
                            self.structured_trunk_output_head_hf_head_cfg.get(
                                'reuse_output_gate',
                                True,
                            )
                        )
                        self.structured_trunk_output_head_hf_head_init_from_output_head = bool(
                            self.structured_trunk_output_head_hf_head_cfg.get(
                                'init_from_output_head',
                                True,
                            )
                        )
                        self.structured_trunk_output_head_hf_head_init_scale = float(
                            self.structured_trunk_output_head_hf_head_cfg.get(
                                'init_scale',
                                0.15,
                            )
                        )
                        self.structured_trunk_output_head_hf_head_region_boost_cfg = (
                            _normalize_named_scalar_cfg(
                                self.structured_trunk_output_head_hf_head_cfg.get(
                                    'region_boost',
                                    None,
                                )
                            )
                        )
                        self.structured_trunk_output_head_hf_head_region_boost_max = float(
                            self.structured_trunk_output_head_hf_head_cfg.get(
                                'region_boost_max',
                                0.0,
                            )
                        )
                        hf_head_input_cfg = self.structured_trunk_output_head_hf_head_cfg.get(
                            'input',
                            None,
                        )
                        if hf_head_input_cfg is not None:
                            self.structured_trunk_output_head_hf_head_use_output_fusion = bool(
                                hf_head_input_cfg.get('use_output_fusion', True)
                            )
                            self.structured_trunk_output_head_hf_head_use_shared_raw = bool(
                                hf_head_input_cfg.get('use_shared_raw', False)
                            )
                            self.structured_trunk_output_head_hf_head_use_structure_raw = bool(
                                hf_head_input_cfg.get('use_structure_raw', False)
                            )
                            self.structured_trunk_output_head_hf_head_use_local_geometry_raw = bool(
                                hf_head_input_cfg.get('use_local_geometry_raw', False)
                            )

                fusion_dim = 0
                local_color_geometry_input_dim = self._encoded_local_color_geometry_dim()
                hf_head_local_geometry_keys = []
                self.structured_trunk_output_head_local_color_input_dims = {}
                self.structured_trunk_output_head_local_color_owner_head_input_dims = {}
                self.structured_trunk_output_head_local_color_owner_head_boundary_input_dims = {}
                if self.structured_trunk_output_head_use_base_input:
                    if self.structured_trunk_output_head_component_dim != d_in:
                        self.structured_trunk_output_head_base_proj = nn.Linear(
                            d_in,
                            self.structured_trunk_output_head_component_dim,
                        )
                        nn.init.normal_(
                            self.structured_trunk_output_head_base_proj.weight,
                            mean=0.0,
                            std=0.02,
                        )
                        nn.init.constant_(self.structured_trunk_output_head_base_proj.bias, 0.0)
                    fusion_dim += self.structured_trunk_output_head_component_dim

                if self.structured_trunk_output_head_use_shared and shared_input_dim > 0:
                    self.structured_trunk_output_head_shared_fusion_proj = self._build_feature_fusion_proj(
                        shared_input_dim,
                        self.structured_trunk_output_head_component_dim,
                    )
                    fusion_dim += self.structured_trunk_output_head_component_dim

                if (
                    self.structured_trunk_output_head_use_structure
                    and self.structured_trunk_structure_enable
                    and structure_raw_dim > 0
                ):
                    self.structured_trunk_output_head_structure_fusion_proj = self._build_feature_fusion_proj(
                        structure_raw_dim,
                        self.structured_trunk_output_head_component_dim,
                    )
                    fusion_dim += self.structured_trunk_output_head_component_dim

                if self.structured_trunk_output_head_use_local:
                    for local_cfg in self.structured_trunk_local_cfgs:
                        if not bool(local_cfg.get('enable', False)):
                            continue
                        local_key = str(local_cfg.get('_key', ''))
                        if not local_key:
                            continue
                        local_raw_dim = self._encoded_local_carrier_dim(local_cfg)
                        if local_raw_dim <= 0:
                            continue
                        self.structured_trunk_output_head_local_fusion_projs[local_key] = (
                            self._build_feature_fusion_proj(
                                local_raw_dim,
                                self.structured_trunk_output_head_component_dim,
                            )
                        )
                        if self.structured_trunk_output_head_use_local_raw_feature:
                            fusion_dim += self.structured_trunk_output_head_component_dim
                        if self.structured_trunk_output_head_use_local_gated_feature:
                            fusion_dim += self.structured_trunk_output_head_component_dim
                        if self.structured_trunk_output_head_use_local_gate_feature:
                            fusion_dim += 1
                        if local_color_geometry_input_dim > 0:
                            hf_head_local_geometry_keys.append(local_key)
                        local_color_input_dim = 0
                        if (
                            self.structured_trunk_output_head_local_color_enable
                            or self.structured_trunk_output_head_local_color_owner_head_enable
                        ):
                            if self.structured_trunk_output_head_local_color_use_base_feature:
                                local_color_input_dim += self.structured_trunk_output_head_component_dim
                            if self.structured_trunk_output_head_local_color_use_shared_feature:
                                local_color_input_dim += self.structured_trunk_output_head_component_dim
                            if self.structured_trunk_output_head_local_color_use_structure_feature:
                                local_color_input_dim += self.structured_trunk_output_head_component_dim
                            if self.structured_trunk_output_head_local_color_use_raw_feature:
                                local_color_input_dim += self.structured_trunk_output_head_component_dim
                            if self.structured_trunk_output_head_local_color_use_gated_feature:
                                local_color_input_dim += self.structured_trunk_output_head_component_dim
                            if self.structured_trunk_output_head_local_color_use_gate_feature:
                                local_color_input_dim += 1
                            if local_color_geometry_input_dim > 0:
                                self.structured_trunk_output_head_local_geometry_fusion_projs[local_key] = (
                                    self._build_feature_fusion_proj(
                                        local_color_geometry_input_dim,
                                        self.structured_trunk_output_head_component_dim,
                                    )
                                )
                                local_color_input_dim += self.structured_trunk_output_head_component_dim
                            self.structured_trunk_output_head_local_color_input_dims[local_key] = (
                                local_color_input_dim
                            )
                            if (
                                self.structured_trunk_output_head_local_color_owner_head_enable
                                and local_key
                            ):
                                owner_head_input_dim = 0
                                if (
                                    self.structured_trunk_output_head_local_color_owner_head_use_local_color_input
                                    and local_color_input_dim > 0
                                ):
                                    owner_head_input_dim += local_color_input_dim
                                if (
                                    self.structured_trunk_output_head_local_color_owner_head_use_local_color_output
                                ):
                                    owner_head_input_dim += d_out
                                if (
                                    self.structured_trunk_output_head_local_color_owner_head_use_local_geometry_raw
                                    and local_color_geometry_input_dim > 0
                                ):
                                    owner_head_input_dim += local_color_geometry_input_dim
                                if (
                                    self.structured_trunk_output_head_local_color_owner_head_use_support_feature
                                ):
                                    owner_head_input_dim += 1
                                if (
                                    self.structured_trunk_output_head_local_color_owner_head_use_region_gate_feature
                                ):
                                    owner_head_input_dim += 1
                                if owner_head_input_dim > 0:
                                    owner_head_out_dim = (
                                        d_out
                                        if self.structured_trunk_output_head_local_color_owner_head_mode == 'rgb'
                                        else (d_out + 1)
                                    )
                                    self.structured_trunk_output_head_local_color_owner_head_mlps[local_key] = (
                                        self._build_structured_trunk_head_mlp(
                                            self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                                'mlp',
                                                self.structured_trunk_output_head_local_color_cfg.get(
                                                    'mlp',
                                                    self.structured_trunk_output_head_cfg.get('mlp', None),
                                                ),
                                            ),
                                            owner_head_input_dim,
                                            owner_head_out_dim,
                                        )
                                    )
                                    self.structured_trunk_output_head_local_color_owner_head_gate_mlps[local_key] = (
                                        self._build_structured_trunk_head_mlp(
                                            self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                                'gate_mlp',
                                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                                    'mlp',
                                                    self.structured_trunk_output_head_cfg.get(
                                                        'gate_mlp',
                                                        self.structured_trunk_output_head_cfg.get(
                                                            'mlp',
                                                            None,
                                                        ),
                                                    ),
                                                ),
                                            ),
                                            owner_head_input_dim,
                                            1,
                                        )
                                    )
                                    self.structured_trunk_output_head_local_color_owner_head_input_dims[local_key] = (
                                        owner_head_input_dim
                                    )
                            if (
                                self.structured_trunk_output_head_local_color_owner_head_boundary_enable
                                and local_key
                            ):
                                boundary_head_input_dim = 0
                                if (
                                    self.structured_trunk_output_head_local_color_owner_head_boundary_use_local_color_input
                                    and local_color_input_dim > 0
                                ):
                                    boundary_head_input_dim += local_color_input_dim
                                if (
                                    self.structured_trunk_output_head_local_color_owner_head_boundary_use_local_color_output
                                ):
                                    boundary_head_input_dim += d_out
                                if (
                                    self.structured_trunk_output_head_local_color_owner_head_boundary_use_local_geometry_raw
                                    and local_color_geometry_input_dim > 0
                                ):
                                    boundary_head_input_dim += local_color_geometry_input_dim
                                if (
                                    self.structured_trunk_output_head_local_color_owner_head_boundary_use_support_feature
                                ):
                                    boundary_head_input_dim += 1
                                if (
                                    self.structured_trunk_output_head_local_color_owner_head_boundary_use_region_gate_feature
                                ):
                                    boundary_head_input_dim += 1
                                if (
                                    self.structured_trunk_output_head_local_color_owner_head_boundary_use_boundary_feature
                                ):
                                    boundary_head_input_dim += 1
                                if boundary_head_input_dim > 0:
                                    boundary_head_out_dim = (
                                        d_out
                                        if self.structured_trunk_output_head_local_color_owner_head_boundary_mode == 'rgb'
                                        else (d_out + 1)
                                    )
                                    self.structured_trunk_output_head_local_color_owner_head_boundary_mlps[local_key] = (
                                        self._build_structured_trunk_head_mlp(
                                            self.structured_trunk_output_head_local_color_owner_head_boundary_cfg.get(
                                                'mlp',
                                                self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                                    'mlp',
                                                    self.structured_trunk_output_head_local_color_cfg.get(
                                                        'mlp',
                                                        self.structured_trunk_output_head_cfg.get('mlp', None),
                                                    ),
                                                ),
                                            ),
                                            boundary_head_input_dim,
                                            boundary_head_out_dim,
                                        )
                                    )
                                    self.structured_trunk_output_head_local_color_owner_head_boundary_gate_mlps[local_key] = (
                                        self._build_structured_trunk_head_mlp(
                                            self.structured_trunk_output_head_local_color_owner_head_boundary_cfg.get(
                                                'gate_mlp',
                                                self.structured_trunk_output_head_local_color_owner_head_boundary_cfg.get(
                                                    'mlp',
                                                    self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                                        'gate_mlp',
                                                        self.structured_trunk_output_head_local_color_owner_head_cfg.get(
                                                            'mlp',
                                                            self.structured_trunk_output_head_cfg.get(
                                                                'gate_mlp',
                                                                self.structured_trunk_output_head_cfg.get(
                                                                    'mlp',
                                                                    None,
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                            boundary_head_input_dim,
                                            1,
                                        )
                                    )
                                    self.structured_trunk_output_head_local_color_owner_head_boundary_input_dims[local_key] = (
                                        boundary_head_input_dim
                                    )
                            if local_color_input_dim > 0:
                                if self.structured_trunk_output_head_local_color_enable:
                                    self.structured_trunk_output_head_local_color_mlps[local_key] = (
                                        self._build_structured_trunk_head_mlp(
                                            self.structured_trunk_output_head_local_color_cfg.get(
                                                'mlp',
                                                self.structured_trunk_output_head_cfg.get('mlp', None),
                                            ),
                                            local_color_input_dim,
                                            d_out,
                                        )
                                    )

                local_input_cursor = 0
                self.structured_trunk_output_head_local_input_slices = {}
                self.structured_trunk_output_head_global_input_slices = {}
                if self.structured_trunk_output_head_use_base_input:
                    self.structured_trunk_output_head_global_input_slices['base'] = (
                        local_input_cursor,
                        local_input_cursor + self.structured_trunk_output_head_component_dim,
                    )
                    local_input_cursor += self.structured_trunk_output_head_component_dim
                if self.structured_trunk_output_head_shared_fusion_proj is not None:
                    self.structured_trunk_output_head_global_input_slices['shared'] = (
                        local_input_cursor,
                        local_input_cursor + self.structured_trunk_output_head_component_dim,
                    )
                    local_input_cursor += self.structured_trunk_output_head_component_dim
                if self.structured_trunk_output_head_structure_fusion_proj is not None:
                    self.structured_trunk_output_head_global_input_slices['structure'] = (
                        local_input_cursor,
                        local_input_cursor + self.structured_trunk_output_head_component_dim,
                    )
                    local_input_cursor += self.structured_trunk_output_head_component_dim
                deferred_local_raw_keys = []
                if self.structured_trunk_output_head_use_local:
                    for local_cfg in self.structured_trunk_local_cfgs:
                        if not bool(local_cfg.get('enable', False)):
                            continue
                        local_key = str(local_cfg.get('_key', ''))
                        if local_key not in self.structured_trunk_output_head_local_fusion_projs:
                            continue
                        local_input_slices = self.structured_trunk_output_head_local_input_slices.setdefault(
                            local_key,
                            {},
                        )
                        if self.structured_trunk_output_head_use_local_gate_feature:
                            local_input_slices['gate'] = (
                                local_input_cursor,
                                local_input_cursor + 1,
                            )
                            local_input_cursor += 1
                        if self.structured_trunk_output_head_use_local_gated_feature:
                            local_input_slices['gated'] = (
                                local_input_cursor,
                                local_input_cursor + self.structured_trunk_output_head_component_dim,
                            )
                            local_input_cursor += self.structured_trunk_output_head_component_dim
                        if self.structured_trunk_output_head_use_local_raw_feature:
                            deferred_local_raw_keys.append(local_key)
                    for local_key in deferred_local_raw_keys:
                        local_input_slices = self.structured_trunk_output_head_local_input_slices.setdefault(
                            local_key,
                            {},
                        )
                        local_input_slices['raw'] = (
                            local_input_cursor,
                            local_input_cursor + self.structured_trunk_output_head_component_dim,
                        )
                        local_input_cursor += self.structured_trunk_output_head_component_dim

                if fusion_dim <= 0:
                    raise ValueError(
                        "Structured trunk output head requires at least one enabled fusion input."
                    )
                if local_input_cursor != fusion_dim:
                    raise ValueError(
                        "Structured trunk output head local slice bookkeeping mismatch: "
                        f"cursor={local_input_cursor}, fusion_dim={fusion_dim}"
                    )
                self.structured_trunk_output_head_fusion_dim = fusion_dim
                self.structured_trunk_output_head_hf_head_shared_input_dim = (
                    int(shared_input_dim)
                    if self.structured_trunk_output_head_hf_head_use_shared_raw
                    else 0
                )
                self.structured_trunk_output_head_hf_head_structure_raw_dim = (
                    int(structure_raw_dim)
                    if self.structured_trunk_output_head_hf_head_use_structure_raw
                    else 0
                )
                self.structured_trunk_output_head_hf_head_local_geometry_raw_dim = (
                    int(local_color_geometry_input_dim)
                    if self.structured_trunk_output_head_hf_head_use_local_geometry_raw
                    else 0
                )
                self.structured_trunk_output_head_hf_head_local_geometry_keys = list(
                    hf_head_local_geometry_keys
                    if self.structured_trunk_output_head_hf_head_use_local_geometry_raw
                    else []
                )

                head_out_dim = d_out if self.structured_trunk_output_head_mode == 'rgb' else (d_out + 1)
                self.structured_trunk_output_head_mlp = self._build_structured_trunk_head_mlp(
                    self.structured_trunk_output_head_cfg.get('mlp', None),
                    fusion_dim,
                    head_out_dim,
                )
                self.structured_trunk_output_head_gate_mlp = self._build_structured_trunk_head_mlp(
                    self.structured_trunk_output_head_cfg.get(
                        'gate_mlp',
                        self.structured_trunk_output_head_cfg.get('mlp', None),
                    ),
                    fusion_dim,
                    1,
                )
                if self.structured_trunk_output_head_dual_head_enable and self.structured_trunk_output_head_hf_head_enable:
                    hf_input_dim = 0
                    if self.structured_trunk_output_head_hf_head_use_output_fusion:
                        hf_input_dim += fusion_dim
                    hf_input_dim += self.structured_trunk_output_head_hf_head_shared_input_dim
                    hf_input_dim += self.structured_trunk_output_head_hf_head_structure_raw_dim
                    hf_input_dim += (
                        self.structured_trunk_output_head_hf_head_local_geometry_raw_dim
                        * len(self.structured_trunk_output_head_hf_head_local_geometry_keys)
                    )
                    if hf_input_dim <= 0:
                        raise ValueError(
                            "Structured trunk dual-head high-frequency branch requires at least one enabled input."
                        )
                    self.structured_trunk_output_head_hf_input_dim = hf_input_dim
                    hf_head_out_dim = (
                        d_out
                        if self.structured_trunk_output_head_hf_head_mode == 'rgb'
                        else (d_out + 1)
                    )
                    self.structured_trunk_output_head_hf_head_mlp = (
                        self._build_structured_trunk_head_mlp(
                            self.structured_trunk_output_head_hf_head_cfg.get(
                                'mlp',
                                self.structured_trunk_output_head_cfg.get('mlp', None),
                            ),
                            hf_input_dim,
                            hf_head_out_dim,
                        )
                    )
                    if not self.structured_trunk_output_head_hf_head_reuse_output_gate:
                        self.structured_trunk_output_head_hf_head_gate_mlp = (
                            self._build_structured_trunk_head_mlp(
                                self.structured_trunk_output_head_hf_head_cfg.get(
                                    'gate_mlp',
                                    self.structured_trunk_output_head_hf_head_cfg.get(
                                        'mlp',
                                        self.structured_trunk_output_head_cfg.get('mlp', None),
                                    ),
                                ),
                                hf_input_dim,
                                1,
                            )
                        )
        self.detail_residual_cfg = cfg.get('detail_residual', None)
        self.detail_residual_enable = bool(self.detail_residual_cfg.get('enable', False)) if self.detail_residual_cfg is not None else False
        self.detail_mlp = None
        self.detail_high_freq_cfg = None
        self.detail_high_freq_enable = False
        self.detail_high_freq_mlp = None
        self.detail_high_freq_gate_mlp = None
        self.detail_high_freq_context_proj = None
        self.detail_high_freq_carrier_proj = None
        self.detail_high_freq_luma_mlp = None
        self.detail_high_freq_face_mlp = None
        self.detail_high_freq_face_gate_mlp = None
        self.detail_high_freq_face_local_proj = None
        self.detail_scale_cfg = 1.0
        self.detail_max_residual = 0.0
        self.detail_high_freq_scale_cfg = 1.0
        self.detail_high_freq_max_residual = 0.0
        self.detail_high_freq_schedule_use_local_iteration = False
        self.detail_high_freq_use_input_context = True
        self.detail_high_freq_input_context_dim = 0
        self.detail_high_freq_use_canonical_xyz = True
        self.detail_high_freq_use_view_dir = True
        self.detail_high_freq_xyz_multires = 0
        self.detail_high_freq_xyz_include_input = True
        self.detail_high_freq_xyz_scale = 1.0
        self.detail_high_freq_view_multires = 0
        self.detail_high_freq_view_include_input = True
        self.detail_high_freq_view_scale = 1.0
        self.detail_high_freq_gate_bias = 0.0
        self.detail_high_freq_min_gate = 0.0
        self.detail_high_freq_chroma_center = False
        self.detail_high_freq_chroma_scale = 1.0
        self.detail_high_freq_luma_branch_cfg = None
        self.detail_high_freq_luma_branch_enable = False
        self.detail_high_freq_luma_scale = 0.0
        self.detail_high_freq_face_branch_cfg = None
        self.detail_high_freq_face_branch_enable = False
        self.detail_high_freq_face_scale = 0.0
        self.detail_high_freq_face_channels = 1
        self.detail_high_freq_face_gate_bias = 0.0
        self.detail_high_freq_face_min_gate = 0.0
        self.detail_high_freq_face_init_from = 'none'
        self.detail_high_freq_face_gate_init_from = 'none'
        self.detail_high_freq_face_init_missing_only = True
        self.detail_high_freq_face_local_cfg = None
        self.detail_high_freq_face_local_enable = False
        self.detail_high_freq_face_local_use_canonical_xyz = True
        self.detail_high_freq_face_local_use_view_dir = False
        self.detail_high_freq_face_local_use_radius = True
        self.detail_high_freq_face_local_xyz_multires = 0
        self.detail_high_freq_face_local_xyz_include_input = True
        self.detail_high_freq_face_local_xyz_scale = 1.0
        self.detail_high_freq_face_local_view_multires = 0
        self.detail_high_freq_face_local_view_include_input = True
        self.detail_high_freq_face_local_view_scale = 1.0
        self.detail_high_freq_face_local_radius_multires = 0
        self.detail_high_freq_face_local_radius_include_input = True
        self.detail_high_freq_face_local_radius_scale = 1.0
        self.detail_high_freq_face_local_inject_scale = 0.0
        self.detail_high_freq_face_local_region_cfg = None
        self.detail_high_freq_face_extra_local_cfgs = []
        self.detail_high_freq_face_extra_local_projs = nn.ModuleDict()
        self.detail_high_freq_point_gate_cfg = None
        self.detail_high_freq_point_gate_enable = False
        self.detail_high_freq_face_point_gate_cfg = None
        self.detail_high_freq_face_point_gate_enable = False
        self.detail_high_freq_inherit_point_gate = True
        self.detail_high_freq_structure_cfg = None
        self.detail_high_freq_structure_enable = False
        self.detail_high_freq_structure_detach = True
        self.detail_high_freq_structure_inject_scale = 0.0
        self.detail_high_freq_structure_features = []
        self.detail_high_freq_structure_proj = None
        self.detail_tiny_repair_scale_cfg = 1.0
        self.detail_high_freq_tiny_repair_scale_cfg = 1.0
        self.detail_high_freq_face_tiny_repair_scale_cfg = 1.0
        self.detail_high_freq_boundary_floor_cfg = None
        self.detail_high_freq_boundary_floor_enable = False
        self.detail_high_freq_boundary_floor_value_cfg = 0.0
        self.detail_high_freq_boundary_floor_threshold = 0.0
        self.detail_high_freq_boundary_floor_power = 1.0
        self.detail_high_freq_boundary_floor_max = 1.0
        self.detail_high_freq_boundary_floor_detach = True
        self.last_detail_scale = 0.0
        self.last_detail_schedule_iteration = 0
        self.last_detail_residual_abs_mean = None
        self.last_detail_tiny_repair_abs_mean = None
        self.last_detail_gate_mean = None
        self.last_detail_gate_fraction = None
        self.last_detail_high_freq_scale = 0.0
        self.last_detail_high_freq_residual_abs_mean = None
        self.last_detail_high_freq_gate_mean = None
        self.last_detail_high_freq_gate_fraction = None
        self.last_detail_high_freq_point_gate_mean = None
        self.last_detail_high_freq_point_gate_fraction = None
        self.last_detail_high_freq_carrier_abs_mean = None
        self.last_detail_high_freq_chroma_abs_mean = None
        self.last_detail_high_freq_luma_abs_mean = None
        self.last_detail_high_freq_face_abs_mean = None
        self.last_detail_high_freq_face_raw_abs_mean = None
        self.last_detail_high_freq_face_after_gate_abs_mean = None
        self.last_detail_high_freq_face_gate_mean = None
        self.last_detail_high_freq_face_gate_fraction = None
        self.last_detail_high_freq_face_point_gate_mean = None
        self.last_detail_high_freq_face_point_gate_fraction = None
        self.last_detail_high_freq_face_local_abs_mean = None
        self.last_detail_high_freq_face_local_raw_abs_mean = None
        self.last_detail_high_freq_face_extra_local_abs_mean = None
        self.last_detail_high_freq_face_extra_local_raw_abs_mean = None
        self.last_detail_high_freq_face_extra_local_gate_mean = None
        self.last_detail_high_freq_face_extra_local_debug = ''
        self.last_detail_high_freq_structure_abs_mean = None
        self.last_detail_high_freq_structure_raw_abs_mean = None
        self.last_detail_high_freq_structure_debug = ''
        self.last_detail_high_freq_boundary_floor_mean = None
        self.last_partial_load_init_events = []
        self._detail_schedule_local_iteration = None
        if self.detail_residual_enable:
            detail_mlp_cfg = self.detail_residual_cfg.get('mlp', cfg.mlp)
            detail_mlp_cfg = OmegaConf.create(OmegaConf.to_container(detail_mlp_cfg, resolve=True))
            detail_mlp_cfg.last_layer_init = bool(detail_mlp_cfg.get('last_layer_init', True))
            self.detail_mlp = VanillaCondMLP(d_in, 0, d_out, detail_mlp_cfg)
            self.detail_scale_cfg = self.detail_residual_cfg.get('scale', 1.0)
            self.detail_max_residual = float(self.detail_residual_cfg.get('max_residual', 0.35))
            self.detail_tiny_repair_scale_cfg = self.detail_residual_cfg.get(
                'tiny_repair_scale',
                self.detail_residual_cfg.get('repair_scale', 1.0),
            )
            self.detail_point_gate_cfg = self.detail_residual_cfg.get('point_gate', None)
            self.detail_point_gate_enable = bool(
                self.detail_point_gate_cfg.get('enable', False)
            ) if self.detail_point_gate_cfg is not None else False
            self.detail_schedule_use_local_iteration = bool(
                self.detail_residual_cfg.get('schedule_use_local_iteration', False)
            )
            self.detail_high_freq_cfg = self.detail_residual_cfg.get('high_frequency', None)
            self.detail_high_freq_enable = bool(
                self.detail_high_freq_cfg.get('enable', False)
            ) if self.detail_high_freq_cfg is not None else False
            if self.detail_high_freq_enable:
                self.detail_high_freq_scale_cfg = self.detail_high_freq_cfg.get('scale', 1.0)
                self.detail_high_freq_tiny_repair_scale_cfg = self.detail_high_freq_cfg.get(
                    'tiny_repair_scale',
                    self.detail_high_freq_cfg.get('repair_scale', self.detail_tiny_repair_scale_cfg),
                )
                self.detail_high_freq_max_residual = float(
                    self.detail_high_freq_cfg.get(
                        'max_residual',
                        min(max(self.detail_max_residual * 0.7, 0.06), 0.18),
                    )
                )
                self.detail_high_freq_schedule_use_local_iteration = bool(
                    self.detail_high_freq_cfg.get(
                        'schedule_use_local_iteration',
                        self.detail_schedule_use_local_iteration,
                    )
                )
                self.detail_high_freq_use_input_context = bool(
                    self.detail_high_freq_cfg.get('use_input_context', True)
                )
                self.detail_high_freq_input_context_dim = int(
                    self.detail_high_freq_cfg.get(
                        'input_context_dim',
                        self.detail_high_freq_cfg.get('context_dim', 32),
                    )
                )
                self.detail_high_freq_use_canonical_xyz = bool(
                    self.detail_high_freq_cfg.get('use_canonical_xyz', True)
                )
                self.detail_high_freq_use_view_dir = bool(
                    self.detail_high_freq_cfg.get('use_view_dir', True)
                )
                self.detail_high_freq_xyz_multires = int(
                    self.detail_high_freq_cfg.get('xyz_multires', 6)
                )
                self.detail_high_freq_xyz_include_input = bool(
                    self.detail_high_freq_cfg.get('xyz_include_input', True)
                )
                self.detail_high_freq_xyz_scale = float(
                    self.detail_high_freq_cfg.get('xyz_frequency_scale', 1.0)
                )
                self.detail_high_freq_view_multires = int(
                    self.detail_high_freq_cfg.get('view_multires', 4)
                )
                self.detail_high_freq_view_include_input = bool(
                    self.detail_high_freq_cfg.get('view_include_input', True)
                )
                self.detail_high_freq_view_scale = float(
                    self.detail_high_freq_cfg.get('view_frequency_scale', 1.0)
                )
                self.detail_high_freq_gate_bias = float(
                    self.detail_high_freq_cfg.get('gate_bias', 0.0)
                )
                self.detail_high_freq_min_gate = float(
                    self.detail_high_freq_cfg.get('min_gate', 0.0)
                )
                self.detail_high_freq_chroma_center = bool(
                    self.detail_high_freq_cfg.get('chroma_center', False)
                )
                self.detail_high_freq_chroma_scale = float(
                    self.detail_high_freq_cfg.get('chroma_scale', 1.0)
                )
                self.detail_high_freq_luma_branch_cfg = self.detail_high_freq_cfg.get('luma_branch', None)
                self.detail_high_freq_luma_branch_enable = bool(
                    self.detail_high_freq_luma_branch_cfg.get('enable', False)
                ) if self.detail_high_freq_luma_branch_cfg is not None else False
                if self.detail_high_freq_luma_branch_enable:
                    self.detail_high_freq_luma_scale = float(
                        self.detail_high_freq_luma_branch_cfg.get('scale', 0.75)
                    )
                self.detail_high_freq_face_branch_cfg = self.detail_high_freq_cfg.get('face_branch', None)
                self.detail_high_freq_face_branch_enable = bool(
                    self.detail_high_freq_face_branch_cfg.get('enable', False)
                ) if self.detail_high_freq_face_branch_cfg is not None else False
                if self.detail_high_freq_face_branch_enable:
                    self.detail_high_freq_face_scale = float(
                        self.detail_high_freq_face_branch_cfg.get('scale', 0.25)
                    )
                    self.detail_high_freq_face_tiny_repair_scale_cfg = (
                        self.detail_high_freq_face_branch_cfg.get(
                            'tiny_repair_scale',
                            self.detail_high_freq_face_branch_cfg.get(
                                'repair_scale',
                                self.detail_high_freq_tiny_repair_scale_cfg,
                            ),
                        )
                    )
                    self.detail_high_freq_face_channels = int(
                        self.detail_high_freq_face_branch_cfg.get('channels', 1)
                    )
                    self.detail_high_freq_face_gate_bias = float(
                        self.detail_high_freq_face_branch_cfg.get('gate_bias', -0.20)
                    )
                    self.detail_high_freq_face_min_gate = float(
                        self.detail_high_freq_face_branch_cfg.get('min_gate', 0.0)
                    )
                    self.detail_high_freq_face_init_from = str(
                        self.detail_high_freq_face_branch_cfg.get('init_from', 'none')
                    ).lower()
                    self.detail_high_freq_face_gate_init_from = str(
                        self.detail_high_freq_face_branch_cfg.get('gate_init_from', 'none')
                    ).lower()
                    self.detail_high_freq_face_init_missing_only = bool(
                        self.detail_high_freq_face_branch_cfg.get('init_missing_only', True)
                    )
                    self.detail_high_freq_face_local_cfg = self.detail_high_freq_face_branch_cfg.get(
                        'local_carrier',
                        None,
                    )
                    self.detail_high_freq_face_local_enable = bool(
                        self.detail_high_freq_face_local_cfg.get('enable', False)
                    ) if self.detail_high_freq_face_local_cfg is not None else False
                    if self.detail_high_freq_face_local_enable:
                        self.detail_high_freq_face_local_use_canonical_xyz = bool(
                            self.detail_high_freq_face_local_cfg.get('use_canonical_xyz', True)
                        )
                        self.detail_high_freq_face_local_use_view_dir = bool(
                            self.detail_high_freq_face_local_cfg.get('use_view_dir', False)
                        )
                        self.detail_high_freq_face_local_use_radius = bool(
                            self.detail_high_freq_face_local_cfg.get('use_radius', True)
                        )
                        self.detail_high_freq_face_local_xyz_multires = int(
                            self.detail_high_freq_face_local_cfg.get('xyz_multires', 8)
                        )
                        self.detail_high_freq_face_local_xyz_include_input = bool(
                            self.detail_high_freq_face_local_cfg.get('xyz_include_input', True)
                        )
                        self.detail_high_freq_face_local_xyz_scale = float(
                            self.detail_high_freq_face_local_cfg.get('xyz_frequency_scale', 1.0)
                        )
                        self.detail_high_freq_face_local_view_multires = int(
                            self.detail_high_freq_face_local_cfg.get('view_multires', 4)
                        )
                        self.detail_high_freq_face_local_view_include_input = bool(
                            self.detail_high_freq_face_local_cfg.get('view_include_input', True)
                        )
                        self.detail_high_freq_face_local_view_scale = float(
                            self.detail_high_freq_face_local_cfg.get('view_frequency_scale', 1.0)
                        )
                        self.detail_high_freq_face_local_radius_multires = int(
                            self.detail_high_freq_face_local_cfg.get('radius_multires', 4)
                        )
                        self.detail_high_freq_face_local_radius_include_input = bool(
                            self.detail_high_freq_face_local_cfg.get('radius_include_input', True)
                        )
                        self.detail_high_freq_face_local_radius_scale = float(
                            self.detail_high_freq_face_local_cfg.get('radius_frequency_scale', 1.0)
                        )
                        self.detail_high_freq_face_local_inject_scale = float(
                            self.detail_high_freq_face_local_cfg.get('inject_scale', 0.4)
                        )
                        face_local_region_cfg = self.detail_high_freq_face_local_cfg.get(
                            'region',
                            _default_face_local_region_cfg(),
                        )
                        if OmegaConf.is_config(face_local_region_cfg):
                            face_local_region_cfg = OmegaConf.to_container(
                                face_local_region_cfg,
                                resolve=True,
                            )
                        self.detail_high_freq_face_local_region_cfg = OmegaConf.create(
                            face_local_region_cfg
                        )
                    face_local_template = {
                        'enable': False,
                        'use_canonical_xyz': self.detail_high_freq_face_local_use_canonical_xyz,
                        'use_view_dir': self.detail_high_freq_face_local_use_view_dir,
                        'use_radius': self.detail_high_freq_face_local_use_radius,
                        'xyz_multires': self.detail_high_freq_face_local_xyz_multires,
                        'xyz_include_input': self.detail_high_freq_face_local_xyz_include_input,
                        'xyz_frequency_scale': self.detail_high_freq_face_local_xyz_scale,
                        'view_multires': self.detail_high_freq_face_local_view_multires,
                        'view_include_input': self.detail_high_freq_face_local_view_include_input,
                        'view_frequency_scale': self.detail_high_freq_face_local_view_scale,
                        'radius_multires': self.detail_high_freq_face_local_radius_multires,
                        'radius_include_input': self.detail_high_freq_face_local_radius_include_input,
                        'radius_frequency_scale': self.detail_high_freq_face_local_radius_scale,
                        'inject_scale': self.detail_high_freq_face_local_inject_scale,
                        'region': OmegaConf.to_container(
                            self.detail_high_freq_face_local_region_cfg,
                            resolve=True,
                        ) if self.detail_high_freq_face_local_region_cfg is not None else _default_face_local_region_cfg(),
                    }
                    extra_local_cfgs = _normalize_local_carrier_cfg_list(
                        self.detail_high_freq_face_branch_cfg.get('extra_local_carriers', None)
                    )
                    for local_cfg in extra_local_cfgs:
                        merged_cfg = OmegaConf.merge(
                            OmegaConf.create(face_local_template),
                            OmegaConf.create(OmegaConf.to_container(local_cfg, resolve=True)),
                        )
                        region_cfg = merged_cfg.get('region', face_local_template['region'])
                        if OmegaConf.is_config(region_cfg):
                            region_cfg = OmegaConf.to_container(region_cfg, resolve=True)
                        merged_cfg.region = OmegaConf.create(region_cfg)
                        self.detail_high_freq_face_extra_local_cfgs.append(merged_cfg)
                    self.detail_high_freq_face_point_gate_cfg = self.detail_high_freq_face_branch_cfg.get(
                        'point_gate',
                        None,
                    )
                    self.detail_high_freq_face_point_gate_enable = bool(
                        self.detail_high_freq_face_point_gate_cfg.get('enable', False)
                    ) if self.detail_high_freq_face_point_gate_cfg is not None else False
                self.detail_high_freq_point_gate_cfg = self.detail_high_freq_cfg.get('point_gate', None)
                self.detail_high_freq_point_gate_enable = bool(
                    self.detail_high_freq_point_gate_cfg.get('enable', False)
                ) if self.detail_high_freq_point_gate_cfg is not None else False
                self.detail_high_freq_inherit_point_gate = bool(
                    self.detail_high_freq_cfg.get('inherit_point_gate', True)
                )
                self.detail_high_freq_boundary_floor_cfg = self.detail_high_freq_cfg.get(
                    'boundary_floor',
                    None,
                )
                self.detail_high_freq_boundary_floor_enable = bool(
                    self.detail_high_freq_boundary_floor_cfg.get('enable', False)
                ) if self.detail_high_freq_boundary_floor_cfg is not None else False
                if self.detail_high_freq_boundary_floor_enable:
                    self.detail_high_freq_boundary_floor_value_cfg = (
                        self.detail_high_freq_boundary_floor_cfg.get('value', 0.0)
                    )
                    self.detail_high_freq_boundary_floor_threshold = float(
                        self.detail_high_freq_boundary_floor_cfg.get('threshold', 0.0)
                    )
                    self.detail_high_freq_boundary_floor_power = float(
                        self.detail_high_freq_boundary_floor_cfg.get('power', 1.0)
                    )
                    self.detail_high_freq_boundary_floor_max = float(
                        self.detail_high_freq_boundary_floor_cfg.get('max', 1.0)
                    )
                    self.detail_high_freq_boundary_floor_detach = bool(
                        self.detail_high_freq_boundary_floor_cfg.get('detach', True)
                    )
                self.detail_high_freq_structure_cfg = self.detail_high_freq_cfg.get(
                    'structure_carrier',
                    None,
                )
                self.detail_high_freq_structure_enable = bool(
                    self.detail_high_freq_structure_cfg.get('enable', False)
                ) if self.detail_high_freq_structure_cfg is not None else False
                if self.detail_high_freq_structure_enable:
                    self.detail_high_freq_structure_detach = bool(
                        self.detail_high_freq_structure_cfg.get('detach', True)
                    )
                    self.detail_high_freq_structure_inject_scale = float(
                        self.detail_high_freq_structure_cfg.get('inject_scale', 0.35)
                    )
                    self.detail_high_freq_structure_features = _normalize_structure_carrier_cfg_list(
                        self.detail_high_freq_structure_cfg.get('features', None),
                        default_detach=self.detail_high_freq_structure_detach,
                    )

                context_dim = 0
                if self.detail_high_freq_use_input_context:
                    if self.detail_high_freq_input_context_dim > 0:
                        self.detail_high_freq_context_proj = nn.Linear(d_in, self.detail_high_freq_input_context_dim)
                        nn.init.normal_(self.detail_high_freq_context_proj.weight, mean=0.0, std=0.02)
                        nn.init.constant_(self.detail_high_freq_context_proj.bias, 0.0)
                        context_dim = self.detail_high_freq_input_context_dim
                    else:
                        context_dim = d_in

                raw_carrier_dim = 0
                if self.detail_high_freq_use_canonical_xyz:
                    raw_carrier_dim += _encoded_feature_dim(
                        3,
                        self.detail_high_freq_xyz_multires,
                        include_input=self.detail_high_freq_xyz_include_input,
                    )
                if self.detail_high_freq_use_view_dir:
                    raw_carrier_dim += _encoded_feature_dim(
                        3,
                        self.detail_high_freq_view_multires,
                        include_input=self.detail_high_freq_view_include_input,
                    )

                if raw_carrier_dim > 0 and context_dim > 0:
                    self.detail_high_freq_carrier_proj = nn.Linear(raw_carrier_dim, context_dim)
                    nn.init.normal_(self.detail_high_freq_carrier_proj.weight, mean=0.0, std=0.02)
                    nn.init.constant_(self.detail_high_freq_carrier_proj.bias, 0.0)

                high_freq_in_dim = raw_carrier_dim + context_dim
                if raw_carrier_dim > 0 and context_dim > 0:
                    high_freq_in_dim += context_dim
                if high_freq_in_dim <= 0:
                    high_freq_in_dim = d_in
                if self.detail_high_freq_structure_enable:
                    structure_raw_dim = self._encoded_structure_carrier_dim(
                        self.detail_high_freq_structure_features
                    )
                    if structure_raw_dim > 0:
                        self.detail_high_freq_structure_proj = nn.Linear(
                            structure_raw_dim,
                            high_freq_in_dim,
                        )
                        # Keep resume safe: start as a no-op and let the structure
                        # bridge grow only if optimization finds it useful.
                        nn.init.constant_(self.detail_high_freq_structure_proj.weight, 0.0)
                        nn.init.constant_(self.detail_high_freq_structure_proj.bias, 0.0)
                    else:
                        self.detail_high_freq_structure_enable = False

                face_local_raw_dim = 0
                if self.detail_high_freq_face_local_enable:
                    if self.detail_high_freq_face_local_use_canonical_xyz:
                        face_local_raw_dim += _encoded_feature_dim(
                            3,
                            self.detail_high_freq_face_local_xyz_multires,
                            include_input=self.detail_high_freq_face_local_xyz_include_input,
                        )
                    if self.detail_high_freq_face_local_use_view_dir:
                        face_local_raw_dim += _encoded_feature_dim(
                            3,
                            self.detail_high_freq_face_local_view_multires,
                            include_input=self.detail_high_freq_face_local_view_include_input,
                        )
                    if self.detail_high_freq_face_local_use_radius:
                        face_local_raw_dim += _encoded_feature_dim(
                            1,
                            self.detail_high_freq_face_local_radius_multires,
                            include_input=self.detail_high_freq_face_local_radius_include_input,
                        )
                    if face_local_raw_dim > 0:
                        self.detail_high_freq_face_local_proj = nn.Linear(
                            face_local_raw_dim,
                            high_freq_in_dim,
                        )
                        # Start as a no-op so v108b can hot-start cleanly and
                        # learn local face injection progressively.
                        nn.init.constant_(self.detail_high_freq_face_local_proj.weight, 0.0)
                        nn.init.constant_(self.detail_high_freq_face_local_proj.bias, 0.0)
                for local_cfg in self.detail_high_freq_face_extra_local_cfgs:
                    if not bool(local_cfg.get('enable', False)):
                        continue
                    extra_local_raw_dim = self._encoded_local_carrier_dim(local_cfg)
                    if extra_local_raw_dim <= 0:
                        continue
                    local_proj = nn.Linear(extra_local_raw_dim, high_freq_in_dim)
                    nn.init.constant_(local_proj.weight, 0.0)
                    nn.init.constant_(local_proj.bias, 0.0)
                    self.detail_high_freq_face_extra_local_projs[str(local_cfg.get('_key'))] = local_proj

                detail_high_freq_mlp_cfg = self.detail_high_freq_cfg.get('mlp', detail_mlp_cfg)
                detail_high_freq_mlp_cfg = OmegaConf.create(
                    OmegaConf.to_container(detail_high_freq_mlp_cfg, resolve=True)
                )
                detail_high_freq_mlp_cfg.last_layer_init = bool(
                    detail_high_freq_mlp_cfg.get('last_layer_init', True)
                )
                self.detail_high_freq_mlp = VanillaCondMLP(
                    high_freq_in_dim,
                    0,
                    d_out,
                    detail_high_freq_mlp_cfg,
                )

                gate_mlp_cfg = self.detail_high_freq_cfg.get('gate_mlp', detail_high_freq_mlp_cfg)
                gate_mlp_cfg = OmegaConf.create(OmegaConf.to_container(gate_mlp_cfg, resolve=True))
                gate_mlp_cfg.last_layer_init = bool(gate_mlp_cfg.get('last_layer_init', True))
                self.detail_high_freq_gate_mlp = VanillaCondMLP(
                    high_freq_in_dim,
                    0,
                    1,
                    gate_mlp_cfg,
                )
                if self.detail_high_freq_luma_branch_enable:
                    luma_mlp_cfg = self.detail_high_freq_luma_branch_cfg.get(
                        'mlp',
                        self.detail_high_freq_cfg.get('mlp', detail_mlp_cfg),
                    )
                    luma_mlp_cfg = OmegaConf.create(OmegaConf.to_container(luma_mlp_cfg, resolve=True))
                    luma_mlp_cfg.last_layer_init = bool(luma_mlp_cfg.get('last_layer_init', True))
                    self.detail_high_freq_luma_mlp = VanillaCondMLP(
                        high_freq_in_dim,
                        0,
                        1,
                        luma_mlp_cfg,
                    )
                if self.detail_high_freq_face_branch_enable:
                    face_mlp_cfg = self.detail_high_freq_face_branch_cfg.get(
                        'mlp',
                        self.detail_high_freq_luma_branch_cfg.get('mlp', detail_high_freq_mlp_cfg)
                        if self.detail_high_freq_luma_branch_cfg is not None else detail_high_freq_mlp_cfg,
                    )
                    face_mlp_cfg = OmegaConf.create(OmegaConf.to_container(face_mlp_cfg, resolve=True))
                    face_mlp_cfg.last_layer_init = bool(face_mlp_cfg.get('last_layer_init', True))
                    self.detail_high_freq_face_mlp = VanillaCondMLP(
                        high_freq_in_dim,
                        0,
                        self.detail_high_freq_face_channels,
                        face_mlp_cfg,
                    )
                    face_gate_mlp_cfg = self.detail_high_freq_face_branch_cfg.get(
                        'gate_mlp',
                        gate_mlp_cfg,
                    )
                    face_gate_mlp_cfg = OmegaConf.create(
                        OmegaConf.to_container(face_gate_mlp_cfg, resolve=True)
                    )
                    face_gate_mlp_cfg.last_layer_init = bool(
                        face_gate_mlp_cfg.get('last_layer_init', True)
                    )
                    self.detail_high_freq_face_gate_mlp = VanillaCondMLP(
                        high_freq_in_dim,
                        0,
                        1,
                        face_gate_mlp_cfg,
                    )
        else:
            self.detail_point_gate_cfg = None
            self.detail_point_gate_enable = False
            self.detail_schedule_use_local_iteration = False
        self.color_activation = nn.Sigmoid()

    def set_schedule_context(self, local_iteration=None, schedule_iteration=None):
        del schedule_iteration
        self._detail_schedule_local_iteration = None if local_iteration is None else int(local_iteration)

    def _resolve_detail_schedule_iteration(self, iteration):
        if self.detail_schedule_use_local_iteration and self._detail_schedule_local_iteration is not None:
            return self._detail_schedule_local_iteration
        return int(iteration)

    def _resolve_detail_high_freq_schedule_iteration(self, iteration):
        if self.detail_high_freq_schedule_use_local_iteration and self._detail_schedule_local_iteration is not None:
            return self._detail_schedule_local_iteration
        return int(iteration)

    def _build_point_gate(self, gaussians, point_gate_cfg):
        if point_gate_cfg is None or not bool(point_gate_cfg.get('enable', False)):
            return None
        if not torch.is_tensor(gaussians.get_xyz) or gaussians.get_xyz.numel() <= 0:
            return None

        point_count = int(gaussians.get_xyz.shape[0])
        device = gaussians.get_xyz.device
        gate_terms = []

        joint_mask = _weighted_id_mask(
            getattr(gaussians, 'binding_dominant_joint', None),
            point_gate_cfg.get(
                'joint_id_weights',
                point_gate_cfg.get('joint_ids', None),
            ),
        )
        if joint_mask is not None:
            gate_terms.append(joint_mask)

        semantic_mask = _weighted_id_mask(
            getattr(gaussians, 'binding_compact_semantic_ids', None),
            point_gate_cfg.get(
                'semantic_id_weights',
                point_gate_cfg.get('semantic_ids', None),
            ),
        )
        if semantic_mask is not None:
            gate_terms.append(semantic_mask)

        semantic_name_mask = _weighted_name_mask(
            getattr(gaussians, 'binding_compact_semantic_ids', None),
            getattr(gaussians, 'binding_compact_semantic_names', None),
            point_gate_cfg.get(
                'semantic_name_weights',
                point_gate_cfg.get('semantic_names', None),
            ),
        )
        if semantic_name_mask is not None:
            gate_terms.append(semantic_name_mask)

        layer_mask = _weighted_id_mask(
            getattr(gaussians, 'binding_layer_ids', None),
            point_gate_cfg.get(
                'layer_id_weights',
                point_gate_cfg.get('layer_ids', None),
            ),
        )
        if layer_mask is not None:
            gate_terms.append(layer_mask)

        gate = _combine_gate_terms(
            gate_terms,
            mode=point_gate_cfg.get(
                'combine_mode',
                point_gate_cfg.get('mode', 'max'),
            ),
        )
        if gate is None:
            if bool(point_gate_cfg.get('fallback_to_full', True)):
                return torch.ones((point_count,), device=device, dtype=torch.float32)
            return torch.zeros((point_count,), device=device, dtype=torch.float32)

        exclude_joint_mask = _weighted_id_mask(
            getattr(gaussians, 'binding_dominant_joint', None),
            point_gate_cfg.get(
                'exclude_joint_id_weights',
                point_gate_cfg.get('exclude_joint_ids', None),
            ),
        )
        if exclude_joint_mask is not None:
            gate = gate * (1.0 - exclude_joint_mask.clamp(0.0, 1.0))

        exclude_semantic_mask = _weighted_id_mask(
            getattr(gaussians, 'binding_compact_semantic_ids', None),
            point_gate_cfg.get(
                'exclude_semantic_id_weights',
                point_gate_cfg.get('exclude_semantic_ids', None),
            ),
        )
        if exclude_semantic_mask is not None:
            gate = gate * (1.0 - exclude_semantic_mask.clamp(0.0, 1.0))

        exclude_semantic_name_mask = _weighted_name_mask(
            getattr(gaussians, 'binding_compact_semantic_ids', None),
            getattr(gaussians, 'binding_compact_semantic_names', None),
            point_gate_cfg.get(
                'exclude_semantic_name_weights',
                point_gate_cfg.get('exclude_semantic_names', None),
            ),
        )
        if exclude_semantic_name_mask is not None:
            gate = gate * (1.0 - exclude_semantic_name_mask.clamp(0.0, 1.0))

        exclude_layer_mask = _weighted_id_mask(
            getattr(gaussians, 'binding_layer_ids', None),
            point_gate_cfg.get(
                'exclude_layer_id_weights',
                point_gate_cfg.get('exclude_layer_ids', None),
            ),
        )
        if exclude_layer_mask is not None:
            gate = gate * (1.0 - exclude_layer_mask.clamp(0.0, 1.0))

        min_gate = float(point_gate_cfg.get('min_gate', 0.0))
        if min_gate > 0.0:
            gate = torch.where(gate > 0.0, gate.clamp(min=min_gate), gate)

        gate_scale = float(point_gate_cfg.get('scale', 1.0))
        if gate_scale != 1.0:
            gate = gate * gate_scale

        return gate.clamp(0.0, 1.0)

    def _build_detail_point_gate(self, gaussians):
        if not self.detail_point_gate_enable or self.detail_point_gate_cfg is None:
            return None
        return self._build_point_gate(gaussians, self.detail_point_gate_cfg)

    def _build_detail_high_freq_point_gate(self, gaussians, base_gate=None):
        point_gate = None
        if self.detail_high_freq_inherit_point_gate and torch.is_tensor(base_gate):
            point_gate = base_gate

        extra_gate = self._build_point_gate(gaussians, self.detail_high_freq_point_gate_cfg)
        if point_gate is None:
            return extra_gate
        if extra_gate is None:
            return point_gate

        combine_mode = str(
            self.detail_high_freq_cfg.get('point_gate_combine_mode', 'mul')
        ).lower() if self.detail_high_freq_cfg is not None else 'mul'
        return _combine_gate_terms([point_gate, extra_gate], mode=combine_mode)

    def _get_binding_boundary_score(self, gaussians, template, detach=True):
        if gaussians is None or not torch.is_tensor(template):
            return None

        boundary_score = None
        for attr_name in (
            'binding_boundary_mixed_score',
            'binding_boundary_live_score',
            'binding_boundary_score',
        ):
            candidate = getattr(gaussians, attr_name, None)
            if torch.is_tensor(candidate):
                boundary_score = candidate
                break
        if not torch.is_tensor(boundary_score):
            return None
        if boundary_score.shape[0] != template.shape[0]:
            return None
        if boundary_score.dim() == 1:
            boundary_score = boundary_score.unsqueeze(-1)
        else:
            boundary_score = _flatten_feature_tensor(boundary_score)[:, :1]
        boundary_score = boundary_score.to(
            device=template.device,
            dtype=template.dtype,
        ).clamp(0.0, 1.0)
        if detach:
            boundary_score = boundary_score.detach()
        return boundary_score

    def _build_boundary_focus(
        self,
        boundary_score,
        threshold=0.0,
        power=1.0,
        min_focus=0.0,
        max_focus=1.0,
    ):
        if not torch.is_tensor(boundary_score):
            return None

        threshold = float(threshold)
        focus = (
            (boundary_score - threshold)
            / max(1.0 - threshold, 1.0e-6)
        ).clamp(0.0, 1.0)
        power = max(float(power), 1.0e-6)
        if power != 1.0:
            focus = focus.pow(power)

        min_focus = float(min_focus)
        if min_focus > 0.0:
            focus = torch.where(focus > 0.0, focus.clamp(min=min_focus), focus)
        max_focus = float(max_focus)
        if max_focus > 0.0:
            focus = focus.clamp(max=max_focus)
        return focus.clamp(0.0, 1.0)

    def _get_structured_trunk_owner_boundary_focus_score(self, gaussians, template, boundary_score=None):
        source = str(
            self.structured_trunk_output_head_local_color_owner_head_boundary_focus_source
        ).lower()
        if source in ('tag', 'tags', 'boundary_tag', 'boundary_tags'):
            tag_score = None
            if gaussians is not None and hasattr(gaussians, 'get_boundary_tags'):
                tag_score = gaussians.get_boundary_tags()
            elif gaussians is not None:
                tag_score = getattr(gaussians, '_boundary_tag', None)
            if torch.is_tensor(tag_score) and torch.is_tensor(template) and tag_score.shape[0] == template.shape[0]:
                if tag_score.dim() == 1:
                    tag_score = tag_score.unsqueeze(-1)
                else:
                    tag_score = _flatten_feature_tensor(tag_score)[:, :1]
                return tag_score.to(device=template.device, dtype=template.dtype).clamp(0.0, 1.0).detach()
            if source in ('tag', 'tags', 'boundary_tag', 'boundary_tags'):
                return None
        if source in ('tag_or_score', 'boundary_tag_or_score'):
            tag_score = None
            if gaussians is not None and hasattr(gaussians, 'get_boundary_tags'):
                tag_score = gaussians.get_boundary_tags()
            elif gaussians is not None:
                tag_score = getattr(gaussians, '_boundary_tag', None)
            if torch.is_tensor(tag_score) and torch.is_tensor(template) and tag_score.shape[0] == template.shape[0]:
                if tag_score.dim() == 1:
                    tag_score = tag_score.unsqueeze(-1)
                else:
                    tag_score = _flatten_feature_tensor(tag_score)[:, :1]
                return tag_score.to(device=template.device, dtype=template.dtype).clamp(0.0, 1.0).detach()
        return boundary_score

    def _build_detail_high_freq_boundary_floor(self, gaussians, template, iteration=0):
        zero_scalar = template.new_tensor(0.0)
        if not self.detail_high_freq_boundary_floor_enable:
            return None, zero_scalar

        floor_value = _resolve_scheduled_scalar(
            iteration,
            self.detail_high_freq_boundary_floor_value_cfg,
            default=0.0,
        )
        if floor_value <= 0.0:
            return None, zero_scalar

        boundary_score = self._get_binding_boundary_score(
            gaussians,
            template,
            detach=self.detail_high_freq_boundary_floor_detach,
        )
        if not torch.is_tensor(boundary_score):
            return None, zero_scalar

        boundary_floor = self._build_boundary_focus(
            boundary_score,
            threshold=self.detail_high_freq_boundary_floor_threshold,
            power=self.detail_high_freq_boundary_floor_power,
            min_focus=0.0,
            max_focus=self.detail_high_freq_boundary_floor_max,
        )
        if not torch.is_tensor(boundary_floor):
            return None, zero_scalar

        boundary_floor = boundary_floor * float(floor_value)
        if self.detail_high_freq_boundary_floor_max > 0.0:
            boundary_floor = boundary_floor.clamp(
                max=float(self.detail_high_freq_boundary_floor_max)
            )
        return boundary_floor.clamp(0.0, 1.0), boundary_floor.detach().mean()

    def _view_direction(self, gaussians, camera, iteration=0):
        n_points = gaussians.get_xyz.shape[0]
        dir_pp = gaussians.get_xyz - camera.camera_center.repeat(n_points, 1)
        if self.cano_view_dir:
            T_fwd = gaussians.fwd_transform
            R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
            dir_pp = torch.matmul(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
            view_noise_scale = _resolve_scheduled_scalar(iteration, self.cfg.get('view_noise', 0.))
            if self.training and view_noise_scale > 0.:
                view_noise = torch.tensor(
                    augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
                    dtype=torch.float32,
                    device=dir_pp.device,
                ).transpose(0, 1)
                dir_pp = torch.matmul(dir_pp, view_noise)
        return dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-12)

    def _normalized_canonical_xyz(self, gaussians):
        canonical_xyz = getattr(gaussians, 'canonical_xyz', gaussians.get_xyz)
        aabb = self.metadata.get("aabb", None)
        if aabb is not None:
            canonical_xyz = aabb.normalize(canonical_xyz, sym=True)
        return canonical_xyz.clamp(-1.0, 1.0)

    def _encoded_local_carrier_dim(self, local_cfg):
        if local_cfg is None:
            return 0

        raw_dim = 0
        if bool(local_cfg.get('use_canonical_xyz', True)):
            raw_dim += _encoded_feature_dim(
                3,
                int(local_cfg.get('xyz_multires', 8)),
                include_input=bool(local_cfg.get('xyz_include_input', True)),
            )
        if bool(local_cfg.get('use_view_dir', False)):
            raw_dim += _encoded_feature_dim(
                3,
                int(local_cfg.get('view_multires', 4)),
                include_input=bool(local_cfg.get('view_include_input', True)),
            )
        if bool(local_cfg.get('use_radius', True)):
            raw_dim += _encoded_feature_dim(
                1,
                int(local_cfg.get('radius_multires', 4)),
                include_input=bool(local_cfg.get('radius_include_input', True)),
            )
        return raw_dim

    def _encoded_local_color_geometry_dim(self):
        raw_dim = 0
        if self.structured_trunk_output_head_local_color_use_current_xyz_feature:
            raw_dim += _encoded_feature_dim(
                3,
                self.structured_trunk_output_head_local_color_current_xyz_multires,
                include_input=self.structured_trunk_output_head_local_color_current_xyz_include_input,
            )
        if self.structured_trunk_output_head_local_color_use_current_radius_feature:
            raw_dim += _encoded_feature_dim(
                1,
                self.structured_trunk_output_head_local_color_current_radius_multires,
                include_input=self.structured_trunk_output_head_local_color_current_radius_include_input,
            )
        return raw_dim

    def _build_structured_trunk_residual_mlp(self, residual_cfg, dim_in, dim_out):
        if residual_cfg is None or dim_in <= 0 or dim_out <= 0:
            return None, 1.0
        if not bool(residual_cfg.get('enable', False)):
            return None, 1.0

        base_cfg = _clone_cfg_node(self.cfg.mlp)
        override_cfg = _clone_cfg_node(residual_cfg.get('mlp', residual_cfg))
        for meta_key in ('enable', 'scale', 'mlp'):
            if meta_key in override_cfg:
                del override_cfg[meta_key]

        mlp_cfg = OmegaConf.merge(base_cfg, override_cfg)
        mlp_cfg.multires = int(override_cfg.get('multires', 0))
        mlp_cfg.skip_in = list(override_cfg.get('skip_in', []))
        mlp_cfg.cond_in = list(override_cfg.get('cond_in', []))
        mlp_cfg.last_layer_init = bool(override_cfg.get('last_layer_init', True))
        return VanillaCondMLP(dim_in, 0, dim_out, mlp_cfg), float(
            residual_cfg.get('scale', 1.0)
        )

    def _build_feature_fusion_proj(self, dim_in, dim_out):
        if dim_in <= 0 or dim_out <= 0:
            return None
        proj = nn.Linear(dim_in, dim_out)
        nn.init.normal_(proj.weight, mean=0.0, std=0.02)
        nn.init.constant_(proj.bias, 0.0)
        return proj

    def _build_structured_trunk_head_mlp(self, mlp_cfg_override, dim_in, dim_out):
        base_cfg = _clone_cfg_node(self.cfg.mlp)
        override_cfg = _clone_cfg_node(mlp_cfg_override)
        mlp_cfg = OmegaConf.merge(base_cfg, override_cfg)
        mlp_cfg.multires = int(mlp_cfg.get('multires', 0))
        mlp_cfg.skip_in = list(mlp_cfg.get('skip_in', []))
        mlp_cfg.cond_in = list(mlp_cfg.get('cond_in', []))
        mlp_cfg.last_layer_init = bool(mlp_cfg.get('last_layer_init', True))
        return VanillaCondMLP(dim_in, 0, dim_out, mlp_cfg)

    def _decode_structured_trunk_head_color(
        self,
        head_logits,
        mode,
        compose_mode,
        scale,
        max_residual,
        chroma_center,
        band_luma_scale,
        band_chroma_scale,
    ):
        if compose_mode == 'residual':
            head_amplitude = max_residual * scale
            if mode == 'band':
                head_chroma = torch.tanh(head_logits[:, :3])
                if chroma_center:
                    head_chroma = head_chroma - head_chroma.mean(dim=-1, keepdim=True)
                head_color = head_chroma * (head_amplitude * band_chroma_scale)
                head_luma = torch.tanh(head_logits[:, 3:4]) * (
                    head_amplitude * band_luma_scale
                )
                return head_color + head_luma.expand(-1, 3)
            return torch.tanh(head_logits) * head_amplitude

        if mode == 'band':
            head_chroma = head_logits[:, :3]
            if chroma_center:
                head_chroma = head_chroma - head_chroma.mean(dim=-1, keepdim=True)
            head_color = head_chroma * (band_chroma_scale * scale)
            head_luma = head_logits[:, 3:4] * (band_luma_scale * scale)
            return head_color + head_luma.expand(-1, 3)
        return head_logits * scale

    def _compute_structured_trunk_head_gate(
        self,
        output,
        head_inp,
        gate_mlp,
        gate_gain,
        gate_bias,
        min_gate,
        max_gate=1.0,
        gate_boost=None,
    ):
        if gate_mlp is None:
            return output.new_ones((output.shape[0], 1)), output.new_tensor(0.0)

        gate_logits = gate_mlp(head_inp) * gate_gain + gate_bias
        gate_boost_mean = output.new_tensor(0.0)
        if torch.is_tensor(gate_boost):
            gate_logits = gate_logits + gate_boost
            gate_boost_mean = gate_boost.detach().mean()
        gate = torch.sigmoid(gate_logits)
        if min_gate > 0.0:
            gate = gate * (1.0 - min_gate) + min_gate
        if 0.0 < max_gate < 1.0:
            gate = gate.clamp(max=max(float(max_gate), float(min_gate)))
        return gate, gate_boost_mean

    def _get_structured_trunk_output_head_local_color_source_columns(self, local_key):
        local_input_slices = self.structured_trunk_output_head_local_input_slices.get(local_key, None)
        global_input_slices = self.structured_trunk_output_head_global_input_slices
        if not local_input_slices and not global_input_slices:
            return []

        source_columns = []
        ordered_features = (
            ('base', self.structured_trunk_output_head_local_color_use_base_feature),
            ('shared', self.structured_trunk_output_head_local_color_use_shared_feature),
            ('structure', self.structured_trunk_output_head_local_color_use_structure_feature),
            ('gate', self.structured_trunk_output_head_local_color_use_gate_feature),
            ('gated', self.structured_trunk_output_head_local_color_use_gated_feature),
            ('raw', self.structured_trunk_output_head_local_color_use_raw_feature),
        )
        for feature_name, feature_enable in ordered_features:
            if not feature_enable:
                continue
            bounds = None
            if feature_name in ('base', 'shared', 'structure'):
                bounds = global_input_slices.get(feature_name, None)
            elif local_input_slices is not None:
                bounds = local_input_slices.get(feature_name, None)
            if bounds is None:
                continue
            source_columns.extend(range(int(bounds[0]), int(bounds[1])))
        return source_columns

    def _get_structured_trunk_hf_head_source_columns(self):
        if not bool(self.structured_trunk_output_head_hf_head_use_output_fusion):
            return []
        fusion_dim = int(self.structured_trunk_output_head_fusion_dim)
        if fusion_dim <= 0:
            return []
        return list(range(fusion_dim))

    def _init_structured_trunk_local_color_from_output_head(self, local_key):
        if not self.structured_trunk_output_head_local_color_init_from_output_head:
            return False, 'disabled'
        if self.structured_trunk_output_head_mlp is None:
            return False, 'missing_output_head'
        if local_key not in self.structured_trunk_output_head_local_color_mlps:
            return False, 'missing_local_color_mlp'

        source_columns = self._get_structured_trunk_output_head_local_color_source_columns(local_key)
        if len(source_columns) <= 0:
            return False, 'no_source_columns'

        copied, reason = _copy_mlp_parameters_with_input_subset(
            self.structured_trunk_output_head_mlp,
            self.structured_trunk_output_head_local_color_mlps[local_key],
            source_columns,
            output_scale=self.structured_trunk_output_head_local_color_init_scale,
        )
        if not copied:
            return False, reason
        return True, (
            "output_head_slice"
            f"(cols={len(source_columns)},scale={self.structured_trunk_output_head_local_color_init_scale:.2f})"
        )

    def _init_structured_trunk_owner_head_from_local_color(self, local_key):
        if not self.structured_trunk_output_head_local_color_owner_head_enable:
            return False, 'owner_head_disabled'
        if not self.structured_trunk_output_head_local_color_owner_head_init_from_local_color:
            return False, 'disabled'
        if local_key not in self.structured_trunk_output_head_local_color_mlps:
            return False, 'missing_local_color_mlp'
        if local_key not in self.structured_trunk_output_head_local_color_owner_head_mlps:
            return False, 'missing_owner_head_mlp'

        local_color_input_dim = int(
            self.structured_trunk_output_head_local_color_input_dims.get(local_key, 0)
        )
        if local_color_input_dim <= 0:
            return False, 'missing_local_color_input_dim'
        if not self.structured_trunk_output_head_local_color_owner_head_use_local_color_input:
            return False, 'owner_head_without_local_color_input'

        copied, reason = _copy_mlp_parameters_with_input_subset(
            self.structured_trunk_output_head_local_color_mlps[local_key],
            self.structured_trunk_output_head_local_color_owner_head_mlps[local_key],
            list(range(local_color_input_dim)),
            output_scale=self.structured_trunk_output_head_local_color_owner_head_init_scale,
        )
        if not copied:
            return False, reason
        return True, (
            "local_color_copy"
            f"(cols={local_color_input_dim},scale={self.structured_trunk_output_head_local_color_owner_head_init_scale:.2f})"
        )

    def _init_structured_trunk_owner_gate_from_output_gate(self, local_key):
        if not self.structured_trunk_output_head_local_color_owner_head_enable:
            return False, 'owner_head_disabled'
        if self.structured_trunk_output_head_gate_mlp is None:
            return False, 'missing_output_gate'
        if local_key not in self.structured_trunk_output_head_local_color_owner_head_gate_mlps:
            return False, 'missing_owner_gate'
        if not self.structured_trunk_output_head_local_color_owner_head_use_local_color_input:
            return False, 'owner_head_without_local_color_input'

        source_columns = self._get_structured_trunk_output_head_local_color_source_columns(local_key)
        if len(source_columns) <= 0:
            return False, 'no_source_columns'
        copied, reason = _copy_mlp_parameters_with_input_subset(
            self.structured_trunk_output_head_gate_mlp,
            self.structured_trunk_output_head_local_color_owner_head_gate_mlps[local_key],
            source_columns,
            output_scale=1.0,
        )
        if not copied:
            return False, reason
        return True, f"output_gate_subset(cols={len(source_columns)})"

    def _init_structured_trunk_owner_boundary_head_from_local_color(self, local_key):
        if not self.structured_trunk_output_head_local_color_owner_head_boundary_enable:
            return False, 'boundary_head_disabled'
        if not self.structured_trunk_output_head_local_color_owner_head_boundary_init_from_local_color:
            return False, 'disabled'
        if local_key not in self.structured_trunk_output_head_local_color_mlps:
            return False, 'missing_local_color_mlp'
        if local_key not in self.structured_trunk_output_head_local_color_owner_head_boundary_mlps:
            return False, 'missing_boundary_head_mlp'

        local_color_input_dim = int(
            self.structured_trunk_output_head_local_color_input_dims.get(local_key, 0)
        )
        if local_color_input_dim <= 0:
            return False, 'missing_local_color_input_dim'
        if not self.structured_trunk_output_head_local_color_owner_head_boundary_use_local_color_input:
            return False, 'boundary_head_without_local_color_input'

        copied, reason = _copy_mlp_parameters_with_input_subset(
            self.structured_trunk_output_head_local_color_mlps[local_key],
            self.structured_trunk_output_head_local_color_owner_head_boundary_mlps[local_key],
            list(range(local_color_input_dim)),
            output_scale=self.structured_trunk_output_head_local_color_owner_head_boundary_init_scale,
        )
        if not copied:
            return False, reason
        return True, (
            "local_color_copy"
            f"(cols={local_color_input_dim},scale={self.structured_trunk_output_head_local_color_owner_head_boundary_init_scale:.2f})"
        )

    def _init_structured_trunk_owner_boundary_gate_from_output_gate(self, local_key):
        if not self.structured_trunk_output_head_local_color_owner_head_boundary_enable:
            return False, 'boundary_head_disabled'
        if self.structured_trunk_output_head_gate_mlp is None:
            return False, 'missing_output_gate'
        if local_key not in self.structured_trunk_output_head_local_color_owner_head_boundary_gate_mlps:
            return False, 'missing_boundary_gate'
        if not self.structured_trunk_output_head_local_color_owner_head_boundary_use_local_color_input:
            return False, 'boundary_head_without_local_color_input'

        source_columns = self._get_structured_trunk_output_head_local_color_source_columns(local_key)
        if len(source_columns) <= 0:
            return False, 'no_source_columns'
        copied, reason = _copy_mlp_parameters_with_input_subset(
            self.structured_trunk_output_head_gate_mlp,
            self.structured_trunk_output_head_local_color_owner_head_boundary_gate_mlps[local_key],
            source_columns,
            output_scale=1.0,
        )
        if not copied:
            return False, reason
        return True, f"output_gate_subset(cols={len(source_columns)})"

    def _init_structured_trunk_hf_head_from_output_head(self):
        if not self.structured_trunk_output_head_dual_head_enable:
            return False, 'dual_head_disabled'
        if not self.structured_trunk_output_head_hf_head_init_from_output_head:
            return False, 'disabled'
        if self.structured_trunk_output_head_mlp is None:
            return False, 'missing_output_head'
        if self.structured_trunk_output_head_hf_head_mlp is None:
            return False, 'missing_hf_head'

        source_columns = self._get_structured_trunk_hf_head_source_columns()
        if source_columns:
            copied, reason = _copy_mlp_parameters_with_input_subset(
                self.structured_trunk_output_head_mlp,
                self.structured_trunk_output_head_hf_head_mlp,
                source_columns,
                output_scale=self.structured_trunk_output_head_hf_head_init_scale,
            )
        else:
            copied, reason = _copy_mlp_parameters(
                self.structured_trunk_output_head_mlp,
                self.structured_trunk_output_head_hf_head_mlp,
            )
        if not copied:
            return False, reason

        hf_layers = _iter_named_linear_layers(self.structured_trunk_output_head_hf_head_mlp)
        if (
            not source_columns
            and hf_layers
            and abs(float(self.structured_trunk_output_head_hf_head_init_scale) - 1.0) > 1.0e-8
        ):
            with torch.no_grad():
                hf_last = hf_layers[-1][1]
                hf_last.weight.data.mul_(float(self.structured_trunk_output_head_hf_head_init_scale))
                hf_last.bias.data.mul_(float(self.structured_trunk_output_head_hf_head_init_scale))
        if source_columns:
            return True, (
                "output_head_subset"
                f"(cols={len(source_columns)},scale={self.structured_trunk_output_head_hf_head_init_scale:.2f})"
            )
        return True, (
            "output_head_copy"
            f"(scale={self.structured_trunk_output_head_hf_head_init_scale:.2f})"
        )

    def _init_structured_trunk_hf_gate_from_output_gate(self):
        if not self.structured_trunk_output_head_dual_head_enable:
            return False, 'dual_head_disabled'
        if self.structured_trunk_output_head_hf_head_reuse_output_gate:
            return False, 'reuse_output_gate'
        if self.structured_trunk_output_head_gate_mlp is None:
            return False, 'missing_output_gate'
        if self.structured_trunk_output_head_hf_head_gate_mlp is None:
            return False, 'missing_hf_gate'

        source_columns = self._get_structured_trunk_hf_head_source_columns()
        if source_columns:
            copied, reason = _copy_mlp_parameters_with_input_subset(
                self.structured_trunk_output_head_gate_mlp,
                self.structured_trunk_output_head_hf_head_gate_mlp,
                source_columns,
                output_scale=1.0,
            )
        else:
            copied, reason = _copy_mlp_parameters(
                self.structured_trunk_output_head_gate_mlp,
                self.structured_trunk_output_head_hf_head_gate_mlp,
            )
        if not copied:
            return False, reason
        if source_columns:
            return True, f"output_gate_subset(cols={len(source_columns)})"
        return True, "output_gate_copy"

    def _encoded_structure_carrier_dim(self, structure_cfgs):
        if structure_cfgs is None:
            return 0

        raw_dim = 0
        for feature_cfg in structure_cfgs:
            if feature_cfg is None:
                continue
            raw_dim += _encoded_feature_dim(
                int(feature_cfg.get('dims', 0)),
                int(feature_cfg.get('multires', 0)),
                include_input=bool(feature_cfg.get('include_input', True)),
            )
        return raw_dim

    def _compose_detail_high_frequency_structure_delta(self, gaussians, base_input):
        if (
            not self.detail_high_freq_structure_enable
            or self.detail_high_freq_structure_proj is None
            or not torch.is_tensor(base_input)
        ):
            return None, None, ''

        raw_terms = []
        debug_parts = []
        for feature_cfg in self.detail_high_freq_structure_features:
            attr_name = str(feature_cfg.get('attr', '')).strip()
            if not attr_name:
                continue
            feature_value = getattr(gaussians, attr_name, None)
            if not torch.is_tensor(feature_value) or feature_value.numel() <= 0:
                continue

            if feature_value.ndim == 1:
                feature_value = feature_value.unsqueeze(-1)
            else:
                feature_value = _flatten_feature_tensor(feature_value)
            feature_value = feature_value.to(
                device=base_input.device,
                dtype=base_input.dtype,
            )
            if bool(feature_cfg.get('detach', self.detail_high_freq_structure_detach)):
                feature_value = feature_value.detach()
            if bool(feature_cfg.get('abs', False)):
                feature_value = feature_value.abs()

            center = float(feature_cfg.get('center', 0.0))
            if center != 0.0:
                feature_value = feature_value - center
            scale = float(feature_cfg.get('scale', 1.0))
            if scale != 1.0:
                feature_value = feature_value * scale
            clamp_value = feature_cfg.get('clamp', None)
            if clamp_value is not None:
                clamp_value = float(clamp_value)
                feature_value = feature_value.clamp(-clamp_value, clamp_value)

            encoded_value = _fourier_encode(
                feature_value,
                int(feature_cfg.get('multires', 0)),
                include_input=bool(feature_cfg.get('include_input', True)),
                frequency_scale=float(feature_cfg.get('frequency_scale', 1.0)),
            )
            weight = float(feature_cfg.get('weight', 1.0))
            if weight != 1.0:
                encoded_value = encoded_value * weight

            raw_terms.append(encoded_value)
            debug_parts.append(
                f"{feature_cfg.get('name', attr_name)}={float(encoded_value.detach().abs().mean().item()):.5f}"
            )

        if len(raw_terms) <= 0:
            return None, None, ''

        raw_structure = torch.cat(raw_terms, dim=1)
        structure_delta = torch.tanh(self.detail_high_freq_structure_proj(raw_structure))
        inject_scale = float(self.detail_high_freq_structure_inject_scale)
        if inject_scale != 1.0:
            structure_delta = structure_delta * inject_scale
        return raw_structure, structure_delta, ' | '.join(debug_parts)

    def _compose_structured_trunk_structure_delta(self, gaussians, base_input):
        if (
            not self.structured_trunk_structure_enable
            or self.structured_trunk_structure_proj is None
            or not torch.is_tensor(base_input)
        ):
            return None, None, ''

        raw_terms = []
        debug_parts = []
        for feature_cfg in self.structured_trunk_structure_features:
            attr_name = str(feature_cfg.get('attr', '')).strip()
            if not attr_name:
                continue
            feature_value = getattr(gaussians, attr_name, None)
            if not torch.is_tensor(feature_value) or feature_value.numel() <= 0:
                continue

            if feature_value.ndim == 1:
                feature_value = feature_value.unsqueeze(-1)
            else:
                feature_value = _flatten_feature_tensor(feature_value)
            feature_value = feature_value.to(
                device=base_input.device,
                dtype=base_input.dtype,
            )
            if bool(feature_cfg.get('detach', self.structured_trunk_structure_detach)):
                feature_value = feature_value.detach()
            if bool(feature_cfg.get('abs', False)):
                feature_value = feature_value.abs()

            center = float(feature_cfg.get('center', 0.0))
            if center != 0.0:
                feature_value = feature_value - center
            scale = float(feature_cfg.get('scale', 1.0))
            if scale != 1.0:
                feature_value = feature_value * scale
            clamp_value = feature_cfg.get('clamp', None)
            if clamp_value is not None:
                clamp_value = float(clamp_value)
                feature_value = feature_value.clamp(-clamp_value, clamp_value)

            encoded_value = _fourier_encode(
                feature_value,
                int(feature_cfg.get('multires', 0)),
                include_input=bool(feature_cfg.get('include_input', True)),
                frequency_scale=float(feature_cfg.get('frequency_scale', 1.0)),
            )
            weight = float(feature_cfg.get('weight', 1.0))
            if weight != 1.0:
                encoded_value = encoded_value * weight

            raw_terms.append(encoded_value)
            debug_parts.append(
                f"{feature_cfg.get('name', attr_name)}={float(encoded_value.detach().abs().mean().item()):.5f}"
            )

        if len(raw_terms) <= 0:
            return None, None, ''

        raw_structure = torch.cat(raw_terms, dim=1)
        structure_delta = torch.tanh(self.structured_trunk_structure_proj(raw_structure))
        inject_scale = float(self.structured_trunk_structure_inject_scale)
        if inject_scale != 1.0:
            structure_delta = structure_delta * inject_scale
        return raw_structure, structure_delta, ' | '.join(debug_parts)

    def _normalized_region_xyz(self, xyz, gaussians, region_cfg):
        if not torch.is_tensor(xyz):
            return None, None
        if region_cfg is None:
            return xyz, None

        region_gate = self._build_point_gate(gaussians, region_cfg)
        if not torch.is_tensor(region_gate):
            return xyz, None

        region_gate = region_gate.to(
            device=xyz.device,
            dtype=xyz.dtype,
        ).clamp(0.0, 1.0)
        region_gate_mean = region_gate.mean()
        min_gate_mean = float(region_cfg.get('min_gate_mean', 0.0))
        if (
            min_gate_mean > 0.0
            and float(region_gate_mean.detach().item()) < min_gate_mean
            and bool(region_cfg.get('fallback_to_global', False))
        ):
            return xyz, region_gate_mean

        region_mass = region_gate.sum().clamp_min(1e-6)
        center = (xyz * region_gate.unsqueeze(-1)).sum(dim=0, keepdim=True) / region_mass
        centered_xyz = xyz - center
        second_moment = (
            centered_xyz.square() * region_gate.unsqueeze(-1)
        ).sum(dim=0) / region_mass
        region_scale = torch.sqrt(second_moment.mean().clamp_min(1e-4)).clamp_min(0.05)
        return (centered_xyz / region_scale).clamp(-3.0, 3.0), region_gate_mean

    def _normalized_region_local_xyz(self, gaussians, region_cfg):
        canonical_xyz = self._normalized_canonical_xyz(gaussians)
        return self._normalized_region_xyz(canonical_xyz, gaussians, region_cfg)

    def _normalized_current_region_xyz(self, gaussians, region_cfg):
        return self._normalized_region_xyz(gaussians.get_xyz, gaussians, region_cfg)

    def _normalized_face_local_xyz(self, gaussians):
        local_xyz, _ = self._normalized_region_local_xyz(
            gaussians,
            self.detail_high_freq_face_local_region_cfg,
        )
        return local_xyz

    def _compose_detail_high_frequency_input(self, gaussians, camera, inp, iteration=0):
        carrier_terms = []
        if self.detail_high_freq_use_canonical_xyz:
            carrier_terms.append(
                _fourier_encode(
                    self._normalized_canonical_xyz(gaussians),
                    self.detail_high_freq_xyz_multires,
                    include_input=self.detail_high_freq_xyz_include_input,
                    frequency_scale=self.detail_high_freq_xyz_scale,
                )
            )
        if self.detail_high_freq_use_view_dir:
            carrier_terms.append(
                _fourier_encode(
                    self._view_direction(gaussians, camera, iteration=iteration),
                    self.detail_high_freq_view_multires,
                    include_input=self.detail_high_freq_view_include_input,
                    frequency_scale=self.detail_high_freq_view_scale,
                )
            )

        carrier_terms = [term for term in carrier_terms if torch.is_tensor(term) and term.shape[1] > 0]
        raw_carrier = torch.cat(carrier_terms, dim=1) if carrier_terms else None

        context = None
        if self.detail_high_freq_use_input_context:
            if self.detail_high_freq_context_proj is not None:
                context = torch.tanh(self.detail_high_freq_context_proj(inp))
            else:
                context = inp

        high_freq_parts = []
        carrier_monitor = None
        if context is not None:
            high_freq_parts.append(context)
        if raw_carrier is not None:
            high_freq_parts.append(raw_carrier)
        if context is not None and raw_carrier is not None:
            if self.detail_high_freq_carrier_proj is not None:
                modulated_carrier = context * torch.tanh(self.detail_high_freq_carrier_proj(raw_carrier))
            else:
                shared_dim = min(context.shape[1], raw_carrier.shape[1])
                modulated_carrier = context[:, :shared_dim] * raw_carrier[:, :shared_dim]
            high_freq_parts.append(modulated_carrier)
            carrier_monitor = modulated_carrier
        elif raw_carrier is not None:
            carrier_monitor = raw_carrier
        elif context is not None:
            carrier_monitor = context

        if len(high_freq_parts) <= 0:
            high_freq_input = inp
        else:
            high_freq_input = torch.cat(high_freq_parts, dim=1)

        structure_raw = None
        structure_delta = None
        structure_debug = ''
        if self.detail_high_freq_structure_enable:
            structure_raw, structure_delta, structure_debug = self._compose_detail_high_frequency_structure_delta(
                gaussians,
                high_freq_input,
            )
            if torch.is_tensor(structure_delta):
                high_freq_input = high_freq_input + structure_delta

        if carrier_monitor is None:
            carrier_monitor = high_freq_input
        return high_freq_input, carrier_monitor, structure_raw, structure_delta, structure_debug

    def _compose_single_region_local_delta(self, gaussians, camera, local_cfg, local_proj, iteration=0):
        if local_cfg is None or local_proj is None:
            return None, None, None

        local_terms = []
        local_xyz = None
        region_gate_mean = None
        if bool(local_cfg.get('use_canonical_xyz', True)) or bool(local_cfg.get('use_radius', True)):
            local_xyz, region_gate_mean = self._normalized_region_local_xyz(
                gaussians,
                local_cfg.get('region', None),
            )
        if bool(local_cfg.get('use_canonical_xyz', True)) and torch.is_tensor(local_xyz):
            local_terms.append(
                _fourier_encode(
                    local_xyz,
                    int(local_cfg.get('xyz_multires', 8)),
                    include_input=bool(local_cfg.get('xyz_include_input', True)),
                    frequency_scale=float(local_cfg.get('xyz_frequency_scale', 1.0)),
                )
            )
        if bool(local_cfg.get('use_view_dir', False)):
            local_terms.append(
                _fourier_encode(
                    self._view_direction(gaussians, camera, iteration=iteration),
                    int(local_cfg.get('view_multires', 4)),
                    include_input=bool(local_cfg.get('view_include_input', True)),
                    frequency_scale=float(local_cfg.get('view_frequency_scale', 1.0)),
                )
            )
        if bool(local_cfg.get('use_radius', True)) and torch.is_tensor(local_xyz):
            local_radius = local_xyz.norm(dim=1, keepdim=True).clamp(0.0, 3.0)
            local_terms.append(
                _fourier_encode(
                    local_radius,
                    int(local_cfg.get('radius_multires', 4)),
                    include_input=bool(local_cfg.get('radius_include_input', True)),
                    frequency_scale=float(local_cfg.get('radius_frequency_scale', 1.0)),
                )
            )

        local_terms = [term for term in local_terms if torch.is_tensor(term) and term.shape[1] > 0]
        if len(local_terms) <= 0:
            return None, None, region_gate_mean

        local_raw = torch.cat(local_terms, dim=1)
        local_delta = torch.tanh(local_proj(local_raw))
        inject_scale = float(local_cfg.get('inject_scale', 1.0))
        if inject_scale != 1.0:
            local_delta = local_delta * inject_scale
        return local_raw, local_delta, region_gate_mean

    def _compose_structured_trunk_local_geometry_input(self, gaussians, local_cfg):
        if local_cfg is None:
            return None, None

        geometry_terms = []
        local_current_xyz = None
        region_gate_mean = None
        if (
            self.structured_trunk_output_head_local_color_use_current_xyz_feature
            or self.structured_trunk_output_head_local_color_use_current_radius_feature
        ):
            local_current_xyz, region_gate_mean = self._normalized_current_region_xyz(
                gaussians,
                local_cfg.get('region', None),
            )
        if (
            self.structured_trunk_output_head_local_color_use_current_xyz_feature
            and torch.is_tensor(local_current_xyz)
        ):
            geometry_terms.append(
                _fourier_encode(
                    local_current_xyz,
                    self.structured_trunk_output_head_local_color_current_xyz_multires,
                    include_input=self.structured_trunk_output_head_local_color_current_xyz_include_input,
                    frequency_scale=self.structured_trunk_output_head_local_color_current_xyz_scale,
                )
            )
        if (
            self.structured_trunk_output_head_local_color_use_current_radius_feature
            and torch.is_tensor(local_current_xyz)
        ):
            local_current_radius = local_current_xyz.norm(dim=1, keepdim=True).clamp(0.0, 3.0)
            geometry_terms.append(
                _fourier_encode(
                    local_current_radius,
                    self.structured_trunk_output_head_local_color_current_radius_multires,
                    include_input=self.structured_trunk_output_head_local_color_current_radius_include_input,
                    frequency_scale=self.structured_trunk_output_head_local_color_current_radius_scale,
                )
            )

        geometry_terms = [
            term for term in geometry_terms if torch.is_tensor(term) and term.shape[1] > 0
        ]
        if len(geometry_terms) <= 0:
            return None, region_gate_mean
        return torch.cat(geometry_terms, dim=1), region_gate_mean

    def _compose_detail_high_frequency_face_input(self, gaussians, camera, base_input, iteration=0):
        face_local_raw = None
        face_local_delta = None
        extra_local_raw_means = []
        extra_local_delta_means = []
        extra_local_gate_means = []
        extra_local_debug_parts = []

        if not torch.is_tensor(base_input):
            return base_input, face_local_raw, face_local_delta

        combined_input = base_input
        if self.detail_high_freq_face_local_enable and self.detail_high_freq_face_local_proj is not None:
            face_local_raw, face_local_delta, _ = self._compose_single_region_local_delta(
                gaussians,
                camera,
                self.detail_high_freq_face_local_cfg,
                self.detail_high_freq_face_local_proj,
                iteration=iteration,
            )
            if torch.is_tensor(face_local_delta):
                combined_input = combined_input + face_local_delta

        for local_cfg in self.detail_high_freq_face_extra_local_cfgs:
            if not bool(local_cfg.get('enable', False)):
                continue
            local_key = str(local_cfg.get('_key', ''))
            if local_key not in self.detail_high_freq_face_extra_local_projs:
                continue
            local_raw, local_delta, region_gate_mean = self._compose_single_region_local_delta(
                gaussians,
                camera,
                local_cfg,
                self.detail_high_freq_face_extra_local_projs[local_key],
                iteration=iteration,
            )
            if torch.is_tensor(local_delta):
                combined_input = combined_input + local_delta
                extra_local_delta_means.append(local_delta.detach().abs().mean())
            if torch.is_tensor(local_raw):
                extra_local_raw_means.append(local_raw.detach().abs().mean())
            gate_mean_value = 0.0 if region_gate_mean is None else float(region_gate_mean.detach().item())
            if region_gate_mean is not None:
                extra_local_gate_means.append(region_gate_mean.detach())
            extra_local_debug_parts.append(
                (
                    f"{local_cfg.get('name', local_key)}:"
                    f"gate={gate_mean_value:.4f},"
                    f"raw={0.0 if local_raw is None else float(local_raw.detach().abs().mean().item()):.5f},"
                    f"delta={0.0 if local_delta is None else float(local_delta.detach().abs().mean().item()):.5f}"
                )
            )

        if extra_local_raw_means:
            self.last_detail_high_freq_face_extra_local_raw_abs_mean = torch.stack(
                extra_local_raw_means,
                dim=0,
            ).mean()
        if extra_local_delta_means:
            self.last_detail_high_freq_face_extra_local_abs_mean = torch.stack(
                extra_local_delta_means,
                dim=0,
            ).mean()
        if extra_local_gate_means:
            self.last_detail_high_freq_face_extra_local_gate_mean = torch.stack(
                extra_local_gate_means,
                dim=0,
            ).mean()
        self.last_detail_high_freq_face_extra_local_debug = ' | '.join(extra_local_debug_parts)
        return combined_input, face_local_raw, face_local_delta

    def _compose_structured_trunk_shared_input(self, gaussians, camera, base_input, iteration=0):
        if not torch.is_tensor(base_input):
            return None, None

        carrier_terms = []
        if self.structured_trunk_use_canonical_xyz:
            carrier_terms.append(
                _fourier_encode(
                    self._normalized_canonical_xyz(gaussians),
                    self.structured_trunk_xyz_multires,
                    include_input=self.structured_trunk_xyz_include_input,
                    frequency_scale=self.structured_trunk_xyz_scale,
                )
            )
        if self.structured_trunk_use_view_dir:
            carrier_terms.append(
                _fourier_encode(
                    self._view_direction(gaussians, camera, iteration=iteration),
                    self.structured_trunk_view_multires,
                    include_input=self.structured_trunk_view_include_input,
                    frequency_scale=self.structured_trunk_view_scale,
                )
            )

        carrier_terms = [term for term in carrier_terms if torch.is_tensor(term) and term.shape[1] > 0]
        raw_carrier = torch.cat(carrier_terms, dim=1) if carrier_terms else None

        context = None
        if self.structured_trunk_use_input_context:
            if self.structured_trunk_context_proj is not None:
                context = torch.tanh(self.structured_trunk_context_proj(base_input))
            else:
                context = base_input

        modulated_carrier = None
        shared_parts = []
        if context is not None:
            shared_parts.append(context)
        if raw_carrier is not None:
            shared_parts.append(raw_carrier)
        if (
            self.structured_trunk_use_modulated_carrier
            and context is not None
            and raw_carrier is not None
        ):
            if self.structured_trunk_carrier_proj is not None:
                modulated_carrier = context * torch.tanh(
                    self.structured_trunk_carrier_proj(raw_carrier)
                )
            else:
                shared_dim = min(context.shape[1], raw_carrier.shape[1])
                modulated_carrier = context[:, :shared_dim] * raw_carrier[:, :shared_dim]
            shared_parts.append(modulated_carrier)

        shared_input = None
        if shared_parts:
            shared_input = torch.cat(shared_parts, dim=1)
        carrier_monitor = modulated_carrier
        if carrier_monitor is None:
            carrier_monitor = raw_carrier
        if carrier_monitor is None:
            carrier_monitor = context
        return shared_input, carrier_monitor

    def _compose_structured_trunk_delta(self, gaussians, camera, base_input, iteration=0):
        if not self.structured_trunk_enable or not torch.is_tensor(base_input):
            return None, ''

        trunk_delta = torch.zeros_like(base_input)
        has_delta = False
        debug_parts = []

        shared_input = None
        carrier_monitor = None
        if self.structured_trunk_shared_proj is not None or self.structured_trunk_shared_mlp is not None:
            shared_input, carrier_monitor = self._compose_structured_trunk_shared_input(
                gaussians,
                camera,
                base_input,
                iteration=iteration,
            )
            if torch.is_tensor(shared_input):
                shared_delta = None
                if self.structured_trunk_shared_proj is not None:
                    shared_delta = torch.tanh(self.structured_trunk_shared_proj(shared_input))
                shared_residual_delta = None
                if self.structured_trunk_shared_mlp is not None:
                    shared_residual_delta = torch.tanh(
                        self.structured_trunk_shared_mlp(shared_input)
                    )
                    shared_residual_scale = float(self.structured_trunk_shared_mlp_scale)
                    if shared_residual_scale != 1.0:
                        shared_residual_delta = shared_residual_delta * shared_residual_scale
                    self.last_structured_trunk_shared_residual_abs_mean = (
                        shared_residual_delta.detach().abs().mean()
                    )
                if not torch.is_tensor(shared_delta):
                    shared_delta = torch.zeros_like(base_input)
                if torch.is_tensor(shared_residual_delta):
                    shared_delta = shared_delta + shared_residual_delta
                inject_scale = float(self.structured_trunk_inject_scale)
                if inject_scale != 1.0:
                    shared_delta = shared_delta * inject_scale
                trunk_delta = trunk_delta + shared_delta
                has_delta = True
                self.last_structured_trunk_shared_abs_mean = shared_delta.detach().abs().mean()
                debug_parts.append(
                    f"shared={float(self.last_structured_trunk_shared_abs_mean.item()):.5f}"
                )
                if self.last_structured_trunk_shared_residual_abs_mean is not None:
                    debug_parts.append(
                        f"shared_res={float(self.last_structured_trunk_shared_residual_abs_mean.item()):.5f}"
                    )

        if torch.is_tensor(carrier_monitor):
            self.last_structured_trunk_carrier_abs_mean = carrier_monitor.detach().abs().mean()
            debug_parts.append(
                f"carrier={float(self.last_structured_trunk_carrier_abs_mean.item()):.5f}"
            )

        structure_raw = None
        structure_delta = None
        structure_debug = ''
        if self.structured_trunk_structure_enable:
            structure_raw, structure_delta, structure_debug = self._compose_structured_trunk_structure_delta(
                gaussians,
                base_input,
            )
            if torch.is_tensor(structure_raw):
                self.last_structured_trunk_structure_raw_abs_mean = (
                    structure_raw.detach().abs().mean()
                )
            structure_residual_delta = None
            if (
                torch.is_tensor(structure_raw)
                and self.structured_trunk_structure_mlp is not None
            ):
                structure_residual_delta = torch.tanh(
                    self.structured_trunk_structure_mlp(structure_raw)
                )
                structure_residual_scale = float(self.structured_trunk_structure_mlp_scale)
                if structure_residual_scale != 1.0:
                    structure_residual_delta = structure_residual_delta * structure_residual_scale
                self.last_structured_trunk_structure_residual_abs_mean = (
                    structure_residual_delta.detach().abs().mean()
                )
                if structure_debug:
                    structure_debug = (
                        f"{structure_debug} | "
                        f"mlp={float(self.last_structured_trunk_structure_residual_abs_mean.item()):.5f}"
                    )
                else:
                    structure_debug = (
                        f"mlp={float(self.last_structured_trunk_structure_residual_abs_mean.item()):.5f}"
                    )
            if not torch.is_tensor(structure_delta):
                structure_delta = None
            if torch.is_tensor(structure_residual_delta):
                if structure_delta is None:
                    structure_delta = structure_residual_delta
                else:
                    structure_delta = structure_delta + structure_residual_delta
            if torch.is_tensor(structure_delta):
                self.last_structured_trunk_structure_abs_mean = (
                    structure_delta.detach().abs().mean()
                )
                trunk_delta = trunk_delta + structure_delta
                has_delta = True
            if structure_debug:
                debug_parts.append(structure_debug)

        local_raw_means = []
        local_delta_means = []
        local_gate_means = []
        local_residual_means = []
        for local_cfg in self.structured_trunk_local_cfgs:
            if not bool(local_cfg.get('enable', False)):
                continue
            local_key = str(local_cfg.get('_key', ''))
            if local_key not in self.structured_trunk_local_projs:
                continue

            local_raw, local_delta, region_gate_mean = self._compose_single_region_local_delta(
                gaussians,
                camera,
                local_cfg,
                self.structured_trunk_local_projs[local_key],
                iteration=iteration,
            )
            local_residual_delta = None
            if (
                torch.is_tensor(local_raw)
                and local_key in self.structured_trunk_local_mlps
            ):
                local_residual_delta = torch.tanh(
                    self.structured_trunk_local_mlps[local_key](local_raw)
                )
                local_residual_scale = float(
                    local_cfg.get('mlp_scale', self.structured_trunk_local_mlp_scale)
                )
                if local_residual_scale != 1.0:
                    local_residual_delta = local_residual_delta * local_residual_scale
                local_residual_means.append(local_residual_delta.detach().abs().mean())
                if torch.is_tensor(local_delta):
                    local_delta = local_delta + local_residual_delta
                else:
                    local_delta = local_residual_delta
            region_gate = self._build_point_gate(gaussians, local_cfg.get('region', None))
            if torch.is_tensor(region_gate):
                region_gate = region_gate.to(
                    device=base_input.device,
                    dtype=base_input.dtype,
                ).clamp(0.0, 1.0)

            if torch.is_tensor(local_raw):
                local_raw_means.append(local_raw.detach().abs().mean())
            if torch.is_tensor(local_delta):
                gate_to_region = bool(
                    local_cfg.get('gate_to_region', self.structured_trunk_local_gate_to_region)
                )
                if gate_to_region and torch.is_tensor(region_gate):
                    local_delta = local_delta * region_gate.unsqueeze(-1)
                trunk_delta = trunk_delta + local_delta
                has_delta = True
                local_delta_means.append(local_delta.detach().abs().mean())
            if region_gate_mean is not None:
                local_gate_means.append(region_gate_mean.detach())
            debug_parts.append(
                (
                    f"{local_cfg.get('name', local_key)}:"
                    f"gate={0.0 if region_gate_mean is None else float(region_gate_mean.detach().item()):.4f},"
                    f"raw={0.0 if local_raw is None else float(local_raw.detach().abs().mean().item()):.5f},"
                    f"res={0.0 if local_residual_delta is None else float(local_residual_delta.detach().abs().mean().item()):.5f},"
                    f"delta={0.0 if local_delta is None else float(local_delta.detach().abs().mean().item()):.5f}"
                )
            )

        if local_raw_means:
            self.last_structured_trunk_local_raw_abs_mean = torch.stack(
                local_raw_means,
                dim=0,
            ).mean()
        if local_delta_means:
            self.last_structured_trunk_local_abs_mean = torch.stack(
                local_delta_means,
                dim=0,
            ).mean()
        if local_gate_means:
            self.last_structured_trunk_local_gate_mean = torch.stack(
                local_gate_means,
                dim=0,
            ).mean()
        if local_residual_means:
            self.last_structured_trunk_local_residual_abs_mean = torch.stack(
                local_residual_means,
                dim=0,
            ).mean()

        if not has_delta:
            return None, ' | '.join(debug_parts)
        return trunk_delta, ' | '.join(debug_parts)

    def _compose_structured_trunk_output_head_input(self, gaussians, camera, base_input, iteration=0):
        if (
            not self.structured_trunk_output_head_enable
            or not torch.is_tensor(base_input)
        ):
            return base_input, '', {}

        fusion_parts = []
        deferred_local_raw_fusion_parts = []
        debug_parts = []
        local_gate_means = []
        local_feature_means = []
        local_raw_feature_means = []
        base_feature = None
        shared_feature = None
        structure_feature = None
        local_color_total = None
        local_color_outputs = {}
        local_owner_input_outputs = {}
        local_geometry_raw_outputs = {}
        local_color_means = []
        local_gate_boost_terms = []
        hf_extra_parts = []
        hf_debug_parts = []
        region_gate_outputs = {}
        region_support_outputs = {}
        local_color_scale = _resolve_scheduled_scalar(
            iteration,
            self.structured_trunk_output_head_local_color_scale_cfg,
            default=1.0,
        )
        region_support_sources = set()
        if self.structured_trunk_output_head_region_support_enable:
            region_support_source = str(
                self.structured_trunk_output_head_region_support_source or 'hybrid'
            ).lower()
            normalized_support_source = (
                region_support_source
                .replace('+', ',')
                .replace('|', ',')
                .replace('/', ',')
            )
            region_support_sources = {
                token.strip()
                for token in normalized_support_source.split(',')
                if token.strip()
            }
            if not region_support_sources:
                region_support_sources = {'hybrid'}

        if self.structured_trunk_output_head_use_base_input:
            base_feature = base_input
            if self.structured_trunk_output_head_base_proj is not None:
                base_feature = torch.tanh(self.structured_trunk_output_head_base_proj(base_input))
            fusion_parts.append(base_feature)
            debug_parts.append(
                f"base={float(base_feature.detach().abs().mean().item()):.5f}"
            )

        shared_input = None
        carrier_monitor = None
        if (
            self.structured_trunk_output_head_shared_fusion_proj is not None
            or self.structured_trunk_output_head_hf_head_use_shared_raw
        ):
            shared_input, carrier_monitor = self._compose_structured_trunk_shared_input(
                gaussians,
                camera,
                base_input,
                iteration=iteration,
            )
            if (
                torch.is_tensor(shared_input)
                and self.structured_trunk_output_head_shared_fusion_proj is not None
            ):
                shared_feature = torch.tanh(
                    self.structured_trunk_output_head_shared_fusion_proj(shared_input)
                )
                fusion_parts.append(shared_feature)
                self.last_structured_trunk_shared_abs_mean = shared_feature.detach().abs().mean()
                debug_parts.append(
                    f"shared_fuse={float(self.last_structured_trunk_shared_abs_mean.item()):.5f}"
                )
            if self.structured_trunk_output_head_hf_head_use_shared_raw:
                if torch.is_tensor(shared_input):
                    hf_extra_parts.append(shared_input)
                    hf_debug_parts.append(
                        f"shared_raw={float(shared_input.detach().abs().mean().item()):.5f}"
                    )
                elif self.structured_trunk_output_head_hf_head_shared_input_dim > 0:
                    hf_extra_parts.append(
                        base_input.new_zeros(
                            (
                                base_input.shape[0],
                                self.structured_trunk_output_head_hf_head_shared_input_dim,
                            )
                        )
                    )
                    hf_debug_parts.append("shared_raw=0.00000")
        if torch.is_tensor(carrier_monitor):
            self.last_structured_trunk_carrier_abs_mean = carrier_monitor.detach().abs().mean()
            debug_parts.append(
                f"carrier={float(self.last_structured_trunk_carrier_abs_mean.item()):.5f}"
            )

        if (
            self.structured_trunk_output_head_structure_fusion_proj is not None
            or self.structured_trunk_output_head_hf_head_use_structure_raw
        ):
            structure_raw, _structure_delta, structure_debug = self._compose_structured_trunk_structure_delta(
                gaussians,
                base_input,
            )
            if (
                torch.is_tensor(structure_raw)
                and self.structured_trunk_output_head_structure_fusion_proj is not None
            ):
                structure_feature = torch.tanh(
                    self.structured_trunk_output_head_structure_fusion_proj(structure_raw)
                )
                fusion_parts.append(structure_feature)
                self.last_structured_trunk_structure_raw_abs_mean = (
                    structure_raw.detach().abs().mean()
                )
                self.last_structured_trunk_structure_abs_mean = (
                    structure_feature.detach().abs().mean()
                )
                if structure_debug:
                    debug_parts.append(structure_debug)
                debug_parts.append(
                    f"struct_fuse={float(self.last_structured_trunk_structure_abs_mean.item()):.5f}"
                )
            if self.structured_trunk_output_head_hf_head_use_structure_raw:
                if torch.is_tensor(structure_raw):
                    hf_extra_parts.append(structure_raw)
                    hf_debug_parts.append(
                        f"struct_raw={float(structure_raw.detach().abs().mean().item()):.5f}"
                    )
                elif self.structured_trunk_output_head_hf_head_structure_raw_dim > 0:
                    hf_extra_parts.append(
                        base_input.new_zeros(
                            (
                                base_input.shape[0],
                                self.structured_trunk_output_head_hf_head_structure_raw_dim,
                            )
                        )
                    )
                    hf_debug_parts.append("struct_raw=0.00000")

        for local_cfg in self.structured_trunk_local_cfgs:
            if not bool(local_cfg.get('enable', False)):
                continue
            local_key = str(local_cfg.get('_key', ''))
            if local_key not in self.structured_trunk_output_head_local_fusion_projs:
                continue
            region_name = str(local_cfg.get('name', local_key) or local_key)
            local_raw, _local_delta, region_gate_mean = self._compose_single_region_local_delta(
                gaussians,
                camera,
                local_cfg,
                self.structured_trunk_local_projs[local_key],
                iteration=iteration,
            )
            if not torch.is_tensor(local_raw):
                continue

            local_feature_raw = torch.tanh(
                self.structured_trunk_output_head_local_fusion_projs[local_key](local_raw)
            )
            local_raw_feature_means.append(local_feature_raw.detach().abs().mean())
            local_override_cfg = self.structured_trunk_output_head_local_region_overrides.get(
                local_key,
                None,
            )
            local_raw_scale = float(local_override_cfg.get('raw_scale', 1.0)) if local_override_cfg is not None else 1.0
            local_gated_scale = float(local_override_cfg.get('gated_scale', 1.0)) if local_override_cfg is not None else 1.0
            local_gate_feature_scale = float(
                local_override_cfg.get('gate_feature_scale', 1.0)
            ) if local_override_cfg is not None else 1.0
            local_gate_bias = float(local_override_cfg.get('gate_bias', 0.0)) if local_override_cfg is not None else 0.0
            local_gate_boost = float(local_override_cfg.get('gate_boost', 0.0)) if local_override_cfg is not None else 0.0
            local_color_region_scale = float(
                local_override_cfg.get('color_scale', 1.0)
            ) if local_override_cfg is not None else 1.0
            region_gate = self._build_point_gate(gaussians, local_cfg.get('region', None))
            gate_mean_value = 1.0
            gate_feature = None
            local_feature_raw_scaled = local_feature_raw * local_raw_scale
            local_feature_gated_scaled = local_feature_raw * local_gated_scale
            if torch.is_tensor(region_gate):
                region_gate = region_gate.to(
                    device=base_input.device,
                    dtype=base_input.dtype,
                ).clamp(0.0, 1.0)
                soft_gate_min = (
                    local_override_cfg.get('soft_gate_min', None)
                    if local_override_cfg is not None
                    else None
                )
                if soft_gate_min is None:
                    soft_gate_min = float(
                        local_cfg.get(
                            'soft_gate_min',
                            self.structured_trunk_output_head_local_soft_gate_min,
                        )
                    )
                else:
                    soft_gate_min = float(soft_gate_min)
                if soft_gate_min > 0.0:
                    region_gate = region_gate * (1.0 - soft_gate_min) + soft_gate_min
                if local_gate_bias != 0.0:
                    region_gate = (region_gate + local_gate_bias).clamp(0.0, 1.0)
                gate_feature = region_gate.unsqueeze(-1) * local_gate_feature_scale
                local_feature_gated_scaled = local_feature_raw * gate_feature * local_gated_scale
                gate_mean_value = float(region_gate.detach().mean().item())
                local_gate_means.append(region_gate.detach().mean())
                if self.structured_trunk_output_head_use_local_gate_feature:
                    fusion_parts.append(gate_feature)
                if local_gate_boost != 0.0:
                    local_gate_boost_terms.append(region_gate.unsqueeze(-1) * local_gate_boost)
            elif self.structured_trunk_output_head_use_local_gate_feature:
                ones_gate = base_input.new_ones((base_input.shape[0], 1))
                gate_feature = ones_gate * local_gate_feature_scale
                fusion_parts.append(gate_feature)
                if local_gate_boost != 0.0:
                    local_gate_boost_terms.append(ones_gate * local_gate_boost)

            local_debug_parts = []
            local_geometry_feature = None
            local_geometry_raw = None
            local_color = None
            if self.structured_trunk_output_head_use_local_raw_feature:
                deferred_local_raw_fusion_parts.append(local_feature_raw_scaled)
                local_feature_means.append(local_feature_raw_scaled.detach().abs().mean())
                local_debug_parts.append(
                    f"raw={float(local_feature_raw_scaled.detach().abs().mean().item()):.5f}"
                )
            if self.structured_trunk_output_head_use_local_gated_feature:
                fusion_parts.append(local_feature_gated_scaled)
                local_feature_means.append(local_feature_gated_scaled.detach().abs().mean())
                local_debug_parts.append(
                    f"gated={float(local_feature_gated_scaled.detach().abs().mean().item()):.5f}"
                )
            use_hf_local_geometry = (
                self.structured_trunk_output_head_hf_head_use_local_geometry_raw
                and local_key in self.structured_trunk_output_head_hf_head_local_geometry_keys
                and self.structured_trunk_output_head_hf_head_local_geometry_raw_dim > 0
            )
            if (
                local_key in self.structured_trunk_output_head_local_geometry_fusion_projs
                or use_hf_local_geometry
            ):
                local_geometry_raw, _ = self._compose_structured_trunk_local_geometry_input(
                    gaussians,
                    local_cfg,
                )
                if (
                    torch.is_tensor(local_geometry_raw)
                    and local_key in self.structured_trunk_output_head_local_geometry_fusion_projs
                ):
                    local_geometry_feature = torch.tanh(
                        self.structured_trunk_output_head_local_geometry_fusion_projs[local_key](
                            local_geometry_raw
                        )
                    )
            if use_hf_local_geometry:
                if torch.is_tensor(local_geometry_raw):
                    hf_extra_parts.append(local_geometry_raw)
                    local_debug_parts.append(
                        f"hf_geom={float(local_geometry_raw.detach().abs().mean().item()):.5f}"
                    )
                    hf_debug_parts.append(
                        f"{local_cfg.get('name', local_key)}_geom={float(local_geometry_raw.detach().abs().mean().item()):.5f}"
                    )
                else:
                    hf_extra_parts.append(
                        base_input.new_zeros(
                            (
                                base_input.shape[0],
                                self.structured_trunk_output_head_hf_head_local_geometry_raw_dim,
                            )
                        )
                    )
                    local_debug_parts.append("hf_geom=0.00000")
                    hf_debug_parts.append(f"{local_cfg.get('name', local_key)}_geom=0.00000")
            local_color_inputs = []
            if (
                self.structured_trunk_output_head_local_color_use_base_feature
                and torch.is_tensor(base_feature)
            ):
                local_color_inputs.append(base_feature)
            if (
                self.structured_trunk_output_head_local_color_use_shared_feature
                and torch.is_tensor(shared_feature)
            ):
                local_color_inputs.append(shared_feature)
            if (
                self.structured_trunk_output_head_local_color_use_structure_feature
                and torch.is_tensor(structure_feature)
            ):
                local_color_inputs.append(structure_feature)
            if self.structured_trunk_output_head_local_color_use_gate_feature:
                if gate_feature is None:
                    gate_feature = base_input.new_ones((base_input.shape[0], 1)) * local_gate_feature_scale
                local_color_inputs.append(gate_feature)
            if self.structured_trunk_output_head_local_color_use_gated_feature:
                local_color_inputs.append(local_feature_gated_scaled)
            if self.structured_trunk_output_head_local_color_use_raw_feature:
                local_color_inputs.append(local_feature_raw_scaled)
            if torch.is_tensor(local_geometry_feature):
                local_color_inputs.append(local_geometry_feature)
            local_color_input_tensor = None
            if local_color_inputs:
                local_color_input_tensor = torch.cat(local_color_inputs, dim=1)
                local_owner_input_outputs[region_name] = local_color_input_tensor
                if region_name != local_key:
                    local_owner_input_outputs[local_key] = local_color_input_tensor
            if torch.is_tensor(local_geometry_raw):
                local_geometry_raw_outputs[region_name] = local_geometry_raw
                if region_name != local_key:
                    local_geometry_raw_outputs[local_key] = local_geometry_raw
            if (
                self.structured_trunk_output_head_local_color_enable
                and local_key in self.structured_trunk_output_head_local_color_mlps
                and local_color_scale > 0.0
                and torch.is_tensor(local_color_input_tensor)
            ):
                local_color = self.structured_trunk_output_head_local_color_mlps[local_key](
                    local_color_input_tensor
                ) * (local_color_scale * local_color_region_scale)
                if (
                    self.structured_trunk_output_head_local_color_gate_with_region
                    and torch.is_tensor(region_gate)
                ):
                    local_color = local_color * region_gate.unsqueeze(-1)
                local_color_means.append(local_color.detach().abs().mean())
                if local_color_total is None:
                    local_color_total = local_color
                else:
                    local_color_total = local_color_total + local_color
                if region_name in local_color_outputs:
                    local_color_outputs[region_name] = (
                        local_color_outputs[region_name] + local_color
                    )
                else:
                    local_color_outputs[region_name] = local_color
                local_debug_parts.append(
                    f"color={float(local_color.detach().abs().mean().item()):.5f}"
                )
            if torch.is_tensor(local_geometry_feature):
                local_debug_parts.append(
                    f"geom={float(local_geometry_feature.detach().abs().mean().item()):.5f}"
                )
            if self.structured_trunk_output_head_region_support_enable:
                support_components = []
                use_hybrid_support = (
                    'hybrid' in region_support_sources
                    or 'auto' in region_support_sources
                    or 'default' in region_support_sources
                )
                if (
                    use_hybrid_support
                    or 'gated_feature' in region_support_sources
                    or 'gated' in region_support_sources
                ):
                    support_components.append(local_feature_gated_scaled)
                if (
                    use_hybrid_support
                    or 'local_color' in region_support_sources
                    or 'color' in region_support_sources
                ) and torch.is_tensor(local_color):
                    support_components.append(local_color)
                if (
                    'raw_feature' in region_support_sources
                    or 'raw' in region_support_sources
                ):
                    support_components.append(local_feature_raw_scaled)
                if (
                    'geometry' in region_support_sources
                    or 'geom' in region_support_sources
                    or 'local_geometry' in region_support_sources
                ) and torch.is_tensor(local_geometry_feature):
                    support_components.append(local_geometry_feature)
                if (
                    'gate_feature' in region_support_sources
                    or 'gate' in region_support_sources
                    or 'region_gate' in region_support_sources
                ) and gate_feature is not None:
                    support_components.append(gate_feature)
                if not support_components and gate_feature is not None:
                    support_components.append(gate_feature)

                support_terms = []
                for support_component in support_components:
                    if not torch.is_tensor(support_component):
                        continue
                    support_term = support_component
                    if self.structured_trunk_output_head_region_support_detach:
                        support_term = support_term.detach()
                    if support_term.dim() == 1:
                        support_term = support_term.unsqueeze(-1)
                    support_terms.append(support_term.abs().mean(dim=1, keepdim=True))
                if support_terms:
                    local_region_support = torch.stack(support_terms, dim=0).mean(dim=0)
                    local_debug_parts.append(
                        f"support={float(local_region_support.detach().mean().item()):.4f}"
                    )
                else:
                    local_region_support = None
            else:
                local_region_support = None
            if region_gate_mean is not None and not torch.is_tensor(region_gate):
                local_gate_means.append(region_gate_mean.detach())
            if torch.is_tensor(region_gate):
                region_gate_outputs[region_name] = region_gate.unsqueeze(-1)
                if region_name != local_key:
                    region_gate_outputs[local_key] = region_gate.unsqueeze(-1)
            if torch.is_tensor(local_region_support):
                region_support_outputs[region_name] = local_region_support
                if region_name != local_key:
                    region_support_outputs[local_key] = local_region_support
            debug_parts.append(
                (
                    f"{local_cfg.get('name', local_key)}:"
                    f"{','.join(local_debug_parts) if local_debug_parts else 'disabled'},"
                    f"gate={gate_mean_value:.4f}"
                )
            )

        if local_raw_feature_means:
            self.last_structured_trunk_local_raw_abs_mean = torch.stack(
                local_raw_feature_means,
                dim=0,
            ).mean()
        if local_feature_means:
            self.last_structured_trunk_local_abs_mean = torch.stack(
                local_feature_means,
                dim=0,
            ).mean()
        if local_gate_means:
            self.last_structured_trunk_local_gate_mean = torch.stack(
                local_gate_means,
                dim=0,
            ).mean()

        if len(fusion_parts) <= 0:
            return base_input, 'no_fusion_parts', {}

        if deferred_local_raw_fusion_parts:
            # Keep legacy gate/gated ordering at the front so older trunk-head
            # checkpoints can warm-start the expanded fusion input safely.
            fusion_parts.extend(deferred_local_raw_fusion_parts)

        fusion_input = torch.cat(fusion_parts, dim=1)
        extra_outputs = {}
        if local_color_total is not None:
            extra_outputs['local_color'] = local_color_total
            self.last_structured_trunk_head_local_color_abs_mean = (
                local_color_total.detach().abs().mean()
            )
        if local_color_outputs:
            extra_outputs['local_colors'] = local_color_outputs
        if local_owner_input_outputs:
            extra_outputs['local_owner_inputs'] = local_owner_input_outputs
        if local_geometry_raw_outputs:
            extra_outputs['local_geometry_raws'] = local_geometry_raw_outputs
        if local_gate_boost_terms:
            extra_outputs['gate_boost'] = torch.stack(local_gate_boost_terms, dim=0).sum(dim=0)
            self.last_structured_trunk_head_gate_boost_mean = (
                extra_outputs['gate_boost'].detach().mean()
            )
        if region_gate_outputs:
            extra_outputs['region_gates'] = region_gate_outputs
        if region_support_outputs:
            extra_outputs['region_supports'] = region_support_outputs
        hf_input_parts = []
        if self.structured_trunk_output_head_hf_head_use_output_fusion:
            hf_input_parts.append(fusion_input)
        if hf_extra_parts:
            hf_input_parts.extend(hf_extra_parts)
        if hf_input_parts:
            extra_outputs['hf_input'] = torch.cat(hf_input_parts, dim=1)
            extra_outputs['hf_debug'] = ' | '.join(hf_debug_parts)
        return fusion_input, ' | '.join(debug_parts), extra_outputs

    def _compose_structured_trunk_local_color_owner_head(
        self,
        template,
        gaussians,
        local_owner_inputs,
        local_color_outputs,
        local_geometry_raws,
        region_gates,
        region_supports,
        iteration=0,
    ):
        zero_color = torch.zeros_like(template)
        zero_scalar = template.new_tensor(0.0)
        if not self.structured_trunk_output_head_local_color_owner_head_enable:
            return zero_color, zero_color, zero_scalar, zero_scalar, 'disabled'

        owner_scale = _resolve_scheduled_scalar(
            iteration,
            self.structured_trunk_output_head_local_color_owner_head_scale_cfg,
            default=1.0,
        )
        if owner_scale <= 0.0:
            return zero_color, zero_color, zero_scalar, zero_scalar, 'disabled_by_scale'

        owner_local_color_output_scale = _resolve_scheduled_scalar(
            iteration,
            self.structured_trunk_output_head_local_color_owner_head_local_color_output_scale_cfg,
            default=1.0,
        )
        owner_input_stats = []
        raw_parts = []
        owner_parts = []
        support_stats = []
        gate_stats = []
        debug_parts = []
        boundary_score = self._get_binding_boundary_score(
            gaussians,
            template,
            detach=True,
        ) if self.structured_trunk_output_head_local_color_owner_head_boundary_enable else None
        boundary_scale = _resolve_scheduled_scalar(
            iteration,
            self.structured_trunk_output_head_local_color_owner_head_boundary_scale_cfg,
            default=1.0,
        ) if self.structured_trunk_output_head_local_color_owner_head_boundary_enable else 0.0
        boundary_local_color_output_scale = _resolve_scheduled_scalar(
            iteration,
            self.structured_trunk_output_head_local_color_owner_head_boundary_local_color_output_scale_cfg,
            default=1.0,
        ) if self.structured_trunk_output_head_local_color_owner_head_boundary_enable else 0.0
        boundary_contrib_scale = _resolve_scheduled_scalar(
            iteration,
            self.structured_trunk_output_head_local_color_owner_head_boundary_contrib_scale_cfg,
            default=1.0,
        ) if self.structured_trunk_output_head_local_color_owner_head_boundary_enable else 1.0
        boundary_input_stats = []
        boundary_raw_stats = []
        boundary_contrib_stats = []
        boundary_focus_stats = []
        boundary_gate_stats = []
        boundary_debug_parts = []
        owner_geometry_raw_dim = self._encoded_local_color_geometry_dim()

        for local_cfg in self.structured_trunk_local_cfgs:
            if not bool(local_cfg.get('enable', False)):
                continue
            local_key = str(local_cfg.get('_key', ''))
            if local_key not in self.structured_trunk_output_head_local_color_owner_head_mlps:
                continue
            region_name = str(local_cfg.get('name', local_key) or local_key)
            owner_input_parts = []
            owner_input_dim = int(
                self.structured_trunk_output_head_local_color_owner_head_input_dims.get(local_key, 0)
            )

            local_owner_input = _lookup_named_tensor(local_owner_inputs, local_key, region_name)
            local_color = _lookup_named_tensor(local_color_outputs, local_key, region_name)
            local_geometry_raw = _lookup_named_tensor(local_geometry_raws, local_key, region_name)
            region_support = _lookup_named_tensor(region_supports, local_key, region_name)
            region_gate = _lookup_named_tensor(region_gates, local_key, region_name)

            if self.structured_trunk_output_head_local_color_owner_head_use_local_color_input:
                local_color_input_dim = int(
                    self.structured_trunk_output_head_local_color_input_dims.get(local_key, 0)
                )
                if torch.is_tensor(local_owner_input):
                    owner_input_parts.append(local_owner_input)
                elif local_color_input_dim > 0:
                    owner_input_parts.append(
                        template.new_zeros((template.shape[0], local_color_input_dim))
                    )
            if self.structured_trunk_output_head_local_color_owner_head_use_local_color_output:
                if torch.is_tensor(local_color):
                    if owner_local_color_output_scale > 0.0:
                        owner_input_parts.append(local_color * owner_local_color_output_scale)
                    else:
                        owner_input_parts.append(
                            template.new_zeros((template.shape[0], template.shape[1]))
                        )
                else:
                    owner_input_parts.append(template.new_zeros((template.shape[0], template.shape[1])))
            if self.structured_trunk_output_head_local_color_owner_head_use_local_geometry_raw:
                if torch.is_tensor(local_geometry_raw):
                    owner_input_parts.append(local_geometry_raw)
                elif owner_geometry_raw_dim > 0:
                    owner_input_parts.append(
                        template.new_zeros((template.shape[0], owner_geometry_raw_dim))
                    )
            if self.structured_trunk_output_head_local_color_owner_head_use_support_feature:
                if torch.is_tensor(region_support):
                    owner_input_parts.append(region_support)
                    support_stats.append(region_support.detach().mean())
                else:
                    owner_input_parts.append(template.new_zeros((template.shape[0], 1)))
            if self.structured_trunk_output_head_local_color_owner_head_use_region_gate_feature:
                if torch.is_tensor(region_gate):
                    owner_input_parts.append(region_gate)
                else:
                    owner_input_parts.append(template.new_zeros((template.shape[0], 1)))

            if len(owner_input_parts) <= 0:
                continue
            owner_head_inp = torch.cat(owner_input_parts, dim=1)
            if owner_input_dim > 0 and owner_head_inp.shape[1] != owner_input_dim:
                debug_parts.append(
                    f"{region_name}:shape_mismatch(inp={owner_head_inp.shape[1]},expected={owner_input_dim})"
                )
                continue
            owner_input_stats.append(owner_head_inp.detach().abs().mean())

            owner_logits = self.structured_trunk_output_head_local_color_owner_head_mlps[local_key](
                owner_head_inp
            )
            owner_color = self._decode_structured_trunk_head_color(
                owner_logits,
                self.structured_trunk_output_head_local_color_owner_head_mode,
                self.structured_trunk_output_head_local_color_owner_head_compose_mode,
                owner_scale,
                0.0,
                self.structured_trunk_output_head_local_color_owner_head_chroma_center,
                self.structured_trunk_output_head_local_color_owner_head_band_luma_scale,
                self.structured_trunk_output_head_local_color_owner_head_band_chroma_scale,
            )
            owner_gate, _ = self._compute_structured_trunk_head_gate(
                template,
                owner_head_inp,
                self.structured_trunk_output_head_local_color_owner_head_gate_mlps[local_key]
                if local_key in self.structured_trunk_output_head_local_color_owner_head_gate_mlps
                else None,
                self.structured_trunk_output_head_local_color_owner_head_gate_gain,
                self.structured_trunk_output_head_local_color_owner_head_gate_bias,
                self.structured_trunk_output_head_local_color_owner_head_min_gate,
                self.structured_trunk_output_head_local_color_owner_head_max_gate,
                gate_boost=None,
            )
            owner_contrib = owner_color * owner_gate
            raw_parts.append(owner_color)
            owner_parts.append(owner_contrib)
            gate_stats.append(owner_gate.detach().mean())
            debug_parts.append(
                (
                    f"{region_name}:inp={float(owner_head_inp.detach().abs().mean().item()):.5f},"
                    f"seed={0.0 if local_color is None else float(local_color.detach().abs().mean().item()):.5f},"
                    f"geom_raw={0.0 if local_geometry_raw is None else float(local_geometry_raw.detach().abs().mean().item()):.5f},"
                    f"support={0.0 if region_support is None else float(region_support.detach().mean().item()):.4f},"
                    f"gate={float(owner_gate.detach().mean().item()):.4f},"
                    f"color={float(owner_color.detach().abs().mean().item()):.5f}"
                )
            )

        if (
            self.structured_trunk_output_head_local_color_owner_head_boundary_enable
            and boundary_scale > 0.0
            and torch.is_tensor(boundary_score)
        ):
            for local_cfg in self.structured_trunk_local_cfgs:
                if not bool(local_cfg.get('enable', False)):
                    continue
                local_key = str(local_cfg.get('_key', ''))
                if local_key not in self.structured_trunk_output_head_local_color_owner_head_boundary_mlps:
                    continue
                region_name = str(local_cfg.get('name', local_key) or local_key)
                boundary_input_parts = []
                boundary_input_dim = int(
                    self.structured_trunk_output_head_local_color_owner_head_boundary_input_dims.get(local_key, 0)
                )

                local_owner_input = _lookup_named_tensor(local_owner_inputs, local_key, region_name)
                local_color = _lookup_named_tensor(local_color_outputs, local_key, region_name)
                local_geometry_raw = _lookup_named_tensor(local_geometry_raws, local_key, region_name)
                region_support = _lookup_named_tensor(region_supports, local_key, region_name)
                region_gate = _lookup_named_tensor(region_gates, local_key, region_name)

                boundary_focus_score = self._get_structured_trunk_owner_boundary_focus_score(
                    gaussians,
                    template,
                    boundary_score,
                )
                local_boundary_focus = self._build_boundary_focus(
                    boundary_focus_score,
                    threshold=self.structured_trunk_output_head_local_color_owner_head_boundary_focus_threshold,
                    power=self.structured_trunk_output_head_local_color_owner_head_boundary_focus_power,
                    min_focus=self.structured_trunk_output_head_local_color_owner_head_boundary_focus_min,
                    max_focus=self.structured_trunk_output_head_local_color_owner_head_boundary_focus_max,
                )
                if not torch.is_tensor(local_boundary_focus):
                    continue
                if (
                    self.structured_trunk_output_head_local_color_owner_head_boundary_focus_use_region_gate
                    and torch.is_tensor(region_gate)
                ):
                    local_boundary_focus = local_boundary_focus * region_gate.to(
                        device=template.device,
                        dtype=template.dtype,
                    ).clamp(0.0, 1.0)
                if (
                    self.structured_trunk_output_head_local_color_owner_head_boundary_focus_use_support
                    and torch.is_tensor(region_support)
                ):
                    boundary_support = region_support
                    if self.structured_trunk_output_head_local_color_owner_head_boundary_focus_support_detach:
                        boundary_support = boundary_support.detach()
                    local_boundary_focus = local_boundary_focus * torch.sigmoid(
                        boundary_support.to(device=template.device, dtype=template.dtype)
                    )
                local_boundary_focus = local_boundary_focus.clamp(0.0, 1.0)

                if self.structured_trunk_output_head_local_color_owner_head_boundary_use_local_color_input:
                    local_color_input_dim = int(
                        self.structured_trunk_output_head_local_color_input_dims.get(local_key, 0)
                    )
                    if torch.is_tensor(local_owner_input):
                        boundary_input_parts.append(local_owner_input)
                    elif local_color_input_dim > 0:
                        boundary_input_parts.append(
                            template.new_zeros((template.shape[0], local_color_input_dim))
                        )
                if self.structured_trunk_output_head_local_color_owner_head_boundary_use_local_color_output:
                    if torch.is_tensor(local_color):
                        if boundary_local_color_output_scale > 0.0:
                            boundary_input_parts.append(local_color * boundary_local_color_output_scale)
                        else:
                            boundary_input_parts.append(
                                template.new_zeros((template.shape[0], template.shape[1]))
                            )
                    else:
                        boundary_input_parts.append(
                            template.new_zeros((template.shape[0], template.shape[1]))
                        )
                if self.structured_trunk_output_head_local_color_owner_head_boundary_use_local_geometry_raw:
                    if torch.is_tensor(local_geometry_raw):
                        boundary_input_parts.append(local_geometry_raw)
                    elif owner_geometry_raw_dim > 0:
                        boundary_input_parts.append(
                            template.new_zeros((template.shape[0], owner_geometry_raw_dim))
                        )
                if self.structured_trunk_output_head_local_color_owner_head_boundary_use_support_feature:
                    if torch.is_tensor(region_support):
                        boundary_input_parts.append(region_support)
                    else:
                        boundary_input_parts.append(template.new_zeros((template.shape[0], 1)))
                if self.structured_trunk_output_head_local_color_owner_head_boundary_use_region_gate_feature:
                    if torch.is_tensor(region_gate):
                        boundary_input_parts.append(region_gate)
                    else:
                        boundary_input_parts.append(template.new_zeros((template.shape[0], 1)))
                if self.structured_trunk_output_head_local_color_owner_head_boundary_use_boundary_feature:
                    boundary_input_parts.append(local_boundary_focus)

                if len(boundary_input_parts) <= 0:
                    continue
                boundary_head_inp = torch.cat(boundary_input_parts, dim=1)
                if boundary_input_dim > 0 and boundary_head_inp.shape[1] != boundary_input_dim:
                    boundary_debug_parts.append(
                        f"{region_name}:shape_mismatch(inp={boundary_head_inp.shape[1]},expected={boundary_input_dim})"
                    )
                    continue
                boundary_input_stats.append(boundary_head_inp.detach().abs().mean())
                boundary_focus_stats.append(local_boundary_focus.detach().mean())

                boundary_logits = self.structured_trunk_output_head_local_color_owner_head_boundary_mlps[local_key](
                    boundary_head_inp
                )
                boundary_color = self._decode_structured_trunk_head_color(
                    boundary_logits,
                    self.structured_trunk_output_head_local_color_owner_head_boundary_mode,
                    self.structured_trunk_output_head_local_color_owner_head_boundary_compose_mode,
                    boundary_scale,
                    0.0,
                    self.structured_trunk_output_head_local_color_owner_head_boundary_chroma_center,
                    self.structured_trunk_output_head_local_color_owner_head_boundary_band_luma_scale,
                    self.structured_trunk_output_head_local_color_owner_head_boundary_band_chroma_scale,
                )
                boundary_color = boundary_color * local_boundary_focus
                boundary_gate, _ = self._compute_structured_trunk_head_gate(
                    template,
                    boundary_head_inp,
                    self.structured_trunk_output_head_local_color_owner_head_boundary_gate_mlps[local_key]
                    if local_key in self.structured_trunk_output_head_local_color_owner_head_boundary_gate_mlps
                    else None,
                    self.structured_trunk_output_head_local_color_owner_head_boundary_gate_gain,
                    self.structured_trunk_output_head_local_color_owner_head_boundary_gate_bias,
                    self.structured_trunk_output_head_local_color_owner_head_boundary_min_gate,
                    self.structured_trunk_output_head_local_color_owner_head_boundary_max_gate,
                    gate_boost=None,
                )
                boundary_effective_gate = boundary_gate
                boundary_contrib_gate_mode = (
                    self.structured_trunk_output_head_local_color_owner_head_boundary_contrib_gate_mode
                )
                if boundary_contrib_gate_mode in ('none', 'focus_only', 'mask_only', 'direct'):
                    boundary_effective_gate = torch.ones_like(boundary_gate)
                elif boundary_contrib_gate_mode in ('floor', 'min', 'gate_floor'):
                    min_gate = self.structured_trunk_output_head_local_color_owner_head_boundary_contrib_min_gate
                    if min_gate > 0.0:
                        boundary_effective_gate = torch.clamp(boundary_gate, min=float(min_gate), max=1.0)
                elif boundary_contrib_gate_mode not in ('multiply', 'gate', 'default'):
                    boundary_effective_gate = boundary_gate
                boundary_contrib = boundary_color * boundary_effective_gate * float(boundary_contrib_scale)
                raw_parts.append(boundary_color)
                owner_parts.append(boundary_contrib)
                boundary_raw_stats.append(boundary_color.detach().abs().mean())
                boundary_contrib_stats.append(boundary_contrib.detach().abs().mean())
                boundary_gate_stats.append(boundary_effective_gate.detach().mean())
                boundary_debug_parts.append(
                    (
                        f"{region_name}:inp={float(boundary_head_inp.detach().abs().mean().item()):.5f},"
                        f"focus={float(local_boundary_focus.detach().mean().item()):.4f},"
                        f"gate={float(boundary_gate.detach().mean().item()):.4f},"
                        f"eff_gate={float(boundary_effective_gate.detach().mean().item()):.4f},"
                        f"color={float(boundary_color.detach().abs().mean().item()):.5f}"
                    )
                )

        self.last_structured_trunk_owner_input_abs_mean = (
            torch.stack(owner_input_stats, dim=0).mean()
            if owner_input_stats
            else zero_scalar
        )
        owner_support_mean = (
            torch.stack(support_stats, dim=0).mean()
            if support_stats
            else zero_scalar
        )
        owner_gate_mean = (
            torch.stack(gate_stats, dim=0).mean()
            if gate_stats
            else zero_scalar
        )
        self.last_structured_trunk_owner_boundary_input_abs_mean = (
            torch.stack(boundary_input_stats, dim=0).mean()
            if boundary_input_stats
            else zero_scalar
        )
        self.last_structured_trunk_owner_boundary_color_abs_mean = (
            torch.stack(boundary_raw_stats, dim=0).mean()
            if boundary_raw_stats
            else zero_scalar
        )
        self.last_structured_trunk_owner_boundary_abs_mean = (
            torch.stack(boundary_contrib_stats, dim=0).mean()
            if boundary_contrib_stats
            else zero_scalar
        )
        self.last_structured_trunk_owner_boundary_focus_mean = (
            torch.stack(boundary_focus_stats, dim=0).mean()
            if boundary_focus_stats
            else zero_scalar
        )
        self.last_structured_trunk_owner_boundary_gate_mean = (
            torch.stack(boundary_gate_stats, dim=0).mean()
            if boundary_gate_stats
            else zero_scalar
        )
        self.last_structured_trunk_owner_boundary_debug = ' | '.join(boundary_debug_parts)
        if not raw_parts or not owner_parts:
            return zero_color, zero_color, owner_support_mean, owner_gate_mean, 'missing_owner_head_parts'
        all_debug_parts = list(debug_parts)
        if boundary_debug_parts:
            all_debug_parts.append('boundary=' + ' | '.join(boundary_debug_parts))
        return (
            torch.stack(raw_parts, dim=0).sum(dim=0),
            torch.stack(owner_parts, dim=0).sum(dim=0),
            owner_support_mean,
            owner_gate_mean,
            'mode=head | ' + ' | '.join(all_debug_parts),
        )

    def _compose_structured_trunk_local_color_owner(
        self,
        template,
        local_color_outputs,
        local_color_total,
        region_gates,
        region_support_modulators,
        iteration=0,
    ):
        zero_color = torch.zeros_like(template)
        zero_scalar = template.new_tensor(0.0)
        self.last_structured_trunk_owner_input_abs_mean = zero_scalar
        if not self.structured_trunk_output_head_local_color_owner_enable:
            return zero_color, zero_color, zero_scalar, zero_scalar, ''

        owner_scale = _resolve_scheduled_scalar(
            iteration,
            self.structured_trunk_output_head_local_color_owner_scale_cfg,
            default=1.0,
        )
        if owner_scale <= 0.0:
            return zero_color, zero_color, zero_scalar, zero_scalar, 'disabled_by_scale'

        named_local_colors = {}
        if isinstance(local_color_outputs, dict):
            for name, local_color in local_color_outputs.items():
                if torch.is_tensor(local_color):
                    named_local_colors[str(name)] = local_color
        if not named_local_colors and torch.is_tensor(local_color_total):
            named_local_colors['global'] = local_color_total
        if not named_local_colors:
            return zero_color, zero_color, zero_scalar, zero_scalar, 'missing_local_color'

        fallback_support = None
        if (
            self.structured_trunk_output_head_local_color_owner_use_support
            and isinstance(region_support_modulators, dict)
            and len(region_support_modulators) > 0
        ):
            support_terms = [
                support
                for support in region_support_modulators.values()
                if torch.is_tensor(support)
            ]
            if support_terms:
                fallback_support = torch.stack(support_terms, dim=0).mean(dim=0)

        fallback_gate = None
        if (
            self.structured_trunk_output_head_local_color_owner_use_region_gate
            and isinstance(region_gates, dict)
            and len(region_gates) > 0
        ):
            gate_terms = [
                gate
                for gate in region_gates.values()
                if torch.is_tensor(gate)
            ]
            if gate_terms:
                fallback_gate = torch.stack(gate_terms, dim=0).mean(dim=0)

        raw_parts = []
        owner_parts = []
        support_stats = []
        gate_stats = []
        debug_parts = []
        gate_base = float(self.structured_trunk_output_head_local_color_owner_gate_base)

        for region_name, local_color in named_local_colors.items():
            raw_parts.append(local_color)
            owner_color = local_color
            support_value = 0.0
            gate_value = 0.0

            if self.structured_trunk_output_head_local_color_owner_use_support:
                region_support = None
                if isinstance(region_support_modulators, dict):
                    region_support = region_support_modulators.get(str(region_name), None)
                if not torch.is_tensor(region_support):
                    region_support = fallback_support
                if torch.is_tensor(region_support):
                    owner_color = owner_color * region_support
                    support_value = float(region_support.detach().mean().item())
                    support_stats.append(region_support.detach().mean())

            if self.structured_trunk_output_head_local_color_owner_use_region_gate:
                region_gate = None
                if isinstance(region_gates, dict):
                    region_gate = region_gates.get(str(region_name), None)
                if not torch.is_tensor(region_gate):
                    region_gate = fallback_gate
                if torch.is_tensor(region_gate):
                    if gate_base > 0.0:
                        region_gate = region_gate * (1.0 - gate_base) + gate_base
                    owner_color = owner_color * region_gate
                    gate_value = float(region_gate.detach().mean().item())
                    gate_stats.append(region_gate.detach().mean())

            owner_parts.append(owner_color)
            debug_parts.append(
                (
                    f"{region_name}:raw={float(local_color.detach().abs().mean().item()):.5f},"
                    f"support={support_value:.4f},gate={gate_value:.4f}"
                )
            )

        if not raw_parts or not owner_parts:
            return zero_color, zero_color, zero_scalar, zero_scalar, 'missing_owner_parts'

        raw_owner_color = torch.stack(raw_parts, dim=0).sum(dim=0) * owner_scale
        owner_contrib = torch.stack(owner_parts, dim=0).sum(dim=0) * owner_scale
        owner_support_mean = (
            torch.stack(support_stats, dim=0).mean()
            if support_stats
            else zero_scalar
        )
        owner_gate_mean = (
            torch.stack(gate_stats, dim=0).mean()
            if gate_stats
            else zero_scalar
        )
        return (
            raw_owner_color,
            owner_contrib,
            owner_support_mean,
            owner_gate_mean,
            'mode=legacy | ' + ' | '.join(debug_parts),
        )

    def _compose_structured_trunk_local_color_owner_takeover_mix(
        self,
        template,
        gaussians,
        region_gates,
        region_supports,
        iteration=0,
    ):
        zero_mix = template.new_zeros((template.shape[0], 1))
        zero_scalar = template.new_tensor(0.0)
        if not (
            self.structured_trunk_output_head_local_color_owner_head_takeover_enable
            or self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_enable
        ):
            return zero_mix, zero_scalar, ''
        takeover_mix = zero_mix
        debug_parts = []
        support_mean = zero_scalar

        if self.structured_trunk_output_head_local_color_owner_head_takeover_enable:
            takeover_scale = _resolve_scheduled_scalar(
                iteration,
                self.structured_trunk_output_head_local_color_owner_head_takeover_scale_cfg,
                default=1.0,
            )
            takeover_strength = _resolve_scheduled_scalar(
                iteration,
                self.structured_trunk_output_head_local_color_owner_head_takeover_strength_cfg,
                default=1.0,
            )
            if takeover_scale > 0.0 and takeover_strength > 0.0:
                region_strength_cfg = (
                    self.structured_trunk_output_head_local_color_owner_head_takeover_region_strength_cfg
                )
                named_scales = {}
                if isinstance(region_strength_cfg, dict) and region_strength_cfg:
                    named_scales.update(region_strength_cfg)
                elif isinstance(region_gates, dict):
                    for region_name, region_gate in region_gates.items():
                        if torch.is_tensor(region_gate):
                            named_scales[str(region_name)] = 1.0
                if named_scales:
                    takeover_supports = {}
                    if (
                        self.structured_trunk_output_head_local_color_owner_head_takeover_use_support
                        and isinstance(region_supports, dict)
                    ):
                        for region_name, region_support in region_supports.items():
                            if not torch.is_tensor(region_support):
                                continue
                            if self.structured_trunk_output_head_local_color_owner_head_takeover_support_detach:
                                region_support = region_support.detach()
                            takeover_supports[str(region_name)] = region_support

                    support_modulators = _build_named_region_support_modulators(
                        takeover_supports,
                        template,
                        offset=self.structured_trunk_output_head_local_color_owner_head_takeover_support_offset,
                        gain=self.structured_trunk_output_head_local_color_owner_head_takeover_support_gain,
                        power=self.structured_trunk_output_head_local_color_owner_head_takeover_support_power,
                        base_mix=0.0,
                        max_value=1.0,
                    )
                    takeover_mix = _compose_named_region_scalar(
                        region_gates,
                        named_scales,
                        template,
                        base=0.0,
                        additive=True,
                        min_value=0.0,
                        max_value=1.0,
                        named_modulators=support_modulators,
                    )
                    takeover_mix = (
                        self.structured_trunk_output_head_local_color_owner_head_takeover_base_mix
                        + takeover_mix * takeover_scale * takeover_strength
                    )
                    takeover_max = (
                        self.structured_trunk_output_head_local_color_owner_head_takeover_max
                    )
                    if takeover_max > 0.0:
                        takeover_mix = takeover_mix.clamp(max=takeover_max)
                    takeover_mix = takeover_mix.clamp(0.0, 1.0)

                    debug_parts.append(
                        (
                            f"scale={takeover_scale:.3f},strength={takeover_strength:.3f},"
                            f"base={self.structured_trunk_output_head_local_color_owner_head_takeover_base_mix:.3f},"
                            f"mean={float(takeover_mix.detach().mean().item()):.4f}"
                        )
                    )
                    support_stats = []
                    for region_name, region_strength in named_scales.items():
                        region_gate = region_gates.get(str(region_name), None) if isinstance(region_gates, dict) else None
                        if not torch.is_tensor(region_gate):
                            continue
                        region_gate = region_gate.to(device=template.device, dtype=template.dtype)
                        if region_gate.dim() == 1:
                            region_gate = region_gate.unsqueeze(-1)
                        support_modulator = support_modulators.get(str(region_name), None)
                        region_support_mean = None
                        if torch.is_tensor(support_modulator):
                            region_support_mean = float(support_modulator.detach().mean().item())
                            support_stats.append(support_modulator.detach().mean())
                            region_gate = region_gate * support_modulator.to(
                                device=template.device,
                                dtype=template.dtype,
                            )
                        region_mix = (
                            region_gate
                            * float(region_strength)
                            * takeover_scale
                            * takeover_strength
                        ).clamp(0.0, 1.0)
                        debug_parts.append(
                            f"{region_name}={float(region_mix.detach().mean().item()):.4f}"
                            + (
                                f"/support={region_support_mean:.4f}"
                                if region_support_mean is not None
                                else ''
                            )
                        )

                    support_mean = (
                        torch.stack(support_stats, dim=0).mean()
                        if support_stats
                        else zero_scalar
                    )
                else:
                    debug_parts.append('missing_region_strength')
            else:
                debug_parts.append('disabled_by_scale')

        boundary_takeover_mean = zero_scalar
        if self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_enable:
            boundary_takeover_scale = _resolve_scheduled_scalar(
                iteration,
                self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_scale_cfg,
                default=1.0,
            )
            if boundary_takeover_scale > 0.0:
                boundary_focus_score = self._get_structured_trunk_owner_boundary_focus_score(
                    gaussians,
                    template,
                    self._get_binding_boundary_score(gaussians, template, detach=True),
                )
                boundary_focus = self._build_boundary_focus(
                    boundary_focus_score,
                    threshold=self.structured_trunk_output_head_local_color_owner_head_boundary_focus_threshold,
                    power=self.structured_trunk_output_head_local_color_owner_head_boundary_focus_power,
                    min_focus=self.structured_trunk_output_head_local_color_owner_head_boundary_focus_min,
                    max_focus=self.structured_trunk_output_head_local_color_owner_head_boundary_focus_max,
                )
                if torch.is_tensor(boundary_focus):
                    boundary_terms = []
                    if isinstance(region_gates, dict) and region_gates:
                        for region_gate in region_gates.values():
                            if not torch.is_tensor(region_gate):
                                continue
                            boundary_term = boundary_focus
                            if self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_use_region_gate:
                                boundary_term = boundary_term * region_gate.to(
                                    device=template.device,
                                    dtype=template.dtype,
                                ).clamp(0.0, 1.0)
                            boundary_terms.append(boundary_term)
                    if boundary_terms:
                        boundary_takeover_mix = torch.stack(boundary_terms, dim=0).amax(dim=0)
                    else:
                        boundary_takeover_mix = boundary_focus
                    boundary_takeover_mix = boundary_takeover_mix * boundary_takeover_scale
                    if self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_max > 0.0:
                        boundary_takeover_mix = boundary_takeover_mix.clamp(
                            max=self.structured_trunk_output_head_local_color_owner_head_boundary_takeover_max
                        )
                    boundary_takeover_mix = boundary_takeover_mix.clamp(0.0, 1.0)
                    takeover_mix = torch.maximum(
                        takeover_mix.to(device=template.device, dtype=template.dtype),
                        boundary_takeover_mix,
                    )
                    boundary_takeover_mean = boundary_takeover_mix.detach().mean()
                    debug_parts.append(
                        f"boundary_takeover={float(boundary_takeover_mean.item()):.4f}/scale={boundary_takeover_scale:.3f}"
                    )

        self.last_structured_trunk_owner_boundary_takeover_mean = boundary_takeover_mean
        return takeover_mix, support_mean, ' | '.join(debug_parts)

    def _compose_structured_trunk_local_color_owner_takeover_legacy_scale(
        self,
        template,
        takeover_mix,
    ):
        legacy_scale = template.new_ones((template.shape[0], 1))
        zero_scalar = template.new_tensor(0.0)
        if not (
            self.structured_trunk_output_head_local_color_owner_head_takeover_enable
            and self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_enable
            and torch.is_tensor(takeover_mix)
        ):
            return legacy_scale, zero_scalar, ''

        legacy_scale = takeover_mix.to(device=template.device, dtype=template.dtype)
        if legacy_scale.dim() == 1:
            legacy_scale = legacy_scale.unsqueeze(-1)
        legacy_min_scale = float(
            self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_min_scale
        )
        legacy_min_scale = max(0.0, min(1.0, legacy_min_scale))
        legacy_scale = (1.0 - legacy_scale).clamp(min=legacy_min_scale, max=1.0)
        legacy_power = max(
            float(self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_power),
            1.0e-6,
        )
        if abs(legacy_power - 1.0) > 1.0e-6:
            legacy_scale = legacy_scale.pow(legacy_power)
        legacy_scale_mean = legacy_scale.detach().mean()
        debug = (
            f"legacy={float(legacy_scale_mean.item()):.4f},"
            f"min={legacy_min_scale:.3f},pow={legacy_power:.3f}"
        )
        return legacy_scale, legacy_scale_mean, debug

    def _set_structured_trunk_output_head_zero_stats(self, output, debug=''):
        zero = output.new_tensor(0.0)
        self.last_structured_trunk_head_abs_mean = zero
        self.last_structured_trunk_head_color_abs_mean = zero
        self.last_structured_trunk_head_gate_mean = zero
        self.last_structured_trunk_head_gate_boost_mean = zero
        self.last_structured_trunk_head_local_color_abs_mean = zero
        self.last_structured_trunk_head_fusion_abs_mean = zero
        self.last_structured_trunk_head_debug = debug
        self.last_structured_trunk_owner_abs_mean = zero
        self.last_structured_trunk_owner_input_abs_mean = zero
        self.last_structured_trunk_owner_color_abs_mean = zero
        self.last_structured_trunk_owner_support_mean = zero
        self.last_structured_trunk_owner_gate_mean = zero
        self.last_structured_trunk_owner_takeover_mean = zero
        self.last_structured_trunk_owner_takeover_legacy_scale_mean = zero
        self.last_structured_trunk_owner_boundary_abs_mean = zero
        self.last_structured_trunk_owner_boundary_input_abs_mean = zero
        self.last_structured_trunk_owner_boundary_color_abs_mean = zero
        self.last_structured_trunk_owner_boundary_focus_mean = zero
        self.last_structured_trunk_owner_boundary_gate_mean = zero
        self.last_structured_trunk_owner_boundary_takeover_mean = zero
        self.last_structured_trunk_scaffold_abs_mean = zero
        self.last_structured_trunk_coarse_abs_mean = zero
        self.last_structured_trunk_hf_abs_mean = zero
        self.last_structured_trunk_hf_color_abs_mean = zero
        self.last_structured_trunk_hf_gate_mean = zero
        self.last_structured_trunk_hf_local_color_abs_mean = zero
        self.last_structured_trunk_hf_fusion_abs_mean = zero
        self.last_structured_trunk_hf_region_gain_mean = zero
        self.last_structured_trunk_coarse_region_scale_mean = zero
        self.last_structured_trunk_region_support_mean = zero
        self.last_structured_trunk_hf_debug = ''
        self.last_structured_trunk_owner_takeover_debug = ''
        self.last_structured_trunk_owner_boundary_debug = ''

    def _set_empty_forward_stats(self, output, debug='empty_points'):
        zero = output.new_tensor(0.0)
        self.last_structured_trunk_shared_abs_mean = zero
        self.last_structured_trunk_shared_residual_abs_mean = zero
        self.last_structured_trunk_carrier_abs_mean = zero
        self.last_structured_trunk_structure_abs_mean = zero
        self.last_structured_trunk_structure_raw_abs_mean = zero
        self.last_structured_trunk_structure_residual_abs_mean = zero
        self.last_structured_trunk_local_abs_mean = zero
        self.last_structured_trunk_local_raw_abs_mean = zero
        self.last_structured_trunk_local_gate_mean = zero
        self.last_structured_trunk_local_residual_abs_mean = zero
        self.last_structured_trunk_total_abs_mean = zero
        self.last_structured_trunk_debug = debug
        self._set_structured_trunk_output_head_zero_stats(output, debug=debug)
        self.last_detail_residual_abs_mean = zero
        self.last_detail_tiny_repair_abs_mean = zero
        self.last_detail_gate_mean = zero
        self.last_detail_gate_fraction = zero
        self.last_detail_high_freq_residual_abs_mean = zero
        self.last_detail_high_freq_gate_mean = zero
        self.last_detail_high_freq_gate_fraction = zero
        self.last_detail_high_freq_point_gate_mean = zero
        self.last_detail_high_freq_point_gate_fraction = zero
        self.last_detail_high_freq_carrier_abs_mean = zero
        self.last_detail_high_freq_chroma_abs_mean = zero
        self.last_detail_high_freq_luma_abs_mean = zero
        self.last_detail_high_freq_face_abs_mean = zero
        self.last_detail_high_freq_face_raw_abs_mean = zero
        self.last_detail_high_freq_face_after_gate_abs_mean = zero
        self.last_detail_high_freq_face_gate_mean = zero
        self.last_detail_high_freq_face_gate_fraction = zero
        self.last_detail_high_freq_face_point_gate_mean = zero
        self.last_detail_high_freq_face_point_gate_fraction = zero
        self.last_detail_high_freq_face_local_abs_mean = zero
        self.last_detail_high_freq_face_local_raw_abs_mean = zero
        self.last_detail_high_freq_face_extra_local_abs_mean = zero
        self.last_detail_high_freq_face_extra_local_raw_abs_mean = zero
        self.last_detail_high_freq_face_extra_local_gate_mean = zero
        self.last_detail_high_freq_face_extra_local_debug = ''
        self.last_detail_high_freq_structure_abs_mean = zero
        self.last_detail_high_freq_structure_raw_abs_mean = zero
        self.last_detail_high_freq_structure_debug = ''
        self.last_detail_high_freq_boundary_floor_mean = zero

    def _apply_structured_trunk_output_head(self, output, gaussians, camera, base_input, iteration=0):
        if self.structured_trunk_output_head_mlp is None:
            return output

        trunk_compose_mode = self.structured_trunk_output_head_compose_mode
        trunk_head_scale = _resolve_scheduled_scalar(
            iteration,
            self.structured_trunk_output_head_scale_cfg,
            default=1.0,
        )
        trunk_head_needs_residual_cap = trunk_compose_mode == 'residual'
        if trunk_head_scale <= 0.0 or (
            trunk_head_needs_residual_cap
            and self.structured_trunk_output_head_max_residual <= 0.0
        ):
            self._set_structured_trunk_output_head_zero_stats(output, debug='disabled_by_scale')
            return output

        trunk_head_inp, trunk_head_debug, trunk_head_extra = (
            self._compose_structured_trunk_output_head_input(
                gaussians,
                camera,
                base_input,
                iteration=iteration,
            )
        )
        if not torch.is_tensor(trunk_head_inp):
            self._set_structured_trunk_output_head_zero_stats(output, debug='invalid_fusion_input')
            return output

        self.last_structured_trunk_head_fusion_abs_mean = (
            trunk_head_inp.detach().abs().mean()
        )
        local_trunk_color = trunk_head_extra.get('local_color', None)
        local_trunk_colors = trunk_head_extra.get('local_colors', {})
        if not isinstance(local_trunk_colors, dict):
            local_trunk_colors = {}
        local_owner_inputs = trunk_head_extra.get('local_owner_inputs', {})
        if not isinstance(local_owner_inputs, dict):
            local_owner_inputs = {}
        local_geometry_raws = trunk_head_extra.get('local_geometry_raws', {})
        if not isinstance(local_geometry_raws, dict):
            local_geometry_raws = {}
        gate_boost = trunk_head_extra.get('gate_boost', None)
        region_gates = trunk_head_extra.get('region_gates', {})
        if not isinstance(region_gates, dict):
            region_gates = {}
        region_supports = trunk_head_extra.get('region_supports', {})
        if not isinstance(region_supports, dict):
            region_supports = {}
        hf_head_inp = trunk_head_extra.get('hf_input', trunk_head_inp)
        if not torch.is_tensor(hf_head_inp):
            hf_head_inp = trunk_head_inp
        self.last_structured_trunk_hf_fusion_abs_mean = (
            hf_head_inp.detach().abs().mean()
        )
        self.last_structured_trunk_hf_debug = str(
            trunk_head_extra.get('hf_debug', '') or ''
        )
        region_support_modulators = {}
        if self.structured_trunk_output_head_region_support_enable:
            region_support_modulators = _build_named_region_support_modulators(
                region_supports,
                output,
                offset=self.structured_trunk_output_head_region_support_offset,
                gain=self.structured_trunk_output_head_region_support_gain,
                power=self.structured_trunk_output_head_region_support_power,
                base_mix=self.structured_trunk_output_head_region_support_base_mix,
                max_value=self.structured_trunk_output_head_region_support_max,
            )
        region_support_stat_names = list(
            dict.fromkeys(
                list(self.structured_trunk_output_head_coarse_region_suppress_cfg.keys())
                + list(self.structured_trunk_output_head_hf_head_region_boost_cfg.keys())
            )
        )
        if not region_support_stat_names:
            region_support_stat_names = list(region_support_modulators.keys())
        region_support_stats = []
        region_support_debug_parts = []
        for region_name in region_support_stat_names:
            region_support_modulator = region_support_modulators.get(str(region_name), None)
            if not torch.is_tensor(region_support_modulator):
                continue
            region_support_stats.append(region_support_modulator.detach().mean())
            region_support_debug_parts.append(
                f"{region_name}={float(region_support_modulator.detach().mean().item()):.4f}"
            )
        if region_support_stats:
            self.last_structured_trunk_region_support_mean = torch.stack(
                region_support_stats,
                dim=0,
            ).mean()
        else:
            self.last_structured_trunk_region_support_mean = output.new_tensor(0.0)
        if self.structured_trunk_output_head_local_color_owner_head_enable:
            (
                local_owner_raw_color,
                local_owner_contrib,
                local_owner_support_mean,
                local_owner_gate_mean,
                local_owner_debug,
            ) = self._compose_structured_trunk_local_color_owner_head(
                output,
                gaussians,
                local_owner_inputs,
                local_trunk_colors,
                local_geometry_raws,
                region_gates,
                region_supports,
                iteration=iteration,
            )
        else:
            (
                local_owner_raw_color,
                local_owner_contrib,
                local_owner_support_mean,
                local_owner_gate_mean,
                local_owner_debug,
            ) = self._compose_structured_trunk_local_color_owner(
                output,
                local_trunk_colors,
                local_trunk_color,
                region_gates,
                region_support_modulators,
                iteration=iteration,
            )
        self.last_structured_trunk_owner_color_abs_mean = (
            local_owner_raw_color.detach().abs().mean()
            if torch.is_tensor(local_owner_raw_color)
            else output.new_tensor(0.0)
        )
        self.last_structured_trunk_owner_abs_mean = (
            local_owner_contrib.detach().abs().mean()
            if torch.is_tensor(local_owner_contrib)
            else output.new_tensor(0.0)
        )
        self.last_structured_trunk_owner_support_mean = local_owner_support_mean
        self.last_structured_trunk_owner_gate_mean = local_owner_gate_mean
        (
            local_owner_takeover_mix,
            _local_owner_takeover_support_mean,
            local_owner_takeover_debug,
        ) = self._compose_structured_trunk_local_color_owner_takeover_mix(
            output,
            gaussians,
            region_gates,
            region_supports,
            iteration=iteration,
        )
        self.last_structured_trunk_owner_takeover_mean = (
            local_owner_takeover_mix.detach().mean()
            if torch.is_tensor(local_owner_takeover_mix)
            else output.new_tensor(0.0)
        )
        (
            local_owner_legacy_scale,
            local_owner_legacy_scale_mean,
            local_owner_legacy_debug,
        ) = self._compose_structured_trunk_local_color_owner_takeover_legacy_scale(
            output,
            local_owner_takeover_mix,
        )
        self.last_structured_trunk_owner_takeover_legacy_scale_mean = (
            local_owner_legacy_scale_mean
        )
        self.last_structured_trunk_owner_takeover_debug = (
            local_owner_takeover_debug
        )

        if not self.structured_trunk_output_head_dual_head_enable:
            self.last_structured_trunk_scaffold_abs_mean = output.new_tensor(0.0)
            self.last_structured_trunk_coarse_abs_mean = output.new_tensor(0.0)
            self.last_structured_trunk_coarse_region_scale_mean = output.new_tensor(1.0)
            self.last_structured_trunk_hf_abs_mean = output.new_tensor(0.0)
            self.last_structured_trunk_hf_color_abs_mean = output.new_tensor(0.0)
            self.last_structured_trunk_hf_gate_mean = output.new_tensor(0.0)
            self.last_structured_trunk_hf_local_color_abs_mean = output.new_tensor(0.0)
            self.last_structured_trunk_hf_region_gain_mean = output.new_tensor(1.0)
            self.last_structured_trunk_head_debug = (
                f"compose={trunk_compose_mode} | {trunk_head_debug}"
            )
            if region_support_debug_parts:
                self.last_structured_trunk_head_debug = (
                    f"{self.last_structured_trunk_head_debug} | support="
                    f"{','.join(region_support_debug_parts)}"
                )
            if local_owner_debug:
                self.last_structured_trunk_head_debug = (
                    f"{self.last_structured_trunk_head_debug} | owner={local_owner_debug}"
                )
            if local_owner_takeover_debug:
                self.last_structured_trunk_head_debug = (
                    f"{self.last_structured_trunk_head_debug} | takeover={local_owner_takeover_debug}"
                )
            if local_owner_legacy_debug:
                self.last_structured_trunk_head_debug = (
                    f"{self.last_structured_trunk_head_debug} | {local_owner_legacy_debug}"
                )

            trunk_head_logits = self.structured_trunk_output_head_mlp(trunk_head_inp)
            trunk_head_color = self._decode_structured_trunk_head_color(
                trunk_head_logits,
                self.structured_trunk_output_head_mode,
                trunk_compose_mode,
                trunk_head_scale,
                self.structured_trunk_output_head_max_residual,
                self.structured_trunk_output_head_chroma_center,
                self.structured_trunk_output_head_band_luma_scale,
                self.structured_trunk_output_head_band_chroma_scale,
            )
            if self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_apply_to_coarse:
                trunk_head_color = trunk_head_color * local_owner_legacy_scale.to(
                    device=trunk_head_color.device,
                    dtype=trunk_head_color.dtype,
                )
            if (
                torch.is_tensor(local_trunk_color)
                and not self.structured_trunk_output_head_local_color_owner_enable
            ):
                trunk_head_color = trunk_head_color + local_trunk_color
                self.last_structured_trunk_head_local_color_abs_mean = (
                    local_trunk_color.detach().abs().mean()
                )
            else:
                self.last_structured_trunk_head_local_color_abs_mean = (
                    output.new_tensor(0.0)
                )
            self.last_structured_trunk_head_color_abs_mean = (
                trunk_head_color.detach().abs().mean()
            )

            trunk_head_gate, gate_boost_mean = self._compute_structured_trunk_head_gate(
                output,
                trunk_head_inp,
                self.structured_trunk_output_head_gate_mlp,
                self.structured_trunk_output_head_gate_gain,
                self.structured_trunk_output_head_gate_bias,
                self.structured_trunk_output_head_min_gate,
                self.structured_trunk_output_head_max_gate,
                gate_boost=gate_boost,
            )
            trunk_head_contrib = trunk_head_color * trunk_head_gate
            output = output + trunk_head_contrib
            combined_trunk_contrib = trunk_head_contrib
            combined_trunk_color = trunk_head_color
            if (
                self.structured_trunk_output_head_local_color_owner_enable
                and torch.is_tensor(local_owner_contrib)
            ):
                if self.structured_trunk_output_head_local_color_owner_head_takeover_enable:
                    takeover_mix = local_owner_takeover_mix.to(
                        device=output.device,
                        dtype=output.dtype,
                    )
                    if takeover_mix.dim() == 1:
                        takeover_mix = takeover_mix.unsqueeze(-1)
                    output = output + (local_owner_contrib - trunk_head_contrib) * takeover_mix
                    combined_trunk_contrib = (
                        trunk_head_contrib * (1.0 - takeover_mix)
                        + local_owner_contrib * takeover_mix
                    )
                    if torch.is_tensor(local_owner_raw_color):
                        combined_trunk_color = (
                            trunk_head_color * (1.0 - takeover_mix)
                            + local_owner_raw_color * takeover_mix
                        )
                else:
                    output = output + local_owner_contrib
                    combined_trunk_contrib = combined_trunk_contrib + local_owner_contrib
                    combined_trunk_color = combined_trunk_color + local_owner_raw_color
                self.last_structured_trunk_head_local_color_abs_mean = (
                    self.last_structured_trunk_owner_abs_mean
                )
            self.last_structured_trunk_head_abs_mean = (
                combined_trunk_contrib.detach().abs().mean()
            )
            self.last_structured_trunk_head_color_abs_mean = (
                combined_trunk_color.detach().abs().mean()
            )
            self.last_structured_trunk_head_gate_mean = trunk_head_gate.detach().mean()
            self.last_structured_trunk_head_gate_boost_mean = gate_boost_mean
            if self.structured_trunk_output_head_disable_input_residual:
                self.last_structured_trunk_total_abs_mean = (
                    self.last_structured_trunk_head_abs_mean
                )
            return output

        base_scaffold_scale = _resolve_scheduled_scalar(
            iteration,
            self.structured_trunk_output_head_base_scaffold_scale_cfg,
            default=1.0,
        )
        coarse_scale = _resolve_scheduled_scalar(
            iteration,
            self.structured_trunk_output_head_coarse_scale_cfg,
            default=1.0,
        )
        output = output * base_scaffold_scale
        self.last_structured_trunk_scaffold_abs_mean = output.detach().abs().mean()
        self.last_structured_trunk_head_debug = f"compose=dual_head | {trunk_head_debug}"
        if self.last_structured_trunk_hf_debug:
            self.last_structured_trunk_head_debug = (
                f"{self.last_structured_trunk_head_debug} | hf_in={self.last_structured_trunk_hf_debug}"
            )
        if region_support_debug_parts:
            self.last_structured_trunk_head_debug = (
                f"{self.last_structured_trunk_head_debug} | support="
                f"{','.join(region_support_debug_parts)}"
            )
        if local_owner_debug:
            self.last_structured_trunk_head_debug = (
                f"{self.last_structured_trunk_head_debug} | owner={local_owner_debug}"
            )
        if local_owner_takeover_debug:
            self.last_structured_trunk_head_debug = (
                f"{self.last_structured_trunk_head_debug} | takeover={local_owner_takeover_debug}"
            )
        if local_owner_legacy_debug:
            self.last_structured_trunk_head_debug = (
                f"{self.last_structured_trunk_head_debug} | {local_owner_legacy_debug}"
            )

        coarse_logits = self.structured_trunk_output_head_mlp(trunk_head_inp)
        coarse_color = self._decode_structured_trunk_head_color(
            coarse_logits,
            self.structured_trunk_output_head_mode,
            trunk_compose_mode,
            trunk_head_scale * coarse_scale,
            self.structured_trunk_output_head_max_residual,
            self.structured_trunk_output_head_chroma_center,
            self.structured_trunk_output_head_band_luma_scale,
            self.structured_trunk_output_head_band_chroma_scale,
        )
        if self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_apply_to_coarse:
            coarse_color = coarse_color * local_owner_legacy_scale.to(
                device=coarse_color.device,
                dtype=coarse_color.dtype,
            )
        coarse_region_scale = _compose_named_region_scalar(
            region_gates,
            self.structured_trunk_output_head_coarse_region_suppress_cfg,
            output,
            base=1.0,
            additive=False,
            min_value=self.structured_trunk_output_head_coarse_region_min_scale,
            named_modulators=region_support_modulators,
        )
        coarse_color = coarse_color * coarse_region_scale
        self.last_structured_trunk_coarse_region_scale_mean = (
            coarse_region_scale.detach().mean()
        )
        output = output + coarse_color
        self.last_structured_trunk_coarse_abs_mean = coarse_color.detach().abs().mean()

        hf_contrib = torch.zeros_like(output)
        hf_color = None
        hf_gate = None
        if torch.is_tensor(local_trunk_color) and self.structured_trunk_output_head_hf_head_use_local_color:
            hf_local_color_scale = _resolve_scheduled_scalar(
                iteration,
                self.structured_trunk_output_head_hf_head_local_color_scale_cfg,
                default=1.0,
            )
            if hf_local_color_scale > 0.0:
                local_trunk_color = local_trunk_color * hf_local_color_scale
                self.last_structured_trunk_hf_local_color_abs_mean = (
                    local_trunk_color.detach().abs().mean()
                )
            else:
                local_trunk_color = None
                self.last_structured_trunk_hf_local_color_abs_mean = output.new_tensor(0.0)
        else:
            local_trunk_color = None
            self.last_structured_trunk_hf_local_color_abs_mean = output.new_tensor(0.0)

        if (
            self.structured_trunk_output_head_hf_head_enable
            and self.structured_trunk_output_head_hf_head_mlp is not None
        ):
            hf_scale = _resolve_scheduled_scalar(
                iteration,
                self.structured_trunk_output_head_hf_head_scale_cfg,
                default=1.0,
            )
            hf_compose_mode = self.structured_trunk_output_head_hf_head_compose_mode
            hf_needs_residual_cap = hf_compose_mode == 'residual'
            if hf_scale > 0.0 and (
                not hf_needs_residual_cap
                or self.structured_trunk_output_head_hf_head_max_residual > 0.0
            ):
                hf_logits = self.structured_trunk_output_head_hf_head_mlp(hf_head_inp)
                hf_color = self._decode_structured_trunk_head_color(
                    hf_logits,
                    self.structured_trunk_output_head_hf_head_mode,
                    hf_compose_mode,
                    hf_scale,
                    self.structured_trunk_output_head_hf_head_max_residual,
                    self.structured_trunk_output_head_hf_head_chroma_center,
                    self.structured_trunk_output_head_hf_head_band_luma_scale,
                    self.structured_trunk_output_head_hf_head_band_chroma_scale,
                )
                hf_region_gain = _compose_named_region_scalar(
                    region_gates,
                    self.structured_trunk_output_head_hf_head_region_boost_cfg,
                    output,
                    base=1.0,
                    additive=True,
                    min_value=0.0,
                    max_value=self.structured_trunk_output_head_hf_head_region_boost_max,
                    named_modulators=region_support_modulators,
                )
                hf_color = hf_color * hf_region_gain
                self.last_structured_trunk_hf_region_gain_mean = (
                    hf_region_gain.detach().mean()
                )
                if self.structured_trunk_output_head_local_color_owner_head_takeover_legacy_decay_apply_to_hf:
                    hf_color = hf_color * local_owner_legacy_scale.to(
                        device=hf_color.device,
                        dtype=hf_color.dtype,
                    )
                if torch.is_tensor(local_trunk_color):
                    hf_color = hf_color + local_trunk_color
                gate_module = (
                    self.structured_trunk_output_head_gate_mlp
                    if self.structured_trunk_output_head_hf_head_reuse_output_gate
                    else self.structured_trunk_output_head_hf_head_gate_mlp
                )
                gate_gain = (
                    self.structured_trunk_output_head_gate_gain
                    if self.structured_trunk_output_head_hf_head_reuse_output_gate
                    else self.structured_trunk_output_head_hf_head_gate_gain
                )
                gate_bias = (
                    self.structured_trunk_output_head_gate_bias
                    if self.structured_trunk_output_head_hf_head_reuse_output_gate
                    else self.structured_trunk_output_head_hf_head_gate_bias
                )
                min_gate = (
                    self.structured_trunk_output_head_min_gate
                    if self.structured_trunk_output_head_hf_head_reuse_output_gate
                    else self.structured_trunk_output_head_hf_head_min_gate
                )
                max_gate = (
                    self.structured_trunk_output_head_max_gate
                    if self.structured_trunk_output_head_hf_head_reuse_output_gate
                    else self.structured_trunk_output_head_hf_head_max_gate
                )
                hf_gate, gate_boost_mean = self._compute_structured_trunk_head_gate(
                    output,
                    trunk_head_inp
                    if self.structured_trunk_output_head_hf_head_reuse_output_gate
                    else hf_head_inp,
                    gate_module,
                    gate_gain,
                    gate_bias,
                    min_gate,
                    max_gate,
                    gate_boost=gate_boost,
                )
                hf_contrib = hf_color * hf_gate
                output = output + hf_contrib
                self.last_structured_trunk_hf_color_abs_mean = hf_color.detach().abs().mean()
                self.last_structured_trunk_hf_abs_mean = hf_contrib.detach().abs().mean()
                self.last_structured_trunk_hf_gate_mean = hf_gate.detach().mean()
                self.last_structured_trunk_head_gate_mean = hf_gate.detach().mean()
                self.last_structured_trunk_head_gate_boost_mean = gate_boost_mean
            else:
                self.last_structured_trunk_hf_color_abs_mean = output.new_tensor(0.0)
                self.last_structured_trunk_hf_abs_mean = output.new_tensor(0.0)
                self.last_structured_trunk_hf_gate_mean = output.new_tensor(0.0)
                self.last_structured_trunk_hf_region_gain_mean = output.new_tensor(1.0)
                self.last_structured_trunk_head_gate_mean = output.new_tensor(0.0)
                self.last_structured_trunk_head_gate_boost_mean = output.new_tensor(0.0)
        else:
            self.last_structured_trunk_hf_color_abs_mean = output.new_tensor(0.0)
            self.last_structured_trunk_hf_abs_mean = output.new_tensor(0.0)
            self.last_structured_trunk_hf_gate_mean = output.new_tensor(0.0)
            self.last_structured_trunk_hf_region_gain_mean = output.new_tensor(1.0)
            self.last_structured_trunk_head_gate_mean = output.new_tensor(0.0)
            self.last_structured_trunk_head_gate_boost_mean = output.new_tensor(0.0)

        combined_trunk_contrib = coarse_color + hf_contrib
        combined_trunk_color = coarse_color
        if torch.is_tensor(hf_color):
            combined_trunk_color = combined_trunk_color + hf_color
        if (
            self.structured_trunk_output_head_local_color_owner_enable
            and torch.is_tensor(local_owner_contrib)
        ):
            if self.structured_trunk_output_head_local_color_owner_head_takeover_enable:
                takeover_mix = local_owner_takeover_mix.to(
                    device=output.device,
                    dtype=output.dtype,
                )
                if takeover_mix.dim() == 1:
                    takeover_mix = takeover_mix.unsqueeze(-1)
                output = output + (local_owner_contrib - combined_trunk_contrib) * takeover_mix
                combined_trunk_contrib = (
                    combined_trunk_contrib * (1.0 - takeover_mix)
                    + local_owner_contrib * takeover_mix
                )
                if torch.is_tensor(local_owner_raw_color):
                    combined_trunk_color = (
                        combined_trunk_color * (1.0 - takeover_mix)
                        + local_owner_raw_color * takeover_mix
                    )
            else:
                output = output + local_owner_contrib
                combined_trunk_contrib = combined_trunk_contrib + local_owner_contrib
                combined_trunk_color = combined_trunk_color + local_owner_raw_color
            self.last_structured_trunk_head_local_color_abs_mean = (
                self.last_structured_trunk_owner_abs_mean
            )
        else:
            self.last_structured_trunk_head_local_color_abs_mean = (
                self.last_structured_trunk_hf_local_color_abs_mean
            )
        self.last_structured_trunk_head_abs_mean = (
            combined_trunk_contrib.detach().abs().mean()
        )
        self.last_structured_trunk_head_color_abs_mean = (
            combined_trunk_color.detach().abs().mean()
        )
        if self.structured_trunk_output_head_disable_input_residual:
            self.last_structured_trunk_total_abs_mean = (
                self.last_structured_trunk_head_abs_mean
            )
        return output

    def _resolve_face_init_source_module(self, source_name):
        source_name = str(source_name or 'none').lower()
        if source_name in ('none', '', 'off', 'false'):
            return None, source_name
        if source_name in ('trunk_hf_head', 'structured_trunk_hf_head', 'hf_head'):
            return self.structured_trunk_output_head_hf_head_mlp, 'trunk_hf_head'
        if source_name in ('trunk_output_head', 'output_head', 'trunk_coarse', 'coarse_head'):
            return self.structured_trunk_output_head_mlp, 'trunk_output_head'
        if source_name in ('luma_branch', 'shared_luma', 'luma'):
            return self.detail_high_freq_luma_mlp, 'luma_branch'
        if source_name in ('shared_chroma', 'high_frequency', 'detail_high_freq_mlp', 'chroma'):
            return self.detail_high_freq_mlp, 'shared_chroma'
        if source_name in ('face_branch', 'face'):
            return self.detail_high_freq_face_mlp, 'face_branch'
        return None, source_name

    def _resolve_face_gate_init_source_module(self, source_name):
        source_name = str(source_name or 'none').lower()
        if source_name in ('none', '', 'off', 'false'):
            return None, source_name
        if source_name in ('trunk_hf_gate', 'structured_trunk_hf_gate', 'hf_gate'):
            if (
                self.structured_trunk_output_head_hf_head_reuse_output_gate
                or self.structured_trunk_output_head_hf_head_gate_mlp is None
            ):
                return self.structured_trunk_output_head_gate_mlp, 'trunk_hf_gate'
            return self.structured_trunk_output_head_hf_head_gate_mlp, 'trunk_hf_gate'
        if source_name in ('trunk_output_gate', 'output_gate', 'coarse_gate'):
            return self.structured_trunk_output_head_gate_mlp, 'trunk_output_gate'
        if source_name in ('gate_mlp', 'shared_gate', 'detail_high_freq_gate_mlp'):
            return self.detail_high_freq_gate_mlp, 'gate_mlp'
        if source_name in ('face_gate', 'face_branch'):
            return self.detail_high_freq_face_gate_mlp, 'face_gate'
        return None, source_name

    def on_partial_load(self, missing_keys=None, unexpected_keys=None):
        del unexpected_keys

        self.last_partial_load_init_events = []
        missing_keys = set(missing_keys or [])
        if not missing_keys:
            return

        if self.detail_high_freq_face_branch_enable:
            face_mlp_missing = any('detail_high_freq_face_mlp.' in key for key in missing_keys)
            face_gate_missing = any('detail_high_freq_face_gate_mlp.' in key for key in missing_keys)

            if not self.detail_high_freq_face_init_missing_only:
                face_mlp_missing = True
                face_gate_missing = True

            if face_mlp_missing and self.detail_high_freq_face_mlp is not None:
                src_module, src_name = self._resolve_face_init_source_module(
                    self.detail_high_freq_face_init_from
                )
                if src_module is not None and src_module is not self.detail_high_freq_face_mlp:
                    copied, reason = _copy_mlp_parameters(src_module, self.detail_high_freq_face_mlp)
                    if copied:
                        event = (
                            f"face_mlp<-{src_name}"
                            f"(channels={self.detail_high_freq_face_channels})"
                        )
                        self.last_partial_load_init_events.append(event)
                        print(f"[ClarityInit] {event}", flush=True)
                    else:
                        print(
                            f"[ClarityInit] face_mlp init from {src_name} skipped ({reason})",
                            flush=True,
                        )

            if face_gate_missing and self.detail_high_freq_face_gate_mlp is not None:
                src_module, src_name = self._resolve_face_gate_init_source_module(
                    self.detail_high_freq_face_gate_init_from
                )
                if src_module is not None and src_module is not self.detail_high_freq_face_gate_mlp:
                    copied, reason = _copy_mlp_parameters(src_module, self.detail_high_freq_face_gate_mlp)
                    if copied:
                        event = f"face_gate<-{src_name}"
                        self.last_partial_load_init_events.append(event)
                        print(f"[ClarityInit] {event}", flush=True)
                    else:
                        print(
                            f"[ClarityInit] face_gate init from {src_name} skipped ({reason})",
                            flush=True,
                        )

            for local_cfg in self.detail_high_freq_face_extra_local_cfgs:
                local_key = str(local_cfg.get('_key', ''))
                if local_key not in self.detail_high_freq_face_extra_local_projs:
                    continue
                local_missing = any(
                    f'detail_high_freq_face_extra_local_projs.{local_key}.' in key
                    for key in missing_keys
                )
                if not local_missing or self.detail_high_freq_face_local_proj is None:
                    continue
                local_proj = self.detail_high_freq_face_extra_local_projs[local_key]
                copied = _copy_linear_layer_params(
                    self.detail_high_freq_face_local_proj,
                    local_proj,
                    allow_output_adapter=False,
                )
                if copied:
                    event = f"extra_local_proj[{local_key}]<-face_local_proj"
                    self.last_partial_load_init_events.append(event)
                    print(f"[ClarityInit] {event}", flush=True)
                else:
                    print(
                        f"[ClarityInit] extra_local_proj[{local_key}] init from face_local_proj skipped (shape_mismatch)",
                        flush=True,
                    )

        if (
            self.structured_trunk_shared_mlp is not None
            and any('structured_trunk_shared_mlp.' in key for key in missing_keys)
        ):
            event = 'trunk_shared_mlp=zero_init'
            self.last_partial_load_init_events.append(event)
            print(f"[ClarityInit] {event}", flush=True)
        if (
            self.structured_trunk_structure_mlp is not None
            and any('structured_trunk_structure_mlp.' in key for key in missing_keys)
        ):
            event = 'trunk_structure_mlp=zero_init'
            self.last_partial_load_init_events.append(event)
            print(f"[ClarityInit] {event}", flush=True)
        local_mlp_keys = sorted(
            {
                key.split('structured_trunk_local_mlps.', 1)[1].split('.', 1)[0]
                for key in missing_keys
                if 'structured_trunk_local_mlps.' in key
            }
        )
        if local_mlp_keys:
            event = f"trunk_local_mlp=zero_init[{','.join(local_mlp_keys)}]"
            self.last_partial_load_init_events.append(event)
            print(f"[ClarityInit] {event}", flush=True)
        local_color_keys = sorted(
            {
                key.split('structured_trunk_output_head_local_color_mlps.', 1)[1].split('.', 1)[0]
                for key in missing_keys
                if 'structured_trunk_output_head_local_color_mlps.' in key
            }
        )
        for local_key in local_color_keys:
            copied, reason = self._init_structured_trunk_local_color_from_output_head(local_key)
            if copied:
                event = f"trunk_local_color[{local_key}]<-{reason}"
                self.last_partial_load_init_events.append(event)
                print(f"[ClarityInit] {event}", flush=True)
            else:
                print(
                    f"[ClarityInit] trunk_local_color[{local_key}] init skipped ({reason})",
                    flush=True,
                )
        owner_head_keys = sorted(
            {
                key.split('structured_trunk_output_head_local_color_owner_head_mlps.', 1)[1].split('.', 1)[0]
                for key in missing_keys
                if 'structured_trunk_output_head_local_color_owner_head_mlps.' in key
            }
        )
        for local_key in owner_head_keys:
            copied, reason = self._init_structured_trunk_owner_head_from_local_color(local_key)
            if copied:
                event = f"trunk_owner_head[{local_key}]<-{reason}"
                self.last_partial_load_init_events.append(event)
                print(f"[ClarityInit] {event}", flush=True)
            else:
                print(
                    f"[ClarityInit] trunk_owner_head[{local_key}] init skipped ({reason})",
                    flush=True,
                )
        owner_gate_keys = sorted(
            {
                key.split('structured_trunk_output_head_local_color_owner_head_gate_mlps.', 1)[1].split('.', 1)[0]
                for key in missing_keys
                if 'structured_trunk_output_head_local_color_owner_head_gate_mlps.' in key
            }
        )
        for local_key in owner_gate_keys:
            copied, reason = self._init_structured_trunk_owner_gate_from_output_gate(local_key)
            if copied:
                event = f"trunk_owner_gate[{local_key}]<-{reason}"
                self.last_partial_load_init_events.append(event)
                print(f"[ClarityInit] {event}", flush=True)
            else:
                print(
                    f"[ClarityInit] trunk_owner_gate[{local_key}] init skipped ({reason})",
                    flush=True,
                )
        owner_boundary_head_keys = sorted(
            {
                key.split('structured_trunk_output_head_local_color_owner_head_boundary_mlps.', 1)[1].split('.', 1)[0]
                for key in missing_keys
                if 'structured_trunk_output_head_local_color_owner_head_boundary_mlps.' in key
            }
        )
        for local_key in owner_boundary_head_keys:
            copied, reason = self._init_structured_trunk_owner_boundary_head_from_local_color(local_key)
            if copied:
                event = f"trunk_owner_boundary_head[{local_key}]<-{reason}"
                self.last_partial_load_init_events.append(event)
                print(f"[ClarityInit] {event}", flush=True)
            else:
                print(
                    f"[ClarityInit] trunk_owner_boundary_head[{local_key}] init skipped ({reason})",
                    flush=True,
                )
        owner_boundary_gate_keys = sorted(
            {
                key.split('structured_trunk_output_head_local_color_owner_head_boundary_gate_mlps.', 1)[1].split('.', 1)[0]
                for key in missing_keys
                if 'structured_trunk_output_head_local_color_owner_head_boundary_gate_mlps.' in key
            }
        )
        for local_key in owner_boundary_gate_keys:
            copied, reason = self._init_structured_trunk_owner_boundary_gate_from_output_gate(local_key)
            if copied:
                event = f"trunk_owner_boundary_gate[{local_key}]<-{reason}"
                self.last_partial_load_init_events.append(event)
                print(f"[ClarityInit] {event}", flush=True)
            else:
                print(
                    f"[ClarityInit] trunk_owner_boundary_gate[{local_key}] init skipped ({reason})",
                    flush=True,
                )
        hf_head_missing = any(
            'structured_trunk_output_head_hf_head_mlp.' in key
            for key in missing_keys
        )
        if hf_head_missing:
            copied, reason = self._init_structured_trunk_hf_head_from_output_head()
            if copied:
                event = f"trunk_hf_head<-{reason}"
                self.last_partial_load_init_events.append(event)
                print(f"[ClarityInit] {event}", flush=True)
            else:
                print(
                    f"[ClarityInit] trunk_hf_head init skipped ({reason})",
                    flush=True,
                )
        hf_gate_missing = any(
            'structured_trunk_output_head_hf_head_gate_mlp.' in key
            for key in missing_keys
        )
        if hf_gate_missing:
            copied, reason = self._init_structured_trunk_hf_gate_from_output_gate()
            if copied:
                event = f"trunk_hf_gate<-{reason}"
                self.last_partial_load_init_events.append(event)
                print(f"[ClarityInit] {event}", flush=True)
            else:
                print(
                    f"[ClarityInit] trunk_hf_gate init skipped ({reason})",
                    flush=True,
                )

    def _reset_texture_debug_state(self, iteration):
        self.last_structured_trunk_shared_abs_mean = None
        self.last_structured_trunk_shared_residual_abs_mean = None
        self.last_structured_trunk_carrier_abs_mean = None
        self.last_structured_trunk_structure_abs_mean = None
        self.last_structured_trunk_structure_raw_abs_mean = None
        self.last_structured_trunk_structure_residual_abs_mean = None
        self.last_structured_trunk_local_abs_mean = None
        self.last_structured_trunk_local_raw_abs_mean = None
        self.last_structured_trunk_local_gate_mean = None
        self.last_structured_trunk_local_residual_abs_mean = None
        self.last_structured_trunk_total_abs_mean = None
        self.last_structured_trunk_debug = ''
        self.last_structured_trunk_head_abs_mean = None
        self.last_structured_trunk_head_color_abs_mean = None
        self.last_structured_trunk_head_gate_mean = None
        self.last_structured_trunk_head_gate_boost_mean = None
        self.last_structured_trunk_head_local_color_abs_mean = None
        self.last_structured_trunk_head_fusion_abs_mean = None
        self.last_structured_trunk_head_debug = ''
        self.last_structured_trunk_owner_abs_mean = None
        self.last_structured_trunk_owner_input_abs_mean = None
        self.last_structured_trunk_owner_color_abs_mean = None
        self.last_structured_trunk_owner_support_mean = None
        self.last_structured_trunk_owner_gate_mean = None
        self.last_structured_trunk_owner_takeover_mean = None
        self.last_structured_trunk_owner_takeover_legacy_scale_mean = None
        self.last_structured_trunk_owner_boundary_abs_mean = None
        self.last_structured_trunk_owner_boundary_input_abs_mean = None
        self.last_structured_trunk_owner_boundary_color_abs_mean = None
        self.last_structured_trunk_owner_boundary_focus_mean = None
        self.last_structured_trunk_owner_boundary_gate_mean = None
        self.last_structured_trunk_owner_boundary_takeover_mean = None
        self.last_structured_trunk_scaffold_abs_mean = None
        self.last_structured_trunk_coarse_abs_mean = None
        self.last_structured_trunk_hf_abs_mean = None
        self.last_structured_trunk_hf_color_abs_mean = None
        self.last_structured_trunk_hf_gate_mean = None
        self.last_structured_trunk_hf_local_color_abs_mean = None
        self.last_structured_trunk_hf_fusion_abs_mean = None
        self.last_structured_trunk_hf_region_gain_mean = None
        self.last_structured_trunk_coarse_region_scale_mean = None
        self.last_structured_trunk_hf_debug = ''
        self.last_structured_trunk_owner_takeover_debug = ''
        self.last_structured_trunk_owner_boundary_debug = ''
        self.last_detail_scale = 0.0
        self.last_detail_schedule_iteration = int(iteration)
        self.last_detail_residual_abs_mean = None
        self.last_detail_tiny_repair_abs_mean = None
        self.last_detail_gate_mean = None
        self.last_detail_gate_fraction = None
        self.last_detail_high_freq_scale = 0.0
        self.last_detail_high_freq_residual_abs_mean = None
        self.last_detail_high_freq_gate_mean = None
        self.last_detail_high_freq_gate_fraction = None
        self.last_detail_high_freq_point_gate_mean = None
        self.last_detail_high_freq_point_gate_fraction = None
        self.last_detail_high_freq_carrier_abs_mean = None
        self.last_detail_high_freq_chroma_abs_mean = None
        self.last_detail_high_freq_luma_abs_mean = None
        self.last_detail_high_freq_face_abs_mean = None
        self.last_detail_high_freq_face_raw_abs_mean = None
        self.last_detail_high_freq_face_after_gate_abs_mean = None
        self.last_detail_high_freq_face_gate_mean = None
        self.last_detail_high_freq_face_gate_fraction = None
        self.last_detail_high_freq_face_point_gate_mean = None
        self.last_detail_high_freq_face_point_gate_fraction = None
        self.last_detail_high_freq_face_local_abs_mean = None
        self.last_detail_high_freq_face_local_raw_abs_mean = None
        self.last_detail_high_freq_face_extra_local_abs_mean = None
        self.last_detail_high_freq_face_extra_local_raw_abs_mean = None
        self.last_detail_high_freq_face_extra_local_gate_mean = None
        self.last_detail_high_freq_face_extra_local_debug = ''
        self.last_detail_high_freq_structure_abs_mean = None
        self.last_detail_high_freq_structure_raw_abs_mean = None
        self.last_detail_high_freq_structure_debug = ''
        self.last_detail_high_freq_boundary_floor_mean = None

    def compose_input(self, gaussians, camera, iteration=0):
        features = gaussians.get_features.squeeze(-1)
        n_points = features.shape[0]
        if self.use_xyz:
            aabb = self.metadata["aabb"]
            xyz_norm = aabb.normalize(gaussians.get_xyz, sym=True)
            features = torch.cat([features, xyz_norm], dim=1)
        if self.use_cov:
            cov = gaussians.get_covariance()
            features = torch.cat([features, cov], dim=1)
        if self.use_normal:
            scale = gaussians._scaling
            rot = build_rotation(gaussians._rotation)
            normal = torch.gather(rot, dim=2, index=scale.argmin(1).reshape(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
            features = torch.cat([features, normal], dim=1)
        if self.sh_degree > 0:
            dir_pp = (gaussians.get_xyz - camera.camera_center.repeat(n_points, 1))
            if self.cano_view_dir:
                T_fwd = gaussians.fwd_transform
                R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
                dir_pp = torch.matmul(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
                view_noise_scale = _resolve_scheduled_scalar(iteration, self.cfg.get('view_noise', 0.))
                if self.training and view_noise_scale > 0.:
                    view_noise = torch.tensor(augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
                                              dtype=torch.float32,
                                              device=dir_pp.device).transpose(0, 1)
                    dir_pp = torch.matmul(dir_pp, view_noise)
            dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-12)
            dir_embed = self.sh_embed(dir_pp_normalized)
            features = torch.cat([features, dir_embed], dim=1)
        if self.non_rigid_dim > 0:
            assert hasattr(gaussians, "non_rigid_feature")
            features = torch.cat([features, gaussians.non_rigid_feature], dim=1)
        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx = len(self.frame_dict) - 1
            else:
                latent_idx = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx]).long().to(features.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(features.shape[0], -1)
            features = torch.cat([features, latent_code], dim=1)

        if self.structured_trunk_enable:
            use_structured_trunk_input_residual = not (
                self.structured_trunk_output_head_enable
                and self.structured_trunk_output_head_disable_input_residual
            )
            if use_structured_trunk_input_residual:
                structured_trunk_delta, structured_trunk_debug = self._compose_structured_trunk_delta(
                    gaussians,
                    camera,
                    features,
                    iteration=iteration,
                )
                if torch.is_tensor(structured_trunk_delta):
                    features = features + structured_trunk_delta
                    self.last_structured_trunk_total_abs_mean = (
                        structured_trunk_delta.detach().abs().mean()
                    )
                elif torch.is_tensor(features):
                    self.last_structured_trunk_total_abs_mean = features.new_tensor(0.0)
                self.last_structured_trunk_debug = structured_trunk_debug
            elif torch.is_tensor(features):
                self.last_structured_trunk_total_abs_mean = features.new_tensor(0.0)
                self.last_structured_trunk_debug = 'input_residual=disabled_for_output_head'

        return features


    def forward(self, gaussians, camera, iteration=0):
        self._reset_texture_debug_state(iteration)
        point_count = int(gaussians.get_xyz.shape[0])
        if point_count <= 0:
            zero_output = gaussians.get_xyz.new_zeros((0, 3))
            self._set_empty_forward_stats(zero_output, debug='empty_points')
            return self.color_activation(zero_output)
        inp = self.compose_input(gaussians, camera, iteration=iteration)
        output = self.mlp(inp)
        if self.structured_trunk_output_head_mlp is not None:
            output = self._apply_structured_trunk_output_head(
                output,
                gaussians,
                camera,
                inp,
                iteration=iteration,
            )
            if self.last_structured_trunk_debug:
                self.last_structured_trunk_debug = (
                    f"{self.last_structured_trunk_debug} | head={self.last_structured_trunk_head_debug}"
                )
            else:
                self.last_structured_trunk_debug = f"head={self.last_structured_trunk_head_debug}"
        tiny_repair = torch.zeros_like(output)
        has_tiny_repair = False
        if self.detail_mlp is not None:
            detail_schedule_iteration = self._resolve_detail_schedule_iteration(iteration)
            self.last_detail_schedule_iteration = int(detail_schedule_iteration)
            detail_scale = _resolve_scheduled_scalar(detail_schedule_iteration, self.detail_scale_cfg, default=1.0)
            detail_tiny_repair_scale = _resolve_scheduled_scalar(
                detail_schedule_iteration,
                self.detail_tiny_repair_scale_cfg,
                default=1.0,
            )
            self.last_detail_scale = float(detail_scale)
            detail_gate = self._build_detail_point_gate(gaussians)
            if torch.is_tensor(detail_gate):
                detail_gate = detail_gate.to(device=output.device, dtype=output.dtype)
                self.last_detail_gate_mean = detail_gate.detach().mean()
                self.last_detail_gate_fraction = (detail_gate.detach() > 0.0).float().mean()
            if (
                detail_scale > 0.0
                and self.detail_max_residual > 0.0
                and detail_tiny_repair_scale > 0.0
            ):
                detail_logits = self.detail_mlp(inp)
                detail_residual = torch.tanh(detail_logits) * (self.detail_max_residual * detail_scale)
                if torch.is_tensor(detail_gate):
                    detail_residual = detail_residual * detail_gate.unsqueeze(-1)
                detail_residual = detail_residual * detail_tiny_repair_scale
                tiny_repair = tiny_repair + detail_residual
                has_tiny_repair = True
                self.last_detail_residual_abs_mean = detail_residual.detach().abs().mean()
            else:
                self.last_detail_residual_abs_mean = output.new_tensor(0.0)
        if self.detail_high_freq_mlp is not None:
            detail_high_freq_schedule_iteration = self._resolve_detail_high_freq_schedule_iteration(iteration)
            self.last_detail_schedule_iteration = int(
                max(self.last_detail_schedule_iteration, detail_high_freq_schedule_iteration)
            )
            detail_high_freq_scale = _resolve_scheduled_scalar(
                detail_high_freq_schedule_iteration,
                self.detail_high_freq_scale_cfg,
                default=1.0,
            )
            detail_high_freq_tiny_repair_scale = _resolve_scheduled_scalar(
                detail_high_freq_schedule_iteration,
                self.detail_high_freq_tiny_repair_scale_cfg,
                default=1.0,
            )
            self.last_detail_high_freq_scale = float(detail_high_freq_scale)
            (
                detail_high_freq_boundary_floor,
                self.last_detail_high_freq_boundary_floor_mean,
            ) = self._build_detail_high_freq_boundary_floor(
                gaussians,
                output,
                iteration=detail_high_freq_schedule_iteration,
            )
            detail_gate = self._build_detail_point_gate(gaussians)
            if torch.is_tensor(detail_gate):
                detail_gate = detail_gate.to(device=output.device, dtype=output.dtype)
            detail_high_freq_point_gate = self._build_detail_high_freq_point_gate(
                gaussians,
                base_gate=detail_gate,
            )
            if torch.is_tensor(detail_high_freq_point_gate):
                detail_high_freq_point_gate = detail_high_freq_point_gate.to(
                    device=output.device,
                    dtype=output.dtype,
                )
                self.last_detail_high_freq_point_gate_mean = detail_high_freq_point_gate.detach().mean()
                self.last_detail_high_freq_point_gate_fraction = (
                    detail_high_freq_point_gate.detach() > 0.0
                ).float().mean()
            detail_high_freq_face_point_gate = None
            if self.detail_high_freq_face_point_gate_enable and self.detail_high_freq_face_point_gate_cfg is not None:
                detail_high_freq_face_point_gate = self._build_point_gate(
                    gaussians,
                    self.detail_high_freq_face_point_gate_cfg,
                )
                if torch.is_tensor(detail_high_freq_face_point_gate):
                    detail_high_freq_face_point_gate = detail_high_freq_face_point_gate.to(
                        device=output.device,
                        dtype=output.dtype,
                    )
                    self.last_detail_high_freq_face_point_gate_mean = (
                        detail_high_freq_face_point_gate.detach().mean()
                    )
                    self.last_detail_high_freq_face_point_gate_fraction = (
                        detail_high_freq_face_point_gate.detach() > 0.0
                    ).float().mean()
            if (
                detail_high_freq_scale > 0.0
                and self.detail_high_freq_max_residual > 0.0
                and (
                    detail_high_freq_tiny_repair_scale > 0.0
                    or torch.is_tensor(detail_high_freq_boundary_floor)
                )
            ):
                (
                    detail_high_freq_inp,
                    carrier_monitor,
                    structure_raw,
                    structure_delta,
                    structure_debug,
                ) = self._compose_detail_high_frequency_input(
                    gaussians,
                    camera,
                    inp,
                    iteration=iteration,
                )
                self.last_detail_high_freq_carrier_abs_mean = carrier_monitor.detach().abs().mean()
                if torch.is_tensor(structure_raw):
                    self.last_detail_high_freq_structure_raw_abs_mean = (
                        structure_raw.detach().abs().mean()
                    )
                else:
                    self.last_detail_high_freq_structure_raw_abs_mean = output.new_tensor(0.0)
                if torch.is_tensor(structure_delta):
                    self.last_detail_high_freq_structure_abs_mean = (
                        structure_delta.detach().abs().mean()
                    )
                else:
                    self.last_detail_high_freq_structure_abs_mean = output.new_tensor(0.0)
                self.last_detail_high_freq_structure_debug = structure_debug
                detail_high_freq_amplitude = self.detail_high_freq_max_residual * detail_high_freq_scale
                detail_high_freq_logits = self.detail_high_freq_mlp(detail_high_freq_inp)
                detail_high_freq_chroma = torch.tanh(detail_high_freq_logits)
                if self.detail_high_freq_chroma_center:
                    detail_high_freq_chroma = detail_high_freq_chroma - detail_high_freq_chroma.mean(
                        dim=-1,
                        keepdim=True,
                    )
                detail_high_freq_chroma = detail_high_freq_chroma * (
                    detail_high_freq_amplitude * self.detail_high_freq_chroma_scale
                )
                self.last_detail_high_freq_chroma_abs_mean = detail_high_freq_chroma.detach().abs().mean()
                detail_high_freq_residual = detail_high_freq_chroma
                if self.detail_high_freq_luma_mlp is not None:
                    detail_high_freq_luma = torch.tanh(
                        self.detail_high_freq_luma_mlp(detail_high_freq_inp)
                    ) * (
                        detail_high_freq_amplitude * self.detail_high_freq_luma_scale
                    )
                    detail_high_freq_residual = detail_high_freq_residual + detail_high_freq_luma.expand(-1, 3)
                    self.last_detail_high_freq_luma_abs_mean = detail_high_freq_luma.detach().abs().mean()
                else:
                    self.last_detail_high_freq_luma_abs_mean = output.new_tensor(0.0)
                if self.detail_high_freq_gate_mlp is not None:
                    detail_high_freq_gate = torch.sigmoid(
                        self.detail_high_freq_gate_mlp(detail_high_freq_inp) + self.detail_high_freq_gate_bias
                    )
                    if self.detail_high_freq_min_gate > 0.0:
                        detail_high_freq_gate = (
                            detail_high_freq_gate * (1.0 - self.detail_high_freq_min_gate)
                            + self.detail_high_freq_min_gate
                        )
                else:
                    detail_high_freq_gate = output.new_ones((output.shape[0], 1))
                self.last_detail_high_freq_gate_mean = detail_high_freq_gate.detach().mean()
                self.last_detail_high_freq_gate_fraction = (
                    detail_high_freq_gate.detach() > 0.5
                ).float().mean()
                detail_high_freq_residual = detail_high_freq_residual * detail_high_freq_gate
                if torch.is_tensor(detail_high_freq_point_gate):
                    detail_high_freq_residual = detail_high_freq_residual * detail_high_freq_point_gate.unsqueeze(-1)
                if torch.is_tensor(detail_high_freq_boundary_floor):
                    detail_high_freq_residual_scale = torch.maximum(
                        output.new_full(
                            (output.shape[0], 1),
                            float(detail_high_freq_tiny_repair_scale),
                        ),
                        detail_high_freq_boundary_floor.to(
                            device=output.device,
                            dtype=output.dtype,
                        ),
                    )
                else:
                    detail_high_freq_residual_scale = detail_high_freq_tiny_repair_scale
                detail_high_freq_residual = (
                    detail_high_freq_residual * detail_high_freq_residual_scale
                )
                tiny_repair = tiny_repair + detail_high_freq_residual
                has_tiny_repair = True
                self.last_detail_high_freq_residual_abs_mean = detail_high_freq_residual.detach().abs().mean()
                if self.detail_high_freq_face_mlp is not None and self.detail_high_freq_face_scale > 0.0:
                    detail_high_freq_face_tiny_repair_scale = _resolve_scheduled_scalar(
                        detail_high_freq_schedule_iteration,
                        self.detail_high_freq_face_tiny_repair_scale_cfg,
                        default=detail_high_freq_tiny_repair_scale,
                    )
                    (
                        detail_high_freq_face_inp,
                        detail_high_freq_face_local_raw,
                        detail_high_freq_face_local_delta,
                    ) = self._compose_detail_high_frequency_face_input(
                        gaussians,
                        camera,
                        detail_high_freq_inp,
                        iteration=iteration,
                    )
                    if torch.is_tensor(detail_high_freq_face_local_raw):
                        self.last_detail_high_freq_face_local_raw_abs_mean = (
                            detail_high_freq_face_local_raw.detach().abs().mean()
                        )
                    else:
                        self.last_detail_high_freq_face_local_raw_abs_mean = output.new_tensor(0.0)
                    if torch.is_tensor(detail_high_freq_face_local_delta):
                        self.last_detail_high_freq_face_local_abs_mean = (
                            detail_high_freq_face_local_delta.detach().abs().mean()
                        )
                    else:
                        self.last_detail_high_freq_face_local_abs_mean = output.new_tensor(0.0)
                    if self.last_detail_high_freq_face_extra_local_raw_abs_mean is None:
                        self.last_detail_high_freq_face_extra_local_raw_abs_mean = output.new_tensor(0.0)
                    if self.last_detail_high_freq_face_extra_local_abs_mean is None:
                        self.last_detail_high_freq_face_extra_local_abs_mean = output.new_tensor(0.0)
                    if self.last_detail_high_freq_face_extra_local_gate_mean is None:
                        self.last_detail_high_freq_face_extra_local_gate_mean = output.new_tensor(0.0)
                    detail_high_freq_face_residual = torch.tanh(
                        self.detail_high_freq_face_mlp(detail_high_freq_face_inp)
                    ) * (
                        detail_high_freq_amplitude * self.detail_high_freq_face_scale
                    )
                    self.last_detail_high_freq_face_raw_abs_mean = (
                        detail_high_freq_face_residual.detach().abs().mean()
                    )
                    if self.detail_high_freq_face_gate_mlp is not None:
                        detail_high_freq_face_gate = torch.sigmoid(
                            self.detail_high_freq_face_gate_mlp(detail_high_freq_face_inp)
                            + self.detail_high_freq_face_gate_bias
                        )
                        if self.detail_high_freq_face_min_gate > 0.0:
                            detail_high_freq_face_gate = (
                                detail_high_freq_face_gate * (1.0 - self.detail_high_freq_face_min_gate)
                                + self.detail_high_freq_face_min_gate
                            )
                    else:
                        detail_high_freq_face_gate = output.new_ones((output.shape[0], 1))
                    self.last_detail_high_freq_face_gate_mean = detail_high_freq_face_gate.detach().mean()
                    self.last_detail_high_freq_face_gate_fraction = (
                        detail_high_freq_face_gate.detach() > 0.5
                    ).float().mean()
                    detail_high_freq_face_residual = (
                        detail_high_freq_face_residual * detail_high_freq_face_gate
                    )
                    self.last_detail_high_freq_face_after_gate_abs_mean = (
                        detail_high_freq_face_residual.detach().abs().mean()
                    )
                    if torch.is_tensor(detail_high_freq_face_point_gate):
                        detail_high_freq_face_residual = (
                            detail_high_freq_face_residual
                            * detail_high_freq_face_point_gate.unsqueeze(-1)
                        )
                    if detail_high_freq_face_residual.shape[1] == 1:
                        detail_high_freq_face_residual = detail_high_freq_face_residual.expand(-1, 3)
                    detail_high_freq_face_residual = (
                        detail_high_freq_face_residual
                        * detail_high_freq_face_tiny_repair_scale
                    )
                    tiny_repair = tiny_repair + detail_high_freq_face_residual
                    has_tiny_repair = True
                    self.last_detail_high_freq_face_abs_mean = (
                        detail_high_freq_face_residual.detach().abs().mean()
                    )
                else:
                    self.last_detail_high_freq_face_abs_mean = output.new_tensor(0.0)
                    self.last_detail_high_freq_face_raw_abs_mean = output.new_tensor(0.0)
                    self.last_detail_high_freq_face_after_gate_abs_mean = output.new_tensor(0.0)
                    self.last_detail_high_freq_face_gate_mean = output.new_tensor(0.0)
                    self.last_detail_high_freq_face_gate_fraction = output.new_tensor(0.0)
                    self.last_detail_high_freq_face_local_abs_mean = output.new_tensor(0.0)
                    self.last_detail_high_freq_face_local_raw_abs_mean = output.new_tensor(0.0)
                    self.last_detail_high_freq_face_extra_local_abs_mean = output.new_tensor(0.0)
                    self.last_detail_high_freq_face_extra_local_raw_abs_mean = output.new_tensor(0.0)
                    self.last_detail_high_freq_face_extra_local_gate_mean = output.new_tensor(0.0)
            else:
                self.last_detail_high_freq_residual_abs_mean = output.new_tensor(0.0)
                self.last_detail_high_freq_carrier_abs_mean = output.new_tensor(0.0)
                self.last_detail_high_freq_gate_mean = output.new_tensor(0.0)
                self.last_detail_high_freq_gate_fraction = output.new_tensor(0.0)
                self.last_detail_high_freq_chroma_abs_mean = output.new_tensor(0.0)
                self.last_detail_high_freq_luma_abs_mean = output.new_tensor(0.0)
                self.last_detail_high_freq_face_abs_mean = output.new_tensor(0.0)
                self.last_detail_high_freq_face_raw_abs_mean = output.new_tensor(0.0)
                self.last_detail_high_freq_face_after_gate_abs_mean = output.new_tensor(0.0)
                self.last_detail_high_freq_face_gate_mean = output.new_tensor(0.0)
                self.last_detail_high_freq_face_gate_fraction = output.new_tensor(0.0)
                self.last_detail_high_freq_face_local_abs_mean = output.new_tensor(0.0)
                self.last_detail_high_freq_face_local_raw_abs_mean = output.new_tensor(0.0)
                self.last_detail_high_freq_face_extra_local_abs_mean = output.new_tensor(0.0)
                self.last_detail_high_freq_face_extra_local_raw_abs_mean = output.new_tensor(0.0)
                self.last_detail_high_freq_face_extra_local_gate_mean = output.new_tensor(0.0)
                self.last_detail_high_freq_structure_abs_mean = output.new_tensor(0.0)
                self.last_detail_high_freq_structure_raw_abs_mean = output.new_tensor(0.0)
                self.last_detail_high_freq_structure_debug = ''
                if self.last_detail_high_freq_point_gate_mean is None:
                    self.last_detail_high_freq_point_gate_mean = output.new_tensor(0.0)
                if self.last_detail_high_freq_point_gate_fraction is None:
                    self.last_detail_high_freq_point_gate_fraction = output.new_tensor(0.0)
                if self.last_detail_high_freq_face_point_gate_mean is None:
                    self.last_detail_high_freq_face_point_gate_mean = output.new_tensor(0.0)
                if self.last_detail_high_freq_face_point_gate_fraction is None:
                    self.last_detail_high_freq_face_point_gate_fraction = output.new_tensor(0.0)
                self.last_detail_high_freq_boundary_floor_mean = output.new_tensor(0.0)
        if has_tiny_repair:
            output = output + tiny_repair
            self.last_detail_tiny_repair_abs_mean = tiny_repair.detach().abs().mean()
        else:
            self.last_detail_tiny_repair_abs_mean = output.new_tensor(0.0)
        color = self.color_activation(output)
        return color


def get_texture(cfg, metadata):
    name = cfg.name
    model_dict = {
        "sh2rgb": SH2RGB,
        "mlp": ColorMLP,
    }
    return model_dict[name](cfg, metadata)
