import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from collections.abc import Sequence
from omegaconf import OmegaConf

import trimesh
import igl

from utils.general_utils import build_rotation, get_body_model_misc_path
from utils.pytorch3d_compat import ops
from models.network_utils import VanillaCondMLP, get_skinning_mlp


def transform_points(T, xyz):
    homo_coord = torch.ones(xyz.shape[0], 1, dtype=xyz.dtype, device=xyz.device)
    xyz_homo = torch.cat([xyz, homo_coord], dim=-1).view(xyz.shape[0], 4, 1)
    return torch.matmul(T, xyz_homo)[:, :3, 0]


def compose_pointwise_transform(R, xyz, x_bar):
    T = torch.eye(4, dtype=xyz.dtype, device=xyz.device).unsqueeze(0).repeat(xyz.shape[0], 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = x_bar - torch.bmm(R, xyz.unsqueeze(-1)).squeeze(-1)
    return T


def point_to_segment_distance(points, starts, ends):
    seg = ends - starts
    seg_norm = torch.sum(seg * seg, dim=-1, keepdim=True).clamp_min(1e-8)
    t = torch.sum((points - starts) * seg, dim=-1, keepdim=True) / seg_norm
    t = torch.clamp(t, 0., 1.)
    proj = starts + t * seg
    return torch.norm(points - proj, dim=-1)


def closest_point_on_triangles(points, tri_verts):
    p = points[:, None, :]
    a = tri_verts[:, :, 0]
    b = tri_verts[:, :, 1]
    c = tri_verts[:, :, 2]

    ab = b - a
    ac = c - a
    bc = c - b

    bary = torch.zeros((*tri_verts.shape[:2], 3), dtype=tri_verts.dtype, device=tri_verts.device)
    closest = torch.zeros_like(p.expand(-1, tri_verts.shape[1], -1))
    assigned = torch.zeros(tri_verts.shape[:2], dtype=torch.bool, device=tri_verts.device)

    ap = p - a
    d1 = (ab * ap).sum(dim=-1)
    d2 = (ac * ap).sum(dim=-1)

    mask = (d1 <= 0.0) & (d2 <= 0.0) & (~assigned)
    bary[mask] = bary.new_tensor((1.0, 0.0, 0.0))
    closest[mask] = a[mask]
    assigned |= mask

    bp = p - b
    d3 = (ab * bp).sum(dim=-1)
    d4 = (ac * bp).sum(dim=-1)
    mask = (d3 >= 0.0) & (d4 <= d3) & (~assigned)
    bary[mask] = bary.new_tensor((0.0, 1.0, 0.0))
    closest[mask] = b[mask]
    assigned |= mask

    vc = d1 * d4 - d3 * d2
    v_ab = d1 / (d1 - d3).clamp_min(1e-8)
    bary_ab = torch.stack([1.0 - v_ab, v_ab, torch.zeros_like(v_ab)], dim=-1)
    closest_ab = a + v_ab.unsqueeze(-1) * ab
    mask = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0) & (~assigned)
    bary[mask] = bary_ab[mask]
    closest[mask] = closest_ab[mask]
    assigned |= mask

    cp = p - c
    d5 = (ab * cp).sum(dim=-1)
    d6 = (ac * cp).sum(dim=-1)
    mask = (d6 >= 0.0) & (d5 <= d6) & (~assigned)
    bary[mask] = bary.new_tensor((0.0, 0.0, 1.0))
    closest[mask] = c[mask]
    assigned |= mask

    vb = d5 * d2 - d1 * d6
    w_ac = d2 / (d2 - d6).clamp_min(1e-8)
    bary_ac = torch.stack([1.0 - w_ac, torch.zeros_like(w_ac), w_ac], dim=-1)
    closest_ac = a + w_ac.unsqueeze(-1) * ac
    mask = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0) & (~assigned)
    bary[mask] = bary_ac[mask]
    closest[mask] = closest_ac[mask]
    assigned |= mask

    va = d3 * d6 - d5 * d4
    w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(1e-8)
    bary_bc = torch.stack([torch.zeros_like(w_bc), 1.0 - w_bc, w_bc], dim=-1)
    closest_bc = b + w_bc.unsqueeze(-1) * bc
    mask = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0) & (~assigned)
    bary[mask] = bary_bc[mask]
    closest[mask] = closest_bc[mask]
    assigned |= mask

    denom = (va + vb + vc).clamp_min(1e-8)
    v_face = vb / denom
    w_face = vc / denom
    bary_face = torch.stack([1.0 - v_face - w_face, v_face, w_face], dim=-1)
    closest_face = (
        tri_verts[:, :, 0] * bary_face[..., [0]]
        + tri_verts[:, :, 1] * bary_face[..., [1]]
        + tri_verts[:, :, 2] * bary_face[..., [2]]
    )
    mask = ~assigned
    bary[mask] = bary_face[mask]
    closest[mask] = closest_face[mask]

    dist = torch.norm(p - closest, dim=-1)
    return bary, closest, dist


def weighted_reduce(values, weights):
    return torch.sum(values * weights) / weights.sum().clamp_min(1e-6)


def safe_normalize(v, eps=1e-8):
    return v / torch.norm(v, dim=-1, keepdim=True).clamp_min(eps)


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

    current_value = seq[0]
    idx = 1
    current_iteration = int(iteration)
    while idx + 1 < len(seq):
        step = int(seq[idx])
        next_value = seq[idx + 1]
        if current_iteration >= step:
            current_value = next_value
            idx += 2
        else:
            break
    return current_value


COMPACT_SEMANTIC_NAMES = ('hair', 'face', 'skin', 'upper', 'lower', 'shoes')


def joint_group_mask(joint_ids, members):
    member_tensor = joint_ids.new_tensor(members)
    return (joint_ids.unsqueeze(-1) == member_tensor.unsqueeze(0)).any(dim=-1).float()


def sample_neighbor_consistency(values, positions, k=6, max_samples=2048):
    if values.shape[0] <= 1:
        return values.new_tensor(0.)

    if values.shape[0] > max_samples:
        sample_ids = torch.randperm(values.shape[0], device=values.device)[:max_samples]
        values = values[sample_ids]
        positions = positions[sample_ids]

    k = min(k + 1, values.shape[0])
    if k <= 1:
        return values.new_tensor(0.)

    knn = ops.knn_points(positions.detach().unsqueeze(0), positions.detach().unsqueeze(0), K=k)
    nn_idx = knn.idx[0, :, 1:]
    nn_dists = torch.sqrt(knn.dists[0, :, 1:].clamp_min(1e-8))
    scale = torch.quantile(nn_dists.detach().reshape(-1), 0.5).clamp_min(1e-6)
    weights = torch.exp(-nn_dists / scale)
    neigh = values[nn_idx]
    if values.dim() == 1:
        diff = (values.unsqueeze(1) - neigh).abs()
    else:
        diff = (values.unsqueeze(1) - neigh).abs().mean(dim=-1)
    return (diff * weights).sum() / weights.sum().clamp_min(1e-6)


class RigidDeform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, gaussians, iteration, camera):
        raise NotImplementedError

    def regularization(self):
        return NotImplementedError

class Identity(RigidDeform):
    """ Identity mapping for single frame reconstruction """
    def __init__(self, cfg, metadata):
        super().__init__(cfg)

    def forward(self, gaussians, iteration, camera):
        return gaussians

    def regularization(self):
        return {}

class SMPLNN(RigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.smpl_verts = torch.from_numpy(metadata["smpl_verts"]).float().cuda()
        self.skinning_weights = torch.from_numpy(metadata["skinning_weights"]).float().cuda()

    def query_weights(self, xyz):
        # find the nearest vertex
        knn_ret = ops.knn_points(xyz.unsqueeze(0), self.smpl_verts.unsqueeze(0))
        p_idx = knn_ret.idx.squeeze()
        pts_W = self.skinning_weights[p_idx, :]

        return pts_W

    def forward(self, gaussians, iteration, camera):
        bone_transforms = camera.bone_transforms

        xyz = gaussians.get_xyz
        n_pts = xyz.shape[0]
        pts_W = self.query_weights(xyz)
        T_fwd = torch.matmul(pts_W, bone_transforms.view(-1, 16)).view(n_pts, 4, 4).float()

        deformed_gaussians = gaussians.clone()
        deformed_gaussians.set_fwd_transform(T_fwd.detach())

        homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz.device)
        x_hat_homo = torch.cat([xyz, homo_coord], dim=-1).view(n_pts, 4, 1)
        x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
        deformed_gaussians._xyz = x_bar

        rotation_hat = build_rotation(gaussians._rotation)
        rotation_bar = torch.matmul(T_fwd[:, :3, :3], rotation_hat)
        setattr(deformed_gaussians, 'rotation_precomp', rotation_bar)
        # deformed_gaussians._rotation = tf.matrix_to_quaternion(rotation_bar)
        # deformed_gaussians._rotation = rotation_matrix_to_quaternion(rotation_bar)

        return deformed_gaussians

    def regularization(self):
        return {}

def create_voxel_grid(d, h, w, device='cpu'):
    x_range = (torch.linspace(-1,1,steps=w,device=device)).view(1, 1, 1, w).expand(1, d, h, w)  # [1, H, W, D]
    y_range = (torch.linspace(-1,1,steps=h,device=device)).view(1, 1, h, 1).expand(1, d, h, w)  # [1, H, W, D]
    z_range = (torch.linspace(-1,1,steps=d,device=device)).view(1, d, 1, 1).expand(1, d, h, w)  # [1, H, W, D]
    grid = torch.cat((x_range, y_range, z_range), dim=0).reshape(1, 3,-1).permute(0,2,1)

    return grid

''' Hierarchical softmax following the kinematic tree of the human body. Imporves convergence speed'''
def hierarchical_softmax(x):
    def softmax(x):
        return F.softmax(x, dim=-1)

    def sigmoid(x):
        return torch.sigmoid(x)

    n_point, n_dim = x.shape

    prob_all = torch.ones(n_point, 24, device=x.device)
    # softmax_x = F.softmax(x, dim=-1)
    sigmoid_x = sigmoid(x).float()

    prob_all[:, [1, 2, 3]] = sigmoid_x[:, [0]] * softmax(x[:, [1, 2, 3]])
    prob_all[:, [0]] = 1 - sigmoid_x[:, [0]]

    prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (sigmoid_x[:, [4, 5, 6]])
    prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid_x[:, [4, 5, 6]])

    prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (sigmoid_x[:, [7, 8, 9]])
    prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid_x[:, [7, 8, 9]])

    prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (sigmoid_x[:, [10, 11]])
    prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid_x[:, [10, 11]])

    prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid_x[:, [24]] * softmax(x[:, [12, 13, 14]])
    prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid_x[:, [24]])

    prob_all[:, [15]] = prob_all[:, [12]] * (sigmoid_x[:, [15]])
    prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid_x[:, [15]])

    prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (sigmoid_x[:, [16, 17]])
    prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid_x[:, [16, 17]])

    prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (sigmoid_x[:, [18, 19]])
    prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid_x[:, [18, 19]])

    prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (sigmoid_x[:, [20, 21]])
    prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid_x[:, [20, 21]])

    prob_all[:, [22, 23]] = prob_all[:, [20, 21]] * (sigmoid_x[:, [22, 23]])
    prob_all[:, [20, 21]] = prob_all[:, [20, 21]] * (1 - sigmoid_x[:, [22, 23]])

    # prob_all = prob_all.reshape(n_batch, n_point, prob_all.shape[-1])
    return prob_all

class SkinningField(RigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.smpl_verts = metadata["smpl_verts"]
        self.skinning_weights = metadata["skinning_weights"]
        self.aabb = metadata["aabb"]
        self.faces = np.load(get_body_model_misc_path('faces.npz'))['faces']
        self.cano_mesh = metadata["cano_mesh"]

        self.distill = cfg.distill
        d, h, w = cfg.res // cfg.z_ratio, cfg.res, cfg.res
        self.resolution = (d, h, w)
        if self.distill:
            self.grid = create_voxel_grid(d, h, w).cuda()

        self.lbs_network = get_skinning_mlp(3, cfg.d_out, cfg.skinning_network)


    def precompute(self, recompute_skinning=True):
        if recompute_skinning or not hasattr(self, "lbs_voxel_final"):
            d, h, w = self.resolution

            lbs_voxel_final = self.lbs_network(self.grid[0]).float()
            lbs_voxel_final = self.cfg.soft_blend * lbs_voxel_final

            lbs_voxel_final = self.softmax(lbs_voxel_final)

            self.lbs_voxel_final = lbs_voxel_final.permute(1, 0).reshape(1, 24, d, h, w)

    def get_forward_transform(self, xyz, tfs):
        if self.distill:
            self.precompute(recompute_skinning=self.training)
            fwd_grid = torch.einsum("bcdhw,bcxy->bxydhw", self.lbs_voxel_final, tfs[None])
            fwd_grid = fwd_grid.reshape(1, -1, *self.resolution)
            T_fwd = F.grid_sample(fwd_grid, xyz.reshape(1, 1, 1, -1, 3), padding_mode='border')
            T_fwd = T_fwd.reshape(4, 4, -1).permute(2, 0, 1)
        else:
            pts_W = self.lbs_network(xyz)
            pts_W = self.softmax(pts_W)
            T_fwd = torch.matmul(pts_W, tfs.view(-1, 16)).view(-1, 4, 4).float()
        return T_fwd

    def sample_skinning_loss(self):
        points_skinning, face_idx = self.cano_mesh.sample(self.cfg.n_reg_pts, return_index=True)
        points_skinning = points_skinning.view(np.ndarray).astype(np.float32)
        bary_coords = igl.barycentric_coordinates_tri(
            points_skinning,
            self.smpl_verts[self.faces[face_idx, 0], :],
            self.smpl_verts[self.faces[face_idx, 1], :],
            self.smpl_verts[self.faces[face_idx, 2], :],
        )
        vert_ids = self.faces[face_idx, ...]
        pts_W = (self.skinning_weights[vert_ids] * bary_coords[..., None]).sum(axis=1)

        points_skinning = torch.from_numpy(points_skinning).cuda()
        pts_W = torch.from_numpy(pts_W).cuda()
        return points_skinning, pts_W

    def softmax(self, logit):
        if logit.shape[-1] == 25:
            w = hierarchical_softmax(logit)
        elif logit.shape[-1] == 24:
            w = F.softmax(logit, dim=-1)
        else:
            raise ValueError
        return w

    def get_skinning_loss(self):
        pts_skinning, sampled_weights = self.sample_skinning_loss()
        pts_skinning = self.aabb.normalize(pts_skinning, sym=True)

        if self.distill:
            pred_weights = F.grid_sample(self.lbs_voxel_final, pts_skinning.reshape(1, 1, 1, -1, 3), padding_mode='border')
            pred_weights = pred_weights.reshape(24, -1).permute(1, 0)
        else:
            pred_weights = self.lbs_network(pts_skinning)
            pred_weights = self.softmax(pred_weights)
        skinning_loss = torch.nn.functional.mse_loss(
            pred_weights, sampled_weights, reduction='none').sum(-1).mean()
        # breakpoint()

        return skinning_loss


    def forward(self, gaussians, iteration, camera):
        tfs = camera.bone_transforms

        xyz = gaussians.get_xyz
        n_pts = xyz.shape[0]
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        T_fwd = self.get_forward_transform(xyz_norm, tfs)

        deformed_gaussians = gaussians.clone()
        deformed_gaussians.set_fwd_transform(T_fwd.detach())

        homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz.device)
        x_hat_homo = torch.cat([xyz, homo_coord], dim=-1).view(n_pts, 4, 1)
        x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
        deformed_gaussians._xyz = x_bar

        rotation_hat = build_rotation(gaussians._rotation)
        rotation_bar = torch.matmul(T_fwd[:, :3, :3], rotation_hat)
        setattr(deformed_gaussians, 'rotation_precomp', rotation_bar)
        # deformed_gaussians._rotation = tf.matrix_to_quaternion(rotation_bar)
        # deformed_gaussians._rotation = rotation_matrix_to_quaternion(rotation_bar)

        return deformed_gaussians

    def regularization(self):
        loss_skinning = self.get_skinning_loss()
        return {
            'loss_skinning': loss_skinning
        }


class ExplicitBinding(RigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.register_buffer('smpl_verts', torch.from_numpy(metadata['smpl_verts']).float())
        self.register_buffer('faces', torch.from_numpy(metadata['faces']).long())
        self.register_buffer('skinning_weights', torch.from_numpy(metadata['skinning_weights']).float())
        self.register_buffer('cano_joints', self._build_canonical_joints(metadata))
        parents = np.load(get_body_model_misc_path('kintree_table.npy'))[0].astype(np.int64)
        self.register_buffer('parents', torch.from_numpy(parents))
        self.register_buffer('joint_hop_distance', self._build_joint_hop_distance(parents))
        self.register_buffer('vertex_face_ids', self._build_vertex_face_adjacency(metadata['faces'], metadata['smpl_verts'].shape[0]))
        self.register_buffer('face_neighbor_ids', self._build_face_face_adjacency(metadata['faces'], metadata['smpl_verts'].shape[0]))

        self.soft_rigid_blend = cfg.get('soft_rigid_blend', 0.5)
        self.soft_normal_blend = cfg.get('soft_normal_blend', 0.8)
        self.rigid_bone_threshold = cfg.get('rigid_bone_threshold', 0.03)
        self.soft_bone_threshold = cfg.get('soft_bone_threshold', 0.08)
        self.transition_width = cfg.get('transition_width', 0.015)
        self.rigid_surface_threshold = cfg.get('rigid_surface_threshold', 0.01)
        self.free_surface_threshold = cfg.get('free_surface_threshold', 0.03)
        self.surface_transition_width = cfg.get('surface_transition_width', 0.01)
        self.dominance_power = cfg.get('dominance_power', 1.0)
        self.rebind_interval = cfg.get('rebind_interval', 0)
        self.temporal_momentum = cfg.get('temporal_momentum', 0.9)
        self.temporal_cache_size = cfg.get('temporal_cache_size', 8)
        self.semantic_knn = cfg.get('semantic_knn', 4)
        self.semantic_geo_weight = cfg.get('semantic_geo_weight', 1.0)
        self.semantic_skinning_weight = cfg.get('semantic_skinning_weight', 0.03)
        self.semantic_normal_weight = cfg.get('semantic_normal_weight', 0.02)
        self.semantic_prior_weight = cfg.get('semantic_prior_weight', 0.03)
        self.semantic_switch_score_margin = cfg.get('semantic_switch_score_margin', 0.002)
        self.semantic_switch_skinning_margin = cfg.get('semantic_switch_skinning_margin', 0.02)
        self.semantic_switch_surface_tolerance = cfg.get('semantic_switch_surface_tolerance', 0.002)
        self.semantic_same_group_score_margin = cfg.get('semantic_same_group_score_margin', 0.0)
        self.semantic_same_group_surface_tolerance = cfg.get('semantic_same_group_surface_tolerance', 0.006)
        self.semantic_same_group_normal_min_dot = float(resolve_schedule_value(
            0,
            cfg.get('semantic_same_group_normal_min_dot', 0.0),
            default=0.0,
        ))
        self.semantic_same_group_max_hops = int(resolve_schedule_value(
            0,
            cfg.get('semantic_same_group_max_hops', -1),
            default=-1,
        ))
        self.semantic_arm_group_max_hops = int(resolve_schedule_value(
            0,
            cfg.get('semantic_arm_group_max_hops', 1),
            default=1,
        ))
        self.body_confidence_threshold = cfg.get('body_confidence_threshold', 0.7)
        self.body_surface_threshold = cfg.get('body_surface_threshold', 0.018)
        self.body_semantic_threshold = cfg.get('body_semantic_threshold', 0.012)
        self.cloth_surface_threshold = cfg.get('cloth_surface_threshold', 0.02)
        self.cloth_semantic_threshold = cfg.get('cloth_semantic_threshold', 0.04)
        self.region_transition_width = cfg.get('region_transition_width', 0.015)
        self.cloth_semantic_mix = cfg.get('cloth_semantic_mix', 0.35)
        self.free_confidence_bias = cfg.get('free_confidence_bias', 0.15)
        self.free_surface_power = cfg.get('free_surface_power', 0.5)
        self.body_rigid_boost = cfg.get('body_rigid_boost', 0.35)
        self.cloth_free_boost = cfg.get('cloth_free_boost', 0.45)
        self.torso_rigid_boost = cfg.get('torso_rigid_boost', 0.18)
        self.arm_rigid_boost = cfg.get('arm_rigid_boost', 0.28)
        self.head_rigid_boost = cfg.get('head_rigid_boost', 0.12)
        self.pelvis_free_boost = cfg.get('pelvis_free_boost', 0.12)
        self.leg_free_boost = cfg.get('leg_free_boost', 0.08)
        self.thin_scale_threshold = cfg.get('thin_scale_threshold', 0.0035)
        self.thin_scale_width = cfg.get('thin_scale_width', 0.0015)
        self.thin_shell_threshold = cfg.get('thin_shell_threshold', 0.008)
        self.thin_confidence_threshold = cfg.get('thin_confidence_threshold', 0.60)
        self.thin_semantic_threshold = cfg.get('thin_semantic_threshold', 0.03)
        self.thin_score_width = cfg.get('thin_score_width', 0.01)
        self.thin_accessory_boost = cfg.get('thin_accessory_boost', 0.60)
        self.thin_rigid_suppression = cfg.get('thin_rigid_suppression', 0.35)
        self.boundary_score_soft_weight = cfg.get('boundary_score_soft_weight', 0.70)
        self.boundary_score_shell_weight = cfg.get('boundary_score_shell_weight', 0.20)
        self.boundary_score_thin_weight = cfg.get('boundary_score_thin_weight', 0.10)
        self.boundary_score_surface_threshold = cfg.get('boundary_score_surface_threshold', self.free_surface_threshold)
        self.boundary_score_surface_width = cfg.get('boundary_score_surface_width', self.surface_transition_width)
        self.boundary_score_confidence_suppress = cfg.get('boundary_score_confidence_suppress', 0.15)
        self.non_rigid_delta_rigid_preserve = float(cfg.get('non_rigid_delta_rigid_preserve', 0.0))
        self.non_rigid_delta_soft_preserve = float(cfg.get('non_rigid_delta_soft_preserve', 0.0))
        self.non_rigid_geometry_blend = float(cfg.get('non_rigid_geometry_blend', 0.0))
        self.non_rigid_geometry_delta_clip = float(cfg.get('non_rigid_geometry_delta_clip', 0.0))
        self.non_rigid_layer_sharpen_strength = float(cfg.get('non_rigid_layer_sharpen_strength', 0.0))
        self.non_rigid_layer_sharpen_threshold = float(cfg.get('non_rigid_layer_sharpen_threshold', 0.003))
        self.non_rigid_layer_sharpen_width = float(cfg.get('non_rigid_layer_sharpen_width', 0.0015))
        self.non_rigid_layer_sharpen_min_dominance = float(cfg.get('non_rigid_layer_sharpen_min_dominance', 0.45))
        self.non_rigid_layer_sharpen_dominance_width = float(
            cfg.get('non_rigid_layer_sharpen_dominance_width', 0.05)
        )
        self.non_rigid_delta_debug_interval = int(cfg.get('non_rigid_delta_debug_interval', 0))

        joint_rigid_prior = torch.zeros(24, dtype=torch.float32)
        joint_free_prior = torch.zeros(24, dtype=torch.float32)
        joint_thin_prior = torch.zeros(24, dtype=torch.float32)
        joint_rigid_prior[[0, 3, 6, 9, 12]] = self.torso_rigid_boost
        joint_rigid_prior[[13, 14, 16, 17, 18, 19, 20, 21, 22, 23]] = self.arm_rigid_boost
        joint_rigid_prior[[15]] = self.head_rigid_boost
        joint_free_prior[[0, 1, 2]] = self.pelvis_free_boost
        joint_free_prior[[4, 5, 7, 8, 10, 11]] = self.leg_free_boost
        joint_thin_prior[[0, 1, 2, 4, 5, 7, 8, 10, 11]] = 1.0
        self.register_buffer('joint_rigid_prior', joint_rigid_prior)
        self.register_buffer('joint_free_prior', joint_free_prior)
        self.register_buffer('joint_thin_prior', joint_thin_prior)

        joint_anchor_group = torch.zeros(24, dtype=torch.long)
        joint_anchor_group[[0, 1, 2]] = 0
        joint_anchor_group[[3, 6, 9, 12]] = 1
        joint_anchor_group[[4, 5, 7, 8, 10, 11]] = 2
        joint_anchor_group[[13, 14, 16, 17, 18, 19, 20, 21, 22, 23]] = 3
        joint_anchor_group[[15]] = 4
        self.register_buffer('joint_anchor_group', joint_anchor_group)
        face_anchor_weights = self.skinning_weights[self.faces].mean(dim=1)
        face_dominant_joint = torch.argmax(face_anchor_weights, dim=-1)
        self.register_buffer('face_dominant_joint', face_dominant_joint)
        self.register_buffer('face_joint_group', joint_anchor_group[face_dominant_joint])

        self.binding_cache = {}
        self.temporal_cache = OrderedDict()
        self.latest_subset_refresh_info = None
        self.latest_face_anchor_switch_mask = None
        self.latest_face_anchor_keep_prior_mask = None
        self.latest_face_anchor_best_joint_changed_mask = None
        self.latest_face_anchor_keep_prior_best_face_changed_mask = None
        self.latest_face_anchor_keep_prior_best_joint_changed_mask = None
        self.latest_face_anchor_keep_prior_best_anchor_shift = None
        self.loss_reg = {}
        self.latest_temporal_slip = None

        self.residual_field_enable = bool(cfg.get('residual_field_enable', False))
        self.residual_field_anchor_enable = bool(cfg.get('residual_field_anchor_enable', self.residual_field_enable))
        self.residual_field_xbar_enable = bool(cfg.get('residual_field_xbar_enable', False))
        self.residual_field_update_dominant_joint = bool(cfg.get('residual_field_update_dominant_joint', False))
        self.residual_field_feature_detach = bool(cfg.get('residual_field_feature_detach', True))
        self.residual_field_detach_gate = bool(cfg.get('residual_field_detach_gate', True))
        self.residual_field_boundary_gate_enable = bool(cfg.get('residual_field_boundary_gate_enable', True))
        self.residual_field_boundary_gate_power = float(cfg.get('residual_field_boundary_gate_power', 1.0))
        self.residual_field_boundary_gate_min = float(cfg.get('residual_field_boundary_gate_min', 0.0))
        self.residual_field_surface_gate_enable = bool(cfg.get('residual_field_surface_gate_enable', False))
        self.residual_field_surface_gate_width = float(
            cfg.get('residual_field_surface_gate_width', self.surface_transition_width)
        )
        self.residual_anchor_delta_scale = float(cfg.get('residual_anchor_delta_scale', 0.12))
        self.residual_anchor_delta_max = float(cfg.get('residual_anchor_delta_max', 0.20))
        self.residual_xbar_scale = float(cfg.get('residual_xbar_scale', 0.0035))
        self.residual_xbar_max = float(cfg.get('residual_xbar_max', 0.008))
        self.residual_xbar_tangent_scale = float(cfg.get('residual_xbar_tangent_scale', 1.0))
        self.residual_xbar_normal_scale = float(cfg.get('residual_xbar_normal_scale', 0.5))

        self.residual_field_mlp = None
        self.residual_field_input_dim = 41
        self.residual_field_output_dim = 0
        if self.residual_field_enable and (self.residual_field_anchor_enable or self.residual_field_xbar_enable):
            self.residual_field_output_dim += 24 if self.residual_field_anchor_enable else 0
            self.residual_field_output_dim += 3 if self.residual_field_xbar_enable else 0
            residual_field_mlp_cfg = cfg.get('residual_field_mlp', None)
            if residual_field_mlp_cfg is None:
                residual_field_mlp_cfg = OmegaConf.create({
                    'n_neurons': 64,
                    'n_hidden_layers': 2,
                    'skip_in': [],
                    'cond_in': [],
                    'multires': 0,
                    'last_layer_init': True,
                })
            self.residual_field_mlp = VanillaCondMLP(
                self.residual_field_input_dim,
                0,
                self.residual_field_output_dim,
                residual_field_mlp_cfg,
            )

        self.hybrid_field_enable = bool(cfg.get('hybrid_field_enable', False))
        self.hybrid_field_anchor_enable = bool(cfg.get('hybrid_field_anchor_enable', self.hybrid_field_enable))
        self.hybrid_field_support_enable = bool(cfg.get('hybrid_field_support_enable', False))
        self.hybrid_field_layer_enable = bool(cfg.get('hybrid_field_layer_enable', False))
        self.hybrid_field_update_dominant_joint = bool(cfg.get('hybrid_field_update_dominant_joint', True))
        self.hybrid_field_feature_detach = bool(cfg.get('hybrid_field_feature_detach', False))
        self.hybrid_field_detach_gate = bool(cfg.get('hybrid_field_detach_gate', False))
        self.hybrid_field_gate_mode = str(cfg.get('hybrid_field_gate_mode', 'global'))
        self.hybrid_field_gate_min = float(cfg.get('hybrid_field_gate_min', 0.0))
        self.hybrid_field_gate_power = float(cfg.get('hybrid_field_gate_power', 1.0))
        self.hybrid_field_confidence_suppress = float(cfg.get('hybrid_field_confidence_suppress', 0.0))
        self.hybrid_anchor_mode = str(cfg.get('hybrid_anchor_mode', 'residual'))
        self.hybrid_anchor_delta_scale = float(cfg.get('hybrid_anchor_delta_scale', 0.10))
        self.hybrid_anchor_delta_max = float(cfg.get('hybrid_anchor_delta_max', 0.16))
        self.hybrid_anchor_direct_logit_scale = float(cfg.get('hybrid_anchor_direct_logit_scale', 1.0))
        self.hybrid_layer_mode = str(cfg.get('hybrid_layer_mode', 'residual'))
        self.hybrid_layer_delta_scale = float(cfg.get('hybrid_layer_delta_scale', 0.75))
        self.hybrid_layer_delta_max = float(cfg.get('hybrid_layer_delta_max', 1.20))
        self.hybrid_layer_direct_logit_scale = float(cfg.get('hybrid_layer_direct_logit_scale', 1.0))
        self.hybrid_support_scale = float(cfg.get('hybrid_support_scale', 0.0025))
        self.hybrid_support_max = float(cfg.get('hybrid_support_max', 0.006))
        self.hybrid_support_tangent_scale = float(cfg.get('hybrid_support_tangent_scale', 1.0))
        self.hybrid_support_normal_scale = float(cfg.get('hybrid_support_normal_scale', 0.5))

        self.hybrid_field_mlp = None
        self.hybrid_field_input_dim = 47
        self.hybrid_field_output_dim = 0
        if self.hybrid_field_enable and (
            self.hybrid_field_anchor_enable
            or self.hybrid_field_support_enable
            or self.hybrid_field_layer_enable
        ):
            self.hybrid_field_output_dim += 24 if self.hybrid_field_anchor_enable else 0
            self.hybrid_field_output_dim += 3 if self.hybrid_field_support_enable else 0
            self.hybrid_field_output_dim += 3 if self.hybrid_field_layer_enable else 0
            hybrid_field_mlp_cfg = cfg.get('hybrid_field_mlp', None)
            if hybrid_field_mlp_cfg is None:
                hybrid_field_mlp_cfg = OmegaConf.create({
                    'n_neurons': 64,
                    'n_hidden_layers': 2,
                    'skip_in': [],
                    'cond_in': [],
                    'multires': 0,
                    'last_layer_init': True,
                })
            self.hybrid_field_mlp = VanillaCondMLP(
                self.hybrid_field_input_dim,
                0,
                self.hybrid_field_output_dim,
                hybrid_field_mlp_cfg,
            )

        self.forward_trunk_enable = bool(cfg.get('forward_trunk_enable', False))
        self.forward_trunk_anchor_enable = bool(cfg.get('forward_trunk_anchor_enable', self.forward_trunk_enable))
        self.forward_trunk_support_enable = bool(cfg.get('forward_trunk_support_enable', False))
        self.forward_trunk_layer_enable = bool(cfg.get('forward_trunk_layer_enable', False))
        self.forward_trunk_update_dominant_joint = bool(cfg.get('forward_trunk_update_dominant_joint', True))
        self.forward_trunk_feature_detach = bool(cfg.get('forward_trunk_feature_detach', False))
        self.forward_trunk_blend_alpha_cfg = cfg.get('forward_trunk_blend_alpha', 1.0)
        self.forward_trunk_confidence_suppress = float(cfg.get('forward_trunk_confidence_suppress', 0.0))
        self.forward_trunk_output_clamp = float(cfg.get('forward_trunk_output_clamp', 0.0))
        self.forward_trunk_anchor_alpha_scale_cfg = cfg.get('forward_trunk_anchor_alpha_scale', 1.0)
        self.forward_trunk_support_alpha_scale_cfg = cfg.get('forward_trunk_support_alpha_scale', 1.0)
        self.forward_trunk_layer_alpha_scale_cfg = cfg.get('forward_trunk_layer_alpha_scale', 1.0)
        self.forward_trunk_xbar_alpha_scale_cfg = cfg.get('forward_trunk_xbar_alpha_scale', 1.0)
        self.forward_trunk_anchor_logit_scale = float(cfg.get('forward_trunk_anchor_logit_scale', 1.0))
        self.forward_trunk_layer_logit_scale = float(cfg.get('forward_trunk_layer_logit_scale', 1.0))
        self.forward_trunk_anchor_residual = bool(cfg.get('forward_trunk_anchor_residual', False))
        self.forward_trunk_layer_residual = bool(cfg.get('forward_trunk_layer_residual', False))
        self.forward_trunk_anchor_delta_logit_max = float(cfg.get('forward_trunk_anchor_delta_logit_max', 0.0))
        self.forward_trunk_layer_delta_logit_max = float(cfg.get('forward_trunk_layer_delta_logit_max', 0.0))
        self.forward_trunk_layer_min_prob = float(cfg.get('forward_trunk_layer_min_prob', 0.0))
        self.forward_trunk_support_scale = float(cfg.get('forward_trunk_support_scale', 0.0035))
        self.forward_trunk_support_max = float(cfg.get('forward_trunk_support_max', 0.008))
        self.forward_trunk_support_tangent_scale = float(cfg.get('forward_trunk_support_tangent_scale', 1.0))
        self.forward_trunk_support_normal_scale = float(cfg.get('forward_trunk_support_normal_scale', 0.5))
        self.forward_trunk_xbar_enable = bool(cfg.get('forward_trunk_xbar_enable', False))
        self.forward_trunk_xbar_scale = float(cfg.get('forward_trunk_xbar_scale', 0.0025))
        self.forward_trunk_xbar_max = float(cfg.get('forward_trunk_xbar_max', 0.006))
        self.forward_trunk_xbar_tangent_scale = float(cfg.get('forward_trunk_xbar_tangent_scale', 1.0))
        self.forward_trunk_xbar_normal_scale = float(cfg.get('forward_trunk_xbar_normal_scale', 0.5))

        self.forward_trunk_mlp = None
        self.forward_trunk_input_dim = 47
        self.forward_trunk_output_dim = 0
        if self.forward_trunk_enable and (
            self.forward_trunk_anchor_enable
            or self.forward_trunk_support_enable
            or self.forward_trunk_layer_enable
            or self.forward_trunk_xbar_enable
        ):
            self.forward_trunk_output_dim += 24 if self.forward_trunk_anchor_enable else 0
            self.forward_trunk_output_dim += 3 if self.forward_trunk_support_enable else 0
            self.forward_trunk_output_dim += 3 if self.forward_trunk_layer_enable else 0
            self.forward_trunk_output_dim += 3 if self.forward_trunk_xbar_enable else 0
            forward_trunk_mlp_cfg = cfg.get('forward_trunk_mlp', None)
            if forward_trunk_mlp_cfg is None:
                forward_trunk_mlp_cfg = OmegaConf.create({
                    'n_neurons': 128,
                    'n_hidden_layers': 4,
                    'skip_in': [],
                    'cond_in': [],
                    'multires': 0,
                    'last_layer_init': True,
                })
            self.forward_trunk_mlp = VanillaCondMLP(
                self.forward_trunk_input_dim,
                0,
                self.forward_trunk_output_dim,
                forward_trunk_mlp_cfg,
            )

    def _build_canonical_joints(self, metadata):
        Jtr = torch.from_numpy(metadata['Jtr']).float()
        bone_transforms_02v = torch.from_numpy(metadata['bone_transforms_02v']).float()
        joints = torch.matmul(bone_transforms_02v[:, :3, :3], Jtr.unsqueeze(-1)).squeeze(-1)
        joints = joints + bone_transforms_02v[:, :3, 3]
        return joints

    def _build_vertex_face_adjacency(self, faces, n_verts):
        face_lists = [[] for _ in range(n_verts)]
        for face_id, face in enumerate(faces.tolist()):
            for vid in face:
                face_lists[vid].append(face_id)
        max_faces = max(len(face_ids) for face_ids in face_lists)
        adjacency = np.full((n_verts, max_faces), -1, dtype=np.int64)
        for vid, face_ids in enumerate(face_lists):
            adjacency[vid, :len(face_ids)] = face_ids
        return torch.from_numpy(adjacency)

    def _build_face_face_adjacency(self, faces, n_verts):
        face_lists = [[] for _ in range(n_verts)]
        for face_id, face in enumerate(faces.tolist()):
            for vid in face:
                face_lists[vid].append(face_id)

        neighbor_lists = []
        for face_id, face in enumerate(faces.tolist()):
            neighbors = set()
            for vid in face:
                neighbors.update(face_lists[vid])
            neighbors.discard(face_id)
            neighbor_lists.append(sorted(neighbors))

        max_neighbors = max((len(neighbors) for neighbors in neighbor_lists), default=0)
        adjacency = np.full((len(neighbor_lists), max_neighbors), -1, dtype=np.int64)
        for face_id, neighbors in enumerate(neighbor_lists):
            if neighbors:
                adjacency[face_id, :len(neighbors)] = neighbors
        return torch.from_numpy(adjacency)

    def _expand_face_neighbor_ids(self, face_ids, hops):
        if face_ids is None:
            return None
        face_ids = face_ids.to(dtype=torch.long)
        frontier = face_ids[:, None]
        expanded = [frontier]
        if hops <= 0:
            return frontier

        invalid_fill = torch.full_like(frontier, -1)
        for _ in range(hops):
            frontier_valid = frontier >= 0
            if not bool(frontier_valid.any().item()):
                expanded.append(invalid_fill.expand_as(frontier))
                frontier = invalid_fill.expand_as(frontier)
                continue
            frontier_safe = frontier.clamp_min(0)
            neighbor_ids = self.face_neighbor_ids[frontier_safe.reshape(-1)].view(frontier.shape[0], frontier.shape[1], -1)
            neighbor_ids = torch.where(
                frontier_valid.unsqueeze(-1),
                neighbor_ids,
                torch.full_like(neighbor_ids, -1),
            )
            frontier = neighbor_ids.reshape(frontier.shape[0], -1)
            expanded.append(frontier)
        return torch.cat(expanded, dim=1)

    def _build_joint_hop_distance(self, parents):
        parents_np = parents.astype(np.int64)
        n_joints = int(parents_np.shape[0])
        large = np.int64(1_000_000)
        hop = np.full((n_joints, n_joints), large, dtype=np.int64)
        np.fill_diagonal(hop, 0)

        for child, parent in enumerate(parents_np.tolist()):
            if parent < 0:
                continue
            hop[child, parent] = 1
            hop[parent, child] = 1

        for mid in range(n_joints):
            hop = np.minimum(hop, hop[:, [mid]] + hop[[mid], :])
        return torch.from_numpy(hop)

    def _force_rebind(self, iteration):
        return self.rebind_interval > 0 and self.training and iteration > 0 and iteration % self.rebind_interval == 0

    def _scheduled_cfg(self, key, iteration, default=None, cast=None):
        value = resolve_schedule_value(iteration, self.cfg.get(key, default), default=default)
        if cast is not None and value is not None:
            value = cast(value)
        return value

    def _state_matches(self, binding_state, canonical_xyz):
        if not binding_state:
            return False
        for value in binding_state.values():
            if torch.is_tensor(value):
                return value.shape[0] == canonical_xyz.shape[0]
        return False

    def _debug_binding_state_summary(self, binding_state, canonical_xyz):
        if not binding_state:
            return 'empty'
        point_count = None
        refresh_count = 0
        risky_count = 0
        for value in binding_state.values():
            if torch.is_tensor(value):
                point_count = int(value.shape[0])
                break
        refresh_mask = binding_state.get('anchor_refresh_mask', None)
        if torch.is_tensor(refresh_mask):
            refresh_count = int(refresh_mask.to(dtype=torch.bool).sum().item())
        risky_mask = binding_state.get('densify_risky_child_mask', None)
        if torch.is_tensor(risky_mask):
            risky_count = int(risky_mask.to(dtype=torch.bool).sum().item())
        canonical_count = int(canonical_xyz.shape[0]) if torch.is_tensor(canonical_xyz) else -1
        return (
            f'points={point_count} canonical={canonical_count} '
            f'refresh={refresh_count} risky={risky_count}'
        )

    def _slice_binding_state(self, binding_state, mask):
        if not binding_state:
            return {}
        sliced_state = {}
        for key, value in binding_state.items():
            if torch.is_tensor(value) and value.shape[0] == mask.shape[0]:
                sliced_state[key] = value[mask]
            elif not torch.is_tensor(value):
                sliced_state[key] = value
        return sliced_state

    def _merge_binding_state(self, base_state, subset_state, mask):
        if not base_state:
            return subset_state

        point_count = int(mask.shape[0])
        selected_count = int(mask.sum().item())
        merged_state = {}
        all_keys = set(base_state.keys()) | set(subset_state.keys())

        for key in all_keys:
            base_value = base_state.get(key, None)
            subset_value = subset_state.get(key, None)

            if torch.is_tensor(base_value) and base_value.shape[0] == point_count:
                merged_value = base_value.clone()
                if torch.is_tensor(subset_value):
                    merged_value[mask] = subset_value.to(device=merged_value.device, dtype=merged_value.dtype)
                merged_state[key] = merged_value
                continue

            if torch.is_tensor(subset_value) and subset_value.shape[0] == selected_count:
                target_device = subset_value.device
                target_dtype = subset_value.dtype
                if torch.is_tensor(base_value):
                    target_device = base_value.device
                    target_dtype = base_value.dtype
                merged_shape = (point_count,) + tuple(subset_value.shape[1:])
                merged_value = torch.zeros(merged_shape, device=target_device, dtype=target_dtype)
                merged_value[mask] = subset_value.to(device=target_device, dtype=target_dtype)
                merged_state[key] = merged_value
                continue

            if torch.is_tensor(base_value):
                merged_state[key] = base_value.clone()
            elif torch.is_tensor(subset_value):
                merged_state[key] = subset_value.clone()
            else:
                merged_state[key] = subset_value if subset_value is not None else base_value

        return merged_state

    def _compute_bone_distance(self, bound_xyz, dominant_joint):
        joints = self.cano_joints[dominant_joint]
        parents = self.parents[dominant_joint]
        parent_joints = joints.clone()
        valid_parent = parents >= 0
        if valid_parent.any():
            parent_joints[valid_parent] = self.cano_joints[parents[valid_parent]]

        dist = torch.norm(bound_xyz - joints, dim=-1)
        if valid_parent.any():
            dist[valid_parent] = point_to_segment_distance(
                bound_xyz[valid_parent],
                parent_joints[valid_parent],
                joints[valid_parent],
            )
        return dist

    def _compute_region_probs(self, confidence, surface_distance, semantic_distance):
        width = max(float(self.region_transition_width), 1e-6)
        conf_score = torch.sigmoid((confidence - self.body_confidence_threshold) / width)
        body_surface = torch.sigmoid((self.body_surface_threshold - surface_distance) / width)
        body_semantic = torch.sigmoid((self.body_semantic_threshold - semantic_distance) / width)
        surface_cloth = torch.sigmoid((surface_distance - self.cloth_surface_threshold) / width)
        semantic_cloth = torch.sigmoid((semantic_distance - self.cloth_semantic_threshold) / width)

        body_prob = conf_score * body_surface * body_semantic
        cloth_prob = surface_cloth * ((1. - self.cloth_semantic_mix) + self.cloth_semantic_mix * semantic_cloth)
        overlap = torch.sqrt((body_prob * cloth_prob).clamp_min(0.0))
        body_prob = torch.clamp(body_prob * (1. - 0.72 * cloth_prob), min=0.0)
        cloth_prob = torch.clamp(cloth_prob * (1. - 0.78 * body_prob), min=0.0)
        soft_prob = torch.clamp(1. - body_prob - cloth_prob, min=0.) + 0.24 * overlap
        probs = torch.stack([body_prob, soft_prob, cloth_prob], dim=-1)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        body_prob, soft_prob, cloth_prob = probs.unbind(dim=-1)
        dominance_gap = (body_prob - cloth_prob).abs()
        dominant_max = torch.maximum(body_prob, cloth_prob)
        interior = torch.sigmoid((dominant_max - 0.42) / 0.05) * torch.sigmoid((dominance_gap - 0.14) / 0.05)
        soft_prob = soft_prob * (1. - 0.38 * interior)
        probs = torch.stack([body_prob, soft_prob, cloth_prob], dim=-1)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _compute_compact_semantic_probs(self, dominant_joint, region_probs, confidence, surface_distance, semantic_distance, thin_score):
        width = max(float(self.region_transition_width), 1e-6)
        body_prob, soft_prob, cloth_prob = region_probs.unbind(dim=-1)
        conf_score = torch.sigmoid((confidence - self.body_confidence_threshold) / width)
        body_surface = torch.sigmoid((self.body_surface_threshold - surface_distance) / width)
        body_semantic = torch.sigmoid((self.body_semantic_threshold - semantic_distance) / width)
        cloth_surface = torch.sigmoid((surface_distance - self.cloth_surface_threshold) / width)
        cloth_semantic = torch.sigmoid((semantic_distance - self.cloth_semantic_threshold) / width)

        head_mask = joint_group_mask(dominant_joint, [15])
        torso_mask = joint_group_mask(dominant_joint, [0, 1, 2, 3, 6, 9, 12])
        upper_body_mask = joint_group_mask(dominant_joint, [3, 6, 9, 12, 13, 14, 16, 17, 18, 19])
        arm_mask = joint_group_mask(dominant_joint, [13, 14, 16, 17, 18, 19, 20, 21, 22, 23])
        leg_mask = joint_group_mask(dominant_joint, [4, 5, 7, 8, 10, 11])
        pelvis_mask = joint_group_mask(dominant_joint, [0, 1, 2, 4, 5, 7, 8])
        foot_mask = joint_group_mask(dominant_joint, [10, 11])

        face_score = head_mask * torch.clamp(body_prob * (0.52 + 0.48 * conf_score) * body_surface * (0.58 + 0.42 * body_semantic), 0.0, 1.0)
        hair_score = head_mask * torch.clamp(0.42 * soft_prob + 0.26 * cloth_prob + 0.22 * (1.0 - conf_score) + 0.18 * cloth_surface + 0.10 * cloth_semantic, 0.0, 1.0)
        hair_score = hair_score * (1.0 - 0.72 * face_score) + 0.01 * head_mask

        skin_score = torch.clamp(body_prob * ((0.88 * arm_mask) + (0.84 * leg_mask) + (0.06 * torso_mask)) * (0.54 + 0.46 * conf_score) * (0.58 + 0.42 * body_surface), 0.0, 1.0)

        upper_support = torch.clamp(
            0.96 * torso_mask
            + 0.20 * upper_body_mask
            + 0.08 * arm_mask
            - 0.42 * pelvis_mask
            - 0.24 * leg_mask,
            0.0,
            1.0,
        )
        lower_support = torch.clamp(
            1.08 * leg_mask
            + 0.92 * pelvis_mask
            + 0.10 * torso_mask
            - 0.18 * head_mask,
            0.0,
            1.0,
        )

        upper_score = torch.clamp(
            cloth_prob
            * upper_support
            * (1.0 - 0.90 * foot_mask)
            * (0.54 + 0.46 * cloth_surface)
            * (0.56 + 0.44 * cloth_semantic),
            0.0,
            1.0,
        )
        lower_score = torch.clamp(
            cloth_prob
            * lower_support
            * (1.0 - 0.86 * foot_mask)
            * (0.60 + 0.40 * cloth_surface)
            * (0.54 + 0.46 * cloth_semantic),
            0.0,
            1.0,
        )
        shoes_score = torch.clamp((0.94 * cloth_prob + 0.20 * thin_score) * foot_mask * (0.60 + 0.40 * cloth_surface), 0.0, 1.0) + 0.01 * foot_mask

        compact_scores = torch.stack([hair_score, face_score, skin_score, upper_score, lower_score, shoes_score], dim=-1)
        compact_scores[..., 3] = compact_scores[..., 3] * (1.0 - 0.24 * lower_support)
        compact_scores[..., 4] = compact_scores[..., 4] * (1.0 + 0.18 * pelvis_mask + 0.12 * leg_mask)
        compact_scores = compact_scores + compact_scores.new_tensor(1e-6)
        return compact_scores / compact_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _compute_layer_weights(self, bone_distance, surface_distance, confidence, semantic_distance):
        width = max(float(self.transition_width), 1e-6)
        surface_width = max(float(self.surface_transition_width), 1e-6)
        confidence = confidence.clamp(0., 1.).pow(self.dominance_power)

        rigid_bone = torch.sigmoid((self.rigid_bone_threshold - bone_distance) / width)
        rigid_surface = torch.sigmoid((self.rigid_surface_threshold - surface_distance) / surface_width)
        rigid_weight = rigid_bone * rigid_surface
        rigid_weight = rigid_weight * confidence

        free_bone = torch.sigmoid((bone_distance - self.soft_bone_threshold) / width)
        free_surface = torch.sigmoid((surface_distance - self.free_surface_threshold) / surface_width)
        free_weight = free_bone * torch.pow(free_surface.clamp_min(1e-6), self.free_surface_power)
        free_weight = torch.clamp(free_weight + (1. - confidence) * self.free_confidence_bias, 0., 1.)

        region_probs = self._compute_region_probs(confidence, surface_distance, semantic_distance)
        region_guidance = region_probs.detach()
        rigid_weight = torch.clamp(rigid_weight * (1. + self.body_rigid_boost * region_guidance[:, 0]) + 0.10 * region_guidance[:, 0] * (1. - free_weight), 0., 1.)
        free_weight = torch.clamp(free_weight * (1. + self.cloth_free_boost * region_guidance[:, 2]), 0., 1.)

        soft_weight = torch.clamp(1. - rigid_weight - free_weight, min=0.)
        weights = torch.stack([rigid_weight, soft_weight, free_weight], dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        rigid_weight, soft_weight, free_weight = weights.unbind(dim=-1)
        dominance_gap = (rigid_weight - free_weight).abs()
        dominant_max = torch.maximum(rigid_weight, free_weight)
        interior = torch.sigmoid((dominant_max - 0.44) / 0.05) * torch.sigmoid((dominance_gap - 0.12) / 0.05)
        soft_weight = soft_weight * (1. - 0.42 * interior)
        rigid_weight = rigid_weight + 0.16 * interior * region_guidance[:, 0]
        free_weight = free_weight + 0.16 * interior * region_guidance[:, 2]
        weights = torch.stack([rigid_weight, soft_weight, free_weight], dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return weights, region_probs

    def _compute_thin_accessory_score(self, point_scale, surface_distance, confidence, semantic_distance, dominant_joint):
        scale_width = max(float(self.thin_scale_width), 1e-6)
        score_width = max(float(self.thin_score_width), 1e-6)
        scale_score = torch.sigmoid((self.thin_scale_threshold - point_scale) / scale_width)
        shell_score = torch.sigmoid((surface_distance - self.thin_shell_threshold) / score_width)
        conf_score = torch.sigmoid((self.thin_confidence_threshold - confidence) / score_width)
        semantic_score = torch.sigmoid((semantic_distance - self.thin_semantic_threshold) / score_width)
        part_score = self.joint_thin_prior[dominant_joint]
        thin_score = scale_score * (0.5 * shell_score + 0.25 * conf_score + 0.25 * semantic_score)
        thin_score = torch.clamp(thin_score * (1. + 0.5 * part_score), 0., 1.)
        return thin_score

    def _apply_v41_priors(self, layer_weights, region_probs, dominant_joint, point_scale, confidence, surface_distance, semantic_distance):
        part_rigid = self.joint_rigid_prior[dominant_joint]
        part_free = self.joint_free_prior[dominant_joint]
        thin_score = self._compute_thin_accessory_score(point_scale, surface_distance, confidence, semantic_distance, dominant_joint)

        rigid_transfer = self.thin_rigid_suppression * thin_score * layer_weights[:, 0]
        rigid_weight = torch.clamp(layer_weights[:, 0] * (1. + part_rigid) - rigid_transfer, min=0.)
        soft_weight = torch.clamp(layer_weights[:, 1] + 0.35 * rigid_transfer, min=0.)
        free_weight = torch.clamp(layer_weights[:, 2] * (1. + part_free + self.thin_accessory_boost * thin_score) + 0.65 * rigid_transfer, min=0.)

        refined_weights = torch.stack([rigid_weight, soft_weight, free_weight], dim=-1)
        refined_weights = refined_weights / refined_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        body_prob = region_probs[:, 0] * (1. + part_rigid)
        soft_prob = region_probs[:, 1]
        cloth_prob = region_probs[:, 2] * (1. + part_free + self.thin_accessory_boost * thin_score)
        refined_regions = torch.stack([body_prob, soft_prob, cloth_prob], dim=-1)
        refined_regions = refined_regions / refined_regions.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return refined_weights, refined_regions, thin_score, part_rigid, part_free

    def _compute_boundary_score(self, layer_weights, surface_distance, thin_score, confidence):
        shell_width = max(float(self.boundary_score_surface_width), 1e-6)
        soft_score = layer_weights[:, 1]
        shell_score = torch.sigmoid((surface_distance - self.boundary_score_surface_threshold) / shell_width)
        score = (
            self.boundary_score_soft_weight * soft_score
            + self.boundary_score_shell_weight * shell_score
            + self.boundary_score_thin_weight * thin_score
        )
        if self.boundary_score_confidence_suppress > 0.0:
            score = score * (1.0 - self.boundary_score_confidence_suppress * confidence.clamp(0.0, 1.0))
        return torch.clamp(score, 0.0, 1.0)

    def _project_points_to_triangles(self, points, tri_verts):
        return closest_point_on_triangles(points, tri_verts)

    def _build_geometric_face_anchor(self, canonical_xyz):
        point_count = int(canonical_xyz.shape[0]) if canonical_xyz.ndim > 1 else int(canonical_xyz.numel() > 0)
        if point_count <= 0:
            empty_long = torch.zeros((0,), dtype=torch.long, device=self.faces.device)
            empty_tri = torch.zeros((0, 3), dtype=torch.long, device=self.faces.device)
            empty_bary = torch.zeros((0, 3), dtype=self.smpl_verts.dtype, device=self.smpl_verts.device)
            empty_xyz = torch.zeros((0, 3), dtype=self.smpl_verts.dtype, device=self.smpl_verts.device)
            empty_weights = torch.zeros(
                (0, self.skinning_weights.shape[-1]),
                dtype=self.skinning_weights.dtype,
                device=self.skinning_weights.device,
            )
            empty_scalar = torch.zeros((0,), dtype=self.smpl_verts.dtype, device=self.smpl_verts.device)
            return (
                empty_long,
                empty_tri,
                empty_bary,
                empty_xyz,
                empty_weights,
                empty_xyz,
                empty_scalar,
                empty_scalar,
                empty_scalar,
            )

        points_np = canonical_xyz.detach().cpu().numpy().astype(np.float32).reshape(-1, 3)
        verts_np = self.smpl_verts.detach().cpu().numpy().astype(np.float32).reshape(-1, 3)
        faces_np = self.faces.detach().cpu().numpy().astype(np.int32).reshape(-1, 3)
        if verts_np.shape[0] == 0 or faces_np.shape[0] == 0:
            raise RuntimeError(
                'geometric face anchor requires non-empty SMPL verts/faces, '
                f'got verts={verts_np.shape} faces={faces_np.shape}'
            )

        sqr_dists, face_ids_np, closest_points_np = igl.point_mesh_squared_distance(points_np, verts_np, faces_np)
        sqr_dists = np.asarray(sqr_dists, dtype=np.float32).reshape(-1)
        face_ids_np = np.asarray(face_ids_np, dtype=np.int64).reshape(-1)
        closest_points_np = np.asarray(closest_points_np, dtype=np.float32).reshape(-1, 3)
        if face_ids_np.shape[0] != point_count or closest_points_np.shape[0] != point_count:
            raise RuntimeError(
                'point_mesh_squared_distance returned inconsistent shapes '
                f'for point_count={point_count}: face_ids={face_ids_np.shape} '
                f'closest_points={closest_points_np.shape}'
            )

        tri_vids_np = np.asarray(faces_np[face_ids_np], dtype=np.int32).reshape(-1, 3)
        if tri_vids_np.shape[0] != point_count:
            raise RuntimeError(
                'triangle lookup returned inconsistent shape '
                f'for point_count={point_count}: tri_vids={tri_vids_np.shape}'
            )

        closest_points_np = np.asfortranarray(closest_points_np.astype(np.float32))
        tri_a_np = np.asfortranarray(verts_np[tri_vids_np[:, 0]].astype(np.float32))
        tri_b_np = np.asfortranarray(verts_np[tri_vids_np[:, 1]].astype(np.float32))
        tri_c_np = np.asfortranarray(verts_np[tri_vids_np[:, 2]].astype(np.float32))
        bary_np = igl.barycentric_coordinates_tri(
            closest_points_np,
            tri_a_np,
            tri_b_np,
            tri_c_np,
        ).astype(np.float32)

        face_ids = torch.from_numpy(face_ids_np).to(self.faces.device)
        tri_vids = self.faces[face_ids]
        bary = torch.from_numpy(bary_np).to(self.smpl_verts.device)
        tri_verts = self.smpl_verts[tri_vids]
        tri_weights = self.skinning_weights[tri_vids]

        anchor_xyz = (tri_verts * bary[..., None]).sum(dim=1)
        anchor_weights = (tri_weights * bary[..., None]).sum(dim=1)
        face_normal = torch.cross(tri_verts[:, 1] - tri_verts[:, 0], tri_verts[:, 2] - tri_verts[:, 0], dim=-1)
        face_normal = safe_normalize(face_normal)
        surface_distance = torch.from_numpy(np.sqrt(np.clip(sqr_dists, a_min=0., a_max=None)).astype(np.float32)).to(anchor_xyz.device)
        semantic_score = torch.zeros_like(surface_distance)
        semantic_distance = torch.zeros_like(surface_distance)

        return face_ids, tri_vids, bary, anchor_xyz, anchor_weights, face_normal, surface_distance, semantic_score, semantic_distance

    def _build_face_anchor(self, canonical_xyz, iteration, prior_state=None):
        self.latest_face_anchor_switch_mask = None
        self.latest_face_anchor_keep_prior_mask = None
        self.latest_face_anchor_best_joint_changed_mask = None
        self.latest_face_anchor_keep_prior_best_face_changed_mask = None
        self.latest_face_anchor_keep_prior_best_joint_changed_mask = None
        self.latest_face_anchor_keep_prior_best_anchor_shift = None
        geom_face_ids, geom_tri_vids, geom_bary, geom_anchor_xyz, geom_anchor_weights, geom_face_normal, geom_surface_distance, _, _ = self._build_geometric_face_anchor(canonical_xyz)
        if self.semantic_knn <= 0:
            return geom_face_ids, geom_tri_vids, geom_bary, geom_anchor_xyz, geom_anchor_weights, geom_face_normal, geom_surface_distance, torch.zeros_like(geom_surface_distance), torch.zeros_like(geom_surface_distance)

        risky_child_mask = None
        risky_child_escape_mask = None
        prior_dominant_joint = None
        prior_face_ids = None
        prior_tri_vids = None
        prior_barycentric = None
        prior_anchor_xyz = None
        prior_weights = None
        prior_normal = None
        prior_refresh_mask = None
        prior_confidence = None
        prior_semantic_score = None
        prior_semantic_distance = None
        prior_surface_distance = None
        source_parent_joint = None
        source_root_parent_joint = None
        risky_child_escape_refresh_mask = None
        risky_child_escape_confidence_mask = None
        risky_child_escape_semantic_mask = None
        risky_child_escape_surface_mask = None
        risky_child_escape_weight_gap_mask = None
        if prior_state is not None and self._state_matches(prior_state, canonical_xyz):
            prior_weights = prior_state.get('anchor_weights', None)
            if not torch.is_tensor(prior_weights) or prior_weights.shape[0] != canonical_xyz.shape[0]:
                prior_weights = None
            else:
                prior_weights = prior_weights.to(device=canonical_xyz.device)
            prior_normal = prior_state.get('anchor_normal', None)
            if not torch.is_tensor(prior_normal) or prior_normal.shape[0] != canonical_xyz.shape[0]:
                prior_normal = None
            else:
                prior_normal = prior_normal.to(device=canonical_xyz.device)
            prior_refresh_mask = prior_state.get('anchor_refresh_mask', None)
            if not torch.is_tensor(prior_refresh_mask) or prior_refresh_mask.shape[0] != canonical_xyz.shape[0]:
                prior_refresh_mask = None
            else:
                prior_refresh_mask = prior_refresh_mask.to(device=canonical_xyz.device, dtype=torch.bool)
            prior_confidence = prior_state.get('anchor_confidence', None)
            if not torch.is_tensor(prior_confidence) or prior_confidence.shape[0] != canonical_xyz.shape[0]:
                prior_confidence = None
            else:
                prior_confidence = prior_confidence.to(device=canonical_xyz.device)
            prior_semantic_score = prior_state.get('semantic_score', None)
            if not torch.is_tensor(prior_semantic_score) or prior_semantic_score.shape[0] != canonical_xyz.shape[0]:
                prior_semantic_score = None
            else:
                prior_semantic_score = prior_semantic_score.to(device=canonical_xyz.device)
            prior_semantic_distance = prior_state.get('semantic_distance', None)
            if not torch.is_tensor(prior_semantic_distance) or prior_semantic_distance.shape[0] != canonical_xyz.shape[0]:
                prior_semantic_distance = None
            else:
                prior_semantic_distance = prior_semantic_distance.to(device=canonical_xyz.device)
            prior_surface_distance = prior_state.get('surface_distance', None)
            if not torch.is_tensor(prior_surface_distance) or prior_surface_distance.shape[0] != canonical_xyz.shape[0]:
                prior_surface_distance = None
            else:
                prior_surface_distance = prior_surface_distance.to(device=canonical_xyz.device)
            risky_child_mask = prior_state.get('densify_risky_child_mask', None)
            if torch.is_tensor(risky_child_mask) and risky_child_mask.shape[0] == canonical_xyz.shape[0]:
                risky_child_mask = risky_child_mask.to(device=canonical_xyz.device, dtype=torch.bool)
            else:
                risky_child_mask = None
            prior_dominant_joint = prior_state.get('dominant_joint', None)
            if torch.is_tensor(prior_dominant_joint) and prior_dominant_joint.shape[0] == canonical_xyz.shape[0]:
                prior_dominant_joint = prior_dominant_joint.to(device=canonical_xyz.device, dtype=torch.long)
            else:
                prior_dominant_joint = None
            prior_face_ids = prior_state.get('anchor_face_ids', None)
            if torch.is_tensor(prior_face_ids) and prior_face_ids.shape[0] == canonical_xyz.shape[0]:
                prior_face_ids = prior_face_ids.to(device=canonical_xyz.device, dtype=torch.long)
            else:
                prior_face_ids = None
            prior_tri_vids = prior_state.get('anchor_vertex_ids', None)
            if torch.is_tensor(prior_tri_vids) and prior_tri_vids.shape[0] == canonical_xyz.shape[0]:
                prior_tri_vids = prior_tri_vids.to(device=canonical_xyz.device, dtype=torch.long)
            else:
                prior_tri_vids = None
            prior_barycentric = prior_state.get('anchor_barycentric', None)
            if torch.is_tensor(prior_barycentric) and prior_barycentric.shape[0] == canonical_xyz.shape[0]:
                prior_barycentric = prior_barycentric.to(device=canonical_xyz.device)
            else:
                prior_barycentric = None
            prior_anchor_xyz = prior_state.get('anchor_xyz', None)
            if torch.is_tensor(prior_anchor_xyz) and prior_anchor_xyz.shape[0] == canonical_xyz.shape[0]:
                prior_anchor_xyz = prior_anchor_xyz.to(device=canonical_xyz.device)
            else:
                prior_anchor_xyz = None
            source_parent_joint = prior_state.get('source_parent_joint', None)
            if torch.is_tensor(source_parent_joint) and source_parent_joint.shape[0] == canonical_xyz.shape[0]:
                source_parent_joint = source_parent_joint.to(device=canonical_xyz.device, dtype=torch.long)
            else:
                source_parent_joint = None
            source_root_parent_joint = prior_state.get('source_root_parent_joint', None)
            if torch.is_tensor(source_root_parent_joint) and source_root_parent_joint.shape[0] == canonical_xyz.shape[0]:
                source_root_parent_joint = source_root_parent_joint.to(device=canonical_xyz.device, dtype=torch.long)
            else:
                source_root_parent_joint = None
            if bool(self.cfg.get('semantic_risky_child_wide_escape_enable', True)) and risky_child_mask is not None and bool(risky_child_mask.any().item()):
                risky_child_escape_mask = torch.zeros_like(risky_child_mask)
                if (
                    prior_refresh_mask is not None
                    and bool(self.cfg.get('semantic_risky_child_wide_escape_include_refresh_mask', False))
                ):
                    risky_child_escape_refresh_mask = prior_refresh_mask
                    risky_child_escape_mask |= prior_refresh_mask
                if prior_confidence is not None:
                    risky_child_escape_confidence_mask = prior_confidence < float(self._scheduled_cfg(
                        'semantic_risky_child_wide_escape_confidence_threshold',
                        iteration,
                        default=0.72,
                        cast=float,
                    ))
                    risky_child_escape_mask |= risky_child_escape_confidence_mask
                if prior_semantic_distance is not None:
                    risky_child_escape_semantic_mask = prior_semantic_distance > float(self._scheduled_cfg(
                        'semantic_risky_child_wide_escape_semantic_distance_threshold',
                        iteration,
                        default=0.025,
                        cast=float,
                    ))
                    risky_child_escape_mask |= risky_child_escape_semantic_mask
                if prior_surface_distance is not None:
                    risky_child_escape_surface_mask = prior_surface_distance > float(self._scheduled_cfg(
                        'semantic_risky_child_wide_escape_surface_distance_threshold',
                        iteration,
                        default=0.01,
                        cast=float,
                    ))
                    risky_child_escape_mask |= risky_child_escape_surface_mask
                if prior_weights is not None and prior_weights.ndim == 2 and prior_weights.shape[1] >= 2:
                    top2 = torch.topk(prior_weights, k=2, dim=-1).values
                    risky_child_escape_weight_gap_mask = (top2[:, 0] - top2[:, 1]) < float(self._scheduled_cfg(
                        'semantic_risky_child_wide_escape_weight_gap_threshold',
                        iteration,
                        default=0.22,
                        cast=float,
                    ))
                    risky_child_escape_mask |= risky_child_escape_weight_gap_mask
                risky_child_escape_mask &= risky_child_mask
                if bool(self.cfg.get('semantic_risky_child_wide_escape_force_any_risky', False)):
                    risky_child_escape_mask |= risky_child_mask

        knn_k = max(int(self.semantic_knn), 1)
        knn_ret = ops.knn_points(canonical_xyz.unsqueeze(0), self.smpl_verts.unsqueeze(0), K=knn_k)
        vertex_ids = knn_ret.idx.squeeze(0)
        vertex_dists = torch.sqrt(knn_ret.dists.squeeze(0).clamp_min(1e-8))
        vertex_weights = 1. / vertex_dists.clamp_min(1e-6)
        vertex_weights = vertex_weights / vertex_weights.sum(dim=-1, keepdim=True)
        query_weights = (self.skinning_weights[vertex_ids] * vertex_weights[..., None]).sum(dim=1)

        candidate_face_ids = self.vertex_face_ids[vertex_ids].reshape(canonical_xyz.shape[0], -1)
        candidate_face_ids = torch.cat([candidate_face_ids, geom_face_ids[:, None]], dim=1)
        geom_dominant_joint = torch.argmax(geom_anchor_weights, dim=-1)

        risky_source_joint = None
        risky_source_joint_valid_mask = None
        risky_source_joint_consensus_mask = None
        risky_source_joint_disagreement_mask = None
        risky_source_prior_mismatch_mask = None
        risky_reference_joint = None
        if risky_child_mask is not None and bool(risky_child_mask.any().item()):
            use_source_joint = bool(self.cfg.get('semantic_risky_child_candidate_use_source_joint_enable', True))
            prefer_source_consensus = bool(self.cfg.get('semantic_risky_child_candidate_prefer_source_consensus', True))
            if use_source_joint and (source_parent_joint is not None or source_root_parent_joint is not None):
                consensus_fallback = str(
                    self.cfg.get('semantic_risky_child_candidate_consensus_fallback', 'parent')
                ).lower()
                risky_source_joint = torch.full(
                    (canonical_xyz.shape[0],),
                    -1,
                    dtype=torch.long,
                    device=canonical_xyz.device,
                )
                parent_valid = torch.zeros_like(risky_source_joint, dtype=torch.bool)
                root_valid = torch.zeros_like(risky_source_joint, dtype=torch.bool)
                if source_parent_joint is not None:
                    parent_valid = source_parent_joint >= 0
                if source_root_parent_joint is not None:
                    root_valid = source_root_parent_joint >= 0
                risky_source_joint_consensus_mask = parent_valid & root_valid
                if source_parent_joint is not None and source_root_parent_joint is not None:
                    risky_source_joint_disagreement_mask = risky_source_joint_consensus_mask & (
                        source_parent_joint != source_root_parent_joint
                    )
                    risky_source_joint_consensus_mask &= source_parent_joint == source_root_parent_joint
                else:
                    risky_source_joint_disagreement_mask = torch.zeros_like(risky_source_joint_consensus_mask)
                    risky_source_joint_consensus_mask &= False
                if bool(prefer_source_consensus):
                    if source_parent_joint is not None:
                        risky_source_joint[risky_source_joint_consensus_mask] = source_parent_joint[risky_source_joint_consensus_mask]
                    unresolved_mask = risky_source_joint < 0
                    if source_parent_joint is not None and consensus_fallback in ('parent', 'either'):
                        risky_source_joint[unresolved_mask & parent_valid] = source_parent_joint[unresolved_mask & parent_valid]
                        unresolved_mask = risky_source_joint < 0
                    if source_root_parent_joint is not None and consensus_fallback in ('root', 'either'):
                        risky_source_joint[unresolved_mask & root_valid] = source_root_parent_joint[unresolved_mask & root_valid]
                        unresolved_mask = risky_source_joint < 0
                    if source_parent_joint is not None and source_root_parent_joint is None and consensus_fallback == 'none':
                        risky_source_joint[parent_valid] = source_parent_joint[parent_valid]
                    if source_root_parent_joint is not None and source_parent_joint is None and consensus_fallback == 'none':
                        risky_source_joint[root_valid] = source_root_parent_joint[root_valid]
                else:
                    if source_parent_joint is not None:
                        risky_source_joint[parent_valid] = source_parent_joint[parent_valid]
                    if source_root_parent_joint is not None:
                        root_only_mask = root_valid & (~parent_valid)
                        risky_source_joint[root_only_mask] = source_root_parent_joint[root_only_mask]
                risky_source_joint_valid_mask = risky_source_joint >= 0
                if prior_dominant_joint is not None:
                    risky_source_prior_mismatch_mask = risky_source_joint_valid_mask & (prior_dominant_joint != risky_source_joint)

        risky_append_prior_face_neighbor_mask = None
        risky_append_prior_face_neighbor_count = None
        if prior_face_ids is not None and risky_child_mask is not None and bool(risky_child_mask.any().item()):
            append_prior_face_neighbor_enable = bool(self.cfg.get(
                'semantic_risky_child_candidate_append_prior_face_neighbor_enable',
                True,
            ))
            append_prior_face_neighbor_with_source_joint = bool(self.cfg.get(
                'semantic_risky_child_candidate_append_prior_face_neighbor_with_source_joint',
                True,
            ))
            append_prior_face_neighbor_hops = int(self._scheduled_cfg(
                'semantic_risky_child_candidate_append_prior_face_neighbor_hops',
                iteration,
                default=1,
                cast=int,
            ))
            risky_append_prior_face_neighbor_mask = risky_child_mask.clone()
            if (
                not append_prior_face_neighbor_with_source_joint
                and risky_source_joint_valid_mask is not None
            ):
                risky_append_prior_face_neighbor_mask &= (~risky_source_joint_valid_mask)
            if (
                append_prior_face_neighbor_enable
                and append_prior_face_neighbor_hops >= 0
                and bool(risky_append_prior_face_neighbor_mask.any().item())
            ):
                expanded_prior_face_ids = self._expand_face_neighbor_ids(prior_face_ids, append_prior_face_neighbor_hops)
                expanded_prior_face_ids = torch.where(
                    risky_append_prior_face_neighbor_mask[:, None],
                    expanded_prior_face_ids,
                    torch.full_like(expanded_prior_face_ids, -1),
                )
                risky_append_prior_face_neighbor_count = (expanded_prior_face_ids >= 0).sum(dim=-1)
                candidate_face_ids = torch.cat([candidate_face_ids, expanded_prior_face_ids], dim=1)

        valid_face_mask = candidate_face_ids >= 0
        safe_face_ids = candidate_face_ids.clamp_min(0)

        risky_switch_guard_mask = None
        risky_source_missing_force_geom_mask = None
        risky_source_mismatch_force_geom_mask = None
        risky_required_joint = None
        risky_same_group_score_ok_mask = None
        risky_same_group_surface_ok_mask = None
        risky_cross_group_score_ok_mask = None
        risky_cross_group_skinning_ok_mask = None
        risky_cross_group_surface_ok_mask = None
        risky_cross_group_prior_reference_mask = None
        risky_cross_group_score_failopen_mask = None
        risky_required_joint_candidate_available_mask = None
        risky_required_joint_failopen_mask = None
        risky_seed_override_candidate_mask = None
        risky_same_face_seed_mask = None
        risky_same_face_joint_change_mask = None
        risky_same_face_weight_shift_mask = None
        risky_same_face_anchor_shift_mask = None
        risky_same_face_bary_shift_mask = None
        if (prior_dominant_joint is not None or risky_source_joint_valid_mask is not None) and risky_child_mask is not None and bool(risky_child_mask.any().item()):
            restricted_risky_child_mask = risky_child_mask
            if (
                bool(self.cfg.get('semantic_risky_child_candidate_apply_wide_escape', False))
                and risky_child_escape_mask is not None
                and bool(risky_child_escape_mask.any().item())
            ):
                restricted_risky_child_mask = risky_child_mask & (~risky_child_escape_mask)
            candidate_face_dominant_joint = self.face_dominant_joint[safe_face_ids]
            candidate_face_group = self.face_joint_group[safe_face_ids]
            risky_reference_joint = geom_dominant_joint.clone()
            if prior_dominant_joint is not None:
                risky_reference_joint = prior_dominant_joint.clone()
            if risky_source_joint_valid_mask is not None and bool(risky_source_joint_valid_mask.any().item()):
                risky_reference_joint = torch.where(
                    risky_source_joint_valid_mask,
                    risky_source_joint,
                    risky_reference_joint,
                )
            prior_group = self.joint_anchor_group[risky_reference_joint][:, None]
            candidate_max_hops = int(self._scheduled_cfg(
                'semantic_risky_child_candidate_max_hops',
                iteration,
                default=1,
                cast=int,
            ))
            require_prior_joint_match = bool(self.cfg.get('semantic_risky_child_candidate_require_prior_joint_match', False))
            allow_cross_group = bool(self.cfg.get('semantic_risky_child_candidate_allow_cross_group', False))
            allow_prior_face = bool(self.cfg.get('semantic_risky_child_candidate_allow_prior_face', True))
            local_neighbor_hops = int(self._scheduled_cfg(
                'semantic_risky_child_candidate_face_neighbor_hops',
                iteration,
                default=1,
                cast=int,
            ))
            require_prior_face_neighbor = bool(self.cfg.get('semantic_risky_child_candidate_require_prior_face_neighbor', True))
            require_prior_face_neighbor_with_source_joint = bool(
                self.cfg.get('semantic_risky_child_candidate_require_prior_face_neighbor_with_source_joint', False)
            )

            base_valid_face_mask = valid_face_mask.clone()
            risky_valid_face_mask = base_valid_face_mask.clone()
            risky_joint_gate_mask = risky_valid_face_mask.clone()
            if require_prior_joint_match:
                risky_joint_gate_mask &= candidate_face_dominant_joint == risky_reference_joint[:, None]
            else:
                same_group = candidate_face_group == prior_group
                if not allow_cross_group:
                    risky_joint_gate_mask &= same_group
                if candidate_max_hops >= 0:
                    candidate_hop_distance = self.joint_hop_distance[risky_reference_joint[:, None], candidate_face_dominant_joint]
                    risky_joint_gate_mask &= candidate_hop_distance <= candidate_max_hops
            risky_valid_face_mask = risky_joint_gate_mask.clone()

            apply_prior_face_mask = restricted_risky_child_mask
            if bool(self.cfg.get('semantic_risky_child_candidate_disable_prior_face_on_source_mismatch', True)) and risky_source_prior_mismatch_mask is not None:
                apply_prior_face_mask = apply_prior_face_mask & (~risky_source_prior_mismatch_mask)
            if (
                not require_prior_face_neighbor_with_source_joint
                and risky_source_joint_valid_mask is not None
            ):
                apply_prior_face_mask = apply_prior_face_mask & (~risky_source_joint_valid_mask)
            if require_prior_face_neighbor and prior_face_ids is not None and local_neighbor_hops >= 0:
                local_face_ids = self._expand_face_neighbor_ids(prior_face_ids, local_neighbor_hops)
                local_face_valid = local_face_ids >= 0
                face_local_mask = (
                    (candidate_face_ids[:, :, None] == local_face_ids[:, None, :])
                    & local_face_valid[:, None, :]
                ).any(dim=-1)
                risky_prior_face_mask = torch.where(
                    apply_prior_face_mask[:, None],
                    risky_valid_face_mask & face_local_mask,
                    risky_valid_face_mask,
                )
                risky_valid_face_mask = torch.where(
                    apply_prior_face_mask[:, None],
                    risky_prior_face_mask,
                    risky_valid_face_mask,
                )
            else:
                risky_prior_face_mask = risky_valid_face_mask.clone()

            if allow_prior_face and prior_face_ids is not None:
                prior_face_match = candidate_face_ids == prior_face_ids[:, None]
                prior_face_match &= apply_prior_face_mask[:, None]
                risky_valid_face_mask |= prior_face_match
            risky_pre_fallback_face_mask = risky_valid_face_mask.clone()

            # Keep the geometric anchor candidate as a fallback so risky children never end up candidate-empty.
            risky_valid_face_mask[:, -1] = valid_face_mask[:, -1]
            valid_face_mask = torch.where(restricted_risky_child_mask[:, None], risky_valid_face_mask, valid_face_mask)

            if bool(self.cfg.get('semantic_risky_child_debug_verbose', False)) and bool(restricted_risky_child_mask.any().item()):
                restricted_counts = restricted_risky_child_mask
                base_candidate_count = base_valid_face_mask[restricted_counts].sum(dim=-1)
                joint_gate_candidate_count = risky_joint_gate_mask[restricted_counts].sum(dim=-1)
                prior_face_candidate_count = risky_prior_face_mask[restricted_counts].sum(dim=-1)
                pre_fallback_candidate_count = risky_pre_fallback_face_mask[restricted_counts].sum(dim=-1)
                final_candidate_count = risky_valid_face_mask[restricted_counts].sum(dim=-1)
                append_prior_face_candidate_count = None
                append_prior_face_point_count = 0
                if (
                    torch.is_tensor(risky_append_prior_face_neighbor_mask)
                    and torch.is_tensor(risky_append_prior_face_neighbor_count)
                ):
                    append_prior_face_candidate_count = risky_append_prior_face_neighbor_count[restricted_counts]
                    append_prior_face_point_count = int(
                        (risky_append_prior_face_neighbor_mask & restricted_counts).sum().item()
                    )
                source_backed_count = int(
                    (restricted_counts & risky_source_joint_valid_mask).sum().item()
                ) if torch.is_tensor(risky_source_joint_valid_mask) else 0
                debug_message = (
                    '[ExplicitBinding] risky-child candidate mask '
                    f'iter={iteration} risky={int(restricted_counts.sum().item())} '
                    f'source_backed={source_backed_count} '
                    f'base_mean={float(base_candidate_count.float().mean().item()):.2f} '
                    f'joint_mean={float(joint_gate_candidate_count.float().mean().item()):.2f} '
                    f'prior_mean={float(prior_face_candidate_count.float().mean().item()):.2f} '
                    f'prefallback_mean={float(pre_fallback_candidate_count.float().mean().item()):.2f} '
                    f'final_mean={float(final_candidate_count.float().mean().item()):.2f} '
                )
                if append_prior_face_candidate_count is not None:
                    debug_message += (
                        f'append_points={append_prior_face_point_count} '
                        f'append_mean={float(append_prior_face_candidate_count.float().mean().item()):.2f} '
                    )
                debug_message += (
                    f'prefallback_empty={int((pre_fallback_candidate_count == 0).sum().item())} '
                    f'geom_only={int((final_candidate_count == 1).sum().item())}'
                )
                print(debug_message)

        tri_vids = self.faces[safe_face_ids]
        tri_verts = self.smpl_verts[tri_vids]
        tri_weights = self.skinning_weights[tri_vids]
        bary, anchor_xyz, surface_distance = self._project_points_to_triangles(canonical_xyz, tri_verts)
        anchor_weights = (tri_weights * bary[..., None]).sum(dim=2)
        face_normal = torch.cross(tri_verts[:, :, 1] - tri_verts[:, :, 0], tri_verts[:, :, 2] - tri_verts[:, :, 0], dim=-1)
        face_normal = safe_normalize(face_normal)

        score_query_weights = query_weights
        risky_query_prior_reference_mask = None
        if (
            bool(self.cfg.get('semantic_risky_child_query_weights_use_prior_on_geom_mismatch', True))
            and risky_child_mask is not None
            and bool(risky_child_mask.any().item())
            and risky_reference_joint is not None
            and prior_weights is not None
        ):
            risky_query_prior_reference_mask = risky_child_mask.clone()
            if risky_source_joint_valid_mask is not None:
                risky_query_prior_reference_mask &= risky_source_joint_valid_mask
            if (
                bool(self.cfg.get('semantic_risky_child_query_weights_prior_require_source_consensus', True))
                and risky_source_joint_consensus_mask is not None
            ):
                risky_query_prior_reference_mask &= risky_source_joint_consensus_mask
            if risky_source_prior_mismatch_mask is not None:
                risky_query_prior_reference_mask &= (~risky_source_prior_mismatch_mask)
            risky_query_prior_reference_mask &= (geom_dominant_joint != risky_reference_joint)
            if prior_dominant_joint is not None:
                risky_query_prior_reference_mask &= (prior_dominant_joint == risky_reference_joint)
            score_query_weights = torch.where(
                risky_query_prior_reference_mask[:, None],
                prior_weights,
                score_query_weights,
            )

        geom_skinning_distance = torch.mean(torch.abs(geom_anchor_weights - score_query_weights), dim=-1)
        skinning_distance = torch.mean(torch.abs(anchor_weights - score_query_weights[:, None, :]), dim=-1)
        total_score = self.semantic_geo_weight * surface_distance + self.semantic_skinning_weight * skinning_distance
        geom_total_score = self.semantic_geo_weight * geom_surface_distance + self.semantic_skinning_weight * geom_skinning_distance

        if prior_weights is not None and prior_normal is not None:
            prior_distance = torch.mean(torch.abs(anchor_weights - prior_weights[:, None, :]), dim=-1)
            normal_consistency = 1. - torch.abs((face_normal * prior_normal[:, None, :]).sum(dim=-1))
            prior_penalty = self.semantic_prior_weight * prior_distance + self.semantic_normal_weight * normal_consistency
            if risky_child_escape_mask is not None and bool(risky_child_escape_mask.any().item()):
                prior_penalty = torch.where(
                    risky_child_escape_mask[:, None],
                    torch.zeros_like(prior_penalty),
                    prior_penalty,
                )
            total_score = total_score + prior_penalty
            geom_prior_distance = torch.mean(torch.abs(geom_anchor_weights - prior_weights), dim=-1)
            geom_normal_consistency = 1. - torch.abs((geom_face_normal * prior_normal).sum(dim=-1))
            geom_prior_penalty = self.semantic_prior_weight * geom_prior_distance + self.semantic_normal_weight * geom_normal_consistency
            if risky_child_escape_mask is not None and bool(risky_child_escape_mask.any().item()):
                geom_prior_penalty = torch.where(
                    risky_child_escape_mask,
                    torch.zeros_like(geom_prior_penalty),
                    geom_prior_penalty,
                )
            geom_total_score = geom_total_score + geom_prior_penalty

        total_score[~valid_face_mask] = 1e6
        best_idx = torch.argmin(total_score, dim=1)
        gather_idx = torch.arange(canonical_xyz.shape[0], device=canonical_xyz.device)

        best_face_ids = safe_face_ids[gather_idx, best_idx]
        best_surface_distance = surface_distance[gather_idx, best_idx]
        best_skinning_distance = skinning_distance[gather_idx, best_idx]
        best_total_score = total_score[gather_idx, best_idx]
        best_anchor_xyz = anchor_xyz[gather_idx, best_idx]
        best_anchor_weights = anchor_weights[gather_idx, best_idx]
        best_face_normal = face_normal[gather_idx, best_idx]

        best_dominant_joint = torch.argmax(best_anchor_weights, dim=-1)
        geom_group = self.joint_anchor_group[geom_dominant_joint]
        best_group = self.joint_anchor_group[best_dominant_joint]
        same_group = best_group == geom_group
        hop_distance = self.joint_hop_distance[geom_dominant_joint, best_dominant_joint]
        semantic_switch_score_margin = float(self._scheduled_cfg('semantic_switch_score_margin', iteration, default=self.semantic_switch_score_margin, cast=float))
        semantic_switch_skinning_margin = float(self._scheduled_cfg('semantic_switch_skinning_margin', iteration, default=self.semantic_switch_skinning_margin, cast=float))
        semantic_switch_surface_tolerance = float(self._scheduled_cfg('semantic_switch_surface_tolerance', iteration, default=self.semantic_switch_surface_tolerance, cast=float))
        semantic_same_group_score_margin = float(self._scheduled_cfg('semantic_same_group_score_margin', iteration, default=self.semantic_same_group_score_margin, cast=float))
        semantic_same_group_surface_tolerance = float(self._scheduled_cfg('semantic_same_group_surface_tolerance', iteration, default=self.semantic_same_group_surface_tolerance, cast=float))
        semantic_same_group_normal_min_dot = float(self._scheduled_cfg('semantic_same_group_normal_min_dot', iteration, default=self.semantic_same_group_normal_min_dot, cast=float))
        semantic_same_group_max_hops = int(self._scheduled_cfg('semantic_same_group_max_hops', iteration, default=self.semantic_same_group_max_hops, cast=int))
        semantic_arm_group_max_hops = int(self._scheduled_cfg('semantic_arm_group_max_hops', iteration, default=self.semantic_arm_group_max_hops, cast=int))

        same_group_max_hops = torch.full_like(hop_distance, semantic_same_group_max_hops)
        if semantic_arm_group_max_hops >= 0:
            arm_group_mask = geom_group == 3
            same_group_max_hops = torch.where(
                arm_group_mask,
                torch.full_like(same_group_max_hops, semantic_arm_group_max_hops),
                same_group_max_hops,
            )
        same_group_hop_ok = (same_group_max_hops < 0) | (hop_distance <= same_group_max_hops)
        same_side = (best_face_normal * geom_face_normal).sum(dim=-1) >= semantic_same_group_normal_min_dot

        base_switch_mask = best_face_ids != geom_face_ids
        same_face_mask = ~base_switch_mask
        anchor_weight_shift = torch.mean(torch.abs(best_anchor_weights - geom_anchor_weights), dim=-1)
        anchor_position_shift = torch.norm(anchor_xyz[gather_idx, best_idx] - geom_anchor_xyz, dim=-1)
        bary_shift = torch.mean(torch.abs(bary[gather_idx, best_idx] - geom_bary), dim=-1)
        dominant_joint_change_mask = best_dominant_joint != geom_dominant_joint
        same_group_switch_seed = base_switch_mask & same_group & same_side & same_group_hop_ok
        same_group_switch = same_group_switch_seed & (
            best_total_score + semantic_same_group_score_margin < geom_total_score
        )
        same_group_switch = same_group_switch & (
            best_surface_distance <= geom_surface_distance + semantic_same_group_surface_tolerance
        )

        cross_group_switch_seed = base_switch_mask & (~same_group)
        cross_group_switch = cross_group_switch_seed & (
            best_total_score + semantic_switch_score_margin < geom_total_score
        )
        cross_group_switch = cross_group_switch & (
            best_skinning_distance + semantic_switch_skinning_margin < geom_skinning_distance
        )
        cross_group_switch = cross_group_switch & (
            best_surface_distance <= geom_surface_distance + semantic_switch_surface_tolerance
        )

        if bool(self.cfg.get('semantic_risky_child_switch_guard_enable', False)) and risky_child_mask is not None and bool(risky_child_mask.any().item()):
            guarded_risky_child_mask = risky_child_mask
            if (
                bool(self.cfg.get('semantic_risky_child_switch_guard_apply_wide_escape', False))
                and risky_child_escape_mask is not None
                and bool(risky_child_escape_mask.any().item())
            ):
                guarded_risky_child_mask = risky_child_mask & (~risky_child_escape_mask)
            risky_switch_guard_mask = guarded_risky_child_mask
            risky_same_group_max_hops = int(self._scheduled_cfg(
                'semantic_risky_child_same_group_max_hops',
                iteration,
                default=1,
                cast=int,
            ))
            risky_score_margin_relax = float(self._scheduled_cfg(
                'semantic_risky_child_score_margin_relax',
                iteration,
                default=0.004,
                cast=float,
            ))
            risky_skinning_margin_relax = float(self._scheduled_cfg(
                'semantic_risky_child_skinning_margin_relax',
                iteration,
                default=0.03,
                cast=float,
            ))
            risky_surface_tolerance = float(self._scheduled_cfg(
                'semantic_risky_child_surface_tolerance',
                iteration,
                default=0.008,
                cast=float,
            ))
            risky_allow_cross_group_switch = bool(self.cfg.get('semantic_risky_child_allow_cross_group_switch', True))
            risky_require_prior_joint_match = bool(self.cfg.get('semantic_risky_child_require_prior_joint_match', True))
            risky_force_geometric_without_source_joint = bool(self.cfg.get(
                'semantic_risky_child_force_geometric_without_source_joint',
                True,
            ))
            risky_force_geometric_on_source_mismatch = bool(self.cfg.get(
                'semantic_risky_child_force_geometric_on_source_mismatch',
                True,
            ))
            risky_seed_override_enable = bool(self.cfg.get(
                'semantic_risky_child_switch_seed_override_with_source_joint_match',
                True,
            ))
            risky_seed_override_ignore_same_side = bool(self.cfg.get(
                'semantic_risky_child_switch_seed_override_ignore_same_side',
                True,
            ))
            risky_seed_override_ignore_geom_hop = bool(self.cfg.get(
                'semantic_risky_child_switch_seed_override_ignore_geom_hop',
                True,
            ))

            risky_same_group_switch = same_group_switch_seed.clone()
            if risky_same_group_max_hops >= 0:
                risky_same_group_switch = risky_same_group_switch & (hop_distance <= risky_same_group_max_hops)
            risky_same_group_score_margin = semantic_same_group_score_margin - risky_score_margin_relax
            risky_same_group_score_ok_mask = (
                best_total_score + risky_same_group_score_margin < geom_total_score
            )
            risky_same_group_switch = risky_same_group_switch & risky_same_group_score_ok_mask
            risky_same_group_surface_ok_mask = (
                best_surface_distance <= geom_surface_distance + max(semantic_same_group_surface_tolerance, risky_surface_tolerance)
            )
            risky_same_group_switch = risky_same_group_switch & risky_same_group_surface_ok_mask
            if risky_source_joint_valid_mask is not None and bool(risky_source_joint_valid_mask.any().item()) and risky_source_joint is not None:
                risky_required_joint = torch.where(
                    risky_source_joint_valid_mask,
                    risky_source_joint,
                    prior_dominant_joint if prior_dominant_joint is not None else geom_dominant_joint,
                )
            elif prior_dominant_joint is not None:
                risky_required_joint = prior_dominant_joint
            risky_same_face_seed_enable = bool(self.cfg.get(
                'semantic_risky_child_same_face_seed_enable',
                False,
            ))
            risky_same_face_joint_change_enable = bool(self.cfg.get(
                'semantic_risky_child_same_face_joint_change_enable',
                True,
            ))
            risky_same_face_require_source_joint = bool(self.cfg.get(
                'semantic_risky_child_same_face_seed_require_source_joint',
                True,
            ))
            risky_same_face_require_required_joint_match = bool(self.cfg.get(
                'semantic_risky_child_same_face_seed_require_required_joint_match',
                True,
            ))
            risky_same_face_weight_delta_threshold = float(self._scheduled_cfg(
                'semantic_risky_child_same_face_weight_delta_threshold',
                iteration,
                default=0.02,
                cast=float,
            ))
            risky_same_face_anchor_shift_threshold = float(self._scheduled_cfg(
                'semantic_risky_child_same_face_anchor_shift_threshold',
                iteration,
                default=0.003,
                cast=float,
            ))
            risky_same_face_bary_shift_threshold = float(self._scheduled_cfg(
                'semantic_risky_child_same_face_bary_shift_threshold',
                iteration,
                default=0.04,
                cast=float,
            ))
            if risky_same_face_seed_enable:
                risky_same_face_joint_change_mask = same_face_mask & same_group & dominant_joint_change_mask
                risky_same_face_weight_shift_mask = same_face_mask & same_group & (
                    anchor_weight_shift >= risky_same_face_weight_delta_threshold
                )
                risky_same_face_anchor_shift_mask = same_face_mask & same_group & (
                    anchor_position_shift >= risky_same_face_anchor_shift_threshold
                )
                risky_same_face_bary_shift_mask = same_face_mask & same_group & (
                    bary_shift >= risky_same_face_bary_shift_threshold
                )
                risky_same_face_seed_mask = guarded_risky_child_mask & same_group
                if not risky_seed_override_ignore_same_side:
                    risky_same_face_seed_mask &= same_side
                if not risky_seed_override_ignore_geom_hop:
                    risky_same_face_seed_mask &= same_group_hop_ok
                if risky_same_face_require_source_joint and risky_source_joint_valid_mask is not None:
                    risky_same_face_seed_mask &= risky_source_joint_valid_mask
                if risky_same_face_require_required_joint_match and risky_required_joint is not None:
                    risky_same_face_seed_mask &= (best_dominant_joint == risky_required_joint)
                risky_same_face_trigger_mask = (
                    risky_same_face_weight_shift_mask
                    | risky_same_face_anchor_shift_mask
                    | risky_same_face_bary_shift_mask
                )
                if risky_same_face_joint_change_enable:
                    risky_same_face_trigger_mask = risky_same_face_trigger_mask | risky_same_face_joint_change_mask
                risky_same_face_seed_mask &= risky_same_face_trigger_mask
                risky_same_group_switch = risky_same_group_switch | risky_same_face_seed_mask
            if (
                risky_seed_override_enable
                and risky_required_joint is not None
                and risky_source_joint_valid_mask is not None
            ):
                risky_seed_override_candidate_mask = (
                    guarded_risky_child_mask
                    & risky_source_joint_valid_mask
                    & (best_dominant_joint == risky_required_joint)
                )
                risky_override_seed = base_switch_mask & same_group
                if not risky_seed_override_ignore_same_side:
                    risky_override_seed &= same_side
                if not risky_seed_override_ignore_geom_hop:
                    risky_override_seed &= same_group_hop_ok
                risky_same_group_switch = torch.where(
                    risky_seed_override_candidate_mask,
                    risky_same_group_switch | risky_override_seed,
                    risky_same_group_switch,
                )
            risky_required_joint_failopen_enable = bool(self.cfg.get(
                'semantic_risky_child_required_joint_failopen_enable',
                True,
            ))
            risky_required_joint_failopen_require_source_consensus = bool(self.cfg.get(
                'semantic_risky_child_required_joint_failopen_require_source_consensus',
                True,
            ))
            risky_required_joint_failopen_same_group_only = bool(self.cfg.get(
                'semantic_risky_child_required_joint_failopen_same_group_only',
                True,
            ))
            if (
                risky_required_joint_failopen_enable
                and risky_required_joint is not None
                and risky_source_joint_valid_mask is not None
                and bool(risky_source_joint_valid_mask.any().item())
            ):
                risky_required_joint_candidate_available_mask = (
                    risky_pre_fallback_face_mask
                    & (candidate_face_dominant_joint == risky_required_joint[:, None])
                ).any(dim=-1)
                risky_required_joint_failopen_mask = guarded_risky_child_mask & risky_source_joint_valid_mask
                if risky_required_joint_failopen_require_source_consensus and risky_source_joint_consensus_mask is not None:
                    risky_required_joint_failopen_mask &= risky_source_joint_consensus_mask
                if risky_required_joint_failopen_same_group_only:
                    required_group = self.joint_anchor_group[risky_required_joint]
                    risky_required_joint_failopen_mask &= best_group == required_group
                risky_required_joint_failopen_mask &= ~risky_required_joint_candidate_available_mask
            else:
                risky_required_joint_failopen_mask = torch.zeros_like(guarded_risky_child_mask)
            if risky_require_prior_joint_match and risky_required_joint is not None:
                risky_same_group_switch = torch.where(
                    risky_required_joint_failopen_mask,
                    risky_same_group_switch,
                    risky_same_group_switch & (best_dominant_joint == risky_required_joint),
                )
            if risky_force_geometric_without_source_joint and risky_source_joint_valid_mask is not None:
                risky_source_missing_force_geom_mask = guarded_risky_child_mask & (~risky_source_joint_valid_mask)
                risky_same_group_switch = torch.where(
                    risky_source_missing_force_geom_mask,
                    torch.zeros_like(risky_same_group_switch),
                    risky_same_group_switch,
                )
            if risky_force_geometric_on_source_mismatch and risky_source_prior_mismatch_mask is not None:
                risky_source_mismatch_force_geom_mask = guarded_risky_child_mask & risky_source_prior_mismatch_mask
                risky_same_group_switch = torch.where(
                    risky_source_mismatch_force_geom_mask,
                    torch.zeros_like(risky_same_group_switch),
                    risky_same_group_switch,
                )
            same_group_switch = torch.where(guarded_risky_child_mask, risky_same_group_switch, same_group_switch)

            if risky_allow_cross_group_switch:
                risky_cross_group_switch = cross_group_switch_seed.clone()
                risky_cross_group_score_margin = semantic_switch_score_margin - risky_score_margin_relax
                risky_cross_group_skinning_margin = max(
                    semantic_switch_skinning_margin - risky_skinning_margin_relax,
                    0.0,
                )
                risky_cross_group_score_reference = geom_total_score
                risky_cross_group_skinning_reference = geom_skinning_distance
                risky_cross_group_surface_reference = geom_surface_distance
                risky_cross_group_prior_reference_mask = torch.zeros_like(risky_cross_group_switch)
                risky_cross_group_use_prior_reference = bool(self.cfg.get(
                    'semantic_risky_child_cross_group_use_prior_reference_on_geom_mismatch',
                    True,
                ))
                risky_cross_group_prior_reference_require_source_consensus = bool(self.cfg.get(
                    'semantic_risky_child_cross_group_prior_reference_require_source_consensus',
                    True,
                ))
                if (
                    risky_cross_group_use_prior_reference
                    and risky_required_joint is not None
                    and prior_dominant_joint is not None
                    and prior_semantic_score is not None
                    and prior_semantic_distance is not None
                    and prior_surface_distance is not None
                ):
                    risky_cross_group_prior_reference_mask = guarded_risky_child_mask.clone()
                    if risky_source_joint_valid_mask is not None:
                        risky_cross_group_prior_reference_mask &= risky_source_joint_valid_mask
                    if (
                        risky_cross_group_prior_reference_require_source_consensus
                        and risky_source_joint_consensus_mask is not None
                    ):
                        risky_cross_group_prior_reference_mask &= risky_source_joint_consensus_mask
                    risky_cross_group_prior_reference_mask &= (geom_dominant_joint != risky_required_joint)
                    risky_cross_group_prior_reference_mask &= (prior_dominant_joint == risky_required_joint)
                    if risky_source_prior_mismatch_mask is not None:
                        risky_cross_group_prior_reference_mask &= (~risky_source_prior_mismatch_mask)
                    risky_cross_group_score_reference = torch.where(
                        risky_cross_group_prior_reference_mask,
                        torch.maximum(geom_total_score, prior_semantic_score),
                        risky_cross_group_score_reference,
                    )
                    risky_cross_group_skinning_reference = torch.where(
                        risky_cross_group_prior_reference_mask,
                        torch.maximum(geom_skinning_distance, prior_semantic_distance),
                        risky_cross_group_skinning_reference,
                    )
                    risky_cross_group_surface_reference = torch.where(
                        risky_cross_group_prior_reference_mask,
                        torch.maximum(geom_surface_distance, prior_surface_distance),
                        risky_cross_group_surface_reference,
                    )
                risky_cross_group_score_ok_mask = (
                    best_total_score + risky_cross_group_score_margin < risky_cross_group_score_reference
                )
                risky_cross_group_switch = risky_cross_group_switch & risky_cross_group_score_ok_mask
                risky_cross_group_skinning_ok_mask = (
                    best_skinning_distance + risky_cross_group_skinning_margin < risky_cross_group_skinning_reference
                )
                risky_cross_group_switch = risky_cross_group_switch & risky_cross_group_skinning_ok_mask
                risky_cross_group_surface_ok_mask = (
                    best_surface_distance <= risky_cross_group_surface_reference + max(semantic_switch_surface_tolerance, risky_surface_tolerance)
                )
                risky_cross_group_switch = risky_cross_group_switch & risky_cross_group_surface_ok_mask
                risky_cross_group_score_failopen_enable = bool(self.cfg.get(
                    'semantic_risky_child_cross_group_score_failopen_with_prior_query',
                    True,
                ))
                if risky_cross_group_score_failopen_enable:
                    risky_cross_group_score_failopen_mask = cross_group_switch_seed.clone()
                    if torch.is_tensor(risky_query_prior_reference_mask):
                        risky_cross_group_score_failopen_mask &= risky_query_prior_reference_mask
                    if risky_required_joint is not None:
                        risky_cross_group_score_failopen_mask &= (best_dominant_joint == risky_required_joint)
                    risky_cross_group_score_failopen_mask &= risky_cross_group_skinning_ok_mask
                    risky_cross_group_score_failopen_mask &= risky_cross_group_surface_ok_mask
                    risky_cross_group_switch = risky_cross_group_switch | risky_cross_group_score_failopen_mask
                if risky_force_geometric_without_source_joint and risky_source_joint_valid_mask is not None:
                    risky_cross_group_switch = torch.where(
                        risky_source_missing_force_geom_mask,
                        torch.zeros_like(risky_cross_group_switch),
                        risky_cross_group_switch,
                    )
                if risky_force_geometric_on_source_mismatch and risky_source_prior_mismatch_mask is not None:
                    risky_cross_group_switch = torch.where(
                        risky_source_mismatch_force_geom_mask,
                        torch.zeros_like(risky_cross_group_switch),
                        risky_cross_group_switch,
                    )
                cross_group_switch = torch.where(guarded_risky_child_mask, risky_cross_group_switch, cross_group_switch)
            else:
                cross_group_switch = cross_group_switch & (~guarded_risky_child_mask)

        switch_mask = same_group_switch | cross_group_switch
        if torch.is_tensor(switch_mask) and switch_mask.shape[0] == canonical_xyz.shape[0]:
            if torch.is_tensor(risky_child_mask) and risky_child_mask.shape[0] == canonical_xyz.shape[0]:
                self.latest_face_anchor_switch_mask = (switch_mask & risky_child_mask).detach().clone()
            else:
                self.latest_face_anchor_switch_mask = switch_mask.detach().clone()
        risky_geom_fallback_mask = None
        risky_geom_fallback_joint_match_mask = None
        risky_geom_fallback_joint_mismatch_mask = None
        risky_geom_fallback_keep_prior_mask = None
        risky_geom_fallback_missing_prior_mask = None
        risky_best_joint_changed_mask = None
        risky_geom_fallback_keep_prior_best_face_changed_mask = None
        risky_geom_fallback_keep_prior_best_joint_changed_mask = None
        risky_geom_fallback_keep_prior_best_anchor_shift = None

        face_ids = torch.where(switch_mask, best_face_ids, geom_face_ids)
        tri_vids = torch.where(switch_mask.unsqueeze(-1), tri_vids[gather_idx, best_idx], geom_tri_vids)
        bary = torch.where(switch_mask.unsqueeze(-1), bary[gather_idx, best_idx], geom_bary)
        anchor_xyz = torch.where(switch_mask.unsqueeze(-1), anchor_xyz[gather_idx, best_idx], geom_anchor_xyz)
        anchor_weights = torch.where(switch_mask.unsqueeze(-1), anchor_weights[gather_idx, best_idx], geom_anchor_weights)
        face_normal = torch.where(switch_mask.unsqueeze(-1), face_normal[gather_idx, best_idx], geom_face_normal)
        surface_distance = torch.where(switch_mask, best_surface_distance, geom_surface_distance)
        semantic_score = torch.where(switch_mask, best_total_score, geom_total_score)
        semantic_distance = torch.where(switch_mask, best_skinning_distance, geom_skinning_distance)
        if prior_dominant_joint is not None and prior_dominant_joint.shape[0] == canonical_xyz.shape[0]:
            risky_best_joint_changed_mask = (best_dominant_joint != prior_dominant_joint)
            if torch.is_tensor(risky_child_mask) and risky_child_mask.shape[0] == canonical_xyz.shape[0]:
                risky_best_joint_changed_mask &= risky_child_mask

        risky_geom_fallback_joint_gate_enable = bool(self.cfg.get(
            'semantic_risky_child_geom_fallback_joint_gate_enable',
            True,
        ))
        risky_geom_fallback_keep_prior_enable = bool(self.cfg.get(
            'semantic_risky_child_geom_fallback_keep_prior_enable',
            True,
        ))
        if (
            risky_geom_fallback_joint_gate_enable
            and risky_required_joint is not None
            and risky_switch_guard_mask is not None
            and bool(risky_switch_guard_mask.any().item())
            and risky_source_joint_valid_mask is not None
            and bool(risky_source_joint_valid_mask.any().item())
        ):
            risky_geom_fallback_mask = risky_switch_guard_mask & risky_source_joint_valid_mask & (~switch_mask)
            risky_geom_fallback_joint_match_mask = risky_geom_fallback_mask & (geom_dominant_joint == risky_required_joint)
            risky_geom_fallback_joint_mismatch_mask = risky_geom_fallback_mask & (geom_dominant_joint != risky_required_joint)
            prior_binding_available = (
                prior_face_ids is not None
                and prior_tri_vids is not None
                and prior_barycentric is not None
                and prior_anchor_xyz is not None
                and prior_weights is not None
                and prior_normal is not None
                and prior_surface_distance is not None
                and prior_semantic_distance is not None
            )
            risky_geom_fallback_keep_prior_mask = torch.zeros_like(risky_geom_fallback_mask)
            risky_geom_fallback_missing_prior_mask = torch.zeros_like(risky_geom_fallback_mask)
            if bool(risky_geom_fallback_joint_mismatch_mask.any().item()):
                if torch.is_tensor(risky_required_joint_failopen_mask) and risky_required_joint_failopen_mask.shape[0] == risky_geom_fallback_joint_mismatch_mask.shape[0]:
                    risky_geom_fallback_joint_mismatch_mask = (
                        risky_geom_fallback_joint_mismatch_mask
                        & (~risky_required_joint_failopen_mask)
                    )
                if risky_geom_fallback_keep_prior_enable and prior_binding_available:
                    risky_geom_fallback_keep_prior_mask = risky_geom_fallback_joint_mismatch_mask
                else:
                    risky_geom_fallback_missing_prior_mask = risky_geom_fallback_joint_mismatch_mask
            if bool(risky_geom_fallback_keep_prior_mask.any().item()):
                risky_geom_fallback_keep_prior_best_face_changed_mask = torch.zeros_like(risky_geom_fallback_keep_prior_mask)
                if prior_face_ids is not None and prior_face_ids.shape[0] == canonical_xyz.shape[0]:
                    risky_geom_fallback_keep_prior_best_face_changed_mask = (
                        risky_geom_fallback_keep_prior_mask & (best_face_ids != prior_face_ids)
                    )
                risky_geom_fallback_keep_prior_best_joint_changed_mask = torch.zeros_like(risky_geom_fallback_keep_prior_mask)
                if prior_dominant_joint is not None and prior_dominant_joint.shape[0] == canonical_xyz.shape[0]:
                    risky_geom_fallback_keep_prior_best_joint_changed_mask = (
                        risky_geom_fallback_keep_prior_mask & (best_dominant_joint != prior_dominant_joint)
                    )
                risky_geom_fallback_keep_prior_best_anchor_shift = torch.zeros_like(geom_surface_distance)
                if prior_anchor_xyz is not None and prior_anchor_xyz.shape == best_anchor_xyz.shape:
                    risky_geom_fallback_keep_prior_best_anchor_shift = torch.where(
                        risky_geom_fallback_keep_prior_mask,
                        torch.norm(best_anchor_xyz - prior_anchor_xyz, dim=-1),
                        torch.zeros_like(geom_surface_distance),
                    )
                face_ids = torch.where(risky_geom_fallback_keep_prior_mask, prior_face_ids, face_ids)
                tri_vids = torch.where(risky_geom_fallback_keep_prior_mask.unsqueeze(-1), prior_tri_vids, tri_vids)
                bary = torch.where(risky_geom_fallback_keep_prior_mask.unsqueeze(-1), prior_barycentric, bary)
                anchor_xyz = torch.where(risky_geom_fallback_keep_prior_mask.unsqueeze(-1), prior_anchor_xyz, anchor_xyz)
                anchor_weights = torch.where(risky_geom_fallback_keep_prior_mask.unsqueeze(-1), prior_weights, anchor_weights)
                face_normal = torch.where(risky_geom_fallback_keep_prior_mask.unsqueeze(-1), prior_normal, face_normal)
                surface_distance = torch.where(risky_geom_fallback_keep_prior_mask, prior_surface_distance, surface_distance)
                if prior_semantic_score is not None:
                    semantic_score = torch.where(risky_geom_fallback_keep_prior_mask, prior_semantic_score, semantic_score)
                semantic_distance = torch.where(risky_geom_fallback_keep_prior_mask, prior_semantic_distance, semantic_distance)
        if torch.is_tensor(risky_geom_fallback_keep_prior_mask) and risky_geom_fallback_keep_prior_mask.shape[0] == canonical_xyz.shape[0]:
            if torch.is_tensor(risky_child_mask) and risky_child_mask.shape[0] == canonical_xyz.shape[0]:
                self.latest_face_anchor_keep_prior_mask = (risky_geom_fallback_keep_prior_mask & risky_child_mask).detach().clone()
            else:
                self.latest_face_anchor_keep_prior_mask = risky_geom_fallback_keep_prior_mask.detach().clone()
        if (
            torch.is_tensor(risky_best_joint_changed_mask)
            and risky_best_joint_changed_mask.shape[0] == canonical_xyz.shape[0]
        ):
            self.latest_face_anchor_best_joint_changed_mask = (
                risky_best_joint_changed_mask.detach().clone()
            )
        if (
            torch.is_tensor(risky_geom_fallback_keep_prior_best_face_changed_mask)
            and risky_geom_fallback_keep_prior_best_face_changed_mask.shape[0] == canonical_xyz.shape[0]
        ):
            self.latest_face_anchor_keep_prior_best_face_changed_mask = (
                risky_geom_fallback_keep_prior_best_face_changed_mask.detach().clone()
            )
        if (
            torch.is_tensor(risky_geom_fallback_keep_prior_best_joint_changed_mask)
            and risky_geom_fallback_keep_prior_best_joint_changed_mask.shape[0] == canonical_xyz.shape[0]
        ):
            self.latest_face_anchor_keep_prior_best_joint_changed_mask = (
                risky_geom_fallback_keep_prior_best_joint_changed_mask.detach().clone()
            )
        if (
            torch.is_tensor(risky_geom_fallback_keep_prior_best_anchor_shift)
            and risky_geom_fallback_keep_prior_best_anchor_shift.shape[0] == canonical_xyz.shape[0]
        ):
            self.latest_face_anchor_keep_prior_best_anchor_shift = (
                risky_geom_fallback_keep_prior_best_anchor_shift.detach().clone()
            )

        if bool(self.cfg.get('semantic_risky_child_debug_verbose', False)) and risky_child_mask is not None and bool(risky_child_mask.any().item()):
            risky_count = int(risky_child_mask.sum().item())
            guarded_count = int(risky_switch_guard_mask.sum().item()) if torch.is_tensor(risky_switch_guard_mask) else risky_count
            escaped_count = int(risky_child_escape_mask.sum().item()) if torch.is_tensor(risky_child_escape_mask) else 0
            escaped_refresh_count = int(risky_child_escape_refresh_mask.sum().item()) if torch.is_tensor(risky_child_escape_refresh_mask) else 0
            escaped_confidence_count = int(risky_child_escape_confidence_mask.sum().item()) if torch.is_tensor(risky_child_escape_confidence_mask) else 0
            escaped_semantic_count = int(risky_child_escape_semantic_mask.sum().item()) if torch.is_tensor(risky_child_escape_semantic_mask) else 0
            escaped_surface_count = int(risky_child_escape_surface_mask.sum().item()) if torch.is_tensor(risky_child_escape_surface_mask) else 0
            escaped_weight_gap_count = int(risky_child_escape_weight_gap_mask.sum().item()) if torch.is_tensor(risky_child_escape_weight_gap_mask) else 0
            required_joint_match_count = 0
            if risky_child_mask is not None and risky_required_joint is not None:
                required_joint_match_count = int((best_dominant_joint[risky_child_mask] == risky_required_joint[risky_child_mask]).sum().item())
            source_valid_count = int(risky_source_joint_valid_mask.sum().item()) if torch.is_tensor(risky_source_joint_valid_mask) else 0
            source_consensus_count = int(risky_source_joint_consensus_mask.sum().item()) if torch.is_tensor(risky_source_joint_consensus_mask) else 0
            source_disagreement_count = int(risky_source_joint_disagreement_mask.sum().item()) if torch.is_tensor(risky_source_joint_disagreement_mask) else 0
            source_mismatch_count = int(risky_source_prior_mismatch_mask.sum().item()) if torch.is_tensor(risky_source_prior_mismatch_mask) else 0
            risky_query_prior_reference_count = int((risky_query_prior_reference_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_query_prior_reference_mask) else 0
            source_missing_geom_count = int(risky_source_missing_force_geom_mask.sum().item()) if torch.is_tensor(risky_source_missing_force_geom_mask) else 0
            source_mismatch_geom_count = int(risky_source_mismatch_force_geom_mask.sum().item()) if torch.is_tensor(risky_source_mismatch_force_geom_mask) else 0
            risky_switch_count = int((switch_mask & risky_child_mask).sum().item())
            risky_same_seed_count = int((same_group_switch_seed & risky_child_mask).sum().item())
            risky_same_score_ok_count = int((risky_same_group_score_ok_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_same_group_score_ok_mask) else 0
            risky_same_surface_ok_count = int((risky_same_group_surface_ok_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_same_group_surface_ok_mask) else 0
            risky_cross_seed_count = int((cross_group_switch_seed & risky_child_mask).sum().item())
            risky_cross_score_ok_count = int((risky_cross_group_score_ok_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_cross_group_score_ok_mask) else 0
            risky_cross_skinning_ok_count = int((risky_cross_group_skinning_ok_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_cross_group_skinning_ok_mask) else 0
            risky_cross_surface_ok_count = int((risky_cross_group_surface_ok_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_cross_group_surface_ok_mask) else 0
            risky_cross_prior_reference_count = int((risky_cross_group_prior_reference_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_cross_group_prior_reference_mask) else 0
            risky_cross_score_failopen_count = int((risky_cross_group_score_failopen_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_cross_group_score_failopen_mask) else 0
            risky_base_switch_count = int((base_switch_mask & risky_child_mask).sum().item())
            risky_same_group_count = int((same_group & risky_child_mask).sum().item())
            risky_same_side_count = int((same_side & risky_child_mask).sum().item())
            risky_same_hop_count = int((same_group_hop_ok & risky_child_mask).sum().item())
            risky_seed_override_count = int((risky_seed_override_candidate_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_seed_override_candidate_mask) else 0
            risky_required_joint_candidate_available_count = int((risky_required_joint_candidate_available_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_required_joint_candidate_available_mask) else 0
            risky_required_joint_failopen_count = int((risky_required_joint_failopen_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_required_joint_failopen_mask) else 0
            risky_same_face_seed_count = int((risky_same_face_seed_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_same_face_seed_mask) else 0
            risky_same_face_joint_change_count = int((risky_same_face_joint_change_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_same_face_joint_change_mask) else 0
            risky_same_face_weight_shift_count = int((risky_same_face_weight_shift_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_same_face_weight_shift_mask) else 0
            risky_same_face_anchor_shift_count = int((risky_same_face_anchor_shift_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_same_face_anchor_shift_mask) else 0
            risky_same_face_bary_shift_count = int((risky_same_face_bary_shift_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_same_face_bary_shift_mask) else 0
            geom_required_joint_match_count = 0
            prior_required_joint_match_count = 0
            geom_vs_prior_face_change_count = 0
            geom_vs_prior_joint_change_count = 0
            best_vs_prior_face_change_count = 0
            best_vs_prior_joint_change_count = 0
            final_required_joint_match_count = 0
            final_vs_prior_joint_change_count = 0
            risky_geom_fallback_count = int((risky_geom_fallback_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_geom_fallback_mask) else 0
            risky_geom_fallback_joint_match_count = int((risky_geom_fallback_joint_match_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_geom_fallback_joint_match_mask) else 0
            risky_geom_fallback_joint_mismatch_count = int((risky_geom_fallback_joint_mismatch_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_geom_fallback_joint_mismatch_mask) else 0
            risky_geom_fallback_keep_prior_count = int((risky_geom_fallback_keep_prior_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_geom_fallback_keep_prior_mask) else 0
            risky_geom_fallback_missing_prior_count = int((risky_geom_fallback_missing_prior_mask & risky_child_mask).sum().item()) if torch.is_tensor(risky_geom_fallback_missing_prior_mask) else 0
            final_dominant_joint = torch.argmax(anchor_weights, dim=-1)
            if risky_required_joint is not None:
                geom_required_joint_match_count = int((geom_dominant_joint[risky_child_mask] == risky_required_joint[risky_child_mask]).sum().item())
                final_required_joint_match_count = int((final_dominant_joint[risky_child_mask] == risky_required_joint[risky_child_mask]).sum().item())
                if prior_dominant_joint is not None and prior_dominant_joint.shape[0] == canonical_xyz.shape[0]:
                    prior_required_joint_match_count = int((prior_dominant_joint[risky_child_mask] == risky_required_joint[risky_child_mask]).sum().item())
            if prior_face_ids is not None and prior_face_ids.shape[0] == canonical_xyz.shape[0]:
                geom_vs_prior_face_change_count = int((geom_face_ids[risky_child_mask] != prior_face_ids[risky_child_mask]).sum().item())
                best_vs_prior_face_change_count = int((best_face_ids[risky_child_mask] != prior_face_ids[risky_child_mask]).sum().item())
            if prior_dominant_joint is not None and prior_dominant_joint.shape[0] == canonical_xyz.shape[0]:
                geom_vs_prior_joint_change_count = int((geom_dominant_joint[risky_child_mask] != prior_dominant_joint[risky_child_mask]).sum().item())
                best_vs_prior_joint_change_count = int((best_dominant_joint[risky_child_mask] != prior_dominant_joint[risky_child_mask]).sum().item())
                final_vs_prior_joint_change_count = int((final_dominant_joint[risky_child_mask] != prior_dominant_joint[risky_child_mask]).sum().item())
            print(
                '[ExplicitBinding] risky-child source gate '
                f'iter={iteration} risky={risky_count} guarded={guarded_count} '
                f'escaped={escaped_count} escaped_by_refresh={escaped_refresh_count} '
                f'escaped_by_conf={escaped_confidence_count} '
                f'escaped_by_sem={escaped_semantic_count} '
                f'escaped_by_surf={escaped_surface_count} '
                f'escaped_by_gap={escaped_weight_gap_count} '
                f'required_joint_match={required_joint_match_count} '
                f'source_valid={source_valid_count} source_consensus={source_consensus_count} '
                f'source_disagreement={source_disagreement_count} '
                f'prior_source_mismatch={source_mismatch_count} '
                f'query_prior_ref={risky_query_prior_reference_count} '
                f'geom_required_joint_match={geom_required_joint_match_count} '
                f'prior_required_joint_match={prior_required_joint_match_count} '
                f'final_required_joint_match={final_required_joint_match_count} '
                f'geom_vs_prior_face_change={geom_vs_prior_face_change_count} '
                f'geom_vs_prior_joint_change={geom_vs_prior_joint_change_count} '
                f'best_vs_prior_face_change={best_vs_prior_face_change_count} '
                f'best_vs_prior_joint_change={best_vs_prior_joint_change_count} '
                f'final_vs_prior_joint_change={final_vs_prior_joint_change_count} '
                f'base_switch={risky_base_switch_count} same_group={risky_same_group_count} '
                f'same_side={risky_same_side_count} same_hop_ok={risky_same_hop_count} '
                f'same_seed={risky_same_seed_count} same_score_ok={risky_same_score_ok_count} '
                f'same_surface_ok={risky_same_surface_ok_count} '
                f'same_face_seed={risky_same_face_seed_count} '
                f'same_face_joint_change={risky_same_face_joint_change_count} '
                f'same_face_weight_shift={risky_same_face_weight_shift_count} '
                f'same_face_anchor_shift={risky_same_face_anchor_shift_count} '
                f'same_face_bary_shift={risky_same_face_bary_shift_count} '
                f'cross_seed={risky_cross_seed_count} cross_score_ok={risky_cross_score_ok_count} '
                f'cross_skin_ok={risky_cross_skinning_ok_count} cross_surface_ok={risky_cross_surface_ok_count} '
                f'cross_prior_ref={risky_cross_prior_reference_count} '
                f'cross_score_failopen={risky_cross_score_failopen_count} '
                f'seed_override_candidates={risky_seed_override_count} '
                f'required_joint_candidates={risky_required_joint_candidate_available_count} '
                f'required_joint_failopen={risky_required_joint_failopen_count} '
                f'geom_fallback={risky_geom_fallback_count} '
                f'geom_fallback_joint_match={risky_geom_fallback_joint_match_count} '
                f'geom_fallback_joint_mismatch={risky_geom_fallback_joint_mismatch_count} '
                f'geom_fallback_keep_prior={risky_geom_fallback_keep_prior_count} '
                f'geom_fallback_missing_prior={risky_geom_fallback_missing_prior_count} '
                f'force_geom_missing_source={source_missing_geom_count} '
                f'force_geom_source_mismatch={source_mismatch_geom_count} '
                f'switched={risky_switch_count}'
            )

        return face_ids, tri_vids, bary, anchor_xyz, anchor_weights, face_normal, surface_distance, semantic_score, semantic_distance

    def _binding_from_state(self, canonical_xyz, binding_state):
        anchor_xyz = binding_state['anchor_xyz']
        anchor_weights = binding_state['anchor_weights']
        anchor_normal = binding_state['anchor_normal']
        dominant_joint = binding_state['dominant_joint']
        bound_xyz = canonical_xyz.detach()
        local_offset = bound_xyz - anchor_xyz
        normal_offset_mag = torch.sum(local_offset * anchor_normal, dim=-1, keepdim=True)
        normal_offset = normal_offset_mag * anchor_normal
        tangent_offset = local_offset - normal_offset
        bone_distance = self._compute_bone_distance(anchor_xyz, dominant_joint)
        surface_distance = torch.abs(normal_offset_mag.squeeze(-1))
        confidence = binding_state.get('anchor_confidence', anchor_weights.max(dim=-1).values)
        semantic_distance = binding_state.get('semantic_distance', torch.zeros_like(surface_distance))
        layer_weights, region_probs = self._compute_layer_weights(bone_distance, surface_distance, confidence, semantic_distance)

        binding = dict(binding_state)
        binding.update({
            'bound_xyz': bound_xyz,
            'local_offset': local_offset,
            'normal_offset': normal_offset,
            'tangent_offset': tangent_offset,
            'bone_distance': bone_distance,
            'surface_distance': surface_distance,
            'layer_weights': layer_weights,
            'region_probs': region_probs,
        })
        return binding

    def _build_binding_state(self, canonical_xyz, face_ids, tri_vids, barycentric, anchor_xyz, anchor_weights, anchor_normal, semantic_score, semantic_distance):
        dominant_joint = torch.argmax(anchor_weights, dim=-1)
        region_labels = torch.argmax(self._compute_region_probs(anchor_weights.max(dim=-1).values, torch.norm(anchor_xyz - canonical_xyz.detach(), dim=-1), semantic_distance), dim=-1)
        binding_state = {
            'anchor_ids': tri_vids[:, 0],
            'anchor_face_ids': face_ids,
            'anchor_vertex_ids': tri_vids,
            'anchor_barycentric': barycentric,
            'anchor_xyz': anchor_xyz,
            'anchor_weights': anchor_weights,
            'anchor_normal': anchor_normal,
            'dominant_joint': dominant_joint,
            'anchor_confidence': anchor_weights.max(dim=-1).values,
            'semantic_score': semantic_score,
            'semantic_distance': semantic_distance,
            'region_labels': region_labels,
            'anchor_refresh_mask': torch.zeros_like(dominant_joint, dtype=torch.bool),
        }
        return self._binding_from_state(canonical_xyz, binding_state)

    def _refresh_binding(self, canonical_xyz, iteration, canonical_owner=None, prior_state=None):
        with torch.no_grad():
            self.latest_subset_refresh_info = None
            self.latest_face_anchor_switch_mask = None
            self.latest_face_anchor_keep_prior_mask = None
            self.latest_face_anchor_best_joint_changed_mask = None
            self.latest_face_anchor_keep_prior_best_face_changed_mask = None
            self.latest_face_anchor_keep_prior_best_joint_changed_mask = None
            self.latest_face_anchor_keep_prior_best_anchor_shift = None
            if prior_state is None and canonical_owner is not None and hasattr(canonical_owner, 'has_binding_state') and canonical_owner.has_binding_state():
                owner_state = canonical_owner.get_binding_state()
                if self._state_matches(owner_state, canonical_xyz):
                    prior_state = owner_state
            face_ids, tri_vids, barycentric, anchor_xyz, anchor_weights, anchor_normal, _, semantic_score, semantic_distance = self._build_face_anchor(
                canonical_xyz,
                iteration,
                prior_state=prior_state,
            )
            binding = self._build_binding_state(
                canonical_xyz,
                face_ids,
                tri_vids,
                barycentric,
                anchor_xyz,
                anchor_weights,
                anchor_normal,
                semantic_score,
                semantic_distance,
            )
            if canonical_owner is not None and hasattr(canonical_owner, 'set_binding_state'):
                canonical_owner.set_binding_state(binding)
            self.binding_cache = binding
            self.temporal_cache.clear()

    def _refresh_binding_subset(self, canonical_xyz, iteration, refresh_mask, canonical_owner=None, base_state=None):
        self.latest_subset_refresh_info = None
        self.latest_face_anchor_switch_mask = None
        self.latest_face_anchor_keep_prior_mask = None
        self.latest_face_anchor_best_joint_changed_mask = None
        self.latest_face_anchor_keep_prior_best_face_changed_mask = None
        self.latest_face_anchor_keep_prior_best_joint_changed_mask = None
        self.latest_face_anchor_keep_prior_best_anchor_shift = None
        debug_verbose = bool(self.cfg.get('partial_refresh_debug_verbose', False))
        if refresh_mask is None:
            if debug_verbose:
                print(
                    '[ExplicitBinding] subset refresh skipped '
                    f'iter={iteration} reason=no_refresh_mask '
                    f'base=({self._debug_binding_state_summary(base_state, canonical_xyz)})'
                )
            return None
        refresh_mask = refresh_mask.to(device=canonical_xyz.device, dtype=torch.bool)
        refresh_count = int(refresh_mask.sum().item())
        if refresh_mask.shape[0] != canonical_xyz.shape[0]:
            if debug_verbose:
                print(
                    '[ExplicitBinding] subset refresh skipped '
                    f'iter={iteration} reason=shape_mismatch refresh_shape={refresh_mask.shape[0]} '
                    f'canonical_shape={canonical_xyz.shape[0]} '
                    f'base=({self._debug_binding_state_summary(base_state, canonical_xyz)})'
                )
            return None
        if refresh_count <= 0:
            if debug_verbose:
                print(
                    '[ExplicitBinding] subset refresh skipped '
                    f'iter={iteration} reason=empty_mask '
                    f'base=({self._debug_binding_state_summary(base_state, canonical_xyz)})'
                )
            return None
        if not self._state_matches(base_state, canonical_xyz):
            if debug_verbose:
                print(
                    '[ExplicitBinding] subset refresh fallback-to-full '
                    f'iter={iteration} reason=base_state_mismatch refresh={refresh_count} '
                    f'base=({self._debug_binding_state_summary(base_state, canonical_xyz)})'
                )
            self._refresh_binding(canonical_xyz, iteration, canonical_owner=canonical_owner, prior_state=base_state)
            return self.binding_cache
        if bool(refresh_mask.all().item()):
            if debug_verbose:
                print(
                    '[ExplicitBinding] subset refresh fallback-to-full '
                    f'iter={iteration} reason=full_mask refresh={refresh_count} '
                    f'base=({self._debug_binding_state_summary(base_state, canonical_xyz)})'
                )
            self._refresh_binding(canonical_xyz, iteration, canonical_owner=canonical_owner, prior_state=base_state)
            return self.binding_cache

        with torch.no_grad():
            subset_xyz = canonical_xyz[refresh_mask]
            subset_prior_state = self._slice_binding_state(base_state, refresh_mask)
            risky_child_mask = None
            source_parent_index = None
            source_root_parent_index = None
            source_parent_joint = None
            source_root_parent_joint = None
            source_parent_joint_origin = 'none'
            source_root_parent_joint_origin = 'none'
            source_root_lineage_id = None
            densify_birth_iter = None
            risky_mask_full = base_state.get('densify_risky_child_mask', None)
            if torch.is_tensor(risky_mask_full) and risky_mask_full.shape[0] == canonical_xyz.shape[0]:
                risky_child_mask = risky_mask_full[refresh_mask].to(device=canonical_xyz.device, dtype=torch.bool)
            source_parent_full = base_state.get('densify_parent_index', None)
            if torch.is_tensor(source_parent_full) and source_parent_full.shape[0] == canonical_xyz.shape[0]:
                source_parent_index = source_parent_full[refresh_mask].to(device=canonical_xyz.device, dtype=torch.long)
            source_root_parent_full = base_state.get('densify_root_parent_index', None)
            if torch.is_tensor(source_root_parent_full) and source_root_parent_full.shape[0] == canonical_xyz.shape[0]:
                source_root_parent_index = source_root_parent_full[refresh_mask].to(device=canonical_xyz.device, dtype=torch.long)
            source_parent_joint_full = base_state.get('source_parent_joint', None)
            if torch.is_tensor(source_parent_joint_full) and source_parent_joint_full.shape[0] == canonical_xyz.shape[0]:
                source_parent_joint = source_parent_joint_full[refresh_mask].to(device=canonical_xyz.device, dtype=torch.long)
                source_parent_joint_origin = 'direct'
            source_root_parent_joint_full = base_state.get('source_root_parent_joint', None)
            if (
                torch.is_tensor(source_root_parent_joint_full)
                and source_root_parent_joint_full.shape[0] == canonical_xyz.shape[0]
            ):
                source_root_parent_joint = source_root_parent_joint_full[refresh_mask].to(
                    device=canonical_xyz.device,
                    dtype=torch.long,
                )
                source_root_parent_joint_origin = 'direct'
            dominant_joint_full = base_state.get('dominant_joint', None)
            if torch.is_tensor(dominant_joint_full) and dominant_joint_full.shape[0] == canonical_xyz.shape[0]:
                dominant_joint_full = dominant_joint_full.to(device=canonical_xyz.device, dtype=torch.long)
                if (
                    source_parent_joint is None
                    and torch.is_tensor(source_parent_index)
                    and source_parent_index.shape[0] == refresh_count
                ):
                    source_parent_joint = torch.full(
                        (refresh_count,),
                        -1,
                        dtype=torch.long,
                        device=canonical_xyz.device,
                    )
                    valid_parent = (source_parent_index >= 0) & (source_parent_index < dominant_joint_full.shape[0])
                    if bool(valid_parent.any().item()):
                        source_parent_joint[valid_parent] = dominant_joint_full[source_parent_index[valid_parent].long()]
                    source_parent_joint_origin = 'reconstructed'
                if (
                    source_root_parent_joint is None
                    and torch.is_tensor(source_root_parent_index)
                    and source_root_parent_index.shape[0] == refresh_count
                ):
                    source_root_parent_joint = torch.full(
                        (refresh_count,),
                        -1,
                        dtype=torch.long,
                        device=canonical_xyz.device,
                    )
                    valid_root_parent = (source_root_parent_index >= 0) & (source_root_parent_index < dominant_joint_full.shape[0])
                    if bool(valid_root_parent.any().item()):
                        source_root_parent_joint[valid_root_parent] = dominant_joint_full[source_root_parent_index[valid_root_parent].long()]
                    source_root_parent_joint_origin = 'reconstructed'
            source_root_lineage_full = base_state.get('densify_root_lineage_id', None)
            if torch.is_tensor(source_root_lineage_full) and source_root_lineage_full.shape[0] == canonical_xyz.shape[0]:
                source_root_lineage_id = source_root_lineage_full[refresh_mask].to(device=canonical_xyz.device, dtype=torch.long)
            densify_birth_full = base_state.get('densify_birth_iter', None)
            if torch.is_tensor(densify_birth_full) and densify_birth_full.shape[0] == canonical_xyz.shape[0]:
                densify_birth_iter = densify_birth_full[refresh_mask].to(device=canonical_xyz.device, dtype=torch.long)
            if torch.is_tensor(source_parent_joint) and source_parent_joint.shape[0] == refresh_count:
                subset_prior_state['source_parent_joint'] = source_parent_joint.clone()
            if torch.is_tensor(source_root_parent_joint) and source_root_parent_joint.shape[0] == refresh_count:
                subset_prior_state['source_root_parent_joint'] = source_root_parent_joint.clone()
            if debug_verbose:
                source_parent_valid_count = int((source_parent_joint >= 0).sum().item()) if torch.is_tensor(source_parent_joint) else 0
                source_root_parent_valid_count = int((source_root_parent_joint >= 0).sum().item()) if torch.is_tensor(source_root_parent_joint) else 0
                print(
                    '[ExplicitBinding] subset refresh source joints '
                    f'iter={iteration} refresh={refresh_count} '
                    f'parent_origin={source_parent_joint_origin} parent_valid={source_parent_valid_count} '
                    f'root_origin={source_root_parent_joint_origin} root_valid={source_root_parent_valid_count}'
                )
            try:
                face_ids, tri_vids, barycentric, anchor_xyz, anchor_weights, anchor_normal, _, semantic_score, semantic_distance = self._build_face_anchor(
                    subset_xyz,
                    iteration,
                    prior_state=subset_prior_state,
                )
            except Exception as exc:
                print(
                    '[ExplicitBinding] subset refresh fallback-to-full '
                    f'iter={iteration} reason=anchor_build_error refresh={refresh_count} '
                    f'subset_points={subset_xyz.shape[0]} error={type(exc).__name__}: {exc}'
                )
                self._refresh_binding(
                    canonical_xyz,
                    iteration,
                    canonical_owner=canonical_owner,
                    prior_state=base_state,
                )
                return self.binding_cache
            subset_binding = self._build_binding_state(
                subset_xyz,
                face_ids,
                tri_vids,
                barycentric,
                anchor_xyz,
                anchor_weights,
                anchor_normal,
                semantic_score,
                semantic_distance,
            )
            anchor_shift = None
            anchor_face_changed = None
            dominant_joint_changed = None
            switched_child_mask = None
            kept_prior_child_mask = None
            best_joint_changed_mask = None
            kept_prior_best_face_changed_mask = None
            kept_prior_best_joint_changed_mask = None
            kept_prior_best_anchor_shift = None
            if torch.is_tensor(subset_prior_state.get('anchor_xyz', None)) and subset_prior_state['anchor_xyz'].shape == subset_binding['anchor_xyz'].shape:
                anchor_shift = torch.norm(
                    subset_binding['anchor_xyz'] - subset_prior_state['anchor_xyz'],
                    dim=-1,
                )
            if torch.is_tensor(subset_prior_state.get('anchor_face_ids', None)) and subset_prior_state['anchor_face_ids'].shape == subset_binding['anchor_face_ids'].shape:
                anchor_face_changed = subset_binding['anchor_face_ids'] != subset_prior_state['anchor_face_ids']
            if torch.is_tensor(subset_prior_state.get('dominant_joint', None)) and subset_prior_state['dominant_joint'].shape == subset_binding['dominant_joint'].shape:
                dominant_joint_changed = subset_binding['dominant_joint'] != subset_prior_state['dominant_joint']
            if (
                torch.is_tensor(self.latest_face_anchor_switch_mask)
                and self.latest_face_anchor_switch_mask.shape[0] == refresh_count
            ):
                switched_child_mask = self.latest_face_anchor_switch_mask.to(
                    device=canonical_xyz.device,
                    dtype=torch.bool,
                )
            if (
                torch.is_tensor(self.latest_face_anchor_keep_prior_mask)
                and self.latest_face_anchor_keep_prior_mask.shape[0] == refresh_count
            ):
                kept_prior_child_mask = self.latest_face_anchor_keep_prior_mask.to(
                    device=canonical_xyz.device,
                    dtype=torch.bool,
                )
            if (
                torch.is_tensor(self.latest_face_anchor_best_joint_changed_mask)
                and self.latest_face_anchor_best_joint_changed_mask.shape[0] == refresh_count
            ):
                best_joint_changed_mask = self.latest_face_anchor_best_joint_changed_mask.to(
                    device=canonical_xyz.device,
                    dtype=torch.bool,
                )
            if (
                torch.is_tensor(self.latest_face_anchor_keep_prior_best_face_changed_mask)
                and self.latest_face_anchor_keep_prior_best_face_changed_mask.shape[0] == refresh_count
            ):
                kept_prior_best_face_changed_mask = self.latest_face_anchor_keep_prior_best_face_changed_mask.to(
                    device=canonical_xyz.device,
                    dtype=torch.bool,
                )
            if (
                torch.is_tensor(self.latest_face_anchor_keep_prior_best_joint_changed_mask)
                and self.latest_face_anchor_keep_prior_best_joint_changed_mask.shape[0] == refresh_count
            ):
                kept_prior_best_joint_changed_mask = self.latest_face_anchor_keep_prior_best_joint_changed_mask.to(
                    device=canonical_xyz.device,
                    dtype=torch.bool,
                )
            if (
                torch.is_tensor(self.latest_face_anchor_keep_prior_best_anchor_shift)
                and self.latest_face_anchor_keep_prior_best_anchor_shift.shape[0] == refresh_count
            ):
                kept_prior_best_anchor_shift = self.latest_face_anchor_keep_prior_best_anchor_shift.to(
                    device=canonical_xyz.device,
                    dtype=torch.float32,
                )
            merged_state = self._merge_binding_state(base_state, subset_binding, refresh_mask)
            merged_refresh_mask = merged_state.get('anchor_refresh_mask', None)
            if torch.is_tensor(merged_refresh_mask) and merged_refresh_mask.shape[0] == canonical_xyz.shape[0]:
                merged_refresh_mask = merged_refresh_mask.to(device=canonical_xyz.device, dtype=torch.bool)
                merged_refresh_mask[refresh_mask] = False
                merged_state['anchor_refresh_mask'] = merged_refresh_mask
            merged_risky_child_mask = merged_state.get('densify_risky_child_mask', None)
            if torch.is_tensor(merged_risky_child_mask) and merged_risky_child_mask.shape[0] == canonical_xyz.shape[0]:
                merged_risky_child_mask = merged_risky_child_mask.to(device=canonical_xyz.device, dtype=torch.bool)
                merged_risky_child_mask[refresh_mask] = False
                merged_state['densify_risky_child_mask'] = merged_risky_child_mask
            binding = self._binding_from_state(canonical_xyz, merged_state)
            if canonical_owner is not None and hasattr(canonical_owner, 'set_binding_state'):
                canonical_owner.set_binding_state(binding)
            self.binding_cache = binding
            self.temporal_cache.clear()
            self.latest_subset_refresh_info = {
                'refresh_mask': refresh_mask.detach().clone(),
                'anchor_shift': anchor_shift.detach().clone() if torch.is_tensor(anchor_shift) else None,
                'anchor_face_changed': anchor_face_changed.detach().clone() if torch.is_tensor(anchor_face_changed) else None,
                'dominant_joint_changed': dominant_joint_changed.detach().clone() if torch.is_tensor(dominant_joint_changed) else None,
                'best_joint_changed_mask': best_joint_changed_mask.detach().clone() if torch.is_tensor(best_joint_changed_mask) else None,
                'switched_child_mask': switched_child_mask.detach().clone() if torch.is_tensor(switched_child_mask) else None,
                'kept_prior_child_mask': kept_prior_child_mask.detach().clone() if torch.is_tensor(kept_prior_child_mask) else None,
                'kept_prior_best_face_changed_mask': kept_prior_best_face_changed_mask.detach().clone() if torch.is_tensor(kept_prior_best_face_changed_mask) else None,
                'kept_prior_best_joint_changed_mask': kept_prior_best_joint_changed_mask.detach().clone() if torch.is_tensor(kept_prior_best_joint_changed_mask) else None,
                'kept_prior_best_anchor_shift': kept_prior_best_anchor_shift.detach().clone() if torch.is_tensor(kept_prior_best_anchor_shift) else None,
                'risky_child_mask': risky_child_mask.detach().clone() if torch.is_tensor(risky_child_mask) else None,
                'source_parent_index': source_parent_index.detach().clone() if torch.is_tensor(source_parent_index) else None,
                'source_root_parent_index': source_root_parent_index.detach().clone() if torch.is_tensor(source_root_parent_index) else None,
                'source_parent_joint': source_parent_joint.detach().clone() if torch.is_tensor(source_parent_joint) else None,
                'source_root_parent_joint': source_root_parent_joint.detach().clone() if torch.is_tensor(source_root_parent_joint) else None,
                'source_root_lineage_id': source_root_lineage_id.detach().clone() if torch.is_tensor(source_root_lineage_id) else None,
                'densify_birth_iter': densify_birth_iter.detach().clone() if torch.is_tensor(densify_birth_iter) else None,
            }
            if bool(self.cfg.get('partial_refresh_verbose', False)):
                refresh_count = int(refresh_mask.sum().item())
                risky_count = int(risky_child_mask.sum().item()) if torch.is_tensor(risky_child_mask) else 0
                face_changed_count = int(anchor_face_changed.sum().item()) if torch.is_tensor(anchor_face_changed) else 0
                joint_changed_count = int(dominant_joint_changed.sum().item()) if torch.is_tensor(dominant_joint_changed) else 0
                anchor_shift_mean = float(anchor_shift.mean().item()) if torch.is_tensor(anchor_shift) and anchor_shift.numel() > 0 else 0.0
                anchor_shift_max = float(anchor_shift.max().item()) if torch.is_tensor(anchor_shift) and anchor_shift.numel() > 0 else 0.0
                lineage_count = int(source_root_lineage_id.unique().numel()) if torch.is_tensor(source_root_lineage_id) and source_root_lineage_id.numel() > 0 else 0
                print(
                    '[ExplicitBinding] partial refresh summary '
                    f'iter={iteration} refresh={refresh_count} risky={risky_count} '
                    f'face_changed={face_changed_count} joint_changed={joint_changed_count} '
                    f'shift_mean={anchor_shift_mean:.5f} '
                    f'shift_max={anchor_shift_max:.5f} root_lineages={lineage_count}'
                )
        return self.binding_cache

    def _binding_state_needs_refresh(self, binding_state, canonical_xyz):
        if not self._state_matches(binding_state, canonical_xyz):
            return False
        refresh_mask = binding_state.get('anchor_refresh_mask', None)
        return (
            torch.is_tensor(refresh_mask)
            and refresh_mask.shape[0] == canonical_xyz.shape[0]
            and bool(refresh_mask.any().item())
        )

    def _get_binding(self, canonical_xyz, iteration, canonical_owner=None):
        owner_state = {}
        if canonical_owner is not None and hasattr(canonical_owner, 'has_binding_state') and canonical_owner.has_binding_state():
            owner_state = canonical_owner.get_binding_state()

        owner_state_matches = self._state_matches(owner_state, canonical_xyz)
        owner_state_needs_refresh = self._binding_state_needs_refresh(owner_state, canonical_xyz)
        cache_matches = self._state_matches(self.binding_cache, canonical_xyz)
        cache_needs_refresh = self._binding_state_needs_refresh(self.binding_cache, canonical_xyz)
        partial_refresh_enable = bool(self.cfg.get('partial_refresh_enable', True))

        if not self._force_rebind(iteration):
            if owner_state_matches and not owner_state_needs_refresh:
                binding = self._binding_from_state(canonical_xyz, owner_state)
                self.binding_cache = binding
                return binding
            if owner_state_matches and owner_state_needs_refresh and partial_refresh_enable:
                refreshed = self._refresh_binding_subset(
                    canonical_xyz,
                    iteration,
                    owner_state.get('anchor_refresh_mask', None),
                    canonical_owner=canonical_owner,
                    base_state=owner_state,
                )
                if refreshed is not None:
                    return refreshed
            if cache_matches and not cache_needs_refresh:
                self.binding_cache = self._binding_from_state(canonical_xyz, self.binding_cache)
                return self.binding_cache
            if cache_matches and cache_needs_refresh and partial_refresh_enable:
                refreshed = self._refresh_binding_subset(
                    canonical_xyz,
                    iteration,
                    self.binding_cache.get('anchor_refresh_mask', None),
                    canonical_owner=canonical_owner,
                    base_state=self.binding_cache,
                )
                if refreshed is not None:
                    return refreshed

        prior_state = owner_state if owner_state_matches else (self.binding_cache if cache_matches else None)
        self._refresh_binding(canonical_xyz, iteration, canonical_owner=canonical_owner, prior_state=prior_state)
        return self.binding_cache

    def refresh_pending_binding(self, canonical_xyz, iteration, canonical_owner=None):
        owner_state = {}
        if canonical_owner is not None and hasattr(canonical_owner, 'has_binding_state') and canonical_owner.has_binding_state():
            owner_state = canonical_owner.get_binding_state()

        debug_verbose = bool(self.cfg.get('partial_refresh_debug_verbose', False))
        owner_matches = self._state_matches(owner_state, canonical_xyz)
        owner_needs = self._binding_state_needs_refresh(owner_state, canonical_xyz)
        cache_matches = self._state_matches(self.binding_cache, canonical_xyz)
        cache_needs = self._binding_state_needs_refresh(self.binding_cache, canonical_xyz)
        if debug_verbose:
            print(
                '[ExplicitBinding] refresh_pending_binding '
                f'iter={iteration} owner_match={int(owner_matches)} owner_need={int(owner_needs)} '
                f'cache_match={int(cache_matches)} cache_need={int(cache_needs)} '
                f'owner=({self._debug_binding_state_summary(owner_state, canonical_xyz)}) '
                f'cache=({self._debug_binding_state_summary(self.binding_cache, canonical_xyz)})'
            )

        if owner_matches and owner_needs:
            return self._refresh_binding_subset(
                canonical_xyz,
                iteration,
                owner_state.get('anchor_refresh_mask', None),
                canonical_owner=canonical_owner,
                base_state=owner_state,
            )

        if cache_matches and cache_needs:
            return self._refresh_binding_subset(
                canonical_xyz,
                iteration,
                self.binding_cache.get('anchor_refresh_mask', None),
                canonical_owner=canonical_owner,
                base_state=self.binding_cache,
            )
        if debug_verbose:
            print(
                '[ExplicitBinding] refresh_pending_binding no-op '
                f'iter={iteration} owner_match={int(owner_matches)} owner_need={int(owner_needs)} '
                f'cache_match={int(cache_matches)} cache_need={int(cache_needs)}'
            )
        return None

    def consume_latest_subset_refresh_info(self):
        info = self.latest_subset_refresh_info
        self.latest_subset_refresh_info = None
        return info

    def _update_temporal_cache(self, frame_id, local_motion):
        local_motion = local_motion.detach().cpu()
        point_count = int(local_motion.shape[0])
        cached_point_count = getattr(self, '_temporal_point_count', None)
        if cached_point_count is not None and cached_point_count != point_count:
            self.temporal_cache.clear()
        self._temporal_point_count = point_count

        if frame_id in self.temporal_cache:
            cached = self.temporal_cache.pop(frame_id)
            if cached.shape == local_motion.shape:
                local_motion = self.temporal_momentum * cached + (1. - self.temporal_momentum) * local_motion
        self.temporal_cache[frame_id] = local_motion
        while len(self.temporal_cache) > self.temporal_cache_size:
            self.temporal_cache.popitem(last=False)

    def _temporal_loss(self, frame_id, local_motion, layer_weights):
        if torch.is_tensor(frame_id):
            frame_id = int(frame_id.item())
        else:
            frame_id = int(frame_id)

        losses = []
        diff_maps = []
        temporal_weight = layer_weights[:, 1] + 0.5 * layer_weights[:, 2]
        for neighbor in (frame_id - 1, frame_id + 1):
            if neighbor not in self.temporal_cache:
                continue
            neighbor_motion = self.temporal_cache[neighbor]
            if neighbor_motion.shape != local_motion.shape:
                continue
            neighbor_motion = neighbor_motion.to(local_motion.device)
            diff = torch.norm(local_motion - neighbor_motion, dim=-1)
            diff_maps.append(diff)
            losses.append(weighted_reduce(diff, temporal_weight))

        if diff_maps:
            self.latest_temporal_slip = torch.stack(diff_maps, dim=0).mean(dim=0).detach()
        else:
            self.latest_temporal_slip = torch.zeros(local_motion.shape[0], device=local_motion.device, dtype=local_motion.dtype)

        self._update_temporal_cache(frame_id, local_motion)
        if not losses:
            return local_motion.new_tensor(0.)
        return sum(losses) / len(losses)

    def get_loss_reg(self):
        return self.loss_reg

    def _compute_non_rigid_carry(self, canonical_xyz, xyz, anchor_normal, R_dom, R_lbs):
        if canonical_xyz.shape != xyz.shape:
            zero = torch.zeros_like(xyz)
            zero_scalar = torch.zeros((xyz.shape[0],), dtype=xyz.dtype, device=xyz.device)
            return zero, zero, zero, zero_scalar

        non_rigid_delta = xyz - canonical_xyz
        delta_norm = torch.norm(non_rigid_delta, dim=-1)
        if (
            self.non_rigid_delta_rigid_preserve <= 0.0
            and self.non_rigid_delta_soft_preserve <= 0.0
        ):
            zero = torch.zeros_like(non_rigid_delta)
            return non_rigid_delta, zero, zero, delta_norm

        delta_normal_mag = torch.sum(non_rigid_delta * anchor_normal, dim=-1, keepdim=True)
        delta_normal = delta_normal_mag * anchor_normal
        delta_tangent = non_rigid_delta - delta_normal

        rigid_carry = torch.zeros_like(non_rigid_delta)
        if self.non_rigid_delta_rigid_preserve > 0.0:
            rigid_carry = self.non_rigid_delta_rigid_preserve * torch.bmm(
                R_dom,
                non_rigid_delta.unsqueeze(-1),
            ).squeeze(-1)

        soft_carry = torch.zeros_like(non_rigid_delta)
        if self.non_rigid_delta_soft_preserve > 0.0:
            soft_tangent = (
                self.soft_rigid_blend * torch.bmm(R_dom, delta_tangent.unsqueeze(-1)).squeeze(-1)
                + (1.0 - self.soft_rigid_blend) * torch.bmm(R_lbs, delta_tangent.unsqueeze(-1)).squeeze(-1)
            )
            soft_normal = (
                self.soft_normal_blend * torch.bmm(R_dom, delta_normal.unsqueeze(-1)).squeeze(-1)
                + (1.0 - self.soft_normal_blend) * torch.bmm(R_lbs, delta_normal.unsqueeze(-1)).squeeze(-1)
            )
            soft_carry = self.non_rigid_delta_soft_preserve * (soft_tangent + soft_normal)

        return non_rigid_delta, rigid_carry, soft_carry, delta_norm

    def _compute_non_rigid_geometry_state(
        self,
        canonical_xyz,
        xyz,
        anchor_xyz,
        anchor_normal,
        dominant_joint,
    ):
        if canonical_xyz.shape != xyz.shape:
            zero_scalar = torch.zeros((xyz.shape[0],), dtype=xyz.dtype, device=xyz.device)
            return canonical_xyz.detach(), zero_scalar, zero_scalar, zero_scalar

        geometry_delta = (xyz - canonical_xyz).detach()
        if self.non_rigid_geometry_delta_clip > 0.0:
            geometry_delta_norm = torch.norm(geometry_delta, dim=-1, keepdim=True)
            geometry_delta = geometry_delta * torch.clamp(
                self.non_rigid_geometry_delta_clip / geometry_delta_norm.clamp_min(1e-8),
                max=1.0,
            )

        geometry_xyz = canonical_xyz.detach() + self.non_rigid_geometry_blend * geometry_delta
        geometry_shift = torch.norm(geometry_xyz - canonical_xyz.detach(), dim=-1)
        geometry_local_offset = geometry_xyz - anchor_xyz
        geometry_normal_mag = torch.sum(geometry_local_offset * anchor_normal, dim=-1, keepdim=True)
        geometry_surface_distance = torch.abs(geometry_normal_mag.squeeze(-1))
        geometry_bone_distance = self._compute_bone_distance(geometry_xyz, dominant_joint)
        return geometry_xyz, geometry_shift, geometry_bone_distance, geometry_surface_distance

    def _apply_non_rigid_layer_sharpen(self, layer_weights, non_rigid_delta_norm):
        if (
            self.non_rigid_layer_sharpen_strength <= 0.0
            or layer_weights.numel() == 0
            or non_rigid_delta_norm.shape[0] != layer_weights.shape[0]
        ):
            gate = torch.zeros(
                (layer_weights.shape[0],),
                dtype=layer_weights.dtype,
                device=layer_weights.device,
            )
            power = torch.ones_like(gate)
            return layer_weights, gate, power

        delta_gate = torch.sigmoid(
            (non_rigid_delta_norm.detach() - self.non_rigid_layer_sharpen_threshold)
            / max(self.non_rigid_layer_sharpen_width, 1e-6)
        )
        dominant_gate = torch.sigmoid(
            (layer_weights.detach().max(dim=-1).values - self.non_rigid_layer_sharpen_min_dominance)
            / max(self.non_rigid_layer_sharpen_dominance_width, 1e-6)
        )
        sharpen_gate = delta_gate * dominant_gate
        sharpen_power = 1.0 + self.non_rigid_layer_sharpen_strength * sharpen_gate
        sharpened = torch.pow(layer_weights.clamp_min(1e-6), sharpen_power.unsqueeze(-1))
        sharpened = sharpened / sharpened.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return sharpened, sharpen_gate, sharpen_power

    def _build_residual_field_input(
        self,
        canonical_xyz,
        anchor_xyz,
        anchor_normal,
        anchor_weights,
        layer_weights,
        region_probs,
        boundary_score,
        confidence,
        semantic_distance,
        surface_distance,
        non_rigid_delta_norm,
    ):
        local_offset = canonical_xyz - anchor_xyz
        features = [
            local_offset,
            anchor_normal,
            anchor_weights,
            layer_weights,
            region_probs,
            boundary_score.unsqueeze(-1),
            confidence.unsqueeze(-1),
            semantic_distance.unsqueeze(-1),
            surface_distance.unsqueeze(-1),
            non_rigid_delta_norm.unsqueeze(-1),
        ]
        if self.residual_field_feature_detach:
            features = [feat.detach() if torch.is_tensor(feat) else feat for feat in features]
        return torch.cat(features, dim=-1)

    def _compute_residual_field_gate(self, boundary_score, surface_distance):
        if self.residual_field_boundary_gate_enable:
            gate = boundary_score.clamp(0.0, 1.0)
            gate = torch.pow(gate, self.residual_field_boundary_gate_power)
        else:
            gate = torch.ones_like(boundary_score)

        if self.residual_field_surface_gate_enable:
            surface_gate = torch.sigmoid(
                (surface_distance - self.boundary_score_surface_threshold)
                / max(self.residual_field_surface_gate_width, 1e-6)
            )
            gate = gate * surface_gate

        if self.residual_field_boundary_gate_min > 0.0:
            gate = torch.clamp(gate, min=self.residual_field_boundary_gate_min, max=1.0)
        else:
            gate = torch.clamp(gate, 0.0, 1.0)

        if self.residual_field_detach_gate:
            gate = gate.detach()
        return gate

    def _apply_residual_anchor_field(self, anchor_weights, anchor_delta_raw, gate):
        anchor_delta = torch.tanh(anchor_delta_raw) * self.residual_anchor_delta_scale
        if self.residual_anchor_delta_max > 0.0:
            anchor_delta = anchor_delta.clamp(
                min=-self.residual_anchor_delta_max,
                max=self.residual_anchor_delta_max,
            )
        anchor_delta = anchor_delta * gate.unsqueeze(-1)
        anchor_logits = torch.log(anchor_weights.clamp_min(1e-6))
        forward_anchor_weights = F.softmax(anchor_logits + anchor_delta, dim=-1)
        return forward_anchor_weights, anchor_delta

    def _build_local_frame(self, anchor_normal, tangent_offset):
        normal = safe_normalize(anchor_normal)
        tangent_proj = tangent_offset - torch.sum(tangent_offset * normal, dim=-1, keepdim=True) * normal
        tangent_norm = torch.norm(tangent_proj, dim=-1, keepdim=True)

        fallback_x = torch.zeros_like(normal)
        fallback_x[:, 0] = 1.0
        fallback_y = torch.zeros_like(normal)
        fallback_y[:, 1] = 1.0
        tangent_x = torch.cross(normal, fallback_x, dim=-1)
        tangent_y = torch.cross(normal, fallback_y, dim=-1)
        tangent_fallback = torch.where(
            torch.norm(tangent_x, dim=-1, keepdim=True) > 1e-4,
            tangent_x,
            tangent_y,
        )
        tangent = torch.where(
            tangent_norm > 1e-6,
            tangent_proj / tangent_norm.clamp_min(1e-8),
            safe_normalize(tangent_fallback),
        )
        bitangent = safe_normalize(torch.cross(normal, tangent, dim=-1))
        tangent = safe_normalize(torch.cross(bitangent, normal, dim=-1))
        return tangent, bitangent, normal

    def _apply_residual_xbar_field(self, xbar_delta_raw, tangent_u, tangent_v, normal, gate):
        xbar_delta_local = torch.tanh(xbar_delta_raw)
        component_scale = xbar_delta_local.new_tensor([
            self.residual_xbar_tangent_scale,
            self.residual_xbar_tangent_scale,
            self.residual_xbar_normal_scale,
        ])
        xbar_delta_local = xbar_delta_local * component_scale
        xbar_delta_local = xbar_delta_local * (self.residual_xbar_scale * gate).unsqueeze(-1)
        if self.residual_xbar_max > 0.0:
            delta_norm = torch.norm(xbar_delta_local, dim=-1, keepdim=True)
            xbar_delta_local = xbar_delta_local * torch.clamp(
                self.residual_xbar_max / delta_norm.clamp_min(1e-8),
                max=1.0,
            )
        xbar_delta_world = (
            xbar_delta_local[:, [0]] * tangent_u
            + xbar_delta_local[:, [1]] * tangent_v
            + xbar_delta_local[:, [2]] * normal
        )
        return xbar_delta_world, xbar_delta_local

    def _build_hybrid_field_input(
        self,
        canonical_xyz,
        xyz,
        anchor_xyz,
        anchor_normal,
        anchor_weights,
        layer_weights,
        region_probs,
        boundary_score,
        confidence,
        semantic_distance,
        surface_distance,
        non_rigid_delta_norm,
    ):
        local_offset = canonical_xyz - anchor_xyz
        non_rigid_delta = xyz - canonical_xyz
        features = [
            canonical_xyz,
            local_offset,
            anchor_normal,
            anchor_weights,
            layer_weights,
            region_probs,
            confidence.unsqueeze(-1),
            semantic_distance.unsqueeze(-1),
            surface_distance.unsqueeze(-1),
            non_rigid_delta,
            non_rigid_delta_norm.unsqueeze(-1),
            boundary_score.unsqueeze(-1),
        ]
        if self.hybrid_field_feature_detach:
            features = [feat.detach() if torch.is_tensor(feat) else feat for feat in features]
        return torch.cat(features, dim=-1)

    def _compute_hybrid_field_gate(self, layer_weights, boundary_score, confidence):
        mode = self.hybrid_field_gate_mode
        if mode == 'global':
            gate = torch.ones_like(boundary_score)
        elif mode == 'softfree':
            gate = (layer_weights[:, 1] + layer_weights[:, 2]).clamp(0.0, 1.0)
        elif mode == 'boundary_softfree':
            gate = torch.maximum(
                boundary_score.clamp(0.0, 1.0),
                (layer_weights[:, 1] + layer_weights[:, 2]).clamp(0.0, 1.0),
            )
        else:
            raise ValueError(f'Unsupported hybrid_field_gate_mode: {mode}')

        if self.hybrid_field_confidence_suppress > 0.0:
            gate = gate * (1.0 - self.hybrid_field_confidence_suppress * confidence.clamp(0.0, 1.0))
        gate = torch.pow(gate.clamp(0.0, 1.0), self.hybrid_field_gate_power)
        gate = torch.clamp(gate, min=self.hybrid_field_gate_min, max=1.0)
        if self.hybrid_field_detach_gate:
            gate = gate.detach()
        return gate

    def _apply_hybrid_probability_field(
        self,
        base_weights,
        field_raw,
        gate,
        *,
        mode,
        delta_scale,
        delta_max,
        direct_logit_scale,
        name,
    ):
        gate = gate.unsqueeze(-1)
        if mode == 'residual':
            field_delta = torch.tanh(field_raw) * delta_scale
            if delta_max > 0.0:
                field_delta = field_delta.clamp(
                    min=-delta_max,
                    max=delta_max,
                )
            field_delta = field_delta * gate
            base_logits = torch.log(base_weights.clamp_min(1e-6))
            forward_weights = F.softmax(base_logits + field_delta, dim=-1)
            return forward_weights, field_delta

        if mode == 'direct':
            predicted_logits = field_raw * direct_logit_scale
            predicted_weights = F.softmax(predicted_logits, dim=-1)
            forward_weights = torch.lerp(base_weights, predicted_weights, gate)
            forward_weights = forward_weights.clamp_min(1e-8)
            forward_weights = forward_weights / forward_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            field_delta = forward_weights - base_weights
            return forward_weights, field_delta

        raise ValueError(f'Unsupported {name} mode: {mode}')

    def _apply_hybrid_anchor_field(self, anchor_weights, anchor_delta_raw, gate):
        return self._apply_hybrid_probability_field(
            anchor_weights,
            anchor_delta_raw,
            gate,
            mode=self.hybrid_anchor_mode,
            delta_scale=self.hybrid_anchor_delta_scale,
            delta_max=self.hybrid_anchor_delta_max,
            direct_logit_scale=self.hybrid_anchor_direct_logit_scale,
            name='hybrid_anchor',
        )

    def _apply_hybrid_layer_field(self, layer_weights, layer_delta_raw, gate):
        return self._apply_hybrid_probability_field(
            layer_weights,
            layer_delta_raw,
            gate,
            mode=self.hybrid_layer_mode,
            delta_scale=self.hybrid_layer_delta_scale,
            delta_max=self.hybrid_layer_delta_max,
            direct_logit_scale=self.hybrid_layer_direct_logit_scale,
            name='hybrid_layer',
        )

    def _apply_hybrid_support_field(self, support_delta_raw, tangent_u, tangent_v, normal, gate):
        support_delta_local = torch.tanh(support_delta_raw)
        component_scale = support_delta_local.new_tensor([
            self.hybrid_support_tangent_scale,
            self.hybrid_support_tangent_scale,
            self.hybrid_support_normal_scale,
        ])
        support_delta_local = support_delta_local * component_scale
        support_delta_local = support_delta_local * (self.hybrid_support_scale * gate).unsqueeze(-1)
        if self.hybrid_support_max > 0.0:
            delta_norm = torch.norm(support_delta_local, dim=-1, keepdim=True)
            support_delta_local = support_delta_local * torch.clamp(
                self.hybrid_support_max / delta_norm.clamp_min(1e-8),
                max=1.0,
            )
        support_delta_world = (
            support_delta_local[:, [0]] * tangent_u
            + support_delta_local[:, [1]] * tangent_v
            + support_delta_local[:, [2]] * normal
        )
        return support_delta_world, support_delta_local

    def _build_forward_trunk_input(
        self,
        canonical_xyz,
        xyz,
        anchor_xyz,
        anchor_normal,
        anchor_weights,
        layer_weights,
        region_probs,
        boundary_score,
        confidence,
        semantic_distance,
        surface_distance,
        non_rigid_delta_norm,
    ):
        local_offset = canonical_xyz - anchor_xyz
        non_rigid_delta = xyz - canonical_xyz
        features = [
            canonical_xyz,
            local_offset,
            anchor_normal,
            anchor_weights,
            layer_weights,
            region_probs,
            confidence.unsqueeze(-1),
            semantic_distance.unsqueeze(-1),
            surface_distance.unsqueeze(-1),
            non_rigid_delta,
            non_rigid_delta_norm.unsqueeze(-1),
            boundary_score.unsqueeze(-1),
        ]
        if self.forward_trunk_feature_detach:
            features = [feat.detach() if torch.is_tensor(feat) else feat for feat in features]
        return torch.cat(features, dim=-1)

    def _compute_forward_trunk_alpha(self, iteration, confidence):
        alpha = float(resolve_schedule_value(iteration, self.forward_trunk_blend_alpha_cfg, default=0.0))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        alpha_tensor = confidence.new_full(confidence.shape, alpha)
        if self.forward_trunk_confidence_suppress > 0.0:
            alpha_tensor = alpha_tensor * (
                1.0 - self.forward_trunk_confidence_suppress * confidence.clamp(0.0, 1.0)
            )
        return alpha_tensor.clamp(0.0, 1.0)

    def _resolve_forward_trunk_alpha_scale(self, iteration, value, default=1.0):
        resolved = resolve_schedule_value(iteration, value, default=default)
        if resolved is None:
            resolved = default
        return float(resolved)

    def _apply_forward_trunk_probability_field(
        self,
        base_weights,
        field_raw,
        alpha,
        *,
        logit_scale,
        residual=False,
        delta_logit_max=0.0,
        min_prob=0.0,
    ):
        base_logits = torch.log(base_weights.clamp_min(1e-6))
        if delta_logit_max > 0.0:
            learned_field = torch.tanh(field_raw) * delta_logit_max
        else:
            learned_field = field_raw
        learned_logits = learned_field * logit_scale
        if residual:
            blended_logits = base_logits + alpha.unsqueeze(-1) * learned_logits
        else:
            blended_logits = torch.lerp(base_logits, learned_logits, alpha.unsqueeze(-1))
        forward_weights = F.softmax(blended_logits, dim=-1)
        if min_prob > 0.0:
            class_count = float(forward_weights.shape[-1])
            floor = float(np.clip(min_prob, 0.0, max(0.0, (1.0 - 1e-6) / class_count)))
            retain = max(1.0 - class_count * floor, 0.0)
            forward_weights = forward_weights * retain + floor
            forward_weights = forward_weights / forward_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return forward_weights, forward_weights - base_weights

    def _apply_forward_trunk_support_field(self, support_delta_raw, tangent_u, tangent_v, normal, alpha):
        support_delta_local = torch.tanh(support_delta_raw)
        component_scale = support_delta_local.new_tensor([
            self.forward_trunk_support_tangent_scale,
            self.forward_trunk_support_tangent_scale,
            self.forward_trunk_support_normal_scale,
        ])
        support_delta_local = support_delta_local * component_scale
        support_delta_local = support_delta_local * (self.forward_trunk_support_scale * alpha).unsqueeze(-1)
        if self.forward_trunk_support_max > 0.0:
            delta_norm = torch.norm(support_delta_local, dim=-1, keepdim=True)
            support_delta_local = support_delta_local * torch.clamp(
                self.forward_trunk_support_max / delta_norm.clamp_min(1e-8),
                max=1.0,
            )
        support_delta_world = (
            support_delta_local[:, [0]] * tangent_u
            + support_delta_local[:, [1]] * tangent_v
            + support_delta_local[:, [2]] * normal
        )
        return support_delta_world, support_delta_local

    def _apply_forward_trunk_xbar_field(self, xbar_delta_raw, tangent_u, tangent_v, normal, alpha):
        xbar_delta_local = torch.tanh(xbar_delta_raw)
        component_scale = xbar_delta_local.new_tensor([
            self.forward_trunk_xbar_tangent_scale,
            self.forward_trunk_xbar_tangent_scale,
            self.forward_trunk_xbar_normal_scale,
        ])
        xbar_delta_local = xbar_delta_local * component_scale
        xbar_delta_local = xbar_delta_local * (self.forward_trunk_xbar_scale * alpha).unsqueeze(-1)
        if self.forward_trunk_xbar_max > 0.0:
            delta_norm = torch.norm(xbar_delta_local, dim=-1, keepdim=True)
            xbar_delta_local = xbar_delta_local * torch.clamp(
                self.forward_trunk_xbar_max / delta_norm.clamp_min(1e-8),
                max=1.0,
            )
        xbar_delta_canonical = (
            xbar_delta_local[:, [0]] * tangent_u
            + xbar_delta_local[:, [1]] * tangent_v
            + xbar_delta_local[:, [2]] * normal
        )
        return xbar_delta_canonical, xbar_delta_local

    def _forward_empty_gaussians(self, gaussians, canonical_owner=None):
        xyz = gaussians.get_xyz
        device = xyz.device
        dtype = xyz.dtype
        point_count = int(xyz.shape[0])
        joint_count = int(self.cano_joints.shape[0]) if torch.is_tensor(self.cano_joints) else 24

        if canonical_owner is not None and hasattr(canonical_owner, 'clear_binding_state'):
            canonical_owner.clear_binding_state()

        self.temporal_cache.clear()
        self.latest_temporal_slip = torch.zeros((point_count,), device=device, dtype=dtype)

        deformed_gaussians = gaussians.clone()
        T_fwd = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(point_count, 1, 1)
        R_precomp = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).repeat(point_count, 1, 1)
        empty_scalar = torch.zeros((point_count,), dtype=dtype, device=device)
        empty_long = torch.zeros((point_count,), dtype=torch.long, device=device)
        empty_xyz = torch.zeros((point_count, 3), dtype=dtype, device=device)
        empty_weights = torch.zeros((point_count, 3), dtype=dtype, device=device)
        empty_joint_weights = torch.zeros((point_count, joint_count), dtype=dtype, device=device)
        empty_semantic = torch.zeros((point_count, len(COMPACT_SEMANTIC_NAMES)), dtype=dtype, device=device)

        deformed_gaussians.set_fwd_transform(T_fwd.detach())
        deformed_gaussians._xyz = xyz
        setattr(deformed_gaussians, 'rotation_precomp', R_precomp)
        setattr(deformed_gaussians, 'binding_weights', empty_weights)
        setattr(deformed_gaussians, 'binding_weights_raw', empty_weights)
        setattr(deformed_gaussians, 'binding_distance', empty_scalar)
        setattr(deformed_gaussians, 'binding_surface_distance', empty_scalar)
        setattr(deformed_gaussians, 'binding_boundary_score', empty_scalar)
        setattr(deformed_gaussians, 'binding_boundary_live_score', empty_scalar)
        setattr(deformed_gaussians, 'binding_boundary_mixed_score', empty_scalar)
        setattr(deformed_gaussians, 'binding_anchor_ids', empty_long)
        setattr(deformed_gaussians, 'binding_anchor_face_ids', empty_long)
        setattr(deformed_gaussians, 'binding_barycentric', empty_xyz)
        setattr(deformed_gaussians, 'binding_dominant_joint', empty_long)
        setattr(deformed_gaussians, 'binding_layer_ids', empty_long)
        setattr(deformed_gaussians, 'binding_region_probs', empty_weights)
        setattr(deformed_gaussians, 'binding_region_probs_raw', empty_weights)
        setattr(deformed_gaussians, 'binding_region_ids', empty_long)
        setattr(deformed_gaussians, 'binding_compact_semantic_probs', empty_semantic)
        setattr(deformed_gaussians, 'binding_compact_semantic_probs_raw', empty_semantic)
        setattr(deformed_gaussians, 'binding_compact_semantic_ids', empty_long)
        setattr(deformed_gaussians, 'binding_compact_semantic_names', COMPACT_SEMANTIC_NAMES)
        setattr(deformed_gaussians, 'binding_semantic_score', empty_scalar)
        setattr(deformed_gaussians, 'binding_semantic_distance', empty_scalar)
        setattr(deformed_gaussians, 'binding_thin_score', empty_scalar)
        setattr(deformed_gaussians, 'binding_thin_score_raw', empty_scalar)
        setattr(deformed_gaussians, 'binding_part_rigid_prior', empty_scalar)
        setattr(deformed_gaussians, 'binding_part_free_prior', empty_scalar)
        setattr(deformed_gaussians, 'binding_forward_dominant_joint', empty_long)
        setattr(deformed_gaussians, 'binding_forward_anchor_weights', empty_joint_weights)
        setattr(deformed_gaussians, 'binding_forward_trunk_alpha', empty_scalar)
        setattr(deformed_gaussians, 'binding_forward_trunk_anchor_delta', empty_joint_weights)
        setattr(deformed_gaussians, 'binding_forward_trunk_layer_delta', empty_weights)
        setattr(deformed_gaussians, 'binding_forward_trunk_support_delta', empty_xyz)
        setattr(deformed_gaussians, 'binding_forward_trunk_support_delta_local', empty_xyz)
        setattr(deformed_gaussians, 'binding_forward_trunk_xbar_delta', empty_xyz)
        setattr(deformed_gaussians, 'binding_forward_trunk_xbar_delta_local', empty_xyz)
        setattr(deformed_gaussians, 'binding_hybrid_gate', empty_scalar)
        setattr(deformed_gaussians, 'binding_hybrid_anchor_delta', empty_joint_weights)
        setattr(deformed_gaussians, 'binding_hybrid_layer_delta', empty_weights)
        setattr(deformed_gaussians, 'binding_hybrid_support_delta', empty_xyz)
        setattr(deformed_gaussians, 'binding_hybrid_support_delta_local', empty_xyz)
        setattr(deformed_gaussians, 'binding_residual_gate', empty_scalar)
        setattr(deformed_gaussians, 'binding_residual_anchor_delta', empty_joint_weights)
        setattr(deformed_gaussians, 'binding_residual_xbar_delta', empty_xyz)
        setattr(deformed_gaussians, 'binding_residual_xbar_delta_local', empty_xyz)
        setattr(deformed_gaussians, 'binding_non_rigid_delta', empty_scalar)
        setattr(deformed_gaussians, 'binding_non_rigid_rigid_carry', empty_scalar)
        setattr(deformed_gaussians, 'binding_non_rigid_soft_carry', empty_scalar)
        setattr(deformed_gaussians, 'binding_non_rigid_geometry_shift', empty_scalar)
        setattr(deformed_gaussians, 'binding_non_rigid_layer_sharpen_gate', empty_scalar)
        setattr(deformed_gaussians, 'binding_non_rigid_layer_sharpen_power', empty_scalar)
        setattr(deformed_gaussians, 'binding_geometry_xyz', empty_xyz)
        setattr(deformed_gaussians, 'binding_temporal_slip', empty_scalar)

        zero = xyz.new_tensor(0.0)
        self.loss_reg = {
            'binding_rigid': zero,
            'binding_soft': zero,
            'binding_surface': zero,
            'binding_entropy': zero,
            'binding_temporal': zero,
            'binding_body': zero,
            'binding_cloth': zero,
            'binding_layer_sharp': zero,
            'binding_layer_balance': zero,
            'binding_region_sharp': zero,
            'binding_bodycloth_exclusive': zero,
            'binding_semantic_smooth': zero,
            'binding_semantic_cluster': zero,
            'binding_thin_boundary': zero,
            'binding_thin_sparse': zero,
            'binding_compact_entropy': zero,
            'binding_non_rigid_delta_mean': zero,
            'binding_non_rigid_rigid_carry_mean': zero,
            'binding_non_rigid_soft_carry_mean': zero,
            'binding_non_rigid_geometry_shift_mean': zero,
            'binding_non_rigid_geometry_surface_mean': zero,
            'binding_non_rigid_layer_sharpen_gate_mean': zero,
            'binding_non_rigid_layer_sharpen_power_mean': zero,
            'binding_forward_trunk_alpha_mean': zero,
            'binding_forward_trunk_anchor_reg': zero,
            'binding_forward_trunk_anchor_shift_mean': zero,
            'binding_forward_trunk_layer_reg': zero,
            'binding_forward_trunk_layer_shift_mean': zero,
            'binding_forward_trunk_support_reg': zero,
            'binding_forward_trunk_support_abs_mean': zero,
            'binding_forward_trunk_xbar_reg': zero,
            'binding_forward_trunk_xbar_abs_mean': zero,
            'binding_hybrid_gate_mean': zero,
            'binding_hybrid_anchor_reg': zero,
            'binding_hybrid_anchor_shift_mean': zero,
            'binding_hybrid_layer_reg': zero,
            'binding_hybrid_layer_shift_mean': zero,
            'binding_hybrid_support_reg': zero,
            'binding_hybrid_support_abs_mean': zero,
            'binding_residual_gate_mean': zero,
            'binding_residual_anchor_reg': zero,
            'binding_residual_anchor_shift_mean': zero,
            'binding_residual_xbar_reg': zero,
            'binding_residual_xbar_abs_mean': zero,
            'binding_canonical': zero,
            'binding_semantic': zero,
        }
        return deformed_gaussians

    def forward(self, gaussians, iteration, camera):
        bone_transforms = camera.bone_transforms.float()
        xyz = gaussians.get_xyz
        canonical_xyz = getattr(gaussians, 'canonical_xyz', xyz).detach()
        canonical_owner = getattr(gaussians, 'canonical_gaussians', None)
        if xyz.shape[0] == 0 or canonical_xyz.shape[0] == 0:
            return self._forward_empty_gaussians(gaussians, canonical_owner=canonical_owner)
        binding = self._get_binding(canonical_xyz, iteration, canonical_owner=canonical_owner)
        if canonical_owner is not None and hasattr(canonical_owner, 'set_binding_state'):
            canonical_owner.set_binding_state(binding)

        anchor_xyz = binding['anchor_xyz']
        anchor_weights = binding['anchor_weights']
        dominant_joint = binding['dominant_joint']
        layer_weights = binding['layer_weights']
        region_probs = binding['region_probs']
        bound_xyz = binding['bound_xyz']
        normal_offset = binding['normal_offset']
        tangent_offset = binding['tangent_offset']
        confidence = binding.get('anchor_confidence', anchor_weights.max(dim=-1).values)
        semantic_distance = binding['semantic_distance']

        T_lbs = torch.matmul(anchor_weights, bone_transforms.view(-1, 16)).view(-1, 4, 4).float()
        T_dom = bone_transforms[dominant_joint].float()

        x_anchor_dom = transform_points(T_dom, anchor_xyz)
        x_anchor_lbs = transform_points(T_lbs, anchor_xyz)
        R_dom = T_dom[:, :3, :3]
        R_lbs = T_lbs[:, :3, :3]
        non_rigid_delta, rigid_nr_carry, soft_nr_carry, non_rigid_delta_norm = self._compute_non_rigid_carry(
            canonical_xyz,
            xyz,
            binding['anchor_normal'],
            R_dom,
            R_lbs,
        )
        geometry_xyz = canonical_xyz.detach()
        geometry_shift = torch.zeros_like(non_rigid_delta_norm)
        effective_bone_distance = binding['bone_distance']
        effective_surface_distance = binding['surface_distance']
        if self.non_rigid_geometry_blend > 0.0:
            (
                geometry_xyz,
                geometry_shift,
                effective_bone_distance,
                effective_surface_distance,
            ) = self._compute_non_rigid_geometry_state(
                canonical_xyz,
                xyz,
                anchor_xyz,
                binding['anchor_normal'],
                dominant_joint,
            )
            layer_weights, region_probs = self._compute_layer_weights(
                effective_bone_distance,
                effective_surface_distance,
                confidence,
                semantic_distance,
            )

        point_scale = gaussians.get_scaling.min(dim=-1).values.detach()
        layer_weights, region_probs, thin_score, part_rigid, part_free = self._apply_v41_priors(
            layer_weights,
            region_probs,
            dominant_joint,
            point_scale,
            confidence,
            effective_surface_distance,
            semantic_distance,
        )
        layer_weights, layer_sharpen_gate, layer_sharpen_power = self._apply_non_rigid_layer_sharpen(
            layer_weights,
            non_rigid_delta_norm,
        )
        boundary_score = self._compute_boundary_score(
            layer_weights,
            effective_surface_distance,
            thin_score,
            confidence,
        )
        compact_semantic_probs = self._compute_compact_semantic_probs(
            dominant_joint,
            region_probs,
            confidence,
            effective_surface_distance,
            semantic_distance,
            thin_score,
        )

        forward_anchor_weights = anchor_weights
        forward_layer_weights = layer_weights
        forward_dominant_joint = dominant_joint
        forward_source_xyz = xyz
        forward_tangent_offset = tangent_offset
        forward_normal_offset = normal_offset
        forward_trunk_alpha = torch.zeros_like(boundary_score)
        forward_trunk_anchor_delta = torch.zeros_like(anchor_weights)
        forward_trunk_layer_delta = torch.zeros_like(layer_weights)
        forward_trunk_support_delta_world = torch.zeros_like(xyz)
        forward_trunk_support_delta_local = torch.zeros((xyz.shape[0], 3), dtype=xyz.dtype, device=xyz.device)
        forward_trunk_xbar_delta_canonical = torch.zeros_like(xyz)
        forward_trunk_xbar_delta_world = torch.zeros_like(xyz)
        forward_trunk_xbar_delta_local = torch.zeros((xyz.shape[0], 3), dtype=xyz.dtype, device=xyz.device)
        if self.forward_trunk_mlp is not None:
            forward_trunk_input = self._build_forward_trunk_input(
                canonical_xyz,
                xyz,
                anchor_xyz,
                binding['anchor_normal'],
                anchor_weights,
                layer_weights,
                region_probs,
                boundary_score,
                confidence,
                semantic_distance,
                effective_surface_distance,
                non_rigid_delta_norm,
            )
            forward_trunk_output = self.forward_trunk_mlp(forward_trunk_input)
            forward_trunk_output = torch.nan_to_num(forward_trunk_output, nan=0.0, posinf=0.0, neginf=0.0)
            if self.forward_trunk_output_clamp > 0.0:
                forward_trunk_output = forward_trunk_output.clamp(
                    min=-self.forward_trunk_output_clamp,
                    max=self.forward_trunk_output_clamp,
                )
            forward_trunk_alpha = self._compute_forward_trunk_alpha(iteration, confidence)
            forward_anchor_alpha_scale = self._resolve_forward_trunk_alpha_scale(
                iteration,
                self.forward_trunk_anchor_alpha_scale_cfg,
                default=1.0,
            )
            forward_support_alpha_scale = self._resolve_forward_trunk_alpha_scale(
                iteration,
                self.forward_trunk_support_alpha_scale_cfg,
                default=1.0,
            )
            forward_layer_alpha_scale = self._resolve_forward_trunk_alpha_scale(
                iteration,
                self.forward_trunk_layer_alpha_scale_cfg,
                default=1.0,
            )
            forward_xbar_alpha_scale = self._resolve_forward_trunk_alpha_scale(
                iteration,
                self.forward_trunk_xbar_alpha_scale_cfg,
                default=1.0,
            )
            forward_anchor_alpha = (forward_trunk_alpha * forward_anchor_alpha_scale).clamp(0.0, 1.0)
            forward_support_alpha = (forward_trunk_alpha * forward_support_alpha_scale).clamp(0.0, 1.0)
            forward_layer_alpha = (forward_trunk_alpha * forward_layer_alpha_scale).clamp(0.0, 1.0)
            forward_xbar_alpha = (forward_trunk_alpha * forward_xbar_alpha_scale).clamp(0.0, 1.0)
            output_offset = 0
            tangent_u = None
            tangent_v = None
            normal = None
            if self.forward_trunk_anchor_enable:
                forward_anchor_weights, forward_trunk_anchor_delta = self._apply_forward_trunk_probability_field(
                    anchor_weights,
                    forward_trunk_output[:, output_offset:output_offset + 24],
                    forward_anchor_alpha,
                    logit_scale=self.forward_trunk_anchor_logit_scale,
                    residual=self.forward_trunk_anchor_residual,
                    delta_logit_max=self.forward_trunk_anchor_delta_logit_max,
                )
                if self.forward_trunk_update_dominant_joint:
                    forward_dominant_joint = torch.argmax(forward_anchor_weights, dim=-1)
                output_offset += 24
            if self.forward_trunk_support_enable:
                tangent_u, tangent_v, normal = self._build_local_frame(binding['anchor_normal'], tangent_offset)
                (
                    forward_trunk_support_delta_world,
                    forward_trunk_support_delta_local,
                ) = self._apply_forward_trunk_support_field(
                    forward_trunk_output[:, output_offset:output_offset + 3],
                    tangent_u,
                    tangent_v,
                    normal,
                    forward_support_alpha,
                )
                forward_source_xyz = xyz + forward_trunk_support_delta_world
                forward_local_offset = tangent_offset + normal_offset + forward_trunk_support_delta_world
                forward_normal_mag = torch.sum(forward_local_offset * binding['anchor_normal'], dim=-1, keepdim=True)
                forward_normal_offset = forward_normal_mag * binding['anchor_normal']
                forward_tangent_offset = forward_local_offset - forward_normal_offset
                output_offset += 3
            if self.forward_trunk_layer_enable:
                forward_layer_weights, forward_trunk_layer_delta = self._apply_forward_trunk_probability_field(
                    layer_weights,
                    forward_trunk_output[:, output_offset:output_offset + 3],
                    forward_layer_alpha,
                    logit_scale=self.forward_trunk_layer_logit_scale,
                    residual=self.forward_trunk_layer_residual,
                    delta_logit_max=self.forward_trunk_layer_delta_logit_max,
                    min_prob=self.forward_trunk_layer_min_prob,
                )
                output_offset += 3
            if self.forward_trunk_xbar_enable:
                if tangent_u is None or tangent_v is None or normal is None:
                    tangent_u, tangent_v, normal = self._build_local_frame(binding['anchor_normal'], tangent_offset)
                (
                    forward_trunk_xbar_delta_canonical,
                    forward_trunk_xbar_delta_local,
                ) = self._apply_forward_trunk_xbar_field(
                    forward_trunk_output[:, output_offset:output_offset + 3],
                    tangent_u,
                    tangent_v,
                    normal,
                    forward_xbar_alpha,
                )
                output_offset += 3

        hybrid_base_anchor_weights = forward_anchor_weights
        hybrid_base_layer_weights = forward_layer_weights
        hybrid_gate = torch.zeros_like(boundary_score)
        hybrid_anchor_delta = torch.zeros_like(anchor_weights)
        hybrid_layer_delta = torch.zeros_like(layer_weights)
        hybrid_support_delta_world = torch.zeros_like(xyz)
        hybrid_support_delta_local = torch.zeros((xyz.shape[0], 3), dtype=xyz.dtype, device=xyz.device)
        if self.hybrid_field_mlp is not None:
            hybrid_input = self._build_hybrid_field_input(
                canonical_xyz,
                xyz,
                anchor_xyz,
                binding['anchor_normal'],
                anchor_weights,
                layer_weights,
                region_probs,
                boundary_score,
                confidence,
                semantic_distance,
                effective_surface_distance,
                non_rigid_delta_norm,
            )
            hybrid_output = self.hybrid_field_mlp(hybrid_input)
            hybrid_gate = self._compute_hybrid_field_gate(layer_weights, boundary_score, confidence)
            output_offset = 0
            if self.hybrid_field_anchor_enable:
                forward_anchor_weights, hybrid_anchor_delta = self._apply_hybrid_anchor_field(
                    anchor_weights,
                    hybrid_output[:, output_offset:output_offset + 24],
                    hybrid_gate,
                )
                if self.hybrid_field_update_dominant_joint:
                    forward_dominant_joint = torch.argmax(forward_anchor_weights, dim=-1)
                output_offset += 24
            if self.hybrid_field_support_enable:
                tangent_u, tangent_v, normal = self._build_local_frame(binding['anchor_normal'], tangent_offset)
                hybrid_support_delta_world, hybrid_support_delta_local = self._apply_hybrid_support_field(
                    hybrid_output[:, output_offset:output_offset + 3],
                    tangent_u,
                    tangent_v,
                    normal,
                    hybrid_gate,
                )
                forward_source_xyz = xyz + hybrid_support_delta_world
                forward_local_offset = tangent_offset + normal_offset + hybrid_support_delta_world
                forward_normal_mag = torch.sum(forward_local_offset * binding['anchor_normal'], dim=-1, keepdim=True)
                forward_normal_offset = forward_normal_mag * binding['anchor_normal']
                forward_tangent_offset = forward_local_offset - forward_normal_offset
                output_offset += 3
            if self.hybrid_field_layer_enable:
                forward_layer_weights, hybrid_layer_delta = self._apply_hybrid_layer_field(
                    layer_weights,
                    hybrid_output[:, output_offset:output_offset + 3],
                    hybrid_gate,
                )
                output_offset += 3

        residual_base_anchor_weights = forward_anchor_weights
        residual_base_layer_weights = forward_layer_weights
        residual_gate = torch.zeros_like(boundary_score)
        residual_anchor_delta = torch.zeros_like(anchor_weights)
        residual_xbar_delta_world = torch.zeros_like(xyz)
        residual_xbar_delta_local = torch.zeros((xyz.shape[0], 3), dtype=xyz.dtype, device=xyz.device)
        if self.residual_field_mlp is not None:
            residual_input = self._build_residual_field_input(
                canonical_xyz,
                anchor_xyz,
                binding['anchor_normal'],
                residual_base_anchor_weights,
                residual_base_layer_weights,
                region_probs,
                boundary_score,
                confidence,
                semantic_distance,
                effective_surface_distance,
                non_rigid_delta_norm,
            )
            residual_output = self.residual_field_mlp(residual_input)
            residual_gate = self._compute_residual_field_gate(boundary_score, effective_surface_distance)
            output_offset = 0
            if self.residual_field_anchor_enable:
                forward_anchor_weights, residual_anchor_delta = self._apply_residual_anchor_field(
                    residual_base_anchor_weights,
                    residual_output[:, output_offset:output_offset + 24],
                    residual_gate,
                )
                if self.residual_field_update_dominant_joint:
                    forward_dominant_joint = torch.argmax(forward_anchor_weights, dim=-1)
                output_offset += 24
            if self.residual_field_xbar_enable:
                tangent_u, tangent_v, normal = self._build_local_frame(binding['anchor_normal'], tangent_offset)
                residual_xbar_delta_world, residual_xbar_delta_local = self._apply_residual_xbar_field(
                    residual_output[:, output_offset:output_offset + 3],
                    tangent_u,
                    tangent_v,
                    normal,
                    residual_gate,
                )

        T_lbs = torch.matmul(forward_anchor_weights, bone_transforms.view(-1, 16)).view(-1, 4, 4).float()
        T_dom = bone_transforms[forward_dominant_joint].float()
        x_anchor_dom = transform_points(T_dom, anchor_xyz)
        x_anchor_lbs = transform_points(T_lbs, anchor_xyz)
        R_dom = T_dom[:, :3, :3]
        R_lbs = T_lbs[:, :3, :3]
        _, rigid_nr_carry, soft_nr_carry, _ = self._compute_non_rigid_carry(
            canonical_xyz,
            xyz,
            binding['anchor_normal'],
            R_dom,
            R_lbs,
        )

        rigid_offset = torch.bmm(R_dom, (forward_tangent_offset + forward_normal_offset).unsqueeze(-1)).squeeze(-1)
        x_rigid = x_anchor_dom + rigid_offset + rigid_nr_carry
        x_free = transform_points(T_lbs, forward_source_xyz)
        soft_tangent = (
            self.soft_rigid_blend * torch.bmm(R_dom, forward_tangent_offset.unsqueeze(-1)).squeeze(-1) +
            (1. - self.soft_rigid_blend) * torch.bmm(R_lbs, forward_tangent_offset.unsqueeze(-1)).squeeze(-1)
        )
        soft_normal = (
            self.soft_normal_blend * torch.bmm(R_dom, forward_normal_offset.unsqueeze(-1)).squeeze(-1) +
            (1. - self.soft_normal_blend) * torch.bmm(R_lbs, forward_normal_offset.unsqueeze(-1)).squeeze(-1)
        )
        x_soft_anchor = self.soft_rigid_blend * x_anchor_dom + (1. - self.soft_rigid_blend) * x_anchor_lbs
        x_soft = x_soft_anchor + soft_tangent + soft_normal + soft_nr_carry

        x_bar = (
            forward_layer_weights[:, [0]] * x_rigid +
            forward_layer_weights[:, [1]] * x_soft +
            forward_layer_weights[:, [2]] * x_free
        )
        x_bar_base = x_bar

        R_soft = self.soft_rigid_blend * R_dom + (1. - self.soft_rigid_blend) * R_lbs
        R_bar = (
            forward_layer_weights[:, [0]].unsqueeze(-1) * R_dom +
            forward_layer_weights[:, [1]].unsqueeze(-1) * R_soft +
            forward_layer_weights[:, [2]].unsqueeze(-1) * R_lbs
        )
        if self.forward_trunk_xbar_enable:
            forward_trunk_xbar_delta_world = torch.bmm(
                R_bar,
                forward_trunk_xbar_delta_canonical.unsqueeze(-1),
            ).squeeze(-1)
        x_bar = x_bar + forward_trunk_xbar_delta_world
        x_bar_after_forward_trunk = x_bar
        x_bar = x_bar + residual_xbar_delta_world
        T_fwd = compose_pointwise_transform(R_bar, xyz, x_bar)

        deformed_gaussians = gaussians.clone()
        deformed_gaussians.set_fwd_transform(T_fwd.detach())
        deformed_gaussians._xyz = x_bar
        setattr(deformed_gaussians, 'rotation_precomp', torch.matmul(R_bar, build_rotation(gaussians._rotation)))
        setattr(deformed_gaussians, 'binding_weights', forward_layer_weights.detach())
        setattr(deformed_gaussians, 'binding_weights_raw', forward_layer_weights)
        setattr(deformed_gaussians, 'binding_distance', effective_bone_distance.detach())
        setattr(deformed_gaussians, 'binding_surface_distance', effective_surface_distance.detach())
        setattr(deformed_gaussians, 'binding_boundary_score', boundary_score.detach())
        setattr(deformed_gaussians, 'binding_anchor_ids', binding['anchor_ids'].detach())
        setattr(deformed_gaussians, 'binding_anchor_face_ids', binding['anchor_face_ids'].detach())
        setattr(deformed_gaussians, 'binding_barycentric', binding['anchor_barycentric'].detach())
        setattr(deformed_gaussians, 'binding_dominant_joint', dominant_joint.detach())
        setattr(deformed_gaussians, 'binding_layer_ids', torch.argmax(forward_layer_weights, dim=-1).detach())
        setattr(deformed_gaussians, 'binding_region_probs', region_probs.detach())
        setattr(deformed_gaussians, 'binding_region_probs_raw', region_probs)
        setattr(deformed_gaussians, 'binding_region_ids', torch.argmax(region_probs, dim=-1).detach())
        setattr(deformed_gaussians, 'binding_compact_semantic_probs', compact_semantic_probs.detach())
        setattr(deformed_gaussians, 'binding_compact_semantic_probs_raw', compact_semantic_probs)
        setattr(deformed_gaussians, 'binding_compact_semantic_ids', torch.argmax(compact_semantic_probs, dim=-1).detach())
        setattr(deformed_gaussians, 'binding_compact_semantic_names', COMPACT_SEMANTIC_NAMES)
        setattr(deformed_gaussians, 'binding_semantic_score', binding['semantic_score'].detach())
        setattr(deformed_gaussians, 'binding_semantic_distance', binding['semantic_distance'].detach())
        setattr(deformed_gaussians, 'binding_thin_score', thin_score.detach())
        setattr(deformed_gaussians, 'binding_thin_score_raw', thin_score)
        setattr(deformed_gaussians, 'binding_part_rigid_prior', part_rigid.detach())
        setattr(deformed_gaussians, 'binding_part_free_prior', part_free.detach())
        setattr(deformed_gaussians, 'binding_forward_dominant_joint', forward_dominant_joint.detach())
        setattr(deformed_gaussians, 'binding_forward_anchor_weights', forward_anchor_weights.detach())
        setattr(deformed_gaussians, 'binding_forward_trunk_alpha', forward_trunk_alpha.detach())
        setattr(deformed_gaussians, 'binding_forward_trunk_anchor_delta', forward_trunk_anchor_delta.detach())
        setattr(deformed_gaussians, 'binding_forward_trunk_layer_delta', forward_trunk_layer_delta.detach())
        setattr(deformed_gaussians, 'binding_forward_trunk_support_delta', forward_trunk_support_delta_world.detach())
        setattr(deformed_gaussians, 'binding_forward_trunk_support_delta_local', forward_trunk_support_delta_local.detach())
        setattr(deformed_gaussians, 'binding_forward_trunk_xbar_delta', forward_trunk_xbar_delta_world.detach())
        setattr(deformed_gaussians, 'binding_forward_trunk_xbar_delta_local', forward_trunk_xbar_delta_local.detach())
        setattr(deformed_gaussians, 'binding_hybrid_gate', hybrid_gate.detach())
        setattr(deformed_gaussians, 'binding_hybrid_anchor_delta', hybrid_anchor_delta.detach())
        setattr(deformed_gaussians, 'binding_hybrid_layer_delta', hybrid_layer_delta.detach())
        setattr(deformed_gaussians, 'binding_hybrid_support_delta', hybrid_support_delta_world.detach())
        setattr(deformed_gaussians, 'binding_hybrid_support_delta_local', hybrid_support_delta_local.detach())
        setattr(deformed_gaussians, 'binding_residual_gate', residual_gate.detach())
        setattr(deformed_gaussians, 'binding_residual_anchor_delta', residual_anchor_delta.detach())
        setattr(deformed_gaussians, 'binding_residual_xbar_delta', residual_xbar_delta_world.detach())
        setattr(deformed_gaussians, 'binding_residual_xbar_delta_local', residual_xbar_delta_local.detach())
        setattr(deformed_gaussians, 'binding_non_rigid_delta', non_rigid_delta_norm.detach())
        setattr(deformed_gaussians, 'binding_non_rigid_rigid_carry', torch.norm(rigid_nr_carry, dim=-1).detach())
        setattr(deformed_gaussians, 'binding_non_rigid_soft_carry', torch.norm(soft_nr_carry, dim=-1).detach())
        setattr(deformed_gaussians, 'binding_non_rigid_geometry_shift', geometry_shift.detach())
        setattr(deformed_gaussians, 'binding_non_rigid_layer_sharpen_gate', layer_sharpen_gate.detach())
        setattr(deformed_gaussians, 'binding_non_rigid_layer_sharpen_power', layer_sharpen_power.detach())
        setattr(deformed_gaussians, 'binding_geometry_xyz', geometry_xyz.detach())

        slip_distance = torch.norm(x_free - x_rigid, dim=-1)
        entropy = -(forward_layer_weights * torch.log(forward_layer_weights.clamp_min(1e-8))).sum(dim=-1)
        local_motion = torch.bmm(R_dom.transpose(1, 2), (x_bar - x_anchor_dom).unsqueeze(-1)).squeeze(-1)
        surface_distance = effective_surface_distance
        temporal_loss = self._temporal_loss(camera.frame_id, local_motion, forward_layer_weights)
        temporal_slip = self.latest_temporal_slip
        if temporal_slip is None:
            temporal_slip = torch.zeros_like(slip_distance)
        setattr(deformed_gaussians, 'binding_temporal_slip', temporal_slip.detach())

        layer_sharp = (forward_layer_weights * (1. - forward_layer_weights)).sum(dim=-1).mean()
        region_sharp = (region_probs * (1. - region_probs)).sum(dim=-1).mean()
        bodycloth_exclusive = (region_probs[:, 0] * region_probs[:, 2]).mean()
        layer_target = forward_layer_weights.new_tensor([0.40, 0.20, 0.40])
        layer_balance = torch.abs(forward_layer_weights.mean(dim=0) - layer_target).mean()
        semantic_smooth = sample_neighbor_consistency(binding['semantic_distance'], canonical_xyz, k=6, max_samples=2048)
        semantic_cluster = sample_neighbor_consistency(binding['semantic_score'], canonical_xyz, k=6, max_samples=2048)
        overlap = torch.sqrt((region_probs[:, 0] * region_probs[:, 2]).clamp_min(0.0))
        thin_target = torch.clamp(
            0.65 * overlap
            + 0.35 * torch.sqrt((region_probs[:, 2] * forward_layer_weights[:, 2]).clamp_min(0.0)),
            0.0,
            1.0,
        )
        thin_boundary = F.l1_loss(thin_score, thin_target)
        thin_sparse = thin_score.mean()
        compact_entropy = -(compact_semantic_probs * torch.log(compact_semantic_probs.clamp_min(1e-8))).sum(dim=-1).mean()
        forward_trunk_anchor_shift = torch.abs(hybrid_base_anchor_weights - anchor_weights).sum(dim=-1)
        forward_trunk_layer_shift = torch.abs(hybrid_base_layer_weights - layer_weights).sum(dim=-1)
        forward_trunk_support_shift = torch.norm(forward_trunk_support_delta_local, dim=-1)
        forward_trunk_xbar_shift = torch.norm(forward_trunk_xbar_delta_local, dim=-1)
        hybrid_anchor_shift = torch.abs(residual_base_anchor_weights - hybrid_base_anchor_weights).sum(dim=-1)
        hybrid_layer_shift = torch.abs(residual_base_layer_weights - hybrid_base_layer_weights).sum(dim=-1)
        hybrid_support_shift = torch.norm(hybrid_support_delta_local, dim=-1)
        residual_anchor_shift = torch.abs(forward_anchor_weights - residual_base_anchor_weights).sum(dim=-1)
        residual_xbar_shift = torch.norm(x_bar - x_bar_after_forward_trunk, dim=-1)

        self.loss_reg = {
            'binding_rigid': weighted_reduce(slip_distance, forward_layer_weights[:, 0]),
            'binding_soft': weighted_reduce(slip_distance, forward_layer_weights[:, 1]),
            'binding_surface': weighted_reduce(surface_distance, forward_layer_weights[:, 0] + 0.5 * forward_layer_weights[:, 1]),
            'binding_entropy': entropy.mean(),
            'binding_temporal': temporal_loss,
            'binding_body': region_probs[:, 0].mean(),
            'binding_cloth': region_probs[:, 2].mean(),
            'binding_layer_sharp': layer_sharp,
            'binding_layer_balance': layer_balance,
            'binding_region_sharp': region_sharp,
            'binding_bodycloth_exclusive': bodycloth_exclusive,
            'binding_semantic_smooth': semantic_smooth,
            'binding_semantic_cluster': semantic_cluster,
            'binding_thin_boundary': thin_boundary,
            'binding_thin_sparse': thin_sparse,
            'binding_compact_entropy': compact_entropy,
            'binding_non_rigid_delta_mean': non_rigid_delta_norm.mean(),
            'binding_non_rigid_rigid_carry_mean': torch.norm(rigid_nr_carry, dim=-1).mean(),
            'binding_non_rigid_soft_carry_mean': torch.norm(soft_nr_carry, dim=-1).mean(),
            'binding_non_rigid_geometry_shift_mean': geometry_shift.mean(),
            'binding_non_rigid_geometry_surface_mean': effective_surface_distance.mean(),
            'binding_non_rigid_layer_sharpen_gate_mean': layer_sharpen_gate.mean(),
            'binding_non_rigid_layer_sharpen_power_mean': layer_sharpen_power.mean(),
            'binding_forward_trunk_alpha_mean': forward_trunk_alpha.mean(),
            'binding_forward_trunk_anchor_reg': forward_trunk_anchor_delta.abs().mean(),
            'binding_forward_trunk_anchor_shift_mean': forward_trunk_anchor_shift.mean(),
            'binding_forward_trunk_layer_reg': forward_trunk_layer_delta.abs().mean(),
            'binding_forward_trunk_layer_shift_mean': forward_trunk_layer_shift.mean(),
            'binding_forward_trunk_support_reg': forward_trunk_support_shift.mean(),
            'binding_forward_trunk_support_abs_mean': forward_trunk_support_delta_local.abs().mean(),
            'binding_forward_trunk_xbar_reg': forward_trunk_xbar_shift.mean(),
            'binding_forward_trunk_xbar_abs_mean': forward_trunk_xbar_delta_local.abs().mean(),
            'binding_hybrid_gate_mean': hybrid_gate.mean(),
            'binding_hybrid_anchor_reg': hybrid_anchor_delta.abs().mean(),
            'binding_hybrid_anchor_shift_mean': hybrid_anchor_shift.mean(),
            'binding_hybrid_layer_reg': hybrid_layer_delta.abs().mean(),
            'binding_hybrid_layer_shift_mean': hybrid_layer_shift.mean(),
            'binding_hybrid_support_reg': hybrid_support_shift.mean(),
            'binding_hybrid_support_abs_mean': hybrid_support_delta_local.abs().mean(),
            'binding_residual_gate_mean': residual_gate.mean(),
            'binding_residual_anchor_reg': residual_anchor_delta.abs().mean(),
            'binding_residual_anchor_shift_mean': residual_anchor_shift.mean(),
            'binding_residual_xbar_reg': residual_xbar_shift.mean(),
            'binding_residual_xbar_abs_mean': residual_xbar_delta_local.abs().mean(),
        }

        if (
            self.training
            and self.non_rigid_delta_debug_interval > 0
            and iteration >= 0
            and iteration % self.non_rigid_delta_debug_interval == 0
        ):
            rigid_layer_mass = float(forward_layer_weights[:, 0].mean().item())
            soft_layer_mass = float(forward_layer_weights[:, 1].mean().item())
            free_layer_mass = float(forward_layer_weights[:, 2].mean().item())
            print(
                '[ExplicitBinding] non-rigid carry '
                f'iter={iteration} '
                f'delta_mean={float(non_rigid_delta_norm.mean().item()):.6f} '
                f'rigid_carry_mean={float(torch.norm(rigid_nr_carry, dim=-1).mean().item()):.6f} '
                f'soft_carry_mean={float(torch.norm(soft_nr_carry, dim=-1).mean().item()):.6f} '
                f'geom_shift_mean={float(geometry_shift.mean().item()):.6f} '
                f'sharpen_gate_mean={float(layer_sharpen_gate.mean().item()):.6f} '
                f'sharpen_power_mean={float(layer_sharpen_power.mean().item()):.6f} '
                f'fwd_alpha_mean={float(forward_trunk_alpha.mean().item()):.6f} '
                f'layer_mean=({rigid_layer_mass:.4f},{soft_layer_mass:.4f},{free_layer_mass:.4f}) '
                f'preserve=({self.non_rigid_delta_rigid_preserve:.3f},{self.non_rigid_delta_soft_preserve:.3f}) '
                f'geom_blend={self.non_rigid_geometry_blend:.3f}'
            )

        slip_canonical = torch.norm(xyz - bound_xyz, dim=-1)
        self.loss_reg['binding_canonical'] = weighted_reduce(
            slip_canonical,
            forward_layer_weights[:, 0] + 0.5 * forward_layer_weights[:, 1],
        )
        self.loss_reg['binding_semantic'] = binding['semantic_distance'].mean()
        return deformed_gaussians

    def regularization(self):
        return {}

def get_rigid_deform(cfg, metadata):
    name = cfg.name
    model_dict = {
        "identity": Identity,
        "smpl_nn": SMPLNN,
        "skinning_field": SkinningField,
        "explicit_binding": ExplicitBinding,
    }
    return model_dict[name](cfg, metadata)
