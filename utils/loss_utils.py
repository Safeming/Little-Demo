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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

import numpy as np
import cv2
from utils.pytorch3d_compat import knn_points

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def full_aiap_loss(gs_can, gs_obs, n_neighbors=5, max_points=0):
    xyz_can = gs_can.get_xyz
    xyz_obs = gs_obs.get_xyz

    cov_can = gs_can.get_covariance()
    cov_obs = gs_obs.get_covariance()

    if xyz_can.shape[0] < 2 or xyz_obs.shape[0] < 2:
        device = xyz_can.device if xyz_can.numel() > 0 else xyz_obs.device
        zero = torch.tensor(0.0, device=device)
        return zero, zero

    if max_points is not None and int(max_points) > 0 and xyz_can.shape[0] > int(max_points):
        sample_count = int(max_points)
        # Deterministic uniform subsampling keeps cost bounded without introducing extra RNG drift.
        sample_idx = torch.linspace(0, xyz_can.shape[0] - 1, sample_count, device=xyz_can.device).long()
        xyz_can = xyz_can.index_select(0, sample_idx)
        xyz_obs = xyz_obs.index_select(0, sample_idx)
        cov_can = cov_can.index_select(0, sample_idx)
        cov_obs = cov_obs.index_select(0, sample_idx)

    k = min(int(n_neighbors), int(xyz_can.shape[0]))
    if k < 2:
        zero = torch.tensor(0.0, device=xyz_can.device)
        return zero, zero

    _, nn_ix, _ = knn_points(xyz_can.unsqueeze(0),
                             xyz_can.unsqueeze(0),
                             K=k,
                             return_sorted=True)
    nn_ix = nn_ix.squeeze(0)

    loss_xyz = aiap_loss(xyz_can, xyz_obs, nn_ix=nn_ix)
    loss_cov = aiap_loss(cov_can, cov_obs, nn_ix=nn_ix)

    return loss_xyz, loss_cov

def aiap_loss(x_canonical, x_deformed, n_neighbors=5, nn_ix=None):
    if x_canonical.shape != x_deformed.shape:
        raise ValueError("Input point sets must have the same shape.")

    if x_canonical.shape[0] < 2:
        return torch.tensor(0.0, device=x_canonical.device)

    if nn_ix is None:
        k = min(int(n_neighbors) + 1, int(x_canonical.shape[0]))
        if k < 2:
            return torch.tensor(0.0, device=x_canonical.device)
        _, nn_ix, _ = knn_points(x_canonical.unsqueeze(0),
                                 x_canonical.unsqueeze(0),
                                 K=k,
                                 return_sorted=True)
        nn_ix = nn_ix.squeeze(0)

    if nn_ix.shape[-1] < 2:
        return torch.tensor(0.0, device=x_canonical.device)

    dists_canonical = torch.cdist(x_canonical.unsqueeze(1), x_canonical[nn_ix])[:,0,1:]
    dists_deformed = torch.cdist(x_deformed.unsqueeze(1), x_deformed[nn_ix])[:,0,1:]

    if dists_canonical.numel() == 0 or dists_deformed.numel() == 0:
        return torch.tensor(0.0, device=x_canonical.device)

    loss = F.l1_loss(dists_canonical, dists_deformed)

    return loss
