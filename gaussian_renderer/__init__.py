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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def _build_rasterizer(data, pipe, bg_color, pc, scaling_modifier):
    tanfovx = math.tan(data.FoVx * 0.5)
    tanfovy = math.tan(data.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(data.image_height),
        image_width=int(data.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data.world_view_transform,
        projmatrix=data.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=data.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    return GaussianRasterizer(raster_settings=raster_settings)


def rasterize_gaussians(data, pc, pipe, bg_color, colors_precomp=None, scaling_modifier=1.0, return_opacity=False):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device='cuda') + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    rasterizer = _build_rasterizer(data, pipe, bg_color, pc, scaling_modifier)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    opacity_image = None
    if return_opacity:
        opacity_image, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=torch.ones(opacity.shape[0], 3, device=opacity.device),
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        opacity_image = opacity_image[:1]

    return {
        'render': rendered_image,
        'viewspace_points': screenspace_points,
        'visibility_filter': radii > 0,
        'radii': radii,
        'opacity_render': opacity_image,
    }


def render(data,
           iteration,
           scene,
           pipe,
           bg_color: torch.Tensor,
           scaling_modifier=1.0,
           override_color=None,
           compute_loss=True,
           return_opacity=False):
    """Render the scene. Background tensor (bg_color) must be on GPU!"""
    pc, loss_reg, colors_precomp = scene.convert_gaussians(data, iteration, compute_loss)
    if override_color is not None:
        colors_precomp = override_color

    raster_pkg = rasterize_gaussians(
        data,
        pc,
        pipe,
        bg_color,
        colors_precomp=colors_precomp,
        scaling_modifier=scaling_modifier,
        return_opacity=return_opacity,
    )
    raster_pkg.update({
        'deformed_gaussian': pc,
        'loss_reg': loss_reg,
    })
    return raster_pkg
