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
import os
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


def resolve_raster_opacity(pc, opacities_precomp=None):
    if opacities_precomp is None:
        return pc.get_opacity
    if int(opacities_precomp.shape[0]) != int(pc.get_xyz.shape[0]):
        raise ValueError("opacity override point count must match Gaussian point count")
    return opacities_precomp


def rasterize_gaussians(
    data,
    pc,
    pipe,
    bg_color,
    colors_precomp=None,
    scaling_modifier=1.0,
    return_opacity=False,
    opacities_precomp=None,
):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device='cuda') + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    rasterizer = _build_rasterizer(data, pipe, bg_color, pc, scaling_modifier)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = resolve_raster_opacity(pc, opacities_precomp)

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        covariance_mode = getattr(pipe, "covariance_mode", "default")
        anisotropy_clamp = getattr(pipe, "covariance_anisotropy_clamp", 0.0)
        isotropic_reduce = getattr(pipe, "covariance_isotropic_reduce", "geom")
        polar_det_min = getattr(pipe, "covariance_polar_det_min", 0.0)
        polar_det_max = getattr(pipe, "covariance_polar_det_max", 0.0)
        polar_det_power = getattr(pipe, "covariance_polar_det_power", 1.0)
        polar_anisotropy_clamp = getattr(pipe, "covariance_polar_anisotropy_clamp", 1.25)
        signed_point_json = getattr(pipe, "covariance_signed_point_json", "")
        signed_shrink_factor = getattr(pipe, "covariance_signed_shrink_factor", 1.0)
        signed_grow_factor = getattr(pipe, "covariance_signed_grow_factor", 1.0)
        signed_max_shrink_points = getattr(pipe, "covariance_signed_max_shrink_points", -1)
        signed_max_grow_points = getattr(pipe, "covariance_signed_max_grow_points", -1)
        signed_anisotropic_axis = getattr(pipe, "covariance_signed_anisotropic_axis", "all")
        signed_point_screen_actuator_enable = getattr(pipe, "covariance_signed_point_screen_actuator_enable", False)
        signed_point_screen_actuator_drop_images = getattr(pipe, "covariance_signed_point_screen_actuator_drop_images", "")
        signed_dynamic_enable = getattr(pipe, "covariance_signed_dynamic_enable", False)
        signed_dynamic_component_csv = getattr(pipe, "covariance_signed_dynamic_component_csv", "")
        signed_dynamic_point_csv = getattr(pipe, "covariance_signed_dynamic_point_csv", "")
        signed_dynamic_component_signature_enable = getattr(pipe, "covariance_signed_dynamic_component_signature_enable", False)
        signed_dynamic_over_layer_ids = getattr(pipe, "covariance_signed_dynamic_over_layer_ids", "soft,free")
        signed_dynamic_over_region_ids = getattr(pipe, "covariance_signed_dynamic_over_region_ids", "cloth")
        signed_dynamic_over_joint_ids = getattr(pipe, "covariance_signed_dynamic_over_joint_ids", "6,9,12,14,15")
        signed_dynamic_under_layer_ids = getattr(pipe, "covariance_signed_dynamic_under_layer_ids", "soft")
        signed_dynamic_under_region_ids = getattr(pipe, "covariance_signed_dynamic_under_region_ids", "body,cloth,soft")
        signed_dynamic_under_joint_ids = getattr(pipe, "covariance_signed_dynamic_under_joint_ids", "4,7,8")
        signed_dynamic_over_drop_images = getattr(pipe, "covariance_signed_dynamic_over_drop_images", "")
        signed_dynamic_under_drop_images = getattr(pipe, "covariance_signed_dynamic_under_drop_images", "")
        signed_dynamic_component_row_guard_json = getattr(pipe, "covariance_signed_dynamic_component_row_guard_json", "")
        signed_dynamic_component_local_asset_json = getattr(
            pipe, "covariance_signed_dynamic_component_local_asset_json", ""
        )
        signed_dynamic_boundary_min = getattr(pipe, "covariance_signed_dynamic_boundary_min", 0.0)
        signed_dynamic_surface_min = getattr(pipe, "covariance_signed_dynamic_surface_min", None)
        signed_dynamic_surface_max = getattr(pipe, "covariance_signed_dynamic_surface_max", None)
        signed_dynamic_component_pad_px = getattr(pipe, "covariance_signed_dynamic_component_pad_px", 10.0)
        signed_dynamic_component_ellipse_scale = getattr(pipe, "covariance_signed_dynamic_component_ellipse_scale", 1.25)
        signed_dynamic_component_max_over = getattr(pipe, "covariance_signed_dynamic_component_max_over", 16)
        signed_dynamic_component_max_under = getattr(pipe, "covariance_signed_dynamic_component_max_under", 16)
        signed_dynamic_component_min_area = getattr(pipe, "covariance_signed_dynamic_component_min_area", 1.0)
        signed_dynamic_component_required = getattr(pipe, "covariance_signed_dynamic_component_required", False)
        signed_dynamic_component_top_ids_enable = getattr(pipe, "covariance_signed_dynamic_component_top_ids_enable", False)
        signed_dynamic_component_top_ids_only = getattr(pipe, "covariance_signed_dynamic_component_top_ids_only", False)
        signed_dynamic_component_action_filter = getattr(pipe, "covariance_signed_dynamic_component_action_filter", "")
        signed_dynamic_component_action_required = getattr(
            pipe, "covariance_signed_dynamic_component_action_required", False
        )
        signed_dynamic_score_weighting_enable = getattr(pipe, "covariance_signed_dynamic_score_weighting_enable", False)
        signed_dynamic_score_weight_power = getattr(pipe, "covariance_signed_dynamic_score_weight_power", 1.0)
        signed_dynamic_score_weight_min = getattr(pipe, "covariance_signed_dynamic_score_weight_min", 0.0)
        signed_dynamic_score_weight_quantile = getattr(pipe, "covariance_signed_dynamic_score_weight_quantile", 0.90)
        signed_dynamic_max_over_points = getattr(pipe, "covariance_signed_dynamic_max_over_points", -1)
        signed_dynamic_max_under_points = getattr(pipe, "covariance_signed_dynamic_max_under_points", -1)
        signed_dynamic_guard_enable = getattr(pipe, "covariance_signed_dynamic_guard_enable", False)
        signed_dynamic_guard_shrink_mode = getattr(pipe, "covariance_signed_dynamic_guard_shrink_mode", "aniso_clamp")
        signed_dynamic_guard_grow_mode = getattr(pipe, "covariance_signed_dynamic_guard_grow_mode", "canonical_blend")
        signed_dynamic_guard_shrink_strength = getattr(pipe, "covariance_signed_dynamic_guard_shrink_strength", 0.0)
        signed_dynamic_guard_grow_strength = getattr(pipe, "covariance_signed_dynamic_guard_grow_strength", 0.0)
        signed_dynamic_guard_power = getattr(pipe, "covariance_signed_dynamic_guard_power", 1.0)
        signed_dynamic_guard_quantile = getattr(pipe, "covariance_signed_dynamic_guard_quantile", 0.90)
        signed_dynamic_guard_min_weight = getattr(pipe, "covariance_signed_dynamic_guard_min_weight", 0.0)
        signed_dynamic_guard_anisotropy_clamp = getattr(pipe, "covariance_signed_dynamic_guard_anisotropy_clamp", 1.20)
        signed_center_offset_enable = getattr(pipe, "covariance_signed_center_offset_enable", False)
        signed_center_offset_outer_px = getattr(pipe, "covariance_signed_center_offset_outer_px", 0.0)
        signed_center_offset_inner_px = getattr(pipe, "covariance_signed_center_offset_inner_px", 0.0)
        signed_center_offset_outer_direction = getattr(pipe, "covariance_signed_center_offset_outer_direction", "view_center")
        signed_center_offset_inner_direction = getattr(pipe, "covariance_signed_center_offset_inner_direction", "component_center")
        signed_center_offset_score_weight_power = getattr(pipe, "covariance_signed_center_offset_score_weight_power", 1.0)
        signed_center_offset_score_weight_min = getattr(pipe, "covariance_signed_center_offset_score_weight_min", 0.0)
        signed_center_offset_score_weight_quantile = getattr(pipe, "covariance_signed_center_offset_score_weight_quantile", 0.90)
        signed_center_offset_jacobian_eps = getattr(pipe, "covariance_signed_center_offset_jacobian_eps", 1.0e-3)
        signed_center_offset_jacobian_damping = getattr(pipe, "covariance_signed_center_offset_jacobian_damping", 1.0e-5)
        signed_center_offset_max_world_step = getattr(pipe, "covariance_signed_center_offset_max_world_step", 0.003)
        signed_virtual_grow_clone_enable = getattr(pipe, "covariance_signed_virtual_grow_clone_enable", False)
        signed_virtual_grow_clone_opacity_scale = getattr(pipe, "covariance_signed_virtual_grow_clone_opacity_scale", 0.35)
        signed_virtual_grow_clone_max_points = getattr(pipe, "covariance_signed_virtual_grow_clone_max_points", -1)
        signed_virtual_grow_clone_min_score = getattr(pipe, "covariance_signed_virtual_grow_clone_min_score", 0.0)
        signed_virtual_grow_clone_inner_px = getattr(pipe, "covariance_signed_virtual_grow_clone_inner_px", 0.0)
        signed_virtual_grow_clone_action_filter = getattr(
            pipe, "covariance_signed_virtual_grow_clone_action_filter", "virtual_grow_clone"
        )
        signed_virtual_grow_clone_drop_base_inner = getattr(
            pipe, "covariance_signed_virtual_grow_clone_drop_base_inner", True
        )
        signed_virtual_grow_clone_drop_base_inner_mode = str(
            getattr(pipe, "covariance_signed_virtual_grow_clone_drop_base_inner_mode", "image") or "image"
        ).strip().lower()
        signed_virtual_grow_clone_opacity_score_weighting_enable = getattr(
            pipe, "covariance_signed_virtual_grow_clone_opacity_score_weighting_enable", False
        )
        signed_virtual_grow_clone_opacity_score_weight_power = getattr(
            pipe, "covariance_signed_virtual_grow_clone_opacity_score_weight_power", 1.0
        )
        signed_virtual_grow_clone_opacity_score_weight_min = getattr(
            pipe, "covariance_signed_virtual_grow_clone_opacity_score_weight_min", 0.0
        )
        signed_virtual_grow_clone_opacity_score_weight_quantile = getattr(
            pipe, "covariance_signed_virtual_grow_clone_opacity_score_weight_quantile", 0.90
        )
        split_child_component_enable = getattr(pipe, "split_child_component_enable", False)
        split_child_component_asset_json = getattr(pipe, "split_child_component_asset_json", "")
        split_child_component_action_filter = getattr(pipe, "split_child_component_action_filter", "")
        split_child_component_action_required = getattr(pipe, "split_child_component_action_required", False)
        split_child_component_opacity = getattr(pipe, "split_child_component_opacity", 0.18)
        split_child_component_radius_scale = getattr(pipe, "split_child_component_radius_scale", 0.38)
        split_child_component_max_children = getattr(pipe, "split_child_component_max_children", -1)
        signed_screen_actuator_enable = getattr(pipe, "covariance_signed_screen_actuator_enable", False)
        signed_screen_normal_shrink_factor = getattr(pipe, "covariance_signed_screen_normal_shrink_factor", 1.0)
        signed_screen_normal_grow_factor = getattr(pipe, "covariance_signed_screen_normal_grow_factor", 1.0)
        signed_screen_tangent_factor = getattr(pipe, "covariance_signed_screen_tangent_factor", 1.0)
        boundary_cov_residual_enable = getattr(pipe, "boundary_cov_residual_enable", False)
        boundary_cov_residual_max_abs = getattr(pipe, "boundary_cov_residual_max_abs", 0.12)
        binding_covariance_guard_enable = getattr(pipe, "binding_covariance_guard_enable", False)
        binding_covariance_guard_mode = getattr(pipe, "binding_covariance_guard_mode", "canonical_blend")
        binding_covariance_guard_strength = getattr(pipe, "binding_covariance_guard_strength", 0.5)
        binding_covariance_guard_boundary_min = getattr(pipe, "binding_covariance_guard_boundary_min", 0.08)
        binding_covariance_guard_layer_ids = getattr(pipe, "binding_covariance_guard_layer_ids", "soft,free")
        binding_covariance_guard_region_ids = getattr(pipe, "binding_covariance_guard_region_ids", "cloth,soft")
        binding_covariance_guard_joint_ids = getattr(pipe, "binding_covariance_guard_joint_ids", "")
        binding_covariance_guard_thin_min = getattr(pipe, "binding_covariance_guard_thin_min", None)
        binding_covariance_guard_surface_min = getattr(pipe, "binding_covariance_guard_surface_min", None)
        binding_covariance_guard_surface_max = getattr(pipe, "binding_covariance_guard_surface_max", None)
        binding_covariance_guard_power = getattr(pipe, "binding_covariance_guard_power", 1.0)
        binding_covariance_guard_max_points = getattr(pipe, "binding_covariance_guard_max_points", -1)
        binding_covariance_guard_anisotropy_clamp = getattr(pipe, "binding_covariance_guard_anisotropy_clamp", 1.25)
        dynamic_under_drop_for_base = signed_dynamic_under_drop_images
        dynamic_under_action_exclude_filter_for_base = ""
        clone_drop_images = ""
        if (
            bool(signed_virtual_grow_clone_enable)
            and bool(signed_dynamic_enable)
            and bool(signed_virtual_grow_clone_drop_base_inner)
            and str(signed_dynamic_component_local_asset_json or "").strip()
        ):
            if signed_virtual_grow_clone_drop_base_inner_mode in ("row", "action", "component", "local"):
                dynamic_under_action_exclude_filter_for_base = signed_virtual_grow_clone_action_filter
            else:
                clone_drop_images = ",".join(
                    pc.component_local_asset_image_names(
                        signed_dynamic_component_local_asset_json,
                        direction="inner",
                        action_filter=signed_virtual_grow_clone_action_filter,
                    )
                )
                if clone_drop_images:
                    dynamic_under_drop_for_base = (
                        clone_drop_images
                        if not str(signed_dynamic_under_drop_images or "").strip()
                        else f"{signed_dynamic_under_drop_images},{clone_drop_images}"
                    )
        cov3D_precomp = pc.get_covariance(
            scaling_modifier,
            mode=covariance_mode,
            anisotropy_clamp=anisotropy_clamp,
            isotropic_reduce=isotropic_reduce,
            polar_det_min=polar_det_min,
            polar_det_max=polar_det_max,
            polar_det_power=polar_det_power,
            polar_anisotropy_clamp=polar_anisotropy_clamp,
            signed_point_json=signed_point_json,
            signed_shrink_factor=signed_shrink_factor,
            signed_grow_factor=signed_grow_factor,
            signed_max_shrink_points=signed_max_shrink_points,
            signed_max_grow_points=signed_max_grow_points,
            signed_anisotropic_axis=signed_anisotropic_axis,
            signed_point_screen_actuator_enable=signed_point_screen_actuator_enable,
            signed_point_screen_actuator_drop_images=signed_point_screen_actuator_drop_images,
            signed_dynamic_enable=signed_dynamic_enable,
            signed_dynamic_component_csv=signed_dynamic_component_csv,
            signed_dynamic_point_csv=signed_dynamic_point_csv,
            signed_dynamic_component_signature_enable=signed_dynamic_component_signature_enable,
            signed_dynamic_over_layer_ids=signed_dynamic_over_layer_ids,
            signed_dynamic_over_region_ids=signed_dynamic_over_region_ids,
            signed_dynamic_over_joint_ids=signed_dynamic_over_joint_ids,
            signed_dynamic_under_layer_ids=signed_dynamic_under_layer_ids,
            signed_dynamic_under_region_ids=signed_dynamic_under_region_ids,
            signed_dynamic_under_joint_ids=signed_dynamic_under_joint_ids,
            signed_dynamic_over_drop_images=signed_dynamic_over_drop_images,
            signed_dynamic_under_drop_images=dynamic_under_drop_for_base,
            signed_dynamic_component_row_guard_json=signed_dynamic_component_row_guard_json,
            signed_dynamic_component_local_asset_json=signed_dynamic_component_local_asset_json,
            signed_dynamic_boundary_min=signed_dynamic_boundary_min,
            signed_dynamic_surface_min=signed_dynamic_surface_min,
            signed_dynamic_surface_max=signed_dynamic_surface_max,
            signed_dynamic_component_pad_px=signed_dynamic_component_pad_px,
            signed_dynamic_component_ellipse_scale=signed_dynamic_component_ellipse_scale,
            signed_dynamic_component_max_over=signed_dynamic_component_max_over,
            signed_dynamic_component_max_under=signed_dynamic_component_max_under,
            signed_dynamic_component_min_area=signed_dynamic_component_min_area,
            signed_dynamic_component_required=signed_dynamic_component_required,
            signed_dynamic_component_top_ids_enable=signed_dynamic_component_top_ids_enable,
            signed_dynamic_component_top_ids_only=signed_dynamic_component_top_ids_only,
            signed_dynamic_component_action_filter=signed_dynamic_component_action_filter,
            signed_dynamic_component_action_exclude_filter=dynamic_under_action_exclude_filter_for_base,
            signed_dynamic_component_action_required=signed_dynamic_component_action_required,
            signed_dynamic_score_weighting_enable=signed_dynamic_score_weighting_enable,
            signed_dynamic_score_weight_power=signed_dynamic_score_weight_power,
            signed_dynamic_score_weight_min=signed_dynamic_score_weight_min,
            signed_dynamic_score_weight_quantile=signed_dynamic_score_weight_quantile,
            signed_dynamic_max_over_points=signed_dynamic_max_over_points,
            signed_dynamic_max_under_points=signed_dynamic_max_under_points,
            signed_dynamic_guard_enable=signed_dynamic_guard_enable,
            signed_dynamic_guard_shrink_mode=signed_dynamic_guard_shrink_mode,
            signed_dynamic_guard_grow_mode=signed_dynamic_guard_grow_mode,
            signed_dynamic_guard_shrink_strength=signed_dynamic_guard_shrink_strength,
            signed_dynamic_guard_grow_strength=signed_dynamic_guard_grow_strength,
            signed_dynamic_guard_power=signed_dynamic_guard_power,
            signed_dynamic_guard_quantile=signed_dynamic_guard_quantile,
            signed_dynamic_guard_min_weight=signed_dynamic_guard_min_weight,
            signed_dynamic_guard_anisotropy_clamp=signed_dynamic_guard_anisotropy_clamp,
            signed_screen_actuator_enable=signed_screen_actuator_enable,
            signed_screen_normal_shrink_factor=signed_screen_normal_shrink_factor,
            signed_screen_normal_grow_factor=signed_screen_normal_grow_factor,
            signed_screen_tangent_factor=signed_screen_tangent_factor,
            boundary_cov_residual_enable=boundary_cov_residual_enable,
            boundary_cov_residual_max_abs=boundary_cov_residual_max_abs,
            binding_covariance_guard_enable=binding_covariance_guard_enable,
            binding_covariance_guard_mode=binding_covariance_guard_mode,
            binding_covariance_guard_strength=binding_covariance_guard_strength,
            binding_covariance_guard_boundary_min=binding_covariance_guard_boundary_min,
            binding_covariance_guard_layer_ids=binding_covariance_guard_layer_ids,
            binding_covariance_guard_region_ids=binding_covariance_guard_region_ids,
            binding_covariance_guard_joint_ids=binding_covariance_guard_joint_ids,
            binding_covariance_guard_thin_min=binding_covariance_guard_thin_min,
            binding_covariance_guard_surface_min=binding_covariance_guard_surface_min,
            binding_covariance_guard_surface_max=binding_covariance_guard_surface_max,
            binding_covariance_guard_power=binding_covariance_guard_power,
            binding_covariance_guard_max_points=binding_covariance_guard_max_points,
            binding_covariance_guard_anisotropy_clamp=binding_covariance_guard_anisotropy_clamp,
            camera=data,
        )
        if bool(signed_center_offset_enable):
            means3D = means3D + pc.get_signed_center_offset(
                camera=data,
                signed_point_json=signed_point_json,
                signed_dynamic_enable=signed_dynamic_enable,
                signed_dynamic_component_csv=signed_dynamic_component_csv,
                signed_dynamic_point_csv=signed_dynamic_point_csv,
                signed_dynamic_component_signature_enable=signed_dynamic_component_signature_enable,
                signed_dynamic_over_layer_ids=signed_dynamic_over_layer_ids,
                signed_dynamic_over_region_ids=signed_dynamic_over_region_ids,
                signed_dynamic_over_joint_ids=signed_dynamic_over_joint_ids,
                signed_dynamic_under_layer_ids=signed_dynamic_under_layer_ids,
                signed_dynamic_under_region_ids=signed_dynamic_under_region_ids,
                signed_dynamic_under_joint_ids=signed_dynamic_under_joint_ids,
                signed_dynamic_over_drop_images=signed_dynamic_over_drop_images,
                signed_dynamic_under_drop_images=dynamic_under_drop_for_base,
                signed_dynamic_component_row_guard_json=signed_dynamic_component_row_guard_json,
                signed_dynamic_component_local_asset_json=signed_dynamic_component_local_asset_json,
                signed_dynamic_boundary_min=signed_dynamic_boundary_min,
                signed_dynamic_surface_min=signed_dynamic_surface_min,
                signed_dynamic_surface_max=signed_dynamic_surface_max,
                signed_dynamic_component_pad_px=signed_dynamic_component_pad_px,
                signed_dynamic_component_ellipse_scale=signed_dynamic_component_ellipse_scale,
                signed_dynamic_component_max_over=signed_dynamic_component_max_over,
                signed_dynamic_component_max_under=signed_dynamic_component_max_under,
                signed_dynamic_component_min_area=signed_dynamic_component_min_area,
                signed_dynamic_component_required=signed_dynamic_component_required,
                signed_dynamic_component_top_ids_enable=signed_dynamic_component_top_ids_enable,
                signed_dynamic_component_top_ids_only=signed_dynamic_component_top_ids_only,
                signed_dynamic_max_over_points=signed_dynamic_max_over_points,
                signed_dynamic_max_under_points=signed_dynamic_max_under_points,
                signed_max_shrink_points=signed_max_shrink_points,
                signed_max_grow_points=signed_max_grow_points,
                outer_offset_px=signed_center_offset_outer_px,
                inner_offset_px=signed_center_offset_inner_px,
                outer_direction=signed_center_offset_outer_direction,
                inner_direction=signed_center_offset_inner_direction,
                score_weight_power=signed_center_offset_score_weight_power,
                score_weight_min=signed_center_offset_score_weight_min,
                score_weight_quantile=signed_center_offset_score_weight_quantile,
                jacobian_eps=signed_center_offset_jacobian_eps,
                jacobian_damping=signed_center_offset_jacobian_damping,
                max_world_step=signed_center_offset_max_world_step,
            )
        if bool(signed_virtual_grow_clone_enable) and bool(signed_dynamic_enable):
            clone_pkg = pc.build_signed_virtual_grow_clone_tensors(
                camera=data,
                colors_precomp=colors_precomp,
                scaling_modifier=scaling_modifier,
                mode=covariance_mode,
                anisotropy_clamp=anisotropy_clamp,
                isotropic_reduce=isotropic_reduce,
                polar_det_min=polar_det_min,
                polar_det_max=polar_det_max,
                polar_det_power=polar_det_power,
                polar_anisotropy_clamp=polar_anisotropy_clamp,
                signed_dynamic_component_csv=signed_dynamic_component_csv,
                signed_dynamic_point_csv=signed_dynamic_point_csv,
                signed_dynamic_component_signature_enable=signed_dynamic_component_signature_enable,
                signed_dynamic_over_layer_ids=signed_dynamic_over_layer_ids,
                signed_dynamic_over_region_ids=signed_dynamic_over_region_ids,
                signed_dynamic_over_joint_ids=signed_dynamic_over_joint_ids,
                signed_dynamic_under_layer_ids=signed_dynamic_under_layer_ids,
                signed_dynamic_under_region_ids=signed_dynamic_under_region_ids,
                signed_dynamic_under_joint_ids=signed_dynamic_under_joint_ids,
                signed_dynamic_over_drop_images=signed_dynamic_over_drop_images,
                signed_dynamic_under_drop_images=signed_dynamic_under_drop_images,
                signed_dynamic_component_row_guard_json=signed_dynamic_component_row_guard_json,
                signed_dynamic_component_local_asset_json=signed_dynamic_component_local_asset_json,
                signed_dynamic_boundary_min=signed_dynamic_boundary_min,
                signed_dynamic_surface_min=signed_dynamic_surface_min,
                signed_dynamic_surface_max=signed_dynamic_surface_max,
                signed_dynamic_component_pad_px=signed_dynamic_component_pad_px,
                signed_dynamic_component_ellipse_scale=signed_dynamic_component_ellipse_scale,
                signed_dynamic_component_max_over=signed_dynamic_component_max_over,
                signed_dynamic_component_max_under=signed_dynamic_component_max_under,
                signed_dynamic_component_min_area=signed_dynamic_component_min_area,
                signed_dynamic_component_required=signed_dynamic_component_required,
                signed_dynamic_component_top_ids_enable=signed_dynamic_component_top_ids_enable,
                signed_dynamic_component_top_ids_only=signed_dynamic_component_top_ids_only,
                signed_dynamic_component_action_filter=signed_virtual_grow_clone_action_filter,
                signed_dynamic_component_action_required=True,
                signed_dynamic_max_under_points=signed_dynamic_max_under_points,
                signed_dynamic_score_weighting_enable=signed_dynamic_score_weighting_enable,
                signed_dynamic_score_weight_power=signed_dynamic_score_weight_power,
                signed_dynamic_score_weight_min=signed_dynamic_score_weight_min,
                signed_dynamic_score_weight_quantile=signed_dynamic_score_weight_quantile,
                signed_screen_normal_grow_factor=signed_screen_normal_grow_factor,
                signed_screen_tangent_factor=signed_screen_tangent_factor,
                signed_center_offset_inner_px=signed_virtual_grow_clone_inner_px,
                signed_center_offset_inner_direction=signed_center_offset_inner_direction,
                signed_center_offset_score_weight_power=signed_center_offset_score_weight_power,
                signed_center_offset_score_weight_min=signed_center_offset_score_weight_min,
                signed_center_offset_score_weight_quantile=signed_center_offset_score_weight_quantile,
                signed_center_offset_jacobian_eps=signed_center_offset_jacobian_eps,
                signed_center_offset_jacobian_damping=signed_center_offset_jacobian_damping,
                signed_center_offset_max_world_step=signed_center_offset_max_world_step,
                opacity_scale=signed_virtual_grow_clone_opacity_scale,
                opacity_score_weighting_enable=signed_virtual_grow_clone_opacity_score_weighting_enable,
                opacity_score_weight_power=signed_virtual_grow_clone_opacity_score_weight_power,
                opacity_score_weight_min=signed_virtual_grow_clone_opacity_score_weight_min,
                opacity_score_weight_quantile=signed_virtual_grow_clone_opacity_score_weight_quantile,
                max_clones=signed_virtual_grow_clone_max_points,
                min_score=signed_virtual_grow_clone_min_score,
            )
            if clone_pkg is not None:
                means3D = torch.cat((means3D, clone_pkg["means3D"]), dim=0)
                opacity = torch.cat((opacity, clone_pkg["opacities"]), dim=0)
                cov3D_precomp = torch.cat((cov3D_precomp, clone_pkg["cov3D_precomp"]), dim=0)
                if colors_precomp is not None and clone_pkg.get("colors_precomp", None) is not None:
                    colors_precomp = torch.cat((colors_precomp, clone_pkg["colors_precomp"]), dim=0)
        if bool(split_child_component_enable) and str(split_child_component_asset_json or "").strip():
            child_pkg = pc.build_split_child_component_tensors(
                camera=data,
                colors_precomp=colors_precomp,
                asset_json=split_child_component_asset_json,
                action_filter=split_child_component_action_filter,
                action_required=split_child_component_action_required,
                opacity=split_child_component_opacity,
                radius_scale=split_child_component_radius_scale,
                max_children=split_child_component_max_children,
                activation_component_csv=signed_dynamic_component_csv,
                activation_point_csv=signed_dynamic_point_csv,
            )
            if child_pkg is not None:
                child_start = int(means3D.shape[0])
                means3D = torch.cat((means3D, child_pkg["means3D"]), dim=0)
                opacity = torch.cat((opacity, child_pkg["opacities"]), dim=0)
                cov3D_precomp = torch.cat((cov3D_precomp, child_pkg["cov3D_precomp"]), dim=0)
                if colors_precomp is not None and child_pkg.get("colors_precomp", None) is not None:
                    colors_precomp = torch.cat((colors_precomp, child_pkg["colors_precomp"]), dim=0)
                if str(os.environ.get("STAGEB_SPLIT_CHILD_DEBUG", "") or "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                ):
                    print(
                        "[split_child_debug] "
                        f"renderer_appended={int(child_pkg['means3D'].shape[0])} "
                        f"base_count={child_start} total_count={int(means3D.shape[0])}",
                        flush=True,
                    )
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    if means2D.shape[0] != means3D.shape[0]:
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=means3D.device) + 0
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass
        means2D = screenspace_points

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
    if str(os.environ.get("STAGEB_SPLIT_CHILD_DEBUG", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        base_count = int(pc.get_xyz.shape[0])
        if int(radii.shape[0]) > base_count:
            child_radii = radii[base_count:]
            print(
                "[split_child_debug] "
                f"radii_child_count={int(child_radii.shape[0])} "
                f"radii_positive={int((child_radii > 0).sum().item())} "
                f"radii_max={float(child_radii.float().max().item()) if child_radii.numel() else 0.0}",
                flush=True,
            )
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
