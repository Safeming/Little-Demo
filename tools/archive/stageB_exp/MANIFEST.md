# StageB 实验入口归档索引

本文件只做入口分层记录，暂不物理移动历史脚本。

正式 377 no-train signed geometry 路径请使用：

```text
tools/run_377_formal_signed_geometry_render.sh
tools/run_377_formal_signed_geometry_export.sh
```

下面这些脚本属于 StageB / explicit binding 历史实验入口，不作为正式默认入口。后续清理时可按本索引迁移到 `tools/archive/stageB_exp/` 或删除明确 rejected 的分支。

脚本数量：121

## 历史实验脚本

- `tools/append_377_stageB_v262_ray_carve_support.py`
- `tools/append_377_stageB_v266_micro_patch_support.py`
- `tools/append_377_stageB_v277_verified_support.py`
- `tools/audit_377_stageB_v274_contributors.py`
- `tools/audit_377_stageB_v312_geometry_mismatch.py`
- `tools/launch_377_stageB_v243_boundary_residual_serial.sh`
- `tools/launch_377_stageB_v245_boundary_support_projector_serial.sh`
- `tools/launch_377_stageB_v246_boundary_footprint_projector_serial.sh`
- `tools/launch_377_stageB_v247_boundary_contributor_projector_serial.sh`
- `tools/launch_377_stageB_v248_boundary_component_support_serial.sh`
- `tools/launch_377_stageB_v249_boundary_component_multiview_support_serial.sh`
- `tools/launch_377_stageB_v250_gapaware_component_support_9h_serial.sh`
- `tools/launch_377_stageB_v251_projector_order_outer_shrink_serial.sh`
- `tools/launch_377_stageB_v252_under_protected_outer_shrink_serial.sh`
- `tools/launch_377_stageB_v253_support_activation_probe_serial.sh`
- `tools/launch_377_stageB_v254_diagnostic_queue_serial.sh`
- `tools/launch_377_stageB_v255_support_audit_queue_serial.sh`
- `tools/launch_377_stageB_v256_target_project_component_support_serial.sh`
- `tools/launch_377_stageB_v257_parent_consensus_target_support_serial.sh`
- `tools/launch_377_stageB_v258_multiview_gain_gated_support_serial.sh`
- `tools/launch_377_stageB_v259_clustered_multiview_support_serial.sh`
- `tools/launch_377_stageB_v260_candidate_multiview_support_serial.sh`
- `tools/launch_377_stageB_v261_candidate_validator_serial.sh`
- `tools/launch_377_stageB_v262_ray_carve_validator_serial.sh`
- `tools/make_377_stageB_v226_cleanup_ckpts.py`
- `tools/make_377_stageB_v268_shifted_3d_candidates.py`
- `tools/make_377_stageB_v269_visual_hull_candidates.py`
- `tools/make_377_stageB_v275_local_signed_checkpoint.py`
- `tools/make_377_stageB_v276_directional_checkpoint.py`
- `tools/make_377_stageB_v288_boundary_cov_checkpoint.py`
- `tools/make_377_stageB_v311_projected_component_prior.py`
- `tools/make_377_stageB_v325_view_signed_point_prior.py`
- `tools/make_377_stageB_v329_target_raw_inner_prior.py`
- `tools/make_377_stageB_v330_target_raw_component_csv.py`
- `tools/run_377_explicit_binding_v271_color_texture_only.sh`
- `tools/run_377_explicit_binding_v272_rotation_orth_ab.sh`
- `tools/run_377_explicit_binding_v273_footprint_factor_probe.sh`
- `tools/run_377_explicit_binding_v279_covariance_footprint_ab.sh`
- `tools/run_377_explicit_binding_v280_signed_covariance_actuator.sh`
- `tools/run_377_explicit_binding_v281_dynamic_screen_covariance_actuator.sh`
- `tools/run_377_explicit_binding_v282_teacher_distill_refine.sh`
- `tools/run_377_explicit_binding_v283_opacity_footprint_audit.sh`
- `tools/run_377_explicit_binding_v284_signed_footprint_residual_refine.sh`
- `tools/run_377_explicit_binding_v285_opacity_gated_rgb_refine.sh`
- `tools/run_377_explicit_binding_v286_raw_support_hinge_segment_gate.sh`
- `tools/run_377_explicit_binding_v287_deformer_factor_probe.sh`
- `tools/run_377_explicit_binding_v288_boundary_cov_residual.sh`
- `tools/run_377_explicit_binding_v289_train_component_cov_residual.sh`
- `tools/run_377_explicit_binding_v290_component_topid_cov_residual.sh`
- `tools/run_377_explicit_binding_v291_signed2_component_cov_residual.sh`
- `tools/run_377_explicit_binding_v292_score_weighted_actuator_ab.sh`
- `tools/run_377_explicit_binding_v293_binding_covariance_guard_ab.sh`
- `tools/run_377_explicit_binding_v294_signed_dynamic_guard_ab.sh`
- `tools/run_377_explicit_binding_v295_component_center_offset_refine.sh`
- `tools/run_377_explicit_binding_v296_checkpoint_consistent_center_offset_refine.sh`
- `tools/run_377_explicit_binding_v297_teacher_raw_support_guard_refine.sh`
- `tools/run_377_explicit_binding_v298_boundary_protected_feature_refine.sh`
- `tools/run_377_explicit_binding_v299_screen_space_protected_feature_refine.sh`
- `tools/run_377_explicit_binding_v300_polar_covariance_stabilizer_ab.sh`
- `tools/run_377_explicit_binding_v301_geometry_fidelity_gate_ab.sh`
- `tools/run_377_explicit_binding_v302_component_geometry_fidelity_ab.sh`
- `tools/run_377_explicit_binding_v303_component_geometry_guarded_refine.sh`
- `tools/run_377_explicit_binding_v304_consistent_component_geometry_refine.sh`
- `tools/run_377_explicit_binding_v305_boundary_safe_sh_refine.sh`
- `tools/run_377_explicit_binding_v306_adopted_geometry_render.sh`
- `tools/run_377_explicit_binding_v307_adopted_preset_render.sh`
- `tools/run_377_explicit_binding_v308_binding_internal_ab.sh`
- `tools/run_377_explicit_binding_v309_point_prior_ab.sh`
- `tools/run_377_explicit_binding_v310_point_shrink_hybrid_ab.sh`
- `tools/run_377_explicit_binding_v311_projected_component_ab.sh`
- `tools/run_377_explicit_binding_v312_geometry_mismatch_audit.sh`
- `tools/run_377_explicit_binding_v313_view_conditioned_xbar_teacher.sh`
- `tools/run_377_explicit_binding_v314_internal_geometry_gate_ab.sh`
- `tools/run_377_explicit_binding_v315_layer_logits_calibration.sh`
- `tools/run_377_explicit_binding_v316_identity_preserve.sh`
- `tools/run_377_explicit_binding_v317_pose_render_factor_ab.sh`
- `tools/run_377_explicit_binding_v318_component_greedy_selector.py`
- `tools/run_377_explicit_binding_v318_component_greedy_selector.sh`
- `tools/run_377_explicit_binding_v319_component_greedy_fullframes.sh`
- `tools/run_377_explicit_binding_v320_paired_signed_selector.py`
- `tools/run_377_explicit_binding_v320_paired_signed_selector.sh`
- `tools/run_377_explicit_binding_v321_teacher_locked_paired_signed_train.sh`
- `tools/run_377_explicit_binding_v322_boundary_color_locked_paired_signed_train.sh`
- `tools/run_377_explicit_binding_v323_static_contributor_locked_train.sh`
- `tools/run_377_explicit_binding_v324_interior_polish_signed_train.sh`
- `tools/run_377_explicit_binding_v325_view_signed_point_prior_ab.sh`
- `tools/run_377_explicit_binding_v326_v328_signed_point_fix_ab.sh`
- `tools/run_377_explicit_binding_v329_target_raw_inner_gate_ab.sh`
- `tools/run_377_explicit_binding_v330_target_raw_component_adopted_ab.sh`
- `tools/run_377_explicit_binding_v331_target_raw_component_selector_train.sh`
- `tools/run_377_explicit_binding_v332_formal_signed_geometry_render.sh`
- `tools/run_377_explicit_binding_v333_full_formal_signed_geometry_gate.sh`
- `tools/run_377_stageB_v221_rootcause_4gpu.sh`
- `tools/run_377_stageB_v226_gaussian_cleanup_4gpu.sh`
- `tools/run_377_stageB_v234_opacity_silhouette_refine_probe.sh`
- `tools/run_377_stageB_v235_silhouette_support_refine.sh`
- `tools/run_377_stageB_v236_compact_semantic_localfix_2gpu.sh`
- `tools/run_377_stageB_v237_v238_boundary_support_4gpu.sh`
- `tools/run_377_stageB_v239_v240_boundary_repair_3gpu.sh`
- `tools/run_377_stageB_v243_boundary_residual_serial.sh`
- `tools/run_377_stageB_v245_boundary_support_projector_serial.sh`
- `tools/run_377_stageB_v246_boundary_footprint_projector_serial.sh`
- `tools/run_377_stageB_v247_boundary_contributor_projector_serial.sh`
- `tools/run_377_stageB_v248_boundary_component_support_serial.sh`
- `tools/run_377_stageB_v261_candidate_validator_serial.sh`
- `tools/run_377_stageB_v262_ray_carve_validator_serial.sh`
- `tools/run_377_stageB_v265_verified_support_only_serial.sh`
- `tools/run_377_stageB_v265b_forced_support_only_train_serial.sh`
- `tools/run_377_stageB_v266_micro_patch_support_train_serial.sh`
- `tools/run_377_stageB_v267_setlevel_renderloop_train_serial.sh`
- `tools/run_377_stageB_v268_renderloop_3d_search_train_serial.sh`
- `tools/run_377_stageB_v269_visual_hull_renderloop_train_serial.sh`
- `tools/run_377_stageB_v274_contributor_audit.sh`
- `tools/run_377_stageB_v275_local_signed_ab.sh`
- `tools/run_377_stageB_v276_component_directional_ab.sh`
- `tools/run_377_stageB_v277_verified_support_ab.sh`
- `tools/run_377_stageB_v278_renderloop_selector.sh`
- `tools/validate_377_stageB_v261_support_candidates.py`
- `tools/validate_377_stageB_v262_ray_carve_candidates.py`
- `tools/validate_377_stageB_v263_footprint_candidates.py`
- `tools/validate_377_stageB_v264_actual_radii_candidates.py`
