#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RUN_ID="${RUN_ID:-$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
BASE_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v187b_v179c_heldout_camera_geometry_perceptual_20260508_v187b_heldout_geom_perc"
EXP_DIR="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v188a_v187b_geom_perceptual_texture_restore_$RUN_ID}"
START_CKPT="${2:-$BASE_EXP/best_ckpt.pth}"
ITERATIONS="${3:-1600}"

if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then shift; fi
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then shift; fi
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then shift; fi

mkdir -p "$EXP_DIR"
cd "$ROOT"

exec env PYTHONUNBUFFERED=1 "$PYTHON_BIN" train.py \
  --config-path "$BASE_EXP/.hydra" \
  --config-name config \
  mode=train \
  "dataset.root_dir=$ROOT/data/ZJUMoCap" \
  "dataset.preload=false" \
  "start_checkpoint=$START_CKPT" \
  "exp_dir=$EXP_DIR" \
  "wandb_disable=true" \
  "++resume.restore_converter_optimizer_state=false" \
  "++resume.restore_converter_scheduler_state=false" \
  "++resume.partial_converter_missing_keys_allow_patterns=[structured_trunk_output_head_local_color_owner_head_boundary]" \
  "opt.iterations=$ITERATIONS" \
  "pipeline.pose_noise=0.0" \
  "opt.position_lr_init=0.0" \
  "opt.position_lr_final=0.0" \
  "opt.feature_lr=0.0" \
  "opt.opacity_lr=0.0" \
  "opt.scaling_lr=0.0" \
  "opt.rotation_lr=0.0" \
  "opt.rigid_lr=0.0" \
  "opt.non_rigid_lr=0.0" \
  "opt.nr_latent_lr=0.0" \
  "opt.pose_correction_lr=0.0" \
  "opt.texture_lr=1.8e-05" \
  "opt.tex_latent_lr=0.0" \
  "++opt.camera_geometry_lr=8.0e-05" \
  "++opt.camera_geometry_strength=0.85" \
  "++opt.lambda_camera_geometry_reg=0.020" \
  "++opt.camera_affine_enable=false" \
  "++opt.camera_affine_lr=0.0" \
  "++opt.boundary_opacity_residual_lr=0.0" \
  "++opt.boundary_scaling_residual_lr=0.0" \
  "++opt.latent_weight_decay=0.0" \
  "opt.lambda_l1=0.030" \
  "++opt.lambda_l1_fg=0.10" \
  "opt.lambda_perceptual=0.26" \
  "opt.lambda_dssim=0.0" \
  "++opt.perceptual_masked=true" \
  "++opt.perceptual_mask_dilate=3" \
  "++opt.perceptual_crop_pad=20" \
  "++opt.lambda_l1_boundary=0.08" \
  "opt.lambda_mask=0.0" \
  "++opt.lambda_mask_boundary=0.006" \
  "++opt.lambda_mask_boundary_hard=0.010" \
  "++opt.lambda_silhouette_outer=0.006" \
  "++opt.lambda_silhouette_outer_shell=0.008" \
  "++opt.lambda_silhouette_inner=0.004" \
  "++opt.lambda_edge_face=0.0" \
  "++opt.lambda_edge_shoulder_arm=0.0" \
  "++opt.lambda_edge_waist=0.0" \
  "++opt.lambda_detail_face=0.0" \
  "++opt.lambda_detail_shoulder_arm=0.0" \
  "++opt.lambda_detail_waist=0.0" \
  "++opt.lambda_detail_face_luma_dog=0.0" \
  "++opt.lambda_detail_shoulder_arm_luma_dog=0.0" \
  "++opt.lambda_detail_waist_luma_dog=0.0" \
  "++opt.lambda_detail_upper_torso_luma_dog=0.0" \
  "++opt.lambda_detail_upper_torso_core_luma_dog=0.0" \
  "++opt.lambda_perceptual_face=0.12" \
  "++opt.lambda_perceptual_shoulder_arm=0.05" \
  "++opt.lambda_perceptual_waist=0.035" \
  "++opt.lambda_perceptual_face_patch=0.08" \
  "++opt.lambda_perceptual_shoulder_arm_patch=0.045" \
  "++opt.lambda_perceptual_waist_patch=0.030" \
  "++opt.lambda_perceptual_upper_torso_patch=0.045" \
  "++opt.lambda_perceptual_upper_torso_core_patch=0.045" \
  "++opt.reliable_view_highfreq_enable=true" \
  "++opt.reliable_view_camera_ids=[21,22,23]" \
  "++opt.reliable_view_apply_patch_perceptual=true" \
  "++opt.reliable_view_apply_detail_luma_dog=false" \
  "++opt.reliable_view_highfreq_min_weight=1.0" \
  "++opt.reliable_view_highfreq_max_weight=1.25" \
  "opt.grad_clip=0.015" \
  "++opt.train_sample_camera_min_prob=0.25" \
  "++opt.train_sample_camera_max_prob=0.40" \
  "++opt.photometric_contour_debug_interval=200" \
  "test_interval=400" \
  "test_iterations=[400,800,1200,1600]" \
  "save_iterations=[800,1600]" \
  "checkpoint_iterations=[400,800,1200,1600]" \
  "++validation_image_log_limit=0" \
  "$@"
