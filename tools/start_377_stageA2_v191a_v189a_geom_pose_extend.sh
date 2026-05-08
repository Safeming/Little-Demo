#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RUN_ID="${RUN_ID:-$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
BASE_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v189a_v188a_geom_pose_probe_20260508_v189a_geom_pose_probe"
EXP_DIR="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v191a_v189a_geom_pose_extend_$RUN_ID}"
START_CKPT="${2:-$BASE_EXP/best_ckpt.pth}"
ITERATIONS="${3:-1200}"

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
  "dataset.train_views=[21,22,23]" \
  "dataset.val_views=[21,22,23]" \
  "dataset.test_views.view=[21,22,23]" \
  "dataset.train_frames=[0,570,1]" \
  "dataset.val_frames=[0,570,30]" \
  "dataset.test_frames.view=[0,570,30]" \
  "start_checkpoint=$START_CKPT" \
  "exp_dir=$EXP_DIR" \
  "wandb_disable=true" \
  "++resume.restore_converter_optimizer_state=false" \
  "++resume.restore_converter_scheduler_state=false" \
  "++resume.partial_converter_missing_keys_allow_patterns=[]" \
  "opt.iterations=$ITERATIONS" \
  "pipeline.pose_noise=0.0" \
  "model.pose_correction.delay=1" \
  "++model.pose_correction.train_root_orient=false" \
  "++model.pose_correction.train_pose_body=true" \
  "++model.pose_correction.train_pose_hand=false" \
  "++model.pose_correction.train_trans=false" \
  "++model.pose_correction.train_betas=false" \
  "++model.pose_correction.pose_body_train_joint_ids=[12,13,14,15,16,17]" \
  "opt.position_lr_init=0.0" \
  "opt.position_lr_final=0.0" \
  "opt.feature_lr=0.0" \
  "opt.opacity_lr=0.0" \
  "opt.scaling_lr=0.0" \
  "opt.rotation_lr=0.0" \
  "opt.rigid_lr=0.0" \
  "opt.non_rigid_lr=0.0" \
  "opt.nr_latent_lr=0.0" \
  "opt.pose_correction_lr=4.0e-07" \
  "opt.texture_lr=0.0" \
  "opt.tex_latent_lr=0.0" \
  "++opt.camera_geometry_lr=3.0e-05" \
  "++opt.camera_geometry_strength=0.78" \
  "++opt.camera_geometry_rot_max_deg=0.28" \
  "++opt.camera_geometry_trans_max=0.006" \
  "++opt.lambda_camera_geometry_reg=0.030" \
  "++opt.camera_affine_enable=false" \
  "++opt.camera_affine_lr=0.0" \
  "++opt.boundary_opacity_residual_lr=0.0" \
  "++opt.boundary_scaling_residual_lr=0.0" \
  "++opt.latent_weight_decay=0.0" \
  "opt.lambda_pose=0.30" \
  "opt.lambda_l1=0.026" \
  "++opt.lambda_l1_fg=0.085" \
  "opt.lambda_perceptual=0.31" \
  "opt.lambda_dssim=0.0" \
  "++opt.perceptual_masked=true" \
  "++opt.perceptual_mask_dilate=3" \
  "++opt.perceptual_crop_pad=20" \
  "++opt.lambda_l1_boundary=0.070" \
  "opt.lambda_mask=0.0" \
  "++opt.lambda_mask_boundary=0.005" \
  "++opt.lambda_mask_boundary_hard=0.007" \
  "++opt.lambda_silhouette_outer=0.005" \
  "++opt.lambda_silhouette_outer_shell=0.007" \
  "++opt.lambda_silhouette_inner=0.003" \
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
  "++opt.lambda_perceptual_face=0.13" \
  "++opt.lambda_perceptual_shoulder_arm=0.055" \
  "++opt.lambda_perceptual_waist=0.035" \
  "++opt.lambda_perceptual_face_patch=0.09" \
  "++opt.lambda_perceptual_shoulder_arm_patch=0.050" \
  "++opt.lambda_perceptual_waist_patch=0.030" \
  "++opt.lambda_perceptual_upper_torso_patch=0.050" \
  "++opt.lambda_perceptual_upper_torso_core_patch=0.050" \
  "++opt.reliable_view_highfreq_enable=true" \
  "++opt.reliable_view_camera_ids=[21,22,23]" \
  "++opt.reliable_view_apply_patch_perceptual=true" \
  "++opt.reliable_view_apply_detail_luma_dog=false" \
  "++opt.reliable_view_highfreq_min_weight=1.0" \
  "++opt.reliable_view_highfreq_max_weight=1.16" \
  "opt.grad_clip=0.008" \
  "++opt.train_sample_camera_min_prob=0.25" \
  "++opt.train_sample_camera_max_prob=0.40" \
  "++opt.photometric_contour_debug_interval=200" \
  "test_interval=300" \
  "test_iterations=[300,600,900,1200]" \
  "save_iterations=[600,1200]" \
  "checkpoint_iterations=[300,600,900,1200]" \
  "++validation_image_log_limit=0" \
  "$@"
