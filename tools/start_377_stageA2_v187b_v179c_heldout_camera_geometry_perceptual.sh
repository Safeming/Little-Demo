#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RUN_ID="${RUN_ID:-$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
BASE_EXP="$ROOT/exp/377_multiview_explicit_hq_rootfix_resume_v179c_v178c_k6_camquality_tightphoto_screen3000"
EXP_DIR="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v187b_v179c_heldout_camera_geometry_perceptual_$RUN_ID}"
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
  "pipeline.pose_noise=0.0" \
  "start_checkpoint=$START_CKPT" \
  "exp_dir=$EXP_DIR" \
  "wandb_disable=true" \
  "++resume.restore_converter_optimizer_state=false" \
  "++resume.restore_converter_scheduler_state=false" \
  "++resume.partial_converter_missing_keys_allow_patterns=[camera_geometry.]" \
  "opt.iterations=$ITERATIONS" \
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
  "opt.texture_lr=0.0" \
  "opt.tex_latent_lr=0.0" \
  "++opt.camera_affine_enable=false" \
  "++opt.camera_affine_lr=0.0" \
  "++opt.camera_geometry_enable=true" \
  "++opt.camera_geometry_train_camera_ids=[21,22,23]" \
  "++opt.camera_geometry_max_camera_id=23" \
  "++opt.camera_geometry_strength=[0.15,1,0.45,250,0.70,700,0.85]" \
  "++opt.camera_geometry_rot_max_deg=0.30" \
  "++opt.camera_geometry_trans_max=0.007" \
  "++opt.camera_geometry_rot_reg_weight=1.0" \
  "++opt.camera_geometry_trans_reg_weight=1.0" \
  "++opt.camera_geometry_lr=2.0e-04" \
  "++opt.lambda_camera_geometry_reg=0.014" \
  "++opt.boundary_opacity_residual_lr=0.0" \
  "++opt.boundary_scaling_residual_lr=0.0" \
  "++opt.latent_weight_decay=0.0" \
  "opt.lambda_l1=0.035" \
  "++opt.lambda_l1_fg=0.12" \
  "opt.lambda_perceptual=0.22" \
  "opt.lambda_dssim=0.0" \
  "++opt.perceptual_masked=true" \
  "++opt.perceptual_mask_dilate=3" \
  "++opt.perceptual_crop_pad=20" \
  "++opt.lambda_l1_boundary=0.10" \
  "opt.lambda_mask=0.0" \
  "++opt.lambda_mask_boundary=0.010" \
  "++opt.lambda_mask_boundary_hard=0.015" \
  "++opt.lambda_silhouette_outer=0.010" \
  "++opt.lambda_silhouette_outer_shell=0.012" \
  "++opt.lambda_silhouette_inner=0.006" \
  "++opt.silhouette_outer_ring_width=9" \
  "++opt.silhouette_outer_shell_start_width=1" \
  "++opt.silhouette_outer_shell_end_width=17" \
  "++opt.silhouette_inner_ring_width=7" \
  "++opt.boundary_band_width=11" \
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
  "++opt.lambda_perceptual_face=0.10" \
  "++opt.lambda_perceptual_shoulder_arm=0.04" \
  "++opt.lambda_perceptual_waist=0.03" \
  "++opt.lambda_perceptual_face_patch=0.06" \
  "++opt.lambda_perceptual_shoulder_arm_patch=0.035" \
  "++opt.lambda_perceptual_waist_patch=0.025" \
  "++opt.lambda_perceptual_upper_torso_patch=0.035" \
  "++opt.lambda_perceptual_upper_torso_core_patch=0.035" \
  "opt.grad_clip=0.02" \
  "++opt.train_sample_camera_min_prob=0.25" \
  "++opt.train_sample_camera_max_prob=0.40" \
  "++opt.photometric_contour_debug_interval=200" \
  "test_interval=300" \
  "test_iterations=[300,600,900,1200]" \
  "save_iterations=[600,1200]" \
  "checkpoint_iterations=[300,600,900,1200]" \
  "++validation_image_log_limit=0" \
  "$@"
