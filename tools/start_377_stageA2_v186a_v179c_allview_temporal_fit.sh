#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RUN_ID="${RUN_ID:-$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
BASE_EXP="$ROOT/exp/377_multiview_explicit_hq_rootfix_resume_v179c_v178c_k6_camquality_tightphoto_screen3000"
EXP_DIR="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v186a_v179c_allview_temporal_fit_$RUN_ID}"
START_CKPT="${2:-$BASE_EXP/best_ckpt.pth}"
ITERATIONS="${3:-2400}"

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
  "dataset.train_views=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]" \
  "dataset.val_views=[21,22,23]" \
  "dataset.test_views.view=[21,22,23]" \
  "dataset.val_frames=[0,570,30]" \
  "dataset.test_frames.view=[0,570,30]" \
  "start_checkpoint=$START_CKPT" \
  "exp_dir=$EXP_DIR" \
  "wandb_disable=true" \
  "++resume.restore_converter_optimizer_state=false" \
  "++resume.restore_converter_scheduler_state=false" \
  "++resume.partial_converter_missing_keys_allow_patterns=[structured_trunk_output_head_local_color_owner_head_boundary,camera_affine.]" \
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
  "opt.texture_lr=3.0e-05" \
  "opt.tex_latent_lr=0.0" \
  "++opt.camera_affine_enable=true" \
  "++opt.camera_affine_train_camera_ids=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]" \
  "++opt.camera_affine_max_camera_id=23" \
  "++opt.camera_affine_strength=[0.25,1,0.55,700,0.85,1600,1.0]" \
  "++opt.camera_affine_scale_max_delta=0.045" \
  "++opt.camera_affine_shift_max_abs=0.024" \
  "++opt.camera_affine_clamp_colors=true" \
  "++opt.camera_affine_apply_unknown=false" \
  "++opt.camera_affine_scale_reg_weight=1.0" \
  "++opt.camera_affine_shift_reg_weight=1.0" \
  "++opt.camera_affine_lr=5.0e-05" \
  "++opt.lambda_camera_affine_reg=0.003" \
  "++opt.camera_geometry_enable=false" \
  "++opt.camera_geometry_lr=0.0" \
  "++opt.boundary_opacity_residual_lr=0.0" \
  "++opt.boundary_scaling_residual_lr=0.0" \
  "++opt.latent_weight_decay=0.0" \
  "++opt.train_sample_camera_min_prob=0.020" \
  "++opt.train_sample_camera_max_prob=0.095" \
  "++opt.photometric_contour_debug_interval=400" \
  "test_interval=400" \
  "test_iterations=[400,800,1200,1600,2000,2400]" \
  "save_iterations=[1200,2400]" \
  "checkpoint_iterations=[400,800,1200,1600,2000,2400]" \
  "++validation_image_log_limit=0" \
  "$@"
