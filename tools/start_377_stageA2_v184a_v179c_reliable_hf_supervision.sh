#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RUN_ID="${RUN_ID:-$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
BASE_EXP="$ROOT/exp/377_multiview_explicit_hq_rootfix_resume_v179c_v178c_k6_camquality_tightphoto_screen3000"
EXP_DIR="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v184a_v179c_reliable_hf_supervision_$RUN_ID}"
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
  "start_checkpoint=$START_CKPT" \
  "exp_dir=$EXP_DIR" \
  "wandb_disable=true" \
  "++resume.restore_converter_optimizer_state=true" \
  "++resume.restore_converter_scheduler_state=false" \
  "++resume.restore_converter_optimizer_preserve_config_lrs=true" \
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
  "opt.texture_lr=4.0e-05" \
  "opt.tex_latent_lr=0.0" \
  "++opt.camera_affine_enable=false" \
  "++opt.camera_affine_lr=0.0" \
  "++opt.camera_geometry_enable=false" \
  "++opt.camera_geometry_lr=0.0" \
  "++opt.boundary_opacity_residual_lr=0.0" \
  "++opt.boundary_scaling_residual_lr=0.0" \
  "++opt.latent_weight_decay=0.0" \
  "++opt.reliable_view_supervision_enable=true" \
  "++opt.reliable_view_camera_quality_weights={1:1.25,2:0.55,3:0.90,4:0.45,5:1.20,6:0.90,7:0.95,8:0.95,9:1.08,10:1.20,11:0.62,12:0.78,13:0.90,14:1.18,15:0.62,16:0.82,17:0.55,18:0.55,19:0.78,20:0.55}" \
  "++opt.reliable_view_default_highfreq_weight=0.85" \
  "++opt.reliable_view_unknown_highfreq_weight=0.75" \
  "++opt.reliable_view_highfreq_min_weight=0.45" \
  "++opt.reliable_view_highfreq_max_weight=1.25" \
  "++opt.reliable_view_highfreq_power=1.15" \
  "++opt.reliable_view_apply_edge=true" \
  "++opt.reliable_view_apply_detail=true" \
  "++opt.reliable_view_apply_luma_dog=true" \
  "++opt.reliable_view_apply_patch_perceptual=true" \
  "++opt.reliable_view_apply_region_perceptual=false" \
  "++opt.photometric_contour_debug_interval=250" \
  "test_interval=400" \
  "test_iterations=[400,800,1200,1600,2000,2400]" \
  "save_iterations=[1200,2400]" \
  "checkpoint_iterations=[400,800,1200,1600,2000,2400]" \
  "++validation_image_log_limit=0" \
  "$@"
