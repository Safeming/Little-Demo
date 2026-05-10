#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RUN_ID="${RUN_ID:-$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
BASE_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v198a_v197a_boundary_substrate_continue_20260508_224752_bjt_v198a_boundary_substrate_continue"
EXP_DIR="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v200a_v198a_parser_roi_safe_probe_$RUN_ID}"
START_CKPT="${2:-$BASE_EXP/best_ckpt.pth}"
ITERATIONS="${3:-1600}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"

if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then shift; fi
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then shift; fi
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then shift; fi

if [ ! -f "$START_CKPT" ]; then
  echo "missing checkpoint: $START_CKPT" >&2
  exit 2
fi

for cam in $(seq 1 20); do
  parser_dir="$PARSER_ROOT/CoreView_377/mask_cihp/Camera_B${cam}"
  if [ ! -d "$parser_dir" ]; then
    echo "missing parser dir for Camera_B${cam}: $parser_dir" >&2
    exit 3
  fi
done

mkdir -p "$EXP_DIR"
cd "$ROOT"

exec env PYTHONUNBUFFERED=1 "$PYTHON_BIN" train.py \
  --config-path "$BASE_EXP/.hydra" \
  --config-name config \
  mode=train \
  "dataset.root_dir=$ROOT/data/ZJUMoCap" \
  "dataset.preload=false" \
  "dataset.train_views=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]" \
  "dataset.val_views=[21,22,23]" \
  "dataset.test_views.view=[21,22,23]" \
  "dataset.train_frames=[0,570,1]" \
  "dataset.val_frames=[0,570,30]" \
  "dataset.test_frames.view=[0,570,30]" \
  "dataset.parsing_prior.enable=false" \
  "dataset.parsing_prior.roi_enable=true" \
  "dataset.parsing_prior.parser_root=$PARSER_ROOT" \
  "dataset.parsing_prior.parser_layout=cihp_subject" \
  "dataset.parsing_prior.use_direct_parser_labels=true" \
  "dataset.parsing_prior.compact_mapping_file=" \
  "dataset.parsing_prior.skip_empty_samples=false" \
  "start_checkpoint=$START_CKPT" \
  "exp_dir=$EXP_DIR" \
  "wandb_disable=true" \
  "++resume.restore_converter_optimizer_state=false" \
  "++resume.restore_converter_scheduler_state=false" \
  "++resume.partial_converter_missing_keys_allow_patterns=[]" \
  "++resume.disable_densify_on_resume=true" \
  "++resume.disable_opacity_reset_on_resume=true" \
  "++resume.require_no_densify_on_resume=true" \
  "++resume.clear_boundary_tags_on_resume=false" \
  "opt.iterations=$ITERATIONS" \
  "pipeline.pose_noise=0.0" \
  "model.pose_correction.delay=1" \
  "++model.pose_correction.train_root_orient=false" \
  "++model.pose_correction.train_pose_body=false" \
  "++model.pose_correction.train_pose_hand=false" \
  "++model.pose_correction.train_trans=false" \
  "++model.pose_correction.train_betas=false" \
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
  "++opt.camera_affine_enable=false" \
  "++opt.camera_affine_lr=0.0" \
  "++opt.camera_geometry_enable=true" \
  "++opt.camera_geometry_lr=0.0" \
  "++opt.boundary_opacity_residual_lr=0.0" \
  "++opt.boundary_scaling_residual_lr=0.0" \
  "++opt.latent_weight_decay=0.0" \
  "opt.tex_latent_lr=0.0" \
  "opt.texture_lr=2.5e-06" \
  "++opt.texture_trainable_name_patterns=[detail_high_freq_context_proj.*,detail_high_freq_carrier_proj.*,detail_high_freq_mlp.*,detail_high_freq_gate_mlp.*,detail_high_freq_luma_mlp.*,detail_high_freq_face_mlp.*,detail_high_freq_face_gate_mlp.*,detail_high_freq_face_local_proj.*,detail_high_freq_face_extra_local_projs.*,detail_high_freq_structure_proj.*,structured_trunk_output_head_hf_head_mlp.*,structured_trunk_output_head_hf_head_gate_mlp.*]" \
  "++opt.lambda_binding_parsing=0.0" \
  "++opt.face_region_source=parser_prefer" \
  "++opt.face_region_parser_dilate=3" \
  "++opt.face_region_source_aware_validity_enable=true" \
  "++opt.face_region_min_pixels_parser=24" \
  "++opt.face_region_debug_enable=true" \
  "++opt.face_region_debug_interval=200" \
  "++opt.shoulder_arm_region_source=parser_prefer" \
  "++opt.shoulder_arm_region_parser_dilate=5" \
  "++opt.shoulder_arm_region_source_aware_validity_enable=true" \
  "++opt.shoulder_arm_region_min_pixels_parser=40" \
  "++opt.shoulder_arm_region_debug_enable=true" \
  "++opt.shoulder_arm_region_debug_interval=200" \
  "++opt.upper_torso_region_source=parser_prefer" \
  "++opt.upper_torso_region_parser_dilate=5" \
  "++opt.upper_torso_region_debug_enable=true" \
  "++opt.upper_torso_region_debug_interval=200" \
  "++opt.waist_region_source=parser_prefer" \
  "++opt.waist_region_parser_dilate=3" \
  "++opt.waist_region_debug_enable=true" \
  "++opt.waist_region_debug_interval=200" \
  "++opt.lambda_detail_face=0.0" \
  "++opt.lambda_detail_shoulder_arm=0.0" \
  "++opt.lambda_detail_waist=0.0" \
  "++opt.lambda_detail_face_luma_dog=0.0" \
  "++opt.lambda_detail_shoulder_arm_luma_dog=0.0" \
  "++opt.lambda_detail_waist_luma_dog=0.0" \
  "++opt.lambda_detail_upper_torso_luma_dog=0.0" \
  "++opt.lambda_detail_upper_torso_core_luma_dog=0.0" \
  "++opt.reliable_view_apply_detail=false" \
  "++opt.reliable_view_apply_luma_dog=false" \
  "opt.grad_clip=0.0045" \
  "test_interval=400" \
  "test_iterations=[400,800,1200,1600]" \
  "save_iterations=[800,1600]" \
  "checkpoint_iterations=[400,800,1200,1600]" \
  "++validation_image_log_limit=0" \
  "$@"
