#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run this script directly with bash, not via source" >&2
  return 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
GPU="${GPU:-1}"
RUN_ID="${RUN_ID:-stageB_v235_silhouette_support_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
ITERATIONS="${ITERATIONS:-3000}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-1000,2000,3000}"
DO_RENDER="${DO_RENDER:-1}"
WAIT_FOR_FREE_GPU="${WAIT_FOR_FREE_GPU:-1}"
GPU_MAX_USED_MB_START="${GPU_MAX_USED_MB_START:-5000}"
GPU_MAX_UTIL_START="${GPU_MAX_UTIL_START:-25}"
GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS:-60}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
COMPACT_MAPPING="${COMPACT_MAPPING:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_hulk_light_v233d_shoes_preserve_control_stageB_compact_v233_skincloth_20260512_161652_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt135710.pth}"
BASE_ITER="${BASE_ITER:-135710}"

EXP_DIR="${EXP_DIR:-$ROOT/exp/stageB/377_hulk_light_v235a_silhouette_support_refine_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_v235_silhouette_support_${RUN_ID}}"
SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
mkdir -p "$EXP_DIR" "$LOG_DIR"

CHECKPOINT_LIST="[$CHECKPOINT_STEPS]"
FINAL_ITER=$((BASE_ITER + ITERATIONS))
FINAL_CKPT="$EXP_DIR/ckpt${FINAL_ITER}.pth"

SELECT=(
  render_c21_f000240.png
  render_c21_f000300.png
  render_c22_f000240.png
  render_c23_f000300.png
  render_c23_f000420.png
)
BINDING_MAPS="[layer,region,compact_semantic,body_prob,soft_prob,cloth_prob,semantic,temporal,thin]"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$DATA_ROOT" "$PARSER_ROOT" "$COMPACT_MAPPING"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

printf 'time_bjt\tgpu\tphase\tdetail\n' > "$EVENTS"
printf 'name\tkind\texp_dir\trender_exp\tckpt\tstatus\tdetail\n' > "$SUMMARY"

log_event() {
  printf '%s\t%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$GPU" "$1" "$2" | tee -a "$EVENTS"
}

gpu_stats() {
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$GPU" \
    | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
}

wait_for_gpu() {
  if [ "$WAIT_FOR_FREE_GPU" != "1" ]; then
    return 0
  fi
  local used util
  while true; do
    read -r used util < <(gpu_stats)
    used="${used:-999999}"
    util="${util:-100}"
    if [ "$used" -le "$GPU_MAX_USED_MB_START" ] && [ "$util" -le "$GPU_MAX_UTIL_START" ]; then
      log_event "gpu_ready" "used=${used}MiB util=${util}%"
      return 0
    fi
    log_event "gpu_wait" "used=${used}MiB util=${util}% threshold=${GPU_MAX_USED_MB_START}MiB/${GPU_MAX_UTIL_START}%"
    sleep "$GPU_WAIT_POLL_SECONDS"
  done
}

{
  echo "RUN_ID=$RUN_ID"
  echo "START_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
  echo "GPU=$GPU"
  echo "ITERATIONS=$ITERATIONS"
  echo "CHECKPOINT_STEPS=$CHECKPOINT_STEPS"
  echo "BASE_EXP=$BASE_EXP"
  echo "BASE_CKPT=$BASE_CKPT"
  echo "BASE_ITER=$BASE_ITER"
  echo "EXP_DIR=$EXP_DIR"
  echo "FINAL_CKPT=$FINAL_CKPT"
  echo "PURPOSE=v235 silhouette/support only; semantic logits frozen; small boundary scaling residual enabled"
  echo "KEY_LR=opacity_lr=0.000100 boundary_opacity_residual_lr=0.000055 boundary_scaling_residual_lr=0.000020"
} | tee "$LOG_DIR/run_info.txt"

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=$GPU"
  "OMP_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "MKL_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "OPENBLAS_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "NUMEXPR_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "PYTHONUNBUFFERED=1"
  "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64"
)

wait_for_gpu
log_event "train_start" "iterations=$ITERATIONS checkpoints=$CHECKPOINT_LIST"
env "${COMMON_ENV[@]}" "$PYTHON_BIN" "$ROOT/train.py" \
  --config-path "$BASE_EXP/.hydra" \
  --config-name config \
  mode=train \
  "dataset.root_dir=$DATA_ROOT" \
  "dataset.preload=false" \
  "dataset.train_views=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]" \
  "dataset.val_views=[21,22,23]" \
  "dataset.test_views.view=[21,22,23]" \
  "dataset.train_frames=[0,570,1]" \
  "dataset.val_frames=[0,570,60]" \
  "dataset.test_frames.view=[0,570,60]" \
  "dataset.parsing_prior.enable=true" \
  "dataset.parsing_prior.roi_enable=true" \
  "dataset.parsing_prior.parser_root=$PARSER_ROOT" \
  "dataset.parsing_prior.parser_layout=cihp_subject" \
  "dataset.parsing_prior.use_direct_parser_labels=true" \
  "dataset.parsing_prior.compact_mapping_file=$COMPACT_MAPPING" \
  "dataset.parsing_prior.skip_empty_samples=false" \
  "start_checkpoint=$BASE_CKPT" \
  "exp_dir=$EXP_DIR" \
  "seed=-1" \
  "hydra.run.dir=$LOG_DIR/hydra_train" \
  "wandb_disable=true" \
  "++resume.allow_partial_converter_load=false" \
  "++resume.restore_gaussian_optimizer_state=false" \
  "++resume.restore_converter_optimizer_state=false" \
  "++resume.restore_converter_scheduler_state=false" \
  "++resume.disable_densify_on_resume=true" \
  "++resume.disable_opacity_reset_on_resume=true" \
  "++resume.require_no_densify_on_resume=true" \
  "++resume.use_checkpoint_iteration_as_offset=true" \
  "++resume.clear_boundary_tags_on_resume=true" \
  "pipeline.pose_noise=0.0" \
  "model.gaussian.delay=0" \
  "++model.pose_correction.train_root_orient=false" \
  "++model.pose_correction.train_pose_body=false" \
  "++model.pose_correction.train_pose_hand=false" \
  "++model.pose_correction.train_trans=false" \
  "++model.pose_correction.train_betas=false" \
  "opt.iterations=$ITERATIONS" \
  "opt.position_lr_init=0.0" \
  "opt.position_lr_final=0.0" \
  "opt.feature_lr=0.0" \
  "opt.rotation_lr=0.0" \
  "opt.rigid_lr=0.0" \
  "opt.non_rigid_lr=0.0" \
  "opt.nr_latent_lr=0.0" \
  "opt.pose_correction_lr=0.0" \
  "opt.texture_lr=0.0" \
  "opt.tex_latent_lr=0.0" \
  "++opt.texture_trainable_name_patterns=[__freeze_texture_no_match__]" \
  "++opt.camera_affine_enable=false" \
  "++opt.camera_affine_lr=0.0" \
  "++opt.camera_geometry_enable=true" \
  "++opt.camera_geometry_lr=0.0" \
  "++opt.semantic_region_logits_lr=0.0" \
  "++opt.semantic_compact_logits_lr=0.0" \
  "++opt.stageB_semantic_loss_enable=false" \
  "++opt.lambda_binding_semantic_adapter_reg=0.0" \
  "++opt.train_sample_mode=frame_balanced_camera_weighted" \
  "++opt.train_sample_camera_min_prob=0.020" \
  "++opt.train_sample_camera_max_prob=0.160" \
  "++opt.train_sample_log_interval=100" \
  "++opt.train_sample_accumulation_steps=1" \
  "++opt.foreground_mask_source=hard" \
  "++opt.global_mask_source=hard" \
  "++opt.boundary_target_mask_source=hard" \
  "++opt.boundary_region_source=binary" \
  "++opt.boundary_band_width=9" \
  "opt.lambda_mask=0.008" \
  "++opt.mask_loss_type=l1" \
  "++opt.lambda_mask_boundary=0.016" \
  "++opt.lambda_mask_boundary_hard=0.011" \
  "++opt.lambda_silhouette_outer=0.014" \
  "++opt.lambda_silhouette_outer_shell=0.026" \
  "++opt.lambda_silhouette_outer_spike=0.015" \
  "++opt.silhouette_outer_ring_width=7" \
  "++opt.silhouette_outer_shell_start_width=1" \
  "++opt.silhouette_outer_shell_end_width=23" \
  "++opt.silhouette_outer_shell_soft_weights=false" \
  "++opt.boundary_image_error_score_enable=true" \
  "++opt.boundary_image_error_score_signed_enable=true" \
  "++opt.boundary_image_error_score_mix=1.0" \
  "++opt.boundary_image_error_score_gain=1.18" \
  "++opt.boundary_image_error_score_power=1.0" \
  "++opt.boundary_image_error_score_min=0.012" \
  "++opt.boundary_image_error_score_band_width=8" \
  "++opt.boundary_image_error_pred_threshold=0.32" \
  "++opt.boundary_image_error_target_threshold=0.50" \
  "++opt.boundary_image_error_score_focus_dilate=4" \
  "++opt.boundary_image_error_score_smooth_k=10" \
  "++opt.boundary_image_error_score_smooth_blend=0.18" \
  "++opt.boundary_image_error_score_prior_floor=0.05" \
  "++opt.boundary_aware_enable=true" \
  "++opt.boundary_aware_gate_mask_boundary=true" \
  "++opt.boundary_aware_gate_mask_boundary_hard=true" \
  "++opt.boundary_aware_gate_silhouette_outer=true" \
  "++opt.boundary_aware_gate_silhouette_outer_shell=true" \
  "++opt.boundary_aware_gate_silhouette_outer_spike=true" \
  "++opt.boundary_aware_threshold=0.015" \
  "++opt.boundary_aware_score_power=0.90" \
  "++opt.boundary_tag_schedule_use_local_iteration=true" \
  "++opt.boundary_tag_enable=true" \
  "++opt.boundary_tag_init_iter=1" \
  "++opt.boundary_tag_update_interval=20" \
  "++opt.boundary_tag_update_until_iter=260" \
  "++opt.boundary_tag_mode=topk_ratio" \
  "++opt.boundary_tag_topk_ratio=0.14" \
  "++opt.boundary_tag_min_ratio=0.08" \
  "++opt.boundary_tag_threshold=0.16" \
  "++opt.boundary_tag_binary=true" \
  "++opt.boundary_tag_use_score_within_subset=true" \
  "++opt.boundary_tag_score_smooth_blend=0.18" \
  "++opt.boundary_tag_score_smooth_k=10" \
  "++opt.boundary_tag_support_k=8" \
  "++opt.boundary_tag_support_threshold=0.16" \
  "++opt.boundary_signed_routing_enable=true" \
  "++opt.boundary_signed_mixed_loss_scale=0.035" \
  "++opt.boundary_signed_shrink_loss_scale=1.10" \
  "++opt.boundary_signed_grow_loss_scale=0.28" \
  "++opt.boundary_signed_share_gain=1.00" \
  "++opt.boundary_signed_share_power=1.0" \
  "++opt.boundary_signed_shrink_share_gain=1.10" \
  "++opt.boundary_signed_shrink_share_power=1.15" \
  "++opt.boundary_aware_freeze_converter_for_boundary_loss=true" \
  "++opt.boundary_aware_feature_dc_scale=0.0" \
  "++opt.boundary_aware_feature_rest_scale=0.0" \
  "++opt.boundary_aware_xyz_scale=0.0" \
  "++opt.boundary_aware_opacity_scale=0.18" \
  "++opt.boundary_aware_scaling_scale=0.0" \
  "++opt.boundary_aware_boundary_opacity_residual_scale=0.95" \
  "++opt.boundary_aware_boundary_scaling_residual_scale=0.32" \
  "opt.opacity_lr=0.000100" \
  "opt.scaling_lr=0.0" \
  "++opt.boundary_opacity_residual_lr=0.000055" \
  "++opt.boundary_scaling_residual_lr=0.000020" \
  "++opt.lambda_boundary_opacity_residual_reg=0.0012" \
  "++opt.lambda_boundary_scaling_residual_reg=0.0018" \
  "++opt.lambda_boundary_opacity_residual_smooth=0.0010" \
  "++opt.lambda_boundary_scaling_residual_smooth=0.0016" \
  "opt.lambda_l1=0.0" \
  "opt.lambda_l1_fg=0.0" \
  "opt.lambda_l1_boundary=0.0" \
  "opt.lambda_l1_face=0.0" \
  "opt.lambda_l1_shoulder_arm=0.0" \
  "opt.lambda_l1_waist=0.0" \
  "opt.lambda_perceptual=0.0" \
  "++opt.lambda_perceptual_face=0.0" \
  "++opt.lambda_perceptual_shoulder_arm=0.0" \
  "++opt.lambda_perceptual_waist=0.0" \
  "++opt.lambda_perceptual_face_patch=0.0" \
  "++opt.lambda_perceptual_shoulder_arm_patch=0.0" \
  "++opt.lambda_perceptual_waist_patch=0.0" \
  "++opt.lambda_perceptual_upper_torso_patch=0.0" \
  "++opt.lambda_perceptual_upper_torso_core_patch=0.0" \
  "opt.lambda_edge_face=0.0" \
  "opt.lambda_edge_shoulder_arm=0.0" \
  "opt.lambda_edge_waist=0.0" \
  "++opt.lambda_detail_face=0.0" \
  "++opt.lambda_detail_shoulder_arm=0.0" \
  "++opt.lambda_detail_waist=0.0" \
  "++opt.lambda_detail_face_luma_dog=0.0" \
  "++opt.lambda_detail_shoulder_arm_luma_dog=0.0" \
  "++opt.lambda_detail_waist_luma_dog=0.0" \
  "++opt.lambda_detail_upper_torso_luma_dog=0.0" \
  "++opt.lambda_detail_upper_torso_core_luma_dog=0.0" \
  "opt.lambda_skinning=0.0" \
  "opt.lambda_aiap_xyz=0.0" \
  "opt.lambda_aiap_cov=0.0" \
  "opt.percent_dense=0.0" \
  "opt.densify_until_iter=0" \
  "opt.densify_from_iter=1000000" \
  "opt.opacity_reset_interval=1000000" \
  "best_eval_split=test" \
  "best_metric=lpips_fg" \
  "best_metric_mode=min" \
  "best_metric_source=best_eval" \
  "test_interval=1000" \
  "test_iterations=$CHECKPOINT_LIST" \
  "save_iterations=$CHECKPOINT_LIST" \
  "checkpoint_iterations=$CHECKPOINT_LIST" \
  "++validation_image_log_limit=0" \
  "opt.grad_clip=0.0025" \
  > "$LOG_DIR/train.log" 2>&1
log_event "train_done" "$FINAL_CKPT"
printf 'v235a_silhouette_support\ttrain\t%s\t\t%s\tok\titerations=%s\n' "$EXP_DIR" "$FINAL_CKPT" "$ITERATIONS" >> "$SUMMARY"

render_checkpoint() {
  local local_step="$1"
  local global_iter=$((BASE_ITER + local_step))
  local ckpt="$EXP_DIR/ckpt${global_iter}.pth"
  local render_exp="${EXP_DIR}_render_silhouette_support_ckpt${global_iter}"
  if [ ! -f "$ckpt" ]; then
    log_event "render_skip" "missing=$ckpt"
    printf 'v235a_silhouette_support\trender\t%s\t%s\t%s\tskipped\tmissing_ckpt\n' "$EXP_DIR" "$render_exp" "$ckpt" >> "$SUMMARY"
    return 0
  fi
  log_event "render_start" "ckpt=$ckpt render_exp=$render_exp"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" "$ROOT/render.py" \
    --config-path "$EXP_DIR/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.test_views.view=[21,22,23]" \
    "dataset.test_frames.view=[0,570,60]" \
    "dataset.parsing_prior.enable=true" \
    "dataset.parsing_prior.roi_enable=true" \
    "dataset.parsing_prior.parser_root=$PARSER_ROOT" \
    "dataset.parsing_prior.parser_layout=cihp_subject" \
    "dataset.parsing_prior.use_direct_parser_labels=true" \
    "dataset.parsing_prior.compact_mapping_file=$COMPACT_MAPPING" \
    "export_interpretability=true" \
    "++binding_map_names=$BINDING_MAPS" \
    "++binding_map_use_opacity_mask=true" \
    "++binding_map_hard_fg_opacity_threshold=0.030" \
    "++binding_map_opacity_threshold=0.030" \
    "++binding_map_compact_semantic_opacity_threshold=0.025" \
    "++binding_map_hard_fg_close_kernel=5" \
    "++binding_map_hard_fg_erode_kernel=1" \
    "++binding_map_support_close_kernel=5" \
    "++binding_map_support_erode_kernel=1" \
    "++render_export_opacity_threshold=0.025" \
    "++render_export_close_kernel=5" \
    "++render_export_erode_kernel=1" \
    "export_semantic_editable_assets=true" \
    "semantic_editable_parser_root=$PARSER_ROOT" \
    "semantic_editable_parser_layout=cihp_subject" \
    "semantic_editable_direct_parser_mode=false" \
    "semantic_editable_export_compact_head=true" \
    "semantic_editable_include_binding_summary=true" \
    "++semantic_editable_compact_opacity_threshold=0.025" \
    "++semantic_editable_compact_confidence_threshold=0.0" \
    "+semantic_editable_preview_min_area=18" \
    "hydra.run.dir=$LOG_DIR/hydra_render_ckpt${global_iter}" \
    "wandb_disable=true" \
    > "$LOG_DIR/render_ckpt${global_iter}.log" 2>&1
  "$PYTHON_BIN" "$ROOT/tools/analyze_377_render_contours.py" \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --band-width 7 \
    --topk 12 \
    --out-dir "$render_exp/diagnostics/contours" \
    > "$LOG_DIR/contours_ckpt${global_iter}.log" 2>&1 || true
  "$PYTHON_BIN" "$ROOT/tools/make_binding_paper_montage.py" \
    --exp-dir "$render_exp" \
    --gt-root "$DATA_ROOT/CoreView_377" \
    --split test-view \
    --panels gt render layer region body_prob cloth_prob compact_semantic thin semantic \
    --select "${SELECT[@]}" \
    --output-dir "$render_exp/test-view/paper_montages_selected" \
    > "$LOG_DIR/montage_ckpt${global_iter}.log" 2>&1 || true
  log_event "render_done" "$render_exp"
  printf 'v235a_silhouette_support\trender\t%s\t%s\t%s\tok\tlocal_step=%s\n' "$EXP_DIR" "$render_exp" "$ckpt" "$local_step" >> "$SUMMARY"
}

if [ "$DO_RENDER" = "1" ]; then
  IFS=',' read -ra STEPS <<< "$CHECKPOINT_STEPS"
  for step in "${STEPS[@]}"; do
    render_checkpoint "$step"
  done
fi

{
  echo "END_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
  echo "EXP_DIR=$EXP_DIR"
  echo "FINAL_CKPT=$FINAL_CKPT"
  echo "SUMMARY=$SUMMARY"
} | tee -a "$LOG_DIR/run_info.txt"
