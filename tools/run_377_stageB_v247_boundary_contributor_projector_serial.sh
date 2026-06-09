#!/usr/bin/env bash
set -u
set -o pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run this script directly with bash, not via source" >&2
  return 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x /opt/miniconda3/envs/ictrl/bin/python ]; then
    PYTHON_BIN=/opt/miniconda3/envs/ictrl/bin/python
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    PYTHON_BIN=python3
  fi
fi

RUN_ID="${RUN_ID:-stageB_v247_boundary_contributor_projector_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
GPU="${GPU:-0}"
SEED="${SEED:--1}"
DO_RENDER="${DO_RENDER:-1}"
WAIT_FOR_FREE_GPU="${WAIT_FOR_FREE_GPU:-1}"
GPU_MAX_USED_MB_START="${GPU_MAX_USED_MB_START:-18000}"
GPU_MAX_UTIL_START="${GPU_MAX_UTIL_START:-65}"
GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS:-60}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-8}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
COMPACT_MAPPING="${COMPACT_MAPPING:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_hulk_light_v233d_shoes_preserve_control_stageB_compact_v233_skincloth_20260512_161652_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt135710.pth}"
BASE_ITER="${BASE_ITER:-135710}"
BASE_RENDER_EXP="${BASE_RENDER_EXP:-$ROOT/exp/stageB/377_hulk_light_v233d_stageC_softer_stageC_v235_v236_20260513_114036_bjt_render_compact_final}"

LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_v247_boundary_contributor_projector_${RUN_ID}}"
SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
STATUS_JSON="$LOG_DIR/status.json"
mkdir -p "$LOG_DIR"

ALL20="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]"
SELECT=(render_c21_f000240.png render_c21_f000300.png render_c22_f000240.png render_c23_f000300.png render_c23_f000420.png)
BINDING_MAPS="[layer,region,compact_semantic,body_prob,soft_prob,cloth_prob,semantic,temporal,thin]"

printf 'time_bjt\tgpu\tname\tphase\tdetail\n' > "$EVENTS"
printf 'name\tkind\texp_dir\trender_exp\tckpt\tstatus\tdetail\n' > "$SUMMARY"

log_event() {
  local gpu="$1"
  local name="$2"
  local phase="$3"
  local detail="$4"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$gpu" "$name" "$phase" "$detail" | tee -a "$EVENTS"
}

write_status() {
  "$PYTHON_BIN" - "$STATUS_JSON" "$RUN_ID" "$GPU" "$1" "$2" <<'PY' || true
import json, sys, time
from pathlib import Path
path, run_id, gpu, phase, detail = sys.argv[1:]
Path(path).write_text(json.dumps({
    "run_id": run_id,
    "gpu": gpu,
    "phase": phase,
    "detail": detail,
    "now_epoch": int(time.time()),
}, indent=2), encoding="utf-8")
PY
}

summary_row() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7" >> "$SUMMARY"
}

gpu_stats() {
  nvidia-smi --id="$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
}

wait_for_gpu() {
  local name="$1"
  if [ "$WAIT_FOR_FREE_GPU" != "1" ]; then
    return 0
  fi
  local used util
  while true; do
    read -r used util < <(gpu_stats)
    used="${used:-0}"
    util="${util:-0}"
    if [ "$used" -le "$GPU_MAX_USED_MB_START" ] && [ "$util" -le "$GPU_MAX_UTIL_START" ]; then
      log_event "$GPU" "$name" "gpu_ready" "used=${used}MiB util=${util}%"
      return 0
    fi
    log_event "$GPU" "$name" "gpu_wait" "used=${used}MiB util=${util}% threshold=${GPU_MAX_USED_MB_START}MiB/${GPU_MAX_UTIL_START}%"
    sleep "$GPU_WAIT_POLL_SECONDS"
  done
}

common_env() {
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    OMP_NUM_THREADS="$CPU_THREADS_PER_JOB" \
    MKL_NUM_THREADS="$CPU_THREADS_PER_JOB" \
    OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB" \
    NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB" \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
    "$@"
}

check_required() {
  local missing=0
  for required in "$PYTHON_BIN" "$DATA_ROOT" "$PARSER_ROOT" "$COMPACT_MAPPING" "$BASE_RENDER_EXP/test-view/renders" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT"; do
    if [ ! -e "$required" ]; then
      echo "missing required path: $required" >&2
      log_event "$GPU" "preflight" "missing" "$required"
      missing=1
    fi
  done
  if [ "$missing" -ne 0 ]; then
    write_status "blocked" "missing required paths; see $EVENTS"
    return 2
  fi
}

run_residual_mining() {
  local out_dir="$LOG_DIR/v247a_residual_mining"
  log_event "$GPU" "v247a_residual_mining" "start" "render_exp=$BASE_RENDER_EXP"
  "$PYTHON_BIN" "$ROOT/tools/analyze_377_boundary_residuals.py" \
    --render-exp "$BASE_RENDER_EXP" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --band-width 7 \
    --search-band-width 24 \
    --render-support-threshold 0.025 \
    --close-kernel 5 \
    --topk 16 \
    --panel-width 220 \
    --out-dir "$out_dir" \
    > "$LOG_DIR/v247a_residual_mining.log" 2>&1
  local status=$?
  if [ "$status" -eq 0 ]; then
    log_event "$GPU" "v247a_residual_mining" "done" "$out_dir"
    summary_row "v247a_residual_mining" "diagnostic" "" "$BASE_RENDER_EXP" "" "ok" "$out_dir"
  else
    log_event "$GPU" "v247a_residual_mining" "failed" "status=$status"
    summary_row "v247a_residual_mining" "diagnostic" "" "$BASE_RENDER_EXP" "" "failed" "status=$status"
  fi
  return "$status"
}

render_checkpoint() {
  local name="$1"
  local exp_dir="$2"
  local local_step="$3"
  local global_iter=$((BASE_ITER + local_step))
  local ckpt="$exp_dir/ckpt${global_iter}.pth"
  local render_exp="${exp_dir}_render_v247_ckpt${global_iter}"
  if [ ! -f "$ckpt" ]; then
    log_event "$GPU" "$name" "render_skip" "missing=$ckpt"
    summary_row "$name" "render" "$exp_dir" "$render_exp" "$ckpt" "skipped" "missing_ckpt"
    return 0
  fi

  wait_for_gpu "${name}_render_${global_iter}"
  log_event "$GPU" "$name" "render_start" "ckpt=$ckpt"
  common_env "$PYTHON_BIN" "$ROOT/render.py" \
    --config-path "$exp_dir/.hydra" \
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
    "hydra.run.dir=$LOG_DIR/hydra_${name}_render_ckpt${global_iter}" \
    "wandb_disable=true" \
    > "$LOG_DIR/${name}_render_ckpt${global_iter}.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    log_event "$GPU" "$name" "render_failed" "status=$status"
    summary_row "$name" "render" "$exp_dir" "$render_exp" "$ckpt" "failed" "status=$status"
    return "$status"
  fi

  "$PYTHON_BIN" "$ROOT/tools/analyze_377_render_contours.py" \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --band-width 7 \
    --topk 12 \
    --out-dir "$render_exp/diagnostics/contours" \
    > "$LOG_DIR/${name}_contours_ckpt${global_iter}.log" 2>&1 || true
  "$PYTHON_BIN" "$ROOT/tools/analyze_377_boundary_residuals.py" \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --band-width 7 \
    --search-band-width 24 \
    --topk 12 \
    --out-dir "$render_exp/diagnostics/boundary_residuals" \
    > "$LOG_DIR/${name}_residuals_ckpt${global_iter}.log" 2>&1 || true
  "$PYTHON_BIN" "$ROOT/tools/make_binding_paper_montage.py" \
    --exp-dir "$render_exp" \
    --gt-root "$DATA_ROOT/CoreView_377" \
    --split test-view \
    --panels gt render layer region body_prob cloth_prob compact_semantic thin semantic \
    --select "${SELECT[@]}" \
    --output-dir "$render_exp/test-view/paper_montages_selected" \
    > "$LOG_DIR/${name}_montage_ckpt${global_iter}.log" 2>&1 || true

  log_event "$GPU" "$name" "render_done" "$render_exp"
  summary_row "$name" "render" "$exp_dir" "$render_exp" "$ckpt" "ok" "local_step=$local_step"
}

train_v247b() {
  local name="v247b_boundary_contributor_projector"
  local iterations="${V247_ITERATIONS:-360}"
  local checkpoint_steps="${V247_CHECKPOINT_STEPS:-120,240,360}"
  local checkpoint_list="[$checkpoint_steps]"
  local exp_dir="$ROOT/exp/stageB/377_hulk_light_${name}_${RUN_ID}"
  local final_ckpt="$exp_dir/ckpt$((BASE_ITER + iterations)).pth"
  mkdir -p "$exp_dir"

  wait_for_gpu "$name"
  log_event "$GPU" "$name" "train_start" "iterations=$iterations checkpoints=$checkpoint_steps"
  common_env "$PYTHON_BIN" "$ROOT/train.py" \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=train \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.train_views=$ALL20" \
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
    "exp_dir=$exp_dir" \
    "seed=$SEED" \
    "hydra.run.dir=$LOG_DIR/hydra_${name}_train" \
    "wandb_disable=true" \
    "++resume.allow_partial_converter_load=false" \
    "++resume.restore_gaussian_optimizer_state=false" \
    "++resume.restore_converter_optimizer_state=false" \
    "++resume.restore_converter_scheduler_state=false" \
    "++resume.disable_densify_on_resume=false" \
    "++resume.disable_opacity_reset_on_resume=true" \
    "++resume.require_no_densify_on_resume=false" \
    "++resume.use_checkpoint_iteration_as_offset=true" \
    "++resume.clear_boundary_tags_on_resume=true" \
    "pipeline.pose_noise=0.0" \
    "model.gaussian.delay=0" \
    "++model.pose_correction.train_root_orient=false" \
    "++model.pose_correction.train_pose_body=false" \
    "++model.pose_correction.train_pose_hand=false" \
    "++model.pose_correction.train_trans=false" \
    "++model.pose_correction.train_betas=false" \
    "opt.iterations=$iterations" \
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
    "++opt.train_sample_mode=frame_balanced_camera_weighted" \
    "++opt.train_sample_log_interval=100" \
    "++opt.train_sample_accumulation_steps=1" \
    "++opt.train_sample_camera_min_prob=0.020" \
    "++opt.train_sample_camera_max_prob=0.160" \
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
    "++opt.boundary_image_error_score_enable=true" \
    "++opt.boundary_image_error_score_signed_enable=true" \
    "++opt.boundary_image_error_score_mix=1.0" \
    "++opt.boundary_image_error_score_gain=1.25" \
    "++opt.boundary_image_error_score_min=0.008" \
    "++opt.boundary_image_error_score_band_width=8" \
    "++opt.boundary_image_error_pred_threshold=0.32" \
    "++opt.boundary_image_error_target_threshold=0.50" \
    "++opt.boundary_image_error_score_focus_dilate=4" \
    "++opt.boundary_image_error_score_neighborhood_radius=4" \
    "++opt.boundary_image_error_score_neighborhood_max_radius=6" \
    "++opt.boundary_image_error_score_use_radii_radius=true" \
    "++opt.boundary_image_error_score_radii_quantile=0.50" \
    "++opt.boundary_image_error_score_radii_scale=0.30" \
    "++opt.boundary_image_error_score_contributor_enable=true" \
    "++opt.boundary_image_error_score_contributor_mix=1.0" \
    "++opt.boundary_image_error_score_contributor_interval=20" \
    "++opt.boundary_image_error_contributor_residual_threshold=0.10" \
    "++opt.boundary_image_error_contributor_max_pixels=3072" \
    "++opt.boundary_image_error_contributor_pixel_chunk=384" \
    "++opt.boundary_image_error_contributor_point_chunk=8192" \
    "++opt.boundary_image_error_contributor_topk=6" \
    "++opt.boundary_image_error_contributor_radius_scale=1.35" \
    "++opt.boundary_image_error_contributor_min_radius=2.0" \
    "++opt.boundary_image_error_contributor_max_radius=16.0" \
    "++opt.boundary_image_error_contributor_sigma_scale=0.55" \
    "++opt.boundary_image_error_contributor_opacity_power=0.45" \
    "++opt.boundary_image_error_contributor_score_power=1.0" \
    "++opt.boundary_image_error_contributor_verbose=true" \
    "++opt.boundary_image_error_score_smooth_k=10" \
    "++opt.boundary_image_error_score_smooth_blend=0.18" \
    "++opt.boundary_image_error_score_prior_floor=0.05" \
    "++opt.boundary_aware_enable=true" \
    "++opt.boundary_aware_gate_mask_boundary=true" \
    "++opt.boundary_aware_gate_mask_boundary_hard=true" \
    "++opt.boundary_aware_gate_silhouette_outer=true" \
    "++opt.boundary_aware_gate_silhouette_outer_shell=true" \
    "++opt.boundary_aware_gate_silhouette_outer_spike=true" \
    "++opt.boundary_aware_threshold=0.012" \
    "++opt.boundary_aware_score_power=0.90" \
    "++opt.boundary_tag_schedule_use_local_iteration=true" \
    "++opt.boundary_tag_enable=true" \
    "++opt.boundary_tag_init_iter=1" \
    "++opt.boundary_tag_update_interval=20" \
    "++opt.boundary_tag_update_until_iter=260" \
    "++opt.boundary_tag_mode=topk_ratio" \
    "++opt.boundary_tag_topk_ratio=0.12" \
    "++opt.boundary_tag_min_ratio=0.06" \
    "++opt.boundary_tag_threshold=0.16" \
    "++opt.boundary_tag_binary=true" \
    "++opt.boundary_tag_use_score_within_subset=true" \
    "++opt.boundary_signed_routing_enable=true" \
    "++opt.boundary_signed_mixed_loss_scale=0.030" \
    "++opt.boundary_signed_shrink_loss_scale=0.90" \
    "++opt.boundary_signed_grow_loss_scale=0.44" \
    "++opt.boundary_aware_freeze_converter_for_boundary_loss=true" \
    "++opt.boundary_aware_feature_dc_scale=0.0" \
    "++opt.boundary_aware_feature_rest_scale=0.0" \
    "++opt.boundary_aware_xyz_scale=0.015" \
    "++opt.boundary_aware_opacity_scale=0.10" \
    "++opt.boundary_aware_scaling_scale=0.035" \
    "++opt.boundary_aware_boundary_opacity_residual_scale=0.55" \
    "++opt.boundary_aware_boundary_scaling_residual_scale=0.16" \
    "opt.position_lr_init=0.0000006" \
    "opt.position_lr_final=0.00000015" \
    "opt.opacity_lr=0.000060" \
    "opt.scaling_lr=0.0000018" \
    "++opt.boundary_opacity_residual_lr=0.000025" \
    "++opt.boundary_scaling_residual_lr=0.000008" \
    "++opt.lambda_boundary_opacity_residual_reg=0.0015" \
    "++opt.lambda_boundary_scaling_residual_reg=0.0025" \
    "++opt.lambda_boundary_opacity_residual_smooth=0.0012" \
    "++opt.lambda_boundary_scaling_residual_smooth=0.0025" \
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
    "++opt.lambda_detail_face=0.0" \
    "++opt.lambda_detail_shoulder_arm=0.0" \
    "++opt.lambda_detail_waist=0.0" \
    "opt.lambda_skinning=0.0" \
    "opt.lambda_aiap_xyz=0.0" \
    "opt.lambda_aiap_cov=0.0" \
    "best_eval_split=test" \
    "best_metric=lpips_fg" \
    "best_metric_mode=min" \
    "best_metric_source=best_eval" \
    "test_interval=1000" \
    "test_iterations=$checkpoint_list" \
    "save_iterations=$checkpoint_list" \
    "checkpoint_iterations=$checkpoint_list" \
    "++validation_image_log_limit=0" \
    "opt.percent_dense=0.030" \
    "opt.densification_interval=120" \
    "opt.densify_from_iter=$((BASE_ITER + 90))" \
    "opt.densify_until_iter=$((BASE_ITER + 370))" \
    "opt.densify_grad_threshold=0.00110" \
    "opt.opacity_threshold=0.000001" \
    "opt.opacity_reset_interval=1000000" \
    "++model.gaussian.binding_densify_strict_candidate_gate_enable=true" \
    "++model.gaussian.binding_densify_strict_candidate_boundary_only=true" \
    "++model.gaussian.binding_densify_strict_candidate_require_boundary_tag=true" \
    "++model.gaussian.binding_densify_strict_candidate_boundary_threshold=0.10" \
    "++model.gaussian.binding_densify_strict_candidate_arm_only=false" \
    "++model.gaussian.binding_densify_strict_candidate_max_points_per_lineage=3" \
    "++model.gaussian.binding_densify_strict_candidate_max_points=36" \
    "++model.gaussian.binding_densify_strict_candidate_max_ratio=0.0008" \
    "++model.gaussian.binding_densify_boundary_only=true" \
    "++model.gaussian.binding_densify_require_boundary_tag=true" \
    "++model.gaussian.binding_densify_boundary_threshold=0.10" \
    "++model.gaussian.binding_densify_debug_verbose=true" \
    "++model.gaussian.boundary_support_projector_enable=true" \
    "++model.gaussian.boundary_support_projector_verbose=true" \
    "++model.gaussian.boundary_support_projector_use_anchor_normal=true" \
    "++model.gaussian.boundary_support_projector_anchor_normal_outward_flip=true" \
    "++model.gaussian.boundary_support_projector_under_threshold=0.10" \
    "++model.gaussian.boundary_support_projector_over_threshold=0.12" \
    "++model.gaussian.boundary_support_projector_boundary_threshold=0.04" \
    "++model.gaussian.boundary_support_projector_inner_max_points=96" \
    "++model.gaussian.boundary_support_projector_inner_max_ratio=0.0020" \
    "++model.gaussian.boundary_support_projector_inner_max_points_per_lineage=2" \
    "++model.gaussian.boundary_support_projector_outer_max_points=96" \
    "++model.gaussian.boundary_support_projector_outer_max_ratio=0.0020" \
    "++model.gaussian.boundary_support_projector_outer_max_points_per_lineage=2" \
    "++model.gaussian.boundary_support_projector_child_opacity_factor=0.58" \
    "++model.gaussian.boundary_support_projector_child_opacity_floor=0.018" \
    "++model.gaussian.boundary_support_projector_child_opacity_ceiling=0.34" \
    "++model.gaussian.boundary_support_projector_child_scale_factor=0.55" \
    "++model.gaussian.boundary_support_projector_offset_scale=0.42" \
    "++model.gaussian.boundary_support_projector_outer_opacity_factor=0.74" \
    "++model.gaussian.boundary_support_projector_outer_scale_factor=0.88" \
    "opt.grad_clip=0.0022" \
    > "$LOG_DIR/${name}.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    log_event "$GPU" "$name" "train_failed" "status=$status"
    summary_row "$name" "train" "$exp_dir" "" "$final_ckpt" "failed" "status=$status"
    return "$status"
  fi

  log_event "$GPU" "$name" "train_done" "$final_ckpt"
  summary_row "$name" "train" "$exp_dir" "" "$final_ckpt" "ok" "iterations=$iterations"
  if [ "$DO_RENDER" = "1" ]; then
    IFS=',' read -ra STEPS <<< "$checkpoint_steps"
    for step in "${STEPS[@]}"; do
      render_checkpoint "$name" "$exp_dir" "$step" || true
    done
  fi
}

{
  echo "RUN_ID=$RUN_ID"
  echo "START_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
  echo "GPU=$GPU"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "BASE_EXP=$BASE_EXP"
  echo "BASE_CKPT=$BASE_CKPT"
  echo "BASE_RENDER_EXP=$BASE_RENDER_EXP"
  echo "LOG_DIR=$LOG_DIR"
  echo "PURPOSE=v247 boundary contributor projector: contributor-weighted residual score + anchor-normal support clones + over-score shrink"
} | tee "$LOG_DIR/run_info.txt"

write_status "starting" "preflight"
check_required || exit $?

run_residual_mining || true
train_v247b || true

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
log_event "$GPU" "queue" "all_done" "summary=$SUMMARY end=$END_BJT"
write_status "done" "END_BJT=$END_BJT SUMMARY=$SUMMARY"
{
  echo "END_BJT=$END_BJT"
  echo "SUMMARY=$SUMMARY"
  echo "EVENTS=$EVENTS"
  echo "STATUS_JSON=$STATUS_JSON"
} | tee -a "$LOG_DIR/run_info.txt"
